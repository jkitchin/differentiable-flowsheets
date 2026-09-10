"""The problem statement: what is decided before, and what is decided after.

The whole content of a two-stage stochastic program is one distinction, and
it is a modelling decision that nobody but the engineer can make:

*First stage* --- **here and now**.  Decided once, before anything is
revealed, and lived with in every scenario.  Number of stages in a cascade,
settler volume, whether the scrub section exists.  There is exactly one value,
shared by every scenario, and that sharing *is* non-anticipativity: it is
enforced structurally, by there being one array, not by constraints that could
be written down wrongly.

*Second stage* --- **wait and see**.  Re-decided per scenario, after the
uncertain parameters are known.  Operating pH, solvent-to-feed ratio, scrub
acid --- the knobs an operator still has once today's assay comes back.  Each
scenario gets its own row.

Getting this split wrong is the classic error, and it goes wrong in one
direction far more often than the other: putting an operating variable in the
first stage prices a plant that cannot be adjusted, which understates its
worth and over-designs it; putting a *design* variable in the second stage
prices a plant that rebuilds itself every shift, which is worth strictly more
than anything you can build.  The second mistake is the dangerous one because
its answer looks *better*.

Which uncertainties admit recourse at all
------------------------------------------
Recourse is only real when the parameter is *observed* before the recourse
must be committed, and this is the same distinction
:mod:`difflow.flexibility` draws between feed uncertainty and parameter
uncertainty.  A feed assay is measured on arrival, so pH can genuinely be
re-optimized against it.  A distribution coefficient is never revealed: it is
a property of the chemistry, identical in every campaign and unknown in all of
them.

That does not make a stochastic program over distribution coefficients wrong
--- but it does change what the recourse variable means, and the honest
formulations are these two:

1. **No recourse at all** (``recourse=None``): the design is scored against
   the parameter distribution with a single fixed operating point.  This is
   the right model for an unobservable parameter and a plant with no feedback.
2. **Recourse on an observable proxy**: the operator does not see ``D``, but
   does see the raffinate assay, and turns pH until the assay is on target.
   Then pH is genuinely second-stage --- not because ``D`` was revealed, but
   because its *consequence* was.

The second is what a real plant does, and it is what makes the difference
between the stochastic answer and the deterministic one worth having.  State
which one you mean; the module will not guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.flexibility.sets import ControlSpec, as_control_spec
from difflow.stochastic.risk import (
    Expectation, RiskMeasure, as_constraint, as_risk_measure,
)
from difflow.stochastic.scenarios import ScenarioSet


@dataclass
class TwoStageProblem:
    """A two-stage stochastic program over a differentiable model.

    Attributes:
        model: ``model(x, u, theta) -> {name: scalar}``, where ``x``, ``u``
            and ``theta`` are dicts keyed by the first-stage, recourse and
            parameter names.  Must be JAX-traceable: it is ``vmap``-ed over
            the scenarios and differentiated through.  A difflow flowsheet
            solve qualifies --- ``Flowsheet.solve`` swaps its Python
            acceleration loops for an implicitly-differentiated fixed point
            the moment it sees a tracer.
        first_stage: The here-and-now decisions, as ``{name: (lower, upper)}``
            or a :class:`~difflow.flexibility.sets.ControlSpec`.
        recourse: The wait-and-see decisions, in the same form, or ``None``
            for a design scored with no recourse at all.
        objective: Name of the model output to optimize.
        risk: How the per-scenario objective values are collapsed to one
            number --- a :class:`~difflow.stochastic.risk.RiskMeasure` or a
            shorthand such as ``("cvar", 0.95)``.  Defaults to the
            expectation.
        constraints: Constraints on named model outputs; see
            :func:`~difflow.stochastic.risk.as_constraint`.
        maximize: Maximize the objective instead of minimizing it.  The risk
            measure is applied to the *minimization* form throughout, so
            ``CVaR`` under ``maximize=True`` is the mean of the worst
            (lowest) tail, which is what anyone asking for it means.

    Example:
        >>> def model(x, u, theta):
        ...     yield_ = theta["k"] * u["T"] * x["V"] / (1.0 + x["V"])
        ...     return {"profit": yield_ - 0.1 * x["V"], "purity": yield_}
        >>> prob = TwoStageProblem(
        ...     model=model,
        ...     first_stage={"V": (0.1, 10.0)},
        ...     recourse={"T": (0.5, 1.5)},
        ...     objective="profit", maximize=True,
        ...     constraints=[("purity", ">=", 0.5, 0.9)],
        ... )
        >>> prob.n_first_stage, prob.n_recourse
        (1, 1)
    """

    model: Callable
    first_stage: Mapping | ControlSpec
    objective: str
    recourse: Mapping | ControlSpec | None = None
    risk: RiskMeasure | str | tuple = field(default_factory=Expectation)
    constraints: Sequence = ()
    maximize: bool = False

    def __post_init__(self):
        if not callable(self.model):
            raise TypeError("model must be callable: model(x, u, theta).")
        self.first_stage = as_control_spec(self.first_stage)
        if self.first_stage.n == 0:
            raise ValueError(
                "A two-stage problem needs at least one first-stage decision. "
                "With none, every scenario is independent and the answer is "
                "the wait-and-see bound; compute that directly with "
                "difflow.stochastic.diagnostics.wait_and_see."
            )
        self.recourse = as_control_spec(self.recourse)
        self.risk = as_risk_measure(self.risk)
        self.constraints = tuple(as_constraint(c) for c in self.constraints)
        self.objective = str(self.objective)
        overlap = set(self.first_stage.names) & set(self.recourse.names)
        if overlap:
            raise ValueError(
                f"{sorted(overlap)} appear in both the first stage and the "
                f"recourse. A variable is decided either before the "
                f"uncertainty is revealed or after it, not both."
            )

    # -- shape ----------------------------------------------------------

    @property
    def n_first_stage(self) -> int:
        """Number of here-and-now decisions."""
        return self.first_stage.n

    @property
    def n_recourse(self) -> int:
        """Number of wait-and-see decisions, per scenario."""
        return self.recourse.n

    @property
    def sense(self) -> float:
        """``-1`` when maximizing, ``+1`` when minimizing."""
        return -1.0 if self.maximize else 1.0

    # -- evaluation -----------------------------------------------------

    def outputs(self, x, U, scenarios: ScenarioSet) -> dict[str, Array]:
        """Every model output, for every scenario.

        This is the one place the model is called.  It ``vmap``s over
        scenarios, pairing scenario ``s`` with recourse row ``s``, so a
        flowsheet solve inside ``model`` runs batched rather than in a Python
        loop.

        Args:
            x: ``(n_first_stage,)`` here-and-now decisions.
            U: ``(n_scenarios, n_recourse)`` recourse, one row per scenario.
                A ``(n_recourse,)`` vector is broadcast to every scenario,
                which is what a no-recourse problem wants.
            scenarios: The sample.

        Returns:
            ``{output name: (n_scenarios,) array}``.
        """
        x = jnp.atleast_1d(jnp.asarray(x, dtype=float))
        U = jnp.asarray(U, dtype=float)
        if U.ndim == 1:
            U = jnp.broadcast_to(U, (scenarios.n_scenarios, U.shape[0]))
        x_names = self.first_stage.names
        u_names = self.recourse.names

        def one(u_row, theta_row):
            xd = {n: x[i] for i, n in enumerate(x_names)}
            ud = {n: u_row[i] for i, n in enumerate(u_names)}
            td = scenarios.as_dict(theta_row)
            out = self.model(xd, ud, td)
            if not isinstance(out, Mapping):
                raise TypeError(
                    f"model must return a mapping of named outputs, got "
                    f"{type(out).__name__}. Even a single-output model should "
                    f"return {{'{self.objective}': value}} so constraints can "
                    f"name what they constrain."
                )
            return {k: jnp.asarray(v, dtype=float).reshape(())
                    for k, v in out.items()}

        return jax.vmap(one)(U, scenarios.draws)

    def objective_values(self, outputs: Mapping[str, Array]) -> Array:
        """The per-scenario objective, in minimization sense.

        Args:
            outputs: What :meth:`outputs` returned.

        Returns:
            A ``(n_scenarios,)`` array, already sign-flipped when
            :attr:`maximize` is set.

        Raises:
            KeyError: If the model did not produce the named objective.
        """
        if self.objective not in outputs:
            raise KeyError(
                f"The model produced no output named {self.objective!r}. It "
                f"returned {sorted(outputs)}."
            )
        return self.sense * outputs[self.objective]

    def constraint_values(self, outputs: Mapping[str, Array]
                          ) -> list[Array]:
        """The per-scenario value of each constrained output.

        Args:
            outputs: What :meth:`outputs` returned.

        Returns:
            One ``(n_scenarios,)`` array per constraint, in order.

        Raises:
            KeyError: If a constraint names an output the model does not
                produce.
        """
        vals = []
        for c in self.constraints:
            if c.name not in outputs:
                raise KeyError(
                    f"Constraint {c.describe()!r} names the output "
                    f"{c.name!r}, which the model does not produce. It "
                    f"returned {sorted(outputs)}."
                )
            vals.append(outputs[c.name])
        return vals

    # -- auxiliary variable bookkeeping ---------------------------------

    @property
    def n_aux(self) -> int:
        """Auxiliary variables introduced by the risk measure and constraints.

        One per Rockafellar--Uryasev term: the risk measure's value at risk,
        plus one for every :class:`~difflow.stochastic.risk.Chance`
        constraint.
        """
        return int(self.risk.n_aux
                   + sum(c.n_aux for c in self.constraints))

    def aux_slices(self) -> tuple[slice, list[slice]]:
        """Where each term's auxiliary variables live in the flat aux vector.

        Returns:
            ``(risk slice, [constraint slices])``.
        """
        k = self.risk.n_aux
        risk_slice = slice(0, k)
        con_slices = []
        for c in self.constraints:
            con_slices.append(slice(k, k + c.n_aux))
            k += c.n_aux
        return risk_slice, con_slices

    # -- reporting ------------------------------------------------------

    def describe(self) -> str:
        """State the problem: sense, objective, stages, risk, constraints."""
        sense = "maximize" if self.maximize else "minimize"
        lines = [
            f"TwoStageProblem: {sense} {self.risk.describe()} "
            f"on '{self.objective}'",
            f"  first stage (here and now), {self.n_first_stage} decisions:",
        ]
        for i, n in enumerate(self.first_stage.names):
            lines.append(f"    {n:<20s}"
                         f"[{float(self.first_stage.lower[i]):g}, "
                         f"{float(self.first_stage.upper[i]):g}]")
        if self.n_recourse:
            lines.append(f"  recourse (wait and see), {self.n_recourse} "
                         f"decisions per scenario:")
            for i, n in enumerate(self.recourse.names):
                lines.append(f"    {n:<20s}"
                             f"[{float(self.recourse.lower[i]):g}, "
                             f"{float(self.recourse.upper[i]):g}]")
        else:
            lines.append("  recourse: none -- one operating point for every "
                         "scenario")
        if self.constraints:
            lines.append(f"  constraints ({len(self.constraints)}):")
            for c in self.constraints:
                lines.append(f"    {c.describe()}")
        else:
            lines.append("  constraints: none (bounds only)")
        if self.n_aux:
            lines.append(f"  auxiliary variables: {self.n_aux} "
                         f"(Rockafellar-Uryasev)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"TwoStageProblem(objective={self.objective!r}, "
                f"n_first_stage={self.n_first_stage}, "
                f"n_recourse={self.n_recourse}, "
                f"risk={self.risk.describe()!r}, "
                f"constraints={len(self.constraints)})")
