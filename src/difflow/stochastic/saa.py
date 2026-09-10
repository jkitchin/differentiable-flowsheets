"""Solving the sample average approximation, and reporting what it means.

Once the sample is fixed the stochastic program is an ordinary smooth
box-constrained problem in ``(x, U)`` --- the here-and-now decisions and one
recourse row per scenario.  It is solved here with projected Adam from
:mod:`difflow.flexibility.inner`, the same primitive the flexibility index
uses, under an augmented Lagrangian on the stochastic constraints.  The
Rockafellar--Uryasev auxiliaries are *not* decision variables: each is
recomputed at its own closed-form optimum at every iterate and frozen with
``stop_gradient``, which is exact by the envelope theorem and keeps a variable
with an unknown range out of a projected method's step-size calculation.  See
:mod:`difflow.stochastic.risk`.

Three decisions decide whether this works on a real flowsheet.

**Everything is scaled, and the scaling is exact.**  A profit in dollars per
year and a purity in mole fraction differ by eight orders of magnitude, and an
unscaled penalty term makes the whole problem one constraint.  Objective
values and constraint residuals are therefore affinely rescaled at the
starting point.  This is not an approximation: every risk measure here
satisfies :math:`\\rho(a + bZ) = a + b\\,\\rho(Z)` for ``b > 0``, so the
rescaled problem has the *same* solution, and the reported numbers are
unscaled back exactly.

**An augmented Lagrangian, not a growing quadratic penalty.**  This is not a
stylistic preference; a plain penalty continuation *fails* here, and it fails
quietly.  A projected Adam step is normalized by the running second moment of
the gradient, so a single excursion into a region where the penalty weight is
:math:`10^4` inflates that second moment by eight orders of magnitude and every
subsequent step shrinks to nothing --- the iterate freezes wherever the spike
left it, reports a feasible point, and is nowhere near the optimum.  An
augmented Lagrangian absorbs the constraint multiplier into ``lambda`` instead
of into the weight, so ``rho`` stays at its modest starting value, gradients
stay the size of the multiplier, and the converged point is the exactly
constrained one rather than an offset from it.

**Constraint violations are scored from the model, never from the multipliers.**
The augmented objective is what generates steps; what decides whether a
candidate is feasible, and which multi-start wins, is a fresh evaluation of
the model at that point.  This is the same rule :mod:`difflow.planning` states
for its LP slacks, for the same reason.

**The sample never changes.**  Not between iterations, not between rounds, not
between multi-starts.  See :mod:`difflow.stochastic.scenarios`.

What is deliberately not here
-----------------------------
No L-shaped or Benders decomposition, no scenario reduction, no multistage
tree.  The scenarios are handled by ``vmap``, which is the right answer up to
the point where the batch stops fitting in memory and the wrong answer after
it; :func:`~difflow.stochastic.diagnostics.check_scenario_health` will tell
you where you are.  No integer recourse: that is a different solver, and
:mod:`difflow.solvers.discopt_bridge` explains why an implicit flowsheet block
cannot be handed to one and still carry a global certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.flexibility.inner import box_adam
from difflow.flexibility.sets import ControlSpec
from difflow.params_mixin import ParamsMixin
from difflow.stochastic.problem import TwoStageProblem
from difflow.stochastic.risk import Chance
from difflow.stochastic.scenarios import ScenarioSet

#: A constraint counts as satisfied when its residual is at or below this,
#: taken relative to the size of the constraint's own right-hand side so it is
#: a tolerance and not a hidden unit.
FEASIBILITY_TOL = 1e-6


def _satisfied(residual: float, bound: float) -> bool:
    """Whether a residual is at or below the tolerance for that bound."""
    return bool(residual <= FEASIBILITY_TOL * (abs(bound) + 1.0))


@dataclass
class SAAOptions(ParamsMixin):
    """Settings for the sample average approximation solve.

    Attributes:
        steps: Projected-Adam iterations per penalty round.
        learning_rate: Step size as a fraction of each box width, cosine
            decayed to zero within a round.
        rounds: Augmented-Lagrangian rounds.  Each restarts Adam from the
            previous round's point, then updates the multipliers.
        penalty: The augmented-Lagrangian weight ``rho`` on the scaled
            constraint residuals.  Kept modest deliberately; the multipliers,
            not the weight, are what tighten the constraints.
        penalty_growth: Factor applied to ``rho`` in a round where the worst
            violation did not fall by at least a quarter --- the usual
            safeguard, and normally never triggered.
        tau: Initial smoothing scale for the hinges and maxima, as a fraction
            of the scaled residual range.  Annealed to
            ``tau * tau_decay`` within each round.
        tau_decay: Annealing factor over a round.
        n_starts: Deterministic multi-starts.  The SAA problem is convex only
            when the model is, which a flowsheet is not.
        seed: Unused by the search itself, which is deterministic; carried so
            that a result records the whole configuration.

    Example:
        >>> SAAOptions(steps=100, rounds=2)["steps"]
        100
    """

    steps: int = 300
    learning_rate: float = 0.06
    rounds: int = 4
    penalty: float = 10.0
    penalty_growth: float = 4.0
    tau: float = 0.05
    tau_decay: float = 1e-3
    n_starts: int = 3
    seed: int = 0


DEFAULT_SAA_OPTIONS = SAAOptions()


@dataclass
class SAAResult:
    """The solved plan, and the distribution it produced.

    Attributes:
        first_stage: ``{name: value}`` here-and-now decisions.
        recourse: ``{name: (n_scenarios,) array}`` wait-and-see decisions.
        risk_value: The risk measure at the solution, in the problem's own
            sense (already un-flipped when maximizing).
        objective_values: ``(n_scenarios,)`` objective in the problem's sense.
        outputs: Every model output at the solution, per scenario.
        constraint_residuals: Exact scalar residual per constraint, ``<= 0``
            when satisfied, scored from the model.
        violation_rates: Per-scenario violation frequency per constraint.
        feasible: Whether every constraint residual is at or below its
            tolerance.  For a
            :class:`~difflow.stochastic.risk.Chance` constraint this is the
            *CVaR surrogate*, which is conservative: a design can fail this and
            still meet the probability it was asked for, which is why
            :attr:`violation_rates` is reported beside it and
            :meth:`summary` says so explicitly when it happens.
        risk_report: Extra numbers from the risk measure, e.g. the value at
            risk and how many scenarios the CVaR tail actually contains.
        scenarios: The sample this was solved against.
        problem: The problem that was solved.
        options: The settings used.
        starts: Per-multi-start ``(risk value, total violation)``, so a
            scattered set of starts is visible rather than silently reduced to
            its best.
    """

    first_stage: dict[str, float]
    recourse: dict[str, np.ndarray]
    risk_value: float
    objective_values: np.ndarray
    outputs: dict[str, np.ndarray]
    constraint_residuals: np.ndarray
    violation_rates: np.ndarray
    feasible: bool
    risk_report: dict
    scenarios: ScenarioSet
    problem: TwoStageProblem
    options: SAAOptions
    starts: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Expected objective at the solution, in the problem's sense."""
        w = np.asarray(self.scenarios.weights)
        return float(w @ self.objective_values)

    @property
    def std(self) -> float:
        """Standard deviation of the objective across scenarios."""
        w = np.asarray(self.scenarios.weights)
        m = w @ self.objective_values
        return float(np.sqrt(max(float(w @ (self.objective_values - m) ** 2),
                                 0.0)))

    def quantile(self, q: float) -> float:
        """A quantile of the objective distribution.

        Args:
            q: Level in ``[0, 1]``.

        Returns:
            The empirical quantile of :attr:`objective_values`.
        """
        return float(np.quantile(self.objective_values, q))

    def recourse_saturation(self) -> dict[str, float]:
        """Fraction of scenarios in which each recourse variable sits at a bound.

        A recourse variable pinned at a bound in most scenarios is not
        providing recourse: the plant has run out of the adjustment the model
        was crediting it with.  That is usually the most actionable single
        number in a stochastic run --- it names the knob to widen, or the
        instrument to buy.

        Returns:
            ``{name: fraction in [0, 1]}``.
        """
        spec = self.problem.recourse
        out = {}
        for i, name in enumerate(spec.names):
            u = np.asarray(self.recourse[name])
            lo, hi = float(spec.lower[i]), float(spec.upper[i])
            span = max(hi - lo, 1e-30)
            at = ((u - lo) / span < 1e-6) | ((hi - u) / span < 1e-6)
            out[name] = float(np.mean(at))
        return out

    def summary(self) -> str:
        """The whole result as a report: plan, distribution, constraints."""
        p = self.problem
        sense = "maximize" if p.maximize else "minimize"
        lines = [
            f"SAA solution ({sense} {p.risk.describe()} on '{p.objective}')",
            f"  scenarios      {self.scenarios.n_scenarios} "
            f"({self.scenarios.distribution}"
            + (f", seed {self.scenarios.seed}" if self.scenarios.seed
               is not None else "") + ")",
            f"  risk value     {self.risk_value:12.6g}",
            f"  mean           {self.mean:12.6g}",
            f"  sd             {self.std:12.6g}",
            f"  p05 / p95      {self.quantile(0.05):12.6g} / "
            f"{self.quantile(0.95):.6g}",
            "  first stage (one value, every scenario):",
        ]
        for n, v in self.first_stage.items():
            lines.append(f"    {n:<20s}{v:12.6g}")
        if p.n_recourse:
            lines.append("  recourse (per scenario):")
            sat = self.recourse_saturation()
            lines.append(f"    {'variable':<20s}{'mean':>12s}{'min':>12s}"
                         f"{'max':>12s}{'at bound':>11s}")
            for n in p.recourse.names:
                u = np.asarray(self.recourse[n])
                lines.append(f"    {n:<20s}{u.mean():12.6g}{u.min():12.6g}"
                             f"{u.max():12.6g}{sat[n]:10.1%}")
        if p.constraints:
            lines.append("  constraints (scored from the model):")
            lines.append(f"    {'constraint':<44s}{'residual':>12s}"
                         f"{'violated':>10s}")
            conservative = []
            for i, c in enumerate(p.constraints):
                ok = _satisfied(self.constraint_residuals[i], c.bound)
                lines.append(f"   {' ' if ok else '*'}{c.describe():<44s}"
                             f"{self.constraint_residuals[i]:12.4g}"
                             f"{self.violation_rates[i]:9.1%}")
                if not ok and isinstance(c, Chance) and \
                        self.violation_rates[i] <= 1.0 - c.alpha:
                    conservative.append(c.name)
            lines.append(f"  feasible: {self.feasible}")
            if conservative:
                lines.append(
                    f"  note: {', '.join(conservative)} fails the CVaR "
                    f"surrogate but meets its probability empirically. The "
                    f"surrogate is conservative by construction, so this "
                    f"design may well be acceptable -- the decision is "
                    f"yours, not the solver's.")
        for k, v in self.risk_report.items():
            lines.append(f"  {k:<20s}{v:12.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"SAAResult(risk_value={self.risk_value:.6g}, "
                f"feasible={self.feasible}, "
                f"n_scenarios={self.scenarios.n_scenarios})")


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(problem: TwoStageProblem, first_stage, recourse,
             scenarios: ScenarioSet) -> SAAResult:
    """Score a plan you already have, without optimizing anything.

    This is the honest scorer: every number in the result comes from calling
    the model at the given point, with the auxiliaries set to their exact
    optima.  :func:`solve_saa` uses it to rank multi-starts, and
    :mod:`difflow.stochastic.diagnostics` uses it to evaluate one problem's
    solution under another's scenarios --- which is how the value of the
    stochastic solution is defined.

    Args:
        problem: The problem.
        first_stage: ``{name: value}`` or a vector in
            :attr:`~difflow.stochastic.problem.TwoStageProblem.first_stage`
            order.
        recourse: ``{name: value or (n_scenarios,) array}``, a
            ``(n_scenarios, n_recourse)`` array, or ``None`` for the box
            midpoint.
        scenarios: The sample to score against.

    Returns:
        An :class:`SAAResult` with :attr:`SAAResult.options` at the defaults;
        nothing was searched.
    """
    x = _pack_named(first_stage, problem.first_stage, "first_stage")
    U = _pack_recourse(recourse, problem.recourse, scenarios.n_scenarios)
    return _result(problem, scenarios, x, U, DEFAULT_SAA_OPTIONS, starts=[])


def _pack_named(values, spec, what: str) -> Array:
    """A named mapping or a vector, as an array in the spec's own order."""
    if values is None:
        return jnp.asarray(0.5 * (np.asarray(spec.lower)
                                  + np.asarray(spec.upper)))
    if isinstance(values, dict):
        missing = [n for n in spec.names if n not in values]
        if missing:
            raise KeyError(
                f"{what} is missing {missing}; it needs {list(spec.names)}."
            )
        extra = [k for k in values if k not in spec.names]
        if extra:
            raise KeyError(
                f"{what} carries {extra}, which are not declared decisions "
                f"({list(spec.names)})."
            )
        return jnp.asarray([float(values[n]) for n in spec.names])
    v = jnp.atleast_1d(jnp.asarray(values, dtype=float))
    if v.shape[0] != spec.n:
        raise ValueError(
            f"{what} has {v.shape[0]} entries, expected {spec.n}."
        )
    return v


def _pack_recourse(recourse, spec, n_scenarios: int) -> Array:
    """Recourse in any accepted form, as ``(n_scenarios, n_recourse)``."""
    if spec.n == 0:
        return jnp.zeros((n_scenarios, 0))
    if recourse is None:
        mid = 0.5 * (np.asarray(spec.lower) + np.asarray(spec.upper))
        return jnp.broadcast_to(jnp.asarray(mid), (n_scenarios, spec.n))
    if isinstance(recourse, dict):
        cols = []
        for name in spec.names:
            if name not in recourse:
                raise KeyError(
                    f"recourse is missing {name!r}; it needs "
                    f"{list(spec.names)}."
                )
            col = jnp.asarray(recourse[name], dtype=float)
            cols.append(jnp.broadcast_to(col.reshape(-1), (n_scenarios,))
                        if col.size == 1 else col.reshape(-1))
        return jnp.stack(cols, axis=1)
    U = jnp.asarray(recourse, dtype=float)
    if U.ndim == 1:
        U = jnp.broadcast_to(U, (n_scenarios, spec.n))
    if U.shape != (n_scenarios, spec.n):
        raise ValueError(
            f"recourse has shape {U.shape}, expected "
            f"({n_scenarios}, {spec.n})."
        )
    return U


def _result(problem, scenarios, x, U, options, starts) -> SAAResult:
    """Build an :class:`SAAResult` by calling the model at ``(x, U)``."""
    out = problem.outputs(x, U, scenarios)
    z = problem.objective_values(out)          # minimization sense
    w = scenarios.weights
    z_np, w_np = np.asarray(z), np.asarray(w)

    aux = problem.risk.exact_aux(z, w)
    risk_min = float(problem.risk.value(z, w, aux))

    residuals, rates = [], []
    for c, g in zip(problem.constraints, problem.constraint_values(out)):
        residuals.append(float(c.residual(g, w, c.exact_aux(g, w))))
        rates.append(c.violation_rate(np.asarray(g)))
    residuals = np.asarray(residuals, dtype=float)
    rates = np.asarray(rates, dtype=float)

    return SAAResult(
        first_stage={n: float(x[i])
                     for i, n in enumerate(problem.first_stage.names)},
        recourse={n: np.asarray(U[:, i])
                  for i, n in enumerate(problem.recourse.names)},
        risk_value=problem.sense * risk_min,
        objective_values=problem.sense * z_np,
        outputs={k: np.asarray(v) for k, v in out.items()},
        constraint_residuals=residuals,
        violation_rates=rates,
        feasible=bool(all(_satisfied(r, c.bound) for r, c
                          in zip(residuals, problem.constraints))),
        risk_report=problem.risk.report(z_np, w_np, np.asarray(aux)),
        scenarios=scenarios, problem=problem, options=options, starts=starts,
    )


# =============================================================================
# The solve
# =============================================================================


def solve_recourse(problem: TwoStageProblem, scenarios: ScenarioSet,
                   first_stage, *,
                   options: SAAOptions = DEFAULT_SAA_OPTIONS) -> SAAResult:
    """Optimize only the recourse, with the design held where you put it.

    The operating question rather than the design question: the plant is
    built, what is the best it can be run to under this distribution?  It is
    also half of the value of the stochastic solution, which is exactly this
    computation performed on the design a deterministic study produced --- see
    :func:`~difflow.stochastic.diagnostics.value_of_stochastic_solution`.

    Args:
        problem: The problem.
        scenarios: The fixed sample.
        first_stage: ``{name: value}`` or a vector, in the first stage's own
            order.
        options: Search settings.

    Returns:
        An :class:`SAAResult` whose first stage is exactly what was passed in.
    """
    from dataclasses import replace as _replace

    x = np.asarray(_pack_named(first_stage, problem.first_stage,
                               "first_stage"))
    pinned = _replace(problem,
                      first_stage=ControlSpec(lower=x, upper=x,
                                              names=problem.first_stage.names,
                                              start=x))
    return solve_saa(pinned, scenarios,
                     options=_replace(options, n_starts=1),
                     first_stage_start=x)


def _scales(problem, scenarios, x0, U0):
    """Affine rescalings of the objective and each constraint residual.

    Returns:
        ``(z_ref, z_scale, [constraint scales])``, all strictly positive
        scales.  Every risk measure and constraint form here is positively
        homogeneous, and the risk measures are translation-equivariant, so
        rescaling by these leaves the solution set unchanged.
    """
    out = problem.outputs(x0, U0, scenarios)
    z = np.asarray(problem.objective_values(out))
    z_ref = float(np.mean(z))
    z_scale = float(np.std(z) + np.abs(z_ref) + 1.0)
    c_scales = []
    for c, g in zip(problem.constraints, problem.constraint_values(out)):
        r = np.asarray(c.signed(jnp.asarray(g)))
        s = float(np.std(r) + np.mean(np.abs(r)))
        c_scales.append(s if s > 0.0 else 1.0)
    return z_ref, z_scale, np.asarray(c_scales, dtype=float)


def solve_saa(problem: TwoStageProblem, scenarios: ScenarioSet, *,
              options: SAAOptions = DEFAULT_SAA_OPTIONS,
              first_stage_start=None, verbose: bool = False) -> SAAResult:
    """Solve the sample average approximation of a two-stage problem.

    Args:
        problem: The problem.
        scenarios: The fixed sample.  Drawn once and reused throughout; see
            :mod:`difflow.stochastic.scenarios` for why there is no
            resample-per-iteration option.
        options: Search settings.
        first_stage_start: Optional ``{name: value}`` or vector to use as the
            first multi-start, e.g. a deterministic design being improved on.
        verbose: Print each start's exact risk value and total violation as it
            finishes.

    Returns:
        An :class:`SAAResult`.  Feasibility and the reported risk are scored
        by calling the model at the returned point, not read off the penalty.

    Raises:
        ValueError: If the model's outputs do not carry the objective or a
            constrained name (raised from the first evaluation).

    Example:
        >>> import difflow.stochastic as st
        >>> def model(x, u, theta):
        ...     return {"cost": (x["size"] - theta["load"]) ** 2}
        >>> prob = st.TwoStageProblem(model=model,
        ...                           first_stage={"size": (0.0, 10.0)},
        ...                           objective="cost")
        >>> scen = st.ScenarioSet.normal({"load": (5.0, 1.0)}, n=64, seed=0)
        >>> res = st.solve_saa(prob, scen, options=st.SAAOptions(steps=200))
        >>> abs(res.first_stage["size"] - 5.0) < 0.2   # the sample mean
        True
    """
    fs, rc = problem.first_stage, problem.recourse
    S = scenarios.n_scenarios
    n_x, n_u = fs.n, rc.n

    x_lo, x_hi = np.asarray(fs.lower), np.asarray(fs.upper)
    u_lo, u_hi = np.asarray(rc.lower), np.asarray(rc.upper)

    # -- starting points, and the scaling read off the first one -------
    x_starts = np.array(fs.starts(max(int(options.n_starts), 1)),
                        dtype=float, copy=True)
    if first_stage_start is not None:
        x_starts[0] = np.asarray(_pack_named(first_stage_start, fs,
                                             "first_stage_start"))
    U0 = np.broadcast_to(0.5 * (u_lo + u_hi), (S, n_u)).copy() \
        if n_u else np.zeros((S, 0))
    z_ref, z_scale, c_scales = _scales(problem, scenarios,
                                       jnp.asarray(x_starts[0]),
                                       jnp.asarray(U0))

    def scaled_pieces(x, U):
        """Scaled objective values, and scaled signed residual per constraint."""
        out = problem.outputs(x, U, scenarios)
        z = (problem.objective_values(out) - z_ref) / z_scale
        rs = [c.signed(g) / c_scales[i]
              for i, (c, g) in enumerate(zip(problem.constraints,
                                             problem.constraint_values(out)))]
        return z, rs

    w = scenarios.weights

    def unpack(v):
        x = v[:n_x]
        U = v[n_x:].reshape(S, n_u) if n_u else jnp.zeros((S, 0))
        return x, U

    def scalar_residuals(x, U, tau, exact: bool):
        """One scalar per constraint, on the scaled residual."""
        _, rs = scaled_pieces(x, U)
        out = []
        for i, c in enumerate(problem.constraints):
            ca = jax.lax.stop_gradient(c.exact_aux_from_signed(rs[i], w))
            out.append(c.residual_from_signed(rs[i], w, ca) if exact
                       else c.surrogate_from_signed(rs[i], w, ca, tau))
        return out

    def augmented(v, progress, rho, lam):
        x, U = unpack(v)
        tau = options.tau * options.tau_decay ** progress
        z, rs = scaled_pieces(x, U)
        # Each Rockafellar-Uryasev auxiliary at its own exact optimum, frozen:
        # the envelope theorem makes this the exact gradient, and keeps a
        # variable whose range nobody knows out of the projected step.
        a = jax.lax.stop_gradient(problem.risk.exact_aux(z, w))
        phi = problem.risk.surrogate(z, w, a, tau)
        for i, c in enumerate(problem.constraints):
            ca = jax.lax.stop_gradient(c.exact_aux_from_signed(rs[i], w))
            r = c.surrogate_from_signed(rs[i], w, ca, tau)
            # Hestenes-Powell-Rockafellar form for an inequality r <= 0.
            phi = phi + (jnp.maximum(0.0, lam[i] + rho * r) ** 2
                         - lam[i] ** 2) / (2.0 * rho)
        return phi

    # -- solve from each start -----------------------------------------
    best, records = None, []
    for si in range(x_starts.shape[0]):
        x0 = x_starts[si]
        v = np.concatenate([x0, U0.ravel()])
        lo = np.concatenate([x_lo, np.tile(u_lo, S)])
        hi = np.concatenate([x_hi, np.tile(u_hi, S)])

        rho = float(options.penalty)
        lam = np.zeros(len(problem.constraints))
        prev_violation = np.inf
        run = jax.jit(lambda v0, r, l: box_adam(
            lambda vv, pr: augmented(vv, pr, r, l),
            v0, jnp.asarray(lo), jnp.asarray(hi),
            int(options.steps), float(options.learning_rate)))
        for _ in range(max(int(options.rounds), 1)):
            v = np.asarray(run(jnp.asarray(v), rho, jnp.asarray(lam)))
            if not problem.constraints:
                break
            xv, Uv = unpack(jnp.asarray(v))
            r = np.asarray([float(t) for t in
                            scalar_residuals(xv, Uv, 0.0, True)])
            lam = np.maximum(0.0, lam + rho * r)
            violation = float(np.max(np.maximum(r, 0.0)))
            if violation > 0.25 * prev_violation:
                rho *= float(options.penalty_growth)
            prev_violation = violation

        x, U = unpack(jnp.asarray(v))
        cand = _result(problem, scenarios, x, U, options, starts=[])
        total_violation = float(np.sum(np.maximum(cand.constraint_residuals,
                                                  0.0)))
        records.append((cand.risk_value, total_violation))
        if verbose:
            print(f"  start {si}: risk {cand.risk_value:12.6g}  "
                  f"violation {total_violation:10.4g}")
        key = (not cand.feasible,
               problem.sense * cand.risk_value + total_violation)
        if best is None or key < best[0]:
            best = (key, cand)

    result = best[1]
    result.starts = records
    return result

