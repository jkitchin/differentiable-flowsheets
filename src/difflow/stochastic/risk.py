"""What "the best plan" means once the answer is a distribution.

A deterministic design has one number to minimize.  A stochastic one has a
histogram, and *which* functional of that histogram you minimize is a
modelling choice with real consequences, not a solver setting.  This module
carries the four that actually get used, plus the constraint forms that go
with them.

Two conventions run through everything here, both inherited from
:mod:`difflow.flexibility` because they are the same conventions for the same
reasons.

**Smoothed for the search, exact for the answer.**  ``CVaR`` and
``WorstCase`` are built on kinks --- a hinge and a maximum --- whose exact
subgradients are carried by whichever handful of scenarios sits on the kink at
that iterate, which makes for a jumpy search direction.  Each therefore
exposes a smoothed :meth:`~RiskMeasure.surrogate` used to generate steps, with
the smoothing annealed to nothing over the run, and an exact
:meth:`~RiskMeasure.value` which is the only thing ever reported.

The smoothing is deliberately one-sided: ``softplus >= relu`` and
``smooth_max >= max``, so a smoothed risk is always *above* the true one.  That
is the safe direction --- an imperfectly annealed run over-states the risk and
buys a little too much margin.  Had it been smoothed from below, the same run
would have made a risky plan look safe, which is why the reported number is the
exact one regardless.

**Rockafellar--Uryasev, evaluated at its own optimum.**  CVaR is written in
the Rockafellar--Uryasev form

.. math:: \\mathrm{CVaR}_\\alpha(Z) = \\min_t\\ t
          + \\frac{1}{1-\\alpha}\\,\\mathbb{E}\\,[(Z - t)_+] ,

which introduces the value at risk ``t`` and is jointly convex in ``(x, t)``.
The textbook move is to hand ``t`` to the optimizer as one more decision
variable.  This module does not, and the reason is worth stating: ``t`` lives
on the scale of the objective, its useful range is not known before the run,
and a projected method takes steps proportional to the width of the box you
give it --- so a generously-sized box for ``t`` makes the auxiliary swing
harder than the design variables and the search follows it instead of the
design.  That failure is silent and looks like a converged answer at a bound.

Instead ``t`` is recomputed at every iterate as the empirical value at risk,
which is its exact minimizer at fixed ``Z``, and wrapped in
``stop_gradient``.  By the envelope theorem the derivative of a minimum with
respect to everything but the minimizing variable is the partial derivative at
the minimizer, so this is the *exact* gradient of CVaR, not an approximation
of it --- and the value reported is exactly ``min_t``, never a value inherited
from a partially converged auxiliary.  :mod:`difflow.flexibility.inner` makes
the same trade for the same reason.

Reference:
    Rockafellar and Uryasev, J. Risk 2 (2000) 21,
    doi:10.21314/JOR.2000.038.
    Nemirovski and Shapiro, SIAM J. Optim. 17 (2006) 969,
    doi:10.1137/050622328 (CVaR as the convex conservative approximation of a
    chance constraint).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.flexibility.inner import smooth_max

#: Comparison operators a constraint may use.
OPERATORS = ("<=", ">=", "==")

#: Relative floor on a smoothing temperature, so annealing never divides by
#: zero on the last step.
_TAU_FLOOR = 1e-12


def _softplus(z: Array, tau: Array | float) -> Array:
    """``(z)_+`` smoothed at scale ``tau``, numerically safe for large ``z``.

    Args:
        z: Values.
        tau: Smoothing scale.  As ``tau -> 0`` this tends to
            ``jnp.maximum(z, 0)`` from above.

    Returns:
        The smoothed positive part, elementwise.
    """
    t = jnp.maximum(jnp.asarray(tau, dtype=float), _TAU_FLOOR)
    u = z / t
    # log(1 + e^u) written so neither branch overflows.
    return t * (jnp.maximum(u, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(u))))


def weighted_quantile(z: Array, weights: Array, alpha: float) -> Array:
    """The ``alpha``-quantile of a weighted sample, by the empirical CDF.

    Written in ``jnp`` throughout so it can be evaluated on a tracer: it is
    called inside the objective at every iterate, not only for reporting.

    Args:
        z: Sample values.
        weights: Probabilities, summing to one.
        alpha: Quantile level in ``[0, 1]``.

    Returns:
        The smallest sample value whose cumulative weight reaches ``alpha``,
        as a scalar array.
    """
    v = jnp.asarray(z, dtype=float).reshape(-1)
    w = jnp.asarray(weights, dtype=float).reshape(-1)
    order = jnp.argsort(v)
    vs, ws = v[order], w[order]
    c = jnp.cumsum(ws)
    idx = jnp.searchsorted(c, jnp.clip(alpha, 0.0, 1.0) * c[-1])
    return vs[jnp.clip(idx, 0, vs.shape[0] - 1)]


# =============================================================================
# Risk measures
# =============================================================================


@dataclass(frozen=True)
class RiskMeasure:
    """Base class: a functional of the per-scenario objective values.

    A risk measure sees ``z``, the ``(n_scenarios,)`` vector of objective
    values the model produced, and ``w``, the scenario probabilities, and
    returns a scalar.  Some need auxiliary decision variables of their own;
    those are optimized jointly with the design and are reported back so the
    number they converged to can be read (for CVaR it is the value at risk,
    which is worth seeing).
    """

    #: How many auxiliary decision variables this measure introduces. A plain
    #: class attribute, deliberately *not* an annotated dataclass field: as a
    #: field on the base class it would take the first positional slot, and
    #: ``CVaR(0.9)`` would silently set the auxiliary count rather than alpha.
    n_aux = 0

    def exact_aux(self, z: Array, w: Array) -> Array:
        """The auxiliary values that minimize :meth:`value` at fixed ``z``.

        The Rockafellar--Uryasev auxiliary is a *minimization* variable, so at
        any fixed sample its optimum is available in closed form --- for CVaR
        it is the empirical value at risk.  The solver calls this at every
        iterate and freezes the result with ``stop_gradient``; see the module
        docstring for why that is the exact gradient and not an approximation.

        Traceable, so it may be called on a tracer inside an objective.

        Args:
            z: Per-scenario objective values.
            w: Scenario probabilities.

        Returns:
            An array of length :attr:`n_aux`.
        """
        return jnp.zeros(self.n_aux)

    def value(self, z: Array, w: Array, aux: Array) -> Array:
        """The exact risk value.  This is what gets reported."""
        raise NotImplementedError

    def surrogate(self, z: Array, w: Array, aux: Array,
                  tau: Array | float) -> Array:
        """The smoothed value used to generate search directions.

        Args:
            z: Per-scenario objective values.
            w: Scenario probabilities.
            aux: Auxiliary variables.
            tau: Smoothing scale, annealed toward zero over a run.

        Returns:
            A scalar, never below :meth:`value` by more than the smoothing.
        """
        return self.value(z, w, aux)

    def describe(self) -> str:
        """One line naming the measure."""
        return type(self).__name__

    def report(self, z, w, aux) -> dict:
        """Extra numbers worth showing alongside the value."""
        return {}


@dataclass(frozen=True)
class Expectation(RiskMeasure):
    """Plain expected value --- risk neutral.

    The right choice when the plant runs many campaigns and only the average
    matters.  It is the wrong choice when a single bad campaign is
    unrecoverable, which is what the others are for.

    Example:
        >>> import jax.numpy as jnp
        >>> z = jnp.array([1.0, 2.0, 6.0]); w = jnp.full((3,), 1/3)
        >>> float(Expectation().value(z, w, jnp.zeros(0)))
        3.0
    """

    n_aux = 0

    def value(self, z, w, aux):
        return jnp.sum(w * z)

    def describe(self) -> str:
        return "E[objective]"


@dataclass(frozen=True)
class MeanStd(RiskMeasure):
    """``E[Z] + kappa * sd[Z]`` --- the engineer's risk premium.

    Cheap, smooth everywhere, and needs no auxiliary variable, which makes it
    a good first thing to try.  What it is not is *coherent*: it penalizes
    upside as heavily as downside, so a design that is occasionally much
    better than nominal is scored as if that were a problem.  Use
    :class:`CVaR` when the tail is what you care about.

    Attributes:
        kappa: Standard deviations of premium.  ``kappa = 2`` is the usual
            engineering default.
    """

    kappa: float = 2.0
    n_aux = 0

    def value(self, z, w, aux):
        mean = jnp.sum(w * z)
        var = jnp.sum(w * (z - mean) ** 2)
        return mean + self.kappa * jnp.sqrt(jnp.maximum(var, 0.0))

    def describe(self) -> str:
        return f"E[objective] + {self.kappa:g} sd[objective]"


@dataclass(frozen=True)
class CVaR(RiskMeasure):
    """Conditional value at risk: the mean of the worst ``1 - alpha`` tail.

    ``CVaR(0.95)`` is "the average of my worst 5% of campaigns".  It is
    coherent, convex, and --- unlike a plain quantile --- it is sensitive to
    *how bad* the tail is rather than only to where it starts.

    The tail is estimated from the scenarios that fall in it, so the sample
    size that matters is not ``S`` but ``(1 - alpha) S``.  At ``alpha = 0.99``
    with 200 scenarios that is two.  :func:`~difflow.stochastic.diagnostics.
    check_scenario_health` reports this number and complains when it is small,
    because nothing else in the pipeline will.

    Attributes:
        alpha: Confidence level in ``(0, 1)``.  Larger is more conservative.

    Example:
        >>> import jax.numpy as jnp
        >>> z = jnp.arange(10.0); w = jnp.full((10,), 0.1)
        >>> t = jnp.array([8.0])                  # the value at risk
        >>> float(CVaR(0.8).value(z, w, t))       # mean of {8, 9}
        8.5
    """

    alpha: float = 0.95
    n_aux = 1

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"CVaR alpha must lie strictly in (0, 1), got {self.alpha}. "
                f"alpha = 0 is the expectation (use Expectation) and alpha = 1 "
                f"is the worst case (use WorstCase)."
            )

    def exact_aux(self, z, w):
        return weighted_quantile(z, w, self.alpha).reshape(1)

    def value(self, z, w, aux):
        t = aux[0]
        return t + jnp.sum(w * jnp.maximum(z - t, 0.0)) / (1.0 - self.alpha)

    def surrogate(self, z, w, aux, tau):
        t = aux[0]
        return t + jnp.sum(w * _softplus(z - t, tau)) / (1.0 - self.alpha)

    def describe(self) -> str:
        return f"CVaR_{self.alpha:g}[objective]"

    def report(self, z, w, aux) -> dict:
        return {"value_at_risk": float(np.asarray(aux).reshape(-1)[0]),
                "tail_scenarios": float((1.0 - self.alpha)
                                        * np.size(np.asarray(z)))}


@dataclass(frozen=True)
class WorstCase(RiskMeasure):
    """The maximum over the sample --- robust optimization, sampled.

    Honest about what it is: this is the worst of the ``S`` scenarios drawn,
    not the worst of the distribution, and it gets more pessimistic as ``S``
    grows without ever converging to anything.  When a genuine guarantee over
    a set is what you want, :mod:`difflow.flexibility` computes it over the
    set's vertices instead of hoping a sample found the corner.
    """

    n_aux = 0

    def value(self, z, w, aux):
        return jnp.max(z)

    def surrogate(self, z, w, aux, tau):
        scale = jnp.maximum(jnp.max(jnp.abs(z)), 1.0)
        return smooth_max(z, jnp.maximum(tau, _TAU_FLOOR) * scale)

    def describe(self) -> str:
        return "max_s[objective]"


#: Risk measures the string shorthand understands.
RISK_MEASURES = {
    "expectation": Expectation,
    "mean": Expectation,
    "mean_std": MeanStd,
    "cvar": CVaR,
    "worst_case": WorstCase,
}


def as_risk_measure(obj) -> RiskMeasure:
    """Coerce a shorthand into a :class:`RiskMeasure`.

    Args:
        obj: A :class:`RiskMeasure`, a name from :data:`RISK_MEASURES`, or a
            ``(name, parameter)`` pair such as ``("cvar", 0.95)``.

    Returns:
        The risk measure.

    Raises:
        ValueError: For an unknown name.

    Example:
        >>> as_risk_measure(("cvar", 0.9)).describe()
        'CVaR_0.9[objective]'
    """
    if isinstance(obj, RiskMeasure):
        return obj
    if isinstance(obj, str):
        name, arg = obj, None
    else:
        try:
            name, arg = obj
        except (TypeError, ValueError):
            raise ValueError(
                f"Cannot read {obj!r} as a risk measure. Give a RiskMeasure, "
                f"a name from {sorted(RISK_MEASURES)}, or a (name, parameter) "
                f"pair."
            ) from None
    key = str(name).lower()
    if key not in RISK_MEASURES:
        raise ValueError(
            f"Unknown risk measure {name!r}. Available: "
            f"{sorted(RISK_MEASURES)}."
        )
    cls = RISK_MEASURES[key]
    return cls() if arg is None else cls(arg)


# =============================================================================
# Constraints
# =============================================================================


@dataclass(frozen=True)
class StochasticConstraint:
    """Base class: a constraint on a model output that varies by scenario.

    A constraint is written against one named output of the model, and the
    subclass decides what "satisfied" means when that output is a
    distribution: on average (:class:`Expected`), with a stated probability
    (:class:`Chance`), or in every scenario drawn (:class:`Robust`).

    Every method comes in two forms.  The public one takes the raw output
    ``g``; the ``*_from_signed`` one takes the residual ``self.signed(g)``
    directly, which is what the solver calls after rescaling it.  All three
    forms here are positively homogeneous in that residual, so a positive
    rescaling changes the number but never the sign --- which is the whole
    meaning of the constraint.

    Attributes:
        name: The model output this constrains.
        op: One of :data:`OPERATORS`.
        bound: The right-hand side.
    """

    name: str
    op: str
    bound: float
    n_aux = 0

    def __post_init__(self):
        if self.op not in OPERATORS:
            raise ValueError(
                f"Constraint operator must be one of {OPERATORS}, got "
                f"{self.op!r}."
            )

    def signed(self, g: Array) -> Array:
        """Per-scenario residual, written so that ``<= 0`` means satisfied.

        Args:
            g: ``(n_scenarios,)`` values of the constrained output.

        Returns:
            The signed residual, same shape.
        """
        if self.op == "<=":
            return g - self.bound
        if self.op == ">=":
            return self.bound - g
        return jnp.abs(g - self.bound)

    # -- the signed-residual forms the solver uses ----------------------

    def exact_aux_from_signed(self, r: Array, w: Array) -> Array:
        """The auxiliary values minimizing the residual at fixed ``r``."""
        return jnp.zeros(self.n_aux)

    def residual_from_signed(self, r: Array, w: Array, aux: Array) -> Array:
        """The exact scalar residual: satisfied exactly when ``<= 0``."""
        raise NotImplementedError

    def surrogate_from_signed(self, r, w, aux, tau) -> Array:
        """The smoothed scalar residual used to generate search directions."""
        return self.residual_from_signed(r, w, aux)

    # -- the public forms, in terms of the raw output --------------------

    def exact_aux(self, g, w) -> Array:
        """The auxiliary values that minimize :meth:`residual` at fixed ``g``.

        Args:
            g: ``(n_scenarios,)`` values of the constrained output.
            w: Scenario probabilities.

        Returns:
            An array of length :attr:`n_aux`.
        """
        return self.exact_aux_from_signed(self.signed(jnp.asarray(g)), w)

    def residual(self, g: Array, w: Array, aux: Array) -> Array:
        """The exact scalar residual: satisfied exactly when ``<= 0``.

        Args:
            g: ``(n_scenarios,)`` values of the constrained output.
            w: Scenario probabilities.
            aux: This constraint's auxiliary variables.

        Returns:
            A scalar.
        """
        return self.residual_from_signed(self.signed(g), w, aux)

    def surrogate(self, g, w, aux, tau) -> Array:
        """The smoothed residual used to generate search directions.

        Args:
            g: ``(n_scenarios,)`` values of the constrained output.
            w: Scenario probabilities.
            aux: This constraint's auxiliary variables.
            tau: Smoothing scale, annealed toward zero over a run.

        Returns:
            A scalar, never below :meth:`residual` by more than the smoothing.
        """
        return self.surrogate_from_signed(self.signed(g), w, aux, tau)

    def describe(self) -> str:
        """One line stating the constraint."""
        return f"{self.name} {self.op} {self.bound:g}"

    def violation_rate(self, g) -> float:
        """Fraction of scenarios in which the *per-scenario* form is violated.

        Reported for every constraint kind, including the ones that were never
        posed per scenario, because it is the number an operator recognizes.

        Args:
            g: ``(n_scenarios,)`` values of the constrained output.

        Returns:
            A fraction in ``[0, 1]``.
        """
        return float(np.mean(np.asarray(self.signed(jnp.asarray(g))) > 0.0))


@dataclass(frozen=True)
class Expected(StochasticConstraint):
    """Hold the constraint *on average* --- the weakest useful form.

    Appropriate for a quantity that is genuinely pooled downstream: a month of
    product blended into one shipment meets spec on the month's average.  It
    is the wrong form for anything settled per batch, where half the batches
    on the wrong side of the line is the literal meaning of "satisfied on
    average".
    """

    n_aux = 0

    def residual_from_signed(self, r, w, aux):
        return jnp.sum(w * r)

    def describe(self) -> str:
        return f"E[{self.name}] {self.op} {self.bound:g}"


@dataclass(frozen=True)
class Chance(StochasticConstraint):
    """Hold the constraint with probability at least ``alpha``.

    Enforced through its CVaR surrogate, ``CVaR_alpha(residual) <= 0``, which
    is the standard convex *conservative* approximation (Nemirovski and
    Shapiro): satisfying it implies ``P(satisfied) >= alpha``, but not
    conversely, so a design this constraint rejects may still be acceptable.
    The exact empirical probability is reported next to it in
    :meth:`~difflow.stochastic.saa.SAAResult.summary`, so the gap between the
    two is visible rather than assumed away.

    The alternative --- counting violations with an indicator --- is exact and
    non-convex, has zero gradient almost everywhere, and turns the problem
    into a MINLP.  That is deliberately not implemented.

    Attributes:
        alpha: Required probability of satisfaction, in ``(0, 1)``.
    """

    alpha: float = 0.95
    n_aux = 1

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"Chance alpha must lie strictly in (0, 1), got {self.alpha}."
            )
        if self.op == "==":
            raise ValueError(
                "A chance constraint on an equality is satisfied with "
                "probability zero for any continuous output. Pose it as a "
                "two-sided pair of inequalities, or use Expected."
            )

    def _cvar(self) -> CVaR:
        return CVaR(self.alpha)

    def exact_aux_from_signed(self, r, w):
        return self._cvar().exact_aux(r, w)

    def residual_from_signed(self, r, w, aux):
        return self._cvar().value(r, w, aux)

    def surrogate_from_signed(self, r, w, aux, tau):
        return self._cvar().surrogate(r, w, aux, tau)

    def describe(self) -> str:
        return (f"P({self.name} {self.op} {self.bound:g}) >= {self.alpha:g} "
                f"[CVaR surrogate]")


@dataclass(frozen=True)
class Robust(StochasticConstraint):
    """Hold the constraint in every scenario drawn.

    The sampled counterpart of a hard specification.  Same caveat as
    :class:`WorstCase`: this binds on the worst of ``S`` draws, which is not
    the worst of the distribution.  For a guarantee over a whole set, use
    :func:`difflow.flexibility.feasibility_function`, which searches the set
    rather than sampling it.
    """

    n_aux = 0

    def residual_from_signed(self, r, w, aux):
        return jnp.max(r)

    def surrogate_from_signed(self, r, w, aux, tau):
        scale = jnp.maximum(jnp.max(jnp.abs(r)), 1.0)
        return smooth_max(r, jnp.maximum(tau, _TAU_FLOOR) * scale)

    def describe(self) -> str:
        return f"{self.name} {self.op} {self.bound:g} in every scenario"


#: Constraint kinds the tuple shorthand understands.
CONSTRAINT_KINDS = {
    "expected": Expected,
    "chance": Chance,
    "robust": Robust,
}


def as_constraint(obj) -> StochasticConstraint:
    """Coerce a shorthand into a :class:`StochasticConstraint`.

    Args:
        obj: A :class:`StochasticConstraint`, or a tuple
            ``(name, op, bound)`` --- an :class:`Expected` constraint --- or
            ``(name, op, bound, alpha)``, a :class:`Chance` constraint at that
            probability.

    Returns:
        The constraint.

    Raises:
        ValueError: If the tuple has the wrong length or an unknown operator.

    Example:
        >>> as_constraint(("purity", ">=", 0.99, 0.95)).describe()
        'P(purity >= 0.99) >= 0.95 [CVaR surrogate]'
    """
    if isinstance(obj, StochasticConstraint):
        return obj
    try:
        parts = tuple(obj)
    except TypeError:
        raise ValueError(
            f"Cannot read {obj!r} as a constraint. Give a "
            f"StochasticConstraint, or a (name, op, bound[, alpha]) tuple."
        ) from None
    if len(parts) == 3:
        return Expected(str(parts[0]), str(parts[1]), float(parts[2]))
    if len(parts) == 4:
        return Chance(str(parts[0]), str(parts[1]), float(parts[2]),
                      alpha=float(parts[3]))
    raise ValueError(
        f"A constraint tuple is (name, op, bound) for an expectation "
        f"constraint or (name, op, bound, alpha) for a chance constraint; got "
        f"{len(parts)} entries: {obj!r}."
    )
