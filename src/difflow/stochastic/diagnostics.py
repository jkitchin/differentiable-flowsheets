"""Was the stochastic program worth solving, and is its answer trustworthy?

A stochastic program costs perhaps two orders of magnitude more than the
deterministic one it replaces.  Two classical numbers say whether that was
money well spent, and they answer different questions.

**The value of the stochastic solution** answers *should I have bothered?*
Solve the deterministic problem at the mean parameters, take the design it
produced, and run it --- with recourse --- against the real distribution.  The
gap between that and the stochastic optimum is what modelling the uncertainty
bought.

.. math:: \\mathrm{VSS} = \\mathrm{EEV} - \\mathrm{SP}

A VSS of zero is a completely legitimate answer, and finding it early is the
point: it says the mean-value design was already right and the plant should
stop paying for scenario runs.

**The expected value of perfect information** answers *should I buy an
instrument instead?*  It compares the stochastic optimum against the
unattainable plan that knows each realization in advance.

.. math:: \\mathrm{EVPI} = \\mathrm{SP} - \\mathrm{WS}

EVPI is an upper bound on what any measurement, assay or online analyzer can
possibly be worth --- if EVPI is $40k/yr, a $200k analyzer cannot pay back no
matter how good it is.  That is usually the most decision-relevant number in
the whole run.

Both obey ``WS <= SP <= EEV`` for a minimization.  If a computed set violates
that ordering, a solve did not converge, and the functions here say so rather
than reporting a negative EVPI as though it meant something.

Then: is the *sample* big enough?
---------------------------------
:func:`check_scenario_health` is the counterpart of
:func:`difflow.planning.health.check_delta_health` --- the things that go
wrong quietly as a problem grows, each with a number attached:

* **Dead levers.**  A first-stage variable whose SAA gradient is exactly zero
  is one the optimizer cannot move, usually because a ``clip``, ``where`` or
  ``minimum`` on an active path has flattened it.  Identical failure to the
  delta-vector one, identical cause.
* **A tail with nothing in it.**  ``CVaR(0.99)`` on 200 scenarios averages
  two of them.  The reported number will be confident and meaningless.
* **Saturated recourse.**  A recourse variable pinned at a bound in most
  scenarios is not providing recourse --- the credit the model took for it is
  not there.
* **Sample sensitivity.**  The risk value on half the sample against the whole
  sample.  Cheap, and a large split means ``S`` is too small before any
  formal bound is needed.

And :func:`optimality_gap` is the formal version: the Mak--Morton--Wood
replication bound, which is the honest way to state how far an SAA answer
might be from the true optimum.

Reference:
    Birge, Oper. Res. 30 (1982) 989, doi:10.1287/opre.30.5.989 (EVPI, VSS).
    Mak, Morton and Wood, Oper. Res. Lett. 24 (1999) 47,
    doi:10.1016/S0167-6377(98)00054-6 (the replication gap bound).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import numpy as np

from difflow.flexibility.sets import ControlSpec
from difflow.stochastic.problem import TwoStageProblem
from difflow.stochastic.saa import (
    DEFAULT_SAA_OPTIONS, SAAOptions, SAAResult, evaluate, solve_recourse,
    solve_saa,
)
from difflow.stochastic.scenarios import ScenarioSet

#: Below this many scenarios in a CVaR tail, the estimate is noise.
MIN_TAIL_SCENARIOS = 20

#: Above this fraction of scenarios at a bound, a recourse variable is not
#: really providing recourse.
SATURATION_WARN = 0.5


# =============================================================================
# The classical bounds
# =============================================================================


def wait_and_see(problem: TwoStageProblem, scenarios: ScenarioSet, *,
                 options: SAAOptions = DEFAULT_SAA_OPTIONS) -> SAAResult:
    """The plan that knows the future: every decision made per scenario.

    Implemented by moving the first stage into the recourse, so it is one
    batched solve rather than ``S`` separate ones, and it is exact for the risk
    measures here: all of them are monotone in each scenario's objective, so
    minimizing every scenario independently minimizes the risk of the vector.

    The constraints are carried across **unchanged**, which is the point that
    makes this a bound at all.  Dropping non-anticipativity and changing
    nothing else makes the two-stage problem's feasible set a subset of this
    one --- a shared first stage is a per-scenario first stage that happens to
    take the same value --- so ``WS`` cannot be worse than ``SP``, whatever the
    solver does.  Rewriting a chance constraint as a per-scenario one instead
    would break that: it can make the relaxation *infeasible* where the
    original was fine, since a chance constraint is precisely the statement
    that some scenarios are allowed to fail.

    This is not an implementable plan.  It is the bound that says what perfect
    information would be worth.

    Args:
        problem: The problem.
        scenarios: The fixed sample.
        options: Search settings.

    Returns:
        An :class:`SAAResult` whose "first stage" is the mean over scenarios
        of the per-scenario optimum, reported only so the object is complete;
        the number that matters is
        :attr:`~difflow.stochastic.saa.SAAResult.risk_value`.
    """
    fs, rc = problem.first_stage, problem.recourse
    free = ControlSpec(
        lower=np.concatenate([np.asarray(fs.lower), np.asarray(rc.lower)]),
        upper=np.concatenate([np.asarray(fs.upper), np.asarray(rc.upper)]),
        names=tuple(fs.names) + tuple(rc.names),
    )
    anchor = float(0.5 * (float(fs.lower[0]) + float(fs.upper[0])))

    def model(x, u, theta):
        return problem.model({n: u[n] for n in fs.names},
                             {n: u[n] for n in rc.names}, theta)

    # One dummy first-stage variable, pinned: a TwoStageProblem must have a
    # first stage, and this one is not allowed to do anything.
    ws_problem = replace(
        problem, model=model,
        first_stage=ControlSpec(lower=[anchor], upper=[anchor],
                                names=("_ws_anchor",), start=[anchor]),
        recourse=free,
    )
    res = solve_saa(ws_problem, scenarios,
                    options=replace(options, n_starts=1))
    res.first_stage = {n: float(np.mean(res.recourse[n])) for n in fs.names}
    res.recourse = {n: res.recourse[n] for n in rc.names}
    res.problem = problem
    return res


def expected_value_solution(problem: TwoStageProblem, scenarios: ScenarioSet,
                            *, options: SAAOptions = DEFAULT_SAA_OPTIONS
                            ) -> SAAResult:
    """The deterministic design: solve at the mean parameters and stop.

    This is what ignoring the uncertainty produces, and it is the design the
    value of the stochastic solution is measured against.

    Args:
        problem: The problem.
        scenarios: The sample, used only for its mean.
        options: Search settings.

    Returns:
        An :class:`SAAResult` over the single mean scenario.
    """
    return solve_saa(problem, scenarios.mean_scenario(), options=options)


@dataclass
class BoundReport:
    """The three bounds, and what the gaps between them mean.

    Attributes:
        stochastic: ``SP``, the two-stage optimum.
        expected_value: ``EEV``, the mean-value design run under the real
            distribution with recourse re-optimized.
        wait_and_see: ``WS``, the unattainable perfect-information plan.
        vss: ``EEV - SP``, in the problem's own sense: what modelling the
            uncertainty was worth.
        evpi: ``SP - WS``: the ceiling on what any measurement can be worth.
        ordered: Whether ``WS <= SP <= EEV`` held (for a minimization).  False
            means a solve did not converge, and the gaps below are not
            meaningful.
        ev_first_stage: The mean-value design, for comparison with the
            stochastic one.
        sp_first_stage: The stochastic design.
    """

    stochastic: float
    expected_value: float
    wait_and_see: float
    vss: float
    evpi: float
    ordered: bool
    ev_first_stage: dict
    sp_first_stage: dict

    def summary(self) -> str:
        """The three bounds and the two gaps, with the designs side by side."""
        lines = [
            "Stochastic-programming bounds",
            f"  {'wait and see  (WS)':<34s}{self.wait_and_see:14.6g}",
            f"  {'stochastic    (SP)':<34s}{self.stochastic:14.6g}",
            f"  {'mean-value design (EEV)':<34s}{self.expected_value:14.6g}",
            f"  {'value of the stochastic solution':<34s}{self.vss:14.6g}",
            f"  {'expected value of perfect info':<34s}{self.evpi:14.6g}",
        ]
        if not self.ordered:
            lines.append("  ! WS <= SP <= EEV does not hold: at least one "
                         "solve did not converge, so the gaps above are not "
                         "meaningful. Raise SAAOptions.steps or n_starts.")
        lines.append(f"  {'design':<22s}{'mean-value':>14s}{'stochastic':>14s}")
        for n in self.sp_first_stage:
            lines.append(f"  {n:<22s}{self.ev_first_stage.get(n, float('nan')):14.6g}"
                         f"{self.sp_first_stage[n]:14.6g}")
        return "\n".join(lines)


def bounds(problem: TwoStageProblem, scenarios: ScenarioSet, *,
           result: SAAResult | None = None,
           options: SAAOptions = DEFAULT_SAA_OPTIONS) -> BoundReport:
    """Compute ``WS``, ``SP`` and ``EEV``, and the two gaps between them.

    Three solves (or two, if you pass one in).  On anything but a trivial
    model this is the expensive call in the module, and it is the one worth
    paying for: it is what tells you whether to keep running stochastic
    programs or to buy an analyzer instead.

    Args:
        problem: The problem.
        scenarios: The fixed sample.
        result: An already-solved stochastic result, to avoid re-solving.
        options: Search settings.

    Returns:
        A :class:`BoundReport`.

    Example:
        >>> import difflow.stochastic as st
        >>> def model(x, u, theta):
        ...     return {"cost": (x["size"] - theta["load"]) ** 2}
        >>> prob = st.TwoStageProblem(model=model,
        ...                           first_stage={"size": (0.0, 10.0)},
        ...                           objective="cost")
        >>> scen = st.ScenarioSet.normal({"load": (5.0, 1.0)}, n=32, seed=0)
        >>> rep = st.bounds(prob, scen, options=st.SAAOptions(steps=200))
        >>> rep.vss >= -1e-6 and rep.evpi >= -1e-6
        True
    """
    sp = result if result is not None else solve_saa(problem, scenarios,
                                                    options=options)
    ev = expected_value_solution(problem, scenarios, options=options)
    eev = solve_recourse(problem, scenarios, ev.first_stage, options=options)
    ws = wait_and_see(problem, scenarios, options=options)

    sense = problem.sense                       # +1 minimize, -1 maximize
    sp_v, eev_v, ws_v = sp.risk_value, eev.risk_value, ws.risk_value
    tol = 1e-8 * max(1.0, abs(sp_v))
    ordered = bool(sense * ws_v <= sense * sp_v + tol
                   and sense * sp_v <= sense * eev_v + tol)
    return BoundReport(
        stochastic=sp_v, expected_value=eev_v, wait_and_see=ws_v,
        vss=sense * (eev_v - sp_v), evpi=sense * (sp_v - ws_v),
        ordered=ordered, ev_first_stage=dict(ev.first_stage),
        sp_first_stage=dict(sp.first_stage),
    )


def value_of_stochastic_solution(problem, scenarios, *, result=None,
                                 options: SAAOptions = DEFAULT_SAA_OPTIONS
                                 ) -> float:
    """``EEV - SP``: what modelling the uncertainty was worth.

    Args:
        problem: The problem.
        scenarios: The fixed sample.
        result: An already-solved stochastic result.
        options: Search settings.

    Returns:
        A non-negative number, in the problem's own sense.  Zero means the
        mean-value design was already optimal.
    """
    return bounds(problem, scenarios, result=result, options=options).vss


def expected_value_of_perfect_information(problem, scenarios, *, result=None,
                                          options: SAAOptions
                                          = DEFAULT_SAA_OPTIONS) -> float:
    """``SP - WS``: the ceiling on what any measurement can be worth.

    Args:
        problem: The problem.
        scenarios: The fixed sample.
        result: An already-solved stochastic result.
        options: Search settings.

    Returns:
        A non-negative number, in the problem's own sense.
    """
    return bounds(problem, scenarios, result=result, options=options).evpi


# =============================================================================
# Is the sample big enough?
# =============================================================================


@dataclass
class HealthReport:
    """What is quietly wrong with a stochastic run.

    Attributes:
        n_scenarios: Sample size.
        dead_levers: First-stage variables whose SAA gradient is exactly zero.
        gradients: ``{name: dRisk/dx}`` at the solution.
        tail_scenarios: Scenarios in the CVaR tail, or ``None`` if the risk
            measure has no tail.
        saturated: ``{name: fraction at a bound}`` for recourse variables at a
            bound in more than :data:`SATURATION_WARN` of scenarios.
        half_sample_risk: The risk value recomputed on the first half of the
            sample, at the same design.
        risk_value: The risk value on the whole sample.
        standard_error: Bootstrap standard error of the risk value at the
            solution --- the sampling noise, not the optimality gap.
        warnings: Human-readable findings, most important first.
    """

    n_scenarios: int
    dead_levers: list[str]
    gradients: dict[str, float]
    tail_scenarios: float | None
    saturated: dict[str, float]
    half_sample_risk: float
    risk_value: float
    standard_error: float
    warnings: list[str]

    @property
    def healthy(self) -> bool:
        """True when nothing was flagged."""
        return not self.warnings

    def summary(self) -> str:
        """The findings, with the numbers behind each."""
        lines = [f"Scenario health: {self.n_scenarios} scenarios",
                 f"  risk value        {self.risk_value:14.6g}"
                 f"  +/- {self.standard_error:.3g} (sampling)",
                 f"  half-sample risk  {self.half_sample_risk:14.6g}"]
        if self.tail_scenarios is not None:
            lines.append(f"  CVaR tail         {self.tail_scenarios:14.6g}"
                         f"  scenarios")
        lines.append(f"  {'first-stage lever':<24s}{'dRisk/dx':>14s}")
        for n, g in self.gradients.items():
            mark = "  dead" if n in self.dead_levers else ""
            lines.append(f"  {n:<24s}{g:14.6g}{mark}")
        if self.warnings:
            lines.append("  findings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        else:
            lines.append("  findings: none")
        return "\n".join(lines)


def check_scenario_health(problem: TwoStageProblem, scenarios: ScenarioSet,
                          result: SAAResult, *,
                          n_bootstrap: int = 200,
                          seed: int = 0) -> HealthReport:
    """The things that go wrong quietly, each with a number attached.

    Never raises and never re-solves: it examines the solution you already
    have.  Run it on every stochastic result, the way
    :func:`difflow.planning.health.check_delta_health` is run on every plan.

    Args:
        problem: The problem.
        scenarios: The sample it was solved against.
        result: The solved result.
        n_bootstrap: Resamples used for the standard error of the risk value.
        seed: PRNG seed for the bootstrap.

    Returns:
        A :class:`HealthReport`.
    """
    w = scenarios.weights
    x0 = jnp.asarray([result.first_stage[n]
                      for n in problem.first_stage.names])
    U = jnp.stack([jnp.asarray(result.recourse[n])
                   for n in problem.recourse.names], axis=1) \
        if problem.n_recourse else jnp.zeros((scenarios.n_scenarios, 0))

    def risk_of(x):
        out = problem.outputs(x, U, scenarios)
        z = problem.objective_values(out)
        a = jax.lax.stop_gradient(problem.risk.exact_aux(z, w))
        return problem.risk.value(z, w, a)

    grad = np.asarray(jax.grad(risk_of)(x0))
    gradients = {n: float(problem.sense * grad[i])
                 for i, n in enumerate(problem.first_stage.names)}
    dead = [n for n, g in gradients.items() if g == 0.0]

    # Bootstrap the risk value at the fixed design: sampling noise only.
    z = problem.objective_values(problem.outputs(x0, U, scenarios))
    key = jax.random.PRNGKey(int(seed))
    S = scenarios.n_scenarios
    idx = jax.random.randint(key, (int(n_bootstrap), S), 0, S)
    wb = jnp.full((S,), 1.0 / S)

    def one(i):
        zz = z[i]
        return problem.risk.value(zz, wb,
                                  jax.lax.stop_gradient(
                                      problem.risk.exact_aux(zz, wb)))

    se = float(jnp.std(jax.vmap(one)(idx)))

    half = scenarios.subset(np.arange(max(S // 2, 1)))
    half_recourse = {n: np.asarray(result.recourse[n])[:half.n_scenarios]
                     for n in problem.recourse.names} or None
    half_risk = float(evaluate(problem, result.first_stage, half_recourse,
                               half).risk_value)

    tail = result.risk_report.get("tail_scenarios")
    sat = {n: v for n, v in result.recourse_saturation().items()
           if v > SATURATION_WARN}

    warnings = []
    if dead:
        warnings.append(
            f"dead levers {dead}: the SAA gradient is exactly zero, so the "
            f"optimizer cannot move them. Look for a clip / where / minimum "
            f"on an active path -- the same failure "
            f"difflow.planning.health.check_delta_health reports for delta "
            f"columns."
        )
    if tail is not None and tail < MIN_TAIL_SCENARIOS:
        warnings.append(
            f"the CVaR tail holds about {tail:.1f} scenarios; below "
            f"{MIN_TAIL_SCENARIOS} the tail estimate is noise. Draw more "
            f"scenarios or lower alpha."
        )
    if sat:
        pinned = ", ".join(f"{k} ({v:.0%})" for k, v in sat.items())
        warnings.append(
            f"recourse saturated: {pinned} of scenarios sit at a bound, so "
            f"the model is not getting the recourse it is being credited "
            f"with. Widen the box, or accept that the variable is effectively "
            f"first stage."
        )
    if se > 0.0 and abs(result.risk_value) > 0.0 and \
            se / max(abs(result.risk_value), 1e-30) > 0.05:
        warnings.append(
            f"sampling standard error is {se:.4g}, "
            f"{se / abs(result.risk_value):.1%} of the risk value; differences "
            f"smaller than this between two designs are not real."
        )
    if abs(half_risk - result.risk_value) > 2.0 * se + 1e-12:
        warnings.append(
            f"the risk value moves from {half_risk:.6g} on half the sample to "
            f"{result.risk_value:.6g} on all of it, further than the "
            f"bootstrap error accounts for; the sample is probably too small."
        )
    return HealthReport(
        n_scenarios=S, dead_levers=dead, gradients=gradients,
        tail_scenarios=tail, saturated=sat, half_sample_risk=half_risk,
        risk_value=result.risk_value, standard_error=se, warnings=warnings,
    )


@dataclass
class GapReport:
    """A confidence bound on how far the SAA answer is from the true optimum.

    Attributes:
        candidate_value: The candidate design's risk on a large independent
            sample.  An unbiased estimate of what this design is actually
            worth, and therefore a bound on the true optimum from the
            achievable side.
        replication_bound: The mean of the replication optima.  Each
            replication optimizes against its own sampling noise, so this is
            biased on the *optimistic* side of the true optimum --- below it
            when minimizing, above it when maximizing --- and that bias is
            exactly what makes it a bound.
        gap: The distance between the two, in the problem's own sense, so it
            is non-negative whichever way the problem points.
        gap_stderr: Standard error of the gap.
        confidence: One-sided confidence level of :attr:`gap_upper`.
        gap_upper: The one-sided confidence limit on the gap.
        replications: Each replication's optimum.
        n_evaluation: Size of the independent evaluation sample.
    """

    candidate_value: float
    replication_bound: float
    gap: float
    gap_stderr: float
    confidence: float
    gap_upper: float
    replications: np.ndarray
    n_evaluation: int

    def summary(self) -> str:
        """The bound, stated as a sentence anyone can act on."""
        return "\n".join([
            f"SAA optimality gap ({len(self.replications)} replications, "
            f"{self.n_evaluation}-scenario evaluation)",
            f"  candidate value        {self.candidate_value:14.6g}",
            f"  replication bound      {self.replication_bound:14.6g}",
            f"  gap                    {self.gap:14.6g}"
            f"  +/- {self.gap_stderr:.3g}",
            f"  {self.confidence:.0%} one-sided limit    "
            f"{self.gap_upper:14.6g}",
            f"  i.e. the design is within {abs(self.gap_upper):.6g} of the "
            f"true optimum with {self.confidence:.0%} confidence.",
        ])


def optimality_gap(problem: TwoStageProblem, scenarios: ScenarioSet,
                   result: SAAResult, *, n_replications: int = 5,
                   n_evaluation: int = 2000, seed: int = 10_000,
                   confidence: float = 0.95,
                   options: SAAOptions = DEFAULT_SAA_OPTIONS) -> GapReport:
    """The Mak--Morton--Wood bound: how far off can this answer be?

    The one place in this module that deliberately redraws the sample.  Solve
    the SAA problem on ``n_replications`` *independent* samples; because each
    optimizes against its own noise, the average of those optima lands on the
    optimistic side of the true optimum, which is what makes it a bound.
    Evaluate the candidate design on one large independent sample for an
    unbiased estimate from the achievable side.  The distance between them,
    with the replication standard error, is a confidence bound on the
    optimality gap.

    Args:
        problem: The problem.
        scenarios: The sample the candidate was solved against.  Must have
            come from a distribution that can be redrawn --- see
            :meth:`~difflow.stochastic.scenarios.ScenarioSet.redraw`.
        result: The candidate solution.
        n_replications: Independent samples to solve.  Each costs a full
            solve; 5 to 10 is usual.
        n_evaluation: Size of the independent evaluation sample.
        seed: Base seed for the independent draws.  Offset well away from the
            solving seed so no replication reuses it.
        confidence: One-sided confidence level.
        options: Search settings for the replication solves.

    Returns:
        A :class:`GapReport`.

    Raises:
        ValueError: If ``scenarios`` cannot be redrawn, or fewer than two
            replications are requested (the bound needs a variance).
    """
    if int(n_replications) < 2:
        raise ValueError(
            "The gap bound needs at least two replications to have a standard "
            "error; one replication gives a point with no bound on it."
        )
    reps = []
    for m in range(int(n_replications)):
        sample = scenarios.redraw(int(seed) + m)
        reps.append(solve_saa(problem, sample, options=options).risk_value)
    reps = np.asarray(reps, dtype=float)

    big = _resize(scenarios, int(seed) + int(n_replications) + 1,
                  int(n_evaluation))
    cand = solve_recourse(problem, big, result.first_stage, options=options) \
        if problem.n_recourse else evaluate(problem, result.first_stage, None,
                                            big)

    sense = problem.sense
    lower = float(np.mean(reps))
    gap = sense * (cand.risk_value - lower)
    se = float(np.std(reps, ddof=1) / np.sqrt(reps.size))
    try:
        from scipy.stats import t as _t
        crit = float(_t.ppf(confidence, reps.size - 1))
    except Exception:                            # scipy is optional here
        crit = 1.645 if confidence >= 0.95 else 1.282
    return GapReport(
        candidate_value=cand.risk_value, replication_bound=lower, gap=gap,
        gap_stderr=se, confidence=float(confidence),
        gap_upper=gap + crit * se, replications=reps,
        n_evaluation=big.n_scenarios,
    )


def _resize(scenarios: ScenarioSet, seed: int, n: int) -> ScenarioSet:
    """An independent draw of a different size from the same distribution."""
    spec = getattr(scenarios, "_spec", None)
    if spec is None:
        raise ValueError(
            "optimality_gap needs a ScenarioSet it can redraw from; this one "
            "was supplied directly. Rebuild it with ScenarioSet.normal / "
            ".lognormal / .uniform / .from_covariance."
        )
    kind, payload = spec
    kwargs = dict(payload["kwargs"])
    kwargs["n"] = int(n)
    return getattr(ScenarioSet, kind)(*payload["args"], seed=seed, **kwargs)
