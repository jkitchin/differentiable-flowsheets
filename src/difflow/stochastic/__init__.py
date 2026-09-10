"""Design under uncertainty: sample average approximation over a difflow model.

A design that is optimal at the nominal parameters is optimal at exactly one
point of a distribution nobody knows.  This module replaces that point with a
sample and optimizes the resulting histogram --- the *sample average
approximation* of a two-stage stochastic program --- with the model, the
gradients and the scenario batching all coming from the same JAX code
difflow already runs.

The four objects
----------------
:class:`~difflow.stochastic.scenarios.ScenarioSet`
    The sample.  Built from a posterior, a fitted covariance, or a nominal
    value and a spread; drawn once and reused, because a resampled objective
    is not a function and a descent method on it stalls.
:class:`~difflow.stochastic.problem.TwoStageProblem`
    The split between what is decided *here and now* and what is decided
    *wait and see*.  Non-anticipativity is structural: there is one
    first-stage array, shared by every scenario, so it cannot be written down
    wrongly.
:mod:`~difflow.stochastic.risk`
    What "best" means once the answer is a histogram --- expectation, CVaR,
    mean plus a premium, worst case --- and the constraint forms that go with
    them: on average, with a probability, or in every scenario.
:func:`~difflow.stochastic.saa.solve_saa`
    Projected Adam under an augmented Lagrangian, over ``(x, U)``.  Feasibility
    is always scored by re-evaluating the model, never read off the
    multipliers.

Quick start
-----------
::

    import difflow.stochastic as st

    scen = st.ScenarioSet.from_covariance(
        ["logD_Nd", "logD_Dy"], mean, Sigma, n=256, seed=0)

    prob = st.TwoStageProblem(
        model=flowsheet_fn,                    # (x, u, theta) -> {name: value}
        first_stage={"n_stages": (4.0, 14.0)}, # decided once
        recourse={"pH": (1.5, 4.5)},           # re-decided per campaign
        objective="profit", maximize=True,
        risk=("cvar", 0.9),
        constraints=[("purity", ">=", 0.995, 0.95)],
    )
    res = st.solve_saa(prob, scen)
    print(res.summary())
    print(st.bounds(prob, scen, result=res).summary())     # VSS and EVPI
    print(st.check_scenario_health(prob, scen, res).summary())

Why this works on a flowsheet at all
------------------------------------
:meth:`difflow.Flowsheet.solve` detects a tracer and swaps its Anderson and
Wegstein loops --- Python loops that branch on the residual, and so cannot be
traced --- for an optimistix fixed point carrying an implicit-differentiation
rule.  So a recycle solve ``vmap``s over scenarios and differentiates through
to the design, and the gradient comes from the converged solution rather than
from an unrolled iteration.  That is the whole reason the scenario dimension
is free here and expensive everywhere else.

What to reach for instead
-------------------------
:mod:`difflow.flexibility`
    A *guarantee* over an uncertainty set, not a probability over a sample:
    it searches vertices rather than hoping a draw found the corner.  Its
    :func:`~difflow.flexibility.expected_feasibility` is the cheap first look
    --- does a design I already have meet its specs often enough? --- and
    needs no optimization at all.
:mod:`difflow.planning.backoff`
    The linear-model answer: propagate a parameter covariance onto the
    constraint values and buy ``kappa * sigma`` of margin.  Far cheaper, and
    right whenever the constraint is nearly linear in the parameters over
    their spread.
:mod:`difflow.uncertainty`
    Just propagating a distribution through a fixed design, with no
    optimization anywhere.

Out of scope by design
----------------------
No L-shaped or Benders decomposition, no scenario reduction, no multistage
scenario trees, no integer recourse, no distributionally robust formulations.
Scenarios are handled by ``vmap`` --- the right answer until the batch stops
fitting in memory, and :func:`check_scenario_health` reports where you are.
Do not add them here; a decomposition belongs with a solver that can exploit
it, and :mod:`difflow.solvers.discopt_bridge` explains why an implicit
flowsheet block cannot be handed to an integer solver and still carry a
certificate.

Docs: ``docs/stochastic.md``.  Example:
``examples/32_stochastic_ree_separation.ipynb``.
"""

from difflow.stochastic.diagnostics import (
    MIN_TAIL_SCENARIOS, SATURATION_WARN, BoundReport, GapReport, HealthReport,
    bounds, check_scenario_health, expected_value_of_perfect_information,
    expected_value_solution, optimality_gap, value_of_stochastic_solution,
    wait_and_see,
)
from difflow.stochastic.problem import TwoStageProblem
from difflow.stochastic.risk import (
    CONSTRAINT_KINDS, OPERATORS, RISK_MEASURES, CVaR, Chance, Expectation,
    Expected, MeanStd, RiskMeasure, Robust, StochasticConstraint, WorstCase,
    as_constraint, as_risk_measure, weighted_quantile,
)
from difflow.stochastic.saa import (
    DEFAULT_SAA_OPTIONS, FEASIBILITY_TOL, SAAOptions, SAAResult, evaluate,
    solve_recourse, solve_saa,
)
from difflow.stochastic.scenarios import DISTRIBUTIONS, ScenarioSet

__all__ = [
    # The sample
    "ScenarioSet",
    "DISTRIBUTIONS",
    # The problem
    "TwoStageProblem",
    # Risk measures
    "RiskMeasure",
    "Expectation",
    "MeanStd",
    "CVaR",
    "WorstCase",
    "as_risk_measure",
    "RISK_MEASURES",
    "weighted_quantile",
    # Constraints
    "StochasticConstraint",
    "Expected",
    "Chance",
    "Robust",
    "as_constraint",
    "CONSTRAINT_KINDS",
    "OPERATORS",
    # The solve
    "solve_saa",
    "solve_recourse",
    "evaluate",
    "SAAOptions",
    "SAAResult",
    "DEFAULT_SAA_OPTIONS",
    "FEASIBILITY_TOL",
    # Diagnostics
    "bounds",
    "BoundReport",
    "wait_and_see",
    "expected_value_solution",
    "value_of_stochastic_solution",
    "expected_value_of_perfect_information",
    "check_scenario_health",
    "HealthReport",
    "optimality_gap",
    "GapReport",
    "MIN_TAIL_SCENARIOS",
    "SATURATION_WARN",
]
