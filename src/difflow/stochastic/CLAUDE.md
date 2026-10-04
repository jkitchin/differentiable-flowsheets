# difflow.stochastic

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Stochastic Programming (`difflow.stochastic`)

Design under uncertainty: the sample average approximation of a two-stage
stochastic program, over any pure JAX `model(x, u, theta) -> {name: value}`.
Like `difflow.planning`, a module alongside `flexibility/`, **not** a
`difflow.plugins` entry point.

```python
import difflow.stochastic as st
scen = st.ScenarioSet.from_covariance(["a_Nd", "a_Dy"], mean, Sigma, n=256, seed=0)
prob = st.TwoStageProblem(model=circuit,
                          first_stage={"n_stages": (2., 16.)},   # here and now
                          recourse={"pH": (0.1, 1.2)},           # wait and see
                          objective="profit", maximize=True, risk=("cvar", 0.9),
                          constraints=[("purity", ">=", 0.93, 0.90)])
res = st.solve_saa(prob, scen)
st.bounds(prob, scen, result=res)          # VSS and EVPI
st.check_scenario_health(prob, scen, res)  # dead levers, empty tails, saturation
```

Invariants encoded in the module (do not weaken them):
- The scenario sample is drawn ONCE and reused. There is no
  resample-per-iteration option: a resampled SAA objective is not a function,
  and a descent method on it stalls at a distance set by the noise.
  `optimality_gap` is the only thing that redraws, and it reports a bound
  rather than a better point.
- Non-anticipativity is STRUCTURAL — one first-stage array shared by every
  scenario — never a constraint that could be written down wrongly.
- The Rockafellar-Uryasev auxiliary is NOT a decision variable. It is
  recomputed at its closed-form optimum every iterate and frozen with
  `stop_gradient`; that is exact by the envelope theorem. Handing it to a
  projected method whose step scales with the box width makes the auxiliary
  swing harder than the design, and the run ends at a bound looking converged.
- CVaR's smoothed positive part is `t * logaddexp(z/t, 0)`, never the branchless
  `max(u,0) + log1p(exp(-|u|))`. The two have the same VALUE everywhere and the
  second has a derivative of exactly ZERO at `u = 0` — `max`'s JVP breaks the
  tie toward the constant and `abs`'s is 0, so the halves of the kink cancel
  instead of averaging to the sigmoid's 0.5. `z - t` is exactly zero for the
  scenario at the quantile, and with a SINGLE scenario — which is what
  `expected_value_solution`, and hence every VSS, solves — for the only scenario
  there is: the whole gradient vanishes and the design never leaves its
  initialization while reporting convergence.
  `tests/test_stochastic.py::TestRisk::test_a_one_scenario_cvar_solve_would_have_frozen_at_its_start`
  is the regression.
- `WS <= SP <= EEV` holds among ADMISSIBLE designs only. A mean-value design
  that misses the chance constraint scores better than the stochastic one for
  exactly the reason it is not allowed; `BoundReport.ev_feasible` says so and
  `summary()` prints "not a value" rather than a negative VSS that blames the
  solver for a modelling fact. Without recourse this is the common case.
- The constraint handler is an AUGMENTED LAGRANGIAN, not a growing penalty. A
  penalty weight of 1e4 inflates Adam's second moment by eight decades and
  every later step shrinks to nothing — the iterate freezes and reports itself
  feasible. `tests/test_stochastic.py::TestSolve::test_a_growing_penalty_would_have_frozen_here`
  is the regression.
- Objective and constraint residuals are rescaled, and the rescaling is EXACT:
  every risk measure is translation-equivariant and positively homogeneous, and
  every constraint form is positively homogeneous in its residual.
- Feasibility is scored by re-evaluating the model, never read off the
  multipliers — the same rule `difflow.planning` states for LP slacks.
- `wait_and_see` carries the constraints across UNCHANGED. That is what makes
  it a bound (drop non-anticipativity, change nothing else). Rewriting a chance
  constraint per-scenario can make the relaxation infeasible where the original
  was fine, and then EVPI comes out negative.
- `n_aux` on a risk measure is a plain class attribute, never an annotated
  dataclass field: as a base-class field it takes the first positional slot and
  `CVaR(0.9)` silently sets the auxiliary count instead of alpha.
- Out of scope by design: L-shaped/Benders decomposition, scenario reduction,
  multistage trees, integer recourse, distributionally robust formulations.
  Do not add them.

Where the uncertain parameters come from: `difflow.estimation.predicted_covariance`
and `difflow.reconciliation.reconciled_covariance` both return the `(mean, Sigma)`
that `ScenarioSet.from_covariance` wants. Prefer it over the per-parameter
constructors — correlated coefficients have a *difference* variance that a
diagonal Sigma gets wrong by a factor of a few.

Related, and cheaper — try these first: `difflow.uncertainty` (propagate a
distribution through a fixed design), `difflow.planning.backoff` (`kappa*sigma`
margin from one Jacobian), `difflow.flexibility.expected_feasibility` (does a
design I already have meet spec often enough?), `difflow.flexibility` proper (a
guarantee over an envelope, by vertex search rather than sampling).

Docs: `docs/stochastic.md`. Example: `examples/32_stochastic_ree_separation.ipynb`.
Tests: `tests/test_stochastic.py`, `tests/ree/test_coefficient_overrides.py`.
