# Moving-Horizon Estimation

`difflow.dynamic` integrates a DAE flowsheet forward and `difflow.reconciliation`
reconciles a steady-state data set. Between them sits the question a running
plant actually asks: *given a stream of noisy, sparse and late measurements,
where is the process now?*

`difflow.mhe` answers it. At each sampling time it solves

$$
\min_{x_0,\,w}\;
\underbrace{\|x_0 - \bar x\|^2_{P^{-1}}}_{\text{arrival cost}}
+ \sum_{k}\|w_k\|^2_{Q^{-1}}
+ \sum_{k}\|y_k - h(x_k)\|^2_{R^{-1}}
\quad\text{s.t.}\quad
x_{k+1} = f(x_k, u_k, w_k),\; x_k \in [\ell, u]
$$

over the last $K$ intervals, with an *arrival cost* summarising everything
before them. The reference is Rao, Rawlings and Mayne, *IEEE Trans. Automat.
Contr.* **48** (2003) 246, [doi:10.1109/TAC.2002.808470](https://doi.org/10.1109/TAC.2002.808470).

## Table of contents

1. [Why an optimisation and not a filter](#why-an-optimisation-and-not-a-filter)
2. [Quick start](#quick-start)
3. [The model an estimator sees](#the-model-an-estimator-sees)
4. [Measurements: multi-rate, and late](#measurements-multi-rate-and-late)
5. [The extended Kalman filter](#the-extended-kalman-filter)
6. [The arrival cost](#the-arrival-cost)
7. [Constraints](#constraints)
8. [Joint state and parameter estimation](#joint-state-and-parameter-estimation)
9. [Observability: can this window see that state?](#observability-can-this-window-see-that-state)
10. [Consistency testing](#consistency-testing)
11. [Sliding a horizon over a campaign](#sliding-a-horizon-over-a-campaign)
12. [`jit`, `grad`, and feeding a planner](#jit-grad-and-feeding-a-planner)
13. [What is reused rather than duplicated](#what-is-reused-rather-than-duplicated)
14. [Tuning, and what goes wrong](#tuning-and-what-goes-wrong)
15. [API summary](#api-summary)

---

## Why an optimisation and not a filter

The case that motivates this is slow, high-order, sparsely measured and delayed.
In a solvent-extraction train the organic inventory turns over on the order of a
day, a feed disturbance takes many residence times to traverse thirty stages
with recycle, and the assay that would reveal what happened comes back hours
later on a shift cadence. Meanwhile the parameters drift, through extractant
degradation, entrainment and changing stage efficiency.

An extended Kalman filter linearises once, at the current mean, and then commits.
It cannot revisit a bad linearisation, it cannot express "this concentration is
not negative" — the update is a linear correction that will happily produce one —
and it has no notion of a measurement drawn six hours ago, because its state is a
single mean and covariance *at the current time*.

A horizon can do all three, at the cost of an optimisation per sample instead of
a linear solve. `difflow.mhe` ships the filter too (`run_ekf`), because "MHE was
better" is a claim only if the alternative was computed — and because the
filter's covariance recursion is what supplies the arrival cost.

## Quick start

```python
import jax.numpy as jnp
import numpy as np
from difflow.mhe import (ArrivalCost, MHEProblem, Measurement,
                         build_window, linear_model, run_ekf, solve_mhe)

# x+ = A x + w, only the first state is metered.
A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
C = jnp.array([[1.0, 0.0]])
model = linear_model(A, C, x_names=["hot", "cold"], y_names=["T_hot"])

readings = [0.98, 1.02, 0.81, 0.77, 0.60, 0.55, 0.44, 0.39]
records = [Measurement(time=float(k), values=[y], sigma=[0.05])
           for k, y in enumerate(readings)]
window, dropped = build_window(np.arange(8.0), records, y_names=["T_hot"])

problem = MHEProblem(
    model,
    ArrivalCost.diagonal(jnp.array([1.0, 0.0]), 0.5),
    process_std=jnp.array([0.02, 0.02]),
)
result = solve_mhe(problem, window)
print(result.summary())
print(result.x_named)          # {'hot': ..., 'cold': ...} -- cold is unmetered
```

Three objects do the work: a `StateSpaceModel`, a `MeasurementWindow` built from
timestamped `Measurement` records, and an `MHEProblem` that pairs the model with
an `ArrivalCost` and the process-noise level.

## The model an estimator sees

Both estimators read the plant through one object,

$$x_{k+1} = f(x_k, u_k, w_k, \theta),\qquad y_k = h(x_k, u_k, \theta),$$

so anything expressible as a pair of JAX-traceable callables can be estimated.

A dynamic flowsheet becomes that pair through `StateSpaceModel.from_ode`, which
discretises the right-hand side with `difflow.dynamic.integrate` over one
sampling interval. **This module owns no integrator**; every solver
`difflow.dynamic` offers, including the diffrax backends for stiff systems, is
available by name:

```python
from difflow.mhe import StateSpaceModel

model = StateSpaceModel.from_ode(
    lambda t, x, u, theta: -theta["k"] * x,     # dx/dt
    lambda x, u, theta: x,                      # y = x
    n_x=1, n_y=1, dt=0.5,
    method="diffrax:kvaerno5",                  # stiff
    lb=0.0,                                     # a concentration
)
```

`noise=` says how $w$ enters: `None` adds it to the state at the end of the
interval, an array $G$ adds $G w$, and a callable `(x, w) -> x` does whatever you
say.

## Measurements: multi-rate, and late

A running plant does not hand an estimator a tidy matrix. Two facts follow, and
both are built in rather than bolted on.

**A measurement has two times.** `Measurement.time` is when the sample was
*taken* — the state it constrains — and `Measurement.reported` is when the value
became available. An estimator running at `now` may use every record with
`reported <= now`, but each one is placed against the state at its own `time`:

```python
assay = Measurement(time=1.0, values={"Nd_org": 0.31}, sigma={"Nd_org": 0.004},
                    reported=7.0, label="shift assay")
window, not_yet = build_window(times, records, y_names=[...], now=7.0)
```

Placing that assay against the *current* state is not a small error: it asserts
the plant is where it was six hours ago. On a plant whose state halves every
interval it is wrong by a factor of 32, and `tests/test_mhe.py` checks exactly
that discrepancy.

**A channel that is not sampled is not a missing value.** It is a measurement of
infinite variance. Written that way, every array shape stays fixed under `jit`,
and "unmeasured" means the same thing here as in steady-state reconciliation —
`build_window` fills `sigma` with `inf` wherever a channel carries no
information at a grid time, and both the filter and the horizon read that through
`difflow.reconciliation.measured_mask`.

Two records landing on the same grid time and channel are combined by
inverse-variance weighting, which is the exact posterior for independent Gaussian
readings of the same quantity, not an approximation. A record that falls outside
the grid — including one older than the window, whose information now lives in
the arrival cost — is *returned* rather than silently misplaced:

```python
window, dropped = build_window(times, records, y_names=names, now=t_now)
if dropped:
    print(f"{len(dropped)} record(s) not placed")
print(window.summary())     # per-channel sampling counts, and the delays seen
```

## The extended Kalman filter

```python
run = run_ekf(model, window, x0=x0, P0=P0, process_std=q)
run.final.x, run.final.std, run.innovations
```

Update-then-predict at each grid point, in one `lax.scan`. Multi-rate and missing
data change no shape: a channel whose sigma is infinite has its row of $H$ zeroed
and its noise variance set to one, which makes the corresponding gain column
exactly zero. The covariance update is in Joseph form, so it stays symmetric
positive semi-definite even when the gain is not the optimal one — which it is
not once the model is nonlinear or the mean has been clipped to its bounds.

## The arrival cost

Full-information estimation uses every measurement ever taken and its cost grows
without bound. A moving horizon keeps the last $K$ intervals and replaces
everything before them with

$$\Gamma_{k-K}(x) = (x - \bar x)^{T} P^{-1} (x - \bar x).$$

Choosing it well is the whole difficulty of the method. The exact arrival cost is
generally unavailable for a nonlinear model, so what is used is its Gaussian
approximation, whose covariance follows the EKF recursion —
`advance_arrival_cost` — with one difference from running a filter on its own:

* the Jacobians are taken **along the trajectory the optimiser found**, not along
  the filter's mean. When the two differ — which is exactly when constraints bind
  or the model is strongly nonlinear, i.e. when MHE is worth its cost — the MHE
  trajectory is the better linearisation point;
* the mean is the optimiser's **smoothed** estimate of the second state in the
  window, informed by every measurement in it.

A warning the theory makes explicit: summarising a *constrained* problem by an
unconstrained quadratic can be over-confident, because information the
constraints supplied is not represented in $P$. `ArrivalCost.condition` is how
you notice, and `ArrivalCost.inflate(factor)` is the blunt, standard remedy.

`ArrivalCost.vague(x_bar)` is a deliberately uninformative prior — useful for a
first window, and for checking that a result is driven by the data rather than by
the prior: run it twice, once vague, and see how far the estimate moves.

## Constraints

Bounds on the state are handled two ways, for two reasons.

* The **initial state** is reparameterised through a smooth bijection — a sigmoid
  on a two-sided bound, a softplus on a one-sided one — so $x_0$ cannot leave its
  bounds at *any* iterate.
* **Later states** are the image of the dynamics and cannot be reparameterised,
  so they carry a penalty residual weighted by `MHEProblem.constraint_weight`
  (default `1e3`, relative to one sigma).

`MHEResult.max_violation` reports what the penalty left. It is not zero by
construction, so check it; raise `constraint_weight` if it is not small enough.

```python
model = linear_model(A, C, lb=0.0)          # a concentration
res = solve_mhe(problem, window)
assert res.max_violation < 1e-3
```

An unconstrained fit of a nearly-unobservable concentration will happily return a
negative number, and reporting a negative concentration is worse than reporting a
slightly biased one.

## Joint state and parameter estimation

Augmenting the state with slowly drifting parameters is the standard way to
*detect* degradation rather than merely suffer it. `augment_parameters` appends
them as a random walk, $p_{k+1} = p_k + w^p_k$:

```python
from difflow.mhe import augment_parameters, run_mhe

base = StateSpaceModel(f=lambda x, u, w, th: th["a"] * x + 0.1 + w,
                       h=lambda x, u, th: x,
                       n_x=1, n_y=1, x_names=["c"], y_names=["yc"])
model = augment_parameters(base, ["a"], lb=0.0, ub=1.0)

run = run_mhe(model, record, horizon=8,
              process_std=jnp.array([1e-4, 3e-3]),   # state, then parameter
              theta={"a": 0.0},                      # overwritten by the estimate
              x0=jnp.array([0.5, 0.95]), P0=jnp.diag(jnp.array([0.05, 0.05])**2))

run.parameters["a"]          # the estimated drift, shape (N + 1,)
run.windows[-1].parameters   # {'a': 0.758} -- the current estimate
```

The process-noise standard deviation on $w^p$ is the tuning knob and a modelling
choice with real consequences: too large and the parameter absorbs sensor noise
and the state estimate stops correcting; too small and a genuine drift is
rejected as noise. It is an explicit argument, never a default.

`inject=` controls how the estimated vector is turned into the object the base
model expects. The default handles the two common cases: `theta=None` passes the
raw vector through (so write `theta[0]`), and a mapping `theta` is copied with
the named entries overwritten.

## Observability: can this window see that state?

Whether a parameter can be recovered at all is a question to ask *before* the
solve, not by inspecting a NaN afterwards. `check_observability` returns the rank
of the window observability matrix taken along the trajectory:

```python
report = check_observability(model, window, x_now)
report.observable            # False
report.unobservable          # ['stage_efficiency']
report.raise_if_unobservable()
print(report.summary())
```

If the matrix is rank deficient, the estimate of the deficient directions comes
entirely from the arrival cost, and tightening the sensors will not help: those
directions need a different sensor, a longer horizon, or a moving input.
`report.gramian_std` names, direction by direction, how much standard deviation
the measurements alone leave.

## Consistency testing

Every term in the objective is whitened before it is summed, so the objective is
dimensionless and is a $\chi^2$ statistic on the number of scalar readings in the
window. `mhe_global_test` reads it as one, and returns the *same*
`difflow.reconciliation.GlobalTestResult` that steady-state reconciliation
produces, so a dynamic and a steady-state consistency check can be read side by
side:

```python
from difflow.mhe import mhe_global_test
print(mhe_global_test(result))
# global test: chi2 = 14.0 on 14 dof, critical = 23.7, p = 0.45 -> no gross error
```

A rejection says the window's data and the model disagree by more than the stated
noise — a failed sensor, a disturbance the process noise does not cover, or a
parameter that has drifted out from under a fixed model.
`MHEResult.arrival_objective`, `.process_objective` and `.measurement_objective`
say *where* a large objective came from.

## Sliding a horizon over a campaign

```python
run = run_mhe(model, record, horizon=12, process_std=q, x0=x0, P0=P0)
run.x            # the estimate of x_j made at time j, shape (N + 1, n_x)
run.source       # 'ekf' before the first full window, 'mhe' after
run.ekf          # the filter over the same record, as the baseline
run.converged
print(run.summary())
```

The first `horizon` grid points have no full window behind them, so they are
filtered rather than optimised; `source` records which, so a plot never silently
mixes the two. From then on each sampling time solves one window and rolls the
arrival cost forward. The window solve is **traced once and reused**, so a run
costs $N$ optimisations and one compilation, not $N$ of each; `warm_start=True`
(the default) seeds each window from the previous solution, shifted one step.

## `jit`, `grad`, and feeding a planner

The dynamics are *eliminated*, not imposed as equality constraints: the decision
variables are $x_0$ and the noise sequence, and the states follow from a
`lax.scan`. That makes the problem a least-squares problem whose residual is the
whitened concatenation of the three terms, so it goes to
`optimistix.LevenbergMarquardt` and inherits Gauss-Newton convergence *and*
implicit differentiation.

`estimate` is the pure array-in / array-out form — no diagnostics, no Python
floats — so it composes:

```python
from dataclasses import replace
from difflow.mhe import estimate

def current_state(y):
    return estimate(problem, replace(window, y=y))[-1]

jax.jit(current_state)(window.y)
jax.jacobian(current_state)(window.y)     # implicit, not unrolled
```

Differentiation goes through the optimality conditions, not through the iterates,
so its cost does not grow with the iteration count. An estimate can therefore be
a link in a larger differentiable chain rather than the end of one.

`MHEResult.parameters` returns `{name: value}` — the shape
`difflow.planning.Block.theta` takes, and the shape
`difflow.planning.update_modifiers` accepts as its `theta` override — so the
current estimate goes straight into the modifier-adaptation loop, no adapter:

```python
from difflow.planning import update_modifiers
mods = update_modifiers(block, u_plan, plant_fn,
                        theta=run.windows[-1].parameters)
```

That loop is the point of estimating parameters at all: an optimiser acting on a
model whose parameters drifted last week is optimising the wrong plant.

## What is reused rather than duplicated

| Borrowed from | Used for |
|---|---|
| `difflow.dynamic.integrate` | the flow map in `StateSpaceModel.from_ode`; this module owns no integrator |
| `difflow.reconciliation.measured_mask` and the `sigma = inf` convention | "not sampled" means the same in `build_window`, `run_ekf`, `solve_mhe` and `check_observability` |
| `difflow.reconciliation.Scaling` | decision-variable scaling in `MHEProblem.scaling`, column scaling in `check_observability` |
| `difflow.reconciliation.structure._rank_and_spectrum` | one definition of "rank deficient" — SVD of the scaled matrix, never eigenvalues of $O^TO$ |
| `difflow.reconciliation.GlobalTestResult` | `mhe_global_test`, so dynamic and steady-state tests read alike |
| `optimistix.LevenbergMarquardt` | the window solve and its implicit derivatives |

## Tuning, and what goes wrong

| Symptom | Likely cause | What to do |
|---|---|---|
| Estimate stops responding to new data | the arrival cost has become over-confident | check `ArrivalCost.condition`; `inflate()` it |
| A state or parameter sits at its prior | that direction is unobservable from the window | `check_observability`; lengthen the horizon or add a sensor |
| A parameter tracks sensor noise | its `process_std` is too large | shrink it; it sets the allowed drift rate |
| A real drift is rejected | its `process_std` is too small | raise it |
| `max_violation` not small | penalty too weak | raise `constraint_weight` |
| `success=False` on a window | Levenberg-Marquardt hit `max_steps` or a tolerance | the run continues; inspect `MHERunResult.summary()` before trusting it |
| Objective far above the $\chi^2$ critical value | data and model disagree beyond the stated noise | `mhe_global_test`, then read the three objective parts |

The estimator does **not** raise on a failed window: `MHEResult.success` reports
it so a sliding run continues rather than stopping on one bad sample.

## API summary

```python
from difflow.mhe import (
    # the model
    StateSpaceModel, augment_parameters, linear_model,
    # measurements
    Measurement, MeasurementWindow, build_window, slice_window,
    # extended Kalman filter
    EKFState, EKFRunResult, ekf_predict, ekf_update, run_ekf,
    # arrival cost
    ArrivalCost, advance_arrival_cost, CHOLESKY_JITTER,
    # moving-horizon estimation
    MHEProblem, MHEResult, MHERunResult,
    solve_mhe, estimate, run_mhe, mhe_global_test, CONSTRAINT_WEIGHT,
    # observability
    ObservabilityReport, check_observability,
)
```

Tests: `tests/test_mhe.py`. The load-bearing one is
`test_mhe_full_information_matches_kalman`: on a linear model with Gaussian noise
and no active constraints, moving-horizon estimation over the whole record *is*
the Kalman filter, and the two agree to about `1e-13` in the mean and `1e-15` in
the covariance.
