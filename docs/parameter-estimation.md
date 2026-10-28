# Parameter Estimation

The `difflow.estimation` module provides a JAX-powered parameter estimation
framework inspired by `pyomo.parmest`. It combines automatic differentiation with
structured APIs for model fitting, uncertainty quantification, and diagnostics, so
you can calibrate models to experimental data and propagate the resulting
uncertainty through differentiable flowsheets.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Core Components](#core-components)
   - [Supporting Modules](#supporting-modules)
3. [Basic Workflow](#basic-workflow)
4. [Uncertainty Quantification](#uncertainty-quantification)
5. [Use in Chemical Engineering](#use-in-chemical-engineering)
6. [Design Principle: Fit to Measured Quantities](#design-principle-fit-to-measured-quantities)
7. [Current Limitations](#current-limitations)
8. [Example](#example)

---

## Overview

The module fits the parameters of a user-defined model to one or more
`Experiment` observations, using `scipy.optimize` driven by **exact JAX
gradients** (no finite differences). Because every operation is differentiable
end-to-end, the same machinery works whether you are fitting a simple algebraic
model or calibrating parameters inside a complex flowsheet with recycles.

Key capabilities:

- Parameter estimation with box bounds and multiple objective functions
- Fisher-information confidence intervals from the exact Hessian
- Nonparametric and residual bootstrap uncertainty
- Cross-validation for predictive performance
- Regression diagnostics (R², RMSE, AIC, BIC)
- Multi-output and nonlinear model support

## Architecture

### Core Components

1. **`Experiment`** (`experiment.py`)
   - Encapsulates a single experimental observation
   - Stores inputs, observed outputs, uncertainties, and metadata
   - Inherits from `ParamsMixin` for a dict-like interface
   - Convenience properties: `observed_array`, `n_obs`, `weights`, `output_names`
   - An observed value may be an array (a profile); every entry is one
     observation, and `n_obs` counts entries

2. **`Estimator`** (`estimator.py`)
   - Main orchestrator class
   - Wraps fitting, confidence intervals, bootstrap, cross-validation, and diagnostics
   - Uses `scipy.optimize` with exact JAX gradients
   - Supports parameter bounds and multiple objective functions
   - Evaluates the model over all experiments in one `jax.vmap` (see
     [One trace for all experiments](#one-trace-for-all-experiments))

3. **`EstimationResult`** (`estimator.py`)
   - Returns fitted parameters (dict and array forms)
   - Reports convergence status, iterations, and objective value
   - Inherits from `ParamsMixin`

### Supporting Modules

4. **Objectives** (`objectives.py`) — all fully differentiable
   - `sum_squared_errors` (SSE)
   - `weighted_sum_squared_errors` (WSSE)
   - `negative_log_likelihood` (NLL)

5. **Confidence Intervals** (`confidence.py`)
   - Fisher information matrix approach
   - Computes covariance, standard errors, and confidence intervals
   - Uses the JAX Hessian for exact second derivatives

6. **Diagnostics** (`diagnostics.py`)
   - R², adjusted R², RMSE
   - AIC and BIC information criteria
   - Residuals analysis

7. **Bootstrap** (`bootstrap.py`)
   - Nonparametric (resample experiments)
   - Residual (`method="residual"`, formerly "parametric", still accepted):
     fitted predictions plus fitted residuals resampled with replacement. No
     distribution is assumed, and the residuals have variance `S/n` rather
     than `S/(n-p)`, so its spread runs about `sqrt((n-p)/n)` below the
     linearized standard errors
   - Every refit reuses one compiled objective
   - Percentile confidence intervals
   - Takes an `objective`, and it must be the one `fit` minimized (see
     [Matching the objective](#matching-the-objective))

8. **Cross-Validation** (`cross_validation.py`)
   - Leave-N-out cross-validation
   - Predictive performance metrics
   - Optional `max_folds` limit for large datasets

9. **Identifiability** (`identifiability.py`)
   - `check_identifiability` — rank test on the sensitivity matrix
   - Answers whether the parameters are separately estimable *at all*
   - Reuses the SVD rank machinery of `difflow.reconciliation.structure`

10. **Experiment Design** (`design.py`)
    - `design_experiments` — which runs to do next (D/A/E/modified-E optimal,
      or c-optimal for one derived quantity)
    - `predicted_covariance` — the confidence intervals a campaign would buy
    - `design_discrimination` (`discrimination.py`) — runs that tell rival
      models apart (Hunter–Reiner, Buzzi-Ferraris)
    - Selects from a *candidate list*, for any model JAX can differentiate.
      Continuous design optimization, profile likelihood, sequential model
      discrimination and classical designs are in the separate `discopt-doe` plugin
      (`discopt.doe`), which needs a `discopt.modeling` model rather than a JAX
      one.
    - See [Experiment Design and Identifiability](experiment-design.md)

## Basic Workflow

Before fitting anything, ask whether the parameters can be told apart at all:
`check_identifiability` runs a rank test on the sensitivity matrix, and when it
fails no estimator on this page can help — see
[Experiment Design and Identifiability](experiment-design.md), which also covers
choosing the *next* experiment.

The fitting API then follows a simple fit → quantify → diagnose pattern:

```python
import jax.numpy as jnp
from difflow.estimation import Estimator, Experiment

# model_fn(theta, experiment) -> predicted outputs (must be JAX-differentiable)
def model_fn(theta, exp):
    return {'y': theta['k'] * jnp.exp(-theta['b'] * exp.inputs['t'])}

param_names = ['k', 'b']
param_bounds = {'k': (0.0, 10.0), 'b': (0.0, 5.0)}
theta_init = {'k': 1.0, 'b': 0.5}
experiments = [
    Experiment(inputs={'t': t}, observed={'y': 2.0 * float(jnp.exp(-0.7 * t)) + 0.01 * (-1) ** i})
    for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
]

est = Estimator(model_fn, param_names, param_bounds)

result = est.fit(experiments, theta_init)           # fit parameters
ci = est.confidence_intervals(result, experiments)  # Fisher-information CIs
diag = est.diagnostics(result, experiments)         # R², RMSE, AIC, BIC
bs = est.bootstrap(result, experiments, n_bootstrap=20)  # bootstrap uncertainty

print(est.summary(result, experiments))
```

`result` exposes the fitted parameters as both a dict and an array, along with the
convergence status and final objective value.

### Matching the objective

`fit`, `confidence_intervals`, `bootstrap` and `summary` each take an `objective`,
and all four default to `'sse'`. **A weighted fit must carry `objective='wsse'`
through every one of them.** The default is kept for backwards compatibility, and
it is silently wrong in exactly the case where weighting mattered:

```python
result = est.fit(experiments, theta_init, objective='wsse')
ci = est.confidence_intervals(result, experiments, objective='wsse')
bs = est.bootstrap(result, experiments, n_bootstrap=20, objective='wsse')
print(est.summary(result, experiments, objective='wsse'))
```

The intervals come from the Hessian of the objective at the optimum, so an
unweighted Hessian over a weighted fit reports the standard errors of numbers
nobody computed. The bootstrap is worse than that: it *refits*, so on the default
it returns the sampling distribution of a different estimator. In
`examples/22_ree_parameter_estimation.ipynb`, where the measured concentrations
span four decades, the mismatch inflates the bootstrap standard errors several-fold
and reads as a failed linearization.

The objective also fixes how the Hessian $H$ becomes
a covariance: `'sse'` gives the textbook $s^2 (J^T J)^{-1} = 2 s^2 H^{-1}$ with
$s^2 = \mathrm{SSE}/(n_\mathrm{obs} - n_p)$ (the sum of squares has no ½, so
$H = 2 J^T J$); `'wsse'` takes the declared `uncertainties` as known and gives
$(J^T W J)^{-1} = 2 H^{-1}$, with no $s^2$ -- the same covariance
`predicted_covariance` predicts; `'nll'` gives $H^{-1}$.

## Uncertainty Quantification

The module offers three complementary approaches, each with different assumptions
and cost:

- **Fisher information** — asymptotic and fast; derived from the exact Hessian of
  the objective at the optimum. Best when the model is approximately linear near
  the solution.
- **Bootstrap** — distribution-free and robust; resamples experiments
  (nonparametric) or residuals (residual bootstrap) and refits. More expensive but makes
  fewer assumptions.
- **Cross-validation** — assesses predictive performance rather than parameter
  precision, and helps detect over-fitting.

Using more than one gives complementary insight into how well the parameters are
determined.

## One trace for all experiments

A model is written for one experiment, `model(theta, exp) -> dict`, and the
objectives sum over experiments. Calling it once per experiment inside `jit`
would trace and compile one copy of the model per run, which for a difflow unit
(a `CSTR` with its internal Newton solve) is most of the cost of a fit. When
the experiments share their structure -- the same input keys and observed
outputs, numeric inputs of equal shape (a non-numeric input must be the same in
every run) -- `Estimator`, the objective functions, `sensitivity_matrix`,
`check_identifiability` and `design_experiments` stack the inputs and
`jax.vmap` the model over them, so it is traced once however many runs there
are. The model still receives an `Experiment`; its `inputs` hold one run's
traced values.

```python
def model(theta, exp):                       # written for one run
    return {"X": conversion(theta, exp.inputs["T"], exp.inputs["tau"])}

est = Estimator(model, ["lnk", "Ea"])        # one trace of the CSTR for all runs
```

When the runs cannot be stacked, or the model cannot be traced that way (a
Python `if` on an input value, reading `exp.name` or `exp.metadata` that
differ between runs), they fall back to one call per experiment, so a model
that works one run at a time keeps working. `Estimator(..., vectorize=False)`
forces the loop. The bootstrap and cross-validation pass the resampled data as
an argument of one compiled objective (a resample with replacement is a vector
of per-experiment counts), so refits do not recompile either.

## Use in Chemical Engineering

The module is well suited to calibrating process models against measured data:

- **Kinetics** — fit Arrhenius parameters (A, Ea) from batch-reactor concentration
  profiles; estimate catalyst deactivation kinetics.
- **Thermodynamics** — calibrate distribution coefficients, equilibrium constants,
  and activity-coefficient model parameters.
- **Transport** — estimate mass- and heat-transfer coefficients from measured
  profiles.
- **Model calibration** — fit flowsheet models and unit-operation efficiency
  factors to plant data.

Because `difflow` units are already differentiable, estimated parameters can be
fit inside complete flowsheets — including those with recycles — without any extra
machinery.

## Design Principle: Fit to Measured Quantities

**Fit to the quantities you actually measure, not to derived values.**

- ✅ **Good:** fit to concentrations, temperatures, pressures.
- ❌ **Avoid:** fitting to derived quantities such as distribution coefficients
  `D = C_org / C_aq`, reaction rates, or heat-transfer rates.

Fitting raw measurements:

- handles error propagation correctly,
- provides more information (e.g. two measured concentrations rather than one
  ratio),
- matches the actual experimental workflow, and
- enables mass-balance and other physical constraints.

The example notebook demonstrates this by fitting equilibrium concentrations
(`C_aq`, `C_org`) directly rather than the distribution coefficients derived from
them.

## Current Limitations

The current implementation prioritizes clarity and correctness. Known limitations
to be aware of:

- Optimization uses `scipy.optimize.minimize`; JAX-native optimizers
  (`optax`, `jaxopt`) are not yet wired in.
- Constraints are limited to parameter box bounds (no linear/nonlinear
  constraints).
- Weighting supports inverse-variance only (no robust M-estimators or correlated
  errors).
- Bootstrap refits run one after another (each reuses the compiled objective,
  so a refit costs only its optimizer iterations).
- Identifiability diagnostics are local and rank-based (`check_identifiability`,
  see [Experiment Design and Identifiability](experiment-design.md)); there is
  no profile likelihood and no global symbolic identifiability analysis.
- Model discrimination is a design from a candidate list
  (`design_discrimination`); there is no estimability ranking, no classical or
  screening design, and no continuous optimization of experimental conditions.
  Those live in the `discopt-doe` plugin; see
  [what is here and what is in `discopt-doe`](experiment-design.md#related-tools-discopt-doe-and-sensor-placement)
  for the division of labour and when to reach for which.

## Example

A complete worked example is available in the Examples section:
[REE parameter estimation](../examples/22_ree_parameter_estimation.ipynb), which
recovers the shipped PC88A coefficients for La, Nd and Dy from measured
liquid–liquid extraction concentrations. It runs the identifiability check first,
then fits a ladder of four models — free quadratic, linear, shared slope, and the
one-parameter mass-action form the database uses — and lets AIC and BIC choose,
with Fisher and bootstrap uncertainty and censoring of below-LOQ measurements.
