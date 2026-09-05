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
- Parametric and nonparametric bootstrap uncertainty
- Cross-validation for predictive performance
- Regression diagnostics (R², RMSE, AIC, BIC)
- Multi-output and nonlinear model support

## Architecture

### Core Components

1. **`Experiment`** (`experiment.py`)
   - Encapsulates a single experimental observation
   - Stores inputs, observed outputs, uncertainties, and metadata
   - Inherits from `ParamsMixin` for a dict-like interface
   - Convenience properties: `observed_array`, `weights`, `output_names`

2. **`Estimator`** (`estimator.py`)
   - Main orchestrator class
   - Wraps fitting, confidence intervals, bootstrap, cross-validation, and diagnostics
   - Uses `scipy.optimize` with exact JAX gradients
   - Supports parameter bounds and multiple objective functions

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
   - Parametric (resample residuals)
   - Percentile confidence intervals

8. **Cross-Validation** (`cross_validation.py`)
   - Leave-N-out cross-validation
   - Predictive performance metrics
   - Optional `max_folds` limit for large datasets

9. **Identifiability** (`identifiability.py`)
   - `check_identifiability` — rank test on the sensitivity matrix
   - Answers whether the parameters are separately estimable *at all*
   - Reuses the SVD rank machinery of `difflow.reconciliation.structure`

10. **Experiment Design** (`design.py`)
    - `design_experiments` — which runs to do next (D/A/E/modified-E optimal)
    - `predicted_covariance` — the confidence intervals a campaign would buy
    - Selects from a *candidate list*, for any model JAX can differentiate.
      Continuous design optimization, profile likelihood, model discrimination
      and classical designs are in the separate `discopt-doe` plugin
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
from difflow.estimation import Estimator

# model_fn(theta, inputs) -> predicted outputs (must be JAX-differentiable)
est = Estimator(model_fn, param_names, param_bounds)

result = est.fit(experiments, theta_init)           # fit parameters
ci = est.confidence_intervals(result, experiments)  # Fisher-information CIs
diag = est.diagnostics(result, experiments)         # R², RMSE, AIC, BIC
bs = est.bootstrap(result, experiments)             # bootstrap uncertainty

print(est.summary(result, experiments))
```

`result` exposes the fitted parameters as both a dict and an array, along with the
convergence status and final objective value.

## Uncertainty Quantification

The module offers three complementary approaches, each with different assumptions
and cost:

- **Fisher information** — asymptotic and fast; derived from the exact Hessian of
  the objective at the optimum. Best when the model is approximately linear near
  the solution.
- **Bootstrap** — distribution-free and robust; resamples experiments
  (nonparametric) or residuals (parametric) and refits. More expensive but makes
  fewer assumptions.
- **Cross-validation** — assesses predictive performance rather than parameter
  precision, and helps detect over-fitting.

Using more than one gives complementary insight into how well the parameters are
determined.

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
- Bootstrap runs sequentially, so large resample counts can be slow.
- Identifiability diagnostics are local and rank-based (`check_identifiability`,
  see [Experiment Design and Identifiability](experiment-design.md)); there is
  no profile likelihood and no global symbolic identifiability analysis.
- There is no model discrimination, no estimability ranking, no classical or
  screening design, and no continuous optimization of experimental conditions.
  Those live in the `discopt-doe` plugin; see
  [what is here and what is in `discopt-doe`](experiment-design.md#related-tools-discopt-doe-and-sensor-placement)
  for the division of labour and when to reach for which.

## Example

A complete worked example is available in the Examples section:
[REE parameter estimation](../examples/22_ree_parameter_estimation.ipynb), which
fits pH-dependent distribution coefficients for La, Nd, and Dy from measured
liquid–liquid extraction concentrations, including Fisher and bootstrap
uncertainty, diagnostics, and cross-validation.
