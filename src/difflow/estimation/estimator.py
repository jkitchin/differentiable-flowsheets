"""Estimator class — main orchestrator for parameter estimation.

Provides a high-level API inspired by pyomo.parmest, powered by
JAX autodiff for exact gradients and JIT compilation.
"""

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize

from difflow.params_mixin import ParamsMixin
from difflow.estimation.experiment import Experiment
from difflow.estimation.objectives import (
    sum_squared_errors,
    weighted_sum_squared_errors,
    negative_log_likelihood,
)
from difflow.estimation.confidence import fisher_confidence_intervals, ConfidenceResult
from difflow.estimation.diagnostics import compute_diagnostics, DiagnosticsResult
from difflow.estimation.bootstrap import (
    nonparametric_bootstrap,
    parametric_bootstrap,
    BootstrapResult,
)
from difflow.estimation.cross_validation import leave_n_out_cv, CrossValidationResult


_OBJECTIVE_MAP = {
    'sse': sum_squared_errors,
    'wsse': weighted_sum_squared_errors,
    'nll': negative_log_likelihood,
}


@dataclass
class EstimationResult(ParamsMixin):
    """Result of parameter estimation.

    Attributes:
        theta_opt: dict of optimal parameter values
        theta_array: optimal parameter array
        objective_value: value of objective at optimum
        converged: whether the optimizer converged
        n_iterations: number of optimizer iterations
        message: optimizer status message
        param_names: list of parameter names
    """

    theta_opt: dict[str, float]
    theta_array: Array
    objective_value: float
    converged: bool
    n_iterations: int
    message: str
    param_names: list[str] = field(default_factory=list)


class Estimator:
    """Parameter estimation orchestrator.

    Wraps model fitting, confidence intervals, bootstrap, cross-validation,
    and diagnostics into a cohesive API.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
            - theta_dict: parameter names to values
            - experiment: an Experiment object
            - Returns: predictions dict matching experiment.observed keys
        param_names: list of parameter names
        param_bounds: optional dict of (lo, hi) bounds per parameter

    Example:
        >>> est = Estimator(my_model, ['k', 'Ea'], param_bounds={'k': (0, 10)})
        >>> result = est.fit(experiments, theta_init={'k': 1.0, 'Ea': 5000.0})
        >>> ci = est.confidence_intervals(result, experiments)
        >>> print(est.summary(result, experiments))
    """

    def __init__(self, model_fn, param_names, param_bounds=None):
        self.model_fn = model_fn
        self.param_names = list(param_names)
        self.param_bounds = param_bounds

    def fit(
        self,
        experiments,
        theta_init,
        objective='sse',
        method='L-BFGS-B',
        jit_objective=True,
    ):
        """Fit parameters to experimental data.

        Args:
            experiments: list of Experiment objects
            theta_init: dict or array of initial parameter guesses
            objective: 'sse', 'wsse', or 'nll'
            method: scipy.optimize.minimize method
            jit_objective: whether to JIT-compile the objective

        Returns:
            EstimationResult
        """
        # Convert theta_init to array
        if isinstance(theta_init, dict):
            theta0 = jnp.array([float(theta_init[name]) for name in self.param_names])
        else:
            theta0 = jnp.array(theta_init, dtype=jnp.float64)

        # Select objective function
        obj_fn = _OBJECTIVE_MAP[objective]

        # Build closed-over objective
        def objective_fn(theta):
            return obj_fn(self.model_fn, theta, experiments, self.param_names)

        if jit_objective:
            objective_fn = jax.jit(objective_fn)

        grad_fn = jax.jit(jax.grad(
            lambda theta: obj_fn(self.model_fn, theta, experiments, self.param_names)
        ))

        # scipy interface
        def scipy_obj(x):
            return float(objective_fn(jnp.array(x)))

        def scipy_grad(x):
            return np.array(grad_fn(jnp.array(x)), dtype=np.float64)

        # Build bounds
        bounds_list = None
        if self.param_bounds is not None:
            bounds_list = [self.param_bounds.get(name, (None, None))
                           for name in self.param_names]

        result = minimize(
            scipy_obj,
            np.array(theta0, dtype=np.float64),
            jac=scipy_grad,
            method=method,
            bounds=bounds_list,
        )

        theta_opt_arr = jnp.array(result.x)
        theta_opt_dict = {name: float(result.x[i])
                          for i, name in enumerate(self.param_names)}

        return EstimationResult(
            theta_opt=theta_opt_dict,
            theta_array=theta_opt_arr,
            objective_value=float(result.fun),
            converged=bool(result.success),
            n_iterations=int(result.get('nit', 0)),
            message=str(result.message),
            param_names=self.param_names,
        )

    def confidence_intervals(self, result, experiments, alpha=0.05, objective='sse'):
        """Compute Fisher information confidence intervals.

        Args:
            result: EstimationResult from fit()
            experiments: list of Experiment objects
            alpha: significance level (0.05 = 95% CI)
            objective: which objective to use for Hessian ('sse', 'wsse', 'nll')

        Returns:
            ConfidenceResult
        """
        obj_fn = _OBJECTIVE_MAP[objective]

        def closed_obj(theta):
            return obj_fn(self.model_fn, theta, experiments, self.param_names)

        return fisher_confidence_intervals(
            closed_obj, result.theta_array, experiments, self.param_names, alpha
        )

    def bootstrap(
        self,
        result,
        experiments,
        n_bootstrap=200,
        method='nonparametric',
        alpha=0.05,
        seed=42,
    ):
        """Bootstrap uncertainty quantification.

        Args:
            result: EstimationResult from fit()
            experiments: list of Experiment objects
            n_bootstrap: number of bootstrap resamples
            method: 'nonparametric' or 'parametric'
            alpha: significance level
            seed: random seed

        Returns:
            BootstrapResult
        """
        theta_init = np.array(result.theta_array, dtype=np.float64)

        if method == 'nonparametric':
            return nonparametric_bootstrap(
                self.model_fn, theta_init, experiments, self.param_names,
                self.param_bounds, n_bootstrap, alpha, seed,
            )
        elif method == 'parametric':
            return parametric_bootstrap(
                self.model_fn, theta_init, experiments, self.param_names,
                self.param_bounds, n_bootstrap, alpha, seed,
            )
        else:
            raise ValueError(f"Unknown bootstrap method: {method}")

    def cross_validate(self, experiments, theta_init, n=1, max_folds=None):
        """Leave-N-out cross-validation.

        Args:
            experiments: list of Experiment objects
            theta_init: dict or array of initial parameter guesses
            n: number of experiments to hold out per fold
            max_folds: maximum folds to evaluate (None = all)

        Returns:
            CrossValidationResult
        """
        if isinstance(theta_init, dict):
            theta0 = np.array([float(theta_init[name]) for name in self.param_names])
        else:
            theta0 = np.array(theta_init, dtype=np.float64)

        return leave_n_out_cv(
            self.model_fn, theta0, experiments, self.param_names,
            n, self.param_bounds, max_folds,
        )

    def diagnostics(self, result, experiments):
        """Compute diagnostic statistics.

        Args:
            result: EstimationResult from fit()
            experiments: list of Experiment objects

        Returns:
            DiagnosticsResult
        """
        return compute_diagnostics(
            self.model_fn, result.theta_opt, experiments, len(self.param_names)
        )

    def summary(self, result, experiments, alpha=0.05):
        """Generate a text summary of the estimation results.

        Args:
            result: EstimationResult from fit()
            experiments: list of Experiment objects
            alpha: significance level for CIs

        Returns:
            Formatted summary string.
        """
        diag = self.diagnostics(result, experiments)
        ci = self.confidence_intervals(result, experiments, alpha)

        lines = []
        lines.append("=" * 60)
        lines.append("Parameter Estimation Summary")
        lines.append("=" * 60)
        lines.append(f"Converged: {result.converged}")
        lines.append(f"Objective value: {result.objective_value:.6g}")
        lines.append(f"Iterations: {result.n_iterations}")
        lines.append("")
        lines.append("Parameters:")
        lines.append(f"  {'Name':<15} {'Value':>12} {'Std Err':>12} "
                      f"{'CI Lower':>12} {'CI Upper':>12}")
        lines.append("  " + "-" * 63)
        for name in self.param_names:
            val = result.theta_opt[name]
            se = ci.std_errors[name]
            lo = ci.ci_lower[name]
            hi = ci.ci_upper[name]
            lines.append(f"  {name:<15} {val:>12.6g} {se:>12.6g} {lo:>12.6g} {hi:>12.6g}")
        lines.append("")
        lines.append("Diagnostics:")
        lines.append(f"  R-squared:     {diag.r_squared:.6f}")
        lines.append(f"  Adj R-squared: {diag.r_squared_adj:.6f}")
        lines.append(f"  RMSE:          {diag.rmse:.6g}")
        lines.append(f"  AIC:           {diag.aic:.4f}")
        lines.append(f"  BIC:           {diag.bic:.4f}")
        lines.append(f"  N obs:         {diag.n_obs}")
        lines.append(f"  N params:      {diag.n_params}")
        lines.append("=" * 60)

        return "\n".join(lines)
