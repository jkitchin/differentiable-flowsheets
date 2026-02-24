"""Fisher information confidence intervals.

Uses jax.hessian for exact second derivatives at the optimum to compute
parameter covariance and confidence intervals.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array
from scipy import stats

from difflow.params_mixin import ParamsMixin


@dataclass
class ConfidenceResult(ParamsMixin):
    """Result of confidence interval computation.

    Attributes:
        covariance: Parameter covariance matrix
        std_errors: Standard errors for each parameter
        ci_lower: Lower confidence bounds {name: value}
        ci_upper: Upper confidence bounds {name: value}
        correlation: Correlation matrix
        alpha: Significance level used
        param_names: Parameter names
    """

    covariance: Array
    std_errors: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    correlation: Array
    alpha: float
    param_names: list[str] = field(default_factory=list)


def fisher_confidence_intervals(
    objective_fn,
    theta_opt,
    experiments,
    param_names,
    alpha=0.05,
):
    """Compute confidence intervals from the Fisher information matrix.

    The covariance is estimated as:
        cov = 2 * s^2 * H^{-1}

    where H is the Hessian of the objective at the optimum and s^2 is the
    estimated variance of residuals.

    Args:
        objective_fn: objective(theta) -> scalar (already closed over experiments)
        theta_opt: optimal parameter array
        experiments: list of Experiment objects (for degrees of freedom)
        param_names: list of parameter names
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        ConfidenceResult with covariance, std errors, and CIs.
    """
    n_params = len(param_names)

    # Count total observations
    n_obs = sum(len(exp.output_names) for exp in experiments)
    dof = n_obs - n_params

    # Hessian at optimum
    hessian = jax.hessian(objective_fn)(theta_opt)

    # Estimated residual variance: s^2 = SSE / dof
    sse_opt = objective_fn(theta_opt)
    s2 = float(sse_opt) / max(dof, 1)

    # Covariance = s^2 * H^{-1}  (factor of 2 cancels with 1/2 in Hessian of SSE)
    cov = s2 * jnp.linalg.inv(hessian)

    # Standard errors
    std_arr = jnp.sqrt(jnp.diag(cov))
    std_errors = {name: float(std_arr[i]) for i, name in enumerate(param_names)}

    # t-statistic for CIs
    t_val = stats.t.ppf(1 - alpha / 2, max(dof, 1))

    ci_lower = {}
    ci_upper = {}
    for i, name in enumerate(param_names):
        ci_lower[name] = float(theta_opt[i] - t_val * std_arr[i])
        ci_upper[name] = float(theta_opt[i] + t_val * std_arr[i])

    # Correlation matrix
    d = jnp.sqrt(jnp.diag(cov))
    d_safe = jnp.where(d > 0, d, 1.0)
    correlation = cov / jnp.outer(d_safe, d_safe)

    return ConfidenceResult(
        covariance=cov,
        std_errors=std_errors,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        correlation=correlation,
        alpha=alpha,
        param_names=param_names,
    )
