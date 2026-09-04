"""Fisher information confidence intervals.

Uses jax.hessian for exact second derivatives at the optimum to compute
parameter covariance and confidence intervals.

The step from a parameter covariance matrix to a
:class:`ConfidenceResult` -- standard errors, Student-t intervals and the
correlation matrix -- is factored into
:func:`confidence_result_from_covariance` so that the *predicted*
covariance of an experiment that has not been run
(:func:`difflow.estimation.predicted_covariance`) and the *achieved*
covariance of a fit are assembled by the same code and can be compared
field by field.
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


def confidence_result_from_covariance(
    covariance,
    theta,
    param_names,
    alpha=0.05,
    dof=1,
):
    """Assemble a :class:`ConfidenceResult` from a parameter covariance.

    Shared by the two places a covariance is turned into intervals: a fit
    that has happened (:func:`fisher_confidence_intervals`, where the
    covariance comes from the Hessian and an estimated residual variance)
    and a campaign that has not (:func:`difflow.estimation.predicted_covariance`,
    where it comes from ``inv(FIM)``). Keeping one assembler is what makes
    the predicted and achieved intervals directly comparable.

    Args:
        covariance: Parameter covariance, shape ``(n, n)``, or ``None`` for
            a singular information matrix -- in which case the standard
            errors are ``inf``, the intervals are the whole real line, and
            the correlation matrix is ``nan``.
        theta: Parameter values the intervals are centered on (array or
            dict keyed by ``param_names``).
        param_names: Parameter names, in the order of ``covariance``.
        alpha: Significance level; 0.05 gives 95% intervals.
        dof: Degrees of freedom for the Student-t quantile
            (``n_obs - n_params``); clipped below at 1.

    Returns:
        ConfidenceResult.

    Example:
        >>> import jax.numpy as jnp
        >>> r = confidence_result_from_covariance(
        ...     jnp.diag(jnp.array([4.0, 9.0])), [1.0, 2.0], ['a', 'b'],
        ...     dof=1000)
        >>> r.std_errors['a'], r.std_errors['b']
        (2.0, 3.0)
    """
    names = list(param_names)
    n_p = len(names)
    if isinstance(theta, dict):
        theta_vals = [float(theta[n]) for n in names]
    else:
        theta_vals = [float(v) for v in jnp.asarray(theta).ravel()]

    t_val = float(stats.t.ppf(1 - alpha / 2, max(int(dof), 1)))

    if covariance is None:
        cov = jnp.full((n_p, n_p), jnp.inf)
        std_errors = {n: float("inf") for n in names}
        correlation = jnp.full((n_p, n_p), jnp.nan)
    else:
        cov = jnp.asarray(covariance, dtype=float)
        std_arr = jnp.sqrt(jnp.diag(cov))
        std_errors = {n: float(std_arr[i]) for i, n in enumerate(names)}
        d = jnp.sqrt(jnp.diag(cov))
        d_safe = jnp.where(d > 0, d, 1.0)
        correlation = cov / jnp.outer(d_safe, d_safe)

    ci_lower = {}
    ci_upper = {}
    for i, name in enumerate(names):
        se = std_errors[name]
        ci_lower[name] = theta_vals[i] - t_val * se
        ci_upper[name] = theta_vals[i] + t_val * se

    return ConfidenceResult(
        covariance=cov,
        std_errors=std_errors,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        correlation=correlation,
        alpha=alpha,
        param_names=names,
    )


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

    return confidence_result_from_covariance(
        cov, theta_opt, param_names, alpha=alpha, dof=dof
    )
