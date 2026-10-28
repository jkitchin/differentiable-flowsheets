"""Fisher information confidence intervals.

Uses jax.hessian for exact second derivatives at the optimum to compute
parameter covariance and confidence intervals. For the default
sum-of-squares objective the covariance is ``s^2 (J^T J)^{-1}``, the
textbook least-squares result (``2 s^2 H^{-1}``, since ``H = 2 J^T J``).

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
    objective='sse',
):
    """Compute confidence intervals from the Fisher information matrix.

    ``H`` is the exact Hessian of ``objective_fn`` at the optimum. How it
    becomes a covariance depends on which objective it is the Hessian of:

    - ``'sse'``, ``S = sum r_i^2``: ``H = 2 J^T J`` at a least-squares
      optimum (exactly so for a linear model), so
      ``cov = 2 s^2 H^{-1} = s^2 (J^T J)^{-1}``, with the residual variance
      estimated from the fit, ``s^2 = S / (n_obs - n_params)``. For a
      straight line this is the textbook ``s^2 (X^T X)^{-1}``.
    - ``'wsse'``, ``S = sum r_i^2 / sigma_i^2`` with the declared 1-sigma
      ``uncertainties`` taken as known: ``cov = 2 H^{-1} = (J^T W J)^{-1}``,
      the same covariance :func:`difflow.estimation.predicted_covariance`
      predicts. No ``s^2`` is estimated; the uncertainties *are* the error
      model.
    - ``'nll'``, the Gaussian negative log-likelihood: ``H`` is the
      observed Fisher information, so ``cov = H^{-1}``.

    Args:
        objective_fn: objective(theta) -> scalar (already closed over experiments)
        theta_opt: optimal parameter array
        experiments: list of Experiment objects (for degrees of freedom)
        param_names: list of parameter names
        alpha: significance level (default 0.05 for 95% CI)
        objective: which objective ``objective_fn`` is -- 'sse' (default),
            'wsse' or 'nll'; it fixes the scaling above.

    Returns:
        ConfidenceResult with covariance, std errors, and CIs.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.0, 1.0, 2.0, 3.0])
        >>> y = jnp.array([1.0, 2.9, 5.2, 6.9])
        >>> sse = lambda th: jnp.sum((th[0] * x + th[1] - y) ** 2)
        >>> X = jnp.stack([x, jnp.ones(4)], axis=1)
        >>> th = jnp.linalg.lstsq(X, y)[0]
        >>> from difflow.estimation import Experiment
        >>> exps = [Experiment(observed={'y': float(v)}) for v in y]
        >>> ci = fisher_confidence_intervals(sse, th, exps, ['a', 'b'])
        >>> s2 = float(sse(th)) / 2
        >>> bool(jnp.isclose(ci.covariance[0, 0], s2 * jnp.linalg.inv(X.T @ X)[0, 0]))
        True
    """
    if objective not in ('sse', 'wsse', 'nll'):
        raise ValueError(
            f"Unknown objective {objective!r}; expected 'sse', 'wsse' or 'nll'"
        )
    n_params = len(param_names)

    # Count total observations: every entry of an array-valued output
    n_obs = sum(exp.n_obs for exp in experiments)
    dof = n_obs - n_params

    # Hessian at optimum, compiled once rather than run op by op
    hessian = jax.jit(jax.hessian(objective_fn))(theta_opt)
    h_inv = jnp.linalg.inv(hessian)

    if objective == 'sse':
        # S = sum r^2 carries no 1/2, so H = 2 J^T J and the factor of 2
        # does NOT cancel: cov = s^2 (J^T J)^{-1} = 2 s^2 H^{-1}.
        sse_opt = objective_fn(theta_opt)
        s2 = float(sse_opt) / max(dof, 1)
        cov = 2.0 * s2 * h_inv
    elif objective == 'wsse':
        # Known sigmas: H = 2 J^T W J, cov = (J^T W J)^{-1}.
        cov = 2.0 * h_inv
    else:
        # Gaussian NLL already has the 1/2: H is the Fisher information.
        cov = h_inv

    return confidence_result_from_covariance(
        cov, theta_opt, param_names, alpha=alpha, dof=dof
    )
