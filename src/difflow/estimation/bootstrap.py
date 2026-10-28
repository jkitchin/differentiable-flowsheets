"""Bootstrap resampling for uncertainty quantification.

Two bootstraps, both refitting the model to many resampled data sets:

- **nonparametric** -- resample the experiments with replacement;
- **residual** -- keep the fitted predictions and add residuals resampled,
  with replacement, from the fit's own. It used to be called
  *parametric*, and ``method='parametric'`` / :func:`parametric_bootstrap`
  still work, but no error distribution is assumed, so the name was wrong.

Every refit reuses one compiled objective. A resample with replacement is
a vector of per-experiment counts, a residual resample is a new set of
observations, and both are *arguments* of the objective
(:meth:`difflow.estimation._batch.Batch.objective`), so neither forces a
retrace.
"""

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize

from difflow.params_mixin import ParamsMixin
from difflow.estimation.objectives import OBJECTIVES
from difflow.estimation._batch import Batch


@dataclass
class BootstrapResult(ParamsMixin):
    """Result of bootstrap analysis.

    Attributes:
        theta_samples: (n_bootstrap, n_params) array of fitted parameters
        mean: dict of parameter means
        std: dict of parameter standard deviations
        ci_lower: dict of lower CI bounds
        ci_upper: dict of upper CI bounds
        alpha: significance level
        param_names: parameter names
        n_bootstrap: number of bootstrap samples
        method: 'nonparametric' or 'residual' ('parametric' when it was
            asked for by its old name)
    """

    theta_samples: Array
    mean: dict[str, float]
    std: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    alpha: float
    param_names: list[str] = field(default_factory=list)
    n_bootstrap: int = 0
    method: str = "nonparametric"


def _refitter(batch, objective, theta0, param_names, param_bounds):
    """``refit(data) -> theta``: L-BFGS-B on one compiled value-and-grad."""
    OBJECTIVES[objective]  # KeyError on an unknown name
    value_and_grad = jax.jit(jax.value_and_grad(batch.objective(objective)))
    bounds_list = None
    if param_bounds is not None:
        bounds_list = [param_bounds.get(name, (None, None)) for name in param_names]
    x0 = np.array(theta0, dtype=np.float64)

    def refit(data):
        def fun(x):
            v, g = value_and_grad(jnp.asarray(x), data)
            return float(v), np.array(g, dtype=np.float64)
        return minimize(fun, x0, jac=True, method='L-BFGS-B',
                        bounds=bounds_list).x

    return refit


def nonparametric_bootstrap(
    model_fn,
    theta_init,
    experiments,
    param_names,
    param_bounds=None,
    n_bootstrap=200,
    alpha=0.05,
    seed=42,
    objective='sse',
    *,
    vectorize=True,
):
    """Nonparametric bootstrap: resample experiments with replacement.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta_init: initial parameter array
        experiments: list of Experiment objects
        param_names: list of parameter names
        param_bounds: optional dict of (lo, hi) bounds
        n_bootstrap: number of bootstrap resamples
        alpha: significance level
        seed: random seed
        objective: 'sse', 'wsse' or 'nll'; must match the objective the
            original fit minimized, or the resampled spread describes a
            different estimator than the one whose value is being reported.
        vectorize: evaluate the model over the experiments in one
            ``jax.vmap`` when they can be stacked (see :class:`Estimator`).

    Returns:
        BootstrapResult
    """
    rng = np.random.default_rng(seed)
    n_exp = len(experiments)
    batch = Batch(model_fn, experiments, param_names, vectorize)
    refit = _refitter(batch, objective, theta_init, param_names, param_bounds)

    theta_samples = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_exp, size=n_exp, replace=True)
        # Experiment i drawn c_i times contributes c_i times its term.
        counts = np.bincount(indices, minlength=n_exp)
        theta_samples.append(refit(batch.data_for(count=counts)))

    theta_samples = jnp.array(theta_samples)
    return _build_bootstrap_result(
        theta_samples, param_names, alpha, "nonparametric", n_bootstrap
    )


def residual_bootstrap(
    model_fn,
    theta_opt,
    experiments,
    param_names,
    param_bounds=None,
    n_bootstrap=200,
    alpha=0.05,
    seed=42,
    objective='sse',
    *,
    vectorize=True,
    method_name='residual',
):
    """Residual bootstrap: fitted predictions plus resampled fitted residuals.

    Each synthetic data set is ``f(x_i; theta_opt) + r*``, with the ``r*``
    drawn with replacement from the residuals of the original fit (every
    scalar residual of every experiment in one pool), and is refitted.

    No error distribution is assumed -- this is not a *parametric*
    bootstrap, which would draw normal noise of the estimated variance
    ``s^2 = S/(n - p)``. The resampled residuals have variance ``S/n``, so
    the spread of the refits comes out smaller than the linearized standard
    errors by about ``sqrt((n - p)/n)`` (0.93 for 15 observations and 2
    parameters) on top of its own sampling error.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta_opt: optimal parameter array (from initial fit)
        experiments: list of Experiment objects
        param_names: list of parameter names
        param_bounds: optional dict of (lo, hi) bounds
        n_bootstrap: number of bootstrap resamples
        alpha: significance level
        seed: random seed
        objective: 'sse', 'wsse' or 'nll'; must match the objective the
            original fit minimized.
        vectorize: evaluate the model over the experiments in one
            ``jax.vmap`` when they can be stacked (see :class:`Estimator`).
        method_name: the label stored in ``BootstrapResult.method``.

    Returns:
        BootstrapResult
    """
    rng = np.random.default_rng(seed)
    batch = Batch(model_fn, experiments, param_names, vectorize)

    # Predictions and residuals at the optimum, every scalar entry pooled.
    preds = batch.predictions(theta_opt)
    shapes, fitted, residuals = [], [], []
    for exp, p in zip(experiments, preds):
        for key in exp.output_names:
            obs = np.asarray(exp.observed[key], dtype=float)
            pk = np.broadcast_to(np.asarray(p[key], dtype=float), obs.shape)
            shapes.append((key, obs.shape))
            fitted.append(pk.ravel())
            residuals.append((pk - obs).ravel())
    fitted = np.concatenate(fitted) if fitted else np.zeros(0)
    residuals = np.concatenate(residuals) if residuals else np.zeros(0)

    refit = _refitter(batch, objective, theta_opt, param_names, param_bounds)

    theta_samples = []
    for _ in range(n_bootstrap):
        boot = fitted + rng.choice(residuals, size=residuals.size, replace=True)
        observed, pos, it = [], 0, iter(shapes)
        for exp in experiments:
            new_obs = {}
            for key, shape in (next(it) for _ in exp.output_names):
                size = int(np.prod(shape))
                vals = boot[pos:pos + size]
                new_obs[key] = vals.reshape(shape) if shape else float(vals[0])
                pos += size
            observed.append(new_obs)
        theta_samples.append(refit(batch.data_for(observed=observed)))

    theta_samples = jnp.array(theta_samples)
    return _build_bootstrap_result(
        theta_samples, param_names, alpha, method_name, n_bootstrap
    )


def parametric_bootstrap(*args, **kwargs):
    """The old name of :func:`residual_bootstrap`, kept for compatibility.

    It resamples the fitted residuals; it does not draw from a fitted
    distribution. The result's ``method`` reads ``'parametric'``.
    """
    kwargs.setdefault('method_name', 'parametric')
    return residual_bootstrap(*args, **kwargs)


def _build_bootstrap_result(theta_samples, param_names, alpha, method, n_bootstrap):
    """Build BootstrapResult from samples."""
    mean = {name: float(jnp.mean(theta_samples[:, i]))
            for i, name in enumerate(param_names)}
    std = {name: float(jnp.std(theta_samples[:, i]))
           for i, name in enumerate(param_names)}

    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)
    ci_lower = {name: float(jnp.percentile(theta_samples[:, i], lo_pct))
                for i, name in enumerate(param_names)}
    ci_upper = {name: float(jnp.percentile(theta_samples[:, i], hi_pct))
                for i, name in enumerate(param_names)}

    return BootstrapResult(
        theta_samples=theta_samples,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        param_names=param_names,
        n_bootstrap=n_bootstrap,
        method=method,
    )
