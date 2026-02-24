"""Bootstrap resampling for uncertainty quantification.

Nonparametric and parametric bootstrap methods for estimating
parameter uncertainty distributions.
"""

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize
from scipy import stats

from difflow.params_mixin import ParamsMixin
from difflow.estimation.objectives import sum_squared_errors


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
        method: 'nonparametric' or 'parametric'
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


def nonparametric_bootstrap(
    model_fn,
    theta_init,
    experiments,
    param_names,
    param_bounds=None,
    n_bootstrap=200,
    alpha=0.05,
    seed=42,
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

    Returns:
        BootstrapResult
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n_exp = len(experiments)

    bounds_list = None
    if param_bounds is not None:
        bounds_list = [param_bounds.get(name, (None, None)) for name in param_names]

    theta_samples = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_exp, size=n_exp, replace=True)
        boot_exps = [experiments[i] for i in indices]

        def obj(theta):
            return float(sum_squared_errors(model_fn, jnp.array(theta), boot_exps, param_names))

        def grad_obj(theta):
            g = jax.grad(lambda t: sum_squared_errors(model_fn, t, boot_exps, param_names))(
                jnp.array(theta)
            )
            return np.array(g, dtype=np.float64)

        result = minimize(obj, np.array(theta_init, dtype=np.float64),
                          jac=grad_obj, method='L-BFGS-B', bounds=bounds_list)
        theta_samples.append(result.x)

    theta_samples = jnp.array(theta_samples)
    return _build_bootstrap_result(
        theta_samples, param_names, alpha, "nonparametric", n_bootstrap
    )


def parametric_bootstrap(
    model_fn,
    theta_opt,
    experiments,
    param_names,
    param_bounds=None,
    n_bootstrap=200,
    alpha=0.05,
    seed=42,
):
    """Parametric bootstrap: resample residuals, create synthetic data.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta_opt: optimal parameter array (from initial fit)
        experiments: list of Experiment objects
        param_names: list of parameter names
        param_bounds: optional dict of (lo, hi) bounds
        n_bootstrap: number of bootstrap resamples
        alpha: significance level
        seed: random seed

    Returns:
        BootstrapResult
    """
    import numpy as np
    from copy import deepcopy
    rng = np.random.default_rng(seed)

    theta_dict = {name: float(theta_opt[i]) for i, name in enumerate(param_names)}

    # Compute residuals at optimum
    residuals = []
    for exp in experiments:
        preds = model_fn(theta_dict, exp)
        for key in exp.output_names:
            residuals.append(float(preds[key]) - float(exp.observed[key]))
    residuals = np.array(residuals)

    bounds_list = None
    if param_bounds is not None:
        bounds_list = [param_bounds.get(name, (None, None)) for name in param_names]

    theta_samples = []
    for _ in range(n_bootstrap):
        # Resample residuals
        boot_resids = rng.choice(residuals, size=len(residuals), replace=True)

        # Create synthetic experiments with perturbed observations
        boot_exps = []
        r_idx = 0
        for exp in experiments:
            preds = model_fn(theta_dict, exp)
            new_obs = {}
            for key in exp.output_names:
                new_obs[key] = float(preds[key]) + boot_resids[r_idx]
                r_idx += 1
            new_exp = deepcopy(exp)
            object.__setattr__(new_exp, 'observed', new_obs)
            boot_exps.append(new_exp)

        def obj(theta):
            return float(sum_squared_errors(model_fn, jnp.array(theta), boot_exps, param_names))

        def grad_obj(theta):
            g = jax.grad(lambda t: sum_squared_errors(model_fn, t, boot_exps, param_names))(
                jnp.array(theta)
            )
            return np.array(g, dtype=np.float64)

        result = minimize(obj, np.array(theta_opt, dtype=np.float64),
                          jac=grad_obj, method='L-BFGS-B', bounds=bounds_list)
        theta_samples.append(result.x)

    theta_samples = jnp.array(theta_samples)
    return _build_bootstrap_result(
        theta_samples, param_names, alpha, "parametric", n_bootstrap
    )


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
