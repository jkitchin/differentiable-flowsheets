"""Diagnostics for parameter estimation results.

Residual analysis, R-squared, RMSE, AIC, BIC.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin


@dataclass
class DiagnosticsResult(ParamsMixin):
    """Result of diagnostic analysis.

    Attributes:
        residuals: list of dicts [{output_name: residual}, ...] per experiment
        r_squared: Coefficient of determination
        r_squared_adj: Adjusted R-squared
        rmse: Root mean squared error
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        n_obs: Number of observations
        n_params: Number of parameters
    """

    residuals: list[dict[str, float]]
    r_squared: float
    r_squared_adj: float
    rmse: float
    aic: float
    bic: float
    n_obs: int
    n_params: int


def compute_diagnostics(model_fn, theta_dict, experiments, n_params):
    """Compute diagnostic statistics for a fitted model.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta_dict: dict of optimal parameter values
        experiments: list of Experiment objects
        n_params: number of estimated parameters

    Returns:
        DiagnosticsResult
    """
    all_residuals = []
    obs_values = []
    resid_values = []

    for exp in experiments:
        preds = model_fn(theta_dict, exp)
        exp_resid = {}
        for key in exp.output_names:
            r = float(preds[key]) - float(exp.observed[key])
            exp_resid[key] = r
            resid_values.append(r)
            obs_values.append(float(exp.observed[key]))
        all_residuals.append(exp_resid)

    resid_arr = jnp.array(resid_values)
    obs_arr = jnp.array(obs_values)
    n_obs = len(resid_values)

    # Sum of squared residuals
    ss_res = float(jnp.sum(resid_arr ** 2))

    # Total sum of squares
    ss_tot = float(jnp.sum((obs_arr - jnp.mean(obs_arr)) ** 2))

    # R-squared
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Adjusted R-squared
    if n_obs - n_params - 1 > 0:
        r_squared_adj = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params - 1)
    else:
        r_squared_adj = r_squared

    # RMSE
    rmse = float(jnp.sqrt(ss_res / n_obs))

    # AIC = n*ln(SSR/n) + 2*k
    aic = n_obs * float(jnp.log(ss_res / n_obs)) + 2 * n_params if n_obs > 0 else float('inf')

    # BIC = n*ln(SSR/n) + k*ln(n)
    bic = (n_obs * float(jnp.log(ss_res / n_obs)) + n_params * float(jnp.log(n_obs))
           if n_obs > 0 else float('inf'))

    return DiagnosticsResult(
        residuals=all_residuals,
        r_squared=r_squared,
        r_squared_adj=r_squared_adj,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n_obs=n_obs,
        n_params=n_params,
    )
