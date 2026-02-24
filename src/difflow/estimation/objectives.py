"""Objective functions for parameter estimation.

Pure functions compatible with jax.grad and jax.jit.
"""

import jax.numpy as jnp
from jax import Array


def sum_squared_errors(model_fn, theta, experiments, param_names):
    """Sum of squared errors between model predictions and observations.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta: 1-D array of parameter values
        experiments: list of Experiment objects
        param_names: list of parameter names (maps theta indices to names)

    Returns:
        Scalar SSE value.
    """
    theta_dict = {name: theta[i] for i, name in enumerate(param_names)}
    sse = 0.0
    for exp in experiments:
        preds = model_fn(theta_dict, exp)
        for key in exp.output_names:
            sse = sse + (preds[key] - exp.observed[key]) ** 2
    return sse


def weighted_sum_squared_errors(model_fn, theta, experiments, param_names):
    """Weighted sum of squared errors (inverse-variance weighting).

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta: 1-D array of parameter values
        experiments: list of Experiment objects
        param_names: list of parameter names

    Returns:
        Scalar weighted SSE value.
    """
    theta_dict = {name: theta[i] for i, name in enumerate(param_names)}
    wsse = 0.0
    for exp in experiments:
        preds = model_fn(theta_dict, exp)
        weights = exp.weights
        for j, key in enumerate(exp.output_names):
            wsse = wsse + weights[j] * (preds[key] - exp.observed[key]) ** 2
    return wsse


def negative_log_likelihood(model_fn, theta, experiments, param_names):
    """Negative log-likelihood assuming Gaussian measurement errors.

    NLL = 0.5 * sum_i [ ((y_i - f_i) / sigma_i)^2 + log(2*pi*sigma_i^2) ]

    If no uncertainties are provided on an experiment, sigma=1 is assumed.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta: 1-D array of parameter values
        experiments: list of Experiment objects
        param_names: list of parameter names

    Returns:
        Scalar NLL value.
    """
    theta_dict = {name: theta[i] for i, name in enumerate(param_names)}
    nll = 0.0
    for exp in experiments:
        preds = model_fn(theta_dict, exp)
        for key in exp.output_names:
            residual = preds[key] - exp.observed[key]
            if exp.uncertainties is not None and key in exp.uncertainties:
                sigma = exp.uncertainties[key]
            else:
                sigma = 1.0
            nll = nll + 0.5 * ((residual / sigma) ** 2 + jnp.log(2 * jnp.pi * sigma ** 2))
    return nll
