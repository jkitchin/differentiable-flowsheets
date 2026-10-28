"""Objective functions for parameter estimation.

Pure functions compatible with jax.grad and jax.jit.

Each is a sum over experiments, but the model is not called once per
experiment: when the experiments share their structure (the same input
keys and observed outputs, numeric inputs of equal shape) the model is
``jax.vmap``-ed over the stacked inputs, so ``jit`` traces and compiles it
once however many experiments there are. Otherwise -- or when the model
cannot be vectorized, e.g. it branches in Python on an input value -- they
fall back to calling it one experiment at a time. See
:mod:`difflow.estimation._batch`.

An observed value may be an array (a profile, a composition); every entry
is a residual, and its uncertainty -- a scalar or an array of the same
shape -- applies entry by entry.
"""

from difflow.estimation._batch import Batch


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
    return Batch(model_fn, experiments, param_names).objective('sse')(theta)


def weighted_sum_squared_errors(model_fn, theta, experiments, param_names):
    """Weighted sum of squared errors (inverse-variance weighting).

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta: 1-D array of parameter values
        experiments: list of Experiment objects
        param_names: list of parameter names

    Returns:
        Scalar weighted SSE value, ``sum (r / sigma)^2`` with sigma = 1
        where no uncertainty is given.
    """
    return Batch(model_fn, experiments, param_names).objective('wsse')(theta)


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
    return Batch(model_fn, experiments, param_names).objective('nll')(theta)


#: The objective names :meth:`Estimator.fit`, ``confidence_intervals``,
#: ``summary`` and the bootstraps all dispatch on. One table, so a name added
#: here cannot reach one caller and not another.
OBJECTIVES = {
    'sse': sum_squared_errors,
    'wsse': weighted_sum_squared_errors,
    'nll': negative_log_likelihood,
}
