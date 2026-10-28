"""Leave-N-out cross-validation for parameter estimation.

Evaluates predictive performance by holding out N experiments,
fitting on the rest, and measuring prediction error on the held-out set.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize

from difflow.params_mixin import ParamsMixin
from difflow.estimation._batch import Batch


@dataclass
class CrossValidationResult(ParamsMixin):
    """Result of cross-validation analysis.

    Attributes:
        cv_scores: prediction SSE for each fold
        fold_params: fitted parameters for each fold
        mean_score: mean CV score across folds
        std_score: standard deviation of CV scores
        n_folds: number of folds evaluated
        n_holdout: number held out per fold
    """

    cv_scores: list[float]
    fold_params: list[dict[str, float]]
    mean_score: float
    std_score: float
    n_folds: int
    n_holdout: int


def leave_n_out_cv(
    model_fn,
    theta_init,
    experiments,
    param_names,
    n=1,
    param_bounds=None,
    max_folds=None,
    *,
    vectorize=True,
):
    """Leave-N-out cross-validation.

    For each fold, hold out N experiments, fit on the rest,
    and compute prediction SSE on the held-out experiments.

    Args:
        model_fn: model_fn(theta_dict, experiment) -> dict[str, float]
        theta_init: initial parameter array
        experiments: list of Experiment objects
        param_names: list of parameter names
        n: number of experiments to hold out per fold
        param_bounds: optional dict of (lo, hi) bounds
        max_folds: maximum number of folds to evaluate (None = all)
        vectorize: evaluate the model over the experiments in one
            ``jax.vmap`` when they can be stacked (see :class:`Estimator`).

    Every fold reuses one compiled objective: a fold is a 0/1 weight per
    experiment, an argument of the objective, not a new list to trace.

    Returns:
        CrossValidationResult
    """
    n_exp = len(experiments)
    folds = list(combinations(range(n_exp), n))

    if max_folds is not None and len(folds) > max_folds:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(folds), size=max_folds, replace=False)
        folds = [folds[i] for i in sorted(indices)]

    bounds_list = None
    if param_bounds is not None:
        bounds_list = [param_bounds.get(name, (None, None)) for name in param_names]

    batch = Batch(model_fn, experiments, param_names, vectorize)
    sse = batch.objective('sse')
    value_and_grad = jax.jit(jax.value_and_grad(sse))
    sse_jit = jax.jit(sse)

    cv_scores = []
    fold_params = []

    for holdout_indices in folds:
        test = np.zeros(n_exp)
        test[list(holdout_indices)] = 1.0
        train_data = batch.data_for(count=1.0 - test)

        # Fit on the training set
        def obj(theta, train_data=train_data):
            v, g = value_and_grad(jnp.asarray(theta), train_data)
            return float(v), np.array(g, dtype=np.float64)

        result = minimize(obj, np.array(theta_init, dtype=np.float64),
                          jac=True, method='L-BFGS-B', bounds=bounds_list)

        theta_dict = {name: float(result.x[i]) for i, name in enumerate(param_names)}
        fold_params.append(theta_dict)

        # Prediction error on the held-out experiments
        score = float(sse_jit(jnp.asarray(result.x), batch.data_for(count=test)))
        cv_scores.append(score)

    return CrossValidationResult(
        cv_scores=cv_scores,
        fold_params=fold_params,
        mean_score=float(np.mean(cv_scores)),
        std_score=float(np.std(cv_scores)),
        n_folds=len(folds),
        n_holdout=n,
    )
