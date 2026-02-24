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
from difflow.estimation.objectives import sum_squared_errors


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

    cv_scores = []
    fold_params = []

    for holdout_indices in folds:
        holdout_set = set(holdout_indices)
        train_exps = [experiments[i] for i in range(n_exp) if i not in holdout_set]
        test_exps = [experiments[i] for i in holdout_indices]

        # Fit on training set
        def obj(theta):
            return float(sum_squared_errors(model_fn, jnp.array(theta), train_exps, param_names))

        def grad_obj(theta):
            g = jax.grad(lambda t: sum_squared_errors(model_fn, t, train_exps, param_names))(
                jnp.array(theta)
            )
            return np.array(g, dtype=np.float64)

        result = minimize(obj, np.array(theta_init, dtype=np.float64),
                          jac=grad_obj, method='L-BFGS-B', bounds=bounds_list)

        theta_dict = {name: float(result.x[i]) for i, name in enumerate(param_names)}
        fold_params.append(theta_dict)

        # Prediction error on test set
        score = float(sum_squared_errors(
            model_fn, jnp.array(result.x), test_exps, param_names
        ))
        cv_scores.append(score)

    return CrossValidationResult(
        cv_scores=cv_scores,
        fold_params=fold_params,
        mean_score=float(np.mean(cv_scores)),
        std_score=float(np.std(cv_scores)),
        n_folds=len(folds),
        n_holdout=n,
    )
