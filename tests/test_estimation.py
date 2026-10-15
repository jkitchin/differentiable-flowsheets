"""Smoke tests for difflow.estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.estimation import (
    BootstrapResult, ConfidenceResult, CrossValidationResult, DiagnosticsResult,
    Estimator, Experiment, negative_log_likelihood, sum_squared_errors,
    weighted_sum_squared_errors,
)

NAMES = ["a", "b"]


def linear_model(theta, exp):
    return {"y": theta["a"] * exp.inputs["x"] + theta["b"]}


def _experiments(n=8, noise=0.05):
    rng = np.random.default_rng(0)
    return [Experiment(inputs={"x": float(x)},
                       observed={"y": float(2 * x + 1 + noise * rng.standard_normal())})
            for x in np.linspace(0.5, 5.0, n)]


@pytest.fixture(scope="module")
def fitted():
    exps = _experiments()
    est = Estimator(linear_model, NAMES)
    return est, exps, est.fit(exps, {"a": 1.0, "b": 0.0})


def test_experiment():
    exp = Experiment(inputs={"x": 1.0}, observed={"y": 2.0}, uncertainties={"y": 0.5},
                     name="e")
    assert exp.output_names == ["y"] and exp.observed_array.shape == (1,)
    np.testing.assert_allclose(exp.weights, [4.0])
    assert "inputs" in exp and exp["name"] == "e"


def test_objectives_and_grad():
    exps = _experiments(n=4)
    theta = jnp.array([1.0, 0.0])
    for f in (sum_squared_errors, weighted_sum_squared_errors, negative_log_likelihood):
        assert np.isfinite(float(f(linear_model, theta, exps, NAMES)))
        g = jax.grad(lambda t: f(linear_model, t, exps, NAMES))(theta)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_fit_and_analysis(fitted):
    est, exps, result = fitted
    assert result["converged"] is True and "theta_opt" in result
    ci = est.confidence_intervals(result, exps)
    assert isinstance(ci, ConfidenceResult) and ci.covariance.shape == (2, 2)
    diag = est.diagnostics(result, exps)
    assert isinstance(diag, DiagnosticsResult) and len(diag.residuals) == len(exps)
    assert "Parameter Estimation Summary" in est.summary(result, exps)


def test_cross_validate(fitted):
    est, exps, _ = fitted
    cv = est.cross_validate(exps, {"a": 1.0, "b": 0.0}, n=2, max_folds=3)
    assert isinstance(cv, CrossValidationResult) and cv.n_folds == 3 and cv.n_holdout == 2


def test_bootstrap(fitted):
    est, exps, result = fitted
    bs = est.bootstrap(result, exps, n_bootstrap=3, method="nonparametric", seed=0)
    assert isinstance(bs, BootstrapResult) and bs.theta_samples.shape == (3, 2)
