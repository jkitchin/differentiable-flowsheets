"""Smoke tests for difflow.estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.estimation import (
    BootstrapResult, ConfidenceResult, CrossValidationResult, DiagnosticsResult,
    Estimator, Experiment, fisher_confidence_intervals, negative_log_likelihood,
    sum_squared_errors,
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


def test_fisher_se_match_least_squares_formula(fitted):
    # Regression for #512: the 'sse' covariance was s^2 H^-1 = s^2 (2 X^T X)^-1,
    # i.e. standard errors too small by sqrt(2).
    est, exps, result = fitted
    x = np.array([e.inputs["x"] for e in exps])
    y = np.array([e.observed["y"] for e in exps])
    X = np.stack([x, np.ones_like(x)], axis=1)
    resid = y - X @ np.asarray(result.theta_array)
    cov = resid @ resid / (len(y) - 2) * np.linalg.inv(X.T @ X)
    ci = est.confidence_intervals(result, exps)
    np.testing.assert_allclose([ci.std_errors[n] for n in NAMES],
                               np.sqrt(np.diag(cov)), rtol=1e-6)


def test_fisher_wsse_nll_use_known_sigma():
    # Known sigma: wsse and nll both give (X^T W X)^-1, with no s^2.
    exps = [Experiment(inputs={"x": float(x)}, observed={"y": 2 * x + 1.1},
                       uncertainties={"y": 0.2}) for x in np.linspace(0.5, 5.0, 6)]
    x = np.array([e.inputs["x"] for e in exps])
    X = np.stack([x, np.ones_like(x)], axis=1)
    expected = np.linalg.inv(X.T @ X / 0.2**2)
    est = Estimator(linear_model, NAMES)
    result = est.fit(exps, {"a": 1.0, "b": 0.0}, objective="wsse")
    for objective in ("wsse", "nll"):
        ci = est.confidence_intervals(result, exps, objective=objective)
        np.testing.assert_allclose(ci.covariance, expected, rtol=1e-6)


def test_fisher_unknown_objective_raises(fitted):
    _, exps, result = fitted
    with pytest.raises(ValueError, match="objective"):
        fisher_confidence_intervals(lambda t: jnp.sum(t**2), result.theta_array,
                                    exps, NAMES, objective="mse")
