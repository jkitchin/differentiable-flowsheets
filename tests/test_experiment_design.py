"""Smoke tests for model-based design of experiments in difflow.estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.estimation import (
    CRITERIA, DesignResult, Estimator, Experiment, IdentifiabilityError,
    IdentifiabilityReport, check_identifiability, design_criterion, design_experiments,
    fisher_information, log_det, predicted_covariance, sensitivity_matrix,
)

THETA = {"a": 1.0, "b": 0.0}


def linear_model(theta, exp):
    return {"y": theta["a"] * exp.inputs["x"] + theta["b"]}


def product_model(theta, exp):
    """A and B are not separately identifiable."""
    return {"y": theta["A"] * theta["B"] * exp.inputs["x"]}


def line_pool(xs):
    return [Experiment.candidate({"x": float(x)}, ["y"], {"y": 0.5}) for x in xs]


def test_candidate_experiment():
    c = Experiment.candidate({"x": 1.0}, ["y", "z"], {"y": 0.25})
    assert c.is_candidate and c.observed == {} and c.measured_names == ["y", "z"]
    np.testing.assert_allclose(np.asarray(c.sigma_array), [0.25, 1.0])


def test_sensitivity_fim_and_criteria():
    pool = line_pool([1.0, 3.0])
    assert np.asarray(sensitivity_matrix(linear_model, THETA, pool)).shape == (2, 2)
    fim = fisher_information(linear_model, THETA, pool)
    assert fim.shape == (2, 2) and np.isfinite(float(log_det(fim)))
    for c in CRITERIA:
        assert np.isfinite(float(design_criterion(fim, c)))
    with pytest.raises(ValueError, match="unknown criterion"):
        design_criterion(jnp.eye(2), "Z")


def test_design_and_predicted_covariance():
    pool = line_pool(np.linspace(0.0, 10.0, 6))
    res = design_experiments(linear_model, THETA, pool, n=3)
    assert isinstance(res, DesignResult) and len(res.selected) == 3
    assert res.identifiability.identifiable and isinstance(res.summary(), str)
    ci = predicted_covariance(linear_model, THETA, res.selected)
    assert np.all(np.isfinite(np.asarray(ci.covariance)))
    with pytest.raises(ValueError, match="empty"):
        design_experiments(linear_model, THETA, [], n=2)
    with pytest.raises(ValueError, match="n must be positive"):
        design_experiments(linear_model, THETA, pool, n=0)
    with pytest.raises(ValueError, match="without replacement"):
        design_experiments(linear_model, THETA, pool, n=99, replace=False)


def test_identifiability():
    pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 5)]
    rep = check_identifiability(product_model, {"A": 2.0, "B": 3.0}, pool)
    assert isinstance(rep, IdentifiabilityReport) and not rep.identifiable
    assert isinstance(rep.summary(), str)
    with pytest.raises(IdentifiabilityError):
        rep.raise_if_unidentifiable()
    with pytest.raises(IdentifiabilityError):
        design_experiments(product_model, {"A": 2.0, "B": 3.0}, pool, n=2)


def test_criterion_grad_is_finite():
    pool = line_pool([0.0, 1.0, 2.0])
    def d(theta):
        return design_criterion(fisher_information(linear_model, theta, pool, ["a", "b"]), "D")
    g = jax.grad(jax.jit(d))(jnp.array([1.0, 0.0]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_estimator_delegates():
    est = Estimator(linear_model, ["a", "b"])
    pool = line_pool(np.linspace(0.0, 10.0, 6))
    assert est.check_identifiability(THETA, pool).identifiable
    design = est.design_experiments(THETA, pool, n=2)
    assert est.predicted_covariance(THETA, design.selected).std_errors["a"] > 0
