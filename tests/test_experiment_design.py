"""Smoke tests for model-based design of experiments in difflow.estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.estimation import (
    CRITERIA, DISCRIMINATION_CRITERIA, DesignResult, DiscriminationResult, Estimator, Experiment, IdentifiabilityError,
    IdentifiabilityReport, check_identifiability, design_criterion, design_discrimination, design_experiments,
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
        cvec = [1.0, 1.0] if c == "C" else None
        assert np.isfinite(float(design_criterion(fim, c, c=cvec)))
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


def test_design_traces_model_independent_of_pool_size():
    # Regression (#519): sensitivities were one jax.jacobian per candidate.
    counts = []
    for n in (4, 12):
        traces = []

        def model(theta, exp, traces=traces):
            traces.append(1)                  # runs once per trace
            return linear_model(theta, exp)

        design_experiments(model, THETA, line_pool(np.linspace(0, 1, n)), n=2)
        counts.append(len(traces))
    assert counts[0] == counts[1]


def test_vectorized_sensitivities_match_per_experiment():
    pool = line_pool([0.5, 1.0, 3.0]) + [Experiment.candidate({"x": 2.0}, ["y"], {"y": 0.1})]
    s = np.asarray(sensitivity_matrix(linear_model, THETA, pool))
    rows = [np.asarray(sensitivity_matrix(linear_model, THETA, [e])) for e in pool]
    np.testing.assert_allclose(s, np.concatenate(rows), rtol=1e-12)


# -- c-optimal criterion -------------------------------------------------------


def test_c_criterion_is_c_inv_fim_c():
    pool = line_pool([0.0, 1.0, 4.0])
    fim = np.asarray(fisher_information(linear_model, THETA, pool))
    c = np.array([4.0, 1.0])
    np.testing.assert_allclose(float(design_criterion(fim, "C", c=c)),
                               c @ np.linalg.inv(fim) @ c, rtol=1e-10)
    assert float(design_criterion(np.diag([1.0, 0.0]), "C", c=c)) == np.inf


def test_c_optimal_design_picks_the_point_of_interest():
    pool = line_pool([0.0, 1.0, 2.0, 3.0, 4.0])
    g = lambda th: 4.0 * th["a"] + th["b"]            # the prediction at x = 4
    res = design_experiments(linear_model, THETA, pool, n=2, criterion="C",
                             existing=pool[:1], quantity=g)
    assert [e.inputs["x"] for e in res.selected] == [4.0, 4.0]
    np.testing.assert_allclose(np.asarray(res.c), [4.0, 1.0])
    rows = np.array([[0.0, 1.0], [4.0, 1.0], [4.0, 1.0]]) / 0.5
    c = np.array([4.0, 1.0])
    np.testing.assert_allclose(res.criterion_value, c @ np.linalg.inv(rows.T @ rows) @ c,
                               rtol=1e-10)
    assert "std of the quantity" in res.summary()
    # c given directly gives the same design.
    res2 = design_experiments(linear_model, THETA, pool, n=2, criterion="C",
                              existing=pool[:1], quantity=[4.0, 1.0])
    assert res2.indices == res.indices


def test_c_criterion_validation():
    pool = line_pool([0.0, 4.0])
    with pytest.raises(ValueError, match="needs quantity"):
        design_experiments(linear_model, THETA, pool, n=1, criterion="C")
    with pytest.raises(ValueError, match="only used with criterion='C'"):
        design_experiments(linear_model, THETA, pool, n=1, quantity=[1.0, 0.0])
    with pytest.raises(ValueError, match="expected 2"):
        design_experiments(linear_model, THETA, pool, n=1, criterion="C", quantity=[1.0])
    with pytest.raises(ValueError, match="not finite"):
        design_experiments(linear_model, THETA, pool, n=1, criterion="C",
                           quantity=lambda th: jnp.sqrt(th["b"]))
    with pytest.raises(ValueError, match="needs c"):
        design_criterion(jnp.eye(2), "C")
    with pytest.raises(ValueError, match="only used with criterion 'C'"):
        design_criterion(jnp.eye(2), "D", c=[1.0, 0.0])


def test_c_criterion_grad_is_finite():
    pool = line_pool([0.0, 1.0, 2.0])
    def v(theta):
        fim = fisher_information(linear_model, theta, pool, ["a", "b"])
        return design_criterion(fim, "C", c=jnp.array([2.0, 1.0]))
    assert bool(jnp.all(jnp.isfinite(jax.grad(v)(jnp.array([1.0, 0.0])))))


# -- model discrimination -----------------------------------------------------


def lin(theta, exp):
    return {"y": theta["a"] * exp.inputs["x"]}


def quad(theta, exp):
    return {"y": theta["b"] * exp.inputs["x"] ** 2}


def test_hunter_reiner_picks_largest_difference():
    pool = line_pool([0.0, 0.5, 1.0, 2.0])                 # sigma 0.5
    res = design_discrimination([lin, quad], [{"a": 1.0}, {"b": 1.0}], pool, n=2)
    assert isinstance(res, DiscriminationResult) and res.criterion == "HR"
    assert res.indices == [3, 3]                          # x = 2: 2 against 4
    np.testing.assert_allclose(res.scores, [0.0, 0.25 ** 2, 0.0, 4.0] / np.array(0.25))
    np.testing.assert_allclose(res.criterion_value, 2 * (2.0 ** 2) / 0.25)
    np.testing.assert_allclose(res.predictions["quad"][:, 0], [0.0, 0.25, 1.0, 4.0])
    assert isinstance(res.summary(), str)
    distinct = design_discrimination({"l": lin, "q": quad}, {"l": {"a": 1.0}, "q": {"b": 1.0}},
                                     pool, n=2, criterion="hunter_reiner", replace=False)
    assert distinct.indices == [3, 1] and distinct.model_names == ["l", "q"]


def test_buzzi_ferraris():
    pool = line_pool([0.5, 1.0, 2.0])
    hr = design_discrimination([lin, quad], [{"a": 1.0}, {"b": 1.0}], pool)
    bf0 = design_discrimination([lin, quad], [{"a": 1.0}, {"b": 1.0}], pool, criterion="BF",
                                covariances=[np.zeros((1, 1)), np.zeros((1, 1))])
    np.testing.assert_allclose(bf0.scores, hr.scores / 2)    # no parameter uncertainty
    bf = design_discrimination([lin, quad], [{"a": 1.0}, {"b": 1.0}], pool, n=2, criterion="BF",
                               existing=line_pool([1.5]))
    assert set(DISCRIMINATION_CRITERIA) == {"HR", "BF"}
    assert np.all(bf.scores <= bf0.scores + 1e-12) and np.isfinite(bf.criterion_value)


def test_discrimination_validation():
    pool = line_pool([1.0, 2.0])
    th = [{"a": 1.0}, {"b": 1.0}]
    with pytest.raises(ValueError, match="at least two"):
        design_discrimination([lin], th[:1], pool)
    with pytest.raises(ValueError, match="parameter sets"):
        design_discrimination([lin, quad], th[:1], pool)
    with pytest.raises(ValueError, match="unknown discrimination criterion"):
        design_discrimination([lin, quad], th, pool, criterion="D")
    with pytest.raises(ValueError, match="empty"):
        design_discrimination([lin, quad], th, [])
    with pytest.raises(ValueError, match="n must be positive"):
        design_discrimination([lin, quad], th, pool, n=0)
    with pytest.raises(ValueError, match="without replacement"):
        design_discrimination([lin, quad], th, pool, n=3, replace=False)
    with pytest.raises(ValueError, match="covariance"):
        design_discrimination([lin, quad], th, pool, criterion="BF")
    with pytest.raises(ValueError, match="shape"):
        design_discrimination([lin, quad], th, pool, criterion="BF",
                              covariances=[np.eye(2), np.eye(1)])
