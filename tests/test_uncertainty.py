"""Smoke tests for the uncertainty propagation module."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from difflow.uncertainty import (
    linear_propagation,
    monte_carlo_propagation,
    propagate_covariance,
    sensitivity_analysis,
    sobol_indices,
)


def model(p):
    return p["a"] * p["x"] + p["b"]


NOMINAL = {"a": 2.0, "x": 5.0, "b": 1.0}
SIGMA = {"a": 0.1, "x": 0.5, "b": 0.2}


def test_linear_propagation():
    y, y_std, info = linear_propagation(model, NOMINAL, SIGMA)
    assert abs(float(y) - 11.0) < 1e-10
    assert np.isfinite(float(y_std)) and "jacobian" in info


def test_monte_carlo_propagation():
    mean, std, info = monte_carlo_propagation(model, NOMINAL, SIGMA, n_samples=200, seed=0)
    assert np.isfinite(float(mean)) and np.isfinite(float(std))
    assert info["n_samples"] == 200


def test_sensitivity_analysis():
    res = sensitivity_analysis(model, NOMINAL, n_points=3)
    assert set(res) == set(NOMINAL)
    assert np.isfinite(float(res["a"]["gradient"]))
    assert len(res["a"]["oat_x"]) == 3


def test_sobol_indices():
    res = sobol_indices(model, {"a": (1.0, 3.0), "x": (4.0, 6.0), "b": (0.0, 2.0)},
                        n_samples=64, seed=0)
    assert res["x"]["bounds"] == (4.0, 6.0)
    assert np.isfinite(float(res["x"]["S1"]))


def test_propagate_covariance():
    y, y_cov, J = propagate_covariance(
        lambda p: p["x"] + p["y"], {"x": 5.0, "y": 10.0}, jnp.diag(jnp.array([1.0, 4.0])),
        param_order=["x", "y"])
    assert np.asarray(J).shape == (1, 2)
    assert np.all(np.isfinite(np.asarray(y_cov)))
