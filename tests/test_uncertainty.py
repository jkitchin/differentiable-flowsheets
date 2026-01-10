"""Tests for uncertainty propagation module.

Tests cover:
- Linear propagation with Jacobian
- Monte Carlo propagation with sampling
- Sensitivity analysis (local and OAT)
- Sobol sensitivity indices
- Covariance propagation
"""

import pytest
import jax
import jax.numpy as jnp
from jax import Array
import numpy.testing as npt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from difflow.uncertainty import (
    linear_propagation,
    monte_carlo_propagation,
    sensitivity_analysis,
    sobol_indices,
    propagate_covariance,
)


class TestLinearPropagation:
    """Tests for linear uncertainty propagation."""

    def test_linear_model(self):
        """Test linear propagation on a linear model (exact)."""
        # y = a*x + b
        def linear_model(params):
            return params["a"] * params["x"] + params["b"]

        nominal = {"a": 2.0, "x": 5.0, "b": 1.0}
        sigma = {"a": 0.1, "x": 0.5, "b": 0.2}

        y, y_std, info = linear_propagation(linear_model, nominal, sigma)

        # Expected: y = 2*5 + 1 = 11
        assert abs(y - 11.0) < 1e-10

        # For linear model, exact variance is:
        # Var(y) = x^2 * sigma_a^2 + a^2 * sigma_x^2 + sigma_b^2
        # = 25*0.01 + 4*0.25 + 0.04 = 0.25 + 1.0 + 0.04 = 1.29
        expected_std = jnp.sqrt(25 * 0.01 + 4 * 0.25 + 0.04)
        assert abs(y_std - float(expected_std)) < 1e-10

    def test_quadratic_model(self):
        """Test linear propagation on a quadratic model."""
        # y = x^2
        def quadratic(params):
            return params["x"] ** 2

        nominal = {"x": 10.0}
        sigma = {"x": 1.0}

        y, y_std, info = linear_propagation(quadratic, nominal, sigma)

        # y = 100
        assert abs(y - 100.0) < 1e-10

        # dy/dx = 2x = 20, so sigma_y = 20 * sigma_x = 20
        assert abs(y_std - 20.0) < 1e-10

        # Check gradient in info
        assert "jacobian" in info
        # Jacobian is 2D array even for scalar-in, scalar-out
        assert jnp.abs(info["jacobian"].flatten()[0] - 20.0) < 1e-10

    def test_multivariate_output(self):
        """Test propagation with multiple outputs."""
        def multi_output(params):
            x = params["x"]
            return jnp.array([x, x**2, x**3])

        nominal = {"x": 2.0}
        sigma = {"x": 0.1}

        y, y_std, info = linear_propagation(multi_output, nominal, sigma)

        # Expected outputs
        npt.assert_allclose(y, [2.0, 4.0, 8.0], rtol=1e-10)

        # Gradients: [1, 2x, 3x^2] = [1, 4, 12]
        # sigma_y = |grad| * sigma_x = [0.1, 0.4, 1.2]
        npt.assert_allclose(y_std, [0.1, 0.4, 1.2], rtol=1e-10)

    def test_sensitivity_info(self):
        """Test that sensitivity info is computed correctly."""
        def model(params):
            return params["a"] * params["b"]

        nominal = {"a": 2.0, "b": 3.0}
        sigma = {"a": 0.2, "b": 0.3}

        y, y_std, info = linear_propagation(model, nominal, sigma)

        assert "sensitivities" in info
        assert "a" in info["sensitivities"]
        assert "b" in info["sensitivities"]

        # Check gradient values: dy/da = b = 3, dy/db = a = 2
        grad_a = jnp.atleast_1d(info["sensitivities"]["a"]["gradient"])
        grad_b = jnp.atleast_1d(info["sensitivities"]["b"]["gradient"])
        assert jnp.abs(grad_a[0] - 3.0) < 1e-10
        assert jnp.abs(grad_b[0] - 2.0) < 1e-10

        # Check elasticity: (dy/da)*(a/y) = 3*2/6 = 1.0
        # (dy/db)*(b/y) = 2*3/6 = 1.0
        elast_a = jnp.atleast_1d(info["sensitivities"]["a"]["elasticity"])
        elast_b = jnp.atleast_1d(info["sensitivities"]["b"]["elasticity"])
        assert jnp.abs(elast_a[0] - 1.0) < 1e-10
        assert jnp.abs(elast_b[0] - 1.0) < 1e-10


class TestMonteCarloPropagation:
    """Tests for Monte Carlo uncertainty propagation."""

    def test_normal_distribution(self):
        """Test MC propagation with normal distribution."""
        def linear_model(params):
            return params["x"] + params["y"]

        nominal = {"x": 10.0, "y": 20.0}
        sigma = {"x": 1.0, "y": 2.0}

        mean, std, info = monte_carlo_propagation(
            linear_model, nominal, sigma,
            n_samples=10000, seed=42
        )

        # Expected mean = 30, std = sqrt(1 + 4) = sqrt(5) ≈ 2.236
        assert abs(mean - 30.0) < 0.5  # Statistical tolerance
        assert abs(std - jnp.sqrt(5.0)) < 0.2

        assert info["n_samples"] == 10000
        assert info["distribution"] == "normal"

    def test_uniform_distribution(self):
        """Test MC propagation with uniform distribution."""
        def identity(params):
            return params["x"]

        nominal = {"x": 10.0}
        sigma = {"x": 1.0}  # ±3 sigma range

        mean, std, info = monte_carlo_propagation(
            identity, nominal, sigma,
            n_samples=10000,
            distribution="uniform",
            seed=42
        )

        # Uniform over [-3, 3] has std = sqrt(12)/2 * sigma ≈ 1.73 * sigma
        assert abs(mean - 10.0) < 0.3
        assert info["distribution"] == "uniform"

    def test_percentiles(self):
        """Test that percentiles are computed correctly."""
        def identity(params):
            return params["x"]

        nominal = {"x": 0.0}
        sigma = {"x": 1.0}

        _, _, info = monte_carlo_propagation(
            identity, nominal, sigma,
            n_samples=10000, seed=42
        )

        # For standard normal, 2.5th percentile ≈ -1.96, 97.5th ≈ 1.96
        assert float(jnp.atleast_1d(info["p2.5"])[0]) < -1.5
        assert float(jnp.atleast_1d(info["p97.5"])[0]) > 1.5
        assert abs(float(jnp.atleast_1d(info["p50"])[0])) < 0.2  # Median close to 0

    def test_return_samples(self):
        """Test returning samples."""
        def model(params):
            return params["x"]

        nominal = {"x": 5.0}
        sigma = {"x": 1.0}

        _, _, info = monte_carlo_propagation(
            model, nominal, sigma,
            n_samples=100,
            return_samples=True,
            seed=42
        )

        assert "samples" in info
        assert "input_samples" in info
        # Samples are (n_samples, n_outputs) - atleast_1d wraps scalar outputs
        assert info["samples"].shape[0] == 100
        assert info["input_samples"].shape == (100, 1)

    def test_nonlinear_model(self):
        """Test MC vs linear propagation for nonlinear model."""
        def nonlinear(params):
            return jnp.exp(params["x"])

        nominal = {"x": 0.0}
        sigma = {"x": 0.5}  # Large sigma makes linear approx inaccurate

        # Linear propagation
        y_lin, std_lin, _ = linear_propagation(nonlinear, nominal, sigma)

        # MC propagation (more accurate for nonlinear)
        mean_mc, std_mc, _ = monte_carlo_propagation(
            nonlinear, nominal, sigma,
            n_samples=10000, seed=42
        )

        # For exp(x) with x ~ N(0, 0.5), true mean = exp(0 + 0.5^2/2) = exp(0.125) ≈ 1.133
        # Linear approximation gives 1.0
        # MC should be closer to true value
        assert abs(mean_mc - jnp.exp(0.125)) < 0.1


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis."""

    def test_local_sensitivity(self):
        """Test local sensitivity (gradient) computation."""
        def model(params):
            return params["a"] ** 2 + 2 * params["b"]

        nominal = {"a": 3.0, "b": 2.0}

        results = sensitivity_analysis(model, nominal)

        # dy/da = 2a = 6, dy/db = 2
        assert abs(results["a"]["gradient"] - 6.0) < 1e-8
        assert abs(results["b"]["gradient"] - 2.0) < 1e-8

    def test_elasticity(self):
        """Test normalized sensitivity (elasticity)."""
        def model(params):
            return params["x"] ** 2

        nominal = {"x": 5.0}

        results = sensitivity_analysis(model, nominal)

        # y = 25, dy/dx = 10
        # elasticity = (dy/dx) * (x/y) = 10 * 5/25 = 2
        assert abs(results["x"]["elasticity"] - 2.0) < 1e-8

    def test_oat_curves(self):
        """Test one-at-a-time sensitivity curves."""
        def model(params):
            return params["x"]

        nominal = {"x": 10.0}
        ranges = {"x": (5.0, 15.0)}

        results = sensitivity_analysis(model, nominal, param_ranges=ranges, n_points=5)

        # OAT should give x values from 5 to 15
        assert len(results["x"]["oat_x"]) == 5
        npt.assert_allclose(results["x"]["oat_x"], [5, 7.5, 10, 12.5, 15], rtol=1e-10)
        npt.assert_allclose(results["x"]["oat_y"], [5, 7.5, 10, 12.5, 15], rtol=1e-10)

    def test_default_ranges(self):
        """Test default ±20% parameter ranges."""
        def model(params):
            return params["x"]

        nominal = {"x": 100.0}

        results = sensitivity_analysis(model, nominal, n_points=3)

        # Default range: 0.8*100 to 1.2*100 = [80, 120]
        npt.assert_allclose(results["x"]["oat_x"], [80, 100, 120], rtol=1e-10)


class TestSobolIndices:
    """Tests for Sobol sensitivity indices."""

    def test_additive_model(self):
        """Test Sobol indices for additive model (should sum to ~1)."""
        def additive(params):
            return params["a"] + 2 * params["b"]

        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}

        results = sobol_indices(additive, bounds, n_samples=2048, seed=42)

        # For y = a + 2b:
        # Var(a) = 1/12, Var(2b) = 4/12 = 1/3
        # Total Var = 1/12 + 1/3 = 5/12
        # S_a = (1/12) / (5/12) = 0.2
        # S_b = (1/3) / (5/12) = 0.8
        assert abs(results["a"]["S1"] - 0.2) < 0.15
        assert abs(results["b"]["S1"] - 0.8) < 0.15

        # Sum should be close to 1 for additive model
        total = results["a"]["S1"] + results["b"]["S1"]
        assert abs(total - 1.0) < 0.2

    def test_single_important_param(self):
        """Test model where one parameter dominates."""
        def dominated(params):
            return params["important"] ** 2 + 0.001 * params["minor"]

        bounds = {"important": (0.0, 10.0), "minor": (0.0, 1.0)}

        results = sobol_indices(dominated, bounds, n_samples=1024, seed=42)

        # 'important' should have much higher S1 than 'minor'
        assert results["important"]["S1"] > results["minor"]["S1"]

    def test_bounds_recorded(self):
        """Test that bounds are recorded in results."""
        def model(params):
            return params["x"]

        bounds = {"x": (5.0, 15.0)}

        results = sobol_indices(model, bounds, n_samples=256, seed=42)

        assert results["x"]["bounds"] == (5.0, 15.0)


class TestPropagateCovariance:
    """Tests for full covariance propagation."""

    def test_uncorrelated(self):
        """Test propagation with diagonal covariance (uncorrelated)."""
        def model(params):
            return params["x"] + params["y"]

        nominal = {"x": 5.0, "y": 10.0}
        cov = jnp.array([
            [1.0, 0.0],  # Var(x) = 1, Cov(x,y) = 0
            [0.0, 4.0],  # Cov(y,x) = 0, Var(y) = 4
        ])

        y, y_cov, jacobian = propagate_covariance(
            model, nominal, cov, param_order=["x", "y"]
        )

        # y = 15
        npt.assert_allclose(y, [15.0], rtol=1e-10)

        # Jacobian = [1, 1]
        npt.assert_allclose(jacobian, [[1.0, 1.0]], rtol=1e-10)

        # Output variance = J @ cov @ J.T = [1,1] @ [[1,0],[0,4]] @ [1,1].T = 1 + 4 = 5
        npt.assert_allclose(y_cov, [[5.0]], rtol=1e-10)

    def test_correlated(self):
        """Test propagation with correlated inputs."""
        def model(params):
            return params["x"] - params["y"]

        nominal = {"x": 10.0, "y": 10.0}
        # Positively correlated: when x goes up, y goes up
        cov = jnp.array([
            [1.0, 0.8],
            [0.8, 1.0],
        ])

        y, y_cov, jacobian = propagate_covariance(
            model, nominal, cov, param_order=["x", "y"]
        )

        # Jacobian = [1, -1] for x - y
        npt.assert_allclose(jacobian, [[1.0, -1.0]], rtol=1e-10)

        # Output variance = [1,-1] @ [[1,0.8],[0.8,1]] @ [1,-1].T
        # = 1*1*1 + 1*(-1)*0.8 + (-1)*1*0.8 + (-1)*(-1)*1 = 1 - 0.8 - 0.8 + 1 = 0.4
        npt.assert_allclose(y_cov, [[0.4]], rtol=1e-10)

    def test_multivariate_output(self):
        """Test with multiple outputs."""
        def model(params):
            x = params["x"]
            return jnp.array([x, 2*x])

        nominal = {"x": 5.0}
        cov = jnp.array([[4.0]])  # Var(x) = 4

        y, y_cov, jacobian = propagate_covariance(
            model, nominal, cov, param_order=["x"]
        )

        # Jacobian = [[1], [2]]
        npt.assert_allclose(jacobian, [[1.0], [2.0]], rtol=1e-10)

        # Output covariance = J @ cov @ J.T = [[1],[2]] @ [[4]] @ [[1,2]]
        # = [[4, 8], [8, 16]]
        npt.assert_allclose(y_cov, [[4.0, 8.0], [8.0, 16.0]], rtol=1e-10)


class TestGradientCompatibility:
    """Test that uncertainty functions work with JAX autodiff."""

    def test_mc_propagation_uses_jax_random(self):
        """Test that MC propagation uses JAX random (deterministic with seed)."""
        def model(params):
            return params["x"] ** 2

        nominal = {"x": 5.0}
        sigma = {"x": 1.0}

        # Two runs with same seed should give identical results
        mean1, std1, _ = monte_carlo_propagation(
            model, nominal, sigma, n_samples=100, seed=42
        )
        mean2, std2, _ = monte_carlo_propagation(
            model, nominal, sigma, n_samples=100, seed=42
        )

        assert abs(mean1 - mean2) < 1e-10
        assert abs(std1 - std2) < 1e-10

    def test_sobol_is_deterministic(self):
        """Test that Sobol indices are deterministic with seed."""
        def model(params):
            return params["x"] + params["y"]

        bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

        result1 = sobol_indices(model, bounds, n_samples=256, seed=123)
        result2 = sobol_indices(model, bounds, n_samples=256, seed=123)

        assert result1["x"]["S1"] == result2["x"]["S1"]
        assert result1["y"]["S1"] == result2["y"]["S1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
