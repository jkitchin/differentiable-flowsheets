"""Tests and examples for diffrax backend integration.

This module provides comprehensive tests for the diffrax integration,
serving both as tests and as usage examples.

Examples include:
1. Basic ODE integration with different solvers
2. Stiff system integration
3. Chemical reaction kinetics (CSTR simulation)
4. Gradient computation through integration
5. Comparison of different solvers
"""

import pytest
import jax
import jax.numpy as jnp
from jax import Array

from difflow.dynamic import (
    integrate,
    integrate_unit,
    DynamicCSTR,
    DynamicFlowsheet,
    IntegrationResult,
)
from difflow.streams import make_stream

# Check if diffrax is available
try:
    from difflow.dynamic import (
        HAS_DIFFRAX,
        integrate_diffrax,
        integrate_diffrax_unit,
        integrate_stiff,
        integrate_dopri5,
        integrate_tsit5,
        list_diffrax_solvers,
        check_diffrax_available,
    )
except ImportError:
    HAS_DIFFRAX = False

# Skip all tests if diffrax not installed
pytestmark = pytest.mark.skipif(
    not HAS_DIFFRAX,
    reason="diffrax not installed"
)


# =============================================================================
# Example 1: Basic ODE - Harmonic Oscillator
# =============================================================================

class TestBasicODE:
    """Example: Solving a simple harmonic oscillator.

    System: d²x/dt² = -x
    As first order: dx/dt = v, dv/dt = -x

    This is a classic test case because we know the exact solution:
    x(t) = x0*cos(t) + v0*sin(t)
    """

    def test_harmonic_oscillator_dopri5(self):
        """Solve harmonic oscillator with Dopri5."""
        def harmonic(t, y):
            x, v = y[0], y[1]
            return jnp.array([v, -x])

        y0 = jnp.array([1.0, 0.0])  # x=1, v=0
        t_span = (0.0, 2 * jnp.pi)  # One full period

        result = integrate(harmonic, y0, t_span, method="diffrax:dopri5")

        # After one period, should return to initial state
        assert jnp.allclose(result.y_final, y0, atol=1e-4)
        assert result.info.success

    def test_harmonic_oscillator_tsit5(self):
        """Solve harmonic oscillator with Tsit5 (recommended solver)."""
        def harmonic(t, y):
            return jnp.array([y[1], -y[0]])

        y0 = jnp.array([1.0, 0.0])
        result = integrate(harmonic, y0, (0.0, 10.0), method="diffrax:tsit5")

        # Energy should be conserved: E = (x² + v²)/2 = 0.5
        x_final, v_final = result.y_final
        E_final = 0.5 * (x_final**2 + v_final**2)
        assert jnp.allclose(E_final, 0.5, atol=1e-4)

    def test_default_diffrax_solver(self):
        """Using 'diffrax' without specifying solver uses Tsit5."""
        def harmonic(t, y):
            return jnp.array([y[1], -y[0]])

        y0 = jnp.array([1.0, 0.0])
        result = integrate(harmonic, y0, (0.0, 5.0), method="diffrax")

        assert jnp.all(jnp.isfinite(result.y_final))
        assert result.info.success


# =============================================================================
# Example 2: Stiff System - Chemical Kinetics
# =============================================================================

class TestStiffSystem:
    """Example: Stiff chemical reaction system.

    Robertson problem - a classic stiff test case:
    A -> B  (fast, k1=0.04)
    B + B -> C + B  (very fast, k2=3e7)
    B + C -> A + C  (moderate, k3=1e4)

    The very different rate constants make this stiff.
    """

    def test_robertson_problem_kvaerno5(self):
        """Solve Robertson problem with implicit Kvaerno5."""
        # Rate constants
        k1 = 0.04
        k2 = 3e7
        k3 = 1e4

        def robertson(t, y):
            A, B, C = y[0], y[1], y[2]
            dA = -k1*A + k3*B*C
            dB = k1*A - k2*B*B - k3*B*C
            dC = k2*B*B
            return jnp.array([dA, dB, dC])

        y0 = jnp.array([1.0, 0.0, 0.0])

        # Use implicit solver for stiff system
        result = integrate(
            robertson, y0,
            t_span=(0.0, 1e5),
            method="diffrax:kvaerno5",
            rtol=1e-4,
            atol=1e-6,
        )

        # Mass conservation: A + B + C = 1
        total = jnp.sum(result.y_final)
        assert jnp.allclose(total, 1.0, atol=1e-3)

    def test_integrate_stiff_convenience(self):
        """Use integrate_stiff convenience function."""
        def stiff_decay(t, y):
            # y' = -1000*y (very stiff)
            return -1000 * y

        y0 = jnp.array([1.0])
        result = integrate_stiff(stiff_decay, y0, (0.0, 0.1))

        # Should decay to near zero
        assert result.y_final[0] < 1e-10


# =============================================================================
# Example 3: Chemical Reactor (CSTR)
# =============================================================================

class TestCSTRExample:
    """Example: Dynamic CSTR simulation with diffrax.

    Simulates a continuous stirred tank reactor with first-order
    reaction A -> B using diffrax adaptive solver.
    """

    @pytest.fixture
    def cstr_setup(self):
        """Create CSTR and feed stream."""
        def rate_fn(C, T, params):
            k = params.get("k", 0.1)
            return jnp.array([k * C["A"]])

        cstr = DynamicCSTR(
            volume=1.0,
            rate_fn=rate_fn,
            stoich=jnp.array([[-1.0], [1.0]]),  # A -> B
            species_order=["A", "B"],
            rate_params={"k": 0.1},
            name="reactor",
        )

        feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

        return cstr, feed

    def test_cstr_dopri5(self, cstr_setup):
        """Simulate CSTR with Dopri5."""
        cstr, feed = cstr_setup

        result = integrate_diffrax_unit(
            cstr,
            inputs={"inlet": feed},
            t_span=(0.0, 100.0),
            solver="dopri5",
            rtol=1e-5,
        )

        assert jnp.all(jnp.isfinite(result.y_final))
        # Product B should increase
        assert result.y_final[1] > 0

    def test_cstr_via_integrate(self, cstr_setup):
        """Use generic integrate() with diffrax for CSTR."""
        cstr, feed = cstr_setup

        result = integrate_unit(
            cstr,
            inputs={"inlet": feed},
            t_span=(0.0, 100.0),
            method="diffrax:tsit5",
            rtol=1e-5,
        )

        assert result.info.success

    def test_cstr_compare_solvers(self, cstr_setup):
        """Compare different solvers on same CSTR problem."""
        cstr, feed = cstr_setup
        inputs = {"inlet": feed}
        t_span = (0.0, 50.0)

        # Get initial state
        y0 = cstr.initial_state(inputs)

        def f(t, y):
            return cstr.derivatives(t, y, inputs)

        # Compare solvers
        results = {}
        for solver in ["dopri5", "tsit5", "heun"]:
            results[solver] = integrate(f, y0, t_span, method=f"diffrax:{solver}")

        # All should give similar results
        ref = results["dopri5"].y_final
        for solver, result in results.items():
            assert jnp.allclose(result.y_final, ref, atol=0.1), f"{solver} differs"


# =============================================================================
# Example 4: Gradient Computation
# =============================================================================

class TestGradientsThroughDiffrax:
    """Example: Computing gradients through diffrax integration.

    This is a key use case - optimizing parameters by differentiating
    through the ODE solution.
    """

    def test_gradient_wrt_initial_condition(self):
        """Compute gradient of final state w.r.t. initial condition."""
        def exponential_decay(t, y):
            return -0.5 * y

        def loss(y0):
            result = integrate(
                exponential_decay, y0,
                t_span=(0.0, 2.0),
                method="diffrax:tsit5",
            )
            return jnp.sum(result.y_final**2)

        y0 = jnp.array([1.0])
        grad = jax.grad(loss)(y0)

        assert jnp.isfinite(grad[0])
        # Gradient should be positive (larger y0 -> larger final value)
        assert grad[0] > 0

    def test_gradient_wrt_parameter(self):
        """Optimize reaction rate parameter using gradient descent."""
        def make_rate_system(k):
            def f(t, y):
                return -k * y
            return f

        def loss(k):
            """Loss: final value after decay should equal target."""
            f = make_rate_system(k)
            y0 = jnp.array([1.0])
            result = integrate(f, y0, (0.0, 1.0), method="diffrax:dopri5")
            target = 0.5  # Want y(1) = 0.5
            return (result.y_final[0] - target)**2

        # Compute gradient
        k_init = jnp.array(0.5)
        grad = jax.grad(loss)(k_init)

        assert jnp.isfinite(grad)


# =============================================================================
# Example 5: Flowsheet with Diffrax
# =============================================================================

class TestFlowsheetWithDiffrax:
    """Example: DynamicFlowsheet simulation using diffrax."""

    def test_flowsheet_diffrax(self):
        """Simulate a multi-unit flowsheet with diffrax."""
        from difflow.dynamic import DynamicTank

        # Create units
        def rate_fn(C, T, params):
            return jnp.array([params["k"] * C["A"]])

        cstr = DynamicCSTR(
            volume=1.0,
            rate_fn=rate_fn,
            stoich=jnp.array([[-1.0], [1.0]]),
            species_order=["A", "B"],
            rate_params={"k": 0.1},
            name="reactor",
        )

        tank = DynamicTank(
            max_volume=10.0,
            species_order=["A", "B"],
            name="storage",
        )

        # Build flowsheet
        fs = DynamicFlowsheet(species_order=["A", "B"])
        feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
        fs.add_feed("feed", feed)
        fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(tank, inlet_names=["reactor_out"], outlet_names=["product"])

        # Simulate with diffrax
        y0 = fs.initial_state()

        def f(t, y):
            return fs.derivatives(t, y)

        result = integrate(f, y0, (0.0, 100.0), method="diffrax:tsit5")

        assert jnp.all(jnp.isfinite(result.y_final))


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_list_solvers(self):
        """Can list available diffrax solvers."""
        solvers = list_diffrax_solvers()
        assert "dopri5" in solvers
        assert "tsit5" in solvers
        assert "kvaerno5" in solvers

    def test_check_available(self):
        """check_diffrax_available returns True when installed."""
        assert check_diffrax_available() is True

    def test_invalid_solver_raises(self):
        """Invalid solver name raises clear error."""
        def f(t, y):
            return -y

        with pytest.raises(ValueError, match="Unknown diffrax solver"):
            integrate(f, jnp.array([1.0]), (0, 1), method="diffrax:invalid_solver")


# =============================================================================
# Solver Comparison
# =============================================================================

class TestSolverComparison:
    """Compare accuracy and speed of different solvers."""

    def test_accuracy_comparison(self):
        """Compare accuracy of adaptive solvers on known problem."""
        # Exponential decay: y' = -y, y(0) = 1
        # Exact solution: y(t) = exp(-t)

        def decay(t, y):
            return -y

        y0 = jnp.array([1.0])
        t_final = 2.0
        exact = jnp.exp(-t_final)

        errors = {}
        # Only test adaptive solvers (Euler/Heun need constant stepsize)
        for solver in ["dopri5", "tsit5", "dopri8"]:
            result = integrate(
                decay, y0, (0.0, t_final),
                method=f"diffrax:{solver}",
                rtol=1e-6, atol=1e-8,
            )
            errors[solver] = float(jnp.abs(result.y_final[0] - exact))

        # All should be very accurate with these tolerances
        for solver, error in errors.items():
            assert error < 1e-4, f"{solver} error {error} too large"

        # Print for informational purposes
        print(f"Errors: {errors}")

    def test_solver_selection_guide(self):
        """Demonstrate when to use different solvers."""
        # Non-stiff: Use tsit5 or dopri5
        def simple_ode(t, y):
            return jnp.cos(t) - y

        result = integrate(
            simple_ode,
            jnp.array([0.0]),
            (0.0, 10.0),
            method="diffrax:tsit5",  # Recommended for most problems
        )
        assert result.info.success

        # Stiff: Use kvaerno5
        def stiff_ode(t, y):
            return -100 * (y - jnp.sin(t)) + jnp.cos(t)

        result = integrate(
            stiff_ode,
            jnp.array([0.0]),
            (0.0, 10.0),
            method="diffrax:kvaerno5",  # For stiff problems
        )
        assert result.info.success
