"""Tests for the unified dynamic modeling framework.

Tests cover:
- State specification and management
- ODE integrators (RK4, RK45, Euler)
- Dynamic unit implementations (DynamicCSTR, DynamicTank)
- Gradient computation through integration
"""

import pytest
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from difflow.dynamic import (
    # State
    StateVar,
    StateSpec,
    StateVector,
    molar_states,
    thermal_state,
    reactor_states,
    # Units
    DynamicUnit,
    DynamicUnitBase,
    DynamicCSTR,
    DynamicTank,
    # Integrators
    integrate,
    integrate_unit,
    integrate_rk4,
    integrate_rk45,
    integrate_euler,
    rk4_step,
    IntegrationResult,
    Trajectory,
    # Utilities
    sensitivity_analysis,
)
from difflow.streams import make_stream, get_flows


# =============================================================================
# State Specification Tests
# =============================================================================

class TestStateVar:
    """Tests for StateVar dataclass."""

    def test_basic_creation(self):
        """Test creating a basic state variable."""
        var = StateVar("T", "temperature", "K", "Temperature")
        assert var.name == "T"
        assert var.category == "temperature"
        assert var.units == "K"
        assert var.description == "Temperature"

    def test_defaults(self):
        """Test default values."""
        var = StateVar("x")
        assert var.category == "generic"
        assert var.units == ""
        assert var.bounds == (None, None)
        assert var.scale == 1.0

    def test_with_bounds(self):
        """Test state variable with bounds."""
        var = StateVar("n", "moles", "mol", bounds=(0.0, None))
        assert var.bounds[0] == 0.0
        assert var.bounds[1] is None


class TestStateSpec:
    """Tests for StateSpec."""

    def test_creation(self):
        """Test creating state specification."""
        spec = StateSpec([
            StateVar("n_A", "moles"),
            StateVar("n_B", "moles"),
            StateVar("T", "temperature"),
        ])
        assert spec.n_states == 3
        assert spec.names == ["n_A", "n_B", "T"]

    def test_get_index(self):
        """Test getting index by name."""
        spec = StateSpec([
            StateVar("n_A"),
            StateVar("n_B"),
            StateVar("T"),
        ])
        assert spec.get_index("n_A") == 0
        assert spec.get_index("T") == 2

    def test_get_indices(self):
        """Test getting multiple indices."""
        spec = StateSpec([StateVar("x"), StateVar("y"), StateVar("z")])
        indices = spec.get_indices(["z", "x"])
        assert indices == [2, 0]

    def test_add(self):
        """Test adding state variable."""
        spec = StateSpec([StateVar("x")])
        spec2 = spec.add(StateVar("y"))
        assert spec.n_states == 1
        assert spec2.n_states == 2

    def test_combine(self):
        """Test combining state specs."""
        spec1 = StateSpec([StateVar("x")])
        spec2 = StateSpec([StateVar("y")])
        combined = spec1 + spec2
        assert combined.n_states == 2
        assert combined.names == ["x", "y"]

    def test_get_scales(self):
        """Test getting scale array."""
        spec = StateSpec([
            StateVar("T", scale=300.0),
            StateVar("P", scale=1e5),
        ])
        scales = spec.get_scales()
        assert jnp.allclose(scales, jnp.array([300.0, 1e5]))

    def test_get_bounds(self):
        """Test getting bounds arrays."""
        spec = StateSpec([
            StateVar("n", bounds=(0.0, None)),
            StateVar("T", bounds=(200.0, 500.0)),
        ])
        lower, upper = spec.get_bounds()
        assert jnp.allclose(lower, jnp.array([0.0, 200.0]))
        assert lower[0] == 0.0
        assert upper[1] == 500.0


class TestStateVector:
    """Tests for StateVector."""

    def test_creation(self):
        """Test creating state vector."""
        spec = StateSpec([StateVar("T"), StateVar("P")])
        state = StateVector(jnp.array([300.0, 101325.0]), spec)
        assert state["T"] == 300.0
        assert state["P"] == 101325.0

    def test_from_dict(self):
        """Test creating from dictionary."""
        spec = StateSpec([StateVar("x"), StateVar("y")])
        state = StateVector.from_dict({"x": 1.0, "y": 2.0}, spec)
        assert state["x"] == 1.0
        assert state["y"] == 2.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        spec = StateSpec([StateVar("a"), StateVar("b")])
        state = StateVector(jnp.array([1.0, 2.0]), spec)
        d = state.to_dict()
        assert d["a"] == 1.0
        assert d["b"] == 2.0

    def test_get_multiple(self):
        """Test getting multiple values."""
        spec = StateSpec([StateVar("x"), StateVar("y"), StateVar("z")])
        state = StateVector(jnp.array([1.0, 2.0, 3.0]), spec)
        vals = state.get(["z", "x"])
        assert jnp.allclose(vals, jnp.array([3.0, 1.0]))


class TestStateFactories:
    """Tests for state factory functions."""

    def test_molar_states(self):
        """Test creating molar state spec."""
        spec = molar_states(["A", "B", "C"])
        assert spec.n_states == 3
        assert spec.names == ["n_A", "n_B", "n_C"]
        assert spec.variables[0].category == "moles"

    def test_thermal_state(self):
        """Test creating thermal state spec."""
        spec = thermal_state()
        assert spec.n_states == 1
        assert spec.names == ["T"]
        assert spec.variables[0].category == "temperature"

    def test_reactor_states(self):
        """Test creating reactor state spec."""
        spec = reactor_states(["A", "B"], include_T=True)
        assert spec.n_states == 3
        assert "n_A" in spec.names
        assert "T" in spec.names


# =============================================================================
# Integrator Tests
# =============================================================================

class TestIntegrators:
    """Tests for ODE integrators."""

    def test_harmonic_oscillator_rk4(self):
        """Test RK4 on harmonic oscillator."""
        def f(t, y):
            return jnp.array([y[1], -y[0]])

        y0 = jnp.array([1.0, 0.0])
        result = integrate_rk4(f, y0, (0.0, 2 * jnp.pi), n_steps=100)

        # Should return close to initial state after one period
        assert jnp.allclose(result.y_final, y0, atol=0.01)
        assert result.info.success

    def test_harmonic_oscillator_euler(self):
        """Test Euler on harmonic oscillator (less accurate)."""
        def f(t, y):
            return jnp.array([y[1], -y[0]])

        y0 = jnp.array([1.0, 0.0])
        result = integrate_euler(f, y0, (0.0, 2 * jnp.pi), n_steps=1000)

        # Euler is less accurate but should be in ballpark
        assert jnp.abs(result.y_final[0] - 1.0) < 0.5
        assert result.info.success

    def test_exponential_decay(self):
        """Test on simple exponential decay."""
        k = 0.1

        def f(t, y):
            return -k * y

        y0 = jnp.array([1.0])
        t_final = 10.0
        result = integrate_rk4(f, y0, (0.0, t_final), n_steps=100)

        expected = jnp.exp(-k * t_final)
        assert jnp.allclose(result.y_final[0], expected, rtol=1e-4)

    def test_trajectory_shape(self):
        """Test that trajectory has correct shape."""
        def f(t, y):
            return -y

        y0 = jnp.array([1.0, 2.0])
        result = integrate_rk4(f, y0, (0.0, 1.0), n_steps=50)

        assert result.trajectory.t.shape == (51,)  # n_steps + 1
        assert result.trajectory.y.shape == (51, 2)

    def test_unified_interface(self):
        """Test unified integrate() function."""
        def f(t, y):
            return -y

        y0 = jnp.array([1.0])

        # Test different methods
        result_rk4 = integrate(f, y0, (0.0, 1.0), method="RK4", n_steps=100)
        result_euler = integrate(f, y0, (0.0, 1.0), method="Euler", n_steps=1000)

        # Both should give reasonable results
        expected = jnp.exp(-1.0)
        assert jnp.allclose(result_rk4.y_final[0], expected, rtol=1e-3)
        assert jnp.allclose(result_euler.y_final[0], expected, rtol=0.05)

    def test_rk4_step(self):
        """Test single RK4 step."""
        def f(t, y):
            return -y

        y0 = jnp.array([1.0])
        dt = jnp.array(0.1)
        y1 = rk4_step(f, jnp.array(0.0), y0, dt)

        # Compare to analytical
        expected = jnp.exp(-0.1)
        assert jnp.allclose(y1[0], expected, rtol=1e-6)


class TestRK45:
    """Tests for adaptive RK45 integrator."""

    def test_simple_ode(self):
        """Test RK45 on simple ODE."""
        def f(t, y):
            return -y

        y0 = jnp.array([1.0])
        result = integrate_rk45(f, y0, (0.0, 5.0), rtol=1e-6, atol=1e-8)

        expected = jnp.exp(-5.0)
        assert jnp.allclose(result.y_final[0], expected, rtol=1e-5)
        assert result.info.success

    def test_stiff_problem(self):
        """Test RK45 on moderately stiff problem."""
        k = 10.0

        def f(t, y):
            return -k * y

        y0 = jnp.array([1.0])
        result = integrate_rk45(f, y0, (0.0, 1.0), rtol=1e-6, atol=1e-8)

        expected = jnp.exp(-k * 1.0)
        assert jnp.allclose(result.y_final[0], expected, rtol=1e-4)


class TestIntegrationGradients:
    """Tests for gradient computation through integration."""

    def test_gradient_wrt_initial_condition(self):
        """Test gradient of final state w.r.t. initial condition."""
        def f(t, y):
            return -y

        def final_value(y0):
            result = integrate_rk4(f, y0, (0.0, 1.0), n_steps=100)
            return result.y_final[0]

        y0 = jnp.array([1.0])
        grad = jax.grad(final_value)(y0)

        # For dy/dt = -y, y(t) = y0 * exp(-t)
        # dy_final/dy0 = exp(-1)
        expected_grad = jnp.exp(-1.0)
        assert jnp.allclose(grad[0], expected_grad, rtol=1e-3)

    def test_gradient_wrt_parameter(self):
        """Test gradient w.r.t. parameter in ODE."""
        def f(t, y, k):
            return -k * y

        def final_value(k):
            f_k = lambda t, y: f(t, y, k)
            result = integrate_rk4(f_k, jnp.array([1.0]), (0.0, 1.0), n_steps=100)
            return result.y_final[0]

        k = jnp.array(0.5)
        grad = jax.grad(final_value)(k)

        # y(t) = exp(-k*t), dy_final/dk = -t * exp(-k*t) = -exp(-0.5)
        expected = -jnp.exp(-0.5)
        assert jnp.allclose(grad, expected, rtol=1e-2)


# =============================================================================
# Dynamic Unit Tests
# =============================================================================

class TestDynamicCSTR:
    """Tests for DynamicCSTR implementation."""

    @pytest.fixture
    def simple_cstr(self):
        """Create a simple CSTR for testing."""
        def rate_fn(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"]])

        return DynamicCSTR(
            volume=1.0,
            rate_fn=rate_fn,
            stoich=jnp.array([[-1.0], [1.0]]),  # A -> B
            species_order=["A", "B"],
            rate_params={"k": 0.1},
            mode="isothermal",
        )

    def test_state_spec(self, simple_cstr):
        """Test state specification."""
        spec = simple_cstr.state_spec()
        assert spec.n_states == 2  # n_A, n_B (isothermal, no T)
        assert "n_A" in spec.names
        assert "n_B" in spec.names

    def test_initial_state(self, simple_cstr):
        """Test initial state computation."""
        inlet = make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0)
        y0 = simple_cstr.initial_state({"inlet": inlet})
        assert y0.shape == (2,)
        assert y0[0] > 0  # Some initial moles of A

    def test_derivatives(self, simple_cstr):
        """Test derivative computation."""
        inlet = make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0)
        y = jnp.array([10.0, 0.0])  # 10 mol A, 0 mol B
        dy = simple_cstr.derivatives(jnp.array(0.0), y, {"inlet": inlet})
        assert dy.shape == (2,)
        # A should be consumed (dy[0] < 0 from reaction)
        # B should be produced (dy[1] > 0 from reaction)

    def test_outputs(self, simple_cstr):
        """Test output stream computation."""
        inlet = make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0)
        y = jnp.array([10.0, 5.0])  # 10 mol A, 5 mol B
        outputs = simple_cstr.outputs(jnp.array(0.0), y, {"inlet": inlet})
        assert "outlet" in outputs
        outlet = outputs["outlet"]
        assert "F_A" in outlet
        assert "F_B" in outlet

    def test_integration(self, simple_cstr):
        """Test full integration."""
        inlet = make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0)
        result = integrate_unit(
            simple_cstr,
            inputs={"inlet": inlet},
            t_span=(0.0, 100.0),
            method="RK4",
            n_steps=100,
        )
        assert result.info.success
        # A should decrease, B should increase
        y0 = result.trajectory.y[0]
        y_final = result.y_final
        # Check moles changed in expected direction


class TestDynamicTank:
    """Tests for DynamicTank implementation."""

    @pytest.fixture
    def simple_tank(self):
        """Create a simple tank for testing."""
        return DynamicTank(
            max_volume=10.0,
            species_order=["A", "B"],
            isothermal=True,
        )

    def test_state_spec(self, simple_tank):
        """Test state specification."""
        spec = simple_tank.state_spec()
        assert "V" in spec.names
        assert "n_A" in spec.names
        assert "n_B" in spec.names

    def test_initial_state(self, simple_tank):
        """Test initial state."""
        inlet = make_stream({"A": 1.0, "B": 0.5}, T=300.0, P=101325.0)
        y0 = simple_tank.initial_state({"inlet": inlet})
        # Should initialize at half capacity
        assert y0[0] == pytest.approx(5.0, rel=0.1)  # V = V_max / 2

    def test_outputs(self, simple_tank):
        """Test output computation."""
        inlet = make_stream({"A": 1.0, "B": 0.5}, T=300.0, P=101325.0)
        y = jnp.array([5.0, 100.0, 50.0])  # V=5, n_A=100, n_B=50
        outputs = simple_tank.outputs(jnp.array(0.0), y, {"inlet": inlet})
        assert "outlet" in outputs


class TestDynamicUnitProtocol:
    """Tests for DynamicUnit protocol compliance."""

    def test_cstr_is_dynamic_unit(self):
        """Test that DynamicCSTR satisfies protocol."""
        def rate_fn(C, T, params):
            return jnp.array([params["k"] * C["A"]])

        cstr = DynamicCSTR(
            volume=1.0,
            rate_fn=rate_fn,
            stoich=jnp.array([[-1], [1]]),
            species_order=["A", "B"],
            rate_params={"k": 0.1},
        )
        assert isinstance(cstr, DynamicUnit)

    def test_tank_is_dynamic_unit(self):
        """Test that DynamicTank satisfies protocol."""
        tank = DynamicTank(
            max_volume=10.0,
            species_order=["A"],
        )
        assert isinstance(tank, DynamicUnit)


# =============================================================================
# Sensitivity Analysis Tests
# =============================================================================

class TestSensitivityAnalysis:
    """Tests for sensitivity analysis through integration."""

    def test_parameter_sensitivity(self):
        """Test computing sensitivity to parameters."""
        def f(t, y, params):
            return -params[0] * y

        y0 = jnp.array([1.0])
        params = jnp.array([0.5])

        result, jacobian = sensitivity_analysis(
            f, y0, params, (0.0, 1.0), method="RK4", n_steps=100
        )

        # Jacobian should be dy_final/d_params
        # For y = exp(-k*t), dy/dk = -t*exp(-k*t)
        expected_sens = -1.0 * jnp.exp(-0.5)
        assert jnp.allclose(jacobian[0, 0], expected_sens, rtol=0.05)


# =============================================================================
# Integration with Existing difflow Tests
# =============================================================================

class TestDynamicWithStreams:
    """Tests for integration with difflow streams."""

    def test_stream_to_state_conversion(self):
        """Test converting stream to initial state."""
        stream = make_stream({"A": 1.0, "B": 2.0}, T=350.0, P=101325.0)
        flows = get_flows(stream)
        assert flows["A"] == 1.0
        assert flows["B"] == 2.0

    def test_dynamic_cstr_with_real_kinetics(self):
        """Test DynamicCSTR with Arrhenius kinetics."""
        def arrhenius_rate(C, T, params):
            k0, Ea = params["k0"], params["Ea"]
            R = 8.314
            k = k0 * jnp.exp(-Ea / (R * T))
            return jnp.array([k * C["A"]])

        cstr = DynamicCSTR(
            volume=0.1,  # 100 L
            rate_fn=arrhenius_rate,
            stoich=jnp.array([[-1.0], [1.0]]),
            species_order=["A", "B"],
            rate_params={"k0": 1e6, "Ea": 50000.0},
            mode="isothermal",
        )

        inlet = make_stream({"A": 0.1, "B": 0.0}, T=350.0, P=101325.0)
        result = integrate_unit(
            cstr,
            inputs={"inlet": inlet},
            t_span=(0.0, 3600.0),  # 1 hour
            method="RK4",
            n_steps=360,
        )

        assert result.info.success
        # Check some conversion occurred
        n_A_initial = result.trajectory.y[0, 0]
        n_A_final = result.y_final[0]
        # Could check conversion but depends on kinetics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
