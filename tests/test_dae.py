"""Tests for DAE (Differential-Algebraic Equation) support.

Tests cover:
- Algebraic state specification
- Newton solver for algebraic constraints
- DAE integration (Euler and RK4)
- DynamicFlashDrum example unit
- Gradients through DAE systems
"""

import pytest
import jax
import jax.numpy as jnp

from difflow.dynamic import (
    # State management
    StateSpec,
    StateVar,
    AlgebraicVar,
    AlgebraicSpec,
    AlgebraicVector,
    DAEUnit,
    DAEUnitBase,
    integrate_dae,
    DAEResult,
    newton_solve,
    vapor_fraction_algebraic,
    k_value_algebraic,
    DynamicFlashDrum,
)
from difflow.streams import make_stream


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_algebraic_spec():
    """Simple algebraic specification."""
    return AlgebraicSpec([
        AlgebraicVar("z1", units="-", description="Variable 1", initial_guess=1.0),
        AlgebraicVar("z2", units="-", description="Variable 2", initial_guess=0.5),
    ])


@pytest.fixture
def feed_stream():
    """Standard feed stream for flash tests."""
    return make_stream({"A": 0.5, "B": 0.5}, T=350.0, P=101325.0)


# =============================================================================
# Algebraic Specification Tests
# =============================================================================

class TestAlgebraicSpec:
    """Tests for AlgebraicSpec and AlgebraicVar."""


    def test_algebraic_spec_index(self, simple_algebraic_spec):
        """Can get index by name."""
        spec = simple_algebraic_spec
        assert spec.get_index("z1") == 0
        assert spec.get_index("z2") == 1


    def test_algebraic_spec_combine(self):
        """Can combine AlgebraicSpecs."""
        spec1 = AlgebraicSpec([AlgebraicVar("a")])
        spec2 = AlgebraicSpec([AlgebraicVar("b")])
        combined = spec1 + spec2
        assert combined.n_algebraic == 2
        assert combined.names == ["a", "b"]


    def test_algebraic_vector_to_dict(self, simple_algebraic_spec):
        """AlgebraicVector can convert to dict."""
        spec = simple_algebraic_spec
        values = jnp.array([1.0, 2.0])
        vec = AlgebraicVector(values, spec)

        d = vec.to_dict()
        assert "z1" in d
        assert "z2" in d


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestAlgebraicUtilities:
    """Tests for algebraic specification utility functions."""

    def test_vapor_fraction_algebraic(self):
        """vapor_fraction_algebraic creates beta spec."""
        spec = vapor_fraction_algebraic()
        assert spec.n_algebraic == 1
        assert "beta" in spec.names
        assert spec.get_var("beta").bounds == (0.0, 1.0)

    def test_k_value_algebraic(self):
        """k_value_algebraic creates K-value specs."""
        spec = k_value_algebraic(["A", "B", "C"])
        assert spec.n_algebraic == 3
        assert "K_A" in spec.names
        assert "K_B" in spec.names
        assert "K_C" in spec.names


# =============================================================================
# Newton Solver Tests
# =============================================================================

class TestNewtonSolver:
    """Tests for Newton solver."""


    def test_newton_multivariate(self):
        """Newton solver works for multivariate system."""
        # Solve: x + y = 3, x - y = 1 (x=2, y=1)
        def residual(z):
            x, y = z[0], z[1]
            return jnp.array([x + y - 3.0, x - y - 1.0])

        z0 = jnp.array([0.0, 0.0])
        z_sol, info = newton_solve(residual, z0, tol=1e-10, max_iter=10)

        assert jnp.allclose(z_sol, jnp.array([2.0, 1.0]), atol=1e-6)


# =============================================================================
# Simple DAE Unit for Testing
# =============================================================================

class SimplePendulum(DAEUnitBase):
    """Simple pendulum as DAE for testing.

    State variables (differential):
    - theta: angle (rad)
    - omega: angular velocity (rad/s)

    Algebraic variables:
    - T: tension (N) - constrained by length

    The constraint is: L^2 = x^2 + y^2 (constant length)
    which in angle form becomes: T = m*g*cos(theta) + m*L*omega^2
    """

    def __init__(self, length: float = 1.0, mass: float = 1.0, g: float = 9.81):
        params = {"L": length, "m": mass, "g": g}
        super().__init__(params, "pendulum")

    def _build_state_spec(self) -> StateSpec:
        return StateSpec([
            StateVar("theta", category="generic", units="rad", initial_value=0.1),
            StateVar("omega", category="generic", units="rad/s", initial_value=0.0),
        ])

    def _build_algebraic_spec(self) -> AlgebraicSpec:
        return AlgebraicSpec([
            AlgebraicVar("T", units="N", description="Tension", initial_guess=10.0),
        ])

    def _derivatives(self, t, x, z, inputs):
        p = self.params
        theta = x["theta"]
        omega = x["omega"]
        T = z["T"]

        # d(theta)/dt = omega
        # d(omega)/dt = -g/L * sin(theta)
        dtheta_dt = omega
        domega_dt = -p["g"] / p["L"] * jnp.sin(theta)

        return jnp.array([dtheta_dt, domega_dt])

    def _algebraic_residual(self, t, x, z, inputs):
        p = self.params
        theta = x["theta"]
        omega = x["omega"]
        T = z["T"]

        # Tension constraint: T = m*g*cos(theta) + m*L*omega^2
        T_required = p["m"] * p["g"] * jnp.cos(theta) + p["m"] * p["L"] * omega**2
        residual = T - T_required

        return jnp.array([residual])

    def _outputs(self, t, x, z, inputs):
        # No output streams for pendulum
        return {}

    def initial_state(self, inputs, params=None):
        return jnp.array([0.3, 0.0])  # 0.3 rad initial angle


# =============================================================================
# DAE Integration Tests
# =============================================================================

class TestDAEIntegration:
    """Tests for DAE integration."""

    def test_integrate_dae_euler(self):
        """Can integrate DAE with Euler method."""
        pendulum = SimplePendulum(length=1.0)

        result = integrate_dae(
            pendulum,
            inputs={},
            t_span=(0.0, 1.0),
            method="Euler",
            n_steps=100,
        )

        assert isinstance(result, DAEResult)
        assert result.x_final.shape == (2,)
        assert result.z_final.shape == (1,)
        assert result.t_history.shape == (101,)
        assert result.x_history.shape == (101, 2)
        assert result.z_history.shape == (101, 1)


# =============================================================================
# DynamicFlashDrum Tests
# =============================================================================

class TestDynamicFlashDrum:
    """Tests for DynamicFlashDrum DAE unit."""


    def test_flash_drum_integrate(self, feed_stream):
        """Can integrate flash drum."""
        flash = DynamicFlashDrum(
            volume=1.0,
            species_order=["A", "B"],
        )

        # Use shorter time span and more steps for stability
        result = integrate_dae(
            flash,
            inputs={"inlet": feed_stream},
            t_span=(0.0, 1.0),
            method="RK4",
            n_steps=100,
        )

        assert jnp.all(jnp.isfinite(result.x_final))


# =============================================================================
# Gradient Tests
# =============================================================================

class TestDAEGradients:
    """Tests for gradients through DAE systems."""

    def test_gradient_through_newton(self):
        """Can compute gradient through Newton solver."""
        # Solve: z - a = 0 where a is a parameter
        def residual(z, a):
            return z - a

        def solve_for_z(a):
            z0 = jnp.array([0.0])
            # Custom residual with parameter
            z_sol, _ = newton_solve(lambda z: residual(z, a), z0)
            return z_sol[0]

        grad = jax.grad(solve_for_z)(jnp.array(5.0))
        assert jnp.isfinite(grad)


# =============================================================================
# DAE Unit Protocol Tests
# =============================================================================

class TestDAEUnitProtocol:
    """Tests for DAE unit protocol compliance."""

    def test_pendulum_is_dae_unit(self):
        """SimplePendulum implements DAEUnit protocol."""
        pendulum = SimplePendulum()
        assert isinstance(pendulum, DAEUnit)

    def test_flash_drum_is_dae_unit(self):
        """DynamicFlashDrum implements DAEUnit protocol."""
        flash = DynamicFlashDrum(volume=1.0, species_order=["A", "B"])
        assert isinstance(flash, DAEUnit)


# =============================================================================
# Edge Cases
# =============================================================================

class TestDAEEdgeCases:
    """Tests for edge cases in DAE support."""


    def test_custom_initial_conditions(self):
        """Can use custom initial conditions."""
        pendulum = SimplePendulum()

        x0_custom = jnp.array([0.5, 0.1])
        z0_custom = jnp.array([10.0])

        result = integrate_dae(
            pendulum,
            inputs={},
            t_span=(0.0, 0.5),
            x0=x0_custom,
            z0=z0_custom,
            method="RK4",
            n_steps=50,
        )

        assert jnp.all(jnp.isfinite(result.x_final))
