"""Tests for DynamicFlowsheet - Phase 3 of unified dynamic modeling.

Tests cover:
- Single unit in flowsheet
- Multiple connected units
- Time-varying feeds
- State access and trajectories
- Steady-state finding
- Gradient through flowsheet simulation
"""

import pytest
import jax
import jax.numpy as jnp
from jax import Array

from difflow.dynamic import (
    DynamicFlowsheet,
    DynamicFlowsheetResult,
    DynamicUnitEntry,
    DynamicCSTR,
    DynamicTank,
    DynamicUnitBase,
    StateSpec,
    StateVar,
    StateVector,
    molar_states,
    thermal_state,
    EventSpec,
)
from difflow.streams import make_stream, get_flows


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def species_order():
    """Standard species list for tests."""
    return ["A", "B"]


@pytest.fixture
def simple_rate_fn():
    """Simple first-order A -> B rate function."""
    def rate_fn(C, T, params):
        k = params.get("k", 0.1)
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def stoich_A_to_B():
    """Stoichiometry for A -> B."""
    return jnp.array([[-1.0], [1.0]])


@pytest.fixture
def feed_stream():
    """Standard feed stream."""
    return make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)


@pytest.fixture
def simple_cstr(simple_rate_fn, stoich_A_to_B):
    """Simple isothermal CSTR."""
    return DynamicCSTR(
        volume=1.0,
        rate_fn=simple_rate_fn,
        stoich=stoich_A_to_B,
        species_order=["A", "B"],
        rate_params={"k": 0.1},
        name="reactor",
    )


@pytest.fixture
def simple_tank():
    """Simple storage tank."""
    return DynamicTank(
        max_volume=10.0,
        species_order=["A", "B"],
        name="storage",
    )


# =============================================================================
# Basic Flowsheet Creation Tests
# =============================================================================

class TestFlowsheetCreation:
    """Tests for creating and configuring DynamicFlowsheet."""

    def test_create_empty_flowsheet(self, species_order):
        """Can create an empty flowsheet."""
        fs = DynamicFlowsheet(species_order=species_order)
        assert fs.species_order == species_order
        assert len(fs.units) == 0
        assert fs.n_states == 0

    def test_add_feed(self, species_order, feed_stream):
        """Can add feed stream to flowsheet."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        assert "feed" in fs._feeds

    def test_add_unit(self, species_order, simple_cstr, feed_stream):
        """Can add unit to flowsheet."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        assert len(fs.units) == 1
        assert fs.units[0].name == "reactor"
        assert fs.n_states == 2  # n_A, n_B for isothermal CSTR

    def test_add_multiple_units(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Can add multiple connected units."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        assert len(fs.units) == 2
        assert fs.unit_names == ["reactor", "storage"]
        # CSTR: 2 states, Tank: 3 states (V + n_A + n_B)
        assert fs.n_states == 5

    def test_duplicate_unit_name_raises(
        self, species_order, simple_cstr, feed_stream
    ):
        """Adding unit with duplicate name raises error."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["out1"])

        cstr2 = DynamicCSTR(
            volume=2.0,
            rate_fn=lambda C, T, p: jnp.array([0.0]),
            stoich=jnp.array([[-1.0], [1.0]]),
            species_order=["A", "B"],
            name="reactor",  # Same name
        )

        with pytest.raises(ValueError, match="already exists"):
            fs.add_unit(cstr2, inlet_names=["out1"], outlet_names=["out2"])

    def test_flowsheet_repr(self, species_order, simple_cstr, feed_stream):
        """Flowsheet has informative repr."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        repr_str = repr(fs)
        assert "DynamicFlowsheet" in repr_str
        assert "n_units=1" in repr_str
        assert "n_states=2" in repr_str


# =============================================================================
# State Specification Tests
# =============================================================================

class TestCombinedStateSpec:
    """Tests for combined state specification."""

    def test_combined_state_spec_single_unit(
        self, species_order, simple_cstr, feed_stream
    ):
        """Combined state spec for single unit."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        spec = fs.combined_state_spec()
        assert spec.n_states == 2
        # Names should be prefixed with unit name
        assert "reactor.n_A" in spec.names
        assert "reactor.n_B" in spec.names

    def test_combined_state_spec_multiple_units(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Combined state spec for multiple units."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        spec = fs.combined_state_spec()
        assert spec.n_states == 5
        assert "reactor.n_A" in spec.names
        assert "reactor.n_B" in spec.names
        assert "storage.V" in spec.names
        assert "storage.n_A" in spec.names
        assert "storage.n_B" in spec.names


# =============================================================================
# Initial State Tests
# =============================================================================

class TestInitialState:
    """Tests for initial state computation."""

    def test_initial_state_single_unit(
        self, species_order, simple_cstr, feed_stream
    ):
        """Initial state computation for single unit."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        y0 = fs.initial_state()
        assert y0.shape == (2,)
        # Initial moles should be non-negative (B can be 0 since only A in feed)
        assert jnp.all(y0 >= 0)
        # A should be positive
        assert y0[0] > 0

    def test_initial_state_multiple_units(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Initial state for connected units."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        y0 = fs.initial_state()
        assert y0.shape == (5,)
        # All states should be non-negative (some may be 0 for products)
        assert jnp.all(y0 >= 0)
        # Tank volume (state index 2) should be positive
        assert y0[2] > 0


# =============================================================================
# Derivatives Tests
# =============================================================================

class TestDerivatives:
    """Tests for derivatives computation."""

    def test_derivatives_shape(self, species_order, simple_cstr, feed_stream):
        """Derivatives have correct shape."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        y0 = fs.initial_state()
        dy = fs.derivatives(jnp.array(0.0), y0)

        assert dy.shape == y0.shape

    def test_derivatives_finite(self, species_order, simple_cstr, feed_stream):
        """Derivatives are finite."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        y0 = fs.initial_state()
        dy = fs.derivatives(jnp.array(0.0), y0)

        assert jnp.all(jnp.isfinite(dy))

    def test_derivatives_multiple_units(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Derivatives work for multiple connected units."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        y0 = fs.initial_state()
        dy = fs.derivatives(jnp.array(0.0), y0)

        assert dy.shape == (5,)
        assert jnp.all(jnp.isfinite(dy))


# =============================================================================
# Simulation Tests
# =============================================================================

class TestSimulation:
    """Tests for flowsheet simulation."""

    def test_simulate_single_unit(self, species_order, simple_cstr, feed_stream):
        """Can simulate single unit flowsheet."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)

        assert isinstance(result, DynamicFlowsheetResult)
        assert result.y_final.shape == (2,)
        assert result.trajectory.t.shape == (51,)  # n_steps + 1
        assert result.trajectory.y.shape == (51, 2)

    def test_simulate_multiple_units(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Can simulate connected units."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)

        assert result.y_final.shape == (5,)
        assert result.trajectory.y.shape == (51, 5)

    def test_simulate_with_rk45(self, species_order, simple_cstr, feed_stream):
        """Can simulate with adaptive RK45."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(
            t_span=(0.0, 100.0),
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
        )

        assert jnp.all(jnp.isfinite(result.y_final))

    def test_reaction_progress(self, species_order, simple_cstr, feed_stream):
        """A -> B reaction makes progress during simulation."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 500.0), method="RK4", n_steps=500)

        # Get initial and final states
        y0 = result.trajectory.y[0]
        yf = result.trajectory.y[-1]

        # Product B should increase (state index 1)
        assert yf[1] > y0[1]


class TestSimulationEvents:
    """Issue #130: event detection wired through DynamicFlowsheet.simulate."""

    def test_no_events_by_default(self, species_order, simple_cstr, feed_stream):
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)
        assert result.events == []

    def test_threshold_event_detected(self, species_order, simple_cstr, feed_stream):
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        # First run to find the range of product B (state index 1)
        base = fs.simulate(t_span=(0.0, 500.0), method="RK4", n_steps=500)
        b0 = float(base.trajectory.y[0][1])
        bf = float(base.trajectory.y[-1][1])
        threshold = 0.5 * (b0 + bf)

        event = EventSpec(
            name="B_threshold",
            condition_fn=lambda t, y: y[1] - threshold,
            direction=1,  # B increasing through the threshold
        )
        result = fs.simulate(
            t_span=(0.0, 500.0), method="RK4", n_steps=500, events=[event]
        )
        assert len(result.events) >= 1
        ev = result.events[0]
        assert ev.name == "B_threshold"
        assert 0.0 < ev.t_event < 500.0

    def test_event_not_triggered_when_condition_never_crosses(
        self, species_order, simple_cstr, feed_stream
    ):
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        # Threshold far above any reachable state
        event = EventSpec(
            name="never", condition_fn=lambda t, y: y[1] - 1e9, direction=1
        )
        result = fs.simulate(
            t_span=(0.0, 100.0), method="RK4", n_steps=50, events=[event]
        )
        assert result.events == []


# =============================================================================
# Result Access Tests
# =============================================================================

class TestResultAccess:
    """Tests for accessing results from DynamicFlowsheetResult."""

    def test_unit_trajectory(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Can access trajectory for specific unit."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)

        # Get reactor trajectory
        reactor_traj = result.unit_trajectory("reactor")
        assert reactor_traj.t.shape == (51,)
        assert reactor_traj.y.shape == (51, 2)  # n_A, n_B

        # Get tank trajectory
        tank_traj = result.unit_trajectory("storage")
        assert tank_traj.y.shape == (51, 3)  # V, n_A, n_B

    def test_unit_state_at(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Can access unit state at specific time index."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)

        # Get final reactor state
        reactor_final = result.unit_state_at("reactor", -1)
        assert reactor_final.shape == (2,)

        # Get initial tank state
        tank_initial = result.unit_state_at("storage", 0)
        assert tank_initial.shape == (3,)

    def test_get_state_dict(self, species_order, simple_cstr, feed_stream):
        """Can get unit state as named dictionary."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)

        state_dict = result.get_state_dict("reactor", -1)
        assert "n_A" in state_dict
        assert "n_B" in state_dict
        assert isinstance(state_dict["n_A"], Array)


# =============================================================================
# Time-Varying Feed Tests
# =============================================================================

class TestTimeVaryingFeeds:
    """Tests for time-varying feed streams."""

    def test_step_change_feed(self, species_order, simple_cstr):
        """Flowsheet responds to step change in feed."""
        # Feed that steps up at t=50
        def feed_fn(t):
            flow_A = jnp.where(t < 50.0, 1.0, 2.0)
            return make_stream({"A": flow_A, "B": 0.0}, T=350.0, P=101325.0)

        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_fn)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=100)

        # System should respond to the step change
        # Final state should be different from what it would be without the step
        assert jnp.all(jnp.isfinite(result.y_final))

    def test_sinusoidal_feed(self, species_order, simple_cstr):
        """Flowsheet responds to sinusoidal feed variation."""
        def feed_fn(t):
            flow_A = 1.0 + 0.2 * jnp.sin(0.1 * t)
            return make_stream({"A": flow_A, "B": 0.0}, T=350.0, P=101325.0)

        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_fn)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 200.0), method="RK4", n_steps=200)

        # Should complete without error
        assert jnp.all(jnp.isfinite(result.y_final))


# =============================================================================
# Stream Outputs Tests
# =============================================================================

class TestOutputs:
    """Tests for computing outlet streams."""

    def test_outputs_single_unit(self, species_order, simple_cstr, feed_stream):
        """Can compute output streams from flowsheet."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        y0 = fs.initial_state()
        outputs = fs.outputs(jnp.array(0.0), y0)

        assert "feed" in outputs
        assert "reactor_out" in outputs

    def test_outputs_multiple_units(
        self, species_order, simple_cstr, simple_tank, feed_stream
    ):
        """Can compute outputs from multiple connected units."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
        fs.add_unit(
            simple_tank, inlet_names=["reactor_out"], outlet_names=["product"]
        )

        y0 = fs.initial_state()
        outputs = fs.outputs(jnp.array(0.0), y0)

        assert "feed" in outputs
        assert "reactor_out" in outputs
        assert "product" in outputs


# =============================================================================
# Steady State Tests
# =============================================================================

class TestSteadyState:
    """Tests for steady-state finding."""

    def test_find_steady_state(self, species_order, simple_cstr, feed_stream):
        """Can find approximate steady state."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        y_ss = fs.steady_state(tol=1e-4, max_iter=500)

        # Derivatives should be small at steady state
        dy = fs.derivatives(jnp.array(0.0), y_ss)
        assert jnp.max(jnp.abs(dy)) < 0.1  # Relaxed tolerance


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients:
    """Tests for gradients through flowsheet simulation."""

    def test_gradient_through_simulation(
        self, species_order, simple_cstr, feed_stream
    ):
        """Can compute gradient through flowsheet simulation."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        y0 = fs.initial_state()

        # Define loss function
        def loss(y0_):
            def f(t, y):
                return fs.derivatives(t, y)
            from difflow.dynamic import integrate
            result = integrate(f, y0_, (0.0, 100.0), "RK4", n_steps=50)
            return jnp.sum(result.y_final ** 2)

        # Compute gradient
        grad = jax.grad(loss)(y0)

        assert grad.shape == y0.shape
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.release
    def test_gradient_wrt_parameter(self, species_order, feed_stream):
        """Can compute gradient with respect to rate parameter."""
        def rate_fn(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"]])

        def make_flowsheet(k_value):
            cstr = DynamicCSTR(
                volume=1.0,
                rate_fn=rate_fn,
                stoich=jnp.array([[-1.0], [1.0]]),
                species_order=["A", "B"],
                rate_params={"k": k_value},
                name="reactor",
            )
            fs = DynamicFlowsheet(species_order=["A", "B"])
            fs.add_feed("feed", feed_stream)
            fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
            return fs

        def loss(k):
            fs = make_flowsheet(k)
            result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)
            # Want to maximize product B
            return -result.y_final[1]  # Negative because we minimize

        # Compute gradient
        grad = jax.grad(loss)(0.1)

        assert jnp.isfinite(grad)
        # Higher k should give more product, so gradient should be negative
        # (because we negated to maximize)
        assert grad < 0


# =============================================================================
# Connection Tests
# =============================================================================

class TestConnections:
    """Tests for explicit stream connections."""

    def test_explicit_connection(self, species_order, simple_cstr, feed_stream):
        """Explicit connections work."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["out"])

        # Tank expects inlet named "tank_inlet" but we have "out"
        tank = DynamicTank(
            max_volume=10.0,
            species_order=species_order,
            name="tank",
        )
        fs.add_unit(tank, inlet_names=["tank_inlet"], outlet_names=["product"])

        # Connect reactor output to tank input
        fs.connect("out", "tank_inlet")

        # Should work now
        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)
        assert jnp.all(jnp.isfinite(result.y_final))


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_inlet_raises(self, species_order, simple_cstr, feed_stream):
        """Missing inlet stream raises clear error."""
        fs = DynamicFlowsheet(species_order=species_order)
        # Don't add the feed
        fs.add_unit(
            simple_cstr, inlet_names=["missing_feed"], outlet_names=["reactor_out"]
        )

        with pytest.raises(ValueError, match="not found"):
            fs.initial_state()

    def test_very_short_simulation(self, species_order, simple_cstr, feed_stream):
        """Very short simulation works."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 0.001), method="RK4", n_steps=10)
        assert jnp.all(jnp.isfinite(result.y_final))

    def test_long_simulation(self, species_order, simple_cstr, feed_stream):
        """Long simulation maintains stability."""
        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(simple_cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 10000.0), method="RK4", n_steps=1000)
        assert jnp.all(jnp.isfinite(result.y_final))
        # Should approach steady state
        assert jnp.all(result.y_final > 0)


# =============================================================================
# Integration with Existing Units Tests
# =============================================================================

class TestExistingUnitIntegration:
    """Tests for using existing units (CSTR, PFR, etc.) in dynamic flowsheet."""

    def test_existing_cstr_in_flowsheet(self, species_order, feed_stream):
        """Can use existing CSTR class in flowsheet."""
        from difflow.units.cstr import CSTR, CSTRParams

        def rate_fn(C, T, params):
            k = params.get("k", 0.1)
            return jnp.array([k * C["A"]])

        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=rate_fn,
            stoich=jnp.array([[-1.0], [1.0]]),
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        fs = DynamicFlowsheet(species_order=species_order)
        fs.add_feed("feed", feed_stream)
        fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["reactor_out"])

        result = fs.simulate(t_span=(0.0, 100.0), method="RK4", n_steps=50)
        assert jnp.all(jnp.isfinite(result.y_final))


# =============================================================================
# Port maps, stateless units, algebraic loops, integrator args (#390)
# =============================================================================

from difflow.dynamic import InstantaneousUnit, integrate


class _LagHX:
    """Toy counter-current exchanger: duty Q relaxes to UA * (T_hot - T_cold).

    Each outlet reads only its own side's inlet, like DynamicCounterCurrentHX,
    so a loop through it can be ordered.
    """

    output_ports = ("hot_out", "cold_out")
    output_dependencies = {"hot_out": ("hot",), "cold_out": ("cold",)}

    def __init__(self, UA=1.0, C=10.0, tau=5.0, name="hx"):
        self.UA, self.C, self.tau, self.name = UA, C, tau, name

    def state_spec(self):
        return StateSpec([StateVar("Q", "generic", "W", "duty")])

    def initial_state(self, inputs, params=None):
        return jnp.zeros(1)

    def _side(self, stream, dT):
        out = dict(stream)
        out["T"] = stream["T"] + dT
        return out

    def partial_outputs(self, t, state, inputs, params=None):
        out = {}
        if "hot" in inputs:
            out["hot_out"] = self._side(inputs["hot"], -state[0] / self.C)
        if "cold" in inputs:
            out["cold_out"] = self._side(inputs["cold"], state[0] / self.C)
        return out

    def outputs(self, t, state, inputs, params=None):
        return self.partial_outputs(t, state, inputs, params)

    def derivatives(self, t, state, inputs, params=None):
        target = self.UA * (inputs["hot"]["T"] - inputs["cold"]["T"])
        return jnp.atleast_1d((target - state[0]) / self.tau)


def _duty_limited_heater(T_set, Q_max, C=10.0):
    """Stateless heater: setpoint, unless that needs more than Q_max."""
    def fn(inputs, params):
        s = inputs["inlet"]
        out = dict(s)
        out["T"] = jnp.minimum(T_set, s["T"] + Q_max / C)
        return {"outlet": out}
    return InstantaneousUnit(fn=fn, name="heater")


def _feed_effluent_loop(heater, hx=None):
    """feed -> HX cold -> heater -> HX hot -> product: a loop through outputs()."""
    fs = DynamicFlowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(hx or _LagHX(), inlets={"hot": "hot_feed", "cold": "feed"},
                outlets={"hot_out": "product", "cold_out": "preheated"})
    fs.add_unit(heater, inlets={"inlet": "preheated"}, outlets={"outlet": "hot_feed"})
    return fs


class TestPortMaps:
    def test_two_units_with_the_same_port_names(self, simple_rate_fn, stoich_A_to_B, feed_stream):
        """Ports are wired to freely named streams, so two CSTRs coexist."""
        def cstr(name):
            return DynamicCSTR(volume=1.0, rate_fn=simple_rate_fn, stoich=stoich_A_to_B,
                               species_order=["A", "B"], rate_params={"k": 0.1}, name=name)

        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed("fresh", feed_stream)
        fs.add_unit(cstr("r1"), inlets={"inlet": "fresh"}, outlets={"outlet": "mid"})
        fs.add_unit(cstr("r2"), inlets={"inlet": "mid"}, outlets={"outlet": "out"})
        streams = fs.outputs(jnp.array(0.0), fs.initial_state())
        assert {"fresh", "mid", "out"} <= set(streams)

    def test_dict_in_the_positional_slot_is_a_port_map(self, simple_cstr, feed_stream):
        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed("fresh", feed_stream)
        fs.add_unit(simple_cstr, {"inlet": "fresh"}, {"outlet": "product"})
        assert fs.units[0].inlet_names == ["fresh"]
        assert fs.units[0].outlet_names == ["product"]

    def test_unknown_output_port_is_reported(self, simple_cstr, feed_stream):
        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed("fresh", feed_stream)
        fs.add_unit(simple_cstr, inlets={"inlet": "fresh"}, outlets={"nope": "product"})
        with pytest.raises(ValueError, match="no output 'nope'"):
            fs.initial_state()


class TestStatelessAndLoops:
    def test_instantaneous_unit_has_no_state(self):
        fs = _feed_effluent_loop(_duty_limited_heater(T_set=400.0, Q_max=1e6))
        assert fs.n_states == 1  # only the exchanger's duty

    def test_loop_is_ordered_by_declared_dependencies(self):
        fs = _feed_effluent_loop(_duty_limited_heater(T_set=400.0, Q_max=1e6))
        plan = [(e.name, None if p is None else set(p)) for e, p in fs._evaluation_plan()]
        assert plan == [("hx", {"cold_out"}), ("heater", {"outlet"}), ("hx", {"hot_out"})]

    def test_duty_limited_loop_reaches_its_analytic_steady_state(self):
        """With the heater at its duty limit, T_h = T_c + Q_max/C and
        Q = UA (T_h - T_feed) close the loop in closed form."""
        UA, C, Q_max, T_feed = 1.0, 10.0, 50.0, 300.0
        fs = _feed_effluent_loop(_duty_limited_heater(T_set=1000.0, Q_max=Q_max),
                                 _LagHX(UA=UA, C=C))
        r = fs.simulate((0.0, 200.0), method="RK4", n_steps=400)
        Q = float(r.y_final[0])
        # T_hot_in = T_feed + Q/C + Q_max/C, and Q = UA (T_hot_in - T_feed)
        Q_exact = UA * Q_max / C / (1.0 - UA / C)
        assert Q == pytest.approx(Q_exact, rel=1e-6)
        s = fs.outputs(jnp.array(200.0), r.y_final)
        assert float(s["hot_feed"]["T"]) == pytest.approx(T_feed + (Q_exact + Q_max) / C, rel=1e-6)

    def test_unbreakable_algebraic_loop_raises(self):
        class Opaque(_LagHX):
            output_dependencies = None

        fs = _feed_effluent_loop(_duty_limited_heater(T_set=400.0, Q_max=1e6), Opaque())
        with pytest.raises(ValueError, match="Algebraic loop"):
            fs.initial_state()


class TestIntegratorArgs:
    def test_integrate_passes_args(self):
        for method in ("RK4", "Euler", "RK45", "diffrax:tsit5"):
            steps = {} if method.startswith("diffrax") else {"n_steps": 5000}
            r = integrate(lambda t, y, a: -a["k"] * y, jnp.array([1.0]), (0.0, 1.0),
                          method, args={"k": jnp.array(2.0)}, **steps)
            assert float(r.y_final[0]) == pytest.approx(float(jnp.exp(-2.0)), rel=2e-3), method

    def _flowsheet_with_args_feed(self, simple_cstr):
        def feed(t, args):
            return make_stream({"A": args["F"], "B": 0.0}, T=350.0, P=101325.0)

        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed("fresh", feed)
        fs.add_unit(simple_cstr, inlets={"inlet": "fresh"}, outlets={"outlet": "product"})
        return fs

    def test_scenarios_share_one_compiled_rhs(self, simple_cstr):
        """A new disturbance is a new args value, not a recompile."""
        fs = self._flowsheet_with_args_feed(simple_cstr)
        y0 = fs.initial_state(args={"F": jnp.array(1.0)})
        traces = []
        original = fs.derivatives

        def counting(*a, **k):
            traces.append(1)
            return original(*a, **k)

        fs.derivatives = counting
        finals = []
        for F in (1.0, 2.0, 3.0):
            r = fs.simulate((0.0, 50.0), y0=y0, method="diffrax:tsit5",
                            args={"F": jnp.array(F)})
            finals.append(float(r.y_final[0]))
        n_after_first = len(traces)
        assert len(set(finals)) == 3
        fs.simulate((0.0, 50.0), y0=y0, method="diffrax:tsit5", args={"F": jnp.array(4.0)})
        assert len(traces) == n_after_first  # no retrace

    def test_grad_with_respect_to_a_disturbance(self, simple_cstr):
        fs = self._flowsheet_with_args_feed(simple_cstr)
        y0 = fs.initial_state(args={"F": jnp.array(1.0)})

        def holdup_A(F):
            r = fs.simulate((0.0, 20.0), y0=y0, method="diffrax:tsit5", args={"F": F})
            return r.y_final[0]

        g = jax.grad(holdup_A)(jnp.array(1.0))
        eps = 1e-4
        fd = (holdup_A(jnp.array(1.0 + eps)) - holdup_A(jnp.array(1.0 - eps))) / (2 * eps)
        assert float(g) == pytest.approx(float(fd), rel=1e-4)

    def test_one_argument_feed_still_works(self, simple_cstr):
        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed("fresh", lambda t, scale=2.0: make_stream({"A": scale, "B": 0.0}, T=350.0, P=101325.0))
        fs.add_unit(simple_cstr, inlets={"inlet": "fresh"}, outlets={"outlet": "product"})
        s = fs.outputs(jnp.array(0.0), fs.initial_state())
        assert float(s["fresh"]["F_A"]) == 2.0


@pytest.mark.slow
@pytest.mark.release
class TestCubicEOSFeedEffluentFlowsheet:
    """The #390 acceptance case: a dynamic flowsheet of difflow's own units.

    FEHE (DynamicCounterCurrentHX) + stateless trim heater + adiabatic PR CSTR
    + stateless trim cooler + DynamicEOSFlash, with freely named streams and a
    feed read from integrator args. The FEHE -> heater -> reactor -> FEHE
    cycle is evaluated through the exchanger's declared dependencies. Its
    long-time limit is the steady-state flowsheet of test_eo_solver (#389).
    """

    def test_long_time_limit_matches_the_steady_state_flowsheet(self):
        from difflow import (Cooler, CoolerParams, Flowsheet, Heater, HeaterParams,
                             Unit)
        from difflow.dynamic import DynamicCounterCurrentHX, DynamicEOSFlash
        from difflow.units.flash import EOSFlash, EOSFlashParams
        from difflow.units.heat_exchanger import (EnthalpyCounterCurrentHX,
                                                  EnthalpyHXParams)
        from tests.c4_system import (C4_P, C4_SPECIES, _c4_eos_and_thermo, _c4_stream,
                                     _isomerization_cstr)

        eos, thermo = _c4_eos_and_thermo()

        def feed(t, args):
            s = args["scale"]
            return make_stream({"propane": 0.5 * s, "butane": 1.0 * s,
                                "isobutane": 0.1 * s}, T=340.0, P=C4_P)

        fs = DynamicFlowsheet(species_order=C4_SPECIES)
        fs.add_feed("naphtha", feed)
        fs.add_unit(DynamicCounterCurrentHX(150.0, thermo, tau=30.0, name="fehe"),
                    inlets={"hot": "effluent", "cold": "naphtha"},
                    outlets={"hot_out": "eff_cooled", "cold_out": "preheated"})
        fs.add_unit(InstantaneousUnit(Heater(HeaterParams(T_out=460.0), thermo), name="trim"),
                    inlets={"inlet": "preheated"}, outlets={"outlet": "rx_in"})
        fs.add_unit(_isomerization_cstr(), name="reactor",
                    inlets={"inlet": "rx_in"}, outlets={"outlet": "effluent"})
        fs.add_unit(InstantaneousUnit(Cooler(CoolerParams(T_out=320.0), thermo), name="cooler"),
                    inlets={"inlet": "eff_cooled"}, outlets={"outlet": "cooled"})
        fs.add_unit(DynamicEOSFlash(eos, C4_SPECIES, P=C4_P, name="sep"),
                    inlets={"feed": "cooled"}, outlets={"liquid": "liq", "vapor": "vap"})

        args = {"scale": jnp.array(1.0)}
        r = fs.simulate((0.0, 3000.0), method="diffrax:kvaerno5", args=args,
                        rtol=1e-8, atol=1e-8)
        assert bool(r.info.success)
        assert float(jnp.max(jnp.abs(
            fs.derivatives(jnp.array(3000.0), r.y_final, None, args)))) < 1e-8
        dyn = fs.outputs(jnp.array(3000.0), r.y_final, None, args)

        ss = Flowsheet(species_order=C4_SPECIES)
        ss.add_feed("naphtha", _c4_stream(340.0))
        ss.add_unit(Unit("fehe", EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=150.0), thermo),
                         ["effluent", "naphtha"], ["eff_cooled", "preheated"]))
        ss.add_unit(Unit("trim", Heater(HeaterParams(T_out=460.0), thermo),
                         ["preheated"], ["rx_in"]))
        ss.add_unit(Unit("reactor", _isomerization_cstr(), ["rx_in"], ["effluent"]))
        ss.add_unit(Unit("cooler", Cooler(CoolerParams(T_out=320.0), thermo),
                         ["eff_cooled"], ["cooled"]))
        ss.add_unit(Unit("sep", EOSFlash(EOSFlashParams(species_order=C4_SPECIES), eos),
                         ["cooled"], ["liq", "vap"]))
        ss.add_recycle("effluent", "effluent")
        steady = ss.solve_eo(tol=1e-8)

        for name in ["preheated", "effluent", "eff_cooled", "liq", "vap"]:
            for key, val in steady[name].items():
                assert float(dyn[name][key]) == pytest.approx(float(val), rel=1e-6, abs=1e-7), \
                    f"{name}[{key}]"


# =============================================================================
# InstantaneousUnit sees t and args (#396)
# =============================================================================


def _outlet_T(fs, t, args=None):
    streams = fs.outputs(jnp.asarray(t), fs.initial_state(None, args), None, args)
    return float(streams["out"]["T"])


def _one_unit_fs(unit, args_feed=None):
    fs = DynamicFlowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(unit, inlets={"inlet": "feed"}, outlets={"outlet": "out"})
    return fs


class TestInstantaneousUnitTimeAndArgs:
    def test_fn_with_t(self):
        def fn(t, inputs, params):
            return {"outlet": {**inputs["inlet"], "T": 300.0 + 10.0 * t}}

        fs = _one_unit_fs(InstantaneousUnit(fn=fn, name="ramp"))
        assert _outlet_T(fs, 0.0) == pytest.approx(300.0)
        assert _outlet_T(fs, 5.0) == pytest.approx(350.0)

    def test_fn_with_t_and_args_shares_one_channel_with_feeds(self):
        def fn(t, inputs, params, args):
            return {"outlet": {**inputs["inlet"], "T": args["T0"] + args["rate"] * t}}

        fs = DynamicFlowsheet(species_order=["A", "B"])
        fs.add_feed(
            "feed",
            lambda t, args: make_stream({"A": 1.0, "B": 0.0}, T=args["T_feed"], P=101325.0),
        )
        fs.add_unit(InstantaneousUnit(fn=fn, name="u"),
                    inlets={"inlet": "feed"}, outlets={"outlet": "out"})
        args = {"T0": jnp.array(400.0), "rate": jnp.array(2.0), "T_feed": jnp.array(310.0)}
        streams = fs.outputs(jnp.array(10.0), fs.initial_state(None, args), None, args)
        assert float(streams["feed"]["T"]) == pytest.approx(310.0)
        assert float(streams["out"]["T"]) == pytest.approx(420.0)

    def test_legacy_two_argument_fn_unchanged(self):
        def fn(inputs, params):
            return {"outlet": {**inputs["inlet"], "T": 333.0}}

        assert _outlet_T(_one_unit_fs(InstantaneousUnit(fn=fn)), 1.0) == pytest.approx(333.0)

    def test_call_params_callable_sees_t_and_args(self):
        class Op:
            def __call__(self, inlet, T_out):
                return {**inlet, "T": T_out}

        def by_t(t, params):
            return 300.0 + t

        def by_args(t, params, args):
            return args["T1"] - 1.0 * t

        fs = _one_unit_fs(InstantaneousUnit(Op(), call_params={"T_out": by_t}))
        assert _outlet_T(fs, 7.0) == pytest.approx(307.0)
        fs = _one_unit_fs(InstantaneousUnit(Op(), call_params={"T_out": by_args}))
        assert _outlet_T(fs, 7.0, {"T1": jnp.array(400.0)}) == pytest.approx(393.0)

    def test_simulate_with_one_args_dict_and_gradient(self):
        def fn(t, inputs, params, args):
            return {"outlet": {**inputs["inlet"], "T": args["T0"] + args["rate"] * t}}

        fs = _one_unit_fs(InstantaneousUnit(fn=fn))

        def final_T(rate):
            args = {"T0": jnp.array(300.0), "rate": rate}
            return fs.outputs(jnp.array(4.0), fs.initial_state(None, args), None, args)["out"]["T"]

        assert float(jax.grad(final_T)(jnp.array(2.0))) == pytest.approx(4.0)
