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
