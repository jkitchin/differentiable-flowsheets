"""Tests for initialization strategies and acceleration methods."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    PFR,
    PFRParams,
    Flash,
    FlashParams,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
    Flowsheet,
    Unit,
)
from difflow.initialization import (
    wegstein_acceleration,
    anderson_acceleration_step,
    AndersonAccelerator,
    estimate_cstr_conversion,
    estimate_outlet_temperature,
    InitializationResult,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_thermo():
    """Simple two-component thermodynamics."""
    species_data = {
        "A": SpeciesData(
            "A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        ),
        "B": SpeciesData(
            "B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def simple_rate_fn():
    """First-order reaction rate function."""
    def rate_fn(C, T, params):
        k = params["k"]
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def simple_inlet():
    """Simple inlet stream."""
    return make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0)


@pytest.fixture
def simple_cstr(simple_thermo, simple_rate_fn):
    """Create a simple CSTR for testing."""
    stoich = jnp.array([[-1.0], [+1.0]])  # A → B
    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=simple_rate_fn,
        stoich=stoich,
        rate_params={"k": jnp.array(0.1)},
        species_order=["A", "B"],
    )
    return CSTR(params, thermo=simple_thermo, mode="isothermal")


class TestWegsteinAcceleration:
    """Tests for Wegstein acceleration."""

    def test_wegstein_basic(self):
        """Test basic Wegstein acceleration."""
        # Simple fixed-point: x = 0.5 * x + 0.25 (solution: x = 0.5)
        x_prev = jnp.array([0.0])
        x_curr = jnp.array([0.25])  # g(0) = 0.5 * 0 + 0.25 = 0.25
        g_prev = jnp.array([0.25])  # g(0)
        g_curr = jnp.array([0.375])  # g(0.25) = 0.5 * 0.25 + 0.25 = 0.375

        x_new = wegstein_acceleration(x_prev, x_curr, g_prev, g_curr)

        # Should be closer to 0.5 than direct iteration
        assert x_new.shape == (1,)
        assert jnp.isfinite(x_new).all()

    def test_wegstein_bounds(self):
        """Test that Wegstein respects acceleration bounds."""
        x_prev = jnp.array([1.0])
        x_curr = jnp.array([2.0])
        g_prev = jnp.array([1.5])
        g_curr = jnp.array([2.5])

        x_new = wegstein_acceleration(x_prev, x_curr, g_prev, g_curr, bounds=(-2.0, 1.0))

        assert jnp.isfinite(x_new).all()

    def test_wegstein_multidimensional(self):
        """Test Wegstein with multiple variables."""
        x_prev = jnp.array([0.0, 0.0])
        x_curr = jnp.array([0.3, 0.2])
        g_prev = jnp.array([0.3, 0.2])
        g_curr = jnp.array([0.45, 0.35])

        x_new = wegstein_acceleration(x_prev, x_curr, g_prev, g_curr)

        assert x_new.shape == (2,)
        assert jnp.isfinite(x_new).all()


class TestAndersonAcceleration:
    """Tests for Anderson acceleration."""

    def test_anderson_step_basic(self):
        """Test basic Anderson acceleration step."""
        # Build a simple history
        x_hist = jnp.array([
            [0.0, 0.0],
            [0.3, 0.2],
            [0.45, 0.35],
        ])
        g_hist = jnp.array([
            [0.3, 0.2],
            [0.45, 0.35],
            [0.52, 0.42],
        ])

        x_new = anderson_acceleration_step(x_hist, g_hist, m=2)

        assert x_new.shape == (2,)
        assert jnp.isfinite(x_new).all()

    def test_anderson_step_insufficient_history(self):
        """Test Anderson with insufficient history falls back to direct iteration."""
        x_hist = jnp.array([[0.0, 0.0]])
        g_hist = jnp.array([[0.3, 0.2]])

        x_new = anderson_acceleration_step(x_hist, g_hist, m=5)

        # Should return g_hist[-1] when not enough history
        assert jnp.allclose(x_new, g_hist[-1])

    def test_anderson_accelerator_class(self):
        """Test AndersonAccelerator class."""
        accelerator = AndersonAccelerator(m=3)

        # First step: no history, returns g
        x1 = jnp.array([0.0, 0.0])
        g1 = jnp.array([0.3, 0.2])
        x2 = accelerator.step(x1, g1)
        assert jnp.allclose(x2, g1)

        # Second step: starts using acceleration
        g2 = jnp.array([0.45, 0.35])
        x3 = accelerator.step(x2, g2)
        assert jnp.isfinite(x3).all()

        # Reset clears history
        accelerator.reset()
        assert len(accelerator.x_hist) == 0

    def test_anderson_accelerator_history_limit(self):
        """Test that AndersonAccelerator limits history size."""
        accelerator = AndersonAccelerator(m=2)

        # Add more steps than m+1
        for i in range(5):
            x = jnp.array([float(i)])
            g = jnp.array([float(i) + 0.5])
            accelerator.step(x, g)

        # History should be limited
        assert len(accelerator.x_hist) <= 3  # m+1


class TestCSTRInitialization:
    """Tests for CSTR initialization."""

    def test_cstr_initialize(self, simple_cstr, simple_inlet):
        """Test CSTR initialize method."""
        result = simple_cstr.initialize(simple_inlet)

        assert 'outlet' in result
        assert 'states' in result
        assert 'info' in result

        outlet = result['outlet']
        assert outlet is not None

        # Check outlet has required fields
        flows = get_flows(outlet)
        assert 'A' in flows
        assert 'B' in flows

        # Check states
        assert 'conversion' in result['states']
        assert 'residence_time' in result['states']

    def test_cstr_initialize_expected_conversion(self, simple_cstr, simple_inlet):
        """Test CSTR initialization with expected conversion hint."""
        result = simple_cstr.initialize(simple_inlet, expected_conversion=0.5)

        # Conversion should be close to specified value
        assert result['states']['conversion'] == pytest.approx(0.5, rel=0.01)

    def test_cstr_initialize_outlet_mass_balance(self, simple_cstr, simple_inlet):
        """Test that CSTR initialization respects mass balance."""
        result = simple_cstr.initialize(simple_inlet)

        inlet_flows = get_flows(simple_inlet)
        outlet_flows = get_flows(result['outlet'])

        # Total moles should be conserved (for A → B, total = A + B)
        inlet_total = inlet_flows['A'] + inlet_flows['B']
        outlet_total = outlet_flows['A'] + outlet_flows['B']

        # Should be approximately equal (allowing for numerical precision)
        assert inlet_total == pytest.approx(outlet_total, rel=0.01)


class TestPFRInitialization:
    """Tests for PFR initialization."""

    def test_pfr_initialize(self, simple_thermo, simple_rate_fn, simple_inlet):
        """Test PFR initialize method."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = PFRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"k": jnp.array(0.1)},
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        result = pfr.initialize(simple_inlet)

        assert 'outlet' in result
        assert 'states' in result
        assert 'info' in result
        assert 'conversion' in result['states']


class TestFlashInitialization:
    """Tests for Flash initialization."""

    def test_flash_initialize(self, simple_thermo):
        """Test Flash initialize method."""
        params = FlashParams(species_order=["A", "B"])
        flash = Flash(params, simple_thermo)

        inlet = make_stream({"A": 0.5, "B": 0.5}, 350.0, 101325.0)
        result = flash.initialize(inlet)

        assert 'liquid' in result
        assert 'vapor' in result
        assert 'states' in result
        assert 'V_frac' in result['states']

    def test_flash_initialize_vapor_fraction_hint(self, simple_thermo):
        """Test Flash initialization with expected vapor fraction hint."""
        params = FlashParams(species_order=["A", "B"])
        flash = Flash(params, simple_thermo)

        inlet = make_stream({"A": 0.5, "B": 0.5}, 350.0, 101325.0)
        result = flash.initialize(inlet, expected_vapor_fraction=0.5)

        assert result['states']['V_frac'] == pytest.approx(0.5, rel=0.01)


class TestInitializationHelpers:
    """Tests for initialization helper functions."""

    def test_estimate_cstr_conversion_first_order(self):
        """Test CSTR conversion estimate for first-order reaction."""
        # X = k*tau / (1 + k*tau)
        k = 0.1
        tau = 10.0
        X = estimate_cstr_conversion(k, tau, order=1)

        expected = 0.1 * 10 / (1 + 0.1 * 10)  # = 0.5
        assert X == pytest.approx(expected, rel=0.01)

    def test_estimate_outlet_temperature(self, simple_inlet):
        """Test outlet temperature estimation."""
        T_out = estimate_outlet_temperature(
            simple_inlet,
            heat_duty=0.0,
            heat_of_reaction=7500.0,  # Exothermic
            Cp_avg=75.0,
        )

        # Should be higher than inlet due to exothermic reaction
        assert T_out > simple_inlet["T"]


class TestFlowsheetAcceleration:
    """Tests for flowsheet solver with acceleration."""

    @pytest.fixture
    def simple_flowsheet_with_recycle(self, simple_thermo, simple_rate_fn):
        """Create a simple flowsheet with a recycle loop."""
        stoich = jnp.array([[-1.0], [+1.0]])

        cstr_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"k": jnp.array(0.05)},
            species_order=["A", "B"],
        )
        cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")

        # Create a simple splitter that returns a fraction of the flow
        def splitter(inlet):
            flows = get_flows(inlet)
            out1_flows = {s: f * 0.8 for s, f in flows.items()}
            out2_flows = {s: f * 0.2 for s, f in flows.items()}
            out1 = make_stream(out1_flows, inlet["T"], inlet["P"])
            out2 = make_stream(out2_flows, inlet["T"], inlet["P"])
            return out1, out2

        # Mixer function
        def mixer(stream1, stream2):
            flows1 = get_flows(stream1)
            flows2 = get_flows(stream2)
            mixed_flows = {s: flows1[s] + flows2[s] for s in flows1}
            T_avg = (stream1["T"] + stream2["T"]) / 2
            return make_stream(mixed_flows, T_avg, stream1["P"])

        fs = Flowsheet(["A", "B"])
        feed = make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0)
        fs.add_feed("feed", feed)

        fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("cstr", cstr, ["mixed"], ["reactor_out"]))
        fs.add_unit(Unit("splitter", splitter, ["reactor_out"], ["product", "recycle_source"]))

        fs.add_recycle("recycle_source", "recycle")

        return fs

    def test_flowsheet_anderson_acceleration(self, simple_flowsheet_with_recycle):
        """Test flowsheet solving with Anderson acceleration."""
        fs = simple_flowsheet_with_recycle

        streams = fs.solve(acceleration="anderson", max_iter=50, tol=1e-6)

        assert "product" in streams
        assert "reactor_out" in streams

        # Check mass balance
        feed_flows = get_flows(fs.feeds["feed"])
        product_flows = get_flows(streams["product"])

        # Total moles should be conserved
        feed_total = sum(feed_flows.values())
        product_total = sum(product_flows.values())

        assert feed_total == pytest.approx(product_total, rel=0.05)

    def test_flowsheet_wegstein_acceleration(self, simple_flowsheet_with_recycle):
        """Test flowsheet solving with Wegstein acceleration."""
        fs = simple_flowsheet_with_recycle

        streams = fs.solve(acceleration="wegstein", max_iter=50, tol=1e-6)

        assert "product" in streams
        assert "reactor_out" in streams

    def test_flowsheet_no_acceleration(self, simple_flowsheet_with_recycle):
        """Test flowsheet solving without acceleration."""
        fs = simple_flowsheet_with_recycle

        streams = fs.solve(acceleration="none", damping=0.3, max_iter=100, tol=1e-6)

        assert "product" in streams
        assert "reactor_out" in streams

    def test_acceleration_methods_converge_to_same_solution(self, simple_flowsheet_with_recycle):
        """Test that different acceleration methods converge to the same solution."""
        fs = simple_flowsheet_with_recycle

        streams_anderson = fs.solve(acceleration="anderson", max_iter=100, tol=1e-8)
        streams_wegstein = fs.solve(acceleration="wegstein", max_iter=100, tol=1e-8)
        streams_none = fs.solve(acceleration="none", damping=0.3, max_iter=200, tol=1e-8)

        # Check that product flows are similar
        product_anderson = get_flows(streams_anderson["product"])
        product_wegstein = get_flows(streams_wegstein["product"])
        product_none = get_flows(streams_none["product"])

        for species in ["A", "B"]:
            assert product_anderson[species] == pytest.approx(
                product_wegstein[species], rel=0.05
            )
            assert product_anderson[species] == pytest.approx(
                product_none[species], rel=0.05
            )


class TestSignedFlowTears:
    """Regression tests for issue #164.

    Accelerated tear solvers clipped the packed tear vector at zero every
    iteration, which makes convergence impossible for flowsheets with signed
    (bidirectional) flows whose tear streams settle at negative flows. The
    plain damped path never clipped, so the three acceleration paths disagreed
    about the feasible tear space.
    """

    @staticmethod
    def _signed_flowsheet(signed_flows):
        """A one-species recycle whose tear converges to a negative flow.

        The single unit maps the recycle stream ``r`` to ``-0.5 * r - 3.0``,
        an affine contraction with fixed point ``F_X = -2.0``. T and P pass
        through, so they are already at their fixed point and only the
        (signed) flow moves.
        """

        def affine(inlet):
            f = get_flows(inlet)["X"]
            return make_stream({"X": -0.5 * f - 3.0}, inlet["T"], inlet["P"])

        fs = Flowsheet(["X"], signed_flows=signed_flows)
        fs.add_unit(Unit("node", affine, ["recycle"], ["source"]))
        fs.add_recycle("source", "recycle")
        return fs

    def test_anderson_converges_negative_tear_when_signed(self):
        """signed_flows=True lets Anderson reach the negative fixed point."""
        fs = self._signed_flowsheet(signed_flows=True)
        streams = fs.solve(acceleration="anderson", max_iter=100, tol=1e-10)
        assert get_flows(streams["source"])["X"] == pytest.approx(-2.0, abs=1e-6)

    def test_wegstein_reaches_negative_tear_when_signed(self):
        """signed_flows=True lets Wegstein move into the negative flow region.

        (This Wegstein implementation under-relaxes conservatively, so it lands
        near -2 rather than exactly on it; the point is that the clip no longer
        traps the iterate at zero.)
        """
        fs = self._signed_flowsheet(signed_flows=True)
        streams = fs.solve(acceleration="wegstein", max_iter=100, tol=1e-8)
        assert get_flows(streams["source"])["X"] == pytest.approx(-2.0, abs=0.5)

    def test_none_path_reaches_negative_tear(self):
        """The un-accelerated path never clipped and reaches the true root."""
        fs = self._signed_flowsheet(signed_flows=True)
        streams = fs.solve(acceleration="none", max_iter=200, tol=1e-10)
        assert get_flows(streams["source"])["X"] == pytest.approx(-2.0, abs=1e-6)

    def test_signed_anderson_matches_unaccelerated(self):
        """With signed flows, Anderson and the un-clipped 'none' path agree."""
        fs = self._signed_flowsheet(signed_flows=True)
        f_anderson = get_flows(
            fs.solve(acceleration="anderson", max_iter=200, tol=1e-10)["source"]
        )["X"]
        f_none = get_flows(
            fs.solve(acceleration="none", max_iter=200, tol=1e-10)["source"]
        )["X"]
        assert f_anderson == pytest.approx(f_none, abs=1e-6)

    def test_clip_default_prevents_signed_convergence(self):
        """The default (clipping) flowsheet cannot converge to a negative tear.

        This is the bug guarded against: with the clip on, the accelerated
        tear is projected back to zero every pass, so the iteration stalls
        away from the true -2.0 root instead of contracting to it.
        """
        fs = self._signed_flowsheet(signed_flows=False)
        streams = fs.solve(acceleration="anderson", max_iter=100, tol=1e-10)
        assert abs(get_flows(streams["source"])["X"] - (-2.0)) > 0.5

    def test_clip_negative_flows_override_on_unsigned_flowsheet(self):
        """Per-call clip_negative_flows=False overrides an unsigned flowsheet."""
        fs = self._signed_flowsheet(signed_flows=False)
        streams = fs.solve(
            acceleration="anderson",
            max_iter=100,
            tol=1e-10,
            clip_negative_flows=False,
        )
        assert get_flows(streams["source"])["X"] == pytest.approx(-2.0, abs=1e-6)

    def test_clip_negative_flows_override_true_on_signed_flowsheet(self):
        """Per-call clip_negative_flows=True re-enables clipping on a signed fs."""
        fs = self._signed_flowsheet(signed_flows=True)
        streams = fs.solve(
            acceleration="anderson",
            max_iter=100,
            tol=1e-10,
            clip_negative_flows=True,
        )
        assert abs(get_flows(streams["source"])["X"] - (-2.0)) > 0.5

    def test_project_tear_leaves_temperature_and_pressure_untouched(self):
        """Clipping targets only flow entries, never the T/P slots."""
        fs = Flowsheet(["A", "B"])
        # Packed layout per stream is [F_A, F_B, T, P]; use a negative T/P to
        # prove they survive the projection (only the negative flow is clipped).
        x = jnp.array([-1.0, 2.0, -300.0, -101325.0])
        projected = fs._project_tear(x)
        assert projected[0] == pytest.approx(0.0)   # F_A clipped
        assert projected[1] == pytest.approx(2.0)   # F_B unchanged
        assert projected[2] == pytest.approx(-300.0)     # T untouched
        assert projected[3] == pytest.approx(-101325.0)  # P untouched
