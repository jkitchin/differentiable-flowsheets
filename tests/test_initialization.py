"""Tests for initialization strategies and acceleration methods."""

import warnings

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

        assert x_new.shape == (1,)
        assert jnp.isfinite(x_new).all()
        # Wegstein is exact for a linear map: one step lands on the root (0.5),
        # and must be at least as close as plain substitution (g_curr = 0.375).
        assert float(x_new[0]) == pytest.approx(0.5, abs=1e-6)
        assert abs(float(x_new[0]) - 0.5) <= abs(float(g_curr[0]) - 0.5)

    def test_wegstein_converges_positive_slope_contraction(self):
        """Iterating Wegstein converges for a 0 < s < 1 contraction (#164).

        Regression guard: the update previously swapped its weights, giving an
        effective slope of 1 + s that diverged for exactly this common case.
        """
        # g(x) = 0.5 * x + 1  ->  fixed point x* = 2.0
        def g(x):
            return 0.5 * x + 1.0

        x_prev = jnp.array([0.0])
        g_prev = g(x_prev)
        x_curr = g_prev
        for _ in range(20):
            g_curr = g(x_curr)
            x_next = wegstein_acceleration(x_prev, x_curr, g_prev, g_curr)
            x_prev, g_prev, x_curr = x_curr, g_curr, x_next
        assert float(x_curr[0]) == pytest.approx(2.0, abs=1e-6)

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


class TestTearInitialization:
    """Tests for the initial guess a recycle starts from (#247).

    The guess used to come out of ``_make_zero_stream()`` every single time,
    because it was looked up by the recycle's *destination* --- an inlet, which
    by construction no unit computes.  These check that the guess now comes
    from the source end, and that the fallbacks behave when a unit in the loop
    cannot run on the near-empty stream of a first pass.
    """

    @pytest.fixture
    def recycle_flowsheet(self, simple_thermo, simple_rate_fn):
        """mixer -> CSTR -> splitter, with 20% of the reactor effluent recycled."""
        cstr_params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=jnp.array([[-1.0], [+1.0]]),
            rate_params={"k": jnp.array(0.05)},
            species_order=["A", "B"],
        )
        cstr = CSTR(cstr_params, thermo=simple_thermo, mode="isothermal")

        def splitter(inlet):
            flows = get_flows(inlet)
            return (
                make_stream({s: f * 0.8 for s, f in flows.items()},
                            inlet["T"], inlet["P"]),
                make_stream({s: f * 0.2 for s, f in flows.items()},
                            inlet["T"], inlet["P"]),
            )

        def mixer(stream1, stream2):
            flows1, flows2 = get_flows(stream1), get_flows(stream2)
            return make_stream(
                {s: flows1[s] + flows2[s] for s in flows1},
                (stream1["T"] + stream2["T"]) / 2,
                stream1["P"],
            )

        fs = Flowsheet(["A", "B"])
        fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0))
        fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("cstr", cstr, ["mixed"], ["reactor_out"]))
        fs.add_unit(Unit("splitter", splitter, ["reactor_out"],
                         ["product", "recycle_source"]))
        fs.add_recycle("recycle_source", "recycle")
        return fs

    def test_guess_is_not_the_flowsheet_default(self, recycle_flowsheet):
        """The regression itself: the guess has to say something about the loop.

        This is the check from the issue.  Before the fix both sides of it were
        the 0.01 mol/s default stream, on every flowsheet, and the whole
        ``use_initialization`` path was dead code that no test noticed.
        """
        fs = recycle_flowsheet
        guess = get_flows(fs._initialize_tear_stream("recycle_source"))
        default = get_flows(fs._make_zero_stream())

        assert guess != pytest.approx(default)
        # One pass from a pure-A feed through a 5% reactor: mostly A, and a
        # fifth of the effluent, so well under the feed's 1.0 mol/s.
        assert 0.0 < float(guess["A"]) < 1.0

    def test_destination_end_has_nothing_to_propagate(self, recycle_flowsheet):
        """Asking by the destination name is what used to be wrong.

        It still answers the default, and correctly so --- ``recycle`` is an
        inlet and the propagation walk never assigns one.  The bug was the
        caller passing this end, not the answer given for it.
        """
        fs = recycle_flowsheet
        guess = get_flows(fs._initialize_tear_stream("recycle"))
        assert guess == pytest.approx(get_flows(fs._make_zero_stream()))

    def test_explicit_tear_initial_still_wins(self, recycle_flowsheet):
        """A caller who says what the tear is does not get overruled by a guess."""
        fs = recycle_flowsheet
        mine = make_stream({"A": 0.42, "B": 0.17}, 360.0, 101325.0)

        streams = fs.solve(tear_initial={"recycle": mine}, tol=1e-8, max_iter=100)

        assert fs.last_solve_converged
        assert "product" in streams

    def test_same_fixed_point_either_way(self, recycle_flowsheet):
        """A better start may not change where the iteration lands.

        Changing the default guess for every recycle flowsheet in the package
        is only safe if it moves the starting point and nothing else.
        """
        fs = recycle_flowsheet
        with_init = get_flows(
            fs.solve(use_initialization=True, tol=1e-10, max_iter=200)["product"])
        without = get_flows(
            fs.solve(use_initialization=False, tol=1e-10, max_iter=200)["product"])

        for species in ("A", "B"):
            assert with_init[species] == pytest.approx(without[species], rel=1e-6)


class TestInitializedOutletMapping:
    """Tests for reading a unit's ``initialize()`` result positionally."""

    def test_flash_outlets_map_onto_outlet_names(self, simple_thermo):
        """A two-outlet unit is only usable if its guesses can be named.

        Flash's initializer answered in ``liquid``/``vapor``, which a caller
        holding ``Unit.outlet_names`` has no way to match up --- the names on
        the flowsheet are whatever the user called them.  ``outlets`` is the
        sequence in ``__call__`` order, so the mapping is positional.
        """
        from difflow.flowsheet import _initialized_outlets

        flash = Flash(FlashParams(species_order=["A", "B"]), simple_thermo)
        unit = Unit("flash", flash, ["feed"], ["liq", "vap"])
        result = flash.initialize(make_stream({"A": 0.5, "B": 0.5}, 350.0, 101325.0))

        outlets = _initialized_outlets(result, unit)

        assert list(outlets) == ["liq", "vap"]
        assert get_flows(outlets["liq"]) == pytest.approx(get_flows(result["liquid"]))
        assert get_flows(outlets["vap"]) == pytest.approx(get_flows(result["vapor"]))

    def test_singular_outlet_key_is_read_too(self, simple_cstr, simple_inlet):
        """``outlet`` means a one-element ``outlets``, for a unit with one."""
        from difflow.flowsheet import _initialized_outlets

        unit = Unit("cstr", simple_cstr, ["feed"], ["effluent"])
        result = simple_cstr.initialize(simple_inlet)

        outlets = _initialized_outlets(result, unit)

        assert list(outlets) == ["effluent"]
        assert get_flows(outlets["effluent"]) == pytest.approx(
            get_flows(result["outlet"]))

    def test_a_unit_with_no_guesses_to_give(self, simple_cstr):
        """Nothing usable in, nothing out --- the caller falls back on its own."""
        from difflow.flowsheet import _initialized_outlets

        unit = Unit("cstr", simple_cstr, ["feed"], ["effluent"])

        assert _initialized_outlets(None, unit) == {}
        assert _initialized_outlets({"states": {}}, unit) == {}


class TestTearInitializationFallbacks:
    """What happens when a unit in the loop cannot run on the first pass.

    The propagation pass runs before any recycle is known, so a unit inside a
    loop sees the near-empty default arriving.  A CSTR's root find and a
    flash's Rachford-Rice can genuinely fail there.  That is the one place
    ``initialize()`` earns its keep.
    """

    @staticmethod
    def _build(operation):
        def splitter(inlet):
            flows = get_flows(inlet)
            return (
                make_stream({s: f * 0.7 for s, f in flows.items()},
                            inlet["T"], inlet["P"]),
                make_stream({s: f * 0.3 for s, f in flows.items()},
                            inlet["T"], inlet["P"]),
            )

        def mixer(a, b):
            fa, fb = get_flows(a), get_flows(b)
            return make_stream({s: fa[s] + fb[s] for s in fa}, a["T"], a["P"])

        fs = Flowsheet(["A", "B"])
        fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0))
        fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("op", operation, ["mixed"], ["op_out"]))
        fs.add_unit(Unit("splitter", splitter, ["op_out"],
                         ["product", "recycle_source"]))
        fs.add_recycle("recycle_source", "recycle")
        return fs

    def test_initialize_stands_in_for_a_unit_that_raised(self):
        """An analytic estimate that cannot fail, in place of the call that did."""
        from difflow.initialization import TearInitializationWarning

        class FailsButEstimates:
            def __call__(self, inlet):
                raise RuntimeError("root find did not converge")

            def initialize(self, inlet, **kwargs):
                flows = get_flows(inlet)
                return {"outlets": (make_stream(
                    {s: f * 0.9 for s, f in flows.items()},
                    inlet["T"], inlet["P"]),)}

        fs = self._build(FailsButEstimates())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            guess = get_flows(fs._initialize_tear_stream("recycle_source"))

        # An estimate that worked is not worth warning about.
        assert [w for w in caught
                if issubclass(w.category, TearInitializationWarning)] == []

        # The pass runs with the recycle still unknown, so the mixer sees the
        # feed plus the default: (1.0 + 0.01) * 0.9 through the estimate, then
        # * 0.3 out of the splitter.
        assert float(guess["A"]) == pytest.approx(1.01 * 0.9 * 0.3, rel=1e-6)

    def test_no_initializer_warns_rather_than_guessing_quietly(self):
        """A tear guess that is quietly worse than it looks is the thing to avoid."""
        from difflow.initialization import TearInitializationWarning

        class JustFails:
            def __call__(self, inlet):
                raise RuntimeError("root find did not converge")

        fs = self._build(JustFails())

        with pytest.warns(TearInitializationWarning, match="no initialize"):
            guess = get_flows(fs._initialize_tear_stream("recycle_source"))

        # The default propagated through the splitter, which is the best that
        # can be said -- but said out loud.
        assert float(guess["A"]) == pytest.approx(0.01 * 0.3, rel=1e-6)

    def test_a_failing_initializer_warns_too(self):
        """Both halves of the fallback are allowed to fail; neither may be silent."""
        from difflow.initialization import TearInitializationWarning

        class FailsTwice:
            def __call__(self, inlet):
                raise RuntimeError("root find did not converge")

            def initialize(self, inlet, **kwargs):
                raise ValueError("no bracketing interval")

        fs = self._build(FailsTwice())

        with pytest.warns(TearInitializationWarning, match="neither could its"):
            fs._initialize_tear_stream("recycle_source")

