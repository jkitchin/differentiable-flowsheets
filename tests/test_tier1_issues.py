"""Tests for Tier 1 issue fixes.

Covers:
- #154: Tear stream initial guess configuration
- #114: REE cascade mass balance verification
- #85/#86: Flash EOS integration & phase detection
- #125: Dynamic state non-negativity clipping
"""

import pytest
import jax
import jax.numpy as jnp

# ============================================================================
# #154: Tear stream initial guess configuration
# ============================================================================


class TestTearStreamConfig:
    """Tests for configurable tear stream defaults."""

    def test_default_flow_propagates(self):
        """Custom default_flow is used in _make_zero_stream."""
        from difflow.flowsheet import Flowsheet

        fs = Flowsheet(["A", "B"], default_flow=0.1)
        stream = fs._make_zero_stream()
        assert float(stream["F_A"]) == pytest.approx(0.1, abs=1e-10)
        assert float(stream["F_B"]) == pytest.approx(0.1, abs=1e-10)

    def test_default_T_P_propagates(self):
        """Custom default_T and default_P are used in _make_zero_stream."""
        from difflow.flowsheet import Flowsheet

        fs = Flowsheet(["A"], default_T=400.0, default_P=200000.0)
        stream = fs._make_zero_stream()
        assert float(stream["T"]) == pytest.approx(400.0, abs=1e-10)
        assert float(stream["P"]) == pytest.approx(200000.0, abs=1e-10)

    def test_backward_compat_defaults(self):
        """Default values match the original hardcoded values."""
        from difflow.flowsheet import Flowsheet

        fs = Flowsheet(["A"])
        stream = fs._make_zero_stream()
        assert float(stream["F_A"]) == pytest.approx(0.01, abs=1e-10)
        assert float(stream["T"]) == pytest.approx(300.0, abs=1e-10)
        assert float(stream["P"]) == pytest.approx(101325.0, abs=1e-10)

    def test_apply_params_preserves_defaults(self):
        """_apply_params copies custom defaults to new flowsheet."""
        from difflow.flowsheet import Flowsheet

        fs = Flowsheet(["A"], default_flow=0.5, default_T=350.0)
        new_fs = fs._apply_params({})
        assert new_fs.default_flow == 0.5
        assert new_fs.default_T == 350.0


# ============================================================================
# #114: REE cascade mass balance verification
# ============================================================================


class TestREEMassBalance:
    """Tests for mass balance verification in REE flowsheets."""

    @pytest.fixture
    def ree_feed(self):
        """Create a simple REE feed stream."""
        from difflow.streams import make_stream

        return make_stream(
            {"La": 1.0, "Ce": 2.0, "Nd": 0.5, "H2O": 100.0},
            298.15, 101325.0,
        )

    def test_extract_strip_mass_balance(self, ree_feed):
        """ExtractStripCircuit returns mass_balance dict."""
        from difflow_ree.flowsheets.extract_strip import (
            ExtractStripCircuit,
            ExtractStripParams,
        )

        params = ExtractStripParams(
            extractant="D2EHPA",
            elements=("La", "Ce", "Nd"),
            n_extraction_stages=5,
            n_stripping_stages=3,
        )
        circuit = ExtractStripCircuit(params)
        results = circuit(ree_feed)

        assert "mass_balance" in results
        mb = results["mass_balance"]
        assert "feed" in mb
        assert "output" in mb
        assert "closure" in mb

        # Closure should be close to 1.0 for each element
        for elem in ("La", "Ce", "Nd"):
            assert float(mb["closure"][elem]) == pytest.approx(1.0, abs=0.05)

    def test_extract_scrub_strip_mass_balance(self, ree_feed):
        """ExtractScrubStripCircuit returns mass_balance dict."""
        from difflow_ree.flowsheets.extract_scrub_strip import (
            ExtractScrubStripCircuit,
            ExtractScrubStripParams,
        )

        params = ExtractScrubStripParams(
            extractant="D2EHPA",
            elements=("La", "Ce", "Nd"),
            target_elements=("Nd",),
            n_extraction_stages=5,
            n_scrubbing_stages=3,
            n_stripping_stages=3,
        )
        circuit = ExtractScrubStripCircuit(params)
        results = circuit(ree_feed)

        assert "mass_balance" in results
        mb = results["mass_balance"]
        for elem in ("La", "Ce", "Nd"):
            assert float(mb["closure"][elem]) == pytest.approx(1.0, abs=0.05)

    def test_split_shell_mass_balance(self):
        """SplitShellCascade returns mass_balance dict."""
        from difflow.streams import make_stream
        from difflow_ree.flowsheets.split_shell import (
            SplitShellCascade,
            SplitShellParams,
        )

        feed = make_stream(
            {"La": 1.0, "Ce": 2.0, "Nd": 0.5, "Dy": 0.3, "H2O": 100.0},
            298.15, 101325.0,
        )
        solvent = make_stream(
            {"kerosene": 100.0, "D2EHPA": 50.0},
            298.15, 101325.0,
        )

        params = SplitShellParams(
            extractant="D2EHPA",
            elements=("La", "Ce", "Nd", "Dy"),
            n_stages=12,
            split_points=(4, 8),
        )
        cascade = SplitShellCascade(params)
        results = cascade(feed, solvent)

        assert "mass_balance" in results
        mb = results["mass_balance"]
        for elem in ("La", "Ce", "Nd", "Dy"):
            assert float(mb["closure"][elem]) == pytest.approx(1.0, abs=0.05)


# ============================================================================
# #85/#86: Flash EOS integration & phase detection
# ============================================================================


def _make_binary_thermo():
    """Create a binary light/heavy thermo for flash tests.

    Antoine coefficients are chosen so that:
    - Light Psat ≈ 1 atm at ~310 K
    - Heavy Psat ≈ 1 atm at ~400 K
    This gives a clear two-phase region around 350 K at 1 atm.
    """
    from difflow.thermo import IdealThermo, SpeciesData

    # log10(Psat/Pa) = A - B/(T+C)
    # For Light: 5.01 = A - B/(310+C) => Psat ≈ 101325 at 310 K
    # For Heavy: 5.01 = A - B/(400+C) => Psat ≈ 101325 at 400 K
    species_data = {
        "Light": SpeciesData(
            name="Light",
            MW=72.0,
            Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 470.0),
            antoine_coeffs=(12.0, 2170.0, -40.0),
        ),
        "Heavy": SpeciesData(
            name="Heavy",
            MW=114.0,
            Cp_coeffs=(200.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(40000.0, 0.38, 570.0),
            antoine_coeffs=(12.0, 2520.0, -50.0),
        ),
    }
    return IdealThermo(species_data)


class TestFlashPhaseDetection:
    """Tests for phase detection flags in Flash info dict."""

    @pytest.fixture
    def flash_setup(self):
        """Create a Flash with ideal thermo for testing."""
        from difflow.units.flash import Flash, FlashParams
        from difflow.streams import make_stream

        thermo = _make_binary_thermo()
        params = FlashParams(species_order=["Light", "Heavy"])
        flash = Flash(params, thermo)
        feed = make_stream({"Light": 1.0, "Heavy": 1.0}, 375.0, 101325.0)
        return flash, feed

    def test_phase_flag_in_info(self, flash_setup):
        """Flash info dict contains phase_flag, bubble_check, dew_check."""
        flash, feed = flash_setup
        liquid, vapor, info = flash(feed)

        assert "phase_flag" in info
        assert "bubble_check" in info
        assert "dew_check" in info

    def test_phase_flag_subcooled(self):
        """Low temperature gives subcooled (phase_flag=1)."""
        from difflow.units.flash import Flash, FlashParams
        from difflow.streams import make_stream

        thermo = _make_binary_thermo()
        params = FlashParams(species_order=["Light", "Heavy"])
        flash = Flash(params, thermo)

        # Very low temperature -> subcooled liquid
        feed = make_stream({"Light": 1.0, "Heavy": 1.0}, 250.0, 101325.0)
        _, _, info = flash(feed)
        assert int(info["phase_flag"]) == 1

    def test_phase_flag_superheated(self):
        """High temperature gives superheated vapor (phase_flag=2)."""
        from difflow.units.flash import Flash, FlashParams
        from difflow.streams import make_stream

        thermo = _make_binary_thermo()
        params = FlashParams(species_order=["Light", "Heavy"])
        flash = Flash(params, thermo)

        # At 450K both K >> 1, dew_check < 1 -> superheated vapor
        feed = make_stream({"Light": 1.0, "Heavy": 1.0}, 450.0, 101325.0)
        _, _, info = flash(feed)
        assert int(info["phase_flag"]) == 2

    def test_backward_compat_no_eos(self, flash_setup):
        """Flash without eos still works identically."""
        flash, feed = flash_setup
        liquid, vapor, info = flash(feed)

        # Basic sanity: mass balance
        total_in = float(feed["F_Light"]) + float(feed["F_Heavy"])
        total_out = (
            float(liquid["F_Light"]) + float(liquid["F_Heavy"])
            + float(vapor["F_Light"]) + float(vapor["F_Heavy"])
        )
        assert total_out == pytest.approx(total_in, rel=1e-6)

    def test_flash_with_eos(self):
        """Flash with EOS parameter uses flash_TP_eos."""
        from difflow.units.flash import Flash, FlashParams
        from difflow.thermo import IdealThermo, SpeciesData
        from difflow.eos import PengRobinson, CriticalProperties
        from difflow.streams import make_stream

        species_data = {
            "methane": CriticalProperties("methane", 190.6, 4.6e6, 0.011),
            "ethane": CriticalProperties("ethane", 305.4, 4.9e6, 0.099),
        }
        eos = PengRobinson(species_data)

        # Thermo needed for constructor but not used when eos is provided
        thermo_data = {
            "methane": SpeciesData(
                name="methane", MW=16.04,
                Cp_coeffs=(35.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(8200.0, 0.38, 190.6),
                antoine_coeffs=(8.7, 574.0, -7.0),
            ),
            "ethane": SpeciesData(
                name="ethane", MW=30.07,
                Cp_coeffs=(52.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(14700.0, 0.38, 305.4),
                antoine_coeffs=(9.1, 1511.0, -17.0),
            ),
        }
        thermo = IdealThermo(thermo_data)
        params = FlashParams(species_order=["methane", "ethane"])
        flash = Flash(params, thermo, eos=eos)

        # 250 K / 50 bar is genuinely two-phase for this 60/40 mixture (the PR
        # two-phase window here is ~40-60 bar; methane is supercritical, so the
        # split occurs at high pressure). At the earlier 20 bar the mixture is a
        # single vapor phase -- the ungated flash used to leave x as an
        # incipient-liquid composition there, but flash_TP_eos now runs a
        # phase-stability test and correctly reports V=1 with x = feed.
        feed = make_stream({"methane": 0.6, "ethane": 0.4}, 250.0, 5e6)
        liquid, vapor, info = flash(feed)

        # Genuinely two-phase, and methane enriched in the vapor.
        assert 0.0 < float(info["V_frac"]) < 1.0
        assert float(info["y"]["methane"]) > float(info["x"]["methane"])

        # Mass balance
        total_in = 1.0
        total_out = float(info["L"]) + float(info["V"])
        assert total_out == pytest.approx(total_in, rel=1e-4)


# ============================================================================
# #125: Dynamic state non-negativity clipping
# ============================================================================


class TestDynamicNonNegativity:
    """Tests for state bounds enforcement in integrators."""

    def test_enforce_bounds(self):
        """StateSpec.enforce_bounds clips to declared bounds."""
        from difflow.dynamic.state import StateSpec, StateVar

        spec = StateSpec([
            StateVar("n_A", "moles", bounds=(0.0, None)),
            StateVar("n_B", "moles", bounds=(0.0, None)),
            StateVar("T", "temperature", bounds=(0.0, None)),
        ])
        state = jnp.array([-0.5, 1.0, -100.0])
        clipped = spec.enforce_bounds(state)
        assert float(clipped[0]) == pytest.approx(0.0)
        assert float(clipped[1]) == pytest.approx(1.0)
        assert float(clipped[2]) == pytest.approx(0.0)

    def test_rk4_bounds_prevent_negative(self):
        """RK4 with bounds prevents negative states."""
        from difflow.dynamic.integrators import integrate_rk4

        # Stiff system: fast A -> B with large rate
        # Without clipping, explicit methods overshoot to negative
        def fast_decay(t, y):
            k = 100.0  # Very fast rate
            return jnp.array([-k * y[0], k * y[0]])

        y0 = jnp.array([1.0, 0.0])
        bounds = (jnp.array([0.0, 0.0]), jnp.array([jnp.inf, jnp.inf]))

        # With only 10 steps, large dt will cause overshoot
        result = integrate_rk4(fast_decay, y0, (0.0, 1.0), n_steps=10, bounds=bounds)
        # All states should be non-negative
        assert jnp.all(result.trajectory.y >= -1e-10)
        assert jnp.all(result.y_final >= -1e-10)

    def test_euler_bounds_prevent_negative(self):
        """Euler with bounds prevents negative states."""
        from difflow.dynamic.integrators import integrate_euler

        def fast_decay(t, y):
            k = 50.0
            return jnp.array([-k * y[0], k * y[0]])

        y0 = jnp.array([1.0, 0.0])
        bounds = (jnp.array([0.0, 0.0]), jnp.array([jnp.inf, jnp.inf]))

        result = integrate_euler(fast_decay, y0, (0.0, 1.0), n_steps=20, bounds=bounds)
        assert jnp.all(result.trajectory.y >= -1e-10)

    def test_grad_through_clipped_rk4(self):
        """jax.grad works through clipped RK4 integration."""
        from difflow.dynamic.integrators import integrate_rk4

        def fast_decay(t, y):
            k = 50.0
            return jnp.array([-k * y[0], k * y[0]])

        bounds = (jnp.array([0.0, 0.0]), jnp.array([jnp.inf, jnp.inf]))

        def objective(y0_A):
            y0 = jnp.array([y0_A, 0.0])
            result = integrate_rk4(fast_decay, y0, (0.0, 0.5), n_steps=20, bounds=bounds)
            return result.y_final[1]  # Final amount of B

        grad_fn = jax.grad(objective)
        g = grad_fn(1.0)
        # Gradient should be finite
        assert jnp.isfinite(g)
        # More initial A should produce more B (positive gradient)
        assert float(g) > 0

    def test_integrate_unified_bounds(self):
        """The unified integrate() function passes bounds through."""
        from difflow.dynamic.integrators import integrate

        def fast_decay(t, y):
            k = 100.0
            return jnp.array([-k * y[0], k * y[0]])

        y0 = jnp.array([1.0, 0.0])
        bounds = (jnp.array([0.0, 0.0]), jnp.array([jnp.inf, jnp.inf]))

        result = integrate(fast_decay, y0, (0.0, 1.0), method="RK4", n_steps=10, bounds=bounds)
        assert jnp.all(result.y_final >= -1e-10)
