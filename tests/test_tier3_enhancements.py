"""Tests for Tier 3 enhancements: core, economics, and dynamic.

10 test classes covering issues #89, #91, #92, #126, #127, #128, #129, #130, #152, #153.
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# =====================================================================
# #89 — EOS binary interaction parameters
# =====================================================================


class TestEOSBinaryInteraction:
    """Tests for EOS k_ij storage and build_kij_matrix."""

    def test_build_kij_matrix(self):
        from difflow.eos import build_kij_matrix

        species = ["methane", "ethane", "propane"]
        kij_dict = {("methane", "ethane"): 0.02, ("methane", "propane"): 0.03}
        mat = build_kij_matrix(species, kij_dict)

        assert mat.shape == (3, 3)
        assert float(mat[0, 1]) == pytest.approx(0.02)
        assert float(mat[1, 0]) == pytest.approx(0.02)  # symmetric
        assert float(mat[0, 2]) == pytest.approx(0.03)
        assert float(mat[2, 0]) == pytest.approx(0.03)
        assert float(mat[1, 2]) == pytest.approx(0.0)  # not set

    def test_pr_without_kij_backward_compat(self):
        from difflow.eos import PengRobinson, CriticalProperties

        species = {
            "methane": CriticalProperties("methane", 190.6, 4.599e6, 0.011),
            "ethane": CriticalProperties("ethane", 305.3, 4.872e6, 0.099),
        }
        pr = PengRobinson(species)
        assert pr.params.k_ij is None

        # Should still work (uses zeros internally)
        y = jnp.array([0.5, 0.5])
        Z = pr.solve_Z(200.0, 1e6, y)
        assert float(Z) > 0

    def test_pr_with_kij_dict(self):
        from difflow.eos import PengRobinson, CriticalProperties

        species = {
            "methane": CriticalProperties("methane", 190.6, 4.599e6, 0.011),
            "ethane": CriticalProperties("ethane", 305.3, 4.872e6, 0.099),
        }
        kij_dict = {("methane", "ethane"): 0.02}

        pr_no_kij = PengRobinson(species)
        pr_kij = PengRobinson(species, k_ij=kij_dict)

        assert pr_kij.params.k_ij is not None

        y = jnp.array([0.5, 0.5])
        T = 250.0

        a_mix_no = pr_no_kij.a_mix(T, y)
        a_mix_yes = pr_kij.a_mix(T, y)

        # k_ij > 0 should reduce a_mix
        assert float(a_mix_yes) < float(a_mix_no)

    def test_srk_with_kij(self):
        from difflow.eos import SRK, CriticalProperties

        species = {
            "methane": CriticalProperties("methane", 190.6, 4.599e6, 0.011),
            "ethane": CriticalProperties("ethane", 305.3, 4.872e6, 0.099),
        }
        kij = {("methane", "ethane"): 0.01}
        srk = SRK(species, k_ij=kij)
        assert srk.params.k_ij is not None
        assert float(srk.params.k_ij[0, 1]) == pytest.approx(0.01)

    def test_gradient_through_kij(self):
        from difflow.eos import PengRobinson, CriticalProperties

        species = {
            "methane": CriticalProperties("methane", 190.6, 4.599e6, 0.011),
            "ethane": CriticalProperties("ethane", 305.3, 4.872e6, 0.099),
        }
        pr = PengRobinson(species)

        def a_mix_fn(k_val):
            k_ij = jnp.array([[0.0, k_val], [k_val, 0.0]])
            return pr.a_mix(250.0, jnp.array([0.5, 0.5]), k_ij)

        grad_fn = jax.grad(a_mix_fn)
        g = grad_fn(0.02)
        assert jnp.isfinite(g)


# =====================================================================
# #91 — Antoine coefficient temperature range validation
# =====================================================================


class TestAntoineValidation:
    """Tests for Antoine temperature range validation."""

    def test_psat_with_info_in_range(self):
        from difflow.database import get_species_data
        from difflow.thermo import IdealThermo

        data = {"water": get_species_data("water")}
        thermo = IdealThermo(data)

        Psat, info = thermo.Psat_with_info("water", 373.0)
        assert float(Psat) > 0
        assert info['antoine_in_range'] is True

    def test_psat_with_info_out_of_range(self):
        from difflow.database import get_species_data
        from difflow.thermo import IdealThermo

        data = {"water": get_species_data("water")}
        thermo = IdealThermo(data)

        # 600 K is above the 473 K max for water
        Psat, info = thermo.Psat_with_info("water", 600.0)
        assert float(Psat) > 0  # Still computes, no exception
        assert info['antoine_in_range'] is False

    def test_validate_antoine(self):
        from difflow.database import get_species_data
        from difflow.thermo import IdealThermo

        data = {"water": get_species_data("water")}
        thermo = IdealThermo(data)

        result = thermo.validate_antoine("water", 350.0)
        assert result['in_range'] is True
        assert result['T_min'] == pytest.approx(273.0)
        assert result['T_max'] == pytest.approx(473.0)

    def test_default_range_always_valid(self):
        from difflow.thermo import IdealThermo, SpeciesData

        # Species with default range (0 to 1e6) — always valid
        data = {
            "X": SpeciesData(
                name="X", MW=100.0,
                Cp_coeffs=(50.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(30000.0, 0.38, 500.0),
                antoine_coeffs=(9.0, 1000.0, -30.0),
            )
        }
        thermo = IdealThermo(data)
        result = thermo.validate_antoine("X", 5000.0)
        assert result['in_range'] is True


# =====================================================================
# #92 — Stream mixing phase compatibility check
# =====================================================================


class TestStreamPhase:
    """Tests for stream phase labeling and combine_streams compatibility."""

    def test_make_stream_no_phase(self):
        from difflow.streams import make_stream

        s = make_stream({"A": 1.0}, T=300.0, P=1e5)
        assert "phase" not in s

    def test_make_stream_with_phase(self):
        from difflow.streams import make_stream

        s = make_stream({"A": 1.0}, T=300.0, P=1e5, phase="liquid")
        assert s["phase"] == "liquid"

    def test_combine_same_phase(self):
        from difflow.streams import make_stream, combine_streams

        s1 = make_stream({"A": 1.0}, T=300.0, P=1e5, phase="liquid")
        s2 = make_stream({"A": 2.0}, T=310.0, P=1e5, phase="liquid")
        result = combine_streams(s1, s2)
        assert result["phase"] == "liquid"
        assert "phase_mismatch" not in result

    def test_combine_mixed_phases(self):
        from difflow.streams import make_stream, combine_streams

        s1 = make_stream({"A": 1.0}, T=300.0, P=1e5, phase="liquid")
        s2 = make_stream({"A": 2.0}, T=400.0, P=1e5, phase="vapor")
        result = combine_streams(s1, s2)
        assert result["phase"] == "two_phase"
        assert result.get("phase_mismatch") is True

    def test_combine_none_plus_labeled(self):
        from difflow.streams import make_stream, combine_streams

        s1 = make_stream({"A": 1.0}, T=300.0, P=1e5)
        s2 = make_stream({"A": 2.0}, T=310.0, P=1e5, phase="liquid")
        result = combine_streams(s1, s2)
        # Only one labeled → should get that phase
        assert result["phase"] == "liquid"

    def test_combine_no_phase_info(self):
        from difflow.streams import make_stream, combine_streams

        s1 = make_stream({"A": 1.0}, T=300.0, P=1e5)
        s2 = make_stream({"A": 2.0}, T=310.0, P=1e5)
        result = combine_streams(s1, s2)
        assert "phase" not in result


# =====================================================================
# #127 — CAPEX scaling exponent validation
# =====================================================================


class TestCAPEXValidation:
    """Tests for validate_cost_params."""

    def test_valid_params(self):
        from difflow.economics.capital import (
            REACTOR_COSTS, validate_cost_params,
        )

        params = REACTOR_COSTS["cstr_jacketed"]
        result = validate_cost_params(params, 5.0)
        assert result['exponent_valid'] is True
        assert result['size_in_range'] is True

    def test_extreme_exponent(self):
        from difflow.economics.capital import CostParams, validate_cost_params

        params = CostParams(a=1000, b=100, n=2.0, S_min=1, S_max=100,
                            S_units="m³", base_year=2019)
        result = validate_cost_params(params, 50.0)
        assert result['exponent_valid'] is False

    def test_size_out_of_range(self):
        from difflow.economics.capital import REACTOR_COSTS, validate_cost_params

        params = REACTOR_COSTS["cstr_jacketed"]
        result = validate_cost_params(params, 500.0)  # S_max is 100
        assert result['size_in_range'] is False

    def test_valid_exponent_range_constant(self):
        from difflow.economics.capital import VALID_EXPONENT_RANGE

        assert VALID_EXPONENT_RANGE == (0.3, 1.2)


# =====================================================================
# #128 — CEPCI cost index update
# =====================================================================


class TestCEPCIUpdate:
    """Tests for CEPCI 2025/2026 data and estimation."""

    def test_cepci_2026_available(self):
        from difflow.economics.indices import get_cepci

        val = get_cepci(2026)
        assert val == pytest.approx(820.0)

    def test_cepci_2025_available(self):
        from difflow.economics.indices import get_cepci

        val = get_cepci(2025)
        assert val == pytest.approx(810.0)

    def test_default_current_year(self):
        from difflow.economics.indices import DEFAULT_CURRENT_YEAR

        assert DEFAULT_CURRENT_YEAR == 2026

    def test_estimate_cepci_known_year(self):
        from difflow.economics.indices import estimate_cepci

        val = estimate_cepci(2024)
        assert val == pytest.approx(800.0)

    def test_estimate_cepci_future_year(self):
        from difflow.economics.indices import estimate_cepci

        val = estimate_cepci(2030)
        # Should extrapolate linearly from last 3 points (2024, 2025, 2026)
        assert val > 820.0  # Should be above 2026
        assert val < 1000.0  # Reasonable bound

    def test_cepci_available_years(self):
        from difflow.economics.indices import cepci_available_years

        years = cepci_available_years()
        assert 2000 in years
        assert 2026 in years
        assert len(years) >= 27


# =====================================================================
# #153 — Utility costs by region
# =====================================================================


class TestRegionalPrices:
    """Tests for regional utility price presets."""

    def test_default_matches_gulf_coast(self):
        from difflow.economics.utilities import (
            DEFAULT_PRICES, REGIONAL_PRICES,
        )

        gulf = REGIONAL_PRICES['us_gulf_coast']
        assert gulf.electricity == pytest.approx(DEFAULT_PRICES.electricity)
        assert gulf.steam_high_pressure == pytest.approx(
            DEFAULT_PRICES.steam_high_pressure
        )

    def test_regional_prices_differ(self):
        from difflow.economics.utilities import REGIONAL_PRICES

        gulf = REGIONAL_PRICES['us_gulf_coast']
        europe = REGIONAL_PRICES['europe_west']
        # Europe electricity should be ~2x US
        assert europe.electricity > gulf.electricity

    def test_from_region_classmethod(self):
        from difflow.economics.utilities import UtilityPrices

        prices = UtilityPrices.from_region('china')
        assert prices.electricity < 0.07  # Less than US Gulf Coast

    def test_from_region_invalid(self):
        from difflow.economics.utilities import UtilityPrices

        with pytest.raises(ValueError, match="Unknown region"):
            UtilityPrices.from_region("antarctica")


# =====================================================================
# #126 — Diffrax backend error control parameters
# =====================================================================


class TestDiffraxErrorControl:
    """Tests for diffrax rtol/atol/dtmin/dtmax passthrough."""

    def test_default_tolerances_rk45(self):
        from difflow.dynamic.integrators import integrate

        def f(t, y):
            return -0.1 * y

        y0 = jnp.array([1.0])
        result = integrate(f, y0, (0.0, 10.0), method="RK45")
        expected = jnp.exp(-1.0)
        assert float(result.y_final[0]) == pytest.approx(float(expected), rel=1e-3)

    def test_tight_tolerance_more_accurate(self):
        from difflow.dynamic.integrators import integrate

        def f(t, y):
            return -0.1 * y

        y0 = jnp.array([1.0])
        expected = float(jnp.exp(-1.0))

        result_loose = integrate(f, y0, (0.0, 10.0), "RK45", rtol=1e-3, atol=1e-5)
        result_tight = integrate(f, y0, (0.0, 10.0), "RK45", rtol=1e-10, atol=1e-12)

        err_loose = abs(float(result_loose.y_final[0]) - expected)
        err_tight = abs(float(result_tight.y_final[0]) - expected)
        assert err_tight <= err_loose

    def test_diffrax_dtmin_dtmax_accepted(self):
        """Test that dtmin/dtmax kwargs are accepted (no error)."""
        try:
            from difflow.dynamic.diffrax_backend import HAS_DIFFRAX
            if not HAS_DIFFRAX:
                pytest.skip("diffrax not installed")
        except ImportError:
            pytest.skip("diffrax not installed")

        from difflow.dynamic.integrators import integrate

        def f(t, y):
            return -0.1 * y

        y0 = jnp.array([1.0])
        result = integrate(
            f, y0, (0.0, 10.0), "diffrax:tsit5",
            dtmin=1e-10, dtmax=1.0,
        )
        assert result.info.success


# =====================================================================
# #129 — Integrator stiffness auto-detection
# =====================================================================


class TestStiffnessDetection:
    """Tests for estimate_stiffness and method='auto'."""

    def test_nonstiff_system(self):
        from difflow.dynamic.integrators import estimate_stiffness

        # Simple harmonic oscillator — not stiff
        def f(t, y):
            return jnp.array([y[1], -y[0]])

        result = estimate_stiffness(f, jnp.array([1.0, 0.0]))
        assert result['stiffness_ratio'] < 100
        assert result['is_stiff'] is False

    def test_stiff_system(self):
        from difflow.dynamic.integrators import estimate_stiffness

        # Stiff system: fast + slow reactions
        def f(t, y):
            return jnp.array([-1000.0 * y[0] + y[1], y[0] - 0.001 * y[1]])

        result = estimate_stiffness(f, jnp.array([1.0, 1.0]))
        assert result['stiffness_ratio'] > 100
        assert result['is_stiff'] is True

    def test_auto_selects_method(self):
        from difflow.dynamic.integrators import integrate

        # Non-stiff: auto should select RK45
        def f(t, y):
            return jnp.array([y[1], -y[0]])

        result = integrate(f, jnp.array([1.0, 0.0]), (0.0, 1.0), "auto")
        assert "auto" in result.info.message
        assert result.info.success


# =====================================================================
# #152 — Dynamic CSTR variable-volume operation
# =====================================================================


class TestDynamicCSTRVariableVolume:
    """Tests for variable-volume DynamicCSTR."""

    def _make_cstr(self, variable_volume=False):
        from difflow.dynamic.base import DynamicCSTR
        from difflow.streams import make_stream

        def rate_fn(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"]])

        cstr = DynamicCSTR(
            volume=1.0,
            rate_fn=rate_fn,
            stoich=jnp.array([[-1], [1]]),
            species_order=["A", "B"],
            rate_params={"k": 0.1},
            variable_volume=variable_volume,
        )
        inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
        return cstr, {"inlet": inlet}

    def test_fixed_volume_backward_compat(self):
        from difflow.dynamic.integrators import integrate_unit

        cstr, inputs = self._make_cstr(variable_volume=False)
        y0 = cstr.initial_state(inputs)
        spec = cstr.state_spec()

        # Should NOT have V in state
        assert "V" not in spec.names
        assert len(y0) == 2  # n_A, n_B

        result = integrate_unit(cstr, inputs, (0.0, 100.0), method="RK4", n_steps=50)
        assert result.info.success

    def test_variable_volume_has_V_state(self):
        cstr, inputs = self._make_cstr(variable_volume=True)
        spec = cstr.state_spec()

        assert "V" in spec.names
        y0 = cstr.initial_state(inputs)
        assert len(y0) == 3  # V, n_A, n_B
        assert float(y0[0]) == pytest.approx(1.0)  # Initial volume

    def test_variable_volume_equal_flows(self):
        """With equal in/out flows, volume should stay constant."""
        from difflow.dynamic.integrators import integrate_unit

        cstr, inputs = self._make_cstr(variable_volume=True)
        result = integrate_unit(cstr, inputs, (0.0, 100.0), method="RK4", n_steps=100)

        # Volume should stay near 1.0 since F_in = F_out by assumption
        V_final = float(result.y_final[0])
        assert V_final == pytest.approx(1.0, abs=0.1)


# =====================================================================
# #130 — Dynamic flowsheet event handling
# =====================================================================


class TestEventHandling:
    """Tests for event detection in integration."""

    def test_threshold_crossing(self):
        from difflow.dynamic.integrators import (
            integrate, detect_events, EventSpec,
        )

        # Decaying exponential: y(t) = exp(-0.1*t), crosses 0.5 at t=ln(2)/0.1
        def f(t, y):
            return -0.1 * y

        y0 = jnp.array([1.0])
        result = integrate(f, y0, (0.0, 20.0), "RK4", n_steps=200)

        events = [EventSpec(
            name="half_life",
            condition_fn=lambda t, y: y[0] - 0.5,
            direction=-1,
        )]
        detected = detect_events(result, events)
        assert len(detected) >= 1
        assert detected[0].name == "half_life"
        expected_t = jnp.log(2.0) / 0.1
        assert detected[0].t_event == pytest.approx(float(expected_t), rel=0.05)

    def test_direction_filtering(self):
        from difflow.dynamic.integrators import (
            integrate, detect_events, EventSpec,
        )

        # Oscillator: sin crosses zero in both directions
        def f(t, y):
            return jnp.array([y[1], -y[0]])

        y0 = jnp.array([0.0, 1.0])  # sin, cos
        result = integrate(f, y0, (0.0, 10.0), "RK4", n_steps=200)

        # Only detect upward crossings of y[0] = 0
        events_up = [EventSpec("up_cross", lambda t, y: y[0], direction=1)]
        events_down = [EventSpec("down_cross", lambda t, y: y[0], direction=-1)]

        up = detect_events(result, events_up)
        down = detect_events(result, events_down)

        # Both should find crossings but they should be different
        assert len(up) > 0
        assert len(down) > 0
        # Up crossings at t=0, 2pi, etc.; down crossings at t=pi, 3pi, etc.
        if len(up) > 0 and len(down) > 0:
            assert abs(up[0].t_event - down[0].t_event) > 1.0

    def test_no_events_when_none_specified(self):
        from difflow.dynamic.integrators import integrate, detect_events

        def f(t, y):
            return -0.1 * y

        y0 = jnp.array([1.0])
        result = integrate(f, y0, (0.0, 10.0), "RK4", n_steps=50)
        detected = detect_events(result, [])
        assert detected == []
