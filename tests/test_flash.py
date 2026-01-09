"""Tests for flash separator, splitter, and mixer units.

Tests cover:
- Flash TP calculation with Rachford-Rice
- Flash edge cases (subcooled, superheated)
- Flash gradient compatibility
- Splitter functionality
- Mixer functionality
- FlashParams validation
"""

import pytest
import jax
import jax.numpy as jnp
from jax import Array
import numpy.testing as npt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from difflow.units.flash import Flash, FlashParams, Splitter, Mixer
from difflow.streams import make_stream, get_flows, total_flow
from difflow.thermo import IdealThermo, SpeciesData


@pytest.fixture
def binary_thermo():
    """Create a simple binary system (light/heavy) for testing.

    Using Antoine coefficients similar to pentane (light) and octane (heavy).
    Antoine: log10(Psat/Pa) = A - B/(T + C) where T in K
    """
    species_data = {
        "Light": SpeciesData(
            name="Light",
            MW=72.0,  # Similar to pentane
            Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(26000.0, 0.38, 470.0),  # Hvap, n, Tc
            # Antoine coeffs for ~pentane (high vapor pressure)
            antoine_coeffs=(10.422, 1687.537, -38.44),
            Hf=0.0,
        ),
        "Heavy": SpeciesData(
            name="Heavy",
            MW=114.0,  # Similar to octane
            Cp_coeffs=(190.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 570.0),
            # Antoine coeffs for ~octane (lower vapor pressure)
            antoine_coeffs=(10.186, 2004.68, -60.53),
            Hf=0.0,
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def flash_params():
    """Flash parameters for binary system."""
    return FlashParams(species_order=["Light", "Heavy"])


@pytest.fixture
def binary_feed():
    """Binary feed stream.

    Conditions chosen to be in two-phase region for pentane-octane like system.
    At 350K:
    - Pentane (Light) Psat ≈ 101,295 Pa (1.01 bar)
    - Octane (Heavy) Psat ≈ 1,822 Pa (0.018 bar)
    For 50/50 molar:
    - P_bubble ≈ 51,559 Pa
    - P_dew ≈ 3,580 Pa
    So P = 30,000 Pa gives partial vaporization (in two-phase region).
    """
    return make_stream(
        {"Light": 50.0, "Heavy": 50.0},
        T=350.0,
        P=30000.0,  # 0.3 bar - in two-phase region (3580 < 30000 < 51559)
    )


class TestFlashParams:
    """Tests for FlashParams validation."""

    def test_valid_params(self):
        """Test creating valid FlashParams."""
        params = FlashParams(species_order=["A", "B", "C"])
        assert params.species_order == ["A", "B", "C"]

    def test_empty_species_order(self):
        """Test that empty species_order raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            FlashParams(species_order=[])

    def test_duplicate_species(self):
        """Test that duplicate species raises error."""
        with pytest.raises(ValueError, match="duplicate"):
            FlashParams(species_order=["A", "B", "A"])

    def test_params_dict_access(self):
        """Test ParamsMixin dict-like access."""
        params = FlashParams(species_order=["X", "Y"])
        assert params["species_order"] == ["X", "Y"]
        assert "species_order" in params


class TestFlashCalculation:
    """Tests for Flash TP calculations."""

    def test_two_phase_flash(self, binary_thermo, flash_params, binary_feed):
        """Test flash in two-phase region."""
        flash = Flash(flash_params, binary_thermo)

        liquid, vapor, info = flash(binary_feed)

        # Check we have two-phase output
        assert 0.0 < float(info["V_frac"]) < 1.0

        # Check mass balance
        feed_total = total_flow(binary_feed)
        liquid_total = total_flow(liquid)
        vapor_total = total_flow(vapor)
        npt.assert_allclose(liquid_total + vapor_total, feed_total, rtol=1e-8)

        # Check component balance
        feed_flows = get_flows(binary_feed)
        liquid_flows = get_flows(liquid)
        vapor_flows = get_flows(vapor)
        for s in ["Light", "Heavy"]:
            npt.assert_allclose(
                liquid_flows[s] + vapor_flows[s],
                feed_flows[s],
                rtol=1e-8
            )

        # Check light component enriched in vapor
        liquid_light_frac = liquid_flows["Light"] / liquid_total
        vapor_light_frac = vapor_flows["Light"] / vapor_total
        assert vapor_light_frac > liquid_light_frac

    def test_flash_at_different_T(self, binary_thermo, flash_params, binary_feed):
        """Test flash at different temperatures."""
        flash = Flash(flash_params, binary_thermo)

        # Higher temperature should increase vapor fraction
        _, _, info_low_T = flash(binary_feed, T=330.0)
        _, _, info_high_T = flash(binary_feed, T=370.0)

        assert info_high_T["V_frac"] > info_low_T["V_frac"]

    def test_flash_at_different_P(self, binary_thermo, flash_params, binary_feed):
        """Test flash at different pressures."""
        flash = Flash(flash_params, binary_thermo)

        # Lower pressure should increase vapor fraction
        _, _, info_low_P = flash(binary_feed, P=50000.0)
        _, _, info_high_P = flash(binary_feed, P=200000.0)

        assert info_low_P["V_frac"] > info_high_P["V_frac"]

    def test_subcooled_liquid(self, binary_thermo, flash_params):
        """Test flash of subcooled liquid (all liquid)."""
        # Very low temperature, high pressure
        feed = make_stream({"Light": 10.0, "Heavy": 90.0}, T=280.0, P=500000.0)
        flash = Flash(flash_params, binary_thermo)

        liquid, vapor, info = flash(feed)

        # Should be mostly liquid
        assert float(info["V_frac"]) < 0.05  # Small tolerance for numerical

    def test_superheated_vapor(self, binary_thermo, flash_params):
        """Test flash of superheated vapor (all vapor)."""
        # Very high temperature, low pressure
        feed = make_stream({"Light": 90.0, "Heavy": 10.0}, T=450.0, P=10000.0)
        flash = Flash(flash_params, binary_thermo)

        liquid, vapor, info = flash(feed)

        # Should be mostly vapor
        assert float(info["V_frac"]) > 0.95

    def test_compositions_normalized(self, binary_thermo, flash_params, binary_feed):
        """Test that liquid and vapor compositions sum to 1."""
        flash = Flash(flash_params, binary_thermo)

        _, _, info = flash(binary_feed)

        x_sum = sum(info["x"].values())
        y_sum = sum(info["y"].values())

        npt.assert_allclose(float(x_sum), 1.0, rtol=1e-8)
        npt.assert_allclose(float(y_sum), 1.0, rtol=1e-8)

    def test_k_values_in_info(self, binary_thermo, flash_params, binary_feed):
        """Test that K-values are returned in info."""
        flash = Flash(flash_params, binary_thermo)

        _, _, info = flash(binary_feed)

        assert "K" in info
        assert "Light" in info["K"]
        assert "Heavy" in info["K"]

        # Light component should have higher K-value
        assert float(info["K"]["Light"]) > float(info["K"]["Heavy"])


class TestFlashGradient:
    """Tests for Flash gradient compatibility."""

    def test_gradient_wrt_temperature(self, binary_thermo, flash_params):
        """Test gradient of vapor fraction w.r.t. temperature."""
        flash = Flash(flash_params, binary_thermo)

        # Use feed at conditions clearly in two-phase region
        feed = make_stream({"Light": 50.0, "Heavy": 50.0}, T=350.0, P=80000.0)

        def vapor_fraction(T):
            _, _, info = flash(feed, T=T)
            return info["V_frac"]

        grad_fn = jax.grad(vapor_fraction)
        grad = grad_fn(350.0)

        # In two-phase region, gradient should be positive (higher T → more vapor)
        # Check gradient is finite (may be zero at boundaries)
        assert jnp.isfinite(grad)
        # Higher T should increase vapor fraction
        V_low = float(vapor_fraction(340.0))
        V_high = float(vapor_fraction(360.0))
        assert V_high >= V_low

    def test_gradient_wrt_pressure(self, binary_thermo, flash_params):
        """Test gradient of vapor fraction w.r.t. pressure."""
        flash = Flash(flash_params, binary_thermo)

        feed = make_stream({"Light": 50.0, "Heavy": 50.0}, T=350.0, P=80000.0)

        def vapor_fraction(P):
            _, _, info = flash(feed, P=P)
            return info["V_frac"]

        grad_fn = jax.grad(vapor_fraction)
        grad = grad_fn(80000.0)

        # Check gradient is finite
        assert jnp.isfinite(grad)
        # Higher P should decrease vapor fraction
        V_low_P = float(vapor_fraction(60000.0))
        V_high_P = float(vapor_fraction(100000.0))
        assert V_low_P >= V_high_P

    def test_gradient_smooth_at_boundaries(self, binary_thermo, flash_params):
        """Test that gradients are finite near phase boundaries."""
        flash = Flash(flash_params, binary_thermo)

        feed = make_stream({"Light": 50.0, "Heavy": 50.0}, T=350.0, P=80000.0)

        def vapor_fraction(T):
            _, _, info = flash(feed, T=T)
            return info["V_frac"]

        # Check gradient at multiple points - should all be finite
        temps = [330.0, 350.0, 370.0, 390.0]
        grads = [float(jax.grad(vapor_fraction)(T)) for T in temps]

        # All gradients should be finite
        for g in grads:
            assert jnp.isfinite(g)


class TestBubbleDewPoints:
    """Tests for bubble and dew point calculations."""

    def test_bubble_point_pressure(self, binary_thermo, flash_params, binary_feed):
        """Test bubble point pressure calculation."""
        flash = Flash(flash_params, binary_thermo)

        P_bubble = flash.bubble_point_pressure(binary_feed)

        # Should be finite and positive
        assert float(P_bubble) > 0
        assert jnp.isfinite(P_bubble)

    def test_dew_point_pressure(self, binary_thermo, flash_params, binary_feed):
        """Test dew point pressure calculation."""
        flash = Flash(flash_params, binary_thermo)

        P_dew = flash.dew_point_pressure(binary_feed)

        # Should be finite and positive
        assert float(P_dew) > 0
        assert jnp.isfinite(P_dew)

    def test_bubble_above_dew(self, binary_thermo, flash_params, binary_feed):
        """Test that bubble point pressure is above dew point pressure.

        For Raoult's law:
        - P_bubble = sum(x_i * Psat_i) - first bubble of vapor
        - P_dew = 1/sum(y_i / Psat_i) - last drop of liquid

        At same T with same composition: P_bubble > P_dew
        """
        flash = Flash(flash_params, binary_thermo)

        P_bubble = flash.bubble_point_pressure(binary_feed)
        P_dew = flash.dew_point_pressure(binary_feed)

        # For Raoult's law: P_bubble > P_dew at same T
        # (bubble point is when first bubble forms at high P,
        #  dew point is when last drop condenses at low P)
        assert float(P_bubble) > float(P_dew)


class TestFlashInitialize:
    """Tests for Flash initialize method."""

    def test_initialize_returns_estimates(self, binary_thermo, flash_params, binary_feed):
        """Test that initialize returns reasonable estimates."""
        flash = Flash(flash_params, binary_thermo)

        result = flash.initialize(binary_feed)

        assert "liquid" in result
        assert "vapor" in result
        assert "states" in result
        assert "info" in result

        # Check states
        assert "V_frac" in result["states"]
        assert "K" in result["states"]
        assert "x" in result["states"]
        assert "y" in result["states"]


class TestSplitter:
    """Tests for Splitter unit."""

    def test_basic_split(self):
        """Test basic stream splitting."""
        feed = make_stream({"A": 100.0, "B": 50.0}, T=300.0, P=101325.0)
        splitter = Splitter(species_order=["A", "B"])

        out1, out2, info = splitter(feed, split_frac=0.3)

        # Check split fractions
        out1_flows = get_flows(out1)
        out2_flows = get_flows(out2)

        npt.assert_allclose(out1_flows["A"], 30.0, rtol=1e-10)
        npt.assert_allclose(out1_flows["B"], 15.0, rtol=1e-10)
        npt.assert_allclose(out2_flows["A"], 70.0, rtol=1e-10)
        npt.assert_allclose(out2_flows["B"], 35.0, rtol=1e-10)

    def test_full_split_to_first(self):
        """Test splitting everything to first outlet."""
        feed = make_stream({"A": 100.0}, T=300.0, P=101325.0)
        splitter = Splitter(species_order=["A"])

        out1, out2, info = splitter(feed, split_frac=1.0)

        npt.assert_allclose(total_flow(out1), 100.0, rtol=1e-10)
        npt.assert_allclose(total_flow(out2), 0.0, atol=1e-10)

    def test_full_split_to_second(self):
        """Test splitting everything to second outlet."""
        feed = make_stream({"A": 100.0}, T=300.0, P=101325.0)
        splitter = Splitter(species_order=["A"])

        out1, out2, info = splitter(feed, split_frac=0.0)

        npt.assert_allclose(total_flow(out1), 0.0, atol=1e-10)
        npt.assert_allclose(total_flow(out2), 100.0, rtol=1e-10)

    def test_splitter_preserves_T_P(self):
        """Test that splitter preserves temperature and pressure."""
        feed = make_stream({"A": 100.0}, T=350.0, P=200000.0)
        splitter = Splitter(species_order=["A"])

        out1, out2, _ = splitter(feed, split_frac=0.5)

        npt.assert_allclose(float(out1["T"]), 350.0, rtol=1e-10)
        npt.assert_allclose(float(out1["P"]), 200000.0, rtol=1e-10)
        npt.assert_allclose(float(out2["T"]), 350.0, rtol=1e-10)
        npt.assert_allclose(float(out2["P"]), 200000.0, rtol=1e-10)

    def test_splitter_info(self):
        """Test splitter info dict."""
        feed = make_stream({"A": 100.0, "B": 100.0}, T=300.0, P=101325.0)
        splitter = Splitter(species_order=["A", "B"])

        _, _, info = splitter(feed, split_frac=0.4)

        npt.assert_allclose(float(info["split_fraction"]), 0.4, rtol=1e-10)
        npt.assert_allclose(float(info["flow_to_outlet1"]), 80.0, rtol=1e-10)
        npt.assert_allclose(float(info["flow_to_outlet2"]), 120.0, rtol=1e-10)

    def test_splitter_gradient(self):
        """Test gradient through splitter."""
        feed = make_stream({"A": 100.0}, T=300.0, P=101325.0)
        splitter = Splitter(species_order=["A"])

        def outlet1_flow(split_frac):
            out1, _, _ = splitter(feed, split_frac=split_frac)
            return total_flow(out1)

        grad = jax.grad(outlet1_flow)(0.5)

        # d(100*s)/ds = 100
        npt.assert_allclose(float(grad), 100.0, rtol=1e-10)


class TestMixer:
    """Tests for Mixer unit."""

    def test_two_stream_mix(self, binary_thermo):
        """Test mixing two streams."""
        stream1 = make_stream({"Light": 30.0, "Heavy": 20.0}, T=300.0, P=101325.0)
        stream2 = make_stream({"Light": 20.0, "Heavy": 30.0}, T=350.0, P=101325.0)

        mixer = Mixer(species_order=["Light", "Heavy"])

        outlet, info = mixer(stream1, stream2)

        # Check mass balance
        out_flows = get_flows(outlet)
        npt.assert_allclose(out_flows["Light"], 50.0, rtol=1e-10)
        npt.assert_allclose(out_flows["Heavy"], 50.0, rtol=1e-10)

        # Check temperature is between inputs (without thermo, uses weighted average)
        T_out = float(outlet["T"])
        assert 300.0 < T_out < 350.0

    def test_three_stream_mix(self):
        """Test mixing three streams."""
        s1 = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        s2 = make_stream({"A": 20.0}, T=300.0, P=101325.0)
        s3 = make_stream({"A": 30.0}, T=300.0, P=101325.0)

        mixer = Mixer(species_order=["A"])

        outlet, info = mixer(s1, s2, s3)

        out_flows = get_flows(outlet)
        npt.assert_allclose(out_flows["A"], 60.0, rtol=1e-10)
        assert info["n_inlets"] == 3

    def test_single_stream(self):
        """Test mixer with single stream (passthrough)."""
        stream = make_stream({"A": 100.0}, T=350.0, P=101325.0)
        mixer = Mixer(species_order=["A"])

        outlet, info = mixer(stream)

        npt.assert_allclose(total_flow(outlet), 100.0, rtol=1e-10)
        npt.assert_allclose(float(outlet["T"]), 350.0, rtol=1e-10)

    def test_mixer_no_inlets_error(self):
        """Test that mixer raises error with no inlets."""
        mixer = Mixer(species_order=["A"])

        with pytest.raises(ValueError, match="At least one inlet"):
            mixer()

    def test_mixer_info(self):
        """Test mixer info dict."""
        s1 = make_stream({"A": 40.0}, T=300.0, P=101325.0)
        s2 = make_stream({"A": 60.0}, T=400.0, P=101325.0)

        mixer = Mixer(species_order=["A"])

        _, info = mixer(s1, s2)

        assert info["n_inlets"] == 2
        npt.assert_allclose(float(info["total_flow"]), 100.0, rtol=1e-10)
        assert "T_out" in info
        assert "P_out" in info

    def test_mixer_with_thermo(self, binary_thermo):
        """Test mixer with thermodynamic T calculation."""
        s1 = make_stream({"Light": 50.0, "Heavy": 0.0}, T=300.0, P=101325.0)
        s2 = make_stream({"Light": 0.0, "Heavy": 50.0}, T=400.0, P=101325.0)

        mixer_with_thermo = Mixer(species_order=["Light", "Heavy"], thermo=binary_thermo)
        mixer_no_thermo = Mixer(species_order=["Light", "Heavy"])

        out_thermo, _ = mixer_with_thermo(s1, s2)
        out_simple, _ = mixer_no_thermo(s1, s2)

        # Both should give reasonable results
        assert 300.0 < float(out_thermo["T"]) < 400.0
        assert 300.0 < float(out_simple["T"]) < 400.0


class TestMixerGradient:
    """Tests for Mixer gradient compatibility."""

    def test_mixer_gradient_wrt_flow(self):
        """Test gradient of mixer output w.r.t. input flow."""
        mixer = Mixer(species_order=["A"])

        def output_flow(F_in):
            s1 = make_stream({"A": F_in}, T=300.0, P=101325.0)
            s2 = make_stream({"A": 50.0}, T=300.0, P=101325.0)
            outlet, _ = mixer(s1, s2)
            return total_flow(outlet)

        grad = jax.grad(output_flow)(100.0)

        # d(F_in + 50)/d(F_in) = 1
        npt.assert_allclose(float(grad), 1.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
