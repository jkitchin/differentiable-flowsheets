"""Tests for membrane bug fixes (#139, #143, #144, #150).

Bug #139: Permeate recycle mode discards stage 2 retentate
Bug #143: Hybrid membrane model inconsistency (flux ratios vs perfect-mixing)
Bug #144: Perfect-mixing equation doesn't account for pressure ratio limitation
Bug #150: Per-species 99% cap doesn't recalculate total flows
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream, get_flows, total_flow
from difflow.numerics import safe_divide
from difflow_cc import MembraneParams
from difflow_cc.units.membrane import MembraneSeparator, MultistageMembrane


def _flue_gas_feed(P=1000000.0):
    """Create a typical flue gas feed stream."""
    return make_stream(
        flows={"CO2": 1.0, "N2": 9.0},
        T=298.15,
        P=P,
    )


class TestBug139PermeateRecycleMassBalance:
    """Bug #139: Permeate recycle mode should combine both retentates."""

    def test_mass_balance_permeate_recycle(self):
        """Total feed moles must equal total retentate + permeate moles."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        cascade = MultistageMembrane(
            params, n_stages=2, configuration="permeate_recycle"
        )
        feed = _flue_gas_feed()

        retentate, permeate, info = cascade(feed)

        F_feed = float(total_flow(feed))
        F_ret = float(total_flow(retentate))
        F_perm = float(total_flow(permeate))

        # Mass balance: feed = retentate + permeate
        assert F_ret + F_perm == pytest.approx(F_feed, rel=1e-6), (
            f"Mass balance violated: feed={F_feed}, ret={F_ret}, perm={F_perm}"
        )

    def test_species_balance_permeate_recycle(self):
        """Per-species mass balance must hold."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        cascade = MultistageMembrane(
            params, n_stages=2, configuration="permeate_recycle"
        )
        feed = _flue_gas_feed()

        retentate, permeate, info = cascade(feed)

        feed_flows = get_flows(feed)
        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)

        for species in feed_flows:
            f_in = float(feed_flows[species])
            f_ret = float(ret_flows.get(species, 0.0))
            f_perm = float(perm_flows.get(species, 0.0))
            assert f_ret + f_perm == pytest.approx(f_in, rel=1e-6), (
                f"Species {species}: in={f_in}, ret={f_ret}, perm={f_perm}"
            )

    def test_retentate_includes_stage2_retentate(self):
        """Retentate should be larger than stage 1 retentate alone."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        cascade = MultistageMembrane(
            params, n_stages=2, configuration="permeate_recycle"
        )
        feed = _flue_gas_feed()

        retentate, permeate, info = cascade(feed)

        # Stage 1 alone
        stage1 = MembraneSeparator(params)
        ret_1, perm_1, _ = stage1(feed)

        # The combined retentate should be greater than stage 1 retentate alone
        # because it includes stage 2 retentate
        F_combined = float(total_flow(retentate))
        F_stage1_only = float(total_flow(ret_1))
        assert F_combined > F_stage1_only, (
            f"Combined retentate ({F_combined}) should exceed stage 1 retentate ({F_stage1_only})"
        )


class TestBug143PerfectMixingConsistency:
    """Bug #143: Permeate composition should match perfect-mixing model."""

    def test_co2_purity_matches_perfect_mixing(self):
        """CO2 purity in permeate should be consistent with perfect-mixing equation."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)
        feed = _flue_gas_feed()

        retentate, permeate, info = membrane(feed)

        # Compute expected CO2 purity from perfect-mixing model
        from difflow_cc.database import get_membrane
        mem_data = get_membrane("Matrimid")
        alpha = mem_data.selectivity.get("CO2_N2", 1.0)
        y_CO2_feed = 1.0 / 10.0  # 10% CO2

        y_CO2_perm_expected = alpha * y_CO2_feed / (1.0 + (alpha - 1.0) * y_CO2_feed)

        # Apply pressure ratio limit (bug #144 fix)
        P_feed = 1000000.0
        P_perm = P_feed / 10.0
        pressure_ratio = P_feed / P_perm
        y_CO2_perm_expected = min(y_CO2_perm_expected, y_CO2_feed * pressure_ratio)
        y_CO2_perm_expected = min(max(y_CO2_perm_expected, 0.0), 0.999)

        # Actual CO2 purity
        actual_purity = float(info["CO2_purity"])

        # Should be close to perfect-mixing prediction
        # (not exact due to per-species capping, but close)
        assert actual_purity == pytest.approx(y_CO2_perm_expected, rel=0.05), (
            f"CO2 purity {actual_purity} doesn't match perfect-mixing {y_CO2_perm_expected}"
        )

    def test_non_co2_species_scale_correctly(self):
        """Non-CO2 species in permeate should be proportional to feed fractions."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=200.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)

        # Feed with multiple non-CO2 species
        feed = make_stream(
            flows={"CO2": 1.0, "N2": 7.0, "O2": 2.0},
            T=298.15,
            P=1000000.0,
        )

        retentate, permeate, info = membrane(feed)

        perm_flows = get_flows(permeate)
        # N2 and O2 should be in proportion to their feed mole fractions
        # (both are non-CO2, so they share the remaining permeate fraction)
        F_N2_perm = float(perm_flows.get("N2", 0.0))
        F_O2_perm = float(perm_flows.get("O2", 0.0))

        # Feed ratio N2:O2 = 7:2 = 3.5
        if F_O2_perm > 0:
            ratio = F_N2_perm / F_O2_perm
            assert ratio == pytest.approx(7.0 / 2.0, rel=0.1), (
                f"N2/O2 permeate ratio {ratio} doesn't match feed ratio 3.5"
            )


class TestBug144PressureRatioLimit:
    """Bug #144: Perfect-mixing equation should account for pressure ratio."""

    def test_low_pressure_ratio_limits_enrichment(self):
        """At low pressure ratio, permeate CO2 fraction should be limited."""
        # Low pressure ratio = 2 should severely limit enrichment
        params_low = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=2.0,
            feed_pressure=200000.0,
        )
        membrane_low = MembraneSeparator(params_low)

        # High pressure ratio = 20 for comparison
        params_high = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=20.0,
            feed_pressure=2000000.0,
        )
        membrane_high = MembraneSeparator(params_high)

        feed_low = make_stream(
            flows={"CO2": 1.0, "N2": 9.0}, T=298.15, P=200000.0,
        )
        feed_high = make_stream(
            flows={"CO2": 1.0, "N2": 9.0}, T=298.15, P=2000000.0,
        )

        _, _, info_low = membrane_low(feed_low)
        _, _, info_high = membrane_high(feed_high)

        purity_low = float(info_low["CO2_purity"])
        purity_high = float(info_high["CO2_purity"])

        # With pressure ratio of 2, max enrichment factor is 2
        # So CO2 purity should be at most ~0.2 (= 0.1 * 2)
        # With high pressure ratio, selectivity is the limit
        assert purity_low <= 0.25, (
            f"Low pressure ratio purity {purity_low} exceeds physical limit"
        )
        assert purity_high > purity_low, (
            f"High pressure ratio purity {purity_high} should exceed low {purity_low}"
        )

    def test_pressure_ratio_caps_co2_perm_fraction(self):
        """y_CO2_perm should not exceed y_CO2_feed * pressure_ratio."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=3.0,
            feed_pressure=300000.0,
        )
        membrane = MembraneSeparator(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0}, T=298.15, P=300000.0,
        )

        _, permeate, info = membrane(feed)

        perm_flows = get_flows(permeate)
        F_CO2_perm = float(perm_flows.get("CO2", 0.0))
        F_perm_total = float(total_flow(permeate))

        if F_perm_total > 0:
            y_CO2_actual = F_CO2_perm / F_perm_total
            # Maximum possible is y_CO2_feed * pressure_ratio = 0.1 * 3 = 0.3
            assert y_CO2_actual <= 0.30 + 0.01, (
                f"CO2 perm fraction {y_CO2_actual} exceeds pressure ratio limit 0.30"
            )


class TestBug150TotalFlowConsistency:
    """Bug #150: Total permeate flow should match sum of species flows after capping."""

    def test_permeate_species_sum_matches_total(self):
        """Sum of permeate species flows should equal total permeate flow."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)
        feed = _flue_gas_feed()

        retentate, permeate, info = membrane(feed)

        perm_flows = get_flows(permeate)
        species_sum = sum(float(v) for v in perm_flows.values())
        reported_total = float(info["permeate_flow"])

        assert species_sum == pytest.approx(reported_total, rel=1e-6), (
            f"Species sum {species_sum} != reported total {reported_total}"
        )

    def test_stage_cut_consistent_with_actual_flows(self):
        """Stage cut should equal actual permeate / feed."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)
        feed = _flue_gas_feed()

        retentate, permeate, info = membrane(feed)

        F_feed = float(total_flow(feed))
        F_perm = float(total_flow(permeate))
        expected_cut = F_perm / F_feed
        actual_cut = float(info["stage_cut"])

        assert actual_cut == pytest.approx(expected_cut, rel=1e-4), (
            f"Stage cut {actual_cut} != actual ratio {expected_cut}"
        )

    def test_large_area_with_capping(self):
        """With very large membrane area, per-species capping should activate."""
        # Use very large area to force high permeation and trigger capping
        params = MembraneParams(
            membrane_type="Matrimid",
            area=50000.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)
        feed = _flue_gas_feed()

        retentate, permeate, info = membrane(feed)

        # Even with large area, no species should exceed 99% of feed
        feed_flows = get_flows(feed)
        perm_flows = get_flows(permeate)
        for species in feed_flows:
            f_feed = float(feed_flows[species])
            f_perm = float(perm_flows.get(species, 0.0))
            assert f_perm <= f_feed * 0.99 + 1e-10, (
                f"{species}: perm={f_perm} exceeds 99% of feed={f_feed}"
            )

        # Total flow should still be consistent
        species_sum = sum(float(v) for v in perm_flows.values())
        reported_total = float(info["permeate_flow"])
        assert species_sum == pytest.approx(reported_total, rel=1e-6)
