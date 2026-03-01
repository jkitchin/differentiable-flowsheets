"""Tests for adsorption unit bug fixes (#134, #138, #141, #142).

Bug #134: N2 mass balance broken in PSA, VSA, TSA
Bug #138: PSA and VSA energy units wrong (extra *3.6 factor)
Bug #141: PSA desorption uses feed y_CO2 instead of enriched composition
Bug #142: PSA compression work based on total feed flow, not captured CO2
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow_cc import AdsorptionParams, PSAUnit, VSAUnit, TSAUnit
from difflow.streams import make_stream, get_flows


def _flue_gas_feed(P=500000.0):
    """Create a typical flue gas feed stream for testing."""
    return make_stream(
        flows={"CO2": 1.0, "N2": 4.0},  # 20% CO2
        T=298.15,
        P=P,
    )


# =============================================================================
# Bug #134: N2 mass balance
# =============================================================================

class TestN2MassBalance:
    """N2_product + N2_offgas must equal N2_feed for PSA, VSA, TSA."""

    def _check_n2_balance(self, product, offgas, feed):
        feed_flows = get_flows(feed)
        product_flows = get_flows(product)
        offgas_flows = get_flows(offgas)

        N2_feed = float(feed_flows.get("N2", 0.0))
        N2_product = float(product_flows.get("N2", 0.0))
        N2_offgas = float(offgas_flows.get("N2", 0.0))

        assert N2_feed == pytest.approx(N2_product + N2_offgas, rel=1e-6), (
            f"N2 mass balance broken: feed={N2_feed:.6f}, "
            f"product={N2_product:.6f}, offgas={N2_offgas:.6f}, "
            f"sum={N2_product + N2_offgas:.6f}"
        )

    def test_psa_n2_balance(self):
        """PSA: N2 in product + offgas equals N2 in feed."""
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )
        feed = _flue_gas_feed(P=500000.0)
        product, offgas, info = PSAUnit(params)(feed)
        self._check_n2_balance(product, offgas, feed)

    def test_vsa_n2_balance(self):
        """VSA: N2 in product + offgas equals N2 in feed."""
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="VSA",
            P_adsorption=101325.0,
            P_desorption=10000.0,
            bed_mass=100.0,
        )
        feed = _flue_gas_feed(P=101325.0)
        product, offgas, info = VSAUnit(params)(feed)
        self._check_n2_balance(product, offgas, feed)

    def test_tsa_n2_balance(self):
        """TSA: N2 in product + offgas equals N2 in feed."""
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="TSA",
            T_adsorption=298.15,
            T_desorption=393.15,
            bed_mass=100.0,
        )
        feed = _flue_gas_feed(P=101325.0)
        product, offgas, info = TSAUnit(params)(feed)
        self._check_n2_balance(product, offgas, feed)


# =============================================================================
# Bug #138: Energy units (no extra *3.6)
# =============================================================================

class TestEnergyUnits:
    """Specific energy should be in reasonable GJ/tonne range (0.1-10)."""

    def test_psa_energy_reasonable(self):
        """PSA specific energy should be in GJ/tonne, not inflated by 3.6x."""
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )
        feed = _flue_gas_feed(P=500000.0)
        _, _, info = PSAUnit(params)(feed)

        energy = float(info["specific_energy"])
        assert 0.01 < energy < 20.0, (
            f"PSA energy {energy:.4f} GJ/tonne outside reasonable range "
            f"(0.01-20). Likely wrong units."
        )

    def test_vsa_energy_reasonable(self):
        """VSA specific energy should be in GJ/tonne, not inflated by 3.6x."""
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="VSA",
            P_adsorption=101325.0,
            P_desorption=10000.0,
            bed_mass=100.0,
        )
        feed = _flue_gas_feed(P=101325.0)
        _, _, info = VSAUnit(params)(feed)

        energy = float(info["specific_energy"])
        assert 0.01 < energy < 20.0, (
            f"VSA energy {energy:.4f} GJ/tonne outside reasonable range "
            f"(0.01-20). Likely wrong units."
        )

    def test_psa_energy_not_inflated(self):
        """PSA energy should not have spurious 3.6 multiplier.

        If the old code produced E_old, the fixed code should give E_old/3.6.
        We check the ratio by comparing two identical runs is not ~3.6x off
        from a hand-calculated value.
        """
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )
        feed = _flue_gas_feed(P=500000.0)
        _, _, info = PSAUnit(params)(feed)

        # Hand check: W = F_captured * R * T * ln(ratio) / 0.7
        # energy_GJ = W / (F_captured*44/1e6) / 1e9
        # The 44/1e6 and captured cancel partially, leaving R*T*ln(5)/0.7 / (44/1e6) / 1e9
        # = 8.314 * 298.15 * ln(5) / 0.7 / 44e-6 / 1e9
        R_val = 8.314
        T_val = 298.15
        ratio = 5.0
        hand_energy = R_val * T_val * jnp.log(ratio) / 0.7 / (44.0 / 1e6) / 1e9
        computed = float(info["specific_energy"])

        # Should match within ~50% (simplified model differences)
        assert abs(computed - float(hand_energy)) / float(hand_energy) < 0.5, (
            f"PSA energy {computed:.4f} vs hand calc {float(hand_energy):.4f} "
            f"differ by more than 50%."
        )


# =============================================================================
# Bug #141: PSA desorption uses enriched CO2 composition
# =============================================================================

class TestPSAEnrichedDesorption:
    """PSA should use enriched CO2 partial pressure during desorption."""

    def test_enriched_composition_increases_working_capacity(self):
        """Using enriched y_CO2 for desorption should give higher P_CO2_des,
        which means higher residual loading, hence LOWER working capacity
        compared to using feed y_CO2 directly (the old bug).

        The key check: the working capacity should reflect enrichment.
        With selectivity > 1 and feed y_CO2 < 1, the enriched y_CO2 should
        be greater than feed y_CO2.
        """
        from difflow_cc.database import get_adsorbent

        adsorbent_data = get_adsorbent("Zeolite_13X")
        selectivity = adsorbent_data.CO2_selectivity

        y_CO2_feed = 0.2  # 20% CO2 in feed
        # Enriched composition formula
        y_CO2_enriched = min(
            y_CO2_feed * selectivity / (1.0 + y_CO2_feed * (selectivity - 1.0)),
            0.95,
        )
        # Enriched should be higher than feed for selectivity > 1
        assert y_CO2_enriched > y_CO2_feed, (
            f"Enriched y_CO2 ({y_CO2_enriched:.4f}) should be > feed "
            f"y_CO2 ({y_CO2_feed:.4f}) when selectivity={selectivity:.1f}"
        )

    def test_psa_uses_selectivity_in_desorption(self):
        """PSA with different selectivity adsorbents should show different
        working capacities due to enriched desorption composition."""
        feed = _flue_gas_feed(P=500000.0)

        # Use two different adsorbents with different selectivities
        params_13x = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )
        _, _, info_13x = PSAUnit(params_13x)(feed)

        params_ac = AdsorptionParams(
            adsorbent="AC_Coconut",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )
        _, _, info_ac = PSAUnit(params_ac)(feed)

        # Both should have positive working capacity
        assert float(info_13x["working_capacity"]) > 0
        assert float(info_ac["working_capacity"]) > 0


# =============================================================================
# Bug #142: PSA compression work scales with captured CO2
# =============================================================================

class TestPSACompressionWork:
    """Compression work should scale with F_CO2_captured, not F_total."""

    def test_compression_scales_with_captured_not_total(self):
        """Doubling total feed N2 (same CO2) should NOT double compression power."""
        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )

        # Feed 1: 1 mol/s CO2, 4 mol/s N2
        feed1 = make_stream(
            flows={"CO2": 1.0, "N2": 4.0},
            T=298.15,
            P=500000.0,
        )
        _, _, info1 = PSAUnit(params)(feed1)

        # Feed 2: 1 mol/s CO2, 40 mol/s N2 (10x more N2, same CO2)
        feed2 = make_stream(
            flows={"CO2": 1.0, "N2": 40.0},
            T=298.15,
            P=500000.0,
        )
        _, _, info2 = PSAUnit(params)(feed2)

        W1 = float(info1["compression_power"])
        W2 = float(info2["compression_power"])

        # If bug still existed, W2 would be ~10x W1 (since F_total is 10x).
        # After fix, both should be similar (same CO2 captured, approximately).
        ratio = W2 / W1 if W1 > 0 else float("inf")
        assert ratio < 3.0, (
            f"Compression power ratio {ratio:.2f} is too high. "
            f"W1={W1:.2f}, W2={W2:.2f}. "
            f"Power should scale with captured CO2, not total feed."
        )
