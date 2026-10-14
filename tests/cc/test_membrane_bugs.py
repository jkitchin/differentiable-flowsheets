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
        """Stage 2's retentate comes back through stage 1, so it leaves in
        the cascade retentate: more than stage 1 retentates from the feed.

        Uses a smaller second stage so the recycle is real. With one shared
        500 m2 area the exact flux model has stage 2 permeate essentially
        all of its small feed, the recycle is ~1e-9 mol/s, and the old strict
        comparison against stage 1 alone came down to the last bit (it
        passed on Python 3.11 and failed on 3.12).
        """
        stage_1 = MembraneParams(membrane_type="Matrimid", area=2000.0,
                                 pressure_ratio=10.0, feed_pressure=1000000.0)
        stage_2 = MembraneParams(membrane_type="Matrimid", area=200.0,
                                 pressure_ratio=10.0, feed_pressure=1000000.0)
        cascade = MultistageMembrane(stage_1, n_stages=2,
                                     configuration="permeate_recycle",
                                     stage_params=[stage_1, stage_2])
        feed = _flue_gas_feed()

        retentate, permeate, info = cascade(feed)
        ret_alone, _, _ = MembraneSeparator(stage_1)(feed)

        recycle = float(info["recycle_flow"])
        assert recycle > 0.1
        assert float(info["recycle_residual"]) < 1e-8
        # the recycle leaves through the retentate: the cascade retentate
        # exceeds stage 1 alone by a clear margin, not a rounding difference
        F_combined = float(total_flow(retentate))
        F_stage1_only = float(total_flow(ret_alone))
        assert F_combined - F_stage1_only > 0.1 * recycle, (
            f"Combined retentate ({F_combined}) should exceed stage 1 "
            f"retentate ({F_stage1_only})"
        )


class TestBug143PerfectMixingConsistency:
    """Bug #143: outlets must satisfy the complete-mixing equations.

    The earlier form of these tests pinned the permeate CO2 fraction to the
    feed-composition selectivity formula and the N2:O2 permeate ratio to the
    feed ratio. Both are what the carbon-capture audit (C5) found wrong:
    they ignore depletion of the retentate and the species' permeances, and
    produced reversed CO2 driving forces. The tests now check the
    solution-diffusion balance itself for every species.
    """

    @staticmethod
    def _check_flux_balance(membrane, feed, params):
        retentate, permeate, info = membrane(feed)
        ret, perm = get_flows(retentate), get_flows(permeate)
        R = float(total_flow(retentate))
        V = float(total_flow(permeate))
        P_h, P_l = float(retentate["P"]), float(permeate["P"])
        for sp in get_flows(feed):
            Q = float(info["permeances"][sp])
            x = float(ret[sp]) / R
            y = float(perm[sp]) / V
            flux = params.area * Q * (x * P_h - y * P_l)
            assert x * P_h - y * P_l >= 0.0
            assert float(perm[sp]) == pytest.approx(flux, rel=1e-8), sp
        return info

    def test_co2_purity_matches_perfect_mixing(self):
        """Per-species flux equals A Q_i (x_i P_feed - y_i P_permeate)."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        info = self._check_flux_balance(MembraneSeparator(params), _flue_gas_feed(), params)
        # Purity is below the zero-cut selectivity limit, which assumes an
        # undepleted retentate.
        from difflow_cc.database import get_membrane
        alpha = get_membrane("Matrimid").selectivity["CO2_N2"]
        assert float(info["CO2_purity"]) < alpha * 0.1 / (1.0 + (alpha - 1.0) * 0.1)

    def test_non_co2_species_scale_correctly(self):
        """O2 permeates faster than N2 relative to feed (higher permeance)."""
        params = MembraneParams(
            membrane_type="Matrimid",
            area=200.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)
        feed = make_stream(
            flows={"CO2": 1.0, "N2": 7.0, "O2": 2.0},
            T=298.15,
            P=1000000.0,
        )
        info = self._check_flux_balance(membrane, feed, params)
        _, permeate, _ = membrane(feed)
        perm = get_flows(permeate)
        if float(info["permeances"]["O2"]) > float(info["permeances"]["N2"]):
            assert float(perm["N2"]) / float(perm["O2"]) < 7.0 / 2.0


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
        """With a very large area the stage approaches total permeation.

        The old ad-hoc 99 %-of-feed cap is gone (audit C5): with complete
        mixing and enough area every species permeates, so the limit is the
        feed itself and the retentate never goes negative.
        """
        params = MembraneParams(
            membrane_type="Matrimid",
            area=50000.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)
        feed = _flue_gas_feed()

        retentate, permeate, info = membrane(feed)

        feed_flows = get_flows(feed)
        perm_flows = get_flows(permeate)
        ret_flows = get_flows(retentate)
        for species in feed_flows:
            f_feed = float(feed_flows[species])
            f_perm = float(perm_flows.get(species, 0.0))
            assert 0.0 <= f_perm <= f_feed
            assert float(ret_flows[species]) >= 0.0
            assert f_perm + float(ret_flows[species]) == pytest.approx(f_feed)
        assert float(info["stage_cut"]) > 0.99

        # Total flow should still be consistent
        species_sum = sum(float(v) for v in perm_flows.values())
        reported_total = float(info["permeate_flow"])
        assert species_sum == pytest.approx(reported_total, rel=1e-6)
