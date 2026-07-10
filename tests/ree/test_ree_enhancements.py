"""Tests for REE plugin enhancement issues (acid balance, ionic strength,
filtrate pH, entrainment, co-precipitation, third-phase, kinetics)."""

import pytest
import jax
import jax.numpy as jnp

pytest.importorskip("difflow_ree")

from difflow.streams import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


class TestScrubberAcidConsumption:
    """Issue #115: scrubber should track H+ consumption and resulting pH."""

    def _run(self, pH=1.5):
        from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
        params = ScrubberParams(
            n_stages=5, extractant="D2EHPA",
            elements=("La", "Dy"), target_elements=("Dy",), pH=pH,
        )
        scrubber = REEScrubber(params)
        loaded_org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "La": 1.0, "Dy": 1.0},
            T=298.15, P=101325.0,
        )
        scrub_soln = make_stream({"H2O": 10.0}, T=298.15, P=101325.0)
        return scrubber(loaded_org, scrub_soln)

    def test_acid_consumed_is_three_per_ree(self):
        _, _, info = self._run()
        # 3 H+ per mole REE scrubbed
        assert float(info["acid_consumed"]) == pytest.approx(
            3.0 * float(info["ree_scrubbed_total"]), rel=1e-9
        )
        assert float(info["ree_scrubbed_total"]) > 0.0

    def test_acid_balance_and_ph_shift(self):
        _, _, info = self._run(pH=1.5)
        # Consuming H+ raises the pH of the scrub liquor
        assert float(info["h_plus_remaining"]) <= float(info["h_plus_supplied"])
        assert float(info["pH_final"]) >= 1.5

    def test_acid_consumption_differentiable(self):
        from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams

        def consumed(la_in):
            params = ScrubberParams(
                n_stages=5, extractant="D2EHPA",
                elements=("La", "Dy"), target_elements=("Dy",), pH=1.5,
            )
            org = make_stream(
                {"D2EHPA": 1.0, "kerosene": 5.0, "La": la_in, "Dy": 1.0},
                T=298.15, P=101325.0,
            )
            scrub = make_stream({"H2O": 10.0}, T=298.15, P=101325.0)
            _, _, info = REEScrubber(params)(org, scrub)
            return info["acid_consumed"]

        g = jax.grad(consumed)(1.0)
        assert jnp.isfinite(g) and float(g) > 0.0


class TestPrecipitationFiltratePH:
    """Issue #116: filtrate pH updated for H+/OH- released/consumed."""

    def test_hydroxide_ph_drops(self):
        from difflow_ree.units.precipitation import HydroxidePrecipitator, PrecipitatorParams
        params = PrecipitatorParams(elements=("Nd", "Dy"))
        precip = HydroxidePrecipitator(params)
        feed = make_stream({"H2O": 1.0, "Nd": 0.01, "Dy": 0.01}, T=298.15, P=101325.0)
        base = make_stream({"OH": 1.0}, T=298.15, P=101325.0)
        _, _, info = precip(feed, base, pH=9.0)
        # OH- consumed by precipitation -> filtrate pH below target 9.0
        assert float(info["OH_consumed"]) > 0.0
        assert float(info["pH_final"]) <= 9.0

    def test_oxalate_ph_drops_when_feed_ph_given(self):
        from difflow_ree.units.precipitation import OxalatePrecipitator, PrecipitatorParams
        params = PrecipitatorParams(elements=("Nd", "Dy"))
        precip = OxalatePrecipitator(params)
        feed = make_stream({"H2O": 1.0, "Nd": 0.05, "Dy": 0.05}, T=298.15, P=101325.0)
        oxalic = make_stream({"C2O4": 1.0}, T=298.15, P=101325.0)
        _, _, info = precip(feed, oxalic, feed_pH=3.0)
        # Releases H+ -> filtrate more acidic than feed
        assert float(info["h_plus_released"]) > 0.0
        assert float(info["pH_final"]) < 3.0

    def test_oxalate_backward_compat_no_feed_ph(self):
        from difflow_ree.units.precipitation import OxalatePrecipitator, PrecipitatorParams
        precip = OxalatePrecipitator(PrecipitatorParams(elements=("Nd",)))
        feed = make_stream({"H2O": 1.0, "Nd": 0.05}, T=298.15, P=101325.0)
        oxalic = make_stream({"C2O4": 1.0}, T=298.15, P=101325.0)
        _, _, info = precip(feed, oxalic)
        assert "pH_final" not in info


class TestCoprecipitation:
    """Issue #109: common-ion / co-precipitation boosts trace REE capture."""

    def _run(self, factor):
        from difflow_ree.units.precipitation import OxalatePrecipitator, PrecipitatorParams
        params = PrecipitatorParams(
            elements=("La", "Dy"), coprecipitation_factor=factor,
        )
        precip = OxalatePrecipitator(params)
        feed = make_stream({"H2O": 1.0, "La": 0.5, "Dy": 0.5}, T=298.15, P=101325.0)
        # Sub-stoichiometric oxalate so conversions are below 1
        oxalic = make_stream({"C2O4": 0.3}, T=298.15, P=101325.0)
        return precip(feed, oxalic)

    def test_backward_compat_factor_zero(self):
        _, solid0, info0 = self._run(0.0)
        assert float(info0["coprecipitation_factor"]) == 0.0
        assert float(info0["total_precipitated"]) > 0.0

    def test_coprecipitation_increases_recovery(self):
        _, _, info0 = self._run(0.0)
        _, _, info1 = self._run(1.0)
        # Common-ion coupling recovers more REE than independent precipitation
        assert float(info1["total_precipitated"]) > float(info0["total_precipitated"])

    def test_coprecipitation_differentiable(self):
        from difflow_ree.units.precipitation import OxalatePrecipitator, PrecipitatorParams

        def recovered(factor):
            params = PrecipitatorParams(elements=("La", "Dy"), coprecipitation_factor=factor)
            feed = make_stream({"H2O": 1.0, "La": 0.5, "Dy": 0.5}, T=298.15, P=101325.0)
            oxalic = make_stream({"C2O4": 0.3}, T=298.15, P=101325.0)
            _, _, info = OxalatePrecipitator(params)(feed, oxalic)
            return info["total_precipitated"]

        g = jax.grad(recovered)(0.5)
        assert jnp.isfinite(g) and float(g) > 0.0


class TestMixerSettlerKinetics:
    """Issue #118: kinetic stage efficiency from residence time."""

    def _stage(self, **kw):
        from difflow_ree.units.extraction import REEMixerSettler, MixerSettlerParams
        return REEMixerSettler(MixerSettlerParams(
            extractant="D2EHPA", elements=("Nd",), pH=3.0, **kw))

    def _streams(self):
        aq = make_stream({"H2O": 10.0, "Nd": 1.0}, T=298.15, P=101325.0)
        org = make_stream({"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0)
        return aq, org

    def test_slow_kinetics_undershoots_equilibrium(self):
        aq, org = self._streams()
        # Slow rate + short mixing -> efficiency well below 1
        _, _, info = self._stage(k_extraction=0.001, mixer_residence_time=60.0)(aq, org)
        assert float(info["kinetic_efficiency"]) < 0.1
        assert float(info["efficiency"]) == pytest.approx(1 - jnp.exp(-0.001 * 60.0), rel=1e-6)

    def test_fast_kinetics_reaches_equilibrium(self):
        aq, org = self._streams()
        _, _, fast = self._stage(k_extraction=1.0, mixer_residence_time=120.0)(aq, org)
        assert float(fast["efficiency"]) > 0.99

    def test_backward_compat_murphree(self):
        aq, org = self._streams()
        _, _, info = self._stage(stage_efficiency=0.9)(aq, org)  # no k_extraction
        assert float(info["efficiency"]) == pytest.approx(0.9, rel=1e-6)
        assert "kinetic_efficiency" not in info


class TestMixerSettlerEntrainment:
    """Issue #110: phase entrainment carries REE across, degrading separation."""

    def _run(self, f_oa=0.0, f_ao=0.0):
        from difflow_ree.units.extraction import REEMixerSettler, MixerSettlerParams
        stage = REEMixerSettler(MixerSettlerParams(
            extractant="D2EHPA", elements=("Nd",), pH=3.0,
            entrainment_org_in_aq=f_oa, entrainment_aq_in_org=f_ao,
        ))
        aq = make_stream({"H2O": 10.0, "Nd": 1.0}, T=298.15, P=101325.0)
        org = make_stream({"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0)
        return stage(aq, org)

    def test_entrainment_conserves_mass(self):
        aq_out, org_out, _ = self._run(f_oa=0.02, f_ao=0.01)
        nd = float(get_flows(aq_out).get("Nd", 0.0)) + float(get_flows(org_out).get("Nd", 0.0))
        assert nd == pytest.approx(1.0, rel=1e-6)

    def test_entrainment_moves_ree_back_to_aqueous(self):
        aq0, org0, _ = self._run(0.0, 0.0)
        aqE, orgE, _ = self._run(f_oa=0.05, f_ao=0.0)
        # Organic-in-aqueous entrainment carries extracted Nd back to raffinate
        assert float(get_flows(aqE)["Nd"]) > float(get_flows(aq0)["Nd"])


class TestThirdPhaseFormation:
    """Issue #117: high organic loading flags third-phase onset."""

    def _run(self, extractant_flow, limit=0.5):
        from difflow_ree.units.extraction import REEMixerSettler, MixerSettlerParams
        stage = REEMixerSettler(MixerSettlerParams(
            extractant="D2EHPA", elements=("Nd", "Dy"), pH=3.5,
            third_phase_loading_limit=limit,
        ))
        aq = make_stream({"H2O": 10.0, "Nd": 1.0, "Dy": 1.0}, T=298.15, P=101325.0)
        org = make_stream({"D2EHPA": extractant_flow, "kerosene": 5.0}, T=298.15, P=101325.0)
        return stage(aq, org)

    def test_low_loading_no_third_phase(self):
        _, _, info = self._run(extractant_flow=100.0)  # lots of extractant -> low loading
        assert not bool(info["third_phase_formed"])

    def test_high_loading_flags_third_phase(self):
        _, _, info = self._run(extractant_flow=0.5)  # little extractant -> high loading
        assert bool(info["third_phase_formed"])
        assert float(info["organic_loading"]) > 0.5


class TestDistributionIonicStrength:
    """Issue #111: Davies ionic-strength correction of D."""

    def _dist(self):
        from difflow_ree.equilibrium.distribution import REEDistribution
        return REEDistribution(extractant="D2EHPA", elements=("Nd", "Dy"))

    def test_backward_compat_no_correction(self):
        d = self._dist()
        D0 = float(d.get_D("Nd", pH=3.0))
        D_none = float(d.get_D("Nd", pH=3.0, ionic_strength=None))
        assert D0 == pytest.approx(D_none, rel=1e-9)

    def test_higher_ionic_strength_lowers_D(self):
        d = self._dist()
        D_lo = float(d.get_D("Nd", pH=3.0, ionic_strength=0.01))
        D_hi = float(d.get_D("Nd", pH=3.0, ionic_strength=0.4))
        D_ideal = float(d.get_D("Nd", pH=3.0))
        # Davies gamma < 1 for I>0, decreasing with I
        assert D_lo < D_ideal
        assert D_hi < D_lo

    def test_get_D_all_threads_ionic_strength(self):
        d = self._dist()
        D_ideal = d.get_D_all(pH=3.0)
        D_corr = d.get_D_all(pH=3.0, ionic_strength=0.3)
        for e in ("Nd", "Dy"):
            assert float(D_corr[e]) < float(D_ideal[e])

    def test_differentiable_through_ionic_strength(self):
        d = self._dist()
        g = jax.grad(lambda I: d.get_D("Nd", pH=3.0, ionic_strength=I))(0.1)
        assert jnp.isfinite(g) and float(g) < 0.0


class TestDatabaseThermoProperties:
    """Issue #119: DB can carry optional heats / degradation rate."""

    def test_defaults_are_none(self):
        from difflow_ree.database import get_ree_database, get_extractant_database
        nd = get_ree_database().get("Nd")
        assert nd.heat_of_extraction is None
        assert nd.heat_of_scrubbing is None
        assert nd.heat_of_stripping is None
        d2ehpa = get_extractant_database().get("D2EHPA")
        assert d2ehpa.degradation_rate is None
        assert d2ehpa.heat_of_extraction is None

    def test_element_can_carry_heats(self):
        from difflow_ree.database import REEElement
        # User-supplied values (with citation in real use)
        el = REEElement(
            symbol="Ho", name="Holmium", atomic_number=67, atomic_weight=164.93,
            ionic_radius_pm=90.1, density=8.79, melting_point=1734.0,
            oxidation_states=(3,), group="heavy", oxide_formula="Ho2O3",
            oxide_mw=377.86, price_usd_kg=1400.0,
            heat_of_extraction=-25.0, heat_of_stripping=25.0,
        )
        assert el.heat_of_extraction == -25.0
        assert el.heat_of_stripping == 25.0
        assert el.heat_of_scrubbing is None
