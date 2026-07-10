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
