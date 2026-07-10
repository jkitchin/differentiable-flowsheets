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
