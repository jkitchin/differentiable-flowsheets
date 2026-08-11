"""Tests for the activity-coefficient (gamma-phi) flash path (issue #90).

Demonstrates that with an NRTL liquid model the flash reproduces
non-ideal VLE, including the ethanol-water azeotrope, which the ideal
Raoult's-law path cannot.

NRTL ethanol(1)-water(2) parameters (tau_ij = a_ij + b_ij/T, alpha = 0.3)
are the widely used Aspen/DECHEMA set (Gmehling, J., Onken, U.,
Vapor-Liquid Equilibrium Data Collection, DECHEMA, 1977). Antoine
coefficients are the NIST log10(P[bar]) form shifted by +5 to give Pa.
"""

import pytest
import jax
import jax.numpy as jnp

from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.flash import Flash, FlashParams
from difflow.units.lle import nrtl_activity_coefficients, NRTLParams
from difflow.streams import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


def _thermo():
    ethanol = SpeciesData(
        name="ethanol", MW=46.07, Cp_coeffs=(0, 0, 0, 0),
        Hvap_coeffs=(1.0, 0.38, 514.0), antoine_coeffs=(10.24677, 1598.673, -46.424),
    )
    water = SpeciesData(
        name="water", MW=18.02, Cp_coeffs=(0, 0, 0, 0),
        Hvap_coeffs=(1.0, 0.38, 647.0), antoine_coeffs=(9.6543, 1435.264, -64.848),
    )
    return IdealThermo({"ethanol": ethanol, "water": water})


def _nrtl():
    return NRTLParams(
        species=("ethanol", "water"),
        a=jnp.array([[0.0, 3.4578], [-0.8009, 0.0]]),
        b=jnp.array([[0.0, -586.0809], [246.18, 0.0]]),
        alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
    )


class TestNRTLRelativeVolatility:
    """The NRTL model produces an azeotrope; ideal Raoult does not."""

    def test_relative_volatility_crosses_unity(self):
        th, nrtl = _thermo(), _nrtl()
        T, P = 351.4, 101325.0
        K_ideal = jnp.array([th.Psat("ethanol", T), th.Psat("water", T)]) / P

        def alpha(x1):
            x = jnp.array([x1, 1 - x1])
            g = nrtl_activity_coefficients(x, T, nrtl)
            K = g * K_ideal
            return float(K[0] / K[1])

        # Ethanol-rich liquid: water salts out, alpha drops below 1 (azeotrope)
        assert alpha(0.2) > 1.0
        assert alpha(0.9) < 1.0
        # Ideal relative volatility is constant and never crosses 1
        ideal_alpha = float(K_ideal[0] / K_ideal[1])
        assert ideal_alpha > 1.0


@pytest.mark.slow
class TestGammaPhiFlash:
    """Flash with an NRTL activity model (#90)."""

    def _feed(self, x_etoh):
        return make_stream({"ethanol": x_etoh, "water": 1 - x_etoh}, T=351.4, P=101325.0)

    def test_flash_runs_and_conserves_mass(self):
        flash = Flash(FlashParams(species_order=["ethanol", "water"]),
                      thermo=_thermo(), activity_model=_nrtl())
        feed = self._feed(0.5)
        liquid, vapor, info = flash(feed, T=351.4, P=101325.0)
        lf, vf = get_flows(liquid), get_flows(vapor)
        for s in ("ethanol", "water"):
            total = float(lf.get(s, 0.0)) + float(vf.get(s, 0.0))
            assert total == pytest.approx(float(get_flows(feed)[s]), rel=1e-6)

    def test_azeotrope_pinch_vs_ideal(self):
        """Near the azeotrope the gamma flash barely separates; ideal does."""
        params = FlashParams(species_order=["ethanol", "water"])
        gamma_flash = Flash(params, thermo=_thermo(), activity_model=_nrtl())
        ideal_flash = Flash(params, thermo=_thermo())

        feed = self._feed(0.85)  # near the ethanol-water azeotrope
        _, _, ig = gamma_flash(feed, T=351.4, P=101325.0)
        _, _, ii = ideal_flash(feed, T=351.4, P=101325.0)

        # x and y are close for the azeotropic (gamma) system...
        gap_gamma = abs(float(ig["x"]["ethanol"]) - float(ig["y"]["ethanol"]))
        # ...but the ideal model shows a large ethanol enrichment in vapor.
        gap_ideal = abs(float(ii["x"]["ethanol"]) - float(ii["y"]["ethanol"]))
        assert gap_gamma < gap_ideal

    def test_gamma_changes_k_values(self):
        """The activity path yields different K-values than ideal Raoult."""
        params = FlashParams(species_order=["ethanol", "water"])
        _, _, ig = Flash(params, thermo=_thermo(), activity_model=_nrtl())(
            self._feed(0.3), T=351.4, P=101325.0)
        _, _, ii = Flash(params, thermo=_thermo())(
            self._feed(0.3), T=351.4, P=101325.0)
        assert float(ig["K"]["ethanol"]) != pytest.approx(float(ii["K"]["ethanol"]), rel=1e-3)
