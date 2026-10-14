"""Smoke tests for the activity-coefficient (gamma-phi) flash path (issue #90)."""

import jax.numpy as jnp
import pytest

from difflow.streams import get_flows, make_stream
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.flash import Flash, FlashParams
from difflow.units.lle import NRTLParams, nrtl_activity_coefficients

THERMO = IdealThermo({
    "ethanol": SpeciesData(name="ethanol", MW=46.07, Cp_coeffs=(0, 0, 0, 0),
                           Hvap_coeffs=(1.0, 0.38, 514.0),
                           antoine_coeffs=(10.24677, 1598.673, -46.424)),
    "water": SpeciesData(name="water", MW=18.02, Cp_coeffs=(0, 0, 0, 0),
                         Hvap_coeffs=(1.0, 0.38, 647.0),
                         antoine_coeffs=(9.6543, 1435.264, -64.848)),
})
NRTL = NRTLParams(species=("ethanol", "water"),
                  a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
                  b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
                  alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]))


def test_nrtl_activity_coefficients_are_finite():
    g = nrtl_activity_coefficients(jnp.array([0.3, 0.7]), 351.4, NRTL)
    assert g.shape == (2,) and bool(jnp.all(jnp.isfinite(g)))


def test_gamma_phi_flash_dilute_volatile_feed():
    """Rachford-Rice Newton once left [0, 1] here and reported V ~ 1e-4."""
    flash = Flash(FlashParams(species_order=["ethanol", "water"]), thermo=THERMO,
                  activity_model=NRTL)
    feed = make_stream({"ethanol": 0.05, "water": 0.95}, T=360.0, P=101325.0)
    liquid, vapor, info = flash(feed, T=367.5, P=101325.0)
    assert float(info["V_frac"]) > 0.1
    lf, vf = get_flows(liquid), get_flows(vapor)
    for s in ("ethanol", "water"):
        assert float(lf[s] + vf[s]) == pytest.approx(float(get_flows(feed)[s]), rel=1e-6)
