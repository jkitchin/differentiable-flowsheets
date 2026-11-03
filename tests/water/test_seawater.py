"""Seawater properties and seawater streams: they run, are finite, differentiate."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw
from difflow_water.chemistry import seawater as sw

PROPS = [sw.density, sw.viscosity, sw.osmotic_coefficient, sw.osmotic_pressure]


@pytest.mark.parametrize("fn", PROPS, ids=lambda f: f.__name__)
def test_property_is_finite(fn):
    T = jnp.array([280.0, 298.15, 350.0])
    S = jnp.array([0.01, 0.035, 0.1])
    assert jnp.all(jnp.isfinite(fn(T, S)))


@pytest.mark.parametrize("fn", PROPS, ids=lambda f: f.__name__)
def test_property_gradient_is_finite(fn):
    g = jax.grad(fn, argnums=(0, 1))(298.15, 0.035)
    assert all(jnp.isfinite(x) for x in g)


def test_pure_water_properties_are_finite():
    assert jnp.isfinite(sw.density_water(298.15))
    assert jnp.isfinite(sw.viscosity_water(298.15))
    assert jnp.isfinite(jax.grad(sw.viscosity_water)(298.15))


def test_salinity_from_mass_concentration_inverts_density():
    """API contract: S -> C = S rho(S) -> S."""
    S = jnp.array([0.0, 0.035, 0.12])
    C = S * sw.density(300.0, S)
    assert jnp.allclose(sw.salinity_from_mass_concentration(C, 300.0), S, atol=1e-12)


def test_validity_reports_each_range():
    v = sw.validity(298.15, 0.035)
    assert set(v) == set(sw.RANGES) and all(v.values())
    assert not sw.validity(298.15, 0.001)["osmotic_coefficient"]


def test_seawater_stream_keys_and_conversions():
    s = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035)
    assert set(s) == {"F_H2O", "F_TDS", "T", "P"}
    for f in (dw.mass_flow, dw.salinity, dw.volumetric_flow, dw.concentration_mg_per_L):
        assert jnp.isfinite(f(s))


def test_seawater_stream_needs_exactly_one_flow():
    with pytest.raises(ValueError):
        dw.make_seawater_stream()
    with pytest.raises(ValueError):
        dw.make_seawater_stream(flow_mass=1.0, flow_vol=1e-3)


def test_watertap_state_round_trip():
    s = dw.make_seawater_stream(flow_mass=1.0, salinity=0.035, T=300.0, P=2e5)
    st = dw.to_watertap_state(s)
    back = dw.from_watertap_state(st["flow_mass_phase_comp"][("Liq", "H2O")],
                                  st["flow_mass_phase_comp"][("Liq", "TDS")],
                                  st["temperature"], st["pressure"])
    for k in s:
        assert jnp.allclose(s[k], back[k])
