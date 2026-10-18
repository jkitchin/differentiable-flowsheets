"""Smoke tests for difflow.gas_properties."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.gas_properties import (
    GasRangeWarning,
    gas_density,
    gas_heat_capacity,
    gas_prandtl,
    gas_property_range,
    gas_thermal_conductivity,
    gas_transport_properties,
    gas_viscosity,
    list_gas_species,
    sutherland_conductivity,
    sutherland_viscosity,
)


def test_properties_finite():
    for fn in (gas_viscosity, gas_thermal_conductivity, gas_heat_capacity,
               gas_prandtl, gas_density):
        assert np.isfinite(float(fn("air", 300.0)))
    assert np.isfinite(float(sutherland_viscosity(300.0, "air")))
    assert np.isfinite(float(sutherland_conductivity(300.0, "nitrogen")))
    p = gas_transport_properties("air", 300.0)
    assert {"nu", "Pr", "alpha", "k", "rho", "cp"} <= set(p)
    assert gas_viscosity("air", jnp.linspace(300.0, 600.0, 4)).shape == (4,)
    assert np.isfinite(float(jax.grad(lambda t: gas_viscosity("methane", t))(400.0)))


def test_aliases_and_errors():
    assert float(gas_viscosity("steam", 500.0)) == float(gas_viscosity("water", 500.0))
    assert "air" in list_gas_species()
    with pytest.raises(KeyError, match="sutherland"):
        gas_viscosity("unobtainium", 300.0)


def test_range_warning_concrete_only():
    lo, hi = gas_property_range("methane", "mu")
    with pytest.warns(GasRangeWarning):
        gas_viscosity("methane", hi + 80.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.grad(lambda T: gas_viscosity("methane", T))(hi + 80.0)


def test_natural_gas_components():
    # #495: ethane/propane/butanes/H2S have fits and critical constants, so a
    # natural-gas mixture at pressure runs (it used to raise KeyError).
    from difflow.gas_properties import dense_gas_viscosity, gas_mixture_transport
    for name in ("C2H6", "propane", "n-butane", "isobutane", "H2S"):
        assert np.isfinite(float(gas_prandtl(name, 300.0)))
    names = ["methane", "ethane", "propane"]
    assert np.isfinite(float(dense_gas_viscosity(names, 288.15, 50e5, y=[0.90, 0.06, 0.04])))
    out = gas_mixture_transport(names + ["n_butane", "isobutane", "hydrogen_sulfide"],
                                [0.85, 0.06, 0.04, 0.02, 0.02, 0.01], 288.15, 50e5)
    assert all(np.isfinite(float(v)) for v in out.values())
    g = jax.grad(lambda T: dense_gas_viscosity(names, T, 50e5, y=[0.90, 0.06, 0.04]))(288.15)
    assert np.isfinite(float(g))
