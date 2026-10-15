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
