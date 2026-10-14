"""Smoke and API tests for difflow.liquid_properties."""

import warnings

import jax
import numpy as np
import pytest

from difflow import make_stream
from difflow.database import list_species, track_database_access
from difflow.liquid_properties import (
    LiquidRangeWarning,
    SurfaceTensionWarning,
    check_liquid_range,
    get_liquid_properties,
    liquid_density,
    liquid_property_source,
    liquid_thermal_conductivity,
    liquid_viscosity,
    list_liquid_species,
    mixture_liquid_density,
    mixture_liquid_thermal_conductivity,
    mixture_liquid_viscosity,
    mixture_surface_tension,
    stream_liquid_properties,
    surface_tension,
)

jax.config.update("jax_enable_x64", True)


def test_every_database_species_has_liquid_data():
    assert list_liquid_species() == list_species("any")


def test_aliases_and_unknown_species():
    assert get_liquid_properties("H2O").name == "water"
    with pytest.raises(KeyError, match="No liquid-property data"):
        get_liquid_properties("unobtainium")
    with pytest.raises(ValueError):
        liquid_property_source("water", "cp")


def test_lookups_are_tracked_for_reports():
    with track_database_access() as tracker:
        liquid_density("toluene", 300.0)
    assert tracker.kinds("toluene") == {"liquid"}


def test_pure_properties_finite():
    for f in (liquid_density, liquid_viscosity, liquid_thermal_conductivity, surface_tension):
        assert np.isfinite(float(f("toluene", 300.0)))


def test_mixture_and_stream():
    x = {"water": 0.7, "ethanol": 0.3}
    for f in (mixture_liquid_density, mixture_liquid_viscosity,
              mixture_liquid_thermal_conductivity):
        assert np.isfinite(float(f(x, 300.0)))
    assert np.isfinite(float(mixture_surface_tension({"benzene": 0.4, "toluene": 0.6}, 360.0)))
    s = make_stream({"water": 8.0, "methanol": 2.0}, T=300.0, P=1e5)
    assert set(stream_liquid_properties(s)) == {"rho", "rho_molar", "mu", "k", "Q"}


def test_range_warning_and_check():
    hi = get_liquid_properties("benzene").mu.Tmax
    with pytest.warns(LiquidRangeWarning, match="benzene liquid viscosity"):
        liquid_viscosity("benzene", hi + 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.grad(lambda T: liquid_viscosity("benzene", T))(hi + 20.0)
    assert check_liquid_range("water", 1000.0)["rho"]["in_range"] is False


def test_surface_tension_alcohol_warns():
    with pytest.warns(SurfaceTensionWarning):
        surface_tension("ethanol", 293.15)


def test_grad_finite():
    def f(x1, T):
        x = {"water": 1.0 - x1, "ethanol": x1}
        return mixture_liquid_density(x, T) + 1e6 * mixture_liquid_viscosity(x, T)
    g = jax.grad(f, argnums=(0, 1))(0.3, 320.0)
    assert all(np.isfinite(float(v)) for v in g)
