"""Smoke tests for the pure-component saturation T/P helpers."""

import jax
import pytest

from difflow import IdealThermo, PengRobinson, saturation_pressure, saturation_temperature
from difflow.database import get_critical_props, get_species_data
from difflow.thermo import AntoineRangeWarning

jax.config.update("jax_enable_x64", True)


def test_ideal_smoke_and_grad():
    th = IdealThermo({"benzene": get_species_data("benzene")})
    T = saturation_temperature("benzene", 5e4, thermo=th)
    assert float(saturation_pressure("benzene", T, thermo=th)) == pytest.approx(5e4, rel=1e-8)
    g = jax.grad(lambda P: saturation_temperature("benzene", P, thermo=th))(5e4)
    assert jax.numpy.isfinite(g)


def test_ideal_range_warning():
    with pytest.warns(AntoineRangeWarning):
        saturation_temperature("water", 1e7)


def test_bad_model_and_species():
    with pytest.raises(ValueError):
        saturation_temperature("water", 1e5, model="raoult")
    th = IdealThermo({"water": get_species_data("water")})
    with pytest.raises(KeyError):
        saturation_temperature("benzene", 1e5, thermo=th)
    eos = PengRobinson({"water": get_critical_props("water")})
    with pytest.raises(KeyError):
        saturation_pressure("benzene", 350.0, model="eos", eos=eos)
