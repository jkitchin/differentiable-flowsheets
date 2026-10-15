"""Smoke tests for the Wagner vapor-pressure sets."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import IdealThermo
from difflow.database import get_species_data
from difflow.thermo import AntoineRangeWarning

jax.config.update("jax_enable_x64", True)


def test_wagner_psat_smoke_and_grad():
    th = IdealThermo({"methane": get_species_data("methane", vapor_pressure="wagner")})
    assert float(th.Psat("methane", 150.0)) > 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(lambda t: jnp.log(th.Psat("methane", t)))(150.0)
    assert jnp.isfinite(g)


def test_lookup_modes():
    assert get_species_data("methane").wagner_coeffs is None
    with pytest.raises(KeyError, match="Wagner"):
        get_species_data("water", vapor_pressure="wagner")
    with pytest.raises(ValueError):
        get_species_data("methane", vapor_pressure="raoult")


def test_range_warning():
    th = IdealThermo({"methane": get_species_data("methane", vapor_pressure="wagner")})
    with pytest.warns(AntoineRangeWarning, match="Wagner"):
        th.Psat("methane", 250.0)
