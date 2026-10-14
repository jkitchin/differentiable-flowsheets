"""Smoke tests for the column thermodynamics."""

import jax
import numpy as np
import pytest

from difflow_refinery import Assay, ColumnThermo, characterize, water_vapor_pressure

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def thermo():
    assay = Assay([0, 10, 50, 90, 100], [280.0, 350.0, 550.0, 800.0, 950.0], sg=0.86,
                  light_ends={"propane": 0.5, "n_butane": 1.0})
    return ColumnThermo.from_characterization(characterize(assay))


def test_water_vapor_pressure_is_finite():
    assert np.isfinite(float(water_vapor_pressure(373.0)))


def test_k_values_finite_and_jittable(thermo):
    K = np.asarray(thermo.K(500.0, 1.5e5))
    assert K.shape == (thermo.n_components,)
    assert np.all(np.isfinite(K))
    f = jax.jit(lambda th, T: th.h_liquid(T))
    assert np.all(np.isfinite(np.asarray(f(thermo, 400.0))))


def test_unknown_light_end_raises():
    crude = characterize(Assay([0, 100], [300.0, 900.0], sg=0.85, light_ends={"water": 1.0}))
    with pytest.raises(ValueError, match="light ends"):
        ColumnThermo.from_characterization(crude)
