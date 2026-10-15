"""Smoke tests for the vacuum package's correlations, thermo and characterization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.vacuum import correlations as corr
from difflow_refinery.vacuum.assay import (
    atmospheric_residue,
    characterize,
    heavy_crude,
    product_properties,
    tbp_fraction,
)
from difflow_refinery.vacuum.thermo import ColumnThermo, steam_enthalpy

jax.config.update("jax_enable_x64", True)


def test_correlations_are_finite():
    t = corr.twu_critical_properties(560.0, 0.777)
    om = corr.lee_kesler_acentric(560.0, t["Tc"], t["Pc"], corr.watson_k(560.0, 0.777))
    p = corr.maxwell_bonnell_psat(480.0, 560.0, 11.8)
    for v in (t["Tc"], t["Pc"], t["MW"], om, p):
        assert np.isfinite(float(v))


@pytest.fixture(scope="module")
def char():
    return characterize(heavy_crude())


def test_characterize_and_thermo(char):
    assert np.all(np.isfinite(np.asarray(char.components.Tb)))
    th = ColumnThermo(char.components)
    assert np.all(np.isfinite(np.asarray(th.dh_vap(600.0))))
    assert np.isfinite(float(steam_enthalpy(573.15)))


def test_atmospheric_residue_properties(char):
    feed = atmospheric_residue(char, 100.0, cut_point_C=370.0)
    c = char.components
    m = jnp.stack([feed[f"F_{n}"] for n in c.names]) * c.MW / 1000.0
    props = product_properties(c, m)
    assert np.isfinite(float(props["rate"]))


def test_tbp_gradient_is_finite():
    a = heavy_crude()
    g = jax.grad(lambda t: tbp_fraction(a.with_tbp_point(9, t), 500.0))(500.0)
    assert np.isfinite(float(g))
