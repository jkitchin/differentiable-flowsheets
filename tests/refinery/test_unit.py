"""Smoke tests for product properties and the CrudeUnit assembly."""

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, characterize, products
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

jax.config.update("jax_enable_x64", True)

ASSAY = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)


@pytest.fixture(scope="module")
def small():
    params = cc.CrudeColumnParams(
        n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
        specs=(cc.product_rate("naphtha", 0.25, basis="mole"), cc.overflash(0.05)))
    return dr.CrudeUnit(ASSAY, params)


def test_product_properties_smoke():
    crude = characterize(ASSAY)
    th = ColumnThermo.from_characterization(crude)
    feed = crude.stream(10.0, T=300.0, P=1e5)
    props = products.product_properties({"crude": feed}, th, feed)
    assert np.isfinite(float(props["crude"].sg))


def test_crude_unit_solves(small):
    res = small.solve(1.0, T=480.0, P=4e5, basis="mole")
    assert bool(res.converged)
    assert sum(float(p.yield_mass) for p in res.properties.values()) == pytest.approx(1.0, rel=1e-8)
    assert "naphtha" in res.table()


def test_gradient_is_finite(small):
    g = jax.grad(lambda T: small.solve(1.0, T=T, P=4e5, basis="mole").column.furnace_fired_duty)(480.0)
    assert np.isfinite(float(g))


def test_bad_basis(small):
    with pytest.raises(ValueError, match="basis"):
        small.feed(1.0, T=480.0, P=4e5, basis="gallons")
