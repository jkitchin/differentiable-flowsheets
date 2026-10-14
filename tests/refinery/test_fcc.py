"""FCC unit: smoke, validation and API tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.fcc import FCCFeed, FCCParams, FCCUnit, fcc_block, get_scheme
from difflow_refinery.vacuum import assay as va

jax.config.update("jax_enable_x64", True)

VGO = (343.0 + 273.15, 550.0 + 273.15)
RATE = 50.0


@pytest.fixture(scope="module")
def light_feed():
    char = dr.characterize(va.light_crude().to_assay(heavy_end=dr.HeavyEnd()))
    return FCCFeed.from_characterization(char, RATE, *VGO)


@pytest.fixture(scope="module")
def unit():
    return FCCUnit(FCCParams())


def test_feed_smoke(light_feed):
    assert float(light_feed.mass) == pytest.approx(RATE)
    assert all(np.isfinite(float(v)) for v in light_feed.as_dict().values())


def test_scheme_rates_finite_and_conserve_mass():
    s = get_scheme("ancheyta_5")
    th = s.theta()
    dy = s.rates(jnp.linspace(0.6, 0.05, s.n), 800.0, 0.7, th["k_ref"], th["Ea"])
    assert np.all(np.isfinite(np.asarray(dy))) and abs(float(jnp.sum(dy))) < 1e-15


def test_unknown_scheme_raises():
    with pytest.raises(NotImplementedError):
        get_scheme("jacob_10")


def test_unit_solves(unit, light_feed):
    res = unit.solve(light_feed)
    assert bool(res["converged"])
    assert all(np.isfinite(float(v)) for v in res["outputs"].values())


def test_unit_grad_finite(unit, light_feed):
    g = jax.grad(lambda rot: unit.solve(light_feed, riser_outlet_T=rot)["outputs"]["conversion"])
    assert np.isfinite(float(g(jnp.asarray(800.0))))


def test_unknown_lever(unit, light_feed):
    with pytest.raises(ValueError):
        fcc_block(unit, light_feed, levers=["no_such_knob"])
