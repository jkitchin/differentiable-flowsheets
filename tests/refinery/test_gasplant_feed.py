"""Smoke tests for gas_plant_feed and evolved_h2s (#326)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow_refinery import Assay, characterize
from difflow_refinery.gasplant import evolved_h2s, gas_plant_feed

PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
LE = {"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
      "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5}
LIGHT = ["hydrogen_sulfide", "ethane", "propane", "isobutane", "n_butane",
         "isopentane", "n_pentane"]


@pytest.fixture(scope="module")
def char():
    a = Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LE, sulfur_wt=1.8)
    return characterize(a)


def _products(char):
    def stream(light, cut_total, water):
        s = {f"F_{n}": jnp.asarray(light.get(n, 0.0)) for n in char.light_names}
        for i, n in enumerate(char.pseudo_names):
            s[f"F_{n}"] = jnp.asarray(cut_total * np.exp(-0.9 * i))
        s["F_water"] = jnp.asarray(water)
        s["T"], s["P"] = jnp.asarray(313.0), jnp.asarray(1.3e5)
        return s

    return {"offgas": stream(dict(ethane=0.9, propane=3.0, n_butane=3.5), 0.05, 2.0),
            "naphtha": stream(dict(propane=4.0, n_butane=14.0, n_pentane=22.0), 120.0, 0.5)}


def test_gas_plant_feed_smoke(char):
    feed = gas_plant_feed(_products(char), char, LIGHT, h2s={"offgas": 0.02}, T=313.15, P=1.3e5)
    for n in ("offgas", "naphtha"):
        assert np.all(np.isfinite(np.asarray(feed.flows(feed[n]))))
    assert "folded into" in feed.summary()


def test_gradient_is_finite(char):
    P = _products(char)
    cuts = gas_plant_feed(P, char, LIGHT).cuts

    def f(scale):
        Q = {n: {k: (v * scale if k.startswith("F_") else v) for k, v in s.items()}
             for n, s in P.items()}
        feed = gas_plant_feed(Q, char, LIGHT, cuts=cuts)
        return feed.mass(feed["naphtha"])

    assert np.isfinite(float(jax.grad(f)(1.0)))


def test_unknown_light_end_raises(char):
    with pytest.raises(ValueError, match="neither in light"):
        gas_plant_feed(_products(char), char, LIGHT[:-1])


def test_selection_under_jit_needs_cuts(char):
    P = _products(char)
    with pytest.raises(TypeError, match="cuts="):
        jax.jit(lambda s: gas_plant_feed(
            {n: {k: v * s for k, v in st.items()} for n, st in P.items()},
            char, LIGHT).basis_total)(1.0)


def test_evolved_h2s(char):
    P = _products(char)
    assert np.isfinite(float(evolved_h2s(P, char, 0.01)))
    no_s = characterize(Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LE))
    with pytest.raises(ValueError, match="sulfur"):
        evolved_h2s(P, no_s, 0.01)
