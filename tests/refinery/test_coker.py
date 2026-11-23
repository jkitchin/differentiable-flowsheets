"""Delayed coker (#534): smoke, gradient, input validation, API contract.
Balances, the correlation trend, FD gradients and the Du et al. comparison
are checked by hand in tests/reference/coker_check.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.coking import (PRODUCTS, CokerFeed, CokerParams, CokerRangeWarning,
                                     DelayedCoker, coker_block)

jax.config.update("jax_enable_x64", True)

VR = CokerFeed.from_properties(8.0, sg=1.03, sulfur_wt=3.5, ccr_wt=22.0,
                               nitrogen_wppm=4000.0, nickel_vanadium_wppm=250.0)
SLURRY = CokerFeed.from_properties(2.0, sg=1.08, sulfur_wt=1.2, ccr_wt=6.0)


def test_smoke():
    o = DelayedCoker().solve([VR, SLURRY])
    for k in PRODUCTS:
        assert np.isfinite(float(o[f"yield.{k}"])) and float(o[f"yield.{k}"]) >= 0.0
    assert all(np.isfinite(float(v)) for v in o.values())


def test_grad_finite():
    unit = DelayedCoker()
    g = jax.grad(lambda c: unit.solve(CokerFeed.from_properties(
        1.0, sg=1.0, sulfur_wt=2.0, ccr_wt=c))["yield.naphtha"])(18.0)
    assert np.isfinite(float(g))
    g2 = jax.grad(lambda p: unit.solve(VR, drum_P=p)["yield.coke"])(2.0)
    assert np.isfinite(float(g2))


def test_blend_pools_sg_by_volume():
    f = CokerFeed.blend([VR, SLURRY])
    assert float(f.mass) == pytest.approx(10.0)
    assert float(f.SG) == pytest.approx(10.0 / (8.0 / 1.03 + 2.0 / 1.08))


def test_bad_feed_raises():
    with pytest.raises(ValueError, match="positive"):
        DelayedCoker().solve(CokerFeed.from_properties(0.0, sg=1.0, sulfur_wt=1.0, ccr_wt=10.0))
    with pytest.raises(ValueError, match="nonnegative"):
        DelayedCoker().solve(CokerFeed.from_properties(1.0, sg=1.0, sulfur_wt=-1.0, ccr_wt=10.0))


def test_empty_blend_raises():
    with pytest.raises(ValueError, match="at least one"):
        CokerFeed.blend([])


def test_bad_split_raises():
    with pytest.raises(ValueError, match="sum"):
        DelayedCoker(CokerParams(sulfur_split=(0.5, 0.5, 0.5, 0.0, 0.0))).solve(VR)


def test_out_of_range_ccr_warns():
    with pytest.warns(CokerRangeWarning):
        DelayedCoker().solve(CokerFeed.from_properties(1.0, sg=0.95, sulfur_wt=1.0, ccr_wt=2.0))


def test_coker_block():
    block = coker_block(DelayedCoker(), CokerFeed.blend([VR, SLURRY]),
                        ["feed.ccr", "drum_P", "recycle"], jit=False)
    y = np.asarray(block.fn(jnp.asarray(block.u0)))
    assert y.shape == (len(block.y_names),) and np.all(np.isfinite(y))
    with pytest.raises(ValueError, match="unknown coker lever"):
        coker_block(DelayedCoker(), VR, ["nope"], jit=False)
