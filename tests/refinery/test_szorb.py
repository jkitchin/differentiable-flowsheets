"""S Zorb desulfurizer (#535): smoke, gradient, input validation. Balances,
trends, FD gradients and the Du et al. Upd4 comparison are by hand in
tests/reference/szorb_check.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.desulfurization import SZorbFeed, SZorbParams, SZorbRangeWarning, SZorbUnit

jax.config.update("jax_enable_x64", True)


def _feed(s=500.0, **kw):
    return SZorbFeed.from_properties(50.0, sg=0.73, sulfur_wppm=s, olefins_vol=30.0,
                                     aromatics_vol=22.0, ron=92.0, mon=80.0, **kw)


def test_smoke():
    o = SZorbUnit().solve(_feed())
    assert all(np.isfinite(float(v)) for v in o.values())
    assert 0.0 < float(o["sorbent.activity"]) <= 1.0


def test_grad_finite():
    u = SZorbUnit()
    g = jax.grad(lambda s: u.solve(_feed(s))["product.S_wppm"])(500.0)
    gT = jax.grad(lambda T: u.solve(_feed(), T=T)["product.RON"])(693.15)
    assert np.isfinite(float(g)) and np.isfinite(float(gT))


def test_bad_split_raises():
    with pytest.raises(ValueError, match="sulfur_split"):
        SZorbUnit().solve(_feed(sulfur_split=(0.5, 0.5, 0.5, 0.0, 0.0)))


def test_bad_params_raise():
    with pytest.raises(ValueError, match="entries"):
        SZorbUnit(SZorbParams(k_s=(1.0, 2.0))).solve(_feed())


def test_bad_feed_raises():
    with pytest.raises(ValueError, match="positive"):
        SZorbUnit().solve(SZorbFeed.from_properties(0.0, sg=0.73, sulfur_wppm=500.0,
                                                    olefins_vol=30.0, aromatics_vol=22.0,
                                                    ron=92.0, mon=80.0))


def test_saturated_sorbent_warns():
    with pytest.warns(SZorbRangeWarning):
        SZorbUnit(SZorbParams(circulation=0.001)).solve(_feed(1500.0))
