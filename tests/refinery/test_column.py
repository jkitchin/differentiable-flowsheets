"""Smoke tests for the equation-oriented crude column."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

jax.config.update("jax_enable_x64", True)

ASSAY = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)


@pytest.fixture(scope="module")
def crude():
    return characterize(ASSAY)


@pytest.fixture(scope="module")
def thermo(crude):
    return ColumnThermo.from_characterization(crude)


def _flows(stream, thermo):
    return jnp.stack([jnp.asarray(stream[f"F_{n}"]) for n in thermo.names])


def _small(thermo, specs, **over):
    kw = dict(n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
              P_condenser=1.2e5, specs=specs)
    kw.update(over)
    return cc.CrudeColumn(cc.CrudeColumnParams(**kw), thermo)


def test_small_column_solves_and_balances(crude, thermo):
    col = _small(thermo, (cc.product_rate("naphtha", 0.25, basis="mole"),))
    feed = crude.stream(1.0, T=620.0, P=1.6e5, basis="mole")
    res = col.solve(feed)
    assert bool(res.converged)
    out = sum(np.asarray(_flows(s, thermo)) for s in res.products.values())
    np.testing.assert_allclose(out, np.asarray(_flows(feed, thermo)), rtol=1e-8)


def test_spec_count_must_match_freedoms(thermo):
    with pytest.raises(ValueError, match="degrees of freedom"):
        _small(thermo, ())


def test_unknown_target(thermo):
    with pytest.raises(ValueError, match="no product"):
        _small(thermo, (cc.product_rate("jet", 0.1),))


def test_bad_layout(thermo):
    with pytest.raises(ValueError, match="feed_stage"):
        _small(thermo, (cc.reflux_ratio(1.0),), feed_stage=9)
