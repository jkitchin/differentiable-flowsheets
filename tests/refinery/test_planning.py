"""Smoke tests for the crude unit as a delta-base planning block (#297)."""

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow.planning import Network
from difflow_refinery import Assay
from difflow_refinery import column as cc
from difflow_refinery.planning import available_levers, cdu_block, link_cdu, product_value_block

jax.config.update("jax_enable_x64", True)

KW = dict(rate=10_000.0, T=480.0, P=4e5, jit=False)


@pytest.fixture(scope="module")
def unit():
    params = cc.CrudeColumnParams(
        n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
        specs=(cc.product_rate("naphtha", 0.004), cc.overflash(0.05)))
    return dr.CrudeUnit(Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85), params)


def test_lever_units(unit):
    lv = available_levers(unit)
    assert lv["preheat.T"].to_si(240.0, 1.0) == pytest.approx(513.15)


def test_cdu_block_and_network(unit):
    block = cdu_block(unit, ["crude.rate", "naphtha.yield"], **KW)
    u0 = np.asarray(block.u0)
    assert np.all(np.isfinite(np.asarray(block.fn(u0))))
    value = product_value_block({"naphtha": 70.0, "residue": 55.0}, jit=False)
    Network([block, value], link_cdu(block, value))


def test_bad_lever_raises(unit):
    with pytest.raises(ValueError, match="no lever"):
        cdu_block(unit, ["furnace.cot"], **KW)
