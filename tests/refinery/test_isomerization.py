"""Isomerization unit: smoke, validation and API tests."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.planning import Network
from difflow_refinery.blending import BlendComponent, BlendPool
from difflow_refinery.isomerization import (
    OUTPUT_NAMES,
    IsomerizationHydrogenWarning,
    IsomerizationReactor,
    IsomerizationReactorParams,
    IsomerizationUnit,
    IsomerizationUnitParams,
    constructed_feed,
    isom_block,
    link_isom,
    nc6_fraction,
)
from difflow_refinery.isomerization import thermochem as tc
from difflow_refinery.isomerization.feed import hydrocarbon_flows
from difflow_refinery.isomerization.plant import MIN_H2_HC_OUT
from difflow_refinery.isomerization.reactor import flows_of

jax.config.update("jax_enable_x64", True)

T_BASE = 413.15


def test_unknown_species_is_refused():
    with pytest.raises(ValueError, match="not isomerization species"):
        flows_of({"F_toluene": 1.0, "T": 400.0, "P": 1e5})


def test_unknown_params_are_refused():
    with pytest.raises(ValueError, match="configuration"):
        IsomerizationUnitParams(configuration="tip")
    with pytest.raises(ValueError, match="catalyst"):
        IsomerizationReactorParams(catalyst="platinum_on_wishes")


def test_reactor_smoke():
    F = hydrocarbon_flows(constructed_feed("paraffinic", 10.0))
    F = F.at[tc.idx("hydrogen")].set(0.3 * jnp.sum(F))
    out = IsomerizationReactor(IsomerizationReactorParams()).run(F, T_BASE, P=30e5)
    assert out is not None


def test_hydrogen_warning_nan_counts_as_starved():
    with pytest.warns(IsomerizationHydrogenWarning, match="hydrogen-starved"):
        IsomerizationUnit._check_hydrogen({"reactor": {"H2_HC_out": 0.5 * MIN_H2_HC_OUT}})
    with pytest.warns(IsomerizationHydrogenWarning):
        IsomerizationUnit._check_hydrogen({"reactor": {"H2_HC_out": float("nan")}})


@pytest.fixture(scope="module")
def once_through():
    unit = IsomerizationUnit(IsomerizationUnitParams())
    feed = constructed_feed("paraffinic", 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IsomerizationHydrogenWarning)
        iso, gas, info = unit(feed)
    return unit, feed, iso, gas, info


def test_unit_smoke(once_through):
    out = once_through[-1]["outputs"]
    assert set(out) == set(OUTPUT_NAMES)
    for k, v in out.items():
        assert np.isfinite(float(v)), k


def test_unmodelled_feed_species_is_refused(once_through):
    unit, feed = once_through[:2]
    with pytest.raises(ValueError, match="not isomerization species"):
        unit(dict(feed, F_water=2.0))


def test_unit_jacfwd_finite(once_through):
    unit, feed = once_through[:2]
    g = jax.jacfwd(lambda T: unit.outputs(feed, T, 2.0, x_nc6=float(nc6_fraction(feed)))[
        OUTPUT_NAMES.index("RON")])(T_BASE)  # forward mode: the unit has a while_loop
    assert np.isfinite(float(g))


def test_block_and_pool_link(once_through):
    unit, feed, _, _, info = once_through
    blk = isom_block(unit, feed, levers=("T_in", "LHSV", "x_nC6"),
                     outputs=("isomerate_V", "RON", "dih_duty"))
    assert blk.u_names == ["T_in", "LHSV", "x_nC6"]
    reformate = BlendComponent.from_properties(
        "reformate", SG=0.82, RON=98.0, MON=88.0, RVP_psi=4.0, olefins_vol=0.0,
        aromatics_vol=60.0, benzene_vol=1.0)
    pool = BlendPool("gasoline").as_block(
        [unit.blend_component(info["outputs"]), reformate],
        u0=[float(info["outputs"]["isomerate_volume"]), 50.0])
    blk = isom_block(unit, feed)
    Network([blk, pool], link_isom(blk, pool))
    with pytest.raises(ValueError, match="isomerate_V"):
        link_isom(blk, BlendPool("gasoline").as_block([reformate]))


def test_bad_levers_are_refused(once_through):
    unit, feed, *_ = once_through
    with pytest.raises(ValueError, match="unknown levers"):
        isom_block(unit, feed, levers=("pressure",))
    with pytest.raises(ValueError, match="unknown outputs"):
        isom_block(unit, feed, outputs=("octane",))
    with pytest.raises(ValueError, match="outside its bounds"):
        isom_block(unit, feed, bounds={"T_in": (150.0, 160.0)})
