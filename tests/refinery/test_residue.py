"""Residue desulfurizer and the low-sulfur fuel-oil route (#331): smoke and validation."""

import warnings
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay
from difflow_refinery.composition import CompositionRangeWarning
from difflow_refinery.residue import (
    RDS_ATTRIBUTES, ResidueDesulfurizer, atmospheric_residue_cut, cutter_fraction_for_sulfur,
    fuel_oil_blend, rds_feed, rds_layout)
from difflow_refinery.residue.kinetics import conversion_targets

jax.config.update("jax_enable_x64", True)

C = 273.15
FEED_KG_S = 50.0


def crude_and_cuts():
    """The test crude of examples/35-40, with the Ni+V a residue unit needs; and its cut points."""
    pct = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    t_c = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
    assay = Assay(pct, [t + C for t in t_c], sg=0.86,
                  light_ends={"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
                              "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5},
                  sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=5.0, nickel_vanadium_wppm=40.0)
    return assay, dr.default_cut_points(assay, ((673.15, 30.0), (873.15, 60.0), (np.inf, 150.0)))


@pytest.fixture(scope="module")
def char():
    assay, cuts = crude_and_cuts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionRangeWarning)
        return dr.characterize(assay, cuts, composition=True)


@pytest.fixture(scope="module")
def feed(char):
    return atmospheric_residue_cut(char, FEED_KG_S)


@pytest.fixture(scope="module")
def rds(char, feed):
    return ResidueDesulfurizer(char, feed)


@pytest.fixture(scope="module")
def result(rds, feed):
    return rds.solve(feed)


def fuel_oil(r):
    """Fuel oil = the whole liquid product of the RDS (desulfurized residue + its distillate)."""
    return fuel_oil_blend([r.blend_component("residue"), r.blend_component("distillate")],
                          [r.volume("residue"), r.volume("distillate")])


def test_feed_layout_smoke(char, feed):
    cuts = tuple(n for n in char.pseudo_names if f"F_{n}" in feed)
    lay = rds_layout(char, cuts)
    assert lay.attributes == RDS_ATTRIBUTES
    f = rds_feed(char, feed, lay)
    assert np.all(np.isfinite(np.asarray(f.attr)))


def test_conversion_targets_and_cutter_fraction():
    tgt, mult = conversion_targets(np.array([10.0, 15.0, 20.0, 30.0, 40.0, 60.0]),
                                   np.array([500.0, 600.0, 700.0, 780.0, 850.0, 950.0]), 811.15)
    assert len(tgt) == 6 and len(mult) == 6
    assert 0.0 < float(cutter_fraction_for_sulfur(3.3e4, 15.0, 5000.0)) < 1.0


def test_unit_solves_and_fuel_oil_blends(result):
    assert bool(result.converged)
    assert all(np.isfinite(float(v)) for v in result.balances.values())
    fo = fuel_oil(result)
    assert np.isfinite(float(fo.properties["S_ppm"]))


def test_structure_is_fixed_at_construction(rds, feed):
    with pytest.raises(ValueError, match="T_in"):
        rds.solve(feed, params=rds.params.update(T_in=(640.0, 650.0)))
    with pytest.raises(ValueError, match="quench"):
        rds.solve(feed, params=rds.params.update(quench=None))
    with pytest.raises(ValueError, match="composition"):
        ResidueDesulfurizer(SimpleNamespace(composition=None), feed)


def test_grad_finite(rds, feed):
    def fo_S(T):
        return fuel_oil(rds.solve(feed, params=rds.params.update(T_in=(T,)))).properties["S_ppm"]

    assert np.isfinite(float(jax.jit(jax.grad(fo_S))(rds.params.T_in[0])))
