"""Residue desulfurizer and the low-sulfur fuel-oil route (#331).

The route: atmospheric residue -> :class:`ResidueDesulfurizer` (trickle beds
on the shared hydroprocessing blocks, residue kinetics) -> fuel-oil pool
(:func:`fuel_oil_blend`, a :class:`BlendPool` with the VLSFO specs).

The fuel oil meets 0.5 wt% S, the unit's element, mass and
metals balances close, and the route's sulfur and mass balances close from
the residue feed to the fuel-oil pool and the H2S. The calibration ranges of the illustrative constants are pinned.
"""

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, BlendCharacterization, BlendComponent
from difflow_refinery.composition import SULFUR_CLASSES, CompositionRangeWarning
from difflow_refinery.hydroprocessing.layout import ATOMIC_MASS
from difflow_refinery.residue import (
    RDS_ATTRIBUTES, RDSParams, ResidueDesulfurizer, atmospheric_residue_cut, cutter_fraction_for_sulfur,
    fuel_oil_blend, rds_feed, rds_layout)
from difflow_refinery.residue.kinetics import CCR_MW, METAL_MW, conversion_targets

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


# -----------------------------------------------------------------------------
# Feed and layout (no reactor)
# -----------------------------------------------------------------------------


def test_feed_carries_the_residue_sulfur_metals_and_ccr(char, feed):
    cuts = tuple(n for n in char.pseudo_names if f"F_{n}" in feed)
    lay = rds_layout(char, cuts)
    assert lay.attributes == RDS_ATTRIBUTES
    f = rds_feed(char, feed, lay)
    names = list(char.names)
    idx = np.asarray([names.index(c) for c in cuts])
    F = np.asarray([float(feed[f"F_{c}"]) for c in cuts])
    m = F * np.asarray(char.component_MW)[idx]                       # g/s
    # mass from atoms = F * MW (metals and CCR are not counted twice)
    np.testing.assert_allclose(np.asarray(f.cut_mass(lay)) * 1000.0, m, rtol=1e-12)
    ai = {a: i for i, a in enumerate(lay.attributes)}
    s_cols = [ai[f"S_{k}"] for k in SULFUR_CLASSES] + [ai["S_residue"]]
    S = np.asarray(f.attr)[:, s_cols].sum(1) * ATOMIC_MASS["S"]
    np.testing.assert_allclose(S, m * np.asarray(char.sulfur)[idx], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.attr)[:, ai["NiV"]] * METAL_MW,
                               m * np.asarray(char.nickel_vanadium)[idx], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.attr)[:, ai["CCR"]] * CCR_MW, m * np.asarray(char.ccr)[idx],
                               rtol=1e-12)
    # the refractory share is zero in the VGO range and rises into the vacuum residue
    share = np.asarray(f.attr)[:, ai["S_residue"]] * ATOMIC_MASS["S"] / S
    assert share[0] == 0.0 and share[-1] > 0.3 and np.all(np.diff(share) >= 0)


def test_conversion_sends_heavy_cuts_to_lighter_ones():
    cpm = np.array([10.0, 15.0, 20.0, 30.0, 40.0, 60.0])
    Tb = np.array([500.0, 600.0, 700.0, 780.0, 850.0, 950.0])
    tgt, mult = conversion_targets(cpm, Tb, 811.15)
    assert tgt[:4] == (-1, -1, -1, -1)
    assert tgt[4] == 2 and tgt[5] == 3
    np.testing.assert_allclose(mult[4:], (2.0, 2.0))


def test_lever_rule_and_pool_agree(char, feed):
    """The cutter arithmetic that rules out cutter blending alone (see the docs)."""
    fo_char = BlendCharacterization.from_characterization(char, contaminants=True)
    raw = BlendComponent.from_stream("atmospheric residue", feed, fo_char, estimate=("viscosity_cSt",))
    S_raw = raw.properties["S_ppm"]
    assert float(S_raw) > 3e4                                           # ~3.3 wt%
    x = cutter_fraction_for_sulfur(S_raw, 15.0, 5000.0)
    assert float(x) > 0.8                                               # 85 % ULSD to make 0.5 wt%
    cutter = BlendComponent.from_properties("ULSD", SG=0.84, S_ppm=15.0, N_ppm=10.0, CCR_wt=0.0,
                                            viscosity_cSt=2.5)
    sg_r, sg_c = float(raw.properties["SG"]), 0.84
    # volumes giving cutter mass fraction x
    V = [(1.0 - float(x)) / sg_r, float(x) / sg_c]
    pool = fuel_oil_blend([raw, cutter], V)
    np.testing.assert_allclose(float(pool.properties["S_ppm"]), 5000.0, rtol=1e-12)


# -----------------------------------------------------------------------------
# The unit and the route
# -----------------------------------------------------------------------------


def test_unit_converges_and_balances_close(result):
    assert bool(result.converged)
    for k, v in result.balances.items():
        assert float(v) < 1e-10, (k, float(v))


def test_fuel_oil_meets_the_half_percent_sulfur_spec(char, feed, result):
    fo = fuel_oil(result)
    assert float(fo.properties["S_ppm"]) <= 5000.0
    for spec, margin in fo.margins.items():
        assert float(margin) >= 0.0, spec
    # the residue it replaces does not
    fo_char = BlendCharacterization.from_characterization(char, contaminants=True)
    raw = BlendComponent.from_stream("atmospheric residue", feed, fo_char, estimate=("viscosity_cSt",))
    raw_pool = fuel_oil_blend([raw], [1.0])
    assert float(raw_pool.margins["S_ppm <= 5000"]) < 0.0
    # desulfurizing lowers gravity and viscosity, as hydrogen goes in
    assert float(fo.properties["SG"]) < float(raw.properties["SG"])
    assert float(fo.properties["viscosity_cSt"]) < float(raw.properties["viscosity_cSt"])


def test_route_sulfur_and_mass_balances_close(rds, result):
    """Residue in = fuel oil + H2S (sulfur); residue + treat gas = fuel oil + gas (mass)."""
    lay = rds.layout
    fo = fuel_oil(result)
    o = result.outputs
    S_in = float(o["feed.S_wt"]) / 100.0 * float(jnp.sum(result.streams["feed"].cut_mass(lay)))
    S_fo = float(fo.properties["S_ppm"]) * 1e-6 * float(fo.mass)
    S_h2s = float(result.streams["gas"].gas_flow(lay, "hydrogen_sulfide")) * ATOMIC_MASS["S"] / 1000.0
    np.testing.assert_allclose(S_fo + S_h2s, S_in, rtol=1e-10)
    np.testing.assert_allclose(float(o["h2s.make"]) * ATOMIC_MASS["S"] / 1000.0, S_h2s, rtol=1e-12)
    m_in = float(result.streams["feed"].mass(lay) + result.streams["treat_gas"].mass(lay))
    m_out = float(fo.mass) + float(result.streams["gas"].mass(lay))
    np.testing.assert_allclose(m_out, m_in, rtol=1e-10)
    # the pool's mass is the products' mass
    np.testing.assert_allclose(float(fo.mass), float(o["residue.rate"] + o["distillate.rate"]), rtol=1e-12)


def test_structure_is_fixed_at_construction(rds, feed):
    with pytest.raises(ValueError, match="T_in"):
        rds.solve(feed, params=rds.params.update(T_in=(640.0, 650.0)))
    with pytest.raises(ValueError, match="quench"):
        rds.solve(feed, params=rds.params.update(quench=None))
    with pytest.raises(ValueError, match="composition"):
        ResidueDesulfurizer(SimpleNamespace(composition=None), feed)


# -----------------------------------------------------------------------------
# Release: calibration ranges, trends, the gradient
# -----------------------------------------------------------------------------


def test_illustrative_constants_land_in_ards_ranges(result):
    """The constants were chosen to give ARDS-like severities (see the kinetics docstring)."""
    o = {k: float(v) for k, v in result.outputs.items()}
    assert 380.0 < o["wabt"] - C < 405.0
    assert 0.85 < o["hds.conversion"] < 0.95
    assert 0.70 < o["hdm.conversion"] < 0.90
    assert 0.35 < o["ccr.reduction"] < 0.65
    assert 0.05 < o["conversion"] < 0.25
    assert 80.0 < o["h2.chemical_nm3_m3"] < 200.0
    assert o["residue.S_wt"] < 0.5
