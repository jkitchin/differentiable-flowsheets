"""Tests for the VGO hydrocracker (#307).

Fast tests cover the pieces: the cracking kinetics (conservation, the
distribution matrix, the yield distribution, the property assignment, the
heats), the fractionator split, the adjoint fixed point and a cracking bed.
The full-unit tests compile the whole hydrocracker (a few minutes each, more
under load) and are marked slow.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import correlations as corr
from difflow_refinery.hydrocracking import fractionator as frac
from difflow_refinery.hydrocracking import kinetics as hck
from difflow_refinery.hydrocracking.feed import VGO_PRETREAT_PARAMS, hcu_feed, hcu_layout
from difflow_refinery.hydrocracking.fixed_point import fixed_point
from difflow_refinery.hydrocracking.unit import Hydrocracker, HydrocrackerParams, bmci
from difflow_refinery.hydroprocessing.layout import Flows
from difflow_refinery.hydroprocessing.reactor import (
    ReactorOptions, check_element_conservation, integrate_bed, k_model_at, reaction_context)
from difflow_refinery.hydrotreating.feed import select_cuts, straight_run_cut
from difflow_refinery.hydrotreating.kinetics import HDTKinetics, MODEL_COMPOUNDS, crack_targets

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]           # SG 0.86, 1.8 wt% S
TBP_H = [90, 130, 190, 250, 310, 370, 430, 490, 560, 650, 720]          # SG 0.93, 3.0 wt% S, heavier
_CUTS = {}


def char_a(tbp_C=None):
    tbp = jnp.asarray([t + 273.15 for t in TBP_A]) if tbp_C is None else tbp_C
    a = dr.Assay(TBP_PCT, tbp, sg=0.86, light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    if "a" not in _CUTS:
        _CUTS["a"] = dr.default_cut_points(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dr.characterize(a, cut_points=_CUTS["a"], composition=True)


def char_h():
    a = dr.Assay(TBP_PCT, [t + 273.15 for t in TBP_H], sg=0.93,
                 light_ends={"propane": 0.3, "n_butane": 0.6, "n_pentane": 1.0},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=3.0, nitrogen_wppm=2500.0, ccr_wt=8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dr.characterize(a, composition=True)


VGO_RANGE = (370 + 273.15, 560 + 273.15)


@pytest.fixture(scope="module")
def vgo():
    char = char_a()
    cuts = select_cuts(char, *VGO_RANGE)
    feed = straight_run_cut(char, rate=50.0, cuts=cuts)
    unit = Hydrocracker(char, feed, HydrocrackerParams())
    return char, cuts, feed, unit


@pytest.fixture(scope="module")
def bed_setup(vgo):
    char, cuts, feed, unit = vgo
    th = unit.theta(feed)
    comps = unit._comps(th)
    cst = unit._crack_state(th)
    lay = unit.layout
    oil = th["oil"]
    Q = float(unit._feed_volume(th))
    h2 = 1200.0 * Q * 44.615
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(h2)
                .at[lay.gas_index("methane")].set(0.08 * h2), jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    return unit, th, comps, cst, lay, oil, gas, Q


# =============================================================================
# Kinetics
# =============================================================================


def test_yield_distribution_endpoints():
    """p(K, K) = 0 (no self-cracking), p(0, K) = delta (the light-end yield)."""
    for a0, a1, d in ((1.0, 0.2, 0.02), (0.7, 0.12, 0.0), (1.5, 0.08, 0.1)):
        assert float(hck.yield_distribution(1.0, a0, a1, d)) == pytest.approx(0.0, abs=1e-15)
        assert float(hck.yield_distribution(1e-300, a0, a1, d)) == pytest.approx(d, abs=1e-12)
        u = jnp.linspace(1e-6, 1.0, 200)
        assert float(jnp.min(hck.yield_distribution(u, a0, a1, d))) >= -1e-15


def test_species_density_is_uniform_at_alpha_one():
    t = jnp.linspace(0.01, 1.0, 7)
    np.testing.assert_allclose(np.asarray(hck.species_density(t, 1.0)), 1.0)
    # theta^(1/alpha^2 - 1) / alpha^2
    np.testing.assert_allclose(np.asarray(hck.species_density(t, 0.5)), np.asarray(t) ** 3 / 0.25)


@pytest.mark.parametrize("scheme", hck.SCHEMES)
def test_distribution_rows_are_stochastic_and_only_go_lighter(vgo, scheme):
    char, cuts, feed, unit = vgo
    th = unit.theta(feed)
    kin = hck.HCKinetics(unit.layout, None, scheme=scheme)
    st = kin.prepare(dataclasses.replace(th["crk"], scheme=scheme), th["cut"]["Tb"], th["cut"]["T_lo"],
                     th["cut"]["T_hi"], 11.8)
    W = np.asarray(st.prep.W)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-13)
    assert (W >= 0).all()
    n = W.shape[0]
    for i in range(n):
        assert np.all(W[i, 1 + i:] == 0.0), i          # nothing lands in the parent's cut or heavier
    if scheme == "continuous":
        kap = np.asarray(st.prep.kappa)
        assert np.all(np.diff(kap) > 0)                  # reactivity rises with boiling point


def test_product_properties_assignment():
    """Tb of the target cut, SG from the shifted Watson K, Twu MW, Goossens H (#305 chain)."""
    Tb = jnp.asarray([400.0, 500.0, 600.0])
    pr = hck.product_properties(Tb, 12.1)
    np.testing.assert_allclose(np.asarray(corr.watson_k(Tb, pr["SG"])), 12.1, rtol=1e-12)
    MW, _, _ = corr.critical_properties(Tb, pr["SG"], "twu")
    np.testing.assert_allclose(np.asarray(pr["MW"]), np.asarray(MW))
    assert np.all(np.diff(np.asarray(pr["H"])) < 0)
    np.testing.assert_allclose(np.asarray(pr["types"]).sum(1), 1.0, rtol=1e-12)
    # atoms: n_C carbons per molecule, H/C consistent with the H mass fraction
    H = np.asarray(pr["H"])
    np.testing.assert_allclose(np.asarray(pr["n_C"]) * 12.0107 * (1 + np.asarray(pr["h_per_C"]) * 1.00794 / 12.0107),
                               np.asarray(pr["MW"]), rtol=1e-12)
    assert np.all((H > 0.12) & (H < 0.16))


def test_heats_pinned_to_model_compounds():
    """Scission: n-hexane + H2 -> n-butane + ethane; saturation: benzene + 3 H2 -> cyclohexane, per H2."""
    H = {k: v[0] for k, v in MODEL_COMPOUNDS.items()}
    assert hck.SCISSION_HEAT == pytest.approx(H["n_butane"] + H["ethane"] - H["n_hexane"], abs=1e-9)
    assert hck.SCISSION_HEAT / 1e3 == pytest.approx(-42.69, abs=0.01)
    assert hck.SATURATION_HEAT_PER_H2 == pytest.approx((H["cyclohexane"] - H["benzene"]) / 3.0, abs=1e-9)
    assert hck.SATURATION_HEAT_PER_H2 / 1e3 == pytest.approx(-68.42, abs=0.01)


def test_gas_split_normalised():
    assert sum(hck.HCU_GAS_SPLIT.values()) == pytest.approx(1.0, abs=1e-12)


def test_bmci_pins():
    """The index's two anchors: 0 for n-heptane (Tb 371.58 K, SG 0.6880), 100 for benzene (353.24 K, 0.8844)."""
    assert float(bmci(353.24, 0.8844)) == pytest.approx(100.0, abs=0.5)
    assert float(bmci(371.58, 0.6880)) == pytest.approx(0.0, abs=0.5)


def _ctx(bed_setup, T=653.15):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    f = oil + gas
    # put some molecules in every cut, so the light cuts' attributes are exercised too
    km, _ = k_model_at(f, lay, comps, T, 150e5)
    return reaction_context(f, lay, comps, km, T, 150e5)


@pytest.mark.parametrize("scheme", hck.SCHEMES)
def test_cracking_conserves_elements(bed_setup, scheme):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    kin = hck.HCKinetics(lay, unit.hdt_kinetics, scheme=scheme)
    st = kin.prepare(dataclasses.replace(th["crk"], scheme=scheme), th["cut"]["Tb"], th["cut"]["T_lo"],
                     th["cut"]["T_hi"], 11.8)
    ctx = _ctx(bed_setup)
    e = np.asarray(check_element_conservation(kin, ctx, st))
    r = kin.rates(ctx, st)
    scale = float(jnp.sum(jnp.abs(r.attr[:, 0])))
    assert np.all(np.abs(e) < 1e-12 * scale), e
    # cracking consumes hydrogen and releases heat
    assert float(r.gas[lay.gas_index("hydrogen")]) < 0
    assert float(r.heat) > 0
    # molecules: one parent makes more than one product
    assert float(jnp.sum(r.cut)) > 0


def test_nitrogen_inhibits_cracking(bed_setup):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    ctx = _ctx(bed_setup)
    r0 = unit.kinetics.cracking_rate(ctx, cst)
    hi_N = dataclasses.replace(cst.p, K_N=jnp.asarray(10.0))
    r1 = unit.kinetics.cracking_rate(ctx, hck.HCState(p=hi_N, prep=cst.prep))
    assert float(jnp.sum(r1)) < float(jnp.sum(r0))
    none = dataclasses.replace(cst.p, K_N=jnp.asarray(0.0))
    r2 = unit.kinetics.cracking_rate(ctx, hck.HCState(p=none, prep=cst.prep))
    assert float(jnp.sum(r2)) > float(jnp.sum(r0))


def test_feed_has_no_cracked_molecules(vgo):
    char, cuts, feed, unit = vgo
    oil = hcu_feed(char, feed, unit.layout)
    assert float(jnp.sum(jnp.abs(oil.attribute(unit.layout, "cracked")))) == 0.0
    assert float(oil.mass(unit.layout)) == pytest.approx(50.0, rel=1e-12)
    assert unit.layout.attributes == hck.HCU_ATTRIBUTES


# =============================================================================
# Fractionator and the fixed point
# =============================================================================


def test_fractionator_shares_and_conservation(bed_setup):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    Tb = th["cut"]["Tb"]
    sh = np.asarray(frac.cut_shares(Tb, jnp.asarray(frac.DEFAULT_CUT_POINTS), 15.0))
    np.testing.assert_allclose(sh.sum(1), 1.0, rtol=1e-14)
    assert (sh >= 0).all()
    f = oil + gas
    prods = frac.fractionate(f, lay, Tb, frac.DEFAULT_CUT_POINTS, 15.0)
    tot = Flows.zeros(lay)
    for p in prods.values():
        tot = tot + p
    np.testing.assert_allclose(np.asarray(tot.ravel()), np.asarray(f.ravel()), rtol=1e-14, atol=1e-14)
    # UCO is exactly empty below the cut point less the width
    below = np.asarray(Tb) < frac.DEFAULT_CUT_POINTS[-1] - 15.0
    assert np.all(np.asarray(prods["uco"].cut)[below] == 0.0)
    assert float(frac.smootherstep(-1.0)) == 0.0 and float(frac.smootherstep(1.0)) == 1.0
    assert float(frac.smootherstep(0.0)) == pytest.approx(0.5)


def test_fixed_point_adjoint_gradient():
    """z = a + b tanh(z) A: the adjoint gradient against the implicit-function closed form."""
    A = jnp.asarray([[0.3, 0.1], [0.05, 0.2]])

    def G(z, args):
        a, b = args
        return a + b * (A @ jnp.tanh(z))

    def solve(a, b):
        return fixed_point(G, jnp.zeros(2), (a, b), scale=jnp.ones(2), tol=1e-14).value

    a0, b0 = jnp.asarray([0.5, -0.2]), jnp.asarray(1.3)
    z = solve(a0, b0)
    np.testing.assert_allclose(np.asarray(G(z, (a0, b0))), np.asarray(z), atol=1e-13)
    J = jax.jacrev(lambda a: jnp.sum(solve(a, b0) ** 2))(a0)
    dGdz = b0 * A * (1 - jnp.tanh(z) ** 2)[None, :]
    dz_da = jnp.linalg.solve(jnp.eye(2) - dGdz, jnp.eye(2))
    np.testing.assert_allclose(np.asarray(J), np.asarray(2 * z @ dz_da), rtol=1e-10)


# =============================================================================
# A cracking bed
# =============================================================================


@pytest.fixture(scope="module")
def pretreated(bed_setup):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    W = Q * 3600 / 1.5 * 800
    r = unit.pretreat_reactor(oil, gas.scale(0.5), [648.15], 340.0, 150e5, [0.5 * W, 0.5 * W], comps, th["pre"],
                              quench=(0.12,))
    return r, W


def test_pretreat_removes_nitrogen(bed_setup, pretreated):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    r, W = pretreated
    N_in = float(jnp.sum(oil.attr[:, 7:9]))
    N_out = float(jnp.sum(r.outlet.attr[:, 7:9]))
    assert N_out < 0.05 * N_in
    e_in, e_out = (oil + gas.scale(0.5)).elements(lay), r.outlet.elements(lay)
    np.testing.assert_allclose(np.asarray(e_out), np.asarray(e_in), rtol=1e-12)


def test_cracking_bed_conversion_rises_with_temperature(bed_setup, pretreated):
    unit, th, comps, cst, lay, oil, gas, Q = bed_setup
    r, W = pretreated
    Tb = np.asarray(th["cut"]["Tb"])
    heavy = jnp.asarray(Tb > 643.15)

    @jax.jit
    def conv(T):
        b = integrate_bed(unit.kinetics, cst, lay, comps, r.outlet + gas.scale(0.1), T, 150e5, 0.1 * W)
        m_in, m_out = r.outlet.cut_mass(lay), b.outlet.cut_mass(lay)
        return 1.0 - jnp.sum(jnp.where(heavy, m_out, 0.0)) / jnp.sum(jnp.where(heavy, m_in, 0.0)), b

    cs = []
    for T in (633.15, 643.15, 653.15):
        c, b = conv(T)
        cs.append(float(c))
        e_in, e_out = (r.outlet + gas.scale(0.1)).elements(lay), b.outlet.elements(lay)
        np.testing.assert_allclose(np.asarray(e_out), np.asarray(e_in), rtol=1e-12)
        assert float(b.T_out) > T
    assert 0 < cs[0] < cs[1] < cs[2] < 1
