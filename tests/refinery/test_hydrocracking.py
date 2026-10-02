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


def test_gmres_solves_a_nonsymmetric_system():
    from difflow_refinery.hydrocracking.fixed_point import gmres
    rng = np.random.default_rng(0)
    M = jnp.asarray(np.eye(60) - 0.08 * rng.standard_normal((60, 60)))
    b = jnp.asarray(rng.standard_normal(60))
    x = gmres(lambda v: M @ v, b, k=20, restarts=3)
    np.testing.assert_allclose(np.asarray(M @ x), np.asarray(b), atol=1e-10)


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


# =============================================================================
# The whole unit
# =============================================================================


def vdu_vgo(char, crude_kg_s=150.0):
    """LVGO + HVGO of a VacuumColumn on an idealized atmospheric residue of ``char``."""
    from difflow_refinery.vacuum import VacuumColumn, VacuumColumnParams, atmospheric_residue
    res = atmospheric_residue(char, crude_kg_s)
    out = dict(zip(["vac_overhead", "lvgo", "hvgo", "slop", "vac_residue", "info"],
                   VacuumColumn(VacuumColumnParams(components=char.pseudo_components()))(res)))
    vgo = {k: v for k, v in out["lvgo"].items() if k.startswith("F_")}
    for k, v in out["hvgo"].items():
        if k.startswith("F_"):
            vgo[k] = vgo.get(k, 0.0) + v
    vgo["T"], vgo["P"] = jnp.asarray(298.15), jnp.asarray(101325.0)
    return vgo


def _check_solution(r, rtol=1e-8):
    assert bool(r.converged), (float(r.outputs["tear.residual"]), float(r.outputs["uco.residual"]))
    for k, v in r.balances.items():
        assert float(v) < rtol, (k, float(v))
    o = r.outputs
    assert float(o["h2.chemical"]) == pytest.approx(float(o["h2.consumed_by_balance"]), rel=1e-8)
    assert 0.0 < float(o["conversion.per_pass"]) < 1.0
    assert float(o["crack.gas_left"]) > 0.0
    for p in ("light_naphtha", "heavy_naphtha", "kerosene", "diesel"):
        assert float(o[f"{p}.yield"]) > 0.0
    # the slate is ordered by boiling range
    T50 = [float(o[f"{p}.T50"]) for p in ("heavy_naphtha", "kerosene", "diesel", "uco")]
    assert T50 == sorted(T50)
    assert float(o["pretreat.N_wppm"]) < 0.05 * float(o["feed.N_wppm"])


@pytest.fixture(scope="module")
def light_vdu():
    char = char_a()
    return char, vdu_vgo(char)


@pytest.fixture(scope="module")
def light_once(light_vdu):
    char, feed = light_vdu
    unit = Hydrocracker(char, feed, HydrocrackerParams())
    return unit, feed, unit.solve(feed)


@pytest.mark.slow
def test_light_vgo_once_through_converges_and_balances(light_once):
    unit, feed, r = light_once
    assert unit.dropped_mass_fraction < 1e-3
    _check_solution(r)
    o = r.outputs
    assert float(o["conversion.overall"]) == pytest.approx(float(o["conversion.per_pass"]), rel=1e-12)
    assert 100.0 < float(o["h2.chemical_nm3_m3"]) < 600.0


@pytest.mark.slow
def test_light_vgo_with_uco_recycle(light_vdu, light_once):
    char, feed = light_vdu
    unit = Hydrocracker(char, feed, HydrocrackerParams(uco_recycle=0.6))
    r = unit.solve(feed)
    _check_solution(r)
    o, o1 = r.outputs, light_once[2].outputs
    assert float(o["uco.steps"]) > 1
    assert float(o["uco.recycle_rate"]) > 0
    # the recycle closes: overall = 1 - (1 - rho)(1 - X) / (1 - rho (1 - X)), X the per-pass conversion
    X, rho = float(o["conversion.per_pass"]), 0.6
    assert float(o["conversion.overall"]) == pytest.approx(1 - (1 - rho) * (1 - X) / (1 - rho * (1 - X)), rel=1e-9)
    # on the same catalyst at the same temperature the recycle lowers the per-pass severity,
    # and with it the overcracking: less naphtha per middle distillate
    assert X < float(o1["conversion.per_pass"])
    assert float(o["naphtha_to_middle_distillate"]) < float(o1["naphtha_to_middle_distillate"])


@pytest.mark.slow
@pytest.mark.parametrize("recycle", [0.0, 0.6])
def test_heavy_vgo_converges(recycle):
    """The heavy crude's VGO (SG 0.93 crude, 3 wt% S): heavier, so more reactive per the lumping
    model and more exothermic -- run 15 K cooler (at the light VGO's 380 C inlets its beds run away)."""
    char = char_h()
    feed = vdu_vgo(char)
    unit = Hydrocracker(char, feed, HydrocrackerParams(uco_recycle=recycle, T_crack=638.15))
    r = unit.solve(feed)
    _check_solution(r)


@pytest.mark.slow
@pytest.mark.release
def test_trends_with_cracking_temperature(light_once):
    """Conversion rises with WABT; naphtha/middle distillate and H2 consumption rise with conversion."""
    unit, feed, _ = light_once
    p = unit.params
    rows = []
    for T in (633.15, 643.15, 653.15):
        o = unit.solve(feed, params=dataclasses.replace(p, T_crack=T)).outputs
        rows.append([float(o[k]) for k in ("wabt.crack", "conversion.per_pass", "naphtha_to_middle_distillate",
                                           "h2.chemical")])
    rows = np.asarray(rows)
    for j in range(4):
        assert np.all(np.diff(rows[:, j]) > 0), (j, rows)


#: AD against Richardson-extrapolated central differences.
GRAD_RTOL = 1e-5


def richardson(f, x0, h):
    d = lambda s: (f(x0 + s) - f(x0 - s)) / (2 * s)
    return (4.0 * d(h / 2) - d(h)) / 3.0


@pytest.mark.slow
@pytest.mark.release
def test_gradients_match_central_differences(light_once):
    """d(kerosene yield, conversion, chemical H2)/d(cracking T, pressure, one TBP point of the assay)."""
    unit, feed, _ = light_once
    p0 = unit.params
    tbp0 = jnp.asarray([t + 273.15 for t in TBP_A])
    char0 = char_a()
    names = [n for n in char0.pseudo_names if n in unit.cuts]
    mw0 = dict(zip(char0.names, np.asarray(char0.component_MW)))
    # hold the feed's MASS per cut while the assay moves (the cut's molecular weight moves with it)
    mass = {n: float(feed.get(f"F_{n}", 0.0)) * mw0[n] for n in names}
    keys = ("kerosene.yield", "conversion.per_pass", "h2.chemical")

    def f(v):
        c = char_a(tbp0.at[7].set(v[2]))
        mw = dict(zip(c.names, c.component_MW))
        fd = {f"F_{n}": mass[n] / mw[n] for n in names}
        r = unit.solve(fd, char=c, params=dataclasses.replace(p0, T_crack=v[0], P=v[1]), warn=False).outputs
        return jnp.stack([r[k] for k in keys])

    v0 = jnp.asarray([float(p0.T_crack), float(p0.P), float(tbp0[7])])
    J = np.asarray(jax.jacrev(f)(v0))
    for j, h in enumerate((0.5, 1e5, 1.0)):
        e = jnp.zeros(3).at[j].set(1.0)
        fd = np.asarray(richardson(lambda t: f(v0 + t * e), 0.0, h))
        np.testing.assert_allclose(J[:, j], fd, rtol=GRAD_RTOL, err_msg=f"input {j}")


@pytest.mark.slow
@pytest.mark.release
def test_recycle_ratio_gradient(light_vdu):
    """d(overall conversion, diesel yield, chemical H2)/d(UCO recycle fraction), through the UCO tear's adjoint."""
    char, feed = light_vdu
    unit = Hydrocracker(char, feed, HydrocrackerParams(uco_recycle=0.6))
    p0 = unit.params
    keys = ("conversion.overall", "diesel.yield", "h2.chemical")

    def f(x):
        o = unit.solve(feed, params=dataclasses.replace(p0, uco_recycle=x), warn=False).outputs
        return jnp.stack([o[k] for k in keys])

    g = np.asarray(jax.jacrev(f)(0.6))
    fd = np.asarray(richardson(f, 0.6, 0.02))
    np.testing.assert_allclose(g, fd, rtol=GRAD_RTOL)


@pytest.mark.slow
def test_products_blend_into_jet_and_ulsd_pools(light_once):
    from difflow_refinery.blending import BlendComponent, BlendPool
    unit, feed, r = light_once
    grid = r.product_char
    for name, pool in (("kerosene", "jet"), ("diesel", "ulsd")):
        comp = BlendComponent.from_stream(f"hcu_{name}", r.product_stream(name), grid, flash_C=60.0)
        assert float(comp.properties["SG"]) == pytest.approx(float(r.outputs[f"{name}.sg"]), rel=1e-9)
        assert float(comp.properties["S_ppm"]) == pytest.approx(float(r.outputs[f"{name}.S_wppm"]), rel=1e-9,
                                                               abs=1e-9)
        res = BlendPool(pool, specs=[])([comp], [1.0])     # no specs: freeze/smoke points are not computed
        assert np.isfinite(float(res.properties["SG"]))
    d = BlendComponent.from_stream("hcu_diesel", r.product_stream("diesel"), grid, flash_C=60.0)
    assert float(BlendPool("ulsd", specs=[])([d], [1.0]).properties["cetane_index"]) == pytest.approx(
        float(r.outputs["diesel.cetane_index"]), rel=1e-6)


@pytest.mark.slow
@pytest.mark.release
def test_hcu_block_delta_vectors(light_once):
    from difflow.planning import check_delta_vectors
    from difflow_refinery.hydrocracking.planning import hcu_block
    unit, feed, _ = light_once
    blk = hcu_block(unit, feed, ["crack.T_in", "uco.cut_point"], ["kerosene.bpd", "conversion.per_pass",
                                                                   "h2.chemical"], feed_product="vgo")
    chk = check_delta_vectors(blk, rtol=1e-3)
    assert chk["passed"], chk["max_rel_error"]


@pytest.mark.slow
def test_discrete_lump_scheme_runs_the_unit(light_vdu):
    """The discrete-lump option through the whole unit: converges, closes, and makes a lighter slate when hotter."""
    char, feed = light_vdu
    p = HydrocrackerParams(T_crack=638.15, crack_kinetics=hck.HCKineticParams(scheme="discrete"))
    unit = Hydrocracker(char, feed, p)
    r = unit.solve(feed)
    _check_solution(r)
    hot = unit.solve(feed, params=dataclasses.replace(p, T_crack=p.T_crack + 5.0)).outputs
    assert float(hot["conversion.per_pass"]) > float(r.outputs["conversion.per_pass"])
