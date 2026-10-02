"""Tests for the hydroprocessing building blocks and the hydrotreater (#306).

Fast tests cover the pieces (layout, PR flash, compressor, kinetics,
thermochemistry pins, one bed). The full-unit tests compile the whole
hydrotreater (about a minute each, more under load) and are marked slow.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow.eos import CriticalProperties, PengRobinson
from difflow_refinery.hydroprocessing.layout import ATOMIC_MASS, Flows, Layout
from difflow_refinery.hydroprocessing.reactor import (
    ReactorOptions, check_element_conservation, integrate_bed, k_model_at, reaction_context)
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3, compress, solve_tear
from difflow_refinery.hydroprocessing.separator import flash_components, pr_flash, rachford_rice
from difflow_refinery.hydroprocessing.solve import newton_solve
from difflow_refinery.hydroprocessing.thermo import Components, pr_lnphi
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, TargetSpec, straight_run_cut
from difflow_refinery.hydrotreating import kinetics as hk
from difflow_refinery.hydrotreating.feed import hdt_feed, select_cuts

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]     # 0.86 SG, high sulfur
PCT_B = [5, 10, 30, 50, 70, 90, 95]
TBP_B = [40, 80, 160, 250, 350, 480, 560]                         # 0.83 SG, low sulfur
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


def char_b():
    a = dr.Assay(PCT_B, [t + 273.15 for t in TBP_B], sg=0.83, light_ends={"propane": 0.3, "n_butane": 0.8},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=0.35, nitrogen_wppm=500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dr.characterize(a, composition=True)


@pytest.fixture(scope="module")
def diesel():
    char = char_a()
    cuts = select_cuts(char, 230 + 273.15, 370 + 273.15)
    feed = straight_run_cut(char, rate=50.0, cuts=cuts)
    unit = Hydrotreater(char, feed, HydrotreaterParams())
    return char, cuts, feed, unit


@pytest.fixture(scope="module")
def bed_setup(diesel):
    char, cuts, feed, unit = diesel
    th = unit.theta(feed)
    comps = unit._comps(th)
    kin = unit._kin_params(th["kin"])
    lay = unit.layout
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(700.0)
                .at[lay.gas_index("methane")].set(60.0).at[lay.gas_index("hydrogen_sulfide")].set(5.0),
                jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    return unit, th, comps, kin, lay, th["oil"], gas


# =============================================================================
# Layout and feed
# =============================================================================


def test_feed_mass_from_atoms_equals_char_mass(diesel):
    char, cuts, feed, unit = diesel
    lay = unit.layout
    oil = hdt_feed(char, feed, lay)
    names = list(char.names)
    MW = np.asarray(char.component_MW)[[names.index(c) for c in lay.cuts]]
    F = np.asarray(oil.cut)
    np.testing.assert_allclose(np.asarray(oil.cut_mass(lay)), F * MW / 1000.0, rtol=1e-12, atol=1e-14)
    assert float(oil.mass(lay)) == pytest.approx(50.0, rel=1e-12)
    # sulfur and nitrogen are the characterization's, by class
    comp = char.composition
    idx = [names.index(c) for c in lay.cuts]
    S = float(np.sum(F * MW * np.asarray(comp.sulfur)[idx]))
    S_attr = float(jnp.sum(oil.attr[:, 2:7])) * ATOMIC_MASS["S"]
    assert S_attr == pytest.approx(S, rel=1e-12)


def test_stream_round_trip(diesel):
    char, cuts, feed, unit = diesel
    lay = unit.layout
    oil = hdt_feed(char, feed, lay)
    s = oil.to_stream(lay, 300.0, 1e5)
    back = Flows.from_stream(s, lay)
    np.testing.assert_array_equal(np.asarray(back.attr), np.asarray(oil.attr))
    assert all(k.startswith("F_") for k in lay.keys())
    assert f"F_{cuts[0]}@S_dibenzothiophenes" in s


def test_layout_requires_carbon_and_hydrogen():
    with pytest.raises(ValueError, match="required"):
        Layout(("hydrogen",), ("pc01",), ("S",), ("S",))


# =============================================================================
# Thermodynamics
# =============================================================================


def test_pr_fugacity_matches_difflow_pengrobinson():
    names = ("methane", "propane", "n_hexane")
    crit = {n: dr.thermo.LIGHT_ENDS[n] for n in names}
    species = {n: CriticalProperties(n, crit[n][2], crit[n][3], crit[n][4], crit[n][0]) for n in names}
    eos = PengRobinson(species)
    comps = Components(names=names, n_gas=3, Tb=jnp.asarray([crit[n][1] for n in names]),
                       Tc=jnp.asarray([crit[n][2] for n in names]), Pc=jnp.asarray([crit[n][3] for n in names]),
                       omega=jnp.asarray([crit[n][4] for n in names]), hvap_nb=jnp.ones(3),
                       cp_ig=jnp.zeros((3, 4)), SG=jnp.ones(3), MW=jnp.ones(3), kij=jnp.zeros((3, 3)))
    x = jnp.asarray([0.2, 0.3, 0.5])
    for phase in ("liquid", "vapor"):
        mine, _ = pr_lnphi(350.0, 20e5, x, comps, phase)
        ref = jnp.log(eos.fugacity_coefficient(350.0, 20e5, x, phase))
        np.testing.assert_allclose(np.asarray(mine), np.asarray(ref), rtol=1e-8, atol=1e-10)


def test_ideal_gas_cp_pinned_to_poling_5th_edition():
    from difflow_refinery.hydroprocessing.thermo import GAS_PROPERTIES
    # PPO 5th ed. App. A polynomial values as tabulated in `chemicals` (J/mol/K)
    ref = {"hydrogen": (28.785, 29.312, 29.425), "hydrogen_sulfide": (34.105, 37.297, 40.981),
           "ammonia": (35.736, 41.997, 48.418)}
    for name, vals in ref.items():
        a, b, c, d = GAS_PROPERTIES[name][6]
        for T, v in zip((300.0, 500.0, 700.0), vals):
            assert a + b * T + c * T**2 + d * T**3 == pytest.approx(v, rel=5e-3)


def test_rachford_rice_negative_flash():
    z = jnp.asarray([0.6, 0.3, 0.1])
    K = jnp.asarray([8.0, 3.0, 0.9])
    V = rachford_rice(z, K)
    assert float(jnp.sum(z * (K - 1) / (1 + V * (K - 1)))) == pytest.approx(0.0, abs=1e-12)
    assert float(V) > 1.0                     # a superheated feed: the negative flash gives V > 1


def test_pr_flash_converges_and_differentiates(bed_setup):
    unit, th, comps, kin, lay, oil, gas = bed_setup
    z = flash_components(oil + gas, lay, comps)
    fl = jax.jit(pr_flash)
    r = fl(323.15, 47e5, z, comps)
    assert float(r.residual) < 1e-10
    assert 0.0 < float(r.beta) < 1.0
    g = jax.grad(lambda T: pr_flash(T, 47e5, z, comps).x[0])(323.15)
    fd = (fl(323.4, 47e5, z, comps).x[0] - fl(322.9, 47e5, z, comps).x[0]) / 0.5
    assert float(g) == pytest.approx(float(fd), rel=1e-5)


def test_compressor_isentropic_ideal_gas():
    lay = Layout(("hydrogen", "methane"), ("pc01",), ("C", "H"), ("C", "H"))
    comps = Components.build(lay, [400.0], [0.75], [100.0], [550.0], [3e6], [0.3], [3e4],
                             [[0.0, 0.5, 0.0, 0.0]])
    gas = Flows(jnp.asarray([90.0, 10.0]), jnp.zeros(1), jnp.zeros((1, 2)))
    T2, W = compress(gas, lay, comps, 320.0, 40e5, 50e5, 1.0)
    # isentropic: sum z int Cp/T = n R ln(P2/P1); W = sum z int Cp dT
    from difflow_refinery.hydroprocessing.thermo import GAS_PROPERTIES, R_GAS
    from scipy.integrate import quad
    cps = [GAS_PROPERTIES[g][6] for g in ("hydrogen", "methane")]
    cp = lambda T: 0.9 * np.polyval(cps[0][::-1], T) + 0.1 * np.polyval(cps[1][::-1], T)
    s = quad(lambda T: cp(T) / T, 320.0, float(T2))[0]
    assert s == pytest.approx(R_GAS * np.log(50 / 40), rel=1e-8)
    assert float(W) == pytest.approx(100.0 * quad(cp, 320.0, float(T2))[0], rel=1e-8)
    T2e, We = compress(gas, lay, comps, 320.0, 40e5, 50e5, 0.75)
    assert float(We) == pytest.approx(float(W) / 0.75, rel=1e-10)


def test_newton_solve_implicit_gradient():
    # x^3 + a x - 3 = 0: dx/da = -x / (3 x^2 + a)
    f = lambda x, a: x**3 + a * x - 3.0
    sol = newton_solve(f, jnp.asarray([1.0]), jnp.asarray(2.0))
    x = float(sol.value[0])
    g = jax.grad(lambda a: newton_solve(f, jnp.asarray([1.0]), a).value[0])(2.0)
    assert float(g) == pytest.approx(-x / (3 * x**2 + 2.0), rel=1e-12)
    gf = jax.jacfwd(lambda a: newton_solve(f, jnp.asarray([1.0]), a, jac="fwd").value[0])(2.0)
    assert float(gf) == pytest.approx(float(g), rel=1e-12)


def test_solve_tear_fixed_point():
    g = lambda x, a: 0.9 * x + a                     # fixed point x = 10 a, dx/da = 10
    s = solve_tear(g, jnp.asarray([1.0, 2.0]), jnp.asarray([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(s.value), [5.0, 10.0], rtol=1e-12)
    J = jax.jacrev(lambda a: solve_tear(g, jnp.asarray([1.0, 2.0]), a).value)(jnp.asarray([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(J), 10.0 * np.eye(2), rtol=1e-10)


# =============================================================================
# Kinetics and thermochemistry
# =============================================================================


def test_thermochemistry_pinned_to_model_compounds():
    # kJ/mol, from the Hf of the `chemicals` tables quoted in kinetics.MODEL_COMPOUNDS
    np.testing.assert_allclose(np.asarray(hk.HDS_HEAT) / 1e3, [-104.66, -261.35, -157.0, -83.92, -173.06],
                               atol=0.02)
    np.testing.assert_allclose(np.asarray(hk.HDN_HEAT) / 1e3, [-238.158, -262.958], atol=1e-3)
    assert [round(t[0] / 1e3, 2) for t in hk.AROMATIC_THERMO] == [-115.2, -124.6, -205.26]
    assert hk.AROMATIC_THERMO[2][1] == pytest.approx(298.19 - 269.2 - 3 * 130.7)
    assert hk.AROMATIC_THERMO[1][1] == pytest.approx(366.22 - 333.1 - 2 * 130.7)
    assert hk.OLEFIN_HEAT == pytest.approx(-166940.0 + 43500.0)
    assert hk.HDS_H2 == (2.0, 4.0, 3.0, 2.6, 3.95)
    assert hk.HDN_H2 == (4.0, 5.0)
    # every model reaction is balanced in hydrogen: H2 per event = (H gained by the HC + H in H2S/NH3)/2
    assert sum(hk.CRACK_GAS_SPLIT.values()) == pytest.approx(1.0)


def test_kinetics_conserve_elements(bed_setup):
    unit, th, comps, kin, lay, oil, gas = bed_setup
    stream = oil + gas
    km, _ = k_model_at(stream, lay, comps, 623.15, 50e5)
    ctx = reaction_context(stream, lay, comps, km, 630.0, 50e5)
    r = unit.kinetics.rates(ctx, kin)
    net = check_element_conservation(unit.kinetics, ctx, kin)
    scale = float(jnp.max(jnp.abs(r.attr)))
    assert scale > 0
    np.testing.assert_allclose(np.asarray(net), 0.0, atol=1e-12 * scale * lay.n_cut)
    assert float(r.heat) > 0                        # exothermic
    assert float(r.gas[lay.gas_index("hydrogen")]) < 0
    assert float(r.gas[lay.gas_index("hydrogen_sulfide")]) > 0


def test_power_law_fallback(bed_setup):
    unit, th, comps, kin, lay, oil, gas = bed_setup
    stream = oil + gas
    km, _ = k_model_at(stream, lay, comps, 623.15, 50e5)
    ctx = reaction_context(stream, lay, comps, km, 623.15, 50e5)
    p = dataclasses.replace(kin, hds_form="power", hds_n=1.0, hds_h2_order=0.0, crack_k=0.0)
    r = unit.kinetics.rates(ctx, p)
    kS = np.asarray(p.hds_k)
    np.testing.assert_allclose(np.asarray(-r.attr[:, 2:7]), kS * np.asarray(ctx.c_attr[:, 2:7]), rtol=1e-12)
    assert float(check_element_conservation(unit.kinetics, ctx, p)[2]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
@pytest.mark.release
def test_bed_balances_and_gradient(bed_setup):
    unit, th, comps, kin, lay, oil, gas = bed_setup

    def run(T, h2):
        g = Flows(gas.gas.at[lay.gas_index("hydrogen")].set(h2), gas.cut, gas.attr)
        return integrate_bed(unit.kinetics, kin, lay, comps, oil + g, T, 50e5, 67000.0)

    b = jax.jit(run)(613.15, 700.0)
    np.testing.assert_allclose(np.asarray(b.outlet.elements(lay)), np.asarray(b.inlet.elements(lay)),
                               rtol=1e-11)
    assert float(b.T_out) > 613.15
    S = lambda T, h2: jnp.sum(run(T, h2).outlet.attr[:, 2:7])
    g = jax.jit(jax.grad(S, argnums=(0, 1)))(613.15, 700.0)
    f = jax.jit(S)
    fdT = (f(613.35, 700.0) - f(612.95, 700.0)) / 0.4
    fdH = (f(613.15, 701.0) - f(613.15, 699.0)) / 2.0
    assert float(g[0]) == pytest.approx(float(fdT), rel=1e-4)
    assert float(g[1]) == pytest.approx(float(fdH), rel=1e-4)


@pytest.mark.slow
@pytest.mark.release
def test_aromatics_pass_through_a_minimum_with_temperature(bed_setup):
    """Saturation is equilibrium-limited at high T: outlet aromatics fall, then rise."""
    unit, th, comps, kin, lay, oil, gas = bed_setup
    arom = jax.jit(lambda T: jnp.sum(integrate_bed(unit.kinetics, kin, lay, comps, oil + gas.scale(0.6), T,
                                                   30e5, 67000.0).outlet.attr[:, 9:12]))
    Ts = np.arange(300.0, 481.0, 15.0) + 273.15
    a = np.asarray([float(arom(T)) for T in Ts])
    i = int(np.argmin(a))
    assert 0 < i < len(Ts) - 1, a
    assert a[0] > a[i] and a[-1] > a[i]


# =============================================================================
# The whole unit
# =============================================================================


@pytest.fixture(scope="module")
def diesel_result(diesel):
    char, cuts, feed, unit = diesel
    return unit.solve(feed)


@pytest.mark.slow
def test_diesel_converges_and_balances(diesel_result):
    r = diesel_result
    assert bool(r.converged)
    for k, v in r.balances.items():
        assert float(v) < 1e-8, k
    o = r.outputs
    assert float(o["h2.chemical"]) == pytest.approx(float(o["h2.consumed_by_balance"]), rel=1e-8)
    assert 10.0 < float(o["product.S_wppm"]) < float(o["feed.S_wppm"])
    assert float(o["product.N_wppm"]) < float(o["feed.N_wppm"])
    assert 0.95 < float(o["product.yield"]) < 1.0
    assert float(o["reactor.dT_total"]) > 0
    assert 0 < float(o["recycle.h2_purity"]) < 1


@pytest.mark.slow
@pytest.mark.release  # the second-feed acceptance claim; the diesel covers the code per commit
def test_kerosene_from_second_assay_converges():
    char = char_b()
    feed = straight_run_cut(char, 150 + 273.15, 250 + 273.15, 30.0)
    unit = Hydrotreater(char, feed, HydrotreaterParams(T_in=(593.15,), P=35e5, h2_oil=150.0, lhsv=2.5,
                                                         stripper_feed_T=473.15))
    r = unit.solve(feed)
    assert bool(r.converged)
    for k, v in r.balances.items():
        assert float(v) < 1e-8, k
    assert float(r.outputs["product.S_wppm"]) < float(r.outputs["feed.S_wppm"])


@pytest.mark.slow
@pytest.mark.release
def test_trends_with_severity(diesel):
    char, cuts, feed, unit = diesel
    p = unit.params
    out = lambda **kw: unit.solve(feed, params=dataclasses.replace(p, **kw)).outputs
    lo, mid, hi = out(T_in=(603.15,)), out(T_in=(613.15,)), out(T_in=(623.15,))
    S = [float(x["product.S_wppm"]) for x in (lo, mid, hi)]
    H = [float(x["h2.chemical"]) for x in (lo, mid, hi)]
    assert S[0] > S[1] > S[2]
    assert H[0] < H[1] < H[2]
    pl, ph = out(P=40e5), out(P=60e5)
    assert float(pl["reactor.pH2_in"]) < float(ph["reactor.pH2_in"])
    assert float(pl["product.S_wppm"]) > float(ph["product.S_wppm"])


#: AD against Richardson-extrapolated central differences (truncation O(h^4)).
GRAD_RTOL = 1e-5


def richardson(f, x0, h):
    """Central difference extrapolated from steps h and h/2: ``(4 D(h/2) - D(h)) / 3``."""
    d = lambda s: (f(x0 + s) - f(x0 - s)) / (2 * s)
    return (4.0 * d(h / 2) - d(h)) / 3.0


@pytest.mark.slow
@pytest.mark.release
def test_gradients_match_central_differences(diesel):
    """d(product S, chemical H2, liquid yield) / d(inlet T, pressure, H2/oil, TBP 50 % point).

    One reverse-mode Jacobian of one function of all four (one compile), against
    Richardson-extrapolated central differences of the same solve.
    """
    char, cuts, feed, unit = diesel
    p0 = unit.params
    tbp0 = jnp.asarray([t + 273.15 for t in TBP_A])
    keys = ("product.S_wppm", "h2.chemical", "product.yield")

    def f(v):
        c = char_a(tbp0.at[5].set(v[3]))
        fd = straight_run_cut(c, rate=50.0, cuts=cuts)
        r = unit.solve(fd, char=c, params=dataclasses.replace(p0, T_in=(v[0],), P=v[1], h2_oil=v[2]),
                       warn=False).outputs
        return jnp.stack([r[k] for k in keys])

    v0 = jnp.asarray([613.15, 50e5, 300.0, float(tbp0[5])])
    J = np.asarray(jax.jacrev(f)(v0))
    steps = (0.5, 0.5e5, 3.0, 1.0)
    for j, h in enumerate(steps):
        e = jnp.zeros(4).at[j].set(1.0)
        fd = np.asarray(richardson(lambda t: f(v0 + t * e), 0.0, h))
        np.testing.assert_allclose(J[:, j], fd, rtol=GRAD_RTOL, err_msg=f"input {j}")


@pytest.mark.slow
@pytest.mark.release  # ~10 min on the runner; that the target is met is an answer
def test_wabt_target(diesel):
    char, cuts, feed, unit = diesel
    u = Hydrotreater(char, feed, unit.params, target=TargetSpec("wabt", 630.0))
    r = u.solve(feed)
    assert float(r.outputs["wabt"]) == pytest.approx(630.0, abs=1e-6)


@pytest.mark.slow
def test_product_blends_into_ulsd_pool(diesel_result):
    from difflow_refinery.blending import BlendComponent, BlendPool
    r = diesel_result
    # flash point is not computed by the unit: a measured value is the override
    comp = BlendComponent.from_stream("hdt_diesel", r.product_stream(), r.product_char, flash_C=60.0)
    assert float(comp.properties["S_ppm"]) == pytest.approx(float(r.outputs["product.S_wppm"]), rel=1e-9)
    assert float(comp.properties["SG"]) == pytest.approx(float(r.outputs["product.sg"]), rel=1e-12)
    pool = BlendPool("ulsd")
    res = pool([comp], [1.0])
    assert float(res.properties["S_ppm"]) == pytest.approx(float(r.outputs["product.S_wppm"]), rel=1e-9)
    # the pool computes the cetane index from the composition, as the unit does
    assert float(res.properties["cetane_index"]) == pytest.approx(float(r.outputs["product.cetane_index"]),
                                                                  rel=1e-6)


@pytest.mark.slow
@pytest.mark.release
def test_hdt_block_delta_vectors(diesel):
    from difflow.planning import check_delta_vectors
    from difflow_refinery.hydrotreating.planning import hdt_block
    char, cuts, feed, unit = diesel
    blk = hdt_block(unit, feed, ["reactor.T_in", "h2_oil"], ["product.S_wppm", "h2.chemical", "wabt"],
                    feed_product="diesel")
    chk = check_delta_vectors(blk, rtol=1e-3)
    assert chk["passed"], chk["max_rel_error"]


@pytest.mark.slow
def test_crude_unit_diesel_feeds_the_hydrotreater():
    """A real CDU diesel product (F_<char.names> plus stripping water) goes in as it is."""
    from difflow_refinery import column as cc
    from difflow_refinery.thermo import ColumnThermo
    from .test_column import LIGHT, _atmospheric_params

    assay = dr.Assay(TBP_PCT, [t + 273.15 for t in TBP_A], sg=0.86, light_ends=LIGHT, heavy_end=dr.HeavyEnd(),
                     sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(assay, composition=True)
    th = ColumnThermo.from_characterization(char)
    bpd = 95_000.0
    kg_s = bpd * cc.BARREL / 86400.0 * float(char.bulk_sg) * 999.016
    sized = char.stream(kg_s, T=600.0, P=1.9e5, basis="mass")
    base = _atmospheric_params(sized, th, pa1_duty=15e6)
    params = _atmospheric_params(sized, th, pa1_duty=15e6, specs=base.specs + (cc.overflash(0.05),))
    cdu = dr.CrudeDistillationUnit(dr.CrudeDistillationUnitParams(assay=assay, column=params))
    outs = dict(zip(cdu.outlet_names, cdu(cdu.feed(bpd, T=273.15 + 240.0, P=6e5))))
    diesel = outs["diesel"]
    unit = Hydrotreater(char, diesel, HydrotreaterParams())
    assert unit.dropped_mass_fraction < 1e-6
    r = unit.solve(diesel)
    assert bool(r.converged)
    for k, v in r.balances.items():
        assert float(v) < 1e-8, k
    assert float(r.outputs["product.S_wppm"]) < 0.1 * float(r.outputs["feed.S_wppm"])


def test_volume_increments_pinned_to_model_compounds():
    """Liquid molar volumes at 60 F (DIPPR 105, Perry's 8th ed., as tabulated in `chemicals`), cm3/mol."""
    from difflow_refinery.hydrotreating.unit import VOLUME_INCREMENTS as V
    mono = np.mean([107.68 - 88.52, 126.86 - 105.60, 141.72 - 121.80])     # benzene, toluene, ethylbenzene
    assert -1e6 * V["A_mono"] == pytest.approx(mono, abs=0.05)
    assert 1e6 * (V["A_mono"] - V["A_di"]) == pytest.approx(135.77 - 124.79, abs=0.05)   # naphthalene -> tetralin
    assert 1e6 * (V["A_di"] - V["A_poly"]) == pytest.approx(1e6 * (V["A_mono"] - V["A_di"]))
    olef = np.mean([129.70 - 124.12, 193.62 - 188.19])                     # hexene, decene
    assert -1e6 * V["olefins"] == pytest.approx(olef, abs=0.05)


# =============================================================================
# Products: dissolved gases (#333), yields and charge heater (#332), fractionator (#328)
# =============================================================================


def test_fractionator_shares_telescope_to_one():
    from difflow_refinery.hydrotreating.fractionator import split_shares
    Tb = jnp.linspace(300.0, 700.0, 41)
    s = split_shares(Tb, jnp.asarray([420.0, 520.0, 600.0]), 8.0)
    assert s.shape == (41, 4)
    np.testing.assert_allclose(np.asarray(s.sum(axis=1)), 1.0, rtol=0, atol=1e-15)
    assert np.all(np.asarray(s) >= 0.0)
    assert float(s[0, 0]) > 0.9999 and float(s[-1, -1]) > 0.9999


@pytest.mark.slow
def test_downstream_mass_balance_closes_with_dissolved_gases(diesel_result, diesel):
    """#333: the liquid outlets as streams carry everything the unit sends out in them."""
    r = diesel_result
    lay = diesel[3].layout
    for name in ("product", "wild_naphtha"):
        assert float(r.stream_mass(r.product_stream(name))) == pytest.approx(
            float(r.streams[name].mass(lay)), rel=1e-13)
    # the wild naphtha carries H2, H2S, C1, C2 off the blend grid; leaving them out is a flag
    wn = r.product_stream("wild_naphtha")
    assert "F_hydrogen" in wn and "F_hydrogen_sulfide" in wn and "F_methane" in wn
    blend = r.product_stream("wild_naphtha", gases=False)
    assert not any(k[2:] in ("hydrogen", "hydrogen_sulfide", "methane") for k in blend)
    dropped = float(r.stream_mass(wn) - r.stream_mass(blend))
    assert dropped > 0.0
    assert dropped == pytest.approx(float(r.outputs["naphtha.dissolved_gas_rate"]), rel=1e-12)
    assert "F_hydrogen" not in r.product_stream("product")      # the stripper bottoms has none
    # the whole unit: liquid outlets as streams, the gas and water outlets as Flows
    m_in = sum(float(r.streams[k].mass(lay)) for k in ("feed", "makeup", "steam"))
    m_out = (float(r.stream_mass(r.product_stream("product")) + r.stream_mass(wn))
             + sum(float(r.streams[k].mass(lay)) for k in ("off_gas", "purge", "acid_gas", "hps_water",
                                                           "sour_water")))
    assert abs(m_out - m_in) / m_in < 1e-13


@pytest.mark.slow
def test_yields_add_up(diesel_result):
    """#332: product + wild naphtha + gas = feed (less its H2 and water) + chemical hydrogen."""
    o = diesel_result.outputs
    f = float(o["feed.rate"])
    rhs = 1.0 - float(o["feed.h2_water_rate"]) / f + float(o["h2.consumed_by_balance"]) * 2.01588e-3 / f
    assert float(o["yields.total"]) == pytest.approx(rhs, rel=1e-9)
    assert 0.0 < float(o["gas.yield"]) < 0.05


@pytest.mark.slow
def test_charge_heater_duty(diesel_result, diesel):
    """#332: the heater takes the oil from its feed T and the gas from the compressor to bed 1's inlet."""
    o = diesel_result.outputs
    p = diesel[3].params
    assert float(o["heater.fired_duty"]) == pytest.approx(float(o["heater.duty"]) / p.heater_efficiency,
                                                          rel=1e-14)
    # order of magnitude: 50 kg/s of diesel from 25 C to 340 C at 2.5-3 kJ/kg/K, some of it vaporised
    sensible = 50.0 * 2.6e3 * (p.T_in[0] - 298.15)
    assert 0.7 * sensible < float(o["heater.duty"]) < 1.6 * sensible
    assert float(o["feed_effluent.duty"]) == 0.0


@pytest.mark.slow
def test_fractionator_jet_and_ulsd_pools(diesel_result):
    """#328: named products, mass closure to round-off, and blend pools on the outputs."""
    from difflow_refinery.blending import BlendComponent, BlendPool
    r = diesel_result
    fr = r.fractionate(cut_points=(240.0 + 273.15,), products=("jet", "diesel"))
    assert set(fr.products) == {"jet", "diesel"}
    assert float(fr.balance) < 1e-13
    assert float(fr.rates["off_gas"]) == 0.0          # the stripper bottoms has no gas
    jet = BlendPool("jet")([BlendComponent.from_stream("jet", fr.products["jet"], fr.char, flash_C=42.0,
                                                      freeze_C=-47.0, smoke_mm=22.0)], [1.0])
    ulsd = BlendPool("ulsd")([BlendComponent.from_stream("diesel", fr.products["diesel"], fr.char,
                                                        flash_C=60.0)], [1.0])
    assert any(k.startswith("S_ppm") for k in jet.margins)
    assert any(k.startswith("T90_d86_C") for k in ulsd.margins)
    assert float(jet.properties["T90_d86_C"]) < float(ulsd.properties["T90_d86_C"])
    # sulfur is mass-averaged: the two products average back to the unit's product sulfur
    m_j, m_d = float(fr.rates["jet"]), float(fr.rates["diesel"])
    S = (m_j * float(jet.properties["S_ppm"]) + m_d * float(ulsd.properties["S_ppm"])) / (m_j + m_d)
    assert S == pytest.approx(float(r.outputs["product.S_wppm"]), rel=1e-9)
    # three products off the product and the wild naphtha, whose dissolved gases go to the off-gas
    fr3 = r.fractionate(cut_points=(180.0 + 273.15, 250.0 + 273.15), products=("naphtha", "jet", "diesel"),
                        feeds=("product", "wild_naphtha"))
    assert float(fr3.balance) < 1e-13
    assert float(fr3.rates["off_gas"]) == pytest.approx(float(r.outputs["naphtha.dissolved_gas_rate"]),
                                                        rel=1e-12)
    with pytest.raises(ValueError, match="cut points"):
        r.fractionate(cut_points=(500.0, 450.0), products=("a", "b", "c"))


@pytest.mark.slow
@pytest.mark.release
def test_fractionator_cut_point_gradient_matches_central_differences(diesel_result):
    """#328: d(jet rate, jet S, ULSD T90, ULSD cetane)/d(cut point), AD against Richardson differences."""
    from difflow_refinery.blending import BlendComponent, BlendPool
    r = diesel_result

    def f(T):
        fr = r.fractionate(cut_points=(T,))
        jet = BlendComponent.from_stream("jet", fr.products["jet"], fr.char)
        d = BlendPool("ulsd")([BlendComponent.from_stream("diesel", fr.products["diesel"], fr.char,
                                                          flash_C=60.0)], [1.0]).properties
        return jnp.stack([fr.rates["jet"], jet.properties["S_ppm"], d["T90_d86_C"], d["cetane_index"]])

    T0 = 240.0 + 273.15
    J = np.asarray(jax.jacrev(f)(T0))
    fd = np.asarray(richardson(f, T0, 1.0))
    assert np.all(np.abs(J) > 0)
    np.testing.assert_allclose(J, fd, rtol=GRAD_RTOL)
