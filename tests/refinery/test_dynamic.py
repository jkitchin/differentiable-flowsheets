"""Smoke and gradient tests of the dynamic refinery units (difflow_refinery.dynamic).

Per the test policy these check the code, not the physics: the steady-limit,
balance-along-a-trajectory, trajectory-gradient and time-scale gates are in
``tests/reference/dynamic_refinery_checks.py`` (about 40 min). Here: one bed
of three cells on the docs' diesel layout and one three-cell reforming bed,
no steady solve and no long integration.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.dynamic import (
    BedGeometry, DynamicReformingReactor, DynamicTrickleBedReactor, QuasiSteadyHoldup, evaluate,
    holdup_outflow, signal_args, smoothstep)
from difflow_refinery.dynamic.hydroprocessing import (
    DynamicRestStateWarning, choose_n_sub, flows_scale, holdup_from_outflow, rest_mismatch, warn_rest_state)
from difflow_refinery.hydroprocessing.layout import Flows
from difflow_refinery.hydroprocessing.reactor import k_model_at
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut
from difflow_refinery.hydrotreating.feed import select_cuts

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]


def test_signals_step_and_ramp():
    a = signal_args({"T": 600.0, "s": 1.0}, {"T": [(100.0, 10.0), (300.0, -4.0)]}, ramp=20.0)
    assert float(evaluate(0.0, a)["T"]) == 600.0
    assert float(evaluate(110.0, a)["T"]) == pytest.approx(605.0)
    assert float(evaluate(1e4, a)["T"]) == pytest.approx(606.0)
    assert float(evaluate(1e4, a)["s"]) == 1.0
    assert float(smoothstep(jnp.asarray(-1.0))) == 0.0 and float(smoothstep(jnp.asarray(2.0))) == 1.0
    g = jax.grad(lambda d: evaluate(110.0, signal_args({"T": 600.0}, {"T": [(100.0, d)]}, 20.0))["T"])(10.0)
    assert float(g) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no base"):
        signal_args({"T": 1.0}, {"x": [(0.0, 1.0)]})


@pytest.fixture(scope="module")
def bed():
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, 230 + 273.15, 370 + 273.15))
    unit = Hydrotreater(char, feed, HydrotreaterParams())
    th = unit.theta(feed)
    comps, kin, lay = unit._comps(th), unit._kin_params(th["kin"]), unit.layout
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(700.0),
                jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    inlet = th["oil"] + gas
    T_in = 613.15
    km, _ = k_model_at(inlet, lay, comps, T_in, 50e5)
    geo = BedGeometry(W=60000.0, tau_G=5.0, tau_L=25.0)
    n_cells = 3
    h0 = holdup_from_outflow(inlet, lay, comps, km, T_in, geo.tau_G, geo.tau_L)
    cells = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_cells)[None], h0)
    T = jnp.full((1, n_cells), T_in)
    sc = flows_scale(h0)
    s1 = jnp.concatenate([sc.ravel(), jnp.asarray([100.0])])
    scale = jnp.tile(s1, n_cells)
    rx = DynamicTrickleBedReactor(lay, unit.kinetics, kin, comps, [geo], 50e5, n_cells=n_cells, scale=scale)
    y0 = rx.pack(cells, T)
    return {"rx": rx, "y0": y0, "inlet": inlet.to_stream(lay, T_in, 50e5), "lay": lay, "comps": comps,
            "km": km, "geo": geo, "flows": inlet, "T_in": T_in}


def test_holdup_split_inverts(bed):
    lay, comps, km, geo = bed["lay"], bed["comps"], bed["km"], bed["geo"]
    F = bed["flows"]
    n = holdup_from_outflow(F, lay, comps, km, 640.0, geo.tau_G, geo.tau_L)
    back = holdup_outflow(n, lay, comps, km, 640.0, 1.0 / geo.tau_G, 1.0 / geo.tau_L)
    np.testing.assert_allclose(np.asarray(back.ravel()), np.asarray(F.ravel()), rtol=1e-9, atol=1e-12)


def test_trickle_bed_smoke_and_balance(bed):
    rx, y0, inp = bed["rx"], bed["y0"], {"feed": bed["inlet"]}
    f = jax.jit(lambda y: rx.derivatives(0.0, y, inp))
    outlet = jax.jit(lambda y: rx.outputs(0.0, y, inp)["outlet"])
    dy = f(y0)
    assert dy.shape == y0.shape and bool(jnp.all(jnp.isfinite(dy)))
    out = Flows.from_stream(outlet(y0), bed["lay"])
    # accumulation of every element = in - out, to round-off (the RK4 segments conserve elements)
    lay = bed["lay"]
    held = lambda y: jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=(0, 1)), rx.unpack(y)[0]).elements(lay)
    acc = jax.jvp(held, (y0,), (dy,))[1]
    flux = bed["flows"].elements(lay) - out.elements(lay)
    through = float(jnp.max(jnp.abs(bed["flows"].elements(lay))))
    np.testing.assert_allclose(np.asarray(acc), np.asarray(flux), rtol=0, atol=1e-12 * through)
    # and a perturbed state (a real transient): the identity still holds
    y1 = y0 * (1.0 + 0.05 * jnp.sin(jnp.arange(y0.size)))
    dy1 = f(y1)
    out1 = Flows.from_stream(outlet(y1), lay)
    acc1 = jax.jvp(held, (y1,), (dy1,))[1]
    flux1 = bed["flows"].elements(lay) - out1.elements(lay)
    assert float(jnp.max(jnp.abs(flux1))) > 1e-6 * through
    np.testing.assert_allclose(np.asarray(acc1), np.asarray(flux1), rtol=0, atol=1e-12 * through)


def test_trickle_bed_gradient_finite(bed):
    rx, y0 = bed["rx"], bed["y0"]
    lay = bed["lay"]

    def g(T_in):
        s = dict(bed["inlet"])
        s["T"] = T_in
        return jnp.sum(rx.derivatives(0.0, y0, {"feed": s})[-1::rx.n_per_cell])

    d = jax.jit(jax.grad(g))(jnp.asarray(bed["T_in"]))
    assert np.isfinite(float(d))
    del lay


def test_high_conversion_bed_needs_more_rk4_steps(bed):
    # Five times the catalyst in three cells: one RK4 step per cell is unstable
    # for the fastest sulfur class, and the marched rest state is garbage.
    rx = DynamicTrickleBedReactor(bed["rx"].layout, bed["rx"].kinetics, bed["rx"].kin_params, bed["comps"],
                                  [BedGeometry(W=300000.0, tau_G=5.0, tau_L=25.0)], 50e5, n_cells=3,
                                  scale=bed["rx"].scale)
    F, T = rx.rest_profile(0, bed["flows"], bed["T_in"], n_sub=32)
    ref = (bed["flows"], bed["T_in"], jax.tree_util.tree_map(lambda a: a[-1], F), T[-1])
    coarse = rest_mismatch(rx, 0, *ref, n_sub=1)
    assert not np.isfinite(coarse["S_wppm"]) or coarse["rel_S"] > 1.0
    with pytest.warns(DynamicRestStateWarning, match="bed 1"):
        assert warn_rest_state([coarse], "test bed", 3, 1)
    n, ms = choose_n_sub(rx, [ref])
    assert 1 < n < 32
    assert ms[0]["rel_S"] < 0.02 and abs(ms[0]["dT"]) < 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", DynamicRestStateWarning)
        assert not warn_rest_state(ms, "test bed", 3, n)


def test_feed_sulfur_scaling_is_differentiable(bed):
    # the feed_sulfur signal scales only the sulfur attributes, and has a derivative
    from difflow_refinery.dynamic.hydrotreater import scale_feed_sulfur
    lay, F = bed["lay"], bed["flows"]
    S = lambda x: scale_feed_sulfur(F, lay, x).elements(lay)
    np.testing.assert_allclose(np.asarray(S(1.0)), np.asarray(F.elements(lay)), rtol=1e-14)
    _, dS = jax.jvp(S, (jnp.asarray(1.3),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(float(dS[2]), float(F.elements(lay)[2]), rtol=1e-12)     # d(S)/dx = S
    g = jax.grad(lambda x: S(x)[2])(jnp.asarray(0.7))
    assert np.isfinite(float(g))


@pytest.fixture(scope="module")
def reformer_bed():
    from difflow_refinery.reforming import ReformingKinetics, lean_naphtha
    from difflow_refinery.reforming.reactor import flows_of, stream_of
    from difflow_refinery.reforming import species as sp
    feed = lean_naphtha()
    F = flows_of(feed.stream())
    F = F.at[sp.INDEX["H2"]].add(5.0 * float(feed.hydrocarbon_moles))
    inlet = stream_of(F, jnp.asarray(773.15), jnp.asarray(15e5))
    rx = DynamicReformingReactor(ReformingKinetics(), 3000.0, 15e5, n_cells=3, coke_alpha=10.0,
                                 T0=[740.0, 730.0, 725.0])
    return rx, inlet


def test_reformer_cells_smoke_and_rest(reformer_bed):
    rx, inlet = reformer_bed
    y0 = rx.initial_state()
    f = jax.jit(lambda y: rx.derivatives(0.0, y, {"inlet": inlet}))
    run = jax.jit(lambda y: rx._run(inlet, y))
    dy = f(y0)
    assert bool(jnp.all(jnp.isfinite(dy)))
    assert bool(jnp.all(dy[rx.n_cells:] >= 0.0))                 # coke only accumulates
    # at T = T* (each cell at its own segment's adiabatic outlet) the temperatures are at rest
    _, Ts, _, _ = run(y0)
    for _ in range(6):                                             # Ts depends on the upstream cell's T
        y = jnp.concatenate([Ts / 100.0, y0[rx.n_cells:]])
        _, Ts, _, _ = run(y)
    dT = f(y)[:rx.n_cells]
    assert float(jnp.max(jnp.abs(dT))) < 1e-8


def test_reformer_cells_gradient_finite(reformer_bed):
    rx, inlet = reformer_bed

    def g(T_in):
        s = dict(inlet)
        s["T"] = T_in
        return jnp.sum(rx.derivatives(0.0, rx.initial_state(), {"inlet": s}))

    assert np.isfinite(float(jax.jit(jax.grad(g))(jnp.asarray(773.15))))


def test_quasi_steady_holdup_conserves_and_rests():
    def split(s):
        a = {k: 0.3 * v for k, v in s.items() if k.startswith("F_")}
        b = {k: 0.7 * v for k, v in s.items() if k.startswith("F_")}
        return dict(a, T=s["T"] - 5.0, P=s["P"]), dict(b, T=s["T"], P=s["P"])

    inlet = {"F_A": jnp.asarray(2.0), "F_B": jnp.asarray(1.0), "T": jnp.asarray(400.0), "P": jnp.asarray(1e5)}
    u = QuasiSteadyHoldup.at_steady(split, {"in": inlet}, ["in"], ["top", "bottom"], ["F_A", "F_B"],
                                    {"top": 60.0, "bottom": 300.0})
    y0 = u.initial_state()
    assert float(jnp.max(jnp.abs(u.derivatives(0.0, y0, {"in": inlet})))) == 0.0
    step = dict(inlet, F_A=jnp.asarray(3.0))
    dy = u.derivatives(0.0, y0, {"in": step}) * u._scale
    held = lambda y: sum(v["F_A"] for v in u.inventory(y).values())
    acc = jax.jvp(held, (y0,), (dy / u._scale,))[1]
    out = u.outputs(0.0, y0, {"in": step})
    assert float(acc) == pytest.approx(3.0 - float(out["top"]["F_A"] + out["bottom"]["F_A"]), rel=1e-12)


# ----- phase 2: equilibrium beds, SMR tubes, amine loop, SRU control, crude unit -----------

def _shift_bed(n_cells=3):
    from difflow_refinery.dynamic import DynamicEquilibriumBed
    from difflow_refinery.dynamic.hydrogen import _shift_equilibrium
    from difflow_refinery.hydrogen import smr
    n = jnp.asarray([2.0, 30.0, 45.0, 8.0, 6.0, 0.5, 0.0])
    inlet = smr.to_stream(n, 593.15, 2.8e6)
    bed = DynamicEquilibriumBed(_shift_equilibrium(0.0), smr.enthalpy, smr.to_vec, smr.to_stream, 4.0e7,
                                n_cells, P=2.8e6, T0=[650.0, 640.0, 630.0][:n_cells], name="hts")
    return bed, inlet, n


def test_equilibrium_bed_rests_at_the_adiabatic_shift_and_conserves():
    from difflow_refinery.hydrogen import smr
    from difflow_refinery import thermochemistry as tc
    bed, inlet, n = _shift_bed()
    out, _ = smr.shift(inlet, 593.15)
    T_ad = float(out["T"])
    y = jnp.full(bed.n_cells, T_ad / 100.0)
    dy = jax.jit(lambda y: bed.derivatives(0.0, y, {"inlet": inlet}))(y)
    assert float(jnp.max(jnp.abs(dy))) < 1e-9                 # every cell at the adiabatic equilibrium
    outlet = jax.jit(lambda y: smr.to_vec(bed.outputs(0.0, y, {"inlet": inlet})["outlet"]))
    np.testing.assert_allclose(np.asarray(outlet(y)), np.asarray(smr.to_vec(out)), rtol=1e-8, atol=1e-10)
    # away from rest the cells hold no gas: elements in = out at every state
    y1 = jnp.asarray([6.2, 6.4, 6.0])
    o = outlet(y1)
    E = jnp.asarray([[tc.elements(k).get(e, 0) for k in smr.SYNGAS] for e in "CHO"])
    np.testing.assert_allclose(np.asarray(E @ o), np.asarray(E @ n), rtol=1e-12)
    g = jax.jit(jax.grad(lambda T: jnp.sum(bed.derivatives(0.0, y1, {"inlet": dict(inlet, T=T)}))))(593.15)
    assert np.isfinite(float(g))


def test_reformer_tube_cells_smoke_and_rest():
    from difflow_refinery.dynamic.hydrogen import DynamicReformerTube
    from difflow_refinery.hydrogen import smr
    tp = smr.ReformerTubeParams(rtol=1e-7, atol=1e-9)
    n = jnp.asarray([1.6, 4.4, 0.3, 0.02, 0.1, 0.03, 0.0])      # per tube, after a prereformer
    inlet = smr.to_stream(n * 10.0, 893.15, 3.2e6)                # ten tubes
    tube = DynamicReformerTube(tp, 10.0, 2, 3.0e5, 3.2e6, [1000.0, 1150.0])
    firing = {"Q": jnp.asarray(10 * 3.5e5)}
    inp = {"inlet": inlet, "firing": firing}
    y0 = tube.initial_state()
    dy = jax.jit(lambda y: tube.derivatives(0.0, y, inp))(y0)
    assert bool(jnp.all(jnp.isfinite(dy)))
    # at T = T* the cells are at rest (iterate: each cell's T* depends on the cell upstream)
    y = y0
    t_star = jax.jit(lambda y: tube.report(y, inp)["T_star"])
    for _ in range(4):
        y = t_star(y) / 100.0
    assert float(jnp.max(jnp.abs(jax.jit(lambda y: tube.derivatives(0.0, y, inp))(y)))) < 1e-9
    g = jax.jit(jax.grad(lambda Q: jnp.sum(tube.derivatives(0.0, y0, {"inlet": inlet, "firing": {"Q": Q}}))))(3.5e6)
    assert np.isfinite(float(g))


def test_amine_loop_species_balance_and_gradient():
    from difflow_refinery.dynamic import AmineHoldupParams, DynamicAmineUnit
    from difflow_refinery.treating import TrayParams
    from difflow_refinery.treating.unit import AmineUnitParams, Contactor
    p = AmineUnitParams("MDEA", 0.45, (Contactor("fuel", 60.0, 4, TrayParams(area=200.0, k_G=1e-5, k_L=1e-4)),),
                        Q_reboiler=3e6, n_regenerator=4)
    gas = {"F_methane": 350.0, "F_hydrogen": 50.0, "F_hydrogen_sulfide": 10.0, "F_carbon_dioxide": 10.0,
           "F_water": 0.5, "T": 313.15, "P": 7e5}
    hp = AmineHoldupParams()
    L = 60.0
    x0 = jnp.asarray([L * hp.tau_lean * 1e-4, L * hp.tau_lean * 1e-2, L * hp.tau_rich * 0.01,
                      L * hp.tau_rich * 0.15, L * hp.tau_rich * 400.0 / 60.0, 393.0])
    u = DynamicAmineUnit(p, hp, jnp.ones(6), x0)
    sig_ = {"circulation": jnp.asarray(1.0), "Q_reboiler": jnp.asarray(3e6)}
    inp = {"fuel": gas, "signals": sig_}
    y0 = u.initial_state()
    dy = jax.jit(lambda y: u.derivatives(0.0, y, inp))(y0)
    assert bool(jnp.all(jnp.isfinite(dy)))
    out = jax.jit(lambda y: u.outputs(0.0, y, inp))(y0)
    acc = jax.jvp(lambda yy: jnp.stack([u.inventory(yy)[k] for k in ("hydrogen_sulfide", "carbon_dioxide",
                                                                       "water")]), (y0,), (dy,))[1]
    for i, k in enumerate(("hydrogen_sulfide", "carbon_dioxide", "water")):
        fk = f"F_{k}"
        io = gas.get(fk, 0.0) + out["makeup"].get(fk, 0.0) - sum(out[s].get(fk, 0.0) for s in
                                                                  ("fuel_sweet", "flash_gas", "acid_gas"))
        assert float(acc[i]) == pytest.approx(float(io), abs=1e-9 * 400.0)
    g = jax.jit(jax.grad(lambda c: u.derivatives(0.0, y0, {"fuel": gas, "signals": dict(sig_, circulation=c)})[1]))(1.0)
    assert np.isfinite(float(g))


def test_air_demand_control_trims_toward_the_ratio():
    from difflow_refinery.dynamic import AirDemandControl
    from difflow_refinery.sulfur import claus as cl
    c = AirDemandControl(Ki=1e-3, S_in=100.0)
    tail = cl.to_stream(jnp.zeros(len(cl.SPECIES)).at[cl._IDX["hydrogen_sulfide"]].set(2.0)
                        .at[cl._IDX["sulfur_dioxide"]].set(0.5), 413.15, 1.3e5)
    du = c.derivatives(0.0, c.initial_state(), {"tail_gas": tail, "signals": {"air": 200.0}})
    assert float(du[0]) == pytest.approx(1e-3 * 1.0 / 100.0)      # H2S-rich tail gas: more air
    assert float(c.outputs(0.0, jnp.asarray([0.01]), {"signals": {"air": 200.0}})["air"]["air"]) == pytest.approx(202.0)


def test_crude_unit_starts_at_rest_and_conserves():
    from difflow_refinery import Assay, CrudeUnit, column as cc
    from difflow_refinery.dynamic import DynamicCrudeUnit
    assay = Assay([0, 30, 70, 100], [320., 480., 640., 900.], sg=0.85)
    unit = CrudeUnit(assay, cc.CrudeColumnParams(n_stages=8, feed_stage=7, bottom_steam=0.2,
                                                 specs=(cc.product_rate("naphtha", 0.25, basis="mole"),
                                                        cc.coil_outlet_temperature(620.0))))
    feed = unit.feed(1.0, T=500.0, P=4e5, basis="mole")
    du = DynamicCrudeUnit.from_steady(unit.column, feed, holdup=dr.dynamic.CDUHoldupParams(tau_side=60.0))
    a = du.args(steps={"COT": [(0.0, 5.0)], "feed_rate": [(0.0, 0.1)]}, ramp=1.0)
    rhs = jax.jit(lambda y, a: du.flowsheet.derivatives(jnp.asarray(10.0), y, None, a))
    assert float(jnp.max(jnp.abs(rhs(du.y0, du.args())))) < 1e-8
    dy = rhs(du.y0, a)
    assert bool(jnp.all(jnp.isfinite(dy)))
    b = du.balance(10.0, du.y0, a)
    np.testing.assert_allclose(np.asarray(b["accumulation"]), np.asarray(b["in_minus_out"]), atol=1e-9)


@pytest.fixture(scope="module")
def quench_beds(bed):
    """Two three-cell beds behind an integral quench controller (the hydrocracker's), on the diesel layout."""
    from difflow_refinery.dynamic import QuenchControlledReactor
    lay, comps, km, geo = bed["lay"], bed["comps"], bed["km"], bed["geo"]
    rx1 = bed["rx"]
    h0 = holdup_from_outflow(bed["flows"], lay, comps, km, bed["T_in"], geo.tau_G, geo.tau_L)
    cells = jax.tree_util.tree_map(lambda x: jnp.stack([jnp.stack([x] * 3)] * 2), h0)
    T = jnp.full((2, 3), bed["T_in"])
    scale = jnp.tile(rx1.scale, 2)
    rx = QuenchControlledReactor(lay, rx1.kinetics, rx1.kin_params, comps, [geo, geo], 50e5,
                                 quench_shares=(1.0,), n_cells=3, scale=scale, q0=[0.1], gain=[100.0],
                                 tau_c=60.0, q_max=[0.3], setpoint="T_set")
    y0 = jnp.concatenate([jax.vmap(jax.vmap(lambda f: f.ravel()))(cells).reshape(2, 3, -1),
                          T[..., None]], axis=-1).reshape(-1) / scale
    y0 = jnp.concatenate([y0, jnp.asarray([1.0])])                  # valve at its design opening
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(700.0),
                jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    header = gas.to_stream(lay, 340.0, 50e5)
    return rx, y0, {"feed": bed["inlet"], "quench": header, "signals": {"T_set": jnp.asarray(bed["T_in"])}}


def test_quench_controller_smoke_balance_and_windup(quench_beds, bed):
    rx, y0, inp = quench_beds
    lay = bed["lay"]
    f = jax.jit(lambda y, i: rx.derivatives(0.0, y, i))       # one compile: every call below has inp's structure
    dy = f(y0, inp)
    assert dy.shape == y0.shape and bool(jnp.all(jnp.isfinite(dy)))
    out = jax.jit(lambda y: rx.outputs(0.0, y, inp))(y0)
    draw = Flows.from_stream(out["quench_draw"], lay)
    header = Flows.from_stream(inp["quench"], lay)
    np.testing.assert_allclose(np.asarray(draw.gas), 0.1 * np.asarray(header.gas), rtol=1e-12)
    assert set(rx.partial_outputs(0.0, y0, {"quench": inp["quench"]})) == {"quench_draw"}
    # the beds' material balance with the drawn quench as an inflow
    held = lambda y: jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=(0, 1)), rx.unpack(y)[0]).elements(lay)
    acc = jax.jvp(held, (y0,), (dy,))[1]
    flux = (bed["flows"] + draw).elements(lay) - Flows.from_stream(out["outlet"], lay).elements(lay)
    through = float(jnp.max(jnp.abs(bed["flows"].elements(lay))))
    np.testing.assert_allclose(np.asarray(acc), np.asarray(flux), rtol=0, atol=1e-11 * through)
    # a valve wound past its stop is pulled back to it (back-calculation anti-windup) and reads shut
    hot = dict(inp, signals={"T_set": jnp.asarray(bed["T_in"] + 200.0)})       # inlet far below setpoint
    y_past = y0.at[-1].set(-2.0)                                               # integrator below shut
    assert float(rx.valves(y_past)[0]) < 1e-12
    assert float(f(y0, hot)[-1]) < 0.0                                        # an open valve closes
    assert float(f(y_past, inp)[-1]) > 0.0                                    # at the setpoint: back to the stop
    np.testing.assert_allclose(float(rx.valves(y0)[0]), 0.1, rtol=1e-12)        # inside its range: the state


def test_quench_controller_gradient_finite(quench_beds):
    rx, y0, inp = quench_beds
    g = jax.jit(jax.grad(lambda Ts: rx.derivatives(0.0, y0, dict(inp, signals={"T_set": Ts}))[-1]))(
        jnp.asarray(inp["signals"]["T_set"]))
    assert np.isfinite(float(g)) and float(g) < 0.0


def test_reformer_loop_balance_and_spec():
    from difflow.units.eos_units import Compressor, CompressorParams
    from difflow_refinery.dynamic import DynamicReformerLoop
    from difflow_refinery.reforming import lean_naphtha
    from difflow_refinery.reforming import species as sp
    from difflow_refinery.reforming import thermo as rth
    from difflow_refinery.reforming.reactor import flows_of, stream_of
    from difflow_refinery.reforming.separation import ProductSeparator, ProductSeparatorParams
    feed = lean_naphtha()
    F = flows_of(feed.stream()).at[sp.INDEX["H2"]].add(6.0 * float(feed.hydrocarbon_moles))
    eff = stream_of(F, jnp.asarray(760.0), jnp.asarray(14e5))
    sep = ProductSeparator(ProductSeparatorParams(T=311.15, P=12e5))
    comp = Compressor(CompressorParams(P_out=14e5, eta_isentropic=0.75, T_bounds=(200.0, 800.0)), rth.cubic_thermo())
    liq, vap, _ = sep(eff)
    Fv, Fl = flows_of(vap), flows_of(liq)
    N = 12e5 * 30.0 / (8.314462618 * 311.15)
    loop = DynamicReformerLoop(sep, comp, H2_HC=4.0, hydrocarbon_feed=float(feed.hydrocarbon_moles), tau_L=300.0,
                               N_design=N, net_design=0.3 * float(jnp.sum(Fv)), tau_p=30.0,
                               R_max=2.0 * float(jnp.sum(Fv)), y0=jnp.concatenate([Fl * 300.0, Fv / jnp.sum(Fv) * N]),
                               T_sep=311.15, P_sep=12e5)
    y0 = loop.initial_state()
    sig_ = {"compressor": jnp.asarray(1.0), "feed_rate": jnp.asarray(1.0)}
    inp = {"effluent": eff, "signals": sig_}
    dy = jax.jit(lambda y: loop.derivatives(0.0, y, inp))(y0)
    assert bool(jnp.all(jnp.isfinite(dy)))
    outputs = jax.jit(lambda i: loop.outputs(0.0, y0, i))      # one compile for the running and tripped loop
    out = outputs(inp)
    rec = flows_of(out["recycle"])
    assert float(rec[sp.INDEX["H2"]]) == pytest.approx(4.0 * float(feed.hydrocarbon_moles), rel=1e-10)
    E = jnp.asarray(sp.ELEMENTS)
    held = lambda y: E @ sum(loop.holdups(y))
    acc = jax.jvp(held, (y0,), (dy,))[1]
    flux = E @ (F - rec - flows_of(out["net_gas"]) - flows_of(out["liquid"]))
    np.testing.assert_allclose(np.asarray(acc), np.asarray(flux), rtol=1e-10, atol=1e-9)
    tripped = outputs(dict(inp, signals=dict(sig_, compressor=jnp.asarray(0.0))))
    assert float(jnp.sum(flows_of(tripped["recycle"]))) == 0.0
