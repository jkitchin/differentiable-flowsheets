"""Correctness gates of the dynamic refinery units (difflow_refinery.dynamic); not part of the suite.

Run from the repository root::

    python tests/reference/dynamic_refinery_checks.py [--quick] [--out results.json]

About an hour on a laptop CPU (``--quick``: one cell count each, about 30
minutes). Prints, and writes as JSON, the
numbers the "Dynamic operation" section of docs/unit-operations-refinery.md
quotes:

1. **Steady limit.** From a start far from rest (cells 30 K cold, the loop
   at 80 % of its inventory, the separator half full) the dynamic
   hydrotreater settles to the steady Hydrotreater's answer; the error
   against the cell count; the same for the reformer train; and the rest
   state after a +10 K inlet step against a steady solve at the new inlet.
2. **Balances along a trajectory.** Element accumulation against
   in - out at every saved state of a compressor-trip transient (equal by
   construction, so this checks the code), and the inventory change
   against the trapezoid integral of in - out.
3. **Trajectory gradients.** d(separator-liquid sulfur at 1 h) / d(size of
   a bed-inlet temperature step), reverse mode through the diffrax
   solve, against a central difference; the reformer's d(RON at 1 h) /
   d(first-inlet step) likewise.
4. **Time scales.** The thermal transit time of each bed estimated from
   its heat capacities (sum over cells of ``(W_c c_p + C_hold) / C_flow``)
   against the mean arrival time of the bed outlet's response to an inlet
   step (heater lag set to 1 s for this check); and the reformer's 120-day
   coke cycle on the slow time scale, its coke rate against the steady
   reformer's.

The suite's tests (tests/refinery/test_dynamic.py) only smoke-test the
units and check finite gradients, per the test policy.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import difflow_refinery as dr  # noqa: E402
from difflow_refinery.dynamic import DynamicHydrotreater, DynamicReformerTrain, HoldupParams  # noqa: E402
from difflow_refinery.dynamic.hydroprocessing import holdup_outflow  # noqa: E402
from difflow_refinery.hydroprocessing.layout import Flows  # noqa: E402
from difflow_refinery.hydroprocessing.reactor import stream_enthalpy  # noqa: E402
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut  # noqa: E402
from difflow_refinery.hydrotreating.feed import select_cuts  # noqa: E402

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_C = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]


def test_diesel():
    """The docs' test diesel (230-370 C cut of the docs' crude) and its steady hydrotreater."""
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_C]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, 230 + 273.15, 370 + 273.15))
    unit = Hydrotreater(char, feed, HydrotreaterParams())
    return unit, feed, unit.solve(feed)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-300)))


def perturbed_start(dh: DynamicHydrotreater) -> jnp.ndarray:
    """Cells 30 K cold, loop inventory 80 %, separator holdup 50 %."""
    fs = dh.flowsheet
    fs._update_state_indices()
    y = dh.y0
    for e in fs.units:
        sl = e.state_slice
        if e.name == "reactor":
            n, T = dh.reactor.unpack(y[sl])
            y = y.at[sl].set(dh.reactor.pack(n, T - 30.0))
        elif e.name == "loop":
            y = y.at[sl].multiply(0.8)
        elif e.name == "hps":
            y = y.at[sl].multiply(0.5)
    return y


def hydrotreater_gates(cells, out):
    unit, feed, res = test_diesel()
    ref = None
    out["hdt"] = {"cells": {}}
    for N in cells:
        t0 = time.time()
        dh = DynamicHydrotreater.from_steady(unit, feed, res, n_cells=N)
        ref = dh.steady_reference()
        sim = dh.simulate(8 * 3600.0, y0=perturbed_start(dh), n_save=2, max_steps=400_000)
        ob = dh.observe(sim.trajectory.t[-1], sim.y_final)
        dy = dh.flowsheet.derivatives(jnp.asarray(0.0), sim.y_final, None, dh.args())
        row = {
            "liquid_S_wppm": float(ob["liquid.S_wppm"]), "steady_liquid_S_wppm": float(ref["liquid.S_wppm"]),
            "S_rel_err": rel(ob["liquid.S_wppm"], ref["liquid.S_wppm"]),
            "wabt_err_K": float(ob["wabt"] - ref["wabt"]),
            "bed_T_out_err_K": np.asarray(ob["bed_T_out"] - ref["bed_T_out"]).tolist(),
            "recycle_rel_err": rel(ob["recycle.rate"], ref["recycle.rate"]),
            "makeup_rel_err": rel(ob["h2.makeup"], ref["h2.makeup"]),
            "max_scaled_dydt": float(jnp.max(jnp.abs(dy))), "steps": int(sim.info.n_steps),
            "seconds": time.time() - t0, "n_states": int(dh.y0.size),
        }
        out["hdt"]["cells"][N] = row
        print(f"N={N}: {row}", flush=True)
    return unit, feed, res


def hydrotreater_balances_gradients_timescales(unit, feed, res, out):
    dh = DynamicHydrotreater.from_steady(unit, feed, res, n_cells=10)
    y_rest = dh.settle(4 * 3600.0)
    # --- balances along a compressor trip
    a = dh.args(steps={"compressor": [(600.0, -1.0), (1200.0, 1.0)]}, ramp=30.0)
    lay = dh.layout
    eb = jax.jit(lambda t, y: dh.element_balance(t, y, a))
    inv_el = jax.jit(lambda y: dh.inventory(y).elements(lay))
    worst = 0.0
    drifts = {}
    for n_save in (121, 721):                       # 60 s and 10 s between saved states
        sim = dh.simulate(2 * 3600.0, args=a, y0=y_rest, n_save=n_save, max_steps=400_000)
        flux = []
        for t, y in zip(sim.trajectory.t, sim.trajectory.y):
            b = eb(t, y)
            worst = max(worst, float(jnp.max(jnp.abs(b["accumulation"] - b["in_minus_out"])
                                             / jnp.maximum(jnp.abs(b["in_minus_out"]), 1e-3))))
            flux.append(np.asarray(b["in_minus_out"]))
        flux = np.asarray(flux)
        inv = np.asarray([np.asarray(inv_el(y)) for y in sim.trajectory.y])
        tt = np.asarray(sim.trajectory.t)
        integ = np.concatenate([[np.zeros(4)],
                                np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(tt)[:, None], axis=0)])
        # inventory change against the trapezoid integral of in - out, relative to the largest change
        drifts[float(tt[1] - tt[0])] = (np.max(np.abs((inv - inv[0]) - integ), axis=0)
                                        / np.maximum(np.max(np.abs(inv - inv[0]), axis=0), 1e-300)).tolist()
    out["hdt"]["balances"] = {"pointwise_max_rel": worst,
                              "integrated_rel_of_inventory_change_CHSN_by_spacing_s": drifts}
    print("balances", out["hdt"]["balances"], flush=True)

    # --- off-design rest state: after a +10 K inlet step, against a steady solve at the new inlet
    import dataclasses
    a = dh.args(steps={"T_heater": [(300.0, 10.0)]})
    sim = dh.simulate(8 * 3600.0, args=a, y0=y_rest, n_save=2, max_steps=400_000)
    ob = dh.observe(sim.trajectory.t[-1], sim.y_final, a)
    p2 = dataclasses.replace(unit.params, T_in=(unit.params.T_in[0] + 10.0,))
    u2 = Hydrotreater(unit.char, feed, p2)
    r2 = u2.solve(feed)
    ref2 = DynamicHydrotreater.from_steady(u2, feed, r2, n_cells=10).steady_reference()
    out["hdt"]["off_design"] = {
        "liquid_S_wppm": float(ob["liquid.S_wppm"]), "steady_liquid_S_wppm": float(ref2["liquid.S_wppm"]),
        "S_rel_err": rel(ob["liquid.S_wppm"], ref2["liquid.S_wppm"]), "wabt_err_K": float(ob["wabt"] - ref2["wabt"]),
        "makeup_rel_err": rel(ob["h2.makeup"], ref2["h2.makeup"]),
        "recycle_rel_err": rel(ob["recycle.rate"], ref2["recycle.rate"])}
    print("off design", out["hdt"]["off_design"], flush=True)

    # --- gradient through a trajectory: d(liquid S at 1 h)/d(T step size)
    def S_at_1h(dT):
        a = dh.args(steps={"T_heater": [(300.0, dT)]})
        sim = dh.simulate(3600.0, args=a, y0=y_rest, n_save=2, max_steps=100_000)
        return dh.observe(sim.trajectory.t[-1], sim.y_final, a)["liquid.S_wppm"]

    t0 = time.time()
    g = float(jax.grad(S_at_1h)(jnp.asarray(5.0)))
    t_grad = time.time() - t0
    h = 0.5
    fd = (float(S_at_1h(jnp.asarray(5.0 + h))) - float(S_at_1h(jnp.asarray(5.0 - h)))) / (2 * h)
    out["hdt"]["gradient"] = {"reverse": g, "central_difference": fd, "rel_err": abs(g - fd) / abs(fd),
                              "seconds_reverse": t_grad}
    print("gradient", out["hdt"]["gradient"], flush=True)

    # --- thermal transit time against the step response
    dh1 = DynamicHydrotreater.from_steady(unit, feed, res, n_cells=10,
                                          holdups=HoldupParams(heater_tau=1.0))
    y1 = dh1.settle(4 * 3600.0)
    fs = dh1.flowsheet
    streams = fs.outputs(jnp.asarray(0.0), y1, None, dh1.args())
    us = dh1._split(y1)
    prof = dh1.reactor.profiles(us["reactor"], {"feed": streams["rx_in"], "quench": streams["quench_gas"]})
    n_hold, T = dh1.reactor.unpack(us["reactor"])
    lay, comps = dh1.layout, dh1.comps
    est = []
    for b, geo in enumerate(dh1.reactor.beds):
        km = res.reactor.beds[b].k_model
        Wc = geo.W / dh1.reactor.n_cells
        tot = 0.0
        for c in range(dh1.reactor.n_cells):
            Fc = jax.tree_util.tree_map(lambda x: x[b, c], prof["outflow"])
            nc = jax.tree_util.tree_map(lambda x: x[b, c], n_hold)
            C_flow = jax.jvp(lambda t: stream_enthalpy(Fc, lay, comps, km, t), (T[b, c],), (1.0,))[1]
            C_hold = jax.jvp(lambda t: stream_enthalpy(nc, lay, comps, km, t), (T[b, c],), (1.0,))[1]
            tot += float((Wc * dh1.reactor.cp_cat + C_hold) / C_flow)
        est.append(tot)
    t_step, ramp = 600.0, 60.0
    a = dh1.args(steps={"T_heater": [(t_step, 10.0)]}, ramp=ramp)
    sim = dh1.simulate(4 * 3600.0, args=a, y0=y1, n_save=481, max_steps=400_000)
    tr = dh1.trajectory(sim, a, ["bed_T_out", "bed_T_in"])
    t = np.asarray(tr["t"])
    arrivals = []
    for b in range(len(dh1.reactor.beds)):
        Tin, Tout = np.asarray(tr["bed_T_in"])[:, b], np.asarray(tr["bed_T_out"])[:, b]
        dTin, dTout = np.gradient(Tin, t), np.gradient(Tout, t)
        mean = lambda d: float(np.trapezoid(t * d, t) / np.trapezoid(d, t))
        arrivals.append(mean(dTout) - mean(dTin))
    out["hdt"]["thermal"] = {"transit_estimate_s": est, "mean_arrival_s": arrivals,
                             "outlet_dT_final_K": (np.asarray(tr["bed_T_out"])[-1]
                                                   - np.asarray(tr["bed_T_out"])[0]).tolist()}
    print("thermal", out["hdt"]["thermal"], flush=True)


def reformer_gates(out, quick=False):
    from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
    feed = lean_naphtha()
    ref = CatalyticReformer(ReformerParams())
    res = ref.solve(feed)
    out["ref"] = {"cells": {}}
    for N in (5,) if quick else (3, 5, 10):
        t0 = time.time()
        tr = DynamicReformerTrain.from_steady(ref, feed, res, n_cells=N)
        y = tr.y0.at[:].add(0.0)
        # start 20 K cold in every cell
        fs = tr.flowsheet
        fs._update_state_indices()
        for e in fs.units:
            if e.name.startswith("rx"):
                y = y.at[e.state_slice.start:e.state_slice.start + N].add(-0.2)
        sim = tr.simulate(3 * 3600.0, y0=y, n_save=2, max_steps=200_000)
        ob = tr.observe(sim.trajectory.t[-1], sim.y_final)
        rf = tr.steady_reference()
        row = {"T_out_err_K": np.asarray(ob["T_out"] - rf["T_out"]).tolist(),
               "RON": float(ob["reformate.RON"]), "steady_RON": float(rf["reformate.RON"]),
               "coke_make_rel_err": rel(ob["coke_make"], rf["coke_make"]), "steps": int(sim.info.n_steps),
               "seconds": time.time() - t0}
        out["ref"]["cells"][N] = row
        print(f"reformer N={N}: {row}", flush=True)
    tr = DynamicReformerTrain.from_steady(ref, feed, res, n_cells=5)

    def ron_at_1h(dT):
        a = tr.args(steps={"T_in1": [(300.0, dT)]})
        sim = tr.simulate(3600.0, args=a, n_save=2)
        return tr.observe(sim.trajectory.t[-1], sim.y_final, a)["reformate.RON"]

    t0 = time.time()
    g = float(jax.grad(ron_at_1h)(jnp.asarray(5.0)))
    tg = time.time() - t0
    h = 0.5
    fd = (float(ron_at_1h(jnp.asarray(5.0 + h))) - float(ron_at_1h(jnp.asarray(5.0 - h)))) / (2 * h)
    out["ref"]["gradient"] = {"reverse": g, "central_difference": fd, "rel_err": abs(g - fd) / abs(fd),
                              "seconds_reverse": tg}
    print("reformer gradient", out["ref"]["gradient"], flush=True)

    # --- the slow time scale: 120 days of coking with activity exp(-10 C)
    trc = DynamicReformerTrain.from_steady(ref, feed, res, n_cells=5, coke_alpha=10.0)
    t0 = time.time()
    ts, C, info = trc.cycle(120 * 86400.0, n_save=5)
    o0, o1 = trc.cycle_observe(ts[0], C[0]), trc.cycle_observe(ts[-1], C[-1])
    W = np.asarray([float(r.W) for r in trc.reactors])
    made = np.asarray(C[-1]).mean(axis=1) * W / float(ts[-1])          # kg/s of coke laid down
    dC0, _ = trc._rest_train(jnp.asarray(0.0), jnp.zeros_like(C[0]), trc.args())
    rate0 = (np.asarray(dC0).sum(axis=1) * W / trc.n_cells).tolist()    # fresh catalyst, kg/s
    out["ref"]["cycle"] = {"days": 120, "seconds": time.time() - t0, "steps": int(info.n_steps),
                           "RON_start": float(o0["reformate.RON"]), "RON_end": float(o1["reformate.RON"]),
                           "T_out_start": np.asarray(o0["T_out"]).tolist(), "T_out_end": np.asarray(o1["T_out"]).tolist(),
                           "mean_coke_end": np.asarray(C[-1]).mean(axis=1).tolist(),
                           "coke_rate_mean_kg_s": made.tolist(), "coke_rate_fresh_kg_s": rate0,
                           "steady_coke_rate_kg_s": [float(x["coke"]) for x in res.reactors]}
    print("cycle", out["ref"]["cycle"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="dynamic_refinery_checks.json")
    a = ap.parse_args()
    out = {}
    unit, feed, res = hydrotreater_gates((10,) if a.quick else (5, 10, 20), out)
    hydrotreater_balances_gradients_timescales(unit, feed, res, out)
    reformer_gates(out, quick=a.quick)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
