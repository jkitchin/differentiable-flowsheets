"""Correctness gates of the dynamic hydrocracker and the reformer's dynamic recycle loop; not part of the suite.

Run from the repository root::

    python tests/reference/dynamic_hydrocracker_checks.py [--quick] [--out results.json]

About an hour and a half on a laptop CPU (``--quick``: no 5 x 2 cell case
and no off-design rest state, about an hour). Prints, and writes as JSON,
the numbers the "Dynamic operation" section of
docs/unit-operations-refinery.md quotes for these units (the hydrotreater
and reformer-train gates are ``dynamic_refinery_checks.py``):

1. **Steady limit (hydrocracker).** From the steady profile mapped onto
   the cells (close to, not at, the dynamic model's rest) the dynamic
   hydrocracker settles to the steady Hydrocracker's answer in two hours:
   the error against cells per bed and RK4 steps per cell, once-through and
   with a 50 % UCO recycle, whose tear the fractionator's UCO holdup
   carries as ODE states. (A start far from rest -- every cell 20 K cold,
   as the hydrotreater's gate -- is stiff for the explicit integrator here:
   3000 steps did not reach 10 simulated minutes. Not resolved; see the
   docs' open problems.) Full run: also the rest state after a +5 K
   step of the cracking setpoint against a steady solve at the new
   ``T_crack`` (the quench controllers hold the new inlets).
2. **Balances along a trajectory.** Element accumulation against in - out
   at every saved state of a 10-minute recycle-compressor trip, and the
   inventory change against the trapezoid integral of in - out.
3. **Trajectory gradient.** d(per-pass conversion at 1 h) / d(size of a
   ``T_crack`` step), reverse mode through the diffrax solve, against a
   central difference.
4. **Time scales.** Each pretreat bed's thermal transit time estimated from
   its heat capacities against the mean arrival time of its outlet's
   response to an inlet step (heater lag 1 s); likewise the first cracking
   bed against an interstage step.
5. **A runaway is reported.** The heavy VGO, solved steady at a 365 C
   cracking inlet, then given a +15 K step of ``T_crack``: the hottest
   cell, when it crosses the alarm, and whether the integration completed
   (a failed one returns with ``info.success`` False and a warning, not an
   exception).
6. **Reformer recycle loop.** The reformer train with its dynamic separator
   and recycle loop (``recycle_loop=True``) from a perturbed start against
   the steady reformer; its C and H balances along a compressor trip; and
   d(RON at 1 h) / d(depth of a partial compressor trip) against a central
   difference.

The suite's tests (tests/refinery/test_dynamic.py) only smoke-test the
units and check finite gradients, per the test policy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import difflow_refinery as dr  # noqa: E402
from difflow_refinery.dynamic import DynamicHydrocracker, DynamicReformerTrain  # noqa: E402
from difflow_refinery.dynamic.hydrocracker import HydrocrackerHoldupParams  # noqa: E402
from difflow_refinery.hydrocracking.unit import Hydrocracker, HydrocrackerParams  # noqa: E402
from difflow_refinery.hydroprocessing.reactor import stream_enthalpy  # noqa: E402
from difflow_refinery.hydrotreating.feed import select_cuts, straight_run_cut  # noqa: E402

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_C = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]
VGO = (370 + 273.15, 560 + 273.15)
#: Step cap of every upset (a stiff transient returns unfinished instead of running for hours).
MAX_STEPS = 3000


def test_vgo(heavy=False, **params):
    """A 370-560 C straight-run VGO of the docs' light crude (SG 0.86, 1.8 wt% S) or a heavy one
    (SG 0.93, 3.0 wt% S, 2500 wppm N), 50 kg/s, and its steady hydrocracker."""
    sg, s, n = (0.93, 3.0, 2500.0) if heavy else (0.86, 1.8, 1500.0)
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_C]), sg=sg,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=s, nitrogen_wppm=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, *VGO))
    unit = Hydrocracker(char, feed, HydrocrackerParams(**params))
    return unit, feed, unit.solve(feed)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-300)))


def perturbed_start(dyn: DynamicHydrocracker) -> jnp.ndarray:
    """Every cell 20 K cold, loop inventory 80 %, separator holdup 50 %."""
    fs = dyn.flowsheet
    fs._update_state_indices()
    y = dyn.y0
    for e in fs.units:
        sl = e.state_slice
        if e.name == "pretreat":
            n, T = dyn.pretreat.unpack(y[sl])
            y = y.at[sl].set(dyn.pretreat.pack(n, T - 20.0))
        elif e.name == "crack":
            n, T = dyn.crack.unpack(y[sl])
            y = y.at[sl].set(jnp.concatenate([dyn.crack.pack(n, T - 20.0), y[sl][dyn.crack.n_bed_states:]]))
        elif e.name == "loop":
            y = y.at[sl].multiply(0.8)
        elif e.name == "hps":
            y = y.at[sl].multiply(0.5)
    return y


def compare(dyn, y, args=None) -> dict:
    ob = dyn.observe(0.0, y, args)
    rf = dyn.steady_reference()
    dy = dyn.flowsheet.derivatives(jnp.asarray(0.0), y, None, dyn.args() if args is None else args)
    return {
        "conversion": float(ob["conversion.per_pass"]), "steady_conversion": float(rf["conversion.per_pass"]),
        "conversion_err": float(ob["conversion.per_pass"] - rf["conversion.per_pass"]),
        "wabt_pretreat_err_K": float(ob["wabt.pretreat"] - rf["wabt.pretreat"]),
        "wabt_crack_err_K": float(ob["wabt.crack"] - rf["wabt.crack"]),
        "pretreat_T_out_err_K": np.asarray(ob["pretreat.T_out"] - rf["pretreat.T_out"]).tolist(),
        "crack_T_out_err_K": np.asarray(ob["crack.T_out"] - rf["crack.T_out"]).tolist(),
        "quench_rel_err": rel(ob["crack.quench"], rf["crack.quench"]),
        "pretreat_N_wppm": float(ob["pretreat.N_wppm"]), "steady_pretreat_N_wppm": float(rf["pretreat.N_wppm"]),
        "diesel_rel_err": rel(ob["diesel.rate"], rf["diesel.rate"]),
        "uco_recycle_rel_err": rel(ob["uco.recycle_rate"], rf["uco.recycle_rate"])
        if float(rf["uco.recycle_rate"]) > 0 else 0.0,
        "makeup_rel_err": rel(ob["h2.makeup"], rf["h2.makeup"]),
        "recycle_rel_err": rel(ob["recycle.rate"], rf["recycle.rate"]),
        "max_scaled_dydt": float(jnp.max(jnp.abs(dy))),
    }


def steady_limit(out, quick):
    unit, feed, res = test_vgo()
    out["hcu"] = {"cells": {}}
    for nc, ns in ((3, 1), (5, 1)) if quick else ((3, 1), (5, 1), (5, 2)):
        t0 = time.time()
        dyn = DynamicHydrocracker.from_steady(unit, feed, res, n_cells=nc, n_sub=ns)
        sim = dyn.simulate(2 * 3600.0, n_save=2)
        row = compare(dyn, sim.y_final)
        row.update(steps=int(sim.info.n_steps), seconds=time.time() - t0, n_states=int(dyn.y0.size))
        out["hcu"]["cells"][f"{nc}x{ns}"] = row
        print(f"HCU cells {nc} x RK4 {ns}: {row}", flush=True)
    if True:
        t0 = time.time()
        ur, fr, rr = test_vgo(uco_recycle=0.5)
        dyn = DynamicHydrocracker.from_steady(ur, fr, rr, n_cells=5)
        sim = dyn.simulate(4 * 3600.0, n_save=2)
        row = compare(dyn, sim.y_final)
        row.update(steps=int(sim.info.n_steps), seconds=time.time() - t0)
        out["hcu"]["uco_recycle_0.5"] = row
        print("HCU with UCO recycle", row, flush=True)
    return unit, feed, res


def balances_gradient_timescales(unit, feed, res, out, quick):
    dyn = DynamicHydrocracker.from_steady(unit, feed, res, n_cells=5)
    y_rest = dyn.settle(2 * 3600.0)
    lay = dyn.layout
    # --- balances along a 10-minute compressor trip
    a = dyn.args(steps={"compressor": [(600.0, -1.0), (1200.0, 1.0)]}, ramp=30.0)
    eb = jax.jit(lambda t, y: dyn.element_balance(t, y, a))
    inv_el = jax.jit(lambda y: dyn.inventory(y).elements(lay))
    sim = dyn.simulate(3600.0, args=a, y0=y_rest, n_save=121, max_steps=MAX_STEPS)
    ok = np.all(np.isfinite(np.asarray(sim.trajectory.y)), axis=1)
    worst, flux = 0.0, []
    for t, y in zip(np.asarray(sim.trajectory.t)[ok], np.asarray(sim.trajectory.y)[ok]):
        b = eb(t, y)
        worst = max(worst, float(jnp.max(jnp.abs(b["accumulation"] - b["in_minus_out"])
                                         / jnp.maximum(jnp.abs(b["in_minus_out"]), 1e-3))))
        flux.append(np.asarray(b["in_minus_out"]))
    flux = np.asarray(flux)
    inv = np.asarray([np.asarray(inv_el(y)) for y in np.asarray(sim.trajectory.y)[ok]])
    tt = np.asarray(sim.trajectory.t)[ok]
    integ = np.concatenate([[np.zeros(4)], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(tt)[:, None], axis=0)])
    drift = (np.max(np.abs((inv - inv[0]) - integ), axis=0)
             / np.maximum(np.max(np.abs(inv - inv[0]), axis=0), 1e-300)).tolist()
    tr = dyn.trajectory(sim, a, ["T_max", "conversion.per_pass", "h2.makeup", "vent.rate", "pretreat.N_wppm"])
    out["hcu"]["trip"] = {"pointwise_max_rel": worst, "integrated_rel_of_inventory_change_CHSN": drift,
                          "T_max_K": float(np.max(tr["T_max"])),
                          "conversion_min": float(np.min(tr["conversion.per_pass"])),
                          "conversion_max": float(np.max(tr["conversion.per_pass"])),
                          "N_slip_max_wppm": float(np.max(tr["pretreat.N_wppm"])),
                          "vent_max_mol_s": float(np.max(tr["vent.rate"])),
                          "completed": bool(sim.info.success), "steps": int(sim.info.n_steps),
                          "t_reached_s": float(tt[-1])}
    print("HCU trip / balances", out["hcu"]["trip"], flush=True)

    # --- gradient through a trajectory: d(organic N to the cracker at 15 min)/d(size of a feed-N step
    # at 1 min). (A T_crack step or a compressor trip does not complete in MAX_STEPS: stiff; see the docs.)
    def conv_at_1h(dN):
        a = dyn.args(steps={"feed_nitrogen": [(60.0, dN)]})
        sim = dyn.simulate(900.0, args=a, y0=y_rest, n_save=2, warn=False, max_steps=MAX_STEPS)
        return dyn.observe(sim.trajectory.t[-1], sim.y_final, a)["pretreat.N_wppm"]

    probe = dyn.simulate(900.0, args=dyn.args(steps={"feed_nitrogen": [(60.0, 0.5)]}), y0=y_rest, n_save=2,
                         warn=False, max_steps=MAX_STEPS)
    if not bool(probe.info.success):
        out["hcu"]["gradient"] = f"not taken: the 1 h trajectory did not complete in {MAX_STEPS} steps"
    else:
        t0 = time.time()
        g = float(jax.grad(conv_at_1h)(jnp.asarray(0.5)))
        tg = time.time() - t0
        h = 0.05
        fd = (float(conv_at_1h(jnp.asarray(0.5 + h))) - float(conv_at_1h(jnp.asarray(0.5 - h)))) / (2 * h)
        out["hcu"]["gradient"] = {"reverse": g, "central_difference": fd, "rel_err": abs(g - fd) / abs(fd),
                                  "seconds_reverse": tg, "steps": int(probe.info.n_steps)}
    print("HCU gradient", out["hcu"]["gradient"], flush=True)

    # --- thermal transit: pretreat beds (heater step) and the first cracking bed (interstage step)
    hp = HydrocrackerHoldupParams(heater_tau=1.0, interstage_tau=1.0)
    d1 = DynamicHydrocracker.from_steady(unit, feed, res, n_cells=5, holdups=hp)
    y1 = d1.settle(2 * 3600.0)
    streams = d1.flowsheet.outputs(jnp.asarray(0.0), y1, None, d1.args())
    us = d1._split(y1)
    comps = d1.comps
    est = {}
    for name, rx, inp in (("pretreat", d1.pretreat, {"feed": streams["pretreat_in"], "quench": streams["pretreat_quench"]}),
                          ("crack", d1.crack, {"feed": streams["crack_in"], "quench": streams["header"],
                                               "signals": streams["signals"]})):
        prof = rx.profiles(us[name], inp)
        n_hold, T = rx.unpack(us[name])
        steady_beds = res.pretreat.beds if name == "pretreat" else res.crack.beds
        vals = []
        for b, geo in enumerate(rx.beds[:2] if name == "pretreat" else rx.beds[:1]):
            km = steady_beds[b].k_model
            Wc = geo.W / rx.n_cells
            tot = 0.0
            for c in range(rx.n_cells):
                Fc = jax.tree_util.tree_map(lambda x: x[b, c], prof["outflow"])
                ncl = jax.tree_util.tree_map(lambda x: x[b, c], n_hold)
                C_flow = jax.jvp(lambda t: stream_enthalpy(Fc, lay, comps, km, t), (T[b, c],), (1.0,))[1]
                C_hold = jax.jvp(lambda t: stream_enthalpy(ncl, lay, comps, km, t), (T[b, c],), (1.0,))[1]
                tot += float((Wc * rx.cp_cat + C_hold) / C_flow)
            vals.append(tot)
        est[name] = vals
    arrivals = {}
    for name, signal, n_beds in (("pretreat", "T_pretreat", 2), ("crack", "T_crack", 1)):
        a = d1.args(steps={signal: [(600.0, 5.0)]}, ramp=60.0)
        sim = d1.simulate(2 * 3600.0, args=a, y0=y1, n_save=241, max_steps=MAX_STEPS)
        if not bool(sim.info.success):
            arrivals[name] = f"not completed ({int(sim.info.n_steps)} steps)"
            continue
        tr = d1.trajectory(sim, a, [f"{name}.T_in", f"{name}.T_out"])
        t = np.asarray(tr["t"])
        vals = []
        for b in range(n_beds):
            Tin, Tout = np.asarray(tr[f"{name}.T_in"])[:, b], np.asarray(tr[f"{name}.T_out"])[:, b]
            dTin, dTout = np.gradient(Tin, t), np.gradient(Tout, t)
            mean = lambda d: float(np.trapezoid(t * d, t) / np.trapezoid(d, t))
            vals.append(mean(dTout) - mean(dTin))
        arrivals[name] = vals
    out["hcu"]["thermal"] = {"transit_estimate_s": est, "mean_arrival_s": arrivals}
    print("HCU thermal", out["hcu"]["thermal"], flush=True)

    if not quick:
        # --- off-design rest: +5 K on the cracking setpoint, against a steady solve there
        a = dyn.args(steps={"T_crack": [(300.0, 5.0)]})
        sim = dyn.simulate(4 * 3600.0, args=a, y0=y_rest, n_save=2, max_steps=MAX_STEPS)
        u2 = Hydrocracker(unit.char, feed, dataclasses.replace(unit.params, T_crack=unit.params.T_crack + 5.0))
        r2 = u2.solve(feed)
        d2 = DynamicHydrocracker.from_steady(u2, feed, r2, n_cells=5)
        ob = dyn.observe(sim.trajectory.t[-1], sim.y_final, a)
        rf = d2.steady_reference()
        out["hcu"]["off_design"] = {
            "conversion": float(ob["conversion.per_pass"]), "steady_conversion": float(rf["conversion.per_pass"]),
            "conversion_err": float(ob["conversion.per_pass"] - rf["conversion.per_pass"]),
            "wabt_crack_err_K": float(ob["wabt.crack"] - rf["wabt.crack"]),
            "quench_rel_err": rel(ob["crack.quench"], rf["crack.quench"]),
            "makeup_rel_err": rel(ob["h2.makeup"], rf["h2.makeup"])}
        print("HCU off design", out["hcu"]["off_design"], flush=True)


def gradient_only(unit, feed, res, out):
    dyn = DynamicHydrocracker.from_steady(unit, feed, res, n_cells=5)
    y_rest = dyn.settle(2 * 3600.0)

    def n_at_15min(dN):
        a = dyn.args(steps={"feed_nitrogen": [(60.0, dN)]})
        sim = dyn.simulate(900.0, args=a, y0=y_rest, n_save=2, warn=False, max_steps=MAX_STEPS)
        return dyn.observe(sim.trajectory.t[-1], sim.y_final, a)["pretreat.N_wppm"], sim.info

    _, info = n_at_15min(jnp.asarray(0.5))
    t0 = time.time()
    g = float(jax.grad(lambda d: n_at_15min(d)[0])(jnp.asarray(0.5)))
    tg = time.time() - t0
    fds = {}
    for h in (0.1, 0.05):
        fds[h] = (float(n_at_15min(jnp.asarray(0.5 + h))[0]) - float(n_at_15min(jnp.asarray(0.5 - h))[0])) / (2 * h)
    rich = (4 * fds[0.05] - fds[0.1]) / 3
    out["hcu"]["gradient"] = {"output": "organic N to the cracker at 15 min, wppm per unit feed-N multiplier",
                              "reverse": g, "central_difference": fds, "extrapolated": rich,
                              "rel_err": abs(g - rich) / abs(rich), "seconds_reverse": tg,
                              "steps": int(info.n_steps), "completed": bool(info.success)}
    print("HCU gradient", out["hcu"]["gradient"], flush=True)


def operating_scenarios(out):
    """The hydrocracker's large transients posed as a unit is operated (5 cells, 3 RK4 steps per cell).

    * Setpoint moves as operator ramps (``DynamicHydrocracker.move``): +5 K on
      ``T_crack`` at 5 and 10 K/h.
    * Loss of half the recycle compressor's capacity (a 60 s rundown) with an
      emergency cut of the cracking setpoint, 10 K at 30 K/h.
    * A full compressor trip with the interlock sequence modelled here
      (rundown 60 s, feed cut to 5 % over 2 min, heater and interstage
      setpoints down 60 K over 10 min; depressuring is not modelled).
    * The margin to runaway: ``T_crack`` ramped up 40 K at 15 K/h, the hottest
      cell, each bed's rise and the quench valves' opening against their stops.
    * A gradient on a smooth scenario: d(conversion at 2 h)/d(size of a 5 K/h
      ramp of ``T_crack``), reverse mode against central differences at two
      tolerances.
    """
    from difflow_refinery.dynamic.signals import ramp_for_rate
    unit, feed, res = test_vgo()
    dyn = DynamicHydrocracker.from_steady(unit, feed, res, n_cells=5, n_sub=3)
    t0 = time.time()
    y = dyn.settle(2 * 3600.0, max_steps=MAX_STEPS)
    o = out.setdefault("hcu", {})
    o["scenarios"] = {"settle_s": time.time() - t0}
    R = lambda **kw: {**{k: 60.0 for k in dyn.base_signals}, **kw}       # noqa: E731

    def run(name, a, t_end, keys, n_save=13):
        t0 = time.time()
        s = dyn.simulate(t_end, args=a, y0=y, n_save=n_save, max_steps=MAX_STEPS, warn=False)
        ok = np.all(np.isfinite(np.asarray(s.trajectory.y)), axis=1)
        row = {"completed": bool(s.info.success), "steps": int(s.info.n_steps), "seconds": time.time() - t0,
               "t_reached_s": float(np.asarray(s.trajectory.t)[ok][-1])}
        tr = dyn.trajectory(s, a, keys)
        for k in keys:
            v = np.asarray(tr[k])
            row[k + ".min"], row[k + ".max"], row[k + ".end"] = (v.min(axis=0).tolist(), v.max(axis=0).tolist(),
                                                                v[-1].tolist())
        o["scenarios"][name] = row
        print("HCU", name, row, flush=True)
        return s, tr

    keys = ["T_max", "conversion.per_pass", "pretreat.N_wppm", "crack.quench", "recycle.rate"]
    for rate in (5.0, 10.0):
        st, rp = dyn.move("T_crack", 5.0, max_rate_K_per_h=rate)
        run(f"ramp_+5K_at_{rate:.0f}K_per_h", dyn.args(steps=st, ramp=rp), 2 * 3600.0, keys)
    run("compressor_to_50pct_with_setpoint_cut",
        dyn.args(steps={"compressor": [(300.0, -0.5)], "T_crack": [(300.0, -10.0)]},
                 ramp=R(compressor=60.0, T_crack=ramp_for_rate(10.0, 30.0 / 3600))), 3600.0, keys)
    run("compressor_trip_with_interlocks",
        dyn.args(steps={"compressor": [(300.0, -1.0)], "feed_rate": [(300.0, -0.95)], "T_crack": [(300.0, -60.0)],
                        "T_pretreat": [(300.0, -60.0)]},
                 ramp=R(compressor=60.0, feed_rate=120.0, T_crack=600.0, T_pretreat=600.0)), 3600.0, keys)
    r40 = ramp_for_rate(40.0, 15.0 / 3600)
    _, tr = run("runaway_margin_+40K_at_15K_per_h", dyn.args(steps={"T_crack": [(300.0, 40.0)]}, ramp=R(T_crack=r40)),
                300.0 + r40 + 1800.0, ["T_max", "conversion.per_pass", "crack.quench", "crack.T_in", "crack.T_out"],
                n_save=41)
    T, Tin1 = np.asarray(tr["T_max"]), np.asarray(tr["crack.T_in"])[:, 0]
    hot = np.nonzero(T > dyn.holdups.T_alarm)[0]
    sat = np.asarray(tr["crack.quench"]) / np.asarray(dyn.crack.q_max)
    o["scenarios"]["runaway_margin"] = {
        "T_alarm_K": dyn.holdups.T_alarm, "alarm_at_inlet_K": float(Tin1[hot[0]]) if hot.size else None,
        "first_valve_at_90pct_at_inlet_K": float(Tin1[np.nonzero(sat.max(axis=1) > 0.9)[0][0]])
        if (sat.max(axis=1) > 0.9).any() else None,
        "max_bed_rise_K": float(np.max(np.asarray(tr["crack.T_out"]) - np.asarray(tr["crack.T_in"]))),
        "inlet_K": Tin1.tolist(), "T_max_K": T.tolist()}
    print("HCU runaway margin", o["scenarios"]["runaway_margin"], flush=True)

    def conv(d, rtol=1e-6, atol=1e-8):
        a = dyn.args(steps={"T_crack": [(300.0, d)]}, ramp=R(T_crack=ramp_for_rate(5.0, 5.0 / 3600)))
        s = dyn.simulate(2 * 3600.0, args=a, y0=y, n_save=2, max_steps=MAX_STEPS, warn=False, rtol=rtol, atol=atol)
        return dyn.observe(s.trajectory.t[-1], s.y_final, a)["conversion.per_pass"]

    t0 = time.time()
    g = float(jax.grad(conv)(jnp.asarray(5.0)))
    tg = time.time() - t0
    cds = {}
    for h in (0.5, 0.25):
        for rt in (1e-6, 1e-8):
            cds[f"h={h},rtol={rt}"] = (float(conv(jnp.asarray(5.0 + h), rt, rt * 1e-2))
                                       - float(conv(jnp.asarray(5.0 - h), rt, rt * 1e-2))) / (2 * h)
    o["scenarios"]["gradient_ramp"] = {"output": "conversion at 2 h per K of a 5 K/h T_crack ramp", "reverse": g,
                                       "central_differences": cds, "seconds_reverse": tg}
    print("HCU ramp gradient", o["scenarios"]["gradient_ramp"], flush=True)


def runaway(out):
    unit, feed, res = test_vgo(heavy=True, T_crack=365.0 + 273.15)
    dyn = DynamicHydrocracker.from_steady(unit, feed, res, n_cells=5)
    y = dyn.settle(2 * 3600.0)
    a = dyn.args(steps={"T_crack": [(600.0, 15.0)]})
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = dyn.simulate(2 * 3600.0, args=a, y0=y, n_save=121, max_steps=MAX_STEPS)
    ex = dyn.excursion(sim, a)
    tr = dyn.trajectory(sim, a, ["crack.quench", "crack.T_out"])
    ex.update(seconds=time.time() - t0, warned=[str(x.message)[:200] for x in w],
              steady_conversion=float(res.outputs["conversion.per_pass"]),
              quench_final=np.asarray(tr["crack.quench"])[-1].tolist(),
              quench_max=np.asarray(dyn.crack.q_max).tolist(),
              crack_T_out_final_K=np.asarray(tr["crack.T_out"])[-1].tolist(), t_last=float(tr["t"][-1]))
    out["hcu"]["runaway"] = ex
    print("HCU runaway", ex, flush=True)


def reformer_loop(out):
    from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
    feed = lean_naphtha()
    ref = CatalyticReformer(ReformerParams())
    res = ref.solve(feed)
    out["ref_loop"] = {}
    t0 = time.time()
    tr = DynamicReformerTrain.from_steady(ref, feed, res, n_cells=5, recycle_loop=True)
    fs = tr.flowsheet
    fs._update_state_indices()
    y = tr.y0
    for e in fs.units:
        if e.name.startswith("rx"):
            y = y.at[e.state_slice.start:e.state_slice.start + tr.n_cells].add(-0.2)      # 20 K cold
        elif e.name == "loop":
            y = y.at[e.state_slice].multiply(0.8)
    sim = tr.simulate(3 * 3600.0, y0=y, n_save=2, max_steps=200_000)
    ob = tr.observe(sim.trajectory.t[-1], sim.y_final)
    rf = tr.steady_reference()
    out["ref_loop"]["rest"] = {
        "T_out_err_K": np.asarray(ob["T_out"] - rf["T_out"]).tolist(), "RON": float(ob["reformate.RON"]),
        "steady_RON": float(rf["reformate.RON"]), "recycle_rel_err": rel(ob["recycle.rate"], rf["recycle.rate"]),
        "purity_err": float(ob["recycle.h2_purity"] - rf["recycle.h2_purity"]),
        "net_h2_rel_err": rel(ob["net_gas.h2"], rf["net_gas.h2"]), "steps": int(sim.info.n_steps),
        "seconds": time.time() - t0}
    print("reformer loop rest", out["ref_loop"]["rest"], flush=True)
    y_rest = sim.y_final
    # --- balances along a 5-minute compressor trip
    # to 60 % for ten minutes (a full trip leaves the reactors without hydrogen, and a cell's
    # bed integration then exceeds its step limit: not covered)
    a = tr.args(steps={"compressor": [(600.0, -0.4), (1200.0, 0.4)]}, ramp=30.0)
    sim = tr.simulate(2400.0, args=a, y0=y_rest, n_save=81)
    eb = jax.jit(lambda t, y: tr.element_balance(t, y, a))
    worst = 0.0
    for t, yy in zip(sim.trajectory.t, sim.trajectory.y):
        b = eb(t, yy)
        worst = max(worst, float(jnp.max(jnp.abs(b["accumulation"] - b["in_minus_out"])
                                         / jnp.maximum(jnp.abs(b["in_minus_out"]), 1e-3))))
    trj = tr.trajectory(sim, a, ["T_out", "reformate.RON", "recycle.rate", "h2.make"])
    out["ref_loop"]["trip"] = {"pointwise_max_rel": worst,
                               "RON_min": float(np.min(trj["reformate.RON"])),
                               "RON_max": float(np.max(trj["reformate.RON"])),
                               "T_out_max_K": np.asarray(trj["T_out"]).max(axis=0).tolist(),
                               "T_out_min_K": np.asarray(trj["T_out"]).min(axis=0).tolist()}
    print("reformer loop trip", out["ref_loop"]["trip"], flush=True)

    # --- gradient: d(RON at 1 h)/d(depth of a partial compressor trip at 300 s)
    def ron_at_1h(d):
        a = tr.args(steps={"compressor": [(300.0, -d)]})
        s = tr.simulate(3600.0, args=a, y0=y_rest, n_save=2)
        return tr.observe(s.trajectory.t[-1], s.y_final, a)["reformate.RON"]

    t0 = time.time()
    g = float(jax.grad(ron_at_1h)(jnp.asarray(0.2)))
    tg = time.time() - t0
    h = 0.02
    fd = (float(ron_at_1h(jnp.asarray(0.2 + h))) - float(ron_at_1h(jnp.asarray(0.2 - h)))) / (2 * h)
    out["ref_loop"]["gradient"] = {"reverse": g, "central_difference": fd, "rel_err": abs(g - fd) / abs(fd),
                                   "seconds_reverse": tg}
    print("reformer loop gradient", out["ref_loop"]["gradient"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="hcu,scenarios,ref")
    ap.add_argument("--out", default="dynamic_hydrocracker_checks.json")
    a = ap.parse_args()
    parts = set(a.only.split(","))
    out = {}
    if "grad" in parts:
        unit, feed, res = test_vgo()
        out["hcu"] = {}
        gradient_only(unit, feed, res, out)
    if "hcu" in parts or "upsets" in parts:
        if "hcu" in parts:
            unit, feed, res = steady_limit(out, a.quick)
        else:
            unit, feed, res = test_vgo()
            out["hcu"] = {}
        balances_gradient_timescales(unit, feed, res, out, a.quick)
    if "scenarios" in parts:
        operating_scenarios(out)
    if "runaway" in parts:
        out.setdefault("hcu", {})
        runaway(out)
    if "ref" in parts:
        reformer_loop(out)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
