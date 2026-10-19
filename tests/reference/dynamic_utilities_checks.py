"""Correctness gates of the dynamic hydrogen plant, amine unit, SRU and crude unit; not part of the suite.

Run from the repository root::

    python tests/reference/dynamic_utilities_checks.py [--units smr,amine,sru,cdu] [--out results.json]

About 40 minutes on a laptop CPU, most of it the steady solves the dynamic
units are built from (the amine loop and the SRU compile for a minute or
two each). For each unit it prints, and writes as JSON, the numbers the
"Dynamic operation" section of docs/unit-operations-refinery.md quotes:

1. **Steady limit.** From a start away from rest (cells 30 K cold or 20 K
   hot, inventories at 80-90 %), the unit settles to the steady unit's
   answer; the largest relative error of what ``steady_reference`` lists.
2. **Balances along a trajectory.** At every saved state of an upset, the
   element (or species) accumulation against in - out; the largest
   mismatch relative to the throughput. Equal by construction, so this
   checks the code.
3. **Trajectory gradient.** d(an output at 1 h) / d(the size of a step at
   5 min), reverse mode through the diffrax solve, against a central
   difference extrapolated from two step sizes (Richardson).
4. **Time scales.** The mean arrival time of an output's response to a
   step, ``int (1 - r(t)) dt``, against the estimate from the holdups:
   lags in series add their time constants; a catalyst bed's thermal
   transit is ``C_bed / C_flow`` with the gas's heat flow along its
   equilibrium; the reformer tube's, under a heat-flux step that reaches
   every cell at once, is the heat-share-weighted sum of the downstream
   cells' time constants.

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

from difflow import make_stream  # noqa: E402


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def steady_gate(unit, y_start, t_settle):
    """Settle from ``y_start``; the largest relative error against the steady reference."""
    t0 = time.time()
    sim = unit.simulate(t_settle, y0=y_start, n_save=2, max_steps=400_000)
    ob = unit.observe(sim.trajectory.t[-1], sim.y_final)
    ref = unit.steady_reference()
    errs = {k: rel(ob[k], ref[k]) for k in ref if k in ob and abs(float(ref[k])) > 1e-9}
    worst = max(errs, key=errs.get)
    return {"max_rel_err": errs[worst], "worst": worst, "errors": errs, "seconds": time.time() - t0,
            "steps": int(sim.info.n_steps)}


def balance_gate(unit, args, t_end, n_save=13):
    sim = unit.simulate(t_end, args=args, n_save=n_save)
    worst = 0.0
    for t, y in zip(sim.trajectory.t, sim.trajectory.y):
        b = unit.balance(t, y, args)
        mis = np.abs(np.asarray(b["in_minus_out"] - b["accumulation"]))
        worst = max(worst, float(np.max(mis / np.maximum(np.asarray(b["through"]), 1e-12))))
    moved = 0.0
    b = unit.balance(sim.trajectory.t[n_save // 3], sim.trajectory.y[n_save // 3], args)
    moved = float(np.max(np.abs(np.asarray(b["accumulation"])) / np.maximum(np.asarray(b["through"]), 1e-12)))
    return {"max_rel_mismatch": worst, "accumulation_rel_mid_upset": moved, "states": n_save}


def gradient_gate(unit, signal, d0, h, out_key, t_step=300.0, t_end=3600.0):
    def J(d):
        a = unit.args(steps={signal: [(t_step, d)]})
        sim = unit.simulate(t_end, args=a, n_save=2)
        return unit.observe(sim.trajectory.t[-1], sim.y_final, a)[out_key]

    t0 = time.time()
    g = float(jax.grad(J)(jnp.asarray(d0, dtype=float)))
    t_grad = time.time() - t0
    cd = [(float(J(d0 + s)) - float(J(d0 - s))) / (2 * s) for s in (h, h / 2)]
    ext = (4 * cd[1] - cd[0]) / 3
    return {"output": out_key, "signal": signal, "ad": g, "cd": cd, "cd_extrapolated": ext,
            "rel_diff": abs(g - ext) / abs(ext), "grad_seconds": t_grad}


def arrival_gate(unit, signal, delta, out_key, t_end, estimate, args_base=None, t_step=60.0, n_save=601):
    """Mean arrival time of ``out_key``'s response to a step of ``signal`` at ``t_step``."""
    args_base = args_base or {}
    a = unit.args(steps={signal: [(t_step, delta)]}, ramp=1.0, **args_base)
    sim = unit.simulate(t_end, args=a, n_save=n_save)
    tr = unit.trajectory(sim, a, keys=[out_key])
    t, y = np.asarray(tr["t"]), np.asarray(tr[out_key])
    r = (y - y[0]) / (y[-1] - y[0])
    m = t >= t_step
    t_mean = float(np.trapezoid(1.0 - r[m], t[m]))
    return {"output": out_key, "signal": signal, "simulated_s": t_mean, "estimate_s": float(estimate),
            "settled_fraction_at_end": float(r[-2])}


def linear_arrival(unit, signal, out_key, y=None):
    """Mean arrival time of ``out_key`` after a small step of ``signal``, from the model linearised at rest.

    ``x' = A x + b u``, ``y = c x + d u``: the step response's mean arrival
    time is ``c A^-2 b / (d - c A^-1 b)``. What the nonlinear trajectory
    should reproduce for a small step.
    """
    fs = unit.flowsheet
    y = unit.y0 if y is None else y
    a0 = unit.args()

    def with_u(u):
        a = jax.tree_util.tree_map(lambda v: v, a0)
        a[signal] = dict(a[signal], base=u)
        return a

    u0 = a0[signal]["base"]
    A = jax.jacrev(lambda yy: fs.derivatives(jnp.asarray(0.0), yy, None, a0))(y)
    b = jax.jacrev(lambda u: fs.derivatives(jnp.asarray(0.0), y, None, with_u(u)))(u0)
    c = jax.grad(lambda yy: unit.observe(0.0, yy, a0)[out_key])(y)
    d = jax.grad(lambda u: unit.observe(0.0, y, with_u(u))[out_key])(u0)
    Ab = jnp.linalg.solve(A, b)
    AAb = jnp.linalg.solve(A, Ab)
    return float(jnp.dot(c, AAb) / (d - jnp.dot(c, Ab)))


# ----------------------------------------------------------------- the units


def natural_gas():
    y = {"methane": 0.89, "ethane": 0.07, "propane": 0.01, "n_butane": 0.0011,   # IEAGHG 2017/02 stream 2
         "carbon_dioxide": 0.02, "nitrogen": 0.0089, "hydrogen_sulfide": 5e-6}
    return make_stream({k: v * 1455.8 / 3.6 for k, v in y.items()}, 401.15, 3.71e6)


def smr_gates(out):
    import difflow_refinery.hydrogen as h2
    from difflow_refinery.dynamic import DynamicHydrogenPlant, SMRHoldupParams
    from difflow_refinery.hydrogen import smr
    p = h2.SMRParams(n_tubes=240, psa=h2.PSA(recovery=0.90, purity=0.9999))
    t0 = time.time()
    dp = DynamicHydrogenPlant.from_steady(natural_gas(), p)
    r = {"build_s": time.time() - t0, "n_states": int(dp.y0.size), "tau_tube_s": dp.tau_tube,
         "tau_shift_s": dp.tau_beds}
    fs = dp.flowsheet
    fs._update_state_indices()
    y = dp.y0
    for e in fs.units:
        sl = e.state_slice
        if e.name == "tubes":
            y = y.at[sl].add(-0.3)
        elif e.name == "hts":
            y = y.at[sl].add(0.2)
        elif e.name in ("psa", "furnace_lag"):
            y = y.at[sl].multiply(0.9)
    r["steady"] = steady_gate(dp, y, 6 * 3600.0)
    Q0 = dp.base_signals["firing"]
    r["balance"] = balance_gate(dp, dp.args(steps={"firing": [(300.0, 0.03 * Q0)], "feed_rate": [(1800.0, -0.05)]}),
                                2 * 3600.0)
    r["gradient"] = gradient_gate(dp, "firing", 0.02 * Q0, 0.004 * Q0, "H2")
    # time scales: the tube under a firing step (furnace lag 1 s), the HT shift bed under an inlet step
    fast = DynamicHydrogenPlant.from_steady(natural_gas(), p, result=dp.steady,
                                            holdup=SMRHoldupParams(tau_fire=1.0))
    rep = fast.tubes.report(fast.tubes.initial_state(), {
        "inlet": fs.outputs(jnp.asarray(0.0), dp.y0, None, dp.args())["reformer_feed"],
        "firing": {"Q": jnp.asarray(Q0)}})
    n = rep["n"]
    C_flow = np.asarray(jax.vmap(lambda nk, Tk: jax.jvp(lambda T: smr.enthalpy(nk, T), (Tk,),
                                                        (jnp.asarray(1.0),))[1])(n, rep["T"]))
    taus = np.asarray(fast.tubes.C_cell) / C_flow
    N = len(taus)
    est_tube = float(np.mean([taus[k:].sum() for k in range(N)])) + 1.0
    r["time_tube"] = arrival_gate(fast, "firing", 0.03 * Q0, "T_reformer_out", 3 * 3600.0, est_tube)
    # the frozen-gas estimate counts every cell's heat as reaching the outlet; the tube's gas
    # reforms more as it heats, so heat added upstream is mostly taken up by reaction before the
    # outlet -- the linearised model has that coupling
    r["time_tube"]["cell_time_constants_s"] = taus.tolist()
    r["time_tube"]["linearised_s"] = linear_arrival(fast, "firing", "T_reformer_out")
    hts = fast.beds[0]
    est_hts = float(hts.thermal_time(smr.to_vec(dp.steady["hts"]), dp.steady["hts"]["T"]))
    r["time_shift"] = arrival_gate(fast, "T_hts", 10.0, "T_hts_out", 3 * 3600.0, est_hts)
    out["smr"] = r
    print("smr", json.dumps(r, default=float, indent=1), flush=True)


def amine_gates(out):
    from difflow_refinery.dynamic import DynamicAminePlant
    from difflow_refinery.treating import TrayParams
    from difflow_refinery.treating.unit import AmineUnitParams, Contactor
    tray = TrayParams(area=200.0, k_G=1e-5, k_L=1e-4)    # ILLUSTRATIVE (tests/refinery/test_treating.py)
    p = AmineUnitParams("MDEA", 0.45, (Contactor("fuel", 120.0, 8, tray),), Q_reboiler=5e6, n_regenerator=6)
    gas = {"F_methane": 700.0, "F_ethane": 150.0, "F_hydrogen": 110.0, "F_hydrogen_sulfide": 20.0,
           "F_carbon_dioxide": 20.0, "F_water": 1.0, "T": 313.15, "P": 7e5}
    t0 = time.time()
    dp = DynamicAminePlant.from_steady(p, {"fuel": gas})
    r = {"build_s": time.time() - t0, "n_states": int(dp.y0.size)}
    y = dp.y0.at[:5].multiply(0.8).at[5].add(-0.05)            # inventories at 80 %, hot end 5 K cold
    r["steady"] = steady_gate(dp, y, 6 * 3600.0)
    r["balance"] = balance_gate(dp, dp.args(steps={"gas_rate": [(300.0, 0.3)], "Q_reboiler": [(2400.0, -5e5)]}),
                                2 * 3600.0)
    r["gradient"] = gradient_gate(dp, "gas_rate", 0.1, 0.02, "lean.alpha_H2S")
    hp = dp.holdup
    r["time_lean"] = arrival_gate(dp, "gas_rate", 0.1, "lean.alpha_H2S", 4 * 3600.0, hp.tau_lean + hp.tau_rich,
                                  n_save=241)
    out["amine"] = r
    print("amine", json.dumps(r, default=float, indent=1), flush=True)


def sru_gates(out):
    from difflow_refinery.dynamic import DynamicSRU, SRUHoldupParams
    from difflow_refinery.sulfur import claus as cl, tgtu as tg
    from difflow_refinery.treating import TrayParams
    ag = make_stream({"hydrogen_sulfide": 90.0, "carbon_dioxide": 5.0, "water": 4.0, "methane": 1.0},
                     T=323.15, P=1.6e5)                       # tests/refinery/test_sulfur.py
    beds = (cl.ConverterParams(523.15, hydrolysis=(0.9, 0.7)), cl.ConverterParams(483.15))
    claus = cl.ClausParams(converters=beds, T_condensers=(443.15, 438.15, 413.15), T_air=473.15)
    tgtu = tg.TGTUParams(tray=TrayParams(area=200.0, k_G=1e-5, k_L=1e-4), L=30.0, n_stages=12)
    t0 = time.time()
    ds = DynamicSRU.from_steady(ag, claus, tgtu)
    r = {"build_s": time.time() - t0, "n_states": int(ds.y0.size), "tau_beds_s": ds.tau_beds}
    fs = ds.flowsheet
    fs._update_state_indices()
    y = ds.y0
    for e in fs.units:
        sl = e.state_slice
        if e.name.startswith("bed") or e.name == "hydrogenation":
            y = y.at[sl].add(-0.2)
        elif e.name == "tgtu_solvent":
            y = y.at[sl].multiply(0.8)
    r["steady"] = steady_gate(ds, y, 10 * 3600.0)
    r["balance"] = balance_gate(ds, ds.args(steps={"acid_rate": [(300.0, 0.1)], "T_bed1": [(1800.0, -10.0)]}),
                                2 * 3600.0)
    r["gradient"] = gradient_gate(ds, "T_bed2", 5.0, 1.0, "SO2_emitted")
    fast = DynamicSRU.from_steady(ag, claus, tgtu, result=ds.steady, holdup=SRUHoldupParams(tau_reheat=1.0))
    st = ds.steady["claus"]["bed"][1]
    bed2 = fast.beds[1]
    n2 = cl.to_vec(ds.flowsheet.outputs(jnp.asarray(0.0), ds.y0, None, ds.args())["bed2_out"])
    est = float(bed2.thermal_time(n2, st["T_out"])) + 1.0
    r["time_bed2"] = arrival_gate(fast, "T_bed2", 10.0, "bed2.T_out", 5 * 3600.0, est, n_save=301)
    # air-demand control: an acid-gas step with and without the integral trim
    ctl = DynamicSRU.from_steady(ag, claus, tgtu, result=ds.steady, holdup=SRUHoldupParams(air_Ki=2e-3))
    a = ctl.args(steps={"acid_rate": [(300.0, 0.1)]})
    rows = {}
    for name, u in (("open_loop", ds), ("air_control", ctl)):
        sim = u.simulate(3 * 3600.0, args=a, n_save=2)
        ob = u.observe(sim.trajectory.t[-1], sim.y_final, a)
        rows[name] = {"recovery_3h": float(ob["recovery"]), "ratio_residual_3h": float(ob["ratio_residual"])}
    r["air_control"] = rows
    out["sru"] = r
    print("sru", json.dumps(r, default=float, indent=1), flush=True)


def cdu_gates(out):
    from difflow_refinery import Assay, ColumnThermo, characterize, column as cc
    from difflow_refinery.dynamic import CDUHoldupParams, DynamicCrudeUnit
    assay = Assay([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],          # the docs' test crude
                  [t + 273.15 for t in (-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850)],
                  sg=0.86, light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crude = characterize(assay)
    thermo = ColumnThermo.from_characterization(crude)
    kg_s = 95_000 * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    feed = crude.stream(kg_s, T=273.15 + 240, P=6e5, basis="mass")
    Vf = float(thermo.std_volume(jnp.stack([jnp.asarray(feed[f"F_{n}"]) for n in thermo.names])))
    params = cc.CrudeColumnParams(
        n_stages=30, feed_stage=27, P_top=1.5e5, P_bottom=1.9e5, P_condenser=1.3e5, bottom_steam=150.0,
        steam_T=533.15,
        side_products=(cc.SideProduct("kero", 9, 4, steam=40.0), cc.SideProduct("diesel", 16, 4, steam=40.0),
                       cc.SideProduct("ago", 22, 3, steam=20.0)),
        pumparounds=(cc.Pumparound("pa1", 12, 10), cc.Pumparound("pa2", 19, 17)),
        specs=(cc.product_rate("naphtha", 0.20 * Vf), cc.product_rate("kero", 0.11 * Vf),
               cc.product_rate("diesel", 0.17 * Vf), cc.product_rate("ago", 0.05 * Vf),
               cc.pumparound_duty("pa1", 15e6), cc.pumparound_delta_t("pa1", 60.0),
               cc.pumparound_duty("pa2", 20e6), cc.pumparound_delta_t("pa2", 60.0), cc.overflash(0.05)),
        furnace=cc.Furnace(efficiency=0.85))
    col = cc.CrudeColumn(params, thermo)
    t0 = time.time()
    res = col.solve(feed)
    du = DynamicCrudeUnit.from_steady(col, feed, res)
    r = {"build_s": time.time() - t0, "n_states": int(du.y0.size)}
    fs = du.flowsheet
    fs._update_state_indices()
    y = du.y0
    for e in fs.units:
        sl = e.state_slice
        if e.name == "furnace":
            y = y.at[sl].add(-0.1)
        elif e.name == "column":
            y = y.at[sl].multiply(0.9)
    r["steady"] = steady_gate(du, y, 4 * 3600.0)
    r["balance"] = balance_gate(du, du.args(steps={"COT": [(300.0, 5.0)], "feed_rate": [(1800.0, 0.05)]}),
                                2 * 3600.0, n_save=9)
    r["gradient"] = gradient_gate(du, "COT", 5.0, 1.0, "residue.T")
    h = du.holdup_unit
    i_res = list(du.products).index("residue")
    r["time_residue"] = arrival_gate(du, "COT", 5.0, "residue.T", 3 * 3600.0,
                                     float(du.furnace.tau) + float(h.tau_T[i_res]), n_save=181)
    out["cdu"] = r
    print("cdu", json.dumps(r, default=float, indent=1), flush=True)


GATES = {"smr": smr_gates, "amine": amine_gates, "sru": sru_gates, "cdu": cdu_gates}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="smr,amine,sru,cdu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = {}
    for u in a.units.split(","):
        t0 = time.time()
        GATES[u](results)
        results[u]["total_s"] = time.time() - t0
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, default=float, indent=1)
