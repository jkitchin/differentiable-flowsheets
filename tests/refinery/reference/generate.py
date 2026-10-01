"""Generate ``cdu_reference.json``: the reference answers difflow is checked against.

Run from the repository root (needs IDAES, Pyomo and IPOPT; optionally
Caleb Bell's ``chemicals`` for the layer-1 cross-checks)::

    PYTHONPATH=src:tests python -m refinery.reference.generate [--ipopt PATH] [--chemicals DIR]

What it computes, in the order it needs them:

* **Layer 3** -- the reference column (:mod:`.mesh`, Pyomo + IPOPT) from a
  starting point that owes nothing to difflow's solution, then the same
  column with Watson's latent heat unsmoothed, to measure what the
  smoothing difflow states costs.
* **Layer 4** -- central finite differences of that reference column in
  the diesel yield spec and the overflash spec.
* **Layer 2** -- IDAES generic property packages (:mod:`.idaes_thermo`)
  at states taken from the reference column: the ideal package (the model
  difflow states, assembled and solved by IDAES) and Peng-Robinson on the
  same constants; water saturation by IAPWS-IF97 and its latent heat from
  IAPWS-95 by Clausius-Clapeyron.
* **Layer 1** -- ``chemicals`` evaluated on the case's pseudo-components:
  Lee-Kesler vapour pressure and acentric factor, Riedel latent heat.

The only difflow call is :func:`case.difflow_case` (the characterisation,
whose constants are an input to every layer) and :func:`case.feed_flows`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "cdu_reference.json"
DEFAULT_IPOPT = os.path.expanduser("~/.idaes/bin/ipopt")

#: Main-column stages whose state layer 2 re-evaluates: top, the three side
#: draws, both pumparound draws, the feed stage and the bottom.
from .case import FURNACE_EFFICIENCY  # noqa: E402

STATE_STAGES = ("s1", "s9", "s12", "s16", "s19", "s22", "s27", "s30")


def _quiet():
    import logging

    for name in ("idaes", "pyomo"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import idaes.logger as idaeslog

        idaeslog.getIdaesLogger("init").setLevel(logging.ERROR)
    except Exception:  # pragma: no cover
        pass


def reference_column(cd, feed, props, ipopt, eps=0.01, start=None, specs_over=None):
    """Build, initialise and solve the reference column; ``(model, status, info)``."""
    from . import case, mesh

    Vf = props.std_volume(feed)
    vy = dict(case.SPECS["volume_yield"], **(specs_over or {}).get("volume_yield", {}))
    of = (specs_over or {}).get("overflash", case.SPECS["overflash"])
    specs = {"volume_yield": {k: v * Vf for k, v in vy.items()},
             "pa_duty": case.SPECS["pa_duty"], "pa_delta_T": case.SPECS["pa_delta_T"],
             "overflash": of * Vf}
    from . import formulas as fm

    _, H_in = fm.flash_enthalpy(props, feed, case.T_FURNACE_IN, case.P_FURNACE_IN)
    net = mesh.Network.atmospheric(case.LAYOUT)
    M = mesh.build(cd, net, feed, specs, case.LAYOUT["steam_T"], H_in, eps=eps)
    if start is None:
        mesh.initialise(M, props, feed, INITIAL_GUESS)
    else:
        _copy_values(start, M)
    t = time.time()
    status, res = mesh.solve(M, ipopt=ipopt)
    info = {"status": status, "seconds": round(time.time() - t, 1),
            "furnace_inlet_enthalpy_W": H_in}
    return M, status, info


#: The coarse, difflow-free starting point: a linear 380-580 K main-column
#: profile, uniform 300 mol/s liquid and 500 mol/s vapour, round-number
#: draws. IPOPT gets from here to the solution on its own.
INITIAL_GUESS = dict(T_top=380.0, T_bottom=580.0, T_F=590.0, L=300.0, V=500.0, reflux=300.0,
                     dist=200.0, draws={"kero": 60.0, "diesel": 70.0, "ago": 15.0},
                     pa={"pa1": 300.0, "pa2": 300.0})


def _copy_values(src, dst):
    from pyomo.environ import Var

    for v in src.component_objects(Var):
        d = dst.find_component(v.name)
        for k, vv in v.items():
            d[k].set_value(vv.value)


def column_summary(M, props) -> dict:
    from . import formulas as fm
    from . import mesh

    r = mesh.results(M)
    net = M._meta["net"]
    prods = {p: fm.product_inspection(props, f) for p, f in r["products"].items()}
    feed_vol = sum(p["volume"] for p in prods.values())
    feed_mass = sum(p["mass"] for p in prods.values())
    for p in prods.values():
        p["volume_yield"] = p["volume"] / feed_vol
        p["mass_yield"] = p["mass"] / feed_mass
    order = ["naphtha", "kero", "diesel", "ago", "residue"]
    gaps = {f"{a}-{b}": prods[b]["TBP"]["5"] - prods[a]["TBP"]["95"] for a, b in zip(order, order[1:])}
    main = [k for k in net.nodes if k.startswith("s") and k[1:].isdigit()]
    strippers = {d: [r["T"][k] for k in net.nodes if k.startswith(d)] for d in net.side_draws}
    return {
        "stage_T": [r["T"][k] for k in main],
        "stripper_T": strippers,
        "T_condenser": r["T_condenser"],
        "coil_outlet_T": r["T_F"],
        "feed_vaporized": r["feed_vaporized"],
        "condenser_duty": r["condenser_duty"],
        "furnace_duty": r["furnace_duty"],
        "fired_duty": r["furnace_duty"] / FURNACE_EFFICIENCY,
        "pumparound_duty": r["pumparound_duty"],
        "pumparound_return_T": r["pumparound_return_T"],
        "pumparound_rate": r["pumparound_rate"],
        "side_draw_rate": r["side_draw_rate"],
        "reflux": r["reflux"],
        "free_water": r["free_water"],
        # main column only, as difflow reports it (the strippers' is in
        # stripper_water_saturation)
        "water_saturation": [r["water_saturation"][k] for k in main],
        "max_water_saturation": max(r["water_saturation"][k] for k in main),
        "stripper_water_saturation": {d: [r["water_saturation"][k] for k in net.nodes if k.startswith(d)]
                                      for d in net.side_draws},
        "L": [r["L"][k] for k in main],
        "V": [r["V"][k] for k in main],
        "products": prods,
        "product_flows": r["products"],
        "gaps": gaps,
    }


def stage_states(M) -> dict:
    """``(T, P, x, y, yw, L, V)`` of the stages and the condenser, for layer 2."""
    from pyomo.environ import value

    net = M._meta["net"]
    nc = len(M._meta["comp"]["names"])
    out = {}
    for k in STATE_STAGES:
        out[k] = {"T": value(M.T[k]), "P": net.P[k],
                  "x": [value(M.x[k, i]) for i in range(nc)],
                  "y": [value(M.y[k, i]) for i in range(nc)],
                  "yw": value(M.yw[k]), "L": value(M.L[k]), "V": value(M.V[k])}
    out["condenser"] = {"T": value(M.T0), "P": net.P_condenser,
                        "x": [value(M.x0[i]) for i in range(nc)]}
    return out


# -----------------------------------------------------------------------------
# Layer 2
# -----------------------------------------------------------------------------


def layer2(cd, feed, props, states, cot, P_feed) -> dict:
    from . import case
    from . import formulas as fm
    from . import idaes_thermo as it

    names = cd["names"]
    nc = len(names)
    out = {"ideal": {}, "pr": {}, "water": {}}

    # -- the ideal model through IDAES, at each stage's own state --------------
    # The stage's whole contents (liquid, hydrocarbon vapour and steam) flashed
    # at its T and P. IDAES should split it back into the stage's own L and V:
    # that checks the equilibrium and summation closure, and the K-values and
    # phase enthalpies are compared with difflow's in the test.
    for eps_name, eps in (("smoothed", 0.01), ("exact", 0.0)):
        sp = it.StatePoint(it.ideal_config(cd, water=True, eps=eps))
        res = {}
        for k, s in states.items():
            if k == "condenser":
                continue
            Ft = s["L"] + s["V"]
            z = {n: (s["L"] * s["x"][i] + s["V"] * s["y"][i]) / Ft for i, n in enumerate(names)}
            z["H2O"] = s["V"] * s["yw"] / Ft
            ok = sp.solve(z, s["T"], s["P"], flow=Ft)
            st = sp.s
            res[k] = {
                "ok": ok,
                "V": sp.value(st.flow_mol_phase["Vap"]),
                "L": sp.value(st.flow_mol_phase["Liq"]),
                "K": [sp.value(st.mole_frac_phase_comp["Vap", n] / st.mole_frac_phase_comp["Liq", n])
                      for n in names],
                "x": [sp.value(st.mole_frac_phase_comp["Liq", n]) for n in names],
                "y": [sp.value(st.mole_frac_phase_comp["Vap", n]) for n in names],
                "yw": sp.value(st.mole_frac_phase_comp["Vap", "H2O"]),
                "h_liq": sp.value(st.enth_mol_phase["Liq"]),
                "h_vap": sp.value(st.enth_mol_phase["Vap"]),
            }
        out["ideal"][eps_name] = res

    # -- the crude alone: bubble and dew points, and the flash at the coil outlet
    sp = it.StatePoint(it.ideal_config(cd, water=False, eps=0.01))
    z = dict(zip(names, cd["mole_fraction"]))
    ok = sp.solve(z, cot, P_feed)
    st = sp.s
    out["crude"] = {
        "T": cot, "P": P_feed, "ok": ok,
        "feed_vaporized": sp.value(st.phase_frac["Vap"]),
        "T_bubble": sp.value(st.temperature_bubble["Vap", "Liq"]),
        "T_dew": sp.value(st.temperature_dew["Vap", "Liq"]),
        "h": sp.value(st.enth_mol),
    }

    # -- Peng-Robinson on the same constants, hydrocarbons only ---------------
    # Each stage's hydrocarbons at the stage T and the hydrocarbon partial
    # pressure P (1 - yw): the pressure the hydrocarbons see once steam is
    # set aside, which is where Raoult's K = Psat/P is evaluated too.
    pr_states = {k: s for k, s in states.items() if k != "condenser"}
    pr_states["coil_outlet"] = {"T": cot, "P": P_feed, "L": 1.0, "V": 0.0, "yw": 0.0,
                                "x": cd["mole_fraction"], "y": cd["mole_fraction"]}
    # the furnace inlet: with the coil outlet, the enthalpy rise the furnace
    # duty is, under each model
    pr_states["furnace_inlet"] = {"T": case.T_FURNACE_IN, "P": case.P_FURNACE_IN, "L": 1.0,
                                  "V": 0.0, "yw": 0.0, "x": cd["mole_fraction"],
                                  "y": cd["mole_fraction"]}
    for k, s in pr_states.items():
        Fh = s["L"] + s["V"] * (1 - s["yw"])
        zz = np.array([(s["L"] * s["x"][i] + s["V"] * s["y"][i]) / Fh for i in range(nc)])
        # a component the column has stripped out entirely is exactly zero
        # here; IDAES's log-form equilibrium needs it present, so floor it at a
        # trace (1e-12, what StatePoint fixes it to anyway) before seeding
        zz = np.maximum(zz, 1e-12)
        zz = zz / zz.sum()
        P = s["P"] * (1 - s["yw"])
        K = props.K(s["T"], P)
        b = fm.rachford_rice(zz, K)
        x = zz / (1 + b * (K - 1))
        seed = {"beta": b, "x": dict(zip(names, x)), "y": dict(zip(names, K * x)),
                "T_bubble": s["T"] - 50, "T_dew": s["T"] + 50}
        # a fresh block, from a fresh config, per state: solve_two_phase fixes
        # and deactivates the bubble/dew sub-problem, so a re-used block starts
        # from the previous state's values, and building a parameter block
        # consumes parts of the config dict it is given
        spr = it.StatePoint(it.pr_config(cd))
        ok = spr.solve_two_phase(dict(zip(names, zz)), s["T"], P, seed)
        st = spr.s
        xp = np.array([spr.value(st.mole_frac_phase_comp["Liq", n]) for n in names])
        out["pr"][k] = {
            "ok": ok, "termination": spr.termination, "T": s["T"], "P_hc": P, "z": zz.tolist(),
            "beta": spr.value(st.phase_frac["Vap"]),
            "beta_ideal": b,
            "K": [spr.value(st.mole_frac_phase_comp["Vap", n] / st.mole_frac_phase_comp["Liq", n])
                  for n in names],
            "x": xp.tolist(),
            "h_liq": spr.value(st.enth_mol_phase["Liq"]),
            "h": spr.value(st.enth_mol),
            "h_ideal": fm.flash_enthalpy(props, zz, s["T"], P)[1],
            # the ideal model's liquid enthalpy at PR's own liquid composition
            "h_liq_ideal_same_x": float(np.sum(xp * props.hL(s["T"]))),
            "T_eq": spr.value(st._teq["Vap", "Liq"]),
        }

    # -- water -----------------------------------------------------------------
    out["water"] = water_reference()
    return out


def water_reference() -> dict:
    """IAPWS-IF97 saturation pressure, and IAPWS-95 latent heat by
    Clausius-Clapeyron, ``dHvap = T (v_g - v_l) dPsat/dT`` (exact on the
    saturation line), where ``chemicals`` is importable."""
    from . import formulas as fm

    Ts = [300.0, 320.0, 350.0, 373.124, 400.0, 450.0, 500.0, 550.0]
    out = {"T": Ts, "Psat_IF97": [float(fm.water_psat(fm.NUMPY, T)) for T in Ts]}
    try:
        from chemicals.iapws import iapws95_dPsat_dT, iapws95_Psat, iapws95_rhog_sat, iapws95_rhol_sat

        out["Psat_IAPWS95"] = [float(iapws95_Psat(T)) for T in Ts]

        mw = fm.WATER["MW"] / 1000.0
        hv = []
        for T in Ts:
            dP = iapws95_dPsat_dT(T)
            dP = dP[0] if isinstance(dP, tuple) else dP
            hv.append(float(T * (1 / iapws95_rhog_sat(T) - 1 / iapws95_rhol_sat(T)) * dP * mw))
        out["hvap_IAPWS95"] = hv
    except ImportError:
        pass
    return out


# -----------------------------------------------------------------------------
# Layer 1
# -----------------------------------------------------------------------------


def layer1(cd) -> dict:
    """``chemicals`` on the pseudo-components (where it is importable)."""
    try:
        import chemicals
        from chemicals.acentric import LK_omega
        from chemicals.phase_change import Riedel
        from chemicals.vapor_pressure import Lee_Kesler
    except ImportError:
        return {}
    n = len(cd["names"])
    out = {"chemicals_version": chemicals.__version__, "Lee_Kesler_Psat": {}, "LK_omega": [],
           "Riedel_hvap": []}
    for T in (350.0, 450.0, 550.0, 650.0):
        out["Lee_Kesler_Psat"][str(T)] = [float(Lee_Kesler(T, cd["Tc"][i], cd["Pc"][i], cd["omega_vp"][i]))
                                         for i in range(n)]
    nle = cd["n_light_ends"]
    for i in range(n):
        out["LK_omega"].append(float(LK_omega(cd["Tb"][i], cd["Tc"][i], cd["Pc"][i])))
        out["Riedel_hvap"].append(float(Riedel(cd["Tb"][i], cd["Tc"][i], cd["Pc"][i])) if i >= nle else None)
    return out


# -----------------------------------------------------------------------------
# Layer 4
# -----------------------------------------------------------------------------


def layer4(cd, feed, props, ipopt, base_model) -> dict:
    """Central differences of the reference column in two spec values."""
    out = {}
    cases = {
        "d_diesel_api_d_diesel_yield": ("volume_yield", "diesel", 0.17, 2e-3,
                                        lambda s: s["products"]["diesel"]["API"]),
        "d_fired_duty_d_overflash": ("overflash", None, 0.05, 2e-3,
                                     lambda s: s["fired_duty"]),
        "d_condenser_duty_d_diesel_yield": ("volume_yield", "diesel", 0.17, 2e-3,
                                            lambda s: s["condenser_duty"]),
        "d_residue_api_d_overflash": ("overflash", None, 0.05, 2e-3,
                                      lambda s: s["products"]["residue"]["API"]),
    }
    for name, (kind, key, x0, h, f) in cases.items():
        vals = []
        for x in (x0 + h, x0 - h):
            over = {"volume_yield": {key: x}} if kind == "volume_yield" else {"overflash": x}
            M, status, _ = reference_column(cd, feed, props, ipopt, start=base_model, specs_over=over)
            if status != "optimal":
                raise RuntimeError(f"{name}: reference column did not solve at {x}: {status}")
            vals.append(f(column_summary(M, props)))
        out[name] = {"value": (vals[0] - vals[1]) / (2 * h), "x0": x0, "h": h,
                     "spec": kind if key is None else f"{kind}.{key}",
                     "f_plus": vals[0], "f_minus": vals[1]}
    return out


# -----------------------------------------------------------------------------


def provenance(ipopt) -> dict:
    import idaes
    import pyomo.version

    try:
        v = subprocess.run([ipopt, "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    except Exception:  # pragma: no cover
        v = "unknown"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    try:
        import chemicals

        chem = chemicals.__version__
    except ImportError:
        chem = None
    return {
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/generate.py",
        "difflow_commit": commit,
        "reference_simulator": (
            "IDAES generic property framework + a hand-written Pyomo MESH column solved by IPOPT. "
            "DWSIM (the issue's first choice), HYSYS and PRO/II were not available; IDAES is the "
            "independent open-source equation-oriented platform that was."),
        "idaes": idaes.__version__,
        "pyomo": pyomo.version.version,
        "ipopt": v,
        "chemicals": chem,
        "python": platform.python_version(),
        "property_methods": {
            "ideal": "Raoult K = Psat/P; Lee-Kesler (1975) Psat with the vapour-pressure acentric factor; "
                     "ideal-gas H from the cubic Cp, datum 298.15 K; liquid H = ideal-gas H - Watson "
                     "(1943) latent heat anchored at Tb (smoothed floor eps = 0.01 unless stated); water "
                     "vapour-only, Psat by IAPWS-IF97 region 4",
            "pr": "Peng-Robinson (1976), kij = 0, same Tc/Pc, omega = the characterisation's EOS "
                  "acentric factor, same ideal-gas Cp; hydrocarbons only",
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ipopt", default=DEFAULT_IPOPT)
    ap.add_argument("--chemicals", default=None, help="directory to put on sys.path for chemicals")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-fd", action="store_true")
    a = ap.parse_args(argv)
    if a.chemicals:
        sys.path.insert(0, a.chemicals)
    _quiet()

    import jax

    jax.config.update("jax_enable_x64", True)
    from . import case
    from . import formulas as fm

    unit, crude, thermo, _ = case.difflow_case()
    cd = case.component_data(crude, thermo)
    feed = np.array(case.feed_flows(unit, thermo))
    props = fm.Props(cd, eps=0.01)

    print("layer 3: reference column ...", flush=True)
    M, status, info = reference_column(cd, feed, props, a.ipopt)
    if status != "optimal":
        raise SystemExit(f"reference column failed from the independent start: {status}")
    base = column_summary(M, props)
    states = stage_states(M)
    print("  ", info, flush=True)

    print("layer 3: unsmoothed Watson ...", flush=True)
    Mx, sx, ix = reference_column(cd, feed, fm.Props(cd, eps=1e-6), a.ipopt, eps=1e-6, start=M)
    exact = column_summary(Mx, props) if sx == "optimal" else None

    l4 = {} if a.skip_fd else layer4(cd, feed, props, a.ipopt, M)
    print("layer 2 ...", flush=True)
    l2 = layer2(cd, feed, props, states, base["coil_outlet_T"],
                case.LAYOUT["P_top"] + (case.LAYOUT["feed_stage"] - 1)
                * (case.LAYOUT["P_bottom"] - case.LAYOUT["P_top"]) / (case.LAYOUT["n_stages"] - 1))
    print("layer 1 ...", flush=True)
    l1 = layer1(cd)

    data = {
        "provenance": provenance(a.ipopt),
        "case": {"bpd": case.BPD, "T_furnace_in": case.T_FURNACE_IN, "P_furnace_in": case.P_FURNACE_IN,
                 "furnace_efficiency": case.FURNACE_EFFICIENCY, "layout": case.LAYOUT,
                 "specs": case.SPECS, "feed_mol_s": feed.tolist(),
                 "initial_guess": INITIAL_GUESS},
        "components": cd,
        "layer1": l1,
        "layer2": l2,
        "layer3": {"solve": info, "column": base,
                   "watson_unsmoothed": {"solve": ix, "column": exact}},
        "layer4": l4,
    }
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
