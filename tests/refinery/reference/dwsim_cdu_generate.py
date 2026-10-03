"""Generate ``dwsim_cdu_reference.json``: difflow_refinery's crude side against
DWSIM 9.0.5 -- characterization, thermodynamics on the crude, the
atmospheric column, and the vacuum feed flash.

Run from the repository root (needs DWSIM 9.0.5, .NET 8 and pythonnet; see
``scripts/install_dwsim.sh``)::

    PYTHONPATH=src:tests python -m refinery.reference.dwsim_cdu_generate [--only char,thermo,column,vacuum]

``--only`` regenerates the named sections and keeps the others from the
existing file (the column takes several minutes; the rest seconds).

What each section records (the cases are :mod:`.dwsim_cdu_case`):

* ``characterization`` -- per assay, DWSIM's distillation-curve
  characterization (:func:`.dwsim_columns.distcurve_characterization`) on
  difflow's cut temperatures, with DWSIM's default options and with its
  closest to difflow's; DWSIM's correlations at difflow's own (Tb, SG)
  (:func:`.dwsim_columns.property_methods`); and difflow's tables, frozen.
* ``thermo`` -- (a) DWSIM "Raoult's Law" with the ideal options on difflow's
  constants (every component a :class:`.dwsim_columns.FlatCompound`): pure
  Psat, Cp and Hvap, bubble and dew points, the flash zone, the heating
  path; (b) DWSIM PR, Grayson-Streed, Lee-Kesler-Plocker and its own Raoult
  on DWSIM's own characterization of the same crude (the test crude's
  default run, plus DWSIM's database propane, n-butane, n-pentane), the same
  points.
* ``column`` -- :data:`.dwsim_cdu_case.COLUMN_A` in DWSIM's rigorous column
  (Naphtali-Sandholm) on the same model and constants as difflow.
* ``vacuum`` -- the CDU residue of ``cdu_reference.json`` flashed at vacuum
  flash-zone conditions on both models.

The only difflow calls are the characterizations and the CDU case's
constants and feed: inputs to both sides, frozen into the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "dwsim_cdu_reference.json"
PREFIX = "DF_"
SECTIONS = ("characterization", "thermo", "column", "vacuum")


# ---------------------------------------------------------------------------
# 1. Characterization
# ---------------------------------------------------------------------------

def run_characterization(session, df_tables) -> dict:
    from . import dwsim_cdu_case as dc
    from .dwsim_columns import distcurve_characterization, property_methods

    out = {}
    for key, spec in (("test_crude", dc.TEST_CRUDE), ("heavy_crude", dc.HEAVY_CRUDE)):
        tw = df_tables[key]["twu"]
        edges = tw["cut_edges"]
        T_K = [t + 273.15 for t in spec["T_C"]]
        cum = [p / 100.0 for p in spec["pct"]]
        sg = spec.get("sg") or 141.5 / (spec["api"] + 131.5)
        basis = {"volume": "liquid_volume", "mass": "mass"}[spec["basis"]]
        # DWSIM's cuts start at the curve's first point and end at its last:
        # the interior edges are difflow's, from where difflow's cuts start
        # (above the light ends) to the last edge below the final data point
        cuts = [e for e in edges[:-1] if T_K[0] < e < T_K[-1]]
        runs = {}
        for name, opts in dc.DWSIM_CHAR_RUNS.items():
            runs[name] = distcurve_characterization(session, T_K, cum, sg, cuts, basis=basis, **opts)
        n_pc = len(tw["Tb"]) - (1 if tw["residue_lump"] else 0)
        pm = property_methods(session, tw["Tb"][:n_pc], tw["SG"][:n_pc], dc.T_CP)
        out[key] = {"cut_temps_K": cuts, "sg_bulk": sg, "basis": basis, "runs": runs,
                    "property_methods_at_difflow_cuts": pm}
    return out


# ---------------------------------------------------------------------------
# 2. Thermodynamics on the crude
# ---------------------------------------------------------------------------

def flat_compounds(comp) -> list:
    """difflow's constants as DWSIM :class:`FlatCompound` s."""
    from .dwsim_columns import FlatCompound
    from .dwsim_session import lee_kesler_eq101

    out = []
    for i, n in enumerate(comp["names"]):
        out.append(FlatCompound(
            name=PREFIX + n, MW=comp["MW"][i], Tc=comp["Tc"][i], Pc=comp["Pc"][i],
            omega=comp["omega_vp"][i], Tb=comp["Tb"][i], SG=comp["SG"][i],
            cp_ig=comp["cp_ig"][i],
            psat_eq101=lee_kesler_eq101(comp["Tc"][i], comp["Pc"][i], comp["omega_vp"][i]),
            hvap_A=comp["hvap_A"][i], hvap_n=0.38))
    return out


def _points(fs, z, flows_total, label="") -> dict:
    import time

    from . import dwsim_cdu_case as dc

    out = {"bubble": [], "dew": [], "path": [], "flash_zone": None}
    for P in dc.BUBBLE_DEW_P:
        for kind, vf in (("bubble", 0.0), ("dew", 1.0)):
            t0 = time.time()
            try:
                r = fs.flash_pvf(z, P, vf)
                out[kind].append({"P": P, "T": r["T"], "vapor_fraction": r["vapor_fraction"],
                                  "residual": r["equilibrium_residual"],
                                  "seconds": round(time.time() - t0, 2)})
            except Exception as e:  # noqa: BLE001 - recorded, not fatal
                out[kind].append({"P": P, "T": None, "error": str(e)[:300]})
            print(f"  {label} {kind} {P:.0f} Pa: {out[kind][-1].get('T')} "
                  f"({time.time() - t0:.1f} s)", flush=True)
    for T, P in dc.HEAT_PATH:
        r = fs.flash_tp(z, T, P)
        out["path"].append({"T": T, "P": P, "vapor_fraction": r["vapor_fraction"], "h": r["h"],
                            "h_vap": r["h_vap"], "h_liq": r["h_liq"],
                            "rho_liq": r["rho_liq"], "MW_liq": r["MW_liq"],
                            "x": r["x"], "y": r["y"],
                            "residual": r["equilibrium_residual"], **_h_direct(fs, r)})
    r = fs.flash_tp(z, dc.T_COT, dc.P_FZ)
    out["flash_zone"] = {k: r[k] for k in ("T", "P", "vapor_fraction", "h", "h_vap", "h_liq", "x",
                                           "y", "K", "equilibrium_residual", "MW_vap", "MW_liq")}
    h_in, h_out = out["path"][0]["h"], out["path"][-1]["h"]
    out["furnace_duty_W"] = flows_total * (h_out - h_in)
    hd_in, hd_out = out["path"][0]["h_direct"], out["path"][-1]["h_direct"]
    out["furnace_duty_direct_W"] = flows_total * (hd_out - hd_in)
    return out


def _h_direct(fs, r) -> dict:
    """The flash's mixture enthalpy from DWSIM's DW_CalcEnthalpy on each
    phase (what the column solver uses), beside the stream's own number."""
    from .dwsim_columns import phase_enthalpy

    vf = r["vapor_fraction"]
    hv = phase_enthalpy(fs, r["y"], r["T"], r["P"], "vapor") if r["y"] is not None else 0.0
    hl = phase_enthalpy(fs, r["x"], r["T"], r["P"], "liquid") if r["x"] is not None else 0.0
    return {"h_vap_direct": hv, "h_liq_direct": hl, "h_direct": vf * hv + (1.0 - vf) * hl}


def run_thermo(session, comp, flows, char_default) -> dict:
    from difflow.dwsim_import import DWSIM_NAMES

    from . import dwsim_cdu_case as dc
    from .dwsim_columns import no_ideal_fallback, pf_flowsheet
    from .dwsim_session import IDEAL_RAOULT

    flows = np.asarray(flows, float)
    z = flows / flows.sum()
    out = {}
    # (a) the same model and constants
    fs = no_ideal_fallback(session.flowsheet("RAOULT", hypos=flat_compounds(comp),
                                             options=IDEAL_RAOULT))
    a = _points(fs, z, flows.sum(), "same model")
    a["pure"] = {n: {str(T): fs.pure(PREFIX + n, T) for T in (350.0, 450.0, 550.0, 650.0)}
                 for n in comp["names"]}
    a["dwsim"] = fs.provenance()
    out["same_model"] = a

    # (b) DWSIM's own characterization of the same crude on DWSIM's packages:
    # DWSIM's cuts above difflow's first edge (its first cut is the light-ends
    # region of the curve) as DWSIM petroleum fractions, plus DWSIM's
    # database propane, n-butane, n-pentane. The feed by standard volume:
    # light ends at the assay's vol %, each DWSIM cut at its share of DWSIM's
    # own fitted curve; mass by SG (light ends: difflow's standard SGs), moles
    # by MW; the same total mass flow as difflow's feed.
    nle = comp["n_light_ends"]
    le = comp["names"][:nle]
    le_dw = [DWSIM_NAMES[n] for n in le]
    cuts = char_default["cuts"][1:]
    rows = char_default["rows"][1:]
    pfs = char_default["_constant_properties"][1:]
    vol = np.asarray([dc.TEST_CRUDE["light_ends"][n] / 100.0 for n in le]
                     + [c["fvf"] - c["fv0"] for c in cuts])
    sgs = np.asarray(comp["SG"][:nle] + [r["SG"] for r in rows])
    mws = np.asarray(comp["MW"][:nle] + [r["MW"] for r in rows])
    mass = vol * sgs
    mass_total = float(np.dot(flows, comp["MW"]))                 # g/s
    F_dw = mass / mass.sum() * mass_total / mws                     # mol/s
    own = {"feed": {"names": le_dw + [r["name"] for r in rows], "flows": F_dw.tolist(),
                    "volume_fraction": (vol / vol.sum()).tolist(),
                    "mass_flow_g_s": mass_total}, "packages": {}}
    for pkg in dc.MODEL_PACKAGES:
        try:
            # a looser loop than the harness's 1e-10 / 1000: a PR dew point of
            # the whole crude otherwise runs for many minutes
            fs2 = no_ideal_fallback(pf_flowsheet(session, pkg, compounds=le_dw, pf=pfs,
                                                 kij="dwsim", flash_tol=1e-8, max_iter=100))
            res = _points(fs2, F_dw / F_dw.sum(), F_dw.sum(), pkg)
            # one fraction's pure-component functions as this package sees them
            pf_name = rows[len(rows) // 2]["name"]
            res["pf_pure"] = {"name": pf_name,
                              **{str(T): fs2.pure(pf_name, T) for T in (400.0, 500.0)}}
            res["dwsim"] = fs2.provenance()
            own["packages"][pkg] = res
        except Exception as e:  # noqa: BLE001 - recorded: what DWSIM could not do
            own["packages"][pkg] = {"error": str(e)[:500]}
    out["dwsim_own"] = own
    return out


# ---------------------------------------------------------------------------
# 3. The column
# ---------------------------------------------------------------------------

def run_column(session, comp) -> dict:
    """:data:`.dwsim_cdu_case.COLUMN_SMALL` in DWSIM's rigorous column, on the
    same model and constants as difflow. The starting point owes nothing to
    difflow's solution: DWSIM's own estimates, and bottom-stage temperatures
    2 and 4 K below the feed's for the secant that holds the bottom stage
    adiabatic (:meth:`.dwsim_columns.DWSIMColumn.solve_adiabatic`).

    :data:`.dwsim_cdu_case.COLUMN_A` (the CDU crude, 28 components) is not
    run: no configuration of DWSIM's column solved it
    (:data:`.dwsim_cdu_case.COLUMN_A_ATTEMPTS`, copied into the file)."""
    from . import dwsim_cdu_case as dc
    from .dwsim_columns import DWSIMColumn, no_ideal_fallback
    from .dwsim_session import IDEAL_RAOULT

    cfg = dc.COLUMN_SMALL
    idx = [comp["names"].index(n) for n in cfg["components"]]
    sub = {k: [comp[k][i] for i in idx] for k in ("names", "MW", "Tc", "Pc", "omega_vp", "Tb",
                                                  "SG", "cp_ig", "hvap_A")}
    fs = no_ideal_fallback(session.flowsheet("RAOULT", hypos=flat_compounds(sub),
                                             options=IDEAL_RAOULT))
    col = DWSIMColumn(fs, cfg["n_stages"], cfg["P_top"], cfg["P_bottom"], cfg["P_condenser"],
                      solver="naphtali-sandholm")
    col.add_feed("FEED", cfg["feed_stage"], [cfg["feed_each"]] * len(idx), cfg["T_feed"],
                 cfg["P_feed"])
    for name, stage, rate in cfg["side_draws"]:
        col.add_side_draw(name, stage, rate)
    col.connect_products()
    T0, T1 = cfg["T_feed"] - 2.0, cfg["T_feed"] - 4.0
    col.specs(cfg["distillate"], T0)
    res = col.solve_adiabatic(T0, T1, duty_tol=1.0)
    res["dwsim"] = col.provenance()
    res["package"] = {k: v for k, v in fs.provenance().items() if k != "constants"}
    res["products"] = {{"distillate": "naphtha", "bottoms": "residue"}.get(k, k): v
                       for k, v in res.get("products", {}).items()}
    return {"small": res, "A_attempts": dc.COLUMN_A_ATTEMPTS}


# ---------------------------------------------------------------------------
# 4. The vacuum feed
# ---------------------------------------------------------------------------

def run_vacuum(session, comp, residue) -> dict:
    from . import dwsim_cdu_case as dc
    from .dwsim_session import EOS_ONLY, IDEAL_RAOULT, HypoCompound

    residue = np.asarray(residue, float)
    z = residue / residue.sum()
    out = {"feed_flows": residue.tolist()}
    fs = session.flowsheet("RAOULT", hypos=flat_compounds(comp), options=IDEAL_RAOULT)
    pr = session.flowsheet("PR", hypos=[HypoCompound(name=PREFIX + n, MW=comp["MW"][i],
                                                      Tc=comp["Tc"][i], Pc=comp["Pc"][i],
                                                      omega=comp["omega_eos"][i],
                                                      cp_ig=comp["cp_ig"][i])
                                         for i, n in enumerate(comp["names"])],
                           kij="zero", options=EOS_ONLY)
    for key, f in (("same_model", fs), ("pr_same_constants", pr)):
        pts = []
        for T, P in dc.VACUUM_FLASH:
            r = f.flash_tp(z, T, P)
            pts.append({k: r[k] for k in ("T", "P", "vapor_fraction", "h", "x", "y", "K",
                                          "equilibrium_residual")})
        prov = f.provenance()
        out[key] = {"points": pts, "dwsim": {k: prov[k] for k in ("property_package", "options",
                                                                  "kij", "flash_settings")}}
    return out


# ---------------------------------------------------------------------------

def provenance(session) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    p = session.provenance()
    p.update({
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/dwsim_cdu_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.dwsim_cdu_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            f"{p['dwsim']}: the distillation-curve petroleum characterization (run headless), "
            "the Raoult's Law package with the ideal options on difflow's constants, PR / "
            "Grayson-Streed / Lee-Kesler-Plocker / Raoult on DWSIM's own fractions, and the "
            "rigorous column (refluxed absorber, Naphtali-Sandholm)."),
    })
    return p


def _clean(obj):
    """Drop the .NET handles (keys starting with ``_``) before JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", default=",".join(SECTIONS))
    a = ap.parse_args()
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    for s in only:
        if s not in SECTIONS:
            raise SystemExit(f"unknown section {s!r}; choose from {SECTIONS}")

    import jax

    jax.config.update("jax_enable_x64", True)
    from . import case
    from . import dwsim_cdu_case as dc
    from .dwsim_session import DWSIMSession

    old = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    session = DWSIMSession()
    unit, crude, thermo, Vf, flows = dc.difflow_crude()
    comp = case.component_data(crude, thermo)
    comp["hvap_A"] = [float(v) for v in np.asarray(thermo.hvap_A)]
    df_tables = dc.difflow_characterizations()
    cdu_ref = json.loads((HERE / "cdu_reference.json").read_text())
    residue = cdu_ref["layer3"]["column"]["product_flows"]["residue"]

    data = {"provenance": provenance(session),
            "inputs": {"components": comp, "feed_flows": [float(v) for v in flows],
                       "difflow_characterization": df_tables,
                       "test_crude": dc.TEST_CRUDE, "heavy_crude": dc.HEAVY_CRUDE,
                       "column_A": dc.COLUMN_A, "column_small": dc.COLUMN_SMALL, "char_runs": dc.DWSIM_CHAR_RUNS,
                       "bubble_dew_P": list(dc.BUBBLE_DEW_P), "heat_path": dc.HEAT_PATH,
                       "T_cot": dc.T_COT, "P_fz": dc.P_FZ, "vacuum_flash": dc.VACUUM_FLASH,
                       "vacuum_feed_source": "cdu_reference.json layer3 product_flows.residue",
                       "T_cp": list(dc.T_CP)}}
    for s in SECTIONS:
        if s not in only and s in old:
            data[s] = old[s]
    char = None
    if "characterization" in only or "thermo" in only:
        print("characterization", flush=True)
        char = run_characterization(session, df_tables)
        if "characterization" in only:
            data["characterization"] = char
    if "thermo" in only:
        print("thermo", flush=True)
        data["thermo"] = run_thermo(session, comp, flows,
                                    char["test_crude"]["runs"]["default"])
    if "vacuum" in only:
        print("vacuum", flush=True)
        data["vacuum"] = run_vacuum(session, comp, residue)
    if "column" in only:
        print("column (minutes)", flush=True)
        data["column"] = run_column(session, comp)
    gen = data["provenance"]["generated"]
    sections = dict(old.get("provenance", {}).get("sections_generated", {}))
    sections.update({s: gen for s in only})
    data["provenance"]["sections_generated"] = sections
    Path(a.out).write_text(json.dumps(_clean(data), indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
