"""Generate ``vdu_reference.json``: the vacuum column's independent reference (#294).

Run from the repository root (needs Pyomo and an IPOPT; IDAES's build is
the one the committed file was made with)::

    PYTHONPATH=src:tests python -m refinery.reference.vdu_generate [--ipopt PATH]

What it computes:

* **The column** -- :mod:`.vdu_mesh` (Pyomo + IPOPT) on the case of
  :mod:`.vdu_case`, from :func:`.vdu_mesh.engineering_guess` (the feed flash
  and the specs; nothing of difflow's solution), with difflow's four
  Watson-K passes and blended Maxwell-Bonnell branches -- the model difflow
  states.
* **What the model's two numerical choices cost** -- the same column with
  the Watson-K correction solved to its exact fixed point, and with
  Maxwell-Bonnell's published piecewise branches instead of the blend (both
  warm-started from the column above, which is the reference's own answer).
* **Murphree efficiencies** -- a second column with every bed below 100 %,
  on :data:`.vdu_case.EFFICIENCY_SPECS` (the 450 C LVGO end point is
  unreachable with beds this poor, in both models). Reached by continuation
  from the reference's own equilibrium-bed answer -- the efficiencies and the
  end point moved a quarter of the way to their targets per warm solve --
  because a cold start dries the wash bed. Nothing of difflow's enters.
* **Sensitivities** -- central differences of the reference column in the
  furnace outlet temperature, the flash-zone pressure and the stripping
  steam rate, each point re-solved warm by IPOPT.

The only difflow call is :func:`.vdu_case.characterize`: the pseudo-component
table and the residue feed, an input to both models, frozen into the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "vdu_reference.json"
DEFAULT_IPOPT = os.path.expanduser("~/.idaes/bin/ipopt")

#: Central-difference steps for the sensitivities: large enough that the
#: solve tolerance (1e-10) is negligible over 2h, small enough that the
#: O(h^2) truncation is well inside the tests' 1e-4. A 5 % steam step
#: (2.5e-4) left 5e-4 of truncation in the HVGO rate; 0.5 % does not.
FD_STEPS = {"furnace_T": 0.25, "flash_zone_P": 10.0, "steam_per_feed": 2.5e-5}
#: Fractions of the way from equilibrium beds to EFFICIENCY_CASE.
CONTINUATION = (0.25, 0.5, 0.75, 1.0)
#: The outputs whose sensitivities are recorded (difflow output names). Not
#: the slop: the overflash spec holds it, so its derivative is zero.
FD_OUTPUTS = ("lvgo.rate", "hvgo.rate", "residue.rate", "lvgo_pa.duty",
              "hvgo_pa.duty", "furnace.duty", "flash_zone.T", "hvgo.T95")


def case_numbers(comp, feed, specs=None, **knobs):
    """The plain numbers :func:`.vdu_mesh.build` takes, from :mod:`.vdu_case`."""
    from . import vdu_case as vc
    from . import vdu_formulas as vf

    lay = dict(vc.LAYOUT, **knobs)
    sp = dict(vc.SPECS, **(specs or {}))
    F = vc.feed_mass(comp, feed)
    return lay, {
        "feed_T": vc.CRUDE["feed_T"], "furnace_T": lay["furnace_T"],
        "furnace_P": lay["flash_zone_P"] + lay["transfer_line_dP"],
        "steam_T": lay["steam_T"], "steam_mol_s": lay["steam_per_feed"] * F * 1000.0 / vf.MW_WATER,
        "top_T": sp["top.T"], "overflash": sp["overflash"], "lvgo_T95": sp["lvgo.T95"],
        "lvgo_pa_kg_s": lay["lvgo_pa_per_feed"] * F, "hvgo_pa_kg_s": lay["hvgo_pa_per_feed"] * F,
    }


def reference_column(comp, feed, ipopt, n_pass=4, published=False, efficiency=None,
                     specs=None, start=None, **knobs):
    """Build and solve the reference column; returns ``(model, status, info)``.

    With ``start`` (a solved model of the same structure) it is warm-started
    from that and solved once; otherwise it starts from the engineering
    guess and takes :func:`.vdu_mesh.solve_staged`'s two steps.
    """
    from . import vdu_case as vc
    from . import vdu_mesh as vm

    lay, case = case_numbers(comp, feed, specs, **knobs)
    eff = dict(vc.EFFICIENCY, **(efficiency or {}))
    net = vm.Network.vacuum(lay, eff, case["steam_mol_s"])
    M = vm.build(comp, net, np.asarray(feed), case, n_pass=n_pass, published=published)
    if start is not None and (n_pass is None) == (start._meta["n_pass"] is None):
        vm.copy_values(start, M)
        status, res = vm.solve(M, ipopt)
        info = {"status": status, "start": "warm"}
    elif start is not None:
        # a different pass structure: copy what matches, then re-evaluate the passes
        vm.copy_values(start, M)
        for j in range(net.N):
            for i in range(len(feed)):
                vm.set_passes(M, j, i, M.T[j].value)
        status, res = vm.solve(M, ipopt)
        info = {"status": status, "start": "warm"}
    else:
        vm.initialise(M)
        status, info = vm.solve_staged(M, ipopt)
        info["start"] = "engineering_guess"
    return M, status, info


def column_summary(M) -> dict:
    """The reference's answer, keyed by difflow's output names where it has one."""
    from . import vdu_mesh as vm

    r = vm.results(M)
    net = M._meta["net"]
    out = {f"{k}.rate": v for k, v in r["rates"].items()}
    out.update({f"{p}.duty": r["pumparound_duty"][p] for p in net.pumparounds})
    out.update({f"{p}.return_T": r["pumparound_return_T"][p] for p in net.pumparounds})
    out.update({f"stage{j}.T": t for j, t in enumerate(r["T"])})
    out["top.T"] = r["T"][0]
    out["flash_zone.T"] = r["T"][net.feed_stage]
    out["furnace.duty"] = r["furnace_duty"]
    out["furnace.vapor_fraction"] = r["furnace_vapor_fraction"]
    out["feed.rate"] = r["feed_rate"]
    out["overflash"] = r["overflash"]
    for p, ins in r["inspection"].items():
        out[f"{p}.sg"] = ins["SG"]
        for pct, T in ins["TBP"].items():
            out[f"{p}.T{int(pct):02d}"] = T
    return {"outputs": out, "T": r["T"], "P": r["P"], "L": r["L"], "V": r["V"],
            "products_mol_s": r["products_mol_s"]}


def sensitivities(comp, feed, ipopt, base) -> dict:
    """Central differences of the reference column, warm-started from ``base``."""
    from . import vdu_case as vc

    out = {}
    for knob, h in FD_STEPS.items():
        x0 = vc.LAYOUT[knob]
        vals = []
        for x in (x0 + h, x0 - h):
            M, status, _ = reference_column(comp, feed, ipopt, start=base, **{knob: x})
            if status != "optimal":
                raise SystemExit(f"finite-difference solve failed at {knob}={x}: {status}")
            vals.append(column_summary(M)["outputs"])
        out[knob] = {"x0": x0, "h": h,
                     "d": {o: (vals[0][o] - vals[1][o]) / (2 * h) for o in FD_OUTPUTS}}
        print("  ", knob, {o: f"{v:.6g}" for o, v in out[knob]["d"].items()}, flush=True)
    return out


def provenance(ipopt) -> dict:
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
        import idaes

        idaes_v = idaes.__version__
    except ImportError:  # pragma: no cover
        idaes_v = None
    import pyomo.version

    return {
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/vdu_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.vdu_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            "A hand-written Pyomo MESH model of the vacuum column (tests/refinery/reference/"
            "vdu_mesh.py) solved by IPOPT, on a property model transcribed from the published "
            "correlations (vdu_formulas.py). DWSIM (the issue's first choice) was not available, "
            "and IDAES has no vacuum-column model with pumparounds, wash bed and Murphree beds, "
            "so the column is written in Pyomo, IDAES's modelling layer, and solved by IDAES's "
            "IPOPT build."),
        "idaes": idaes_v,
        "pyomo": pyomo.version.version,
        "ipopt": v,
        "ipopt_options": {"tol": 1e-10, "max_iter": 3000, "nlp_scaling_method": "gradient-based",
                          "bound_push": 1e-8, "mu_init": 1e-3},
        "python": platform.python_version(),
        "property_methods": {
            "K": "Raoult, K = Psat/P; Maxwell-Bonnell (1957) Psat, MNL50 Eqs. 7.22-7.26 in "
                 "Rankine, Watson-K correction by 4 passes from 760 mmHg, branches blended "
                 "over Q-width 1.5e-5 with the upper join at Q = 1/748.1 (difflow's stated model)",
            "liquid_H": "Kesler-Lee (1976) liquid Cp, MNL50 Eq. 7.49, integrated from 0 R",
            "vapour_H": "liquid H + Clausius-Clapeyron heat of vaporisation on the same Psat",
            "steam": "NIST Shomate (water vapour, 500-1700 K) with its own F - H datum",
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ipopt", default=DEFAULT_IPOPT)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-fd", action="store_true")
    ap.add_argument("--skip-variants", action="store_true")
    a = ap.parse_args(argv)

    import logging

    logging.getLogger("pyomo").setLevel(logging.ERROR)
    import jax

    jax.config.update("jax_enable_x64", True)
    from . import vdu_case as vc

    comp, feed = vc.characterize()
    feed = np.asarray(feed)

    print("column (equilibrium beds) ...", flush=True)
    M, status, info = reference_column(comp, feed, a.ipopt)
    if status != "optimal":
        raise SystemExit(f"reference column failed from the engineering guess: {status}")
    base = column_summary(M)
    print("  ", info, flush=True)

    variants = {}
    if not a.skip_variants:
        for name, kw in (("exact_fixed_point", {"n_pass": None}),
                         ("published_branches", {"published": True})):
            print(f"variant {name} ...", flush=True)
            Mv, sv, iv = reference_column(comp, feed, a.ipopt, start=M, **kw)
            variants[name] = {"solve": iv, "column": column_summary(Mv) if sv == "optimal" else None}
            print("  ", iv, flush=True)

    print("column (Murphree beds, continuation in efficiency) ...", flush=True)
    Me, ie = M, {"start": "continuation from the equilibrium-bed reference", "steps": []}
    for s in CONTINUATION:
        eff = {k: 1.0 - s * (1.0 - v) for k, v in vc.EFFICIENCY_CASE.items()}
        t95 = vc.SPECS["lvgo.T95"] + s * (vc.EFFICIENCY_SPECS["lvgo.T95"] - vc.SPECS["lvgo.T95"])
        Me, se, step = reference_column(comp, feed, a.ipopt, efficiency=eff,
                                        specs={"lvgo.T95": t95}, start=Me)
        ie["steps"].append({"fraction": s, **step})
        if se != "optimal":
            raise SystemExit(f"Murphree continuation failed at fraction {s}: {se}")
    ie["status"] = se
    print("  ", ie, flush=True)

    sens = {}
    if not a.skip_fd:
        print("sensitivities ...", flush=True)
        sens = sensitivities(comp, feed, a.ipopt, M)

    data = {
        "provenance": provenance(a.ipopt),
        "case": {"crude": vc.CRUDE, "layout": vc.LAYOUT, "specs": vc.SPECS,
                 "efficiency": vc.EFFICIENCY, "efficiency_case": vc.EFFICIENCY_CASE,
                 "efficiency_specs": vc.EFFICIENCY_SPECS,
                 "feed_mol_s": feed.tolist()},
        "components": comp,
        "column": {"solve": info, **base},
        "variants": variants,
        "murphree": {"solve": ie, **column_summary(Me)},
        "sensitivities": {"steps": FD_STEPS, **sens},
    }
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
