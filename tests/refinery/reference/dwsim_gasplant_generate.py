"""Generate ``dwsim_gasplant_reference.json``: the gas plant's columns,
compressor train and vapour pressures in DWSIM 9.0.5.

Run from the repository root (needs DWSIM 9.0.5, .NET 8 and pythonnet; see
``scripts/install_dwsim.sh``)::

    PYTHONPATH=src:tests python -m refinery.reference.dwsim_gasplant_generate

For the cases of :mod:`.dwsim_gasplant_case`:

* **Columns, comparison (a)** -- DWSIM's rigorous ``DistillationColumn`` on
  "Peng-Robinson (PR)" with every component a hypothetical compound on
  difflow's constants, kij = 0 (like the IDAES reference), the liquid
  density from the EOS, flash loops at 1e-10; reflux and boilup ratios
  specified, total condenser at the bubble point, kettle reboiler,
  equilibrium trays, no pressure drop (:func:`.dwsim_lightends.column`).
  Recorded: every stage's T, P, L, V, x, y, K; the products; the duties;
  DWSIM's iteration counts and the column's component balance.
  The solver is the one that converges the case: Naphtali-Sandholm for the
  debutanizer (Wang-Henke converges it but fails DWSIM's own post-solve
  component-balance check, 1.8e-8 against 1e-8 at loop tolerance 1e-9),
  Wang-Henke for the C3/C4 and naphtha splitters (Naphtali-Sandholm had not
  finished either after four minutes; Wang-Henke takes 2-3 s).
* **Columns, comparison (b)** -- the debutanizer and the C3/C4 splitter on
  DWSIM's database compounds with DWSIM's own kij and package defaults
  except the liquid density (still the EOS): the model's sensitivity to
  the constants and binary parameters, not the implementation.
* **Compressor** -- inlet knock-out, two stages of compressor, cooler and
  knock-out drum (:func:`.dwsim_lightends.compressor_train`): (a) on
  difflow's constants and difflow's kij (H2S with C1-C3 nonzero), (b) on
  DWSIM's compounds and kij.
* **Vapour pressure** -- for each liquid of ``RVP``: DWSIM's bubble pressure
  at 100 F (the TVP), DWSIM's own RVP (its cold-flow utility's correlation
  on that TVP, :func:`.dwsim_lightends.dwsim_rvp_from_tvp`), and difflow's
  D323 construction rebuilt on DWSIM's flashes
  (:func:`.dwsim_lightends.d323_rvp`); (a) and (b).

It is an independent implementation of the same model in (a), and DWSIM's
model in (b). The only difflow calls are the constants
(:func:`.dwsim_gasplant_case.component_data`), an input to both sides,
frozen into the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dwsim_gasplant_reference.json"
PREFIX = "DF_"


def _hypos(comp):
    from .dwsim_session import HypoCompound

    return [HypoCompound(name=PREFIX + n, MW=comp["MW"][i], Tc=comp["Tc"][i], Pc=comp["Pc"][i],
                         omega=comp["omega"][i], cp_ig=comp["cp_ig"][i], Tb=comp["Tb"][i],
                         SG=comp["SG"][i]) for i, n in enumerate(comp["names"])]


def flowsheet(session, comp, mode: str):
    """(a) ``"same"``: hypos on difflow's constants and difflow's kij; (b)
    ``"dwsim"``: DWSIM's database compounds (real components only) and kij."""
    import numpy as np

    from difflow.dwsim_import import DWSIM_NAMES

    from . import dwsim_lightends as dl
    from .dwsim_session import EOS_ONLY

    if mode == "same":
        fs = session.flowsheet("PR", hypos=_hypos(comp), kij="zero", options=EOS_ONLY,
                               flash_tol=1e-10)
        if np.any(np.asarray(comp["kij"])):
            dl.set_kij(fs, comp["kij"])
        return fs
    if any(t is not None for t in comp["Tb"]):
        raise ValueError("mode 'dwsim' is for real components only")
    return session.flowsheet("PR", compounds=[DWSIM_NAMES[n] for n in comp["names"]],
                             kij="dwsim", options=EOS_ONLY, flash_tol=1e-10)


def run_column(session, case, comp, mode):
    from . import dwsim_lightends as dl

    fs = flowsheet(session, comp, mode)
    res = dl.column(fs, z=case["z"], F=case["F"], T=case["T"], P=case["P"],
                    n_trays=case["n_trays"], feed_tray=case["feed_tray"],
                    reflux_ratio=case["reflux_ratio"], boilup_ratio=case["boilup_ratio"],
                    solver=case["solver"])
    res["dwsim"] = fs.provenance()
    res["feed"] = dl.stream_state(fs, fs.stream)
    return res


def run_compressor(session, case, comp, mode):
    from . import dwsim_lightends as dl

    fs = flowsheet(session, comp, mode)
    res = dl.compressor_train(fs, z=case["z"], F=case["F"], T=case["T"], P=case["P"],
                              outlet_P=case["outlet_P"], n_stages=case["n_stages"],
                              efficiency=case["efficiency"], cooler_T=case["cooler_T"],
                              cooler_dP=case["cooler_dP"])
    res["dwsim"] = fs.provenance()
    return res


def run_rvp(session, case, comp, mode):
    from . import dwsim_lightends as dl

    fs = flowsheet(session, comp, mode)
    d323 = dl.d323_rvp(fs, case["z"])
    return {"tvp": d323["tvp"], "rvp_dwsim_correlation": dl.dwsim_rvp_from_tvp(d323["tvp"]),
            "d323": d323, "dwsim": fs.provenance()}


def provenance(session) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    p = session.provenance()
    p.update({
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/dwsim_gasplant_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.dwsim_gasplant_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            f"{p['dwsim']} 'Peng-Robinson (PR)': rigorous DistillationColumn (Naphtali-Sandholm "
            "or Wang-Henke, loop tolerance 1e-9), Compressor (adiabatic, outlet pressure) + "
            "Cooler + Vessel, flashes at 1e-10. (a) every component a hypothetical compound on "
            "difflow's constants and kij, liquid density from the EOS; (b) DWSIM's database "
            "compounds and kij."),
    })
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from . import dwsim_gasplant_case as dc
    from .dwsim_session import DWSIMSession

    session = DWSIMSession()
    data = {"provenance": provenance(session), "columns": {}, "compressor": {}, "rvp": {}}
    for name, case in dc.COLUMNS.items():
        comp = dc.component_data(case["names"], case["pseudo"])
        entry = {"case": case, "components": comp, "same": run_column(session, case, comp, "same")}
        print("column", name, "(a) done", flush=True)
        if not case["pseudo"]:
            entry["dwsim_data"] = run_column(session, case, comp, "dwsim")
            print("column", name, "(b) done", flush=True)
        data["columns"][name] = entry
    c = dc.COMPRESSOR
    comp = dc.component_data(c["names"])
    data["compressor"] = {"case": c, "components": comp,
                          "same": run_compressor(session, c, comp, "same"),
                          "dwsim_data": run_compressor(session, c, comp, "dwsim")}
    print("compressor done", flush=True)
    for name, case in dc.RVP.items():
        comp = dc.component_data(case["names"], case["pseudo"])
        entry = {"case": case, "components": comp, "same": run_rvp(session, case, comp, "same")}
        if not case["pseudo"]:
            entry["dwsim_data"] = run_rvp(session, case, comp, "dwsim")
        data["rvp"][name] = entry
        print("rvp", name, "done", flush=True)
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
