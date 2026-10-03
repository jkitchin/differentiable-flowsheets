"""Generate ``dwsim_smoke_reference.json``: DWSIM 9.0.5 PR flashes on difflow's
constants -- the end-to-end check of the DWSIM harness (:mod:`.dwsim_session`).

Run from the repository root (needs DWSIM 9.0.5, .NET 8 and pythonnet; see
``scripts/install_dwsim.sh``)::

    PYTHONPATH=src:tests python -m refinery.reference.dwsim_smoke_generate

For each case of :mod:`.dwsim_smoke_case` it builds a DWSIM flowsheet on
"Peng-Robinson (PR)" in which **every** component is a hypothetical
compound carrying difflow's MW, Tc, Pc, omega and ideal-gas Cp cubic
(:meth:`.dwsim_session.HypoCompound`), with DWSIM's binary parameters
removed (``kij = 0``), the liquid density taken from the EOS without
Peneloux translation (:data:`.dwsim_session.EOS_ONLY`) and the flash loops
tightened to 1e-10. Recorded, per point: DWSIM's PT flash (vapor fraction,
phase compositions, K = y/x, molar enthalpies of the phases and the mixture,
phase densities and compressibility factors), a PH flash back from the PT
point's enthalpy, and DWSIM's ideal-gas Cp and vapour pressure of each
component (the hypo plumbing: Cp must be difflow's cubic; Psat is the
Lee-Kesler seed, informational).

It is an independent *implementation* of the same model, not an
independent model. The only difflow call is
:func:`.dwsim_smoke_case.component_data` -- the constants, an input to both,
frozen into the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dwsim_smoke_reference.json"
#: Prefix for DWSIM hypo names, so none collides with a database compound.
PREFIX = "DF_"


def run_case(session, case: dict, comp: dict) -> dict:
    from .dwsim_session import EOS_ONLY, HypoCompound

    hypos = [HypoCompound(name=PREFIX + n, MW=comp["MW"][i], Tc=comp["Tc"][i], Pc=comp["Pc"][i],
                          omega=comp["omega"][i], cp_ig=comp["cp_ig"][i], Tb=comp["Tb"][i],
                          SG=comp["SG"][i])
             for i, n in enumerate(comp["names"])]
    fs = session.flowsheet("PR", hypos=hypos, kij="zero", options=EOS_ONLY, flash_tol=1e-10)
    out = {"tp": [], "ph": [], "pure": {}, "dwsim": fs.provenance()}
    for T, P in case["tp"]:
        out["tp"].append(fs.flash_tp(case["z"], T, P))
    for spec in case["ph"]:
        at = fs.flash_tp(case["z"], spec["T_from"], spec["P"])
        back = fs.flash_ph(case["z"], spec["P"], at["h"])
        out["ph"].append({"P": spec["P"], "h": at["h"], "T_from": spec["T_from"], "result": back})
    for n in comp["names"]:
        out["pure"][n] = {str(T): fs.pure(PREFIX + n, T) for T in (300.0, 450.0, 600.0)}
    return out


def provenance(session) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    p = session.provenance()
    p.update({
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/dwsim_smoke_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.dwsim_smoke_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            f"{p['dwsim']} 'Peng-Robinson (PR)' property package (1976 kappa, R = 8.314), "
            "Universal Flash over Nested Loops with loop tolerances 1e-10, every component a "
            "hypothetical compound on difflow's constants, kij removed, liquid density from the "
            "EOS without Peneloux. The same model as difflow's, implemented independently."),
    })
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from . import dwsim_smoke_case as sc
    from .dwsim_session import DWSIMSession

    session = DWSIMSession()
    data = {"provenance": provenance(session), "cases": {}}
    for name, case in sc.CASES.items():
        comp = sc.component_data(case)
        print("flashing", name, flush=True)
        data["cases"][name] = {"case": case, "components": comp,
                               "dwsim": run_case(session, case, comp)}
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
