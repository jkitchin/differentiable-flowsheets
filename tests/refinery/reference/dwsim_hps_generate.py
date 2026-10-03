"""Generate ``dwsim_hps_reference.json``: a hydrotreater's high-pressure
separator flashed by DWSIM 9.0.5.

Run from the repository root (needs DWSIM 9.0.5, .NET 8 and pythonnet; see
``scripts/install_dwsim.sh``; the hydrotreater solve inside takes about two
minutes)::

    PYTHONPATH=src:tests python -m refinery.reference.dwsim_hps_generate

The feed is the reactor effluent of the diesel hydrotreater
(:func:`.dwsim_hps_case.effluent`), frozen into the file with the component
table, and flashed at :data:`.dwsim_hps_case.POINTS` by DWSIM in four
set-ups (see :mod:`.dwsim_hps_case`):

* ``same`` -- comparison (a): "Peng-Robinson 1978 (PR78)", every component
  a hypothetical compound on difflow's constants, difflow's kij, liquid
  density from the EOS. DWSIM's PR78 (``ThermoPlugs.PR78``, read from its
  IL) switches to the 1978 kappa above omega 0.491 as difflow does, but
  writes the 1976 branch's ``1.54226`` as ``1.5422``; its fugacity routine
  has the full-precision sqrt(2) constants and R = 8.314.
* ``pr76`` -- the same on "Peng-Robinson (PR)", whose kappa is the 1976
  quadratic for every omega.
* ``dwsim_compounds`` / ``dwsim_data`` -- comparison (b): the gases from
  DWSIM's database (``difflow.dwsim_import.DWSIM_NAMES``) with difflow's
  kij, then with DWSIM's own; the cuts stay hypos (DWSIM has no kij for a
  pseudo-component, so H2S-cut is 0 there against difflow's 0.0333).

Each flash records DWSIM's phase split, compositions, K, fugacity
coefficients and its own equilibrium residual.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dwsim_hps_reference.json"
PREFIX = "DF_"

SETUPS = {
    "same": {"package": "PR78", "gases": "hypo", "kij": "difflow"},
    "pr76": {"package": "PR", "gases": "hypo", "kij": "difflow"},
    "dwsim_compounds": {"package": "PR78", "gases": "database", "kij": "difflow"},
    "dwsim_data": {"package": "PR78", "gases": "database", "kij": "dwsim"},
}


def build(session, eff: dict, setup: dict):
    import numpy as np

    from difflow.dwsim_import import DWSIM_NAMES

    from . import dwsim_hps_case as hc
    from . import dwsim_lightends as dl
    from .dwsim_session import EOS_ONLY, HypoCompound

    c = eff["components"]
    idx = hc.present(eff)
    gas_idx = [i for i in idx if i < c["n_gas"]]
    cut_idx = [i for i in idx if i >= c["n_gas"]]

    def hypo(i):
        return HypoCompound(name=PREFIX + c["names"][i], MW=c["MW"][i], Tc=c["Tc"][i],
                            Pc=c["Pc"][i], omega=c["omega"][i], Tb=c["Tb"][i],
                            SG=c["SG"][i] if i >= c["n_gas"] else None,
                            cp_ig=c["cp_ig"][i], hvap_tb=c["hvap_nb"][i])

    if setup["gases"] == "hypo":
        compounds, hypos = [], [hypo(i) for i in idx]
    else:
        compounds, hypos = [DWSIM_NAMES[c["names"][i]] for i in gas_idx], [hypo(i) for i in cut_idx]
    kij = "dwsim" if setup["kij"] == "dwsim" else "zero"
    fs = session.flowsheet(setup["package"], compounds=compounds, hypos=hypos, kij=kij,
                           options=EOS_ONLY, flash_tol=1e-10)
    if setup["kij"] == "difflow":
        dl.set_kij(fs, np.asarray(c["kij"])[np.ix_(idx, idx)])
    return fs


def provenance(session) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    p = session.provenance()
    p.update({
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/dwsim_hps_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.dwsim_hps_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            f"{p['dwsim']} PT flashes (Universal Flash, Nested Loops, tolerances 1e-10) of a "
            "hydrotreater reactor effluent: 'Peng-Robinson 1978 (PR78)' on difflow's constants "
            "and kij (a); 'Peng-Robinson (PR)' (1976 kappa throughout); DWSIM's gas compounds "
            "with difflow's and with DWSIM's kij (b)."),
    })
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--effluent", help="reuse the effluent of an existing reference file")
    a = ap.parse_args()

    from . import dwsim_hps_case as hc

    if a.effluent:
        eff = json.loads(Path(a.effluent).read_text())["effluent"]
    else:
        eff = hc.effluent()
        if not eff["converged"]:
            raise RuntimeError("the hydrotreater did not converge")
    print("effluent ready", flush=True)

    from .dwsim_session import DWSIMSession

    session = DWSIMSession()
    idx = hc.present(eff)
    z = [eff["flows"][i] for i in idx]
    data = {"provenance": provenance(session), "effluent": eff, "points": hc.POINTS,
            "names": [eff["components"]["names"][i] for i in idx], "setups": {}}
    for name, setup in SETUPS.items():
        fs = build(session, eff, setup)
        res = {"setup": setup, "dwsim": fs.provenance(), "flashes": []}
        for T, P in hc.POINTS:
            res["flashes"].append(fs.flash_tp(z, T, P))
        data["setups"][name] = res
        print(name, "done", [round(r["vapor_fraction"], 6) for r in res["flashes"]], flush=True)
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
