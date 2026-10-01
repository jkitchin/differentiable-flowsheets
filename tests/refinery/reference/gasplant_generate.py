"""Generate ``gasplant_reference.json``: the gas plant's IDAES reference (#312).

Run from the repository root (needs IDAES, Pyomo and an IPOPT)::

    PYTHONPATH=src:tests python -m refinery.reference.gasplant_generate [--ipopt PATH]

What it computes, for each case of :mod:`.gasplant_case`:

* **The column** -- IDAES's ``TrayColumn`` (total condenser at the bubble
  point, equilibrium trays, kettle reboiler, no pressure change) on IDAES's
  generic Peng-Robinson package (:func:`.idaes_thermo.pr_config`: its
  ``Cubic`` EOS, ``SmoothVLE`` and log-fugacity equilibrium), initialized by
  IDAES's own routine and solved by IPOPT, with the reflux and boilup
  ratios fixed. Recorded: product component flows, condenser and reboiler
  duties, and every stage's temperature, pressure and phase compositions.
* **K-values point by point** -- ``K = y/x`` on every equilibrium stage of
  IDAES's solution, with the state they belong to, so difflow's PR
  ``ln K = ln phi_L - ln phi_V`` can be evaluated at exactly IDAES's (T, P,
  x, y) and compared without either column in the way.

This is an independent *implementation* of the same model (PR 1976 with
zero kij on the same constants, the same column topology), not an
independent model: it checks difflow's EOS, MESH equations and solver
against IDAES's, and says nothing about how well PR describes these
mixtures.

The only difflow call is :func:`.gasplant_case.component_data`: the
constants, an input to both, frozen into the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "gasplant_reference.json"
DEFAULT_IPOPT = os.path.expanduser("~/.idaes/bin/ipopt")


def idaes_column(case: dict, comp: dict, ipopt: str) -> dict:
    from idaes.core import FlowsheetBlock
    from idaes.core.solvers import get_solver
    from idaes.models.properties.modular_properties import GenericParameterBlock
    from idaes.models_extra.column_models import TrayColumn
    from idaes.models_extra.column_models.condenser import CondenserType, TemperatureSpec
    from pyomo.environ import ConcreteModel, value
    from pyomo.environ import units as u

    from . import idaes_thermo as it

    names = comp["names"]
    cfg = it.pr_config({"names": names, "MW": comp["MW"], "Tc": comp["Tc"], "Pc": comp["Pc"],
                        "omega_eos": comp["omega"], "omega_vp": comp["omega"],
                        "cp_ig": comp["cp_ig"]})
    cfg["state_bounds"]["temperature"] = (150, 350, 700, u.K)
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**cfg)
    m.fs.col = TrayColumn(number_of_trays=case["n_trays"], feed_tray_location=case["feed_tray"],
                          condenser_type=CondenserType.totalCondenser,
                          condenser_temperature_spec=TemperatureSpec.atBubblePoint,
                          property_package=m.fs.props, has_heat_transfer=False,
                          has_pressure_change=False)
    col = m.fs.col
    f = col.feed
    f.flow_mol.fix(case["F"])
    f.temperature.fix(case["T"])
    f.pressure.fix(case["P"])
    for n, z in zip(names, case["z"]):
        f.mole_frac_comp[0, n].fix(z)
    col.condenser.reflux_ratio.fix(case["reflux_ratio"])
    col.condenser.condenser_pressure.fix(case["P"])
    col.reboiler.boilup_ratio.fix(case["boilup_ratio"])
    start = case.get("idaes_start", "idaes_initialize")
    if start == "idaes_initialize":
        col.initialize()
    else:
        profile_initialize(col, case, names)
    solver = get_solver(options={"max_iter": 3000, "tol": 1e-10})
    solver.set_executable(ipopt)
    res = solver.solve(m, tee=False)
    status = str(res.solver.termination_condition)

    def phase(sb, p):
        return [value(sb.mole_frac_phase_comp[p, n]) for n in names]

    stages = []
    sb = col.condenser.control_volume.properties_out[0]
    stages.append({"stage": "condenser", "T": value(sb.temperature), "P": value(sb.pressure),
                   "x": phase(sb, "Liq"), "y": None})
    nf = case["feed_tray"]
    for j in range(1, case["n_trays"] + 1):
        tray = (col.rectification_section[j] if j < nf else col.feed_tray if j == nf
                else col.stripping_section[j])
        sb = tray.properties_out[0]
        stages.append({"stage": f"tray{j}", "T": value(sb.temperature),
                       "P": value(sb.pressure), "x": phase(sb, "Liq"), "y": phase(sb, "Vap")})
    sb = col.reboiler.control_volume.properties_out[0]
    stages.append({"stage": "reboiler", "T": value(sb.temperature), "P": value(sb.pressure),
                   "x": phase(sb, "Liq"), "y": phase(sb, "Vap")})
    for s in stages:
        s["K"] = None if s["y"] is None else [y / x for x, y in zip(s["x"], s["y"])]

    D = value(col.condenser.distillate.flow_mol[0])
    B = value(col.reboiler.bottoms.flow_mol[0])
    reflux = value(col.condenser.reflux.flow_mol[0])
    boilup = value(col.reboiler.vapor_reboil.flow_mol[0])
    return {
        "solve": {"status": status, "solver": "ipopt", "start": start},
        "distillate": {n: D * value(col.condenser.distillate.mole_frac_comp[0, n]) for n in names},
        "bottoms": {n: B * value(col.reboiler.bottoms.mole_frac_comp[0, n]) for n in names},
        "reflux_ratio": reflux / D,
        "boilup_ratio": boilup / B,
        "condenser_duty": value(col.condenser.heat_duty[0]),
        "reboiler_duty": value(col.reboiler.heat_duty[0]),
        "stages": stages,
    }


def profile_initialize(col, case: dict, names: list) -> None:
    """A shortcut-column start for IDAES's trays, built without difflow.

    IDAES's own ``TrayColumn.initialize`` gives every rectifying tray the
    feed tray's vapour and the reflux's liquid. On the C3/C4 splitter that
    start ends locally infeasible or at the iteration cap (tried with and
    without ethane, at 12 and 17 bar, with 12 and 20
    trays, and with a 3000-iteration cap). This start is the textbook
    one instead:

    * constant molar overflow from the fixed ratios, for a saturated-liquid
      feed: ``D = boilup B / (R + 1)``;
    * a sharp split by volatility order: the lightest components fill the
      distillate, the rest go to the bottoms;
    * liquid compositions linear in tray number from distillate to bottoms,
      each tray's vapour the composition of the liquid above it;
    * temperatures linear between the condenser's and the reboiler's own
      bubble points, which IDAES's condenser and reboiler initializations
      compute from those compositions.

    Then each tray is initialized by IDAES's ``Tray.initialize`` from its
    inlets. Nothing here comes from difflow's solution.
    """
    from pyomo.environ import value

    n, nf = case["n_trays"], case["feed_tray"]
    F, P, R, bu = case["F"], case["P"], case["reflux_ratio"], case["boilup_ratio"]
    fz = [F * z for z in case["z"]]
    B = F / (1.0 + bu / (R + 1.0))
    D = F - B
    dist, left = [], D
    for f in fz:                                    # names are light to heavy
        take = min(f, left)
        dist.append(take)
        left -= take
    bot = [f - d for f, d in zip(fz, dist)]
    xD = [d / D for d in dist]
    xB = [b / B for b in bot]
    eps = 1e-4                                      # no zero mole fraction
    xD = [(x + eps) / (1 + eps * len(xD)) for x in xD]
    xB = [(x + eps) / (1 + eps * len(xB)) for x in xB]

    def x_on(j):                                    # j = 0 condenser ... n + 1 reboiler
        w = j / (n + 1)
        return [(1 - w) * a + w * b for a, b in zip(xD, xB)]

    def put(sb, flow, x, T):
        sb.flow_mol.value = flow
        sb.temperature.value = T
        sb.pressure.value = P
        for nm, xi in zip(names, x):
            sb.mole_frac_comp[nm].value = xi

    V_rect, L_rect = (R + 1.0) * D, R * D
    V_strip, L_strip = bu * B, bu * B + B
    T_guess = case["T"]
    put(col.condenser.control_volume.properties_in[0], V_rect, xD, T_guess)
    col.condenser.initialize()
    T_top = value(col.condenser.control_volume.properties_out[0].temperature)
    put(col.reboiler.control_volume.properties_in[0], L_strip, xB, T_guess)
    col.reboiler.initialize()
    T_bot = value(col.reboiler.control_volume.properties_out[0].temperature)

    def T_on(j):
        return T_top + (T_bot - T_top) * j / (n + 1)

    def args(flow, x, T):
        return {"flow_mol": flow, "temperature": T, "pressure": P,
                "mole_frac_comp": dict(zip(names, x))}

    for j in range(1, n + 1):
        L_in = L_rect if j <= nf else L_strip
        V_in = V_rect if j < nf else V_strip
        liq = args(L_in, x_on(j - 1), T_on(j - 1))
        vap = args(V_in, x_on(j), T_on(j + 1))
        if j == nf:
            feed = args(F, case["z"], case["T"])
            col.feed_tray.initialize(state_args_feed=feed, state_args_liq=liq,
                                     state_args_vap=vap)
            continue
        tray = col.rectification_section[j] if j < nf else col.stripping_section[j]
        put(tray.properties_in_liq[0], liq["flow_mol"], x_on(j - 1), liq["temperature"])
        put(tray.properties_in_vap[0], vap["flow_mol"], x_on(j), vap["temperature"])
        tray.initialize()


def provenance(ipopt: str) -> dict:
    try:
        v = subprocess.run([ipopt, "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    except Exception:  # pragma: no cover
        v = "unknown"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    import idaes
    import pyomo.version

    return {
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/gasplant_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.gasplant_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            f"IDAES {idaes.__version__} TrayColumn on its generic Peng-Robinson property "
            "package (Cubic EOS, SmoothVLE, log-fugacity equilibrium, kij = 0), solved by "
            "IPOPT from IDAES's own initialization. The same model as difflow's, "
            "implemented independently."),
        "idaes": idaes.__version__,
        "pyomo": pyomo.version.version,
        "ipopt": v,
    }


def main():
    from . import gasplant_case as gc

    ap = argparse.ArgumentParser()
    ap.add_argument("--ipopt", default=DEFAULT_IPOPT)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--case", action="append",
                    help="regenerate only this case (repeatable), keeping the others in --out")
    a = ap.parse_args()
    os.environ["PATH"] = os.path.dirname(a.ipopt) + os.pathsep + os.environ.get("PATH", "")
    cases = {}
    if a.case and Path(a.out).exists():
        cases = json.loads(Path(a.out).read_text())["cases"]
    data = {"provenance": provenance(a.ipopt), "cases": cases}
    for name, case in gc.CASES.items():
        if a.case and name not in a.case:
            continue
        comp = gc.component_data(case["names"])
        print("solving", name, flush=True)
        data["cases"][name] = {"case": case, "components": comp,
                               "idaes": idaes_column(case, comp, a.ipopt)}
        print(" ", data["cases"][name]["idaes"]["solve"]["status"], flush=True)
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
