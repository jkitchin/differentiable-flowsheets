"""Generate ``gasplant_reference.json``: the gas plant's IDAES reference (#312).

Run from the repository root (needs IDAES, Pyomo and an IPOPT)::

    PYTHONPATH=src:tests python -m refinery.reference.gasplant_generate [--ipopt PATH]

What it computes, for each case of :mod:`.gasplant_case` whose
``reference`` is ``"column"`` (the default; ``debutanizer``):

* **The column** -- IDAES's ``TrayColumn`` (total condenser at the bubble
  point, equilibrium trays, kettle reboiler, no pressure change) on IDAES's
  generic Peng-Robinson package (:func:`.idaes_thermo.pr_config`: its
  ``Cubic`` EOS, ``SmoothVLE`` and log-fugacity equilibrium), initialized by
  IDAES's own routine and solved by IPOPT, with the reflux and boilup
  ratios fixed. Recorded: product component flows (the distillate at the
  condenser outlet's overall composition; see :func:`idaes_column`), condenser and reboiler
  duties, and every stage's temperature, pressure and phase compositions.
* **K-values point by point** -- ``K = y/x`` on every equilibrium stage of
  IDAES's solution, with the state they belong to, so difflow's PR
  ``ln K = ln phi_L - ln phi_V`` can be evaluated at exactly IDAES's (T, P,
  x, y) and compared without either column in the way.

For a case whose ``reference`` is ``"state_points"`` (``c3c4_splitter``),
IDAES's flash at each of difflow's stage states instead
(:func:`idaes_state_points`). IDAES's ``TrayColumn`` was not converged on
this column. ``TrayColumn.initialize`` fails at its "column section +
condenser" step on every variant tried: with and without ethane and
propylene; 10, 12 and 17 bar; 10, 12, 15 and 20 trays; reflux/boilup 2/2,
3/2 and 5/3; 3000 IPOPT iterations. Starting every state block from
difflow's own converged profile was tried too: at the case's ratios IPOPT
ends infeasible, and at 4.0/2.5 it reports optimal on a spurious solution
with two trays single phase (``SmoothVLE``'s x == y branch). Neither is a
reference, and a reference seeded from difflow's answer would not be
independent of it anyway, so the splitter's comparison stops at the
thermodynamics, where IDAES solves from its own initialization at every
point.

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
    from pyomo.environ import ConcreteModel, Param, value
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
    col.initialize()
    solver = get_solver(options={"max_iter": 3000, "tol": 1e-10})
    solver.set_executable(ipopt)
    res = solver.solve(m, tee=False)
    # SmoothVLE's smoothing parameters, tightened by continuation. At the
    # defaults (eps_1 = 0.01, eps_2 = 5e-4) the total condenser's bubble-point
    # outlet is left 4e-4 vapor, and since its reflux and distillate ports
    # carry the LIQUID composition at the total flow, the condenser loses
    # components (0.07 % of the propane in the debutanizer) while the total
    # balances. The vapor fraction, and the leak, scale with eps_2.
    eps = [(1e-3, 1e-5), (1e-4, 1e-6), (1e-5, 1e-7), (1e-6, 1e-8), (1e-7, 1e-9)]
    for e1, e2 in eps:
        for o in m.component_data_objects(Param, descend_into=True):
            name = o.parent_component().local_name
            if name == "eps_1_Vap_Liq":
                o.set_value(e1)
            elif name == "eps_2_Vap_Liq":
                o.set_value(e2)
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

    cout = col.condenser.control_volume.properties_out[0]
    D = value(col.condenser.distillate.flow_mol[0])
    B = value(col.reboiler.bottoms.flow_mol[0])
    reflux = value(col.condenser.reflux.flow_mol[0])
    boilup = value(col.reboiler.vapor_reboil.flow_mol[0])
    return {
        "solve": {"status": status, "solver": "ipopt", "start": "idaes_initialize",
                  "smooth_vle_eps": list(eps[-1])},
        # The distillate at the condenser outlet's OVERALL composition, not the
        # port's. At the bubble point SmoothVLE's smoothing leaves the outlet a
        # few 1e-4 vapor, and the total condenser's ports carry the LIQUID
        # phase composition at the total flow, so the port flows break the
        # component balance (by 0.07 % on propane in the debutanizer) while
        # the total balances. The port's values are kept for the record.
        "distillate": {n: D * value(cout.mole_frac_comp[n]) for n in names},
        "distillate_port": {n: D * value(col.condenser.distillate.mole_frac_comp[0, n])
                            for n in names},
        "condenser_outlet_vapor_fraction": value(cout.phase_frac["Vap"]),
        "bottoms": {n: B * value(col.reboiler.bottoms.mole_frac_comp[0, n]) for n in names},
        "reflux_ratio": reflux / D,
        "boilup_ratio": boilup / B,
        "condenser_duty": value(col.condenser.heat_duty[0]),
        "reboiler_duty": value(col.reboiler.heat_duty[0]),
        "stages": stages,
    }


def difflow_profile(case: dict) -> dict:
    """difflow's converged stage profile for ``case``: where the state points are."""
    import jax
    import numpy as np

    from . import gasplant_case as gc

    jax.config.update("jax_enable_x64", True)
    col, feed = gc.difflow_column(case)
    _, _, info = col(feed)
    assert bool(info["converged"]), "difflow did not converge the case's column"
    return {k: np.asarray(info["profiles"][k], dtype=float).tolist()
            for k in ("L", "V", "T", "P", "x", "y")}


def idaes_state_points(case: dict, comp: dict) -> list:
    """IDAES's PR flash at each of difflow's stage states (trays and reboiler).

    For the cases IDAES's ``TrayColumn`` does not converge (see
    :mod:`.gasplant_case`). Each stage's two phases are recombined into the
    stage's overall composition ``z = (L x + V y)/(L + V)``, which is well
    inside the two-phase region, and IDAES flashes ``z`` at the stage's T
    and P on its own generic PR package (its ``SmoothVLE``, log-fugacity
    equilibrium) from its own state-block initialization. Recorded: IDAES's
    phase fraction, phase compositions, ``K = y/x``, and each phase's molar
    enthalpy. difflow's profile picks the points; it is not used as a start.
    """
    from pyomo.environ import value

    from . import idaes_thermo as it

    names = comp["names"]
    cfg = it.pr_config({"names": names, "MW": comp["MW"], "Tc": comp["Tc"], "Pc": comp["Pc"],
                        "omega_eos": comp["omega"], "omega_vp": comp["omega"],
                        "cp_ig": comp["cp_ig"]})
    cfg["state_bounds"]["temperature"] = (150, 350, 700, cfg["state_bounds"]["temperature"][3])
    sp = it.StatePoint(cfg)
    prof = difflow_profile(case)
    out = []
    for j in range(1, case["n_trays"] + 2):         # trays 1..n, then the reboiler
        Lj, Vj = prof["L"][j], prof["V"][j]
        z = [(Lj * a + Vj * b) / (Lj + Vj) for a, b in zip(prof["x"][j], prof["y"][j])]
        T, P = prof["T"][j], prof["P"][j]
        ok = sp.solve(dict(zip(names, z)), T, P)
        s = sp.s
        x = [value(s.mole_frac_phase_comp["Liq", n]) for n in names]
        y = [value(s.mole_frac_phase_comp["Vap", n]) for n in names]
        out.append({"stage": "reboiler" if j == case["n_trays"] + 1 else f"tray{j}",
                    "status": sp.termination, "optimal": ok, "T": T, "P": P, "z": z,
                    "vapor_fraction": value(s.phase_frac["Vap"]), "x": x, "y": y,
                    "K": [b / a for a, b in zip(x, y)],
                    "h_liq": value(s.enth_mol_phase["Liq"]),
                    "h_vap": value(s.enth_mol_phase["Vap"])})
    return out


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
            "IPOPT from IDAES's own initialization; for state-point cases, IDAES's TP flash on "
            "the same package at difflow's stage states. The same model as difflow's, "
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
        entry = {"case": case, "components": comp}
        if case.get("reference", "column") == "column":
            entry["idaes"] = idaes_column(case, comp, a.ipopt)
            print(" ", entry["idaes"]["solve"]["status"], flush=True)
        else:
            entry["idaes_points"] = idaes_state_points(case, comp)
            print(" ", [p["status"] for p in entry["idaes_points"]], flush=True)
        data["cases"][name] = entry
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
