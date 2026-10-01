"""Generate ``preheat_reference.json``: IDAES cross-checks of the preheat train (#313).

Run from the repository root (needs IDAES, Pyomo and an IPOPT; IDAES's
build is the one the committed file was made with)::

    PYTHONPATH=src:tests python -m refinery.reference.preheat_generate [--ipopt PATH]

What it computes, through IDAES unit models on the base crude's
pseudo-components (:func:`.idaes_thermo.ideal_config`, the constants of
:func:`.case.component_data`):

* **The preflash drum** -- an IDAES ``Flash`` (adiabatic, ``deltaP`` to the
  drum pressure) on the dry crude, from a liquid inlet at the train's
  pressure. And the drum's *state*, wet: an IDAES state block at the drum's
  temperature and pressure with the crude's water, which IDAES's package
  carries as a vapour-only component -- the drum's no-free-water branch.
* **Two exchangers** -- IDAES ``HeatExchanger`` (counter-current, exact
  LMTD) at the inlets of the base case's E7 and E8 (the hottest two, residue
  against the drum liquid), with ``U A`` the exchanger's ``A/(1/U + R_f)``.
  The hot side is a liquid-only package (difflow's hot streams are liquid);
  the cold side has the VLE, so a crude that starts to boil in the
  exchanger is split by IDAES's ``SmoothVLE``.

This is an **independent implementation of the same model**, not an
independent model: the same Raoult/Lee-Kesler/Watson property model on the
same constants, assembled and solved by IDAES instead of difflow. IDAES's
liquid enthalpy carries a ``(P - P_ref)/rho`` term difflow's does not; the
liquid density here is made large enough (1e9 mol/m^3) that the term is
below 1e-3 J/mol, so the two models are the same model.

The only difflow calls are the characterisation (the pseudo-component
table, an input to both) and one solve of the coupled base case, to read
the exchangers' *inlet* conditions -- inputs, frozen into the file.
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
OUT = HERE / "preheat_reference.json"
DEFAULT_IPOPT = os.path.expanduser("~/.idaes/bin/ipopt")

#: The drum check: a dry crude at the train's pressure, flashed to the drum's.
DRUM = {"rate": 100.0, "T_in": 500.0, "P_in": 15.0e5, "P": 3.0e5}
#: The wet state check: that crude with 0.2 % (standard volume) water, at
#: the drum's pressure and two temperatures, all water vapour at both.
WET = {"water": 0.002, "points": [(470.0, 3.0e5), (500.0, 2.0e5)]}
EXCHANGERS = ("E7", "E8")
#: Liquid density for IDAES's ``(P - P_ref)/rho`` term (mol/m^3): large
#: enough to make it vanish, which is difflow's model.
BIG_DENSITY = 1.0e9


def _density_class():
    class NoPVDensity:
        @staticmethod
        def build_parameters(cobj):
            pass

        @staticmethod
        def return_expression(b, cobj, T):
            from pyomo.environ import units as u

            return BIG_DENSITY * u.mol / u.m**3

    return NoPVDensity


def config(cd, vle=True, water=False):
    """:func:`.idaes_thermo.ideal_config` (the smoothed Watson branch, as
    difflow's), with no ``P/rho`` term, and liquid-only with ``vle=False``."""
    from idaes.core import LiquidPhase
    from idaes.models.properties.modular_properties.eos.ideal import Ideal

    from . import idaes_thermo as it

    cfg = it.ideal_config(cd, water=water, eps=0.01)
    dens = _density_class()
    for n in cd["names"]:
        cfg["components"][n]["dens_mol_liq_comp"] = dens
        cfg["components"][n]["parameter_data"].pop("sg", None)
    if not vle:
        cfg["phases"] = {"Liq": {"type": LiquidPhase, "equation_of_state": Ideal}}
        for k in ("phases_in_equilibrium", "phase_equilibrium_state", "bubble_dew_method"):
            cfg.pop(k)
        for n in cd["names"]:
            cfg["components"][n].pop("phase_equilibrium_form")
    return cfg


def _solver(ipopt):
    from pyomo.environ import SolverFactory

    s = SolverFactory("ipopt", executable=ipopt)
    s.options.update({"tol": 1e-11, "max_iter": 1000})
    return s


def drum_reference(cd, ipopt) -> dict:
    """IDAES Flash: the adiabatic drum on the dry crude."""
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties import GenericParameterBlock
    from idaes.models.unit_models import Flash
    from pyomo.environ import ConcreteModel, value

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**config(cd))
    m.fs.drum = Flash(property_package=m.fs.props)
    inlet = m.fs.drum.inlet
    inlet.flow_mol.fix(DRUM["rate"])
    inlet.temperature.fix(DRUM["T_in"])
    inlet.pressure.fix(DRUM["P_in"])
    for n, x in zip(cd["names"], cd["mole_fraction"]):
        inlet.mole_frac_comp[0, n].fix(max(x, 1e-12))
    m.fs.drum.heat_duty.fix(0.0)
    m.fs.drum.deltaP.fix(DRUM["P"] - DRUM["P_in"])
    m.fs.drum.initialize(outlvl=0)
    res = _solver(ipopt).solve(m)
    cv = m.fs.drum.control_volume
    out = cv.properties_out[0]
    vap = m.fs.drum.vap_outlet
    liq = m.fs.drum.liq_outlet
    names = cd["names"]
    inn = cv.properties_in[0]
    return {
        "inputs": DRUM,
        "solve": {"status": str(res.solver.termination_condition)},
        "inlet_vapor_fraction": value(inn.phase_frac["Vap"]),
        "T": value(out.temperature),
        "vapor": value(vap.flow_mol[0]),
        "liquid": value(liq.flow_mol[0]),
        "vapor_fraction": value(out.phase_frac["Vap"]),
        "y": [value(vap.mole_frac_comp[0, n]) for n in names],
        "x": [value(liq.mole_frac_comp[0, n]) for n in names],
    }


def wet_reference(cd, water_mol_frac) -> list:
    """IDAES state blocks: the wet crude at the drum's states (water vapour-only)."""
    from . import idaes_thermo as it

    names = cd["names"]
    out = []
    for T, P in WET["points"]:
        sp = it.StatePoint(config(cd, water=True))
        z = {n: x * (1 - water_mol_frac) for n, x in zip(names, cd["mole_fraction"])}
        z["H2O"] = water_mol_frac
        ok = sp.solve(z, T, P)
        s = sp.s
        out.append({
            "T": T, "P": P, "ok": ok, "z_water": water_mol_frac,
            "vapor_fraction": sp.value(s.phase_frac["Vap"]),
            "y_water": sp.value(s.mole_frac_phase_comp["Vap", "H2O"]),
            "x": [sp.value(s.mole_frac_phase_comp["Liq", n]) for n in names],
        })
    return out


def exchanger_reference(cd, ipopt, case_in: dict) -> dict:
    """IDAES HeatExchanger at one exchanger's inlets."""
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties import GenericParameterBlock
    from idaes.models.unit_models import HeatExchanger
    from idaes.models.unit_models.heat_exchanger import (HeatExchangerFlowPattern,
                                                         delta_temperature_lmtd_callback)
    from pyomo.environ import ConcreteModel, value

    names = cd["names"]
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.hot_props = GenericParameterBlock(**config(cd, vle=False))
    m.fs.cold_props = GenericParameterBlock(**config(cd, vle=True))
    m.fs.hx = HeatExchanger(
        hot_side={"property_package": m.fs.hot_props, "has_pressure_change": False},
        cold_side={"property_package": m.fs.cold_props, "has_pressure_change": False},
        delta_temperature_callback=delta_temperature_lmtd_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent)
    for port, side in ((m.fs.hx.hot_side_inlet, "hot"), (m.fs.hx.cold_side_inlet, "cold")):
        f = np.asarray(case_in[f"{side}_flows"])
        port.flow_mol.fix(f.sum())
        port.temperature.fix(case_in[f"{side}_T"])
        port.pressure.fix(case_in[f"{side}_P"])
        for n, x in zip(names, f / f.sum()):
            port.mole_frac_comp[0, n].fix(max(x, 1e-12))
    m.fs.hx.area.fix(1.0)
    m.fs.hx.overall_heat_transfer_coefficient.fix(case_in["UA"])
    m.fs.hx.initialize(outlvl=0)
    res = _solver(ipopt).solve(m)
    co = m.fs.hx.cold_side.properties_out[0]
    return {
        "inputs": case_in,
        "solve": {"status": str(res.solver.termination_condition)},
        "Q": value(m.fs.hx.heat_duty[0]),
        "T_cold_out": value(co.temperature),
        "T_hot_out": value(m.fs.hx.hot_side.properties_out[0].temperature),
        "cold_out_vapor_fraction": value(co.phase_frac["Vap"]),
    }


def base_inputs():
    """The base crude, its component table, and E7/E8's inlets from the coupled base case."""
    from difflow_refinery import characterize
    from difflow_refinery.preheat.train import _water_moles

    from . import case
    from . import preheat_case as pc

    unit, _ = pc.preheated_unit("base")
    crude = characterize(unit.assay, unit.cut_points, unit.method)
    cd = case.component_data(crude, unit.thermo)
    r = unit.solve(case.BPD, pc.T_TANK)
    tp = unit.train_params
    hot_of = {e: hs.source for hs in tp.hot_streams for e in hs.exchangers}
    exch = {}
    for name in EXCHANGERS:
        e = r.train.exchangers[name]
        ex = tp.exchanger(name)
        src = hot_of[name]
        exch[name] = {
            "hot_source": src,
            "hot_flows": [float(v) for v in np.asarray(
                np.stack([r.column.products[src][f"F_{n}"] for n in unit.thermo.names]))],
            "hot_T": float(e["T_hot_in"]), "hot_P": float(tp.P_furnace),
            "cold_flows": [float(v) for v in np.asarray(r.train.crude_out)],
            "cold_T": float(e["T_cold_in"]), "cold_P": float(tp.P_furnace),
            "UA": float(ex.UA), "bypass": float(ex.bypass),
            "difflow_Q": float(e["Q"]),
        }
    f1 = np.asarray(cd["mole_fraction"])
    fw = float(_water_moles(unit.thermo, np.asarray(f1), WET["water"]))
    return unit, cd, exch, fw / (1.0 + fw)


def provenance(ipopt) -> dict:
    import idaes
    import pyomo

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
        except Exception:
            return None

    try:
        ver = subprocess.check_output([ipopt, "--version"], text=True).splitlines()[0]
    except Exception:
        ver = ipopt
    return {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": "tests/refinery/reference/preheat_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.preheat_generate",
        "idaes": idaes.__version__, "pyomo": pyomo.version.version, "ipopt": ver,
        "python": platform.python_version(),
        "difflow_commit": git("rev-parse", "HEAD"),
        "property_methods": "IDAES modular properties: Ideal EOS, SmoothVLE, IdealBubbleDew; "
                            "Lee-Kesler Psat, RPP4 ideal-gas Cp, Watson liquid enthalpy (eps 0.01), "
                            "no (P - Pref)/rho term -- difflow's model, not an independent one",
        "unit_models": "idaes.models.unit_models.Flash, HeatExchanger (countercurrent, exact LMTD)",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ipopt", default=DEFAULT_IPOPT)
    args = ap.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    unit, cd, exch, z_water = base_inputs()
    out = {"provenance": provenance(args.ipopt), "components": cd,
           "drum": drum_reference(cd, args.ipopt),
           "wet": wet_reference(cd, z_water),
           "exchangers": {n: exchanger_reference(cd, args.ipopt, c) for n, c in exch.items()}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps({k: out["drum"][k] for k in ("T", "vapor_fraction", "solve")}, indent=1))
    for n, e in out["exchangers"].items():
        print(n, e["solve"], e["Q"], e["inputs"]["difflow_Q"], e["T_cold_out"], e["cold_out_vapor_fraction"])
    for w in out["wet"]:
        print("wet", w["T"], w["P"], w["ok"], w["vapor_fraction"], w["y_water"])


if __name__ == "__main__":
    main()
