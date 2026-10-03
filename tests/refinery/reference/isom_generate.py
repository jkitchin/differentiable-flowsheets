"""Generate ``isom_reference.json``: IDAES GibbsReactor cross-checks of the
isomerization thermochemistry (#311).

Run from the repository root (needs IDAES, Pyomo and an IPOPT)::

    PYTHONPATH=src:tests python -m refinery.reference.isom_generate [--ipopt PATH]

What it computes, with IDAES's ``GibbsReactor`` (minimisation of the total
Gibbs energy subject to element balances) on an ideal-gas modular property
package (``Ideal`` EOS, ``RPP4`` pure-component methods) given the
constants of :mod:`difflow_refinery.isomerization.thermochem` -- the heats
of formation, absolute entropies and Cp cubics:

* **families** -- each isomer family alone (C5; the five C6 paraffins; MCP
  and cyclohexane), isothermal at several temperatures. difflow's answer is
  the closed form :func:`~difflow_refinery.isomerization.thermochem.family_equilibrium`.
* **c6_ring** -- hydrogen, benzene, MCP, cyclohexane and the five C6
  paraffins together, isothermal at 30 bar: benzene saturation and ring
  opening, whose equilibria depend on the hydrogen partial pressure. difflow's
  answer is its reactor, held isothermal, with the rate constants scaled up
  until the bed reaches equilibrium and hydrocracking switched off.
* **adiabatic** -- the reactor charge of both constructed feeds (fresh feed
  plus hydrogen at H2/HC = 0.3) at 413.15 K and 30 bar, adiabatic: the
  equilibrium temperature and composition. difflow's answer is its reactor,
  adiabatic, the same way.

The element balances of the reaction network are not those of C and H
alone. The reactor has no reaction that changes a carbon number except
hydrocracking, which is irreversible and off here, and does not isomerize
the butanes or the C7+ lump. A Gibbs minimiser given only C and H would
disproportionate pentanes into hexanes, so each conserved skeleton is given
its own element label: ``C5`` for the pentanes, ``C6`` for every C6
species (benzene, the naphthenes and the paraffins share it, with hydrogen
making the difference), and one label each for the species the network
holds fixed (the butanes, the C7+ lump, ethane, propane). The minimisation
is then over exactly the network's reaction space.

This is an **independent implementation, not an independent model**: the
same ideal-gas thermochemistry, assembled and minimised by IDAES instead of
evaluated in closed form or integrated to its steady state by difflow. It
checks the free energies' assembly, the equilibrium-constant convention
(pressures in bar against a 1 bar standard state), the hydrogen-pressure
dependence, the reactor's energy balance and that the bed's rate law
relaxes onto the equilibrium it claims. It does not check the constants
themselves: both sides are handed the same ones.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "isom_reference.json"
DEFAULT_IPOPT = os.path.expanduser("~/.idaes/bin/ipopt")

P_REACTOR = 30.0e5
#: Isothermal family cases (K).
FAMILY_T = (400.0, 450.0, 500.0, 550.0)
#: The C6 ring case: feed (mol/s) and conditions.
C6_RING = {"feed": {"hydrogen": 4.0, "benzene": 0.3, "methylcyclopentane": 0.4,
                    "cyclohexane": 0.3, "n_hexane": 1.0},
           "T": (420.0, 480.0), "P": P_REACTOR}
#: The adiabatic cases: the constructed feeds' reactor charge.
ADIABATIC = {"feeds": ("paraffinic", "benzene_rich"), "mass_rate": 10.0, "H2_HC": 0.3,
             "T_in": 413.15, "P": P_REACTOR}

#: Skeleton label of each species (see the module docstring).
SKELETON = {
    "hydrogen": None, "ethane": "K2", "propane": "K3", "isobutane": "Ki4", "n_butane": "Kn4",
    "isopentane": "C5", "n_pentane": "C5",
    "2_2_dimethylbutane": "C6", "2_3_dimethylbutane": "C6", "2_methylpentane": "C6",
    "3_methylpentane": "C6", "n_hexane": "C6", "methylcyclopentane": "C6", "cyclohexane": "C6",
    "benzene": "C6", "n_heptane": "K7",
}


def component_data() -> dict:
    """The thermochemistry handed to IDAES: difflow's constants (inputs)."""
    from difflow_refinery.isomerization import thermochem as tc

    return {s.name: {"Hf": s.Hf, "S": s.S, "cp": list(s.cp), "H": s.H, "C": s.C,
                     "MW": s.MW} for s in tc.SPECIES}


def adiabatic_charge(kind: str) -> dict:
    """Reactor charge of a constructed feed (mol/s): an input."""
    from difflow_refinery.isomerization import constructed_feed
    from difflow_refinery.isomerization import thermochem as tc

    f = constructed_feed(kind, ADIABATIC["mass_rate"])
    F = {n: float(f[f"F_{n}"]) for n in tc.NAMES}
    hc = sum(v for n, v in F.items() if n != "hydrogen")
    F["hydrogen"] = ADIABATIC["H2_HC"] * hc
    return {n: v for n, v in F.items() if v > 0.0}


def element_counts(cd: dict, name: str) -> dict:
    el = {"H": cd[name]["H"]}
    if SKELETON[name]:
        el[SKELETON[name]] = 1
    return el


def independent_elements(cd: dict, names) -> list:
    """The element labels to give IDAES: a linearly independent subset.

    A family of isomers has as many H atoms per skeleton in every member, so
    its H balance repeats its skeleton balance, and a Gibbs reactor's
    element multipliers are then not unique (IPOPT reports the problem
    unbounded). Labels that add no rank are dropped.
    """
    import numpy as np

    labels = sorted({e for n in names for e in element_counts(cd, n)})
    labels.sort(key=lambda e: e == "H")  # skeletons first, H last
    keep, rows = [], []
    for e in labels:
        row = [element_counts(cd, n).get(e, 0) for n in names]
        if np.linalg.matrix_rank(np.array(rows + [row], dtype=float)) > len(rows):
            keep.append(e)
            rows.append(row)
    return keep


def _config(cd: dict, names) -> dict:
    from idaes.core import Component, VaporPhase
    from idaes.models.properties.modular_properties.eos.ideal import Ideal
    from idaes.models.properties.modular_properties.pure import RPP4
    from idaes.models.properties.modular_properties.state_definitions import FTPx
    from pyomo.environ import units as u

    elements = independent_elements(cd, names)
    comps = {}
    for n in names:
        d = cd[n]
        el = {e: v for e, v in element_counts(cd, n).items() if e in elements}
        a, b, c, dd = d["cp"]
        comps[n] = {
            "type": Component, "valid_phase_types": None,
            "elemental_composition": el,
            "cp_mol_ig_comp": RPP4, "enth_mol_ig_comp": RPP4, "entr_mol_ig_comp": RPP4,
            "parameter_data": {
                "mw": (d["MW"] / 1e3, u.kg / u.mol),
                "pressure_crit": (1e6, u.Pa), "temperature_crit": (500.0, u.K),
                "cp_mol_ig_comp_coeff": {"A": (a, u.J / u.mol / u.K),
                                         "B": (b, u.J / u.mol / u.K**2),
                                         "C": (c, u.J / u.mol / u.K**3),
                                         "D": (dd, u.J / u.mol / u.K**4)},
                "enth_mol_form_vap_comp_ref": (d["Hf"], u.J / u.mol),
                "entr_mol_form_vap_comp_ref": (d["S"], u.J / u.mol / u.K),
            },
        }
    for c in comps.values():
        del c["valid_phase_types"]
    return {
        "components": comps,
        "phases": {"Vap": {"type": VaporPhase, "equation_of_state": Ideal}},
        "base_units": {"time": u.s, "length": u.m, "mass": u.kg, "amount": u.mol,
                       "temperature": u.K},
        "state_definition": FTPx,
        "state_bounds": {"flow_mol": (0, 10, 1e4, u.mol / u.s), "temperature": (250, 400, 900, u.K),
                         "pressure": (1e4, 1e5, 1e8, u.Pa)},
        "pressure_ref": (1e5, u.Pa),
        "temperature_ref": (298.15, u.K),
    }


def gibbs(cd: dict, feed: dict, T_in: float, P: float, T_out: float | None, ipopt: str) -> dict:
    """One IDAES GibbsReactor solve: isothermal at ``T_out``, or adiabatic
    (``T_out=None``). Returns the outlet."""
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties import GenericParameterBlock
    from idaes.models.unit_models import GibbsReactor
    from pyomo.environ import ConcreteModel, SolverFactory, value

    names = list(feed)
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**_config(cd, names))
    m.fs.R = GibbsReactor(property_package=m.fs.props, has_heat_transfer=True,
                          has_pressure_change=False)
    tot = sum(feed.values())
    inlet = m.fs.R.inlet
    inlet.flow_mol.fix(tot)
    inlet.temperature.fix(T_in)
    inlet.pressure.fix(P)
    for n in names:
        inlet.mole_frac_comp[0, n].fix(feed[n] / tot)
    if T_out is None:
        m.fs.R.heat_duty.fix(0.0)
    else:
        m.fs.R.outlet.temperature.fix(T_out)
    m.fs.R.initialize()
    s = SolverFactory("ipopt", executable=ipopt)
    s.options["tol"] = 1e-10
    res = s.solve(m, tee=bool(os.environ.get("ISOM_TEE")), load_solutions=False)
    m.solutions.load_from(res)
    out = m.fs.R.outlet
    F = value(out.flow_mol[0])
    return {
        "T": value(out.temperature[0]),
        "flows": {n: F * value(out.mole_frac_comp[0, n]) for n in names},
        "heat_duty": value(m.fs.R.heat_duty[0]),
        "solve": {"status": str(res.solver.termination_condition),
                  "message": str(res.solver.message)},
    }


def family_reference(cd, ipopt) -> dict:
    from difflow_refinery.isomerization import thermochem as tc

    out = {}
    for fam, names in tc.FAMILIES.items():
        rows = []
        for T in FAMILY_T:
            feed = {n: 1.0 / len(names) for n in names}
            g = gibbs(cd, feed, T, P_REACTOR, T, ipopt)
            tot = sum(g["flows"].values())
            rows.append({"T": T, "x": {n: v / tot for n, v in g["flows"].items()},
                         "solve": g["solve"]})
        out[fam] = rows
    return out


def ring_reference(cd, ipopt) -> list:
    rows = []
    feed = dict(C6_RING["feed"])
    for n in ("2_methylpentane", "3_methylpentane", "2_3_dimethylbutane",
              "2_2_dimethylbutane"):
        feed[n] = 1e-3  # present, so IDAES has a logarithm to start from
    for T in C6_RING["T"]:
        g = gibbs(cd, feed, T, C6_RING["P"], T, ipopt)
        rows.append({"T": T, "feed": feed, **g})
    return rows


def adiabatic_reference(cd, ipopt) -> dict:
    out = {}
    for kind in ADIABATIC["feeds"]:
        feed = adiabatic_charge(kind)
        g = gibbs(cd, feed, ADIABATIC["T_in"], ADIABATIC["P"], None, ipopt)
        out[kind] = {"feed": feed, **g}
    return out


def provenance(ipopt) -> dict:
    import idaes
    import pyomo.version

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    try:
        ver = subprocess.check_output([ipopt, "--version"], text=True).splitlines()[0]
    except Exception:  # noqa: BLE001
        ver = ipopt
    return {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": "tests/refinery/reference/isom_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.isom_generate",
        "idaes": idaes.__version__, "pyomo": pyomo.version.version, "ipopt": ver,
        "python": platform.python_version(),
        "difflow_commit": git("rev-parse", "HEAD"),
        "model": ("IDAES GibbsReactor on a modular ideal-gas package (Ideal EOS, RPP4) given "
                  "difflow's dHf, S and Cp: an independent implementation of the same "
                  "thermochemistry, not an independent one"),
        "elements": ("H plus one skeleton label per conserved carbon skeleton (C5, C6) and per "
                     "species the network holds fixed; see the generator's docstring"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ipopt", default=DEFAULT_IPOPT)
    args = ap.parse_args(argv)
    cd = component_data()
    out = {"provenance": provenance(args.ipopt), "components": cd,
           "cases": {"P": P_REACTOR, "family_T": list(FAMILY_T), "c6_ring": C6_RING,
                     "adiabatic": ADIABATIC, "skeleton": SKELETON},
           "families": family_reference(cd, args.ipopt),
           "c6_ring": ring_reference(cd, args.ipopt),
           "adiabatic": adiabatic_reference(cd, args.ipopt)}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {"T": v["T"], "solve": v["solve"]["status"]}
                      for k, v in out["adiabatic"].items()}, indent=1))


if __name__ == "__main__":
    main()
