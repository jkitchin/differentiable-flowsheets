"""Generate ``gibbs_reference.json``: IDAES GibbsReactor cross-checks of
difflow's :class:`~difflow.units.gibbs.GibbsReactor` (#468).

Run from the repository root (needs IDAES, Pyomo and an IPOPT)::

    PYTHONPATH=src:. python tests/reference/gibbs_generate.py [--ipopt PATH]

IDAES's ``GibbsReactor`` minimises the total Gibbs energy subject to the
element balances. It runs on an ideal-gas modular property package
(``Ideal`` EOS, ``RPP4`` pure-component methods, 1 bar reference pressure).
The package gets the JANAF-fitted constants of
``tests/test_gibbs_reactor.py::janaf_thermo``: Hf(298), S(298) and the Cp
cubic of CH4, H2O, CO, CO2 and H2. The cases are:

* **isothermal**: steam-methane-reforming feed (S/C = 3) at 900-1200 K and
  1 bar, and at 1000-1200 K and 20 bar. Lower temperatures are left out
  because IDAES's isothermal (fixed outlet T) Gibbs system does not converge
  there: at 800 K (and at 900 K and 20 bar) its
  ``initialize()`` fails, and IPOPT (3.13.2, ma27) cycles in restoration with
  infeasibility 1-100 for 5000 iterations, also after a continuation in T from
  1000 K and with other scaling options. Its last iterate is within 2 % of
  difflow's answer. Those cases are left out rather than recorded
  unconverged. The adiabatic case, which ends at 855 K and 20 bar, does
  converge.
* **shift**: a CO/H2O/H2 feed at 600 K and 30 bar, where methanation wins.
* **adiabatic**: the reforming feed at 1100 K and 20 bar, Q = 0.

This is an **independent implementation, not an independent model**. The
same thermochemistry is minimised by IPOPT on IDAES's formulation instead of
by difflow's element-potential Newton. It checks the free-energy assembly,
the standard-state convention, the pressure dependence and the energy
balance. It does not check the constants: both sides get the same ones
(``test_gibbs_reactor.py`` checks those against JANAF's log Kf).
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
OUT = HERE / "gibbs_reference.json"
DEFAULT_IPOPT = os.path.expanduser("~/.idaes/bin/ipopt")

NAMES = ("CH4", "H2O", "CO", "CO2", "H2")
MW = {"CH4": 16.043, "H2O": 18.015, "CO": 28.010, "CO2": 44.010, "H2": 2.016}
#: a trace of every product, so IDAES starts with a logarithm to take
TRACE = 1e-4
CASES = {
    "isothermal": [{"feed": {"CH4": 1.0, "H2O": 3.0}, "T": T, "P": P}
                   for P, Ts in ((1e5, (900.0, 1000.0, 1200.0)), (2e6, (1000.0, 1100.0, 1200.0)))
                   for T in Ts],
    "shift": [{"feed": {"CO": 1.0, "H2O": 1.5, "H2": 0.2}, "T": 600.0, "P": 3e6}],
    "adiabatic": [{"feed": {"CH4": 1.0, "H2O": 3.0}, "T": 1100.0, "P": 2e6}],
}


def full_feed(feed: dict) -> dict:
    """The feed IDAES and difflow both see (mol/s): every species, traces added."""
    return {n: float(feed.get(n, 0.0)) or TRACE for n in NAMES}


def component_data() -> dict:
    from tests.test_gibbs_reactor import janaf_thermo

    th = janaf_thermo(NAMES)
    return {n: {"Hf": float(th.Hf[i]), "S": float(th.S0[i]), "cp": [float(c) for c in th.cp[i]],
                "elements": {e: int(th.ELEMENTS[e][i]) for e in th.ELEMENTS
                             if th.ELEMENTS[e][i]}, "MW": MW[n]}
            for i, n in enumerate(NAMES)}


def _config(cd: dict) -> dict:
    from idaes.core import Component, VaporPhase
    from idaes.models.properties.modular_properties.eos.ideal import Ideal
    from idaes.models.properties.modular_properties.pure import RPP4
    from idaes.models.properties.modular_properties.state_definitions import FTPx
    from pyomo.environ import units as u

    comps = {}
    for n in NAMES:
        d = cd[n]
        a, b, c, dd = d["cp"]
        comps[n] = {
            "type": Component,
            "elemental_composition": d["elements"],
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
    return {
        "components": comps,
        "phases": {"Vap": {"type": VaporPhase, "equation_of_state": Ideal}},
        "base_units": {"time": u.s, "length": u.m, "mass": u.kg, "amount": u.mol,
                       "temperature": u.K},
        "state_definition": FTPx,
        "state_bounds": {"flow_mol": (0, 10, 1e4, u.mol / u.s),
                         "temperature": (250, 900, 2000, u.K),
                         "pressure": (1e4, 1e5, 1e8, u.Pa)},
        "pressure_ref": (1e5, u.Pa),
        "temperature_ref": (298.15, u.K),
    }


def gibbs(cd: dict, feed: dict, T_in: float, P: float, T_out: float | None,
          ipopt: str) -> dict:
    """One IDAES GibbsReactor solve: isothermal at ``T_out``, or adiabatic
    (``T_out=None``)."""
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties import GenericParameterBlock
    from idaes.models.unit_models import GibbsReactor
    from pyomo.environ import ConcreteModel, SolverFactory, value

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**_config(cd))
    m.fs.R = GibbsReactor(property_package=m.fs.props, has_heat_transfer=True,
                          has_pressure_change=False)
    tot = sum(feed.values())
    inlet = m.fs.R.inlet
    inlet.flow_mol.fix(tot)
    inlet.temperature.fix(T_in)
    inlet.pressure.fix(P)
    for n in NAMES:
        inlet.mole_frac_comp[0, n].fix(feed[n] / tot)
    if T_out is None:
        m.fs.R.heat_duty.fix(0.0)
    else:
        m.fs.R.outlet.temperature.fix(T_out)
    try:
        m.fs.R.initialize()
        init = "initialize() converged"
    except Exception as exc:  # noqa: BLE001 -- IDAES's init can stall; the full solve decides
        init = f"initialize() failed ({type(exc).__name__}); solved from its last point"
    s = SolverFactory("ipopt", executable=ipopt)
    s.options["tol"] = 1e-10
    s.options["max_iter"] = 5000
    res = s.solve(m, tee=bool(os.environ.get("GIBBS_TEE")), load_solutions=False)
    m.solutions.load_from(res)
    out = m.fs.R.outlet
    F = value(out.flow_mol[0])
    return {
        "T_out": value(out.temperature[0]),
        "flows": {n: F * value(out.mole_frac_comp[0, n]) for n in NAMES},
        "heat_duty": value(m.fs.R.heat_duty[0]),
        "solve": {"status": str(res.solver.termination_condition), "init": init,
                  "message": str(res.solver.message)},
    }


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
        "script": "tests/reference/gibbs_generate.py",
        "command": "PYTHONPATH=src:. python tests/reference/gibbs_generate.py",
        "idaes": idaes.__version__, "pyomo": pyomo.version.version, "ipopt": ver,
        "python": platform.python_version(),
        "difflow_commit": git("rev-parse", "HEAD"),
        "model": ("IDAES GibbsReactor on a modular ideal-gas package (Ideal EOS, RPP4) given "
                  "JANAF-fitted Hf, S and Cp (tests/test_gibbs_reactor.py::janaf_thermo): an "
                  "independent implementation of the same thermochemistry"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ipopt", default=DEFAULT_IPOPT)
    args = ap.parse_args(argv)
    cd = component_data()
    results = {}
    for kind, cases in CASES.items():
        rows = []
        for c in cases:
            feed = full_feed(c["feed"])
            T_out = None if kind == "adiabatic" else c["T"]
            rows.append({"feed": feed, "T": c["T"], "P": c["P"],
                         **gibbs(cd, feed, c["T"], c["P"], T_out, args.ipopt)})
        results[kind] = rows
    out = {"provenance": provenance(args.ipopt), "components": cd, "trace": TRACE,
           "results": results}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for kind, rows in results.items():
        for r in rows:
            print(kind, r["T"], r["P"], round(r["T_out"], 3), r["solve"]["status"])


if __name__ == "__main__":
    main()
