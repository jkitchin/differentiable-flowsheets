"""WaterTAP reference for the difflow_water GAC adsorber.

Run in an environment with WaterTAP and IPOPT (NOT a difflow dependency)::

    python benchmarks/watertap/gac.py      # writes gac_reference.json

The model is WaterTAP 1.8.0's ``GAC`` in its own example flowsheet
(``flowsheets/gac/gac.py``: ``build`` with fixed film-transfer and
surface-diffusion coefficients, ``initialize``), simulated at the example's
inputs and at variations of EBCT, replacement ratio and Freundlich
parameters. Performance only: costing is not compared yet.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from pyomo.environ import value
from watertap.core.solvers import get_solver
from watertap.flowsheets.gac import gac as gfs

HERE = Path(__file__).parent

CASES = {
    "example": {},
    "long_ebct": {"ebct": 900.0},
    "replace_at_0.2": {"conc_ratio_replace": 0.2},
    "freundlich_ninv_0.6": {"freund_ninv": 0.6, "freund_k": 37.9},
}
OUT_KEYS = ["equil_conc", "dg", "N_Bi", "min_N_St", "min_ebct", "throughput",
            "min_residence_time", "residence_time", "min_operational_time",
            "operational_time", "bed_volumes_treated", "bed_area", "bed_diameter",
            "bed_volume", "bed_mass_gac", "gac_usage_rate", "conc_ratio_avg",
            "mass_adsorbed", "velocity_sup"]


def main() -> None:
    out = {"versions": {"watertap": metadata.version("watertap"),
                        "idaes-pse": metadata.version("idaes-pse"),
                        "pyomo": metadata.version("pyomo"),
                        "python": sys.version.split()[0]},
           "flowsheet": "watertap.flowsheets.gac.gac (build with fixed kf and ds, initialize)",
           "cases": {}}
    solver = get_solver()
    for name, change in CASES.items():
        m = gfs.build()
        g = m.fs.gac
        for k, v in change.items():
            getattr(g, k).fix(v)
        gfs.initialize(m)
        r = solver.solve(m)
        assert str(r.solver.termination_condition) == "optimal", name
        feed = m.fs.feed.properties[0]
        inputs = {k: float(value(getattr(g, k))) for k in
                  ("freund_k", "freund_ninv", "kf", "ds", "particle_dens_app", "particle_dia",
                   "ebct", "bed_voidage", "bed_length", "conc_ratio_replace",
                   "a0", "a1", "b0", "b1", "b2", "b3", "b4")}
        inputs.update(
            flow_mass_H2O=float(value(feed.flow_mass_phase_comp["Liq", "H2O"])),
            flow_mass_solute=float(value(feed.flow_mass_phase_comp["Liq", "solute"])),
            flow_vol=float(value(feed.flow_vol_phase["Liq"])),
            T=float(value(feed.temperature)), P=float(value(feed.pressure)))
        outputs = {k: float(value(getattr(g, k))) for k in OUT_KEYS}
        # the example flowsheet is on a molar basis: the product's mass flow is
        # not a constrained variable (it keeps its initial value); use moles
        outputs["outlet_solute_mass"] = float(value(
            m.fs.product.properties[0].flow_mol_phase_comp["Liq", "solute"])) * 0.08
        out["cases"][name] = {"inputs": inputs, "outputs": outputs}
        print(name, outputs["conc_ratio_avg"], outputs["operational_time"])
    (HERE / "gac_reference.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
