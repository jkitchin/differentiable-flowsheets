"""WaterTAP reference for the difflow_water mechanical vapor compression (MVC) plant.

Run in an environment with WaterTAP and IPOPT (NOT a difflow dependency)::

    pip install watertap==1.8.0
    idaes get-extensions
    python benchmarks/watertap/mvc.py      # writes mvc_reference.json

The plant is WaterTAP's ``flowsheets/mvc/mvc_single_stage.py`` (feed pump,
feed split to distillate and brine preheaters, evaporator, compressor,
condenser, brine and distillate pumps), built with WaterTAP's own ``build``,
``set_operating_conditions`` and ``initialize_system``.

Cases:

- ``optimum``: WaterTAP's ``main()`` (minimum LCOW over evaporator brine
  temperature, compressor pressure ratio and the two preheater areas, no
  external heat). Recorded as an optimization result AND as a simulation
  point (its decision variables are the inputs ``compare_mvc.py`` uses).
- ``sim_*``: simulations at fixed brine temperature, pressure ratio and
  preheater areas with no external heat; the evaporator area is computed.
  At WaterTAP's optimum the condensate leaves exactly saturated (the
  condenser's ``P >= P_sat`` bound is active), so a higher pressure ratio at
  the same design is infeasible: the cases sit at lower pressure ratios.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from pyomo.environ import value
from watertap.flowsheets.mvc import mvc_single_stage as mvc

HERE = Path(__file__).parent

SIM_CASES = {
    "sim_lower_PR": dict(T_brine=348.15, pressure_ratio=1.58, A_hx_distillate=199.0,
                         A_hx_brine=174.0),
    "sim_PR_150": dict(T_brine=348.15, pressure_ratio=1.50, A_hx_distillate=199.0,
                       A_hx_brine=174.0),
    "sim_hotter": dict(T_brine=350.15, pressure_ratio=1.58, A_hx_distillate=199.0,
                       A_hx_brine=174.0),
}


def record(m) -> dict:
    fs = m.fs
    v = lambda x: float(value(x))  # noqa: E731
    ev, cp, cd = fs.evaporator, fs.compressor, fs.condenser
    cost = {
        nm: v(getattr(fs, nm).costing.capital_cost)
        for nm in ("pump_feed", "pump_brine", "pump_distillate", "hx_distillate",
                   "hx_brine", "mixer_feed", "evaporator", "compressor")
    }
    return {
        "inputs": {
            "feed_H2O_kg_s": v(fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]),
            "feed_TDS_kg_s": v(fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]),
            "T_feed": v(fs.feed.properties[0].temperature),
            "P_feed": v(fs.feed.properties[0].pressure),
            "recovery": v(fs.recovery[0]),
            "T_brine": v(ev.properties_brine[0].temperature),
            "pressure_ratio": v(cp.pressure_ratio),
            "compressor_efficiency": v(cp.efficiency),
            "U_evaporator": v(ev.U),
            "A_hx_distillate": v(fs.hx_distillate.area),
            "A_hx_brine": v(fs.hx_brine.area),
            "U_hx": v(fs.hx_distillate.overall_heat_transfer_coefficient[0]),
            "electricity_cost": v(fs.costing.electricity_cost),
        },
        "outputs": {
            "Q_ext": v(fs.Q_ext[0]),
            "evaporator_area": v(ev.area),
            "evaporator_heat": v(ev.heat_transfer),
            "evaporator_lmtd": v(ev.lmtd),
            "T_evaporator_feed": v(ev.properties_feed[0].temperature),
            "P_brine": v(ev.properties_brine[0].pressure),
            "brine_salinity": v(ev.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"]),
            "T_compressed_vapor": v(cp.control_volume.properties_out[0].temperature),
            "P_compressed_vapor": v(cp.control_volume.properties_out[0].pressure),
            "compressor_work": v(cp.control_volume.work[0]),
            "T_condensate": v(cd.control_volume.properties_out[0].temperature),
            "T_distillate_out": v(fs.distillate.properties[0].temperature),
            "T_brine_out": v(fs.brine.properties[0].temperature),
            "pump_feed_work": v(fs.pump_feed.work_mechanical[0]),
            "pump_brine_work": v(fs.pump_brine.work_mechanical[0]),
            "pump_distillate_work": v(fs.pump_distillate.work_mechanical[0]),
            "product_flow_vol": v(fs.distillate.properties[0].flow_vol),
            "SEC": v(fs.costing.specific_energy_consumption),
            "LCOW": v(fs.costing.LCOW),
            "capital": cost,
            "total_capital": v(fs.costing.total_capital_cost),
            "total_operating": v(fs.costing.total_operating_cost),
        },
    }


def main() -> None:
    out = {
        "versions": {
            "watertap": metadata.version("watertap"),
            "idaes-pse": metadata.version("idaes-pse"),
            "pyomo": metadata.version("pyomo"),
            "python": sys.version.split()[0],
        },
        "flowsheet": "watertap.flowsheets.mvc.mvc_single_stage (build, "
                     "set_operating_conditions, initialize_system, main)",
        "cases": {},
    }
    m, res = mvc.main()
    out["cases"]["optimum"] = record(m)
    solver = mvc.get_solver()
    for name, case in SIM_CASES.items():
        m.fs.objective.deactivate()
        m.fs.evaporator.outlet_brine.temperature[0].fix(case["T_brine"])
        m.fs.compressor.pressure_ratio.fix(case["pressure_ratio"])
        m.fs.hx_distillate.area.fix(case["A_hx_distillate"])
        m.fs.hx_brine.area.fix(case["A_hx_brine"])
        m.fs.evaporator.area.unfix()
        m.fs.Q_ext[0].fix(0.0)
        r = solver.solve(m)
        if str(r.solver.termination_condition) != "optimal":
            # infeasible: the compressor delivers more energy than the plant can
            # absorb (WaterTAP's condenser needs subcooled or saturated condensate)
            print(f"{name}: {r.solver.termination_condition}; not recorded")
            continue
        out["cases"][name] = record(m)
    (HERE / "mvc_reference.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: c["outputs"]["LCOW"] for k, c in out["cases"].items()}))


if __name__ == "__main__":
    main()
