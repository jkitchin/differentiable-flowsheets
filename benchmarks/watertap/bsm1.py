"""WaterTAP reference for the difflow_water activated sludge plant (BSM1 layout, ASM1).

Run in an environment with WaterTAP and IPOPT (NOT a difflow dependency)::

    python benchmarks/watertap/bsm1.py      # writes bsm1_reference.json

The model is WaterTAP 1.8.0's ``flowsheets/activated_sludge/ASM1_flowsheet.py``
(its own ``build_flowsheet``): five ASM1 CSTRs, the clarifier as a
component-split separator, internal recycle 60 % of the last reactor's
outflow, external recycle 97.955 % of the clarifier underflow; KLa = 10 1/h
in reactors 3-4 and dissolved oxygen fixed at 0.449 g/m^3 in reactor 5.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from pyomo.environ import value
from watertap.flowsheets.activated_sludge import ASM1_flowsheet as fsm

HERE = Path(__file__).parent
STATES = ["S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P", "S_O", "S_NO", "S_NH",
          "S_ND", "X_ND"]


def conc(port):
    out = {s: float(value(port.conc_mass_comp[0, s])) * 1e3 for s in STATES}   # g/m^3
    out["S_ALK"] = float(value(port.alkalinity[0])) * 1e3                      # mol/m^3
    out["Q"] = float(value(port.flow_vol[0])) * 86400.0                         # m^3/d
    return out


def main() -> None:
    m, res = fsm.build_flowsheet()
    fs = m.fs
    out = {
        "versions": {"watertap": metadata.version("watertap"),
                     "idaes-pse": metadata.version("idaes-pse"),
                     "pyomo": metadata.version("pyomo"), "python": sys.version.split()[0]},
        "flowsheet": "watertap.flowsheets.activated_sludge.ASM1_flowsheet.build_flowsheet",
        "inputs": {
            "influent": conc(fs.FeedWater.outlet),
            "volumes": [float(value(getattr(fs, f"R{k}").volume[0])) for k in range(1, 6)],
            "KLa_per_day": [0.0, 0.0, float(value(fs.R3.KLa)) * 24, float(value(fs.R4.KLa)) * 24],
            "S_O_R5": float(value(fs.R5.outlet.conc_mass_comp[0, "S_O"])) * 1e3,
            "internal_recycle_fraction": float(value(fs.SP5.split_fraction[0, "underflow"])),
            "effluent_split_water": float(value(fs.CL1.split_fraction[0, "effluent", "H2O"])),
            "effluent_split_particulate": float(value(fs.CL1.split_fraction[0, "effluent", "X_I"])),
            "recycle_fraction": float(value(fs.SP6.split_fraction[0, "recycle"])),
        },
        "outputs": {f"R{k}": conc(getattr(fs, f"R{k}").outlet) for k in range(1, 6)},
    }
    out["outputs"]["effluent"] = conc(fs.Treated.inlet)
    out["outputs"]["wastage"] = conc(fs.Sludge.inlet)
    (HERE / "bsm1_reference.json").write_text(json.dumps(out, indent=1))
    print({k: round(v["S_NH"], 4) for k, v in out["outputs"].items()})


if __name__ == "__main__":
    main()
