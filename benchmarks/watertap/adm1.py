"""WaterTAP reference for the difflow_water anaerobic digester (ADM1).

Run in an environment with WaterTAP and IPOPT (NOT a difflow dependency)::

    python benchmarks/watertap/adm1.py      # writes adm1_reference.json

The model is WaterTAP 1.8.0's ``flowsheets/anaerobic_digester/ADM1_flowsheet.py``
(its own ``build_flowsheet``: the AD unit with the ADM1 property and
reaction packages at the inputs of Rosen & Jeppsson's steady-state case).
Concentrations are written in the Rosen & Jeppsson units: kg COD/m^3,
kmol C/m^3 for S_IC (WaterTAP stores kg C), kmol N/m^3 for S_IN (kg N).
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from pyomo.environ import value
from watertap.flowsheets.anaerobic_digester import ADM1_flowsheet as fsm

HERE = Path(__file__).parent
STATES = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC",
          "S_IN", "S_I", "X_c", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro",
          "X_ac", "X_h2", "X_I"]


def conc(port):
    out = {}
    for s in STATES:
        v = float(value(port.conc_mass_comp[0, s]))
        out[s] = v / 12.0 if s == "S_IC" else v / 14.0 if s == "S_IN" else v
    out["S_cat"] = float(value(port.cations[0]))
    out["S_an"] = float(value(port.anions[0]))
    out["Q"] = float(value(port.flow_vol[0])) * 86400.0
    out["T"] = float(value(port.temperature[0]))
    return out


def main() -> None:
    res = fsm.build_flowsheet()
    m = res[0] if isinstance(res, tuple) else res
    r1 = m.fs.R1
    vap = r1.vapor_outlet
    out = {
        "versions": {"watertap": metadata.version("watertap"),
                     "idaes-pse": metadata.version("idaes-pse"),
                     "pyomo": metadata.version("pyomo"), "python": sys.version.split()[0]},
        "flowsheet": "watertap.flowsheets.anaerobic_digester.ADM1_flowsheet.build_flowsheet",
        "inputs": {"influent": conc(r1.inlet),
                   "V_liq": float(value(r1.volume_liquid[0])),
                   "V_gas": float(value(r1.volume_vapor[0]))},
        "outputs": {
            "liquid": conc(r1.liquid_outlet),
            "pH": float(value(r1.liquid_phase.reactions[0].pH)),
            "q_gas": float(value(vap.flow_vol[0])) * 86400.0,
            "P_gas_bar": float(value(vap.pressure[0])) / 1e5,
            "gas_conc": {k: float(value(vap.conc_mass_comp[0, k]))
                         for k in ("S_h2", "S_ch4", "S_co2")},
        },
    }
    (HERE / "adm1_reference.json").write_text(json.dumps(out, indent=1))
    print(out["outputs"]["pH"], out["outputs"]["q_gas"])


if __name__ == "__main__":
    main()
