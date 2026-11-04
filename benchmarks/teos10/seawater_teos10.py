"""TEOS-10 reference values for difflow_water's seawater property correlations.

Run in an environment with the TEOS-10 GSW-Python package (NOT a difflow
dependency)::

    python -m venv gswenv && gswenv/bin/pip install gsw
    gswenv/bin/python benchmarks/teos10/seawater_teos10.py   # writes seawater_teos10.json

TEOS-10 (IOC, SCOR and IAPSO 2010 [S4]) is the international standard for
seawater; its Gibbs function is the IAPWS-2008 formulation against which
Sharqawy et al. (2010) [S1] state the accuracy of their correlations. GSW
is its reference implementation. Values at sea pressure 0 (1 atm), on a
grid of temperature (C) and Absolute Salinity (g/kg; for Reference-
Composition seawater, 1000 times difflow's mass-fraction salinity).

The latent heat (``gsw.latentheat_evap_t``) is per kg of water evaporated,
so it compares with [S1] Eq. 54 (pure water), not Eq. 37 (per kg of
seawater). Osmotic pressure from the chemical potential of water,
``pi = (mu_W(0, t, p) - mu_W(S_A, t, p)) rho_pure(t, p)`` (gsw gives mu_W in
J/g): the same dilute
form, ``-(R T / V_w) ln a_w``, that difflow's osmotic pressure is.
The grid stays inside TEOS-10's range (t <= 40 C for these salinities, S_A
<= 70 g/kg).
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

import gsw

HERE = Path(__file__).parent
T_C = [5.0, 15.0, 25.0, 35.0, 40.0]
SA = [0.0, 10.0, 35.0, 50.0, 70.0]


def main() -> None:
    rows = []
    for t in T_C:
        for sa in SA:
            rho = float(gsw.rho_t_exact(sa, t, 0.0))
            rho_w = float(gsw.rho_t_exact(0.0, t, 0.0))
            mu = float(gsw.chem_potential_water_t_exact(sa, t, 0.0))
            mu0 = float(gsw.chem_potential_water_t_exact(0.0, t, 0.0))
            rows.append({
                "t_C": t, "SA_g_kg": sa, "density": rho,
                "cp": float(gsw.cp_t_exact(sa, t, 0.0)),
                "enthalpy": float(gsw.enthalpy_t_exact(sa, t, 0.0)),
                "latent_heat": float(gsw.latentheat_evap_t(sa, t)),
                # gsw's chemical potential is in J/g
                "osmotic_pressure": (mu0 - mu) * 1000.0 * rho_w,
            })
    out = {"source": "TEOS-10 via GSW-Python (gsw.rho_t_exact, cp_t_exact, enthalpy_t_exact, "
                     "latentheat_evap_t, chem_potential_water_t_exact) at sea pressure 0",
           "versions": {"gsw": metadata.version("gsw"), "python": sys.version.split()[0]},
           "rows": rows}
    (HERE / "seawater_teos10.json").write_text(json.dumps(out, indent=1))
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
