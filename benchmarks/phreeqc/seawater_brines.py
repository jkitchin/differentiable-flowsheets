"""PHREEQC (pitzer.dat) reference for difflow_water's seawater speciation.

Run where phreeqpython is installed (NOT a difflow dependency)::

    pip install phreeqpython==1.6.2
    python benchmarks/phreeqc/seawater_brines.py   # writes seawater_brines_reference.json

Reference-Composition seawater (the same totals difflow_water's
``reference_seawater`` builds: Br in Cl; Sr, F, B dropped) at 35 g/kg and
pH 8.1, evaporated by concentration factors 1-3 (REACTION removing water),
at 15, 25 and 35 C. Records pH, ionic strength, water activity, alkalinity,
total C and saturation indices. ``compare_seawater_brines.py`` runs
difflow_water on the same totals.
"""

import json
from importlib import metadata
from pathlib import Path

from phreeqpython import PhreeqPython

HERE = Path(__file__).parent
X = {"Na": 4188071, "Mg": 471678, "Ca": 91823, "K": 91159, "Cl": 4874839 + 7520,
     "S(6)": 252152, "C(4)": 15340 + 2134 + 86}
M_S = 31.4038218e-3
S = 0.035
CF = [1.0, 1.5, 2.0, 2.5, 3.0]
TEMPS = [15.0, 25.0, 35.0]
PHASES = ["Calcite", "Aragonite", "Gypsum", "Anhydrite", "Brucite", "Magnesite", "Dolomite", "Halite"]


def main():
    pp = PhreeqPython(database="pitzer.dat")
    m = S / ((1 - S) * M_S)
    totals = {k: v * 1e-7 * m for k, v in X.items()}
    out = {"generated_by": "benchmarks/phreeqc/seawater_brines.py",
           "phreeqpython": metadata.version("phreeqpython"), "salinity": S, "pH_feed": 8.1,
           "cases": []}
    for T in TEMPS:
        for cf in CF:
            water_removed = (1.0 - 1.0 / cf) / 0.01801528  # mol H2O from 1 kg
            sol_lines = "\n".join(f"    {k} {v * 1e3:.10g}" for k, v in totals.items())
            inp = f"""
SOLUTION 1
    temp {T}
    pH 8.1
    units mmol/kgw
{sol_lines}
REACTION 1
    H2O -1
    {water_removed:.12g} moles
SELECTED_OUTPUT
    -reset false
    -pH true
    -ionic_strength true
    -alkalinity true
    -totals Na K Mg Ca Cl S(6) C(4)
    -activities H2O
    -saturation_indices {' '.join(PHASES)}
    -water true
END
"""
            pp.ip.run_string(inp)
            rows = pp.ip.get_selected_output_array()
            hdr, vals = rows[0], rows[-1]
            rec = dict(zip(hdr, vals))
            out["cases"].append({"T_C": T, "cf": cf, "row": rec})
    (HERE / "seawater_brines_reference.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
