"""Fit difflow's gas transport-property correlations (difflow.gas_properties).

Reads the committed reference tables and prints the coefficient block that
``src/difflow/gas_properties.py`` holds, plus the worst deviation of each
fit. Needs only numpy::

    python scripts/fit_gas_transport.py

Each property is stored as ``(coeffs, Tmin, Tmax)``, the range being that
of the data fitted. Form: ``ln y = sum_i c_i x^i`` with ``x = ln(T / 300 K)``, cubic for
viscosity and conductivity and quintic for heat capacity (whose curvature
the cubic missed by up to 1 %),
fitted by least squares on ln y (relative error), for y = viscosity (Pa s),
thermal conductivity (W/m/K) and isobaric heat capacity (J/kg/K) at
0.101325 MPa.

Data:
- tests/reference/nist_gases_1atm/<gas>.csv: NIST Chemistry WebBook SRD 69
  isobars at 1 atm (vapour phase only).
- tests/reference/air_engineering_toolbox.csv: dry air mu and k. Air cp is
  the mass-weighted NIST cp of N2/O2/Ar at mole fractions 0.7812/0.2096/
  0.0092 (ideal mixing; good to ~0.1 % at 1 atm).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference"
GASES = ["nitrogen", "oxygen", "argon", "carbon_dioxide", "carbon_monoxide",
         "methane", "hydrogen", "helium", "water"]
MW = {"nitrogen": 28.0134, "oxygen": 31.9988, "argon": 39.948,
      "carbon_dioxide": 44.0095, "carbon_monoxide": 28.0101,
      "methane": 16.0428, "hydrogen": 2.01588, "helium": 4.002602,
      "water": 18.015268}
AIR = {"nitrogen": 0.7812, "oxygen": 0.2096, "argon": 0.0092}


def load_nist(gas):
    d = np.loadtxt(REF / "nist_gases_1atm" / f"{gas}.csv", delimiter=",",
                   comments="#", skiprows=2)
    return d[:, 0], d[:, 1] * 1e3, d[:, 2], d[:, 3]    # T, cp J/kg/K, mu, k


def load_air():
    out = {"mu": [], "k": []}
    with open(REF / "air_engineering_toolbox.csv") as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#"))]
    for prop, T, v in rows[1:]:
        out[prop].append((float(T), float(v)))
    return {p: np.array(v) for p, v in out.items()}


def fit(T, y, order=3):
    x = np.log(T / 300.0)
    c = np.polyfit(x, np.log(y), order)[::-1]
    err = np.abs(np.exp(np.polyval(c[::-1], x)) / y - 1).max()
    return ([float(v) for v in c], float(T.min()), float(T.max())), float(err)


def air_cp(T):
    MWmix = sum(x * MW[g] for g, x in AIR.items())
    cp = 0.0
    for g, x in AIR.items():
        Tg, cpg, _, _ = load_nist(g)
        cp = cp + x * MW[g] / MWmix * np.interp(T, Tg, cpg)
    return cp


def main():
    table = {}
    for g in GASES:
        T, cp, mu, k = load_nist(g)
        entry = {"MW": MW[g]}
        for name, y in (("mu", mu), ("k", k), ("cp", cp)):
            entry[name], e = fit(T, y, 5 if name == "cp" else 3)
            print(f"# {g:16s} {name:2s} max dev {100 * e:.3f} %")
        table[g] = entry
    air = load_air()
    T_air = np.arange(230.0, 1001.0, 10.0)
    entry = {"MW": round(sum(x * MW[g] for g, x in AIR.items()), 4)}
    for name in ("mu", "k"):
        entry[name], e = fit(air[name][:, 0], air[name][:, 1])
        print(f"# {'air':16s} {name:2s} max dev {100 * e:.3f} %")
    entry["cp"], e = fit(T_air, air_cp(T_air), 5)
    print(f"# {'air':16s} cp max dev {100 * e:.3f} % (N2/O2/Ar mix, 230-1000 K)")
    table["air"] = entry
    print("_GAS_FITS = {")
    for g, e in table.items():
        print(f"    {g!r}: {{")
        for key in ("MW", "mu", "k", "cp"):
            print(f"        {key!r}: {e[key]!r},")
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
