"""Compare difflow_water's Pitzer speciation with PHREEQC (seawater_brines_reference.json).

    python benchmarks/phreeqc/compare_seawater_brines.py   # writes seawater_brines_comparison.md

Feeds difflow the totals PHREEQC reports for each case (Na, K, Mg, Ca, Cl,
S(6), C(4) and alkalinity, mol/kgw) and compares pH, ionic strength, water
activity and saturation indices. Hand-run cross-check, not a test.
"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from difflow_water.chemistry import speciation as spc  # noqa: E402

HERE = Path(__file__).parent
KEYS = {"Na": "Na(mol/kgw)", "K": "K(mol/kgw)", "Mg": "Mg(mol/kgw)", "Ca": "Ca(mol/kgw)",
        "Cl": "Cl(mol/kgw)", "SO4": "S(6)(mol/kgw)", "C": "C(4)(mol/kgw)", "Alk": "Alk(eq/kgw)"}


def main():
    ref = json.loads((HERE / "seawater_brines_reference.json").read_text())
    lines = ["| T (C) | CF | pH PHREEQC | pH difflow (MacInnes) | I rel. diff | log a_w diff | "
             + " | ".join(f"SI {m} diff" for m in spc.MINERALS) + " |",
             "|---" * (6 + len(spc.MINERALS)) + "|"]
    worst = {}
    for case in ref["cases"]:
        row = case["row"]
        tot = {k: row[v] for k, v in KEYS.items()}
        r = spc.speciate(tot, T=case["T_C"] + 273.15)
        d = {"pH": float(r["pH_macinnes"]) - row["pH"],
             "I": float(r["ionic_strength"]) / row["mu"] - 1.0,
             "la_w": float(jax.numpy.log10(r["water_activity"])) - row["la_H2O"]}
        for m in spc.MINERALS:
            d[m] = float(r["SI"][m]) - row[f"si_{m}"]
        for k, v in d.items():
            worst[k] = max(worst.get(k, 0.0), abs(v))
        lines.append(f"| {case['T_C']:.0f} | {case['cf']:.1f} | {row['pH']:.3f} | "
                     f"{float(r['pH_macinnes']):.3f} | {d['I']:+.1e} | {d['la_w']:+.1e} | "
                     + " | ".join(f"{d[m]:+.4f}" for m in spc.MINERALS) + " |")
    lines.append("")
    lines.append("Largest absolute difference: " + ", ".join(f"{k} {v:.2g}" for k, v in worst.items()))
    text = "\n".join(lines) + "\n"
    (HERE / "seawater_brines_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
