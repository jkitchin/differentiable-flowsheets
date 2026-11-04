"""Compare difflow_water's GAC adsorber with the WaterTAP reference (``gac_reference.json``).

Run in the difflow environment::

    python benchmarks/watertap/compare_gac.py

Writes ``gac_comparison.md``. A hand-run cross-check, not a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from difflow import make_stream  # noqa: E402
from difflow_water.chemistry.seawater import MW_WATER  # noqa: E402
from difflow_water.treatment import GACAdsorber, GACParams  # noqa: E402

HERE = Path(__file__).parent
MW_SOLUTE = 0.08  # WaterTAP example's solute molar mass (kg/mol)
KEYS = ["equil_conc", "dg", "N_Bi", "min_N_St", "min_ebct", "throughput",
        "operational_time", "bed_volumes_treated", "bed_area", "bed_diameter", "bed_volume",
        "bed_mass_gac", "gac_usage_rate", "conc_ratio_avg", "mass_adsorbed"]


def main() -> None:
    ref = json.loads((HERE / "gac_reference.json").read_text())
    v = ref["versions"]
    lines = [f"# GAC: difflow_water vs WaterTAP {v['watertap']} (IDAES {v['idaes-pse']})", ""]
    worst_all = 0.0
    for name, c in ref["cases"].items():
        i, o = c["inputs"], c["outputs"]
        feed = make_stream({"H2O": i["flow_mass_H2O"] / MW_WATER,
                            "solute": i["flow_mass_solute"] / MW_SOLUTE}, T=i["T"], P=i["P"])
        p = GACParams(mw_target=MW_SOLUTE, velocity_sup=o["velocity_sup"],
                      **{k: i[k] for k in ("freund_k", "freund_ninv", "kf", "ds",
                                           "particle_dens_app", "particle_dia", "ebct",
                                           "bed_voidage", "conc_ratio_replace",
                                           "a0", "a1", "b0", "b1", "b2", "b3", "b4")})
        treated, _, info = GACAdsorber(p)(feed)
        lines += [f"**{name}**", "", "| output | WaterTAP | difflow | rel. diff |", "|---|---|---|---|"]
        worst = 0.0
        pairs = [(k, o[k], float(info[k])) for k in KEYS]
        pairs.append(("outlet solute (kg/s)", o["outlet_solute_mass"],
                      float(treated["F_solute"]) * MW_SOLUTE))
        for k, w, d in pairs:
            rel = (d - w) / abs(w)
            worst = max(worst, abs(rel))
            lines.append(f"| {k} | {w:.6g} | {d:.6g} | {rel:+.1e} |")
        lines.append("")
        worst_all = max(worst_all, worst)
        print(f"{name}: largest relative difference {worst:.1e}")
    lines.insert(2, f"Largest relative difference over all cases and outputs: {worst_all:.1e}\n")
    (HERE / "gac_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
