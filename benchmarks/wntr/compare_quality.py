"""Compare difflow_water's water quality with EPANET 2.2 (eps/*.json quality).

    python benchmarks/wntr/compare_quality.py   # writes quality_comparison.md

Net1 chlorine (bulk -0.5/day and wall -1 ft/day, first order), Net1 water
age, and Net3's trace of the Lake source, over their full durations, at
every report time. EPANET's values come through WNTR (chlorine in kg/m^3,
age in s, trace as WNTR reports it), converted to mg/L, hours and percent.
Hand-run, not a test.
"""

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from difflow_water.network import read_inp, simulate  # noqa: E402
from difflow_water.network.quality import water_quality  # noqa: E402

HERE = Path(__file__).parent
CASES = [("Net1", "Net1.json", None, 1e3, "mg/L"), ("Net1", "Net1_age.json", "AGE", 1.0, "h"),
         ("Net3", "Net3.json", None, None, "%")]


def main():
    lines = ["| network | quality | unit | report times | max abs diff | at (node, hour) "
             "| median abs diff | range of EPANET values |", "|---" * 8 + "|"]
    for name, ref_file, mode, scale, unit in CASES:
        ref = json.loads((HERE / "eps" / ref_file).read_text())
        net, sch = read_inp(HERE / "inp" / f"{name}.inp")
        if mode:
            sch.quality_options["mode"] = mode
        eps = simulate(net, sch)
        wq = water_quality(net, sch, eps)
        if scale is None:   # trace: WNTR may report a fraction or a percent
            mx = max(max(v) for v in ref["quality"].values())
            scale = 100.0 if mx <= 1.0 + 1e-9 else 1.0
        rt = ref["time"]
        diffs, where = [], []
        for k, t in enumerate(wq["time"]):
            if not any(abs(t - x) < 1 for x in rt):
                continue
            r = min(range(len(rt)), key=lambda i: abs(rt[i] - t))
            for j, nd in enumerate(wq["nodes"]):
                if nd in ref["quality"]:
                    diffs.append(abs(wq["quality"][k, j] - ref["quality"][nd][r] * scale))
                    where.append((nd, t / 3600))
        diffs = np.array(diffs)
        i = int(np.argmax(diffs))
        vals = np.array([v * scale for vv in ref["quality"].values() for v in vv])
        lines.append(f"| {name} | {wq['mode'].lower()} | {unit} | {len(wq['time'])} | {diffs.max():.3g} | "
                     f"{where[i][0]}, {where[i][1]:.0f} | {np.median(diffs):.2g} | "
                     f"{vals.min():.3g} - {vals.max():.3g} |")
    text = "\n".join(lines) + "\n"
    (HERE / "quality_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
