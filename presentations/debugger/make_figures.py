"""The deck's two figures: the example flowsheet and the what-if residual plot.

    python make_figures.py        # writes assets/flowsheet.svg and assets/whatif.png
"""

import runpy
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from difflow import debug
from difflow.debug.record import state_at, summarize
from difflow.report.diagram import flowsheet_diagram

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "demo"))
from twostates import bistable

warnings.simplefilter("ignore")


def flowsheet_svg():
    fs = runpy.run_path(str(HERE / "demo" / "plant.py"))["fs"]
    (HERE / "assets" / "flowsheet.svg").write_text(flowsheet_diagram(fs))


def whatif_png():
    runs = []
    for acc, value in [("wegstein", None), ("wegstein", 0.3), ("wegstein", 0.8), ("anderson", 0.4)]:
        kw = {"use_initialization": False, "acceleration": acc}
        base = debug.record(bistable(), kw)
        if value is None:
            runs.append(("recorded, from 0.01", base))
            continue
        k = summarize(base)["iterations"][0]["frame"]
        runs.append((f"what-if: tear = {value} ({acc})",
                     debug.record(bistable(), kw, baseline=base["frames"],
                                  edits=[{"frame": k, "stream": "tear", "field": "F_A",
                                          "value": value}])))
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    for (label, trace), colour in zip(runs, ["#52514e", "#2a78d6", "#eb6834", "#7a3fbf"]):
        res = [f["residual"] for f in trace["frames"] if f["checkpoint"] == "iteration"]
        answer = float(state_at(trace, len(trace["frames"]) - 1)["out"]["F_A"])
        ax.semilogy(range(len(res)), [max(r, 1e-17) for r in res], "o-", color=colour,
                    lw=2, ms=5, label=f"{label} → x = {answer:.2g}")
    ax.set_xlabel("tear iteration")
    ax.set_ylabel("residual  |g(x) − x|")
    ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "assets" / "whatif.png", facecolor="white")


if __name__ == "__main__":
    flowsheet_svg()
    whatif_png()
