"""Compare difflow_water's extended-period simulation with EPANET 2.2 (eps/*.json).

    python benchmarks/wntr/compare_eps.py   # writes eps_comparison.md

Reads each INP file in ``inp/`` with difflow_water's ``read_inp``, simulates
it with ``simulate``, and compares tank levels, junction heads and link
statuses with EPANET at every report time. Hand-run, not a test.
"""

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from difflow_water.network.eps import simulate  # noqa: E402
from difflow_water.network.hydraulics import NetworkModel  # noqa: E402
from difflow_water.network.inp import read_inp  # noqa: E402

HERE = Path(__file__).parent


def compare(name):
    ref = json.loads((HERE / "eps" / f"{name}.json").read_text())
    net, sch = read_inp(HERE / "inp" / f"{name}.inp")
    t0 = time.time()
    res = simulate(net, sch, model=NetworkModel(net))
    dt = time.time() - t0
    times = list(res["time"])
    rt = ref["time"]
    common = [t for t in rt if any(abs(t - s) < 1 for s in times)]
    ri = [rt.index(t) for t in common]
    mi = [min(range(len(times)), key=lambda k: abs(times[k] - t)) for t in common]
    lev = np.asarray(res["tank_level"])
    dlev = max(abs(lev[m, j] - ref["tank_level"][tk][r]) for j, tk in enumerate(res["tanks"])
               for r, m in zip(ri, mi))
    H = np.asarray(res["head"])
    dH = np.array([[H[m, j] - ref["head"][n][r] for j, n in enumerate(res["junctions"])]
                   for r, m in zip(ri, mi)])
    st = np.asarray(res["status"])
    mism = sum(int(st[m, k] != (ref["status"][n][r] > 0)) for k, n in enumerate(res["links"])
               for r, m in zip(ri, mi))
    worst_t = common[int(np.argmax(np.abs(dH).max(axis=1)))] / 3600
    return {"network": name, "reports": len(common), "solves": len(res["steps"]),
            "max_dlevel": dlev, "max_dH": float(np.abs(dH).max()), "worst_h": worst_t,
            "median_dH": float(np.median(np.abs(dH))), "status_mismatches": mism, "seconds": dt}


def main(names=None):
    all_names = sorted(p.stem for p in (HERE / "inp").glob("*.inp"))
    names = names or all_names
    rows = [compare(n) for n in names]
    lines = ["| network | report times | solves | max tank level diff (m) | max head diff (m) "
             "| at hour | median head diff (m) | link-status mismatches | seconds |",
             "|---" * 9 + "|"]
    for r in rows:
        lines.append(f"| {r['network']} | {r['reports']} | {r['solves']} | {r['max_dlevel']:.2e} | "
                     f"{r['max_dH']:.2e} | {r['worst_h']:.1f} | {r['median_dH']:.1e} | "
                     f"{r['status_mismatches']} | {r['seconds']:.0f} |")
    text = "\n".join(lines) + "\n"
    if names == all_names:
        (HERE / "eps_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
