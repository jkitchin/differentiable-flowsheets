"""Compare difflow_water's network solve with EPANET 2.2 (networks/*.json).

    python benchmarks/wntr/compare_networks.py   # writes networks_comparison.md

Hand-run cross-check, not a test. For each network: largest absolute head
difference over the junctions (m), largest absolute flow difference over the
links (L/s) and the largest relative flow difference over links carrying
more than 1 L/s, for both valve formulations.
"""

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from difflow_water.network import NetworkModel, WaterNetwork  # noqa: E402

HERE = Path(__file__).parent


def compare(path, formulation):
    rec = json.loads(path.read_text())
    net = WaterNetwork.from_dict(rec["network"])
    model = NetworkModel(net, valve_formulation=formulation)
    t = time.time()
    res = model.solve()
    jax.block_until_ready(res["head"])
    dt = time.time() - t
    ep = rec["epanet"]
    H = np.asarray(res["head"])
    Q = np.asarray(res["flow"])
    dH = np.array([H[i] - ep["head"][n] for i, n in enumerate(res["junctions"])])
    qe = np.array([ep["flow"][n] for n in res["links"]])
    dQ = Q - qe
    big = np.abs(qe) > 1e-3
    rel = np.abs(dQ[big] / qe[big]).max() if big.any() else 0.0
    i = int(np.argmax(np.abs(dH)))
    k = int(np.argmax(np.abs(dQ)))
    return {"network": path.stem, "formulation": formulation,
            "junctions": len(res["junctions"]), "links": len(res["links"]),
            "max_dH_m": float(np.abs(dH).max()), "at_node": res["junctions"][i],
            "max_dQ_Ls": float(np.abs(dQ).max() * 1e3), "at_link": res["links"][k],
            "max_rel_dQ": float(rel), "residual": float(res["residual"]), "seconds": dt}


def main(names=None):
    paths = sorted((HERE / "networks").glob("*.json"))
    if names:
        paths = [p for p in paths if p.stem in names]
    rows = [compare(p, f) for p in paths for f in ("complementarity", "smooth")]
    lines = ["| network | formulation | junctions | links | max abs dH (m) | at | max abs dQ (L/s) | at "
             "| max rel dQ (Q > 1 L/s) | residual | solve s |", "|---" * 11 + "|"]
    for r in rows:
        lines.append(f"| {r['network']} | {r['formulation']} | {r['junctions']} | {r['links']} | "
                     f"{r['max_dH_m']:.2e} | {r['at_node']} | {r['max_dQ_Ls']:.2e} | {r['at_link']} | "
                     f"{r['max_rel_dQ']:.2e} | {r['residual']:.1e} | {r['seconds']:.1f} |")
    text = "\n".join(lines) + "\n"
    if not names:
        (HERE / "networks_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
