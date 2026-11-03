"""EPANET 2.2 extended-period reference (via WNTR) for difflow_water's EPS.

    python benchmarks/wntr/export_eps.py   # writes eps/<name>.json

Runs EPANET 2.2 on the INP files in ``inp/`` (EPANET's example networks,
U.S. EPA, public domain) for their full duration and records, at every
report time, junction heads, tank levels, link flows and link statuses.
``compare_eps.py`` runs difflow_water's reader and simulator on the same
files.
"""

import json
import tempfile
from importlib import metadata
from pathlib import Path

import wntr

HERE = Path(__file__).parent


def main():
    (HERE / "eps").mkdir(exist_ok=True)
    for inp in sorted((HERE / "inp").glob("*.inp")):
        wn = wntr.network.WaterNetworkModel(str(inp))
        with tempfile.TemporaryDirectory() as tmp:
            r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(tmp) / "epanet"))
        tanks = wn.tank_name_list
        rec = {
            "generated_by": "benchmarks/wntr/export_eps.py", "wntr": metadata.version("wntr"),
            "epanet_version": "2.2 (bundled with WNTR)", "inp": inp.name,
            "time": [float(t) for t in r.node["head"].index],
            "head": {n: [float(v) for v in r.node["head"][n]] for n in wn.junction_name_list},
            "tank_level": {n: [float(v) - wn.get_node(n).elevation for v in r.node["head"][n]]
                           for n in tanks},
            "flow": {n: [float(v) for v in r.link["flowrate"][n]] for n in wn.link_name_list},
            "status": {n: [float(v) for v in r.link["status"][n]] for n in wn.link_name_list},
        }
        (HERE / "eps" / f"{inp.stem}.json").write_text(json.dumps(rec) + "\n")
        print(inp.stem, len(rec["time"]), "report times")


if __name__ == "__main__":
    main()
