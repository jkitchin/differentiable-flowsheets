"""Export EPANET example networks and their EPANET 2.2 solution (via WNTR) to JSON.

Run where WNTR is installed (NOT a difflow dependency)::

    pip install wntr==1.5.0 six
    python benchmarks/wntr/export_networks.py   # writes networks/*.json

For each network: WNTR reads the INP file (converting to SI), EPANET 2.2
(WNTR's EpanetSimulator) solves a single period (duration 0, so demands,
reservoir heads and tank levels are their t = 0 values and controls are
applied at t = 0), and the JSON holds the network in difflow_water's
``WaterNetwork.from_dict`` form with each link's t = 0 status, plus EPANET's
heads, pressures, demands and flows. ``compare_networks.py`` solves the same
networks in difflow_water.
"""

import json
from importlib import metadata
from pathlib import Path

import wntr

HERE = Path(__file__).parent
LIB = Path(wntr.__file__).parent / "library" / "networks"
CASES = {"Net1": LIB / "Net1.inp", "Net3": LIB / "Net3.inp", "ky10": LIB / "ky10.inp",
         "Net1_DW": LIB / "Net1.inp"}


def export(name, inp):
    wn = wntr.network.WaterNetworkModel(str(inp))
    if name.endswith("_DW"):
        # Darcy-Weisbach variant: roughness 0.26 mm for every pipe (ILLUSTRATIVE);
        # WNTR holds D-W roughness in m internally (wntr.epanet.util)
        wn.options.hydraulic.headloss = "D-W"
        for _, p in wn.pipes():
            p.roughness = 0.26e-3
    wn.options.time.duration = 0
    res = wntr.sim.EpanetSimulator(wn).run_sim()
    status = res.link["status"].iloc[0]
    d = {"headloss": wn.options.hydraulic.headloss, "viscosity": 1.0e-6,
         "junctions": {}, "reservoirs": {}, "tanks": {}, "pipes": {}, "pumps": {}, "valves": {}}
    dem = res.node["demand"].iloc[0]
    for n, j in wn.junctions():
        base = j.demand_timeseries_list.at(0, multiplier=wn.options.hydraulic.demand_multiplier)
        d["junctions"][n] = {"elevation": j.elevation, "demand": float(base),
                             "emitter": float(j.emitter_coefficient or 0.0),
                             "emitter_exponent": float(wn.options.hydraulic.emitter_exponent)}
    for n, r in wn.reservoirs():
        d["reservoirs"][n] = {"head": float(r.head_timeseries.at(0))}
    for n, t in wn.tanks():
        d["tanks"][n] = {"elevation": t.elevation, "level": t.init_level,
                         "min_level": t.min_level, "max_level": t.max_level,
                         "diameter": t.diameter}
    for n, p in wn.pipes():
        rough = p.roughness
        d["pipes"][n] = {"start": p.start_node_name, "end": p.end_node_name, "length": p.length,
                         "diameter": p.diameter, "roughness": rough, "minor_loss": p.minor_loss,
                         "check_valve": bool(p.check_valve),
                         "status": "closed" if status[n] == 0 else "open"}
    for n, p in wn.pumps():
        rec = {"start": p.start_node_name, "end": p.end_node_name,
               "speed": float(getattr(p, "base_speed", 1.0)),
               "status": "closed" if status[n] == 0 else "open"}
        if p.pump_type == "POWER":
            rec["power"] = float(p.power)
        else:
            A, B, C = p.get_head_curve_coefficients()
            rec.update(A=float(A), B=float(B), C=float(C))
            rec["_curve"] = [list(pt) for pt in p.get_pump_curve().points]
        d["pumps"][n] = rec
    for n, v in wn.valves():
        kind = v.valve_type
        if kind not in ("PRV", "PSV", "FCV", "TCV"):
            raise NotImplementedError(f"{name}: valve {n} is a {kind}")
        st = status[n]
        d["valves"][n] = {"start": v.start_node_name, "end": v.end_node_name, "kind": kind,
                          "diameter": v.diameter, "setting": float(v.initial_setting),
                          "minor_loss": v.minor_loss,
                          "status": "closed" if st == 0 else "active",
                          "_epanet_status": int(st)}
    sol = {"head": res.node["head"].iloc[0].to_dict(),
           "pressure": res.node["pressure"].iloc[0].to_dict(),
           "demand": dem.to_dict(),
           "flow": res.link["flowrate"].iloc[0].to_dict()}
    return {"network": d, "epanet": sol, "inp": inp.name}


def main():
    out = {"generated_by": "benchmarks/wntr/export_networks.py",
           "wntr": metadata.version("wntr"), "epanet_version": "2.2 (bundled with WNTR)"}
    (HERE / "networks").mkdir(exist_ok=True)
    for name, inp in CASES.items():
        rec = export(name, inp)
        rec.update(out)
        (HERE / "networks" / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
        n = rec["network"]
        print(name, len(n["junctions"]), "junctions", len(n["pipes"]), "pipes",
              len(n["pumps"]), "pumps", len(n["valves"]), "valves")


if __name__ == "__main__":
    main()
