"""Compare difflow_water with the WaterTAP reference (``swro_px_reference.json``).

Run in the difflow environment (no WaterTAP needed)::

    python benchmarks/watertap/compare_swro_px.py

Prints, for each case, the RO stage alone and the whole plant, WaterTAP's
value, difflow's and the relative difference, and writes the same tables to
``swro_px_comparison.md`` (pasted into ``docs/unit-operations-water.md``).
This is a hand-run cross-check, not a test (the test suite does not compare
against reference data).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import difflow_water as dw

HERE = Path(__file__).parent
N_STEPS = 80

RO_KEYS = ["recovery_mass_water", "recovery_vol", "salinity_permeate",
           "salinity_concentrate", "deltaP", "flux_mass_water_avg", "Re_in", "Re_out"]
PLANT_KEYS = ["LCOW", "SEC", "work_P1", "work_P2", "px_hp_pressure", "total_capital",
              "total_operating"]


def _ro_params(c):
    return dw.ROStageParams(A_w=c["A_w"], B=c["B"], area=c["area"], width=c["width"],
                            permeate_pressure=c["permeate_pressure"],
                            channel_height=c["channel_height"],
                            spacer_porosity=c["spacer_porosity"], n_steps=N_STEPS)


def run_case(c) -> tuple[dict, dict]:
    feed_hp = dw.make_seawater_stream(flow_vol=c["flow_vol"], salinity=c["salinity"],
                                      T=c["T"], P=c["P_ro"])
    _, _, info = dw.ROStage(_ro_params(c))(feed_hp)
    ro = {k: float(info[k]) for k in RO_KEYS}
    feed = dw.make_seawater_stream(flow_vol=c["flow_vol"], salinity=c["salinity"],
                                   T=c["T"], P=c["P_feed"])
    r = dw.swro_performance(feed, _ro_params(c), c["P_ro"],
                            px=dw.PressureExchangerParams(efficiency=c["px_efficiency"]),
                            pump_efficiency=c["pump_efficiency"],
                            booster_efficiency=c["booster_efficiency"])
    plant = {
        "LCOW": float(r["LCOW"]), "SEC": float(r["SEC"]),
        "work_P1": float(r["work_P1"]), "work_P2": float(r["work_P2"]),
        "px_hp_pressure": float(r["px"]["deltaP_feed"] + c["P_feed"]),
        "total_capital": float(r["costing"]["total_capital"]),
        "total_operating": float(r["costing"]["total_operating"]),
    }
    return ro, plant


def _table(title, keys, ref, ours) -> str:
    lines = [f"**{title}**", "", "| Quantity | WaterTAP | difflow | rel. diff |",
             "|---|---|---|---|"]
    for k in keys:
        a, b = ref[k], ours[k]
        rel = (b - a) / abs(a) if a else float("nan")
        lines.append(f"| `{k}` | {a:.6g} | {b:.6g} | {rel:+.2%} |")
    return "\n".join(lines) + "\n"


def main():
    ref = json.loads((HERE / "swro_px_reference.json").read_text())
    out = [f"Reference: WaterTAP {ref['versions']['watertap']}, IDAES "
           f"{ref['versions']['idaes-pse']}, Pyomo {ref['versions']['pyomo']}; RO1D with "
           f"{ref['ro_finite_elements']} finite elements. difflow: RK4 with {N_STEPS} steps.", ""]
    for name, case in ref["cases"].items():
        ro, plant = run_case(case["inputs"])
        out.append(f"### Case `{name}`\n")
        out.append(_table("RO stage alone", RO_KEYS, case["ro_alone"], ro))
        out.append(_table("Plant (RO + pressure exchanger + pumps + costing)", PLANT_KEYS,
                          case["plant"], plant))
    text = "\n".join(out)
    (HERE / "swro_px_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
