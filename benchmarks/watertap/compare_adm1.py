"""Compare difflow_water's anaerobic digester (ADM1) with two references.

1. Rosen & Jeppsson's steady-state verification case
   (``benchmarks/adm1/report_steady_state.json``): reproduced with the
   uncompensated gas flow ``k_p (P_gas - P_atm)`` (see the docs).
2. WaterTAP 1.8.0's ADM1 example (``adm1_reference.json`` from ``adm1.py``),
   with WaterTAP's choices: compensated gas flow with ``P_atm = 101325 Pa``
   and Batstone's pH switch (not the BSM2 Hill function). WaterTAP also
   reduces the liquid outflow by evaporated water (1.7e-4 of the flow
   here), which difflow does not; that shows as a uniform offset in the
   conserved states.

Run in the difflow environment::

    python benchmarks/watertap/compare_adm1.py      # writes adm1_comparison.md

A hand-run cross-check, not a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow_water.wastewater.adm1 import (  # noqa: E402
    STATES,
    ADM1Params,
    adm1_test_influent,
    anaerobic_digester,
    state_to_stream,
)

HERE = Path(__file__).parent


def table(pairs) -> tuple[list[str], float]:
    lines = ["| quantity | reference | difflow | rel. diff |", "|---|---|---|---|"]
    worst = 0.0
    for k, w, d in pairs:
        rel = (d - w) / abs(w)
        worst = max(worst, abs(rel))
        lines.append(f"| {k} | {w:.10g} | {d:.10g} | {rel:+.1e} |")
    return lines, worst


def report_case() -> list[str]:
    ref = json.loads((HERE.parent / "adm1" / "report_steady_state.json").read_text())
    r = anaerobic_digester(adm1_test_influent(), ADM1Params(gas_flow="uncompensated"))
    assert bool(r["converged"])
    pairs = [(s, v, float(r["S"][STATES.index(s)])) for s, v in ref["states"].items()]
    o = ref["other"]
    pairs += [("pH", o["pH"], float(r["pH"])), ("S_hco3", o["S_hco3"], float(r["ions"]["hco3"])),
              ("S_nh3", o["S_nh3"], float(r["ions"]["nh3"])), ("S_ac-", o["S_ac_ion"], float(r["ions"]["ac"])),
              ("S_gas_h2", o["S_gas_h2"], float(r["G"][0])), ("S_gas_ch4", o["S_gas_ch4"], float(r["G"][1])),
              ("S_gas_co2", o["S_gas_co2"], float(r["G"][2])), ("p_gas_h2", o["p_gas_h2"], float(r["p_gas_h2"]))]
    lines, worst = table(pairs)
    print(f"Rosen & Jeppsson Section 7: largest relative difference {worst:.1e}")
    rc = anaerobic_digester(adm1_test_influent())
    return [f"Largest relative difference over the 24 states, pH, ions and gas: **{worst:.1e}** "
            f"(uncompensated gas flow). With the compensated gas flow (the default) the same "
            f"case gives pH {float(rc['pH']):.6f} and q_gas {float(rc['q_gas']):.1f} m^3/d "
            f"(uncompensated: {float(r['q_gas']):.1f}).", ""] + lines


def watertap_case() -> list[str]:
    ref = json.loads((HERE / "adm1_reference.json").read_text())
    i = ref["inputs"]["influent"]
    infl = state_to_stream(i["Q"], jnp.array([i[s] for s in STATES]), T=i["T"])
    p = ADM1Params(V_liq=ref["inputs"]["V_liq"], V_gas=ref["inputs"]["V_gas"], T_op=i["T"],
                   P_atm=1.01325, pH_inhibition="switch")
    r = anaerobic_digester(infl, p)
    assert bool(r["converged"])
    o = ref["outputs"]
    pairs = [(s, o["liquid"][s], float(r["S"][STATES.index(s)])) for s in STATES]
    pairs += [("liquid outflow (m^3/d)", o["liquid"]["Q"], float(r["q_out"])),
              ("pH", o["pH"], float(r["pH"])), ("q_gas (m^3/d)", o["q_gas"], float(r["q_gas"])),
              ("P_gas (bar)", o["P_gas_bar"], float(r["P_gas"]))]
    lines, worst = table(pairs)
    print(f"WaterTAP ADM1 example: largest relative difference {worst:.1e}")
    return [f"Largest relative difference over the 24 states, pH, gas flow and pressure: "
            f"**{worst:.1e}**", ""] + lines


def main() -> None:
    v = json.loads((HERE / "adm1_reference.json").read_text())["versions"]
    out = ["# ADM1: difflow_water vs references", "",
           "## Rosen & Jeppsson (2006), Section 7", ""] + report_case()
    out += ["", f"## WaterTAP {v['watertap']} ADM1 example", ""] + watertap_case()
    (HERE / "adm1_comparison.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
