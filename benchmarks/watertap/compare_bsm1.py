"""Compare difflow_water's activated sludge plant with two references.

1. WaterTAP 1.8.0's BSM1 example (``bsm1_reference.json``, from ``bsm1.py``):
   ASM1 reactors with a component-split clarifier, run here with the
   ``"split"`` settler and WaterTAP's recycle fractions converted to flows.
2. The BSM1 report's open-loop steady state (``benchmarks/bsm1/report_tables.json``,
   Tables 6-7 of Alex et al. 2008), with the Takacs settler; compared at the
   report's printed precision.

Run in the difflow environment::

    python benchmarks/watertap/compare_bsm1.py      # writes bsm1_comparison.md

A hand-run cross-check, not a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow_water.wastewater import (  # noqa: E402
    STATES,
    ActivatedSludgeParams,
    activated_sludge_plant,
    bsm1_influent,
    state_to_stream,
    suspended_solids,
)

HERE = Path(__file__).parent


def watertap_case(ref) -> list[str]:
    i = ref["inputs"]
    Q0 = i["influent"]["Q"]
    fa, sw, sp, fr = (i["internal_recycle_fraction"], i["effluent_split_water"],
                      i["effluent_split_particulate"], i["recycle_fraction"])
    # Q1 = Q0 + Qa + Qr, Qa = fa Q1, Qf = (1 - fa) Q1, Qu = (1 - sw) Qf, Qr = fr Qu
    Q1 = Q0 / (1.0 - fa - fr * (1.0 - sw) * (1.0 - fa))
    Qf = (1.0 - fa) * Q1
    Qu = (1.0 - sw) * Qf
    p = ActivatedSludgeParams(KLa=tuple(i["KLa_per_day"]) + (0.0,),
                              DO_setpoint=(None, None, None, None, i["S_O_R5"]),
                              Q_a=fa * Q1, Q_r=fr * Qu, Q_w=(1.0 - fr) * Qu,
                              settler="split", split_particulate=sp)
    infl = state_to_stream(Q0, jnp.array([i["influent"][s] for s in STATES]))
    r = activated_sludge_plant(infl, p)
    assert bool(r["converged"])
    lines = ["| stream | state | WaterTAP | difflow | rel. diff |", "|---|---|---|---|---|"]
    worst = 0.0
    rows = [(f"R{k + 1}", r["reactors"][k]) for k in range(5)] + [("effluent", r["Z_effluent"]),
                                                                   ("wastage", r["Z_underflow"])]
    for name, Z in rows:
        for j, s in enumerate(STATES):
            w = ref["outputs"][name][s]
            d = float(Z[j])
            rel = (d - w) / max(abs(w), 1e-12)
            if abs(w) > 1e-9:
                worst = max(worst, abs(rel))
            if name in ("R5", "effluent", "wastage"):
                lines.append(f"| {name} | {s} | {w:.6g} | {d:.6g} | {rel:+.1e} |")
    print(f"WaterTAP BSM1 example: largest relative difference {worst:.1e}")
    return [f"Largest relative difference over all 7 streams x 13 states: **{worst:.1e}**", ""] + lines


def report_case(tab) -> list[str]:
    r = activated_sludge_plant(bsm1_influent())
    assert bool(r["converged"])
    lines = ["| reactor | state | BSM1 report | difflow | rel. diff |", "|---|---|---|---|---|"]
    worst = 0.0
    for s, vals in tab["reactors"].items():
        for k, w in enumerate(vals):
            d = float(suspended_solids(r["reactors"][k])) if s == "TSS" else \
                float(r["reactors"][k][STATES.index(s)])
            rel = (d - w) / abs(w)
            worst = max(worst, abs(rel))
            if k == 4:
                lines.append(f"| 5 | {s} | {w:.4g} | {d:.4g} | {rel:+.1e} |")
    lines += ["", "| settler layer (bottom = 1) | BSM1 report | difflow | rel. diff |",
              "|---|---|---|---|"]
    for m, w in enumerate(tab["settler_solids_bottom_first"]):
        d = float(r["settler_solids"][m])
        rel = (d - w) / w
        worst = max(worst, abs(rel))
        lines.append(f"| {m + 1} | {w:.4g} | {d:.4g} | {rel:+.1e} |")
    print(f"BSM1 report Tables 6-7: largest relative difference {worst:.1e}")
    return [f"Largest relative difference over Tables 6-7 (70 reactor values, 10 layers): "
            f"**{worst:.1e}** (the tables print 3-4 digits); sludge age "
            f"{float(r['sludge_age']):.2f} d", ""] + lines


def main() -> None:
    ref = json.loads((HERE / "bsm1_reference.json").read_text())
    tab = json.loads((HERE.parent / "bsm1" / "report_tables.json").read_text())
    v = ref["versions"]
    out = ["# Activated sludge (ASM1, BSM1 layout): difflow_water vs references", "",
           f"## WaterTAP {v['watertap']} BSM1 example (split clarifier)", ""]
    out += watertap_case(ref)
    out += ["", "## BSM1 report open-loop steady state (Takacs settler)", ""]
    out += report_case(tab)
    (HERE / "bsm1_comparison.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
