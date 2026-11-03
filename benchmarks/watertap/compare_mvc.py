"""Compare difflow_water's MVC plant with the WaterTAP reference (``mvc_reference.json``).

Run in the difflow environment (no WaterTAP needed)::

    python benchmarks/watertap/compare_mvc.py

Two checks:

1. Simulation: every recorded case (including WaterTAP's optimum, used as a
   simulation point) run with WaterTAP's inputs; WaterTAP's value, difflow's
   and the relative difference for each output.
2. Optimization: minimum LCOW over brine temperature, pressure ratio and the
   preheater areas with WaterTAP's bounds (vapor <= 75 C, compressed vapor
   <= 450 K, areas >= 10 m^2, condensate not above saturation, positive
   evaporator approach), SLSQP with ``jax`` gradients, started from the
   ``sim_lower_PR`` design; compared with WaterTAP's IPOPT optimum.

Writes ``mvc_comparison.md``. A hand-run cross-check, not a test.

WaterTAP's example fixes the preheaters' IDAES ``deltaP = +7 kPa`` (a
pressure gain), so difflow is run with ``hx_dP = -7e3`` to match it.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from difflow_water.streams import from_watertap_state  # noqa: E402
from difflow_water.thermal import MVCParams, mvc_performance  # noqa: E402

HERE = Path(__file__).parent
KEYS = ["evaporator_area", "evaporator_heat", "evaporator_lmtd", "T_evaporator_feed",
        "P_brine", "T_compressed_vapor", "P_compressed_vapor", "compressor_work",
        "T_condensate", "T_distillate_out", "T_brine_out", "pump_feed_work",
        "pump_brine_work", "pump_distillate_work", "product_flow_vol", "SEC", "LCOW",
        "total_capital", "total_operating"]


def _feed(i):
    return from_watertap_state(i["feed_H2O_kg_s"], i["feed_TDS_kg_s"], i["T_feed"], i["P_feed"])


def _params(i, **kw):
    base = dict(recovery=i["recovery"], T_brine=i["T_brine"],
                pressure_ratio=i["pressure_ratio"], A_hx_distillate=i["A_hx_distillate"],
                A_hx_brine=i["A_hx_brine"], compressor_efficiency=i["compressor_efficiency"],
                U_evaporator=i["U_evaporator"], U_hx=i["U_hx"], hx_dP=-7e3,
                mode="fixed_pressure_ratio")
    base.update(kw)
    return MVCParams(**base)


def simulate(ref) -> list[str]:
    lines = []
    for name, c in ref["cases"].items():
        o = mvc_performance(_feed(c["inputs"]), _params(c["inputs"]))
        assert bool(o["converged"]), name
        lines += [f"\n**{name}** (T_brine = {c['inputs']['T_brine']:.2f} K, pressure ratio "
                  f"{c['inputs']['pressure_ratio']:.4f}, preheaters "
                  f"{c['inputs']['A_hx_distillate']:.1f} / {c['inputs']['A_hx_brine']:.1f} m^2)\n",
                  "| output | WaterTAP | difflow | rel. diff |", "|---|---|---|---|"]
        worst = 0.0
        for k in KEYS:
            w, d = c["outputs"][k], float(o[k])
            rel = (d - w) / abs(w)
            worst = max(worst, abs(rel))
            lines.append(f"| {k} | {w:.6g} | {d:.6g} | {rel:+.1e} |")
        for k, w in c["outputs"]["capital"].items():
            d = float(o["capital"][k])
            rel = (d - w) / abs(w)
            worst = max(worst, abs(rel))
            lines.append(f"| capital {k} | {w:.6g} | {d:.6g} | {rel:+.1e} |")
        print(f"{name}: largest relative difference {worst:.1e}")
    return lines


def optimize(ref) -> list[str]:
    opt = ref["cases"]["optimum"]
    i = opt["inputs"]
    start = ref["cases"]["sim_lower_PR"]["inputs"]
    feed = _feed(i)

    def run(x):
        return mvc_performance(feed, _params(i, T_brine=x[0], pressure_ratio=x[1],
                                             A_hx_distillate=x[2], A_hx_brine=x[3]))

    f = jax.jit(lambda x: run(x)["LCOW"])
    g = jax.jit(jax.grad(lambda x: run(x)["LCOW"]))

    def cons(x):
        o = run(x)
        return jnp.stack([o["condensate_subcooling"], o["evaporator_dT_out"],
                          450.0 - o["T_compressed_vapor"]])

    c_fn, c_jac = jax.jit(cons), jax.jit(jax.jacfwd(cons))
    x0 = np.array([start["T_brine"], start["pressure_ratio"], start["A_hx_distillate"],
                   start["A_hx_brine"]])
    r = minimize(lambda x: float(f(x)), x0, jac=lambda x: np.asarray(g(x)), method="SLSQP",
                 bounds=[(273.15, 348.15), (1.0, 5.0), (10.0, None), (10.0, None)],
                 constraints=[{"type": "ineq", "fun": lambda x: np.asarray(c_fn(x)),
                               "jac": lambda x: np.asarray(c_jac(x))}],
                 options={"ftol": 1e-12, "maxiter": 200})
    names = ["T_brine (K)", "pressure_ratio", "A_hx_distillate (m^2)", "A_hx_brine (m^2)"]
    wt = [i["T_brine"], i["pressure_ratio"], i["A_hx_distillate"], i["A_hx_brine"]]
    lines = ["\n| decision | WaterTAP (IPOPT) | difflow (SLSQP + jax.grad) | rel. diff |",
             "|---|---|---|---|"]
    for n, w, d in zip(names, wt, r.x):
        lines.append(f"| {n} | {w:.7g} | {d:.7g} | {(d - w) / w:+.1e} |")
    w = opt["outputs"]["LCOW"]
    lines.append(f"| LCOW (USD/m^3) | {w:.7g} | {r.fun:.7g} | {(r.fun - w) / w:+.1e} |")
    lines.append(f"\nSLSQP: {r.message} ({r.nit} iterations) from the sim_lower_PR design.")
    print(f"optimum: LCOW {r.fun:.7g} vs WaterTAP {w:.7g}; {r.message}")
    return lines


def main() -> None:
    ref = json.loads((HERE / "mvc_reference.json").read_text())
    v = ref["versions"]
    head = [f"# MVC: difflow_water vs WaterTAP {v['watertap']} (IDAES {v['idaes-pse']}, "
            f"Pyomo {v['pyomo']})", "", "## Simulation"]
    out = head + simulate(ref) + ["", "## LCOW optimum"] + optimize(ref)
    (HERE / "mvc_comparison.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
