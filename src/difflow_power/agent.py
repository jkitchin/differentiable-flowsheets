"""Agent support for difflow_power (the ``difflow.agent`` entry point).

An electrical network is not a unit flowsheet: the model is a
:class:`~difflow_power.PowerNetwork` and its solves are power flow and
optimal power flow, so the core build-and-solve tools do not reach it.
These tools do, on the benchmark cases or a MATPOWER case file.
"""

from __future__ import annotations

import json

from difflow.agent.plugins import AgentSupport, PluginTool
from difflow.diagnostics import Symptom


def _network(case: str, matpower_json: str | None):
    from difflow_power.cases import CASES, from_matpower, load_case

    if matpower_json:
        with open(matpower_json) as f:
            return from_matpower(json.load(f))
    if case not in CASES:
        raise ValueError(f"no case {case!r} (cases: {', '.join(CASES)})")
    return load_case(case)


def power_flow(workbench, case: str = "case9", matpower_json: str | None = None) -> dict:
    """Solve an AC power flow (Newton, flat start) and report voltages,
    angles, generation and losses.

    Args:
        case: A benchmark case: case3, case5 (PJM), case9 (WSCC), case14
            (IEEE) or radial_feeder.
        matpower_json: Path to a MATPOWER case as JSON (the mpc struct's
            fields: baseMVA, bus, gen, branch, gencost); used instead of
            case when given.
    """
    from difflow_power import solve_power_flow

    result = solve_power_flow(_network(case, matpower_json))
    return {
        "ok": True, "case": matpower_json or case,
        "converged": bool(result.converged), "newton_steps": int(result.num_steps),
        "max_mismatch_mw": float(result.max_mismatch_mw),
        "losses_mw": float(result.losses_mw),
        "vm_pu": result.vm, "va_deg": result.va_degrees, "pg_mw": result.pg_mw,
        "note": "read max_mismatch_mw before believing anything else here",
    }


def opf(workbench, case: str = "case9", kind: str = "ac",
        matpower_json: str | None = None) -> dict:
    """Solve an optimal power flow and report cost, dispatch and locational
    marginal prices (LMPs).

    The AC problem is solved by difflow's own interior-point method in JAX;
    its LMPs come from the KKT multipliers and are checked against jax.grad
    of the optimal cost (price_check, $/MWh, should be at solver noise).

    Args:
        case: A benchmark case (see power_flow).
        kind: "ac" (nonconvex NLP) or "dc" (linearized, a convex QP).
        matpower_json: A MATPOWER case as JSON, used instead of case.
    """
    from difflow_power import solve_acopf, solve_dcopf

    network = _network(case, matpower_json)
    if kind == "ac":
        result = solve_acopf(network)
    elif kind == "dc":
        result = solve_dcopf(network)
    else:
        return {"ok": False, "error": 'kind is "ac" or "dc"'}
    out = {
        "ok": True, "case": matpower_json or case, "kind": kind,
        "converged": bool(result.converged), "cost": float(result.cost),
        "pg_mw": result.pg_mw, "lmp_mw": result.lmp_mw,
    }
    if kind == "ac" and result.converged:
        gaps = result.check_prices()
        out["price_check"] = max(gaps.values()) if gaps else 0.0
    return out


SUMMARY = """\
Electrical grids: a PowerNetwork of buses, branches (one model for lines,
transformers and phase shifters), generators and loads. Its equations are one
JAX-traceable residual set (difflow_power.residuals.power_flow_residuals) that
power flow, OPF, state estimation and verification all consume.

Use power_flow and power_opf for the benchmark cases or a MATPOWER file.
Anything else (sensitivities, contingencies, a custom network) goes through
run_python with difflow_power: solve_power_flow, solve_acopf, solve_dcopf,
ptdf, lodf, contingency_flows, loss_sensitivity, branch_flow_sensitivity,
demand_sensitivity, voltage_stability_margin, estimate_state. Radial
distribution feeders are real difflow Flowsheets (build_ladder_flowsheet) and
work with the core tools."""

SOLVER_NOTES = """\
Power flow is Newton through optimistix from a flat start, with implicit
gradients; the bus-type specification is written as equations, not by
eliminating variables. Judge a solve by max_mismatch_mw, not only converged.
AC-OPF uses difflow_power.ipm, a primal-dual interior-point NLP solver written
in JAX (no IPOPT, which would end differentiability); DC-OPF is a QP on the
same solver, so AC and DC prices are comparable. Thermal limits are posed on
|S|^2. The core recycle tools (converge, set_solver_options) do not apply to a
PowerNetwork; they do apply to a radial feeder flowsheet."""

SYMPTOMS = (
    Symptom(
        "Power flow did not converge",
        ("max_mismatch", "power flow did not converge", "PowerFlowResult"),
        "A Newton power flow that fails from a flat start usually means the "
        "loading is beyond the network's voltage stability limit or a bus "
        "type is wrong (no slack bus, or a PV bus with no generator). Check "
        "voltage_stability_margin, reduce demand, or warm start from a "
        "lighter-load solution (x0).",
    ),
    Symptom(
        "OPF interior point did not converge",
        ("IPMResult", "interior point", "ipm"),
        "The interior-point OPF failing is usually an infeasible problem: "
        "ratings, voltage or generator limits that no dispatch meets. Solve "
        "with enforce_ratings=False to see whether the ratings are the cause, "
        "then the DC-OPF as a convex check of feasibility.",
    ),
)


def support() -> AgentSupport:
    return AgentSupport(
        plugin="power", summary=SUMMARY, solver_notes=SOLVER_NOTES,
        symptoms=SYMPTOMS,
        tools={"flow": PluginTool(power_flow, "read"),
               "opf": PluginTool(opf, "read")},
    )
