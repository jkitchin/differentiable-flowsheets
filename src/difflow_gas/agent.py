"""Agent support for difflow_gas (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport
from difflow.diagnostics import Symptom

SUMMARY = """\
Gas transmission networks: a GasNetwork of pipes, compressor stations, valves,
control valves, resistors and short pipes, with signed flows. decompose(net,
root) computes the spanning tree, the tear set and the balance schedule from
the topology; build_network_flowsheet turns a network into a
GasNetworkFlowsheet, a real difflow Flowsheet that the core tools can solve,
diagnose and differentiate once it is in a session (build it with
run_python or a .py file opened with open_file). The equation set is one
JAX-traceable definition (difflow_gas.residuals.network_residuals); verify
checks a solution against it (residual_report, bounds_report,
compressor_report). Reconciliation: reconcile_network, monitor_network."""

SOLVER_NOTES = """\
Flows are signed, so solve with clip_negative_flows=False: clipping a tear
that is negative in the right answer pins it at zero and the loop never
closes. Damp the tear map (GasNetworkFlowsheet.solve_differentiable uses
alpha about 0.3). Pose optimization pressure constraints in squared pressure,
which is what the Weymouth equation is linear in. Check a converged answer
with difflow_gas.verify.residual_report before trusting it."""

SYMPTOMS = (
    Symptom(
        "Negative flows in a gas network",
        ("negative flows", "clip_negative_flows", "clipping"),
        "In a gas network a negative flow is a direction, not an error: the "
        "flow runs against the arc's declared orientation. Solve with "
        "clip_negative_flows=False; with clipping on, a tear that should be "
        "negative is held at zero and the loop stalls.",
        ("signed_tears", "damped"),
    ),
)


def support() -> AgentSupport:
    return AgentSupport(plugin="gas", summary=SUMMARY, solver_notes=SOLVER_NOTES,
                        symptoms=SYMPTOMS)
