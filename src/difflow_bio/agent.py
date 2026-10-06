"""Agent support for difflow_bio (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport

SUMMARY = """\
Biomanufacturing: ContinuousBioreactor and FedBatchBioreactor upstream;
Centrifuge and DiscStackCentrifuge; Ultrafiltration, Diafiltration and TFF;
ProteinAChromatography, IonExchangeChromatography and
SizeExclusionChromatography downstream. Units fit the core flowsheet tools.
The bioreactors need a growth model (kinetic_fn, kinetic_params), which is a
function: define it in the code context (describe_operation returns starter
code, a Monod stub) before adding the unit. Process economics live in
difflow_bio.economics (list_api)."""

SOLVER_NOTES = """\
Fed-batch units integrate in time (diffrax) inside the flowsheet call; their
cost is in the integration, not in a recycle, so a slow solve is usually the
ODE (stiff kinetics, a long horizon), not the tear."""


def support() -> AgentSupport:
    return AgentSupport(plugin="bio", summary=SUMMARY, solver_notes=SOLVER_NOTES)
