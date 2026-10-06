"""Agent support for difflow_cc (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport

SUMMARY = """\
Carbon capture: AmineAbsorber and AmineStripper (MEA, DEA, MDEA, PZ, AMP),
MembraneSeparator and MultistageMembrane (nine membrane materials), PSA, TSA,
VSA and TVSA units (eight adsorbents), SolidSorbentDAC and LiquidSolventDAC,
LeanRichExchanger and HeatRecoverySystem, CompressionTrain and Pump. A unit
that names its solvent, membrane or adsorbent needs that name before it can
be built: set it in the code context (for example solvent = "MEA") or as a
parameter; describe_operation says which. Economics (capital, operating cost,
levelized cost of capture) and degradation models are libraries in
difflow_cc (list_api)."""

SOLVER_NOTES = """\
The absorber-stripper loop is a recycle on the lean solvent: a high-gain loop,
so read gain and error_estimate after a solve and prefer tol_basis "error"
when the loading matters."""


def support() -> AgentSupport:
    return AgentSupport(plugin="cc", summary=SUMMARY, solver_notes=SOLVER_NOTES)
