"""Agent support for difflow_refinery (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport
from difflow.diagnostics import Symptom

SUMMARY = """\
Petroleum refining: an Assay (TBP curve, SG, light ends) characterized into
pseudo-components; the crude unit (CrudeDistillationUnit, an equation-oriented
MESH column solved with its furnace), VacuumColumn, preheat train, gas plant
(GasPlantColumn, WetGasCompressor, AmineTreater) and isomerization are unit
operations. Most conversion units are deliberately libraries, not palette
operations: catalytic reforming (CatalyticReformer), FCC (FCCUnit),
hydrotreating, hydrocracking, alkylation, residue, the hydrogen network,
blending and planning. Find them with list_api("difflow_refinery") and its
submodules, and use them through run_python."""

SOLVER_NOTES = """\
Expect long first calls: a column or reactor compiles for seconds to minutes
before its first solve (a hydrotreater about 70 s), after which solves take
milliseconds to seconds; pass a long timeout. Memory, not time, limits long
chains. Each unit reports its own convergence in its info dict and warns with
its own class (HydrocrackerConvergenceWarning, VacuumConvergenceWarning,
GasColumnConvergenceWarning, HydrotreaterConvergenceWarning,
FCCConvergenceWarning, IsomerizationConvergenceWarning, HydrogenLoopWarning).
A crude-unit spec set with no solution (too little overflash for a large
pumparound) returns converged=False rather than an answer: change the specs,
not the solver. jit a stage that holds a Python-level recycle."""

SYMPTOMS = (
    Symptom(
        "A refinery unit's own solve did not converge",
        ("HydrocrackerConvergenceWarning", "VacuumConvergenceWarning",
         "GasColumnConvergenceWarning", "HydrotreaterConvergenceWarning",
         "FCCConvergenceWarning", "IsomerizationConvergenceWarning"),
        "The unit's internal solve (column MESH, reactor, recycle) stopped "
        "short; its outlet is not a solution of the unit. Read its info dict "
        "(get_unit_info) for the residual. Usually the specs are infeasible "
        "together (a product rate or overflash the column cannot meet) or the "
        "feed is outside the correlation's range; adjust the specs before the "
        "solver.",
    ),
    Symptom(
        "The hydrogen loop did not close",
        ("HydrogenLoopWarning",),
        "The hydrogen network's recycle or makeup balance did not close: check "
        "that makeup hydrogen can cover consumption at the purity required.",
    ),
)


def support() -> AgentSupport:
    return AgentSupport(plugin="refinery", summary=SUMMARY,
                        solver_notes=SOLVER_NOTES, symptoms=SYMPTOMS)
