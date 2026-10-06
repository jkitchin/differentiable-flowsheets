"""Agent support for difflow_ree (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport

SUMMARY = """\
Rare earth solvent extraction: REEExtractor, REEMixerSettler, REEScrubber and
REEStripper; oxalate, carbonate and hydroxide precipitators; and the circuits
ExtractStripCircuit, ExtractScrubStripCircuit, SplitShellCascade and
FullSeparationTrain. The database holds 15 elements (the 14 stable
lanthanides and Y) and five extractant systems (D2EHPA, PC88A, Cyanex 272,
TBP, naphthenic acid). Coverage is uneven: only naphthenic acid has
coefficients for all fifteen; the others cover ten (no Ho, Er, Tm, Yb, Lu).
Check coverage before choosing an extractant for a heavy-REE separation."""

SOLVER_NOTES = """\
Distribution ratios use the FREE extractant concentration, closed by a
monotone scalar root with implicit gradients (solve_free_extractant); a
loading past the extractant's capacity is rejected rather than extrapolated.
Cascades converge by correlation, then damped Newton with a trust region and
feed ramping inside the circuit units, so a circuit that fails to converge is
usually loaded beyond capacity or asked for a separation the extractant cannot
make, not short of iterations."""


def support() -> AgentSupport:
    return AgentSupport(plugin="ree", summary=SUMMARY, solver_notes=SOLVER_NOTES)
