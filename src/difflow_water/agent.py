"""Agent support for difflow_water (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport

SUMMARY = """\
Water treatment and desalination. Seawater streams carry F_H2O and F_TDS
(mol/s; one mole of TDS is 31.4038218 g of Reference-Composition sea salt);
make_seawater_stream(flow_vol=..., salinity=...) builds one. Seawater
properties (density, viscosity, osmotic coefficient and pressure) are in
difflow_water.seawater (Sharqawy et al. 2010). Units: ROStage (seawater RO,
solution-diffusion with polarization, plug flow, optional spiral-wound
channel correlations), PressureExchanger (outlets hp_feed, brine, bypass),
WaterPump. swro_performance(feed, ro, ro_pressure) gives the seawater RO +
pressure exchanger plant's recovery, specific energy and LCOW as one
differentiable function; build_swro_flowsheet gives the same plant as a
Flowsheet. Costing follows WaterTAP's default costing package so results
compare with WaterTAP."""

SOLVER_NOTES = """\
The RO stage is integrated with fixed-step RK4 along its area (n_steps,
default 40); near the osmotic limit (outlet driving pressure near zero, see
info['min_driving_pressure']) raise n_steps. The pressure exchanger draws
the brine's volume from its feed; the feed must carry at least that volume
(bypass_fraction >= 0). With pressure-exchanger mixing the concentrate loop
is a real recycle; with mixing = 0 one pass is exact."""


def support() -> AgentSupport:
    return AgentSupport(plugin="water", summary=SUMMARY, solver_notes=SOLVER_NOTES)
