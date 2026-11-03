"""Agent support for difflow_water (the ``difflow.agent`` entry point)."""

from __future__ import annotations

from difflow.agent.plugins import AgentSupport

SUMMARY = """\
Water treatment and desalination. Seawater streams carry F_H2O and F_TDS
(mol/s; one mole of TDS is 31.4038218 g of Reference-Composition sea salt);
make_seawater_stream(flow_vol=..., salinity=...) builds one. Seawater
properties (density, viscosity, osmotic coefficient and pressure) are in
difflow_water.seawater (Sharqawy et al. 2010). Scaling: dw.speciate(totals, T)
(Pitzer model as in PHRQPITZ, PHREEQC pitzer.dat parameters) gives pH and
saturation indices of calcite, gypsum, anhydrite, brucite, ...; totals from
dw.reference_seawater(S) or dw.seawater_feed(S, pH). Units: ROStage (seawater
RO, solution-diffusion with polarization, plug flow; scaling=True reports
saturation indices at the outlet wall), ROTrain (stages in series),
PressureExchanger (outlets hp_feed, brine, bypass; leakage, mixing),
WaterPump. Distribution networks: difflow_water.network (WaterNetwork,
NetworkModel(net).solve(params)) solve heads and flows equation-oriented with
EPANET 2.2's equations, differentiably; load_example("Net3") gives an EPANET
example with EPANET's own answer. Thermal desalination: MVCDesalinator /
mvc_performance (WaterTAP's single-stage MVC plant; LCOW, SEC, evaporator
area) and MEDDesalinator / med_performance (parallel-feed multi-effect
distillation; GOR, areas); thermal seawater properties (cp, enthalpy,
latent heat, vapor pressure, BPE) in difflow_water.seawater. swro_performance(feed, ro, ro_pressure) gives the seawater RO +
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
is a real recycle; with mixing = 0 one pass is exact. Saturation indices are
at 1 atm and assume equal rejection of all solutes; pH_macinnes is the scale
PHREEQC uses. MVC: the default mode solves the compressor pressure ratio
for saturated condensate; with mode="fixed_pressure_ratio" most designs are
unphysical, so check info['feasible'] (condensate_subcooling >= 0,
evaporator_dT_out > 0). Size the preheater areas to the feed (about 4.5 and
3.9 m^2 per kg/s at WaterTAP's optimum)."""


def support() -> AgentSupport:
    return AgentSupport(plugin="water", summary=SUMMARY, solver_notes=SOLVER_NOTES)
