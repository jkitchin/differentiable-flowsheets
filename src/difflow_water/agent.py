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
latent heat, vapor pressure, BPE) in difflow_water.seawater. Conventional
treatment: Sedimentation (ideal basin, Stokes), Filtration (Yao colloid
filtration, Carman-Kozeny), GACAdsorber (WaterTAP's CPHSDM), Disinfection
(Chick-Watson, CT, log inactivation); streams carry F_H2O plus F_<name> per
solute (molar mass given to the unit), particle class (1 mol = 1 g) or
organism (1 mol = 1 organism). Wastewater: ActivatedSludgePlant /
activated_sludge_plant (BSM1: ASM1 reactors + Takacs settler at steady
state; bsm1_indices for effluent quality and energy), ASM1Reactor,
TakacsSettler; ASM1 streams carry F_<state> with 1 mol = 1 g COD or N and
Q = F_H2O M_w / 1000; bsm1_influent() is BSM1's influent. swro_performance(feed, ro, ro_pressure) gives the seawater RO +
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
3.9 m^2 per kg/s at WaterTAP's optimum). The BSM1 plant as a sequential
Flowsheet (build_bsm1_flowsheet) does not converge from default tear guesses
(the sludge recycle gain is near one; Anderson finds autotroph washout):
use activated_sludge_plant or pass its states as tear_initial."""


def support() -> AgentSupport:
    return AgentSupport(plugin="water", summary=SUMMARY, solver_notes=SOLVER_NOTES)
