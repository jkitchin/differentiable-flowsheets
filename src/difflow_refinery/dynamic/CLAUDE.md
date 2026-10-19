# difflow_refinery.dynamic

Notes for Claude Code, loaded when working in this directory.

Holdup-based dynamic versions of the refinery units, on difflow's
`difflow.dynamic` protocol (`state_spec`, `initial_state`, `derivatives`,
`outputs`; `output_dependencies` + `partial_outputs` where an outlet reads a
subset of the inlets), so they run in `DynamicFlowsheet`. Phase 1:
`DynamicHydrotreater` (built from a solved `Hydrotreater`),
`DynamicReformerTrain` (heaters + beds of a solved `CatalyticReformer`),
`QuasiSteadyHoldup` (a steady op behind outlet lags, for fast units), and
`signals` (setpoints/disturbances as integrator `args`). Docs:
`docs/unit-operations-refinery.md`, "Dynamic operation". Gates:
`tests/reference/dynamic_refinery_checks.py` (~40 min, not in the suite);
smoke/grad tests: `tests/refinery/test_dynamic.py`.

Invariants (do not weaken them):
- NO duplicated physics. A hydrotreater cell steps the steady bed's own
  right-hand side (`hydroprocessing.reactor.bed_derivatives`); a reformer
  cell calls the steady `reforming.reactor.integrate_bed` over its catalyst.
  The phase split, enthalpies, K-models, flash, compressor, amine, purge and
  makeup rule are the steady modules' functions.
- The rest state IS the steady unit's: reaction is computed on a cell's
  inflow across the cell (`F*`, `T*`) and the holdup relaxes to it
  (`dn/dt = F* - F_out`, `C dT/dt = H(F*,T*) - H(F*,T)`), so at rest the
  cells are the steady bed integrated in pieces. Do not move the reaction
  onto the holdup composition (that is a CSTR series: first order, 2x off on
  a 99 % HDS bed at 10 cells). The price is a half-cell lag in transients.
- Element balances close EXACTLY along a trajectory (accumulation = in -
  out at every state; tested). Anything new must keep that: e.g. a vent or a
  bleed is a stream, never a sink term.
- Hydrotreater cells: fixed RK4 (`n_sub` per cell) is stable and accurate
  on the diesel (10 steps/bed: bed-outlet S within 5e-5 of adaptive).
  Reformer cells must NOT use fixed RK4: the dehydrogenation equilibria at a
  bed inlet need >1000 steps (NaN otherwise) -- hence `integrate_bed`.
- States are SCALED (design holdups, floored like the steady bed's scale;
  temperatures /100, coke /0.01) so diffrax tolerances (rtol 1e-6, atol 1e-8)
  mean the same thing everywhere.
- Disturbances go through `args` (`signals.signal_args`, smoothstep ramps),
  read by a `"signals"` feed -- never closures, so one compile serves every
  scenario and step sizes/times are differentiable.
- Residence times (`tau_G`, `tau_L`, loop `tau`) are DESIGN values fixed at
  construction; pressure dynamics are not modelled (all pressures fixed).
  Holdup parameters are ILLUSTRATIVE (`HoldupParams`); they set speeds, not
  rest states.
- The steady quench must be given (`HydrotreaterParams.quench`), not solved.

Phase 2b (utilities): `DynamicHydrogenPlant` (`hydrogen.py`), `DynamicAminePlant`
/ `DynamicAmineUnit` (`treating.py`), `DynamicSRU` (`sulfur.py`),
`DynamicCrudeUnit` (`crude.py`), and `DynamicEquilibriumBed` (`beds.py`, the
shift / Claus / hydrogenation beds). Gates:
`tests/reference/dynamic_utilities_checks.py` (~40 min). The same invariants
apply, plus:
- An equilibrium bed's cell takes the steady unit's RESTRICTED equilibrium
  isothermally at the cell temperature; at rest cell 1 is the adiabatic
  equilibrium. Its thermal time uses the gas's heat flow ALONG equilibrium
  (`thermal_time(equilibrium=True)`), which the simulation reproduces.
- The SMR tube cells call `smr._integrate` over `z_span`; the tube is
  flux-mode only (the firing is a signal behind `FiringLag`).
- The amine loop's tear (lean loadings, hot-end T) IS its state: lean and
  rich inventories hold fixed amine; water in the lean inventory is held at
  design by the makeup, so only the rich water is a state.
- The SRU's TGTU solvent inventory is what breaks the furnace <- recycle
  loop (`recycle` reads the state only); the air comes from
  `AirDemandControl` (integral trim, default off).
- The crude column is re-specified with a COT spec (from an overflash or
  duty spec) at its steady COT; rate/duty specs scale with the charge.
