# difflow_water

Notes for Claude Code, loaded when working in this directory. The plan is
`DESIGN.md`; the docs are `docs/unit-operations-water.md` and
`docs/water-summary.md`; tests are `tests/water/`.

## Rules for this plugin (do not weaken them)

- **Every value has a reference with a status.** A coefficient in code
  carries its reference ID (`[S1]`, `[M4]`, ...) from the references table in
  `docs/unit-operations-water.md`, or the word ILLUSTRATIVE. Status terms:
  citation verified (Crossref record checked), content verified (equation
  read against the source), cross-checked, (unverified), ILLUSTRATIVE.
  Something copied from WaterTAP's code is cited to the primary paper AND
  marked unverified until the paper itself is read. Check every new DOI
  against Crossref: two recalled DOIs in this plugin's history were wrong
  (Swamee-Jain 1976, Bartholomew & Mauter 2019).
- **WaterTAP is the benchmark.** Match WaterTAP's equations and conventions
  unless the primary paper says otherwise (then the paper wins and the docs
  say how WaterTAP differs). After changing a unit, rerun
  `make watertap-compare` (reference: `benchmarks/watertap/`) and update the
  comparison section of the docs. Never turn the comparison into a test.
- **Streams are plain difflow streams**: `F_H2O`, `F_TDS` (mol/s, one mole of
  TDS = `M_S` = 31.4038218 g, TEOS-10), `T`, `P`. Do not add a parallel
  stream type.

## Invariants

- Seawater properties (`chemistry/seawater.py`): Sharqawy 2010 Eqs. 8, 22,
  23, 49 with S in kg/kg and t in ITS-90 C; content verified. The osmotic
  pressure is `phi m rho_w R T` on the `M_S` molality basis: no van't Hoff
  factor (m already counts the ions; WaterTAP's code has an unused `i = 2`).
- `ROStage` default `module_type="flat_sheet"` (`f = 0.42 + 189.3/Re`) is
  WaterTAP's default. The spiral-wound `6.23 Re^-0.3` gave 2-14 % different
  pressure drops in the comparison; do not make it the default without
  saying so in the docs.
- The local flux solve is bisection + `_implicit_step` (first-order implicit
  gradients). Keep the salt-flux quadratic in its stable form
  `2c / (b + sqrt(b^2 + 4c))`; it is exact at `J_w = 0`.
- `PressureExchanger` draws the brine's volume from its `feed` and returns
  the rest as `bypass` (WaterTAP's splitter + equal-flow constraint, done
  sequentially). `swro_performance` uses one pass when mixing is 0: the RO
  feed is then the intake at the RO pressure, exactly.
- `economics.py` mirrors WaterTAP 1.8.0's default costing (TIC 2,
  CRF 0.1, utilization 0.9, electricity 0.07 USD/kWh, year = 365.25 d,
  pump work costed as electricity). Change defaults only with WaterTAP.

- Pitzer chemistry (`chemistry/pitzer.py`, `speciation.py`): equations from
  PHRQPITZ (Plummer et al. 1988), content verified; parameters in
  `chemistry/data/pitzer_seawater.py` are GENERATED from PHREEQC pitzer.dat
  by `benchmarks/phreeqc/gen_pitzer_data.py` -- regenerate, never hand-edit.
  `A_phi` uses the defining Eq. 4 with CODATA constants (PHRQPITZ's printed
  Eq. 5 constant is inconsistent; using it put NaCl gamma 0.2 % off PHREEQC).
  PHREEQC prints and reads pH on the MacInnes scale: compare `pH_macinnes`.
  After changing the chemistry, rerun `make phreeqc-compare` (SI within
  ~4e-4 of PHREEQC is the current state).
- Compile time matters (180 s per-test budget): the Chebyshev recursion is
  a `lax.scan`, the gamma passes a `fori_loop`, and `_speciate` is jitted.
  Unrolling them again tripled the gradient compile time.

- `ROStage` runs as one jitted function (`_run_stage`, static config
  flags); the scaling step (`_scaling`) stays OUTSIDE that jit so the
  separately jitted speciation compiles once per process. Inlining it made
  the editor's first solve take ~100 s.
- Editor (GUI): every palette operation needs a drawing in the difflow-book
  repo, `app/frontend/src/lib/nodes/symbols.js` (the app's
  `tests/test_symbols.py` fails on an undrawn catalog operation or
  category), then `npm run build` to refresh the committed bundle. A new
  unit must also be droppable from the palette: required numeric fields get
  a 1.0 placeholder, anything else needs a default (hence `ROTrain`'s
  default stages and `WaterPump`'s `deltaP = 0`). The GUI example is
  `src/difflow/workspace/examples/10_water_swro.json`, written by
  `scripts/make_gui_examples.py` (`water()`); it sets `error_probe: 0`.

- Networks (`network/`): one residual in `hydraulics.py` for both valve
  formulations; do not fork it. EPANET 2.2 [N8] is the reference: rerun
  `make wntr-compare` (benchmarks/wntr) after a change; current state is
  2.7e-5 m on Net1/Net3, 1.6e-3 m on ky10 outside its closed pump pocket.
  Swamee-Jain is log10 (the manual's `ln` is a typo; `ln` put heads 70 m
  off). Closed links are `h = R Q` (not `Q = 0`: that leaves heads behind
  closed links undefined). Pumps with no flow path are compiled closed.
  Newton stages stop on tolerance (`while_loop`), not a fixed count: a
  fixed 40 left ky10 unconverged in the smooth formulation, silently.
  `solve()` rejects parameter arrays of the wrong shape: an out-of-range
  gather is clamped forward and its gradient dropped, so a short array
  gave a zero derivative.
- INP reader (`network/inp.py`): unsupported features raise, never skip.
  Validate a reader change against WNTR's reading (exported networks in
  `benchmarks/wntr/networks/`, all numeric fields to 1e-6).
- EPS (`network/eps.py`): decisions (controls, step lengths) are made on
  concrete values; gradients come from replaying the recorded `plan`, where
  tank-level-cut steps are recomputed in JAX. Do not make `simulate` decide
  on traced values. Rerun `compare_eps.py` after a change (Net1 1e-4 m,
  Net3 3.4e-2 m, 0 status mismatches).
- Water quality (`network/quality.py`) is NumPy and follows the manual's
  Lagrangian algorithm step by step; keep it that way and rerun
  `compare_quality.py` (Net1 chlorine 0.01 mg/L, age 0.16 h). Closed links
  and sub-QZERO flows carry no water: giving them their modelled trickle
  put Net3's node 10 at 100 % a step early.

- Thermal (`thermal/`): `mvc_performance` is WaterTAP 1.8.0's
  `mvc_single_stage`, equation for equation; rerun `make watertap-compare`
  after a change (current state: 4e-7 on every output, optimum to 1e-7).
  Three WaterTAP quirks are matched on purpose and documented: costs in
  USD 2018 (the example sets USD_2020 after costing, which does nothing),
  the example's preheater `deltaP = +7 kPa` is a GAIN (comparison passes
  `hx_dP = -7e3`; difflow's default is a loss), compressor `gamma = 1.3`.
  The plant is ONE `optimistix` Newton solve (8 unknowns); do not split it
  into sequential sub-solves (the condenser-evaporator coupling makes
  that a recycle). Default `mode="saturated_condensate"`: with a fixed
  pressure ratio almost every design is unphysical (see the docs); keep the
  `feasible` flag honest. MED is balances only: do not add fitted
  coefficients to it without a content-verified source.

## Not palette operations

`chemistry`, `economics`, `network`, `swro_performance`/`build_swro_flowsheet`
are libraries. `mvc_performance`, `med_performance` and the thermal properties are
libraries too. Palette: `ROStage`, `ROTrain`, `PressureExchanger`,
`WaterPump`, `MVCDesalinator`, `MEDDesalinator`.
