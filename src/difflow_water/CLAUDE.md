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

## Not palette operations

`chemistry`, `economics`, `swro_performance`/`build_swro_flowsheet` are
libraries. Palette: `ROStage`, `ROTrain`, `PressureExchanger`, `WaterPump`.
