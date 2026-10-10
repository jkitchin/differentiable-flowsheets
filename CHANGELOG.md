# Changelog

All notable changes to difflow are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org) (pre-1.0: a minor bump may break the API).

## [Unreleased]

### Breaking

- **Core database heat capacities moved; results change (#393).** Every
  species now carries an explicit ideal-gas Cp cubic (`SpeciesData
  .Cp_vapor_coeffs`; Poling, Prausnitz & O'Connell 5th ed. Cp/R quartic,
  cubic-fitted over 250-800 K, within 1.2%; styrene by Joback) in addition
  to the liquid Cp (`Cp_coeffs`, where a liquid exists at ambient).
  `CubicThermo` and the entropy integral `IdealThermo.S_ig_T` read the
  ideal-gas field (new `IdealThermo.Cp_ig`, `Cp_mix_ig`, `H_ig`,
  `stream_enthalpy_ig`); `IdealThermo` keeps integrating the liquid one. This
  moves numbers: a `CubicThermo` flowsheet with water or an alcohol used a
  liquid Cp (water 75.3, methanol 81.0) as its ideal-gas Cp (about 33.6 and
  44), now fixed, and the 24 species that were still constants (n-heptane,
  n-octane, benzene, toluene, ...) are temperature dependent, so n-heptane at
  600 K gains about 40%. On `IdealThermo`, n-heptane and n-octane liquid Cp
  are now 225 and 254 (they held ideal-gas 166 and 189), and the species
  #172 gave ideal-gas cubics (C1-C6, N2, CO2, ...) keep that cubic for
  liquid enthalpy too (no ambient liquid exists). Data that stores the
  ideal-gas Cp in `Cp_coeffs` with no `Cp_vapor_coeffs` (the Cantera, DWSIM
  and PyGLenN importers) is unchanged. Downstream models with IDAES or other
  reference data on these species need that data regenerated.

- `REEDistribution(no_data=...)`, threaded as `no_data=` through the REE unit
  and circuit params, says so when there is no data (#384): `"nan"` answers
  NaN for an element with no coefficients (Ho to Lu on D2EHPA, #269) and for
  the elements a record lists in the new `unmeasured_outside_window` (D2EHPA:
  Gd, Tb, Dy, Y) below the fitted pH window, which is where heavy-REE
  stripping has to run; `"raise"` makes the latter an error. A NaN is not
  filtered by anything downstream, so a calculation that uses it returns NaN.
  The default `"warn"` is the previous behaviour.
- `Saponifier` reads a solvent stream's extractant on the record's own basis
  (dimers for D2EHPA, PC88A and Cyanex 272) and converts with
  `Extractant.monomers_per_basis_unit`, so it counts the same exchangeable
  equivalents as the extractor charge (it counted half on a dimeric extractant
  fed from an extract-scrub-strip module; #386). `Saponifier.to_formal_basis`
  hands its output to the formal-monomer `SaponifiedSection`.

- REE strip acid floor (stopgap for #384). A strip pH the code chooses from
  the D curves is floored at `-log10(max_strip_acid)`, default 6 M (pH
  -0.78), on `ExtractStripParams`, `ExtractScrubStripParams`,
  `GroupSeparator`, `SeparationTrainParams`, `design_extract_strip` and
  `design_extract_scrub_strip`. The cut put heavy-REE stripping on D2EHPA at
  pH -1.17 (14.6 M H+), beyond any real strip liquor and 1.2 pH units below
  the fitted window. At the floor the strip leaves REE on the solvent, now
  reported as `results["strip_retained"]` and the train's solvent holdup
  (default D2EHPA Gd/Tb/Dy/Y circuit: Y recovery 97.5 % -> 63.6 %, 34.7 % of
  the Y entering the strip stays on the barren organic; default train
  recovery 1.000 -> 0.972); the design helpers add strip stages at the floor
  (Nd/Dy: 2 -> 4) and raise `StripAcidLimitWarning` when no stage count meets
  the target. `StripperParams` refuses an `acid_conc` above
  `max_strip_acid` when it sets the pH. A pH you pass is never clamped;
  `max_strip_acid=None` restores the unbounded cut.
- REE extractant basis (closes #374, which kept the dimer basis): every
  extractant concentration is documented on the record's own basis (dimer
  for D2EHPA, PC88A, Cyanex272: 0.5 M dimer = 1.0 M formal; the molecule for
  TBP and naphthenic acid), and the boundary to the mass-action layer (formal
  monomer) now converts with the new `Extractant.monomers_per_basis_unit`.
  `log_K_from_correlation` / `MassActionSection` evaluated the correlation at
  the monomer number as if it were dimer (D2EHPA's K 2**2.38 too large,
  Cyanex272's 8x), and `REEExtractor(model="mass_action")` passed its
  record-basis `extractant_conc` and solvent stream unconverted (the closure
  saw half the extractant: twice the loading, half the free extractant).
  Mass-action numbers calibrated from the correlation move accordingly.

- Open-issue sweep (#373, #374, #376, #377, #378, #379, #380, #381). REE: the
  organic phase flow is the diluent volume alone (the extractant entry is a
  moles-per-volume charge), so a stated O/A of 1 runs at 1 and not 1 +
  `extractant_conc`; the loading capacity is counted in the concentration
  basis (`Extractant.basis_units_per_ree`, 3 dimers per REE for D2EHPA, PC88A
  and Cyanex 272), doubling it; `ExtractScrubStripParams.recycle_scrub_liquor`
  closes the scrub-liquor loop with a tear, and `GroupSeparator` /
  `FullSeparationTrain` recycle by default. Carbon capture: `AmineStripper`
  models the overhead condenser (`steam_ratio`, `T_condenser`; `reflux_ratio`
  is deprecated and ignored), and the amine and adsorbent degradation defaults
  are recalibrated. Pinned numbers moved with these. Also: `PlatformDSPParams.
  uf_concentration_factor`, `RadialFeederFlowsheet` refuses downstream PV
  buses, DIP feed hydrogen bypasses the column, SEC bypasses unloaded
  material to the product, the REE palette wrappers run under `jit`/`grad`,
  `tests/conftest.py` imports difflow (xdist deadlock), and
  `tests/test_doc_examples.py` runs every documentation example.

- Flowsheets read a unit's result with one shared reader that tells a Stream
  from an info dict by its keys and understands `((S, S), info)`; a unit that
  returns a different number of streams than it has outlets now raises
  instead of dropping the extras. The catalog counts streams inside a nested
  leading tuple, so Ultrafiltration, Diafiltration, the three chromatography
  units, PowerSplit and BranchFlow have their real outlet counts (was 0).
  `FedBatchReactor`/`SemiBatchReactor` no longer list `C0` as an inlet.
- REE circuits pick any section pH left unset from the extractant's own D
  curves (extraction, scrub and strip each at their cut for the circuit's
  elements and targets), replacing fixed fractions of the fitted window that
  no longer separated after the #270 refit; the design helpers size stages
  from the actual D. `REEStripper`'s `acid_conc` sets the strip pH; TBP
  circuits strip at their own `strip_nitrate_conc`; FullSeparationTrain runs
  its cerium oxidizer at the oxidizer's temperature (Ce removal 8% -> 81%).
  The ExtractStrip, ExtractScrubStrip, SplitShell and SeparationTrain palette
  operations return streams. Precipitators are limited by the reagent fed and
  pass non-REE species through.
- Carbon capture: `AmineAbsorber` capture is bounded by the absorption factor
  (A < 1 gave near-total capture) and uses `solvent_in` when given; the rich
  solvent carries total CO2. `MembraneSeparator` solves the per-species flux
  balance and `feed_pressure` defaults to the stream's; `CompressionTrain`
  compresses from the stream pressure (`P_inlet` defaults to None).
  `SolidSorbentDAC` capture is bounded by the air fed.
- `IsomerizationUnit` refuses feed species it does not model and keeps feed
  H2 above its target; `CrudeUnitWithPreheat` adds the inlet stream's water to
  the tank water; `build_ladder_flowsheet` carries root loads, shunts,
  generators and reversed taps.

- `Ultrafiltration`/`Diafiltration` set each species' rejection from the MWCO
  and its molecular weight (new `difflow_bio.database.BIO_SPECIES_MW_KDA`,
  `molecular_weights` parameter); with no `rejection` given they passed a
  150 kDa mAb freely. `rejection_from_mw` puts R = 0.9 at the cutoff (was
  0.5). `Diafiltration`'s buffer stream is the buffer consumed when
  `n_diavolumes` is omitted (now optional); with it given, the scaled buffer
  is `info['buffer_consumed']`. `TFF.uf_df_uf` returns
  `((product, perm1, perm2, perm3), info)`.
- Chromatography: `load_volume` needs `feed_volume` (it was divided by the
  stream amount), and `info['yield']` is product out over product in.
  `IonExchangeChromatography` uses `column_volume`, `q_max` and `K_d` as a
  binding capacity; Protein A uses `K_d` when the feed concentration is known.
- `mAbDSPTrain`, `PlatformDSP` and `ViralClearanceTrain` return every outlet
  in `side_streams`, so product + side streams = feed. `mAbDSPTrain` step
  yields are against each step's own inlet (adds `tff1`, `tff2`) and its
  final TFF concentration factor is a volume ratio. `PlatformDSP` rejects
  `capture_type="mmc"` and polish steps other than cex/aex, and
  `include_viral_filtration` adds a virus filtration step (default on, 0.97
  recovery). `ViralClearanceTrain` step methods return `((kept, lost), info)`
  and low-pH recovery no longer exceeds 0.98.

### Added

- **Evaporators (#404).** `Evaporator` (single effect, rating by area or design
  by product concentration), `MultiEffectEvaporator` (forward, backward and
  parallel feed; equal-area design solved as one Newton system, or rating with
  given areas; per-effect `U`) and `MechanicalVaporRecompression` (ideal-gas
  isentropic compression, compressor work against steam saved), in
  `difflow/units/evaporator.py`. Each returns `(concentrate, vapor, info)` with
  steam rate, steam economy, per-effect T, P, BPR, vapor, area and Q, and
  closure residuals; `info` is differentiable with respect to `U`, the feed and
  the steam pressure (implicit gradients through `optimistix`). The
  non-volatile solute is an ordinary stream species whose vapor flow is exactly
  zero. `boiling_point_rise` offers Raoult (`ideal`), `colligative`, user
  Duhring lines, polynomial, callable and built-in `naoh`/`nacl`/`sucrose`
  models, plus `bpr_fn=` and `enthalpy_fn=` hooks (heat of dilution). Water
  saturation uses IAPWS-IF97; the enthalpy fits are to recalled steam-table
  points. **Not verified**: the built-in NaOH, NaCl and sucrose BPR data are
  approximate recollections (they raise `UnverifiedDataWarning`), and no
  textbook worked example is claimed to be reproduced. Registered in the
  palette under a new `evaporation` category with its own symbol (GUI bundle
  rebuilt), documented in `docs/unit-operations-chemical.md`, example notebook
  `examples/47_evaporators.ipynb` (optimal number of effects, with illustrative
  placeholder prices). No evaporator cost curve was added to
  `economics/capital.py` (no citable constants at hand); use
  `heat_exchanger_cost` as a stand-in.

- `difflow.solvers.pounce_problem` builds the configured pounce Problem once
  so repeated solves reuse the compiled residual and Jacobian (about 0.1 s
  against 30 s on a nested-EOS flowsheet, #394). `solve_with_pounce` is now a
  thin wrapper over it and still rebuilds on every call; its docstring and
  `docs/external-solvers.md` say so.
- `InstantaneousUnit` sees time and the integrator `args` (#396): an `fn`
  taking `(t, inputs, params)` or `(t, inputs, params, args)`, and
  `call_params` callables taking `(t, params)` or `(t, params, args)`, chosen
  by arity (the 2-argument `fn` and 1-argument callable forms are unchanged).
  `DynamicFlowsheet.simulate(args=...)` is documented as the one place for
  time-varying disturbances, read by feeds and units alike.

### Fixed

- A thermo or EOS object built inside `jax.jit` no longer gets a value key
  (#395): its derived arrays are tracers, and a later equal-valued eager
  object hit a cache entry holding dead tracers (`UnexpectedTracerError`).
  It falls back to identity, a missed cache rather than a wrong hit.
- The missing-dependency message names the PyPI distribution only as
  different from the import name when it is (`asdex`, #394).

## [0.3.0] - 2026-10-06

### Breaking

- The operation registry refuses a second class under a name that is already
  registered (`ValueError`), as its docstring always said; it used to log a
  warning and overwrite. The refinery wet-gas compressor, which silently
  replaced the core `GasCompressor` whenever the refinery plugin was installed,
  is registered as `WetGasCompressor` (the class is still
  `difflow_refinery.GasCompressor`) (#359).
- `FedBatchBioreactor`: with `m_s > 0`, maintenance uptake now tapers as
  `S/(K_m + S)` near substrate exhaustion (new `FedBatchParams.K_m`, default
  0.05 g/L), so results with maintenance change slightly (about 5% at
  S = 1 g/L) (#362, #366).
- `mAbDSPTrain` and `PlatformDSP` load the whole batch onto each column, with
  the column volume acting as a capacity; they used to load one column volume
  of the harvest, which gave yields of a few percent. Purity is now realistic
  (99.4% for the mAb train, not 1.0000) (#360, #366).
- `difflow_bio.economics.estimate_total_opex` prices each chromatography step
  with its own resin (it priced all of them as Protein A), and it and
  `cost_per_gram` emit a `DeprecationWarning` pointing to `cogs_breakdown`
  (#358, #363).

### Added

- **`difflow mcp`**, an MCP server that lets an AI agent (Claude Code, Claude
  Desktop or any stdio MCP client) use difflow: install with
  `pip install "difflow[mcp]"` and register with
  `claude mcp add difflow -- difflow mcp`. The tools are plain Python in
  `difflow.agent` (`Workbench`), with a thin adapter in `difflow.mcp` over the
  MCP SDK 2.x. See `docs/agents.md` (#359, #364, #365, #367, #368, #369).
  - Discovery from the installed code, with no hand-written lists: operations
    from the registry and catalog, library APIs from each package's `__all__`,
    species, documentation search, examples and plugin status.
  - Building and solving: named sessions, units with parameters in one call,
    wiring (a loop closes as a recycle), feeds, every solver option, and solve
    results with the loop gain, error estimate and captured warnings.
  - `diagnose` and `converge`: findings with remedies, and a remedy search
    judged "converged AND correct" that applies only numerics remedies and only
    when asked, reports multiple steady states, and passes every case of the
    convergence corpus.
  - An expression language for objectives (`vapor.F_x / vapor.total_flow`),
    named quantities, and `sensitivity`, `sweep`, `optimize`, `uncertainty`,
    `linearize`, `tea` and `report`.
  - Each plugin contributes its own agent support through a `difflow.agent`
    entry point; `power_flow` and `power_opf` reproduce MATPOWER's benchmarks.
  - Progress notifications for long calls, three workflow prompts, and
    `open_in_editor`. `--no-exec` leaves out the tools that run Python.
- `Flowsheet.last_solve_history` (the tear residual per iteration on the
  Anderson and Wegstein paths) and `Flowsheet.last_solve_unit_info` (each
  unit's info dict from the last concrete evaluation) (#364).
- CSTR and Flash report `balance_residual` and `converged` for their own inner
  solves; the editor's solve audit flags a unit whose balance did not close
  (#364).
- `difflow.diagnostics`: symptom cards, a remedy table, a residual-history
  classifier and `solve_findings`, shared by the editor and the agent tools
  (#364).
- The catalog says what each outlet is (`PortSpec.outlet_roles`), read from the
  unit's docstring (#368), and units may declare a `cost_basis` that the
  catalog carries (#369).
- `difflow.plugins.plugin_status()` reports installed plugins and why any failed
  to load (#359).
- `difflow_bio.economics.cogs_breakdown`: cost of goods by category and step,
  with per-step resins, cycles from load / (DBC x CV), labor from batches and
  steps, buffers, media, QC, single-use items and failed batches, all from data
  (`load_cost_model`, YAML or JSON). Benchmarked against the published mAb
  process of Petrides (Intelligen, 2015, section 11.6.3) in
  `data/petrides2015_mab.yaml` (#358, #363).

### Changed

- The editor's session exposes every setting of `Flowsheet.solve` (not just
  four) and returns the gain, error estimate, clipping and warnings with a
  solve; its messages no longer assume a canvas. `difflow.gui` imports the HTTP
  server lazily (#359).

### Fixed

- diffrax integrations can be jitted and vmapped (`int()` on solver statistics)
  (#361, #366).
- Fed-batch substrate no longer goes negative under starvation; the diffrax and
  RK4 paths agree (#362, #366).
- `mAbDSPTrain`, `PlatformDSP` and `ViralClearanceTrain` are differentiable
  (they converted results with `float()`), and `PlatformDSP(include_sec=True)`
  no longer raises (#360, #366).

## [0.2.2] - 2026-10-05

### Fixed

- Gas plant columns: 0.2.1's retry of a non-converging first pass compiled
  that pass three times (a `lax.cond` per retry). Every jit or `jacfwd`
  through a gas plant column grew by about a quarter -- a `jacfwd` through
  the isomerization DIH went from 10.7 to 13.3 GB -- and the shard of the
  Publish gate holding the DIH tests ran its runner out of memory, so 0.2.1
  never reached PyPI. The retries are now one `while_loop` that traces the
  pass once; same refluxes, same answers.
- C5/C6 isomerization: one pass of the DIH recycle is jitted (it ran eagerly,
  op by op, on every Anderson iteration and every `jvp`), and its implicit
  derivative linearizes the pass once instead of compiling three `jvp`s. A
  recycle solve went from about 390 s to 17 s; a `jacfwd` through the unit
  from 213 s and 13.3 GB to 103 s and 7.1 GB.

## [0.2.1] - 2026-10-05

### Fixed

- Gas plant columns: the initial guess solved its component balances by an LU
  solve, which returns round-off for a heavy cut at the top of a long
  column (true ~e^-600). A 1e-12 change of a feed, or a different CPU's
  rounding, moved those log flows by hundreds and the deisobutanizer's first
  Newton pass could stall on a near-singular Jacobian (this failed the 0.2.0
  Publish gate, so 0.2.0 never reached PyPI). The guess now eliminates the
  M-matrix without pivoting in logs: every gas plant factory case converges in
  the same iterations under feed perturbations. A first pass that still does
  not converge is retried from guesses at half and twice the guessed reflux
  (the C3/C4 splitter on an FCC feed needs it).

## [0.2.0] - 2026-10-05

### Breaking

- Python 3.10 is no longer supported; difflow requires Python >= 3.11 (#261).

### Added

- **`difflow_refinery`**, a new plugin for petroleum refining: crude assay
  characterisation (Twu 1984), differentiable crude and vacuum distillation
  units, a crude preheat train with a desalter and preflash drum, a saturated gas
  plant, C5/C6 isomerization (once-through and with DIH/DIP recycle), a
  hydrotreater, a hydrocracker, an FCC, a reformer, alkylation, a residue
  desulfurizer, a hydrogen network and a product blending pool with nonlinear
  blending rules. The crude unit also works as a delta-base planning block. It is
  validated against independent IDAES, Pyomo/IPOPT and DWSIM 9.0.5 references,
  and there is a whole-refinery example (`examples/40`)
  (#296, #299, #300, #302, #303, #316, #317, #319-#322, #324, #335, #336, #345, #351).
- **`difflow_power`**, a new plugin for electrical grids: AC power flow, DC-OPF
  and AC-OPF (#205).
- **`difflow.planning`**: delta-base planning from differentiable flowsheets,
  with second-order subproblems, feasibility restoration, multi-period inventory,
  delta-vector health checks, delta attribution estimated from plant history,
  and Pyomo export (#184, #185, #187, #252, #293).
- **`difflow.stochastic`**: design under uncertainty by sample average
  approximation (#236).
- **`difflow.reconciliation`**: data reconciliation, monitoring over time,
  pooled parameter estimation, and gating and filtering of parameter updates
  from plant data (#175, #183, #244).
- **Flowsheets as data**: kinetics, a unit catalog, JSON serialization and code
  generation (#182). `Params` field descriptions now appear in the catalog
  schema (#218).
- **Local browser editor** (`difflow.gui`): build, wire, rename and edit
  flowsheets, open Python-script flowsheets, browse example flowsheets, and
  use a code editor, hover cards and per-unit documentation
  (#186, #216, #229, #230, #233, #240-#243, #337, #340-#344).
- **Recycle convergence**: automatic tear-stream selection
  (`fs.solve(tears="auto")`, `fs.tear_analysis()`), loop-gain and error
  estimates for the step tolerance (`TearToleranceWarning`,
  `tol_basis="error"`), a corpus of hard flowsheets for measuring the solver's
  pass rate, and a working `solve(damping=...)` (#258, #260, #262, #273, #274).
- The distillation columns now run on a cubic EOS (#214).
- Heater and Cooler now use a real enthalpy balance (#232).
- Gas network schematics and a worked model-updating example (#176).
- REE: a mass-action closure, saponification, a train graph and four analysis
  modules (#207).
- An "Ask" docs assistant in the Jupyter Book (#325), a refinery units summary
  page, and logos for the package and every plugin (#348, #350).

### Fixed

- Distillation: the feed-stage section convention is set in one place, the
  energy balance uses the feed's thermal condition q, and component balances
  close on the CMO path (#215, #217, #225). Fixed the Hengstebeck-Geddes non-key
  split in `ShortcutColumn` (#323).
- Flowsheet: a recycle starts from the stream being recycled, and a solve that
  did not converge now says so (#254, #256).
- CSTR: the molar density that sets residence time now has a name, and the
  water fallback raises `CSTRDensityWarning` (#234).
- REE: extraction loading, capacity, mechanism and activity models were
  corrected (#204). Extractant correlations were refit against named sources
  (#270, #282, #283, #290). The Kremser two-inlet boundary condition was fixed
  (#285). The capital estimate is now anchored to disclosed project costs
  (#272). Several smaller fixes (#275-#280, #287, #289).
- Solvers: sparsity is derived from the graph instead of defaulting to dense
  (#209).
- Fixed eight defects that adversarial testing found across numerics, streams,
  pytrees, serialization, economics, bio and carbon capture (#292).
- Fact-checked the example notebooks against the literature (#281, #291).

### Changed

- Physics validation (the `release` test marker) runs nightly and before every
  publish, instead of on every commit. CI shards the suite by measured test
  duration (#245, #304, #318, #347, #352, #353).
- The PyPI wheel check covers all seven packages.

## [0.1.0] - 2026-08-10

First public release.

[Unreleased]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jkitchin/differentiable-flowsheets/releases/tag/v0.1.0
