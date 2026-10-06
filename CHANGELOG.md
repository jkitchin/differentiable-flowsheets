# Changelog

All notable changes to difflow are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org) (pre-1.0: a minor bump may break the API).

## [Unreleased]

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
