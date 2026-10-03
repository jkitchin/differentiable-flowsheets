# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**difflow** is a JAX-based differentiable flowsheet framework for chemical process simulation. It enables automatic differentiation through chemical engineering unit operations for gradient-based optimization, sensitivity analysis, and technoeconomic modeling.

## Quick Start Commands

```bash
# Install in development mode
pip install -e ".[dev,examples,solvers]"

# Run tests (parallel; --dist loadfile keeps a module's JAX compilation
# cache in one worker, without which the workers just recompile it each)
make test                      # skips the `slow` marker
make test-all                  # everything
pytest tests/ -v -n auto --dist loadfile

# Run specific test file
pytest tests/test_cstr.py -v

# Run tests with coverage
pytest tests/ --cov=src/difflow

# Re-measure the per-test durations CI shards its three jobs on
make test-durations

# Measure the recycle solver's pass rate over the hard-flowsheet corpus
make convergence

# Build documentation (Jupyter Book); also writes the "Ask" assistant's index
make book
make ask-check                 # its retrieval check over the built book (node)

# Execute all example notebooks
make notebooks
```

## Repository Structure

```
difflow/
├── src/
│   ├── difflow/           # Core package
│   │   ├── streams.py     # Stream representation
│   │   ├── thermo.py      # Thermodynamics (ideal)
│   │   ├── eos.py         # Equations of State (PR, SRK)
│   │   ├── database.py    # Species property database
│   │   ├── flowsheet.py   # Flowsheet with recycle solving
│   │   ├── uncertainty.py # Sensitivity & UQ
│   │   ├── convergence.py # Hard-flowsheet corpus + pass-rate benchmark
│   │   ├── stochastic/    # Two-stage stochastic programming (SAA over scenarios)
│   │   ├── planning/      # Delta-base planning (LP/MILP + trust region)
│   │   ├── catalog.py     # Machine-readable schema of every unit operation
│   │   ├── docstrings.py  # Params field descriptions, read from the Attributes:
│   │   │                   # docstrings (what the catalog reports)
│   │   ├── serialize.py   # Flowsheet <-> JSON round trip
│   │   ├── codegen.py     # Flowsheet -> runnable Python source
│   │   ├── kinetics.py    # Declarative mass-action rate laws (data, not callables)
│   │   ├── publish.py     # Flowsheet -> self-contained interactive HTML (no install)
│   │   ├── gui/           # Local browser editor (python -m difflow.gui)
│   │   │                   # session.py + server.py + layout.py + static/
│   │   │                   # docs_index.py builds static/docs-index.json (committed);
│   │   │                   # doclinks.py resolves a unit -> its section in docs/
│   │   ├── params_mixin.py # ParamsMixin base class for Params dataclasses
│   │   ├── reconciliation/ # Data reconciliation, gross error detection,
│   │   │                   # observability, monitoring, multi-set pooling,
│   │   │                   # gated parameter tracking (the digital-twin loop)
│   │   ├── units/         # Steady-state unit operations
│   │   ├── dynamic/       # Dynamic modeling (DAE)
│   │   ├── economics/     # Technoeconomic analysis
│   │   └── visualization/ # Flowsheet visualization
│   ├── difflow_bio/       # Bio manufacturing plugin (bioreactors, filtration, chromatography)
│   ├── difflow_ree/       # Rare earth element solvent extraction plugin
│   ├── difflow_cc/        # Carbon capture plugin (amine, membrane, adsorption)
│   ├── difflow_gas/       # Gas transmission network plugin (pipes, compressors, computed decomposition)
│   └── difflow_refinery/  # Refinery plugin (crude assay, crude and vacuum distillation units, saturated gas plant, product blending pool,
│                          # hydroprocessing/ building blocks + hydrotreating/, hydrocracking/, fcc/, reforming/, alkylation/, residue/ units,
│                          # hydrogen/ header network)
├── tests/                 # pytest test files (includes tests/bio/, tests/ree/, tests/cc/, tests/gas/, tests/power/, tests/refinery/)
├── examples/              # Jupyter notebook examples
├── jax-tutorials/         # JAX/autodiff tutorials
├── docs/                  # Documentation (Markdown)
└── _ext/                  # Sphinx extension for the book's in-browser "Ask"
                           # assistant (BM25 over the rendered pages + optional
                           # WebLLM); see docs/ask.md
```

## Key Concepts

### 1. Streams
Streams are JAX-compatible data structures representing material flows:
```python
from difflow import Stream, create_experiment_stream

# Create a stream
stream = create_experiment_stream(
    conditions={'T': 350.0, 'P': 101325.0},
    species=['A', 'B'],
    molar_flows=[1.0, 0.5]
)
```

### 2. Unit Operations and Params Classes
All units are differentiable and use Params dataclasses that inherit from `ParamsMixin`:
```python
from difflow import CSTR, CSTRParams
import jax.numpy as jnp

# Define rate function: A -> B, r = k*C_A
def rate_fn(concentrations, T, params):
    k = params['k'] * jnp.exp(-params['Ea'] / (8.314 * T))
    return k * concentrations['A']

# Create params with dict-like access via ParamsMixin
params = CSTRParams(
    V=1.0,  # Reactor volume (m^3)
    rate_fn=rate_fn,
    stoich={'A': -1, 'B': 1},
    molar_density=55500.0,  # concentration basis; see note below
)

# Create and run CSTR
cstr = CSTR(params)
outlet = cstr(inlet_stream)

# The rate law needs a molar density: C_i = F_i / Q_v with Q_v = F_total/rho,
# so rho sets the residence time tau = V*rho/F_total. Give it as
# molar_density=..., or as eos=<cubic EOS> + reaction_phase='liquid'|'vapor'
# for the real EOS density at reactor conditions (a CubicThermo passed as the
# CSTR's thermo supplies the EOS too, once reaction_phase names the phase).
# With neither, the CSTR falls back to 55500 mol/m^3 -- liquid water -- and
# raises CSTRDensityWarning rather than doing it silently.

# Params support dict-like access
print(params['V'])        # -> 1.0
print('V' in params)      # -> True
new_params = params.update(V=2.0)  # Functional update (JAX-compatible)
```

### 3. ParamsMixin Pattern
All `Params` dataclasses should inherit from `ParamsMixin` for consistent API:
```python
from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin

@dataclass
class MyUnitParams(ParamsMixin):
    """Parameters for MyUnit.

    Attributes:
        temperature: Operating temperature (K)
        pressure: Operating pressure (Pa)
    """
    temperature: float
    pressure: float

# The Attributes: section is not just prose: difflow.docstrings reads it, so
# `describe_operation(...).parameters` reports each field's description. A field
# documented only by the comment beside it is read too. Units are separate --
# declare them in the unit class's `parameter_units`.
#
# That pathway reads the source from disk (inspect.getsource), so descriptions
# go quiet -- silently, to None -- in a zipimport or frozen build where no .py
# is on disk. A normal wheel or editable install is fine.

# ParamsMixin provides:
# - params['key'] - dict-style access
# - params.update(key=value) - JAX-compatible functional updates
# - params.keys(), .values(), .items() - dict-like iteration
# - 'key' in params - membership testing
# - Concise __repr__ with JAX array formatting
```

### 4. Automatic Differentiation
Use JAX's `grad`, `jacobian`, `jit` with any difflow function:
```python
import jax
from jax import grad, jit

def conversion(volume):
    params = CSTRParams(V=volume, rate_fn=rate_fn, stoich=stoich)
    cstr = CSTR(params)
    outlet = cstr(inlet)
    return outlet.molar_flows['B'] / inlet.molar_flows['A']

# Gradient of conversion w.r.t. volume
d_conv_d_V = grad(conversion)(1.0)

# JIT compile for speed
fast_conversion = jit(conversion)
```

### 5. Flowsheets with Recycles
```python
from difflow import Flowsheet

fs = Flowsheet()
fs.add_unit('cstr', cstr)
fs.add_unit('flash', flash)
fs.connect('cstr', 'flash')
fs.set_recycle('flash', 'cstr', split_fraction=0.5)

result = fs.solve(feed_stream)
```

Tear placement is the user's call by default (`add_recycle`), because
inferring it silently would change the answer for flowsheets that already
run. `fs.tear_analysis()` reports the loops, the declared tears, what
`select_tear_streams` would pick instead, and any inlet the unit order
cannot supply -- reading only, no unit is called. `fs.solve(tears="auto")`
(or `"minimum"`) picks the tears when none were declared, sequences the
units around them with `calculation_order`, records the choice in
`last_solve_tear_streams`, and leaves `fs.recycles`/`fs.units` as declared.

## Code Conventions

### JAX Compatibility
- All numerical operations use `jax.numpy` (imported as `jnp`)
- Use `@jit` decorator for performance-critical functions
- Avoid in-place operations; use functional updates: `x = x.at[i].set(v)`
- Register custom classes as PyTrees if they contain arrays

### Type Hints
- Use type hints for public APIs
- Common types: `Array` (jax array), `Scalar` (float), `Dict[str, Array]`

### Testing
- Tests use pytest
- Each module has corresponding `test_*.py`
- Test both forward pass and gradients where applicable
- Use `jax.test_util.check_grads()` for gradient verification

### Documentation
- Docstrings follow NumPy style
- Include Args, Returns, and Example sections
- Example notebooks in `examples/` demonstrate usage

## Common Development Tasks

### Adding a New Unit Operation

1. Create file in `src/difflow/units/` (steady-state) or `src/difflow/dynamic/` (dynamic)
2. Inherit from appropriate base class
3. Implement `__call__` method that takes inlet stream(s) and returns outlet stream(s)
4. Ensure all operations are JAX-compatible (use `jnp`, no Python loops over arrays)
5. Add tests in `tests/test_<unit>.py`
6. Add example usage in `examples/`
7. Document it in the right `docs/unit-operations-*.md`, then rerun
   `python3 src/difflow/gui/docs_index.py` (or `make gui-build`) so the
   committed index sees it

A registered operation must have **somewhere of its own to link to**:
either a heading that names it (at `###` or shallower --- the book only
generates heading anchors down to `myst_heading_anchors`), or, for a unit
documented as one row of a reference table, an explicit MyST label
`(op-<lowercase name>)=` before the row (page-prefixed where the bare
name is not unique across the book, as the gas and power plugins do:
`(gas-op-gaspipe)=`). `difflow.gui.doclinks.url_for` is what resolves it
and `tests/test_doclinks.py` asserts all 89 operations resolve, so
skipping step 7 fails the suite rather than shipping a palette entry with
nothing to read.

### Adding to a Plugin (bio, ree, cc, gas, power, refinery)

The project has six domain-specific plugins:
- **difflow_bio**: Bio manufacturing (bioreactors, filtration, chromatography)
- **difflow_ree**: Rare earth element solvent extraction
- **difflow_cc**: Carbon capture (amine absorption, membrane, adsorption)
- **difflow_gas**: Gas transmission networks (pipes, compressors, valves, topology-driven sequential decomposition)
- **difflow_power**: Electrical grids (AC power flow, AC-OPF, DC-OPF, PTDF/LODF, state estimation)
- **difflow_refinery**: Petroleum refining (TBP assay characterisation, crude and vacuum distillation units, saturated gas plant, product blending)

1. Add to appropriate plugin directory (`src/difflow_bio/`, `src/difflow_ree/`, `src/difflow_cc/`, `src/difflow_gas/`, `src/difflow_power/`, or `src/difflow_refinery/`)
2. Create a Params dataclass inheriting from `ParamsMixin`
3. Export in plugin's `__init__.py` and add to `__all__`
4. Add tests in `tests/bio/`, `tests/ree/`, `tests/cc/`, `tests/gas/`, `tests/power/`, or `tests/refinery/`
5. Register in the plugin's `register()` function for plugin discovery
6. Add documentation in `docs/unit-operations-*.md`

Example plugin unit:
```python
from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin

@dataclass
class MyUnitParams(ParamsMixin):
    """Parameters for MyUnit."""
    param1: float
    param2: float = 1.0  # With default

class MyUnit:
    """Description of the unit operation."""

    def __init__(self, params: MyUnitParams):
        self.params = params

    def __call__(self, inlet_stream):
        # Process inlet stream
        return outlet_stream
```

### Plugin Overview

**difflow_bio** - Bio manufacturing:
- Bioreactors: `ContinuousBioreactor`, `FedBatchBioreactor`
- Separation: `Centrifuge`, `DiscStackCentrifuge`
- Filtration: `Ultrafiltration`, `Diafiltration`, `TFF`
- Chromatography: `ProteinAChromatography`, `IonExchangeChromatography`, `SizeExclusionChromatography`

**difflow_ree** - Rare earth element extraction:
- Unit operations: `REEExtractor`, `REEMixerSettler`, `REEScrubber`, `REEStripper`
- Precipitation: `OxalatePrecipitator`, `CarbonatePrecipitator`, `HydroxidePrecipitator`
- Flowsheets: `ExtractStripCircuit`, `ExtractScrubStripCircuit`, `SplitShellCascade`, `FullSeparationTrain`
- Database: 15 REE elements (the 14 stable lanthanides -- no Pm -- plus Y), 5 extractant systems (D2EHPA, PC88A,
  Cyanex 272, TBP, naphthenic acid). Coverage is UNEVEN and `ext_db.coverage()`
  reports it: only naphthenic_acid has coefficients for all fifteen; the other
  four cover ten (no Ho, Er, Tm, Yb, Lu).
- Free extractant (#267): the correlation's `[HA]` is FREE (Q1 Eq. 2.88/2.89),
  not total. `solve_free_extractant(dist, el, c_aq, pH=...)` closes
  `c_org = D([HA]_free) c_aq`, `[HA]_free = [HA]_0 - m c_org` as a monotone
  scalar root through optimistix (implicit diff, so gradients survive). Total
  overpredicts D where the cascade works hardest -- 1.34x at the naphthenic
  anchor. `check_loading_capacity` / `implied_loading_fraction` reject a
  loading past `1/monomers_per_ree`, which a total-basis correlation returns a
  finite D for. Do NOT compose with `LoadingIsotherm.apparent_D`: that caps
  the answer, this changes the input (#190/#204's double count).
- Langmuir constants are DERIVED (#268): `typical_K_L` was a second extractant
  table hand-synced with the YAML, and three of four entries matched the
  coefficients at NO pH (rms log10 residual 0.62/0.85/0.92 at best fit). Now
  `K_L = D(reference)/q_max` computed on access; `EXTRACTANT_CAPACITIES` is a
  derived Mapping, not a dict. Every record declares its basis:
  `reference_concentration` plus `reference_pH` (cation exchange) or
  `reference_nitrate` (solvating). A missing extractant was already tested for;
  a STALE one was not, which is why it drifted silently. `naphthenic_acid` is
  the check that the derivation is right: at its declared `reference_pH` of 4.5
  it reproduces all fifteen of the deleted literals to four figures, which the
  other three could not be made to do at any pH. The memo in front of it is
  keyed on a FINGERPRINT of the record, never on the extractant name --
  `add_element_to_extractant` mutates a record IN PLACE, so identity and
  equality both say "unchanged" while the basis of every derived constant has
  moved, and a name-keyed memo puts the staleness back in memory where no
  drifted number in a file gives it away.
- Uncertain D: `REEDistribution(..., coefficient_overrides={"Nd": {"a": ...}})`
  replaces tabulated log10(D) correlation coefficients, and accepts JAX tracers,
  so a distribution can be put on D and differentiated through. Passed through by
  `REEExtractorParams`, `MixerSettlerParams`, `ScrubberParams`, `StripperParams`.
  `n_stages` is likewise a continuous, traceable decision (Kremser is `E**(N+1)`)

**difflow_cc** - Carbon capture:
- Amine absorption: `AmineAbsorber`, `AmineStripper` (MEA, DEA, MDEA, PZ, AMP)
- Membrane: `MembraneSeparator`, `MultistageMembrane` (9 membrane materials)
- Adsorption: `PSAUnit`, `TSAUnit`, `VSAUnit`, `TVSAUnit` (8 adsorbent materials)
- Direct air capture: `SolidSorbentDAC`, `LiquidSolventDAC`
- Heat integration: `LeanRichExchanger`, `HeatRecoverySystem`
- CO2 compression: `CompressionTrain`, `Pump`
- Economics: CAPEX/OPEX estimation, levelized cost of capture
- Degradation: Amine oxidation, adsorbent capacity fade, membrane aging

**difflow_gas** - Gas transmission networks:
- Network model: `GasNetwork` (pipes, compressor stations, valves, control valves, resistors, short pipes; signed flows)
- Decomposition: `decompose` computes the spanning tree, tear set and balance schedule from the topology
- Units: `GasPipe`, `BackPipe`, `PipePressure`, `PressureDrivenPipe`, `Compressor`, `CompressorBoost`, `OpenValve`, `PressureEqual`, `ControlValveDrop`, `SourceHead`, `AffineFlow`, `Junction`, splits
- Flowsheets: `GasNetworkFlowsheet` (signed-flow Anderson + damped differentiable tear solve), `build_network_flowsheet`
- Physics: `weymouth_beta`, `resistor_xi`, `compressor_power`, `smoothed_power_w`, GasLib unit conversions
- Verification: full equation-oriented residual checks (`difflow_gas.verify`)
- Equations: `difflow_gas.residuals.network_residuals` is the single JAX-traceable definition of the equation set; `verify` is the reporting layer over it
- Plotting: `dg.draw_network(net, pos=..., pressures=..., flows=..., highlight=...)` draws a network schematic
- Reconciliation: `reconcile_network`, `monitor_network` (a campaign against a fixed
  model), `reconcile_network_multi` (pool periods sharing a parameter); all three
  fill in the layout's names and scales (see `difflow.reconciliation`)
- Gotchas encoded in docs: solve with `clip_negative_flows=False` (signed flows), damp the tear map (alpha ~ 0.3), pose optimization pressure constraints in squared pressure

**difflow_power** - Electrical grids:
- Network model: `PowerNetwork` (buses, branches, generators, loads); one branch
  model serves lines, transformers and phase shifters
  (`Yff = (ys + jb/2)/tau^2`, `Yft = -ys/conj(t)`, tap at the FROM end)
- Equations: `difflow_power.residuals.power_flow_residuals` is the single
  JAX-traceable definition of the equation set (2 balance rows per bus plus one
  angle-reference row); `powerflow`, `opf`, `estimation` and `verify` are all
  consumers of it and restate no physics
- Power flow: `solve_power_flow` (Newton via optimistix, implicit-diff gradients);
  the bus-type specification is written as `2 n_gen - 1` EQUATIONS, not by
  eliminating variables
- AC-OPF: `solve_acopf` over `difflow_power.ipm`, a primal-dual interior-point
  NLP solver written in JAX (no IPOPT -- that would end the differentiability).
  LMPs from the equality multipliers; `check_prices()` verifies them against
  `jax.grad` of the optimal cost through the KKT system
- DC: `solve_dcopf`, `ptdf`, `lodf`, `contingency_flows` (same IPM; a QP is an
  NLP with a constant Hessian, so DC and AC prices are comparable)
- Feeders: `RadialFeederFlowsheet` (backward/forward sweep, the voltage profile
  as the tear), `build_ladder_flowsheet` (a real `difflow.Flowsheet`)
- Sensitivity: `loss_sensitivity`, `branch_flow_sensitivity`,
  `demand_sensitivity`, `parameter_sensitivity`, `voltage_stability_margin`
- Estimation: `estimate_state` over `difflow.reconciliation` (state estimation
  and data reconciliation are the same computation)
- Cases: `case3`, `case5` (PJM), `case9` (WSCC), `case14` (IEEE),
  `radial_feeder`, plus `from_matpower`

Invariants encoded in the plugin (do not weaken them):
- Thermal limits are posed on `|S|^2`, never `|S|`: the modulus has curvature
  going as `1/|S|` and early interior-point iterates sit on lightly loaded
  branches.
- The angle-reference row stays in the residual set. Without it the Jacobian is
  one rank short for structural reasons and everything that inverts it fails.
- Limits are NOT equations. Voltage/generator/thermal/angle bounds live in
  `opf.py`, not in `residuals.py`.
- The IPM's inertia test runs on the RUIZ-EQUILIBRATED KKT matrix and uses a
  band (`n_neg <= m_eq <= n_nonpos`), because `Sigma = z/s` spans 12 decades
  near the solution and a weakly convex problem has genuine zero eigenvalues.
- `mu` follows IPOPT's monotone schedule judged on the subproblem, and its floor
  is `tol_comp / (10 m_in)`. Tying `mu` to `s.z` deadlocks on a degenerate
  problem; a floor of `tol_comp` makes the tolerance unreachable.
- Every benchmark number is asserted against MATPOWER's published answer. A
  self-consistent implementation with the phase-shift sign backwards converges
  beautifully to the wrong result.
- Bus/branch/generator order is INSERTION order, not sorted: numeric labels
  sorted as strings interleave "10" between "1" and "2".
- Out of scope by design: unit commitment (integer), security-constrained OPF,
  dynamics/transient stability, unbalanced three-phase. Do not add them.

Docs: `docs/unit-operations-power.md`. Tests: `tests/power/`.

**difflow_refinery** - Petroleum refining:
- Assay: `Assay` (TBP curve, SG or SG curve, light ends) -> `characterize` ->
  pseudo-components (Twu / Riazi-Daubert / Lee-Kesler critical properties)
- Thermo: `ColumnThermo` -- vectorised, Raoult + Lee-Kesler Psat, ideal-gas-path
  enthalpy; water is steam or free water in the drum, never in the HC liquid
- Column: `CrudeColumn` -- EO MESH (Naphtali-Sandholm), side strippers,
  pumparounds, steam; specs from `column.*` builders, one per degree of freedom
- `Furnace`: solved WITH the column (coil outlet T is an unknown), so an
  `overflash` spec sets the furnace. A spec set with no solution (e.g. too little
  overflash for a big pumparound) returns `converged=False`, not an answer
- `CrudeUnit` (assay in, yield table out) and `CrudeDistillationUnit` (the
  registered operation; outlets in `outlet_names` order)
- Products: `product_properties`, `products.gaps` -- TBP, not D86
- Alkylation (#310, `difflow_refinery.alkylation`, a library): Sauer-Colville-
  Burwick correlations transcribed from GAMS `process.gms` (`solve_process_gms`
  reproduces its 1161.3366 optimum); reactor = per-olefin stoichiometry whose
  heavy-end split is set by the correlation's yield; DIB overhead is the
  `Flowsheet` tear. Columns are `KeySplitColumn` shortcuts (Lee-Kesler
  Raoult), NOT difflow's PR `ShortcutColumn`: AD through the recycle with that
  one runs out of memory (its Geddes non-key split is fixed in the base class;
  `GeddesShortcutColumn` is now an alias). The temperature/space-velocity octane terms and the
  selectivities are illustrative, not sourced. FCC (#308) `c3`/`c4` outlets
  feed it via `combine_feeds`.
- Composition (#305, `difflow_refinery.composition`): `characterize(assay,
  composition=True)` / `char.with_composition(CompositionData(...))` puts a
  `Composition` on `char.composition` -- per component, `char.names` order:
  `hc_type` (n,4) vol fractions of `HC_TYPES` (P, N, A, O; rows sum to 1,
  O = 0 straight run), `hydrogen` (n,) mass fraction, `sulfur`/`nitrogen`
  (n,) mass fractions with `sulfur_split` (n,5) over `SULFUR_CLASSES` and
  `nitrogen_split` (n,2) over `NITROGEN_CLASSES`. `comp.of_flows(moles)` /
  `of_stream(s)` -> `StreamComposition` (types by std volume, H/S/N by mass);
  `product_properties(..., composition=comp)`; `BlendCharacterization.
  from_characterization` picks the types up as `*_vol` qualities (vol%).
  A conversion unit edits a copy with `comp.replace(...)`.
  Invariants: C/H for the type correlation is DERIVED from the hydrogen
  (`(1-H-S-N)/H`), so H and PNA are one estimate; correlation branch joins
  (Huang index at 620 K, 2B4.1 at MW 200) are logistic BLENDS because the
  published branches disagree there -- never hard switches; measured
  PIONA/SARA/n20/H override per cut via `jnp.where` (differentiable); the
  default S/N class splits are ILLUSTRATIVE; out-of-range use warns
  (`CompositionRangeWarning`), never clips the answer. MNL50 worked examples
  for these correlations are NOT reproduced -- do not claim they are.

Docs: `docs/unit-operations-refinery.md`. Tests: `tests/refinery/`.

**One characterization (#301).** `dr.Assay(..., heavy_end=dr.HeavyEnd(),
sulfur_wt=..., ccr_wt=...)` -> `dr.characterize` is read by all three
consumers: the CDU, the VDU (`VacuumColumnParams(components=
char.pseudo_components())`) and the blend pool
(`BlendCharacterization.from_characterization(char)`). The CDU's `"residue"`
feeds the VDU in a `Flowsheet` with no re-cut (rename the VDU's outlets:
`"residue"` is taken). Correlations live once, in
`difflow_refinery.correlations`; `vacuum.correlations` and the
`characterization` helpers are re-export shims.
- Opt-in, and pinned: an assay without a heavy end characterizes bit for
  bit as before UNDER `method="twu_legacy"` (`tests/refinery/test_cdu_baseline.py`).
  The default `"twu"` is Twu (1984) as published (alias `"twu_1984"`);
  `"twu_legacy"` is the old crude-unit MW coding (Rankine constants /
  sqrt(1.8), aromatics up to 15% low), kept only to reproduce old numbers.
  Do not make it the default again.
- Water rides under TWO keys: CDU water/steam is `F_water` (WATER_MW
  18.015), VDU steam is `F_H2O` (MW_WATER 18.01528). A flowsheet species list
  is `char.names + ["water", "H2O"]`; a total balance counts each at its own
  molar mass.
- `CrudeColumn` solves a column with a near-involatile component (psat at
  600 K < 0.05 Pa, e.g. the 950 C lump) by a volatility continuation: first
  with those psats floored, then the true ones from there. Without it Newton
  stalls from the bubble-point start.
- `VacuumColumn` sends feed species it does not model (light ends, CDU
  water) out the overhead, so the flowsheet balance closes.

**difflow_refinery.vacuum** - the vacuum unit (stage-network column) and a
compatibility view of the shared characterization:
- Assays: `Assay` (Celsius, wt%; `to_assay()` converts), `characterize`
  (the shared characterization on a 300-800 C grid + a residue lump, returned
  in the old shape), `light_crude`/`heavy_crude` (synthetic),
  `atmospheric_residue` (an idealized CDU cut, for running without a CDU)
- Correlations: names re-exported from `difflow_refinery.correlations` --
  Twu (Tc, Pc, MW), Kesler-Lee (omega, liquid Cp), Maxwell-Bonnell psat (the
  D1160 conversion), Riazi-Daubert/Lee-Kesler for comparison, `fit_antoine`
- Column: `StageColumn` over a `ColumnLayout` of `Route`s, `StageSpec` trades a
  knob or draw rate for a target on any output; `VacuumColumn` builds the VDU

Invariants encoded in the vacuum stack (do not weaken them):
- Component balances are in LOG form (`ln(l+v) - logsumexp(ln ins)`). The
  residue lump in the top stage is 1e-200 of its feed; in linear form its
  rows underflow to zero and the Jacobian is singular (cond 1.7e17).
- Trace log flows (< 1e-7 of their feed) are exempt from the Newton step
  cap and clipped on their own; in the cap, one of them scales every step
  to nothing.
- The TBP curve is C1 (monotone Hermite in `Phi^-1(x)`). The default cut
  grid puts a cut boundary on every assay point; a piecewise-linear curve
  kinks there and the TBP-point gradient had two values (1e-3 off FD).
- The heat of vaporization is the Clausius-Clapeyron slope of the SAME
  Maxwell-Bonnell psat that sets K, so energy and VLE agree on volatility.
- Twu has no root past ~840 C TBP: the residue lump's Tb, SG, MW are SET,
  never correlated.
- Draws are softmax SHARES of stage liquid, never raw rates, so a rate spec
  can never overdraw a stage. Specs replace one knob or rate equation each,
  so degrees of freedom always balance.
- The implicit step reuses the Jacobian the Newton loop evaluated at the
  converged point; do not trace a second Jacobian for it.
- A non-converging solve is usually an infeasible spec (an LVGO end point a
  low-efficiency HVGO bed cannot make), not a solver failure.

Tests: `tests/refinery/test_vacuum*.py`, `test_heavy_end.py`, `test_one_characterization.py`.
Examples: `examples/34_vacuum_distillation.ipynb`, `examples/36_crude_to_vacuum.ipynb`.

### Catalytic Reforming (`difflow_refinery.reforming`, #309)

`CatalyticReformer(ReformerParams()).solve(NaphthaFeed)` -- semi-regen train
(adiabatic beds in diffrax, fired heaters, PR separator, H2 recycle as a
`Flowsheet` tear, component-split stabilizer). Library, not a palette op.
- Lumps C6-C10 nP/iP/N/A (+ MCP, H2, C1-C5), each ONE model compound
  (`species.SPECIES`); thermo from `chemicals` 1.5.2 tables (API TDB Hf, Yaws
  S0), K from Gibbs energies -- never from a kinetic paper.
- One enthalpy basis: Hf + `CubicThermo` (ideal-gas Cp fit + PR departure).
  Reactor T comes from the conserved enthalpy by Newton, so balances close
  to round-off at any ODE tolerance. Keep it that way.
- Beds use `diffrax.ForwardMode`: differentiate a reformer with `jax.jacfwd`
  (the recycle's implicit fixed point needs JVPs), warm-start with
  `tear_initial=res.tear`.
- Kinetic pre-exponentials are ILLUSTRATIVE (this project's); octanes other
  than n-heptane are recalled (unverified). Feed sulfur (#330,
  `reforming.sulfur`) is a TRACE element solved on the converged streams,
  outside the species list and the tear: H2S to net/fuel gas, unconverted S
  to reformate, balance exact. No Padmavathi/Taskar-Riggs
  cross-check is claimed. Separator/recycle are thin and local, to merge with
  #306's shared module later.

Docs: `docs/unit-operations-refinery.md` (Catalytic reforming). Tests:
`tests/refinery/test_reforming.py` (flowsheet tests `slow`).
### Hydroprocessing and the Hydrotreater (`difflow_refinery.hydroprocessing`, `.hydrotreating`)

`hydroprocessing/` is kinetics-agnostic and shared (the hydrocracker of #307
is meant to reuse it): `layout` (`Layout`/`Flows`: gases, cuts, and per-cut
ATTRIBUTE flows `F_<cut>@<attr>` -- C and H atoms compulsory, a cut's mass is
computed from its atoms), `thermo` (vectorised PR on traceable cut constants),
`separator` (`pr_flash`, negative flash, `HPSeparator`), `reactor`
(`TrickleBedReactor` around any object with `attributes`,
`attribute_elements` and `rates(ctx: ReactionContext, params) -> Rates`),
`recycle` (knock-out, amine, purge, ideal-gas compressor, makeup, `solve_tear`),
`stripper` (`StageColumn` steam stripper + PR overhead drum), `solve`
(`newton_solve`). `hydrotreating/` adds `HDTKinetics` (HDS by sulfur class,
LHHW; HDN; reversible aromatics; olefins; cracking leak), `Hydrotreater`,
`hdt_block`. A library, not a palette operation.

Invariants (do not weaken them):
- Attributes are extensive and follow their cut through every split; element
  balances (C, H, S, N) and mass close to round-off -- tested at 1e-8.
- A diffrax solve with `RecursiveCheckpointAdjoint` is reverse-mode only.
  Newton loops around the reactor (tear, quench, targets) use
  `hydroprocessing.solve.newton_solve`: Jacobian from a FORWARD-adjoint copy
  (`f_iter`, `jac="fwd"`), then one implicit step with the reverse-mode
  residual. Residuals must be pure in `(x, args)` -- no traced closures.
  Reverse-mode Jacobians through the checkpointed adjoint compiled for 6 min.
- Root polishing (Rachford-Rice, the PR cubic) takes TWO Newton steps on live
  inputs: the reactor differentiates derivatives (`C_eff = dH/dT`,
  `d ln K/dT`); one step left the second derivatives wrong and the loop's
  implicit gradient was 1-80 % off (high loop gain amplifies it).
- `Flows.per_molecule` uses a safe inverse (`F/(F^2+eps^2)`): `attr/max(F,
  1e-300)` gives a 1e300 cotangent for an empty cut and NaN gradients.
- The stripper feeds 10 % of its steam with the feed: a degassed separator
  liquid is subcooled and `StageColumn`'s feed flash is otherwise singular.
- Outside the two-phase region `reactor.phase_state` returns the stream itself
  and its incipient phase (`y = z, x = z/K` for a vapour), never the negative
  flash's fictitious split: that put a vapour naphtha bed's pH2 6x low and ran
  the aromatics equilibrium backwards (#332).
- `pr_flash` never raises: the Rachford-Rice bracket ignores absent (trace)
  species, a diverged Newton returns its start with a large `residual`, and
  the implicit derivative is the module's own `custom_jvp` (optimistix's
  implicit adjoint raises on a NaN Jacobian). Callers fold the residual into
  `converged` (the hydrotreater's `flash.residual`).
- `Hydrotreater.product_stream()` includes the dissolved real gases by
  default (mass closes downstream, #333); blend the wild naphtha with
  `gases=False` or through `res.fractionate(...)` (#328, a TBP sigmoid split).
  `NAPHTHA_HDT_PARAMS` is the illustrative naphtha constant set.
- Rate constants are ILLUSTRATIVE; thermochemistry is model-compound data from
  the `chemicals` tables. The Korsten-Hoffmann profile cross-check is NOT done.

Docs: `docs/unit-operations-refinery.md` (Hydroprocessing building blocks; The
hydrotreater). Tests: `tests/refinery/test_hydrotreating.py`.

### The Hydrocracker (`difflow_refinery.hydrocracking`, #307)

The same building blocks: pretreat bed (`HDTKinetics` + `VGO_PRETREAT_PARAMS`)
-> cracking bed (`HCKinetics`, a new kinetic model: continuous lumping on the
cut grid after Laxminarasimhan et al. 1996, or discrete lumps; organic-N
inhibition; H2 from the H balance of each event; heat per H2) -> HPS and
recycle gas -> simplified fractionator (smooth TBP split, NOT a column) -> UCO
recycle. Layout attributes are `HDT_ATTRIBUTES + ("cracked",)`. `hcu_block`
for planning. A library, not a palette operation.

Invariants (do not weaken them):
- The Laxminarasimhan forms are UNVERIFIED against the paper (not reachable);
  no equation numbers are given and its parameters are not used. The
  yield-vs-conversion cross-check is NOT done. Constants are illustrative.
- Cracking quench is held to bed-inlet temperature (`quench_crack=None`); its
  total share is one more unknown of the gas tear (no inner Newton). Fixed
  quench rates are a knife-edge (runaway or die-out within a few K).
- The UCO tear (~200 unknowns) is Anderson substitution with a GMRES adjoint
  (`hydrocracking.fixed_point`), never Newton; its test is relative (the bed
  integration's rtol is the noise floor). Balances add nothing for the
  recycle, so an unconverged tear shows in them.
- Recycle at fixed catalyst and T LOWERS per-pass conversion (a recycle
  reactor is less efficient than plug flow); what it buys is selectivity.
- Full-unit tests compile 2-6 min each: slow.
### The Hydrogen Network (`difflow_refinery.hydrogen`, #329)

`HydrogenNetwork(producers, consumers, headers).solve()` -> `H2NetworkResult`
(`outputs["h2.surplus"]`, `<consumer>.purity`, `<consumer>.purity_margin`,
`balances`, `makeup_composition(c, gases)`). `Producer.from_reformer(res)` /
`.of_purity`, `Consumer.from_hydrotreater(name, res, params)`, `PSA(recovery,
purity, target_purity=)`, swing `Import`/`H2Plant` (filled in order, capped),
`Header(min_purge=, purge_to="fuel"|"export")`. `close_hydrotreater_loop(net,
{c: (Hydrotreater, feed, params)})` feeds the header composition into
`HydrotreaterParams.makeup` by substitution on purity; `h2_block` for planning.
A library, not a palette operation.
- Every consumer on a header gets the header's purity; a consumer's demand is
  its makeup H2 FLOW (`h2.makeup`), the impurities ride along at `d/y`.
- The purge is by difference, so total/H2/mass balances close by
  construction; `balances["makeup_h2"]` is the independent check. A deficit
  is returned as a negative surplus, never clipped (`feasible` says so).
- `HydrotreaterParams.makeup` is concrete (`makeup_vector` calls `float()`):
  the loop is Python, and the returned network carries each unit's purity
  response as a LINEAR secant (`d_demand_d_purity`). Reformer gradients go
  through it in forward mode (`jax.jacfwd`).
- `min_pH2` is the MAKEUP's `y P`, not the reactor-inlet pH2 (that is the
  HDT's `reactor.pH2_in`). PSA defaults are illustrative.

Docs: `docs/unit-operations-refinery.md` ("The hydrogen network"). Tests:
`tests/refinery/test_hydrogen.py`, `tests/refinery/test_hydrogen_loop.py` (slow).
### Residue Desulfurizer and Fuel Oil (`difflow_refinery.residue`, #331)

`ResidueDesulfurizer(char, residue).solve(residue)` -> `RDSResult`; fuel oil is
`fuel_oil_blend([res.blend_component("residue"), ...], [res.volume("residue"), ...])`
(a property-mode `BlendPool`, `VLSFO_SPECS`: 0.5 wt% S, 380 cSt, SG 0.991, CCR 18).
`RDSKinetics` = `HDTKinetics` with residue constants + refractory `S_residue`,
`NiV` (HDM onto the catalyst), `CCR` reduction, 538 C+ conversion. Once-through
treat gas, ideal product split (gas / distillate / residue). A library.
- Route (a) chosen over VDU + cutters on the lever rule
  (`cutter_fraction_for_sulfur`): a 3.3 wt% residue needs 85 % ULSD by mass to
  reach 0.5 wt%. Keep that argument in the docs if the route changes.
- `NiV` and `CCR` attributes have NO element (metals outside a cut's mass, CCR a
  subset of C); `S_residue` counts S. Balances incl. Ni+V (with the deposit) close
  to round-off -- tested at 1e-10, keep it that way.
- Conversion moves ALL of a parent's atoms to `m = nC_i/nC_j` lighter molecules
  with `m - 1` H2; the HDT cracking leak is off (`crack_k=0`) so nothing double counts.
- Constants and the refractory-S share table are ILLUSTRATIVE (ARDS ranges, pinned
  by release tests); R1-R5 references are unverified.

Docs: `docs/unit-operations-refinery.md` ("Residue desulfurization and fuel oil").
Tests: `tests/refinery/test_residue.py` (gradient and CDU route: release + slow).

### Crude Preheat Train (`difflow_refinery.preheat`)

`PreheatedCrudeUnit(assay, column, train)` puts a heat-exchanger train,
`Desalter` and `PreflashDrum` in front of the atmospheric column and solves
the coupled problem: an outer Newton on the tear (pumparound return
temperatures, drum temperature, furnace inlet temperature) around the train's
own Newton and the EO column, steps clipped to 30 K, implicit gradients.
`CrudeUnitWithPreheat` is the flowsheet operation. Fouling is a per-exchanger
`Rf`; `fouling_sensitivity` and `cleaning_ranking` give d(fired duty)/dRf and
the exchangers ranked by what cleaning them saves.

Invariants (do not weaken them):
- A pumparound that runs through the train is specified by its rate and its
  RETURN TEMPERATURE; the spec's value is only the loop's starting guess, the
  train sets the answer.
- The LMTD uses `abs()` and a `MIN_DELTA_T` floor (1e-6 K), so a temperature
  cross is NOT prevented -- an undersized hot stream on a large area pinches
  and the answer is the floor, not an error. Check the approach temperatures.
- Free water in the drum is a third phase (all water to vapour or liquid
  water, never dissolved in the oil).
- The Ebert-Panchal fouling constants are illustrative, not fitted.

Validation: `tests/refinery/test_preheat_validation.py` (release) against IDAES
`Flash` and `HeatExchanger` on the same ideal thermo -- an independent
implementation, not an independent model. Tests: `tests/refinery/test_preheat*.py`.
Example: `examples/37_crude_preheat_train.ipynb`.

**difflow_refinery.gasplant** - the saturated gas plant (#312), on a cubic EOS
(PR default, SRK) because Raoult is tens of percent off in K at 10-20 bar:
- `gas_components(light, pseudo=, kij=, cuts=)`: real light ends + naphtha
  pseudocomponents in one table; `cuts=` keeps only the named cuts
- `GasPlantColumn` (the vacuum `StageColumn` machinery on `CubicThermo`:
  total/partial/no condenser, reboiler, any feeds, side draws); factories
  `absorber_deethanizer`, `debutanizer`, `splitter`, `c3c4_splitter`,
  `deisobutanizer` (general enough for the naphtha splitter, #311)
- `GasCompressor` (stages + knock-out condensate), `AmineTreater` (a removal
  FRACTION -- treating chemistry is out of scope), `fuel_gas`, `lpg_quality`
  (GPA 2140, limits marked verify), `reid_vapor_pressure`, `gasplant_block`

Invariants encoded in the gas plant (do not weaken them):
- Three passes: easy specs on equilibrium stages, continuation of targets AND
  Murphree efficiencies, one implicit step. Pass 1's boilup ratio is at least
  one: from the guess's 5 %-of-feed vapor floor a heavy lean oil gets a few
  percent, and pass 1 never converges (the CDU-naphtha absorber of example 38).
- O'Connell is the factories' default tray efficiency; `tray_efficiency=1.0`
  gives theoretical stages. The IDAES cross-check uses 1.0 on both sides.
- RVP is the D323 construction on the same EOS, never a correlation.
- A non-converging solve is usually an infeasible spec: a C2- spec larger than
  the C2- fed, an RVP above what a hot feed allows, an olefin-limited iC4 purity.
- The IDAES reference (`tests/refinery/reference/gasplant_reference.json`) is
  an independent IMPLEMENTATION of the same model (PR, kij 0, same constants),
  not an independent model. Regenerate it, never loosen the staleness checks.
  The debutanizer is a full `TrayColumn` comparison (agreement 1e-7, but only
  after the generator tightens SmoothVLE's eps: at IDAES's defaults the total
  condenser leaks 0.07 % of a component). IDAES's TrayColumn does not converge
  the C3/C4 splitter, so that case is IDAES flashes at difflow's stage states.

Docs: `docs/unit-operations-refinery.md` ("The saturated gas plant").
Tests: `tests/refinery/test_gasplant*.py`. Example: `examples/38_refinery_gas_plant.ipynb`.

### Refinery Blending (`difflow_refinery`)

`BlendPool(product, specs, rules)` blends `BlendComponent`s (from properties,
or from pseudocomponent streams on a shared `BlendCharacterization`) and returns
properties, signed spec margins and, in stream mode, the product stream. Like
`difflow.planning` it is a library, not a palette operation: the plugin's
entry point registers `CrudeDistillationUnit` and `VacuumColumn` only.

Invariants encoded in the module (do not weaken them):
- Volumes are ideal-mixing volumes at 15 degC from SG; product SG is the
  volume average and the stream-mode mass and volume balances close to
  round-off. Both are tested.
- Ethyl RT-70 corrections are spreads, so the rule reduces exactly to the
  linear blend when the components agree -- tested, keep it that way. The MON
  interaction is on MON x SENSITIVITY and b3 = -0.00645 (Maples 2000); the
  pre-#301 code had MON x olefins and -0.0645, which inflated the MON penalty
  ten-fold. RVP index and Refutas are tested against published worked examples.
- Distillation and cetane index are COMPUTED from the blend's composition,
  never blended; the linear view averages each component's own value, and the
  difference is real (T10 especially).
- The smooth violation is `t * logaddexp(-m/t, 0)` (derivative -1/2 at an
  active spec), never the branchless form -- same reason as `difflow.stochastic`.
- Margins for an optimizer over volume flows are `weighted=True` (`V * m`):
  properties are 0/0 at an empty pool.
- `exact=` in `linear_properties`/`backoff` accepts only rules linear in the
  volumes; it describes an LP.
- Blending is nonconvex (with the corrected RT-70 every start in the example
  happens to find one plan, which is not a guarantee); do not present a
  single-start NLP as "the" optimum.

- Property estimates (#330, `difflow_refinery.properties`): `from_stream`
  estimates flash (Riazi-Daubert from D86 T10), freeze (n-paraffin ideal
  solubility on Won 1986), smoke (Riazi), viscosity (Abbott + D341 at
  `viscosity_T_C`) and straight-run RON/MON (the reformer's pure-compound
  octanes by P/N/A/O, RT-70) unless given; `estimate=False` is the old
  behaviour. All but Won's melting points are UNVERIFIED against their
  sources -- keep them marked so, and let a measured value override.

Docs: `docs/unit-operations-refinery.md`. Example:
`examples/33_refinery_gasoline_blending.ipynb`. Tests: `tests/refinery/`.

### Fluid Catalytic Cracker (`difflow_refinery.fcc`, #308)

`FCCUnit(FCCParams(...)).solve(FCCFeed.from_characterization(char, rate, T_lo, T_hi))`:
lumped riser (`ancheyta_5` default, `lee_4`, `weekman_nace_3`; diffrax,
constant-step Tsit5 + `DirectAdjoint`) and coke-burning regenerator solved
TOGETHER by optimistix Newton -- unknowns C/O and T_rg, ROT the spec --
with implicit gradients; simplified main fractionator (TBP sigmoid split,
NOT a StageColumn). Outlets `dry_gas, c3, c4, gasoline, lco, slurry,
sour_water, flue_gas` (C3=/C4= for alkylation #310). `fcc_block(...)` for
planning. A library, not registered.
- Kinetic constants (`ILLUSTRATIVE_5LUMP`) are NOT from any paper; no
  literature cross-check was reproduced. Do not present yields as predictions.
- Balances (mass, C, H, S, N, energy) close by construction: cycle-oil H and
  H2S are BY DIFFERENCE; keep it that way, and keep `cycle_oil_hydrogen`
  reported so an implausible by-difference value is visible.
- The heat balance has multiple steady states; `RegeneratorTemperatureWarning`
  flags a hot one rather than hiding it.
- Under `jax.grad` w.r.t. the assay pass `indices=` (`fcc.feed.cut_indices`).

Docs: `docs/unit-operations-refinery.md` ("The fluid catalytic cracker").
Tests: `tests/refinery/test_fcc.py`.

### Chaining Units (`difflow_refinery.plant`, #334)

`Chain(Stage("nht", f, modes="rev"), Stage("reformer", g, modes="fwd", jit=True)).jacobian(x)`
composes library units (and the pure-JAX adapters between them) into one
differentiable function. `method="auto"`: one `jax.jacfwd`/`jacrev` when every
stage shares the mode, else `"chain"` (the chain rule by unit Jacobians, forward
accumulation). `AD_MODES` / `ad_mode_table()` is the one place each unit's AD mode
is written down. Example 40, section 10, uses it. A library.
- A mixed chain (forward-only reformer, reverse-only default hydrotreater) can NOT
  be traced end to end in either mode; `tests/refinery/test_plant.py` pins both
  failures on toy stages. `HydrotreaterParams(reactor=ReactorOptions(adjoint="forward"))`
  makes the hydrotreater forward-capable (same values).
- `jit=True` on any stage holding a Python-level recycle (the reformer's
  `Flowsheet`): unjitted, it is traced and compiled anew on every JVP and every
  call (measured 671 s / 511 s vs 487 s / 34 s jitted).
- Memory, not time, limits a chain on a 15 GB box: example 40 calls
  `jax.clear_caches()` before differentiating. Keep interfaces after a
  reverse-only stage narrow (one cotangent per output of that stage).

Docs: `docs/unit-operations-refinery.md` ("Chaining units"). Tests:
`tests/refinery/test_plant.py` (per commit), `tests/refinery/test_plant_chain.py` (release, slow).

### Delta-Base Planning (`difflow.planning`)

Turns flowsheets into a planning LP/MILP whose unit submodels are AD Jacobians
("delta vectors"), kept honest with a trust region. It is a module alongside
`eo_solver.py` and `estimation/`, **not** a `difflow.plugins` entry point.

```python
from difflow.planning import Block, Network, DeltaBasePlanner

blk = Block(name="ngl", fn=flowsheet_fn,          # any pure JAX u -> y callable
            u_names=[...], y_names=[...], lb=[...], ub=[...], jit=True)
net = Network([blk, power], links=[("ngl.residue_F", "power.fuel_F")])
res = DeltaBasePlanner(net, prices={...}, specs=[("ngl.T_colfeed", "<=", 236.0)],
                       radius=0.3).solve()
res.plan, res.delta_vectors, res.pyomo_model, res.plan_sensitivity(wrt="prices")
```

Invariants encoded in the module (do not weaken them):
- Trust-region proposals are accepted only after evaluating the caller's own
  *nonlinear* blocks (`accept_test=True`); `accept_test=False` exists only to
  demonstrate the failure mode.
- Constraint violations are scored from the nonlinear model, never from LP slacks.
- Bang-bang levers get vertex-seeded starts; use `price_switch_point` for the
  finite price at which a corner flips.
- AD mode is chosen by shape (`choose_ad_mode`), never hard-coded.
- Blocks with a `phase_fn` raise `PhaseBoundaryWarning` when a proposal crosses
  a phase boundary.
- Inter-block recycles are rejected — merge them into one flowsheet.
- Out of scope by design: pooling/blending bilinearity, assay libraries,
  blending correlations, scheduling. Do not add them.

Scaling to large flowsheets (`difflow.planning.health`): a small *absolute*
composed sensitivity is physics, not a vanishing gradient — difflow runs in
float64 and relative sensitivity survives arbitrary depth. What does degrade
with size is (1) dead levers, where a `clip`/`where`/`minimum` on an active
spec gives an exactly-zero delta column so the LP never moves that lever,
(2) `(I - A)^-1` amplification from recycle loop gains near one, and (3)
constraint-matrix scale spread from mixed units. `check_delta_health(block or
network)` and `DeltaBasePlanner.check_health()` report all three; they never
raise during a solve. Keep blocks small — linearising a whole plant as one
block *does* form the deep chain-rule product and its entries do collapse.

Second order, and multi-period (`difflow.planning.curvature`, and the `Link`
machinery you already have):
- A Hessian-vector product is one `jvp` through `grad`, so the exact Hessian of
  a scalar output costs `O(n_u)` HVPs -- what a central-difference *Jacobian*
  costs. `check_model_order(block, output, sense=...)` measures the linear and
  quadratic models against the block itself and says which earns its cost.
- It recommends `"quadratic"` only when the fit is better AND the Hessian is
  definite in the direction of optimisation. An indefinite Hessian makes the
  subproblem a nonconvex QP, which forfeits the global optimality that the
  duals-as-prices reading and the Eason-Biegler theory both rest on -- so a
  perfect fit is refused, and `.caveat` names the convexification to apply.
- Definiteness is a property of the POINT, not of the model: the same reduced
  AC cost is convex at an incumbent schedule and indefinite under heavy load.
  Check it at the linearisation point each cycle; never cache the verdict.
- Multi-period inventory needs no new machinery. A `Link` is output-to-input
  and the network rejects only cycles, so `tank@t0.level_out ->
  tank@t1.level_in` is an ordinary DAG edge. Make the first period a block with
  no `level_in` so the model starts feasible.
- `Spec` is `elastic=True` by default, which is right for a commercial spec and
  WRONG for a mass balance: an elastic inventory balance lets the planner
  report a better objective by selling from an empty tank, and it converges
  without complaint. Physical constraints are `elastic=False`.
- `model_order="quadratic"` puts that curvature in the subproblem, which
  becomes a QP (`difflow.planning.quadratic`). It is what makes the loop
  TERMINATE: on case9 AC-OPF, linear runs 40 iterations and ends on the
  iteration cap, quadratic ends on its own radius test in 12, same optimum.
  `"auto"` takes curvature only where the Hessian is already definite.
- The subproblem stays a QP, never a QCQP: only the OBJECTIVE gets curvature,
  the constraint rows stay first order. Quadratic rows would make each
  subproblem a nonconvex QCQP and void the global-optimality guarantee.
- `convexify` clips wrong-signed eigenvalues and RECORDS it in
  `qp.convexification`; a convexified model is not the true second-order
  model, and the trust region plus the acceptance test are what keep it
  honest. A block with exactly zero curvature is skipped, never floored --
  flooring hands the solver curvature the model does not have.
- `QPModel.minimised` vs `.objective`: the first is the solver's convention
  (lower is better), the second the caller's. The expansion constant lives in
  the minimised convention and `objective_offset` in the caller's; conflating
  them shifts the reported value by twice the constant.
- Integer columns (piecewise SOS2) would make it a MIQP: those networks fall
  back to linear automatically.
- Feasibility restoration (`difflow.planning.restoration`) handles an
  inelastic spec violated at the start, which otherwise dead-ends: shrinking
  the radius cannot restore feasibility. Phase one relaxes only the SPEC
  rows -- model and link rows are definitional, so an equality-infeasible
  subproblem is a broken model and must be reported, not absorbed. The row
  taxonomy is total and asserted: a label matching neither `RELAXABLE_PREFIXES`
  nor `STRUCTURAL_PREFIXES` raises, because the match is positive and a new
  kind left unclassified would be silently left hard.
- Restoration has its OWN trust region and acceptance test, judged on the
  nonlinear blocks: a phase-one LP given a big enough region proposes points
  it predicts feasible and the blocks are not (measured: predicted violation
  to zero while true violation ROSE). It also keeps its own radius -- the
  search for a feasible point says nothing about where the objective model is
  trustworthy, and resuming from it makes the planner crawl.

Modifiers from plant history (`difflow.planning.attribution`):
`attribute_deltas(block, U, {output: y}, sigma_y=...)` estimates the level and
slope corrections from logged data. Slope estimability is decided by a
column-pivoted QR of the *design* only, never by the fitted answer; most
slopes are not estimable from routine closed-loop data, and saying so is the
point. Do not drop the autocorrelation inflation or the alias report -- both
exist because their absence produced confident false flags.

Reporting and drawings (use these rather than re-deriving them in a notebook):
- `planner.describe()` states the problem — objective, decisions, bounds, links, specs.
- `lp_model.as_text()` writes the assembled LP out row by row.
- `difflow.planning.diagram`: `draw_chain` (process flow diagram of the reference
  chain), `draw_planning_network` (any network as the LP holds it),
  `draw_delta_vectors`, `draw_taylor_model`, `draw_trust_region`. matplotlib is
  imported inside the functions.

From a flowsheet, and out to someone else's LP:
- `Block.from_flowsheet(fs, u=["reactor.V", "feed:feed.total_flow"],
  y=["purge.F_B", ...])` is the bridge. Lever keys are `_apply_params` notation;
  feed streams are levers via the `feed:` prefix (`T`, `P`, `total_flow`,
  `F_<species>`, `x_<species>`). Run `check_delta_vectors` before exporting.
- Under `jax.jacobian` a recycle solve routes to the optimistix fixed-point
  path automatically — the Anderson/Wegstein loops are Python and cannot be
  traced. Never record a solve diagnostic with a bare `float()`; use
  `flowsheet._concrete()`, which returns `None` under tracing.
- `difflow.planning.export`: `DeltaVectorSet.from_result` / `.from_block`, then
  `write_json` / `write_csv` / `write_lp` / `write_mps` /
  `write_iterations_csv`. Also `difflow plan-export`. Units come from
  `Block.metadata["u_units"]`/`["y_units"]`; LP symbols are sanitised and the
  map is in `meta["lp_symbols"]`. The export is one-way — no importer.

Reference model: `difflow.planning.chain.two_plant_chain()`. Docs: `docs/planning.md`.
Example: `examples/30_delta_base_planning.ipynb`. Tests: `tests/test_planning.py`,
`tests/test_planning_export.py`, `tests/test_planning_curvature.py`,
`tests/test_planning_multiperiod.py`, `tests/test_planning_quadratic.py`,
`tests/test_planning_restoration.py`, `tests/test_planning_attribution.py`, `tests/power/test_planning_opf.py` (the
accuracy claim: SLP over AD delta vectors reaches the AC-OPF optimum and beats
DC-OPF, all three dispatches scored in the full AC model; and the termination
claim, linear against quadratic).

### Measuring Convergence (`difflow.convergence`)

How often the recycle solver works, over a corpus of deliberately hard
flowsheets. `make convergence` (or `python -m difflow.convergence`), or:

```python
from difflow.convergence import run_benchmark
report = run_benchmark()          # 99 solves: 11 cases x 3 accelerations x 3 inits
print(report.as_text())
```

Invariants encoded in the module (do not weaken them):
- **Passing is `converged AND correct`.** `Outcome.converged` is the solver's
  own verdict and `Outcome.correct` is an audit of the answer (the overall
  mole balance, or a known analytic solution). Keeping them apart is the
  point: today three of 54 solves report success and land somewhere that does
  not audit, and a benchmark reading only the flag would score those as wins.
- **Every `Case.build` returns a FRESH flowsheet.** `solve` records its
  verdict on the object, so a shared one carries one run's result into the
  next.
- **No case may be rigged.** A strategy must not start on the answer it is
  scored against. `overshoot_loop` originally used `g(x) = 3F - 2x`, whose
  fixed point is `x = F` — exactly the `"feed"` guess, which collected two
  free passes. `TestNoCaseIsRigged` checks all six.
- **Keep a control that is meant to pass.** An all-failing corpus cannot tell
  a hard corpus from a broken solver; `two_phase_flash` passes 9/9.
- **A raise is an outcome, not an error.** `run_case` records it and carries
  on; a benchmark that dies on its hardest case reports the pass rate of the
  cases before it.
- Every `Case` states its own `difficulty`. A hard case that does not say why
  measures something nobody can act on.

What it measured (2026-09, and meant to move): pass rate 46.5%, convergence
rate 49.5%. Anderson 84.8% against plain substitution's 27.3% — acceleration
dominates. The three initialization strategies are separated by one solve, and
`"unit"` (the path #247 wired up) scores exactly what the 0.01 mol/s default
it replaced scores: it saves an iteration or two and flips no verdict. That
held unchanged when the corpus went from six cases to eleven.

**Every case is solved by at least one acceleration.** Eleven cases across the
families #251 names — high gain, sharp splits, multi-loop, phase-regime
switching, near-pinch columns, trace tears, signed tears — and none defeats
all three methods. That negative result is the benchmark's main finding and
the thing #251 turns on.

Two traps it exposes, both worth knowing independently of that:
- `tol` tests the STEP, not the error, so on a loop of gain `g` it understates
  the remaining error by `1/(1-g)` (worse for a non-normal coupling, where the
  worst case is `||(I - M)^-1||`). At `g = 0.97` Wegstein stops inside `1e-8`
  and is 1% out, reporting convergence and meaning it. `trace_recycle` pins it.
  Since #264 `solve` MEASURES that rather than leaving it to be inferred:
  `error_probe` extra substitution passes after convergence give
  `last_solve_gain` and `last_solve_error_estimate`, a `TearToleranceWarning`
  fires when the error is past `tol` by 10x or more (a gain of 0.9 -- below
  that is the step test's ordinary slack, and a warning on every solve is
  noise), and `tol_basis="error"` tightens the step test until the measured
  error is inside `tol`. Opt-in, because it costs iterations. Still an
  ABSOLUTE test: the scale half of the trap is the caller's to set.
  The extrapolation ratio is SIGNED (an oscillating loop is closer than its
  step, not further), works on an expanding loop, and drops entries already at
  round-off -- a ratio of noise lands near one, which is where `1/(1-s)` blows
  up, and a tear carrying a pressure of 1e5 has such an entry most solves.
- `signed_tear` was NOT a second instance of that trap. Its apparent 450x
  non-normal amplification was its reference answer, written to six decimals
  and so 4.4e-6 from `(I - M)^-1 . 1`; the solver lands 7.6e-9 away, inside its
  own residual. Reference now at full float64 and the case audits at 1e-6 like
  the other analytic ones (#264).
- `clip_negative_flows` defaults to True and binds on the Wegstein and
  Anderson paths but NOT on the unaccelerated one -- deliberately (#263): it
  guards an *extrapolated* guess, and substitution makes none. Clipping there
  would clip `g` itself, which invents fixed points and, since a traced solve
  falls back to that path, puts a kink inside the map `optx.fixed_point`
  implicitly differentiates (measured: an exact gradient of 2.0 comes back as
  1.333 on a tear flow that converges to zero). A solve that fails *because*
  of the clip now says so: `last_solve_clip_active` counts the iterations it
  moved the iterate, and the non-convergence warning names the remedy.
  On a tear whose answer is genuinely negative that projection is the
  difference between solving it and not: `signed_tear` is the one case where
  plain substitution beats both accelerated methods, and
  `clip_negative_flows=False` fixes Anderson on it (100 iterations to 13).
  Same point `difflow_gas` already makes for signed flows.

Anderson is not free either — `phase_coupled_flash` fails under it and
converges under Wegstein.

Docs: `docs/convergence.md` ("Measured pass rates"). Tests:
`tests/test_convergence_benchmark.py`, `tests/test_benchmarks_verdict.py`.

### Stochastic Programming (`difflow.stochastic`)

Design under uncertainty: the sample average approximation of a two-stage
stochastic program, over any pure JAX `model(x, u, theta) -> {name: value}`.
Like `difflow.planning`, a module alongside `flexibility/`, **not** a
`difflow.plugins` entry point.

```python
import difflow.stochastic as st
scen = st.ScenarioSet.from_covariance(["a_Nd", "a_Dy"], mean, Sigma, n=256, seed=0)
prob = st.TwoStageProblem(model=circuit,
                          first_stage={"n_stages": (2., 16.)},   # here and now
                          recourse={"pH": (0.1, 1.2)},           # wait and see
                          objective="profit", maximize=True, risk=("cvar", 0.9),
                          constraints=[("purity", ">=", 0.93, 0.90)])
res = st.solve_saa(prob, scen)
st.bounds(prob, scen, result=res)          # VSS and EVPI
st.check_scenario_health(prob, scen, res)  # dead levers, empty tails, saturation
```

Invariants encoded in the module (do not weaken them):
- The scenario sample is drawn ONCE and reused. There is no
  resample-per-iteration option: a resampled SAA objective is not a function,
  and a descent method on it stalls at a distance set by the noise.
  `optimality_gap` is the only thing that redraws, and it reports a bound
  rather than a better point.
- Non-anticipativity is STRUCTURAL — one first-stage array shared by every
  scenario — never a constraint that could be written down wrongly.
- The Rockafellar-Uryasev auxiliary is NOT a decision variable. It is
  recomputed at its closed-form optimum every iterate and frozen with
  `stop_gradient`; that is exact by the envelope theorem. Handing it to a
  projected method whose step scales with the box width makes the auxiliary
  swing harder than the design, and the run ends at a bound looking converged.
- CVaR's smoothed positive part is `t * logaddexp(z/t, 0)`, never the branchless
  `max(u,0) + log1p(exp(-|u|))`. The two have the same VALUE everywhere and the
  second has a derivative of exactly ZERO at `u = 0` — `max`'s JVP breaks the
  tie toward the constant and `abs`'s is 0, so the halves of the kink cancel
  instead of averaging to the sigmoid's 0.5. `z - t` is exactly zero for the
  scenario at the quantile, and with a SINGLE scenario — which is what
  `expected_value_solution`, and hence every VSS, solves — for the only scenario
  there is: the whole gradient vanishes and the design never leaves its
  initialization while reporting convergence.
  `tests/test_stochastic.py::TestRisk::test_a_one_scenario_cvar_solve_would_have_frozen_at_its_start`
  is the regression.
- `WS <= SP <= EEV` holds among ADMISSIBLE designs only. A mean-value design
  that misses the chance constraint scores better than the stochastic one for
  exactly the reason it is not allowed; `BoundReport.ev_feasible` says so and
  `summary()` prints "not a value" rather than a negative VSS that blames the
  solver for a modelling fact. Without recourse this is the common case.
- The constraint handler is an AUGMENTED LAGRANGIAN, not a growing penalty. A
  penalty weight of 1e4 inflates Adam's second moment by eight decades and
  every later step shrinks to nothing — the iterate freezes and reports itself
  feasible. `tests/test_stochastic.py::TestSolve::test_a_growing_penalty_would_have_frozen_here`
  is the regression.
- Objective and constraint residuals are rescaled, and the rescaling is EXACT:
  every risk measure is translation-equivariant and positively homogeneous, and
  every constraint form is positively homogeneous in its residual.
- Feasibility is scored by re-evaluating the model, never read off the
  multipliers — the same rule `difflow.planning` states for LP slacks.
- `wait_and_see` carries the constraints across UNCHANGED. That is what makes
  it a bound (drop non-anticipativity, change nothing else). Rewriting a chance
  constraint per-scenario can make the relaxation infeasible where the original
  was fine, and then EVPI comes out negative.
- `n_aux` on a risk measure is a plain class attribute, never an annotated
  dataclass field: as a base-class field it takes the first positional slot and
  `CVaR(0.9)` silently sets the auxiliary count instead of alpha.
- Out of scope by design: L-shaped/Benders decomposition, scenario reduction,
  multistage trees, integer recourse, distributionally robust formulations.
  Do not add them.

Where the uncertain parameters come from: `difflow.estimation.predicted_covariance`
and `difflow.reconciliation.reconciled_covariance` both return the `(mean, Sigma)`
that `ScenarioSet.from_covariance` wants. Prefer it over the per-parameter
constructors — correlated coefficients have a *difference* variance that a
diagonal Sigma gets wrong by a factor of a few.

Related, and cheaper — try these first: `difflow.uncertainty` (propagate a
distribution through a fixed design), `difflow.planning.backoff` (`kappa*sigma`
margin from one Jacobian), `difflow.flexibility.expected_feasibility` (does a
design I already have meet spec often enough?), `difflow.flexibility` proper (a
guarantee over an envelope, by vertex search rather than sampling).

Docs: `docs/stochastic.md`. Example: `examples/32_stochastic_ree_separation.ipynb`.
Tests: `tests/test_stochastic.py`, `tests/ree/test_coefficient_overrides.py`.

### Keeping a Model Current (`difflow.reconciliation.tracking`)

The digital-twin loop: a parameter updated from plant data as it arrives,
gated on the monitoring verdict and filtered onto a random walk. Four public,
stateless steps — `update_gate`, `time_update`, `parameter_measurement`,
`measurement_update` — plus `track_parameters`, the offline driver over a
campaign, which doubles as the backtest for choosing `drift_std`.

```python
run = track_parameters(F, daily, sigma, names=layout.names,
                       state=TrackerState.initial(["eta"], [1.0], std=[0.02]),
                       drift_std=drift_std_from_time_constant(0.05, 30.0))
run.final.as_params()          # {'eta': 0.71}, the shape a Block takes as theta
```

Invariants encoded in the module (do not weaken them):
- The verdict **gates** the update; it does not advise it. Only `model drift`
  opens it. Re-estimating on a concentrated suspect converts a meter fault into
  a confident statement about equipment, and the chi-squared statistic *falls*
  while it happens — `tests/test_tracking.py::TestGate::test_an_ungated_loop_would_have_invented_a_fouling_factor`
  is the regression. Widening `allow=` is a deliberate act with a known cost.
- Holding on `consistent` is the deadband, not an oversight. A parameter
  re-estimated every period tracks that period's noise and the twin stops being
  a model.
- The **time update always runs**, on held periods too: a held parameter widens
  its error bar at the drift rate, so the next permitted update steps further.
- `drift_std` is required and never defaulted — it is the bandwidth of the twin.
  Same knob, same warning as `process_std` in `difflow.mhe`.
- The covariance is **full**, never diagonal (correlated parameters have a
  difference variance a diagonal gets wrong by a factor of a few), and is
  propagated in **Joseph form** — `(I-K)P` loses symmetry over a long run, and a
  twin is a long run.
- `parameter_measurement` leaves the parameters at `sigma = inf` (free), so the
  prior stays outside `reconcile`: a prior smuggled in through `sigma` would be
  diagonal, and would inflate the objective that is supposed to test data
  against model.
- A weakly informative period is not a special case — large `R`, zero gain.
- Out of scope: this filters parameters, not model *form*. Under structural
  mismatch the statistic never comes back down; the answer is
  `difflow.planning.modifiers`, not a faster filter.

Docs: `docs/data-reconciliation.md`. Tests: `tests/test_tracking.py`.

### Debugging Gradients

```python
# Check for NaN gradients
jax.config.update('jax_debug_nans', True)

# Finite difference gradient check
from jax.test_util import check_grads
check_grads(my_function, (x,), order=1, modes=['rev'])

# Print inside JIT
jax.debug.print("value: {x}", x=value)
```

## Dependencies

**Core:**
- JAX (>=0.4.0) - Automatic differentiation
- diffrax (>=0.6.0) - ODE/DAE solvers
- lineax (>=0.0.7) - Linear solvers
- optimistix (>=0.0.6) - Root finding
- PyYAML (>=6.0) - Configuration

**Development:**
- pytest, pytest-cov - Testing
- jupyter-book - Documentation

**Optional:**
- matplotlib, jupyter - Examples
- cantera - Complex chemistry
- pyglenn - NASA Glenn (CEA) thermo data import (`difflow.pyglenn_import`)
- pythonnet - DWSIM thermo data import (`difflow.dwsim_import`, prototype)
- ipycytoscape, networkx - Visualization

## Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration, dependencies |
| `Makefile` | Build automation (test, book, notebooks) |
| `src/difflow/__init__.py` | Main API exports |
| `src/difflow/params_mixin.py` | ParamsMixin base class for all Params dataclasses |
| `src/difflow/docstrings.py` | Reads Params field descriptions out of the `Attributes:` docstrings and field comments, for the catalog |
| `src/difflow/gui/doclinks.py` | Resolves an operation name to its section in `docs/` (the palette's documentation link) |
| `src/difflow/convergence.py` | Hard-flowsheet corpus and the recycle solver's measured pass rate |
| `src/difflow/planning/` | Delta-base planning: AD delta vectors -> trust-region LP/MILP |
| `src/difflow/stochastic/` | Two-stage stochastic programming: SAA over a scenario sample, VSS/EVPI |
| `src/difflow_bio/__init__.py` | Bio manufacturing plugin exports |
| `src/difflow_ree/__init__.py` | REE extraction plugin exports |
| `src/difflow_cc/__init__.py` | Carbon capture plugin exports |
| `src/difflow_gas/__init__.py` | Gas transmission network plugin exports |
| `src/difflow_power/__init__.py` | Electrical grid plugin exports (AC-OPF) |
| `src/difflow_refinery/__init__.py` | Refinery plugin exports (crude assay, CDU, VDU, blending, hydrotreating) |
| `src/difflow_refinery/__init__.py` | Refinery plugin exports (crude assay, CDU, VDU, gas plant, blending) |
| `tests/` | All pytest tests (includes `bio/`, `ree/`, `cc/`, `gas/`, `power/`, `refinery/` subdirs) |
| `examples/` | Usage examples (Jupyter notebooks) |
| `jax-tutorials/` | JAX autodiff tutorials |
| `docs/` | Documentation source (Markdown, built with Jupyter Book) |

## Performance Tips

1. **JIT compile** hot paths: `@jit` or `jit(fn)`
2. **Vectorize** with `vmap` instead of Python loops
3. **Use 64-bit floats** for numerical stability: `jax.config.update('jax_enable_x64', True)`
4. **Checkpoint** memory-heavy computations: `jax.checkpoint(fn)`
5. **Profile** with `jax.profiler` for bottlenecks

## Troubleshooting

### "TracerArrayConversionError"
- Cause: Using JAX arrays in Python control flow during tracing
- Fix: Use `jax.lax.cond`, `jax.lax.switch`, or `jax.lax.fori_loop`

### "ConcretizationError"
- Cause: Trying to use abstract array values concretely
- Fix: Avoid `if x > 0:` with traced values; use `jnp.where(x > 0, ...)`

### NaN in Gradients
- Enable debug: `jax.config.update('jax_debug_nans', True)`
- Common causes: `log(0)`, `sqrt(negative)`, `0/0`
- Fix: Add small epsilon, use `jnp.clip`, safe functions

### Slow Compilation
- Large functions take time to JIT compile (one-time cost)
- Consider breaking into smaller functions
- Check for Python loops that could be `vmap`/`scan`
