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

# Build documentation (Jupyter Book)
make book

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
│   │   │                   # observability, monitoring, multi-set pooling
│   │   ├── units/         # Steady-state unit operations
│   │   ├── dynamic/       # Dynamic modeling (DAE)
│   │   ├── economics/     # Technoeconomic analysis
│   │   └── visualization/ # Flowsheet visualization
│   ├── difflow_bio/       # Bio manufacturing plugin (bioreactors, filtration, chromatography)
│   ├── difflow_ree/       # Rare earth element solvent extraction plugin
│   ├── difflow_cc/        # Carbon capture plugin (amine, membrane, adsorption)
│   └── difflow_gas/       # Gas transmission network plugin (pipes, compressors, computed decomposition)
├── tests/                 # pytest test files (includes tests/bio/, tests/ree/, tests/cc/, tests/gas/, tests/power/)
├── examples/              # Jupyter notebook examples
├── jax-tutorials/         # JAX/autodiff tutorials
└── docs/                  # Documentation (Markdown)
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
and `tests/test_doclinks.py` asserts all 87 operations resolve, so
skipping step 7 fails the suite rather than shipping a palette entry with
nothing to read.

### Adding to a Plugin (bio, ree, cc, gas, power)

The project has five domain-specific plugins:
- **difflow_bio**: Bio manufacturing (bioreactors, filtration, chromatography)
- **difflow_ree**: Rare earth element solvent extraction
- **difflow_cc**: Carbon capture (amine absorption, membrane, adsorption)
- **difflow_gas**: Gas transmission networks (pipes, compressors, valves, topology-driven sequential decomposition)
- **difflow_power**: Electrical grids (AC power flow, AC-OPF, DC-OPF, PTDF/LODF, state estimation)

1. Add to appropriate plugin directory (`src/difflow_bio/`, `src/difflow_ree/`, `src/difflow_cc/`, `src/difflow_gas/`, or `src/difflow_power/`)
2. Create a Params dataclass inheriting from `ParamsMixin`
3. Export in plugin's `__init__.py` and add to `__all__`
4. Add tests in `tests/bio/`, `tests/ree/`, `tests/cc/`, `tests/gas/`, or `tests/power/`
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
- Database: 10 REE elements, 4 extractant systems
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
`tests/test_planning_export.py`.

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
                          recourse={"pH": (1.8, 2.8)},           # wait and see
                          objective="profit", maximize=True, risk=("cvar", 0.9),
                          constraints=[("purity", ">=", 0.80, 0.90)])
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
| `src/difflow/planning/` | Delta-base planning: AD delta vectors -> trust-region LP/MILP |
| `src/difflow/stochastic/` | Two-stage stochastic programming: SAA over a scenario sample, VSS/EVPI |
| `src/difflow_bio/__init__.py` | Bio manufacturing plugin exports |
| `src/difflow_ree/__init__.py` | REE extraction plugin exports |
| `src/difflow_cc/__init__.py` | Carbon capture plugin exports |
| `src/difflow_gas/__init__.py` | Gas transmission network plugin exports |
| `src/difflow_power/__init__.py` | Electrical grid plugin exports (AC-OPF) |
| `tests/` | All pytest tests (includes `bio/`, `ree/`, `cc/`, `gas/`, `power/` subdirs) |
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
