# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**difflow** is a JAX-based differentiable flowsheet framework for chemical process simulation. It enables automatic differentiation through chemical engineering unit operations for gradient-based optimization, sensitivity analysis, and technoeconomic modeling.

## Quick Start Commands

```bash
# Install in development mode
pip install -e ".[dev,examples]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cstr.py -v

# Run tests with coverage
pytest tests/ --cov=src/difflow

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
│   │   ├── planning/      # Delta-base planning (LP/MILP + trust region)
│   │   ├── catalog.py     # Machine-readable schema of every unit operation
│   │   ├── serialize.py   # Flowsheet <-> JSON round trip
│   │   ├── codegen.py     # Flowsheet -> runnable Python source
│   │   ├── kinetics.py    # Declarative mass-action rate laws (data, not callables)
│   │   ├── publish.py     # Flowsheet -> self-contained interactive HTML (no install)
│   │   ├── gui.py         # Local browser editor (python -m difflow.gui)
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
├── tests/                 # pytest test files (includes tests/bio/, tests/ree/, tests/cc/, tests/gas/)
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
)

# Create and run CSTR
cstr = CSTR(params)
outlet = cstr(inlet_stream)

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

### Adding to a Plugin (bio, ree, cc, gas)

The project has four domain-specific plugins:
- **difflow_bio**: Bio manufacturing (bioreactors, filtration, chromatography)
- **difflow_ree**: Rare earth element solvent extraction
- **difflow_cc**: Carbon capture (amine absorption, membrane, adsorption)
- **difflow_gas**: Gas transmission networks (pipes, compressors, valves, topology-driven sequential decomposition)

1. Add to appropriate plugin directory (`src/difflow_bio/`, `src/difflow_ree/`, `src/difflow_cc/`, or `src/difflow_gas/`)
2. Create a Params dataclass inheriting from `ParamsMixin`
3. Export in plugin's `__init__.py` and add to `__all__`
4. Add tests in `tests/bio/`, `tests/ree/`, `tests/cc/`, or `tests/gas/`
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

Reference model: `difflow.planning.chain.two_plant_chain()`. Docs: `docs/planning.md`.
Example: `examples/30_delta_base_planning.ipynb`. Tests: `tests/test_planning.py`.

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
| `src/difflow/planning/` | Delta-base planning: AD delta vectors -> trust-region LP/MILP |
| `src/difflow_bio/__init__.py` | Bio manufacturing plugin exports |
| `src/difflow_ree/__init__.py` | REE extraction plugin exports |
| `src/difflow_cc/__init__.py` | Carbon capture plugin exports |
| `tests/` | All pytest tests (includes `bio/`, `ree/`, `cc/` subdirs) |
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
