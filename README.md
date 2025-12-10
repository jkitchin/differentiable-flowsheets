# difflow

**Differentiable Flowsheet Framework for Chemical Processes**

A JAX-based framework for building and optimizing chemical process flowsheets with automatic differentiation.

## Features

- **Fully Differentiable**: All unit operations and flowsheet calculations support automatic differentiation via JAX
- **Sensitivity Analysis**: Compute gradients of outputs with respect to any inputs, parameters, or operating conditions
- **Optimization Ready**: Use gradient-based optimization for process design, parameter estimation, and economic optimization
- **Modular Design**: Unit operations can be composed into complex flowsheets with recycle streams
- **Bio Manufacturing**: Specialized unit operations for biopharmaceutical processes (bioreactors, chromatography, filtration)
- **Interactive Visualization**: Generate flowsheet diagrams directly from your code with Plotly

## Installation

```bash
# Clone and install
git clone <repo-url>
cd differentiable-flowsheets
uv venv
uv pip install -e ".[dev]"

# For interactive visualization (includes plotly, networkx)
uv pip install -e ".[visualization]"

# For examples and tutorials (includes matplotlib, jupyter)
uv pip install -e ".[examples]"

# Install everything
uv pip install -e ".[all]"
```

## Quick Start

```python
import jax.numpy as jnp
import jax

from difflow import (
    make_stream, get_flows,
    IdealThermo, SpeciesData,
    CSTR, CSTRParams,
)

# Define species
species_data = {
    "A": SpeciesData("A", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                     Hvap_coeffs=(35000.0, 0.38, 500.0),
                     antoine_coeffs=(10.0, 3000.0, -50.0)),
    "B": SpeciesData("B", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                     Hvap_coeffs=(30000.0, 0.38, 450.0),
                     antoine_coeffs=(10.0, 2800.0, -40.0)),
}
thermo = IdealThermo(species_data)

# Define reaction kinetics
def rate_fn(C, T, params):
    k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
    return jnp.array([k * C["A"]])

# Create CSTR
stoich = jnp.array([[-1.0], [+1.0]])  # A → B
cstr_params = CSTRParams(
    V=jnp.array(1.0),
    rate_fn=rate_fn,
    stoich=stoich,
    rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
    species_order=["A", "B"],
)
cstr = CSTR(cstr_params, thermo=thermo, mode="isothermal")

# Run simulation
inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
outlet, info = cstr(inlet, T_spec=350.0)

print(f"Conversion: {info['conversion']['A']*100:.1f}%")

# Compute gradient of outlet B w.r.t. reactor volume
def outlet_B(V):
    params = CSTRParams(V=V, rate_fn=rate_fn, stoich=stoich,
                        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
                        species_order=["A", "B"])
    cstr = CSTR(params, thermo=thermo, mode="isothermal")
    outlet, _ = cstr(inlet, T_spec=350.0)
    return outlet["F_B"]

dFB_dV = jax.grad(outlet_B)(jnp.array(1.0))
print(f"dF_B/dV = {dFB_dV:.4f} mol/s per m³")
```

## Unit Operations

### CSTR (Continuous Stirred Tank Reactor)
- Multiple reactions with user-defined kinetics
- Isothermal, adiabatic, or specified heat duty modes
- Automatic material and energy balance solving

### Flash Separator
- TP flash (temperature and pressure specified)
- Rachford-Rice equation for VLE
- Ideal thermodynamics (Raoult's law)

### Liquid-Liquid Extraction (LLE)
- **MultistageCascade**: Counter-current or co-current mixer-settler cascade
  - Kremser equation for stage calculations (differentiable in n_stages)
  - Continuous stage relaxation for optimization
- **DifferentialContactor**: Packed column extractor
  - HETP-based equilibrium model
  - Rate-based mass transfer model
- **Equilibrium Models**:
  - Distribution coefficients (K-values) with temperature dependence
  - NRTL activity coefficient model
  - UNIQUAC activity coefficient model

```python
from difflow import (
    MultistageCascade, CascadeParams,
    LLEEquilibrium, DistributionCoeffs,
)

# Define distribution coefficients for rare earth extraction
K_coeffs = DistributionCoeffs(
    species=("La", "Nd", "Dy"),
    K0=(0.5, 2.0, 8.0),  # K at reference temperature
)

equilibrium = LLEEquilibrium(
    solutes=["La", "Nd", "Dy"],
    aqueous_carrier="H2O",
    organic_carrier="Organic",
    K_coeffs=K_coeffs,
)

cascade = MultistageCascade(CascadeParams(
    n_stages=5,
    equilibrium=equilibrium,
    flow_config="counter_current",
))

raffinate, extract, info = cascade(feed_stream, solvent_stream)
```

### Utilities
- **Mixer**: Combine multiple streams
- **Splitter**: Split stream by fraction

## Bio Manufacturing Operations

The `difflow_bio` plugin provides specialized unit operations for biopharmaceutical manufacturing:

### Bioreactors
- **ContinuousBioreactor**: Chemostat with Monod kinetics
- **FedBatchBioreactor**: Fed-batch with substrate feeding strategy

```python
from difflow_bio import (
    ContinuousBioreactor, ContinuousBioreactorParams,
    FedBatchBioreactor, FedBatchBioreactorParams,
    monod_kinetics,
)

# Create a continuous bioreactor (chemostat)
params = ContinuousBioreactorParams(
    V=jnp.array(1000.0),           # Volume (L)
    mu_max=jnp.array(0.3),         # Maximum specific growth rate (1/h)
    Ks=jnp.array(0.5),             # Monod constant (g/L)
    Yxs=jnp.array(0.5),            # Biomass yield
    Yps=jnp.array(0.1),            # Product yield
    D=jnp.array(0.1),              # Dilution rate (1/h)
)
bioreactor = ContinuousBioreactor(params)
outlet = bioreactor(feed_stream)
```

### Downstream Processing
- **DiscStackCentrifuge**: Cell removal with Stokes' law separation
- **Ultrafiltration**: Protein concentration via TFF
- **Diafiltration**: Buffer exchange
- **ProteinAChromatography**: Affinity capture for mAb purification
- **IonExchangeChromatography**: Polishing step (bind-elute or flow-through)
- **SizeExclusionChromatography**: Aggregate removal

```python
from difflow_bio import (
    DiscStackCentrifuge, CentrifugeParams,
    Ultrafiltration, UFParams,
    ProteinAChromatography, ProAParams,
)

# Disc-stack centrifuge for cell removal
centrifuge = DiscStackCentrifuge(CentrifugeParams(
    sigma=jnp.array(5000.0),       # Sigma factor (m²)
    cell_diameter=jnp.array(15e-6), # Cell diameter (m)
))

# Protein A capture
proa = ProteinAChromatography(ProAParams(
    column_volume=jnp.array(10.0),  # CV (L)
    binding_capacity=jnp.array(40.0), # g mAb / L resin
    yield_factor=jnp.array(0.95),
))

# Ultrafiltration for concentration
uf = Ultrafiltration(UFParams(
    membrane_area=jnp.array(1.0),   # m²
    concentration_factor=jnp.array(10.0),
))
```

## Visualization

Generate interactive flowsheet diagrams directly from your code:

```python
from difflow import Flowsheet, Unit, make_stream

# Build and solve flowsheet
fs = Flowsheet(species_order=["A", "B", "C"])
fs.add_feed("feed", feed_stream)
fs.add_unit(Unit("reactor", cstr, ["feed"], ["reactor_out"]))
fs.add_unit(Unit("separator", flash, ["reactor_out"], ["product", "recycle"]))
fs.add_recycle("recycle", "recycle_in")

streams = fs.solve()

# Visualize - graph is generated from your code!
fig = fs.visualize(streams, layout="spring", title="My Process")
fig.show()

# Or get the graph object for more control
graph = fs.to_graph(streams)
```

Features:
- **Automatic**: Graph topology extracted from your flowsheet code
- **Interactive**: Zoom, pan, and hover tooltips with stream data
- **Export**: Save to HTML or PNG for sharing

## Thermodynamics

### Ideal Thermodynamics (for VLE)
- Ideal gas behavior
- Antoine equation for vapor pressures
- Polynomial Cp correlations
- Watson correlation for heat of vaporization

### Activity Coefficient Models (for LLE)
- **NRTL**: Non-Random Two-Liquid model with temperature-dependent parameters
- **UNIQUAC**: Universal Quasi-Chemical model

User provides species data:
```python
SpeciesData(
    name="species_name",
    MW=100.0,                           # Molecular weight (g/mol)
    Cp_coeffs=(a, b, c, d),            # Cp = a + bT + cT² + dT³
    Hvap_coeffs=(A, n, Tc),            # Hvap = A(1 - T/Tc)^n
    antoine_coeffs=(A, B, C),          # log10(Psat) = A - B/(T+C)
    Hf=0.0,                            # Heat of formation (J/mol)
)
```

## Flowsheets with Recycles

```python
from difflow import Flowsheet, make_stream
from difflow.solvers import fixed_point_solve

# Define flowsheet iteration
def flowsheet_step(recycle_arr, args):
    # Unpack recycle, run units, return new recycle
    ...
    return new_recycle_arr

# Solve recycle loop
recycle = fixed_point_solve(
    flowsheet_step,
    initial_guess,
    args,
    max_iter=100,
    damping=0.5,
)
```

## Examples

Jupyter notebooks are in the `examples/` directory:

| Notebook | Description |
|----------|-------------|
| `01_cstr_flash_recycle.ipynb` | Complete flowsheet with CSTR, flash, and recycle |
| `02_cstr_sensitivity.ipynb` | Sensitivity analysis for CSTR parameters |
| `03_optimization.ipynb` | Gradient-based optimization problems |
| `04_rare_earth_extraction.ipynb` | Rare earth recovery using LLE |
| `05_mab_downstream.ipynb` | mAb purification process (bio manufacturing) |
| `08_flowsheet_visualization.ipynb` | Interactive visualization tutorial |

```bash
# Launch Jupyter to explore examples
jupyter notebook examples/
```

## Tutorials

The `tutorials/` directory contains comprehensive JAX tutorials for differentiable programming:

| Notebook | Topics |
|----------|--------|
| `01_jax_fundamentals.ipynb` | grad, jit, vmap, pytrees, lax.scan, lax.cond |
| `02_optimization.ipynb` | Gradient descent, Newton, Adam, constrained optimization |
| `03_differential_equations.ipynb` | ODE solvers, parameter estimation, neural ODEs |
| `04_custom_derivatives.ipynb` | custom_vjp, custom_jvp, stop_gradient |
| `05_machine_learning.ipynb` | Neural networks from scratch, training loops |
| `06_gotchas.ipynb` | Common JAX pitfalls and how to avoid them |

## Key Design Decisions

1. **Streams as Dicts**: Simple `{"F_A": ..., "F_B": ..., "T": ..., "P": ...}` format that's a JAX pytree by default

2. **User-Provided Properties**: No built-in database; you define species data for your system

3. **Function-Based Kinetics**: Maximum flexibility via `rate_fn(C, T, params) → rates`

4. **Unrolled Iteration**: Fixed-point solvers use `lax.scan` for automatic differentiability

5. **Continuous Relaxation**: Discrete parameters (like n_stages) can be relaxed to continuous values for optimization

## Limitations

- VLE only supports ideal thermodynamics (Raoult's law)
- Gradient explosion possible with many iterations (use damping)
- No built-in species database

## Future Work

- Equation of state (Peng-Robinson, SRK) for VLE
- More unit operations (PFR, distillation, heat exchangers)
- Extended bio operations (viral inactivation, sterile filtration)
- GPU acceleration for large flowsheets

## License

MIT
