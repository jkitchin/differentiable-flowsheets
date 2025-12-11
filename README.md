# difflow

**Differentiable Flowsheet Framework for Chemical Processes**

A JAX-based framework for building and optimizing chemical process flowsheets with automatic differentiation.

## Features

- **Fully Differentiable**: All unit operations and flowsheet calculations support automatic differentiation via JAX
- **Sensitivity Analysis**: Compute gradients of outputs with respect to any inputs, parameters, or operating conditions
- **Optimization Ready**: Use gradient-based optimization for process design, parameter estimation, and economic optimization
- **Modular Design**: Unit operations can be composed into complex flowsheets with recycle streams
- **Technoeconomic Analysis**: Comprehensive TEA module with equipment costs, operating costs, and profitability metrics (NPV, IRR, MSP)
- **Bio Manufacturing**: Specialized unit operations for biopharmaceutical processes (bioreactors, chromatography, filtration)

## Installation

```bash
# Clone and install
git clone <repo-url>
cd differentiable-flowsheets
uv venv
uv pip install -e ".[dev]"

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

### PFR (Plug Flow Reactor)
- ODE-based design equation: dF/dV = stoich @ r
- Isothermal or adiabatic operation
- **GasPFR** variant for gas-phase reactions with:
  - Pressure drop (Ergun equation)
  - Variable volumetric flow from mole change
- RK4 integration via `lax.scan` (fully differentiable)

```python
from difflow import PFR, PFRParams, GasPFR, GasPFRParams

# Liquid-phase PFR
pfr = PFR(PFRParams(V=2.0, rate_fn=rate_fn, stoich=stoich,
                    rate_params=params, species_order=["A", "B"]))
outlet, info = pfr(inlet, T_spec=350.0)

# Gas-phase with pressure drop (A → 2B, mole increase)
gas_pfr = GasPFR(GasPFRParams(V=1.0, rate_fn=rate_fn, stoich=stoich,
                              rate_params=params, species_order=["A", "B"],
                              alpha=50000.0))  # Pressure drop parameter
outlet, info = gas_pfr(inlet, T_spec=500.0)
# info contains: conversion, profiles (V, F, T, P, Q), pressure_drop
```

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

## Technoeconomic Analysis (TEA)

The `difflow.economics` module provides comprehensive technoeconomic analysis capabilities, all fully differentiable for gradient-based optimization.

### Capital Costs

Equipment cost correlations with CEPCI escalation and installation factors:

```python
import difflow.economics as econ
import jax.numpy as jnp

# Equipment costs (2024 dollars)
reactor_cost = econ.reactor_cost(jnp.array(5.0), "cstr_jacketed")  # 5 m³
hx_cost = econ.heat_exchanger_cost(jnp.array(100.0), "shell_tube_floating")  # 100 m²
pump_cost = econ.pump_cost(jnp.array(10.0), "centrifugal_single")  # 10 kW

# Installed cost with Lang factor
installed = econ.installed_cost(reactor_cost, lang_factor=4.74)

# Total capital investment
tci = econ.total_capital_investment(
    purchased_equipment_cost=reactor_cost + hx_cost + pump_cost,
    lang_factor=4.74,
    working_capital_fraction=0.15,
)
```

Available equipment types:
- **Reactors**: CSTR (jacketed, coil), PFR, batch
- **Vessels**: Pressure vessels, storage tanks, flash drums
- **Heat Exchangers**: Shell-tube, double-pipe, plate-frame, air coolers
- **Columns**: Tray columns, packed columns
- **Pumps**: Centrifugal, reciprocating, gear
- **Compressors**: Centrifugal, reciprocating, screw
- **Separators**: Mixer-settlers, centrifuges, filters, extraction columns

### Utility Costs

```python
# Steam cost from heat duty
heating_cost = econ.steam_cost_from_duty(jnp.array(1e6), "medium_pressure")  # 1 MW

# Cooling water
cooling_cost = econ.cooling_water_cost(jnp.array(500e3))  # 500 kW

# Electricity
electricity_cost = econ.electricity_cost(jnp.array(100.0))  # 100 kW → $/hour
```

### Profitability Metrics

All metrics are JAX-differentiable:

```python
# Net Present Value
cash_flows = jnp.ones(20) * 500000  # $500k/year for 20 years
npv = econ.npv(cash_flows, jnp.array(0.10), jnp.array(2e6))  # 10% discount, $2M investment

# Internal Rate of Return
irr = econ.irr(cash_flows, jnp.array(2e6))

# Minimum Selling Price
msp = econ.minimum_selling_price(
    total_annual_cost=jnp.array(1e6),
    annual_production=jnp.array(50000.0),  # kg/year
)

# Annualized cost for optimization
tac = econ.annualized_cost(
    capital_cost=jnp.array(5e6),
    annual_opex=jnp.array(1e6),
    discount_rate=jnp.array(0.10),
    plant_life=jnp.array(20.0),
)
```

### Gradient-Based Economic Optimization

```python
import jax

def annual_profit(params):
    V, T = params[0], params[1]

    # Simulate process
    outlet, info = simulate_reactor(V, T)

    # Economics
    capex = econ.reactor_cost(V, "cstr_jacketed")
    installed = econ.installed_cost(capex)

    utility_cost = econ.cooling_water_cost(jnp.abs(info["Q"]))
    annual_utility = utility_cost * 8000 * 3600  # $/year

    revenue = outlet["F_product"] * product_price * 8000 * 3600

    crf = econ.capital_recovery_factor(jnp.array(0.10), jnp.array(20.0))
    return revenue - annual_utility - installed * crf

# Optimize design for maximum profit
grad_profit = jax.grad(annual_profit)
# Use gradient for optimization...
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

| Notebook/Script | Description |
|----------|-------------|
| `00_cstr_pfr_basics.ipynb` | CSTR and PFR basics: conventional vs difflow |
| `01_cstr_flash_recycle.ipynb` | Complete flowsheet with CSTR, flash, and recycle |
| `02_cstr_sensitivity.ipynb` | Sensitivity analysis for CSTR parameters |
| `03_optimization.ipynb` | Gradient-based optimization problems |
| `04_rare_earth_extraction.ipynb` | Rare earth recovery using LLE |
| `05_technoeconomic_analysis.ipynb` | Comprehensive TEA with profit optimization |

```bash
# Launch Jupyter to explore examples
jupyter notebook examples/
```

## Tutorials

The `tutorials/` directory contains comprehensive JAX tutorials for differentiable programming:

| Notebook | Topics |
|----------|--------|
| `01_jax_fundamentals.ipynb` | grad, jit, vmap, pytrees, jacfwd/jacrev, VJP/JVP, HVP |
| `02_inverse_hessian_vector_products.ipynb` | IHVP, conjugate gradient, Newton-CG optimization |
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
- More unit operations (distillation columns, heat exchangers)
- Extended bio operations (viral inactivation, sterile filtration)
- GPU acceleration for large flowsheets

## License

MIT
