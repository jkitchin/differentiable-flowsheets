# difflow

**Differentiable Flowsheet Framework for Chemical Processes**

A JAX-based framework for building and optimizing chemical process flowsheets with automatic differentiation.

## Features

- **Fully Differentiable**: All unit operations and flowsheet calculations support automatic differentiation via JAX
- **Sensitivity Analysis**: Compute gradients of outputs with respect to any inputs, parameters, or operating conditions
- **Optimization Ready**: Use gradient-based optimization for process design, parameter estimation, and economic optimization
- **Modular Design**: Unit operations can be composed into complex flowsheets with recycle streams

## Installation

```bash
# Clone and install
git clone <repo-url>
cd differentiable-flowsheets
uv venv
uv pip install -e ".[dev]"
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

### Utilities
- **Mixer**: Combine multiple streams
- **Splitter**: Split stream by fraction

## Thermodynamics

Currently supports ideal thermodynamics:
- Ideal gas behavior
- Antoine equation for vapor pressures
- Polynomial Cp correlations
- Watson correlation for heat of vaporization

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

### CSTR Sensitivity Analysis
Compute gradients of outputs with respect to:
- Inlet conditions (flow rates, temperature)
- Kinetic parameters (A, Ea)
- Operating conditions (V, T)

```bash
python examples/cstr_sensitivity.py
```

### Optimization
- Single/multi-variable optimization
- Constrained optimization (penalty method)
- Economic optimization (profit maximization)
- Parameter estimation from data
- Multi-objective Pareto analysis

```bash
python examples/optimization_examples.py
```

### CSTR + Flash with Recycle
Complete flowsheet with recycle convergence and sensitivity analysis:

```bash
python examples/cstr_flash_recycle.py
```

## Key Design Decisions

1. **Streams as Dicts**: Simple `{"F_A": ..., "F_B": ..., "T": ..., "P": ...}` format that's a JAX pytree by default

2. **User-Provided Properties**: No built-in database; you define species data for your system

3. **Function-Based Kinetics**: Maximum flexibility via `rate_fn(C, T, params) → rates`

4. **Unrolled Iteration**: Fixed-point solvers use `lax.scan` for automatic differentiability

## Limitations

- Currently only ideal thermodynamics (no activity coefficients or EOS)
- Gradient explosion possible with many iterations (use damping)
- No built-in species database

## Future Work

- Activity coefficient models (NRTL, Wilson)
- Equation of state (Peng-Robinson, SRK)
- More unit operations (PFR, distillation, heat exchangers)
- Implicit differentiation for better gradient accuracy
- Integration with optimization libraries (scipy, optax)

## License

MIT
