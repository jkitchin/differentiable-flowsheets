# Difflow Documentation

**Difflow** is a JAX-based framework for building and optimizing chemical process flowsheets with automatic differentiation support. This enables gradient-based optimization of process designs, sensitivity analysis, and uncertainty propagation.

## Key Features

- **Automatic Differentiation**: All unit operations and calculations are fully differentiable using JAX
- **Comprehensive Unit Operations**: Reactors, separators, heat exchangers, distillation columns, and more
- **Bio-Manufacturing Support**: Specialized operations for bioreactors, centrifugation, filtration, and chromatography
- **Thermodynamic Models**: Ideal thermodynamics and cubic equations of state (Peng-Robinson, SRK)
- **Technoeconomic Analysis**: Equipment costs, operating costs, and profitability metrics
- **Extensible Architecture**: Plugin system for custom unit operations

## Documentation Contents

### Core Modules

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation, quick start, and basic examples |
| [Unit Operations - Chemical](unit-operations-chemical.md) | Reactors, separators, heat exchangers, distillation |
| [Unit Operations - Bio](unit-operations-bio.md) | Bioreactors, centrifugation, filtration, chromatography |
| [Thermodynamics](thermodynamics.md) | Property calculations, equations of state, databases |
| [Technoeconomics](technoeconomics.md) | Capital costs, operating costs, profitability analysis |
| [Streams and Flowsheets](streams-and-flowsheets.md) | Stream handling, flowsheet solver, recycle calculations |
| [Dynamic Modeling](dynamic-modeling.md) | Transient simulation, ODE/DAE integration, diffrax backend |
| [Solvers and Utilities](solvers-and-utilities.md) | Numerical methods, uncertainty propagation |

## Architecture Overview

```
difflow/
├── streams.py          # Stream data structures
├── thermo.py           # Ideal thermodynamics
├── eos.py              # Cubic equations of state
├── database.py         # Species property database
├── solvers.py          # Numerical solvers
├── flowsheet.py        # Flowsheet management
├── uncertainty.py      # Uncertainty propagation
├── cantera_import.py   # Cantera data import
├── units/              # Unit operations
│   ├── cstr.py         # CSTR reactors
│   ├── pfr.py          # PFR reactors
│   ├── fed_batch.py    # Fed-batch reactors
│   ├── flash.py        # Flash, mixer, splitter
│   ├── distillation.py # Distillation columns
│   ├── heat_exchanger.py # Heat exchangers
│   └── lle.py          # Liquid-liquid extraction
├── dynamic/            # Dynamic (transient) simulation
│   ├── state.py        # State variable specification
│   ├── base.py         # DynamicUnit protocol
│   ├── integrators.py  # ODE integration (RK4, RK45)
│   ├── flowsheet.py    # Multi-unit dynamic flowsheets
│   ├── dae.py          # DAE systems and Newton solver
│   └── diffrax_backend.py # Advanced solvers via diffrax
└── economics/          # Technoeconomic analysis
    ├── capital.py      # Equipment costs
    ├── utilities.py    # Utility costs
    ├── opex.py         # Operating costs
    ├── profitability.py # Financial metrics
    └── indices.py      # Cost indices

difflow_bio/
└── units/              # Bio-manufacturing operations
    ├── bioreactors.py  # Bioreactors
    ├── centrifuge.py   # Centrifugation
    ├── filtration.py   # Membrane filtration
    └── chromatography.py # Chromatography
```

## Quick Example

```python
import jax.numpy as jnp
from difflow.streams import make_stream
from difflow.units.cstr import CSTR, CSTRParams
from difflow.thermo import IdealThermo, SpeciesData

# Define species
species_data = {
    'A': SpeciesData(name='A', MW=50.0, Cp_coeffs=(30.0, 0.01, 0.0, 0.0),
                    Hvap_coeffs=(35000.0, 0.38, 500.0), antoine_coeffs=(10.0, 1500.0, -40.0),
                    Hf=0.0, Tref=298.15),
    'B': SpeciesData(name='B', MW=100.0, Cp_coeffs=(40.0, 0.02, 0.0, 0.0),
                    Hvap_coeffs=(40000.0, 0.38, 550.0), antoine_coeffs=(10.5, 1800.0, -50.0),
                    Hf=-50000.0, Tref=298.15)
}
thermo = IdealThermo(species_data)

# Create inlet stream
inlet = make_stream({'A': 1.0, 'B': 0.0}, T=350.0, P=101325.0)

# Define CSTR parameters
params = CSTRParams(
    volume=1.0,  # m³
    stoichiometry=jnp.array([[-1.0], [1.0]]),  # A -> B
    k_ref=0.1,
    E_a=50000.0,
    T_ref=350.0,
    dH_rxn=jnp.array([-50000.0])
)

# Create and run CSTR
cstr = CSTR(params, thermo, species_order=['A', 'B'])
outlet, info = cstr(inlet, T_spec=350.0)

print(f"Conversion: {info['conversion']:.2%}")
print(f"Heat duty: {info['Q']:.2f} W")
```

## Design Philosophy

### JAX-First Approach

Every calculation in Difflow is designed to be compatible with JAX automatic differentiation:

```python
import jax

# Gradient of outlet temperature with respect to inlet temperature
grad_fn = jax.grad(lambda T_in: cstr(make_stream({'A': 1.0}, T=T_in, P=101325.0))[0]['T'])
sensitivity = grad_fn(350.0)
```

### Consistent Interface

All unit operations follow the same calling convention:

```python
outlet, info = unit_operation(inlet, **operating_params)
```

Where:
- `inlet` is a Stream dictionary
- `outlet` is the output Stream
- `info` contains additional results (heat duties, conversions, etc.)

### Pytree Compatibility

Streams are JAX pytrees, allowing seamless use with `jax.vmap`, `jax.jit`, and other transformations.

## Requirements

- Python >= 3.9
- JAX >= 0.4.0
- NumPy
- PyYAML (for Cantera import)

## License

MIT License
