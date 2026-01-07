# Getting Started

This guide will help you get started with Difflow, a JAX-based framework for building and optimizing chemical process flowsheets.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Basic Examples](#basic-examples)
5. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- JAX 0.4.0 or higher

### Install from Source

```bash
git clone https://github.com/jkitchin/differentiable-flowsheets.git
cd differentiable-flowsheets
pip install -e .
```

### Dependencies

Core dependencies are installed automatically:

```
jax>=0.4.0
jaxlib>=0.4.0
numpy
pyyaml
```

For bio-manufacturing operations:

```bash
pip install -e ".[bio]"
```

---

## Quick Start

### Your First Simulation

Let's simulate a simple CSTR reactor for the reaction A → B:

```python
import jax.numpy as jnp
from difflow.streams import make_stream
from difflow.units.cstr import CSTR, CSTRParams
from difflow.thermo import IdealThermo, SpeciesData

# 1. Define species properties
species_data = {
    'A': SpeciesData(
        name='A',
        MW=50.0,                              # g/mol
        Cp_coeffs=(30.0, 0.01, 0.0, 0.0),    # J/mol/K polynomial
        Hvap_coeffs=(35000.0, 0.38, 500.0),  # Watson correlation
        antoine_coeffs=(10.0, 1500.0, -40.0), # Vapor pressure
        Hf=0.0,                               # Heat of formation (J/mol)
        Tref=298.15                           # Reference T (K)
    ),
    'B': SpeciesData(
        name='B',
        MW=50.0,
        Cp_coeffs=(35.0, 0.015, 0.0, 0.0),
        Hvap_coeffs=(38000.0, 0.38, 520.0),
        antoine_coeffs=(10.2, 1600.0, -45.0),
        Hf=-50000.0,
        Tref=298.15
    )
}

# 2. Create thermodynamic model
thermo = IdealThermo(species_data)

# 3. Define reactor parameters
params = CSTRParams(
    volume=1.0,                           # m³
    stoichiometry=jnp.array([[-1.0],      # A consumed
                             [1.0]]),      # B produced
    k_ref=0.1,                            # Rate constant at T_ref (1/s)
    E_a=50000.0,                          # Activation energy (J/mol)
    T_ref=350.0,                          # Reference temperature (K)
    dH_rxn=jnp.array([-50000.0])         # Exothermic (J/mol)
)

# 4. Create the reactor
cstr = CSTR(params, thermo, species_order=['A', 'B'])

# 5. Create inlet stream
inlet = make_stream(
    flows={'A': 1.0, 'B': 0.0},  # mol/s
    T=350.0,                      # K
    P=101325.0                    # Pa
)

# 6. Run simulation (isothermal at 350 K)
outlet, info = cstr(inlet, T_spec=350.0)

# 7. View results
print(f"Inlet A flow: {inlet['F_A']:.4f} mol/s")
print(f"Outlet A flow: {outlet['F_A']:.4f} mol/s")
print(f"Outlet B flow: {outlet['F_B']:.4f} mol/s")
print(f"Conversion: {info['conversion']:.2%}")
print(f"Heat duty: {info['Q']:.2f} W")
```

Output:
```
Inlet A flow: 1.0000 mol/s
Outlet A flow: 0.4762 mol/s
Outlet B flow: 0.5238 mol/s
Conversion: 52.38%
Heat duty: -26190.48 W
```

---

## Core Concepts

### Streams

Streams are Python dictionaries containing molar flows, temperature, and pressure:

```python
from difflow.streams import make_stream, total_flow, mole_fractions

# Create a stream
stream = make_stream(
    flows={'methane': 10.0, 'ethane': 5.0, 'propane': 2.0},
    T=300.0,    # K
    P=500000.0  # Pa (5 bar)
)

# Access properties
print(stream['F_methane'])  # 10.0 mol/s
print(stream['T'])          # 300.0 K
print(stream['P'])          # 500000.0 Pa

# Calculate derived properties
total = total_flow(stream)  # 17.0 mol/s
x = mole_fractions(stream)  # {'methane': 0.588, 'ethane': 0.294, 'propane': 0.118}
```

### Unit Operations

All unit operations follow a consistent interface:

```python
outlet, info = unit_operation(inlet, **params)
```

Where:
- `inlet`: Input stream (or streams for multi-input operations)
- `params`: Operating parameters (temperature, pressure, etc.)
- `outlet`: Output stream (or streams)
- `info`: Dictionary with additional results

### Thermodynamic Models

Two levels of thermodynamic modeling are available:

```python
# Ideal thermodynamics (fast, simple systems)
from difflow.thermo import IdealThermo

thermo = IdealThermo(species_data)
K = thermo.K_values(T=350.0, P=101325.0)  # Raoult's law K-values

# Cubic equation of state (accurate, high-pressure)
from difflow.eos import PengRobinson

pr = PengRobinson(critical_properties)
K = pr.K_value('methane', T=200.0, P=3e6)  # Fugacity-based K-value
```

### Automatic Differentiation

All calculations are JAX-compatible, enabling automatic differentiation:

```python
import jax

# Define a function to differentiate
def conversion_vs_temperature(T):
    inlet = make_stream({'A': 1.0, 'B': 0.0}, T=T, P=101325.0)
    outlet, info = cstr(inlet, T_spec=T)
    return info['conversion']

# Compute gradient
grad_fn = jax.grad(conversion_vs_temperature)
sensitivity = grad_fn(350.0)
print(f"dX/dT at 350K: {sensitivity:.4f} 1/K")
```

---

## Basic Examples

### Example 1: Flash Drum Separation

```python
from difflow.units.flash import Flash
from difflow.database import get_species_data

# Get species from database
species = ['benzene', 'toluene', 'xylene']
species_data = {s: get_species_data(s) for s in species}
thermo = IdealThermo(species_data)

# Create flash drum
flash = Flash(thermo, species)

# Feed stream
feed = make_stream(
    {'benzene': 0.4, 'toluene': 0.35, 'xylene': 0.25},
    T=380.0,
    P=101325.0
)

# Perform flash calculation
liquid, vapor, info = flash(feed)

print(f"Vapor fraction: {info['V_frac']:.3f}")
print(f"K-values: {info['K_values']}")
print(f"Liquid benzene: {info['x']['benzene']:.3f}")
print(f"Vapor benzene: {info['y']['benzene']:.3f}")
```

### Example 2: Heat Exchanger

```python
from difflow.units.heat_exchanger import CounterCurrentHX, HeatExchangerParams

# Create counter-current heat exchanger
hx_params = HeatExchangerParams(mode='rating', UA=5000.0)
hx = CounterCurrentHX(hx_params, thermo, species)

# Hot and cold streams
hot_in = make_stream({'benzene': 1.0}, T=450.0, P=101325.0)
cold_in = make_stream({'toluene': 0.8}, T=300.0, P=101325.0)

# Run heat exchanger
hot_out, cold_out, info = hx(hot_in, cold_in)

print(f"Heat duty: {info['Q']/1000:.2f} kW")
print(f"Hot outlet T: {hot_out['T']:.1f} K")
print(f"Cold outlet T: {cold_out['T']:.1f} K")
print(f"LMTD: {info['LMTD']:.2f} K")
```

### Example 3: Simple Process Flowsheet

```python
from difflow.flowsheet import Flowsheet, Unit
from difflow.units.cstr import CSTR
from difflow.units.flash import Flash

# Create flowsheet
fs = Flowsheet()

# Add feed
fs.add_feed('raw_feed', make_stream({'A': 1.0, 'B': 0.0}, T=350.0, P=101325.0))

# Add units
fs.add_unit(Unit(
    name='reactor',
    operation=cstr,
    inlet_names=['raw_feed'],
    outlet_names=['reactor_out'],
    params={'T_spec': 380.0}
))

fs.add_unit(Unit(
    name='flash',
    operation=flash,
    inlet_names=['reactor_out'],
    outlet_names=['vapor', 'liquid'],
    params={}
))

# Solve flowsheet
results = fs.solve()

print(f"Reactor outlet: {results['reactor_out']}")
print(f"Flash vapor: {results['vapor']}")
print(f"Flash liquid: {results['liquid']}")
```

### Example 4: Economic Analysis

```python
from difflow.economics.capital import reactor_cost, total_capital_investment
from difflow.economics.profitability import full_cash_flow_analysis

# Equipment cost
equip_cost = reactor_cost(volume=10.0, reactor_type='cstr_jacketed')
print(f"Reactor purchase cost: ${equip_cost:,.0f}")

# Total capital investment
tci = total_capital_investment(equip_cost)
print(f"Total capital investment: ${tci:,.0f}")

# Profitability
result = full_cash_flow_analysis(
    FCI=tci * 0.85,
    annual_revenue=500000,
    annual_OPEX=350000,
    tax_rate=0.21,
    discount_rate=0.10,
    plant_life=20
)

print(f"NPV: ${result.npv:,.0f}")
print(f"IRR: {result.irr:.1%}")
print(f"Payback: {result.payback_period:.1f} years")
```

### Example 5: Optimization

```python
import jax
import jax.numpy as jnp
from jax import grad

# Define base parameters once
base_params = CSTRParams(
    volume=1.0,  # Will be optimized
    stoichiometry=jnp.array([[-1.0], [1.0]]),
    k_ref=0.1,
    E_a=50000.0,
    T_ref=350.0,
    dH_rxn=jnp.array([-50000.0])
)

# Objective: Maximize conversion while minimizing heat duty
def objective(opt_params):
    T_reactor, volume = opt_params

    # Use update() to create new params - JAX compatible!
    reactor_params = base_params.update(volume=volume)
    reactor = CSTR(reactor_params, thermo, ['A', 'B'])

    inlet = make_stream({'A': 1.0, 'B': 0.0}, T=350.0, P=101325.0)
    outlet, info = reactor(inlet, T_spec=T_reactor)

    # Maximize conversion, minimize heating cost
    conversion = info['conversion']
    heat_cost = jnp.abs(info['Q']) * 0.0001  # $/W

    return conversion - heat_cost  # Maximize this

# Gradient-based optimization
grad_obj = grad(objective)

opt_params = jnp.array([350.0, 1.0])  # Initial: T=350K, V=1m³
learning_rate = 0.1

for i in range(50):
    gradient = grad_obj(opt_params)
    opt_params = opt_params + learning_rate * gradient

    if i % 10 == 0:
        print(f"Iter {i}: T={opt_params[0]:.1f}K, V={opt_params[1]:.2f}m³, obj={objective(opt_params):.4f}")

# Access parameters with dict-like syntax
print(f"Base volume: {base_params['volume']}")  # Read access
```

---

## Next Steps

### Learn More

1. **[Unit Operations - Chemical](unit-operations-chemical.md)**: Detailed documentation of reactors, separators, heat exchangers, and distillation
2. **[Unit Operations - Bio](unit-operations-bio.md)**: Bioreactors, centrifugation, filtration, and chromatography
3. **[Thermodynamics](thermodynamics.md)**: Property calculations, equations of state, and databases
4. **[Technoeconomics](technoeconomics.md)**: Capital costs, operating costs, and profitability analysis
5. **[Streams and Flowsheets](streams-and-flowsheets.md)**: Stream handling and flowsheet management
6. **[Solvers and Utilities](solvers-and-utilities.md)**: Numerical methods and uncertainty propagation

### Example Notebooks

Check the `examples/` directory for complete worked examples:

- `reactor_optimization.ipynb`: Reactor design optimization
- `distillation_design.ipynb`: Shortcut distillation column design
- `heat_exchanger_network.ipynb`: Heat integration example
- `mab_downstream.ipynb`: Monoclonal antibody purification
- `tea_analysis.ipynb`: Complete technoeconomic analysis

### Common Patterns

#### Using the Database

```python
from difflow.database import get_species_data, get_critical_props, list_species

# See available species
print(list_species())

# Build thermo model from database
species = ['methanol', 'water', 'DME']
species_data = {s: get_species_data(s) for s in species}
thermo = IdealThermo(species_data)

# For equation of state
critical = {s: get_critical_props(s) for s in species}
pr = PengRobinson(critical)
```

#### Vectorization with vmap

```python
from jax import vmap

# Run simulation at multiple temperatures
temperatures = jnp.linspace(300, 450, 50)

def sim_at_T(T):
    inlet = make_stream({'A': 1.0, 'B': 0.0}, T=T, P=101325.0)
    _, info = cstr(inlet, T_spec=T)
    return info['conversion']

# Vectorize over temperature
conversions = vmap(sim_at_T)(temperatures)
```

#### JIT Compilation

```python
from jax import jit

# JIT compile for speed
@jit
def fast_simulation(inlet_T):
    inlet = make_stream({'A': 1.0, 'B': 0.0}, T=inlet_T, P=101325.0)
    outlet, info = cstr(inlet, T_spec=inlet_T)
    return outlet['F_B'], info['Q']

# First call compiles, subsequent calls are fast
F_B, Q = fast_simulation(350.0)
```

---

## Troubleshooting

### Common Issues

**1. JAX not finding GPU**
```python
# Check JAX devices
import jax
print(jax.devices())  # Should show GPU if available
```

**2. Numerical instability in flash calculations**
- Ensure feed composition sums to 1.0
- Check that temperature is between bubble and dew points
- Try adjusting solver tolerances

**3. Reactor not converging**
- Reduce time step for PFR integration
- Check for unreasonable kinetic parameters
- Ensure positive concentrations

### Getting Help

- Check the documentation in `docs/`
- Review examples in `examples/`
- Open an issue on GitHub
