# difflow_cc - Carbon Capture Plugin

A comprehensive plugin for modeling and optimizing carbon capture processes using JAX-based automatic differentiation.

## Installation

```bash
pip install difflow[cc]
# or for development
pip install -e ".[cc]"
```

## Submodules

### `database.py`
Material property databases for carbon capture systems:
- **AmineSolvent**: Properties for 7 amine solvents (MEA, DEA, MDEA, PZ, AMP, amino acids)
- **Adsorbent**: Properties for 8 adsorbent materials with isotherm parameters
- **Membrane**: Properties for 9 membrane materials with permeability data
- Convenience functions: `get_solvent()`, `get_adsorbent()`, `get_membrane()`

### `equilibrium/`
Thermodynamic equilibrium models:
- **isotherms.py**: Langmuir, Sips, Toth, dual-site Langmuir (with temperature dependence)
- **vle.py**: Amine-CO2 vapor-liquid equilibrium, Henry's law constants
- **solubility.py**: CO2 physical solubility, diffusivity in amine solutions

### `kinetics/`
Reaction and mass transfer kinetics:
- **amine_kinetics.py**: Reaction rate constants, enhancement factors, Hatta numbers
- **mass_transfer.py**: Gas/liquid film coefficients, overall mass transfer, interfacial area

### `units/`
Unit operation models (all differentiable):
- **absorber.py**: `AmineAbsorber` - Equilibrium-stage amine absorption column
- **stripper.py**: `AmineStripper` - Amine regeneration/stripping column
- **membrane.py**: `MembraneSeparator`, `MultistageMembrane` - Gas separation membranes
- **adsorption.py**: `PSAUnit`, `TSAUnit`, `VSAUnit`, `TVSAUnit` - Swing adsorption cycles
- **dac.py**: `SolidSorbentDAC`, `LiquidSolventDAC` - Direct air capture units
- **compression.py**: `Compressor`, `CompressionTrain`, `Pump` - CO2 compression
- **heat_integration.py**: `LeanRichExchanger`, `HeatRecoverySystem`, `Intercooler`

### `economics/`
Technoeconomic analysis tools:
- **capex.py**: Equipment cost correlations (absorber, stripper, compressor, membrane modules)
- **opex.py**: Operating costs (steam, electricity, solvent makeup, maintenance)
- **levelized_cost.py**: Levelized cost of capture, cost of CO2 avoided, NPV, IRR

### `integration/`
Process integration with power plants:
- **power_plant.py**: `PowerPlantIntegration` - Flue gas composition, efficiency penalties
- **steam_cycle.py**: `SteamCycleParams` - Steam extraction for reboiler duty

### `degradation/`
Long-term performance degradation models:
- **amine_degradation.py**: Oxidative, thermal, and CO2-induced degradation
- **adsorbent_degradation.py**: Thermal cycling, hydrothermal stability, capacity fade
- **membrane_aging.py**: Physical aging, plasticization, fouling

### `data/`
YAML data files with material properties and parameters.

### `examples/`
Jupyter notebook tutorials demonstrating usage.

## Quick Start

```python
from difflow_cc import (
    get_solvent, get_adsorbent,
    AmineAbsorber, AbsorberParams,
    PSAUnit, AdsorptionParams,
    levelized_cost_capture,
)

# Get MEA solvent properties
mea = get_solvent('MEA')
print(f"Heat of absorption: {mea.heat_of_absorption} kJ/mol")

# Create amine absorber
params = AbsorberParams(
    solvent='MEA',
    n_stages=10,
    solvent_flow=100.0,  # mol/s
)
absorber = AmineAbsorber(params)
rich_solvent, clean_gas = absorber(flue_gas, lean_solvent)

# All operations support automatic differentiation
from jax import grad
d_capture_d_flow = grad(lambda flow: capture_rate(flow))(100.0)
```

## Key Features

- **Fully differentiable**: All models compatible with JAX's `grad`, `jit`, `vmap`
- **Database-driven**: Extensive property databases for amines, adsorbents, membranes
- **Multi-technology**: Supports amine absorption, membrane, and adsorption processes
- **Economics included**: CAPEX, OPEX, and levelized cost calculations
- **Degradation modeling**: Long-term performance prediction

## References

- Rochelle GT (2009). Amine scrubbing for CO2 capture. Science 325:1652
- Ruthven DM (1984). Principles of Adsorption and Adsorption Processes
- Baker RW (2012). Membrane Technology and Applications, 3rd ed.
