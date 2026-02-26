# difflow_ree - Rare Earth Element Separation Plugin

A comprehensive plugin for modeling and optimizing rare earth element (REE) solvent extraction processes using JAX-based automatic differentiation.

## Installation

```bash
pip install difflow[ree]
# or for development
pip install -e ".[ree]"
```

## Submodules

### `database.py`
Property databases for REE separation:
- **REEDatabase**: Properties for 10 commercial rare earth elements (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y)
- **ExtractantDatabase**: Properties for 4 extractant systems (D2EHPA, PC88A, Cyanex272, TBP)
- **SeparationFactorDatabase**: Element pair separation factors
- Convenience functions: `get_element()`, `get_extractant()`, `get_separation_factor()`
- JAX-compatible accessors: `get_atomic_weight_array()`, `get_price_array()`, `get_ionic_radius_array()`

### `equilibrium/`
Thermodynamic equilibrium models for solvent extraction:
- **distribution.py**: `REEDistribution` - pH-dependent distribution coefficient models
- **loading.py**: `LoadingIsotherm` - Organic phase loading corrections, Langmuir isotherms
- **speciation.py**: `REESpeciation` - Aqueous speciation (sulfate, chloride complexes)

### `units/`
Unit operation models (all differentiable):
- **extraction.py**: `REEExtractor`, `REEMixerSettler` - Multi-stage extraction cascades
- **scrubbing.py**: `REEScrubber` - Impurity removal sections
- **stripping.py**: `REEStripper` - Product recovery from organic phase
- **precipitation.py**: `OxalatePrecipitator`, `CarbonatePrecipitator`, `HydroxidePrecipitator` - Solid product recovery
- **cerium.py**: `CeriumOxidizer` - Selective Ce(IV) oxidation and CeO2 precipitation

### `flowsheets/`
Pre-built flowsheet templates for common configurations:
- **extract_strip.py**: `ExtractStripCircuit` - Basic 2-section circuit
- **extract_scrub_strip.py**: `ExtractScrubStripCircuit` - Industrial 3-section circuit
- **split_shell.py**: `SplitShellCascade` - Multi-product split-shell design
- **full_train.py**: `FullSeparationTrain`, `GroupSeparator` - Complete separation train

### `economics/`
Technoeconomic analysis:
- **costs.py**: `REEPricing`, `ReagentCosts`, `OperatingCosts`
- Functions: `estimate_capex()`, `estimate_opex()`, `calculate_revenue()`, `calculate_profit()`, `minimum_selling_price()`

### `data/`
YAML data files with element properties, extractant parameters, and separation factors.

## Quick Start

```python
from difflow_ree import (
    get_element, get_extractant,
    ExtractStripCircuit, ExtractStripParams,
)
from difflow.streams import make_stream

# Get neodymium properties
nd = get_element('Nd')
print(f"Nd price: ${nd.price_usd_kg}/kg")
print(f"Atomic weight: {nd.atomic_weight} g/mol")

# Get D2EHPA extractant
d2ehpa = get_extractant('D2EHPA')
print(f"pKa: {d2ehpa.pKa}")

# Create extraction circuit
params = ExtractStripParams(
    extractant='D2EHPA',
    elements=('La', 'Ce', 'Nd', 'Dy'),
    n_extraction_stages=10,
    n_stripping_stages=5,
)
circuit = ExtractStripCircuit(params)

# Create aqueous feed stream with REE
feed = make_stream(
    flows={"H2O": 10.0, "La": 0.01, "Ce": 0.02, "Nd": 0.02, "Dy": 0.01},
    T=298.15,
    P=101325.0,
)

# Run the circuit (solvent and strip acid are created internally)
results = circuit(feed)
print(f"Overall recovery: {results['recovery']:.3f}")
print(f"Element recoveries: {results['element_recovery']}")
```

## Key Features

- **Fully differentiable**: All models compatible with JAX's `grad`, `jit`, `vmap`
- **Multi-element**: Handles mixtures of any REE combination
- **pH-dependent**: Realistic extraction chemistry with pH effects
- **Loading corrections**: Accounts for organic phase saturation
- **Explicit organic phase**: Extractant + diluent species keys (e.g., `"D2EHPA"` + `"kerosene"`) enable multi-organic flowsheets
- **Pre-built flowsheets**: Common industrial configurations ready to use
- **Economic analysis**: Revenue, profit, and minimum selling price calculations

## Supported Elements

| Element | Symbol | Atomic # | Category |
|---------|--------|----------|----------|
| Lanthanum | La | 57 | Light |
| Cerium | Ce | 58 | Light |
| Praseodymium | Pr | 59 | Light |
| Neodymium | Nd | 60 | Light |
| Samarium | Sm | 62 | Middle |
| Europium | Eu | 63 | Middle |
| Gadolinium | Gd | 64 | Middle |
| Terbium | Tb | 65 | Heavy |
| Dysprosium | Dy | 66 | Heavy |
| Yttrium | Y | 39 | Heavy |

## Supported Extractants

| Extractant | Type | Best For |
|------------|------|----------|
| D2EHPA | Acidic | Light REE separation |
| PC88A | Acidic | General REE separation |
| Cyanex272 | Acidic | Heavy REE, Co/Ni |
| TBP | Neutral | Nitrate systems |

## References

- Xie F et al. (2014). A critical review on solvent extraction of rare earths. Miner. Eng. 56:10-28
- Krishnamurthy N, Gupta CK (2004). Extractive Metallurgy of Rare Earths
