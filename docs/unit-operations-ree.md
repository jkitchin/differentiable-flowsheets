# Rare Earth Element (REE) Unit Operations

This document provides comprehensive documentation for the `difflow_ree` plugin, which provides specialized tools for modeling and optimizing rare earth element solvent extraction processes.

---

## Overview

The `difflow_ree` plugin provides:

- **Database** of 10 commercial REE properties (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y)
- **4 extractant systems**: D2EHPA, PC88A, Cyanex272, TBP
- **pH-dependent distribution coefficient models**
- **Loading and speciation corrections**
- **Unit operations**: extraction, scrubbing, stripping, precipitation
- **Pre-built flowsheet templates**
- **Economic analysis tools**

All operations are fully differentiable using JAX, enabling gradient-based optimization of separation processes.

---

## Installation

The REE plugin is included as an optional dependency:

```bash
pip install difflow[ree]
```

Or install with all extras:

```bash
pip install difflow[all]
```

---

## Database and Properties

(ree-element-database)=
### REE Element Database

Access REE properties using the database functions:

```python
from difflow_ree import get_element, list_ree_elements

# List available elements
print(list_ree_elements())
# ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Y']

# Get element properties
nd = get_element("Nd")
print(f"Atomic weight: {nd.atomic_weight}")
print(f"Ionic radius: {nd.ionic_radius} pm")
print(f"Price: ${nd.price_usd_kg}/kg")
```

#### Available Properties

| Property | Description | Units |
|----------|-------------|-------|
| `atomic_weight` | Atomic mass | g/mol |
| `ionic_radius` | Ionic radius (3+) | pm |
| `price_usd_kg` | Market price | USD/kg |
| `oxide_mw` | Oxide molecular weight | g/mol |
| `oxide_formula` | Oxide formula | - |

(extractant-database)=
### Extractant Database

Four industrial extractants are supported:

```python
from difflow_ree import get_extractant, list_extractants

print(list_extractants())
# ['D2EHPA', 'PC88A', 'Cyanex272', 'TBP']

d2ehpa = get_extractant("D2EHPA")
print(f"Full name: {d2ehpa.full_name}")
print(f"Reference concentration: {d2ehpa.reference_concentration} M")
```

| Extractant | Full Name | Primary Use |
|------------|-----------|-------------|
| D2EHPA | Di(2-ethylhexyl)phosphoric acid | Light/middle REE |
| PC88A | 2-ethylhexyl phosphonic acid mono-2-ethylhexyl ester | Middle REE |
| Cyanex272 | Bis(2,4,4-trimethylpentyl)phosphinic acid | Heavy REE, Co/Ni |
| TBP | Tri-n-butyl phosphate | Ce separation, nuclear |

---

## Equilibrium Models

(distribution-coefficients)=
### Distribution Coefficients

The distribution coefficient D = [REE]_org / [REE]_aq is modeled as a function of pH, temperature, and extractant concentration:

$$\log_{10}(D) = a + b \cdot pH + c \cdot pH^2 + \frac{\Delta H}{R \ln(10)} \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)$$

```python
from difflow_ree import REEDistribution, get_distribution_coefficient

# Create distribution calculator
dist = REEDistribution(
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    concentration=0.5,  # M
)

# Get D value for Nd at pH 3.0
D_nd = dist.get_D("Nd", pH=3.0, T=298.15)
print(f"D(Nd) at pH 3.0: {D_nd:.2f}")

# Get all D values
D_all = dist.get_D_all(pH=3.0)
for elem, D in D_all.items():
    print(f"D({elem}): {D:.2f}")
```

#### pH Dependence

Distribution coefficients are strongly pH-dependent. Higher pH generally increases extraction:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

dist = REEDistribution(extractant="D2EHPA", elements=("La", "Nd", "Dy"))

pH_range = jnp.linspace(1.0, 5.0, 50)
for elem in ["La", "Nd", "Dy"]:
    D_values = [float(dist.get_D(elem, pH)) for pH in pH_range]
    plt.semilogy(pH_range, D_values, label=elem)

plt.xlabel("pH")
plt.ylabel("Distribution Coefficient D")
plt.legend()
plt.grid(True)
```

(separation-factors)=
### Separation Factors

The separation factor SF = D1/D2 determines separation feasibility:

```python
from difflow_ree import REEDistribution

dist = REEDistribution(extractant="PC88A", elements=("Nd", "Pr"))

# Separation factor at pH 3.0
SF = dist.get_separation_factor("Nd", "Pr", pH=3.0)
print(f"SF(Nd/Pr) = {SF:.2f}")

# Find optimal pH for separation
opt_pH, max_SF = dist.optimal_pH_for_separation("Nd", "Pr", pH_range=(1.0, 5.0))
print(f"Optimal pH: {opt_pH:.2f}, Max SF: {max_SF:.2f}")
```

---

## Unit Operations

(reeextractor)=
### REEExtractor

**Location**: `difflow_ree/units/extraction.py`

**Class**: `REEExtractor`

**Description**: Multi-stage counter-current extraction cascade using the Kremser equation.

#### Parameters

```python
@dataclass
class REEExtractorParams:
    n_stages: int              # Number of extraction stages
    extractant: str            # Extractant name (D2EHPA, PC88A, etc.)
    elements: tuple[str, ...]  # REE elements to track
    pH: float = 3.0            # Operating pH
    extractant_conc: float = 0.5  # Extractant concentration (M)
    include_loading: bool = True  # Account for extractant loading
    include_speciation: bool = False  # Account for aqueous speciation
```

#### Usage

```python
from difflow_ree import REEExtractor, REEExtractorParams
from difflow.streams import make_stream

# Create extractor
params = REEExtractorParams(
    n_stages=10,
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    pH=3.0,
)
extractor = REEExtractor(params)

# Create feed and solvent streams
feed = make_stream({"H2O": 1.0, "La": 0.1, "Ce": 0.2, "Nd": 0.15, "Dy": 0.05}, T=298.15, P=101325.0)
solvent = make_stream({"Organic": 1.0}, T=298.15, P=101325.0)

# Run extraction
raffinate, extract, info = extractor(feed, solvent, T=298.15, pH=3.0)

# Check recoveries
for elem, data in info["profiles"].items():
    print(f"{elem}: Recovery = {data['recovery']:.1%}")
```

#### Governing Equations

**Kremser Equation** for counter-current extraction:

$$\frac{x_{out}}{x_{in}} = \frac{E - 1}{E^{N+1} - 1}$$

Where:
- $E = D \cdot (S/F)$ is the extraction factor
- $D$ is the distribution coefficient
- $S/F$ is the solvent-to-feed ratio
- $N$ is the number of stages

(reemixersettler)=
### REEMixerSettler

**Description**: Single mixer-settler stage for REE extraction with efficiency factor.

```python
@dataclass
class MixerSettlerParams:
    extractant: str
    elements: tuple[str, ...]
    pH: float = 3.0
    extractant_conc: float = 0.5
    mixer_residence_time: float = 120.0  # seconds
    settler_residence_time: float = 300.0  # seconds
    stage_efficiency: float = 0.95
```

(reescrubber)=
### REEScrubber

**Description**: Multi-stage scrubbing section for removing impurities from loaded organic.

Scrubbing uses lower pH to selectively strip lighter REE back to aqueous phase while retaining heavier REE in the organic.

```python
from difflow_ree import REEScrubber, ScrubberParams

params = ScrubberParams(
    n_stages=5,
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    target_elements=("Nd", "Dy"),  # Keep these in organic
    pH=2.0,  # Lower pH to reject La, Ce
)
scrubber = REEScrubber(params)
```

(reestripper)=
### REEStripper

**Description**: Multi-stage stripping section for product recovery.

Stripping uses very low pH (strong acid) to transfer all REE from organic back to aqueous phase.

```python
from difflow_ree import REEStripper, StripperParams

params = StripperParams(
    n_stages=5,
    extractant="D2EHPA",
    elements=("Nd", "Dy"),
    pH=0.5,  # Strong acid for complete stripping
)
stripper = REEStripper(params)
```

(ceriumoxidizer)=
### CeriumOxidizer

**Location**: `difflow_ree/units/cerium.py`

**Description**: Oxidizes Ce³⁺ to Ce⁴⁺ and precipitates as CeO₂.

Cerium is unique among lanthanides because it can be oxidized from Ce³⁺ to Ce⁴⁺, enabling selective removal.

#### Parameters

```python
@dataclass
class CeriumOxidizerParams:
    elements: tuple[str, ...]
    oxidant: str = "air"  # air, H2O2, NaOCl, electrolytic
    oxidant_excess: float = 2.0
    pH: float = 8.0  # Alkaline conditions favor oxidation
    temperature: float = 353.15  # 80°C typical
    ce_conversion: float = 0.95
```

#### Usage

```python
from difflow_ree import CeriumOxidizer, CeriumOxidizerParams

params = CeriumOxidizerParams(
    elements=("La", "Ce", "Pr", "Nd"),
    oxidant="air",
    pH=8.0,
    ce_conversion=0.95,
)
oxidizer = CeriumOxidizer(params)

# Run oxidation
filtrate, ceo2_solid, info = oxidizer(feed)

print(f"Ce conversion: {info['ce_conversion']:.1%}")
print(f"CeO2 produced: {info['ceo2_mass_kg_s']:.4f} kg/s")
```

---

(precipitation-operations)=
## Precipitation Operations

(oxalateprecipitator)=
### OxalatePrecipitator

**Description**: Precipitates REE as oxalate, which can be calcined to oxide.

**Reaction**: 2REE³⁺ + 3C₂O₄²⁻ → REE₂(C₂O₄)₃↓

```python
from difflow_ree import OxalatePrecipitator, PrecipitatorParams

params = PrecipitatorParams(
    elements=("Nd", "Dy"),
    precipitant_excess=1.5,  # 50% excess
    target_conversion=0.995,
)
precipitator = OxalatePrecipitator(params)

# Feed is stripped REE solution, precipitant is oxalic acid
filtrate, solid, info = precipitator(feed, oxalic_acid)

print(f"Total precipitated: {info['total_precipitated']:.4f} mol/s")
print(f"Solid composition: {info['solid_composition']}")
```

(carbonateprecipitator)=
### CarbonatePrecipitator

**Reaction**: 2REE³⁺ + 3CO₃²⁻ → REE₂(CO₃)₃↓

Used for group precipitation from leach solutions.

(hydroxideprecipitator)=
### HydroxidePrecipitator

**Reaction**: REE³⁺ + 3OH⁻ → REE(OH)₃↓

Hydroxide precipitation can be selective based on pH - heavy REE precipitate at lower pH than light REE.

```python
from difflow_ree import HydroxidePrecipitator, PrecipitatorParams

params = PrecipitatorParams(elements=("La", "Ce", "Nd", "Dy"))
precipitator = HydroxidePrecipitator(params)

# pH-selective precipitation
filtrate, solid, info = precipitator(feed, naoh_solution, pH=8.5)

# Find selective precipitation pH range
min_pH, max_pH = precipitator.selective_precipitation_pH("Dy", "La")
print(f"pH range for Dy/La separation: {min_pH:.1f} - {max_pH:.1f}")
```

---

(flowsheet-templates)=
## Flowsheet Templates

(extractstripcircuit)=
### ExtractStripCircuit

**Description**: Basic 2-section circuit for simple separations.

```
    Feed                    Product
      ↓                        ↑
┌───────────┐         ┌───────────┐
│           │         │           │
│ EXTRACTION│ ──Org──▶│ STRIPPING │
│           │         │           │
└───────────┘         └───────────┘
      ↓                     ↓
  Raffinate            Strip Acid
```

(extractscrubstripcircuit)=
### ExtractScrubStripCircuit

**Description**: Industrial 3-section circuit for high-purity separations.

```
    Feed                    Scrub                   Product
      ↓                       ↓                        ↑
┌───────────┐         ┌───────────┐         ┌───────────┐
│           │         │           │         │           │
│ EXTRACTION│ ──Org──▶│ SCRUBBING │ ──Org──▶│ STRIPPING │
│           │         │           │         │           │
└───────────┘         └───────────┘         └───────────┘
      ↓                     ↓          ◀──Org──    ↓
  Raffinate            Scrub Liquor          Strip Acid
```

#### Parameters

```python
@dataclass
class ExtractScrubStripParams:
    extractant: str
    elements: tuple[str, ...]
    target_elements: tuple[str, ...]  # Elements to recover
    n_extraction_stages: int = 10
    n_scrubbing_stages: int = 5
    n_stripping_stages: int = 5
    extraction_pH: float = 3.5
    scrubbing_pH: float = 2.0  # Lower pH rejects light REE
    stripping_pH: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    scrub_to_solvent_ratio: float = 0.2
    strip_to_solvent_ratio: float = 0.5
```

#### Usage

```python
from difflow_ree import ExtractScrubStripCircuit, ExtractScrubStripParams
from difflow.streams import make_stream

params = ExtractScrubStripParams(
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    target_elements=("Nd", "Dy"),
    n_extraction_stages=10,
    n_scrubbing_stages=5,
    n_stripping_stages=5,
    extraction_pH=3.5,
    scrubbing_pH=2.0,
    stripping_pH=0.5,
)
circuit = ExtractScrubStripCircuit(params)

# Create feed
feed = make_stream({
    "H2O": 1.0,
    "La": 0.10,
    "Ce": 0.20,
    "Nd": 0.15,
    "Dy": 0.05,
}, T=298.15, P=101325.0)

# Run circuit
results = circuit(feed)

print(f"Target purity: {results['target_purity']:.1%}")
for elem, recovery in results['target_recovery'].items():
    print(f"{elem} recovery: {recovery:.1%}")
```

(splitshellcascade)=
### SplitShellCascade

**Description**: Multi-product split-shell cascade for producing multiple pure REE streams.

---

(custom-elements-and-data)=
## Custom Elements and Data

The built-in database covers 10 commercial REEs and 4 extractant systems, but many applications require elements or extractant data not included by default. The `difflow_ree` plugin provides a runtime API for adding your own literature data, following the same pattern as the existing `create_custom_extractant` / `add_extractant` workflow.

(adding-a-custom-element)=
### Adding a Custom Element

Use `create_custom_element` to build an `REEElement` from known physical properties, then register it with the element database. All physical constants (atomic weight, ionic radius, density, melting point) should come from standard references such as the CRC Handbook or Shannon (1976) ionic radii tables.

```python
from difflow_ree import create_custom_element, get_ree_database

# Create Holmium from literature data
ho = create_custom_element(
    symbol="Ho",
    name="Holmium",
    atomic_number=67,
    atomic_weight=164.930,   # g/mol, CRC Handbook
    ionic_radius_pm=90.1,    # pm, Shannon (1976), CN=6, 3+
    density=8.795,           # g/cm³
    melting_point=1734,      # K
    group="heavy",
    oxide_formula="Ho2O3",
    oxide_mw=377.86,         # g/mol
    price_usd_kg=60.0,      # approximate market price
)

# Register with the database
db = get_ree_database()
db.add_element("Ho", ho)

# Now Ho is available alongside built-in elements
print(db.get("Ho").ionic_radius_pm)  # 90.1
print(db.list_by_group("heavy"))     # [..., 'Ho']
```

Elements can also be updated or removed:

```python
db.update_element("Ho", updated_ho)  # replace with corrected data
db.remove_element("Ho")              # remove entirely
```

(adding-extractant-coefficients-for-a-new-element)=
### Adding Extractant Coefficients for a New Element

After registering an element, you need to provide its pH and temperature coefficients for at least one extractant before it can be used in extraction simulations. These coefficients are empirical and should come from published experimental correlations (e.g., Gupta & Krishnamurthy, 2005; Xie et al., 2014).

You only need to add data for the extractants you plan to use. For example, to add Ho data for PC88A only:

```python
from difflow_ree import get_extractant_database

ext_db = get_extractant_database()

# Add Ho coefficients to PC88A
# Model: log10(D) = a + b*pH + c*pH^2 + d/T
ext_db.add_element_to_extractant(
    "PC88A",
    "Ho",
    ph_coefficients={
        "a": -6.15,   # from your literature source
        "b": 2.95,
        "c": 0.010,
    },
    temperature_coefficient=-2350,  # K, for d*(1/T - 1/T_ref) correction
)

# Verify
extractant = ext_db.get("PC88A")
print("Ho" in extractant.ph_coefficients)  # True

# Other extractants are unaffected
print("Ho" in ext_db.get("D2EHPA").ph_coefficients)  # False
```

If you need to correct values, remove and re-add:

```python
ext_db.remove_element_from_extractant("PC88A", "Ho")
ext_db.add_element_to_extractant("PC88A", "Ho", ...)
```

(adding-separation-factors)=
### Adding Separation Factors

Separation factor data can be added incrementally. You can add individual pairs to existing extractants or create complete entries for new ones.

Adding pairs to an existing extractant:

```python
from difflow_ree import get_sf_database

sf_db = get_sf_database()

# Add Ho separation factors to PC88A (from literature)
sf_db.add_pair("PC88A", "Ho_Dy", 1.4, adjacent=True, stages_99=20)
sf_db.add_pair("PC88A", "Y_Ho", 0.9, adjacent=True)

# Add a non-adjacent group pair
sf_db.add_pair("PC88A", "Ho_Nd", 10.5, adjacent=False)

# Query the new data
print(sf_db.get_sf("PC88A", "Ho_Dy"))           # 1.4
print(sf_db.get_stages_needed("PC88A", "Ho_Dy"))  # 20
```

Creating a complete entry for a new or custom extractant:

```python
sf_db.add_separation_factors(
    extractant="MyExtractant",
    conditions={"pH": 3.0, "temperature_K": 298, "concentration_M": 0.5},
    adjacent_pairs={"Ho_Dy": 1.4, "Y_Ho": 0.9},
    group_pairs={"Ho_La": 50.0},
    stages_for_99_purity={"Ho_Dy": 20},
)
```

(custom-data-complete-workflow)=
### Complete Workflow

Here is a full example of adding Holmium and using it in a separation simulation. In a real application, the pH coefficients and separation factors should come from published experimental data for your specific extractant system.

```python
from difflow_ree import (
    create_custom_element,
    get_ree_database,
    get_extractant_database,
    get_sf_database,
    REEDistribution,
)

# 1. Register the element
ho = create_custom_element(
    symbol="Ho", name="Holmium", atomic_number=67,
    atomic_weight=164.930, ionic_radius_pm=90.1, density=8.795,
    melting_point=1734, group="heavy", oxide_formula="Ho2O3",
    oxide_mw=377.86, price_usd_kg=60.0,
)
get_ree_database().add_element("Ho", ho)

# 2. Add extraction coefficients (from your literature source)
get_extractant_database().add_element_to_extractant(
    "PC88A", "Ho",
    ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
    temperature_coefficient=-2350,
)

# 3. Add separation factor data
sf_db = get_sf_database()
sf_db.add_pair("PC88A", "Ho_Dy", 1.4, stages_99=20)
sf_db.add_pair("PC88A", "Ho_Gd", 2.5, adjacent=False)

# 4. Use in distribution calculations
dist = REEDistribution(
    extractant="PC88A",
    elements=("Gd", "Dy", "Ho", "Y"),
)
D_ho = dist.get_D("Ho", pH=3.5, T=298.15)
print(f"D(Ho) at pH 3.5: {D_ho:.2f}")
```

---

## Economics

The plugin includes economic analysis tools:

```python
from difflow_ree import (
    estimate_capex,
    estimate_opex,
    calculate_revenue,
    calculate_profit,
    minimum_selling_price,
)

# Equipment costs
capex = estimate_capex(
    n_mixer_settlers=20,
    mixer_settler_volume=5.0,  # m³
    precipitation_capacity=100.0,  # kg/hr
)

# Operating costs
opex = estimate_opex(
    ree_throughput=200.0,  # tonnes/year
    extractant_consumption=50.0,  # kg/year
    acid_consumption=1000.0,  # kg/year
)

# Revenue from products
revenue = calculate_revenue(
    nd_production=50.0,  # kg/year
    dy_production=10.0,
    nd_price=100.0,  # $/kg
    dy_price=350.0,
)

# Profitability
profit = calculate_profit(revenue, opex, capex, years=10)
```

---

## Examples

### Example 1: Simple Nd/Pr Separation

```python
from difflow_ree import (
    REEDistribution,
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow.streams import make_stream

# Analyze separation factors
dist = REEDistribution(extractant="PC88A", elements=("Pr", "Nd"))
opt_pH, max_SF = dist.optimal_pH_for_separation("Nd", "Pr")
print(f"Optimal pH for Nd/Pr: {opt_pH:.2f}, SF = {max_SF:.2f}")

# Design separation circuit
params = ExtractScrubStripParams(
    extractant="PC88A",
    elements=("Pr", "Nd"),
    target_elements=("Nd",),
    extraction_pH=opt_pH,
    scrubbing_pH=opt_pH - 1.0,
)
circuit = ExtractScrubStripCircuit(params)

# Run separation
feed = make_stream({"H2O": 1.0, "Pr": 0.3, "Nd": 0.7}, T=298.15, P=101325.0)
results = circuit(feed)

print(f"Nd purity: {results['product_purity']['Nd']:.1%}")
print(f"Nd recovery: {results['target_recovery']['Nd']:.1%}")
```

### Example 2: Cerium Removal from Bastnasite

```python
from difflow_ree import CeriumOxidizer, CeriumOxidizerParams
from difflow.streams import make_stream

# Bastnasite composition (typical)
feed = make_stream({
    "H2O": 1.0,
    "La": 0.25,
    "Ce": 0.50,  # 50% Ce typical
    "Pr": 0.05,
    "Nd": 0.15,
    "Sm": 0.03,
    "Gd": 0.02,
}, T=298.15, P=101325.0)

# Oxidize and remove Ce
params = CeriumOxidizerParams(
    elements=("La", "Ce", "Pr", "Nd", "Sm", "Gd"),
    oxidant="air",
    pH=8.0,
    ce_conversion=0.95,
)
oxidizer = CeriumOxidizer(params)

filtrate, ceo2, info = oxidizer(feed)

print(f"Ce removed: {info['ce_conversion']:.1%}")
print(f"CeO2 produced: {info['ceo2_mass_kg_s']*3600*24*365:.1f} kg/year")
print(f"Ce in filtrate: {info['ce_fraction_out']:.1%}")
```

### Example 3: Gradient-Based Optimization

```python
import jax
import jax.numpy as jnp
from difflow_ree import ExtractScrubStripCircuit, ExtractScrubStripParams
from difflow.streams import make_stream

def separation_objective(pH_values):
    """Objective: maximize Nd purity × recovery."""
    extraction_pH, scrubbing_pH = pH_values

    params = ExtractScrubStripParams(
        extractant="D2EHPA",
        elements=("La", "Ce", "Nd"),
        target_elements=("Nd",),
        extraction_pH=extraction_pH,
        scrubbing_pH=scrubbing_pH,
    )
    circuit = ExtractScrubStripCircuit(params)

    feed = make_stream({"H2O": 1.0, "La": 0.3, "Ce": 0.4, "Nd": 0.3}, T=298.15, P=101325.0)
    results = circuit(feed)

    purity = results['product_purity']['Nd']
    recovery = results['target_recovery']['Nd']

    return -(purity * recovery)  # Negative for minimization

# Compute gradients
grad_fn = jax.grad(separation_objective)
pH_init = jnp.array([3.5, 2.0])
gradients = grad_fn(pH_init)
print(f"Gradients: d/d(ext_pH) = {gradients[0]:.4f}, d/d(scrub_pH) = {gradients[1]:.4f}")
```

---

## See Also

- [Examples: 04_rare_earth_extraction.ipynb](../examples/04_rare_earth_extraction.ipynb) - Basic REE extraction
- [Examples: 09_ree_ndfeb_magnet.ipynb](../examples/09_ree_ndfeb_magnet.ipynb) - NdFeB magnet recycling
- [Examples: 10_bastnasite_separation.ipynb](../examples/10_bastnasite_separation.ipynb) - Bastnasite ore processing
