# Carbon Capture Unit Operations

This document provides comprehensive documentation for the `difflow_cc` plugin, which provides specialized tools for modeling and optimizing carbon capture processes.

---

## Overview

The `difflow_cc` plugin provides:

- **Amine-based absorption**: MEA, DEA, MDEA, piperazine, amino acids
- **Membrane separation**: Polymeric, mixed-matrix, facilitated transport
- **Adsorption systems**: PSA, TSA, VSA, TVSA with various adsorbents
- **Direct Air Capture (DAC)**: Solid sorbent and liquid solvent systems
- **Economic analysis**: CAPEX, OPEX, levelized cost of capture
- **Degradation models**: Amine, adsorbent, and membrane aging

All models are fully differentiable using JAX, enabling gradient-based optimization, sensitivity analysis, and integration with machine learning.

---

## Installation

The carbon capture plugin is included as an optional dependency:

```bash
pip install difflow[cc]
```

Or install with all extras:

```bash
pip install difflow[all]
```

---

## Database and Properties

(amine-solvents)=
### Amine Solvents

Access amine solvent properties using the database functions:

```python
from difflow_cc import get_solvent, list_solvents

# List available solvents
print(list_solvents())
# ['MEA', 'DEA', 'MDEA', 'PZ', 'AMP', 'K2CO3', 'KOH']

# Get solvent properties
mea = get_solvent("MEA")
print(f"Heat of absorption: {mea.heat_of_absorption} kJ/mol")
print(f"Molecular weight: {mea.molecular_weight} g/mol")
```

#### Available Solvent Properties

| Property | Description | Units |
|----------|-------------|-------|
| `molecular_weight` | Molecular mass | g/mol |
| `heat_of_absorption` | Heat released on CO2 absorption | kJ/mol CO2 |
| `reaction_order` | Reaction order with CO2 | - |
| `activation_energy` | Activation energy | kJ/mol |
| `pre_exponential` | Pre-exponential factor | varies |
| `loading_capacity` | Maximum CO2 loading | mol CO2/mol amine |

#### Solvent Comparison

| Solvent | Full Name | Heat (kJ/mol) | Primary Use |
|---------|-----------|---------------|-------------|
| MEA | Monoethanolamine | 82 | Post-combustion (benchmark) |
| DEA | Diethanolamine | 68 | Natural gas sweetening |
| MDEA | Methyldiethanolamine | 55 | Selective H2S removal |
| PZ | Piperazine | 70 | Fast kinetics, blends |
| AMP | 2-amino-2-methyl-1-propanol | 65 | Sterically hindered |

(adsorbent-materials)=
### Adsorbent Materials

```python
from difflow_cc import get_adsorbent, list_adsorbents

print(list_adsorbents())
# ['Zeolite13X', 'ZeoliteNaY', 'Mg-MOF-74', 'SIFSIX-3-Ni',
#  'Activated_Carbon', 'Amine_Silica', 'MIL-101-Cr', 'Solid_Sorbent_DAC']

zeolite = get_adsorbent("Zeolite13X")
print(f"CO2 capacity: {zeolite.capacity_co2} mol/kg")
print(f"Heat of adsorption: {zeolite.heat_of_adsorption} kJ/mol")
```

#### Adsorbent Comparison

| Adsorbent | Capacity (mol/kg) | Heat (kJ/mol) | Primary Use |
|-----------|-------------------|---------------|-------------|
| Zeolite 13X | 5.0 | 36 | PSA, industrial benchmark |
| Mg-MOF-74 | 8.0 | 47 | High capacity, lab scale |
| SIFSIX-3-Ni | 2.5 | 45 | High selectivity |
| Amine-silica | 2.0 | 60 | TSA, DAC |
| Activated Carbon | 3.0 | 25 | Pre-treatment, low cost |

(membrane-materials)=
### Membrane Materials

```python
from difflow_cc import get_membrane, list_membranes

print(list_membranes())
# ['Polyimide', 'Polysulfone', 'PDMS', 'PIM-1', 'Pebax',
#  'MMM_ZIF8', 'FTM_Glycine', 'Cellulose_Acetate', 'TR_Polymer']

pim1 = get_membrane("PIM-1")
print(f"CO2 permeability: {pim1.permeability_co2} Barrer")
print(f"CO2/N2 selectivity: {pim1.selectivity_co2_n2}")
```

#### Membrane Comparison

| Membrane | CO2 Perm. (Barrer) | CO2/N2 Select. | Type |
|----------|-------------------|----------------|------|
| Polyimide | 10 | 30 | Glassy polymer |
| PIM-1 | 3000 | 20 | Polymer of intrinsic microporosity |
| Pebax | 150 | 50 | Block copolymer |
| MMM-ZIF8 | 50 | 40 | Mixed-matrix membrane |
| FTM-Glycine | 1000 | 100+ | Facilitated transport |

---

## Equilibrium Models

(vapor-liquid-equilibrium)=
### Vapor-Liquid Equilibrium

Model CO2-amine vapor-liquid equilibrium:

```python
from difflow_cc import AmineVLE, co2_loading, co2_equilibrium_pressure

# Create VLE model for MEA
vle = AmineVLE(solvent="MEA", concentration=30.0)  # 30 wt%

# Calculate CO2 loading at given conditions
loading = vle.get_loading(T=313.15, P_co2=10000.0)  # 10 kPa CO2
print(f"CO2 loading: {loading:.3f} mol/mol")

# Calculate equilibrium CO2 pressure at given loading
P_eq = vle.get_pressure(T=393.15, loading=0.4)
print(f"Equilibrium CO2 pressure: {P_eq:.0f} Pa")
```

#### Equilibrium Model

The CO2 partial pressure over loaded amine follows a modified Kent-Eisenberg model:

$$P_{CO_2} = K_H \cdot \alpha \cdot \exp\left(\frac{-\Delta H_{abs}}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right) \cdot f(\alpha)$$

Where:
- $\alpha$: CO2 loading (mol CO2/mol amine)
- $K_H$: Henry's constant
- $\Delta H_{abs}$: Heat of absorption
- $f(\alpha)$: Loading correction function

### Adsorption Isotherms

Multiple isotherm models are available:

```python
from difflow_cc import langmuir, sips, toth, dual_site_langmuir

# Langmuir isotherm
q = langmuir(P=100000.0, q_max=5.0, b=0.001)

# Sips isotherm (Freundlich-Langmuir)
q = sips(P=100000.0, q_max=5.0, b=0.001, n=0.8)

# Toth isotherm
q = toth(P=100000.0, q_max=5.0, b=0.001, t=0.7)

# Dual-site Langmuir
q = dual_site_langmuir(P=100000.0, q1=3.0, b1=0.01, q2=2.0, b2=0.0001)
```

#### Temperature-Dependent Isotherms

```python
from difflow_cc import langmuir_T, get_isotherm

# Temperature-dependent Langmuir
q = langmuir_T(P=100000.0, T=298.15, q_max=5.0, b0=0.001, dH=-36000.0)

# Get isotherm for specific adsorbent
isotherm = get_isotherm("Zeolite13X")
q = isotherm(P=100000.0, T=298.15)
```

#### Working Capacity

```python
from difflow_cc import working_capacity_PSA, working_capacity_TSA

# PSA working capacity
wc_psa = working_capacity_PSA(
    isotherm_fn=langmuir,
    P_ads=500000.0,    # 5 bar adsorption
    P_des=100000.0,    # 1 bar desorption
    T=298.15,
    params={'q_max': 5.0, 'b': 0.001}
)

# TSA working capacity
wc_tsa = working_capacity_TSA(
    isotherm_fn=langmuir_T,
    T_ads=298.15,      # 25°C adsorption
    T_des=423.15,      # 150°C desorption
    P=100000.0,
    params={'q_max': 5.0, 'b0': 0.001, 'dH': -36000.0}
)
```

(solubility-models)=
### Solubility Models

```python
from difflow_cc import co2_physical_solubility, diffusivity_co2_amine

# CO2 physical solubility in water
H = co2_physical_solubility(T=298.15)  # mol/(m³·Pa)

# CO2 diffusivity in amine solution
D = diffusivity_co2_amine(T=313.15, amine_conc=30.0)  # m²/s
```

---

(kinetics-models)=
## Kinetics Models

(reaction-kinetics)=
### Reaction Kinetics

Model CO2-amine reaction rates:

```python
from difflow_cc import reaction_rate_constant, enhancement_factor, hatta_number

# Second-order rate constant for MEA
k2 = reaction_rate_constant(
    solvent="MEA",
    T=313.15,
)
print(f"Rate constant: {k2:.0f} m³/(mol·s)")

# Hatta number (reaction vs. diffusion)
Ha = hatta_number(
    k2=k2,
    amine_conc=5000.0,  # mol/m³
    D_co2=1.5e-9,       # m²/s
    k_L=1e-4,           # m/s
)
print(f"Hatta number: {Ha:.1f}")

# Enhancement factor
E = enhancement_factor(Ha=Ha, E_inf=100.0)
print(f"Enhancement factor: {E:.1f}")
```

#### Governing Equations

**Reaction Rate** (second-order):

$$r = k_2 \cdot C_{CO_2} \cdot C_{amine}$$

**Arrhenius Temperature Dependence**:

$$k_2 = A \cdot \exp\left(-\frac{E_a}{RT}\right)$$

**Hatta Number**:

$$Ha = \frac{\sqrt{k_2 \cdot C_{amine} \cdot D_{CO_2}}}{k_L}$$

(mass-transfer)=
### Mass Transfer

```python
from difflow_cc import gas_film_coefficient, liquid_film_coefficient, overall_mass_transfer

# Gas-side mass transfer coefficient
k_G = gas_film_coefficient(
    v_gas=1.0,        # m/s superficial velocity
    d_pack=0.05,      # m packing diameter
    D_gas=1.5e-5,     # m²/s gas diffusivity
)

# Liquid-side coefficient
k_L = liquid_film_coefficient(
    v_liq=0.01,       # m/s liquid velocity
    d_pack=0.05,
    D_liq=1.5e-9,
)

# Overall coefficient
K_G = overall_mass_transfer(k_G=k_G, k_L=k_L, E=E, H=H)
```

---

## Unit Operations

(amineabsorber)=
### AmineAbsorber

**Location**: `difflow_cc/units/absorber.py`

**Class**: `AmineAbsorber`

**Description**: Equilibrium-stage model for amine absorption columns.

#### Parameters

```python
@dataclass
class AbsorberParams:
    solvent: str = "MEA"           # Solvent type
    concentration: float = 30.0    # Solvent concentration (wt%)
    n_stages: int = 10             # Number of equilibrium stages
    stage_efficiency: float = 0.25 # Murphree efficiency
    lean_loading: float = 0.2      # Inlet solvent loading (mol/mol)
    L_G_ratio: float = 3.0         # Liquid-to-gas molar ratio
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `gas_in` | Stream | - | Inlet flue gas |
| `solvent_in` | Stream | - | Lean solvent stream |
| `T` | float | K | Operating temperature |
| `P` | float | Pa | Operating pressure |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `gas_out` | Stream | - | Treated gas |
| `solvent_out` | Stream | - | Rich solvent |
| `info['capture_rate']` | float | - | CO2 capture efficiency |
| `info['rich_loading']` | float | mol/mol | Rich solvent loading |
| `info['profiles']` | dict | - | Stage-by-stage profiles |

#### Example Usage

```python
from difflow_cc import AmineAbsorber, AbsorberParams
from difflow import make_stream

params = AbsorberParams(
    solvent="MEA",
    concentration=30.0,
    n_stages=15,
    stage_efficiency=0.25,
    lean_loading=0.2,
)
absorber = AmineAbsorber(params)

# Create streams
flue_gas = make_stream(
    {'N2': 0.75, 'CO2': 0.12, 'H2O': 0.08, 'O2': 0.05},
    T=313.15, P=101325.0
)
lean_solvent = make_stream(
    {'MEA': 0.3, 'H2O': 0.7, 'CO2': 0.06},  # 20% loading
    T=313.15, P=101325.0
)

# Run absorber
gas_out, rich_solvent, info = absorber(flue_gas, lean_solvent, L_G=3.0)

print(f"CO2 capture rate: {info['capture_rate']:.1%}")
print(f"Rich loading: {info['rich_loading']:.3f} mol/mol")
```

---

(aminestripper)=
### AmineStripper

**Location**: `difflow_cc/units/stripper.py`

**Class**: `AmineStripper`

**Description**: Equilibrium-stage stripper for solvent regeneration.

#### Parameters

```python
@dataclass
class StripperParams:
    solvent: str = "MEA"
    n_stages: int = 8
    stage_efficiency: float = 0.3
    reboiler_duty: float = None     # kW, or calculated from loading
    condenser_temperature: float = 313.15  # K
```

#### Key Outputs

| Parameter | Description | Units |
|-----------|-------------|-------|
| `lean_solvent` | Regenerated solvent | Stream |
| `co2_product` | CO2 product stream | Stream |
| `info['regeneration_energy']` | Specific regen. energy | MJ/kg CO2 |
| `info['lean_loading']` | Lean solvent loading | mol/mol |

#### Example Usage

```python
from difflow_cc import AmineStripper, StripperParams

params = StripperParams(
    solvent="MEA",
    n_stages=10,
    reboiler_duty=4000.0,  # kW
)
stripper = AmineStripper(params)

lean_solvent, co2_product, info = stripper(rich_solvent, T_reboiler=393.15)

print(f"Regen. energy: {info['regeneration_energy']:.2f} MJ/kg CO2")
print(f"Lean loading: {info['lean_loading']:.3f} mol/mol")
```

---

(membraneseparator)=
### MembraneSeparator

**Location**: `difflow_cc/units/membrane.py`

**Class**: `MembraneSeparator`

**Description**: Single-stage membrane gas separator using solution-diffusion model.

#### Parameters

```python
@dataclass
class MembraneParams:
    membrane_type: str = "Polyimide"
    area: float = 1000.0           # m²
    thickness: float = 0.1e-6     # m (100 nm)
    pressure_ratio: float = 10.0   # Feed/permeate pressure
    stage_cut: float = None        # Optional fixed stage cut
```

#### Governing Equations

**Permeate Flux** (solution-diffusion):

$$J_i = \frac{P_i}{\delta} (p_{i,feed} - p_{i,perm})$$

Where:
- $P_i$: Permeability of species i (Barrer)
- $\delta$: Membrane thickness (m)
- $p$: Partial pressures

**Selectivity**:

$$\alpha_{ij} = \frac{P_i}{P_j}$$

**Stage Cut**:

$$\theta = \frac{F_{permeate}}{F_{feed}}$$

#### Example Usage

```python
from difflow_cc import MembraneSeparator, MembraneParams

params = MembraneParams(
    membrane_type="PIM-1",
    area=5000.0,
    thickness=1e-6,
    pressure_ratio=10.0,
)
membrane = MembraneSeparator(params)

flue_gas = make_stream({'N2': 0.85, 'CO2': 0.15}, T=298.15, P=1000000.0)

retentate, permeate, info = membrane(flue_gas)

print(f"Stage cut: {info['stage_cut']:.2%}")
print(f"CO2 purity: {info['co2_purity']:.1%}")
print(f"CO2 recovery: {info['co2_recovery']:.1%}")
```

---

(adsorption-units)=
### Adsorption Units

**Location**: `difflow_cc/units/adsorption.py`

Four swing adsorption variants are available:

| Class | Description | Regeneration |
|-------|-------------|--------------|
| `PSAUnit` | Pressure Swing Adsorption | Pressure reduction |
| `TSAUnit` | Temperature Swing Adsorption | Heating |
| `VSAUnit` | Vacuum Swing Adsorption | Vacuum |
| `TVSAUnit` | Temperature-Vacuum Swing | Both |

#### Parameters

```python
@dataclass
class AdsorptionParams:
    adsorbent: str = "Zeolite13X"
    bed_mass: float = 1000.0       # kg adsorbent
    n_beds: int = 2                # Number of beds
    cycle_time: float = 600.0      # s per cycle
    # PSA/VSA specific
    P_ads: float = 500000.0        # Pa adsorption pressure
    P_des: float = 10000.0         # Pa desorption pressure
    # TSA specific
    T_ads: float = 298.15          # K adsorption temperature
    T_des: float = 423.15          # K desorption temperature
```

#### Example Usage

```python
from difflow_cc import PSAUnit, AdsorptionParams

params = AdsorptionParams(
    adsorbent="Zeolite13X",
    bed_mass=5000.0,
    n_beds=4,
    P_ads=500000.0,
    P_des=100000.0,
    cycle_time=300.0,
)
psa = PSAUnit(params)

flue_gas = make_stream({'N2': 0.85, 'CO2': 0.15}, T=298.15, P=500000.0)

product, offgas, info = psa(flue_gas)

print(f"CO2 purity: {info['co2_purity']:.1%}")
print(f"CO2 recovery: {info['co2_recovery']:.1%}")
print(f"Productivity: {info['productivity']:.2f} mol CO2/(kg·h)")
```

---

(heat-integration)=
## Heat Integration

**Location**: `difflow_cc/units/heat_integration.py`

Efficient heat recovery is critical for minimizing energy penalty:

```python
from difflow_cc import LeanRichExchanger, LeanRichExchangerParams

params = LeanRichExchangerParams(
    approach_temperature=10.0,  # K minimum approach
    effectiveness=0.85,
)
exchanger = LeanRichExchanger(params)

# Heat exchange between rich and lean solvent
lean_hot, rich_cold, info = exchanger(lean_cold, rich_hot)

print(f"Duty: {info['duty']/1e6:.2f} MW")
print(f"Energy saved: {info['energy_saved']:.1%}")
```

---

(co2-compression)=
## CO2 Compression

**Location**: `difflow_cc/units/compression.py`

CO2 must be compressed to pipeline or sequestration pressure:

```python
from difflow_cc import CompressionTrain, CompressionTrainParams

params = CompressionTrainParams(
    n_stages=4,
    P_inlet=101325.0,         # 1 atm
    P_outlet=15000000.0,      # 150 bar (supercritical)
    eta_polytropic=0.85,
    intercooling_T=313.15,    # 40°C
)
compressor = CompressionTrain(params)

co2_stream = make_stream({'CO2': 1.0}, T=313.15, P=101325.0)

compressed, info = compressor(co2_stream)

print(f"Total power: {info['total_power']/1e6:.2f} MW")
print(f"Specific power: {info['specific_power']:.0f} kJ/kg CO2")
```

---

(direct-air-capture)=
## Direct Air Capture

**Location**: `difflow_cc/units/dac.py`

Model direct air capture systems:

```python
from difflow_cc import SolidSorbentDAC, DACParams

params = DACParams(
    adsorbent="Solid_Sorbent_DAC",
    contactor_area=10000.0,    # m² of contactor
    T_ads=298.15,
    T_des=373.15,
    cycle_time=3600.0,         # 1 hour cycle
)
dac = SolidSorbentDAC(params)

air = make_stream({'N2': 0.78, 'O2': 0.21, 'CO2': 0.0004}, T=298.15, P=101325.0)

co2_product, air_out, info = dac(air)

print(f"Capture rate: {info['capture_rate']:.1f} kg CO2/day")
print(f"Energy: {info['energy_per_tonne']:.0f} GJ/tonne CO2")
```

---

## Economics

**Location**: `difflow_cc/economics/`

Comprehensive economic analysis:

```python
from difflow_cc import (
    levelized_cost_capture,
    cost_of_co2_avoided,
    EconomicParams,
)

params = EconomicParams(
    capacity_factor=0.85,
    plant_lifetime=25,          # years
    discount_rate=0.08,
    co2_captured=1000000.0,     # tonnes/year
)

# Capital costs
from difflow_cc import absorber_cost, stripper_cost, installed_cost

capex_absorber = absorber_cost(diameter=10.0, height=30.0)
capex_stripper = stripper_cost(diameter=8.0, height=25.0)
total_capex = installed_cost(capex_absorber + capex_stripper)

# Operating costs
from difflow_cc import steam_cost, electricity_cost, total_operating_cost

opex = total_operating_cost(
    steam_consumption=3.5,      # GJ/tonne CO2
    electricity=150.0,          # kWh/tonne CO2
    solvent_makeup=1.5,         # kg/tonne CO2
)

# Levelized cost
lcoc = levelized_cost_capture(
    capex=total_capex,
    opex_annual=opex * params.co2_captured,
    co2_annual=params.co2_captured,
    lifetime=params.plant_lifetime,
    discount_rate=params.discount_rate,
)
print(f"Levelized cost: ${lcoc:.1f}/tonne CO2")

# Cost of CO2 avoided
cca = cost_of_co2_avoided(
    lcoc=lcoc,
    reference_emission=0.4,    # tonne CO2/MWh
    capture_emission=0.04,     # tonne CO2/MWh with capture
)
print(f"Cost avoided: ${cca:.1f}/tonne CO2")
```

---

(degradation-models)=
## Degradation Models

**Location**: `difflow_cc/degradation/`

(amine-degradation)=
### Amine Degradation

```python
from difflow_cc import (
    oxidative_degradation_rate,
    thermal_degradation_rate,
    total_amine_loss,
    solvent_lifetime,
)

# Oxidative degradation (from O2 in flue gas)
r_ox = oxidative_degradation_rate(
    T=313.15,
    O2_concentration=0.05,     # 5% O2
    amine="MEA",
)

# Thermal degradation
r_th = thermal_degradation_rate(
    T=393.15,                   # Reboiler temperature
    loading=0.4,
    amine="MEA",
)

# Total loss
loss = total_amine_loss(r_ox, r_th, solvent_inventory=100000.0)
print(f"Amine loss: {loss:.1f} kg/day")

# Solvent lifetime
lifetime = solvent_lifetime(loss, solvent_inventory=100000.0)
print(f"Solvent lifetime: {lifetime:.0f} days")
```

(adsorbent-degradation)=
### Adsorbent Degradation

```python
from difflow_cc import capacity_fade, adsorbent_lifetime

# Capacity fade over cycles
fade = capacity_fade(
    n_cycles=10000,
    T_max=423.15,
    humidity=0.1,
)
print(f"Capacity fade: {fade:.1%}")

# Adsorbent lifetime
lifetime = adsorbent_lifetime(
    cycles_per_day=48,
    acceptable_fade=0.2,
)
print(f"Adsorbent lifetime: {lifetime:.0f} days")
```

(membrane-aging)=
### Membrane Aging

```python
from difflow_cc import physical_aging, plasticization, membrane_lifetime

# Physical aging (glassy polymers)
perm_loss = physical_aging(
    time_hours=8760,           # 1 year
    T=298.15,
)

# Plasticization from CO2
perm_change = plasticization(
    P_co2=500000.0,            # 5 bar CO2
    membrane="Polyimide",
)

lifetime = membrane_lifetime(
    perm_loss_rate=0.05,       # 5%/year
    acceptable_loss=0.3,
)
print(f"Membrane lifetime: {lifetime:.1f} years")
```

---

## Examples

### Example 1: Complete Amine Capture Plant

```python
from difflow_cc import (
    AmineAbsorber, AbsorberParams,
    AmineStripper, StripperParams,
    LeanRichExchanger, LeanRichExchangerParams,
    CompressionTrain, CompressionTrainParams,
)
from difflow import make_stream, Flowsheet

# Create units
absorber = AmineAbsorber(AbsorberParams(n_stages=15))
stripper = AmineStripper(StripperParams(n_stages=10))
exchanger = LeanRichExchanger(LeanRichExchangerParams())
compressor = CompressionTrain(CompressionTrainParams(n_stages=4))

# Build flowsheet
fs = Flowsheet()
fs.add_unit('absorber', absorber)
fs.add_unit('exchanger', exchanger)
fs.add_unit('stripper', stripper)
fs.add_unit('compressor', compressor)

# Connect streams
fs.connect('absorber', 'exchanger', stream='rich_solvent')
fs.connect('exchanger', 'stripper', stream='rich_solvent')
fs.connect('stripper', 'exchanger', stream='lean_solvent')
fs.connect('exchanger', 'absorber', stream='lean_solvent')
fs.connect('stripper', 'compressor', stream='co2')

# Solve
results = fs.solve(flue_gas)
print(f"Net capture: {results['capture_efficiency']:.1%}")
print(f"Energy penalty: {results['energy_penalty']:.1f} MJ/kg CO2")
```

### Example 2: Gradient-Based Optimization

```python
import jax
import jax.numpy as jnp
from difflow_cc import AmineAbsorber, AbsorberParams

def capture_cost(params):
    """Minimize cost per tonne CO2 captured."""
    n_stages, L_G_ratio = params

    absorber_params = AbsorberParams(
        n_stages=int(n_stages),
        L_G_ratio=L_G_ratio,
    )
    absorber = AmineAbsorber(absorber_params)

    gas_out, rich, info = absorber(flue_gas, lean_solvent)

    # Cost function: capital + operating
    capex = n_stages * 100000.0  # $/stage
    opex = L_G_ratio * 50.0      # $/mol solvent
    co2_captured = info['capture_rate'] * 100.0  # tonnes/h

    return (capex/8760 + opex) / co2_captured

# Compute gradients
grad_fn = jax.grad(capture_cost)
params = jnp.array([10.0, 3.0])
gradients = grad_fn(params)
print(f"Gradients: d/d(n_stages)={gradients[0]:.2f}, d/d(L_G)={gradients[1]:.2f}")
```

### Example 3: Technology Comparison

```python
from difflow_cc import (
    AmineAbsorber, AbsorberParams,
    MembraneSeparator, MembraneParams,
    PSAUnit, AdsorptionParams,
    levelized_cost_capture,
)

technologies = {
    'Amine (MEA)': AmineAbsorber(AbsorberParams()),
    'Membrane (PIM-1)': MembraneSeparator(MembraneParams(membrane_type='PIM-1')),
    'PSA (13X)': PSAUnit(AdsorptionParams()),
}

for name, unit in technologies.items():
    result = unit(flue_gas)
    lcoc = levelized_cost_capture(result['capex'], result['opex'], result['co2_annual'])
    print(f"{name}: Capture={result['capture']:.1%}, Cost=${lcoc:.0f}/tonne")
```

---

## See Also

- [Examples: 01_amine_capture_fundamentals.ipynb](../src/difflow_cc/examples/01_amine_capture_fundamentals.ipynb) - Amine basics
- [Examples: 02_membrane_separation.ipynb](../src/difflow_cc/examples/02_membrane_separation.ipynb) - Membrane systems
- [Examples: 03_adsorption_processes.ipynb](../src/difflow_cc/examples/03_adsorption_processes.ipynb) - PSA/TSA/VSA
- [Examples: 04_optimization_with_gradients.ipynb](../src/difflow_cc/examples/04_optimization_with_gradients.ipynb) - Optimization
- [Examples: 05_integrated_capture_plant.ipynb](../src/difflow_cc/examples/05_integrated_capture_plant.ipynb) - Full plant
