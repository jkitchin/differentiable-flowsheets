# Thermodynamics

This document provides comprehensive documentation for thermodynamic models, property calculations, and databases available in Difflow.

---

(overview)=
## Overview

Difflow provides two levels of thermodynamic modeling:

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| **Ideal** | Moderate | Fast | Preliminary design, ideal mixtures |
| **Cubic EOS** | Good | Moderate | Non-ideal gases, high pressure |

All thermodynamic calculations are fully differentiable with JAX, enabling gradient-based optimization.

---

(ideal-thermodynamics)=
## Ideal Thermodynamics

**Location**: `difflow/thermo.py`

(speciesdata)=
### SpeciesData

The fundamental data structure for storing species properties.

```python
from typing import NamedTuple

class SpeciesData(NamedTuple):
    name: str              # Species name
    MW: float              # Molecular weight (g/mol)
    Cp_coeffs: tuple       # Heat capacity polynomial (a, b, c, d)
    Hvap_coeffs: tuple     # Heat of vaporization (A, n, Tc)
    antoine_coeffs: tuple  # Antoine equation (A, B, C)
    Hf: float              # Heat of formation (J/mol)
    Tref: float            # Reference temperature (K)
```

#### Heat Capacity Polynomial

$$C_p(T) = a + bT + cT^2 + dT^3$$

Where:
- $C_p$: Heat capacity at constant pressure (J/mol/K)
- $T$: Temperature (K)
- $a, b, c, d$: Polynomial coefficients

**Units**:
- $a$: J/mol/K
- $b$: J/mol/K²
- $c$: J/mol/K³
- $d$: J/mol/K⁴

#### Antoine Equation (Vapor Pressure)

$$\log_{10}(P^{sat}) = A - \frac{B}{T + C}$$

Where:
- $P^{sat}$: Saturation pressure (Pa)
- $T$: Temperature (K)
- $A, B, C$: Antoine coefficients

**Note**: Different sources use different forms. Difflow uses:
- Pressure in Pa
- Temperature in K

#### Watson Correlation (Heat of Vaporization)

$$\Delta H_{vap}(T) = A \left(1 - \frac{T}{T_c}\right)^n$$

Where:
- $\Delta H_{vap}$: Heat of vaporization (J/mol)
- $T_c$: Critical temperature (K)
- $A, n$: Watson correlation parameters

**Example**:

```python
from difflow.thermo import SpeciesData

# Define ethanol
ethanol = SpeciesData(
    name='ethanol',
    MW=46.07,                          # g/mol
    Cp_coeffs=(9.014, 0.2141, -8.39e-5, 1.373e-8),  # J/mol/K
    Hvap_coeffs=(50430.0, 0.4475, 513.9),           # Watson params
    antoine_coeffs=(10.8095, 1592.86, -46.95),      # P in Pa, T in K
    Hf=-277690.0,                      # J/mol
    Tref=298.15                        # K
)
```

---

(idealthermo-class)=
### IdealThermo Class

The main class for ideal thermodynamic calculations.

```python
from difflow.thermo import IdealThermo

class IdealThermo:
    def __init__(self, species_data: dict[str, SpeciesData]):
        """
        Initialize ideal thermodynamic model.

        Args:
            species_data: Dictionary mapping species names to SpeciesData
        """
```

#### Initialization

```python
species_data = {
    'methanol': SpeciesData(...),
    'water': SpeciesData(...),
    'DME': SpeciesData(...)
}

thermo = IdealThermo(species_data)
```

---

(property-calculations)=
### Property Calculations

#### Heat Capacity

```python
# Single species
Cp = thermo.Cp('ethanol', T=350.0)  # J/mol/K

# Mixture (molar average)
Cp_mix = thermo.Cp_mix(mole_fracs={'ethanol': 0.4, 'water': 0.6}, T=350.0)
```

**Equations**:

$$C_p^{pure}(T) = a + bT + cT^2 + dT^3$$

$$C_p^{mix} = \sum_i x_i C_{p,i}$$

#### Enthalpy

```python
# Pure species enthalpy relative to reference
H = thermo.H_pure('ethanol', T=400.0, phase='liquid')  # J/mol

# Stream enthalpy
H_flow = thermo.stream_enthalpy(
    flows={'ethanol': 1.0, 'water': 2.0},  # mol/s
    T=350.0,
    phase='liquid'
)  # W (J/s)
```

**Equations**:

$$H(T) = H_f + \int_{T_{ref}}^T C_p \, dT$$

$$H(T) = H_f + a(T - T_{ref}) + \frac{b}{2}(T^2 - T_{ref}^2) + \frac{c}{3}(T^3 - T_{ref}^3) + \frac{d}{4}(T^4 - T_{ref}^4)$$

For vapor phase, add heat of vaporization:

$$H^V(T) = H^L(T) + \Delta H_{vap}(T)$$

#### Saturation Pressure

```python
P_sat = thermo.Psat('ethanol', T=350.0)  # Pa
```

**Equation** (Antoine):

$$P^{sat} = 10^{A - B/(T + C)}$$

#### Heat of Vaporization

```python
Hvap = thermo.Hvap('ethanol', T=350.0)  # J/mol
```

**Equation** (Watson correlation):

$$\Delta H_{vap} = A \left(1 - \frac{T}{T_c}\right)^n$$

#### K-Values (Vapor-Liquid Equilibrium)

```python
# Single species K-value (Raoult's law)
K = thermo.K_value('ethanol', T=350.0, P=101325.0)

# All species K-values
K_values = thermo.K_values(T=350.0, P=101325.0)  # dict
```

**Equation** (Raoult's Law):

$$K_i = \frac{y_i}{x_i} = \frac{P_i^{sat}(T)}{P}$$

**Assumptions**:
- Ideal liquid mixture (activity coefficient = 1)
- Ideal gas phase (fugacity coefficient = 1)
- Valid for low pressures and similar molecules

#### Bubble Point and Dew Point

```python
# Bubble pressure at given T and liquid composition
P_bubble = thermo.bubble_pressure(x={'ethanol': 0.4, 'water': 0.6}, T=350.0)

# Dew pressure at given T and vapor composition
P_dew = thermo.dew_pressure(y={'ethanol': 0.4, 'water': 0.6}, T=350.0)
```

**Equations**:

Bubble point: $P = \sum_i x_i P_i^{sat}$

Dew point: $\frac{1}{P} = \sum_i \frac{y_i}{P_i^{sat}}$

---

(cubic-equations-of-state)=
## Cubic Equations of State

**Location**: `difflow/eos.py`

Cubic equations of state provide more accurate thermodynamic predictions for non-ideal systems, especially at high pressures.

### Critical Properties

```python
from difflow.eos import CriticalProperties

class CriticalProperties(NamedTuple):
    name: str          # Species name
    Tc: float          # Critical temperature (K)
    Pc: float          # Critical pressure (Pa)
    omega: float       # Acentric factor
    MW: float          # Molecular weight (g/mol)
```

**Acentric Factor** ($\omega$):

$$\omega = -\log_{10}\left(\frac{P^{sat}(T_r=0.7)}{P_c}\right) - 1$$

Measures deviation from simple fluid behavior:
- $\omega \approx 0$: Spherical molecules (Ar, Kr)
- $\omega > 0$: Non-spherical or polar molecules

(peng-robinson-eos)=
### Peng-Robinson EOS

The Peng-Robinson equation of state (1976) is widely used for hydrocarbon systems.

```python
from difflow.eos import PengRobinson, EOSParams

# Initialize with species critical properties
critical_props = {
    'methane': CriticalProperties('methane', 190.6, 4.6e6, 0.011, 16.04),
    'ethane': CriticalProperties('ethane', 305.4, 4.88e6, 0.099, 30.07),
}

pr = PengRobinson(critical_props)
```

#### Equations

**Equation of State**:

$$P = \frac{RT}{V - b} - \frac{a(T)}{V^2 + 2bV - b^2}$$

Or in terms of compressibility factor $Z = PV/RT$:

$$Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0$$

Where:
- $A = aP/(R^2T^2)$
- $B = bP/(RT)$

**Parameters**:

$$a(T) = a_c \cdot \alpha(T)$$

$$a_c = 0.45724 \frac{R^2 T_c^2}{P_c}$$

$$b = 0.07780 \frac{RT_c}{P_c}$$

$$\alpha(T) = \left[1 + \kappa\left(1 - \sqrt{T/T_c}\right)\right]^2$$

$$\kappa = 0.37464 + 1.54226\omega - 0.26992\omega^2$$

**Mixing Rules** (van der Waals one-fluid):

$$a_{mix} = \sum_i \sum_j y_i y_j \sqrt{a_i a_j}(1 - k_{ij})$$

$$b_{mix} = \sum_i y_i b_i$$

Where $k_{ij}$ is the binary interaction parameter (default = 0).

#### Methods

```python
# Compressibility factor
Z = pr.compressibility_factor(V=0.001, T=300.0, P=1e6)

# Fugacity coefficient
phi = pr.fugacity_coefficient(y={'methane': 0.7, 'ethane': 0.3}, T=300.0, P=1e6)

# K-values from fugacity
K = pr.K_value(species='methane', T=300.0, P=1e6, x=x, y=y)

# VLE flash calculation
V_frac, x, y = pr.flash_TP(z={'methane': 0.5, 'ethane': 0.5}, T=250.0, P=2e6)
```

#### Fugacity Calculation

$$\ln \phi_i = \frac{b_i}{b_{mix}}(Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2}B}\left(\frac{2\sum_j y_j a_{ij}}{a_{mix}} - \frac{b_i}{b_{mix}}\right)\ln\left(\frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B}\right)$$

**Equilibrium Condition**:

$$f_i^V = f_i^L$$

$$y_i \phi_i^V P = x_i \phi_i^L P$$

$$K_i = \frac{y_i}{x_i} = \frac{\phi_i^L}{\phi_i^V}$$

---

(soave-redlich-kwong-eos)=
### Soave-Redlich-Kwong EOS

The SRK equation (1972) is another popular cubic EOS.

```python
from difflow.eos import SRK

srk = SRK(critical_props)
```

#### Equations

**Equation of State**:

$$P = \frac{RT}{V - b} - \frac{a(T)}{V(V + b)}$$

**Parameters**:

$$a_c = 0.42748 \frac{R^2 T_c^2}{P_c}$$

$$b = 0.08664 \frac{RT_c}{P_c}$$

$$\alpha(T) = \left[1 + m\left(1 - \sqrt{T/T_c}\right)\right]^2$$

$$m = 0.480 + 1.574\omega - 0.176\omega^2$$

### Comparison: PR vs SRK

| Property | Peng-Robinson | SRK |
|----------|---------------|-----|
| Liquid density | Better | Less accurate |
| Vapor pressure | Good | Good |
| Near critical | Better | Good |
| Polar compounds | Limited | Limited |
| Parameters | $\Omega_a = 0.45724$ | $\Omega_a = 0.42748$ |
| | $\Omega_b = 0.07780$ | $\Omega_b = 0.08664$ |

---

(flash-calculations)=
### Flash Calculations

Flash calculations determine phase split at specified T and P.

```python
# TP Flash using PR EOS
V_frac, x, y = pr.flash_TP(
    z={'methane': 0.3, 'ethane': 0.3, 'propane': 0.4},
    T=250.0,
    P=1.5e6
)

print(f"Vapor fraction: {V_frac:.3f}")
print(f"Liquid composition: {x}")
print(f"Vapor composition: {y}")
```

#### Algorithm

1. **Initial K-values** (Wilson correlation):
   $$K_i = \frac{P_{c,i}}{P} \exp\left[5.373(1 + \omega_i)(1 - T_{c,i}/T)\right]$$

2. **Rachford-Rice equation** (solve for V):
   $$f(V) = \sum_i \frac{z_i(K_i - 1)}{1 + V(K_i - 1)} = 0$$

3. **Phase compositions**:
   $$x_i = \frac{z_i}{1 + V(K_i - 1)}$$
   $$y_i = K_i x_i$$

4. **Update K-values** from fugacity:
   $$K_i^{new} = \frac{\phi_i^L}{\phi_i^V}$$

5. **Iterate** until convergence

---

(species-database)=
## Species Database

**Location**: `difflow/database.py`

(available-species)=
### Available Species

The database contains ~100+ species with complete thermodynamic data:

#### Light Gases
- Hydrogen (H2), Helium (He), Nitrogen (N2), Oxygen (O2)
- Carbon monoxide (CO), Carbon dioxide (CO2)
- Hydrogen sulfide (H2S), Ammonia (NH3), Sulfur dioxide (SO2)

#### Alkanes (C1-C10)
- Methane, Ethane, Propane, n-Butane, i-Butane
- n-Pentane, i-Pentane, Neopentane
- n-Hexane, n-Heptane, n-Octane, n-Nonane, n-Decane

#### Alkenes
- Ethylene, Propylene
- 1-Butene, cis-2-Butene, trans-2-Butene, Isobutylene

#### Aromatics (BTEX)
- Benzene, Toluene
- o-Xylene, m-Xylene, p-Xylene
- Ethylbenzene, Styrene

#### Alcohols
- Methanol, Ethanol
- 1-Propanol, 2-Propanol (Isopropanol)
- 1-Butanol, 2-Butanol

#### Ketones and Aldehydes
- Acetone, Methyl ethyl ketone (MEK)
- Formaldehyde, Acetaldehyde

#### Carboxylic Acids
- Formic acid, Acetic acid

#### Esters
- Methyl acetate, Ethyl acetate

#### Ethers
- Diethyl ether, Dimethyl ether (DME)

#### Water
- Water (H2O)

(database-functions)=
### Database Functions

```python
from difflow.database import (
    get_species_data,
    get_critical_props,
    get_species_info,
    list_species,
    get_alkanes,
    get_btex,
    get_common_solvents
)

# Get SpeciesData for ideal thermo
ethanol_data = get_species_data('ethanol')

# Get CriticalProperties for EOS
ethanol_crit = get_critical_props('ethanol')

# Get all available information
info = get_species_info('ethanol')
print(info)

# List all available species
species_list = list_species()

# Get groups of species
alkanes = get_alkanes()  # ['methane', 'ethane', ..., 'decane']
btex = get_btex()        # ['benzene', 'toluene', 'ethylbenzene', 'xylene']
solvents = get_common_solvents()
```

### Database Contents Example

```python
# Methanol data in database
{
    'name': 'methanol',
    'MW': 32.04,
    'Tc': 512.6,  # K
    'Pc': 8.09e6,  # Pa
    'omega': 0.566,
    'Cp_coeffs': (21.15, 7.092e-2, 2.587e-5, -2.852e-8),
    'Hvap_coeffs': (45050.0, 0.4065, 512.6),
    'antoine_coeffs': (10.2044, 1582.91, -33.50),
    'Hf': -200940.0,  # J/mol
    'Tref': 298.15
}
```

---

(cantera-import)=
## Cantera Import

**Location**: `difflow/cantera_import.py`

Import thermodynamic data from Cantera YAML mechanism files without requiring Cantera installation.

(importing-mechanisms)=
### Importing Mechanisms

```python
from difflow.cantera_import import (
    import_species_data,
    import_critical_props,
    import_reactions,
    load_mechanism,
    list_available_species
)

# List available species in a Cantera file
species = list_available_species('gri30.yaml')

# Import species data for ideal thermo
species_data = import_species_data(
    'gri30.yaml',
    species_list=['CH4', 'O2', 'CO2', 'H2O']
)

# Import reactions with Arrhenius kinetics
reactions = import_reactions('gri30.yaml')

# Load complete mechanism
mechanism = load_mechanism('gri30.yaml')
```

(data-conversion)=
### Data Conversion

Cantera uses NASA polynomial format for thermodynamic properties:

#### NASA 7-Coefficient Polynomial

$$\frac{C_p}{R} = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4$$

$$\frac{H}{RT} = a_1 + \frac{a_2}{2}T + \frac{a_3}{3}T^2 + \frac{a_4}{4}T^3 + \frac{a_5}{5}T^4 + \frac{a_6}{T}$$

$$\frac{S}{R} = a_1 \ln T + a_2 T + \frac{a_3}{2}T^2 + \frac{a_4}{3}T^3 + \frac{a_5}{4}T^4 + a_7$$

The import function converts NASA coefficients to the simpler polynomial form used in Difflow.

### Supported Cantera Data

| Data Type | Support |
|-----------|---------|
| Thermo (NASA 7) | Full |
| Thermo (NASA 9) | Full |
| Transport | Partial |
| Reactions (Arrhenius) | Full |
| Reactions (falloff) | Partial |

---

## Usage Examples

### Complete VLE Flash with Ideal Thermo

```python
from difflow.thermo import IdealThermo
from difflow.database import get_species_data
from difflow.units.flash import Flash
from difflow.streams import make_stream

# Build thermo model from database
species_names = ['benzene', 'toluene', 'xylene']
species_data = {name: get_species_data(name) for name in species_names}
thermo = IdealThermo(species_data)

# Create feed stream
feed = make_stream(
    {'benzene': 0.4, 'toluene': 0.35, 'xylene': 0.25},
    T=380.0,  # K
    P=101325.0  # Pa
)

# Flash calculation
flash = Flash(thermo, species_names)
liquid, vapor, info = flash(feed)

print(f"Vapor fraction: {info['V_frac']:.3f}")
print(f"K-values: {info['K_values']}")
```

### High-Pressure Flash with PR EOS

```python
from difflow.eos import PengRobinson
from difflow.database import get_critical_props

# Build PR model from database
species_names = ['methane', 'ethane', 'propane', 'n-butane']
critical_props = {name: get_critical_props(name) for name in species_names}
pr = PengRobinson(critical_props)

# High-pressure flash
z = {'methane': 0.5, 'ethane': 0.3, 'propane': 0.15, 'n-butane': 0.05}
V_frac, x, y = pr.flash_TP(z, T=250.0, P=3.0e6)  # 30 bar

print(f"Vapor fraction: {V_frac:.3f}")
print(f"Liquid methane: {x['methane']:.4f}")
print(f"Vapor methane: {y['methane']:.4f}")
```

### Sensitivity Analysis with Automatic Differentiation

```python
import jax
import jax.numpy as jnp
from difflow.thermo import IdealThermo
from difflow.database import get_species_data

# Setup
species_data = {name: get_species_data(name) for name in ['ethanol', 'water']}
thermo = IdealThermo(species_data)

# Function to differentiate
def vapor_pressure_ratio(T):
    P_eth = thermo.Psat('ethanol', T)
    P_wat = thermo.Psat('water', T)
    return P_eth / P_wat

# Gradient of vapor pressure ratio w.r.t. temperature
grad_fn = jax.grad(vapor_pressure_ratio)
sensitivity = grad_fn(350.0)
print(f"d(P_eth/P_wat)/dT at 350K: {sensitivity:.6f}")
```

---

## Best Practices

### Model Selection

| Scenario | Recommended Model |
|----------|-------------------|
| Low pressure, ideal mixtures | Ideal Thermo |
| High pressure (> 10 bar) | PR or SRK EOS |
| Hydrocarbons | PR EOS |
| Polar/non-polar mixtures | PR + Binary k_ij |
| Highly polar (water, alcohols) | Activity coefficient models* |

*Activity coefficient models (NRTL, UNIQUAC) available in LLE module.

### Temperature Ranges

- **Cp polynomial**: Valid within fitted range (typically 200-1500 K)
- **Antoine equation**: Limited range (~0.01-2 bar typically)
- **Watson correlation**: Valid T < Tc
- **Cubic EOS**: Valid for all T, better away from critical

### Numerical Stability

```python
# Use jnp.where for safe operations
def safe_K_value(P_sat, P):
    return jnp.where(P > 0, P_sat / P, 0.0)

# Avoid division by zero in LMTD
def safe_lmtd(dT1, dT2):
    ratio = dT1 / jnp.maximum(dT2, 1e-10)
    return jnp.where(
        jnp.abs(dT1 - dT2) < 1e-6,
        0.5 * (dT1 + dT2),  # Limit when dT1 ≈ dT2
        (dT1 - dT2) / jnp.log(ratio)
    )
```
