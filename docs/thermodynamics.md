# Thermodynamics

This document provides comprehensive documentation for thermodynamic models, property calculations, and databases available in Difflow.

---

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

<!-- doc-test: skip: signature listing, redefines the class -->
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
from difflow.thermo import IdealThermo
from difflow.database import get_species_data

# SpeciesData(name, MW, Cp_coeffs, Hvap_coeffs, antoine_coeffs, ...) per species;
# here taken from the built-in database
species_data = {s: get_species_data(s)
                for s in ['methanol', 'ethanol', 'water', 'dimethyl_ether']}

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

`K_values` and `K_values_array` also accept the liquid and vapor compositions
`x` and `y` and ignore them — Raoult K-values are a function of $(T, P)$ alone.
They are in the signature so that a unit written against this interface (the
rigorous [`DistillationColumn`](unit-operations-chemical.md), for one) can pass
its stage compositions unconditionally and run unchanged on a
[`CubicThermo`](#cubic-thermo-k-values), whose K-values do depend on them.
`IdealThermo.K_depends_on_composition` is `False` and `CubicThermo`'s is `True`,
for callers that want to skip a composition iteration they do not need.

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

<!-- doc-test: skip: signature listing, redefines the class -->
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
from difflow.eos import PengRobinson, CriticalProperties

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
import jax.numpy as jnp
from difflow.eos import flash_TP_eos

# Compositions are arrays in pr.species_order (here methane, ethane)
y = jnp.array([0.7, 0.3])

# Compressibility factor (largest root = vapor, smallest = liquid)
Z = pr.solve_Z(T=300.0, P=1e6, y=y, phase='vapor')

# Fugacity coefficients of every species
phi = pr.fugacity_coefficient(T=300.0, P=1e6, y=y, phase='vapor')

# K-values from fugacity, given liquid and vapor compositions
x = jnp.array([0.4, 0.6])
K = pr.K_values(T=250.0, P=2e6, x=x, y=y)

# VLE flash calculation: vapor fraction, liquid and vapor compositions
V_frac, x, y = flash_TP_eos(pr, z=jnp.array([0.5, 0.5]), T=250.0, P=2e6)
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
import jax.numpy as jnp
from difflow.eos import PengRobinson, flash_TP_eos
from difflow.database import get_critical_props

names = ['methane', 'ethane', 'propane']
pr3 = PengRobinson({n: get_critical_props(n) for n in names})

# TP Flash using PR EOS (compositions are arrays in species order)
V_frac, x, y = flash_TP_eos(pr3, z=jnp.array([0.3, 0.3, 0.4]), T=250.0, P=1.5e6)

print(f"Vapor fraction: {float(V_frac):.3f}")
print(f"Liquid composition: {dict(zip(names, map(float, x)))}")
print(f"Vapor composition: {dict(zip(names, map(float, y)))}")
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

(cubic-thermo-k-values)=
### CubicThermo K-Values

`CubicThermo` exposes the same K-value interface as `IdealThermo`, built from
the EOS fugacity coefficients rather than from Raoult's law:

$$K_i = \frac{\hat\phi_i^L(T, P, x)}{\hat\phi_i^V(T, P, y)}$$

```python
from difflow.thermo import CubicThermo

names = ['propane', 'n_butane']
sp = {n: get_species_data(n) for n in names}
crit = {n: get_critical_props(n) for n in names}
x = jnp.array([0.5, 0.5])
y = jnp.array([0.8, 0.2])

thermo = CubicThermo(IdealThermo(sp), PengRobinson(crit))

K = thermo.K_values_array(T=380.0, P=10e5, x=x, y=y)  # both compositions known
K = thermo.K_values_array(T=380.0, P=10e5, x=x)       # bubble-point K at x
K = thermo.K_values_array(T=380.0, P=10e5)            # no composition: Raoult
K = thermo.K_values(T=380.0, P=10e5, x=x)             # same, as a dict
```

That is what lets a unit written against `IdealThermo`'s interface — the
rigorous [`DistillationColumn`](unit-operations-chemical.md) — run on
Peng-Robinson without changing the unit.

Two properties of these K-values shape how they are used:

**They depend on composition.** $K_i$ is a fixed point, not a formula. Pass
whichever compositions you have; whichever you omit is filled in by a bounded
successive substitution ($y = Kx$ renormalised, or $x = y/K$) from a
composition-free starting estimate, which is the standard bubble- or dew-point
K calculation. Pass both when you already have a consistent pair — that is a
single evaluation with no inner loop.

**They only exist inside the two-root window.** The EOS gives a two-phase K
only where its cubic has two distinct roots at this $(T, P, x)$. Away from the
bubble point — a subcooled liquid, a superheated vapor — there is one root,
both phases take it, and $K_i$ comes back identically 1. That is the EOS
correctly reporting a single phase, but it makes $\sum_i K_i x_i - 1$ a flat
zero, which a root finder reads as converged wherever it is standing. A
bubble-point solve on these K-values therefore needs to start inside the window
and be able to retreat if a step leaves it; see the two-pass solve in
`difflow.units.distillation._bubble_T`.

With no composition at all, `CubicThermo` returns the wrapped `IdealThermo`'s
Raoult K-values rather than the EOS's own Wilson estimate. Both are
composition-free, but Antoine coefficients are fitted vapor-pressure data,
while Wilson is a two-constant fit off the critical point that for a heavy
hydrocarbon can be a hundred degrees out — far enough to start the EOS
iteration outside the window.

---

(species-database)=
## Activity-Coefficient Models

Non-ideal liquids use the modified Raoult's law $y_i P = x_i \gamma_i(x,T) P_i^{sat}(T)$.
An activity model is any pytree with a `gamma(x, T)` method (the
`difflow.activity.ActivityModel` protocol; a plain callable `f(x, T)` also works),
and `difflow.activity_gamma(model, x, T)` dispatches on it. `Flash`,
`txy`, `pxy`, `xy_curve` and `find_azeotrope` all take one through
`activity_model=`. The existing `NRTLParams` and `UNIQUACParams` (see
[LLE](unit-operations-chemical.md)) implement `gamma` as adapters over their
functions, so earlier code is unchanged.

| Model | Class | Species | Parameters | Predicts LLE? |
|-------|-------|---------|------------|---------------|
| NRTL | `NRTLParams` | n | $a_{ij}, b_{ij}, \alpha_{ij}$ | yes |
| UNIQUAC | `UNIQUACParams` | n | $r, q, a_{ij}, b_{ij}$ | yes |
| Wilson | `WilsonParams` | n | $V_i$, $\lambda_{ij}-\lambda_{ii}$ (J/mol) | **no** |
| Margules (two-/three-suffix) | `MargulesParams` | 2 | $A_{12}, A_{21}$ | yes |
| van Laar | `VanLaarParams` | 2 | $A_{12}, A_{21}$ | yes |

### Wilson

$$\Lambda_{ij} = \frac{V_j}{V_i}\exp\!\left(-\frac{\lambda_{ij}-\lambda_{ii}}{RT}\right),\qquad
\ln\gamma_i = 1 - \ln\sum_j x_j\Lambda_{ij} - \sum_k \frac{x_k\Lambda_{ki}}{\sum_j x_j\Lambda_{kj}}$$

Wilson's equation **cannot predict liquid-liquid splitting**: its Gibbs energy of
mixing is convex for every parameter set, so it describes one liquid phase. Use
NRTL or UNIQUAC (or Margules / van Laar) for partially miscible systems.

### Margules and van Laar

Binary, dimensionless and temperature independent; $A_{12}=\ln\gamma_1^\infty$,
$A_{21}=\ln\gamma_2^\infty$.

$$\text{Margules (3-suffix):}\quad \ln\gamma_1 = x_2^2\,[A_{12}+2(A_{21}-A_{12})x_1],\quad
\ln\gamma_2 = x_1^2\,[A_{21}+2(A_{12}-A_{21})x_2]$$

$A_{12}=A_{21}=A$ is the two-suffix form $\ln\gamma_1 = A x_2^2$
(`MargulesParams.two_suffix`).

$$\text{van Laar:}\quad \ln\gamma_1 = \frac{A_{12}A_{21}^2x_2^2}{(A_{12}x_1+A_{21}x_2)^2},\quad
\ln\gamma_2 = \frac{A_{21}A_{12}^2x_1^2}{(A_{12}x_1+A_{21}x_2)^2}$$

(the textbook form $A_{12}(1+A_{12}x_1/A_{21}x_2)^{-2}$, rewritten so that the
pure-component limits are finite; $A_{12}$ and $A_{21}$ must share a sign).

```python
import jax.numpy as jnp
from difflow import WilsonParams, MargulesParams, VanLaarParams, activity_gamma

x = jnp.array([0.4, 0.6])
wilson = WilsonParams.binary(("methanol", "water"), [40.73e-6, 18.07e-6], 667.0, 1981.0)
print(activity_gamma(wilson, x, 340.0))
print(activity_gamma(MargulesParams.two_suffix(("a", "b"), 1.0), x, 300.0))
print(activity_gamma(VanLaarParams(("a", "b"), 1.6798, 0.9227), x, 300.0))
```

Validation, stated precisely (see `tests/test_phase_diagrams.py`): closed forms are
checked against hand-computed values, the Gibbs-Duhem equation and the
infinite-dilution identities; ethanol-water NRTL reproduces the azeotrope
(x = 0.893, 351.29 K vs. the experimental 0.894, 351.3 K); the Wilson parameters
above were **regressed** to the 1 atm methanol-water Txy table (Perry's) and the
test checks the model reproduces it to 0.3 K and 0.01 in $y$. No textbook worked
example is reproduced to three figures.

**References**: Wilson, J. Am. Chem. Soc. 86, 127 (1964); van Laar, Z. Phys. Chem.
72, 723 (1910); Margules, Sitzungsber. Akad. Wiss. Wien 104, 1243 (1895); Prausnitz,
Lichtenthaler, de Azevedo, *Molecular Thermodynamics of Fluid-Phase Equilibria*,
3e, Ch. 6-7; Smith, Van Ness, Abbott, 7e, Ch. 12.

## Phase Diagrams

`difflow.phase_diagrams` returns the curves used to read and teach phase
behaviour. All are JAX and differentiable; bubble temperatures are Newton root
finds with implicit differentiation. The vapor is ideal (low pressure) and the
vapor pressures are the thermo object's Antoine equations.

| Function | Returns |
|----------|---------|
| `txy(thermo, species, P, n=51, activity_model=None)` | `dict(x, y, T)`: bubble $T(x)$, $y$ from the K-values |
| `pxy(thermo, species, T, n=51, activity_model=None)` | `dict(x, y, P)` |
| `xy_curve(thermo, species, P, n=51, activity_model=None)` | `dict(x, y, T, alpha)`: $\alpha(x)=K_1/K_2$ |
| `find_azeotrope(thermo, species, P=..., T=..., activity_model=None)` | `dict(found, x, T, P)`; NaN (masked, no Python branch) if none |
| `ternary_lle(lle_model, T, n_tie=15)` | `dict(binodal, tie_lines, plait)` by isoactivity continuation |

`ternary_lle` takes an `LLEEquilibrium` (NRTL/UNIQUAC) or any three-species
activity model; species 0 and 1 are the partially miscible pair, species 2 the
solute. It solves a type-I system by continuation from the binary edge to the
plait point and is not differentiated.

```python
from difflow import txy, xy_curve, find_azeotrope, IdealThermo
from difflow.database import get_species_data

thermo = IdealThermo({n: get_species_data(n) for n in ("benzene", "toluene")})
d = txy(thermo, ["benzene", "toluene"], 101325.0)
print(float(d["T"][0]), float(d["T"][-1]))     # toluene / benzene boiling points
a = xy_curve(thermo, ["benzene", "toluene"], 101325.0)
print(float(a["alpha"].mean()))                # about 2.5
print(bool(find_azeotrope(thermo, ["benzene", "toluene"], P=101325.0)["found"]))
```

The benzene-toluene end points agree with the normal boiling points (353.24 K,
383.78 K) to 0.1 K, and $\bar\alpha\approx 2.49$ (2.35 at the toluene end, 2.60
at the benzene end).

### Plot helpers

`difflow.visualization` (matplotlib, optional) provides `plot_txy`, `plot_pxy`,
`plot_xy` (with the dotted 45 degree line) and `plot_ternary` (equilateral or
right-triangle coordinates, binodal, tie lines, plait point). Colour is never the
only cue: bubble and dew curves differ in line style and marker.

```python
import matplotlib
matplotlib.use("Agg")
from difflow.visualization import plot_txy

ax = plot_txy(d)
```

---

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
alkanes = get_alkanes()  # {'methane': ..., 'ethane': ..., ...}
btex = get_btex()        # CriticalProperties for the BTEX aromatics
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

(liquid-properties)=
## Liquid Properties

**Location**: `difflow/liquid_properties.py`

Liquid density, viscosity and thermal conductivity for every species in the
database, written in JAX so they differentiate and `jit` like the rest of
difflow. Pipe flow, film coefficients and evaporators need them.

```python
from difflow import (liquid_density, liquid_viscosity,
                     liquid_thermal_conductivity, stream_liquid_properties,
                     make_stream)

liquid_density("water", 298.15)               # 997.0 kg/m^3
liquid_viscosity("toluene", 320.0)            # Pa s
liquid_thermal_conductivity("ethanol", 300.0) # W/m/K

s = make_stream({"water": 8.0, "methanol": 2.0}, T=300.0, P=1e5)
stream_liquid_properties(s)   # {'rho', 'rho_molar', 'mu', 'k', 'Q'}
```

Mixtures (`mixture_liquid_density`, `mixture_liquid_viscosity`,
`mixture_liquid_thermal_conductivity` in `difflow.liquid_properties`) use
simple rules:
- density: ideal mixing of molar volumes;
- viscosity: ln μ = Σ xᵢ ln μᵢ;
- thermal conductivity: DIPPR 9H.

All three neglect interactions. For aqueous and hydrogen-bonding mixtures,
treat them as estimates.

**Sources.** The coefficients are generated by
`scripts/generate_liquid_properties.py` into `_liquid_property_data.py`.
difflow does not need `chemicals` or CoolProp at runtime. For each property,
the generator picks the source that agreed best with CoolProp:

| Property | First choice | Then | Estimate if neither table has it |
|---|---|---|---|
| Density | VDI Heat Atlas PPDS (60 species) | Perry's 8e, DIPPR 105 (9) | Rackett / Yamada-Gunn (4) |
| Viscosity | Perry's 8e, DIPPR 101 (66) | VDI-PPDS, PPDS9 (3) | Orrick-Erbar (4) |
| Thermal conductivity | Perry's 8e, DIPPR 100 (66) | VDI-PPDS (3) | Sato-Riedel (4) |

- Both tables come from the `chemicals` package (MIT).
- Heavy water is in neither table. Its three correlations are fitted to
  CoolProp.
- The four estimated species are 2,2,5-trimethylhexane,
  2,3,4-trimethylpentane, 2,4-dimethylpentane and 2,5-dimethylhexane. On
  branched isomers that the tables do cover, the estimates are off by:
  - density: 1–3%;
  - viscosity: up to 25%;
  - thermal conductivity: up to 27%.

`liquid_property_source(name, prop)` returns the citation for any
correlation, and the report layer records liquid lookups with the kind
`"liquid"`.

Against CoolProp's saturated liquid, across the 46 species it covers, the
worst errors are:
- density: 2.2%;
- viscosity: 20%;
- thermal conductivity: 18%.

There is one exception. For the viscosity of n-pentane, isopentane and
dimethyl ether at low temperature, the two tables agree with each other but
not with CoolProp.

**Temperature range.** Each correlation stores Tmin and Tmax, and
`check_liquid_range(name, T)` reports them. A concrete temperature outside the
range raises `LiquidRangeWarning`. The value returned is still the
correlation's: it is never clipped, so the gradient stays smooth.
`range_stated=False` marks a range the source does not give, which is
assumed. Under `jit` or `grad` the check is skipped.

---

(cantera-import)=
## Cantera Import

**Location**: `difflow/cantera_import.py`

Import thermodynamic data from Cantera YAML mechanism files without requiring Cantera installation.

(importing-mechanisms)=
### Importing Mechanisms

<!-- doc-test: skip: needs a Cantera YAML mechanism file (gri30.yaml) -->
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

(pyglenn-import)=
## NASA Glenn (pyglenn) Import

**Location**: `difflow/pyglenn_import.py`

Import ideal-gas thermodynamic data from the NASA Glenn (CEA) thermodynamic
database, as exposed by the [`pyglenn`](https://github.com/ProfLeao/pyglenn)
package (~2030 species, NASA-9 polynomials). `pyglenn` is an **optional**
dependency:

```bash
pip install pyglenn          # or:  pip install "difflow[pyglenn]"
```

Unlike the Cantera importer (which parses a YAML file), this adapter talks to
pyglenn's `ThermochemicalCalculator` at runtime. Because its
`import_species_data` / `list_available_species` names mirror the Cantera ones,
it is exposed as a **namespace** rather than flattened into `difflow`:

<!-- doc-test: skip: needs the optional pyglenn package -->
```python
from difflow.pyglenn_import import import_species_data, list_available_species
from difflow.thermo import IdealThermo

# Find species records (id, name, phase, molecular_weight, ...)
list_available_species("CO2")

# Import ideal-gas SpeciesData for a set of species
species_data = import_species_data(["O2", "CO2", "H2O"])
thermo = IdealThermo(species_data)
```

### What is (and is not) imported

| difflow `SpeciesData` field | Source in pyglenn |
|-----------------------------|-------------------|
| `Cp_coeffs` | Cubic **least-squares fit** of pyglenn's `Cp(T)` over `T_fit_range` (default 300–1000 K) |
| `MW` | `molecular_weight` |
| `Hf` | `heat_of_formation_298K` |
| `Hvap_coeffs`, `antoine_coeffs` | **Not in NASA Glenn data** — filled with neutral placeholders, or estimated from an optional `boiling_points` / `critical_temps` you pass |

The NASA-9 form carries $1/T^2$ and $1/T$ terms that difflow's cubic
$C_p = a + bT + cT^2 + dT^3$ cannot represent exactly, so `Cp_coeffs` come from
a fit over the window you care about — set `T_fit_range` to your operating
range. Samples outside a species' valid interval (where pyglenn raises) are
dropped automatically.

```{note}
pyglenn supplies **no critical properties** (Tc, Pc, ω), so there is no
`import_critical_props` here. To build a `PengRobinson`/`SRK` or a
`CubicThermo`, pair the ideal-gas `SpeciesData` from pyglenn with
`CriticalProperties` from `difflow.database` or `difflow.cantera_import`:

    from difflow.eos import PengRobinson
    from difflow.thermo import IdealThermo, CubicThermo
    from difflow.cantera_import import import_critical_props

    sp   = import_species_data(["CH4", "CO2", "H2O"])            # ideal-gas Cp (pyglenn)
    crit = import_critical_props("gri30.yaml", ["CH4", "CO2", "H2O"])  # Tc/Pc/ω (Cantera)
    thermo = CubicThermo(IdealThermo(sp), PengRobinson(crit))    # real-gas enthalpy
```

### Supported NASA Glenn Data

| Data Type | Support |
|-----------|---------|
| Ideal-gas Cp (NASA 9) | Full (cubic fit) |
| Molecular weight | Full |
| Enthalpy of formation (298 K) | Full |
| Critical properties | Not provided by pyglenn |
| Liquid Hvap / vapor pressure | Placeholder/estimated only |

---

(dwsim-import)=
## DWSIM Import

**Location**: `difflow/dwsim_import.py`

Import compound constants and ideal-gas heat capacities from
[DWSIM](https://dwsim.org)'s thermodynamics library. DWSIM is a .NET
application, so it is reached from Python through
[`pythonnet`](https://github.com/pythonnet/pythonnet) against
`DWSIM.Thermodynamics.dll`, whose `CalculatorInterface.Calculator` is the
"DTL" calculator (an older `DWSIM.Thermodynamics.StandaloneLibrary.dll` is
used if that is what the folder holds). Checked against DWSIM 9.0.5 on Linux:

```bash
scripts/install_dwsim.sh        # DWSIM 9.0.5 .deb unpacked, .NET 8 runtime, pythonnet
export DWSIM_PATH=.../usr/local/lib/dwsim
```

DWSIM 9 is a .NET 8 build: pythonnet has to load CoreCLR (`pythonnet.load
("coreclr")`, which the importer does), not Mono, and only one CLR can be
loaded in a process, so do not `import clr` before the first import.

```{important}
Calls into DWSIM return concrete numbers through the CLR and are **not
differentiable** — JAX cannot trace through them. So, exactly like the Cantera
and pyglenn importers, this adapter uses DWSIM only as a **one-time data
source**: it reads each compound's constants and samples its ideal-gas Cp(T),
then builds difflow's own JAX-native `SpeciesData` / `CriticalProperties`.
difflow stays differentiable end to end; DWSIM is never in the gradient path.
```

Because DWSIM has critical constants (unlike pyglenn), it feeds **both**
`SpeciesData` and `CriticalProperties`, so it can build a full EOS/`CubicThermo`
on its own:

<!-- doc-test: skip: needs a DWSIM installation (.NET runtime) -->
```python
from difflow.dwsim_import import import_species_data, import_critical_props
from difflow.thermo import IdealThermo, CubicThermo
from difflow.eos import PengRobinson

from difflow.dwsim_import import DWSIMBackend, dwsim_name

names = [dwsim_name(n) for n in ("methane", "co2", "water")]   # DWSIM's names
be   = DWSIMBackend()                     # DWSIM_PATH, or dwsim_path=...; ~2 s to start
sp   = import_species_data(names, backend=be)
crit = import_critical_props(names, backend=be)
thermo = CubicThermo(IdealThermo(sp), PengRobinson(crit))
```

`DWSIM_NAMES` maps difflow's database names to DWSIM's for the refinery's
50 species, every one checked to exist in DWSIM 9.0.5 (all in its ChemSep
database). `tests/test_dwsim_import.py` (release) imports them from the real
DWSIM in a subprocess and compares Tc, Pc, omega and MW with
`difflow.database`; the differences are listed under
[Validation against DWSIM](unit-operations-refinery.md#refinery-dwsim-validation).

### What is imported (and DWSIM units)

| difflow field | DWSIM source (`ICompoundConstantProperties`) | Unit conversion |
|---------------|-----------------------------------------------|-----------------|
| `MW` | `Molar_Weight` | kg/kmol ≡ g/mol |
| `CriticalProperties.Tc/Pc/omega` | `Critical_Temperature`, `Critical_Pressure`, `Acentric_Factor` | K, Pa, — |
| `Hf` | `IG_Enthalpy_of_Formation_25C` | kJ/kg × MW → J/mol |
| `Cp_coeffs` | ideal-gas Cp via the calculator's `GetCompoundTDepProp(name, "idealGasHeatCapacity", T)` | J/mol/K, then cubic fit |
| `Hvap_coeffs`, `antoine_coeffs` | estimated from Tb/Tc | — |

```{note}
DWSIM cannot run in difflow's per-commit CI (no .NET runtime). All DWSIM
contact is isolated in `DWSIMBackend`; the import logic is backend-agnostic
and unit-tested against a fake backend every commit, and against DWSIM 9.0.5
itself in the release tier (skipped where DWSIM is not installed). For
another DWSIM build, replace `DWSIMBackend` and pass it via `backend=`.
```

---

## Usage Examples

### Complete VLE Flash with Ideal Thermo

```python
from difflow.thermo import IdealThermo
from difflow.database import get_species_data
from difflow.units.flash import Flash, FlashParams
from difflow.streams import make_stream

# Build thermo model from database
species_names = ['benzene', 'toluene', 'ethylbenzene']
species_data = {name: get_species_data(name) for name in species_names}
thermo = IdealThermo(species_data)

# Create feed stream
feed = make_stream(
    {'benzene': 0.4, 'toluene': 0.35, 'ethylbenzene': 0.25},
    T=380.0,  # K
    P=101325.0  # Pa
)

# Flash calculation
flash = Flash(FlashParams(species_order=species_names), thermo)
liquid, vapor, info = flash(feed)

print(f"Vapor fraction: {float(info['V_frac']):.3f}")
print(f"K-values: {info['K']}")
```

### High-Pressure Flash with PR EOS

```python
from difflow.eos import PengRobinson
from difflow.database import get_critical_props

# Build PR model from database
species_names = ['methane', 'ethane', 'propane', 'n_butane']
critical_props = {name: get_critical_props(name) for name in species_names}
pr = PengRobinson(critical_props)

# High-pressure flash
import jax.numpy as jnp
from difflow.eos import flash_TP_eos

z = jnp.array([0.5, 0.3, 0.15, 0.05])  # in species_names order
V_frac, x, y = flash_TP_eos(pr, z, T=250.0, P=3.0e6)  # 30 bar

print(f"Vapor fraction: {float(V_frac):.3f}")
print(f"Liquid methane: {float(x[0]):.4f}")
print(f"Vapor methane: {float(y[0]):.4f}")
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

*Activity coefficient models (NRTL, UNIQUAC, Wilson, Margules, van Laar): see [Activity-Coefficient Models](#activity-coefficient-models) and `Flash(activity_model=...)`.

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
