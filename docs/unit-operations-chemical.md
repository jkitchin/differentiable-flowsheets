# Chemical Unit Operations

This document provides comprehensive documentation for all chemical unit operations available in Difflow.

---

(reactors)=
## Reactors

(cstr-continuous-stirred-tank-reactor)=
### CSTR (Continuous Stirred-Tank Reactor)

**Location**: `difflow/units/cstr.py`

**Class**: `CSTR`

**Description**: Models an ideal continuous stirred-tank reactor with perfect mixing. The reactor contents are assumed to be at uniform temperature and composition, equal to the outlet conditions.

#### Process Role

CSTRs are widely used in chemical processes for:
- Liquid-phase reactions
- Polymerization reactions
- Fermentation (as idealized model)
- Processes requiring uniform conditions

#### Parameters

```python
@dataclass
class CSTRParams:
    volume: float          # Reactor volume (m³)
    stoichiometry: Array   # Stoichiometric matrix [n_species × n_reactions]
    k_ref: float           # Rate constant at reference temperature (1/s for 1st order)
    E_a: float             # Activation energy (J/mol)
    T_ref: float           # Reference temperature for k_ref (K)
    dH_rxn: Array          # Heat of reaction for each reaction (J/mol)
    order: int = 1         # Reaction order (default: 1)
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Inlet stream with species flows, T, P |
| `T_spec` | float | K | Target outlet temperature (isothermal mode) |
| `Q_spec` | float | W | Specified heat duty (specified_duty mode) |
| `volumetric_flow` | float | m³/s | Volumetric flow rate (optional) |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `outlet` | Stream | - | Outlet stream |
| `info['Q']` | float | W | Heat duty (positive = heating) |
| `info['rates']` | Array | mol/m³/s | Reaction rates |
| `info['conversion']` | float | - | Conversion of limiting reactant |

#### Operating Modes

1. **Isothermal** (`T_spec` provided): Outlet temperature is fixed, heat duty calculated
2. **Adiabatic** (no `T_spec` or `Q_spec`): Q = 0, outlet temperature calculated
3. **Specified Duty** (`Q_spec` provided): Heat duty fixed, outlet temperature calculated

#### Governing Equations

**Material Balance** (steady-state):

$$F_{i,out} = F_{i,in} + V \sum_j \nu_{ij} r_j$$

Where:
- $F_{i,out}$: Outlet molar flow of species i (mol/s)
- $F_{i,in}$: Inlet molar flow of species i (mol/s)
- $V$: Reactor volume (m³)
- $\nu_{ij}$: Stoichiometric coefficient of species i in reaction j
- $r_j$: Rate of reaction j (mol/m³/s)

**Reaction Rate** (Arrhenius kinetics):

$$r_j = k_{ref} \exp\left[\frac{E_a}{R}\left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right] \prod_i C_i^{n_i}$$

Where:
- $k_{ref}$: Rate constant at reference temperature
- $E_a$: Activation energy (J/mol)
- $R$: Gas constant (8.314 J/mol/K)
- $C_i$: Concentration of species i (mol/m³)
- $n_i$: Reaction order with respect to species i

**Energy Balance**:

$$Q = \dot{H}_{out} - \dot{H}_{in} + V \sum_j r_j \Delta H_{rxn,j}$$

Where:
- $Q$: Heat duty (W)
- $\dot{H}$: Enthalpy flow rate (W)
- $\Delta H_{rxn,j}$: Heat of reaction j (J/mol)

**Conversion**:

$$X = \frac{F_{A,in} - F_{A,out}}{F_{A,in}}$$

#### Example Usage

```python
from difflow.units.cstr import CSTR, CSTRParams
from difflow.streams import make_stream
from difflow.thermo import IdealThermo
import jax.numpy as jnp

# A -> B (first-order, exothermic)
params = CSTRParams(
    volume=2.0,  # m³
    stoichiometry=jnp.array([[-1.0], [1.0]]),  # A -> B
    k_ref=0.1,   # 1/s at 350 K
    E_a=50000.0, # J/mol
    T_ref=350.0, # K
    dH_rxn=jnp.array([-80000.0])  # J/mol (exothermic)
)

cstr = CSTR(params, thermo, species_order=['A', 'B'])
inlet = make_stream({'A': 1.0, 'B': 0.0}, T=350.0, P=101325.0)

# Isothermal operation
outlet, info = cstr(inlet, T_spec=350.0)
print(f"Conversion: {info['conversion']:.2%}")
print(f"Heat duty: {info['Q']:.2f} W")

# Adiabatic operation
outlet_adiab, info_adiab = cstr(inlet)
print(f"Outlet temperature: {outlet_adiab['T']:.1f} K")
```

#### Design Considerations

- **Residence Time**: $\tau = V/Q_{vol}$ determines conversion
- **Heat Transfer**: Large exothermic reactions may require cooling coils or jackets
- **Mixing**: Perfect mixing assumption requires adequate agitation
- **Multiple CSTRs**: Series arrangement approaches PFR behavior

---

(pfr-plug-flow-reactor)=
### PFR (Plug Flow Reactor)

**Location**: `difflow/units/pfr.py`

**Class**: `PFR`

**Description**: Models an ideal plug flow reactor where all fluid elements have the same residence time. No axial mixing, but perfect radial mixing is assumed.

#### Process Role

PFRs are preferred for:
- Gas-phase reactions
- High conversion requirements
- Reactions where selectivity depends on conversion
- Fast reactions

#### Parameters

```python
@dataclass
class PFRParams:
    V: float               # Total reactor volume (m³)
    rate_fn: Callable      # Rate function: rate_fn(C, T, rate_params) -> r
    stoich: Array          # Stoichiometry matrix [n_species × n_reactions]
    rate_params: dict      # Parameters passed to rate_fn
    species_order: list    # List of species names
    dH_rxn: Array = None   # Heat of reaction (J/mol), required for non-isothermal
    rtol: float = 1e-6     # Relative tolerance for ODE solver
    atol: float = 1e-8     # Absolute tolerance for ODE solver
    n_save_points: int = 101  # Points to save in profile output
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Inlet stream |
| `T_spec` | float | K | Outlet temperature (isothermal mode) |
| `volumetric_flow` | float | m³/s | Volumetric flow rate |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `outlet` | Stream | - | Outlet stream |
| `info['conversion']` | float | - | Conversion of limiting reactant |
| `info['V_profile']` | Array | m³ | Volume along reactor |
| `info['F_profile']` | Array | mol/s | Molar flows along reactor |
| `info['T_profile']` | Array | K | Temperature along reactor |

#### Governing Equations

**Material Balance** (differential):

$$\frac{dF_i}{dV} = \sum_j \nu_{ij} r_j$$

**Energy Balance** (adiabatic):

$$\frac{dT}{dV} = \frac{-\sum_j r_j \Delta H_{rxn,j}}{F_{total} C_{p,mix}}$$

**Integration Method**: Adaptive ODE integration using diffrax (Tsit5 or Dopri5 solvers)

#### Example Usage

```python
from difflow.units.pfr import PFR, PFRParams
import jax.numpy as jnp

# Define rate function: A -> B (first-order)
def rate_fn(C, T, params):
    """Rate function: r = k * C_A with Arrhenius temperature dependence."""
    k_ref, E_a, T_ref = params['k_ref'], params['E_a'], params['T_ref']
    R = 8.314
    k = k_ref * jnp.exp(E_a / R * (1/T_ref - 1/T))
    return jnp.array([k * C['A']])

params = PFRParams(
    V=5.0,
    rate_fn=rate_fn,
    stoich=jnp.array([[-1.0], [1.0]]),  # A -> B
    rate_params={'k_ref': 0.5, 'E_a': 60000.0, 'T_ref': 400.0},
    species_order=['A', 'B'],
    dH_rxn=jnp.array([-50000.0]),
    n_save_points=201
)

pfr = PFR(params, thermo)
inlet = make_stream({'A': 2.0, 'B': 0.0}, T=400.0, P=200000.0)
outlet, info = pfr(inlet, Q=0.001)  # Q = volumetric flow rate

# Plot conversion profile
import matplotlib.pyplot as plt
plt.plot(info['V_profile'], 1 - info['F_profile'][:, 0] / inlet['F_A'])
plt.xlabel('Volume (m³)')
plt.ylabel('Conversion')
```

---

(gaspfr-gas-phase-pfr-with-pressure-drop)=
### GasPFR (Gas-Phase PFR with Pressure Drop)

**Location**: `difflow/units/pfr.py`

**Class**: `GasPFR`

**Description**: Extended PFR model for gas-phase reactions accounting for:
- Pressure drop (Ergun equation)
- Variable volumetric flow due to mole change and pressure/temperature effects

#### Additional Parameters

```python
@dataclass
class GasPFRParams(PFRParams):
    alpha: float           # Pressure drop parameter (1/m³)
    diameter: float        # Reactor diameter (m)
    void_fraction: float   # Bed void fraction (for packed beds)
    particle_diameter: float  # Catalyst particle diameter (m)
```

#### Additional Equations

**Pressure Drop** (Ergun equation):

$$\frac{dP}{dV} = -\alpha \frac{P_0}{P} \frac{T}{T_0} \frac{F_{total}}{F_{total,0}}$$

Where $\alpha$ combines Ergun parameters:

$$\alpha = \frac{G}{\rho_0 g_c D_p A_c} \left[\frac{150(1-\phi)\mu}{D_p} + 1.75 G\right] \frac{(1-\phi)}{\phi^3}$$

**Variable Volumetric Flow**:

$$Q = Q_0 \frac{F_{total}}{F_{total,0}} \frac{P_0}{P} \frac{T}{T_0}$$

#### Outputs

Additional outputs compared to PFR:

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `info['P_profile']` | Array | Pa | Pressure along reactor |
| `info['pressure_drop']` | float | Pa | Total pressure drop |

---

(fedbatchreactor)=
### FedBatchReactor

**Location**: `difflow/units/fed_batch.py`

**Class**: `FedBatchReactor`, `SemiBatchReactor`

**Description**: Models fed-batch (semi-batch) reactors with time-varying feed profiles. Commonly used when reactant addition rate affects selectivity or safety.

#### Process Role

Fed-batch reactors are used for:
- Controlling exothermic reactions
- Improving selectivity by maintaining low reactant concentration
- Fermentation with substrate feeding
- Polymerization with monomer addition

#### Parameters

```python
@dataclass
class FedBatchParams:
    V0: float              # Initial reactor volume (m³)
    rate_fn: Callable      # Rate function: rate_fn(C, T, rate_params) -> r
    stoich: Array          # Stoichiometry matrix [n_species × n_reactions]
    rate_params: dict      # Parameters passed to rate_fn
    species_order: list    # List of species names
    dH_rxn: Array = None   # Heat of reaction (J/mol), None for isothermal
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream composition |
| `feed_flow` | Callable | mol/s | Feed rate as function of time: `F(t)` |
| `T_profile` | Callable | K | Temperature as function of time: `T(t)` |

#### Governing Equations

**Volume Change**:

$$\frac{dV}{dt} = Q_{feed}$$

**Material Balance**:

$$\frac{d(V C_i)}{dt} = F_{in} C_{in,i} + V \sum_j \nu_{ij} r_j$$

Or equivalently:

$$\frac{dN_i}{dt} = F_{in,i} + V \sum_j \nu_{ij} r_j$$

**Energy Balance**:

$$\frac{d(V \rho C_p T)}{dt} = F_{in} \rho_{in} C_{p,in} T_{in} + V \sum_j r_j (-\Delta H_{rxn,j}) + Q$$

#### Utility Functions

```python
from difflow.units.fed_batch import batch_time_for_conversion, optimal_feed_profile

# Calculate batch time for target conversion
t_batch = batch_time_for_conversion(params, target_X=0.95)

# Generate optimal feed profile (minimize batch time)
feed_profile = optimal_feed_profile(params, constraints={'max_T': 400.0})
```

---

(separators)=
## Separators

(flash-drum)=
### Flash Drum

**Location**: `difflow/units/flash.py`

**Classes**: `Flash`, `EOSFlash`, `PHFlash`

**Description**: Performs vapor-liquid equilibrium (VLE) separation. The feed is separated into vapor and liquid phases at equilibrium conditions.

#### Process Role

Flash drums are used for:
- Separating light and heavy components
- Pressure reduction with phase separation
- Overhead condensers
- Feed preparation for distillation

#### Parameters

```python
@dataclass
class FlashParams:
    species_order: list[str]  # List of species names for array ordering

@dataclass
class EOSFlashParams:
    species_order: list[str]  # List of species names for array ordering
    eos_type: str = "PR"      # "PR" (Peng-Robinson) or "SRK"
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream |
| `T` | float | K | Flash temperature (optional override) |
| `P` | float | Pa | Flash pressure (optional override) |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `liquid` | Stream | - | Liquid product |
| `vapor` | Stream | - | Vapor product |
| `info['V_frac']` | float | - | Vapor fraction |
| `info['K']` | dict | - | K-values for each species |
| `info['x']` | dict | - | Liquid mole fractions |
| `info['y']` | dict | - | Vapor mole fractions |

#### Governing Equations

**Rachford-Rice Equation**:

$$f(V) = \sum_i \frac{z_i (K_i - 1)}{1 + V(K_i - 1)} = 0$$

Where:
- $z_i$: Feed mole fraction of species i
- $K_i$: Equilibrium ratio (K-value) = $y_i / x_i$
- $V$: Vapor fraction (moles vapor / moles feed)

**Phase Compositions**:

$$x_i = \frac{z_i}{1 + V(K_i - 1)}$$

$$y_i = \frac{K_i z_i}{1 + V(K_i - 1)}$$

**K-Value Calculation** (Raoult's Law for `Flash`):

$$K_i = \frac{P_i^{sat}(T)}{P}$$

**K-Value Calculation** (Fugacity-based for `EOSFlash`):

$$K_i = \frac{\phi_i^L}{\phi_i^V}$$

Where $\phi_i$ are fugacity coefficients from Peng-Robinson or SRK equation of state.

**Material Balance**:

$$F = L + V$$

$$F z_i = L x_i + V y_i$$

#### Flash Classes

##### Flash (Ideal)

Uses Raoult's law K-values from IdealThermo. Suitable for ideal or near-ideal mixtures.

```python
from difflow.units.flash import Flash, FlashParams

flash = Flash(FlashParams(species_order=['methane', 'ethane', 'propane']), thermo)
feed = make_stream({'methane': 0.5, 'ethane': 0.3, 'propane': 0.2}, T=300.0, P=500000.0)

liquid, vapor, info = flash(feed)
print(f"Vapor fraction: {info['V_frac']:.3f}")
```

##### EOSFlash (Non-Ideal)

Uses fugacity coefficients from cubic equations of state (Peng-Robinson or SRK) for non-ideal VLE.

```python
from difflow.units.flash import EOSFlash, EOSFlashParams
from difflow.eos import PengRobinson, CriticalProperties

# Define species with critical properties
species_data = {
    "methane": CriticalProperties("methane", 190.6, 4.6e6, 0.011),
    "ethane": CriticalProperties("ethane", 305.4, 4.9e6, 0.099),
    "propane": CriticalProperties("propane", 369.8, 4.2e6, 0.152),
}
eos = PengRobinson(species_data)

params = EOSFlashParams(species_order=["methane", "ethane", "propane"], eos_type="PR")
flash = EOSFlash(params, eos)

feed = make_stream({'methane': 40.0, 'ethane': 30.0, 'propane': 30.0}, T=250.0, P=2e6)
liquid, vapor, info = flash(feed)
```

##### PHFlash (Isenthalpic)

Performs adiabatic flash at constant pressure and enthalpy. Solves for flash temperature.

```python
from difflow.units.flash import PHFlash, FlashParams

ph_flash = PHFlash(FlashParams(species_order=['Light', 'Heavy']), thermo)

# Hot liquid feed, flash to lower pressure
feed = make_stream({'Light': 50.0, 'Heavy': 50.0}, T=380.0, P=101325.0)
liquid, vapor, info = ph_flash(feed, P=30000.0)

print(f"Flash temperature: {info['T_flash']:.1f} K")
print(f"Vapor fraction: {info['V_frac']:.3f}")
```

#### Bubble and Dew Point Methods

The `Flash` class provides methods for calculating phase boundaries:

```python
flash = Flash(FlashParams(species_order=['Light', 'Heavy']), thermo)
feed = make_stream({'Light': 50.0, 'Heavy': 50.0}, T=350.0, P=50000.0)

# Pressure calculations (at specified T)
P_bubble = flash.bubble_point_pressure(feed, T=350.0)  # First bubble forms
P_dew = flash.dew_point_pressure(feed, T=350.0)        # Last drop condenses

# Temperature calculations (at specified P)
T_bubble = flash.bubble_point_temperature(feed, P=50000.0)
T_dew = flash.dew_point_temperature(feed, P=50000.0)

print(f"Bubble point: T={T_bubble:.1f} K, P={P_bubble:.0f} Pa")
print(f"Dew point: T={T_dew:.1f} K, P={P_dew:.0f} Pa")
```

---

### Mixer

**Location**: `difflow/units/flash.py`

**Class**: `Mixer`

**Description**: Combines multiple inlet streams into a single outlet stream by summing molar flows.

#### Governing Equations

**Mass Balance**:

$$F_{out,i} = \sum_k F_{k,i}$$

**Energy Balance** (adiabatic mixing):

$$T_{out} = \frac{\sum_k F_k C_{p,k} T_k}{\sum_k F_k C_{p,k}}$$

(Simplified for ideal mixing with similar heat capacities)

#### Example Usage

```python
from difflow.units.flash import Mixer

mixer = Mixer(thermo, species_order=['A', 'B', 'C'])
stream1 = make_stream({'A': 1.0, 'B': 0.5}, T=350.0, P=101325.0)
stream2 = make_stream({'B': 0.3, 'C': 0.2}, T=360.0, P=101325.0)

outlet, info = mixer([stream1, stream2])
```

---

### Splitter

**Location**: `difflow/units/flash.py`

**Class**: `Splitter`

**Description**: Divides a single inlet stream into multiple outlet streams based on split fractions.

#### Governing Equations

$$F_{out,k} = \alpha_k F_{in}$$

Where $\sum_k \alpha_k = 1$

All outlet streams have the same composition and temperature as the inlet.

#### Example Usage

```python
from difflow.units.flash import Splitter

splitter = Splitter()
inlet = make_stream({'A': 1.0, 'B': 0.5}, T=350.0, P=101325.0)

# Split into 3 streams: 50%, 30%, 20%
outlets, info = splitter(inlet, fractions=[0.5, 0.3, 0.2])
```

---

## Distillation

### ShortcutColumn

**Location**: `difflow/units/distillation.py`

**Class**: `ShortcutColumn`

**Description**: Uses shortcut methods (Fenske-Underwood-Gilliland) for rapid distillation column design and rating calculations.

#### Process Role

Shortcut methods are used for:
- Initial column design estimates
- Optimization studies
- Screening alternatives
- Quick sensitivity analysis

#### Parameters

```python
@dataclass
class ShortcutColumnParams:
    light_key: int         # Index of light key component
    heavy_key: int         # Index of heavy key component
    x_D_lk: float         # Light key recovery in distillate
    x_B_hk: float         # Heavy key recovery in bottoms
    reflux_ratio: float   # Actual reflux ratio (R/R_min multiplier)
```

#### Governing Equations

**Relative Volatility**:

$$\alpha_{ij} = \frac{K_i}{K_j} = \frac{P_i^{sat}}{P_j^{sat}}$$

**Average Relative Volatility** (geometric mean):

$$\bar{\alpha} = (\alpha_{top} \cdot \alpha_{bottom})^{0.5}$$

**Fenske Equation** (minimum stages):

$$N_{min} = \frac{\ln\left[\frac{x_{D,LK}}{x_{B,LK}} \cdot \frac{x_{B,HK}}{x_{D,HK}}\right]}{\ln \bar{\alpha}_{LK/HK}}$$

**Underwood Equations** (minimum reflux):

For each component i:
$$\sum_i \frac{\alpha_i x_{F,i}}{\alpha_i - \theta} = 1 - q$$

$$R_{min} + 1 = \sum_i \frac{\alpha_i x_{D,i}}{\alpha_i - \theta}$$

Where:
- $\theta$: Root between $\alpha_{HK}$ and $\alpha_{LK}$
- $q$: Feed quality (1 for saturated liquid, 0 for saturated vapor)

**Gilliland Correlation** (actual stages):

$$\frac{N - N_{min}}{N + 1} = 1 - \exp\left[\frac{(1 + 54.4X)(X - 1)}{(11 + 117.2X)(X^{0.5})}\right]$$

Where:
$$X = \frac{R - R_{min}}{R + 1}$$

**Feed Stage Location** (Kirkbride correlation):

$$\frac{N_R}{N_S} = \left[\frac{B}{D} \cdot \frac{x_{F,HK}}{x_{F,LK}} \cdot \left(\frac{x_{B,LK}}{x_{D,HK}}\right)^2\right]^{0.206}$$

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `distillate` | Stream | - | Overhead product |
| `bottoms` | Stream | - | Bottom product |
| `info['N_min']` | float | - | Minimum stages |
| `info['N_actual']` | float | - | Actual stages |
| `info['R_min']` | float | - | Minimum reflux ratio |
| `info['feed_stage']` | int | - | Optimal feed stage |
| `info['condenser_duty']` | float | W | Condenser heat duty |
| `info['reboiler_duty']` | float | W | Reboiler heat duty |

#### Example Usage

```python
from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams

params = ShortcutColumnParams(
    light_key=0,    # benzene
    heavy_key=1,    # toluene
    x_D_lk=0.99,    # 99% benzene recovery in distillate
    x_B_hk=0.99,    # 99% toluene recovery in bottoms
    reflux_ratio=1.3  # 1.3 × R_min
)

column = ShortcutColumn(params, thermo, species_order=['benzene', 'toluene', 'xylene'])
feed = make_stream({'benzene': 0.4, 'toluene': 0.35, 'xylene': 0.25}, T=370.0, P=101325.0)

distillate, bottoms, info = column(feed)
print(f"Minimum stages: {info['N_min']:.1f}")
print(f"Actual stages: {info['N_actual']:.1f}")
print(f"Condenser duty: {info['condenser_duty']/1e6:.2f} MW")
```

#### Utility Functions

```python
from difflow.units.distillation import (
    relative_volatility,
    fenske_stages,
    minimum_reflux_ratio,
    gilliland_stages,
    column_diameter
)

# Calculate individual parameters
alpha = relative_volatility(thermo, T=373.0, P=101325.0)
N_min = fenske_stages(x_D, x_B, alpha)
R_min = minimum_reflux_ratio(alpha, x_F, x_D, q=1.0)
N = gilliland_stages(N_min, R, R_min)
D = column_diameter(V_max, rho_V, rho_L, sigma)
```

---

### DistillationColumn (Rigorous)

**Location**: `difflow/units/distillation.py`

**Class**: `DistillationColumn`

**Description**: Stage-by-stage calculation using MESH equations (Material balance, Equilibrium, Summation, enthalpy balance).

#### Parameters

```python
@dataclass
class DistillationColumnParams:
    n_stages: int          # Number of theoretical stages
    feed_stage: int        # Feed stage number (from top)
    reflux_ratio: float    # Reflux ratio (L/D)
    condenser_type: str    # 'total' or 'partial'
    P_top: float           # Top pressure (Pa)
    P_bottom: float        # Bottom pressure (Pa)
```

#### Governing Equations (MESH)

For each stage j:

**Material Balance**:
$$L_{j-1} x_{i,j-1} + V_{j+1} y_{i,j+1} + F_j z_{i,j} = L_j x_{i,j} + V_j y_{i,j}$$

**Equilibrium**:
$$y_{i,j} = K_{i,j} x_{i,j}$$

**Summation**:
$$\sum_i x_{i,j} = 1$$
$$\sum_i y_{i,j} = 1$$

**Enthalpy Balance**:
$$L_{j-1} H^L_{j-1} + V_{j+1} H^V_{j+1} + F_j H^F_j = L_j H^L_j + V_j H^V_j + Q_j$$

---

## Heat Exchangers

### Heater

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `Heater`

**Description**: Single-stream heater that increases stream temperature using an external heat source (steam, hot oil, electric).

#### Parameters

```python
@dataclass
class HeaterParams:
    mode: str              # 'duty', 'outlet_T', or 'lmtd'
    duty: float = None     # Heat duty (W) for 'duty' mode
    T_out: float = None    # Outlet temperature (K) for 'outlet_T' mode
    UA: float = None       # Overall HTC × Area (W/K) for 'lmtd' mode
    T_utility: float = None  # Utility temperature (K) for 'lmtd' mode
```

#### Governing Equations

**Energy Balance**:

$$Q = \dot{m} C_p (T_{out} - T_{in})$$

Or in molar terms:

$$Q = F_{total} C_{p,mix} (T_{out} - T_{in})$$

**LMTD Rating** (for utility heating):

$$Q = UA \cdot LMTD$$

$$LMTD = \frac{(T_U - T_{in}) - (T_U - T_{out})}{\ln\left(\frac{T_U - T_{in}}{T_U - T_{out}}\right)}$$

Where $T_U$ is the utility (steam) temperature.

#### Example Usage

```python
from difflow.units.heat_exchanger import Heater, HeaterParams

# Specified duty mode
heater = Heater(HeaterParams(mode='duty', duty=50000.0), thermo, species_order=['A', 'B'])
outlet, info = heater(inlet)

# Specified outlet temperature mode
heater = Heater(HeaterParams(mode='outlet_T', T_out=400.0), thermo, species_order=['A', 'B'])
outlet, info = heater(inlet)
```

---

### Cooler

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `Cooler`

**Description**: Single-stream cooler that decreases stream temperature using cooling water or refrigeration.

#### Parameters

Same as Heater with appropriate utility temperatures.

#### Governing Equations

Same as Heater (Q is negative for cooling).

---

### CounterCurrentHX

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `CounterCurrentHX`

**Description**: Two-stream heat exchanger with counter-current flow arrangement. Provides maximum temperature driving force.

#### Process Role

Counter-current heat exchangers are preferred for:
- Maximum heat recovery
- Heating/cooling to approach inlet temperature of other stream
- Most efficient use of heat transfer area

#### Parameters

```python
@dataclass
class HeatExchangerParams:
    mode: str              # 'design' or 'rating'
    UA: float = None       # Overall HTC × Area (W/K) for rating
    approach: float = 10.0 # Minimum approach temperature (K) for design
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_stream` | Stream | - | Hot fluid inlet |
| `cold_stream` | Stream | - | Cold fluid inlet |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_outlet` | Stream | - | Hot fluid outlet |
| `cold_outlet` | Stream | - | Cold fluid outlet |
| `info['Q']` | float | W | Heat duty transferred |
| `info['LMTD']` | float | K | Log mean temperature difference |
| `info['UA']` | float | W/K | Required UA (design mode) |

#### Governing Equations

**Energy Balance**:

$$Q = \dot{m}_h C_{p,h} (T_{h,in} - T_{h,out}) = \dot{m}_c C_{p,c} (T_{c,out} - T_{c,in})$$

**LMTD** (Counter-current):

$$LMTD = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}$$

Where:
- $\Delta T_1 = T_{h,in} - T_{c,out}$
- $\Delta T_2 = T_{h,out} - T_{c,in}$

**Heat Transfer Rate**:

$$Q = UA \cdot LMTD$$

**Effectiveness-NTU Method**:

$$\epsilon = \frac{Q}{Q_{max}} = \frac{Q}{C_{min}(T_{h,in} - T_{c,in})}$$

$$\epsilon = \frac{1 - \exp[-NTU(1 - C_r)]}{1 - C_r \exp[-NTU(1 - C_r)]}$$

Where:
- $C_r = C_{min}/C_{max}$
- $NTU = UA/C_{min}$
- $C = \dot{m} C_p$ (heat capacity rate)

#### Example Usage

```python
from difflow.units.heat_exchanger import CounterCurrentHX, HeatExchangerParams

hx = CounterCurrentHX(
    HeatExchangerParams(mode='rating', UA=5000.0),
    thermo,
    species_order=['A', 'B']
)

hot_in = make_stream({'A': 1.0}, T=450.0, P=101325.0)
cold_in = make_stream({'B': 0.8}, T=300.0, P=101325.0)

hot_out, cold_out, info = hx(hot_in, cold_in)
print(f"Heat duty: {info['Q']/1000:.2f} kW")
print(f"LMTD: {info['LMTD']:.2f} K")
```

---

### CoCurrentHX

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `CoCurrentHX`

**Description**: Two-stream heat exchanger with co-current (parallel) flow arrangement.

#### Governing Equations

**LMTD** (Co-current):

$$LMTD = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}$$

Where:
- $\Delta T_1 = T_{h,in} - T_{c,in}$
- $\Delta T_2 = T_{h,out} - T_{c,out}$

**Effectiveness** (Co-current):

$$\epsilon = \frac{1 - \exp[-NTU(1 + C_r)]}{1 + C_r}$$

Note: Co-current flow cannot achieve temperature cross ($T_{c,out} > T_{h,out}$).

---

### CrossFlowHX

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `CrossFlowHX`

**Description**: Two-stream heat exchanger with cross-flow arrangement where fluids flow perpendicular to each other. Effectiveness depends on mixing configuration.

#### Process Role

Cross-flow heat exchangers are used for:
- Air-to-liquid heat transfer (HVAC systems)
- Car radiators and automotive cooling
- Finned-tube heat exchangers
- Applications where cross-flow geometry is advantageous

#### Parameters

```python
@dataclass
class HeatExchangerParams:
    UA: float = None       # Overall HTC × Area (W/K) for rating
    Cp_hot: float = None   # Hot side heat capacity (J/mol·K)
    Cp_cold: float = None  # Cold side heat capacity (J/mol·K)
    min_approach: float = 10.0  # Minimum approach temperature (K)

# CrossFlowHX constructor
CrossFlowHX(params: HeatExchangerParams, mixing: str = "both_unmixed")
```

#### Mixing Configurations

The `mixing` parameter specifies the flow arrangement:

| Configuration | Description | Common Applications |
|--------------|-------------|---------------------|
| `both_unmixed` | Both fluids flow through separate channels (default) | Car radiators, finned-tube HX |
| `cmax_mixed` | Larger heat capacity stream is mixed | Shell-and-tube with mixing in shell |
| `cmin_mixed` | Smaller heat capacity stream is mixed | Special geometries |
| `both_mixed` | Both fluids can mix in flow direction | Compact heat exchangers |

#### Governing Equations

**Energy Balance** (same as other HX types):

$$Q = \dot{m}_h C_{p,h} (T_{h,in} - T_{h,out}) = \dot{m}_c C_{p,c} (T_{c,out} - T_{c,in})$$

**Effectiveness** (both unmixed):

$$\epsilon = 1 - \exp\left[\frac{NTU^{0.22}}{C_r}\left(\exp(-C_r \cdot NTU^{0.78}) - 1\right)\right]$$

**Effectiveness** (Cmax mixed, Cmin unmixed):

$$\epsilon = \frac{1}{C_r}\left[1 - \exp\left(-C_r(1 - \exp(-NTU))\right)\right]$$

**Effectiveness** (Cmin mixed, Cmax unmixed):

$$\epsilon = 1 - \exp\left[-\frac{1}{C_r}(1 - \exp(-C_r \cdot NTU))\right]$$

**Effectiveness** (both mixed):

$$\frac{1}{\epsilon} = \frac{1}{1-\exp(-NTU)} + \frac{C_r}{1-\exp(-C_r \cdot NTU)} - \frac{1}{NTU}$$

Where:
- $NTU = UA/C_{min}$ (Number of Transfer Units)
- $C_r = C_{min}/C_{max}$ (Heat capacity ratio)

**Physical Constraint**: Cross-flow effectiveness is capped at the counter-current value to maintain physical consistency, as cross-flow should never exceed counter-current performance.

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_stream` | Stream | - | Hot fluid inlet |
| `cold_stream` | Stream | - | Cold fluid inlet |
| `UA` | float | W/K | Override UA value |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_outlet` | Stream | - | Hot fluid outlet |
| `cold_outlet` | Stream | - | Cold fluid outlet |
| `info['Q']` | float | W | Heat duty transferred |
| `info['effectiveness']` | float | - | Heat exchanger effectiveness |
| `info['NTU']` | float | - | Number of transfer units |
| `info['LMTD']` | float | K | Log mean temperature difference |
| `info['mixing']` | str | - | Mixing configuration used |
| `info['approach']` | float | K | Minimum temperature approach |

#### Example Usage

```python
from difflow import CrossFlowHX, HeatExchangerParams, make_stream

# Car radiator example (both unmixed - most common)
hx = CrossFlowHX(
    HeatExchangerParams(UA=2000.0, Cp_hot=75.0, Cp_cold=30.0),
    mixing="both_unmixed"  # Default
)

hot_coolant = make_stream({"ethylene_glycol": 10.0}, T=368.0, P=101325.0)  # 95°C
cold_air = make_stream({"air": 50.0}, T=298.0, P=101325.0)  # 25°C

hot_out, cold_out, info = hx(hot_coolant, cold_air)
print(f"Heat rejected: {info['Q']/1000:.2f} kW")
print(f"Effectiveness: {info['effectiveness']:.3f}")
print(f"Air outlet temp: {cold_out['T']:.1f} K")

# Compare mixing configurations
for config in ["both_unmixed", "cmax_mixed", "cmin_mixed", "both_mixed"]:
    hx = CrossFlowHX(HeatExchangerParams(UA=2000.0), mixing=config)
    _, _, info = hx(hot_coolant, cold_air)
    print(f"{config:15s}: ε = {info['effectiveness']:.4f}, Q = {info['Q']/1000:.2f} kW")
```

#### Performance Comparison

For the same UA and inlet conditions, effectiveness ranking:
1. Counter-current (highest)
2. Cross-flow with mixed streams
3. Cross-flow (both unmixed)
4. Co-current (lowest)

Cross-flow heat exchangers offer intermediate performance between counter-current (most efficient) and co-current (simplest), making them practical for applications where perpendicular flow geometry is advantageous.

---

### Heat Exchanger Utility Functions

```python
from difflow.units.heat_exchanger import (
    log_mean_temperature_difference,
    effectiveness_counter_current,
    effectiveness_co_current,
    effectiveness_crossflow_both_unmixed,
    effectiveness_crossflow_cmax_mixed,
    effectiveness_crossflow_cmin_mixed,
    effectiveness_crossflow_both_mixed,
    heat_capacity_rate,
    design_heat_exchanger,
    size_heat_exchanger
)

# Calculate LMTD with numerical stability
lmtd = log_mean_temperature_difference(dT1=50.0, dT2=30.0)

# Calculate effectiveness for different flow configurations
eps_counter = effectiveness_counter_current(NTU=2.0, Cr=0.5)
eps_co = effectiveness_co_current(NTU=2.0, Cr=0.5)
eps_cross = effectiveness_crossflow_both_unmixed(NTU=2.0, Cr=0.5)

# Design for specified duty
UA, area = design_heat_exchanger(Q=100000, LMTD=30, U=500)
```

---

## Liquid-Liquid Extraction

### MultistageCascade

**Location**: `difflow/units/lle.py`

**Class**: `MultistageCascade`

**Description**: Counter-current multistage extraction cascade for liquid-liquid separation.

#### Process Role

LLE is used for:
- Separation of heat-sensitive compounds
- Aromatics extraction (BTX)
- Pharmaceutical purification
- Metal extraction (hydrometallurgy)

#### Parameters

```python
@dataclass
class CascadeParams:
    n_stages: int          # Number of equilibrium stages
    K_values: Array        # Distribution coefficients
    feed_stage: int = 1    # Feed entry stage
    mode: str = 'counter_current'
```

#### Governing Equations

**Distribution Coefficient**:

$$K_i = \frac{C_{i,extract}}{C_{i,raffinate}} = \frac{y_i}{x_i}$$

**Material Balance** (stage j):

$$R_{j-1} x_{i,j-1} + E_{j+1} y_{i,j+1} = R_j x_{i,j} + E_j y_{i,j}$$

**Equilibrium**:

$$y_{i,j} = K_i(T) x_{i,j}$$

**Kremser Equation** (for dilute systems):

$$\frac{x_{in} - x_{out}}{x_{in} - x_{out}^*} = \frac{A^{N+1} - A}{A^{N+1} - 1}$$

Where $A = KE/R$ is the extraction factor.

#### Activity Coefficient Models

**NRTL**:

$$\ln \gamma_i = \frac{\sum_j x_j \tau_{ji} G_{ji}}{\sum_k x_k G_{ki}} + \sum_j \frac{x_j G_{ij}}{\sum_k x_k G_{kj}} \left(\tau_{ij} - \frac{\sum_m x_m \tau_{mj} G_{mj}}{\sum_k x_k G_{kj}}\right)$$

Where:
- $G_{ij} = \exp(-\alpha_{ij} \tau_{ij})$
- $\tau_{ij} = (g_{ij} - g_{jj})/RT$

**UNIQUAC**:

$$\ln \gamma_i = \ln \gamma_i^C + \ln \gamma_i^R$$

Combinatorial and residual contributions based on molecular size and interaction parameters.

#### Example Usage

```python
from difflow.units.lle import MultistageCascade, CascadeParams
import jax.numpy as jnp

params = CascadeParams(
    n_stages=5,
    K_values=jnp.array([0.1, 2.5, 0.05]),  # solute favors extract phase
    mode='counter_current'
)

cascade = MultistageCascade(params)
feed = make_stream({'water': 100.0, 'acetic_acid': 10.0, 'butanol': 0.0}, T=298.0, P=101325.0)
solvent = make_stream({'water': 0.0, 'acetic_acid': 0.0, 'butanol': 50.0}, T=298.0, P=101325.0)

raffinate, extract, info = cascade(feed, solvent)
print(f"Recovery: {info['recovery']:.2%}")
```

#### Utility Functions

```python
from difflow.units.lle import (
    get_K_values,
    nrtl_activity_coefficients,
    uniquac_activity_coefficients,
    separation_factor,
    minimum_solvent_ratio,
    stages_for_recovery
)

# Estimate K-values from activity coefficients
K = get_K_values(gamma_extract, gamma_raffinate, x_eq)

# Calculate minimum solvent for given recovery
S_min = minimum_solvent_ratio(K, target_recovery=0.95)

# Estimate stages needed
N = stages_for_recovery(K, S/F, target_recovery=0.99)
```

---

### DifferentialContactor

**Location**: `difflow/units/lle.py`

**Class**: `DifferentialContactor`

**Description**: Continuous differential contact extraction column (spray, packed, or rotating disc).

#### Governing Equations

**Height of Transfer Unit (HTU)**:

$$HTU = \frac{R}{K_{OC} a A}$$

**Number of Transfer Units (NTU)**:

$$NTU = \int_{x_{out}}^{x_{in}} \frac{dx}{x - x^*}$$

**Column Height**:

$$H = HTU \times NTU$$

---

## Pressure-Change & EOS-Consistent Units

These units close energy balances on the **cubic-EOS enthalpy and entropy**
(ideal-gas Cp + Peng-Robinson/SRK departures) rather than on ideal-K or
constant-Cp models, so they are correct for real gases and near-cryogenic /
gas-processing service (expander plants, NGL recovery, refrigeration). Each
takes a [`CubicThermo`](thermodynamics.md) built from an `IdealThermo` (for the
ideal-gas Cp) and a `PengRobinson`/`SRK` EOS. All internal temperature solves
use `optimistix` root finds on the two-phase enthalpy/entropy, so every outlet
temperature, duty and shaft work is differentiable with respect to feed
conditions, discharge pressures and efficiencies.

```python
from difflow import (
    IdealThermo, CubicThermo, PengRobinson,
    Turboexpander, TurboexpanderParams,
    Compressor, CompressorParams,
    JTValve, JTValveParams,
    ComponentSeparator, ComponentSeparatorParams,
)
from difflow.database import get_critical_props, get_species_data
from difflow.streams import make_stream

names = ["nitrogen", "methane", "ethane", "propane", "n_butane"]
ideal = IdealThermo({c: get_species_data(c) for c in names})
eos = PengRobinson({c: get_critical_props(c) for c in names})
thermo = CubicThermo(ideal, eos)

feed = make_stream({"nitrogen": 0.5, "methane": 86.0, "ethane": 7.0,
                    "propane": 3.0, "n_butane": 1.0}, T=305.0, P=60e5)
```

### Turboexpander

Adiabatic expansion to `P_out` with an isentropic efficiency. The reversible
outlet is found by matching entropy, then the efficiency is applied to the
enthalpy drop:

$$S(T_\text{isen}, P_\text{out}) = S(T_\text{in}, P_\text{in}), \qquad
H_\text{out} = H_\text{in} + \eta\,(H_\text{isen} - H_\text{in})$$

The extracted shaft work is $W = H_\text{in} - H_\text{out} > 0$. Both enthalpy
and entropy are two-phase aware, so an expander whose outlet partly condenses
(common in cryogenic service) is handled correctly.

```python
exp = Turboexpander(TurboexpanderParams(P_out=20e5, eta_isentropic=0.80), thermo)
outlet, info = exp(feed)
# info: {"W", "T_isen", "T_out", "H_in", "H_out"}
```

### Compressor

Adiabatic compression to `P_out` with an isentropic efficiency. Same entropy
match, but the efficiency **inflates** the enthalpy rise (an inefficient machine
needs more work than the reversible one):

$$H_\text{out} = H_\text{in} + \frac{H_\text{isen} - H_\text{in}}{\eta}$$

The required shaft work is $W = H_\text{out} - H_\text{in} > 0$.

```python
comp = Compressor(CompressorParams(P_out=90e5, eta_isentropic=0.75), thermo)
outlet, info = comp(feed)
```

### JTValve (Joule-Thomson valve)

Adiabatic, **isenthalpic** pressure letdown. Holds the two-phase EOS enthalpy
constant across the pressure drop and solves for the outlet temperature,
$H(T_\text{out}, P_\text{out}) = H(T_\text{in}, P_\text{in})$. On a real gas this
produces the Joule-Thomson temperature change that an ideal-gas or ideal-K valve
misses. Because no work is extracted, the same pressure drop cools **less** than
a turboexpander.

```python
valve = JTValve(JTValveParams(P_out=20e5), thermo)
outlet, info = valve(feed)   # info: {"T_out", "H"}
```

### ComponentSeparator

A black-box separator surrogate: each component is split to the product stream
by a fixed recovery, the complement going to the residue. Both products inherit
the inlet T and P, and the reported duty `Q` is the enthalpy imbalance needed to
hold both at the inlet temperature. Useful as a column stand-in when only the
recovery specification is known.

```python
rec = {"propane": 0.95, "n_butane": 0.99}   # heavies to product
sep = ComponentSeparator(
    ComponentSeparatorParams(recovery_to_product=rec, default_recovery=0.0),
    thermo,
)
residue, product, info = sep(feed)   # info: {"Q", "H_in", "H_out"}
```

---

## Summary Tables

### Reactor Comparison

| Reactor | Mixing | Residence Time | Best For |
|---------|--------|----------------|----------|
| CSTR | Perfect | Distribution | Liquid-phase, uniform T |
| PFR | None (axial) | Uniform | Gas-phase, high X |
| GasPFR | None | Variable | Gas with ΔP, mole change |
| Fed-Batch | Perfect | Variable | Selectivity control |

### Heat Exchanger Comparison

| Type | Arrangement | ΔT Driving Force | Max T Approach | Typical Applications |
|------|-------------|------------------|----------------|---------------------|
| Counter-current | Opposite flow | Maximum | T_c,out → T_h,in | Max efficiency, heat recovery |
| Cross-flow (unmixed) | Perpendicular | Good | Intermediate | Car radiators, HVAC, finned-tube HX |
| Cross-flow (mixed) | Perpendicular | Better | Intermediate | Compact HX, special geometries |
| Co-current | Parallel flow | Moderate | T_c,out ≤ T_h,out | Simple applications, temperature control |

**Note**: For the same UA and inlet conditions, effectiveness ranking is:
Counter-current > Cross-flow (mixed) > Cross-flow (unmixed) > Co-current

### Separation Method Selection

| Method | Basis | Typical Application |
|--------|-------|---------------------|
| Flash | VLE | Light/heavy split |
| Distillation | Boiling point | High purity, sharp split |
| LLE | Solubility | Heat-sensitive, azeotropes |
