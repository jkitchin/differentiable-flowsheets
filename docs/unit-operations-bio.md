# Bio-Manufacturing Unit Operations

This document provides comprehensive documentation for all bio-manufacturing unit operations available in the `difflow_bio` module.

## Table of Contents

1. [Bioreactors](#bioreactors)
   - [ContinuousBioreactor (Chemostat)](#continuousbioreactor-chemostat)
   - [FedBatchBioreactor](#fedbatchbioreactor)
   - [Kinetic Models](#kinetic-models)
2. [Centrifugation](#centrifugation)
   - [Centrifuge](#centrifuge)
   - [DiscStackCentrifuge](#discstackcentrifuge)
3. [Membrane Filtration](#membrane-filtration)
   - [Ultrafiltration](#ultrafiltration)
   - [Diafiltration](#diafiltration)
   - [TFF (Tangential Flow Filtration)](#tff-tangential-flow-filtration)
4. [Chromatography](#chromatography)
   - [ProteinAChromatography](#proteinachromatography)
   - [IonExchangeChromatography](#ionexchangechromatography)
   - [SizeExclusionChromatography](#sizeexclusionchromatography)
   - [Adsorption Isotherms](#adsorption-isotherms)

---

## Bioreactors

### ContinuousBioreactor (Chemostat)

**Location**: `difflow_bio/units/bioreactors.py`

**Class**: `ContinuousBioreactor`

**Description**: Models a continuous stirred-tank bioreactor (chemostat) at steady state. The reactor maintains constant volume with continuous feed and withdrawal, allowing for steady-state cell cultivation.

#### Process Role

Continuous bioreactors are used for:
- Large-scale production of microbial products
- Maintaining cells in exponential growth phase
- Steady-state operation for consistent product quality
- High-volume, low-value products (ethanol, organic acids)

#### Parameters

```python
@dataclass
class BioreactorParams:
    volume: float           # Reactor volume (m³)
    mu_max: float          # Maximum specific growth rate (1/h)
    K_s: float             # Monod substrate saturation constant (g/L)
    Y_xs: float            # Biomass yield on substrate (g cell/g substrate)
    Y_px: float            # Product yield on biomass (g product/g cell)
    k_d: float = 0.0       # Cell death rate (1/h)
    m_s: float = 0.0       # Maintenance coefficient (g substrate/g cell/h)
    alpha: float = 0.0     # Growth-associated product formation (g/g)
    beta: float = 0.0      # Non-growth-associated product formation (g/g/h)
    K_i: float = None      # Substrate inhibition constant (g/L)
    K_p: float = None      # Product inhibition constant (g/L)
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream with substrate concentration |
| `D` | float | 1/h | Dilution rate (F/V) |
| `S_f` | float | g/L | Feed substrate concentration |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `outlet` | Stream | - | Outlet stream |
| `info['X']` | float | g/L | Cell concentration |
| `info['S']` | float | g/L | Substrate concentration |
| `info['P']` | float | g/L | Product concentration |
| `info['mu']` | float | 1/h | Specific growth rate |
| `info['productivity']` | float | g/L/h | Volumetric productivity |

#### Governing Equations

**Cell Balance** (steady-state):

$$\frac{dX}{dt} = 0 = (\mu - D - k_d) X$$

At steady state: $\mu = D + k_d$ (washout condition: $D < \mu_{max}$)

**Substrate Balance**:

$$\frac{dS}{dt} = 0 = D(S_f - S) - \frac{\mu X}{Y_{X/S}} - m_s X$$

Solving for steady-state substrate:

$$S = \frac{K_s D}{(\mu_{max} - D)}$$

**Cell Concentration**:

$$X = Y_{X/S}(S_f - S) - m_s \frac{X}{D}$$

Simplified (negligible maintenance):

$$X = Y_{X/S}(S_f - S)$$

**Product Formation** (Luedeking-Piret):

$$\frac{dP}{dt} = (\alpha \mu + \beta) X - D \cdot P$$

At steady state:

$$P = \frac{(\alpha \mu + \beta) X}{D}$$

Where:
- $\alpha$: Growth-associated product formation coefficient
- $\beta$: Non-growth-associated product formation rate

**Productivity**:

$$P_r = D \cdot X$$ (for biomass)

$$P_r = D \cdot P$$ (for product)

#### Critical Dilution Rate

Washout occurs when $D > \mu_{max}$. The critical dilution rate:

$$D_{crit} = \mu_{max} \frac{S_f}{K_s + S_f}$$

**Optimal Dilution Rate** (maximum productivity):

$$D_{opt} = \mu_{max} \left(1 - \sqrt{\frac{K_s}{K_s + S_f}}\right)$$

#### Example Usage

```python
from difflow_bio.units.bioreactors import ContinuousBioreactor, BioreactorParams

params = BioreactorParams(
    volume=10.0,        # m³
    mu_max=0.4,         # 1/h
    K_s=0.5,            # g/L
    Y_xs=0.5,           # g/g
    Y_px=0.1,           # g/g
    alpha=0.05,         # growth-associated
    beta=0.01           # non-growth-associated
)

bioreactor = ContinuousBioreactor(params)
feed = make_stream({'substrate': 50.0, 'cells': 0.0, 'product': 0.0}, T=310.0, P=101325.0)

outlet, info = bioreactor(feed, D=0.2, S_f=50.0)
print(f"Cell concentration: {info['X']:.2f} g/L")
print(f"Productivity: {info['productivity']:.3f} g/L/h")
```

---

### FedBatchBioreactor

**Location**: `difflow_bio/units/bioreactors.py`

**Class**: `FedBatchBioreactor`

**Description**: Models a fed-batch bioreactor where substrate is added during the batch to control growth rate and avoid substrate inhibition.

#### Process Role

Fed-batch bioreactors are used for:
- High-value products (therapeutic proteins, antibodies)
- Avoiding substrate/product inhibition
- Maximizing final product titer
- Most industrial mAb production

#### Parameters

```python
@dataclass
class FedBatchParams:
    V0: float              # Initial volume (L)
    Y_xs: float            # Yield coefficient (g cells / g substrate)
    kinetic_fn: Callable   # Growth kinetics function
    kinetic_params: dict   # Parameters for kinetic function
    k_d: float = 0.0       # Death rate constant (1/h)
    m_s: float = 0.0       # Maintenance coefficient (g/g/h)
    alpha: float = 0.0     # Growth-associated product formation (g/g)
    beta: float = 0.0      # Non-growth-associated product formation (g/g/h)
    species_order: list = ["cells", "substrate", "product"]
```

**Integration**: Uses diffrax (adaptive Tsit5 by default) or RK4 fallback

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream composition |
| `feed_rate` | Callable | L/h | Feed rate as function of time |
| `X_0` | float | g/L | Initial cell concentration |
| `S_0` | float | g/L | Initial substrate concentration |

#### Governing Equations

**Volume Change**:

$$\frac{dV}{dt} = F(t)$$

**Cell Balance**:

$$\frac{d(VX)}{dt} = V(\mu - k_d)X$$

Or: $\frac{dX}{dt} = (\mu - k_d - D)X$ where $D = F/V$

**Substrate Balance**:

$$\frac{d(VS)}{dt} = F \cdot S_f - V\left(\frac{\mu X}{Y_{X/S}} + m_s X\right)$$

**Product Balance**:

$$\frac{d(VP)}{dt} = V(\alpha \mu + \beta)X$$

**Common Feed Strategies**:

1. **Constant Feed**: $F(t) = F_0$
2. **Exponential Feed**: $F(t) = F_0 e^{\mu_{set} t}$ to maintain constant $\mu$
3. **Feedback Control**: Adjust F based on measured substrate or DO

#### Example Usage

```python
from difflow_bio.units.bioreactors import FedBatchBioreactor, FedBatchParams, monod_kinetics
import jax.numpy as jnp

params = FedBatchParams(
    V0=5.0,            # Initial volume (L)
    Y_xs=0.5,          # Yield coefficient
    kinetic_fn=monod_kinetics,
    kinetic_params={'mu_max': 0.4, 'K_s': 0.5},
    k_d=0.01,          # Death rate
    alpha=0.1,         # Growth-associated product formation
    beta=0.02          # Non-growth-associated product formation
)

bioreactor = FedBatchBioreactor(params)

# Exponential feed to maintain mu = 0.1/h
def exponential_feed(t):
    return 0.1 * jnp.exp(0.1 * t)

# Run simulation
outlet, info = bioreactor(
    t_span=(0.0, 72.0),  # hours
    X0=0.5,              # Initial cell concentration (g/L)
    S0=20.0,             # Initial substrate (g/L)
    feed_profile=exponential_feed,
    S_f=200.0            # Feed substrate concentration
)
print(f"Final cell concentration: {info['X_final']:.2f} g/L")
print(f"Final product: {info['P_final']:.2f} g/L")
```

---

### Kinetic Models

**Location**: `difflow_bio/units/bioreactors.py`

#### Monod Kinetics

$$\mu = \mu_{max} \frac{S}{K_s + S}$$

```python
from difflow_bio.units.bioreactors import monod_kinetics

mu = monod_kinetics(S=5.0, mu_max=0.4, K_s=0.5)
```

#### Substrate Inhibition (Andrews/Haldane)

$$\mu = \mu_{max} \frac{S}{K_s + S + S^2/K_i}$$

```python
from difflow_bio.units.bioreactors import substrate_inhibition_kinetics

mu = substrate_inhibition_kinetics(S=5.0, mu_max=0.4, K_s=0.5, K_i=50.0)
```

#### Product Inhibition

$$\mu = \mu_{max} \frac{S}{K_s + S} \left(1 - \frac{P}{P_{max}}\right)^n$$

```python
from difflow_bio.units.bioreactors import product_inhibition_kinetics

mu = product_inhibition_kinetics(S=5.0, P=10.0, mu_max=0.4, K_s=0.5, P_max=100.0, n=1)
```

#### Contois Kinetics (High Cell Density)

$$\mu = \mu_{max} \frac{S}{K_{sX} X + S}$$

```python
from difflow_bio.units.bioreactors import contois_kinetics

mu = contois_kinetics(S=5.0, X=50.0, mu_max=0.4, K_sx=0.1)
```

#### Utility Functions

```python
from difflow_bio.units.bioreactors import (
    dilution_rate,
    residence_time,
    optimal_dilution_rate,
    washout_dilution_rate,
    productivity
)

D = dilution_rate(F=100.0, V=500.0)  # 0.2 1/h
tau = residence_time(V=500.0, F=100.0)  # 5 h
D_opt = optimal_dilution_rate(mu_max=0.4, K_s=0.5, S_f=50.0)
```

---

## Centrifugation

### Centrifuge

**Location**: `difflow_bio/units/centrifuge.py`

**Class**: `Centrifuge`

**Description**: Models centrifugal separation of cells/particles from liquid based on density difference. Uses Sigma factor theory for scale-up.

#### Process Role

Centrifugation is used for:
- Cell harvest (primary recovery)
- Cell debris removal after homogenization
- Precipitate collection
- Clarification before chromatography

#### Parameters

```python
@dataclass
class CentrifugeParams:
    sigma: float           # Sigma factor (equivalent settling area, m²)
    efficiency: float      # Separation efficiency (0-1)
    particle_diameter: float  # Mean particle diameter (m)
    particle_density: float   # Particle density (kg/m³)
    fluid_density: float      # Fluid density (kg/m³)
    fluid_viscosity: float    # Fluid dynamic viscosity (Pa·s)
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed with cells and broth |
| `Q` | float | m³/s | Volumetric throughput |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `concentrate` | Stream | - | Concentrated cells/solids |
| `supernatant` | Stream | - | Clarified liquid |
| `info['recovery']` | float | - | Cell/particle recovery |
| `info['clarity']` | float | - | Supernatant clarity |

#### Governing Equations

**Stokes Settling Velocity** (single particle, laminar flow):

$$v_s = \frac{d_p^2 (\rho_p - \rho_f) g}{18 \mu}$$

Where:
- $d_p$: Particle diameter (m)
- $\rho_p$: Particle density (kg/m³)
- $\rho_f$: Fluid density (kg/m³)
- $\mu$: Fluid viscosity (Pa·s)
- $g$: Gravitational acceleration (9.81 m/s²)

**Centrifugal Enhancement**:

In a centrifuge, $g$ is replaced by centrifugal acceleration:

$$a_c = \omega^2 r$$

**Sigma Factor** (equivalent settling area):

The Sigma factor allows scale-up between different centrifuge types:

$$\Sigma = \frac{Q}{2 v_s}$$

For a given separation: $Q_1/\Sigma_1 = Q_2/\Sigma_2$

**Critical Particle Diameter** (100% capture):

$$d_{crit} = \sqrt{\frac{18 \mu Q}{(\rho_p - \rho_f) \omega^2 \Sigma}}$$

**Separation Efficiency** (particle size distribution):

$$E = 1 - \exp\left(-\frac{d^2}{d_{crit}^2}\right)$$

**G-Force**:

$$G = \frac{\omega^2 r}{g} = \frac{(2\pi N)^2 r}{g}$$

Where N is rotational speed (rev/s).

#### Example Usage

```python
from difflow_bio.units.centrifuge import Centrifuge, CentrifugeParams

params = CentrifugeParams(
    sigma=5000.0,           # m²
    efficiency=0.95,
    particle_diameter=5e-6,  # 5 μm cells
    particle_density=1050.0, # kg/m³
    fluid_density=1000.0,    # kg/m³
    fluid_viscosity=0.001    # Pa·s
)

centrifuge = Centrifuge(params)
feed = make_stream({'cells': 50.0, 'broth': 950.0}, T=298.0, P=101325.0)

concentrate, supernatant, info = centrifuge(feed, Q=0.001)  # 1 L/s = 3.6 m³/h
print(f"Cell recovery: {info['recovery']:.2%}")
```

---

### DiscStackCentrifuge

**Location**: `difflow_bio/units/centrifuge.py`

**Class**: `DiscStackCentrifuge`

**Description**: Specialized model for disc-stack centrifuges, the most common type in bioprocessing for cell harvest.

#### Parameters

```python
@dataclass
class DiscStackParams:
    n_discs: int           # Number of discs
    r_inner: float         # Inner disc radius (m)
    r_outer: float         # Outer disc radius (m)
    cone_angle: float      # Disc cone angle (radians)
    rpm: float             # Rotational speed (rev/min)
```

#### Sigma Factor Calculation

$$\Sigma = \frac{2\pi n \omega^2 (r_o^3 - r_i^3)}{3g \tan\theta}$$

Where:
- $n$: Number of discs
- $\omega$: Angular velocity (rad/s)
- $r_o, r_i$: Outer and inner disc radii
- $\theta$: Half cone angle

#### Example Usage

```python
from difflow_bio.units.centrifuge import DiscStackCentrifuge, DiscStackParams

params = DiscStackParams(
    n_discs=100,
    r_inner=0.05,    # 5 cm
    r_outer=0.15,    # 15 cm
    cone_angle=0.785, # 45 degrees
    rpm=7000
)

centrifuge = DiscStackCentrifuge(params)
# Sigma is calculated internally based on geometry
```

#### Utility Functions

```python
from difflow_bio.units.centrifuge import (
    stokes_velocity,
    critical_particle_diameter,
    disc_stack_sigma,
    tubular_bowl_sigma,
    centrifuge_scale_up,
    g_force
)

# Calculate Stokes velocity
v_s = stokes_velocity(d_p=5e-6, rho_p=1050, rho_f=1000, mu=0.001)

# Calculate Sigma for disc stack
sigma = disc_stack_sigma(n_discs=100, r_i=0.05, r_o=0.15, theta=0.785, omega=733.0)

# Scale-up calculation
Q2 = centrifuge_scale_up(Q1=1.0, Sigma1=1000, Sigma2=10000)

# G-force
G = g_force(rpm=7000, r=0.15)  # ~7900 G
```

---

## Membrane Filtration

### Ultrafiltration

**Location**: `difflow_bio/units/filtration.py`

**Class**: `Ultrafiltration`

**Description**: Pressure-driven membrane separation that concentrates proteins based on molecular weight cutoff (MWCO).

#### Process Role

Ultrafiltration is used for:
- Protein concentration
- Buffer exchange (combined with diafiltration)
- Virus removal (as secondary barrier)
- Harvest clarification (with larger MWCO)

#### Parameters

```python
@dataclass
class UltrafiltrationParams:
    MWCO: float            # Molecular weight cutoff (Da)
    membrane_area: float   # Membrane area (m²)
    TMP: float            # Transmembrane pressure (Pa)
    flux_coefficient: float  # Clean water flux coefficient
    concentration_factor: float  # Target concentration factor
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream |
| `CF` | float | - | Concentration factor (optional override) |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `retentate` | Stream | - | Concentrated product |
| `permeate` | Stream | - | Permeate (removed species) |
| `info['flux']` | float | LMH | Permeate flux (L/m²/h) |
| `info['rejection']` | Array | - | Rejection coefficient per species |

#### Governing Equations

**Permeate Flux** (resistance model):

$$J = \frac{TMP}{\mu (R_m + R_c + R_g)}$$

Where:
- $J$: Permeate flux (L/m²/h or LMH)
- $TMP$: Transmembrane pressure (Pa)
- $\mu$: Permeate viscosity (Pa·s)
- $R_m$: Membrane resistance (1/m)
- $R_c$: Concentration polarization resistance
- $R_g$: Gel layer resistance

**Rejection Coefficient**:

$$R_i = 1 - \frac{C_{i,permeate}}{C_{i,retentate}}$$

For a sharp MWCO:
- Species with $MW > MWCO$: $R \approx 1$ (fully retained)
- Species with $MW < MWCO$: $R \approx 0$ (freely permeating)

**Concentration Factor**:

$$CF = \frac{V_{feed}}{V_{retentate}} = \frac{C_{retentate}}{C_{feed}}$$ (for fully retained species)

**Mass Balance**:

$$V_f C_{f,i} = V_r C_{r,i} + V_p C_{p,i}$$

$$C_{r,i} = \frac{C_{f,i}}{1 - (1-R_i)(1 - 1/CF)}$$

**Concentration Polarization**:

$$\frac{C_w - C_p}{C_b - C_p} = \exp\left(\frac{J}{k}\right)$$

Where:
- $C_w$: Wall concentration
- $C_b$: Bulk concentration
- $k$: Mass transfer coefficient

**Gel Polarization Model** (flux-limited):

$$J = k \ln\left(\frac{C_g}{C_b}\right)$$

Where $C_g$ is the gel concentration (limiting).

#### Example Usage

```python
from difflow_bio.units.filtration import Ultrafiltration, UltrafiltrationParams

params = UltrafiltrationParams(
    MWCO=30000,          # 30 kDa
    membrane_area=5.0,    # m²
    TMP=200000,          # 2 bar
    flux_coefficient=100, # LMH/bar
    concentration_factor=10
)

uf = Ultrafiltration(params)
feed = make_stream({'mAb': 2.0, 'HCP': 0.5, 'buffer': 997.5}, T=298.0, P=101325.0)

retentate, permeate, info = uf(feed)
print(f"mAb concentration: {retentate['mAb']:.1f} g/L")
print(f"Flux: {info['flux']:.1f} LMH")
```

---

### Diafiltration

**Location**: `difflow_bio/units/filtration.py`

**Class**: `Diafiltration`

**Description**: Ultrafiltration with continuous buffer addition to exchange buffer or remove small molecules while retaining proteins.

#### Process Role

Diafiltration is used for:
- Buffer exchange
- Salt removal (desalting)
- Small molecule impurity removal
- Formulation preparation

#### Parameters

```python
@dataclass
class DiafiltrationParams(UltrafiltrationParams):
    diavolumes: float      # Number of diavolumes
    mode: str = 'constant_volume'  # 'constant_volume' or 'discontinuous'
```

#### Governing Equations

**Diavolume Definition**:

$$N_{DV} = \frac{V_{buffer\,added}}{V_{retentate}}$$

**Impurity Removal** (constant volume diafiltration):

$$\frac{C}{C_0} = \exp(-N_{DV}(1-R))$$

For freely permeating species ($R = 0$):

$$\frac{C}{C_0} = \exp(-N_{DV})$$

**Required Diavolumes** for target removal:

$$N_{DV} = -\frac{\ln(C/C_0)}{1-R}$$

| Diavolumes | Removal (R=0) |
|------------|---------------|
| 1 | 63.2% |
| 2 | 86.5% |
| 3 | 95.0% |
| 5 | 99.3% |
| 7 | 99.9% |

#### Example Usage

```python
from difflow_bio.units.filtration import Diafiltration, DiafiltrationParams

params = DiafiltrationParams(
    MWCO=30000,
    membrane_area=5.0,
    TMP=150000,
    diavolumes=5,
    mode='constant_volume'
)

df = Diafiltration(params)
feed = make_stream({'mAb': 10.0, 'salt': 150.0, 'buffer': 840.0}, T=298.0, P=101325.0)
new_buffer = make_stream({'salt': 0.0, 'new_buffer': 1000.0}, T=298.0, P=101325.0)

retentate, permeate, info = df(feed, buffer=new_buffer)
print(f"Salt removal: {1 - retentate['salt']/feed['salt']:.1%}")
```

---

### TFF (Tangential Flow Filtration)

**Location**: `difflow_bio/units/filtration.py`

**Class**: `TFF`

**Description**: Combines ultrafiltration and diafiltration in a single system with tangential flow to minimize fouling.

#### Key Features

- Recirculation loop reduces concentration polarization
- Feed flows parallel to membrane surface
- Multiple passes possible for high concentration

#### Process Modes

1. **Concentration**: UF mode to reduce volume
2. **Diafiltration**: Buffer exchange at constant volume
3. **UF/DF Sequence**: Concentrate → Diafiltrate → Final concentration

---

## Chromatography

### ProteinAChromatography

**Location**: `difflow_bio/units/chromatography.py`

**Class**: `ProteinAChromatography`

**Description**: Affinity chromatography using Protein A ligand for mAb capture. The primary capture step in most mAb processes.

#### Process Role

Protein A chromatography provides:
- High selectivity for Fc-containing antibodies
- >95% purity in single step
- Significant HCP and DNA clearance
- Recovery >90%

#### Parameters

```python
@dataclass
class ProteinAParams:
    column_volume: float   # Column volume (L)
    q_max: float          # Maximum binding capacity (g/L resin)
    K_d: float            # Dissociation constant (M)
    flow_rate: float      # Operating flow rate (CV/h)
    residence_time: float # Column residence time (min)
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `load` | Stream | - | Feed stream with mAb |
| `load_volume` | float | CV | Load volume in column volumes |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `eluate` | Stream | - | Purified mAb |
| `info['recovery']` | float | - | Step recovery |
| `info['purity']` | float | - | Product purity |
| `info['HCP_LRV']` | float | - | HCP log reduction value |
| `info['DNA_LRV']` | float | - | DNA log reduction value |
| `info['DBC']` | float | g/L | Dynamic binding capacity |

#### Process Steps

1. **Equilibration**: Condition column with loading buffer
2. **Load**: Apply feed, mAb binds to resin
3. **Wash**: Remove unbound impurities
4. **Elution**: Release mAb with low pH buffer
5. **Regeneration**: Clean and re-equilibrate column

#### Governing Equations

**Langmuir Isotherm** (equilibrium binding):

$$q = \frac{q_{max} C}{K_d + C}$$

Where:
- $q$: Bound concentration (g/L resin)
- $q_{max}$: Maximum binding capacity
- $C$: Solution concentration
- $K_d$: Dissociation constant

**Dynamic Binding Capacity** (at 10% breakthrough):

$$DBC_{10\%} = q_{max} \cdot f(RT, C_{load}, K_d)$$

Typically 80-95% of equilibrium capacity.

**Column Capacity Utilization**:

$$\text{Load} = \frac{m_{mAb,loaded}}{V_{column} \cdot DBC}$$

Typical loading: 70-90% of DBC.

**Log Reduction Value**:

$$LRV = \log_{10}\left(\frac{C_{in}}{C_{out}}\right)$$

#### Example Usage

```python
from difflow_bio.units.chromatography import ProteinAChromatography, ProteinAParams

params = ProteinAParams(
    column_volume=10.0,    # L
    q_max=35.0,            # g/L (typical for MabSelect)
    K_d=1e-8,              # M (very tight binding)
    flow_rate=2.0,         # CV/h
    residence_time=6.0     # min
)

protein_a = ProteinAChromatography(params)
load = make_stream({'mAb': 5.0, 'HCP': 5.0, 'DNA': 0.1}, T=298.0, P=101325.0)

eluate, info = protein_a(load, load_volume=25)  # 25 CV = 250 L
print(f"Recovery: {info['recovery']:.2%}")
print(f"HCP clearance: {info['HCP_LRV']:.1f} LRV")
```

---

### IonExchangeChromatography

**Location**: `difflow_bio/units/chromatography.py`

**Class**: `IonExchangeChromatography`

**Description**: Separation based on electrostatic interactions between charged proteins and charged resin.

#### Types

- **Cation Exchange (CEX)**: Negatively charged resin, binds positively charged proteins
- **Anion Exchange (AEX)**: Positively charged resin, binds negatively charged proteins

#### Process Role

Ion exchange is used for:
- Aggregate removal (CEX)
- Charge variant separation (CEX)
- DNA/HCP removal (AEX in flow-through mode)
- Viral clearance (AEX)

#### Parameters

```python
@dataclass
class IEXParams:
    column_volume: float   # Column volume (L)
    q_max: float          # Binding capacity (g/L)
    type: str             # 'CEX' or 'AEX'
    mode: str             # 'bind_elute' or 'flow_through'
    selectivity: Array    # Selectivity factors for species
```

#### Operating Modes

**Bind-Elute Mode**:
1. Product binds to resin
2. Impurities wash through or elute at different salt concentrations
3. Product eluted with salt gradient

**Flow-Through Mode**:
1. Product passes through column
2. Impurities bind to resin
3. Common for AEX after Protein A

#### Governing Equations

**Selectivity**:

$$\alpha_{ij} = \frac{K_i}{K_j}$$

**Resolution**:

$$R_s = \frac{t_{R,2} - t_{R,1}}{0.5(w_1 + w_2)} = \frac{\sqrt{N}}{4} \cdot \frac{\alpha - 1}{\alpha} \cdot \frac{k'}{1 + k'}$$

Where:
- $N$: Number of theoretical plates
- $\alpha$: Selectivity
- $k'$: Capacity factor

---

### SizeExclusionChromatography

**Location**: `difflow_bio/units/chromatography.py`

**Class**: `SizeExclusionChromatography`

**Description**: Separation based on molecular size. Large molecules elute first (excluded from pores), small molecules elute last.

#### Process Role

SEC is used for:
- Aggregate/fragment removal (polishing)
- Buffer exchange
- Molecular weight determination (analytical)
- Final formulation preparation

#### Parameters

```python
@dataclass
class SECParams:
    column_volume: float   # Column volume (L)
    void_fraction: float   # Interparticle void (V_0/V_c)
    total_porosity: float  # Total porosity (V_t/V_c)
    exclusion_limit: float # MW exclusion limit (Da)
    permeation_limit: float # MW permeation limit (Da)
```

#### Governing Equations

**Distribution Coefficient**:

$$K_d = \frac{V_e - V_0}{V_t - V_0}$$

Where:
- $V_e$: Elution volume
- $V_0$: Void volume (excluded volume)
- $V_t$: Total accessible volume

**Calibration** (MW vs elution volume):

$$\log(MW) = a - b \cdot K_d$$

Or: $K_d = a' - b' \cdot \log(MW)$

**Resolution Limits**:
- $K_d = 0$: Totally excluded (MW > exclusion limit)
- $K_d = 1$: Fully permeating (MW < permeation limit)

---

### Adsorption Isotherms

**Location**: `difflow_bio/units/chromatography.py`

#### Langmuir Isotherm

$$q = \frac{q_{max} C}{K_d + C}$$

```python
from difflow_bio.units.chromatography import langmuir_isotherm

q = langmuir_isotherm(C=5.0, q_max=35.0, K_d=0.5)
```

#### Linear Isotherm (dilute systems)

$$q = K \cdot C$$

```python
from difflow_bio.units.chromatography import linear_isotherm

q = linear_isotherm(C=0.1, K=100.0)
```

#### Langmuir-Freundlich (heterogeneous sites)

$$q = \frac{q_{max} (C/K_d)^n}{1 + (C/K_d)^n}$$

```python
from difflow_bio.units.chromatography import langmuir_freundlich_isotherm

q = langmuir_freundlich_isotherm(C=5.0, q_max=35.0, K_d=0.5, n=0.8)
```

#### Chromatography Utility Functions

```python
from difflow_bio.units.chromatography import (
    dynamic_binding_capacity,
    column_productivity,
    resolution,
    plate_count,
    hetp
)

# Dynamic binding capacity at 10% breakthrough
DBC = dynamic_binding_capacity(q_max=35.0, C=5.0, K_d=0.5, RT=6.0)

# Column productivity
P = column_productivity(DBC=30.0, cycle_time=4.0)  # g/L/h

# Resolution between peaks
Rs = resolution(t_R1=10.0, t_R2=12.0, w1=0.5, w2=0.6)

# Theoretical plates
N = plate_count(t_R=10.0, w=0.5)

# Height equivalent to theoretical plate
H = hetp(L=20.0, N=10000)  # cm
```

---

## Summary: Typical mAb Downstream Process

```
Harvest (Bioreactor)
        │
        ▼
┌───────────────────┐
│   Centrifugation  │  Cell removal
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Depth Filtration  │  Clarification
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Protein A      │  Capture (>95% purity)
│  Chromatography   │  HCP: 3-4 LRV, DNA: 4-5 LRV
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Low pH Viral     │  Viral inactivation
│   Inactivation    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Cation Exchange  │  Aggregate removal
│  (Bind-Elute)     │  Charge variant control
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Anion Exchange   │  DNA/HCP polishing
│  (Flow-Through)   │  Viral clearance
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Virus Filtration │  Final viral clearance
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    UF/DF          │  Concentration
│                   │  Formulation
└───────────────────┘
        │
        ▼
      Drug Substance
```

### Typical Performance Targets

| Step | Yield | Purity | HCP | DNA |
|------|-------|--------|-----|-----|
| Protein A | >95% | >95% | 100-500 ppm | <10 ppb |
| CEX | >90% | >98% | <50 ppm | <10 ppb |
| AEX | >95% | >99% | <10 ppm | <1 ppb |
| Final | >70% overall | >99.5% | <10 ppm | <10 ppb |
