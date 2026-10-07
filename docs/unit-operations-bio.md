# Bio-Manufacturing Unit Operations

![difflow_bio](../images/plugins/difflow-bio-logo.svg)

This document provides comprehensive documentation for all bio-manufacturing unit operations available in the `difflow_bio` module.

---

(bioreactors)=
## Bioreactors

(continuousbioreactor-chemostat)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class BioreactorParams:
    V: float               # Reactor volume (L)
    Y_xs: float            # Biomass yield on substrate (g cell/g substrate)
    kinetic_fn: Callable   # Growth kinetics mu(S, params) or mu(S, X, params)
    kinetic_params: dict   # Passed to kinetic_fn, e.g. {"mu_max": 0.4, "K_s": 0.5}
    k_d: float = 0.0       # Cell death rate (1/h)
    m_s: float = 0.0       # Maintenance coefficient (g substrate/g cell/h)
    alpha: float = 0.0     # Growth-associated product formation (g/g)
    beta: float = 0.0      # Non-growth-associated product formation (g/g/h)
    species_order: list = ["cells", "substrate", "product"]
```

The growth law is the `kinetic_fn`: `monod_kinetics` reads `mu_max` and
`K_s` from `kinetic_params`, and the inhibition models
(`substrate_inhibition_kinetics`, ...) read their own constants from the same dict.

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream; its `substrate` entry is the feed substrate concentration $S_f$ |
| `D` | float | 1/h | Dilution rate (F/V) |
| `F` | float | L/h | Volumetric feed rate, used when `D` is None ($D = F/V$) |

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
from difflow.streams import make_stream
from difflow_bio.units.bioreactors import (
    BioreactorParams, ContinuousBioreactor, monod_kinetics)

params = BioreactorParams(
    V=10.0,                                       # L
    Y_xs=0.5,                                     # g/g
    kinetic_fn=monod_kinetics,
    kinetic_params={"mu_max": 0.4, "K_s": 0.5},   # 1/h, g/L
    alpha=0.05,         # growth-associated
    beta=0.01,          # non-growth-associated
    species_order=["cells", "substrate", "product"],
)

bioreactor = ContinuousBioreactor(params)
# S_f = 50 g/L is the substrate entry of the feed, not a call argument
feed = make_stream({'substrate': 50.0, 'cells': 0.0, 'product': 0.0}, T=310.0, P=101325.0)

outlet, info = bioreactor(feed, D=0.2)
print(f"Cell concentration: {info['X']:.2f} g/L")
print(f"Productivity: {info['productivity']:.3f} g/L/h")
```

---

(fedbatchbioreactor)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class FedBatchParams:
    V0: float              # Initial volume (L)
    Y_xs: float            # Yield coefficient (g cells / g substrate)
    kinetic_fn: Callable   # Growth kinetics function
    kinetic_params: dict   # Parameters for kinetic function
    k_d: float = 0.0       # Death rate constant (1/h)
    m_s: float = 0.0       # Maintenance coefficient (g/g/h)
    K_m: float = 0.05      # Half-saturation of maintenance uptake (g/L)
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

$$\frac{d(VX)}{dt} = V(\mu - k_d)X - V m_s Y_{X/S} (1 - \phi) X$$

**Substrate Balance**:

$$\frac{d(VS)}{dt} = F \cdot S_f - V\left(\frac{\mu X}{Y_{X/S}} + m_s \phi X\right), \qquad \phi = \frac{S}{K_m + S}$$

**Maintenance under starvation**: maintenance draws on substrate while
there is any ($\phi \approx 1$ for $S \gg K_m$) and on biomass when it
runs out, at the endogenous decay rate $b = m_s Y_{X/S}$ (Pirt, 1965).
Without the $\phi$ factor, uptake would continue at $m_s X$ after the
substrate is gone and drive $S$ negative while product kept
accumulating. `info["S_min"]` reports the lowest substrate concentration
reached, so a starved run is visible.

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
    X0=0.5,              # Initial cell concentration (g/L)
    S0=20.0,             # Initial substrate (g/L)
    P0=0.0,              # Initial product (g/L)
    t_final=72.0,        # hours
    feed_rate_fn=exponential_feed,
    S_feed=200.0,        # Feed substrate concentration (g/L)
)
print(f"Final cell concentration: {info['X_final']:.2f} g/L")
print(f"Final product: {info['P_final']:.2f} g/L")
```

---

(kinetic-models)=
### Kinetic Models

**Location**: `difflow_bio/units/bioreactors.py`

#### Monod Kinetics

$$\mu = \mu_{max} \frac{S}{K_s + S}$$

```python
from difflow_bio.units.bioreactors import monod_kinetics

mu = monod_kinetics(5.0, {"mu_max": 0.4, "K_s": 0.5})
```

#### Substrate Inhibition (Andrews/Haldane)

$$\mu = \mu_{max} \frac{S}{K_s + S + S^2/K_i}$$

```python
from difflow_bio.units.bioreactors import substrate_inhibition_kinetics

mu = substrate_inhibition_kinetics(5.0, {"mu_max": 0.4, "K_s": 0.5, "K_i": 50.0})
```

#### Product Inhibition

$$\mu = \mu_{max} \frac{S}{K_s + S} \left(1 - \frac{P}{P_{max}}\right)^n$$

```python
from difflow_bio.units.bioreactors import product_inhibition_kinetics

mu = product_inhibition_kinetics(
    5.0, 10.0, {"mu_max": 0.4, "K_s": 0.5, "P_max": 100.0, "n": 1})
```

#### Contois Kinetics (High Cell Density)

$$\mu = \mu_{max} \frac{S}{K_{sX} X + S}$$

```python
from difflow_bio.units.bioreactors import contois_kinetics

mu = contois_kinetics(5.0, 50.0, {"mu_max": 0.4, "K_s": 0.1})   # K_s is K_sX here
```

#### Utility Functions

```python
from difflow_bio.units.bioreactors import (
    dilution_rate,
    residence_time,
    optimal_dilution_rate,
)

D = dilution_rate(F=100.0, V=500.0)  # 0.2 1/h
tau = residence_time(D)  # 5 h
D_opt = optimal_dilution_rate({"mu_max": 0.4, "K_s": 0.5, "S_f": 50.0})
```

The washout dilution rate is a method of the reactor,
`ContinuousBioreactor.washout_dilution_rate()`.

```python
```

---

(centrifugation)=
## Centrifugation

(centrifuge)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class CentrifugeParams:
    sigma: float                    # Sigma factor (equivalent settling area, m²)
    efficiency: float = 0.7         # Separation efficiency (0-1)
    species_order: list = None
    cell_species: str = "cells"
    lysis_threshold_g: float = None # RCF above which cells lyse (x g)
    lysis_coefficient: float = 0.0  # Lysed fraction per unit RCF above threshold
    lysate_species: str = "lysate"
```

The particle and fluid properties are call arguments, not parameters:
`d_particle` (m), `rho_particle` and `rho_fluid` (kg/m³), `viscosity`
(Pa·s), plus `concentrate_fraction` and `rcf`.

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
| `info['cell_recovery']` | float | - | Fraction of the cells sent to the concentrate |
| `info['separation_efficiency']` | float | - | Sigma-theory capture efficiency |
| `info['stokes_velocity']` | float | m/s | Settling velocity of the mean particle |
| `info['critical_diameter']` | float | m | Particle size captured at 50% efficiency at this throughput |
| `info['Q_critical']` | float | m³/s | Sigma-theory critical throughput, $2 v_s \Sigma$ |
| `info['concentration_factor']` | float | - | Cell concentration, concentrate over feed |
| `info['lysis_fraction']`, `info['lysate_released']` | float | - | Shear lysis (zero unless `lysis_threshold_g` is set) |

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
from difflow.streams import make_stream
from difflow_bio.units.centrifuge import Centrifuge, CentrifugeParams

params = CentrifugeParams(sigma=5000.0, efficiency=0.95)   # m², -

centrifuge = Centrifuge(params)
feed = make_stream({'cells': 50.0, 'broth': 950.0}, T=298.0, P=101325.0)

# particle and fluid properties are call arguments
concentrate, supernatant, info = centrifuge(
    feed, Q=0.001,              # 1 L/s = 3.6 m³/h
    d_particle=5e-6,            # 5 μm cells
    rho_particle=1050.0, rho_fluid=1000.0, viscosity=0.001,
)
print(f"Cell recovery: {info['cell_recovery']:.2%}")
```

---

(discstackcentrifuge)=
### DiscStackCentrifuge

**Location**: `difflow_bio/units/centrifuge.py`

**Class**: `DiscStackCentrifuge`

**Description**: Specialized model for disc-stack centrifuges, the most common type in bioprocessing for cell harvest.

#### Parameters

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
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
    half_angle=0.785, # 45 degrees
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
v_s = stokes_velocity(d=5e-6, rho_p=1050, rho_f=1000, mu=0.001)

# Calculate Sigma for disc stack
sigma = disc_stack_sigma(n_discs=100, r_inner=0.05, r_outer=0.15, half_angle=0.785, rpm=7000.0)

# Scale-up calculation
Q2 = centrifuge_scale_up(sigma_1=1000.0, Q_1=1.0, sigma_2=10000.0)

# G-force
G = g_force(rpm=7000, r=0.15)  # ~7900 G
```

---

(membrane-filtration)=
## Membrane Filtration

(ultrafiltration)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class UltrafiltrationParams:
    membrane_area: float           # Membrane area (m²); sets the process time only
    MWCO: float = 30.0             # Molecular weight cutoff (kDa), rejected at 90%
    rejection: dict = {}           # species -> R, overrides the MWCO-derived value
    molecular_weights: dict = {}   # species -> MW (kDa), overrides the tables
    Lp: float = 50.0               # Permeability (L/m²/h/bar)
    k_mass: float = 5e-6           # Mass transfer coefficient (m/s)
    sigma: float = 1000.0          # Osmotic pressure coefficient (Pa·m³/kg)
    fouling_coefficient: float = 0.0  # Flux decline per L permeate (1/L)
    species_order: list = None
```

Each species' rejection is its entry in `rejection` if there is one, and
otherwise follows the MWCO against its molecular weight, looked up in
`molecular_weights`, then `difflow_bio.database.BIO_SPECIES_MW_KDA`
(mAb 150 kDa, aggregates 300, fragments 50, HCP 50, DNA 330, and small
solutes; representative values), then the core species database. A species
with no molecular weight anywhere passes freely ($R = 0$), the right default
for buffer components. `membrane_area` does not change the split (sieving is
per unit area); given `feed_volume` (L) in the call, the unit reports
`info['process_time_h']`, the permeate volume over flux times area.

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream |
| `concentration_factor` | float | - | Volume concentration factor (required) |
| `TMP` | float | bar | Transmembrane pressure (default 1.0) |
| `mode` | str | - | `"batch"` (default) or `"continuous"` |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `retentate` | Stream | - | Concentrated product |
| `permeate` | Stream | - | Permeate (removed species) |
| `info['flux']` | float | LMH | Permeate flux (L/m²/h) |
| `info['recovery']` | dict | - | Fraction of each species kept in the retentate |
| `info['concentration_factor']`, `info['volume_reduction']`, `info['retentate_volume_fraction']` | float | - | Volume bookkeeping |
| `info['fouling_factor']` | float | - | Flux decline from fouling (1 when off) |
| `info['rejection']` | dict | - | Rejection used for each species: the `rejection` override, else the MWCO-derived value |

The call returns `((retentate, permeate), info)`: the two streams are
nested in a tuple of their own.

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

From the MWCO (`rejection_from_mw`), a logistic sieving curve in
$\ln MW$ with $R = 0.9$ at the cutoff, the nominal rating:

$$R = \left[1 + \tfrac{1}{9}\left(\frac{MW}{MWCO}\right)^{-3}\right]^{-1}$$

so a solute five times the cutoff (a 150 kDa mAb on a 30 kDa membrane) is
rejected at more than 99.9% and a small solute passes freely.

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
from difflow.streams import make_stream
from difflow_bio.units.filtration import Ultrafiltration, UltrafiltrationParams

params = UltrafiltrationParams(
    membrane_area=5.0,                       # m²
    MWCO=30.0,                               # kDa
    rejection={"mAb": 0.999, "HCP": 0.5},    # buffer passes freely
    Lp=100.0,                                # LMH/bar
)

uf = Ultrafiltration(params)
feed = make_stream({'mAb': 2.0, 'HCP': 0.5, 'buffer': 997.5}, T=298.0, P=101325.0)

(retentate, permeate), info = uf(feed, concentration_factor=10.0, TMP=2.0)
print(f"mAb in retentate: {retentate['F_mAb']:.3f}")
print(f"Flux: {info['flux']:.1f} LMH")
```

---

(diafiltration)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class DiafiltrationParams:
    membrane_area: float           # Membrane area (m²); sets the process time only
    MWCO: float = 30.0             # kDa; rejection as for Ultrafiltration
    rejection: dict = {}
    molecular_weights: dict = {}
    Lp: float = 50.0
    k_mass: float = 5e-6
    sigma: float = 1000.0
    fouling_coefficient: float = 0.0
```

The call is `df(inlet, buffer, n_diavolumes=None, TMP=1.0, feed_volume=None)`
and returns `((retentate, permeate), info)`. With `n_diavolumes=None` the
buffer stream is the buffer fed: it enters the balance and sets
$N_{DV}$ = buffer / inlet (amounts as the volume proxy). With
`n_diavolumes` given, only the buffer's composition is used, scaled to
$N_{DV}$ times the inlet, and `info['buffer_consumed']` is the buffer that
entered, so retentate + permeate = inlet + buffer consumed for every species.

#### Governing Equations

**Diavolume Definition**:

$$N_{DV} = \frac{V_{buffer\,added}}{V_{retentate}}$$

**Impurity Removal** (constant volume diafiltration):

$$\frac{C}{C_0} = \exp(-N_{DV}(1-R))$$

For freely permeating species ($R = 0$):

$$\frac{C}{C_0} = \exp(-N_{DV})$$

**Buffer wash-in** of a species fed at $C_b$ with sieving $s = 1 - R$:

$$\frac{C}{C_b} = \frac{1 - \exp(-s N_{DV})}{s}$$

which tends to $N_{DV}$ (everything added stays) as $R \to 1$.

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

params = DiafiltrationParams(MWCO=30.0, membrane_area=5.0)  # 30 kDa

df = Diafiltration(params)
feed = make_stream({'mAb': 10.0, 'salt': 150.0, 'buffer': 840.0}, T=298.0, P=101325.0)
new_buffer = make_stream({'new_buffer': 5000.0}, T=298.0, P=101325.0)  # 5 diavolumes

(retentate, permeate), info = df(feed, new_buffer)
print(f"Diavolumes: {float(info['n_diavolumes']):.1f}")
print(f"Salt removal: {1 - float(retentate['F_salt'] / feed['F_salt']):.1%}")
print(f"mAb kept: {float(retentate['F_mAb']) / 10.0:.2%}")
```

---

(tff-tangential-flow-filtration)=
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

`TFF.uf_df_uf(inlet, buffer, CF_initial, n_diavolumes, CF_final)` returns
`((product, uf1_permeate, df_permeate, uf2_permeate), info)`; the four
streams sum to the inlet plus `info['buffer_consumed']` for every species.

---

(chromatography)=
## Chromatography

(proteinachromatography)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class ProteinAParams:
    column_volume: float          # Column volume (L)
    q_max: float = 35.0           # Maximum binding capacity (g/L resin)
    K_d: float = 0.1              # Dissociation constant (g/L)
    target_species: str = "mAb"
    yield_factor: float = 0.95    # Elution yield (0-1)
    impurity_clearance: dict      # impurity -> LRV, default HCP 2, DNA 3, cells 4
    k_ads: float | None = None    # Adsorption rate (1/min), for kinetic DBC
    n_plates: float | None = None # Plate count, for elution pool volume
    elution_cv: float = 2.0       # Elution retention volume (CV)
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `inlet` | Stream | - | Feed stream with mAb (component flows as mass) |
| `load_volume` | float or None | L | Volume of feed loaded. `None` (default) loads the whole inlet; the column capacity, $DBC \cdot V_{column}$, then limits what binds and the excess breaks through |
| `feed_volume` | float | L | Total feed volume, so `load_volume / feed_volume` is the fraction loaded. Required with `load_volume` (the stream carries amounts, not a volume; a `ValueError` says so). With it, the capacity uses the Langmuir loading at the feed concentration, $q_{max} C/(K_d + C)$ |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `product` | Stream | - | Elution pool (purified mAb) |
| `waste` | Stream | - | Flow-through, wash and column losses |
| `info['yield']` | float | - | Target in the product / target in the inlet (feed not loaded counts as lost) |
| `info['purity']` | float | - | Product purity |
| `info['mass_loaded']`, `info['mass_bound']` | float | mass | Target loaded and bound (bound is capped by capacity) |
| `info['capacity_utilization']` | float | - | Bound mass / column capacity |
| `info['impurity_clearance']` | dict | LRV | Clearance applied per impurity |
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

from difflow import make_stream

params = ProteinAParams(
    column_volume=10.0,    # L
    q_max=35.0,            # g/L (typical for MabSelect)
)

protein_a = ProteinAChromatography(params)
load = make_stream({'mAb': 250.0, 'HCP': 5.0, 'DNA': 0.1}, T=298.0, P=101325.0)

# Load the whole batch: 250 g of mAb on a 10 L column (346.5 g capacity)
(eluate, waste), info = protein_a(load)
print(f"Yield: {float(info['yield']):.2%}")
print(f"Capacity used: {float(info['capacity_utilization']):.0%}")
print(f"HCP clearance: {info['impurity_clearance']['HCP']:.1f} LRV")
```

---

(ionexchangechromatography)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class IEXParams:
    column_volume: float              # Column volume (L)
    mode: str = "bind_elute"          # 'bind_elute' or 'flow_through'
    q_max: float = 50.0               # Binding capacity (g/L)
    K_d: float = 0.5                  # Dissociation constant (g/L)
    target_species: str = "mAb"
    selectivity: dict = {}            # species -> binding selectivity (0-1)
    yield_factor: float = 0.90        # Step recovery of the target
    impurity_clearance: dict = {}     # impurity -> LRV across the step
```

CEX and AEX are the same unit; `mode` and the parameters say which one it
is. `load_volume` (L) is optional when calling it, and `None` loads the
whole inlet; a partial load needs `feed_volume` (L) as well. The column
holds at most $q \cdot V_{column}$, with $q = q_{max} C / (K_d + C)$ when the
feed concentration is known (`feed_concentration`, or the amount over
`feed_volume`) and $q = q_{max}$ otherwise. In bind-elute mode target beyond
that capacity breaks through to waste; in flow-through mode the impurity
binding is scaled down to fit it. `info['yield']` is the target in the
product over the target in the inlet, and `info['capacity']` reports the
capacity used.

#### Impurity clearance

An impurity listed in `impurity_clearance` reaches the product as
$10^{-\mathrm{LRV}}$ of what was loaded, in either mode; the rest goes to
waste. Impurities not listed fall back to a selectivity rule: in
flow-through mode a fraction `selectivity` binds, and in bind-elute mode
$s^2 (1 - Y)$ co-elutes with the product. That rule caps carry-over at
$1 - Y$, so a 90% step would remove at least 90% of every impurity, far
more than a polishing step removes aggregate. State clearances explicitly
when purity matters.

`TYPICAL_CEX_CLEARANCE` and `TYPICAL_AEX_CLEARANCE` in the same module are
the representative values the packaged trains use (CEX: HCP 0.5, DNA 1.0,
aggregates 0.7; AEX: HCP 1.5, DNA 3.0, aggregates 0.1). They are not
measured clearances. They are chosen so that a harvest at about 5% HCP
and 3% aggregate ends near reported end-of-process levels: about 10 ppm
HCP against a <100 ppm target, and aggregate below the usual 1% target.
The AEX HCP value matches one reported flow-through step, 530 to 15 ppm
(Liu et al., *mAbs* 2:480, 2010, doi:10.4161/mabs.2.5.12645). Replace
them with your own process data. With them, `mAbDSPTrain` reports about
99.4% purity, 6 ppm HCP and 0.6% aggregate from that harvest, and returns
`hcp_ppm` and `aggregate_fraction` alongside `purity`.

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

(sizeexclusionchromatography)=
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

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
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
DBC = dynamic_binding_capacity(q_max=35.0, C_feed=5.0, K_d=0.5, residence_time=6.0, k_ads=1.0)

# Column productivity
P = column_productivity(DBC=30.0, column_volume=1.0, cycle_time=4.0)  # g/L/h

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

## Cost of goods

`difflow_bio.economics.cogs_breakdown` estimates the annual cost of goods of a
mAb process step by step, and cost per gram released. The cost follows the
process rather than the batch count alone:

| Item | How it is computed |
|---|---|
| Resin | Each chromatography step has its own resin, column volume, capacity and lifetime. Cycles per batch are `load / (DBC x CV)`, at least one, and the resin is replaced every `resin_lifetime_cycles` cycles. |
| Product mass | The harvest (`working_volume_L x titer_g_L`) runs through the steps in order, each keeping `step_yield` of what it receives, so a step's load depends on everything upstream. |
| Buffers | Column volumes per cycle for chromatography, litres per m2 for membranes, litres per batch otherwise, each priced from `buffer_usd_L`. |
| Media | Basal medium, fed-batch feed and seed train, per litre of working volume. |
| Labor | Fixed support staff, plus operator hours per batch for the upstream train and for each downstream step. |
| Other | QC per batch, single-use items per step, utilities, liquid waste, maintenance and depreciation on CAPEX. |
| Failed batches | Batches are charged when started and only `batch_success_rate` of them are released. |

Cycles per batch are continuous, `1 + softplus(k (n - 1)) / k` with
`n = load / (DBC x CV)`, so cost has a derivative with respect to titer,
binding capacity, column volume and every yield. `jax.grad` of cost per gram
with respect to titer is therefore not just `-cost / titer`: a higher titer
also loads the columns harder.

```python
import dataclasses
import jax
from difflow_bio.economics import load_cost_model, cogs_breakdown

process, basis = load_cost_model()          # the shipped reference
out = cogs_breakdown(process, basis)
out["cost_per_g"], out["steps"]["protein_a_capture"]["cycles"]

d_titer = jax.grad(lambda t: cogs_breakdown(
    dataclasses.replace(process, titer_g_L=t), basis)["cost_per_g"])(5.0)
```

### Using your own data

Every price, rate and process specification is data. `load_cost_model(path)`
reads a YAML or JSON file with a `process` section (bioreactor, batches and
the ordered `steps`, each with a `type` of `chromatography`, `filtration` or
`yield`) and a `basis` section (prices and labor rates). The shipped
`difflow_bio/economics/data/mab_reference.yaml` is the template: copy it,
replace the numbers with your own plant's, and load the copy. Every number
in it is tagged with its source: `[database]` (copied from the resin
database), `[carried over]` (the value the older functions used) or
`[placeholder]` (an order-of-magnitude assumption). **Most are placeholders**,
and the reference result is not a benchmark.

The benchmark is a published process encoded in the same format:
`difflow_bio/economics/data/petrides2015_mab.yaml`, the large-scale mAb
process of D. Petrides, *Bioprocess Design and Economics* (Intelligen, 2015,
section 11.6.3; an improved version is Chapter 11 of Harrison, Todd, Rudge and
Petrides, *Bioseparations Science and Engineering*, 2nd ed., Oxford University
Press, 2015). Every number in it is tagged with the page that states it, or
marked derived or not stated. From the source's stated inputs (a 15,000 L
fed-batch harvest at about 2 g/L, Protein A, IEX and HIC columns with their
volumes, capacities, yields, buffer volumes, prices and lifetimes, 80 batches
a year), `cogs_breakdown` reproduces:

| Quantity | difflow | Source |
|---|---|---|
| Product per batch / per year | 19.28 kg / 1,542 kg | 19.3 kg / 1,544 kg |
| Cycles per batch, Protein A / IEX / HIC | 3.85 / 3.01 / 3.00 | 4 / 3 / 3 (whole cycles) |
| Protein A / HIC elution buffer per batch | 9,658 / 2,846 L | 10,002 / 2,990 kg (Table 11.15) |
| Resin replacement | $18.5M/yr | within Consumables, $23.6M/yr (Table 11.17) |
| Media | $11.2M/yr | within Raw Materials, $16.7M/yr (Table 11.17) |

The source gives labor, facility-dependent, QC and miscellaneous costs only as
totals, without the inputs behind them, so its total of $84/g is not
reproduced; those categories are zero in the benchmark file.

`ChromatographyStep.from_resin` builds a step from the resin database
(capacity derated to 80 % of `q_max`, price and lifetime), and
`ProcessSpec.from_dict` / `CostBasis.from_dict` build the rest from plain
dictionaries; unknown keys and unpriced buffers are reported by name.

The older `estimate_*` functions remain for coarse estimates.
`estimate_total_opex` now prices each chromatography step with its own resin
(Protein A capture, then cation and anion exchange) and warns that it is
deprecated in favour of `cogs_breakdown`.
