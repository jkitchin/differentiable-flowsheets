# Carbon Capture Unit Operations

![difflow_cc](../images/plugins/difflow-cc-logo.svg)

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

# Get solvent properties
mea = get_solvent("MEA")
print(f"Heat of absorption: {mea.heat_of_absorption} kJ/mol")
print(f"Molecular weight: {mea.MW} g/mol")
```

#### Available Solvent Properties

| Property | Description | Units |
|----------|-------------|-------|
| `MW` | Molecular mass | g/mol |
| `density` | Solution density | kg/m^3 |
| `heat_of_absorption` | Heat released on CO2 absorption | kJ/mol CO2 |
| `loading_capacity` | Maximum CO2 loading | mol CO2/mol amine |
| `kinetics` | Rate-constant data (see `reaction_rate_constant`) | varies |
| `regen_temperature`, `regen_energy` | Typical regeneration conditions | K, GJ/t |

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
# ['Zeolite_13X', 'Zeolite_5A', 'Mg_MOF_74', 'CALF_20', 'ZIF_8',
#  'AC_Coconut', 'AC_Nitrogen_Doped', 'PEI_Silica', 'TEPA_Alumina']

zeolite = get_adsorbent("Zeolite_13X")
print(f"CO2 capacity: {zeolite.CO2_capacity} mol/kg")
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
# ['Matrimid', 'PDMS', 'Cellulose_Acetate', 'PIM_1', 'ZIF8_Matrimid',
#  'MOF74_Polymer', 'PVAm_Carrier', 'IL_SIL_Membrane', 'CMS', 'Zeolite_DDR']

pim1 = get_membrane("PIM_1")
print(f"CO2 permeability: {pim1.permeability['CO2']} Barrer")
print(f"CO2/N2 selectivity: {pim1.selectivity['CO2_N2']}")
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

from difflow_cc.equilibrium.vle import equilibrium_loading

# VLE model for MEA (C_amine in mol/m^3, ~30 wt%)
vle = AmineVLE("MEA", C_amine=5000.0)

# Equilibrium CO2 pressure at a given loading
P_eq = vle.equilibrium_pressure(loading=0.4, T=393.15)
print(f"Equilibrium CO2 pressure: {float(P_eq):.0f} Pa")

# Loading in equilibrium with a CO2 partial pressure (robust inversion,
# used by the stripper's reboiler model)
loading = equilibrium_loading(10000.0, 313.15, "MEA")  # 10 kPa CO2
print(f"CO2 loading: {float(loading):.3f} mol/mol")
```

#### Equilibrium Model

The CO2 partial pressure over loaded amine follows a Kent-Eisenberg type correlation:

$$P_{CO_2} = K_0 \exp\left(\frac{-\Delta H_{abs}}{RT}\right) \frac{\alpha^2}{(1 - \alpha/\alpha_{max})^{1.5}}$$

Where:
- $\alpha$: CO2 loading (mol CO2/mol amine), $\alpha_{max}$ the solvent's capacity
- $\Delta H_{abs}$: Heat of absorption
- $K_0$: calibrated to 1 kPa at $\alpha = 0.4$, 313 K for 30 wt% MEA

### Adsorption Isotherms

Multiple isotherm models are available:

```python
from difflow_cc import langmuir, sips, toth, dual_site_langmuir

# Langmuir isotherm (q_sat in mol/kg, b in 1/Pa)
q = langmuir(P=100000.0, q_sat=5.0, b=0.001)

# Sips isotherm (Freundlich-Langmuir)
q = sips(P=100000.0, q_sat=5.0, b=0.001, n=0.8)

# Toth isotherm
q = toth(P=100000.0, q_sat=5.0, b=0.001, t=0.7)

# Dual-site Langmuir
q = dual_site_langmuir(P=100000.0, q1=3.0, b1=0.01, q2=2.0, b2=0.0001)
```

#### Temperature-Dependent Isotherms

```python
from difflow_cc import langmuir_T, get_isotherm

# Temperature-dependent Langmuir: b = b0 exp(Q / (R T)), Q the heat of
# adsorption (J/mol, positive)
q = langmuir_T(P=100000.0, T=298.15, q_sat=5.0, b0=1e-9, Q=36000.0)

# Fitted isotherm for a database adsorbent (CO2 by default)
isotherm = get_isotherm("Zeolite_13X")
q = isotherm(100000.0, 298.15)   # (P, T)
```

#### Working Capacity

```python
from difflow_cc import working_capacity_PSA, working_capacity_TSA, get_isotherm

isotherm = get_isotherm("Zeolite_13X")

# PSA working capacity: q(P_ads, T) - q(P_des, T)
wc_psa = working_capacity_PSA(isotherm, P_ads=500000.0, P_des=100000.0, T=298.15)

# TSA working capacity: q(P, T_ads) - q(P, T_des)
wc_tsa = working_capacity_TSA(isotherm, P=15000.0, T_ads=298.15, T_des=423.15)
print(f"PSA {float(wc_psa):.2f}, TSA {float(wc_tsa):.2f} mol/kg")
```

(solubility-models)=
### Solubility Models

```python
from difflow_cc import co2_physical_solubility, diffusivity_co2_amine

# CO2 physical solubility in water
H = co2_physical_solubility(T=298.15)

# CO2 diffusivity in amine solution (C_amine in mol/m^3)
D = diffusivity_co2_amine(T=313.15, solvent="MEA", C_amine=5000.0)  # m²/s
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
k2 = reaction_rate_constant(T=313.15, solvent="MEA")
print(f"Rate constant: {float(k2):.0f} L/(mol·s)")

# Hatta number (reaction vs. diffusion); C_amine in mol/m^3, kL in m/s
Ha = hatta_number(T=313.15, solvent="MEA", C_amine=5000.0, kL=1e-4)
print(f"Hatta number: {float(Ha):.1f}")

# Enhancement factor (regime chosen automatically)
E = enhancement_factor(T=313.15, solvent="MEA", C_amine=5000.0, kL=1e-4)
print(f"Enhancement factor: {float(E):.1f}")
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

from difflow_cc import henry_constant

# Gas-side mass transfer coefficient (Onda)
k_G = gas_film_coefficient(
    u_G=1.0,          # m/s superficial gas velocity
    d_p=0.05,         # m packing nominal diameter
    mu_G=1.8e-5,      # Pa·s
    rho_G=1.1,        # kg/m^3
    D_G=1.5e-5,       # m²/s
    a_p=250.0,        # m²/m³ packing specific area
)

# Liquid-side coefficient
k_L = liquid_film_coefficient(
    u_L=0.01,         # m/s superficial liquid velocity
    d_p=0.05,
    mu_L=2e-3,        # Pa·s
    rho_L=1010.0,     # kg/m^3
    D_L=1.5e-9,       # m²/s
    a_p=250.0,
)

# Overall gas-side coefficient (H: Henry's constant, P: total pressure)
H_CO2 = henry_constant(313.15, "MEA")
K_G = overall_mass_transfer(k_G=k_G, k_L=k_L, E=E, H=H_CO2, P=101325.0)
```

---

## Unit Operations

(amineabsorber)=
### AmineAbsorber

**Location**: `difflow_cc/units/absorber.py`

**Class**: `AmineAbsorber`

**Description**: Equilibrium-stage model for amine absorption columns.

#### Parameters

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class AbsorberParams:
    solvent: str                   # Solvent name, e.g. "MEA"
    n_stages: int = 10             # Number of theoretical stages
    solvent_conc: float = 30.0     # Amine concentration (wt%)
    L_G_ratio: float = 3.0         # Liquid-to-gas molar ratio
    T_gas_in: float = 313.15       # K
    T_liquid_in: float = 313.15    # K (operating temperature)
    P_absorber: float = 101325.0   # Pa
    stage_efficiency: float = 0.25 # Murphree efficiency
    lean_loading: float = 0.2      # Lean loading (mol CO2/mol amine)
    model_water_transfer: bool = False
```

`L_G_ratio`, `solvent_conc` and `lean_loading` build the lean solvent
when none is passed. A `solvent_in` stream (species `Amine`, `H2O`,
`CO2_absorbed`), such as the stripper's lean outlet, overrides them: its
flows set the amine and water rates, the lean loading and therefore L/G.

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `gas_in` | Stream | - | Inlet flue gas |
| `solvent_in` | Stream, optional | - | Lean solvent (`Amine`, `H2O`, `CO2_absorbed`) |
| `T_op` | float, optional | K | Operating temperature (default `T_liquid_in`) |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `gas_out` | Stream | - | Treated gas |
| `solvent_out` | Stream | - | Rich solvent; `CO2_absorbed` is the TOTAL CO2 it holds (lean + captured) |
| `info['capture_efficiency']` | float | - | CO2 capture fraction |
| `info['CO2_captured']` | float | mol/s | CO2 moved from gas to liquid |
| `info['rich_loading']`, `info['lean_loading']` | float | mol/mol | Solvent loadings |
| `info['absorption_factor']` | float | - | Kremser $A = L/(mG)$ |
| `info['n_stages_effective']` | float | - | Equilibrium stages after Murphree efficiency |

The Kremser form holds for every absorption factor: when $A < 1$ the
column cannot capture more than the fraction $A$ however many stages it
has, which is what slow, low-capacity solvents such as MDEA show at
flue-gas conditions.

#### Example Usage

```python
from difflow_cc import AmineAbsorber, AbsorberParams
from difflow import make_stream

params = AbsorberParams(
    solvent="MEA",
    solvent_conc=30.0,
    n_stages=15,
    stage_efficiency=0.25,
    lean_loading=0.2,
    L_G_ratio=3.0,
)
absorber = AmineAbsorber(params)

flue_gas = make_stream(
    {'N2': 75.0, 'CO2': 12.0, 'H2O': 8.0, 'O2': 5.0},
    T=313.15, P=101325.0
)

gas_out, rich_solvent, info = absorber(flue_gas)

print(f"CO2 capture: {float(info['capture_efficiency']):.1%}")
print(f"Rich loading: {float(info['rich_loading']):.3f} mol/mol")
```

---

(aminestripper)=
### AmineStripper

**Location**: `difflow_cc/units/stripper.py`

**Class**: `AmineStripper`

**Description**: Equilibrium-stage stripper for solvent regeneration.

#### Parameters

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class StripperParams:
    solvent: str                         # Solvent name
    n_stages: int = 8
    T_reboiler: float = 393.15           # K
    P_stripper: float = 200000.0         # Pa
    steam_ratio: float = 2.0             # mol H2O stripped per mol CO2
    T_condenser: float = 313.15          # K, overhead condenser outlet
    target_lean_loading: float = 0.2     # mol CO2/mol amine
    reboiler_duty: float = None          # W; if set, limits stripping
    cross_exchanger_approach: float = 10.0  # K
```

The lean loading cannot go below the loading in equilibrium with the
reboiler vapour: the CO2 partial pressure there is the column pressure
less the water vapour pressure over the solution, $P_{CO_2} = P -
x_w P^{sat}_w(T_{reb})$, inverted through the solvent's VLE. A hotter
reboiler or a lower column pressure therefore strips deeper; a reboiler
colder than the rich solvent is rejected. A specified `reboiler_duty`
caps the CO2 that can be released after the sensible heat is paid.

The overhead condenser returns the steam it condenses to the column as
reflux. The CO2 product leaves saturated with water at `T_condenser` and
the column pressure (about 96 % CO2 at 2 bar and 40 C) and the solvent
keeps its water, so an absorber-stripper loop needs no make-up water for
the stripping steam. The achieved reflux (mol condensate per mol CO2) is
`info["reflux_ratio"]`; the old `reflux_ratio` input is ignored.

#### Key Outputs

| Parameter | Description | Units |
|-----------|-------------|-------|
| `lean_solvent` | Regenerated solvent (`CO2_absorbed` = CO2 left in it) | Stream |
| `co2_product` | CO2 product stream | Stream |
| `info['specific_energy']` | Specific regeneration energy | GJ/t CO2 |
| `info['lean_loading']` | Achieved lean loading | mol/mol |
| `info['equilibrium_lean_loading']` | Reboiler equilibrium floor | mol/mol |

#### Example Usage

```python
from difflow_cc import AmineStripper, StripperParams

params = StripperParams(solvent="MEA", n_stages=10, T_reboiler=393.15)
stripper = AmineStripper(params)

# rich_solvent from the absorber example above
lean_solvent, co2_product, info = stripper(rich_solvent)

print(f"Regen. energy: {float(info['specific_energy']):.2f} GJ/t CO2")
print(f"Lean loading: {float(info['lean_loading']):.3f} mol/mol")
```

Feeding `lean_solvent` back to `absorber(flue_gas, lean_solvent)` closes
the solvent loop: the absorber then uses the stripper's amine, water and
lean loading, and at steady state the CO2 captured equals the CO2
stripped (water lost with the product needs make-up).

---

(membraneseparator)=
### MembraneSeparator

**Location**: `difflow_cc/units/membrane.py`

**Class**: `MembraneSeparator`

**Description**: Single-stage membrane gas separator using solution-diffusion model.

#### Parameters

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class MembraneParams:
    membrane_type: str                  # Name from list_membranes()
    area: float = 1000.0                # m^2
    thickness: float = None             # micrometres; None uses the database default
    pressure_ratio: float = 10.0        # Feed/permeate pressure
    T_operation: float = 298.15         # K
    feed_pressure: float = None         # Pa; None uses the feed stream's P
    permeate_pressure: float = None     # Pa; from the ratio if None
    stage_cut_target: float = None      # If set, the area is solved to hit it
```

`pressure_ratio` must exceed 1 (and an explicit permeate pressure must be
below the feed pressure); otherwise nothing can permeate and the
parameters are rejected.

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

**Complete mixing.** Both sides are taken as well mixed, so the
retentate composition $x_i$ is the one the membrane sees and every
species obeys

$$F_{perm,i} = A\,Q_i\,(x_i P_{feed} - y_i P_{perm})$$

with $y_i$ the permeate composition. The closure $\sum_i y_i = 1$ is
solved exactly for the stage cut (or, with `stage_cut_target`, for the
area), so no species ever permeates against its partial-pressure
gradient and every retentate flow stays non-negative. With enough area
everything permeates and the permeate tends to the feed composition.

#### Example Usage

```python
from difflow import make_stream
from difflow_cc import MembraneSeparator, MembraneParams

params = MembraneParams(
    membrane_type="PIM_1",     # a name from list_membranes()
    area=200.0,
    thickness=1.0,             # micrometres
    pressure_ratio=10.0,
)
membrane = MembraneSeparator(params)

# Feed at 10 bar; the feed side runs at the stream's pressure
flue_gas = make_stream({'N2': 85.0, 'CO2': 15.0}, T=298.15, P=1000000.0)

retentate, permeate, info = membrane(flue_gas)

print(f"Stage cut: {float(info['stage_cut']):.2%}")       # 11.74%
print(f"CO2 purity: {float(info['CO2_purity']):.1%}")     # 55.0%
print(f"CO2 recovery: {float(info['CO2_recovery']):.1%}") # 43.0%

# Or fix the stage cut and let the unit solve for the area
_, _, cut = MembraneSeparator(MembraneParams(
    membrane_type="PIM_1", thickness=1.0, stage_cut_target=0.2))(flue_gas)
print(f"Area for a 20% cut: {float(cut['area_used']):.0f} m^2")
```

---

(multistagemembrane)=
### MultistageMembrane

**Location**: `difflow_cc/units/membrane.py`

**Class**: `MultistageMembrane`

**Description**: A cascade of `MembraneSeparator` stages, in series or as a two-stage enriching cascade with recycle.

#### Process Role

One membrane stage cannot be both selective and complete. Its purity is
set by the selectivity and the pressure ratio, its recovery by the area,
and pushing the area up to capture the last of the CO2 drags the permeate
composition back towards the feed: the single PIM-1 stage of the previous
section gives 55% purity at 43% recovery over 200 m^2, 31% purity at 81%
recovery over 1000 m^2, and over 5000 m^2 it permeates essentially the
whole feed (15% CO2) --- no separation at all. Staging is the way out,
and which way you stage depends on which of the two you need:

- **`series`** --- each stage treats the previous *retentate*, and the
  permeates are pooled. Recovery rises (every stage gets another chance at
  the CO2 the last one missed) and the pooled purity falls, because the
  later stages are working on an increasingly CO2-lean gas.
- **`permeate_recycle`** --- the stage-1 permeate is recompressed to the
  feed pressure and enriched in stage 2, whose permeate is the product;
  the stage-2 retentate is recycled to the stage-1 inlet and the loop is
  converged (`info['recycle_residual']`). Purity rises (the CO2 is
  enriched twice) and recovery falls. This layout is two stages by
  definition; other `n_stages` are rejected.

#### Parameters

The constructor takes a `MembraneParams` (the same one
`MembraneSeparator` takes, with `area` read **per stage**) plus the
cascade's own arguments:

<!-- doc-test: skip: constructor signature listing, params undefined -->
```python
MultistageMembrane(params, n_stages=2, configuration="series",
                   recycle_iterations=100, stage_params=None)
#  stage_params: optional list of MembraneParams, one per stage
```

Stage 2 of `permeate_recycle` sees only the stage-1 permeate, a small
fraction of the feed, so on the same area it permeates nearly all of
it and enriches nothing; give it its own, smaller area through
`stage_params`.

#### Inputs and Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `feed` | Stream | - | Feed gas |
| `retentate` | Stream | - | Final treated gas |
| `permeate` | Stream | - | Final CO2 product |
| `info['overall_CO2_recovery']` | float | - | CO2 to the product, as a fraction of the feed's |
| `info['overall_CO2_purity']` | float | - | CO2 mole fraction of the product |
| `info['stage_info']` | list | - | Each stage's own `MembraneSeparator` info dict |
| `info['recycle_residual']` | float | - | `permeate_recycle` only: final relative change of the recycle |

#### Governing Equations

There is no new physics here --- each stage is the complete-mixing
solution-diffusion model of `MembraneSeparator`. What the cascade adds is
the composition of stage cuts, which for stages in series is

$$\theta_{overall} = 1 - \prod_k (1 - \theta_k)$$

and a pooled product purity

$$y_{CO_2} = \frac{\sum_k \dot{n}_{CO_2,k}^{perm}}{\sum_k \dot{n}_k^{perm}}$$

#### Example Usage

```python
from difflow import make_stream
from difflow_cc import MembraneSeparator, MultistageMembrane, MembraneParams

params = MembraneParams(membrane_type="Matrimid", area=2000.0, pressure_ratio=10.0)
stage2 = MembraneParams(membrane_type="Matrimid", area=200.0, pressure_ratio=10.0)
flue_gas = make_stream({'N2': 85.0, 'CO2': 15.0}, T=298.15, P=10e5)

_, _, one = MembraneSeparator(params)(flue_gas)
_, _, ser = MultistageMembrane(params, 2, "series")(flue_gas)
_, _, rec = MultistageMembrane(params, 2, "permeate_recycle",
                               stage_params=[params, stage2])(flue_gas)

#                  purity   recovery
# single stage      0.666     0.241
# series            0.629     0.414   <- recovery bought with purity
# permeate_recycle  0.957     0.167   <- purity bought with recovery
```

#### Design Considerations

- **Compression is not included in the duty.** `pressure_ratio` is a
  membrane parameter, not a compressor; the duty that sustains it
  belongs to a `CompressionTrain` ([CO2 Compression](#co2-compression)).
  `permeate_recycle` reports the interstage recompression it assumes in
  `info['interstage_compression']`.

---

(adsorption-units)=
### Adsorption Units

**Location**: `difflow_cc/units/adsorption.py`

Four swing adsorption variants, one per regeneration route. They share
`AdsorptionParams` and the same Langmuir working-capacity calculation;
what differs is *which* variable swings, and therefore which energy term
the cycle pays for.

| Class | Swing | Regeneration | Energy reported |
|-------|-------|--------------|-----------------|
| [`PSAUnit`](#psaunit) | Pressure, above ambient | Blowdown to a lower pressure | `compression_power` |
| [`VSAUnit`](#vsaunit) | Pressure, below ambient | Vacuum | `vacuum_power` |
| [`TSAUnit`](#tsaunit) | Temperature | Heating | `heating_power` |
| [`TVSAUnit`](#tvsaunit) | Both | Warm *and* evacuated | `thermal_power` + `vacuum_power` |

#### Parameters

<!-- doc-test: skip: field listing of a library dataclass, not a runnable example -->
```python
@dataclass
class AdsorptionParams:
    adsorbent: str                   # Name from list_adsorbents()
    cycle_type: str = 'PSA'          # 'PSA' | 'TSA' | 'VSA' | 'TVSA'
    bed_mass: float = 1000.0         # kg adsorbent per bed
    n_beds: int = 2                  # Beds, for continuous operation
    void_fraction: float = 0.4
    # Pressure swing
    P_adsorption: float = 101325.0   # Pa
    P_desorption: float = 10000.0    # Pa
    # Temperature swing
    T_adsorption: float = 298.15     # K
    T_desorption: float = 393.15     # K
    # Cycle timing (s): the four steps whose sum is the cycle time
    t_adsorption: float = 300.0
    t_blowdown: float = 60.0
    t_purge: float = 120.0
    t_repressure: float = 60.0
    # Declared targets. Carried for sizing and costing; the performance
    # the unit reports is computed, not clipped to these.
    CO2_purity_target: float = 0.95
    CO2_recovery_target: float = 0.90
```

#### The quantity all four compute

Every variant reduces to a **working capacity** --- the difference
between what the adsorbent holds at adsorption conditions and what it
still holds at regeneration conditions:

$$q(P, T) = \frac{q_{max} b(T) P}{1 + b(T) P}, \qquad
b(T) = b_0 \exp\left(\frac{-\Delta H_{ads}}{RT}\right)$$

$$\Delta q_{working} = q(P_{ads}, T_{ads}) - q(P_{des}, T_{des})$$

with recovery following from the ratio of what the beds can take to
what the feed brings, through an unused-bed (mass-transfer-zone)
saturation:

$$\phi = \frac{\Delta q_{working}\, m_{bed}\, n_{beds}}
{\dot{n}_{CO_2,feed}\, t_{cycle}}, \qquad
\text{recovery} = R_{max}\left(1 - e^{-\phi/R_{max}}\right),
\quad R_{max} = 0.95$$

A small bed is capacity-limited (recovery $\approx \phi$); an oversized
one approaches $R_{max}$, so bed mass, number of beds and step times
always move the answer.

This is an equilibrium, lumped-capacity model: no intra-particle
diffusion, no breakthrough profile, no bed dynamics. It sizes beds and
compares technologies; it does not replace a cycle simulation. Two
consequences worth knowing before reading its numbers:

- **Recovery saturates towards 95% of the feed CO2** (blowdown and
  breakthrough losses); `info['capacity_ratio']` is $\phi$, and values
  well above 1 mean the bed is oversized.
- **Every species is conserved.** The co-adsorbed impurity implied by
  the purity correlation is drawn from all non-CO2 species in proportion
  to their feed; the offgas is the rest. `info['purity']` is the product
  stream's actual CO2 fraction.
- **Infeasible points are flagged, not hidden.** With no working
  capacity (a TSA with `T_desorption <= T_adsorption`, or a dilute feed
  against too shallow a vacuum) the product is empty, `info['purity']`
  is 0 and `info['feasible']` is False.
- **Purity comes from the adsorbent's selectivity and the swing**, not
  from a breakthrough calculation: it is
  `s' / (s' + 1)` with `s'` the selectivity scaled by the swing ratio.
  That is a reasonable ranking of adsorbents and a poor prediction of a
  real product stream.

(psaunit)=
#### PSAUnit

Adsorption at elevated pressure, regeneration by blowdown to a lower one.
The swing is in $P$ at constant $T$, so the working capacity is the gap
between two points on one isotherm --- which is why PSA wants a steep
isotherm at the feed partial pressure and an already-compressed feed.
Its natural home is pre-combustion capture and hydrogen purification,
where the gas arrives at pressure and the compression is paid for
anyway. `info['compression_power']` is what it costs to hold
`P_adsorption`.

(vsaunit)=
#### VSAUnit

The same pressure swing, moved below atmospheric: adsorb near ambient,
desorb under vacuum. Post-combustion flue gas is at ambient pressure and
there is no compressing 100 times the flow of the CO2 in it, so the
cheaper move is to pull vacuum on the bed instead. The working capacity
is larger than PSA's for the same ratio --- the Langmuir isotherm is
steepest near the origin --- and the price is `info['vacuum_power']`,
which dominates the energy balance.

(tsaunit)=
#### TSAUnit

Temperature swings instead of pressure: $b(T)$ collapses with heating, so
the bed gives up its CO2 at constant pressure. That makes TSA the
variant that works on *dilute* feeds --- direct air capture at 400 ppm,
where no pressure ratio buys a useful working capacity --- and the
variant with the worst energy penalty, because the regeneration duty
includes the sensible heat of the whole bed:

$$Q_{regen} = \left(m_{bed} C_p + m_{CO_2} \Delta H_{ads}\right)
(T_{des} - T_{ads})$$

`info['Q_sensible']` and `info['Q_desorption']` report the two halves.
The thermal mass also sets the cycle time: beds have to be heated and
cooled, which is slow.

(tvsaunit)=
#### TVSAUnit

Both at once --- mildly warm and evacuated. The point is not to add the
two working capacities but to reach a given one at a *lower* desorption
temperature than TSA needs, which cuts the sensible-heat penalty and lets
amine-functionalised sorbents regenerate below the temperature at which
they degrade (see [Adsorbent Degradation](#adsorbent-degradation)). It is
the usual choice for solid-sorbent DAC, and it pays both
`info['thermal_power']` and `info['vacuum_power']`.

#### Example Usage

```python
from difflow import make_stream
from difflow_cc import PSAUnit, VSAUnit, AdsorptionParams

flue_gas = make_stream({'N2': 85.0, 'CO2': 15.0}, T=298.15, P=500000.0)

psa = PSAUnit(AdsorptionParams(
    adsorbent="Zeolite_13X",     # a name from list_adsorbents()
    cycle_type="PSA",
    bed_mass=5000.0,
    n_beds=4,
    P_adsorption=500000.0,
    P_desorption=100000.0,
))

product, offgas, info = psa(flue_gas)

print(f"CO2 purity:   {float(info['purity']):.1%}")          # 99.3%
print(f"CO2 recovery: {float(info['recovery']):.1%}")        # 44.8%
print(f"Productivity: {float(info['productivity']):.2f} "    # 1.64
      f"mol CO2/(kg.h)")
print(f"Working cap.: {float(info['working_capacity']):.3f} mol/kg")
```

The same feed at ambient pressure through a `VSAUnit`
(`P_adsorption=101325`, `P_desorption=10000`) reaches 84% recovery with a
3.4x larger working capacity --- the comparison the four classes exist to
make, and the reason `cycle_type` is a parameter rather than the class
name doing the work.

---

(heat-integration)=
## Heat Integration

**Location**: `difflow_cc/units/heat_integration.py`

Efficient heat recovery is critical for minimizing energy penalty:

```python
from difflow import make_stream
from difflow_cc import LeanRichExchanger, LeanRichExchangerParams

params = LeanRichExchangerParams(
    min_approach=10.0,      # K minimum approach
    effectiveness=0.85,
)
exchanger = LeanRichExchanger(params)

lean_hot = make_stream({"H2O": 200.0}, T=393.15, P=2e5)    # from stripper
rich_cold = make_stream({"H2O": 100.0}, T=313.15, P=2e5)   # from absorber

lean_cold, rich_hot, info = exchanger(lean_hot, rich_cold)

print(f"Duty: {float(info['Q'])/1e6:.3f} MW")
print(f"Effectiveness achieved: {float(info['effectiveness']):.2f}")
print(f"Heat recovery fraction: {float(info['heat_recovery_fraction']):.1%}")
```

One duty serves both sides: the minimum approach caps it at
$C_{min}(\Delta T_{in} - \Delta T_{min})$, both outlet temperatures
follow from it, and `info['effectiveness']` reports what was achieved
(below the requested value when the approach binds).

---

(co2-compression)=
## CO2 Compression

**Location**: `difflow_cc/units/compression.py`

CO2 must be compressed to pipeline or sequestration pressure:

```python
from difflow_cc import CompressionTrain, CompressionTrainParams

from difflow import make_stream

params = CompressionTrainParams(
    n_stages=4,
    P_outlet=15000000.0,      # 150 bar (supercritical)
    eta_isentropic=0.80,
    T_intercool=313.15,       # 40°C
)
compressor = CompressionTrain(params)

# The train compresses from the inlet stream's pressure (here 1 atm).
co2_stream = make_stream({'CO2': 100.0}, T=313.15, P=101325.0)

compressed, info = compressor(co2_stream)

print(f"Outlet: {float(compressed['P'])/1e5:.0f} bar")
print(f"Total power: {float(info['total_power'])/1e6:.2f} MW")
print(f"Specific power: {float(info['specific_power']):.0f} kJ/kg CO2")
```

---

(direct-air-capture)=
## Direct Air Capture

**Location**: `difflow_cc/units/dac.py`

Model direct air capture systems:

```python
from difflow import make_stream
from difflow_cc import SolidSorbentDAC, DACParams, LiquidSolventDAC, LiquidDACParams

params = DACParams(
    sorbent="PEI_Silica",
    cross_section=100.0,       # m² face area per contactor
    n_units=4,
    T_adsorption=298.15,
    T_desorption=373.15,
    cycle_time_ads=1800.0,     # s
    cycle_time_des=900.0,      # s
)
dac = SolidSorbentDAC(params)

# Time-averaged air feed to the plant (mol/s); capture can never exceed
# the CO2 it carries.
air = make_stream({'N2': 12800.0, 'O2': 3440.0, 'CO2': 6.9}, T=298.15, P=101325.0)

co2_product, info = dac(air)

print(f"Captured: {float(info['CO2_captured_tonne_yr']):.0f} t CO2/yr")
print(f"Capture efficiency: {float(info['capture_efficiency']):.1%}")
print(f"Sorbent utilization: {float(info['sorbent_utilization']):.1%}")
print(f"Thermal: {float(info['specific_thermal_GJ_tonne']):.1f} GJ/t CO2")

# Liquid (KOH) DAC: capture from transfer units over the packing depth
_, liq = LiquidSolventDAC(LiquidDACParams(contactor_height=8.0, L_G_ratio=2.0))()
print(f"Liquid DAC capture: {float(liq['capture_efficiency']):.1%}")
```

Solid-sorbent capture is the smaller of what the beds can take (sorbent
mass x working capacity / cycle time) and 95% of the CO2 in the processed
air; without an `ambient_air` stream the air flow is `air_velocity x
cross_section` per unit at 420 ppm. Liquid-solvent capture is
$1 - e^{-NTU}$ with NTU from the packing depth (`contactor_height`), the
air velocity and the liquid wetting (`L_G_ratio`), calibrated to 75% at
the default 8 m, 1.5 m/s, L/G = 2 design point; with no liquid nothing is
captured.

---

## Economics

**Location**: `difflow_cc/economics/`

Comprehensive economic analysis:

```python
from difflow_cc import (
    absorber_cost, stripper_cost, installed_cost,
    total_operating_cost, levelized_cost_capture, cost_of_co2_avoided,
    EconomicParams,
)

params = EconomicParams(
    capacity_factor=0.85,
    lifetime=25,               # years
    discount_rate=0.08,
)
CO2_rate = 1.0e6 * 1e6 / 44.01 / (8760 * 3600 * 0.85)   # mol/s for ~1 Mt/yr

# Capital costs (USD)
equipment = (absorber_cost(diameter=10.0, height=30.0)
             + stripper_cost(diameter=8.0, height=25.0,
                             reboiler_duty=150e6, condenser_duty=60e6))
capex = installed_cost(equipment)["total_overnight_cost"]

# Operating costs (USD/yr); duties in W, CO2 in mol/s
opex = total_operating_cost(
    steam_duty=150e6,
    electricity=20e6,
    cooling_duty=120e6,
    CO2_captured=CO2_rate,
    capital_cost=capex,
)["total_opex"]

# Levelized cost ($/t CO2)
lcoc = levelized_cost_capture(capex, opex, CO2_rate, params)
print(f"Levelized cost: ${float(lcoc['total_cost_per_tonne']):.1f}/tonne CO2")

# Cost of CO2 avoided
cca = cost_of_co2_avoided(
    capture_cost_per_tonne=lcoc["total_cost_per_tonne"],
    reference_emissions=0.80,     # t CO2/MWh without capture
    capture_emissions=0.10,       # t CO2/MWh with capture
    reference_energy=4.0e6,       # MWh/yr without capture
    capture_energy=3.0e6,         # MWh/yr after the energy penalty
)
print(f"Cost avoided: ${float(cca):.1f}/tonne CO2")
```

---

(degradation-models)=
## Degradation Models

**Location**: `difflow_cc/degradation/`

(amine-degradation)=
### Amine Degradation

Each model takes a parameter set describing the solvent and its service
conditions:

```python
from difflow_cc import (
    AmineDegradationParams,
    oxidative_degradation_rate,
    thermal_degradation_rate,
    total_amine_loss,
    solvent_lifetime,
)

deg = AmineDegradationParams(
    solvent="MEA",
    O2_concentration=0.05,     # 5% O2 in the flue gas
    T_absorber=313.15,
    T_stripper=393.15,         # reboiler temperature
    CO2_loading=0.4,
)

r_ox = oxidative_degradation_rate(313.15, deg)   # absorber conditions
r_th = thermal_degradation_rate(393.15, deg)     # stripper conditions

loss = total_amine_loss(deg)
print(f"Amine loss: {float(loss['total_kg_m3_yr']):.1f} kg/m^3/yr")
print(f"Solvent lifetime: {float(solvent_lifetime(deg)):.1f} years")
```

(adsorbent-degradation)=
### Adsorbent Degradation

```python
from difflow_cc import AdsorbentDegradationParams, capacity_fade, adsorbent_lifetime

ads = AdsorbentDegradationParams(
    material_type="amine_silica",
    T_desorption=373.15,
    humidity=0.1,
    cycles_per_day=48.0,
)

fade = capacity_fade(8760.0, ads)            # after one year of operation
print(f"Capacity fade: {float(fade['capacity_loss_percent']):.1f}%")

lifetime_h = adsorbent_lifetime(ads, min_capacity_fraction=0.8)
print(f"Adsorbent lifetime: {float(lifetime_h) / 24:.0f} days")
```

(membrane-aging)=
### Membrane Aging

```python
from difflow_cc import (
    MembraneAgingParams, physical_aging, plasticization, membrane_lifetime,
)

mem = MembraneAgingParams(membrane_type="glassy", T_operating=298.15)

# Physical aging (glassy polymers): permeance fraction after one year
remaining = physical_aging(1.0, mem)

# Plasticization from CO2 partial pressure
plast = plasticization(500000.0, mem)       # 5 bar CO2
print(f"Plasticized: {bool(plast['is_plasticized'])}")

lifetime = membrane_lifetime(mem, min_permeance_fraction=0.7)
print(f"Membrane lifetime: {float(lifetime):.1f} years")
```

---

## Examples

### Example 1: Complete Amine Capture Plant

The absorber, stripper and lean/rich exchanger close on the solvent: the
stripper's lean outlet is the absorber's `solvent_in`. A few
successive-substitution passes (or a `Flowsheet` recycle) converge the
loop, after which CO2 captured equals CO2 stripped.

```python
import jax
jax.config.update("jax_enable_x64", True)
from difflow import make_stream
from difflow.streams import get_flows
from difflow_cc import (
    AmineAbsorber, AbsorberParams,
    AmineStripper, StripperParams,
    LeanRichExchanger, LeanRichExchangerParams,
    CompressionTrain, CompressionTrainParams,
)

flue_gas = make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=101325.0)
absorber = AmineAbsorber(AbsorberParams(solvent="MEA", n_stages=15,
                                        L_G_ratio=3.5, lean_loading=0.25))
stripper = AmineStripper(StripperParams(solvent="MEA", n_stages=10))
exchanger = LeanRichExchanger(LeanRichExchangerParams())
compressor = CompressionTrain(CompressionTrainParams(n_stages=4))

@jax.jit
def loop_pass(rich):
    _, rich_hot, _ = exchanger(make_stream(get_flows(rich), 393.15, 2e5), rich)
    lean, co2, s_info = stripper(rich_hot)
    lean_flows = dict(get_flows(lean))
    lean_flows["H2O"] = lean_flows["H2O"] + get_flows(co2)["H2O"]  # make-up water
    lean = make_stream(lean_flows, T=313.15, P=101325.0)
    gas_out, rich, a_info = absorber(flue_gas, lean)
    return rich, (co2, s_info, a_info)

_, rich, _ = absorber(flue_gas)
for _ in range(200):
    rich, (co2, s_info, a_info) = loop_pass(rich)

compressed, c_info = compressor(co2)
print(f"Capture: {float(a_info['capture_efficiency']):.1%}")
print(f"Captured {float(a_info['CO2_captured']):.3f} = stripped "
      f"{float(s_info['CO2_stripped']):.3f} mol/s")
print(f"Regeneration: {float(s_info['specific_energy']):.2f} GJ/t CO2")
print(f"Compression: {float(c_info['total_power'])/1e3:.0f} kW")
```

### Example 2: Gradient-Based Optimization

```python
import jax
import jax.numpy as jnp
from difflow import make_stream
from difflow_cc import AmineAbsorber, AbsorberParams

flue_gas = make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=101325.0)

def capture_cost(x):
    """Cost per mol CO2 captured (illustrative weights)."""
    n_stages, L_G_ratio = x
    absorber = AmineAbsorber(AbsorberParams(
        solvent="MEA", n_stages=n_stages, L_G_ratio=L_G_ratio))
    _, _, info = absorber(flue_gas)

    capex = n_stages * 1.0      # per stage
    opex = L_G_ratio * 2.0      # per unit solvent circulation
    return (capex + opex) / info["CO2_captured"]

gradients = jax.grad(capture_cost)(jnp.array([10.0, 3.0]))
print(f"d/d(n_stages)={float(gradients[0]):.4f}, d/d(L_G)={float(gradients[1]):.4f}")
```

### Example 3: Technology Comparison

```python
from difflow import make_stream
from difflow_cc import (
    AmineAbsorber, AbsorberParams,
    MembraneSeparator, MembraneParams,
    VSAUnit, AdsorptionParams,
)

flue_gas = make_stream({"CO2": 15.0, "N2": 85.0}, T=313.15, P=101325.0)
compressed_gas = make_stream({"CO2": 15.0, "N2": 85.0}, T=313.15, P=10e5)

_, _, amine = AmineAbsorber(AbsorberParams(solvent="MEA"))(flue_gas)
_, _, mem = MembraneSeparator(MembraneParams(membrane_type="PIM_1", area=200.0,
                                             thickness=1.0))(compressed_gas)
_, _, vsa = VSAUnit(AdsorptionParams(adsorbent="Zeolite_13X", cycle_type="VSA",
                                     bed_mass=5000.0, n_beds=4))(flue_gas)

print(f"Amine (MEA):      capture {float(amine['capture_efficiency']):.1%}")
print(f"Membrane (PIM-1): recovery {float(mem['CO2_recovery']):.1%}, "
      f"purity {float(mem['CO2_purity']):.1%}")
print(f"VSA (13X):        recovery {float(vsa['recovery']):.1%}, "
      f"purity {float(vsa['purity']):.1%}")
```

---

## See Also

- [Examples: 01_amine_capture_fundamentals.ipynb](../src/difflow_cc/examples/01_amine_capture_fundamentals.ipynb) - Amine basics
- [Examples: 02_membrane_separation.ipynb](../src/difflow_cc/examples/02_membrane_separation.ipynb) - Membrane systems
- [Examples: 03_adsorption_processes.ipynb](../src/difflow_cc/examples/03_adsorption_processes.ipynb) - PSA/TSA/VSA
- [Examples: 04_optimization_with_gradients.ipynb](../src/difflow_cc/examples/04_optimization_with_gradients.ipynb) - Optimization
- [Examples: 05_integrated_capture_plant.ipynb](../src/difflow_cc/examples/05_integrated_capture_plant.ipynb) - Full plant
