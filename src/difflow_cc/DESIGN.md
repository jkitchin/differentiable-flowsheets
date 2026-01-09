# Carbon Capture Plugin (difflow_cc) - Design Document

## Overview

This document outlines the design and implementation plan for the `difflow_cc` plugin,
which provides differentiable unit operations for carbon capture technologies:
- **Amine-based absorption** (MEA, DEA, MDEA, piperazine, amino acid salts)
- **Membrane separation** (polymeric, mixed-matrix, facilitated transport, ceramic)
- **Adsorption systems** (PSA, TSA, VSA, TVSA with zeolites, MOFs, activated carbon)

All models are JAX-compatible for automatic differentiation and gradient-based optimization.

---

## Module Structure

```
difflow_cc/
├── __init__.py              # Plugin registration and exports
├── database.py              # Property databases for solvents, adsorbents, membranes
├── data/                    # YAML data files
│   ├── solvents.yaml        # Amine solvent properties
│   ├── adsorbents.yaml      # Adsorbent properties and isotherms
│   └── membranes.yaml       # Membrane material properties
├── equilibrium/
│   ├── __init__.py
│   ├── vle.py               # Vapor-liquid equilibrium for amine systems
│   ├── isotherms.py         # Adsorption isotherm models
│   └── solubility.py        # CO2 solubility correlations
├── kinetics/
│   ├── __init__.py
│   ├── amine_kinetics.py    # Reaction kinetics for amine-CO2
│   └── mass_transfer.py     # Mass transfer correlations
├── units/
│   ├── __init__.py
│   ├── absorber.py          # Amine absorber column
│   ├── stripper.py          # Amine regeneration stripper
│   ├── membrane.py          # Membrane separation units
│   └── adsorption.py        # PSA/TSA/VSA/TVSA units
└── flowsheets/
    ├── __init__.py
    ├── amine_loop.py        # Complete amine capture loop
    └── hybrid.py            # Hybrid membrane-amine systems
```

---

## 1. Database Module (`database.py`)

### 1.1 Amine Solvents

Properties for each solvent (from literature):

| Property | Description | Units | References |
|----------|-------------|-------|------------|
| `MW` | Molecular weight | g/mol | - |
| `density` | Liquid density | kg/m³ | - |
| `viscosity_coeffs` | Arrhenius viscosity params | - | - |
| `pKa` | Acid dissociation constant | - | [1] |
| `heat_of_absorption` | Enthalpy of CO2 absorption | kJ/mol CO2 | [2] |
| `henry_coeffs` | Henry's law coefficients | - | [3] |
| `reaction_rate_coeffs` | Kinetic parameters | varies | [4] |
| `capacity` | Typical CO2 loading capacity | mol CO2/mol amine | - |
| `regen_energy` | Regeneration energy | GJ/tonne CO2 | - |

**Solvents included:**
- **MEA** (monoethanolamine): k₂ ≈ 6000 L/(mol·s) at 25°C, Ea ≈ 50 kJ/mol
- **DEA** (diethanolamine): k₂ ≈ 1500 L/(mol·s) at 25°C
- **MDEA** (methyldiethanolamine): Base-catalyzed, k₂ ≈ 4-8 L/(mol·s)
- **Piperazine (PZ)**: k₂ ≈ 70,000 L/(mol·s), highest rate
- **AMP** (2-amino-2-methyl-1-propanol): Sterically hindered
- **Glycine**: k₂ ≈ 6600 L/(mol·s), amino acid salt
- **Sarcosine**: k₂ ≈ 9900 L/(mol·s), amino acid salt

**References:**
1. Perrin DD (1965). Dissociation Constants of Organic Bases in Aqueous Solution
2. Kim I, Svendsen HF (2007). Heat of absorption of CO2 with aqueous solutions of amines. Int J Greenhouse Gas Control
3. Versteeg GF, Van Swaaij WPM (1988). Solubility and diffusivity of acid gases in aqueous amine solutions. J Chem Eng Data
4. Versteeg GF et al. (1996). On the kinetics between CO2 and alkanolamines. Chem Eng Sci

### 1.2 Adsorbent Materials

| Property | Description | Units |
|----------|-------------|-------|
| `surface_area` | BET surface area | m²/g |
| `pore_volume` | Total pore volume | cm³/g |
| `pore_diameter` | Median pore diameter | Å |
| `density_bulk` | Bulk density | kg/m³ |
| `density_particle` | Particle density | kg/m³ |
| `heat_capacity` | Specific heat | J/(kg·K) |
| `thermal_conductivity` | Thermal conductivity | W/(m·K) |
| `isotherm_type` | Isotherm model type | string |
| `isotherm_params` | Model parameters | dict |
| `heat_of_adsorption` | Isosteric heat | kJ/mol |
| `CO2_selectivity_N2` | Selectivity ratio | - |

**Adsorbents included:**

**Zeolite 13X:**
- BET: 500-700 m²/g
- CO2 capacity: ~5-6 mmol/g at 1 bar, 25°C
- Heat of adsorption: 34.65 kJ/mol
- Best fit: Toth isotherm
- Reference: Cavenati S et al. (2004). J Chem Eng Data

**Mg-MOF-74:**
- BET: 1174 m²/g
- CO2 capacity: 8.61 mmol/g at 1 bar, 25°C
- Heat of adsorption: 36-46 kJ/mol
- Best fit: Dual-site Langmuir
- Reference: Caskey SR et al. (2008). J Am Chem Soc

**Activated Carbon:**
- BET: 800-1500 m²/g
- CO2 capacity: ~2-3 mmol/g at 1 bar, 25°C
- Heat of adsorption: 20-25 kJ/mol
- Best fit: Langmuir or Sips
- Reference: Sevilla M et al. (2011). Energy Environ Sci

**Amine-functionalized silica:**
- Capacity: 1-2 mmol/g
- Works via chemisorption
- Good for direct air capture
- Reference: Choi S et al. (2009). ChemSusChem

### 1.3 Membrane Materials

| Property | Description | Units |
|----------|-------------|-------|
| `permeance_CO2` | CO2 permeance | GPU |
| `selectivity_CO2_N2` | CO2/N2 selectivity | - |
| `selectivity_CO2_CH4` | CO2/CH4 selectivity | - |
| `activation_energy_P` | Permeability activation energy | kJ/mol |
| `thickness` | Typical membrane thickness | μm |
| `max_temperature` | Maximum operating temperature | K |
| `max_pressure` | Maximum pressure differential | bar |

**1 GPU = 10⁻⁶ cm³(STP)/(cm²·s·cmHg)**

**Materials included:**

**Polymeric (Matrimid):**
- CO2 permeance: ~10 GPU (thin film)
- CO2/N2 selectivity: 25-35
- Reference: Dai Y et al. (2012). J Membr Sci

**Mixed-matrix (ZIF-8/polymer):**
- CO2 permeance: 1000+ GPU
- CO2/N2 selectivity: 30-40
- Reference: Ordoñez MJC et al. (2010). J Membr Sci

**Facilitated transport (amine carrier):**
- CO2 permeance: 100-500 GPU
- CO2/N2 selectivity: 100-500
- Humidity dependent
- Reference: Zhao S et al. (2016). J Membr Sci

**Ceramic (carbon molecular sieve):**
- CO2 permeance: 50-200 GPU
- CO2/N2 selectivity: 30-60
- High temperature stable
- Reference: Steel KM (2003). Carbon

---

## 2. Equilibrium Models (`equilibrium/`)

### 2.1 VLE for Amine Systems (`vle.py`)

**Simplified Model (Kent-Eisenberg):**

For initial implementation, use the Kent-Eisenberg approach with fitted equilibrium constants:

```
CO2 + 2RNH2 ⇌ RNHCOO⁻ + RNH3⁺  (carbamate formation)
CO2 + H2O ⇌ HCO3⁻ + H⁺         (bicarbonate formation)
```

The CO2 partial pressure is calculated from:
```
P_CO2 = H_CO2 * [CO2]_free
```

where Henry's constant has temperature dependence:
```
ln(H) = A + B/T + C*ln(T)
```

**Extensibility hook for eNRTL:**
The design includes an abstract `VLEModel` base class that can be extended with:
- Electrolyte NRTL (Chen et al., 1986)
- UNIQUAC with electrolyte extension
- PC-SAFT for more rigorous treatment

**References:**
- Kent RL, Eisenberg B (1976). Hydrocarbon Processing
- Chen CC et al. (1986). AIChE J - eNRTL model

### 2.2 Adsorption Isotherms (`isotherms.py`)

All isotherms implemented as JAX-compatible functions:

**Langmuir:**
```
q = q_sat * b * P / (1 + b * P)
b = b0 * exp(-ΔH/(R*T))
```

**Dual-site Langmuir (DSL):**
```
q = q1 * b1 * P / (1 + b1 * P) + q2 * b2 * P / (1 + b2 * P)
```

**Sips (Langmuir-Freundlich):**
```
q = q_sat * (b * P)^n / (1 + (b * P)^n)
```

**Toth:**
```
q = q_sat * b * P / (1 + (b * P)^t)^(1/t)
```

Temperature dependence via Clausius-Clapeyron:
```
∂ln(P)/∂(1/T)|_q = -ΔH_ads / R
```

**References:**
- Do DD (1998). Adsorption Analysis: Equilibria and Kinetics
- Ruthven DM (1984). Principles of Adsorption and Adsorption Processes

### 2.3 CO2 Solubility (`solubility.py`)

Physical solubility (Henry's law) correlations for different solvents:

```python
def henry_constant(T, solvent):
    """Henry's constant for CO2 in solvent.

    ln(H) = A + B/T + C*ln(T) + D*T

    References:
        Versteeg GF, Van Swaaij WPM (1988). J Chem Eng Data
    """
```

---

## 3. Kinetics Models (`kinetics/`)

### 3.1 Amine-CO2 Reaction Kinetics (`amine_kinetics.py`)

**Zwitterion Mechanism (Caplow, 1968):**

For primary/secondary amines (MEA, DEA, PZ):
```
CO2 + RNH2 ⇌ RNH2⁺COO⁻     (zwitterion formation, k2)
RNH2⁺COO⁻ + B → RNHCOO⁻ + BH⁺  (deprotonation by base B)
```

Overall rate:
```
r = k_obs * [CO2] * [Amine]

where k_obs = k2 / (1 + k2/(k_B*[B]))
```

For fast deprotonation (pseudo-first-order in CO2):
```
k_obs ≈ k2 * [Amine]
```

**Arrhenius Parameters:**
```
k2 = A * exp(-Ea/(R*T))
```

| Amine | A (L/mol·s) | Ea (kJ/mol) | k2 at 25°C |
|-------|-------------|-------------|------------|
| MEA | 4.4×10¹¹ | 44.9 | 5900 |
| DEA | 5.8×10⁹ | 42.0 | 1200 |
| MDEA | - | - | 4-8 (base-catalyzed) |
| PZ | 4.1×10¹³ | 33.6 | 53,000 |

**Termolecular Mechanism (Crooks & Donnellan, 1989):**
For tertiary amines (MDEA) - base-catalyzed hydration:
```
CO2 + R3N + H2O → HCO3⁻ + R3NH⁺
r = k_OH * [CO2] * [OH⁻]
```

**References:**
- Caplow M (1968). J Am Chem Soc 90:6795
- Crooks JE, Donnellan JP (1989). J Chem Soc Perkin Trans 2:331
- Versteeg GF, Oyevaar MH (1989). Chem Eng Sci 44:1264

### 3.2 Mass Transfer (`mass_transfer.py`)

**Overall mass transfer coefficient:**
```
1/KG = 1/kg + H/(E*kl)
```

where:
- `kg`: Gas-side mass transfer coefficient
- `kl`: Liquid-side mass transfer coefficient
- `E`: Enhancement factor due to reaction
- `H`: Henry's constant

**Enhancement Factor (pseudo-first-order regime):**
```
E = Ha / tanh(Ha)

Ha = sqrt(k_obs * D_CO2) / kl  (Hatta number)
```

For instantaneous regime:
```
E_inf = 1 + D_amine/D_CO2 * [Amine]/([CO2]_i * ν)
```

**Correlations for structured packing:**
```
kg = C1 * (Re_g)^a * (Sc_g)^(1/3) * (D_CO2/d_h)
kl = C2 * (Re_l)^b * (Sc_l)^(1/2) * (D_CO2/δ)
```

**References:**
- Danckwerts PV (1970). Gas-Liquid Reactions
- Rocha JA et al. (1996). IEC Research - structured packing correlations

---

## 4. Unit Operations (`units/`)

### 4.1 Amine Absorber (`absorber.py`)

**Simplified Model (Equilibrium-stage):**

Number of theoretical stages from Kremser equation:
```
N = ln[(y_in - y_out*)/(y_out - y_out*)] / ln(A)

where A = L*H / (G*P) (absorption factor)
```

**Parameters:**
```python
@dataclass
class AbsorberParams:
    n_stages: int | float          # Number of stages (can be fractional for NTU)
    solvent: str                    # Solvent name from database
    solvent_conc: float            # Amine concentration (wt% or mol/L)
    L_G_ratio: float               # Liquid/gas molar ratio
    pressure: float                # Operating pressure (Pa)
    T_gas_in: float                # Inlet gas temperature (K)
    T_liquid_in: float             # Inlet liquid temperature (K)
    stage_efficiency: float = 0.7  # Murphree efficiency

    # Extensibility for rate-based
    column_diameter: float | None = None
    packing_type: str | None = None
    packing_height: float | None = None
```

**Outputs:**
- Treated gas stream (CO2 depleted)
- Rich solvent stream (CO2 loaded)
- Stage-by-stage profiles
- CO2 capture efficiency

### 4.2 Amine Stripper (`stripper.py`)

**Simplified Model:**

Energy balance:
```
Q_reboiler = Q_sensible + Q_vaporization + Q_reaction

Q_sensible = m_rich * Cp * (T_reboiler - T_rich)
Q_vaporization = m_steam * ΔH_vap
Q_reaction = m_CO2 * ΔH_absorption
```

Regeneration is typically at 120-140°C under ~2 bar.

**Parameters:**
```python
@dataclass
class StripperParams:
    n_stages: int | float
    solvent: str
    T_reboiler: float              # Reboiler temperature (K)
    P_stripper: float              # Stripper pressure (Pa)
    reflux_ratio: float = 0.3     # Condenser reflux
    reboiler_duty: float | None = None  # If specified, overrides T calc
```

**Outputs:**
- Lean solvent stream
- CO2 product stream
- Reboiler duty
- Specific regeneration energy (GJ/tonne CO2)

### 4.3 Membrane Separator (`membrane.py`)

**Solution-Diffusion Model:**

Flux through membrane:
```
J_i = P_i * (p_i,feed - p_i,permeate) / δ

where:
  P_i = Permeability (Barrer)
  δ = membrane thickness
  p_i = partial pressure of component i
```

For multi-component:
```
Selectivity: α_ij = P_i / P_j
Stage cut: θ = F_permeate / F_feed
```

**Parameters:**
```python
@dataclass
class MembraneParams:
    membrane_type: str             # From database
    area: float                    # Membrane area (m²)
    thickness: float | None = None # Override default (μm)
    n_stages: int = 1              # Multi-stage configuration
    pressure_ratio: float = 10.0  # P_feed / P_permeate
    T: float = 298.15             # Temperature (K)

    # For mixed-matrix membranes
    filler_loading: float = 0.0   # Volume fraction
```

**Models:**
1. **Single-stage**: Direct flux calculation
2. **Multi-stage**: Cascade with recycle options
3. **Counter-current**: More rigorous (extensibility hook)

**References:**
- Baker RW (2012). Membrane Technology and Applications, 3rd ed.
- Robeson LM (2008). J Membr Sci - Upper bound correlation

### 4.4 Adsorption Cycles (`adsorption.py`)

**PSA (Pressure Swing Adsorption):**

Simplified equilibrium model:
```
Working capacity: Δq = q(P_ads) - q(P_des)
CO2 produced per cycle: m_CO2 = Δq * m_adsorbent
Cycle time: t_cycle = t_ads + t_blowdown + t_purge + t_repressure
```

**TSA (Temperature Swing Adsorption):**
```
Working capacity: Δq = q(T_ads) - q(T_des)
Energy for heating: Q = m_ads * Cp_ads * (T_des - T_ads)
```

**VSA (Vacuum Swing Adsorption):**
Same as PSA but desorption at sub-atmospheric pressure.

**TVSA (Combined):**
Both temperature and pressure swing.

**Parameters:**
```python
@dataclass
class AdsorptionParams:
    adsorbent: str                 # From database
    cycle_type: str                # 'PSA', 'TSA', 'VSA', 'TVSA'
    bed_mass: float                # Adsorbent mass (kg)
    bed_void_fraction: float = 0.4

    # PSA/VSA specific
    P_adsorption: float = 101325   # Adsorption pressure (Pa)
    P_desorption: float = 10000    # Desorption pressure (Pa)

    # TSA specific
    T_adsorption: float = 298.15   # Adsorption temperature (K)
    T_desorption: float = 393.15   # Desorption temperature (K)

    # Cycle parameters
    t_adsorption: float = 300      # Adsorption step time (s)
    t_desorption: float = 300      # Desorption step time (s)

    # Multi-bed
    n_beds: int = 2                # Number of beds for continuous operation
```

**Outputs:**
- CO2 purity
- CO2 recovery
- Productivity (mol CO2/kg·h)
- Energy consumption (kWh/tonne CO2)
- Breakthrough curves (extensibility)

**References:**
- Ruthven DM et al. (1994). Pressure Swing Adsorption
- Webley PA (2014). Adsorption 20:225 - Energy analysis

---

## 5. Flowsheet Templates (`flowsheets/`)

### 5.1 Amine Capture Loop (`amine_loop.py`)

Complete absorber-stripper loop with:
- Heat integration (cross-exchanger)
- Solvent make-up
- Blower for flue gas
- CO2 compressor

### 5.2 Hybrid Systems (`hybrid.py`)

Membrane + amine configurations:
- Membrane pre-concentration + amine polishing
- Membrane for bulk removal, amine for deep removal

---

## 6. Implementation Plan

### Phase 1: Database and Core Infrastructure
1. Create data YAML files with validated property data
2. Implement database.py with data classes and loaders
3. Add ParamsMixin support to all parameter classes

### Phase 2: Equilibrium Models
1. Implement Langmuir, Sips, Toth, DSL isotherms
2. Implement Henry's law and Kent-Eisenberg VLE
3. Add temperature dependence to all models

### Phase 3: Unit Operations (Simplified)
1. Implement absorber with Kremser equation
2. Implement stripper with energy balance
3. Implement membrane with solution-diffusion
4. Implement PSA/TSA/VSA/TVSA equilibrium models

### Phase 4: Plugin Integration
1. Add __init__.py with exports
2. Add register() function for difflow plugin system
3. Update pyproject.toml entry points

### Phase 5: Testing
1. Unit tests for each isotherm model
2. Unit tests for each unit operation
3. Integration tests for flowsheets
4. Gradient verification tests

### Phase 6: Extensibility Hooks (Future)
1. Rate-based absorber/stripper columns
2. Dynamic adsorption breakthrough models
3. Counter-current membrane models
4. eNRTL thermodynamic model

---

## Key Design Decisions

1. **Simplified first**: Start with equilibrium-based models that are fast and differentiable.
   Rate-based models can be added later via subclassing.

2. **Database-driven**: All material properties in YAML files for easy updates without code changes.

3. **JAX-native**: All numerical operations use `jax.numpy`. No Python control flow in hot paths.

4. **Consistent interface**: All units follow `(inlet, **kwargs) -> (outlet, info)` pattern.

5. **Extensible parameters**: Use dataclasses with `update()` method for JAX-compatible modifications.

6. **Literature-backed**: All model parameters from peer-reviewed sources with citations in docstrings.
