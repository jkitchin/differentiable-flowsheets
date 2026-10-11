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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class CSTRParams:
    V: float               # Reactor volume (m³)
    rate_fn: Callable      # Rate function: rate_fn(C, T, rate_params) -> r  [n_reactions]
    stoich: Array          # Stoichiometric matrix [n_species × n_reactions]
    rate_params: dict      # Parameters passed to rate_fn (k_ref, E_a, ...)
    species_order: list    # Species names, in the row order of `stoich`
    dH_rxn: Array = None   # Heat of reaction for each reaction (J/mol)
    molar_density: float = None  # Constant molar density (mol/m³), see below
    eos: Any = None        # Cubic EOS for the reaction-phase density
    reaction_phase: str = None   # 'liquid' or 'vapor' (required with eos)
    T_damping: float = ... # Damping on the adiabatic/duty temperature solve
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
| `info['conversion']` | dict | - | Conversion of each species |

#### Operating Modes

The mode is chosen when the unit is built (`CSTR(params, thermo, mode=...)`):

1. **Isothermal** (`mode='isothermal'`, `T_spec` given at call time): Outlet temperature is fixed, heat duty calculated
2. **Adiabatic** (`mode='adiabatic'`): Q = 0, outlet temperature calculated
3. **Specified Duty** (`mode='specified_duty'`, `Q_spec` given at call time): Heat duty fixed, outlet temperature calculated

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
from difflow import CSTR, CSTRParams, IdealThermo, SpeciesData, make_stream
import jax.numpy as jnp

# Two-species thermodynamics (liquid Cp, Antoine and Hvap constants)
species_data = {
    n: SpeciesData(n, MW=100.0, Cp_coeffs=(150.0, 0.0, 0.0, 0.0),
                   Hvap_coeffs=(35000.0, 0.38, 600.0),
                   antoine_coeffs=(10.0, 3000.0, -50.0))
    for n in ('A', 'B')
}
thermo = IdealThermo(species_data)

# A -> B (first-order, exothermic)
def rate_fn(C, T, p):
    k = p['k_ref'] * jnp.exp(p['E_a'] / 8.314 * (1 / p['T_ref'] - 1 / T))
    return jnp.array([k * C['A']])

params = CSTRParams(
    V=2.0,  # m³
    rate_fn=rate_fn,
    stoich=jnp.array([[-1.0], [1.0]]),  # A -> B
    rate_params={'k_ref': 1e-3, 'E_a': 50000.0, 'T_ref': 350.0},  # 1/s at 350 K
    species_order=['A', 'B'],
    dH_rxn=jnp.array([-80000.0]),  # J/mol (exothermic)
    molar_density=1000.0,  # mol/m³
)

inlet = make_stream({'A': 1.0, 'B': 0.0}, T=350.0, P=101325.0)

# Isothermal operation: the mode is set on the unit
cstr = CSTR(params, thermo, mode='isothermal')
outlet, info = cstr(inlet, T_spec=350.0)
print(f"Conversion of A: {info['conversion']['A']:.2%}")
print(f"Heat duty: {info['Q']:.2f} W")

# Adiabatic operation
cstr_adiab = CSTR(params, thermo, mode='adiabatic')
outlet_adiab, info_adiab = cstr_adiab(inlet)
print(f"Outlet temperature: {outlet_adiab['T']:.1f} K")
```

#### Concentration Basis (Molar Density)

The rate law is evaluated at concentrations, so the reactor needs a molar
density: $C_i = F_i / \dot{V}$ with $\dot{V} = F_{total}/\rho$, which makes the
residence time $\tau = V\rho/F_{total}$. An error in $\rho$ is a proportional
error in $\tau$, and so in the conversion. Three ways to set it, in the order
the CSTR resolves them:

1. **An equation of state** -- `eos=<cubic EOS>` with
   `reaction_phase='liquid'` or `'vapor'`. Concentration is then the real
   molarity at reactor $(T, P, y)$, $C_i = y_i\,\rho_{EOS}(T, P, y)$,
   recomputed inside the solve, so the reactor shares the flash's
   thermodynamics. `reaction_phase` is required with `eos`: the liquid and
   vapor molar densities differ by two orders of magnitude, so there is no
   defensible default. Pair it with a `CubicThermo` to make the *enthalpy*
   real-gas too; a `CubicThermo` passed as `thermo` also supplies the EOS
   itself when `reaction_phase` is set and no `eos=` is given.
2. **A constant** -- `molar_density=<mol/m^3>`.
3. **Neither**, in which case the reactor falls back to 55500 mol/m^3 (liquid
   water) and raises a `CSTRDensityWarning`. That fallback is right only for
   aqueous systems: it is ~8x high for a C3-C8 hydrocarbon liquid and ~200x
   high for a gas, and it silently inflates residence time. Treat the warning
   as a request to say which basis you meant.

Passing `volumetric_flow=` to the call sets $\dot{V}$ outright and bypasses all
three; `info['molar_density']` then reports the density that flow implies.

```python
from difflow.eos import PengRobinson
from difflow.database import get_critical_props

species = ['propane', 'n-butane']
eos = PengRobinson({c: get_critical_props(c) for c in species})
params_eos = CSTRParams(V=2.0, rate_fn=lambda C, T, p: jnp.array([p['k'] * C['propane']]),
                        stoich=jnp.array([[-1.0], [1.0]]), rate_params={'k': 1e-3},
                        species_order=species, eos=eos, reaction_phase='liquid')
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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class PFRParams:
    V: float               # Total reactor volume (m³)
    rate_fn: Callable      # Rate function: rate_fn(C, T, rate_params) -> r
    stoich: Array          # Stoichiometry matrix [n_species × n_reactions]
    rate_params: dict      # Parameters passed to rate_fn
    species_order: list    # List of species names
    dH_rxn: Array = None   # Heat of reaction (J/mol), required for non-isothermal
    dP_dV: float = None    # Pressure gradient along the reactor (Pa/m³), optional
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
| `info['conversion']` | dict | - | Conversion of each species |
| `info['profiles']['V']` | Array | m³ | Volume along reactor |
| `info['profiles']['F']` | Array | mol/s | Molar flows along reactor, `[n_save_points, n_species]` |
| `info['profiles']['T']` | Array | K | Temperature along reactor |

#### Governing Equations

**Material Balance** (differential):

$$\frac{dF_i}{dV} = \sum_j \nu_{ij} r_j$$

**Energy Balance** (adiabatic):

$$\frac{dT}{dV} = \frac{-\sum_j r_j \Delta H_{rxn,j}}{F_{total} C_{p,mix}}$$

**Integration Method**: Adaptive ODE integration using diffrax (Tsit5 or Dopri5 solvers)

#### Example Usage

```python
from difflow.units.pfr import PFR, PFRParams
from difflow import make_stream
import jax.numpy as jnp

# Define rate function: A -> B (first-order)
def pfr_rate_fn(C, T, params):
    """Rate function: r = k * C_A with Arrhenius temperature dependence."""
    k_ref, E_a, T_ref = params['k_ref'], params['E_a'], params['T_ref']
    R = 8.314
    k = k_ref * jnp.exp(E_a / R * (1/T_ref - 1/T))
    return jnp.array([k * C['A']])

pfr_params = PFRParams(
    V=0.005,
    rate_fn=pfr_rate_fn,
    stoich=jnp.array([[-1.0], [1.0]]),  # A -> B
    rate_params={'k_ref': 0.5, 'E_a': 60000.0, 'T_ref': 400.0},
    species_order=['A', 'B'],
    dH_rxn=jnp.array([-50000.0]),
    n_save_points=201
)

pfr = PFR(pfr_params, thermo)
pfr_inlet = make_stream({'A': 2.0, 'B': 0.0}, T=400.0, P=200000.0)
outlet, info = pfr(pfr_inlet, volumetric_flow=0.001)  # m³/s

# Plot conversion profile
import matplotlib.pyplot as plt
prof = info['profiles']
plt.plot(prof['V'], 1 - prof['F'][:, 0] / pfr_inlet['F_A'])
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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class GasPFRParams(PFRParams):
    alpha: float           # Pressure drop parameter (Pa/m³); or give the physical
                           # bed instead: d_p, voidage, mu, rho_gas, u_s0, bed_area
```

Instead of a lumped `alpha`, pass `d_p`, `voidage`, `mu`, `rho_gas`,
`bed_area` (and optionally `u_s0` and `sphericity`) and $\alpha$ is computed from the full
Ergun equation; see [Particles and Beds](particles-ergun). Leave `u_s0=None` to
take the superficial velocity from the inlet stream ($Q_0/A_c$); a given `u_s0`
that differs from the stream's by more than 1 % raises `GasPFRBedWarning`.

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
| `info['profiles']['P']` | Array | Pa | Pressure along reactor |
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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
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
from difflow.units.fed_batch import (
    FedBatchParams, batch_time_for_conversion, optimal_feed_profile,
)

fb_params = FedBatchParams(
    V0=1.0,
    rate_fn=lambda C, T, p: jnp.array([p['k'] * C['A']]),
    stoich=jnp.array([[-1.0], [1.0]]),     # A -> B
    rate_params={'k': 1e-3},
    species_order=['A', 'B'],
)
C0 = {'A': 1000.0, 'B': 0.0}               # mol/m³

# Calculate batch time for target conversion (here 95 % of A)
t_batch = batch_time_for_conversion(fb_params, C0, 350.0, 'A', 0.95)

# Generate an optimal piecewise-constant feed profile (maximize product B)
feed_fn, t_opt = optimal_feed_profile(
    'max_yield', fb_params, C0, 350.0, 'B',
    feed_composition={'A': 2000.0, 'B': 0.0}, V_max=2.0, t_max=3600.0,
    n_intervals=4, n_sim_steps=50,
)
```

---

### SemiBatchReactor

**Location**: `difflow/units/fed_batch.py`

**Class**: `SemiBatchReactor`

**Description**: `FedBatchReactor` under the name the process usually goes by. Same model, same parameters, same equations.

A semi-batch reactor is a batch vessel into which one reactant is fed
gradually --- to cap the heat release, to hold a reactant concentration
low for selectivity, or to keep a hazardous intermediate from
accumulating. That is exactly the fed-batch model above, so
`SemiBatchReactor` is a subclass of `FedBatchReactor` that changes
nothing but its display symbol:

```python
from difflow.units.fed_batch import SemiBatchReactor, FedBatchParams

reactor = SemiBatchReactor(fb_params, thermo, mode="isothermal")   # fb_params: FedBatchParams
```

Everything in [FedBatchReactor](#fedbatchreactor) --- parameters, feed
and temperature profiles, the material and energy balances, the utility
functions --- applies unchanged. Pick the name that makes the flowsheet
read correctly; there is no modelling difference to weigh.

---

### StoichiometricReactor

**Location**: `difflow/units/stoichiometric.py`

**Class**: `StoichiometricReactor` (parameters: `StoichiometricReactorParams`)

**Description**: Fixed-conversion reactor (a simulator's `RStoic` block): the
conversion $X$ of a key reactant $k$ is specified and the extent follows. No
kinetics, equilibrium or thermodynamic model; differentiable in the conversion
and the feed. Several reactions (lists of `stoich`, `key`, `conversion`) run in
parallel, each conversion based on the reactor inlet.

#### Governing Equations

$$\xi_r = \frac{X_r F^{in}_{k_r}}{-\nu_{k_r,r}}, \qquad F^{out}_i = F^{in}_i + \sum_r \nu_{i,r}\, \xi_r$$

The outlet is at the inlet temperature (or `T_out`) and the inlet pressure less
`pressure_drop`; no energy balance is solved. Products absent from the inlet are
added to the outlet. `info` reports `extent`, `conversion`, `key_consumed` and
`min_outlet_flow` (negative when the specified conversions consume more of a
species than is fed).

#### Example Usage

```python
import jax
from difflow import StoichiometricReactor, StoichiometricReactorParams, make_stream

feed = make_stream({"CO": 100.0, "H2": 200.0, "MeOH": 0.0}, T=500.0, P=5e6)

def methanol_out(X):
    rx = StoichiometricReactor(StoichiometricReactorParams(
        stoich={"CO": -1.0, "H2": -2.0, "MeOH": 1.0}, key="CO", conversion=X))
    out, info = rx(feed)
    return out["F_MeOH"]

print(float(methanol_out(0.25)), float(jax.grad(methanol_out)(0.25)))   # 25.0, 100.0
```

---

(separators)=
## Declarative Kinetics

The reactors above take `rate_fn` as a Python callable. That is expressive, but it is *code*, not data — it cannot be written to a file, built from a form, or round-tripped through a GUI. `mass_action_kinetics` builds the callable from plain dictionaries instead, so a reaction network can be stored, edited and shared as data.

```python
from difflow import CSTR, CSTRParams, mass_action_kinetics

reactions = [{
    "equation":  "A -> B",
    "reactants": {"A": 1.0},
    "products":  {"B": 1.0},
    "rate_params": {"A": 1.0e6, "Ea": 50_000.0, "n": 0.0},
}]

kin = mass_action_kinetics(reactions, species_order=["A", "B"])
cstr = CSTR(CSTRParams(V=1.5, **kin.params_kwargs()))
```

`params_kwargs()` supplies `rate_fn`, `stoich`, `rate_params` and `species_order` — every rate-law field the reactors need. The result is numerically identical to the equivalent hand-written callable.

The dictionary format is exactly what [`import_reactions`](thermodynamics.md) returns from a Cantera YAML file, so a published mechanism goes straight into a reactor:

<!-- doc-test: skip: needs a Cantera mechanism file (mech.yaml) on disk -->
```python
from difflow import import_reactions

reactions = import_reactions("mech.yaml")
kin = mass_action_kinetics(reactions, reverse="forward_only")
```

### The rate law

$$k_j(T) = A_j \, T^{n_j} \exp\!\left(\frac{-E_{a,j}}{R T}\right), \qquad r_j = k_j \prod_i C_i^{\alpha_{ji}}$$

Orders $\alpha$ come from the reactant stoichiometry unless given explicitly via `orders=`, which covers empirical rate laws where the order is not the coefficient. A reversible reaction subtracts the reverse term scaled by its equilibrium constant:

$$r_j = k_j \left( \prod_i C_i^{\alpha_{ji}} - \frac{1}{K_{eq,j}} \prod_i C_i^{\beta_{ji}} \right)$$

**Parameters:**
- `reactions` — one dict per reaction with `reactants`, `products` and `rate_params` (`A`, `Ea`, `n`); optionally `equation`, `reversible`, `type` and `K_eq`.
- `species_order` — fixes the rows of `stoich`; defaults to the sorted union of every species mentioned.
- `reverse` — `"error"` (default), `"forward_only"`, or `"equilibrium"`.
- `orders` — per-reaction `{species: order}` overrides. `None` keeps stoichiometric orders; `{}` means zeroth order in everything.

### What it refuses, and why

Two specifications raise `KineticsSpecError` rather than being approximated, because in both cases a guess produces a plausible number that is wrong:

- **Reversible reactions, by default.** Forward Arrhenius parameters alone do not determine the reverse rate. Pass `reverse="equilibrium"` with a `K_eq` on each reaction, or `reverse="forward_only"` to drop the reverse term as an explicit, recorded approximation.
- **Three-body, falloff and other pressure-dependent types.** These need their own rate law; mass action covers elementary reactions only.

### Units

Concentrations mol/m³, temperature K, activation energy J/mol, rates mol/m³/s. Cantera files declare their own units in a `units:` block — a mechanism written in cm³ or kcal/mol imports numerically unchanged and will be **silently wrong**, so check that block before trusting a rate constant.

---

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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
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
from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.flash import Flash, FlashParams

thermo = IdealThermo({n: get_species_data(n) for n in ['methane', 'ethane', 'propane']})
flash = Flash(FlashParams(species_order=['methane', 'ethane', 'propane']), thermo)
feed = make_stream({'methane': 0.5, 'ethane': 0.3, 'propane': 0.2}, T=200.0, P=500000.0)

liquid, vapor, info = flash(feed)
print(f"Vapor fraction: {info['V_frac']:.3f}")
```

##### Flash with an activity model (non-ideal liquid)

With `activity_model=` the K-values follow the modified Raoult's law
$K_i = \gamma_i(x,T)\,P_i^{sat}(T)/P$, which captures azeotropes. The argument is
any **activity model**: an object with a `gamma(x, T)` method (the
`difflow.activity.ActivityModel` protocol). `NRTLParams`, `UNIQUACParams`,
`WilsonParams`, `MargulesParams` and `VanLaarParams` all qualify (Margules and
van Laar are binary; Wilson cannot predict liquid-liquid splitting). `species`
of the model must match `species_order`. The NRTL path is unchanged. See
[Activity-Coefficient Models](thermodynamics.md#activity-coefficient-models) and,
for Txy/Pxy/xy diagrams, [Phase Diagrams](thermodynamics.md#phase-diagrams).

```python
from difflow import WilsonParams

wilson = WilsonParams.binary(("methanol", "water"), [40.73e-6, 18.07e-6], 667.0, 1981.0)
th = IdealThermo({n: get_species_data(n) for n in ("methanol", "water")})
f = Flash(FlashParams(species_order=["methanol", "water"]), th, activity_model=wilson)
feed = make_stream({"methanol": 0.5, "water": 0.5}, T=345.0, P=101325.0)
liquid, vapor, info = f(feed)
print(f"Vapor fraction: {float(info['V_frac']):.3f}")
```

(eosflash)=
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

(phflash)=
##### PHFlash (Isenthalpic)

Performs adiabatic flash at constant pressure and enthalpy. Solves for flash temperature.

```python
from difflow.units.flash import PHFlash, FlashParams

thermo_lh = IdealThermo({n: get_species_data(n) for n in ['n_pentane', 'n_heptane']})
ph_flash = PHFlash(FlashParams(species_order=['n_pentane', 'n_heptane']), thermo_lh)

# Hot liquid feed, flash to lower pressure
feed = make_stream({'n_pentane': 50.0, 'n_heptane': 50.0}, T=380.0, P=101325.0)
liquid, vapor, info = ph_flash(feed, P=30000.0)

print(f"Flash temperature: {info['T_flash']:.1f} K")
print(f"Vapor fraction: {info['V_frac']:.3f}")
```

#### Bubble and Dew Point Methods

The `Flash` class provides methods for calculating phase boundaries:

```python
flash = Flash(FlashParams(species_order=['n_pentane', 'n_heptane']), thermo_lh)
feed = make_stream({'n_pentane': 50.0, 'n_heptane': 50.0}, T=350.0, P=50000.0)

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

**Energy Balance** (adiabatic mixing). With a `thermo` model the outlet
temperature solves the enthalpy balance exactly,

$$\sum_i F_{out,i}\, h_i(T_{out}) = \sum_k \sum_i F_{k,i}\, h_i(T_k),$$

by Newton's method on $T_{out}$ started at the flow-weighted mean temperature
(the phase is fixed, so $H$ is smooth), with a bracketed bisection as the
fallback; `jax.grad` differentiates the converged root by the implicit function
theorem. Starting at the mean picks the physical root when the enthalpy is not
monotone: `IdealThermo`'s vapor branch adds a Watson heat of vaporization that
vanishes at $T_c$, so near a species' critical temperature $H_{vap}(T)$ can fall
with $T$ and the balance has a second, spurious root. Without `thermo` the outlet
is at the flow-weighted mean temperature, which is the same balance for equal,
constant heat capacities:

$$T_{out} = \frac{\sum_k F_k T_k}{\sum_k F_k}$$

(Before this was solved exactly, the thermo path evaluated one $c_p$ at the
mean inlet temperature: 1 mol/s of methane at 300 K mixed with 10 mol/s at
800 K came out at 752.7 K instead of 763.6 K.)

**Pressure**: the lowest inlet pressure.

**Missing species**: an inlet that does not carry a species in the mixer's
list contributes zero flow of it, as in `combine_streams`.

#### Example Usage

```python
from difflow.units.flash import Mixer

mixer = Mixer(['A', 'B', 'C'])   # optional: thermo=..., phase='liquid'
stream1 = make_stream({'A': 1.0, 'B': 0.5, 'C': 0.0}, T=350.0, P=101325.0)
stream2 = make_stream({'A': 0.0, 'B': 0.3, 'C': 0.2}, T=360.0, P=101325.0)

# Streams are passed as separate arguments; a species an inlet omits is zero
outlet, info = mixer(stream1, stream2)
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

splitter = Splitter(species_order=['A', 'B'])
inlet = make_stream({'A': 1.0, 'B': 0.5}, T=350.0, P=101325.0)

# Split into 3 streams: 50%, 30%, 20%
out1, out2, out3, info = splitter(inlet, split_frac=[0.5, 0.3, 0.2])
```

---

### ComponentSplitter

**Location**: `difflow/units/flash.py`

**Class**: `ComponentSplitter` (parameters: `ComponentSplitterParams`)

**Description**: Ideal component splitter: a specified fraction $\phi_i$ of each
species goes to the first outlet and the rest to the second. The block-flow
separator of a material-balance calculation ("all the methanol to the liquid,
none of the gas"); no equilibrium and no thermodynamic model, so it is
differentiable in the fractions and the feed. Unlike `Splitter`, each species has
its own split. (`ComponentSeparator` is the same balance with an enthalpy duty and
needs a `CubicThermo`.)

#### Governing Equations

$$F^{(1)}_i = \phi_i F^{in}_i, \qquad F^{(2)}_i = (1 - \phi_i) F^{in}_i$$

Both outlets leave at the inlet temperature and pressure. Species missing from
`split_fractions` use `default_fraction` (default 0: all to the second outlet).

#### Example Usage

```python
from difflow import ComponentSplitter, ComponentSplitterParams, make_stream

sep = ComponentSplitter(ComponentSplitterParams(
    split_fractions={"MeOH": 0.0, "H2O": 0.05}, default_fraction=1.0))
feed = make_stream({"CO": 10.0, "H2": 20.0, "MeOH": 5.0, "H2O": 1.0}, T=320.0, P=5e6)
gas, liquid, info = sep(feed)      # outlets in order: first, second
print(float(gas["F_CO"]), float(liquid["F_MeOH"]), float(gas["F_H2O"]))
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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class ShortcutColumnParams:
    species_order: list[str]   # Species names; sets array ordering
    light_key: str             # Name of the light key component
    heavy_key: str             # Name of the heavy key component
    x_D_LK: float = 0.99       # Fractional recovery of LK in the distillate
    x_B_HK: float = 0.99       # Fractional recovery of HK in the bottoms
```

`x_D_LK` and `x_B_HK` are **recoveries**, not mole fractions, despite the
names: `x_D_LK=0.99` sends 99 % of the feed's light key overhead. The
distillate's actual light-key mole fraction comes back in `info["x_D"]`.

The reflux ratio is not a parameter — it is an argument to the call, along with
the column pressure and the feed quality:

<!-- doc-test: skip: call fragment; the column and feed are built in Example Usage below -->
```python
distillate, bottoms, info = column(feed, R=3.0, P=101325.0, q=1.0)
```

#### Governing Equations

**Relative Volatility**:

$$\alpha_{ij} = \frac{K_i}{K_j}$$

The second equality usually written here, $\alpha_{ij} = P_i^{sat}/P_j^{sat}$,
holds only under Raoult's law. On a `CubicThermo` the K-values are fugacity
coefficient ratios and the vapor pressures cancel out of nothing — which is the
whole reason the volatilities differ between the two packages.

**Average Relative Volatility** (geometric mean):

$$\bar{\alpha} = (\alpha_{top} \cdot \alpha_{bottom})^{0.5}$$

The two ends are the column's actual ends, not estimates around the feed: the
top is a total condenser, so $T_{top}$ is the **bubble point of the
distillate**, and $T_{bot}$ is the **bubble point of the bottoms** in the
reboiler. Those are the temperatures the product streams come out at, reported
as `info["T_condenser"]` and `info["T_reboiler"]`.

They also make the design a small fixed point, since $\bar\alpha$ sets the
product split through Hengstebeck-Geddes and the split sets the bubble points
in turn. The column sweeps it a few times from the feed's own bubble point;
it settles to under a hundredth of a degree by the third sweep, and the whole
loop is unrolled, so `jax.grad` runs through it.

**Non-key distribution (Hengstebeck-Geddes).** The keys go where their
recoveries put them; every other species follows the straight line in
$(\log\alpha, \log d/b)$ through both keys:

$$\log\frac{d_i}{b_i} = A + C\log\alpha_i,\qquad
A = \log\left(\frac{d}{b}\right)_{HK},\qquad
C = \frac{\log(d/b)_{LK} - \log(d/b)_{HK}}{\log\alpha_{LK}}$$

with $\alpha$ relative to the heavy key, so $\alpha_{HK} = 1$ (Geddes, *AIChE
J.* 4, 389 (1958); Hengstebeck, *Distillation*, Reinhold (1961); page and
equation numbers unverified). The distillate share $d_i/(d_i + b_i)$ is the
logistic function of that line, so each species' balance closes exactly.
Before this was corrected the constants were $A = \log(d/b)_{LK} -
\log(d/b)_{HK}$ and $C = \log(d/b)_{LK}/\log\alpha_{LK}$, a line that misses
the heavy key: on a propane/isobutane depropanizer it sent 99.9 % of the
n-butane overhead. `tests/test_distillation.py::TestShortcutColumnNonKeyDistribution`
pins the line through both keys.

Two consequences worth knowing, because the old estimate had neither: the end
temperatures no longer move when you feed the same mixture in hotter, and they
do move with column pressure.

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

| Key | Type | Units | Description |
|-----|------|-------|-------------|
| `distillate` | Stream | - | Overhead product, at the condenser temperature |
| `bottoms` | Stream | - | Bottom product, at the reboiler temperature |
| `info['N_min']` | float | - | Minimum stages (Fenske) |
| `info['R_min']` | float | - | Minimum reflux ratio (Underwood) |
| `info['N']` | float | - | Actual stages (Gilliland) |
| `info['N_feed']` | float | - | Feed stage (Kirkbride) |
| `info['D']`, `info['B']` | float | mol/s | Distillate and bottoms flow |
| `info['x_D']`, `info['x_B']` | dict | - | Product compositions by species |
| `info['T_condenser']` / `info['T_top']` | float | K | Condenser temperature = bubble point of $x_D$ |
| `info['T_reboiler']` / `info['T_bot']` | float | K | Reboiler temperature = bubble point of $x_B$ |
| `info['Q_condenser']` | float | W | Condenser duty (negative: heat removed) |
| `info['Q_reboiler']` | float | W | Reboiler duty (positive: heat added) |
| `info['alpha']` | dict | - | Relative volatilities vs the heavy key |
| `info['alpha_LK']` | float | - | Light key's relative volatility |
| `info['alpha_top']`, `info['alpha_bot']` | dict | - | Volatilities at each column end |
| `info['alpha_variation']` | dict | - | Relative spread between the two ends |
| `info['alpha_varies_significantly']` | bool | - | True if that spread exceeds 0.3 |
| `info['theta']` | float | - | Underwood root |
| `info['close_boiling']` | bool | - | True if $\bar\alpha \approx 1$ capped $N_{min}$ |
| `info['near_min_reflux']` | bool | - | True if $R \approx R_{min}$ |
| `info['negative_flows_detected']` | bool | - | True if the split produced a negative flow |
| `info['feasible']` | bool | - | All three checks above passed |

#### Example Usage

```python
from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams

names = ['benzene', 'toluene', 'ethylbenzene']
thermo = IdealThermo({s: get_species_data(s) for s in names})

params = ShortcutColumnParams(
    species_order=names,
    light_key='benzene',
    heavy_key='toluene',
    x_D_LK=0.99,    # 99 % of the benzene recovered overhead
    x_B_HK=0.99,    # 99 % of the toluene recovered in the bottoms
)

column = ShortcutColumn(params, thermo)
feed = make_stream({'benzene': 40.0, 'toluene': 35.0, 'ethylbenzene': 25.0},
                   T=370.0, P=101325.0)

distillate, bottoms, info = column(feed, R=3.0, P=101325.0, q=1.0)
print(f"Minimum stages:  {float(info['N_min']):.1f}")
print(f"Minimum reflux:  {float(info['R_min']):.2f}")
print(f"Actual stages:   {float(info['N']):.1f}")
print(f"Feed stage:      {float(info['N_feed']):.1f}")
print(f"Condenser:       {float(info['T_condenser']):.1f} K, "
      f"{float(info['Q_condenser'])/1e6:.2f} MW")
print(f"Reboiler:        {float(info['T_reboiler']):.1f} K, "
      f"{float(info['Q_reboiler'])/1e6:.2f} MW")
```

#### Utility Functions

The design correlations are also available on their own, taking plain numbers
rather than a column object. Note the argument order — each takes the
quantities in the order the correlation is written, not the order the column
computes them:

```python
from difflow.units.distillation import (
    fenske_stages,        # (x_D_LK, x_B_LK, alpha)
    minimum_reflux_ratio, # (z_LK, z_HK, x_D_LK, alpha, q=1.0)
    gilliland_stages,     # (R, R_min, N_min)
    column_diameter,      # DEPRECATED, see Column Hardware below
)

N_min = fenske_stages(x_D_LK=0.98, x_B_LK=0.02, alpha=2.4)
R_min = minimum_reflux_ratio(z_LK=0.45, z_HK=0.55, x_D_LK=0.98, alpha=2.4, q=1.0)
N = gilliland_stages(R=1.3 * R_min, R_min=R_min, N_min=N_min)

from difflow.units.column_design import tray_column_diameter
# vapor/liquid rates (mol/s), MW (g/mol), densities (kg/m^3), sigma (N/m)
diameter, _ = tray_column_diameter(V_molar=50.0, MW_V=80.0, rho_V=3.0,
                                   L_molar=50.0, MW_L=85.0, rho_L=800.0, sigma=0.02)
```

Relative volatility is a method on the column, not a free function
(`column.relative_volatility(T, P, x=None)`), because it needs the key
components from the column's parameters — and, on a `CubicThermo`, the phase
composition.

`minimum_reflux_ratio` honours the feed quality `q` (1 saturated liquid, 0
saturated vapor, above 1 subcooled): since #424 it is the McCabe-Thiele pinch
of a constant-alpha curve, computed by `difflow.staged.minimum_reflux`. It
used to accept `q` and ignore it, and to floor the answer at 0.1; neither is
true now (a result below 0.1, or below 0 for an `x_D` the curve cannot reach,
is returned as is). `z_HK` is unused: the feed is `1 - z_LK`. The column
itself (`ShortcutColumn`) uses the Underwood root directly and was not
affected. For a curved equilibrium line (tangent pinch) use
`difflow.staged.minimum_reflux` with a tabulated curve, see
[Graphical Staged Design](#graphical-staged-design).

---

### DistillationColumn (Rigorous)

**Location**: `difflow/units/distillation.py`

**Class**: `DistillationColumn`

**Description**: Stage-by-stage calculation using MESH equations (Material balance, Equilibrium, Summation, enthalpy balance).

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class DistillationColumnParams:
    species_order: list[str]     # Species names; sets array ordering
    n_stages: int                # Equilibrium stages: the trays plus the reboiler
    feed_stage: int              # Zero-based stage index from the bottom
    condenser_type: str = 'total'  # 'total' or 'partial' (an extra equilibrium stage)
    P: float = 101325.0          # One pressure for the whole column
    q: float = 1.0               # Feed quality (1 = saturated liquid)
    stage_efficiency: float | Array = 1.0  # Murphree vapour efficiency, scalar or (n_stages,)
```

**Stage numbering is zero-based from the bottom**: stage 0 is the reboiler,
stage `n_stages - 1` is the top tray, and every profile in `info` is in that
order. The condenser is *not* a stage — it sits outside the cascade — so
`n_stages` counts the trays plus the reboiler, and `feed_stage=10` in a
20-stage column puts the feed halfway up. Stages below the feed stage are the
stripping section and stages above it the rectifying section. There is one
column pressure: no tray pressure drop.

The feed stage itself is split by the feed, which contributes `q F` to the
liquid running down and `(1 - q) F` to the vapour running up. Its liquid is
therefore a **stripping**-section flow and its vapour a **rectifying**-section
one:

$$L_j = \begin{cases} L' = L + qF & j \le n_F \\ L = R D & j > n_F \end{cases}
\qquad
V_j = \begin{cases} V' = V - (1-q)F & j < n_F \\ V = (R+1) D & j \ge n_F \end{cases}$$

That is one convention, not two: both flows follow from asking whether the
horizontal cut a stream crosses has the feed above it. `_is_rectifying_cut` and
`_cmo_section_flows` in `difflow/units/distillation.py` are the single
definition, and `_cmo_flows` — the only thing that builds an L/V profile — is a
thin wrapper over them, so both solver paths read the same boundary.

`q` sets those rates on both paths, and it is also the feed's thermal condition
in the energy balance:

$$h_F = q\, h^L(z, T_F) + (1 - q)\, H^V(z, T_F)$$

both phase enthalpies at the feed stream's own temperature. So a
saturated-vapour feed arrives with its latent heat already in it and the
reboiler is not charged for it: going from `q = 1` to `q = 0` drops
`Q_reboiler` by `F (H^V - h^L)` at the feed temperature (2.4 MW on a 100 mol/s
equimolar benzene/toluene feed at 380 K), and moves the ~`F` step in the
converged profile from `L` to `V` across the feed stage. The shortcut column
forms its feed enthalpy the same way, from the `q` passed to the call.

`q` outside `[0, 1]` is allowed, and means what a textbook means by it: `q > 1`
is a subcooled feed, `q < 0` a superheated one. It has to be allowed, because
on the CMO path (`use_mesh=False`) `q` is the *only* place either can be said —
`_cmo_section_rates` is the whole model there, and `T_feed` never reaches it.
`L_strip = L_rect + q F` with `q > 1` is exactly how the extra internal reflux
of a subcooled feed is written. Underwood's equation takes it as written too.

The one place it is clamped is the **feed enthalpy**, and that clamp is the
physics rather than a guard. Both phase enthalpies are evaluated at the feed's
own temperature, so at `q = 1.3` the honest answer is `h_liquid(z, T_feed)`: an
all-liquid feed below its bubble point, with the subcooling carried by
`T_feed`. Forming `1.3 h^L - 0.3 H^V` would subtract three tenths of a latent
heat that is not there, and count the departure from saturation twice.

#### Partial condenser and Murphree efficiency

**`condenser_type='partial'`** makes the condenser an equilibrium stage at the
column pressure, **outside** `n_stages`. Its vapour is the distillate and its
liquid is the reflux:

$$(R+1)\,D\,y_{i,top} = D\,y_{D,i} + R\,D\,x_{R,i},\qquad
y_{D,i} = K_i(T_c)\,x_{R,i},\qquad \sum_i x_{R,i} = 1$$

so $T_c$ is the **dew point of $y_D$** (and the bubble point of $x_R$). The
condenser is one more row of the same tridiagonal system, so the summed
balances still telescope to $D y_D + B x_B = F z$. Stage counting: the profiles
in `info` keep their `n_stages` rows (nothing shifts when you switch the
condenser type), the condenser is reported separately as `info['y_D']`,
`info['x_R']` and `info['T_condenser']`, and `info['n_equilibrium_stages']`
is `n_stages + 1`. The distillate stream is the vapour at `T_condenser`.
`Q_condenser` is the duty on the partial basis: it removes the latent heat of
the reflux only, not of the vapour distillate. Under CMO a partial condenser
on `N` stages is exactly a total condenser on `N + 1` stages. A mixed
(vapour + liquid distillate) condenser, `condenser_type='mixed'`, is not
implemented and raises `NotImplementedError`; there is no separate condenser
pressure (`P_condenser`) either, the column has one `P`.

**`stage_efficiency`** is a Murphree *vapour* efficiency, a scalar or an
`(n_stages,)` array (the name and the 0-1 convention of
`CascadeParams.stage_efficiency` in the LLE unit):

$$y_{i,j} = y_{i,j-1} + E_j\,\bigl(K_{i,j}\,x_{i,j} - y_{i,j-1}\bigr)$$

with $y_{j-1}$ the *actual* vapour leaving the stage below. The reboiler
(stage 0) and a partial condenser are equilibrium stages, so their $E$ is 1
whatever is given for stage 0. The default $E = 1$ reproduces the previous
results **bit for bit** (it takes the original code path). The MESH energy
balance uses the actual vapour compositions. Each iteration solves the balance
and the Murphree relation together as a block-tridiagonal system (so it is
well posed at $E = 0$, where the trays do nothing and the column is a flash
at the reboiler), and the iteration is a fixed-count homotopy: converge at
$E = 1$, step $E$ to its target in four steps, iterate at the target. It is
differentiable in $E$ (scalar or one element of the array), in $R$, and on both
solver paths. `info['overall_efficiency_equiv']` is a *Fenske-style proxy* for
equivalent theoretical stages over actual stages: the ratio of
$\ln$ of the light/heavy key separation factor (keys = most and least volatile
species at the feed bubble point) to the same column's at $E = 1$. It is not
the theoretical-stage ratio itself and is meant as a trend to compare with an
O'Connell estimate.

What was checked, and against what:

- Constant-$\alpha$ binary under CMO (`use_mesh=False`) with $E = 0.7$,
  total and partial condenser: the tray liquids equal those of
  `difflow.staged.mccabe_thiele` on the pseudo-equilibrium curve to 1e-10
  (observed ~1e-15), and the component balance closes to 1e-12 at
  `cmo_iter=400`.
- $E \to 0$ against an independent flash with vapour fraction $D/F$.
- Partial vs total on $N + 1$ stages (CMO), distillate dew point against an
  independent root find, `check_grads` on the CMO and MESH paths.
- **Not checked** against a published worked example (Wankat, Seader): their
  inputs and printed answers were not available, so no agreement with a
  textbook number is claimed. The depropanizer test (ethane to n-pentane on
  Peng-Robinson at 17 bar, 20 stages, partial condenser) uses an illustrative
  feed, not the book's wex13-1 data: it checks convergence and closure only.

```python
from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.distillation import DistillationColumn, DistillationColumnParams

names = ['benzene', 'toluene']
thermo = IdealThermo({s: get_species_data(s) for s in names})
feed = make_stream({'benzene': 50.0, 'toluene': 50.0}, T=365.0, P=101325.0)

for cond, eff in [('total', 1.0), ('partial', 1.0), ('total', 0.7)]:
    column = DistillationColumn(
        DistillationColumnParams(species_order=names, n_stages=10, feed_stage=5,
                                 condenser_type=cond, stage_efficiency=eff),
        thermo=thermo,
    )
    distillate, bottoms, info = column(feed, R=2.5, D_spec=50.0, use_mesh=False,
                                       cmo_iter=100)
    print(f"{cond:7s} E={eff}: x_D(benzene) = "
          f"{float(distillate['F_benzene'] / 50.0):.4f}, "
          f"T_cond = {float(info['T_condenser']):.1f} K, "
          f"equilibrium stages = {int(info['n_equilibrium_stages'])}")
```

The reflux ratio and the product split are arguments to the call, not
parameters — give exactly one of `D_spec` or `B_spec`:

<!-- doc-test: skip: call fragment; the column is built in the examples below -->
```python
distillate, bottoms, info = column(feed, R=2.0, B_spec=40.0)
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

#### Product temperatures

The bottoms leaves the reboiler, which is a stage, so it is at
`info["T_profile"][0]`. The distillate does not leave a stage — it leaves the
condenser. A total condenser condenses the whole of the top stage's vapor, so
the distillate (and the reflux returned with it) is a **saturated liquid of
composition $x_D$ at its own bubble point**, reported as `info["T_condenser"]`.

That is not the top stage temperature. The top stage sits at the bubble point
of its liquid $x_{top}$, equivalently the dew point of the vapor $y_{top} = x_D$
it sends up, and a mixture's dew point is above its bubble point. The gap is
the boiling range of the distillate itself:

| distillate | top stage $T$ | condenser $T$ | gap |
|---|---|---|---|
| 99.9999 % benzene / toluene, 1 atm | 368.665 K | 368.665 K | 0.0001 K |
| C3-C8 cut, 10 bar (Peng-Robinson) | 400.6 K | 362.9 K | 38 K |

A one-component distillate has no boiling range and so no gap, which is why the
binary row reads as zero: at $\alpha \approx 8$ over 15 stages that column takes
27 µmol/s of toluene overhead and nothing more. The gap is a property of the
cut, not of the column.

The same distinction runs through the energy balance: $Q_{cond}$ takes the top
stage vapor down to that condensed state, and the reflux re-enters the top
stage subcooled, at the condenser temperature rather than the tray's.

#### Thermodynamics: ideal K-values or a cubic EOS

The column takes either an `IdealThermo` or a
[`CubicThermo`](thermodynamics.md), and the choice is the whole of the
difference between a near-ideal separation and a hydrocarbon one:

| | `IdealThermo` | `CubicThermo` |
|---|---|---|
| $K_i$ | $P^{sat}_i(T)/P$ (Raoult) | $\hat\phi^L_i(T,P,x)\,/\,\hat\phi^V_i(T,P,y)$ (PR or SRK) |
| stage enthalpy | ideal-gas $C_p$ + Watson $H_{vap}$ | ideal-gas $C_p$ + EOS departure |
| $K_i$ depends on composition | no | yes |

```python
from difflow import IdealThermo, CubicThermo, PengRobinson, make_stream
from difflow.database import get_critical_props, get_species_data
from difflow.units.distillation import DistillationColumn, DistillationColumnParams

names = ["propane", "isobutane", "n_butane", "isopentane",
         "n_pentane", "n_hexane", "n_heptane", "n_octane"]
ideal = IdealThermo({s: get_species_data(s) for s in names})
eos = PengRobinson({s: get_critical_props(s) for s in names})

column = DistillationColumn(
    DistillationColumnParams(species_order=names, n_stages=20, feed_stage=10,
                             condenser_type="total", P=10e5),
    thermo=CubicThermo(ideal, eos),      # Peng-Robinson K-values and enthalpies
)
feed = make_stream(
    dict(zip(names, [10.0, 7.0, 7.0, 8.0, 8.0, 20.0, 10.0, 30.0])),
    T=380.0, P=10e5,
)
distillate, bottoms, info = column(feed, R=2.0, B_spec=40.0)
```

Use the EOS for light hydrocarbons at pressure. Raoult's law fails there in a
one-sided way: a C3-C8 cut at 10 bar puts its heavy end within a degree of the
EOS answer and its light end tens of degrees off, so an ideal-K column looks
plausible at the reboiler and is wrong at the condenser.

Two things about the EOS path are worth knowing:

- **The K-values depend on composition**, so they are a fixed point rather than
  a formula. `CubicThermo.K_values_array(T, P, x)` runs a short successive
  substitution on $y$ internally to close it at the stage's own composition;
  pass both `x` and `y` if you already have a consistent pair.
- **They only exist where the cubic has two roots.** Away from the bubble point
  -- a subcooled liquid, a superheated vapor -- there is one root, both phases
  take it, and $K_i$ comes back identically 1. That is the EOS reporting a
  single phase, but it makes $\sum_i K_i x_i - 1$ flat, so the column solves
  each stage's bubble point in two passes: a Newton solve on ideal K-values to
  land inside the two-root window, then a step-capped Newton on the EOS
  K-values that bisects back if a step leaves it. Nothing about this is visible
  in the API, but it is why the bubble point is not one `optimistix` call.

The ideal pass is itself in two parts, for a reason that has nothing to do with
the EOS. Vapor pressure is exponential in $-1/T$, so a couple of hundred degrees
below the bubble point both $\sum_i K_i x_i$ and its slope are round-off away
from zero, and a Newton step there divides one tiny number by another and lands
tens of thousands of degrees away. So the solve first takes a few damped steps
on $\log \sum_i K_i x_i$ -- nearly linear in $1/T$, and well scaled over the
whole range -- and only then runs Newton on the residual itself. This is why a
column can be handed a feed far below its own boiling point and still find its
ends.

(solver-paths)=
#### Solver Paths

Both paths run the Wang-Henke bubble-point iteration, which solves the
component material balances for the whole column as a tridiagonal system.
They differ only in where the L/V profiles come from:

| `use_mesh` | L/V profiles | Cost |
|---|---|---|
| `True` (default) | corrected each iteration by the stage enthalpy balances | ~30 % more |
| `False` | frozen at their constant-molar-overflow values, $L' = L + qF$ and $V' = V - (1-q)F$ | cheaper |

The CMO path is a shortcut in the *energy* balance, not in the material
balance: because the tridiagonal solve is the component balance, summing it
over the stages telescopes to $D x_{D,i} + B x_{B,i} = F z_i$. Both paths
satisfy that only to within their iteration count, so both report the
residual:

```python
from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.distillation import DistillationColumn, DistillationColumnParams

names = ['n_pentane', 'n_hexane', 'n_heptane']
column = DistillationColumn(
    DistillationColumnParams(species_order=names, n_stages=20, feed_stage=10,
                             P=101325.0),
    thermo=IdealThermo({s: get_species_data(s) for s in names}),
)
feed = make_stream({'n_pentane': 30.0, 'n_hexane': 40.0, 'n_heptane': 30.0},
                   T=360.0, P=101325.0)

distillate, bottoms, info = column(feed, R=2.0, B_spec=40.0, use_mesh=False)
print(info['balance_error'])      # (n_species,) D_i + B_i - F_i, mol/s
print(info['balance_error_rel'])  # max |error| / F_total
```

If `balance_error_rel` is larger than your problem tolerates, raise `cmo_iter`
(default 30). The residual falls geometrically with it, but at a rate the
column sets, so treat the reported number as the answer rather than assuming a
count is enough:

| case | `cmo_iter` 30 | 60 | 100 |
|---|---|---|---|
| 20-stage ternary, $R = 2$ | 1.6e-4 | 5.0e-10 | — |
| 12-stage binary, $R = 1.2$ (near $R_{min}$) | 1.9e-4 | 1.8e-4 | 1.7e-4 |

Near minimum reflux the fixed-point iteration converges with a rate close to
one, and more sweeps buy almost nothing. That is a property of the column, not
a defect in the solver — but it is exactly why the residual is reported instead
of asserted.

#### Outputs

| Key | Type | Units | Description |
|-----|------|-------|-------------|
| `distillate` | Stream | - | Overhead product, at the condenser temperature |
| `bottoms` | Stream | - | Bottom product, at the reboiler temperature |
| `info['T_profile']` | (n,) array | K | Stage temperatures, reboiler first |
| `info['x_profile']` | (n, nc) array | - | Liquid compositions per stage |
| `info['y_profile']` | (n, nc) array | - | Vapor compositions per stage |
| `info['T_condenser']` | float | K | Condenser temperature = distillate T |
| `info['T_reboiler']` | float | K | Reboiler temperature = bottoms T = `T_profile[0]` |
| `info['D']`, `info['B']` | float | mol/s | Distillate and bottoms flow |
| `info['Q_condenser']` | float | W | Condenser duty (negative: heat removed) |
| `info['Q_reboiler']` | float | W | Reboiler duty (positive: heat added) |
| `info['L_profile']`, `info['V_profile']` | (n,) array | mol/s | Internal flows — `use_mesh=True` only |
| `info['L_rect']`, `info['V_rect']` | float | mol/s | Rectifying flows — `use_mesh=False` only |
| `info['balance_error']` | (nc,) array | mol/s | Component balance residual $D_i + B_i - F_i$ |
| `info['balance_error_rel']` | float | - | `max abs(balance_error) / F_total` |
| `info['n_equilibrium_stages']` | int | - | `n_stages`, plus one for a partial condenser |
| `info['overall_efficiency_equiv']` | float | - | Fenske-style efficiency proxy (1 at $E = 1$), see above |
| `info['y_D']`, `info['x_R']` | (nc,) array | - | Partial condenser only: vapour distillate and reflux compositions |

#### Example Usage

```python
from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.distillation import DistillationColumn, DistillationColumnParams

names = ['n_pentane', 'n_hexane', 'n_heptane']
thermo = IdealThermo({s: get_species_data(s) for s in names})

column = DistillationColumn(
    DistillationColumnParams(
        species_order=names,
        n_stages=20,      # 19 trays plus the reboiler
        feed_stage=10,    # halfway up, counting from the reboiler at 0
        P=101325.0,
    ),
    thermo=thermo,
)

feed = make_stream({'n_pentane': 30.0, 'n_hexane': 40.0, 'n_heptane': 30.0},
                   T=360.0, P=101325.0)

distillate, bottoms, info = column(feed, R=2.0, B_spec=40.0)
print(f"Distillate: {float(distillate['T']):.1f} K, "
      f"{float(distillate['F_n_pentane']):.1f} mol/s n-pentane")
print(f"Bottoms:    {float(bottoms['T']):.1f} K, "
      f"{float(bottoms['F_n_heptane']):.1f} mol/s n-heptane")
print(f"Duties:     {float(info['Q_condenser'])/1e6:.2f} / "
      f"{float(info['Q_reboiler'])/1e6:.2f} MW")
```

Pass `use_mesh=False` for the faster constant-molar-overflow solution, which
skips the energy-balance correction to the internal flows but solves the same
component material balances — see [Solver Paths](#solver-paths) above.

### BatchDistillation

A batch of `F` mol is charged to a still, boiled at a constant rate `V`, and
the vapor passes through `N` equilibrium stages to a total condenser that
returns reflux and draws distillate `D = V/(R+1)`. The model is quasi-steady
(no stage or condenser holdup), constant molar overflow, binary. With the still
holdup `W` and its key fraction `x_W`:

$$
\frac{dW}{dt} = -D, \qquad \frac{d(W x_W)}{dt} = -D\,x_D
\quad\Longrightarrow\quad
\ln\frac{F}{W} = \int_{x_W}^{x_F} \frac{dx}{x_D(x) - x}.
$$

For `N = 0` the still is in equilibrium with its distillate, `x_D = y*(x_W)`,
and this is the **Rayleigh equation** `ln(F/W) = int dx/(y* - x)`; for constant
relative volatility it has the closed form
`ln(F/W) = [ln(x_F/x_W) + alpha ln((1-x_W)/(1-x_F))]/(alpha - 1)`.
`difflow.units.batch_distillation.rayleigh` uses that form for
`constant_alpha` and a 64-point Gauss-Legendre quadrature (in `ln x`, where the
integrand stays smooth as `x_W -> 0`) for any other
`difflow.staged.EquilibriumCurve`; give `W_over_F` or `x_W` and it solves for
the other. Past an azeotrope the integral diverges: it returns `ln_F_over_W =
inf` and `ok = False`, never NaN.

With trays, `x_D(x_W)` is the McCabe-Thiele stage relation inverted: from
`y_1 = x_D` step `x_n = x*(y_n)`, `y_{n+1} = [R x_n + x_D]/(R+1)` through the `N`
trays, then the still, `x_W = x*(y_{N+1})`. A Murphree vapor efficiency
(`stage_efficiency`) applies to the trays, not the still.

* **`mode="constant_reflux"`** (`R`): `D` is constant, so `t = (F - W)(R+1)/V`, and
  the distillate composition falls with time.
* **`mode="constant_composition"`** (`x_D_spec`, `light_key`): `x_D` is fixed, so
  `W/F = (x_D - x_F)/(x_D - x_W)` exactly; the reflux `R(x_W)` is found at each
  still composition (a scalar root find with implicit gradients) and rises with
  time; `t = (1/V) int (R+1) dW`, and `info["boilup_total"] = int V dt` is the
  vapor generated.

**Charge convention.** `charge` is a stream read as *moles* (`F_i` in mol, not
mol/s), the initial contents of the still, as `FedBatchReactor` reads its initial
state. The `still` and `distillate` outlets are final holdup and accumulated
product in moles.

**Stopping** is data, not control flow: `stop` is a dict with any of `"x_W"`
(final still composition), `"x_D_avg"` (accumulated distillate purity),
`"distillate"` (mol) and `"time"` (s) (`t_final=` is shorthand); the first one
reached ends the batch. The stopping composition is a bisection followed by an
implicit-function Newton step, so `jax.grad` of any result with respect to the
spec, `R`, `boilup`, the charge or the efficiency follows
`dt*/d theta = -(dg/d theta)/(dg/dt)`.

```python
import jax
import jax.numpy as jnp
from difflow.staged import constant_alpha
from difflow.units.batch_distillation import (
    BatchDistillation, BatchDistillationParams, rayleigh)

curve = constant_alpha(2.5)                    # an illustration, not a published example
r = rayleigh(curve, 0.5, W_over_F=0.4)         # simple batch distillation
print(float(r["x_W"]), float(r["x_D_avg"]))

def run(R, mode="constant_reflux"):
    p = BatchDistillationParams(species_order=["a", "b"], n_stages=3, R=R,
                                boilup=2.0, mode=mode, x_D_spec=0.9)
    unit = BatchDistillation(p, curve=curve)
    charge = {"F_a": 50.0, "F_b": 50.0, "T": 350.0, "P": 1e5}      # mol
    return unit(charge, stop={"x_W": 0.2}, n_traj=11)

still, distillate, info = run(3.0)
print(float(info["t_batch"]), float(info["x_D_avg"]), float(info["recovery"]))
# balances close: charge = still + distillate
print(float(still["F_a"] + distillate["F_a"]))                      # 50.0

# constant composition: x_D held at 0.9, reflux rises
_, _, cc = run(1.0, "constant_composition")
print(float(cc["x_D"][0]), float(cc["R"][0]), float(cc["R"][-1]))

# design with gradients: d(recovery)/dR at fixed final still composition
print(float(jax.grad(lambda R: run(R)[2]["recovery"])(3.0)))
```

For ethanol-water give `BatchDistillation(params, thermo, activity_model)` (NRTL).
The distillate cannot pass the azeotrope: `info["x_azeotrope"]` reports it,
`info["azeotrope_limited"]` is True when the distillate comes within 0.01 of it
(or a constant-composition spec lies beyond it) and `info["feasible"]` is False
when the operation cannot be carried out (a spec beyond the azeotrope or beyond
what the stages reach, a charge on the far side of the azeotrope, the wrong key).
The numbers stay finite; read the flags.

`batch_cuts(unit, charge, cut_specs)` runs successive cuts, each charged with
the previous still (heads, hearts, tails). Each cut is a stop dict, or
`{"stop": {...}, "R": 6.0}` with per-cut parameter overrides (the reflux
policy); it returns each cut's moles and key composition, the residue and a
`balance` that is zero to round-off.
`rayleigh_multicomponent(thermo, species, x_F, W_over_F, P)` integrates
`dx_i/d ln W = y_i - x_i` with `y_i = K_i(T_b) x_i` by fixed RK4 steps.

Assumptions: binary, total condenser, constant molar overflow, no stage or
condenser holdup (`holdup` must be 0), the key must be the more volatile
component. Not provided: multicomponent tray-to-tray columns, holdup
dynamics, optimal reflux policies other than the two classic modes, inverted
or middle-vessel columns.

What was and was not verified: the constant-alpha closed form and
`scipy.integrate.quad`; the quadrature path against the closed form to 1e-5; a
separate NumPy/SciPy time integration of the batch balances (`solve_ivp`, tray
march by `brentq`) for constant reflux with and without Murphree efficiency; the
exact `W/F` and a NumPy quadrature of the time for constant composition; mass
balances; `check_grads`. **No published textbook worked example (Geankoplis
Sec. 11.3, McCabe-Smith-Harriott Ch. 21, Wankat Ch. 9) is reproduced**: the
numbers could not be retrieved, so none is quoted. See
`tests/test_batch_distillation.py`.

---

## Graphical Staged Design

**Location**: `difflow/staged.py` (a library of functions; not a palette
operation, because it designs a column rather than transforming a stream).

`difflow.staged` is the generic equilibrium-stage graphics and algebra for a
binary (or pseudo-binary) cascade: operating lines, the q-line, a smooth
McCabe-Thiele stepper with a **fractional, differentiable stage count**,
minimum reflux from the pinch, total reflux, Kremser, and a plot helper. The
same stepper serves distillation, absorption/stripping and immiscible
extraction: it takes any equilibrium curve and any pair of straight operating
lines.

Lines are `(a, b, c)` arrays meaning `a x + b y = c` (never slope-intercept,
so the vertical q-line at `q = 1` needs no special case). A curve is an
`EquilibriumCurve` pytree built by `constant_alpha(alpha)`,
`linear_equilibrium(m)` or `tabulated(x, y)` (monotone PCHIP, C1, with an
inverse; `xy_curve` output goes straight in); `curve.with_efficiency(E)` adds a
Murphree vapor efficiency.

### McCabe–Thiele

Benzene–toluene at constant alpha 2.5 (an illustration, not a published
example): `z = 0.4`, `x_D = 0.95`, `x_B = 0.05`, saturated-liquid feed.

```python
import jax
import jax.numpy as jnp
from difflow import staged as S

alpha, z, xD, xB = 2.5, 0.4, 0.95, 0.05
curve = S.constant_alpha(alpha)
R_min = S.minimum_reflux(curve, z, 1.0, xD)        # 1.4444
N_min = S.total_reflux_stages(curve, xD, xB)       # 6.4269 (Fenske)

def column(R, q=1.0):
    return S.mccabe_thiele(curve, S.rectifying_line(R, xD),
                           S.stripping_line(R, xD, xB, z, q), xD, xB,
                           q_line=S.q_line(q, z))

res = column(1.5 * R_min)
print(float(R_min), float(N_min))
print(float(res["N"]), float(res["feed_stage"]))   # 11.72 stages, feed on 6
dNdR = jax.grad(lambda R: column(R)["N"])(1.5 * R_min)
print(float(dNdR))                                 # negative and finite
```

`mccabe_thiele` steps from the top with a fixed-length `lax.scan`: from
`x_{k-1}` the operating line gives `y_k`, and the curve gives
`x_k = x*(y_k)`; steps past the bottom are masked, not branched on. The
result holds `N`, `N_rect`, `N_strip`, a fractional `feed_stage`, the
staircase vertices, and a soft `pinched` flag.

**The fractional-stage convention.** Stage `k` is credited the fraction of
its step needed to reach `x_B`: whole stages count 1, stages past the end
0, and the crossing stage counts the linear interpolation in `x` between
its two compositions (`fraction="linear"`, the textbook "fraction of a
stage"). `fraction="log"` assumes geometric steps (exact for straight lines,
so it reproduces Kremser) and `"log_ratio"` the same in `x/(1-x)` (exact
Fenske). The count is continuous in every input but only piecewise smooth:
its derivative jumps where one more whole stage is needed, and `dN/dR`
depends on the convention, so state it with the answer. The optimal-feed
switch is a sigmoid in `x` of width `feed_width`. A reboiler is the last stage;
with `partial_condenser=True` the first stage is the condenser and
`N_column = N - 1`. With open steam, build the bottom line with
`stripping_line(..., open_steam=True)` (through `(x_B, 0)`). A subcooled feed
is `q > 1`. Specifying a feed stage *earlier* than the optimal one
extends the stripping line into a region where it lies above the curve; the
stepper does not guard against that.

**Pinches.** `minimum_reflux` takes the largest chord slope
`(x_D - y*(x))/(x_D - x)` from `(x_D, x_D)` to the curve over
`[x_q, x_D)`, `x_q` being where the q-line meets the curve. A constant-alpha
curve is concave, so that is the feed-line pinch (the closed form
`R_min = (x_D - y_f)/(y_f - z)` at `q = 1`); a curved curve such as
ethanol–water has an interior maximiser, the **tangent pinch**, with no special
case. The maximiser is found with a stop-gradient search and the value
re-evaluated there, so `jax.grad` follows the envelope theorem. The result is
not floored. `minimum_solvent_ratio(curve, x_in, x_out, y_in)` is the same
construction for absorption and extraction.

```python
import numpy as np
from difflow import staged as S

# A synthetic curve whose volatility falls from 10 to 1 with x (like
# ethanol-water up to its azeotrope, but not data for it): the chord from
# (x_D, x_D) touches it in the interior, so R_min exceeds the feed-line value.
x = np.linspace(0.0, 1.0, 81)
alpha_x = 1.0 + 9.0 * (1.0 - x) ** 4
r = alpha_x * x / (1.0 - x + 1e-300)
y = r / (1.0 + r)
y[-1] = 1.0
curve = S.tabulated(x, y)
z, xD = 0.1, 0.7
R_min = S.minimum_reflux(curve, z, 1.0, xD)
y_feed = curve(z)
R_feed = (xD - y_feed) / (y_feed - z)
print(round(float(R_min), 3), round(float(R_feed), 3))   # 1.059 0.796: tangent pinch governs
```

### Kremser

`kremser_fraction(A, N)` is the recovery of `N` counter-current stages at
absorption (or extraction) factor `A = L/(m V)`, and `kremser_stages(A, f)`
the inverse, `N = ln((A - f)/(1 - f))/ln A - 1`. Both hold for any `A > 0`
(for `A < 1` the recovery tends to `A`, and `kremser_stages` returns `inf`
for `f >= A`) and join smoothly across `A = 1` with a Taylor blend, not a
hard branch, so the gradient is continuous there. They are re-exported from
`difflow.units.lle` and `difflow_ree`; `difflow_ree.stages_kremser` is a
wrapper on them (it used to return 1.0 for every input; see the CHANGELOG; it keeps its floor at one stage).

```python
from difflow.staged import kremser_stages, kremser_fraction

print([round(float(kremser_stages(E, 0.99)), 2) for E in (1.5, 2.0, 5.0)])  # [8.7, 5.66, 2.72]
print(float(kremser_fraction(1.0, 3.0)))   # 0.75, the A -> 1 limit N/(N+1)
```

The worked notebook is `examples/48_mccabe_thiele_staged.ipynb` (stepper vs
`DistillationColumn`, optimum `R/R_min` by gradient).

**What was and was not verified.** `tests/test_staged.py` checks the stepper
against a plain-Python staircase, `minimum_reflux` against the closed-form
pinch at any `q` and (for the tangent pinch) against SciPy PCHIP on a dense
grid, `total_reflux_stages` against Fenske, the stepper against Kremser at
straight lines, and the Murphree efficiency against the Lewis relation
`E_OG = ln(1 + E_MV(lambda - 1))/ln(lambda)` (overall efficiency
`E_OG = N_equilibrium/N_actual`). The wex12-1 numbers are the issue's own
hand calculation. **No published textbook example (Geankoplis, McCabe–Smith–
Harriott, Wankat) was reproduced**: the worked-example inputs and printed
answers were not available, so no agreement is claimed. Out of scope:
Ponchon–Savarit, multicomponent graphical methods, side draws and a second
feed.

---

(absorption-and-stripping)=
## Absorption and Stripping

Physical-solvent gas absorbers and strippers: a gas of carrier plus dilute
solutes rises through a column against a liquid solvent (`gas_in` at the
bottom, `liquid_in` at the top). The keys of `params.henry` are the
transferring solutes; every other species stays on the side it enters (the
carrier in the gas, the solvent in the liquid; give them different names).
Reactive absorption (amines, loading) is the job of the carbon-capture
plugin: see `AmineAbsorber` and `AmineStripper` in
[Carbon Capture Unit Operations](unit-operations-carbon-capture.md). The
generic units here are physical (Henry's law) and are not a replacement for
them.

Equilibrium is Henry's law in mole fractions, `y_i = K_i x_i`, `K_i = H_i(T)/P`,
with a van 't Hoff dependence
`H(T) = H_ref exp(-C (1/T - 1/T_ref))` (`C = -dH_abs/R`, positive for an
exothermic absorption). The column is isothermal; the heat of absorption and
the adiabatic temperature rise it would cause are reported in `info`, not
applied. `difflow.units.absorption.HENRY_WATER` holds a small table (acetone,
NH3, CO2, H2S, SO2, TCE, benzene in water near 25 C) **transcribed from
memory, unverified, +/-20-30 %**: use measured data for design.

### Absorber

**Location**: `difflow/units/absorption.py`

**Class**: `Absorber(params)(gas_in, liquid_in) -> (gas_out, liquid_out, info)`,
with `AbsorptionParams` (shared with `Stripper`; not named `AbsorberParams`
so that it cannot shadow `difflow_cc.AbsorberParams` in the serializer's
by-name lookup).

Three models, chosen by `params.model`:

**`"kremser"`** (closed form per solute). With the effective absorption factor
`A_i = L/(K_i V)` (the geometric mean of its top and bottom values, so the flow
change along the column is carried, Edmister's idea; `variable_flow=False`
uses the inlet flows) and the equivalent equilibrium stages
`N_eq = N ln(1+E(S-1))/ln S`, `S = 1/A`, for a Murphree vapor efficiency `E`:

$$
v_{out} = v_{in}(1-f) + \frac{l_{in}}{A}\,f, \qquad
f = \frac{A^{N_{eq}+1}-A}{A^{N_{eq}+1}-1}.
$$

This includes solute in the entering solvent (`l_in/A = K x_in V` is the gas
in equilibrium with it), holds for every `A > 0` (for `A < 1` the removal
tends to `A`, not 1) and passes smoothly through `A = 1`. `n_stages` is a real
number, so it is differentiable. The effective-stage ratio is the Lewis
relation for a vapor efficiency (`difflow_cc` uses the `A` form, which is the
liquid-side efficiency).

**`"stages"`** (tray-to-tray). All component balances plus the Murphree stage
relation, solved as one linear system per solute and iterated a fixed number of
times on the total flows of every stage. It carries the flow change that the
Kremser mean `A` only approximates, handles any number of solutes at once, and
is the oracle the Kremser mode is checked against (they agree to about `1e-4`
at dilute solute and diverge as the solute gets concentrated). `n_stages`
must be a concrete integer.

**`"packed"`** (rate-based HTU/NTU). `Z = H_OG N_OG` with
`H_OG = H_G + lambda H_L`, `lambda = 1/A = K V/L`. The film heights come from
the Onda correlations of [Packed-Column Mass Transfer](#mass-transfer-packed)
(give `packing`, `diameter`, `D_G`/`D_L` per solute and the fluid properties;
viscosities, density and surface tension default to air and water at
25 C) or you give `H_OG`. For straight lines the outlet follows Colburn's
closed form; `ntu_method="integral"` evaluates `N_OG` by quadrature for the key
solute (`concentrated=True` adds the `(1-y)_lm` correction). **Rating**: give
`height`; **design**: give `recovery` and read `info["height"]`. Flooding and
pressure drop are not part of this unit: give the diameter.

**Design instead of rating.** Setting `recovery` (the fraction of the
`key_solute` removed from the gas) makes the Kremser stage count or the packed
height an output; the other solutes are then rated at that size. A target
below the minimum solvent rate is flagged (`info["feasible"]` is False, the
stage count is `inf`, nothing is NaN).

`info` holds, per solute, `recovery`, `transferred` (mol/s, gas to liquid), `A`,
`S`, `K`, `N_eq` (and `N_OG`, `H_OG` for packed); `n_stages` and `height`;
`L_min` and `L_over_Lmin` for the key solute; the solute-free operating line
and equilibrium curve `X`, `Y`, `Y_eq` for plotting; `Q_absorption` (W),
`delta_T_adiabatic` (K), `converged` and `feasible`.

```python
import jax.numpy as jnp
from difflow import Absorber, AbsorptionParams, make_stream
from difflow.units.absorption import henry_params

# acetone scrubbed from air with water at 25 C (Henry's constants from memory)
henry, henry_C = henry_params(["acetone"])
gas = make_stream({"air": 1.0, "acetone": 0.02}, 298.15, 101325.0)    # mol/s
water = make_stream({"H2O": 2.0}, 298.15, 101325.0)

for model in ("kremser", "stages"):
    absorber = Absorber(AbsorptionParams(henry=henry, henry_C=henry_C,
                                         model=model, n_stages=6))
    treated, rich, info = absorber(gas, water)
    print(model, "recovery", float(info["recovery"]["acetone"]),
          "L/L_min", float(info["L_over_Lmin"]))

# design: how many stages for 95 % recovery at this solvent rate?
design = Absorber(AbsorptionParams(henry=henry, henry_C=henry_C, recovery=0.95))
_, _, info = design(gas, water)
print("stages for 95 %:", float(info["n_stages"]))
```

Packed rating and design, with the Onda films (diffusivities from the
correlations of the mass-transfer section):

```python
from difflow import Absorber, AbsorptionParams, make_stream
from difflow.mass_transfer import (diffusion_volume, fuller_diffusivity,
                                   wilke_chang_diffusivity)
from difflow.units.absorption import henry_params

henry, henry_C = henry_params(["acetone"])
D_G = fuller_diffusivity(298.15, 101325.0, 58.08, 28.96,
                         diffusion_volume({"C": 3, "H": 6, "O": 1}), 19.7)
D_L = wilke_chang_diffusivity(298.15, 8.9e-4, 18.015, 74.0, 2.6)
gas = make_stream({"air": 20.0, "acetone": 0.4}, 298.15, 101325.0)
water = make_stream({"H2O": 50.0}, 298.15, 101325.0)

column = dict(henry=henry, henry_C=henry_C, model="packed",
              packing="pall_metal_50", diameter=0.8,
              D_G={"acetone": float(D_G)}, D_L={"acetone": float(D_L)})
_, _, d = Absorber(AbsorptionParams(recovery=0.95, **column))(gas, water)
print("height for 95 %:", float(d["height"]), "m;  H_OG:", float(d["H_OG"]["acetone"]), "m")
_, _, r = Absorber(AbsorptionParams(height=float(d["height"]), **column))(gas, water)
print("rating at that height:", float(r["recovery"]["acetone"]))
```

Library helpers (not palette units): `solute_free_operating_line(G_s, L_s,
X_ref, Y_ref)` gives the straight operating line `Y = slope X + intercept` in
solute-free ratios; `minimum_solvent_rate(curve, Y_in, Y_out, X_in, G_s)`
finds the pinch (rich end, or tangent for a curved equilibrium) with the
mole-fraction equilibrium curve of `difflow.staged`, returning the minimum
solute-free solvent rate (`inf` if the spec is below equilibrium with the
entering solvent); `minimum_stripping_gas_rate` is its mirror image;
`kremser_absorption(A, N)` and `kremser_stages(A, fraction)` are the
`difflow.staged` functions. The design rule of thumb is `L = 1.2-2 L_min`.

```python
from difflow.staged import tabulated
from difflow.units.absorption import minimum_solvent_rate

# a curved (inflected) equilibrium table: the pinch is a tangent, not the rich end
curve = tabulated([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
                  [0, 0.002, 0.004, 0.009, 0.022, 0.050, 0.080, 0.105, 0.120])
Y_in = 0.12 / 0.88
L_min = minimum_solvent_rate(curve, Y_in=Y_in, Y_out=2e-4, X_in=0.0, G_s=1.0)
print("L_s,min / G_s =", float(L_min))
```

Absorber and stripper with solvent recycle go through `Flowsheet.solve` like
any two-inlet unit; the recycled liquid is the tear stream, and the whole loop
is differentiable with respect to the solvent rate (the L/G trade-off between
absorber size and stripper duty is a `jax.grad` away; see
`examples/52_absorption_stripping.ipynb`).

Assumptions: dilute-to-moderate solutes with constant `K_i` per solute (no
activity-coefficient model: modified Raoult with `thermo` is not implemented),
isothermal, no solvent volatility, no reaction.

What was and was not verified: the Kremser closed form (including `A < 1`
and `A = 1`); the stage-by-stage mode against a separate SciPy `fsolve` of the
nonlinear tray equations; the Kremser/stages agreement for dilute linear
equilibrium, with and without a Murphree efficiency; Colburn against the
numerical `N_OG` integral and the rating/design inversion; the minimum
solvent rate against a dense chord-slope scan (rich-end and tangent pinches);
per-species mass balances in every mode; `check_grads`; an
absorber-stripper recycle flowsheet through `fs.solve`. **No published
textbook worked example (Seader-Henley-Roper Ch. 6, McCabe-Smith-Harriott
Ch. 18, Geankoplis Ch. 10, Wankat Ch. 12/16, Crittenden Ch. 14) is
reproduced**: the issue's numbers are unverified recollections, so none is
quoted. See `tests/test_absorption.py`.

### Stripper

**Location**: `difflow/units/absorption.py`

**Class**: `Stripper(params)(gas_in, liquid_in) -> (gas_out, liquid_out, info)`

The same column run the other way: `gas_in` is the stripping gas (air or
steam) at the bottom and `liquid_in` the rich liquid at the top. It shares the
implementation, the three models and `AbsorptionParams` with `Absorber`; the
differences are that `recovery` is the fraction of the key solute removed from
the **liquid**, and `info` reports the minimum stripping-gas rate `G_min` and
`G_over_Gmin` instead of `L_min`. The stripping factor `S = K V/L` is in
`info["S"]`: with a clean gas the Kremser relation is
`x_out/x_in = (S-1)/(S^(N+1)-1)`, and with `S < 1` infinitely many stages
still remove only the fraction `S`. A volatile, sparingly soluble solute (a
large Henry's constant, TCE in water) needs a small air/water ratio, which is
why air stripping works.

```python
from difflow import Stripper, AbsorptionParams, make_stream
from difflow.units.absorption import HENRY_WATER, henry_params

henry, henry_C = henry_params(["TCE"])          # TCE in water (from memory)
K = HENRY_WATER["TCE"].H298 / 101325.0
L = 1000.0                                        # mol/s of water
air = make_stream({"air": 4.0 * L / K}, 298.15, 101325.0)    # stripping factor S = 4
water = make_stream({"H2O": L, "TCE": 1e-3}, 298.15, 101325.0)

stripper = Stripper(AbsorptionParams(henry=henry, henry_C=henry_C,
                                     recovery=0.99, variable_flow=False))
offgas, treated, info = stripper(air, water)
print("S =", float(info["S"]["TCE"]), " stages for 99 %:", float(info["n_stages"]),
      " G/G_min:", float(info["G_over_Gmin"]))
```

The TCE case is a hand-worked example with the Henry's constant above, not
the textbook example of the issue (which could not be retrieved).

---

## Heat Exchangers

### Heater

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `Heater`

**Description**: Single-stream heater that increases stream temperature using an external heat source (steam, hot oil, electric).

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class HeaterParams:
    duty: float = None       # Heat duty (W); the mode is set by which of
    T_out: float = None      # Outlet temperature (K); these three you give
    UA: float = None         # Overall HTC x Area (W/K), with T_utility
    T_utility: float = None  # Utility temperature (K)
    Cp: float = None         # Constant heat capacity (J/mol/K)
    phase: str = None        # Force 'liquid'/'vapor' for the thermo enthalpy
```

The operating mode is implied by which parameter is set -- `duty`, `T_out`, or
`UA` together with `T_utility` -- there is no `mode` field.

#### Energy models

The heater has two, and the choice matters more than any other parameter:

**Constant Cp** -- `Heater(HeaterParams(T_out=400.0, Cp=75.0))`:

$$Q = F_{total} C_p (T_{out} - T_{in})$$

Sensible heat only. A constant $C_p$ cannot carry latent heat, so this is
wrong -- often by a factor of several -- for any stream that vaporizes or
condenses across the unit.

**Thermo** -- `Heater(HeaterParams(T_out=400.0), thermo=thermo)`:

$$Q = H(T_{out}, P) - H(T_{in}, P)$$

with $H$ from the thermo's stream enthalpy. A
[`CubicThermo`](thermodynamics.md) supplies a two-phase flash enthalpy, so the
duty carries the real temperature dependence of the heat capacity *and* the
latent heat of any phase change. Use this whenever the stream may change
phase, and whenever you are comparing against a rigorous simulator. Pass
`phase='liquid'` or `phase='vapor'` to force a single-phase enthalpy instead
(which is what an `IdealThermo` provides).

With neither `Cp` nor `thermo`, the unit falls back to `DEFAULT_CP`
(75 J/mol/K, roughly liquid water) and raises a `DefaultCpWarning`. Turn that
into an error to make the fallback fatal:

```python
import warnings
from difflow import DefaultCpWarning

warnings.simplefilter("error", DefaultCpWarning)
```

**LMTD Rating** (for utility heating):

$$Q = UA \cdot LMTD$$

$$LMTD = \frac{(T_U - T_{in}) - (T_U - T_{out})}{\ln\left(\frac{T_U - T_{in}}{T_U - T_{out}}\right)}$$

Where $T_U$ is the utility (steam) temperature. On the constant-Cp path this
is solved in closed form by effectiveness-NTU with an infinite-capacity
utility; with a thermo it is a damped fixed point on $Q$, with $T_{out}$ from
inverting the enthalpy.

#### Example Usage

```python
from difflow import IdealThermo, make_stream
from difflow.database import get_species_data
from difflow.units.heat_exchanger import Heater, HeaterParams

thermo = IdealThermo({'water': get_species_data('water')})
inlet = make_stream({'water': 10.0}, T=300.0, P=101325.0)

# Specified duty
heater = Heater(HeaterParams(duty=50000.0, Cp=75.0))
outlet, info = heater(inlet)

# Specified outlet temperature, duty from the thermo (carries latent heat)
heater = Heater(HeaterParams(T_out=400.0), thermo=thermo)
outlet, info = heater(inlet)
info["Q"]  # W

# Rating against a steam utility
heater = Heater(HeaterParams(UA=5000.0, T_utility=450.0), thermo=thermo)
outlet, info = heater(inlet)
info["LMTD"], info["UA_required"]
```

---

### Cooler

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `Cooler`

**Description**: Single-stream cooler that decreases stream temperature using cooling water or refrigeration.

#### Parameters

Same as Heater with appropriate utility temperatures.

#### Governing Equations

Same as Heater, including both energy models and the `DefaultCpWarning`
fallback, with the duty sign reversed: `Q > 0` means heat removed.

```python
from difflow.units.heat_exchanger import Cooler, CoolerParams

cooler = Cooler(CoolerParams(T_out=280.0), thermo=thermo)
outlet, info = cooler(inlet)
```

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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class HeatExchangerParams:
    UA: float = None       # Overall HTC × Area (W/K)
    Cp_hot: float = None   # Hot side heat capacity (J/mol·K), default 75
    Cp_cold: float = None  # Cold side heat capacity (J/mol·K), default 75
    min_approach: float = 10.0  # Minimum temperature approach (K)
```

`CounterCurrentHX(params)` takes only the parameters: there is no thermo
object and no species list, and `UA` can be overridden per call.

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
| `info['UA']` | float | W/K | UA used for the rating |

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

hx = CounterCurrentHX(HeatExchangerParams(UA=50.0, Cp_hot=75.0, Cp_cold=75.0))

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

<!-- doc-test: skip: signature listing (a class sketch), not an executable example -->
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

### ShellAndTubeHX

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `ShellAndTubeHX`

**Description**: Multi-pass shell-and-tube exchanger: counter-current duty reduced by the LMTD correction factor $F$.

#### Process Role

A 1-2N TEMA exchanger is not counter-current. Each tube pass runs with
the shell fluid and the next against it, so part of the area works
against a smaller driving force than the terminal temperatures suggest.
The standard allowance for that is the correction factor $F \le 1$
applied to the counter-current LMTD, and this unit is `CounterCurrentHX`
with that factor in place. Use it when the geometry is a real
shell-and-tube bundle rather than an idealised two-stream exchanger; use
`CounterCurrentHX` when $F$ would be 1 by construction (a true
counter-current double-pipe or a 1-1 arrangement).

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class ShellAndTubeHXParams:
    UA: float                      # Overall HTC x area (W/K)
    Cp_hot: float = None           # Hot side heat capacity (J/mol/K), default 75
    Cp_cold: float = None          # Cold side heat capacity (J/mol/K), default 75
    min_approach: float = 10.0     # Minimum temperature approach (K)
    n_shell_passes: int = 1        # Shell passes: 1-2, 2-4, ... arrangements
```

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_inlet` | Stream | - | Hot fluid inlet |
| `cold_inlet` | Stream | - | Cold fluid inlet |
| `UA` | float | W/K | Optional override of `params.UA` |

#### Outputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_outlet` | Stream | - | Hot fluid outlet |
| `cold_outlet` | Stream | - | Cold fluid outlet |
| `info['Q']` | float | W | Heat duty |
| `info['F_correction']` | float | - | LMTD correction factor |
| `info['R']`, `info['P_param']` | float | - | The two parameters $F$ depends on |
| `info['F_too_low']` | bool | - | `F < 0.75`: add shell passes or re-split the duty |
| `info['effectiveness']` | float | - | $\epsilon_{CC} F$ |
| `info['LMTD']` | float | K | Counter-current LMTD |

#### Governing Equations

**Corrected duty**:

$$Q = UA \cdot F(P, R) \cdot LMTD_{CC}$$

**Correction-factor arguments** (temperature effectiveness and capacity ratio):

$$P = \frac{T_{c,out} - T_{c,in}}{T_{h,in} - T_{c,in}}, \qquad
R = \frac{T_{h,in} - T_{h,out}}{T_{c,out} - T_{c,in}}$$

**Correction factor** (1-2N TEMA, Bowman-Mueller-Nagle):

$$F = \frac{\sqrt{R^2+1}\,\ln\!\frac{1-P}{1-RP}}
{(R-1)\,\ln\!\frac{2-P(R+1-\sqrt{R^2+1})}{2-P(R+1+\sqrt{R^2+1})}}$$

At $R = 1$ that expression is $0/0$; `lmtd_correction_factor` takes the
L'Hopital limit there, so $F$ and its derivative are finite for balanced
flows instead of producing a `nan` in the middle of the usual operating
range.

$F$ falls as $P$ rises: a multi-pass exchanger asked for a close
approach loses area effectiveness quickly, which is why `F_too_low` is
reported rather than silently accepted. The conventional design rule is
to keep $F > 0.75$ and add shell passes otherwise (`n_shell_passes`,
which re-maps $P$ to the equivalent single-shell value before the
formula above).

#### Example Usage

```python
from difflow import make_stream
from difflow.units.heat_exchanger import ShellAndTubeHX, ShellAndTubeHXParams

hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=500.0, Cp_hot=75.0, Cp_cold=75.0))

hot = make_stream({'water': 2.0}, T=400.0, P=101325.0)
cold = make_stream({'water': 1.0}, T=300.0, P=101325.0)

hot_out, cold_out, info = hx(hot, cold)
print(f"Q = {float(info['Q'])/1000:.2f} kW")
print(f"F = {float(info['F_correction']):.3f}")
```

---

### EnthalpyCounterCurrentHX

**Location**: `difflow/units/heat_exchanger.py`

**Class**: `EnthalpyCounterCurrentHX`

**Description**: Counter-current exchanger closed on real, flash-based stream enthalpies, so it stays correct through a phase change.

#### Process Role

`CounterCurrentHX` assumes one constant $C_p$ per side, which makes its
effectiveness-NTU solution closed-form and its answer wrong wherever the
heat capacity is not constant --- above all where a stream boils or
condenses, since latent heat is an infinite apparent $C_p$ that a
constant-$C_p$ model has no way to represent. This unit closes an
enthalpy balance per side instead, using a thermo object's two-phase
stream enthalpy, and therefore matches an equation-oriented exchanger
(IDAES's, say) through the phase change.

Use it for condensers, reboiler-side service, cryogenic/NGL duty, and
anywhere a vapour fraction changes across the exchanger. It costs a
coupled solve rather than a formula, so prefer `CounterCurrentHX` for
single-phase service where the two agree.

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class EnthalpyHXParams:
    UA: float = None       # Overall HTC x area (W/K)
    max_iter: int = 80     # Iterations of the outer fixed point on Q
    damping: float = 0.5   # Damping of the Q update (0 < d <= 1)
```

The constructor takes the thermo alongside the params --- `EnthalpyCounterCurrentHX(params, thermo)` --- and the thermo must provide
`stream_enthalpy_flash(flows, T, P)`. A `CubicThermo` built from an
`IdealThermo` and a `PengRobinson`/`SRK` EOS does
([Thermodynamics](thermodynamics.md)).

#### Inputs

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `hot_inlet` | Stream | - | Hot fluid inlet |
| `cold_inlet` | Stream | - | Cold fluid inlet |
| `UA` | float | W/K | Optional override of `params.UA` |

#### Outputs

Same keys as `CounterCurrentHX` where they mean the same thing:
`info['Q']`, `info['LMTD']`, the four terminal temperatures,
`info['approach']` (the smaller of the two terminal differences) and
`info['flow_arrangement'] == 'counter_current_enthalpy'`.

#### Governing Equations

**Enthalpy balance per side** (not a $\dot{m} C_p \Delta T$ balance):

$$H_{h,out} = H_{h,in} - Q, \qquad H_{c,out} = H_{c,in} + Q$$

**Heat transfer**:

$$Q = UA \cdot LMTD(T_{h,in}, T_{h,out}, T_{c,in}, T_{c,out})$$

The unknowns $(Q, T_{h,out}, T_{c,out})$ are coupled: $Q$ sets the
outlet enthalpies, the enthalpies set the outlet temperatures, and the
temperatures set the LMTD that sets $Q$. The unit solves it as a damped
fixed point on $Q$ with a one-dimensional enthalpy inversion per side
(enthalpy is monotone in $T$, so each inversion is a well-posed root
find). Every solve is an `optimistix` root find, so gradients come from
the implicit function theorem at the converged result rather than from
differentiating the iteration.

#### Example Usage

```python
from difflow import make_stream
from difflow.eos import PengRobinson, CriticalProperties
from difflow.thermo import IdealThermo, CubicThermo, SpeciesData
from difflow.units.heat_exchanger import (
    EnthalpyCounterCurrentHX, EnthalpyHXParams,
)

species = {
    "propane": SpeciesData(name="propane", MW=44.10,
                           Cp_coeffs=(73.0, 0.0, 0.0, 0.0),
                           Hvap_coeffs=(18000.0, 0.38, 369.8),
                           antoine_coeffs=(13.72, 1872.5, -25.16)),
    "butane": SpeciesData(name="butane", MW=58.12,
                          Cp_coeffs=(98.0, 0.0, 0.0, 0.0),
                          Hvap_coeffs=(22000.0, 0.38, 425.1),
                          antoine_coeffs=(13.98, 2292.4, -27.86)),
}
crit = {
    "propane": CriticalProperties(name="propane", Tc=369.8, Pc=4.25e6,
                                  omega=0.152, MW=44.10),
    "butane": CriticalProperties(name="butane", Tc=425.1, Pc=3.80e6,
                                 omega=0.200, MW=58.12),
}
thermo = CubicThermo(IdealThermo(species), PengRobinson(crit))

hot = make_stream({"propane": 1.0, "butane": 1.0}, T=400.0, P=3e5)
cold = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=3e5)

hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=200.0), thermo)
hot_out, cold_out, info = hx(hot, cold)
print(f"Q = {float(info['Q'])/1000:.2f} kW")
```

#### Design Considerations

- **Share the thermo object.** The coupled solve is JIT-compiled and
  cached on the identity of the thermo, so passing the same `CubicThermo`
  to every exchanger compiles once instead of once per unit. The first
  call takes seconds; later ones do not.
- **`damping` is a convergence knob, not a model parameter.** Gradients
  are exact at the fixed point regardless of its value, so lowering it
  costs iterations and nothing else.

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
design = design_heat_exchanger(Q=100000.0, T_hot_in=400.0, T_hot_out=350.0,
                               T_cold_in=300.0, T_cold_out=340.0, U=500.0)
design['UA'], design['A'], design['LMTD']   # W/K, m², K

# A 1-2N shell-and-tube: A = Q / (U F LMTD), F from `lmtd_correction_factor`
st = design_heat_exchanger(Q=100000.0, T_hot_in=400.0, T_hot_out=350.0,
                           T_cold_in=300.0, T_cold_out=340.0, U=500.0,
                           flow_config="shell_and_tube", n_shell_passes=2)
st['F'], st['A']
```

`design_heat_exchanger` and `size_heat_exchanger` take `flow_config` =
`"counter_current"`, `"co_current"` or `"shell_and_tube"` (with
`n_shell_passes`, shells in series); any other value raises `ValueError`.
The result always carries `F` (1 for the pure flow configurations). For a
shell-and-tube, `size_heat_exchanger` inverts the exact 1-2N effectiveness.
A `LowCorrectionFactorWarning` is issued when `F < 0.75` (the usual design
floor) or `F = 0`, which means the duty is past what that many shells can
deliver at any area (`A` is `inf`); add shells. The warning is skipped for
traced values, so the functions stay `jit`/`grad` safe: read `F` from the
result there.

**Phase change in the constant-Cp units.** `CounterCurrentHX`, `CoCurrentHX`,
`CrossFlowHX`, `ShellAndTubeHX` and `Heater`/`Cooler` without a `thermo` carry
no latent heat, so a condensing or vaporising stream is badly mis-rated
(pentane vapour condensing against cooling water comes out 5.5-fold low).
They now raise a `PhaseChangeWarning` when a stream's inlet-to-outlet
temperature span crosses the boiling point, at the stream pressure, of a
database species making up at least 1% of its flow, and point to
`EnthalpyCounterCurrentHX`. The boiling point is the Antoine equation
inverted (species without Antoine coefficients, or a result outside the
equation's range, are not checked). It is an advisory that runs only on
concrete values, never under `jit`/`grad`, so it cannot change a result or a
gradient; silence it with `warnings.simplefilter("ignore", PhaseChangeWarning)`.

(heat-transfer-module)=
### Heat-Transfer Coefficients

The exchangers above take `U` or `UA` as a number. `difflow.heat_transfer`
(pure JAX functions, SI, differentiable, **not** a palette operation) computes
that number from geometry, flow and fluid properties. Properties are inputs
(`k`, `mu`, `Cp`, `rho`), as in [`difflow.fluids`](#fluids-module), which also
supplies the Darcy friction factor Gnielinski needs. Everything is traced:
`jax.grad` of `U` with respect to tube velocity or diameter works, which is
what the velocity and insulation-thickness optimizations use
(`examples/45_heat_transfer_coefficients.ipynb`).

| Group | Functions |
|---|---|
| Dimensionless groups | `reynolds`, `prandtl`, `nusselt_to_h`, `grashof`, `rayleigh` |
| Conduction | `slab_resistance`, `cylinder_resistance`, `sphere_resistance`, `composite_wall`, `critical_insulation_radius`, `fin_efficiency_straight` |
| Tube side | `sieder_tate_laminar`, `dittus_boelter`, `sieder_tate`, `gnielinski`, `internal_nusselt`, `internal_h` |
| External flow | `churchill_bernstein` (cylinder), `flat_plate_nusselt` |
| Natural convection | `churchill_chu_vertical_plate`, `churchill_chu_horizontal_cylinder` |
| Phase change | `nusselt_film_condensation`, `rohsenow_heat_flux`, `rohsenow_superheat`, `mostinski`, `critical_heat_flux` |
| Fouling | `difflow.fouling` (see [Threshold Fouling](#threshold-fouling)) |
| Overall | `overall_U`, `FOULING_RESISTANCES`, `TYPICAL_U_RANGES`, `typical_U_range` |

**Tube side.** `internal_h(Re, Pr, k, D, L=None, mu_ratio=1, rel_roughness=0,
method="gnielinski", heating=True, boundary="T")` returns $h=\mathrm{Nu}\,k/D$:

* $\mathrm{Re}\le2300$: fully developed $\mathrm{Nu}=3.66$ (constant wall
  temperature) or $4.36$ (`boundary="q"`, constant flux); with `L` given, the
  Sieder-Tate entry form $1.86(\mathrm{Re\,Pr}\,D/L)^{1/3}(\mu_b/\mu_w)^{0.14}$
  joined to it as $(\mathrm{Nu}_{fd}^3+\mathrm{Nu}_{en}^3)^{1/3}$ (a smooth
  combination chosen here, not a published correlation).
* $\mathrm{Re}\ge4000$: Gnielinski
  $\mathrm{Nu}=\frac{(f/8)(\mathrm{Re}-1000)\mathrm{Pr}}{1+12.7\sqrt{f/8}\,(\mathrm{Pr}^{2/3}-1)}$
  with $f$ from `fluids.friction_factor`, or `dittus_boelter`
  ($0.023\,\mathrm{Re}^{0.8}\mathrm{Pr}^{0.4/0.3}$) or `sieder_tate`
  ($0.027\,\mathrm{Re}^{0.8}\mathrm{Pr}^{1/3}(\mu_b/\mu_w)^{0.14}$).
* 2300 to 4000: a C2 smootherstep in $\ln\mathrm{Re}$ between the two. There is
  no Python branching, so `Re` may be traced; $h$ and its first two derivatives
  are continuous across the whole range, and the gradient is exactly zero
  where the blend starts. Real transitional flow is intermittent and no
  correlation is accurate there; the blend is a smooth interpolation, not a
  model of transition.

**Viscosity correction (opt-in, exponents unverified).** As published,
Gnielinski and Dittus-Boelter carry no wall-viscosity factor, so by default
`mu_ratio` has no effect on a turbulent tube side (it enters the laminar entry
term and `sieder_tate` only). `internal_h(..., viscosity_correction=True)` (and
`ShellAndTubeDesignParams(tube_viscosity_correction=True)`) multiplies the
Gnielinski or Dittus-Boelter value by $(\mu_b/\mu_w)^n$, with $n=0.11$ when the
fluid is heated and $0.25$ when it is cooled (`VISCOSITY_EXPONENT_HEATING`,
`VISCOSITY_EXPONENT_COOLING`): the liquid form usually attributed to Petukhov
(the VDI Heat Atlas / Gnielinski variant is $(\mathrm{Pr}/\mathrm{Pr}_w)^{0.11}$;
the viscosity ratio stands in for it here). **These exponents are from memory
and were not checked against Petukhov, Gnielinski or the VDI Heat Atlas**, which is
why the factor is off by default and no existing result changes. Check them
before a real design. `sieder_tate` already has its own 0.14 factor and gets no
second one.

```python
from difflow import heat_transfer as ht

Re, Pr = 2.0e4, 6.0                                  # illustrative
base = ht.internal_nusselt(Re, Pr, mu_ratio=1.5)     # default: mu_ratio ignored (turbulent)
corr = ht.internal_nusselt(Re, Pr, mu_ratio=1.5, heating=True, viscosity_correction=True)
float(corr / base)                                   # 1.5**0.11
```

**Overall U.** `overall_U(h_i, h_o, D_i, D_o, k_wall, R_fi=0, R_fo=0,
basis="outer")`:

$$\frac1{U_o}=\frac1{h_o}+R_{f,o}+\frac{D_o\ln(D_o/D_i)}{2k}+R_{f,i}\frac{D_o}{D_i}+\frac{D_o}{D_i\,h_i}$$

and the same referred to the inner area with `basis="inner"`
($U_oA_o=U_iA_i$). Wall conduction uses `cylinder_resistance`'s logarithmic
form; `k_wall=jnp.inf` removes it.

```python
import jax
from difflow import heat_transfer as ht, CounterCurrentHX, HeatExchangerParams

rho, mu, Cp, k = 998.0, 1.0e-3, 4180.0, 0.60    # water, tube side (SI)
h_shell, area = 2000.0, 12.0                    # shell-side h (W/m^2/K), area (m^2)

def U_outer(v, D_i=0.0229):
    Re = ht.reynolds(rho, v, D_i, mu)
    h_i = ht.internal_h(Re, ht.prandtl(Cp, mu, k), k, D_i, L=4.0)
    return ht.overall_U(h_i, h_shell, D_i, D_i + 0.0025, 16.0, R_fi=1e-4)

dU_dv = jax.grad(U_outer)(1.5)                 # W/m^2/K per (m/s)
UA = U_outer(1.5) * area                       # a plain number for the exchanger
hx = CounterCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.3, Cp_cold=75.3))
```

The exchangers still take a float (or array) `U`/`UA`; a callable `U(hot, cold,
params)` is **not** implemented -- compute `U` first, as above (it is already
differentiable, so `jax.grad` flows through to the exchanger).

**Other correlations.**

| Function | Correlation | Source (Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*) |
|---|---|---|
| `churchill_bernstein` | cylinder in cross flow, all $\mathrm{Re\,Pr}>0.2$ | Eq. 7.57 |
| `flat_plate_nusselt` | $0.664\mathrm{Re}^{1/2}\mathrm{Pr}^{1/3}$; $0.037\mathrm{Re}^{4/5}\mathrm{Pr}^{1/3}$; mixed $(0.037\mathrm{Re}^{4/5}-871)\mathrm{Pr}^{1/3}$ joined smoothly over $\mathrm{Re}_L$ 5e5-5.5e5 | Eqs. 7.31, 7.41, 7.38 |
| `churchill_chu_*` | vertical plate, horizontal cylinder, all Ra | Eqs. 9.26, 9.34 |
| `nusselt_film_condensation` | $h=C[g\rho_l(\rho_l-\rho_v)k_l^3h_{fg}'/(\mu_l\,\Delta T\,L)]^{1/4}$, $C=0.943$ plate, $0.729$ horizontal tube, $h_{fg}'=h_{fg}+0.68C_p\Delta T$; $N$ tubes in a column $h_N=h_1N^{-1/4}$ | Eqs. 10.26, 10.40, 10.42 |
| `rohsenow_heat_flux`, `rohsenow_superheat` | $q''=\mu_lh_{fg}[g(\rho_l-\rho_v)/\sigma]^{1/2}[C_p\Delta T_e/(C_{sf}h_{fg}\mathrm{Pr}^n)]^3$ and its closed-form inverse | Eq. 10.5 |
| `mostinski` | $h=0.00417\,P_c^{0.69}q^{0.7}F_p$ ($P_c$ in kPa) | Mostinski (1963); recalled, see below |
| `critical_heat_flux` | Zuber, $C\,h_{fg}\rho_v^{1/2}[\sigma g(\rho_l-\rho_v)]^{1/4}$, $C=\pi/24$ | Eq. 10.7 (constant is a parameter) |

**Limitation: `critical_heat_flux` is the single-surface value.** A tube
bundle (kettle reboiler) has a lower critical flux because vapour from the
lower tubes blankets the upper ones; the published correction is Palen's
bundle factor. It is **not implemented**: its form and constants could not be
checked against a source and none were invented. Treat the Zuber value as an
upper bound for a bundle and derate it with a sourced bundle correction.

Rohsenow's $C_{sf}$ defaults to 0.013 (water on polished copper/platinum),
recalled from memory of Incropera Table 10.1: check it for real work. Not
implemented: annular fins, tube banks (Zukauskas), the Bell-Delaware
shell-side method (Kern is in [`difflow.shell_and_tube`](#shell-and-tube-design)),
radiation, transient conduction.

**What is validated (and what is not).** No textbook worked-example number is
quoted, because none could be reproduced with confidence. Each correlation is
checked against an independent hand/NumPy evaluation of the stated formula
(1e-10 or better). Where a derivation exists the constant itself is checked:
Nusselt's 0.943 against a numerical integral of the film-thickness solution,
the cylinder resistance against numerical integration of Fourier's law, the
fin efficiency against the exact convecting-tip solution, the critical
insulation radius as the stationary maximum of the heat loss. Looser checks
are labelled sanity bands in `tests/test_heat_transfer.py`: Gnielinski vs
Dittus-Boelter within 20% at Re 1e4 to 1e5; Mostinski vs Rohsenow within 30%
for water at 1 atm; Zuber's water CHF within 5% of the commonly quoted 1.1
MW/m². The published shell-and-tube `overall_U` example and the textbook
steam and Rohsenow examples are **not** reproduced; `overall_U` is checked
against the series-resistance network, its limits and $U_oA_o=U_iA_i$.

**Typical fouling resistances** (`FOULING_RESISTANCES`, m² K/W; transcribed
from memory of Incropera Table 11.1, which cites TEMA; **not verified**, use
your own standard for design):

| Service | $R_f$ |
|---|---|
| Seawater, below 50 °C | 0.0001 |
| Seawater, above 50 °C | 0.0002 |
| Treated boiler feedwater, above 50 °C | 0.0002 |
| Fuel oil | 0.0009 |
| Refrigerating liquids | 0.0002 |
| Steam, oil-free | 0.0001 |

**Typical overall U** (`TYPICAL_U_RANGES`, W/m²/K; from memory of Incropera
Table 11.2, **not verified**; a sanity band for a computed `U`, not a design
value):

| Service | U range |
|---|---|
| Water to water | 850-1700 |
| Water to oil | 100-350 |
| Steam condenser, water in tubes | 1000-6000 |
| Ammonia condenser, water in tubes | 800-1400 |
| Alcohol condenser, water in tubes | 250-700 |
| Finned tube, water in tubes, air in cross flow | 25-50 |

(shell-and-tube-design)=
### ShellAndTubeDesign

**Location**: `difflow/shell_and_tube.py` (a library, **not** a palette
operation: it takes mass flows and explicit properties and returns a geometry,
like `difflow.heat_transfer`; `examples/46_shell_and_tube_design.ipynb`)

**Description**: Sizes a 1-2N shell-and-tube exchanger. `ShellAndTubeDesign(params)(hot, cold, Q=...)`
(or `T_hot_out=` / `T_cold_out=`; streams are `{"T": K, "m_dot": kg/s}`)
assumes `U`, gets the area from $Q=UAF\,\mathrm{LMTD}$, builds the geometry,
computes the tube-side film coefficient (`heat_transfer.internal_h`, Gnielinski
with `fluids.friction_factor`), the shell-side one by the Kern method, and
a new `U` from `heat_transfer.overall_U` with fouling, and iterates to a fixed
point with `optimistix.fixed_point`, so the area and everything derived from it
are differentiable (implicit-function-theorem gradients) in baffle spacing,
tube velocity, passes, flows, fouling and properties.

**Equations**

$$A=\frac{Q}{U\,F\,\mathrm{LMTD}_{cc}},\qquad
\frac1{U_o}=\frac1{h_o}+R_{fo}+\frac{d_o\ln(d_o/d_i)}{2k_w}+R_{fi}\frac{d_o}{d_i}+\frac{d_o}{d_ih_i}$$

Kern shell side (segmental baffles, 25 % cut; Kern 1950, Coulson & Richardson
Vol 6 Eqs. 12.21-12.26):

$$D_e=\frac{1.10}{d_o}\left(P_t^2-0.917\,d_o^2\right)\ (\triangle),\quad
D_e=\frac{1.27}{d_o}\left(P_t^2-0.785\,d_o^2\right)\ (\square),\quad
A_s=\frac{(P_t-d_o)D_sl_B}{P_t},\quad G_s=\frac{\dot m_s}{A_s}$$

$$\mathrm{Nu}_s=\frac{h_oD_e}{k}=0.36\,\mathrm{Re}_s^{0.55}\mathrm{Pr}^{1/3}\Big(\frac{\mu}{\mu_w}\Big)^{0.14},\qquad
\Delta P_s=f_s\frac{D_s}{D_e}\frac{L}{l_B}\frac{\rho u_s^2}{2}\Big(\frac{\mu}{\mu_w}\Big)^{-0.14},\quad f_s=e^{0.576-0.19\ln\mathrm{Re}_s}$$

($L/l_B$ is the number of baffle crossings, $N_b+1$; $f_s=8j_f$ of the
Coulson & Richardson chart.) Tube side:

$$\Delta P_t=N_p\Big(f\frac{L}{d_i}+4\Big)\frac{\rho v^2}{2},\qquad v=\frac{\dot m\,N_p}{\rho\,N_t\,\pi d_i^2/4}$$

($f$ Darcy from `fluids.friction_factor`; 4 velocity heads per pass for the
return, Coulson & Richardson use 2.5.) Geometry: $D_b=d_o(N_t/K_1)^{1/n_1}$
(`bundle_diameter`, C&R Table 12.4 constants for $P_t=1.25d_o$), shell
$D_s=D_b+$ `shell_clearance`; `tube_count(shell_ID, ...)` is the Kakac form
$N_t=\mathrm{CTP}\,(\pi/4)D_s^2/(\mathrm{CL}\,P_t^2)$. $F$ is the Bowman 1-2N
factor of `lmtd_correction_factor`; `effectiveness_shell_and_tube` is the same
thing in $\varepsilon$-NTU form.

| Function / class | Purpose |
|---|---|
| `equivalent_diameter`, `kern_shell_side`, `kern_shell_pressure_drop`, `kern_friction_factor` | Kern shell side: $D_e,A_s,G_s,\mathrm{Re}_s,h_o,\Delta P_s$ |
| `tube_side_pressure_drop` | friction (`fluids.friction_factor`) plus return losses per pass |
| `tube_count`, `bundle_diameter` | continuous tube-count and bundle-diameter correlations; `BUNDLE_CONSTANTS`, `BWG_WALL_THICKNESS`, `TEMA_TUBE_PITCH`, `tube_inside_diameter` data |
| `ShellAndTubeDesignParams`, `ShellAndTubeDesign` | the design loop; `.rate(...)` re-rates a given geometry |
| `round_design` | integer tube count, standard passes, baffle spacing and shell ID for building |

```python
import jax
from difflow import shell_and_tube as st

p = st.ShellAndTubeDesignParams(
    tube_OD=0.01905, tube_ID=0.01483, tube_side="cold",
    rho_hot=780.0, mu_hot=4e-4, k_hot=0.13, Cp_hot=2200.0,      # properties are inputs
    rho_cold=830.0, mu_cold=2.9e-3, k_cold=0.13, Cp_cold=2050.0,
    tube_velocity=1.5, tube_length=None, R_fi=1e-4, R_fo=1e-4,
    dP_tube_max=7e4, dP_shell_max=7e4)
design = st.ShellAndTubeDesign(p)
hot, cold = {"T": 473.0, "m_dot": 5.0}, {"T": 303.0, "m_dot": 18.0}
r = design(hot, cold, T_hot_out=373.0)            # U, area, n_tubes, shell_ID, dP_tube, dP_shell, F, flags
dA = jax.grad(lambda lB, v: design(hot, cold, T_hot_out=373.0, baffle_spacing=lB,
                                   tube_velocity=v, warn=False)["area"], argnums=(0, 1))(0.25, 1.2)
geom = st.round_design(r)                         # buildable geometry ...
rated = design.rate(hot, cold, **geom)            # ... and its re-rating
```

**Two modes for the free geometry.** With `tube_length` the tube count follows from the
area; with `tube_velocity` the tube count follows from the flow and the tube length
from the area. The baffle spacing is `baffle_spacing` (m) or `baffle_spacing_ratio`
times the shell ID. Any of these, `n_tube_passes`, `tube_OD`, `tube_ID`,
`pitch_ratio`, `k_wall` and the fouling resistances can be passed as call
overrides, traced or not, to differentiate with respect to them.

**Warnings and flags.** The result carries `F_too_low` (`F < 0.75`), `F_zero`
(`F = 0`), `dP_tube_exceeded` / `dP_shell_exceeded` (against `dP_tube_max` /
`dP_shell_max`), `converged`, `tube_laminar` (`Re_tube < 2300`), `shell_laminar`
(`Re_shell < 2000`, below the range of Kern's power law) and
`shell_Re_out_of_range` (outside $2\times10^3<\mathrm{Re}_s<10^6$, where the
Kern coefficient is an extrapolation). Each also raises a warning for concrete
values (never under `jit`/`grad`). A low or zero $F$ raises
`LowCorrectionFactorWarning` (a `UserWarning` from `difflow.units.heat_exchanger`,
the class the HX design functions use); $F=0$ is the infinite-area limit of
`lmtd_correction_factor` (a temperature cross these shells cannot make, issue
\#419), so the area is `inf` and the other warnings are suppressed.

**Shells in series.** `n_shell_passes=N` (default 1) puts $N$ identical shells in
series, each carrying the full flow of both streams, with
$F=$ `lmtd_correction_factor(R, P, N)`. A cross that one shell cannot make
($F=0$) becomes feasible with two. `area`, `UA` and the pressure drops are
totals over the shells; `n_tubes`, `tube_length`, `shell_ID`, `bundle_ID` and
`baffle_spacing` are per shell (`area_per_shell` is reported), and
`round_design` and `rate` work on that per-shell geometry with the same
`n_shell_passes`.

```python
import warnings
from difflow import shell_and_tube as st

def params(N):
    return st.ShellAndTubeDesignParams(
        tube_OD=0.01905, tube_ID=0.01483, rho_hot=780.0, mu_hot=4e-4, k_hot=0.13, Cp_hot=2050.0,
        rho_cold=830.0, mu_cold=2.9e-3, k_cold=0.13, Cp_cold=2050.0,   # illustrative properties
        tube_velocity=1.2, tube_length=None, n_shell_passes=N)

hot, cold = {"T": 340.0, "m_dot": 5.0}, {"T": 300.0, "m_dot": 5.0}   # hot 340 -> 316 K, cold 300 -> 324 K
Q = 5.0 * 2050.0 * 24.0
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    one = st.ShellAndTubeDesign(params(1))(hot, cold, Q=Q)     # F = 0: LowCorrectionFactorWarning
two = st.ShellAndTubeDesign(params(2))(hot, cold, Q=Q, warn=False)
float(one["F"]), bool(one["F_zero"]), float(two["F"]), float(two["area"])   # 0, True, F > 0, finite area
```

**Non-unique fixed point.** With a fixed tube length, more area means more
tubes, slower tube flow and a lower $h_i$: positive feedback, so the loop can
have a second, large-area, laminar solution. In the test case a start at
$U\le100$ lands on 211 m$^2$ at $\mathrm{Re}_t\approx1500$, against 13 m$^2$ at
$\mathrm{Re}_t\approx23000$ from any typical start. It is a genuine root of the
same equations; `tube_laminar` and a warning flag it, and fixing
`tube_velocity` removes the feedback.

**Integer quantities.** Tube count, tube passes and the number of baffles are
continuous in the design (`n_baffles` $=L/l_B-1$; passes interpolate the
$K_1,n_1$ and CTP tables linearly in $\ln N_p$ / $N_p$, a smoothing choice, not a
published correlation) so that `jax.grad` and gradient-based optimizers work.
To build the result: `round_design(result)` rounds the tube count **up**, the
passes to the nearest standard even count (1, 2, 4, 6, 8), the baffle spacing
**down** and the shell ID **up** to a step (illustrative, not TEMA), and
`design.rate(hot, cold, **geometry)` re-rates that geometry. Rounding changes
the velocities, $h_i$, $h_o$ and both pressure drops, so re-check the duty margin and the
limits after rounding; in the example notebook the rounded 7 m, 54-tube,
single-pass design re-rates to 1.01 times the target duty.

**Phase change (simple mode).** `hot_isothermal=True` (condensing) or
`cold_isothermal=True` (boiling) makes that side isothermal ($F=1$), and `shell_h`
supplies its film coefficient: a number, or a callable of a dict, e.g. built on
`heat_transfer.nusselt_film_condensation` or the boiling correlations. The dict has
the geometry (`tube_OD`, `n_tubes`, `tube_length`, `shell_ID`, `area` per shell,
`area_total`) **and the wall state from the current $U$ iterate**: `q_flux`
($Q/A$), `T_wall` (the shell-side film's wall surface, outside any shell-side
fouling), `dT_film` ($|T_{shell}-T_{wall}|$), `T_shell` and `T_tube` (mean bulk
temperatures). The wall temperature is $T_{wall}=T_{tube,eff}\pm q''R_{chain}$ with
the tube-side film, its fouling, the wall and the shell fouling in $R_{chain}$, and
the tube-side temperature chosen so that $q''=U\,F\,\mathrm{LMTD}$; the loop is
closed inside the same `optimistix.fixed_point` as $U$ (and, for a callable, in
`rate()` on the heat flux), so at convergence $T_{shell}-T_{wall}=q''/h_o$ and the
design needs no outer loop on $\Delta T_{film}$, and is still differentiable. A
callable that reads only the geometry keys is unaffected by the extra keys.
Shell-side $\Delta P$ is then not modelled (reported as 0).

```python
import jax.numpy as jnp
from difflow import heat_transfer as ht, shell_and_tube as st

def shell_h(g):                    # condensing steam on a column of horizontal tubes (illustrative properties)
    return ht.nusselt_film_condensation(k_l=0.6, rho_l=960.0, rho_v=0.6, mu_l=2.8e-4, h_fg=2.26e6,
                                        Cp_l=4216.0, dT=g["dT_film"], L=g["tube_OD"],
                                        n_tubes=jnp.sqrt(g["n_tubes"]))

p = st.ShellAndTubeDesignParams(
    tube_OD=0.01905, tube_ID=0.01483, rho_hot=960.0, mu_hot=2.8e-4, k_hot=0.6, Cp_hot=4216.0,
    rho_cold=830.0, mu_cold=2.9e-3, k_cold=0.13, Cp_cold=2050.0, tube_length=3.0,
    hot_isothermal=True, shell_h=shell_h, tol=1e-13, R_fi=1e-4, R_fo=1e-4)
r = st.ShellAndTubeDesign(p)({"T": 373.15, "m_dot": 1.0}, {"T": 303.0, "m_dot": 18.0}, Q=6.0e5)
float(r["q_flux"] / r["h_o"]), float(r["area"])     # the film dT the loop settled on, and the area
```
 Desuperheating and
subcooling zones are not split out; for a rigorous two-phase *rating* use
`EnthalpyCounterCurrentHX`.

**Design and rating agree.** `design.rate(...)` for the designed geometry returns the
design duty to 1e-8, and `ShellAndTubeHX(UA=U*A, ...)` returns the same duty
and outlet temperatures (`tests/test_shell_and_tube.py`). Making that true required
two fixes to the existing rating code (see the changelog): `lmtd_correction_factor`
returned its $R=1$ value for every $|R-1|>0.075$, and `ShellAndTubeHX` took
$Q=F\,Q_{cc}(UA)$, a few percent off $Q=UAF\,\mathrm{LMTD}$; it now uses the exact
1-2N (and $n$-shell) effectiveness `effectiveness_shell_and_tube`.

**What is validated (and what is not).**

* **Not reproduced: the classic Kern kerosene-crude example** (Kern 1950;
  Coulson & Richardson Vol 6, Ch. 12 worked examples). The inputs and the
  printed answers could not be reproduced from memory with confidence and no copy of
  either book was available, so no agreement with them is claimed. The issue's
  criterion "within 5 % of the textbook example" is therefore **open**: it needs
  someone with the book to enter the example's inputs and compare
  (`tests/test_shell_and_tube.py` has a hand-calculation harness that takes
  the same arguments).
* Checked: every Kern equation ($D_e$, $A_s$, $G_s$, $\mathrm{Re}_s$, $h_o$, $\Delta P_s$) and the
  tube-side equations (velocity, Re, Gnielinski $h_i$ with a brentq-solved
  Colebrook factor, friction and return $\Delta P$) against a plain-Python hand
  calculation (1e-12 to 1e-6); the whole loop against an independent plain-Python
  re-implementation with substitution on $U$ (2e-6, both modes); $F$
  against the closed form; the effectiveness against $Q=UAF\,\mathrm{LMTD}$
  solved by brentq; gradients by `check_grads` and central finite differences.
* **Item list of issue \#435.** Wall-state `shell_h`, the optional viscosity
  correction, the core fouling model, shells in series, the shell-`Re` flag and
  the $F=0$ warning are implemented and tested (`tests/test_shell_and_tube_gaps.py`,
  `tests/test_fouling.py`); the viscosity exponents are unverified (above). The
  Palen bundle factor for boiling CHF is **not** implemented (no sourced
  constants), and the $K_1,n_1$ and tube-count constants remain **unverified**: a
  pass in the issue's work compared the triangular-pitch rows with an independent
  recollection (1 pass 0.319/2.142, 2 passes 0.249/2.207, 4 passes 0.175/2.285,
  6 passes 0.0743/2.499, 8 passes 0.0365/2.675), which agrees, but that is memory
  against memory, not a check against Sinnott's Table 12.4.
* **Recalled from memory and not verified against the books:** the Kern
  constants (0.36, 0.55 for $\mathrm{Nu}$; 0.576, 0.19 for $f_s$, a fit to Kern's chart, not
  the chart), the $D_e$ rounded constants (1.10/0.917/1.27/0.785 are checked
  against the exact pitch-cell hydraulic diameters to 3 %), the $K_1,n_1$ bundle
  constants (sanity-banded against a brute-force tube-lattice count, 15 %), the
  CTP/CL tube-count constants, the BWG wall table and the TEMA pitch table.
  Kern ignores leakage and bypass streams (Bell-Delaware, out of scope,
  handles them): treat it as a screening-level method; its accuracy was not
  assessed here.

**References**: Kern, D.Q. (1950), *Process Heat Transfer*, McGraw-Hill;
Coulson & Richardson, *Chemical Engineering* Vol. 6 (Sinnott), Ch. 12;
Kakac, Liu & Pramuanjaroenkij, *Heat Exchangers: Selection, Rating, and Thermal
Design*; Bowman, Mueller & Nagle, Trans. ASME 62 (1940) 283; Kays & London,
*Compact Heat Exchangers*; Gnielinski, Int. Chem. Eng. 16 (1976) 359.

### Threshold Fouling

**Location**: `difflow/fouling.py` (a library of pure JAX functions, **not** a palette
operation; the core counterpart of the crude-preheat fouling model in
`difflow_refinery.preheat.fouling`, which is unchanged and has its own illustrative
constants).

**Description**: The Ebert-Panchal threshold model: deposition is Arrhenius in the
film temperature, removal goes with the wall shear stress, and below the
threshold a surface does not foul.

$$\frac{dR_f}{dt}=\alpha\,\mathrm{Re}^{-0.66}\mathrm{Pr}^{-0.33}\,e^{-E/(RT_{film})}-\gamma\,\tau_w,
\qquad \tau_w=\frac f8\rho v^2$$

$\alpha$, $\gamma$ and $E$ are **user-supplied and fluid-specific** (no defaults),
fitted to plant or slip-stream data; **every number in the example below is
illustrative**, chosen for the arithmetic, not for any fluid. The $0.55$ film
weight of `film_temperature` is the refinery plugin's convention, unverified, and
the form of the equation is recalled from the literature (Ebert and Panchal, 1995)
and not re-checked against the paper. For the same constants `ebert_panchal_rate`
returns the refinery function's value (a test, skipped when the plugin is not
installed).

| Function | Purpose |
|---|---|
| `ebert_panchal_rate(Re, Pr, T_film, tau_w, alpha, gamma, E, beta=-0.66, floor_at_zero=False)` | $dR_f/dt$ (m$^2$ K/W/s); negative below the threshold |
| `wall_shear_stress(f, rho, v)` | $\tau_w=f\rho v^2/8$ from the Darcy factor |
| `film_temperature(T_bulk, T_wall)` | $T_b+0.55(T_w-T_b)$ |
| `threshold_wall_shear`, `threshold_film_temperature` | the shear (Pa) or film temperature (K) at which the rate is zero |
| `fouling_time(R_target, rate, R_f0=0)` | time to a target resistance at a constant rate (`inf` if the rate is not positive) |
| `integrate_fouling(rate_fn, t_end, R_f0, n_steps)` | RK4 integration of a rate that depends on $R_f$; differentiable in whatever `rate_fn` closes over |

```python
import jax
from difflow import fluids, fouling

alpha, gamma, E = 2.0e-3, 4.0e-13, 48.0e3     # ILLUSTRATIVE constants, not fitted to any fluid
rho, mu, D, Pr = 800.0, 5.0e-3, 0.02, 10.0    # illustrative tube-side properties

def rate(v, T_film=600.0):
    Re = fluids.reynolds_number(rho, v, D, mu)
    tau = fouling.wall_shear_stress(fluids.friction_factor(Re, 0.0), rho, v)
    return fouling.ebert_panchal_rate(Re, Pr, T_film, tau, alpha, gamma, E)

r = rate(1.0)                                  # m^2 K/W per s
days = fouling.fouling_time(3.0e-4, r) / 86400.0
drdv = jax.grad(rate)(1.0)                     # d(rate)/d(velocity): faster flow, less fouling
```

---

(evaporators)=
## Evaporators

An evaporator concentrates a solution of a **non-volatile solute** by boiling
off the solvent (water) with steam: caustic, sugar, pulp liquor, brine,
pharma concentrates. What separates it from a `Heater` followed by a `Flash`
is that the solute raises the boiling point, the heat goes through a surface
with a given `U` and area, and in a multiple-effect train the vapor of one
effect is the heating medium of the next. Source: `difflow/units/evaporator.py`.

**How the solute is represented.** The solute is an ordinary difflow species in
the stream (`F_NaOH`, in mol/s) whose **vapor flow is exactly zero**: the vapor
outlet carries the same species keys with the solute at `0.0`. That is the
difflow-native form of "vapor pressure zero" and needs no special case anywhere
(nothing divides by a vapor pressure or a vapor flow, so gradients stay finite).
The alternative, a mass-fraction field riding on the stream, would not survive a
`Mixer` or a `Splitter`. Mass fractions are computed from the molar flows and
the `solute_MW` on the params; `x` is always the **solute mass fraction**.

**Steam properties** are self-contained correlations in the module
(IAPWS-IF97 region-4 for `Psat(T)`/`Tsat(P)`, fits for `h_f` and `h_fg` valid
20-200 C), not the `IdealThermo` Antoine/Watson water entry, whose latent heat is
too coarse for a steam economy.

**Boiling-point rise** (`bpr_model`, or `boiling_point_rise(x, P, model)`):

| Model | What it is | Status |
|-------|------------|--------|
| `ideal` | Raoult's law with unit activity: the solution boils where `x_w Psat(T) = P`; `vant_hoff_i` counts particles per solute | exact for the ideal solution |
| `colligative` | `dT = R Tw^2 i m / lambda`, the dilute limit | exact in the limit |
| `duhring` | your own lines `T_soln = a(x) + b(x) T_water` (`duhring_x/a/b`) | your data |
| `polynomial` | BPR in K as a polynomial in `x` (`bpr_coeffs`) | your data |
| `naoh`, `nacl` | built-in 1 atm tables turned into Duhring lines | **unverified, see below** |
| `sucrose` | `1.78 x + 6.22 x^2` K (Geankoplis Ex. 8.4-1) | **recalled, unverified** |
| callable | `Evaporator(params, bpr_fn=f)`, `f(x, T_water, P) -> K` | yours |

```{warning}
The built-in `naoh`, `nacl` and `sucrose` data are **approximate recollections,
not transcriptions of the Duhring chart** (McCabe/Perry). Only a couple of
anchor points per table are firm (NaOH 50 wt % boiling near 143 C at 1 atm; the
saturated NaCl solution near 108.7 C). Selecting one raises
`UnverifiedDataWarning`. The acceptance criterion "NaOH BPR matches the cited
chart values" is therefore **not met**: read your chart and pass it through
`duhring=` or `bpr_fn`. The tabulated models are piecewise linear in `x`.
```

**Solution enthalpy** is a Cp mixing rule by default
(`(1-x) h_f(T) + x Cp_solute (T - 273.15)`, no heat of dilution); give
`cp_coeffs` (a polynomial in `x`) or `Evaporator(params, enthalpy_fn=f)`,
`f(T, x) -> J/kg`, for a heat of dilution or an enthalpy-concentration chart
(NaOH is the classic case). The vapor leaves at the solution temperature,
superheated by the BPR (`Cp_vapor`), and condensate leaves saturated.

### Evaporator

**Location**: `difflow/units/evaporator.py`

**Class**: `Evaporator(params, *, bpr_fn=None, enthalpy_fn=None)`; call
`concentrate, vapor, info = evap(feed, steam=None)`.

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class EvaporatorParams:
    solute_species: list[str]   # non-volatile species names (required)
    solute_MW: list[float]      # their molar masses, g/mol (required)
    solvent: str = "water"
    U: float = 2000.0           # W/m^2/K
    A: float = None             # m^2   -> rating
    x_product: float = None     # kg/kg -> design (exactly one of A, x_product)
    P_vapor_space: float = 101325.0
    steam_P: float = None       # Pa (or steam_T, K, or pass a steam stream)
    bpr_model: str = "ideal"    # see the table above
    vant_hoff_i: float = 1.0
    ...                         # bpr_coeffs, duhring_*, Cp_solute, cp_coeffs, Cp_vapor
```

Two modes, as for `design_heat_exchanger`: **design** (`x_product` given: steam,
vapor, duty and the required area in closed form) and **rating** (`A` given:
the product concentration is solved, differentiably).

#### Governing Equations

$$x_F F = x_P L,\quad V = F - L,\quad T = T_{sat}(P_v) + \mathrm{BPR}(x_P, P_v)$$

$$Q = L h_L(T, x_P) + V H_V - F h_F,\qquad H_V = h_g(T_w) + c_{p,v}(T - T_w)$$

$$S \lambda_s = Q,\qquad Q = U A (T_s - T),\qquad \text{economy} = V / S$$

`info` holds `steam` (kg/s), `steam_economy` (vapor/steam), `Q`, `T_soln`, `BPR`,
`A`, `x_product`, `V`, `L`, `driving_force`, and two closure residuals,
`energy_balance_error` and `mass_balance_error`.

```python
from difflow import make_stream
from difflow.units import Evaporator, EvaporatorParams

# 10 kg/s of 10 wt % NaOH, concentrated to 30 wt % at 30 kPa with 300 kPa steam
feed = make_stream({"water": 9000 / 18.015, "NaOH": 1000 / 40.0}, T=330.0, P=2e5)
params = EvaporatorParams(
    solute_species=["NaOH"], solute_MW=[40.0], U=1500.0, x_product=0.30,
    P_vapor_space=30e3, steam_P=300e3, bpr_model="ideal", vant_hoff_i=2.0,
)
concentrate, vapor, info = Evaporator(params)(feed)
float(info["steam"]), float(info["steam_economy"]), float(info["A"]), float(info["BPR"])

# Rating the same vessel: give the area, get the concentration back
rating = EvaporatorParams(
    solute_species=["NaOH"], solute_MW=[40.0], U=1500.0, A=float(info["A"]),
    P_vapor_space=30e3, steam_P=300e3, bpr_model="ideal", vant_hoff_i=2.0,
)
_, _, back = Evaporator(rating)(feed)
float(back["x_product"])          # 0.30
```

### MultiEffectEvaporator

**Location**: `difflow/units/evaporator.py`

**Class**: `MultiEffectEvaporator(params, *, bpr_fn=None, enthalpy_fn=None)`; call
`concentrate, vapor, info = mee(feed, steam=None)`.

N effects; the vapor of effect *i* heats effect *i+1*, effect 1 gets the live
steam and the last effect vents at `P_vapor_space`. Per-effect `U` (a list, or one
value for all). Feed arrangements `feed="forward" | "backward" | "parallel"`.

* **Design, equal areas** (`x_product` given): the intermediate effect
  temperatures, the steam rate, the vapor flows and one common area `A` are
  solved so that every effect has the same area, as one Newton solve
  (`optimistix`) of the stacked energy balances, rate equations and the overall
  solute balance (the textbook iteration, done simultaneously). It is
  differentiable with respect to `U`, the feed and the steam pressure by the
  implicit function theorem. For parallel feed the feed split is solved too.
* **Rating** (`A` list given): areas fixed, the product concentration follows.
  Parallel feed uses `feed_split` (default equal).

`MultiEffectEvaporator` with `n_effects=1` agrees with `Evaporator`. `info` has
per-effect `T`, `T_water`, `P`, `BPR`, `V`, `A`, `Q`, `x` (index 0 is effect 1),
the overall `steam`, `steam_economy` (total vapor / steam), `total_area`,
`converged`, `residual_norm` and the closure residuals. The vapor in effect
*i+1*'s steam chest carries the superheat that the BPR of effect *i* puts into
it; condensate leaves each chest saturated.

```python
from difflow.units import MultiEffectEvaporator, MultiEffectEvaporatorParams

mee_params = MultiEffectEvaporatorParams(
    solute_species=["NaOH"], solute_MW=[40.0], n_effects=3, feed="forward",
    U=[2500.0, 2000.0, 1500.0], x_product=0.30, P_vapor_space=30e3,
    steam_P=300e3, bpr_model="ideal", vant_hoff_i=2.0,
)
concentrate, vapor, info = MultiEffectEvaporator(mee_params)(feed)
print("converged:", bool(info["converged"]))
print("steam economy:", round(float(info["steam_economy"]), 2))
print("areas (m^2):", [round(float(a), 1) for a in info["A"]])
print("effect T (K):", [round(float(t), 1) for t in info["T"]])
```

### MechanicalVaporRecompression

**Location**: `difflow/units/evaporator.py`

**Class**: `MechanicalVaporRecompression(params)`; call
`concentrate, condensate, info = mvr(feed)`.

A single effect whose own vapor is compressed (ideal-gas steam, isentropic,
efficiency `eta`, ratio `gamma`) until it condenses `dT_drive` above the boiling
solution and heats the same surface. Design mode (`x_product` given). `info`
reports the compressor work `W_compressor` against the steam it saves
(`steam_without_mvr`, `steam_makeup`, `steam_saved` when `steam_P` is given). The
compression is an ideal-gas calculation, not the EOS `Compressor`; real-gas
steam deviates by a few percent of the work at these pressures.

```python
from difflow.units import MechanicalVaporRecompression, MVRParams

mvr = MechanicalVaporRecompression(MVRParams(
    solute_species=["NaOH"], solute_MW=[40.0], U=1500.0, x_product=0.30,
    P_vapor_space=101325.0, dT_drive=10.0, eta=0.75, steam_P=300e3,
    bpr_model="ideal", vant_hoff_i=2.0,
))
_, condensate, m = mvr(feed)
print("compressor work (kW):", round(float(m["W_compressor"]) / 1e3, 1))
print("steam saved (kg/s):", round(float(m["steam_saved"]), 2))
```

### What was and was not verified

* **Verified** (`tests/test_evaporator.py`): mass and energy closure; agreement
  with an independent NumPy/SciPy re-implementation of the same equations (own
  steam properties, `fsolve`); `MultiEffectEvaporator(n_effects=1)` equals
  `Evaporator`; steam economy rises with N and stays near 0.8-1.0 N for a
  water-like, preheated feed; design-then-rate round trips; `check_grads`
  (order 1, reverse) of steam rate and area with respect to `U`, feed
  concentration and steam pressure, single and multiple effect.
* **Steam properties**: `Psat`/`Tsat` are the IAPWS-IF97 equations; `h_f` and
  `h_fg` are fits to eight steam-table points recalled from a standard table
  (not regenerated from IAPWS), within about 0.1 % over 20-200 C.
* **NOT verified**: no textbook worked example is claimed to be reproduced to
  2-3 %. The Geankoplis Example 8.4-1 inputs (as recalled) run, and
  `tests/test_evaporator.py` pins them as a regression value, but the book's
  printed answers were not available to compare with. The NaOH, NaCl and
  sucrose BPR data are unverified (above). Nothing here models fouling,
  hydraulics, entrainment, or crystallization.

For an economics pass use `heat_exchanger_cost(area)` and
`steam_cost_from_duty`; the optimal-number-of-effects study is in
`examples/47_evaporators.ipynb`.

---

## Liquid-Liquid Extraction

### LLEEquilibrium

**Location**: `difflow/units/lle.py`

**Class**: `LLEEquilibrium`

**Description**: The two-phase equilibrium model every extraction unit takes as a parameter: which species transfer, which carry each phase, and how the distribution coefficients are computed.

#### Process Role

`LLEEquilibrium` is a model object, not a unit operation: it has no
inlets and no outlets, and it never appears in a flowsheet on its own.
It is the `equilibrium` field of `CascadeParams` and
`DifferentialContactorParams`, and what it decides is the part of an
extraction calculation that is thermodynamics rather than cascade
bookkeeping:

- **which species are solutes** (they partition between the phases) and
  which are the two **carriers** (they do not, so their flows set the
  phase ratio at every stage);
- **where the distribution coefficients come from** --- either tabulated
  $K$ values, optionally with a van't Hoff temperature dependence, or an
  activity-coefficient model (NRTL or UNIQUAC) from which
  $K_i = \gamma_i^{aq} / \gamma_i^{org}$ follows.

Because it is a model object, the editor cannot construct one from a form
alone: `solutes`, `aqueous_carrier` and `organic_carrier` have no
defaults, and the palette reports them as unmet requirements until the
code context binds them.

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class LLEEquilibrium:
    solutes: list[str]                  # Species that transfer between phases
    aqueous_carrier: str                # Species that stays aqueous
    organic_carrier: str                # Species that stays organic
    K_coeffs: DistributionCoeffs = None # For activity_model='K'
    nrtl_params: NRTLParams = None      # For activity_model='NRTL'
    uniquac_params: UNIQUACParams = None# For activity_model='UNIQUAC'
    activity_model: str = 'K'           # 'K' | 'NRTL' | 'UNIQUAC'
    mutual_solubility: dict = None      # Optional carrier cross-solubility
```

`DistributionCoeffs` holds `species`, `K0` at `Tref`, and optionally the
heats of extraction `dH` that make $K$ temperature-dependent:

$$K_i(T) = K_{0,i} \exp\left[-\frac{\Delta H_i}{R}
\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]$$

#### Governing Equations

**Distribution coefficient** (the sign convention: $K > 1$ favours the
organic/extract phase):

$$K_i = \frac{y_i^{org}}{x_i^{aq}}$$

**Component balance across a contact**:

$$F z_i = E\, y_i^{org} + R\, x_i^{aq}$$

**Isoactivity**, when an activity model is used instead of tabulated $K$:

$$\gamma_i^{aq} x_i^{aq} = \gamma_i^{org} y_i^{org}
\qquad \Rightarrow \qquad K_i = \frac{\gamma_i^{aq}}{\gamma_i^{org}}$$

#### Example Usage

```python
from difflow.units.lle import LLEEquilibrium, DistributionCoeffs

equilibrium = LLEEquilibrium(
    solutes=["acetic_acid"],
    aqueous_carrier="water",
    organic_carrier="butanol",
    K_coeffs=DistributionCoeffs(species=("acetic_acid",), K0=(2.5,)),
)

# The one thing it computes, at a temperature:
equilibrium.get_distribution_coefficients({}, {}, 298.15)
# {'acetic_acid': 2.5}
```

With `activity_model='NRTL'` the two composition arguments are used (the
coefficients depend on them); with `'K'` they are ignored, which is why
they can be empty above.

---

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

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class CascadeParams:
    n_stages: int | float        # Equilibrium stages (continuous, for optimization)
    equilibrium: LLEEquilibrium  # The equilibrium model (see above)
    flow_config: str = 'counter_current'  # or 'co_current'
    stage_efficiency: float = 0.8         # Murphree efficiency, co-current only
    model: str = 'immiscible'             # or 'rigorous' (partially miscible, see below)
```

`n_stages` is deliberately continuous: the Kremser solution below is
smooth in it, so stage count is an ordinary design variable that
`jax.grad` can differentiate rather than an integer to enumerate.

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
from difflow import make_stream
from difflow.streams import get_flows
from difflow.units.lle import (
    LLEEquilibrium, DistributionCoeffs, MultistageCascade, CascadeParams,
)

equilibrium = LLEEquilibrium(
    solutes=["acetic_acid"],
    aqueous_carrier="water",
    organic_carrier="butanol",
    K_coeffs=DistributionCoeffs(species=("acetic_acid",), K0=(2.5,)),
)
cascade = MultistageCascade(CascadeParams(n_stages=5, equilibrium=equilibrium))

feed = make_stream({'water': 100.0, 'acetic_acid': 10.0, 'butanol': 0.0},
                   T=298.15, P=101325.0)
solvent = make_stream({'water': 0.0, 'acetic_acid': 0.0, 'butanol': 50.0},
                      T=298.15, P=101325.0)

raffinate, extract, info = cascade(feed, solvent)
recovery = 1.0 - get_flows(raffinate)['acetic_acid'] / 10.0
print(f"Recovery: {float(recovery):.2%}")        # Recovery: 91.12%
print(info['profiles'].keys())  # stage profiles: x, y, carrier_transfer, ...
```

#### Partially Miscible Systems (`model="rigorous"`)

The Kremser solution above assumes immiscible solvents and constant phase flows.
For a type-I system such as acetic acid-water-MIBK the carriers dissolve in each
other and the phase flows change along the cascade, so
`CascadeParams(n_stages=N, equilibrium=..., model="rigorous")` solves $N$
countercurrent ideal stages with the full two-liquid-phase equations,

$$R_{n-1} + E_{n+1} = R_n + E_n, \qquad
\gamma_i^{R} x_i^{R} = \gamma_i^{E} x_i^{E},$$

for every species and stage, from the NRTL or UNIQUAC parameters of the
`LLEEquilibrium`. The raffinate is the phase rich in `aqueous_carrier`, the
extract the one rich in `organic_carrier`. It is a damped Newton solve of the
$2 N n$ stage flows with a fixed iteration count, seeded by a Kremser profile
built from one flash of feed plus solvent; the gradient is the implicit-function
derivative, so `jax.grad` with respect to the solvent rate, the feed or the NRTL
parameters works. `info` carries `converged`, `residual`, `two_phase` and the
stage flows (`info["profiles"]["R"]`, `["E"]`). `mutual_solubility` is
*superseded* (giving it with `model="rigorous"` is an error), and the model
needs an integer `n_stages`, `flow_config="counter_current"`, an NRTL or
UNIQUAC `equilibrium`, and a positive total flow for every species of the
activity model. When too little solvent is used the stages mix into a single
liquid, which a two-phase stage model cannot represent: `info["converged"]` is
`False` then, and the flows should not be used.

The building block is `ternary_lle_flash(lle_model, z, T)`: the isothermal
two-liquid-phase flash of a feed `z` (any number of species; phase I is rich in
`pair[0]`), returning `x_I`, `x_II`, `beta`, the phase amounts, `K`,
`two_phase` and `converged`; a single-phase feed returns a masked result (both
phases equal to the feed, `two_phase=False`) with finite gradients. It reproduces
the tie lines of `ternary_lle` and is differentiable in `z`, `T` and the
activity parameters.

```python
import jax.numpy as jnp
from difflow import make_stream
from difflow.units.lle import (
    CascadeParams, LLEEquilibrium, MultistageCascade, NRTLParams, ternary_lle_flash,
)

# A SYNTHETIC type-I system (carrier_a and carrier_b partially miscible; "solute"
# prefers carrier_b). The NRTL parameters are illustrative, NOT fitted to any
# published data (in particular not to acetic acid-water-MIBK).
tau = jnp.array([[0.0, 2.5, 0.5], [2.5, 0.0, 0.3], [0.8, 0.6, 0.0]])
nrtl = NRTLParams(("carrier_a", "carrier_b", "solute"), tau,
                  jnp.zeros((3, 3)), 0.3 * (1.0 - jnp.eye(3)))
eq = LLEEquilibrium(solutes=["solute"], aqueous_carrier="carrier_a",
                    organic_carrier="carrier_b", nrtl_params=nrtl, activity_model="NRTL")

# One stage: the flash of an overall composition
split = ternary_lle_flash(nrtl, jnp.array([0.4, 0.5, 0.1]), 300.0)
print(bool(split["two_phase"]), f"{float(split['beta']):.3f}")     # True 0.575

# Three countercurrent stages
feed = make_stream({"carrier_a": 80.0, "carrier_b": 0.0, "solute": 20.0}, 300.0, 1e5)
solvent = make_stream({"carrier_a": 0.0, "carrier_b": 100.0, "solute": 0.0}, 300.0, 1e5)
cascade = MultistageCascade(CascadeParams(3, eq, model="rigorous"))
raffinate, extract, info = cascade(feed, solvent)
print(bool(info["converged"]))                                      # True
print(f"solute left in the raffinate: {float(raffinate['F_solute']):.3f}")  # 1.678
print(f"carrier_b lost to the raffinate: {float(raffinate['F_carrier_b']):.3f}")  # 3.069
```

With vanishing mutual solubility and a dilute solute the rigorous model
reduces to the Kremser cascade (checked in `tests/test_lle.py`, where the
discrepancy falls in proportion to the dilution). The ternary construction of
the same stages, the Hunter-Nash stepping of the previous section, gives the
same stage count (to better than 0.01 stage in that test). The system in the tests is
synthetic as well: no published acetic acid-water-MIBK data or NRTL set was
available to check, so no agreement with Sherwood-Evans-Longcor or DECHEMA data
is claimed, and the recalled parameter values are not used.

#### Utility Functions

```python
from difflow.units.lle import (
    DistributionCoeffs,
    get_K_values,
    nrtl_activity_coefficients,
    uniquac_activity_coefficients,
    separation_factor,
    minimum_solvent_ratio,
    stages_for_recovery
)

# Distribution coefficients at a temperature (tabulated K0 at Tref)
coeffs = DistributionCoeffs(species=("acetic_acid",), K0=(2.5,))
K = get_K_values(coeffs, 298.15)["acetic_acid"]

# Minimum solvent-to-feed ratio for a given recovery
S_min = minimum_solvent_ratio(K, recovery=0.95)

# Stages needed (Kremser) at an actual solvent-to-feed ratio above the minimum
N = stages_for_recovery(K, 1.5 * S_min, recovery=0.95)
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

### Ternary Diagrams and the Lever Rule

**Location**: `difflow/phase_diagrams.py`, `difflow/visualization/phase_diagrams.py`

**Functions**: `mix`, `lever_rule`, `tie_line_through`, `difference_point`, `hunter_nash`, `plot_hunter_nash` (library helpers, not palette units)

**Description**: The geometry of a type-I ternary LLE diagram: the mixing point, the lever rule, a tie line through a point, and the Hunter-Nash stage stepping for countercurrent extraction.

#### Process Role

Reading a triangular diagram is mass balance in disguise. Mixing two streams
puts the mixture on the line between them (`mix`); a feed that splits into
two phases lies on the tie line between them, and the amount of each phase is
proportional to the length of the *opposite* segment (`lever_rule`). The same
bookkeeping gives the graphical solution of a countercurrent cascade
(Hunter-Nash): with feed $F$, solvent $S$, mixing point $M = F + S$ and the
final raffinate $R_N$, the first extract $E_1$ is where the line $R_N M$ meets
the binodal, and the difference point

$$\Delta = F - E_1 = R_N - S$$

is the net flow through the cascade, so every operating line
$E_{n+1} - R_n = -\Delta$ passes through it. Stages alternate a tie line
($E_n \to R_n$) with an operating line ($R_n \to E_{n+1}$ through $\Delta$).

`mix` and `lever_rule` are pure `jnp` and differentiable. `tie_line_through`,
`hunter_nash` and `difference_point` take the tie lines tabulated by
`ternary_lle` (like it, they are numeric and not differentiated through the
equilibrium), joined into conjugate curves by monotone cubic interpolation, so
their accuracy is the interpolation's: use enough tie lines (`n_tie`). The
difference point and the operating lines are built from the extensive balance
(amounts times compositions), so they stay valid when $\Delta$ is at infinity
(parallel operating lines). Species 0 and 1 are the partially miscible pair and
species 2 the solute, as in `ternary_lle`; `carrier` says which of 0 and 1
is the feed diluent (its phase is the raffinate). `ternary_xy(...,
coords="right")` gives the right-triangle coordinates (solute on the vertical
axis) for leaching diagrams. `plot_hunter_nash(result)` draws the construction
with the binodal, tie lines, $F$, $S$, $M$, $E_1$, $R_N$, $\Delta$ and the
numbered stages (line style and marker, not colour alone).

#### Example Usage

```python
import numpy as np
import jax.numpy as jnp
from difflow import ternary_lle, lever_rule, tie_line_through, hunter_nash
from difflow.phase_diagrams import mix
from difflow.units.lle import NRTLParams

# A SYNTHETIC type-I system (a, b partially miscible; s the solute); the NRTL
# parameters are illustrative, not fitted to any real mixture.
tau = jnp.array([[0.0, 2.5, 0.5], [2.5, 0.0, 0.3], [0.8, 0.6, 0.0]])
model = NRTLParams(("a", "b", "s"), tau, jnp.zeros((3, 3)), 0.3 * (1 - jnp.eye(3)))
data = ternary_lle(model, 300.0, n_tie=30)

F = np.array([0.8, 0.0, 0.2])          # feed: a with 20 % solute
S = np.array([0.0, 1.0, 0.0])          # pure solvent b
M = mix(jnp.stack([jnp.asarray(F), jnp.asarray(S)]), jnp.array([1.0, 1.0]))
print(np.round(np.asarray(M), 3))      # [0.4 0.5 0.1]

# The mixture splits along a tie line; the lever rule gives the phase amounts.
tie = tie_line_through(data, M)
fI, fII = lever_rule(M, tie["x_I"], tie["x_II"])
print(f"phase I {float(fI):.3f}, phase II {float(fII):.3f}")   # phase I 0.425, phase II 0.575

# Countercurrent cascade: 1 kg solvent per kg feed, raffinate down to 2 % solute.
r = hunter_nash(data, F, S, 0.02, S_over_F=1.0)
print(f"{r['n_stages']:.2f} stages, {r['n_ideal']} ideal stages")   # 3.18 stages, 4 ideal stages
```

#### Limits

No published Hunter-Nash worked example (Seader-Henley-Roper, Treybal,
Geankoplis) is reproduced: their inputs and printed answers could not be
retrieved, so no agreement with them is claimed. The construction is checked
against the lever rule and component balances (exact), against the analytic
Kremser stage count for an immiscible ternary (tie lines from edge to edge with
$Y = KX$), for its own consistency (see `tests/test_phase_diagrams.py`) and
against the rigorous stage-by-stage solution of the same equilibrium model
(`tests/test_lle.py`: same stage count to better than 0.01 stage).
The stepping is limited to the range of the tabulated tie lines (it stops
before the plait point) and reports `feasible=False` at a pinch (solvent at or
below the minimum).

---

## Leaching and Washing

### Leacher

**Location**: `difflow/units/leaching.py`

**Class**: `Leacher`

**Description**: Countercurrent solid-liquid extraction (leaching) and washing with constant or tabulated underflow, as a cascade of ideal stages.

#### Process Role

Sugar from beet cossettes, oil from seeds, metal from ore, and the washing of
a precipitate or a thickener underflow are all the same cascade: the solids
move one way and carry some solution with them (the *underflow*), the
solvent moves the other way (the *overflow*), and in every ideal stage the
solution held by the solids has the composition of the overflow that leaves
it. The unit has two inlets, `solids_feed` and `solvent`, and two outlets,
`underflow` and `overflow`. **Flows are mass flows (kg/s)**: an inert solid
has no molar mass, so the `F_<species>` entries of this unit's streams are
kg/s.

Three underflow laws are available:

- `"constant_solution"`: every stage's underflow holds `underflow` kg of
  *solution* per kg of inert;
- `"constant_solvent"`: it holds `underflow` kg of *solvent* per kg of inert
  (the solution then grows as it gets richer; on a solute-per-solvent basis it
  is the same linear cascade);
- `"variable"`: a table of solution per kg inert against solute mass fraction
  (`underflow_y`, `underflow_s`), the oilseed case. `tabulate_underflow` samples
  a correlation into the tables.

`n_stages` is a real number for the two constant laws (the closed form is
smooth in it, so the stage count is an ordinary design variable); the variable
law is a stage-by-stage solve and needs an integer.

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class LeacherParams:
    inert_species: str          # the solid carrier ("pulp", "meal")
    solute: str                 # what is leached out
    solvent: str                # the leaching or wash liquid
    n_stages: float = 3.0       # ideal stages (continuous for constant underflow)
    underflow_model: str = "constant_solution"  # | "constant_solvent" | "variable"
    underflow: float = 1.0      # kg solution (or solvent) per kg inert
    underflow_y: list = None    # variable underflow table: solute mass fraction
    underflow_s: list = None    # ... and kg solution per kg inert
    stage_efficiency: float = 1.0  # overall: ideal stages = E * n_stages
    recovery: float = None      # design target: fraction of feed solute to the overflow
    design: str = "rating"      # | "stages" (solve N) | "solvent" (scale the solvent)
```

#### Governing Equations

With $B$ kg/s of inert, $L_n$ the solution in the underflow leaving stage $n$,
$V_n$ the overflow leaving stage $n$, $y_n$ the solute mass fraction of both
leaving stage $n$, $L_0$, $m_0$ the solution and solute that come in with the
solids, and $V = V_{N+1}$ the fresh solvent, the balances across any cut are

$$V_{n+1} - L_n = V_1 - L_0, \qquad
V_{n+1}\,y_{n+1} - L_n\,y_n = V_1\,y_1 - m_0 = D .$$

For a constant underflow ($L_n = L = Bu$ for $n \ge 1$) the first stage is
special: $V_{2} = \dots = V_{N+1} = V$ but $V_1 = V + L_0 - L$, which differs
from $V$ unless the feed solids already carry the underflow solution
($L_0 = L$). With $r = L/V$,

$$y_{n+1} = r\,y_n + \frac{D}{V}, \qquad
y_{N+1} = r^{N} y_1 + \frac{D}{V}\,\frac{r^{N}-1}{r-1}, \qquad
y_1 = \frac{D + m_0}{V_1},$$

which fixes $D$ in closed form. For pure washing ($L_0 = L$, solute-free wash
liquid) the fraction of the solute left in the underflow is

$$\frac{m_N}{m_0} = \frac{1-w}{1-w^{N+1}}, \qquad w = \frac{V}{L},$$

so the fraction washed out is the Kremser form. `washing_recovery(w, N)` and
`washing_stages(w, recovery)` are these two (with the $w \to 1$ limit
$N/(N+1)$); for $w<1$ the removal tends to $w$, not to 1.

For the variable underflow the $N$ balances $V_{n+1}y_{n+1}-L_ny_n-D=0$ with
$L_n = B\,s(y_n)$ are nonlinear in $y$; the unit solves them by Newton's
method with a fixed number of iterations (a fixed-size linear solve per step,
seeded by the constant-underflow solution), so the result can be jitted and
differentiated.

#### Design Modes

`recovery` is the fraction of the feed solute that ends in the overflow, net
of any solute that came in with the solvent. With `design="stages"` the
stage count is solved for (closed form, real-valued, `NaN` with
`info["feasible"] == False` when the recovery cannot be reached at that
solvent rate); with `design="solvent"` the solvent stream is scaled
(composition fixed) until the recovery is met at the given `n_stages`
(bisection with implicit gradients). `info` reports the recovery, the washing
efficiency, the per-stage compositions for an integer stage count, the
solution-to-inert ratios, `converged` and `feasible`.

#### Example Usage

```python
from difflow import make_stream, Leacher, LeacherParams
from difflow import washing_recovery, washing_stages

# 100 kg/s of pulp carrying 200 kg/s of sugar solution (10 % sugar), washed
# with 150 kg/s of water; the underflow holds 2 kg solution per kg pulp.
feed = make_stream({"pulp": 100.0, "sugar": 20.0, "water": 180.0}, T=300.0, P=1e5)
water = make_stream({"water": 150.0}, T=300.0, P=1e5)

leacher = Leacher(LeacherParams("pulp", "sugar", "water", n_stages=3.0, underflow=2.0))
underflow, overflow, info = leacher(feed, water)
print(f"recovery {float(info['recovery']):.4f}")          # recovery 0.6343
print(f"liquor {float(info['y_overflow']):.4f} mass fraction")  # liquor 0.0846 mass fraction

# This is a pure wash (the feed holds as much solution as the underflow does),
# so the closed form of the washing helpers must agree.
w = 150.0 / 200.0
assert abs(float(washing_recovery(w, 3.0)) - float(info["recovery"])) < 1e-12

# Design: stages for 95 % recovery with 240 kg/s of water (w = 1.2) ...
big = make_stream({"water": 240.0}, T=300.0, P=1e5)
design = Leacher(LeacherParams("pulp", "sugar", "water", underflow=2.0,
                               recovery=0.95, design="stages"))
_, _, info = design(feed, big)
print(f"stages {float(info['n_stages']):.3f}")            # stages 7.827
print(f"check  {float(washing_stages(1.2, 0.95)):.3f}")   # check  7.827

# ... or the water rate that reaches 95 % in 3 stages
sized = Leacher(LeacherParams("pulp", "sugar", "water", n_stages=3.0, underflow=2.0,
                              recovery=0.95, design="solvent"))
_, _, info = sized(feed, water)
print(f"water {float(info['solvent_rate']):.1f} kg/s")    # water 452.9 kg/s

# Variable underflow (a synthetic, oilseed-like table): 3 stages
var = Leacher(LeacherParams("pulp", "sugar", "water", n_stages=3,
                            underflow_model="variable",
                            underflow_y=[0.0, 0.1, 0.2, 0.4],
                            underflow_s=[1.5, 1.8, 2.2, 3.0]))
_, _, info = var(feed, water)
print(f"recovery {float(info['recovery']):.4f}")          # recovery 0.7688
```

Gradients of the recovery with respect to the solvent rate, the stage count
and the underflow ratio are ordinary `jax.grad` calls, so the solvent-to-solids
ratio and the stage count that minimise a solvent-recovery cost can be found
by gradient descent (worked in `examples/54_leaching_ternary.ipynb`, together
with the ternary helpers and the rigorous cascade).

#### Limits

Ideal stages: the solute is fully dissolved and in equilibrium with the
solution (no mass-transfer rate limit and no adsorption on the solids;
diffusion-controlled leaching kinetics is out of scope). One solute, one
solvent and one inert (other species in the streams are ignored), a common
inert flow in every stage, isothermal. `stage_efficiency` is an overall
efficiency: the number of ideal stages is $E \cdot n_{stages}$. The unit is
checked against an independent dense-linear-system assembly of the stage
balances, the closed washing form and a SciPy solve of the variable-underflow
balances (`tests/test_leaching.py`); no published textbook example is
reproduced (the McCabe-Smith-Harriott and Geankoplis worked examples named in
the issue could not be retrieved, so no agreement with them is claimed), and
the variable-underflow table above is synthetic.

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

### EOSCompressor

**Exported as** `Compressor`; registered in the catalog --- and so labelled
in the editor's palette --- as **`EOSCompressor`**, because `Compressor`
is taken by the gas plugin's fixed-ratio compressor station
([Gas networks](unit-operations-gas.md)) and the two are different
models of different things.

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

## Liquid Pipe Flow

Incompressible (liquid) flow in pipes: Darcy-Weisbach friction with a
Colebrook/Churchill friction factor, fittings, elevation head and
differential-pressure flow meters. The functions live in
[`difflow.fluids`](#fluids-module) (JAX, SI, differentiable) and the unit
operation is [`Pipe`](#op-pipe). Compressible gas flow (nozzles, isothermal
and adiabatic lines) is in [`difflow.compressible`](#compressible-module), and
gas networks are in `difflow_gas`. Two-phase flow, non-Newtonian fluids and
transients are out of scope.

(op-pipe)=
### Pipe

A run of pipe with fittings and an elevation change. The outlet pressure is

$$
P_\mathrm{out} = P_\mathrm{in} - f\frac{L}{D}\frac{\rho v^2}{2}
 - \sum_j n_j K_j \frac{\rho v^2}{2} - \rho g\,\Delta z ,
\qquad v = \frac{4Q}{\pi D^2},\quad Q = \frac{\dot m}{\rho}
$$

Temperature and molar flows pass through unchanged.

| Parameter | Unit | Meaning |
|-----------|------|---------|
| `L` | m | straight pipe length |
| `D` | m | inside diameter |
| `rho` | kg/m³ | liquid density: a number or a callable of the inlet stream |
| `mu` | Pa s | liquid viscosity: a number or a callable of the inlet stream |
| `roughness` | m | absolute roughness, default 4.6e-5 (commercial steel) |
| `dz` | m | elevation gain, positive uphill |
| `fittings` | – | `{name: count}` using the K table below, or a total K |
| `friction_method` | – | `"colebrook"` (default), `"churchill"`, `"haaland"`, `"swamee_jain"` |
| `MW` | g/mol | molar mass (number or `{species: MW}`) to turn molar into mass flow; omit if a `thermo` is passed |

**Design decision: liquid properties are explicit.** Core `SpeciesData` has no
liquid density or viscosity, so `rho` and `mu` are *inputs*, as numbers or as
callables of the stream (for example a correlation in `T`). Putting
Rackett/DIPPR-105 density and Andrade/DIPPR-101 viscosity into `SpeciesData`
(with mixing rules) would also serve heat-transfer correlations and pump models
and is left to a separate issue.

```python
import jax
from difflow import Pipe, PipeParams, make_stream

# 3 L/s of water at 20 C through 100 m of 2-in Sch 40 commercial steel
water = make_stream({"W": 0.003 * 998.2 / 0.018015}, T=293.15, P=3e5)

def dP(D):
    pipe = Pipe(PipeParams(L=100.0, D=D, rho=998.2, mu=1.002e-3, MW=18.015,
                           fittings={"elbow_90_standard": 4, "gate_valve": 1}))
    outlet, info = pipe(water)
    return info["dP"]

outlet, info = Pipe(PipeParams(L=100.0, D=0.0525, rho=998.2, mu=1.002e-3,
                               MW=18.015))(water)
info["v"], info["Re"], info["f"]      # 1.386 m/s, 7.25e4, 0.02254
info["dP_friction"]                    # 41.1 kPa
jax.grad(dP)(0.0525)                   # d(dP)/dD, Pa/m: economic diameter optimization
```

`info` carries `Q`, `v`, `Re`, `f`, `K_total`, `dP_friction`, `dP_minor`,
`dP_static`, `dP` (the sum) and `head_loss` (friction plus minor loss, m of
fluid). `Pipe.eo_residuals` gives `n_species + 2` rows (flows, temperature and
the pressure balance) so the unit works in the [EO solver](eo-solver.md). The
example notebook `examples/42_liquid_pipe_flow.ipynb` regenerates the Moody
chart and finds the economic pipe diameter with `jax.grad`.

(fluids-module)=
### Friction Factor Methods

`difflow.fluids.friction_factor(Re, rel_roughness, method)` returns the
**Darcy** factor (Fanning = Darcy/4).

| Method | Form | Notes |
|--------|------|-------|
| `"colebrook"` | implicit Colebrook-White, $1/\sqrt f=-2\log_{10}\!\left(\frac{\varepsilon/D}{3.7}+\frac{2.51}{\mathrm{Re}\sqrt f}\right)$ | Newton iteration (8 steps) from the Haaland estimate; gradients by the implicit function theorem; blended with 64/Re |
| `"churchill"` | Churchill (1977), one explicit expression | valid laminar, transition and turbulent; no blend |
| `"haaland"` | $1/\sqrt f=-1.8\log_{10}\!\left[(\varepsilon/D/3.7)^{1.11}+6.9/\mathrm{Re}\right]$ | explicit; blended with 64/Re |
| `"swamee_jain"` | $f=0.25/\log_{10}^2\!\left[\varepsilon/D/3.7+5.74/\mathrm{Re}^{0.9}\right]$ | explicit; blended with 64/Re |

**Laminar-turbulent blend.** For the Colebrook, Haaland and Swamee-Jain methods
$f=(1-w)\,64/\mathrm{Re}+w f_\mathrm{turb}$, with $w$ a C² smootherstep of
$\ln\mathrm{Re}$ between Re = 2100 and 4000 (a sigmoid in log Re with compact
support: exactly 0 below 2100, exactly 1 above 4000). Below 2100 the result is
*exactly* 64/Re. No Python `if` is involved, so `jit`, `vmap` and `grad` all
work, and the gradients are finite across the transition (tested at Re = 1e3,
2300, 4000 and 1e5 against `check_grads`). In 2100 < Re < 4000 real flow is
intermittent and no correlation is accurate: the blend is a smooth
interpolation, not a transition model.

**What was verified, and how** (`tests/test_fluids.py`):

- Colebrook agrees with an independent bracketing solution of the same
  equation (scipy `brentq`) to 1e-10 over Re 4e3-1e8, ε/D 0-0.05.
- Its limits are Prandtl's smooth-pipe law and the von Kármán fully-rough law.
- Moody-chart spot values (smooth: 0.0309, 0.0180, 0.0116 at Re = 1e4, 1e5, 1e6;
  fully rough: 0.0716, 0.0380, 0.0196 at ε/D = 0.05, 0.01, 0.001) are matched
  to 1 % (chart reading precision).
- Colebrook vs Churchill: within 2.2 % for Re ≥ 1e4 and 1.5 % for Re ≥ 1e5 over
  ε/D 0-0.05; the worst case is 3.1 % at Re = 4000, ε/D = 0.01. Haaland is
  within 1.4 % and Swamee-Jain within 3.1 % of Colebrook. The 1-2 % agreement
  holds in the fully turbulent range, not right at the onset of turbulence.

Also in the module: `reynolds_number`, `hydraulic_diameter` (4A/P),
`darcy_pressure_drop`, `minor_loss`, `fully_rough_friction_factor` and the L/D
helper `equivalent_length(K, D, f)` = K D / f.

### Fitting Loss Coefficients

`difflow.fluids.FITTINGS` maps a name to a loss coefficient. Entries of kind
`LD` are Crane's equivalent-length form, $K=(L/D)\,f_T$, where $f_T$ is the
*fully rough* friction factor of the connected pipe
(`fully_rough_friction_factor(ε/D)`), so $K$ scales with pipe size and stays
differentiable in $D$. Entries of kind `K` are constants. For a 2-in Sch 40
steel pipe the von Kármán formula gives $f_T=0.0190$, the value Crane tabulates
for 2-in pipe.

| Name | Kind | Value | Source |
|------|------|-------|--------|
| `elbow_90_standard` | LD | 30 | Crane TP-410: 90° standard elbow, K = 30 f_T |
| `elbow_90_long_radius` | LD | 14 | Crane TP-410: 90° bend r/d = 1.5, K = 14 f_T |
| `elbow_45_standard` | LD | 16 | Crane TP-410: 45° standard elbow, K = 16 f_T |
| `return_bend_180` | LD | 50 | Crane TP-410: 180° close return bend, K = 50 f_T |
| `tee_through` | LD | 20 | Crane TP-410: tee, flow through run, K = 20 f_T |
| `tee_branch` | LD | 60 | Crane TP-410: tee, flow through branch, K = 60 f_T |
| `gate_valve` | LD | 8 | Crane TP-410: gate valve, fully open, K = 8 f_T |
| `globe_valve` | LD | 340 | Crane TP-410: globe valve, fully open, K = 340 f_T |
| `angle_valve` | LD | 150 | Crane TP-410: angle valve, fully open, K = 150 f_T |
| `ball_valve` | LD | 3 | Crane TP-410: ball valve, fully open, K = 3 f_T |
| `check_valve_swing` | LD | 50 | Crane TP-410: swing check valve, fully open |
| `entrance_sharp` | K | 0.5 | Crane TP-410: sharp-edged flush entrance (also Perry's 9e Sec. 6) |
| `entrance_rounded` | K | 0.04 | Crane TP-410: well-rounded entrance, r/d ≥ 0.15 |
| `exit` | K | 1.0 | Crane TP-410: exit into a large volume |

Source: Crane Co., *Flow of Fluids Through Valves, Fittings, and Pipe*,
Technical Paper 410; the same tables are reprinted in Perry's *Chemical
Engineers' Handbook* (Sec. 6). **Provenance caveat:** these numbers were
transcribed from the widely reprinted tables and were not machine-checked
against the TP-410 scan; confirm against your edition for tight designs.
Published K-values for fittings are good to about ±25 % at best (the swing
check in particular is quoted differently in different editions).

### Flow Meters

`orifice_flow(dP, D_pipe, d_orifice, rho, Cd=0.61)` and
`venturi_flow(dP, D_pipe, d_throat, rho, Cd=0.98)` return the volumetric flow
$Q=C_d\frac{\pi}{4}d^2\sqrt{\frac{2\,\Delta P}{\rho\,(1-\beta^4)}}$,
$\beta=d/D$. The inverse forms `orifice_dp(Q, ...)` and `venturi_dp(Q, ...)`
give the differential pressure for a known flow. The discharge coefficients
are the usual textbook constants (0.61 sharp-edged orifice at high Re; about
0.98 for a Venturi); ISO 5167 gives $C_d$ as a function of β, Re and tap
geometry, so pass that value when accuracy matters. Incompressible only, no
expansibility factor.

### Pipe Networks: Parallel Branches

A full network solver is out of scope (`difflow_gas` covers gas networks), but a
parallel pair between two nodes is a one-unknown root find with the existing
solvers: the fraction $x$ of the flow in branch A satisfies
$\Delta P_A(xQ)=\Delta P_B((1-x)Q)$.

```python
import jax.numpy as jnp, optimistix as optx
from difflow import Pipe, PipeParams, make_stream

RHO, MU, MW = 998.2, 1.002e-3, 18.015          # water at 20 C
def stream_with_flow(Q):                        # Q in m^3/s
    return make_stream({"W": Q * RHO / (MW * 1e-3)}, 293.15, 3e5)

pipe_a = Pipe(PipeParams(L=80.0, D=0.0525, rho=RHO, mu=MU, MW=MW))
pipe_b = Pipe(PipeParams(L=150.0, D=0.0409, rho=RHO, mu=MU, MW=MW,
                         fittings={"globe_valve": 1}))
Q_total = 6e-3

dP = lambda pipe, Q: pipe(stream_with_flow(Q))[1]["dP"]
res = lambda x, _: dP(pipe_a, x * Q_total) - dP(pipe_b, (1 - x) * Q_total)
x = optx.root_find(res, optx.Newton(rtol=1e-12, atol=1e-10), jnp.asarray(0.5)).value
print(f"branch A carries {float(x):.1%} of the flow")
```

Both branches then have equal ΔP, the flows sum to the total, and the split is
differentiable by the implicit function theorem (`tests/test_pipe.py::
TestParallelBranches`).

(compressible-module)=
## Compressible Gas Flow

`difflow.compressible` covers ideal-gas flow where the density changes along
the path: nozzles and relief devices, isothermal gas lines and adiabatic
(Fanno) ducts. It assumes constant $\gamma=C_p/C_v$ and molar mass `MW`
(kg/mol); friction factors are **Darcy** factors, as in `difflow.fluids`. For an
isothermal line $\mathrm{Re}=GD/\mu$ is constant along the pipe, so $f$ is one
call to `fluids.friction_factor`. JAX, SI and differentiable throughout.

| Function | Relation |
|----------|----------|
| `speed_of_sound(T, MW, gamma)`, `mach_number(v, T, MW, gamma)` | $c=\sqrt{\gamma RT/M}$, $\mathrm{Ma}=v/c$ |
| `stagnation_ratios(M, gamma)` | isentropic $T_0/T$, $P_0/P$, $\rho_0/\rho$ |
| `critical_pressure_ratio(gamma)` | $P^*/P_0=(2/(\gamma+1))^{\gamma/(\gamma-1)}$ (0.528 for air) |
| `choked_mass_flux(P0, T0, MW, gamma)` | $G^*=P_0\sqrt{\gamma M/(RT_0)}\,(2/(\gamma+1))^{(\gamma+1)/(2(\gamma-1))}$ |
| `nozzle_mass_flow(P0, T0, P_back, area, MW, gamma, Cd)` | isentropic converging nozzle, subsonic or choked |
| `isothermal_pipe_mass_flux(P1, P2, T, L, D, f, MW)` | $P_1^2-P_2^2=G^2\frac{RT}{M}\left[\frac{fL}{D}+2\ln\frac{P_1}{P_2}\right]$ |
| `isothermal_outlet_pressure(G, P1, T, L, D, f, MW, gamma)` | the same equation solved for $P_2$, with the choking check |
| `fanno_parameter(M, gamma)`, `fanno_ratios(M, gamma)` | $fL^*/D$ and $P/P^*$, $T/T^*$, $P_0/P_0^*$ |
| `adiabatic_outlet_mach(M1, fL_D, gamma)` | outlet Mach, $P_2/P_1$, $T_2/T_1$ of a subsonic adiabatic duct |

**Choking.** `nozzle_mass_flow` holds $P_\mathrm{back}/P_0$ at no less than the
critical ratio. The isentropic flux has its maximum there, so the clamp is
continuous with a continuous first derivative, and `jax.grad` with respect to
the back pressure is exactly 0 in choked flow. An isothermal line chokes when
its exit velocity reaches $\sqrt{RT/M}$ (Mach $1/\sqrt\gamma$, not 1). Past
that, `isothermal_outlet_pressure` returns `choked=True` and $P_2$ at the choke
pressure; `adiabatic_outlet_mach` returns `choked=True` and $M_2=1$ when
$fL/D$ exceeds $fL^*/D$ at the inlet Mach number. In both cases the real flow
would be lower than the one requested.

**Solvers and gradients.** The two outlet solvers bracket the subsonic root
(between the choke point and the inlet) and bisect in log space. A final
Newton step on the stopped root attaches implicit-function-theorem gradients
with respect to every input.

**What was verified** (`tests/test_compressible.py`):

- The isothermal equation and the Fanno relations agree with an independent
  `solve_ivp` integration of the differential momentum balance, to 1e-6 or
  better. This includes a short, high-velocity line where the acceleration term
  matters.
- $dP_2/dD$ agrees with finite differences of that integration.
- `nozzle_mass_flow` is continuous at the critical ratio, equals
  $\rho^*c^*A$ when choked, and tends to the Bernoulli flow
  $A\sqrt{2\rho\Delta P}$ at small pressure drop.

```python
from difflow import fluids
from difflow.compressible import isothermal_pipe_mass_flux, nozzle_mass_flow

# 50 km natural-gas line, 0.5 m ID, 60 -> 40 bar at 288 K
D, mu, MW = 0.5, 1.1e-5, 0.0175
G = isothermal_pipe_mass_flux(60e5, 40e5, 288.0, 5e4, D, 0.012, MW)
f = fluids.friction_factor(G * D / mu, 4.6e-5 / D)   # check the assumed f
print(float(G) * 3.1416 * D**2 / 4, "kg/s;  f =", float(f))

# air from a 10 bar vessel through a 1 cm^2 nozzle: choked
print(float(nozzle_mass_flow(10e5, 300.0, 1e5, 1e-4, 0.02897, 1.4, Cd=0.97)), "kg/s")
```

---

## Centrifugal Pumps

A centrifugal pump model with performance curves: the pump raises the pressure
of a liquid by $\rho g H(Q)$, the curves scale with speed and impeller diameter
by the affinity laws, NPSH available is checked against NPSH required, and
`operating_point` answers "what flow will this pump deliver into this system?"
The system curve is built from [`Pipe`](#op-pipe) runs, so friction and fittings
come from [`difflow.fluids`](#fluids-module). Positive-displacement pumps, pump
transients and multiphase pumps are out of scope.

(op-centrifugalpump)=
### CentrifugalPump

$$
H(Q) = h_0 + h_1 Q + h_2 Q^2, \qquad
\eta(Q) = \eta_{\max}\Bigl(1-\bigl(\tfrac{Q-Q_\mathrm{bep}}{Q_\mathrm{bep}}\bigr)^2\Bigr), \qquad
P_\mathrm{shaft} = \frac{\rho g Q H}{\eta}
$$

$$
Q \propto N D, \quad H \propto N^2 D^2, \quad P \propto N^3 D^3, \qquad
\mathrm{NPSH}_a = \frac{P_s - P_\mathrm{sat}}{\rho g} + \frac{v_s^2}{2g} + z_s - h_{f,s}
$$

In the default (flow-given) mode the inlet stream fixes $Q=\dot m/\rho$ and the
outlet is $P_\mathrm{in}+\rho g H(Q)$; temperature and molar flows pass through
(or, with `cp`, the liquid warms by the dissipated shaft work).

| Parameter | Unit | Meaning |
|-----------|------|---------|
| `head_coeffs` | m, m/(m³/s), … | ascending polynomial coefficients of $H(Q)$ at the reference speed and diameter (see `fit_pump_curve`) |
| `rho` | kg/m³ | liquid density: a number or a callable of the inlet stream |
| `eta_coeffs` or `eta_max` + `Q_bep` | –, m³/s | efficiency polynomial, or the BEP-centred parabola |
| `npshr_coeffs` | m, … | polynomial of required NPSH (optional) |
| `N_ref`, `D_ref` | rpm, m | speed and impeller diameter the curves belong to |
| `N`, `D` | rpm, m | operating speed and diameter (default: the references) |
| `motor_efficiency` | – | `electric_power = shaft_power / motor_efficiency` |
| `MW` | g/mol | molar mass (number or `{species: MW}`); omit if a `thermo` is passed |
| `Psat` | Pa | vapour pressure for NPSHa (number or callable); otherwise `thermo.Psat(species, T)` |
| `D_suction` | m | suction nozzle diameter for the velocity head in NPSHa |
| `cp` | J/kg/K | liquid heat capacity; enables the temperature rise |

`info` holds `Q`, `head`, `efficiency`, `hydraulic_power`, `shaft_power`,
`electric_power` (W) and, when the curves and a vapour pressure are given,
`NPSHr`, `NPSHa` and `npsh_margin = NPSHa - NPSHr` (negative: cavitation risk).
Like `Pipe`, liquid density is an explicit input; see the design note under
[Pipe](#op-pipe).

```python
from difflow import (CentrifugalPump, CentrifugalPumpParams, Pipe, PipeParams,
                     fit_pump_curve, operating_point, system_curve)

Q_data = [0.0, 0.01, 0.02, 0.03, 0.04]                # vendor points, m^3/s
H_data = [38.0, 36.5, 32.0, 24.5, 14.0]               # head, m (illustrative)
h = fit_pump_curve(Q_data, H_data, order=2)
pump = CentrifugalPump(CentrifugalPumpParams(
    head_coeffs=h, rho=998.2, eta_max=0.72, Q_bep=0.020,
    npshr_coeffs=[1.5, 0.0, 2.5e3], N_ref=1750.0, D_ref=0.25, MW=18.015))

line = Pipe(PipeParams(L=400.0, D=0.1023, rho=998.2, mu=1.002e-3, MW=18.015, fittings=5.3))
sys_ = system_curve(12.0, line)                       # 12 m static lift + friction
Q_star = operating_point(pump, sys_)                  # m^3/s; differentiable
```

**Helpers.**

- `system_curve(static_head, *pipes, delta_P=None, rho=None)` gives
  $H_\mathrm{sys}(Q)=\Delta z+\Delta P/\rho g+\sum_i (h_{f,i}(Q)+\mathrm{dz}_i)$. Pipe `dz`
  is included, so put elevation either in the pipes or in `static_head`.
- `operating_point(pump, system)` solves $H_\mathrm{pump}(Q)=H_\mathrm{sys}(Q)$ by
  Newton started from the pump's zero-head flow. Gradients with respect to
  pump and system parameters (`D`, `N`, pipe diameter, ...) use the implicit
  function theorem, not the iterations. It has no meaningful answer when the
  shutoff head is below the static head.
- `pumps_in_series(*pumps)` adds heads at equal flow; `pumps_in_parallel(*pumps)`
  adds flows at equal head (a small Newton solve for the branch flows,
  `.split(Q)`). Both return curves usable by `operating_point`.
- `npsh_available(P_suction, Psat, rho, v, z_suction, losses)`.
- `pump.curve` is a `difflow.units.pump.PumpCurve` (`head`, `efficiency`, `npshr`,
  `shaft_power` at the operating `N` and `D`); build one directly from
  coefficients to study a pump without a stream.
- Flow meters (`orifice_flow` and its inverse `orifice_dp`) are in
  [`difflow.fluids`](#fluids-module).

**Affinity laws and static head.** Doubling $N$ doubles $Q^*$ only when the
system curve is pure friction ($H_\mathrm{sys}\propto Q^2$, no static head),
because then the whole intersection scales. With a static lift the system
curve does not scale, so $Q^*$ does not follow $N$; the tests check both
(`tests/test_pump.py::TestAffinityLaws`).

**Verification, honestly stated.** The transfer-pump test (12 m lift, 400 m of
4-in Sch 40 steel, fittings $K=5.3$, water at 20 °C) uses an *illustrative*
vendor curve, not a published datasheet or textbook figure; textbook pump
examples (McCabe & Smith, Geankoplis, Crane) are read off a printed curve and
are not reproduced from memory. The operating point is checked instead against
an independent calculation (`numpy.polyfit` for the curve and `scipy.optimize.brentq`
for both the Colebrook-White friction factor and the pump/system intersection,
no difflow code), agreeing to $10^{-6}$ relative, well inside the 2 % the
issue asked for. That validates the implementation, not the vendor data.
The NPSH test uses 95 °C water with steam-table properties
(P_sat = 84.55 kPa, ρ = 961.9 kg/m³; NPSHa = 1.78 m against an NPSHr of 2.5 m at 20 L/s).

**Relation to the carbon-capture `Pump`.** `difflow_cc.Pump` is a different
model: dense-phase CO₂ pumped to a *fixed outlet pressure* with a constant
efficiency, density from the CO₂ EOS ($W=\dot V\,\Delta P/\eta$). It has no curve,
speed or NPSH and cannot say what flow results. It was deliberately **not**
rewritten to delegate to `CentrifugalPump`: its density comes from an EOS rather
than an input, and its specification (outlet pressure) is the inverse of the
curve-driven one. Use `Pump` for CO₂ compression-train duty and `CentrifugalPump`
for liquid transfer and for sizing against a system.

The example notebook `examples/43_centrifugal_pump.ipynb` plots the pump and
system curves with the operating point, speed and trim changes, series/parallel
operation and NPSH margin, and picks the pipe diameter with a VFD-speed
constraint by gradient-based optimisation of `pump_cost` + piping +
`pump_electricity_cost`.

---

## Combustion & Gas-Turbine Units

These units model a **Brayton cycle** working fluid — air and combustion gas at
high temperature and moderate pressure, where the cubic-EOS departure is
negligible. They therefore use **ideal-gas** properties with temperature-dependent
Cp (`difflow.combustion.IdealGasThermo`), the standard model for gas-turbine
cycle analysis, and are named distinctly from the real-gas `Compressor` /
`Turboexpander` above (a different thermodynamic model for a different service).
All internal temperature and air/fuel-ratio solves are `optimistix` root finds,
so shaft work, firing temperature, air/fuel ratio and efficiency are
differentiable with respect to feed conditions, pressures, efficiencies and fuel
composition.

The fuel-hydrocarbon (and N₂/CO₂) ideal-gas Cp come from the database (the same
cubics used by the NGL work); the module adds O₂, H₂O-vapor and Ar Cp, air
composition, and per-fuel lower heating values and combustion stoichiometry.

```python
from difflow import (
    Combustor, CombustorParams,
    GasCompressor, GasCompressorParams,
    GasTurbine, GasTurbineParams,
    brayton_cycle, BraytonCycleParams, make_cycle_thermo,
)
from difflow.combustion import AIR_COMPOSITION
from difflow.streams import make_stream

thermo = make_cycle_thermo()          # ideal-gas thermo over cycle + fuel species
```

### GasCompressor

Adiabatic ideal-gas compression to `pressure_ratio × P_in` with an isentropic
efficiency (the efficiency inflates the enthalpy rise). Work consumed is
$W = H_\text{out} - H_\text{in} > 0$.

```python
air = make_stream(dict(AIR_COMPOSITION), T=288.15, P=101325.0)
comp = GasCompressor(GasCompressorParams(pressure_ratio=18.0, eta_isentropic=0.89), thermo)
compressed, info = comp(air)          # info: {"W", "T_isen", "T_out", ...}
```

### Combustor

Complete-combustion reactor, $C_xH_y + (x + y/4)\,O_2 \to x\,CO_2 + (y/2)\,H_2O$,
with two modes:

- **`"adiabatic"`** — both feeds fixed; solves the adiabatic flame temperature.
- **`"fixed_T"`** — scales the air stream to the air/fuel ratio that hits a
  target firing temperature `T_out` (closed-form, affine in the air amount).

```python
fuel = make_stream({"methane": 1.0}, T=298.15, P=18 * 101325.0)
comb = Combustor(CombustorParams(mode="fixed_T", T_out=1673.15, dp_frac=0.04), thermo)
products, info = comb(fuel, compressed)   # info: {"T_out", "air_scale", "o2_demand", "Q"}
```

### GasTurbine

Adiabatic ideal-gas expansion to a back-pressure `P_out` with an isentropic
efficiency. Work extracted is $W = H_\text{in} - H_\text{out} > 0$.

```python
turb = GasTurbine(GasTurbineParams(P_out=101325.0, eta_isentropic=0.90), thermo)
exhaust, info = turb(products)
```

### brayton_cycle

Assembles compressor → combustor → turbine into an intensive (per mole of fuel)
simple- or combined-cycle solve. Defaults are a modern F-class machine at ISO
conditions and reproduce published performance: simple-cycle η ≈ 0.40
(8470 Btu/kWh), combined-cycle η ≈ 0.568 (6009 Btu/kWh).

```python
fuel_comp = {"methane": 0.95, "ethane": 0.03, "propane": 0.01,
             "nitrogen": 0.005, "carbon_dioxide": 0.005}
result = brayton_cycle(fuel_comp, BraytonCycleParams(combined_cycle=True))
# result: {"eta_thermal", "eta_gt_only", "work_net", "air_fuel_molar", ...}
```

---

## Particles and Beds

**Location**: `difflow/particles.py` (pure JAX functions in SI units; not unit
operations, so they have no palette entry). They are the building blocks for
adsorbers, catalytic beds, settlers, cyclones and filters.
The units that apply them to settling, filtration and gas cleaning are in
[Mechanical Separations](#op-settler) (`Settler`, `CakeFilter`, `Cyclone`).

(particles-drag)=
### Drag and terminal velocity

`drag_coefficient(Re_p, sphericity, method)` is smooth across the Stokes,
intermediate and Newton regimes and tends to $24/Re_p$ as $Re_p \to 0$:

| `method` | Correlation | Range |
|----------|-------------|-------|
| `"haider_levenspiel"` (default) | Haider & Levenspiel (1989), sphericity-dependent | $Re < 2.6\times10^5$, $\phi$ 0.5-1 |
| `"turton_levenspiel"` | Turton & Levenspiel (1986), spheres | $Re < 2.6\times10^5$ |
| `"schiller_naumann"` | Schiller & Naumann (1933), floored at 0.44 | $Re < 800$ then Newton plateau |

`terminal_velocity(d_p, rho_p, rho_f, mu, sphericity)` solves the force balance
$C_D(Re)\,Re^2 = \tfrac43 Ar$ by a Newton root find in $\ln Re$, started from the
explicit Haider-Levenspiel dimensionless velocity, and differentiates it by
implicit differentiation. `hindered_settling_velocity(v_t, voidage, n, Re_t)` is
Richardson & Zaki (1954), $u = v_t\varepsilon^n$ with $n$ from $Re_t$ when not given.

```python
from difflow.particles import terminal_velocity, drag_coefficient
import jax

d = 1e-3                                           # 1 mm sand grain in water
v = terminal_velocity(d, 2650.0, 998.0, 1.0e-3)
dv_dd = jax.grad(terminal_velocity)(d, 2650.0, 998.0, 1.0e-3)
print(f"v_t = {float(v):.3f} m/s, dv/dd = {float(dv_dd):.1f} 1/s")
print(f"C_D(Re=1e-4)*Re/24 = {float(drag_coefficient(1e-4))*1e-4/24:.4f}")
```

(particles-ergun)=
### Packed-bed pressure drop (Ergun)

`ergun_pressure_gradient(u_s, d_p, voidage, rho, mu, sphericity)` returns
$(\Delta P/L,\ \text{info})$ with the full Ergun (1952) equation

$$\frac{\Delta P}{L} = 150\frac{(1-\varepsilon)^2}{\varepsilon^3}\frac{\mu u_s}{(\phi d_p)^2}
+ 1.75\frac{1-\varepsilon}{\varepsilon^3}\frac{\rho u_s^2}{\phi d_p}$$

and `info['viscous']` (Blake-Kozeny), `info['inertial']` (Burke-Plummer),
`info['Re_bed']`. `D_bed=` applies the Mehta & Hawley (1969) wall correction for
narrow beds ($D/d_p \lesssim 10$): $M = 1 + 2\phi d_p/[3D(1-\varepsilon)]$
multiplies the viscous term by $M^2$ and the inertial term by $M$
(`info['wall_factor']`). `kozeny_carman(...)` is the laminar limit with the Kozeny
constant $k_0$ (prefactor $36k_0 = 180$), the piece to reuse for filter-cake
resistance.

```python
from difflow.particles import ergun_pressure_gradient

# 3 mm spheres, eps = 0.4, air, u = 1 m/s: 1687.5 + 6562.5 Pa/m by hand
dpdl, info = ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5)
print(float(dpdl), float(info["viscous"]), float(info["inertial"]))
```

**GasPFR.** `GasPFRParams` accepts either the lumped `alpha` (Pa/m$^3$) or the
physical bed (`d_p`, `voidage`, `mu`, `rho_gas`, `bed_area`, optional `u_s0`,
`sphericity`); in the second case $\alpha = (\Delta P/L)_0/A_c$ is computed from
the Ergun equation at the inlet (`ergun_alpha`, `params.alpha_for(Q0)`, or
`params.effective_alpha` when `u_s0` is given). With `u_s0=None` the velocity is
$Q_0/A_c$ from the inlet stream; a given `u_s0` is checked against it
(`GasPFRBedWarning` above 1 %). Supplying both `alpha` and the bed is an error.

(particles-fluidization)=
### Fluidization

`minimum_fluidization_velocity(d_p, rho_p, rho_f, mu, voidage_mf, sphericity,
method)` equates the Ergun gradient to the bed weight per volume. In terms of
$Ar = d_p^3\rho_f(\rho_p-\rho_f)g/\mu^2$ it is a quadratic in $Re_{mf}$,

$$Ar = \frac{150(1-\varepsilon_{mf})}{\phi^2\varepsilon_{mf}^3}Re_{mf}
+ \frac{1.75}{\phi\varepsilon_{mf}^3}Re_{mf}^2,$$

solved in closed form (`method="ergun"`); `method="wen_yu"` is the Wen & Yu (1966)
correlation $Re_{mf} = \sqrt{33.7^2 + 0.0408\,Ar} - 33.7$. `bed_expansion(u, u_t, n)`
gives the Richardson-Zaki voidage $(u/u_t)^{1/n}$ and height
$H_{mf}(1-\varepsilon_{mf})/(1-\varepsilon)$ (a homogeneous-expansion model). By
default the line is anchored at $u_t$, which can give $\varepsilon < \varepsilon_{mf}$
(a bed shorter than $H_{mf}$) just above $u_{mf}$; pass `u_mf=` to anchor it at
$(u_{mf}, \varepsilon_{mf})$ instead, $\varepsilon = \min[\varepsilon_{mf}(u/u_{mf})^{1/n}, 1]$.
`fluidization_window(...)` returns $(u_{mf}, u_t)$; above $u_t$ particles are
entrained.

`minimum_bubbling_velocity(d_p, rho_f, mu, fines_fraction)` is the Abrahamsen &
Geldart (1980) correlation $u_{mb} = 2.07\,e^{0.716F}d_p\rho_f^{0.06}/\mu^{0.347}$
(gas beds; $F$ is the mass fraction below 45 µm). `geldart_group(d_p, rho_p, rho_f,
mu, fines_fraction)` returns A/B/C/D: C below 20 µm, D above
Grace's (1986) B/D line $Ar > 1.45\times10^5$, and otherwise A when
$u_{mb}/u_{mf} > 1$ (the bed expands before it bubbles), with $u_{mf}$ from the
Baeyens & Geldart (1974) correlation. This is the sloping A/B boundary, so FCC
catalyst (60 µm, 1500 kg/m³) is group A. It is **not differentiable** and not jit-able.

```python
from difflow.particles import fluidization_window, geldart_group

u_mf, u_t = fluidization_window(300e-6, 2600.0, 1.2, 1.8e-5, 0.45, 0.8)
print(f"u_mf = {float(u_mf):.3f} m/s, u_t = {float(u_t):.2f} m/s,",
      geldart_group(300e-6, 2600.0, 1.2))
```

**What is verified** (`tests/test_particles.py`): the Stokes limit; the Ergun
equation against hand arithmetic; $u_{mf}$ against the Ergun residual and an
independent `scipy` root find; terminal velocity against an independent
`scipy.optimize.brentq` solve of the force balance in the Stokes, intermediate
and Newton regimes; mutual agreement of the two sphere drag fits within 8 %;
`check_grads` (reverse mode) across all regimes. No textbook worked-example number
is reproduced from memory; compare against Perry's or McCabe/Geankoplis tables
before relying on a result to better than the correlation scatter (typically
5-10 %). The Richardson-Zaki exponent is piecewise (jumps of at most about 2 %
at $Re_t = 0.2, 1, 500$).

**References**: Haider & Levenspiel, *Powder Technol.* 58, 63 (1989); Turton &
Levenspiel, *Powder Technol.* 47, 83 (1986); Ergun, *Chem. Eng. Prog.* 48(2), 89
(1952); Wen & Yu, *AIChE J.* 12, 610 (1966); Richardson & Zaki, *Trans. Inst.
Chem. Eng.* 32, 35 (1954); Kunii & Levenspiel, *Fluidization Engineering*, 2nd
ed. (1991); Geldart, *Powder Technol.* 7, 285 (1973).

---

## Mechanical Separations

**Location**: `difflow/units/mechanical.py` (the `Settler`, `CakeFilter` and `Cyclone` units and the
pure-JAX library functions behind them, SI units, differentiable). Palette category: `mechanical`
(one module, one category, like `extraction`; the neighbouring per-unit categories `drying` and
`crystallization` each hold one unit).

These are the particle-scale separations of a solids-handling train: gravity settling and
thickening, cake filtration and gas cyclones. The single-particle physics (terminal velocity,
Richardson-Zaki hindered settling, Kozeny-Carman permeability) is in [Particles and Beds](#particles-drag)
(`difflow/particles.py`); these units apply it. **Centrifuges** (Sigma theory for a tubular bowl or a
disc stack, `Centrifuge` and `DiscStackCentrifuge`) are in `difflow_bio` and are documented in
[unit-operations-bio](unit-operations-bio.md#centrifuge); they are not repeated here.

**Streams.** mol/s, K, Pa. A solid is the pseudo-species `F_<solid_species>` (default `F_solid`) with
`solid_molar_mass` in kg/mol (default 1), so `F_solid` is numerically **kg/s** of dry solid. This is the
convention of the `Dryer` (its `info["fines"]` stream), so a dryer's product, fines and a cyclone's dust
connect without a conversion. Every non-solid species is a liquid-phase (or gas-phase, for the cyclone)
species and rides with the fluid. A particle size distribution is **a unit parameter** (fixed size bins),
not a stream property.

**What is and is not verified.** No published worked example (Geankoplis Ch. 14, McCabe, Smith & Harriott
Ch. 29-30, Coulson & Richardson Vol. 2, Cooper & Alley Ch. 4) was reproduced: the issue calls those numbers
"unverified, from recollection" and none could be recalled with confidence. All data below are synthetic
and labelled illustrative. The tests check the closed forms against independent oracles (a NumPy grid
maximum, an analytic Coe-Clevenger area, a batch curve generated *from* a known settling law,
`scipy` quadrature and scalar optimization, hand calculations) and `check_grads`. The Lapple/Shepherd-Lapple
constants (K = 16, the standard proportions, the Stairmand set) are typed from memory and flagged
**unverified** in the code.

(op-settler)=
### Settler

**Location**: `difflow/units/mechanical.py`

**Class**: `Settler`

**Description**: A gravity settler, either an overflow-rate clarifier (Hazen) or a continuous thickener
sized by Coe-Clevenger, Kynch or Talmage-Fitch, from tabulated flux data, a batch interface-height
curve or the particle model (`terminal_velocity` with Richardson-Zaki hindered settling).

#### Process Role

`Settler(params)(feed) -> (overflow, underflow, info)`. The underflow is the slurry at the specified
`underflow_concentration` $C_u$ (kg solid per m$^3$ slurry); the overflow the clarified liquid plus any solid
not captured. Design mode (`area=None`) returns the required area times `safety_factor`; with an `area` it
rates it (`info["area_ratio"]`).

#### Thickener area

With the feed solids rate $F_s$, the Coe-Clevenger area is set by the concentration between feed and
underflow with the smallest solids flux:

$$A = F_s \max_{C_0\le C\le C_u}\;\frac{1/C - 1/C_u}{v(C)}$$

Kynch's limiting-flux form $G_L = \min_C C v/(1 - C/C_u)$, $A = F_s/G_L$ is the same expression, and
`solids_flux(C, v, u)` is the total flux $C(v + u)$. **The maximum is a smooth log-sum-exp** over a fixed
concentration grid (`lse_max`, temperature $\tau = \overline{|a|}/\text{sharpness}$): it never
under-predicts and is within $\tau\ln N$ of the grid maximum (default `sharpness=1000`, 100 grid points:
under 0.5 %), so the area is differentiable in $C_u$, $Q_0$ and the settling-curve parameters, and
`sharpness` $\to\infty$ recovers the grid maximum.

*Particle model.* $v(C) = v_t(1 - C/\rho_p)^n$ with $v_t$ from `terminal_velocity` and the Richardson-Zaki
exponent $n$. **The exponent is piecewise in $Re_t$ (jumps of up to about 2 %): pass
`richardson_zaki_n` explicitly when optimizing over particle size**; `None` selects the piecewise value.

*Batch test.* `kynch_construction(t, z, C0)` returns the flux curve of one interface-height test: the tangent
at $(t, z)$ has slope $-v$ and meets $t=0$ at $z_i = z + v t$, where the concentration is
$C = C_0 z_0/z_i$. `thickener_area_talmage_fitch` reads the underflow time $t_u$ from the tangent that meets
$z_u = C_0 z_0/C_u$ latest and takes $A = Q_0 t_u/z_0$; on the same tangent nodes this is algebraically the
Coe-Clevenger area (the tests verify the identity, which holds exactly in the sharpness limit). Slopes are
PCHIP derivatives of the data (monotone, exact on straight segments); the test must be at or below the feed
concentration, and a compression zone is part of the data, not modelled.

*Overflow rate.* The design particle $d_p$ sets $A = SF\,Q_o/v_t(d_p)$; each size bin $d_i$ is captured in the
proportion $\min(1, v_{t,i}/v_o)$ (a smooth minimum, width $10^{-4}$).

```python
import jax.numpy as jnp
from difflow import Settler, SettlerParams, thickener_area_coe_clevenger

# SYNTHETIC settling data (illustrative): v(C) = v0 (C0/C)^2, whose Coe-Clevenger maximum is at C_u/2
# and whose area is Q0 C_u / (4 v0 C0) in closed form.
Q0, C0, Cu, v0 = 0.20, 20.0, 100.0, 2.0e-4
C = jnp.linspace(C0, Cu, 400)
A = thickener_area_coe_clevenger(Q0, C0, Cu, C, v0 * (C0 / C) ** 2, 1e6)
print(f"smooth-max area {float(A):.2f} m^2, analytic {Q0 * Cu / (4 * v0 * C0):.2f} m^2")

# the particle-model route: a 25 micron solid in water, Richardson-Zaki exponent fixed at 4.65
params = SettlerParams(underflow_concentration=300.0, d_p=25e-6, rho_p=2650.0, richardson_zaki_n=4.65,
                       safety_factor=1.25)
feed = {"F_solid": 5.0, "F_H2O": 50000.0 / 18.015, "T": 300.0, "P": 101325.0}   # 5 kg/s solid, 50 kg/s water
overflow, underflow, info = Settler(params)(feed)
print(f"required area {float(info['area_required']):.1f} m^2, design area {float(info['area']):.1f} m^2")
print(f"underflow {float(underflow['F_solid']):.2f} kg/s solid at {300.0:.0f} kg/m^3, "
      f"overflow {float(overflow['F_H2O'] * 18.015 / 1000.0):.2f} kg/s water")
```

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class SettlerParams:
    underflow_concentration: float      # C_u, kg solid / m^3 slurry
    method: str = "coe_clevenger"       # "overflow_rate" | "coe_clevenger" | "talmage_fitch" | "kynch"
    C_data: list = None                 # kg/m^3, ascending
    v_data: list = None                 # m/s
    t_data: list = None                 # s, batch curve
    z_data: list = None                 # m
    batch_C0: float = None              # kg/m^3 of the batch test
    d_p: float = None                   # m (design particle, or the particle model)
    rho_p: float = 2650.0               # kg/m^3
    liquid_density: float = 998.0       # kg/m^3
    viscosity: float = 1.0e-3           # Pa s
    sphericity: float = 1.0
    richardson_zaki_n: float = None     # None: piecewise in Re_t
    d_bins: list = None                 # m, overflow-rate capture
    mass_fractions: list = None
    capture: float = 1.0                # fraction of solids to the underflow (thickener methods)
    safety_factor: float = 1.0
    area: float = None                  # m^2 (rating); None designs
    sharpness: float = 1000.0
    n_grid: int = 100
    solid_species: str = "solid"        # F_<solid_species>, kg/s with solid_molar_mass = 1
    solid_molar_mass: float = 1.0       # kg/mol
    liquid_species: str = "H2O"
    liquid_MW: float = 18.015           # g/mol
```

**Limitations**: a thickener short of area is reported (`area_ratio`), not modelled; compression, rakes,
flocculation and scale-up factors are the user's `safety_factor`. The overflow rate of the clarifier is
based on the liquid overflow with full capture; the outlet flows then follow from the computed capture and
close the balance.

**Sources**: Coe & Clevenger, *Trans. AIME* 55, 356 (1916); Kynch, *Trans. Faraday Soc.* 48, 166 (1952);
Talmage & Fitch, *Ind. Eng. Chem.* 47, 38 (1955); McCabe, Smith & Harriott Ch. 29; Coulson & Richardson
Vol. 2 Ch. 5 (formulas recalled, no worked example reproduced).

(op-cakefilter)=
### CakeFilter

**Location**: `difflow/units/mechanical.py`

**Class**: `CakeFilter`

**Description**: Cake filtration by Ruth's equation at constant pressure, as a batch filter (press, leaf)
with wash and downtime or a rotary drum, designed or rated, with the throughput-maximizing cycle.

#### Process Role

`CakeFilter(params)(slurry) -> (cake, filtrate, info)`. The cake carries all the solid and `cake_moisture`
kg of liquid per kg dry cake (every non-solid species in the liquid's proportion); the filtrate is the rest.
The wash liquor is not a stream: `info` reports the wash volume and time, and `cake_moisture` is the
post-wash value. For the rotary drum `info["area_for_cost"]` is the drum area for
`SEPARATOR_COSTS["filter_rotary"]` (no cost entry is added here).

#### Ruth's equation

At constant pressure, with $V$ the filtrate volume,

$$\frac{t}{V} = \frac{K_p}{2}\,V + B,\qquad K_p = \frac{\mu\,\alpha\,c_s}{A^2\,\Delta P},\quad B = \frac{\mu R_m}{A\,\Delta P}$$

$c_s = \rho_f w/(1 - m w)$ is the mass of cake solids per volume of filtrate (`cake_solids_per_filtrate`;
the unit computes it from the stream and `cake_moisture` by a mass balance). The volume is the positive root
in the cancellation-safe form $V = 2t/(B + \sqrt{B^2 + 2K_p t})$ (`filtrate_volume`), exact at $B=0$ and at
$K_p \to 0$. At constant rate $q$ the pressure rises as
$\Delta P = (\mu q/A)(\alpha c_s q t/A + R_m)$ (`ruth_constant_rate`, inverse `time_to_pressure`).
`fit_ruth(t, V, A, dP, mu, c_s)` is the **Ruth plot**: a least-squares line of $t/V$ against $V$, slope
$K_p/2$ and intercept $B$, returning $\alpha$ and $R_m$. The specific resistance can also come from
`compressible_cake` ($\alpha = \alpha_0 \Delta P^s$) or `specific_cake_resistance`
($36 k_0 (1-\varepsilon)/(\varepsilon^3\rho_p(\phi d_p)^2)$, the Kozeny-Carman form of
`particles.kozeny_carman`).

#### The batch cycle and its optimum

A batch cycle is filtration, wash and downtime. The wash volume is $rV$ at $\phi$ times the final filtration
rate (`wash_time`; $\phi = 1$ is the standard assumption, plate-and-frame washing is often taken as
$\phi = 1/4$), so the throughput is $V/(t_f + t_w + t_d)$ (`cycle_throughput`, differentiable in $t_f$).
In terms of $V$ the cycle is $T(V) = \tfrac{K_p}{2}(1 + 2r/\phi)V^2 + B(1 + r/\phi)V + t_d$ and the optimum
$T = V\,T'$ gives $\tfrac{K_p}{2}(1 + 2r/\phi)V^2 = t_d$, **independent of $B$**. With no medium
resistance and no wash the optimum filtration time equals the downtime, $t_f = t_d$
(`optimal_cycle`; a user wash function `t_wash_fn(V)` is solved by a bracketed bisection with implicit
gradients). `CakeFilterParams(optimize_cycle=True)` rates a given area at its optimal cycle.

**Design and rating.** With `area=None` the unit sizes the filter so that a cycle of `cycle_time` (batch:
filtration + wash + downtime; rotary: one drum revolution) passes the feed, in closed form (a quadratic in
$1/A$). With an `area` it rates the cycle (`filtration_time`, or `cycle_time` and `submergence` for the
drum) and reports `throughput`, `solids_capacity` and `utilization` (feed over capacity; the streams follow
the feed). In rotary mode the cake forms during the submerged fraction $\psi$ of the revolution, so
$t_f = \psi\,T_c$.

```python
import jax
import jax.numpy as jnp
from difflow import (CakeFilter, CakeFilterParams, fit_ruth, ruth_constants, ruth_constant_pressure,
                     cycle_throughput, optimal_cycle)

# SYNTHETIC constant-pressure data (illustrative): the Ruth plot recovers alpha and R_m
A, dP, mu, c_s = 0.0439, 267e3, 1.0e-3, 23.5
K_p, B = ruth_constants(A, dP, mu, 3.0e10, c_s, 6.0e9)
V = jnp.linspace(0.05, 1.0, 12) * 1e-3
t = ruth_constant_pressure(V, K_p, B)
fit = fit_ruth(t, V, A=A, dP=dP, mu=mu, c_s=c_s)
print(f"alpha = {float(fit['alpha']):.3e} m/kg, R_m = {float(fit['R_m']):.3e} 1/m")

# the optimal filtration time by gradient: d(throughput)/d(t_f) = 0 at the optimum
K_p, B, t_d, r = 15.0, 3.0, 700.0, 0.2
opt = optimal_cycle(K_p, B, t_d, wash_ratio=r)
g = jax.grad(cycle_throughput)(opt["t_f"], K_p, B, t_d, r)
print(f"optimal t_f = {float(opt['t_f']):.1f} s, cycle {float(opt['t_cycle']):.1f} s, "
      f"gradient there {float(g):.1e}")

# the unit: size a press for 2 kg/s of solid in 40 kg/s of liquid, one cycle per hour
feed = {"F_solid": 2.0, "F_H2O": 40000.0 / 18.015, "T": 300.0, "P": 200000.0}
p = CakeFilterParams(delta_P=3e5, alpha=1e11, R_m=1e10, cake_moisture=0.4, cycle_time=3600.0,
                     downtime=600.0, wash_ratio=0.1)
cake, filtrate, info = CakeFilter(p)(feed)
print(f"area {float(info['area']):.1f} m^2, filtration {float(info['t_filtration']):.0f} s, "
      f"wash {float(info['t_wash']):.0f} s, cake {float(info['cake_mass_per_cycle']):.0f} kg per cycle")
```

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class CakeFilterParams:
    delta_P: float                      # Pa
    mode: str = "batch"                 # "batch" | "rotary"
    area: float = None                  # m^2 (rating); None designs from cycle_time
    alpha: float = None                 # m/kg ... or alpha0 (+ s), or d_p (+ voidage, rho_p)
    alpha0: float = None
    s: float = 0.0
    d_p: float = None
    voidage: float = None
    rho_p: float = None
    sphericity: float = 1.0
    k0: float = 5.0
    R_m: float = 0.0                    # 1/m
    viscosity: float = 1.0e-3           # Pa s
    liquid_density: float = 998.0       # kg/m^3
    cake_moisture: float = 0.0          # kg liquid / kg dry cake
    filtration_time: float = None       # s, batch rating
    optimize_cycle: bool = False
    cycle_time: float = None            # s: total cycle (batch design) or revolution (rotary)
    wash_ratio: float = 0.0             # wash volume / filtrate volume
    wash_rate_factor: float = 1.0       # wash rate / final filtration rate
    downtime: float = 0.0               # s
    submergence: float = 0.3            # rotary
    solid_species: str = "solid"
    solid_molar_mass: float = 1.0       # kg/mol
    liquid_species: str = "H2O"
    liquid_MW: float = 18.015           # g/mol
```

**Limitations**: constant pressure, no medium blinding; a compressible cake uses $\alpha$ at the filtration
$\Delta P$; the rotary drum has no wash or dewatering arc in its time budget; streams are time-averaged.
For the filter-press cycle exercise (wex21-2) the structure is `optimal_cycle` with a wash and `CakeFilter`
for the area; the published example was not reproduced.

**Sources**: Ruth, *Ind. Eng. Chem.* 27, 708 (1935); Geankoplis Sec. 14.2; McCabe, Smith & Harriott Ch. 30
(formulas recalled, no worked example reproduced).

(op-cyclone)=
### Cyclone

**Location**: `difflow/units/mechanical.py`

**Class**: `Cyclone`

**Description**: A gas cyclone with the Lapple cut size and grade-efficiency curve, standard geometries and
the Shepherd-Lapple pressure drop, for a fixed set of particle size bins.

#### Process Role

`Cyclone(params)(gas_with_solids) -> (clean_gas, solids, info)`. The inlet is a gas stream (mol/s per gas
species) that also carries the dust as `F_<solid_species>` (kg/s with the default molar mass), the same
convention as the dryer's fines. `solids` is the collected dust; `clean_gas` the gas and the dust that
escapes. The size distribution is the parameters `d_bins` and `mass_fractions`.

#### Lapple theory

With the inlet velocity $v_i = Q/(HW)$, the effective number of turns $N_e = (L_b + L_c/2)/H$ (6 for the
standard Lapple proportions) and the gas viscosity $\mu$,

$$d_{50} = \left[\frac{9\mu W}{2\pi N_e v_i(\rho_p - \rho_g)}\right]^{1/2},\qquad
\eta(d) = \frac{1}{1 + (d_{50}/d)^2},\qquad \Delta P = K\,\frac{HW}{D_e^2}\,\frac{\rho_g v_i^2}{2}$$

The overall efficiency is $\sum_i w_i\,\eta(d_i)$ over the bins, differentiable in $D$, the gas flow and
$\rho_p$. $d_{50}\propto v_i^{-1/2}$, so efficiency rises with flow at fixed $D$ (the pressure drop rises as
$v_i^2$). **Unverified, typed from memory:** $K = 16$ (`K_pressure`), the Lapple proportions
$H = D/2$, $W = D/4$, $D_e = D/2$, $S = D/8$, $L_b = L_c = 2D$, $J = D/4$ and the Stairmand
high-efficiency set (`CYCLONE_GEOMETRIES`); `geometry="custom"` with `ratios` takes your own, and
`delta_P_velocity_heads` replaces the Shepherd-Lapple form by a number of inlet velocity heads. Mass-loading
effects and re-entrainment are not modelled.

#### Dryer exhaust (wex21-3)

`Dryer(DryerParams(entrainment=f))` puts the fraction $f$ of the dry solids in `info["fines"]` (a solids
stream, `F_solid` in kg/s) and **not** in `air_out` (the psychrometric `dry_air_flow` counts every non-water
`F_*` as dry air). The hand-off is explicit: `dust_laden_gas(air_out, info["fines"])` merges them (the
fines' water becomes vapor) into a cyclone inlet. A `Flowsheet` cannot route `info`, so this step is plain
Python.

```python
from difflow import Cyclone, CycloneParams, Dryer, DryerParams, dust_laden_gas
from difflow.psychrometrics import MW_WATER, air_stream_from_state, state

# SYNTHETIC dryer and dust (illustrative numbers, not a textbook example)
wet = {"F_solid": 0.5, "F_H2O": 0.5 * 0.30 / MW_WATER, "T": 300.0, "P": 101325.0}
air = air_stream_from_state(state(T=373.15, W=0.01), 10.0)
dryer = Dryer(DryerParams(mode="continuous", X_c=0.15, X_eq=0.02, product_moisture=0.05, h=60.0,
                          solids_per_area=10.0, max_solids_T=340.0, entrainment=0.05))
dry, air_out, dinfo = dryer(wet, air)
inlet = dust_laden_gas(air_out, dinfo["fines"])

cyc = Cyclone(CycloneParams(D=2.0, rho_p=1500.0, gas_MW=28.5,
                            d_bins=[2e-6, 5e-6, 10e-6, 20e-6, 50e-6],
                            mass_fractions=[0.1, 0.2, 0.3, 0.3, 0.1]))
clean, dust, info = cyc(inlet)
print(f"d50 = {float(info['d50']) * 1e6:.1f} um, overall efficiency {float(info['overall_efficiency']):.3f}, "
      f"dP = {float(info['delta_P']):.0f} Pa, v_i = {float(info['inlet_velocity']):.1f} m/s")
print(f"dry solid out: product {float(dry['F_solid']):.4f} + collected {float(dust['F_solid']):.4f} "
      f"+ escaping {float(clean['F_solid']):.4f} = {float(dry['F_solid'] + dust['F_solid'] + clean['F_solid']):.4f} kg/s")
```

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class CycloneParams:
    D: float                            # body diameter, m
    rho_p: float                        # kg/m^3
    d_bins: list                        # m, fixed-length size bins
    mass_fractions: list                # normalized to 1
    geometry: str = "lapple"            # "lapple" | "stairmand_he" | "custom"
    ratios: list = None                 # [H, W, D_e, S, L_b, L_c, J] / D for "custom"
    n_turns: float = None               # N_e; None: (L_b + L_c/2) / H
    gas_MW: float = 28.96               # g/mol
    viscosity: float = 1.8e-5           # Pa s
    K_pressure: float = 16.0            # Shepherd-Lapple (unverified)
    delta_P_velocity_heads: float = None
    solid_species: str = "solid"
    solid_molar_mass: float = 1.0       # kg/mol
```

**Not included**: hydrocyclones, electrostatic precipitators and bag filters, screening and size
reduction, and any cost correlation (none is cited here, so no cyclone or thickener entry was added to
`economics/capital.py`).

**Sources**: Lapple, *Chem. Eng.* 58(5), 144 (1951); Shepherd & Lapple, *Ind. Eng. Chem.* 31, 972 (1939);
Cooper & Alley Ch. 4; Coulson & Richardson Vol. 2 Ch. 1 (recalled; constants unverified).

---

## Mass Transfer

**Location**: `difflow/mass_transfer.py` (pure JAX functions in SI units; not
unit operations, so they have no palette entry). They are the building blocks
for absorbers, strippers, packed-column hardware and any exercise that needs
a diffusivity, a film coefficient or a transfer-unit height. Properties are
*inputs*: core has no gas viscosity or surface-tension model, so the Onda
functions take `mu_G` and `sigma` as required arguments (there is no hidden
`sigma=0.072`). See the module docstring and `tests/test_mass_transfer.py` for
exactly what was and was not verified; in short, **no textbook worked example
is reproduced**, and the tables (diffusion volumes, Le Bas increments,
packings) are transcribed from memory.

(mass-transfer-diffusivities)=
### Diffusivity Correlations

`fuller_diffusivity(T, P, MW_A, MW_B, V_A, V_B)` is the Fuller-Schettler-Giddings
gas correlation ($D \propto T^{1.75}/P$, 5-10 %). `wilke_chang_diffusivity(T,
mu_B, MW_B, V_A, phi_B)` is the dilute-liquid correlation (10-20 %), with the
solute molar volume at its normal boiling point `V_A` in cm$^3$/mol from
`molar_volume_nbp` (Le Bas increments) and the solvent association factor
`phi_B` from `ASSOCIATION_FACTORS` (water 2.6; there is no silent default for
water). `stokes_einstein_diffusivity(T, mu, r)` is for large solutes. Both
published correlations are implemented in their native units (cm$^2$/s, atm,
cP) and converted at the boundary. Diffusion volumes are in `FULLER_VOLUMES`
and `FULLER_ATOMIC_VOLUMES`; `diffusion_volume` sums atoms.

```python
from difflow.mass_transfer import (
    fuller_diffusivity, wilke_chang_diffusivity, diffusion_volume,
    molar_volume_nbp, FULLER_VOLUMES, ASSOCIATION_FACTORS,
)

# water vapour in air, 25 C, 1 atm: experimental about 2.5e-5 m^2/s
D_air = fuller_diffusivity(298.15, 101325.0, 18.015, 28.96,
                           FULLER_VOLUMES["H2O"], FULLER_VOLUMES["air"])
print(f"D(H2O-air) = {float(D_air):.3e} m^2/s")

# ethanol in water at 25 C (experimental about 1.2e-9; Wilke-Chang runs ~20% high)
V_eth = molar_volume_nbp({"C": 2, "H": 6, "O": 1})          # 59.2 cm^3/mol
D_liq = wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, V_eth, ASSOCIATION_FACTORS["water"])
print(f"D(ethanol-water) = {float(D_liq):.3e} m^2/s")
```

(mass-transfer-film)=
### Film Theory and Overall Coefficients

`equimolar_flux` and `stagnant_film_flux` give the flux across a film of
thickness $\delta$; the stagnant-film (Stefan) form carries the bulk-flow
factor $1/(1-y)_{lm}$, computed with `log_mean`, which is smooth (series form)
when its two arguments coincide. `mass_transfer_coefficient_conversions` moves
a gas coefficient among $k_c$ (m/s), $k_G$ (mol/m$^2$/s/Pa) and $k_y$
(mol/m$^2$/s). Sherwood correlations (`ranz_marshall`, `frossling`,
`linton_sherwood`, `gilliland_sherwood`, `pipe_sherwood`,
`flat_plate_sherwood`, `sherwood_from_heat_transfer`) each state their range
in the docstring; the laminar-turbulent switches use the same smooth blend as
`difflow.heat_transfer`. Two-resistance theory:
`overall_gas_coefficient` ($1/K_y = 1/k_y + m/(E k_x)$),
`overall_liquid_coefficient`, `overall_K_G` (with Henry's constant) and
`resistance_fractions` (which film controls).

```python
from difflow.mass_transfer import (
    fuller_diffusivity, stagnant_film_flux, equimolar_flux, ranz_marshall,
    resistance_fractions,
)
from difflow.units.evaporator import water_saturation_pressure

# evaporation of water at 25 C into dry air across a 1 cm stagnant film
P, T = 101325.0, 298.15
D = fuller_diffusivity(T, P, 18.015, 28.96, 13.1, 19.7)
y_sat = water_saturation_pressure(T) / P
c = P / (8.314462618 * T)
N = stagnant_film_flux(D, c, y_sat, 0.0, 0.01)
print(f"flux = {float(N):.3e} mol/m^2/s (equimolar would be "
      f"{float(equimolar_flux(D, c, y_sat, 0.0, 0.01)):.3e})")

# sphere in stagnant fluid: Sh -> 2
print(float(ranz_marshall(0.0, 1.0)))

# which film controls? soluble (small m) vs sparingly soluble (large m)
for m in (0.01, 1e3):
    g, l = resistance_fractions(k_y=0.05, k_x=5.0, m=m)
    print(f"m = {m:g}: gas {float(g):.3f}, liquid {float(l):.3f}")
```

(mass-transfer-packed)=
### Packed-Column Mass Transfer (Onda, HTU/NTU)

`onda_wetted_area`, `onda_gas_coefficient` and `onda_liquid_coefficient`
implement Onda, Takeuchi & Okumoto (1968) for random packing, taking a
`PackingData` record (`d_p`, `a_p`, void fraction, critical surface tension
`sigma_c`, packing factor `F_p`; the small table `PACKINGS` is from memory and
unverified, so check a data sheet) or explicit `a_p`, `d_p`, `sigma_c`. The gas
function returns $k_c$ in m/s (convert with
`mass_transfer_coefficient_conversions`). **Corrected in #427**: the earlier
`difflow_cc` functions returned a liquid coefficient about $10^4$ too large
(factor $(\rho g/\mu)^{1/3}$ instead of $(\mu g/\rho)^{1/3}$, and $Re$ on the
dry rather than the wetted area), and a gas coefficient too large by $a_p d_p$
(the constant is 2.00, not 5.23, below 15 mm packing; blended smoothly between
12 and 18 mm unless `smooth=False`).

Heights and counts: `htu_og`, `htu_ol`, `htu_og_from_films`
($H_{OG} = H_G + \lambda H_L$), `ntu_og_colburn` (analytic, straight lines,
smooth at $\lambda = 1$), `ntu_og_integral` (Gauss-Legendre quadrature of
$\int dy/(y-y^*)$ for a curved equilibrium line, with an optional
$(1-y)_{lm}$ correction) and `hetp_from_htu` ($\mathrm{HETP} = H_{OG}\ln\lambda
/(\lambda-1)$, continuous through $\lambda = 1$). All differentiable, so
$\partial(\text{height})/\partial(k_G, k_L, a_w, L/G)$ comes from `jax.grad`.

```python
import jax
from difflow.mass_transfer import (
    PACKINGS, onda_wetted_area, onda_gas_coefficient, onda_liquid_coefficient,
    mass_transfer_coefficient_conversions, k_x_from_k_L, htu_og, htu_ol,
    htu_og_from_films, ntu_og_colburn, hetp_from_htu,
)

pk = PACKINGS["pall_metal_50"]          # from memory: verify before design use
a_w = onda_wetted_area(0.01, 1000.0, 1.0e-3, 0.072, pk)
k_c = onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, pk)
k_L = onda_liquid_coefficient(0.01, 1000.0, 1.0e-3, 1.5e-9, 0.072, pk)
print(f"a_w = {float(a_w):.1f} m2/m3, k_c = {float(k_c):.3e} m/s, k_L = {float(k_L):.3e} m/s")

G_m, y_in, y_out, m = 40.0, 0.02, 0.002, 0.8
k_y = mass_transfer_coefficient_conversions(k_c, "c", "y", 300.0, 101325.0)
k_x = k_x_from_k_L(k_L, 55500.0)

def column_height(L_over_G):
    lam = m / L_over_G
    H_OG = htu_og_from_films(htu_og(G_m, k_y, a_w), htu_ol(G_m * L_over_G, k_x, a_w), lam)
    return H_OG * ntu_og_colburn(y_in, y_out, 0.0, m, lam)

print(float(column_height(1.5)), float(jax.grad(column_height)(1.5)))
print(float(hetp_from_htu(0.5, 1.0)))     # HETP -> H_OG as lambda -> 1
```

## Column Hardware

**Location**: `difflow/units/column_design.py` (pure JAX, SI; a library, not
palette operations, so nothing here is registered). These functions turn the
vapor and liquid rates of a column (a `DistillationColumn` result, a
McCabe-Thiele design, an absorber) into a diameter, a pressure drop, margins to
the weeping / entrainment / flooding limits, and the height and shell volume
that `difflow.economics.capital.column_cost` needs. Every limit is a continuous
margin (never an exception on a traced value), so an optimizer can use it as a
smooth constraint, and everything is differentiable in the fraction of flood,
tray spacing, packing factor, rates and surface tension. Properties are inputs:
the surface tension, densities and viscosity are required arguments (there is no
silent `sigma = 0.02`).

**What was and was not verified.** The correlation constants were **written from
memory with no book at hand and are unverified**; each is marked so in its
docstring with the source it is believed to come from. They are fits of
*charts*: expect 10-20 % on a diameter or a flooding velocity, and worse on a
pressure drop. The tests check hand calculations, independent re-implementations,
scaling laws ($D \propto f_{\rm flood}^{-1/2}$, $D \propto (V\,MW/\rho_V)^{1/2}$),
sign changes of margins and gradients. **No published textbook worked example is
reproduced** (the issue's example numbers could not be checked), and two pieces
are placeholders rather than fits: the entrainment function and the GPDC
pressure-drop shape (below).

(op-tray-flooding)=
### Tray Flooding and Column Diameter

`flow_parameter` gives $F_{LV} = (L/V)\sqrt{\rho_V/\rho_L}$.
`fair_capacity_factor` is the flooding capacity factor on the net area,
$C_{SB} = [\alpha \log_{10}(1/F_{LV}) + \beta]\,(\sigma/0.02)^{0.2}$ with
$\alpha = 0.0744\,t + 0.01173$, $\beta = 0.0304\,t + 0.015$ ($t$ the tray
spacing in m): the closed-form fit of Fair's chart as quoted by Treybal
(**from memory, unverified**; believed valid for $0.01 \le F_{LV} \le 0.1$),
with optional `foaming_factor` and `hole_area_factor` multipliers.
`flooding_velocity` is $u_f = C_{SB}\sqrt{(\rho_L-\rho_V)/\rho_V}$.
`tray_column_diameter` sizes at a `fraction_flood` of $u_f$ on the net area
$A_n = A_c(1 - A_d/A_c)$ and returns `(D, info)` (`F_LV`, `C_SB`, `u_flood`,
`A_net`, `A_active`, `percent_flood`, ...); `tray_percent_flood` is the inverse,
for rating an existing column at a new throughput (debottlenecking).

```python
import jax
from difflow.units.column_design import (
    tray_column_diameter, tray_percent_flood, size_column, column_hardware_cost,
    oconnell_efficiency,
)

# Benzene-toluene-like column: rates in mol/s, MW in g/mol, rho in kg/m^3, sigma in N/m
col = dict(V_molar=100.0, MW_V=80.0, rho_V=2.9, L_molar=80.0, MW_L=85.0,
           rho_L=790.0, sigma=0.021)
D, info = tray_column_diameter(**col, tray_spacing=0.6, fraction_flood=0.8)
print(f"F_LV = {float(info['F_LV']):.3f}, C_SB = {float(info['C_SB']):.3f} m/s, "
      f"u_f = {float(info['u_flood']):.2f} m/s, D = {float(D):.2f} m")
print(f"{float(tray_percent_flood(**col, diameter=D)):.0f} % flood at the design rate")

# d(diameter)/d(fraction of flood) and d(cost)/d(fraction of flood)
def cost(ff):
    D, _ = tray_column_diameter(**col, fraction_flood=ff)
    eff = oconnell_efficiency(2.4, 0.3)          # alpha, viscosity in cP
    sizing = size_column(15.0, eff, D, tray_spacing=0.6)
    return column_hardware_cost(sizing)["total"]
print(float(cost(0.8)), float(jax.grad(cost)(0.8)))
```

The old `column_diameter(V, rho_V, rho_L, sigma=0.02, tray_spacing=0.6)` in
`difflow.units.distillation` hard-coded the vapor molecular weight (30 g/mol),
the flow parameter, $C_{SB}$ and ignored `sigma` and `tray_spacing`; it is now a
deprecated wrapper (`DeprecationWarning`) that calls `tray_column_diameter`
and requires `MW_V`, `L` and `MW_L` as keywords and `sigma` positionally.

(op-tray-hydraulics)=
### Tray Hydraulics

`tray_pressure_drop` is the standard textbook breakdown for a sieve tray,
$\Delta P = \rho_L g\,(h_d + h_w + h_{ow} + h_r)$: the dry (orifice) head
$h_d = \rho_V u_h^2 / (2 C_0^2 \rho_L g)$, the weir height plus the Francis
crest $h_{ow} = (Q_L/(1.84\,l_w))^{2/3}$ (`weir_crest`), and a residual head
$h_r = 12.5/\rho_L$ m (the constant is **from memory, unverified**; so is the
orifice coefficient default $C_0 = 0.8$). `info` returns each head and the
components in Pa, which sum to the total. `weep_point` returns the weeping
margin $u_h/u_{min} - 1$ (> 0: no weeping) with the Eduljee-type
$u_{min} = (K_2 - 0.90\,(25.4 - d_h[\text{mm}]))/\sqrt{\rho_V}$; the source reads
$K_2$ from a chart, and here it is a constant (default 30, **from memory,
unverified**, about 15 % on the turndown at which the margin changes sign).

`fractional_entrainment` is a **placeholder, not a published fit**: Fair's
entrainment chart could not be recalled, so it is
$\psi = \psi_{ref}\,(\%\text{flood}/80)^{4}(F_{LV}/0.1)$ with the order of
magnitude $\psi_{ref} = 0.01$ read from the chart; the three constants are
arguments, expect a factor of 2-3, and replace them with a fit from Kister or
Fair before a design depends on it (the usual limit is $\psi < 0.1$).

```python
from difflow.units.column_design import tray_column_diameter, tray_pressure_drop, weep_point

col = dict(V_molar=100.0, MW_V=80.0, rho_V=2.9, L_molar=80.0, MW_L=85.0,
           rho_L=790.0, sigma=0.021)
D, info = tray_column_diameter(**col)
Q_V, Q_L = float(info["Q_V"]), 80.0 * 85e-3 / 790.0       # m^3/s
dP, h = tray_pressure_drop(Q_V, Q_L, 2.9, 790.0, D)
print(f"dP/tray = {float(dP):.0f} Pa (dry {float(h['dP_dry']):.0f}, "
      f"liquid {float(h['dP_liquid']):.0f}, residual {float(h['dP_residual']):.0f})")
for turn in (1.0, 0.6, 0.4):
    margin, w = weep_point(turn * Q_V, Q_L, 2.9, D)
    print(f"{turn:.0%} of design vapor: weeping margin {float(margin):+.2f}")
```

(op-oconnell-efficiency)=
### O'Connell Efficiency

`oconnell_efficiency(alpha, mu_cP)` is $E_o = 0.492\,(\alpha\mu_L)^{-0.245}$
clipped to [0.1, 1] (the fit quoted by Seader, Henley & Roper; **from memory,
unverified**; it gives 0.492 at $\alpha\mu = 1$). It was promoted from the gas
plant (which re-exports it, so `difflow_refinery.gasplant.oconnell_efficiency`
is the same function; see [Refinery Unit Operations](unit-operations-refinery.md)).
The hard clip has a zero gradient outside [0.1, 1]; `smooth=0.02` swaps in a
softplus clip that stays differentiable. The number of actual trays is
`size_column`'s `n_theoretical / efficiency`.

(op-packed-column-flooding)=
### Packed-Column Flooding and Pressure Drop

`gpdc_flood(F_LV, F_p, rho_L, rho_V, mu_L)` is the gas mass flux at flood on
the Eckert / generalized pressure-drop chart: abscissa
$X = (L/G)\sqrt{\rho_V/(\rho_L-\rho_V)}$, ordinate
$Y = G^2 F_p \mu_L^{0.2}/(\rho_V(\rho_L-\rho_V)g)$ ($F_p$ in 1/m, $\mu_L$ in cP
inside the power), and the flood line
$\ln Y = -3.5021 - 1.028\ln X - 0.11093(\ln X)^2$ (a literature fit recalled
without its source: **from memory, unverified**; chart-reading accuracy,
**about 10-20 %** on $G$). `kister_gill_flood_dp(F_p)` is the pressure drop at
flood, $0.115\,F_p^{0.7}$ in H$_2$O/ft with $F_p$ in 1/ft (Kister & Gill, 1991;
**from memory, unverified**), in Pa/m. `gpdc_pressure_drop` is **not a fit of
the GPDC lines** (they could not be recalled): it anchors the flood value,
$\Delta P = \Delta P_{flood}(Y/Y_{flood})^{n}$ at fixed $L/G$, with an assumed
$n = 1.7$ chosen so that 70 % of flood gives about 30 % of the flood drop (the
common 0.25-0.5 in H$_2$O/ft design rule); expect $\pm 30$-50 % away from flood.
`packed_column_diameter(..., packing=, fraction_flood=0.7)` returns `(D, info)`
with the flooding and operating velocities, the pressure drop per metre and the
wetting rate against a minimum (2.2e-5 m$^2$/s, about 0.08 m$^3$/m/h; **from
memory, unverified**). `hetp` has the rules of thumb $18\,d_p$ (random packing)
and $100/a_p + 0.1$ m (structured), both **from memory, unverified**, and
`method="htu"` for `hetp_from_htu` of the mass-transfer section. `size_column`
turns stages, efficiency (or HETP), diameter and spacing into height, shell
volume, tray area and tray count; `column_hardware_cost` feeds them to
`column_cost`.

```python
import jax
from difflow.mass_transfer import PACKINGS
from difflow.units.column_design import (
    packed_column_diameter, size_column, column_hardware_cost, hetp,
)

pk = PACKINGS["pall_metal_25"]       # packing factor from memory: check a data sheet
air = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0,
           rho_L=1000.0, mu_L=1.0e-3)
D, info = packed_column_diameter(**air, packing=pk, fraction_flood=0.7)
print(f"D = {float(D):.2f} m, dP = {float(info['dP_per_m']):.0f} Pa/m "
      f"(flood {float(info['dP_flood_per_m']):.0f}), wetting margin {float(info['wetting_margin']):+.2f}")

def cost(ff):
    D, _ = packed_column_diameter(**air, packing=pk, fraction_flood=ff)
    sizing = size_column(10.0, 1.0, D, hetp=hetp("random", d_p=pk.d_p))
    return column_hardware_cost(sizing, "packing_random")["total"]
print(float(cost(0.7)), float(jax.grad(cost)(0.7)))
```

(op-packing-data)=
### Packing Data

The packing record is `difflow.mass_transfer.PackingData` (`d_p`, `a_p`, void
fraction, critical surface tension, packing factor `F_p`) with the table
`PACKINGS` shared with the Onda functions: ceramic Raschig and Berl 25 mm, Pall
rings in metal (25, 38, 50 mm) and plastic (25 mm), ceramic Intalox saddles (25,
38, 50 mm) and Mellapak 250Y (structured; `d_p = 0`, so the random-packing Onda
functions do not apply to it). **Every value is from memory and unverified**
(`PackingData.source` says so): packing factors after Treybal / Perry, areas
after Billet and Schultes. Build your own `PackingData` from a vendor data sheet
for a real design.

---

## Humidification and Cooling Towers

Two things live here: `difflow.psychrometrics`, a differentiable humid-air
property library, and the `CoolingTower` palette unit, a counterflow tower
designed and rated with the Merkel equation. The water correlations they share
(saturation pressure over liquid and ice, latent heat, liquid enthalpy) are in
`difflow.water`; `difflow.units.evaporator` re-exports its old names.

### Psychrometrics

**Location**: `difflow/psychrometrics.py` (pure JAX, SI; a library, so nothing
is registered in the palette). `T` is in K, `P` in Pa, enthalpy in J per kg of
**dry air**, the humidity ratio `W` in kg water per kg dry air, the specific
volume in m$^3$ per kg dry air.

**Model.** An ideal-gas mixture of dry air and water vapor with the ASHRAE
*Fundamentals* (Ch. 1) constants:

$$W = 0.621945\,\frac{p_w}{P - p_w},\qquad
h = c_{pa}\,t + W\,(h_{fg0} + c_{pv}\,t),\qquad
v = \frac{R_{da} T}{P}\Big(1 + \frac{W}{0.621945}\Big),$$

with $c_{pa} = 1006$, $c_{pv} = 1860$ J/kg/K, $h_{fg0} = 2501$ kJ/kg and $t$ in
degrees C. The saturation pressure is IAPWS-IF97 over liquid water above the
triple point and the IAPWS-2008 sublimation formula over ice below it, joined
by a C2 blend over $\pm 0.25$ K (no Python `if` on a traced temperature).
**No enhancement factor** is applied, so $W_s$ sits 0.3-0.5 % below the
real-gas ASHRAE table at 1 atm (the pressure `P` is an argument of every
function that needs it).

**Pick any two properties.** `state` fixes a state point from any two
independent properties of `T` (dry bulb), `W`, `RH`, `T_wb`, `T_dp` and `h`
(`W` with `T_dp` is not independent and is refused) and returns all of them,
plus `v`, `p_w`, `W_sat` and the degree of saturation `mu`:

```python
import jax
from difflow.psychrometrics import state

s = state(T=303.15, RH=0.5)                       # 30 C, 50 % RH
print(f"W = {float(s['W']) * 1e3:.2f} g/kg   h = {float(s['h']) / 1e3:.1f} kJ/kg")
print(f"dew point = {float(s['T_dp']) - 273.15:.1f} C   wet bulb = {float(s['T_wb']) - 273.15:.1f} C")

back = state(T_wb=s["T_wb"], h=s["h"])            # the same point from a different pair
print(f"round trip: T = {float(back['T']) - 273.15:.3f} C, RH = {float(back['RH']):.3f}")

# everything is differentiable, through the wet-bulb root find as well
dTwb_dRH = jax.grad(lambda rh: state(T=303.15, RH=rh)["T_wb"])(0.5)
print(f"dT_wb/dRH = {float(dTwb_dRH):.2f} K per unit RH")
```

The individual functions are `humidity_ratio(p_w, P)`,
`humidity_ratio_from_rh`, `rh_from_humidity_ratio`,
`saturation_humidity_ratio(T, P)`, `dew_point(W, P)`, `humid_enthalpy(T, W)`,
`humid_heat(W)`, `humid_volume(T, W, P)`, `saturation_enthalpy(T, P)`,
`wet_bulb_temperature(T, W, P, lewis=1.0)` and
`adiabatic_saturation_temperature(T, W, P)`.

**Wet bulb vs. adiabatic saturation (the Lewis relation).** The adiabatic
saturation temperature $T^*$ is the exact enthalpy balance
$h(T, W) = h_s(T^*) - (W_s^* - W)\,h_\ell(T^*)$. The psychrometer wet bulb is
the transfer balance
$\mathrm{Le}_f\,(c_{pa} + W c_{pv})(T - T^*) = (W_s^* - W)\,h_{fg}(T^*)$, where
`lewis` is the Lewis *factor* $\mathrm{Le}_f = h_c/(k_y c_s)$. At
$\mathrm{Le}_f = 1$ the two are the same number for this ideal-gas model, and
`lewis` above (below) 1 moves the wet bulb toward (away from) the dry bulb:

```python
from difflow.psychrometrics import (state, wet_bulb_temperature,
                                    adiabatic_saturation_temperature)

s = state(T=308.15, RH=0.30)
T, W = s["T"], s["W"]
print(f"adiabatic saturation {float(adiabatic_saturation_temperature(T, W)) - 273.15:.3f} C")
for le in (0.8, 1.0, 1.2):
    print(f"wet bulb, Le_f = {le}: {float(wet_bulb_temperature(T, W, lewis=le)) - 273.15:.3f} C")
```

Below the triple point the wick carries ice: the saturation pressure is the ice
curve and the latent heat includes the heat of fusion (333.5 kJ/kg, recalled).

**Processes.** `mix_air(states, dry_air_flows)` conserves dry air, water and
enthalpy exactly (and reports `mu > 1` when the mixture fogs);
`sensible_heat(state, Q, m_dry_air)`; `cool_dehumidify(state, T_coil,
bypass_factor)` for a coil with an apparatus dew point (it returns the leaving
state and the sensible, latent and total duty and the condensate);
`adiabatic_humidify(state, effectiveness=, W_out=)`. A cold-room chain
(cool and dehumidify, then reheat):

```python
from difflow.psychrometrics import state, cool_dehumidify, sensible_heat

warm = state(T=303.15, RH=0.6)                          # 30 C, 60 % RH, 1 kg/s dry air
cold, coil = cool_dehumidify(warm, T_coil=283.15, bypass_factor=0.1)
final = sensible_heat(cold, Q=3.0e3, m_dry_air=1.0)     # 3 kW of reheat
print(f"coil leaves {float(cold['T']) - 273.15:.1f} C at {float(cold['RH']):.0%} RH; "
      f"duty {float(coil['Q_total']) / 1e3:.1f} kW, latent share {float(coil['Q_latent'] / coil['Q_total']):.0%}")
print(f"reheated to {float(final['T']) - 273.15:.1f} C at {float(final['RH']):.0%} RH "
      f"(condensate {float(coil['condensate']) * 1e3:.2f} g/kg)")
```

**The air stream convention.** Humid air flows through a flowsheet as an
ordinary stream with two species, `H2O` (vapor) and `air` (dry air as a single
pseudo-species, molar mass 28.966 g/mol): `{"F_H2O", "F_air", "T", "P"}` in
mol/s, K and Pa. `air_stream_from_state(state, m_dry_air)` builds one from a
state and a dry-air mass flow (kg/s); `state_from_air_stream(stream)` reads the
state back; `dry_air_flow(stream)` is the dry-air mass flow. Every non-`H2O`
species counts as dry air, so an `N2`/`O2` stream also works, and a stream with
no dry air gives `W = 0`, not NaN. Use `species_order=["H2O", "air"]` for a
flowsheet that carries them.

```python
from difflow.psychrometrics import (state, air_stream_from_state,
                                    state_from_air_stream, dry_air_flow)

air = air_stream_from_state(state(T=305.0, RH=0.45), m_dry_air=12.5)
print({k: round(float(v), 3) for k, v in air.items()})
print(f"{float(dry_air_flow(air)):.2f} kg/s dry air, "
      f"T_wb = {float(state_from_air_stream(air)['T_wb']) - 273.15:.2f} C")
```

**The chart.** `psychrometric_chart_data(T_range, P, rh_lines, wb_lines,
h_lines, v_lines)` returns each family as NumPy lines (`W` against `T`, NaN
outside the physical region) and
`difflow.visualization.plot_psychrometric_chart(data, processes=[...])` draws
them with matplotlib (optional), overlaying process lines with arrows.
Relative humidity is solid, wet bulb dashed, enthalpy dash-dot and volume
dotted, so the families differ by line style as well as colour.

<!-- doc-test: skip: draws a figure; the data call it uses is run just above and in the tests -->
```python
import matplotlib.pyplot as plt
from difflow.psychrometrics import (state, cool_dehumidify, sensible_heat,
                                    psychrometric_chart_data)
from difflow.visualization import plot_psychrometric_chart

warm = state(T=303.15, RH=0.6)
cold, _ = cool_dehumidify(warm, 283.15, 0.1)
ax = plot_psychrometric_chart(
    psychrometric_chart_data(T_range=(258.15, 313.15)),
    processes=[{"states": [warm, cold, sensible_heat(cold, 3e3)], "label": "cold room"}])
plt.show()
```

```python
from difflow.psychrometrics import psychrometric_chart_data

d = psychrometric_chart_data()
print({k: len(d[k]) for k in ("rh", "wb", "h", "v")})
```

**What was and was not verified.** There is no humid-air table in the
repository and the textbook examples named in the issue could not be checked,
so nothing here is a transcribed table value. The tests compare against an
independent NumPy implementation of the Hyland-Wexler saturation pressure and
the ASHRAE formulas (constants recalled, flagged in the test), round trips for
every pair of inputs, the Lewis identity, mixing and coil energy balances,
`jax.test_util.check_grads` of $W$, $h$, $T_{wb}$ and $T_{dp}$ with respect to
$T$, RH and $P$, and a few familiar state points (30 C / 50 % RH has a dew
point of 18.4 C and a wet bulb of 22.0 C). Saturated-air $W_s$ at 0, 20 and
30 C is compared with ASHRAE values recalled from memory at **1 %**, not the
0.5 % the issue asked for: this module has no enhancement factor, and a
0.5 % comparison needs the transcribed table. The ice-line coefficients and the
heat of fusion are recalled and flagged as such.

(op-coolingtower)=
### CoolingTower

**Location**: `difflow/units/cooling_tower.py`

**Class**: `CoolingTower`

**Description**: Counterflow cooling tower sized or rated with the Merkel equation.

#### Process Role

Hot process water is cooled by air in a packed fill, mostly by evaporating a
percent or two of it. The tower has two inlets, `water_in` and (optionally)
`air_in`, and two outlets, `water_out` and `air_out`. Merkel combined the
sensible and latent transfer into one enthalpy driving force:

$$\frac{K a V}{L} = \int_{T_{out}}^{T_{in}} \frac{c_{pw}\,dT}{h_s(T) - h_a(T)},
\qquad h_a(T) = h_{a,in} + \frac{L}{G}\,c_{pw}\,(T - T_{out}),$$

where $h_s$ is the enthalpy of air saturated at the water temperature. $KaV/L$
(dimensionless) is the size of the fill. `mode="design"` takes the outlet
water temperature (`T_out`, or an `approach` above the wet bulb) and computes
$KaV/L$; `mode="rating"` takes $KaV/L$ (directly, as `Ka` and `V`, or from the
fill characteristic $C\,(L/G)^{-n}$) and finds `T_out` by a bisection that
differentiates by the implicit function theorem. Without an air stream the
unit takes `L_over_G` and the wet bulb `T_wb` (a call argument) and assumes the
inlet air is saturated at the wet bulb, as the textbook examples do; with one
it reads `L/G` from the two flows and the inlet enthalpy and wet bulb from the
state.

`quadrature="cti"` (the default) is the Cooling Technology Institute
four-point method: the integrand $1/(h_s - h_a)$ is averaged at 10, 40, 60 and
90 % of the range and multiplied by $c_{pw}(T_{in} - T_{out})$.
`quadrature="gauss"` is Gauss-Legendre with `n_quad` points, better when the
approach is close. The air leaves saturated (Merkel's assumption), which gives
the evaporation $E = G(W_{out} - W_{in})$; the make-up is
$E + D + B = E\,C/(C-1)$ for the drift $D$ (`drift_fraction`) and blowdown $B$
at `cycles_of_concentration` $C$.

**Feasibility.** If $L/G$ is too high for the approach the operating line
touches the saturation curve and $h_s - h_a \to 0$: the integral is infinite.
The unit checks the driving force on a 65-point grid and returns
`KaV_L = inf` with `info["feasible"] = False`, never a negative or NaN number;
all other outputs and the gradients stay finite. `info["L_over_G_max"]` is the
pinch limit (also `maximum_L_over_G`).

```python
import numpy as np
from difflow import CoolingTower, CoolingTowerParams, make_stream
from difflow.psychrometrics import MW_WATER

K = 273.15
L = 10e6 / (4186.0 * 8.0)                                  # 10 MW at a range of 8 K: 298.5 kg/s
hot = make_stream({"H2O": L / MW_WATER}, 35.0 + K, 3e5)

for lg in (0.8, 1.0, 1.2, 1.4):
    tower = CoolingTower(CoolingTowerParams(mode="design", L_over_G=lg, T_out=27.0 + K))
    cold, air_out, info = tower(hot, T_wb=22.0 + K)
    print(f"L/G = {lg}: KaV/L = {float(info['KaV_L']):.3f}  (pinch limit L/G = {float(info['L_over_G_max']):.2f})")

print(f"approach {float(info['approach']):.1f} K, effectiveness {float(info['effectiveness']):.2f}, "
      f"evaporation {float(info['evaporation']):.2f} kg/s ({float(info['evaporation_fraction']):.2%}), "
      f"make-up {float(info['makeup']):.2f} kg/s")
```

The example values at $L/G$ = 0.8, 1.0, 1.2 and 1.4 (1.24, 1.39, 1.59, 1.89) are
the issue's own illustration (ideal-gas humid air, IAPWS saturation pressure,
air saturated at 22 C, trapezoid rule), **not** published numbers; the tests
reproduce them to 1 % and against an independent NumPy trapezoid integration of
the same $h_s$ to $10^{-4}$.

Rating the same tower on a hot day, and the approach-versus-size trade-off
(the gradient of the fill volume with respect to the approach):

```python
import jax
from difflow import CoolingTower, CoolingTowerParams, make_stream
from difflow.psychrometrics import MW_WATER

K = 273.15
L = 10e6 / (4186.0 * 8.0)
hot = make_stream({"H2O": L / MW_WATER}, 35.0 + K, 3e5)

# a fixed tower (KaV/L = 1.6 at L/G = 1.2) on a 22 C and on a 26 C wet-bulb day
rate = CoolingTower(CoolingTowerParams(mode="rating", L_over_G=1.2, KaV_L=1.6))
for twb in (22.0, 26.0):
    _, _, info = rate(hot, T_wb=twb + K)
    print(f"wet bulb {twb} C -> cold water {float(info['T_out']) - K:.2f} C")

# fill volume V = (KaV/L) L / Ka for a tighter approach
def volume(approach):
    t = CoolingTower(CoolingTowerParams(mode="design", L_over_G=1.2, approach=approach, Ka=1.0))
    return t(hot, T_wb=22.0 + K)[2]["V"]

print(f"dV/d(approach) at 5 K = {float(jax.grad(volume)(5.0)):.0f} m^3/K")
```

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class CoolingTowerParams:
    mode: str = "design"          # | "rating"
    L_over_G: float = None        # water/dry-air mass ratio (without an air stream)
    KaV_L: float = None           # Merkel number (rating)
    Ka: float = None              # kg/(m^3 s), with V: KaV/L = Ka V / L (rating)
    V: float = None               # fill volume (m^3)
    fill_characteristic: list = None   # [C, n]: KaV/L = C (L/G)^-n (rating)
    T_out: float = None           # cold-water temperature (K) (design)
    approach: float = None        # T_out - T_wb (K) (design)
    P: float = 101325.0           # Pa
    quadrature: str = "cti"       # | "gauss"
    n_quad: int = 8               # Gauss points
    drift_fraction: float = 0.0002
    cycles_of_concentration: float = 4.0
    cp_water: float = 4186.0      # J/kg/K
    makeup_in_outlet: bool = True # outlet flow = inlet flow (basin after make-up)
```

#### Equations and Assumptions

- Merkel (1925): Lewis factor of 1, the air leaves saturated, the evaporated
  water is neglected in the air enthalpy balance (a 1-2 % effect that the
  Poppe method removes; Poppe and the e-NTU model are not included).
- Constant water specific heat and constant $L/G$ through the fill;
  ideal-gas humid air (`difflow.psychrometrics`).
- No fan, fill or spray pressure drop; the cold-water outlet is at the tower
  pressure (the basin), which is what closes a pumped loop.
- The evaporation fraction comes out near the rule of thumb
  $0.00085 \times 1.8 \times$ range (unverified, from recollection; the test
  only checks it to 30 %).

#### A cooling-water loop

A closed loop conserves its flow, so a recycle through `CoolingTower`,
`CentrifugalPump` and `CounterCurrentHX` needs something that sets the
circulation rate, as a level controller does on a real basin: the tower with
`makeup_in_outlet=False` returns the cold water short by the evaporation and
drift and a make-up unit tops it up. The converged loop is in
`tests/test_cooling_tower.py::TestCoolingWaterLoop` (it converges, closes the
exchanger/tower energy balance, and differentiates through the recycle with
respect to the fill size). Use `species_order=["H2O"]` for a water-only loop.

**Gradients.** Everything is differentiable: $KaV/L$ with respect to $L/G$,
$T_{in}$, $T_{out}$ and the wet bulb, rating `T_out` with respect to $KaV/L$,
the pinch limit with respect to the wet bulb, and the whole recycle loop.

---

## Drying

Convective drying removes the water a solid holds into a stream of hot, dry air.
Two things set the design: **how fast** the solid gives up its water (the drying
curve, a time for a batch tray and a residence time for a continuous dryer) and
**what the air can carry** (the water and enthalpy balances, bounded by the dew
point of the exhaust). `difflow` provides the drying-time and isotherm functions
as a library and the `Dryer` unit that applies them to the humid-air streams of
`difflow.psychrometrics`.

### Drying-Time Functions

**Location**: `difflow/units/dryer.py`

**Functions**: `dry_basis`, `wet_basis`, `free_moisture`, `solids_moisture`,
`equilibrium_moisture`, `water_activity`, `constant_drying_rate`,
`heat_transfer_coefficient_tray`, `drying_rate_curve`, `drying_periods`,
`drying_time`

Moisture is on the **dry basis**, $X$ kg water per kg bone-dry solid
(`dry_basis(w)` and `wet_basis(X)` convert); the *equilibrium moisture* $X^*$ is
what the solid holds in air of relative humidity $a_w$, and the *free moisture*
$X - X^*$ is what can be removed. `equilibrium_moisture(a_w, T, model, params)`
takes a GAB isotherm
$X = X_m C K a_w / [(1 - K a_w)(1 - K a_w + C K a_w)]$, a BET isotherm (GAB
with $K = 1$), a monotone table or a callable; the parameters are data you
supply (none are built in), and `water_activity` is the inverse (closed form for
GAB and BET).

The drying rate $R$ (kg water per m$^2$ of surface per s) of a solid with
loading $L_s/A$ (kg dry solid per m$^2$) is constant at
$R_c = h\,(T - T_{wb})/\lambda_{wb}$ while the surface is wet
(`constant_drying_rate`; the wet bulb comes from `difflow.psychrometrics`, the
coefficient $h$ from `difflow.heat_transfer` or the tray correlation
`heat_transfer_coefficient_tray(G, flow)`), and falls linearly in the free
moisture once $X$ drops below the critical moisture $X_c$:

$$t = \frac{L_s}{A\,R_c}\left[(X_1 - X_c) + (X_c - X^*)\ln\frac{X_c - X^*}{X_2 - X^*}\right]$$

(`drying_time`; `falling="integral"` replaces the logarithm by
$(L_s/A)\int dX/R$ over a tabulated rate curve, and `drying_periods` returns the
two periods separately). The Fick-diffusion series for a slab is not implemented.

**The smooth switch.** Which terms apply depends on where $X_c$ falls relative
to $X_1$ and $X_2$: $X_1 > X_c > X_2$, falling only ($X_1 < X_c$) and constant
only ($X_2 > X_c$). The function is one expression with no `if`: with the
softplus $\mathrm{sp}(z) = s\ln(1 + e^{z/s})$ the constant range is
$\mathrm{sp}(X_1 - X_c) - \mathrm{sp}(X_2 - X_c)$ and the falling period runs
from $X_a = X_1 - \mathrm{sp}(X_1 - X_c)$ to
$X_b = X_a - \mathrm{sp}(X_a - X_2)$. The width is $s = (X_c - X^*)/\texttt{sharpness}$
(default 200), so the result is within $e^{-200|X - X_c|/(X_c - X^*)}$ of the exact
piecewise answer away from the corner; exactly at the corner ($X_1 = X_c$) the
first-order offsets of the two periods cancel and the error is second order in $s$
(about $5\times10^{-6}$ of the time on the tests' problem). Larger `sharpness`
recovers the piecewise time, and the
gradient with respect to $X_c$, $X_1$, $X_2$, $X^*$, $R_c$ and $L_s/A$ exists
everywhere.

```python
import math
import jax
from difflow import drying_time, equilibrium_moisture, water_activity

# drying time (hours): X1 = 0.35 -> X2 = 0.10, critical 0.20, equilibrium 0.05 (kg/kg dry solid)
X1, X2, Xc, Xe = 0.35, 0.10, 0.20, 0.05
Rc, LsA = 2.78e-4, 25.0              # kg/m^2/s, kg dry solid/m^2
t = drying_time(X1, X2, Xc, Xe, Rc, LsA)
exact = LsA / Rc * ((X1 - Xc) + (Xc - Xe) * math.log((Xc - Xe) / (X2 - Xe)))
print(f"t = {float(t) / 3600:.4f} h  (closed form {exact / 3600:.4f} h)")

# the critical moisture is a smooth regime boundary: dt/dXc and dt/dRc
dXc, dRc = jax.grad(drying_time, argnums=(2, 4))(X1, X2, Xc, Xe, Rc, LsA)
print(f"dt/dXc = {float(dXc) / 3600:.1f} h per kg/kg,  dt/dRc = {float(dRc) / 3600:.3e} h per (kg/m^2/s)")

# a GAB isotherm (illustrative parameters, NOT those of a real material) and its inverse
gab = {"X_m": 0.07, "C": 10.0, "K": 0.85}
X = equilibrium_moisture(0.5, 300.0, "gab", gab)
print(f"X_eq(a_w = 0.5) = {float(X):.4f}, back to a_w = {float(water_activity(X, 300.0, 'gab', gab)):.3f}")
```

The numbers are an invented illustration, not a published example: the
Geankoplis (Ch. 9) and McCabe, Smith & Harriott (Ch. 24) drying examples were
not reproduced (their inputs could not be recalled with confidence), so the
tests check the closed forms against a numerical $\int dX/R$ of the same
piecewise rate curve instead. The tray correlations in
`heat_transfer_coefficient_tray`, $h = 0.0204\,G^{0.8}$ (parallel flow) and
$h = 1.17\,G^{0.37}$ (perpendicular), with $G$ in kg/m$^2$/h and $h$ in W/m$^2$/K,
are typed from memory and **unverified**; check them against the book.

(op-dryer)=
### Dryer

**Location**: `difflow/units/dryer.py`

**Class**: `Dryer`

**Description**: Tray (batch) or continuous (countercurrent or cocurrent)
convective dryer with air balances.

#### Process Role

Two inlets (`wet_solids`, `air_in`) and two outlets (`dry_solids`, `air_out`).

**Streams.** Air is the humid-air stream of `difflow.psychrometrics`
(`F_H2O`, `F_air`, `T`, `P`, mol/s). Solids are a stream with a dry-solid
pseudo-species `F_solid` and `F_H2O`, **both in mol/s**: the dry solid has no
molar mass, so `solid_molar_mass` (kg/mol, default 1) converts, and with the
default `F_solid` is numerically kg/s of dry solid while `F_H2O` is mol/s like
every other water stream (the moisture is $X = F_{H_2O} M_w / (F_{solid} M_s)$).
This differs from `Leacher`, whose flows are all kg/s. Any species of `air_in`
beyond `H2O` is counted as dry air and passed through.

**Batch** (`mode="batch"`): a tray dryer. The air at the inlet state flows over
a tray of loading $L_s/A$ = `load`/`area` at the constant rate
$R_c$ (`R_c`, or `h`, or `air_mass_velocity` through the tray correlation); the
unit returns the drying time to `product_moisture` (or the moisture reached in
`residence_time`), the cycle time with `turnaround_time`, the tray `area_required` for
the stream's throughput, and the exhaust air from the water and enthalpy balances
over the drying step (the air flow is the flow while the fan runs; the humidity
rise across the trays is assumed small and is reported). **Continuous**: the
overall water balance
$G\,(W_{out} - W_{in}) = L_s\,(X_{in} - X_{out})$ and the enthalpy balance in
humid-air and solids enthalpy fix the exhaust state; the solids leave at
$T_{s,out} = T_a - (T_a - T_{wb})\,R(X_{out})/R_c$ evaluated in the air at their
exit end (the inlet air for countercurrent, the outlet air for cocurrent, which
is solved by bisection with an implicit gradient). With `solids_per_area` the
unit also reports the required residence time, taking the constant rate in the air
at the wet end of the dryer, so countercurrent is slower on the constant rate but
hotter at the product end.

`info` carries the time or residence time, the constant/falling split, the
exhaust state, the dew-point margin (exhaust dry bulb minus its dew point, with a
`supersaturated` flag), the product-temperature margin against `max_solids_T`, the
heat duty `Q_air` of heating the air from `T_ambient`, `energy_per_kg` water
evaporated, `thermal_efficiency`, balance residuals recomputed from the streams, and
for the continuous dryer `fines` (the fraction `entrainment` of the dry solids
split off at the product moisture and temperature; a downstream cyclone takes it
from there).

```python
from difflow import Dryer, DryerParams
from difflow.psychrometrics import air_stream_from_state, state, MW_WATER

K = 273.15
air = air_stream_from_state(state(T=100.0 + K, W=0.01), 10.0)        # 10 kg/s dry air at 100 C
wet = {"F_solid": 0.5, "F_H2O": 0.5 * 0.30 / MW_WATER, "T": 300.0, "P": 101325.0}   # 0.5 kg/s dry, X = 0.30

for flow in ("countercurrent", "cocurrent"):
    dryer = Dryer(DryerParams(mode="continuous", flow=flow, X_c=0.15, X_eq=0.02, product_moisture=0.05,
                              h=60.0, solids_per_area=10.0, max_solids_T=340.0))
    dry, exhaust, info = dryer(wet, air)
    print(f"{flow:15s} exhaust {float(info['T_air_out']) - K:5.1f} C, product {float(info['T_solids_out']) - K:5.1f} C "
          f"(margin {float(info['solids_T_margin']):6.1f} K), dew-point margin {float(info['dew_point_margin']):.1f} K, "
          f"residence {float(info['residence_time_required']) / 60:.0f} min, "
          f"{float(info['energy_per_kg']) / 1e6:.2f} MJ/kg water")
```

A tray dryer: the time to dry a load, and how it responds to the air (the
gradients are exact, through the wet-bulb solve and the smooth switch):

```python
import jax
from difflow import Dryer, DryerParams
from difflow.psychrometrics import air_stream_from_state, state, MW_WATER

K = 273.15
wet = {"F_solid": 0.01, "F_H2O": 0.01 * 0.35 / MW_WATER, "T": 295.0, "P": 101325.0}

def drying_time(T_air, air_velocity):
    tray = Dryer(DryerParams(mode="batch", X_c=0.20, X_eq=0.05, product_moisture=0.10,
                             air_mass_velocity=air_velocity, load=12.5, area=0.5))
    air = air_stream_from_state(state(T=T_air, W=0.01), 0.5)
    return tray(wet, air)[2]["drying_time"]

t = drying_time(70.0 + K, 1.0)
dT, dG = jax.grad(drying_time, argnums=(0, 1))(70.0 + K, 1.0)
print(f"drying time {float(t) / 3600:.2f} h; d(time)/dT = {float(dT) / 3600:.3f} h/K, "
      f"d(time)/d(air mass velocity) = {float(dG) / 3600:.2f} h per kg/m^2/s")
```

**Design with gradients.** A countercurrent dryer of fixed size (residence time
4000 s) must dry the solids to the spec. For each inlet temperature the air flow that
just does it is solved for (with an implicit-function gradient); hotter air needs
less of it, so the energy per kg of water falls with the temperature until the
product reaches `max_solids_T`, which is added to the objective as a soft constraint.
The optimum is found by bisection on the sign of the exact gradient:

```python
import jax
from difflow import Dryer, DryerParams
from difflow.psychrometrics import air_stream_from_state, state, MW_WATER
from difflow.staged import _solve_monotone

K = 273.15
wet = {"F_solid": 0.5, "F_H2O": 0.5 * 0.30 / MW_WATER, "T": 300.0, "P": 101325.0}
T_max, t_avail = 350.0, 4000.0
dryer = Dryer(DryerParams(mode="continuous", flow="countercurrent", X_c=0.15, X_eq=0.02,
                          product_moisture=0.05, h=60.0, solids_per_area=10.0, max_solids_T=T_max))

def run(T, G):
    return dryer(wet, air_stream_from_state(state(T=T, W=0.01), G))[2]

def air_flow(T):                      # the air flow (kg/s) that dries the solids in t_avail
    return _solve_monotone(lambda G: run(T, G)["residence_time_required"] - t_avail, 4.0, 60.0)

def objective(T):                     # MJ/kg water + a soft limit on the product temperature
    info = run(T, air_flow(T))
    return info["energy_per_kg"] / 1e6 + 5.0 * jax.nn.softplus((info["T_solids_out"] - T_max) / 0.1)

grad = jax.jit(jax.grad(objective))
lo, hi = 350.0, 420.0
for _ in range(20):
    T = 0.5 * (lo + hi)
    lo, hi = (T, hi) if grad(T) < 0.0 else (lo, T)
T = 0.5 * (lo + hi)
info = run(T, air_flow(T))
print(f"inlet {T - K:.1f} C with {float(air_flow(T)):.1f} kg/s of air: product {float(info['T_solids_out']) - K:.1f} C "
      f"(limit {T_max - K:.1f} C), {float(info['energy_per_kg']) / 1e6:.2f} MJ/kg water")
```

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class DryerParams:
    mode: str = "batch"                # | "continuous"
    flow: str = "countercurrent"       # | "cocurrent" (continuous)
    X_c: float = 0.2                   # critical moisture, kg/kg dry solid
    X_eq: float = None                 # equilibrium moisture (else from the isotherm at the air RH, else 0)
    isotherm_model: str = None         # "gab" | "bet" | "table"
    isotherm_params: list = None       # [X_m, C, K] | [X_m, C] | [[a_w...], [X...]]
    R_c: float = None                  # constant rate, kg/m^2/s (else h (T - T_wb)/lambda)
    h: float = None                    # W/m^2/K (else from air_mass_velocity)
    air_mass_velocity: float = None    # kg/m^2/s, tray correlation
    tray_flow: str = "parallel"        # | "perpendicular"
    falling: str = "linear"            # | "integral" (with rate_curve = [[X...], [R/R_c...]])
    rate_curve: list = None
    sharpness: float = 200.0           # smooth-switch sharpness
    product_moisture: float = None     # spec, kg/kg dry solid
    residence_time: float = None       # s, instead of product_moisture
    load: float = None                 # kg dry solid per batch (batch)
    area: float = None                 # m^2 (batch); or solids_per_area
    solids_per_area: float = None      # Ls/A, kg/m^2 (continuous: reports the residence time)
    turnaround_time: float = 0.0       # s
    solid_species: str = "solid"       # stream key F_<name>
    solid_molar_mass: float = 1.0      # kg/mol (1: F_solid is kg/s)
    solids_cp: float = 1500.0          # J/kg/K
    cp_water: float = 4186.0
    max_solids_T: float = None         # K, reported as a margin
    heat_loss: float = 0.0             # W, used when adiabatic=False
    adiabatic: bool = True
    T_ambient: float = 293.15          # K, reference of the energy per kg and efficiency
    entrainment: float = 0.0           # fraction of the dry solids carried out as fines (continuous)
    lewis: float = 1.0
```

#### Equations and Assumptions

- Constant-rate period at the wet-bulb temperature, heat-transfer limited (no
  radiation or conduction from the tray); a falling rate linear in the free
  moisture, or a tabulated relative rate curve; the constant/falling switch is
  smooth (above).
- Batch: air at the inlet state over the whole tray. The solids stream's flow is
  the throughput and passes through; the exhaust flows are those of the drying step.
- Continuous: the product-temperature estimate above; no heat of wetting; constant
  solid and liquid-water specific heats; air out must stay above its dew point
  (reported, not enforced).
- Not implemented: the three-zone (preheat, constant-rate, falling-rate) rotary-dryer
  sizing of McCabe, the Fick-diffusion series, deep-bed humidity profiles, spray
  drying, fluid-bed hydrodynamics.
- `psychrometrics.wet_bulb_temperature` now starts its Newton solve below the boiling
  point, so it works for the 100-200 C air of a dryer (it returned NaN before).

**Gradients.** Drying time, energy per kg, the exhaust and product temperatures are
differentiable with respect to the air temperature, humidity and flow, the air
velocity (through $h$), $X_c$ and the load.

---

## Crystallization

Crystallization separates a solute as a solid by making the solution supersaturated,
by cooling, by evaporating solvent, or both. Two questions set the design: **how much**
solid forms (the yield from the solubility curve, including a hydrate that takes water
of crystallization into the crystal) and **how big** the crystals are (nucleation and
growth kinetics and the population balance). `difflow` provides the solubility, yield and
MSMPR functions as a library and the `Crystallizer` unit that applies them to a stream.
The REE oxalate, carbonate and hydroxide precipitators in
[unit-operations-ree](unit-operations-ree.md) (`difflow_ree`) are the *reactive*
precipitation version (a solubility-product-driven conversion, no size distribution);
the `Crystallizer` is the generic solubility-driven one.

**No solubility data, kinetic constant or hydrate transition temperature is built in.**
Every curve and constant in this section is synthetic and labelled illustrative; the
published worked examples (Geankoplis Sec. 12.11, McCabe, Smith & Harriott Ch. 27,
Wankat Ch. 17) were **not reproduced**, because their inputs and printed answers could not be
recalled with confidence. The tests check the closed forms against independent
oracles instead.

(op-crystallizer)=
### Crystallizer

**Location**: `difflow/units/crystallizer.py`

**Class**: `Crystallizer`

**Description**: Solubility-driven crystallizer, cooling, evaporative or both, with an
equilibrium yield (a hydrate or an anhydrous crystal) or an MSMPR kinetic mode.

#### Process Role

One inlet (the feed) and two outlets (`crystals`, `mother_liquor`), plus an `info`
dictionary. An `Evaporator` concentrate goes straight in (give the crystallizer the
evaporator's solvent name), or the unit evaporates itself (`mode="evaporative"`,
`"cooling_evaporative"`) and reports the evaporation duty with
`difflow.water.water_latent_heat`. Palette category: `crystallization` (the one category of
this unit, as `evaporation` and `drying` are for theirs).

**Streams.** mol/s, K, Pa. The crystal stream carries the hydrate as **solute plus bound
water**: `F_<solute>` is the moles of anhydrous solute in the crystals and `F_<solvent>`
is $\nu$ times that, the water of crystallization. Every species appears in every
outlet (zero where absent), so the mole balance closes species by species and a recycle
keeps one species list. The water evaporated leaves as vapor (in `info`, not a stream).

#### Yield balance with hydrates

With $S_0$ the feed solute, $W_0$ the feed solvent, $E$ the solvent evaporated
($W = W_0 - E$; kg/s) and a crystal of solute mass fraction
$f_s = M_s/(M_s + \nu M_w)$ and water fraction $f_w = 1 - f_s$, the solute and water balances

$$S_0 = c\,W_L + f_s C, \qquad W = W_L + f_w C$$

with the liquor saturated, $c = c_{sat}(T)$ (kg anhydrous solute per kg solvent), give the
hydrate mass flow in closed form

$$C = \frac{S_0 - c_{sat}\,W}{f_s - c_{sat}\,f_w}.$$

The numerator is clipped at zero with a softplus of width `soft` $\cdot S_0$ (default
$10^{-6}$), so an undersaturated feed gives zero yield and a finite gradient with no
Python `if`. A feed richer in solute than the hydrate stoichiometry would need an anhydrous
solid; the balance then reports a negative liquor water.

#### Solubility Curves and Hydrate Forms

`Solubility` is a pytree: `Solubility.polynomial(coeffs)` (in $T - 273.15$),
`Solubility.vant_hoff(A, B)` ($\ln c = A + B/T$) or `Solubility.table(T, c)` (PCHIP), each with
`hydrate_waters`, the molar masses and `per_100g=True` for data in g/100 g solvent.
A second solid form (`upper=`, `T_transition=`, `width=`) is a **smooth logistic blend** of
$c_{sat}$ and $\nu$ in $T$ (never a Python `if` on a traced $T$); the explicit alternative is to
pass the curve of the form you mean. The blend makes the crystal composition
fractional within a few widths of the transition, which is an approximation.

```python
import jax
from difflow import Solubility, crystal_balance, crystal_yield, supersaturation

# SYNTHETIC curve (illustrative, not data for any salt): c_sat = 0.35 + 0.0075 (T - 273.15)
sol = Solubility.polynomial([0.35, 0.0075], hydrate_waters=10, solute_MW=142.0)
S0, W0 = 0.60, 1.0                                   # kg/s solute, solvent in the feed
b = crystal_balance(S0, W0, 278.15, sol)
print(f"hydrate {float(b['crystals']):.4f} kg/s, yield {float(b['yield_fraction']):.3f} of the solute")
print(f"solute closes: {float(S0 - b['crystal_solute'] - b['liquor_solute']):.1e},"
      f" water closes: {float(W0 - b['crystal_water'] - b['liquor_water']):.1e}")
print("supersaturation of the feed at 5 C:", float(supersaturation(S0 / W0, 278.15, sol)))

# the yield is differentiable: dC/dT, dC/dE, dC/dS0
g = jax.grad(crystal_yield, argnums=(0, 2, 4))(S0, W0, 278.15, sol, 0.0)
print("dC/dS0, dC/dT, dC/dE =", [round(float(x), 4) for x in g])

# an undersaturated feed: zero yield, finite gradient
print(float(crystal_yield(0.1, 1.0, 278.15, sol)), float(jax.grad(crystal_yield, argnums=2)(0.1, 1.0, 278.15, sol)))
```

#### Parameters

<!-- doc-test: skip: field listing (a class sketch), not an executable example -->
```python
@dataclass
class CrystallizerParams:
    solute: str                         # stream key F_<solute>
    solute_MW: float                    # g/mol, anhydrous
    mode: str = "cooling"               # "cooling" | "evaporative" | "cooling_evaporative"
    T: float = None                     # K (evaporative: defaults to the feed T)
    water_evaporated: float = None      # kg/s ...
    evaporation_fraction: float = None  # ... or a fraction of the feed solvent
    solvent: str = "H2O"
    solvent_MW: float = 18.015
    solubility_model: str = "polynomial"   # "polynomial" | "vant_hoff" | "table"
    solubility_coeffs: list = None
    solubility_T: list = None
    solubility_c: list = None
    solubility_basis: str = "kg_per_kg"    # or "g_per_100g"
    hydrate_waters: float = 0.0
    upper_solubility_model: str = "polynomial"   # a second solid form above transition_T
    upper_solubility_coeffs: list = None
    upper_solubility_T: list = None
    upper_solubility_c: list = None
    upper_hydrate_waters: float = 0.0
    transition_T: float = None
    transition_width: float = 0.5          # K, smooth blend
    kinetics: list = None                  # [k_g, g, k_b, b, j] -> MSMPR kinetic mode
    residence_time: float = None           # s
    volume: float = None                   # m^3 (tau = volume / feed liquor flow)
    crystal_density: float = 2000.0        # kg/m^3
    shape_factor: float = 0.5236           # k_v, pi/6
    liquor_density: float = 1000.0
    cp_solution: float = 4186.0            # J/kg/K
    heat_of_crystallization: float = 0.0   # J per kg of crystal, released
    soft: float = 1e-6
```

#### Equations and Assumptions

- Equilibrium mode: liquor saturated at the outlet temperature; a stoichiometric hydrate
  (or anhydrous) crystal. Kinetic mode: the liquor leaves at $c_{sat} + \Delta C$ (below).
- Duty $Q = m_F c_p (T - T_F) + E\,h_{fg}(T) - \Delta H_c C$ (W; negative is cooling),
  with constant $c_p$ and a user-supplied heat of crystallization (default 0).
- Constant liquor density converts the liquor flow to a volume flow; no fines destruction,
  classification, agglomeration, breakage, size-dependent growth or polymorph kinetics.
- Not implemented: an adiabatic (vacuum flash) mode that solves the evaporation from the
  heat balance; the evaporation is a specification.

```python
import jax
from difflow import Crystallizer, CrystallizerParams, make_stream

# synthetic curve and Na2SO4-like molar masses (illustrative only)
params = CrystallizerParams(solute="Na2SO4", solute_MW=142.0, mode="cooling", T=278.15,
                            solubility_coeffs=[0.35, 0.0075], hydrate_waters=10.0)
feed = make_stream({"Na2SO4": 0.60 * 1000 / 142.0, "H2O": 1.0 * 1000 / 18.015}, T=333.15, P=1e5)
crystals, liquor, info = Crystallizer(params)(feed)
print(f"yield {float(info['yield_kg_s']):.4f} kg/s of solute, hydrate {float(info['crystal_mass']):.4f} kg/s")
print(f"bound water / solute in the crystals = {float(crystals['F_H2O'] / crystals['F_Na2SO4']):.1f} mol/mol")
print(f"cooling duty {float(info['Q']) / 1e3:.1f} kW")

def yield_of_T(T):
    p = CrystallizerParams(solute="Na2SO4", solute_MW=142.0, T=T,
                           solubility_coeffs=[0.35, 0.0075], hydrate_waters=10.0)
    return Crystallizer(p)(feed)[2]["yield_kg_s"]

print("d(yield)/dT =", float(jax.grad(yield_of_T)(278.15)), "kg/s/K")
```

**Evaporator then Crystallizer** (the structure of exercise wex20-2, with synthetic data). The
crystallizer takes the evaporator concentrate; the mother liquor can be recycled to the
evaporator through a `Mixer` and a tear stream, and at steady state the crystals carry
exactly the fresh solute:

```python
from difflow import (Crystallizer, CrystallizerParams, Evaporator, EvaporatorParams, Mixer,
                     make_stream)
from difflow.flowsheet import Flowsheet, Unit

evap = Evaporator(EvaporatorParams(solute_species=["Na2SO4"], solute_MW=[142.0], solvent="water",
                                   x_product=0.30, bpr_model="ideal", vant_hoff_i=3.0,
                                   P_vapor_space=60e3, steam_P=300e3, U=1500.0))
xtal = Crystallizer(CrystallizerParams(solute="Na2SO4", solute_MW=142.0, solvent="water", T=278.15,
                                       solubility_coeffs=[0.35, 0.0075], hydrate_waters=10.0))
fresh = make_stream({"Na2SO4": 0.30 * 1000 / 142.0, "water": 1.7 * 1000 / 18.015}, T=313.15, P=1e5)

fs = Flowsheet(species_order=["water", "Na2SO4"])
fs.add_feed("feed", fresh)
fs.add_unit(Unit("mix", Mixer(["water", "Na2SO4"]), ["feed", "recycle"], ["evap_in"]))
fs.add_unit(Unit("evap", evap, ["evap_in"], ["conc", "vap"]))
fs.add_unit(Unit("xtal", xtal, ["conc"], ["crystals", "liquor"]))
fs.add_recycle("liquor", "recycle")
guess = {"recycle": make_stream({"Na2SO4": 0.5, "water": 40.0}, T=278.15, P=1e5)}
out = fs.solve(tear_initial=guess, tol=1e-12, max_iter=400, on_nonconvergence="raise")
print("crystals / fresh solute =", float(out["crystals"]["F_Na2SO4"] / fresh["F_Na2SO4"]))
```

### MSMPR Functions

**Location**: `difflow/units/crystallizer.py`

**Functions**: `msmpr_distribution`, `msmpr_moments`, `msmpr_dominant_size`,
`msmpr_mass_median_size`, `msmpr_mass_cv`, `msmpr_from_magma`, `fit_msmpr`, `growth_rate`,
`nucleation_rate`, `msmpr_steady_state`

For a steady mixed-suspension, mixed-product-removal crystallizer with residence time
$\tau$, size-independent growth rate $G$ and nuclei density $n_0$ (Randolph & Larson):

$$n(L) = n_0\,e^{-L/(G\tau)}, \qquad m_k = \int_0^\infty L^k n\,dL = k!\,n_0\,(G\tau)^{k+1},$$

the suspension (magma) density is $M_T = 6\rho_c k_v n_0 (G\tau)^4$, the dominant
(mass-mode) size, the maximum of $L^3 n(L)$, is $L_D = 3G\tau$, and the mass median size
$L_{50} = x_{50}\,G\tau$ with $x_{50} = 3.672\ldots$ the root of the cumulative mass
fraction of the gamma(4) law at one half; `msmpr_mass_median_size` computes it (Newton on
the gamma(4) law), it is not typed in. The coefficient of variation of the mass
distribution is $1/\sqrt{4} = 50\%$. `msmpr_from_magma(M_T, rho_c, k_v, G, tau)` gives
$n_0$ from a measured magma density and `fit_msmpr(L, n)` is the "$\ln n$ vs. $L$" line
(slope $-1/(G\tau)$, intercept $\ln n_0$).

**Kinetics.** With the absolute supersaturation $\Delta C = c - c_{sat}$ (kg solute per kg
solvent), $G = k_g\,\Delta C^{g}$ (`growth_rate`) and $B_0 = k_b\,M_T^{\,j}\,\Delta C^{b}$
(`nucleation_rate`, $M_T$ in kg of crystal per m$^3$ of liquor), $B_0 = n_0 G$.
`msmpr_steady_state` closes them: the solute balance gives $M_T(\Delta C)$, the third moment
requires $M_T = 6\rho_c k_v B_0 G^3\tau^4$, and the scalar root in $\Delta C$ is found by a
bisection in $\ln(\Delta C/\Delta C_{max})$ with an implicit-gradient Newton step
(`difflow.staged._solve_monotone`), so gradients with respect to $\tau$, $k_g$ and $k_b$ are
the implicit-function derivatives. In the log form the residual
$(1-j)\ln M_T - (b+3g)\ln\Delta C - \text{const}$ is monotone for $j \le 1$ (larger $j$ is
rejected). Since $G$ falls as $\tau$ grows ($\Delta C$ falls), $L_D(\tau)$ is **not**
$3G\tau$ at a fixed $G$; as $k_g \to \infty$ the supersaturation vanishes and the yield approaches the
equilibrium yield.

```python
import math
import jax
import jax.numpy as jnp
import numpy as np
from difflow import (Solubility, fit_msmpr, msmpr_distribution, msmpr_dominant_size,
                     msmpr_mass_median_size, msmpr_moments, msmpr_steady_state)

# moments from the closed form, and the "ln n vs L" fit
n0, G, tau = 3.0e9, 2.0e-8, 1800.0
print("m0..m3 =", np.asarray(msmpr_moments(n0, G, tau)))
L = np.linspace(1e-5, 8 * G * tau, 40)
Gt, n0_fit = fit_msmpr(L, msmpr_distribution(L, n0, G, tau))
print(f"fit: G tau = {float(Gt):.3e} m (true {G * tau:.3e}), n0 = {float(n0_fit):.3e}")
print(f"L_D = {float(msmpr_dominant_size(G, tau)) * 1e6:.1f} um,"
      f" L_50 = {float(msmpr_mass_median_size(G, tau)) * 1e6:.1f} um")

# residence time versus median size, with SYNTHETIC kinetics (k_g, g, k_b, b, j)
sol = Solubility.polynomial([0.35, 0.0075], hydrate_waters=10, solute_MW=142.0)
kin = (2.0e-6, 1.5, 1.0e9, 2.0, 0.6)

def median_um(t):
    r = msmpr_steady_state(0.60, 1.0, 278.15, sol, kin, t, crystal_density=1460.0, liquor_density=1250.0)
    return r["L_50"] * 1e6

for t in (300.0, 900.0, 3600.0):
    print(f"tau = {t:6.0f} s: L_50 = {float(median_um(t)):7.1f} um, dL_50/dtau = {float(jax.grad(median_um)(t)):.4f} um/s")
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
