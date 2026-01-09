# Rate-Based Operations: Planning Document

## Overview

This document outlines the plan for implementing rate-based models for carbon capture unit operations in difflow_cc. Rate-based models provide higher fidelity than equilibrium-stage models by explicitly accounting for mass and heat transfer limitations.

## Background

### Equilibrium vs Rate-Based Models

| Aspect | Equilibrium-Stage | Rate-Based |
|--------|-------------------|------------|
| Assumption | Phases reach equilibrium at each stage | Finite mass/heat transfer rates |
| Accuracy | Good for high-efficiency trays | Better for packed columns |
| Complexity | Lower (algebraic) | Higher (differential + algebraic) |
| Parameters | Stage efficiency (empirical) | Fundamental transport properties |
| Extrapolation | Limited | More reliable |

### Why Rate-Based for Carbon Capture?

1. **Packed columns dominate** - Most amine absorbers use structured packing
2. **Reaction enhancement** - Fast CO2-amine reactions require proper kinetic treatment
3. **Temperature profiles** - Exothermic absorption creates non-isothermal operation
4. **Scale-up confidence** - Fundamental models extrapolate better to industrial scale

## Architecture

### Proposed Module Structure

```
src/difflow_cc/
├── rate_based/
│   ├── __init__.py
│   ├── film_model.py         # Two-film theory implementation
│   ├── penetration_model.py  # Higbie penetration theory
│   ├── segment.py            # Single column segment
│   ├── column.py             # Full column integration
│   ├── correlations/
│   │   ├── __init__.py
│   │   ├── onda.py           # Onda correlations (random packing)
│   │   ├── billet_schultes.py # Billet-Schultes (structured packing)
│   │   ├── rocha.py          # Rocha-Bravo-Fair (structured)
│   │   └── interfacial.py    # Interfacial area correlations
│   └── packings/
│       ├── __init__.py
│       ├── database.py       # Packing property database
│       └── data/
│           └── packings.yaml # Packing specifications
```

### Core Components

#### 1. Film Model (`film_model.py`)

The two-film model with chemical reaction:

```
Gas Bulk    Gas Film    Interface    Liquid Film    Liquid Bulk
   |           |            |             |              |
  P_CO2 ----> P_i -------> P* <-------- C_i <--------- C_CO2
              kG           H            kL·E
```

Key equations:
- Gas-side flux: `N = kG · (P_bulk - P_i)`
- Liquid-side flux: `N = kL · E · (C_i - C_bulk)`
- Interface equilibrium: `P_i = H · C_i`
- Enhancement factor: `E = f(Ha, E_inf)`

```python
@dataclass
class FilmModelParams:
    """Parameters for two-film model."""
    kG: float  # Gas-side mass transfer coefficient (mol/m²/s/Pa)
    kL: float  # Liquid-side mass transfer coefficient (m/s)
    a: float   # Interfacial area (m²/m³)
    H: float   # Henry's constant (Pa·m³/mol)
    E: float   # Enhancement factor (dimensionless)

def film_model_flux(
    P_bulk: Array,
    C_bulk: Array,
    params: FilmModelParams,
) -> Array:
    """Calculate mass transfer flux using two-film model."""
    # Overall mass transfer coefficient
    KG = 1 / (1/params.kG + params.H/(params.kL * params.E))

    # Equilibrium pressure
    P_eq = params.H * C_bulk

    # Flux
    N = KG * params.a * (P_bulk - P_eq)
    return N
```

#### 2. Mass Transfer Correlations

**Onda Correlation** (random packing):
```python
def onda_kL(
    L: Array,      # Liquid mass velocity (kg/m²/s)
    rho_L: Array,  # Liquid density (kg/m³)
    mu_L: Array,   # Liquid viscosity (Pa·s)
    D_L: Array,    # Liquid diffusivity (m²/s)
    a_p: float,    # Packing specific area (m²/m³)
    d_p: float,    # Packing nominal diameter (m)
    sigma: Array,  # Surface tension (N/m)
    sigma_c: float,# Critical surface tension (N/m)
) -> Array:
    """Onda liquid-side mass transfer coefficient."""
    # Wetted area
    a_w = a_p * (1 - exp(-1.45 * (sigma_c/sigma)**0.75
                         * (L/(a_p*mu_L))**0.1
                         * (L**2 * a_p/(rho_L**2 * g))**-0.05
                         * (L**2/(rho_L * sigma * a_p))**0.2))

    # Liquid-side coefficient
    kL = 0.0051 * (L/(a_w * mu_L))**(2/3) \
               * (mu_L/(rho_L * D_L))**(-0.5) \
               * (a_p * d_p)**0.4 \
               * (rho_L / (mu_L * g))**(-1/3)
    return kL, a_w
```

**Billet-Schultes** (structured packing):
```python
def billet_schultes_kL(
    L: Array,
    rho_L: Array,
    mu_L: Array,
    D_L: Array,
    packing: StructuredPacking,
) -> Array:
    """Billet-Schultes for structured packing."""
    # Reynolds number
    u_L = L / rho_L
    Re_L = u_L * packing.d_h * rho_L / mu_L

    # Liquid holdup
    h_L = (12 * mu_L * u_L * packing.a / (rho_L * g))**(1/3)

    # Mass transfer coefficient
    kL = packing.C_L * (D_L / packing.d_h) * Re_L**0.5 * Sc_L**(1/3)
    return kL
```

#### 3. Column Segment (`segment.py`)

A differential segment of the column:

```python
@dataclass
class SegmentState:
    """State variables for a column segment."""
    # Gas phase
    y_CO2: Array      # CO2 mole fraction
    T_G: Array        # Gas temperature (K)
    G: Array          # Gas molar flow (mol/s)

    # Liquid phase
    loading: Array    # CO2 loading (mol/mol amine)
    T_L: Array        # Liquid temperature (K)
    L: Array          # Liquid molar flow (mol/s)

def segment_equations(
    state: SegmentState,
    z: float,
    params: SegmentParams,
) -> SegmentState:
    """ODEs for a column segment.

    Mass balances:
        d(G·y)/dz = -N·a·A_c
        d(L·x)/dz = +N·a·A_c

    Energy balances:
        d(G·H_G)/dz = -Q_GL - N·ΔH_abs
        d(L·H_L)/dz = +Q_GL + N·ΔH_abs
    """
    # Calculate fluxes
    N_CO2 = calculate_co2_flux(state, params)
    Q_GL = calculate_heat_transfer(state, params)

    # Mass balances
    dG_dy = -N_CO2 * params.a * params.A_cross
    dL_dx = +N_CO2 * params.a * params.A_cross

    # Energy balances
    delta_H = params.heat_of_absorption  # J/mol
    dT_G = (-Q_GL - N_CO2 * delta_H) / (state.G * params.Cp_G)
    dT_L = (+Q_GL + N_CO2 * delta_H) / (state.L * params.Cp_L)

    return SegmentState(
        dy_CO2=dG_dy / state.G,
        dT_G=dT_G,
        dG=...,
        dloading=dL_dx / (state.L * params.C_amine),
        dT_L=dT_L,
        dL=...,
    )
```

#### 4. Full Column (`column.py`)

Integrate segments with boundary conditions:

```python
class RateBasedAbsorber:
    """Rate-based packed absorber column."""

    def __init__(self, params: RateBasedAbsorberParams):
        self.params = params
        self.packing = get_packing(params.packing_type)

    def __call__(
        self,
        gas_in: Stream,
        liquid_in: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Solve counter-current column."""

        # Set up ODE system
        def column_odes(z, state):
            return segment_equations(state, z, self.params)

        # Boundary conditions (counter-current)
        # Gas enters at z=0, liquid enters at z=H
        bc_gas = gas_inlet_conditions(gas_in)
        bc_liquid = liquid_inlet_conditions(liquid_in)

        # Solve BVP using diffrax
        solution = solve_bvp(
            column_odes,
            bc_gas,
            bc_liquid,
            z_span=(0, self.params.height),
        )

        return gas_out, liquid_out, profiles
```

## Implementation Plan

### Phase 1: Foundation (Core Infrastructure)

1. **Packing database**
   - Create `packings.yaml` with common packings (Mellapak, IMTP, Pall rings)
   - Properties: specific area, void fraction, characteristic length, correlation constants

2. **Mass transfer correlations**
   - Implement Onda (random packing)
   - Implement Billet-Schultes (structured packing)
   - Validate against literature data

3. **Enhancement factor models**
   - Pseudo-first-order (fast reaction)
   - Instantaneous reaction limit
   - General Hatta number correlation

### Phase 2: Single Segment Model

4. **Film model implementation**
   - Two-film theory with reaction
   - JAX-compatible for AD
   - Unit tests with analytical limits

5. **Segment state and equations**
   - Define state dataclass (PyTree)
   - Implement mass/energy balance ODEs
   - Verify conservation laws

### Phase 3: Column Integration

6. **Counter-current BVP solver**
   - Use diffrax for ODE integration
   - Shooting method or collocation for BVP
   - Handle stiff systems (fast reactions)

7. **Complete absorber model**
   - Wrapper matching equilibrium-stage API
   - Output profiles (T, composition vs height)
   - Gradient verification

### Phase 4: Extensions

8. **Stripper model**
   - Adapt absorber framework
   - Add reboiler/condenser models
   - Steam stripping option

9. **Heat integration**
   - Cross heat exchanger
   - Lean/rich heat recovery
   - Intercooling

10. **Validation**
    - Compare with ASPEN Rate-Based
    - Validate against pilot plant data
    - Benchmark computational performance

## Key Challenges

### 1. Stiff ODEs
Fast reactions create stiff systems. Mitigations:
- Use implicit ODE solvers (diffrax supports these)
- Adaptive stepping
- Quasi-steady-state for very fast reactions

### 2. Counter-Current BVP
Counter-current flow creates boundary value problem. Approaches:
- Shooting method with Newton iteration
- Orthogonal collocation
- Finite difference discretization

### 3. JAX Compatibility
All operations must be JAX-traceable:
- No Python control flow on traced values
- Use `jax.lax.cond`, `jax.lax.while_loop`
- Register custom classes as PyTrees

### 4. Computational Cost
Rate-based is more expensive than equilibrium:
- JIT compilation amortizes setup cost
- Consider surrogate models for optimization inner loops
- Profile and optimize hot paths

## API Design

### Compatibility with Equilibrium Models

Rate-based models should be drop-in replacements:

```python
# Equilibrium-stage (existing)
from difflow_cc import AbsorberParams, AmineAbsorber

absorber = AmineAbsorber(AbsorberParams(
    solvent='MEA',
    n_stages=15,
    L_G_ratio=3.5,
))
gas_out, liquid_out, info = absorber(gas_in)

# Rate-based (new)
from difflow_cc.rate_based import RateBasedAbsorberParams, RateBasedAbsorber

absorber = RateBasedAbsorber(RateBasedAbsorberParams(
    solvent='MEA',
    packing='Mellapak_250Y',
    height=15.0,  # m
    diameter=2.0,  # m
    L_G_ratio=3.5,
))
gas_out, liquid_out, info = absorber(gas_in)

# Same downstream processing works for both
```

### Profile Access

Rate-based models provide spatial profiles:

```python
info = {
    # Standard outputs (same as equilibrium)
    'capture_efficiency': ...,
    'rich_loading': ...,

    # Rate-based specific
    'z': jnp.array([...]),           # Height coordinates
    'T_gas_profile': jnp.array([...]),
    'T_liquid_profile': jnp.array([...]),
    'y_CO2_profile': jnp.array([...]),
    'loading_profile': jnp.array([...]),
    'flux_profile': jnp.array([...]),
}
```

## References

1. **Mass Transfer Correlations**
   - Onda K et al. (1968). J Chem Eng Japan 1:56-62.
   - Billet R, Schultes M (1999). Chem Eng Res Des 77:498-504.
   - Rocha JA et al. (1996). Ind Eng Chem Res 35:1660-1667.

2. **Rate-Based Modeling**
   - Kenig EY et al. (2001). Chem Eng Sci 56:343-350.
   - Kucka L et al. (2003). Chem Eng Sci 58:3571-3578.
   - Zhang Y et al. (2009). Ind Eng Chem Res 48:9233-9246.

3. **Enhancement Factors**
   - van Swaaij WPM, Versteeg GF (1992). Chem Eng Sci 47:3181-3195.
   - DeCoursey WJ (1974). Chem Eng Sci 29:1867-1872.

4. **Implementation in Process Simulators**
   - Aspen Rate-Based Distillation documentation
   - gPROMS ModelBuilder examples

## Success Criteria

1. **Accuracy**: Match pilot plant data within 10% for key outputs
2. **Differentiability**: Smooth gradients for optimization
3. **Performance**: Full column solve < 1 second (after JIT)
4. **Usability**: API consistent with equilibrium models
5. **Validation**: Comparison with commercial simulators (Aspen, ProTreat)
