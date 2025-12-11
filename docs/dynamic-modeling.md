# Dynamic Modeling

The `difflow.dynamic` module provides a unified framework for transient (time-dependent) simulation of chemical process units. This enables:

- **Startup/shutdown analysis**: Simulate process dynamics from empty to steady state
- **Disturbance response**: Track how processes respond to feed changes
- **Control system design**: Test controllers in simulation before implementation
- **Batch process modeling**: Simulate time-varying batch operations
- **Parameter estimation**: Fit kinetic parameters to dynamic experimental data

All dynamic simulations are fully differentiable via JAX, enabling gradient-based optimization of dynamic systems.

## Table of Contents

1. [Quick Start](#quick-start)
2. [ODE Integration](#ode-integration)
3. [Dynamic Units](#dynamic-units)
4. [State Specification](#state-specification)
5. [Dynamic Flowsheets](#dynamic-flowsheets)
6. [DAE Systems](#dae-systems)
7. [Diffrax Backend](#diffrax-backend)
8. [Gradient Computation](#gradient-computation)
9. [API Reference](#api-reference)

---

## Quick Start

```python
import jax.numpy as jnp
from difflow.dynamic import integrate, DynamicCSTR, integrate_unit
from difflow.streams import make_stream

# Simple ODE: harmonic oscillator
def oscillator(t, y):
    return jnp.array([y[1], -y[0]])

result = integrate(oscillator, jnp.array([1.0, 0.0]), (0.0, 10.0))
print(f"Final position: {result.y_final[0]:.4f}")

# Dynamic reactor simulation
def rate_fn(C, T, params):
    return jnp.array([params["k"] * C["A"]])

cstr = DynamicCSTR(
    volume=1.0,
    rate_fn=rate_fn,
    stoich=jnp.array([[-1], [1]]),
    species_order=["A", "B"],
    rate_params={"k": 0.1},
)

inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
result = integrate_unit(cstr, {"inlet": inlet}, (0.0, 100.0))
```

---

## ODE Integration

### The `integrate()` Function

The unified interface for ODE integration:

```python
from difflow.dynamic import integrate

def f(t, y):
    """Derivative function: dy/dt = f(t, y)"""
    return -0.5 * y  # Exponential decay

result = integrate(
    f,                          # Derivative function
    y0=jnp.array([1.0]),       # Initial state
    t_span=(0.0, 10.0),        # Time interval
    method="RK4",              # Integration method
    n_steps=100,               # Number of steps (fixed-step methods)
)

# Access results
print(f"Final state: {result.y_final}")
print(f"Success: {result.info.success}")
print(f"Steps taken: {result.info.n_steps}")

# Trajectory (intermediate values)
print(f"Time points: {result.trajectory.t.shape}")
print(f"State history: {result.trajectory.y.shape}")
```

### Available Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `"RK4"` | 4th order Runge-Kutta | General purpose, fixed step |
| `"RK45"` | Adaptive RK45 (Dormand-Prince) | When accuracy matters |
| `"Euler"` | Forward Euler | Simple problems, debugging |
| `"diffrax"` | Default diffrax solver (Tsit5) | Advanced features |
| `"diffrax:dopri5"` | Dormand-Prince 5(4) | Good general purpose |
| `"diffrax:kvaerno5"` | Implicit 5th order | Stiff systems |

### Integration Options

```python
# Fixed-step methods (RK4, Euler)
result = integrate(f, y0, t_span, method="RK4", n_steps=1000)

# Adaptive methods (RK45)
result = integrate(f, y0, t_span, method="RK45", rtol=1e-6, atol=1e-8)

# Diffrax methods
result = integrate(
    f, y0, t_span,
    method="diffrax:tsit5",
    rtol=1e-5,
    atol=1e-7,
    max_steps=10000,
)
```

---

## Dynamic Units

### The DynamicUnit Protocol

All dynamic units implement the `DynamicUnit` protocol:

```python
from typing import Protocol

class DynamicUnit(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def state_spec(self) -> StateSpec: ...

    def initial_state(self, inputs: dict) -> Array: ...

    def derivatives(self, t: Array, y: Array, inputs: dict) -> Array: ...

    def outputs(self, t: Array, y: Array, inputs: dict) -> dict[str, Stream]: ...
```

### DynamicCSTR

Continuous stirred-tank reactor with reaction kinetics:

```python
from difflow.dynamic import DynamicCSTR
import jax.numpy as jnp

# Define rate law: r = k * exp(-Ea/RT) * C_A
def rate_fn(C, T, params):
    k = params["k0"] * jnp.exp(-params["Ea"] / (8.314 * T))
    return jnp.array([k * C["A"]])  # Rate of reaction 1

# A -> B (single reaction)
stoich = jnp.array([
    [-1],  # A consumed
    [+1],  # B produced
])

cstr = DynamicCSTR(
    volume=1.0,                    # m³
    rate_fn=rate_fn,
    stoich=stoich,
    species_order=["A", "B"],
    rate_params={"k0": 1e6, "Ea": 50000.0},
    name="reactor",
)

# State variables: [n_A, n_B] (moles of each species)
```

### DynamicTank

Storage tank with variable holdup:

```python
from difflow.dynamic import DynamicTank

tank = DynamicTank(
    max_volume=10.0,              # Maximum volume (m³)
    species_order=["A", "B"],
    initial_volume=1.0,           # Starting volume
    name="storage",
)

# State variables: [V, n_A, n_B] (volume + moles)
```

### Using integrate_unit()

Convenience wrapper for simulating dynamic units:

```python
from difflow.dynamic import integrate_unit
from difflow.streams import make_stream

inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

result = integrate_unit(
    cstr,
    inputs={"inlet": inlet},
    t_span=(0.0, 1000.0),
    method="RK4",
    n_steps=500,
)

# Final moles
n_A_final, n_B_final = result.y_final
print(f"Final A: {n_A_final:.4f} mol")
print(f"Final B: {n_B_final:.4f} mol")
```

---

## State Specification

### StateVar and StateSpec

Define state variables with metadata:

```python
from difflow.dynamic import StateVar, StateSpec, StateCategory

# Manual specification
spec = StateSpec([
    StateVar("n_A", StateCategory.MOLES, "mol", bounds=(0, None)),
    StateVar("n_B", StateCategory.MOLES, "mol", bounds=(0, None)),
    StateVar("T", StateCategory.TEMPERATURE, "K", bounds=(200, 600)),
])

print(f"State dimension: {spec.n_states}")
print(f"State names: {spec.names}")
print(f"Index of T: {spec.index('T')}")
```

### Factory Functions

Convenience functions for common state patterns:

```python
from difflow.dynamic import (
    molar_states,
    concentration_states,
    thermal_state,
    volume_state,
    reactor_states,
)

# Molar holdup for species
spec = molar_states(["A", "B", "C"])  # n_A, n_B, n_C

# Concentration states
spec = concentration_states(["A", "B"])  # C_A, C_B

# Combine specs
spec = molar_states(["A", "B"]) + thermal_state()  # n_A, n_B, T

# Complete reactor state
spec = reactor_states(["A", "B"])  # n_A, n_B, T (moles + temperature)
```

### StateVector

Runtime access to state values by name:

```python
from difflow.dynamic import StateVector

spec = molar_states(["A", "B"]) + thermal_state()
y = jnp.array([1.0, 0.5, 350.0])

state = StateVector(spec, y)
print(f"n_A = {state['n_A']}")
print(f"T = {state['T']}")

# Or use spec directly
idx_A = spec.index("n_A")
n_A = y[idx_A]
```

---

## Dynamic Flowsheets

### Building a Flowsheet

Connect multiple dynamic units:

```python
from difflow.dynamic import DynamicFlowsheet, DynamicCSTR, DynamicTank
from difflow.streams import make_stream

# Create units
cstr = DynamicCSTR(
    volume=1.0,
    rate_fn=rate_fn,
    stoich=stoich,
    species_order=["A", "B"],
    rate_params={"k": 0.1},
    name="reactor",
)

tank = DynamicTank(
    max_volume=10.0,
    species_order=["A", "B"],
    name="storage",
)

# Build flowsheet
fs = DynamicFlowsheet(species_order=["A", "B"])

# Add feed stream
feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
fs.add_feed("feed", feed)

# Add units with connections
fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
fs.add_unit(tank, inlet_names=["reactor_out"], outlet_names=["product"])

# Simulate
result = fs.simulate(t_span=(0.0, 1000.0), method="RK4", n_steps=500)
```

### Accessing Results

```python
# Combined final state
print(f"Final state: {result.y_final}")

# Per-unit states
reactor_state = result.get_unit_state("reactor")
tank_state = result.get_unit_state("storage")

# Trajectory
print(f"Time points: {result.trajectory.t.shape}")

# Output streams at final time
streams = fs.outputs(result.trajectory.t[-1], result.y_final)
print(f"Product stream: {streams['product']}")
```

### Time-Varying Feeds

```python
# Define feed as function of time
def feed_schedule(t):
    """Feed rate doubles after t=500."""
    base = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
    if t > 500:
        return {k: v * 2 for k, v in base.items()}
    return base

fs.add_feed("feed", feed_schedule)  # Pass function instead of stream
```

### Manual Derivative Access

For custom integration or analysis:

```python
# Get combined initial state
y0 = fs.initial_state()

# Define derivative function
def flowsheet_f(t, y):
    return fs.derivatives(t, y)

# Use with any integrator
result = integrate(flowsheet_f, y0, (0.0, 1000.0), method="RK45")
```

---

## DAE Systems

Differential-Algebraic Equations combine ODEs with algebraic constraints:

```
dx/dt = f(t, x, z)    # Differential equations
0 = g(t, x, z)        # Algebraic constraints
```

### DAE Units

```python
from difflow.dynamic import DAEUnitBase, AlgebraicSpec, AlgebraicVar

class MyFlashDrum(DAEUnitBase):
    @property
    def algebraic_spec(self) -> AlgebraicSpec:
        return AlgebraicSpec([
            AlgebraicVar("V_frac", "vapor_fraction", "-", bounds=(0, 1)),
        ])

    def algebraic_residual(self, t, x, z, inputs):
        """Return g(t,x,z) - should equal zero at solution."""
        V_frac = z[0]
        # VLE constraint: sum(z_i * (K_i - 1) / (1 + V*(K_i-1))) = 0
        residual = self._rachford_rice(x, V_frac)
        return jnp.array([residual])

    def differential(self, t, x, z, inputs):
        """Return dx/dt given algebraic variables."""
        # Material balances using vapor fraction
        ...
```

### Built-in: DynamicFlashDrum

```python
from difflow.dynamic import DynamicFlashDrum, integrate_dae

flash = DynamicFlashDrum(
    volume=1.0,
    species_order=["A", "B"],
    K_values={"A": 2.0, "B": 0.5},  # Vapor-liquid K-values
    name="flash",
)

inlet = make_stream({"A": 0.5, "B": 0.5}, T=350.0, P=101325.0)

result = integrate_dae(
    flash,
    inputs={"inlet": inlet},
    t_span=(0.0, 100.0),
    method="RK4",
    n_steps=200,
)

print(f"Final moles: {result.x_final}")
print(f"Final vapor fraction: {result.z_final}")
```

### Newton Solver

The algebraic constraints are solved at each time step:

```python
from difflow.dynamic import newton_solve

def residual(z):
    """System of nonlinear equations."""
    x, y = z[0], z[1]
    return jnp.array([
        x**2 + y**2 - 5.0,  # Circle
        x * y - 2.0,         # Hyperbola
    ])

z0 = jnp.array([2.0, 1.0])
z_solution, info = newton_solve(residual, z0, tol=1e-8, max_iter=50)

print(f"Solution: {z_solution}")
print(f"Converged: {info['converged']}")
print(f"Iterations: {info['iterations']}")
```

### DAE Integration Methods

```python
# Euler method (simpler, may need smaller steps)
result = integrate_dae(unit, inputs, t_span, method="Euler", n_steps=1000)

# RK4 method (more accurate)
result = integrate_dae(unit, inputs, t_span, method="RK4", n_steps=200)
```

---

## Diffrax Backend

[Diffrax](https://github.com/patrick-kidger/diffrax) provides advanced ODE/SDE solvers with adaptive step control.

### Installation

```bash
pip install diffrax
```

### Basic Usage

```python
from difflow.dynamic import integrate

# Use diffrax with default solver (Tsit5)
result = integrate(f, y0, t_span, method="diffrax")

# Specify solver
result = integrate(f, y0, t_span, method="diffrax:dopri5")
```

### Available Solvers

**Explicit (non-stiff problems):**
- `dopri5`: Dormand-Prince 5(4) - good general purpose
- `dopri8`: Dormand-Prince 8(7) - higher accuracy
- `tsit5`: Tsitouras 5(4) - efficient, **recommended default**
- `bosh3`: Bogacki-Shampine 3(2)
- `heun`: Heun's method (2nd order, fixed step)
- `euler`: Forward Euler (1st order, fixed step)

**Implicit (stiff problems):**
- `kvaerno3`: 3rd order implicit
- `kvaerno4`: 4th order implicit
- `kvaerno5`: 5th order implicit - **recommended for stiff**
- `implicit_euler`: Backward Euler

### Stiff Systems

Chemical kinetics often involve very different time scales (stiff):

```python
from difflow.dynamic import integrate, integrate_stiff

# Robertson problem - classic stiff test
def robertson(t, y):
    k1, k2, k3 = 0.04, 3e7, 1e4
    A, B, C = y[0], y[1], y[2]
    return jnp.array([
        -k1*A + k3*B*C,
        k1*A - k2*B*B - k3*B*C,
        k2*B*B,
    ])

y0 = jnp.array([1.0, 0.0, 0.0])

# Using implicit solver
result = integrate(
    robertson, y0, (0.0, 1e5),
    method="diffrax:kvaerno5",
    rtol=1e-4, atol=1e-6,
)

# Or use convenience function
result = integrate_stiff(robertson, y0, (0.0, 1e5))
```

### Tolerance Control

```python
result = integrate(
    f, y0, t_span,
    method="diffrax:tsit5",
    rtol=1e-6,      # Relative tolerance
    atol=1e-8,      # Absolute tolerance
    max_steps=10000, # Maximum integration steps
)
```

### Direct Diffrax API

For more control:

```python
from difflow.dynamic import integrate_diffrax

result = integrate_diffrax(
    f, y0, t_span,
    solver="tsit5",
    rtol=1e-5,
    atol=1e-7,
    dt0=0.01,              # Initial step size
    saveat=jnp.linspace(0, 100, 101),  # Save at specific times
)
```

---

## Gradient Computation

All dynamic simulations are differentiable via JAX.

### Gradients Through Integration

```python
import jax

def loss(y0):
    """Loss based on final state."""
    result = integrate(f, y0, (0.0, 10.0), method="RK4")
    return jnp.sum(result.y_final**2)

# Gradient w.r.t. initial condition
y0 = jnp.array([1.0, 0.0])
grad_y0 = jax.grad(loss)(y0)
```

### Parameter Optimization

```python
def simulate_with_params(k):
    """Simulate reactor with rate constant k."""
    def rate_fn(C, T, params):
        return jnp.array([k * C["A"]])

    cstr = DynamicCSTR(
        volume=1.0, rate_fn=rate_fn, stoich=stoich,
        species_order=["A", "B"], rate_params={},
    )

    result = integrate_unit(cstr, {"inlet": inlet}, (0.0, 100.0))
    return result.y_final[1]  # Final product amount

# Optimize for maximum product
grad_k = jax.grad(simulate_with_params)(jnp.array(0.1))
```

### Sensitivity Analysis

```python
from difflow.dynamic import sensitivity_analysis

def model(params):
    k, Ea = params["k"], params["Ea"]
    # ... simulation ...
    return final_conversion

nominal = {"k": 1e6, "Ea": 50000.0}
sens = sensitivity_analysis(model, nominal)
# Returns gradients and normalized sensitivities
```

### Using integrate_with_grad

Explicit gradient computation:

```python
from difflow.dynamic import integrate_with_grad

# Returns both result and gradient function
result, grad_fn = integrate_with_grad(f, y0, t_span)

# Compute gradient of final state w.r.t. y0
dy_final_dy0 = grad_fn(jnp.ones_like(result.y_final))
```

---

## API Reference

### Integration Functions

| Function | Description |
|----------|-------------|
| `integrate(f, y0, t_span, method, **kwargs)` | Unified ODE integration |
| `integrate_unit(unit, inputs, t_span, **kwargs)` | Integrate DynamicUnit |
| `integrate_dae(unit, inputs, t_span, **kwargs)` | Integrate DAE unit |
| `integrate_rk4(f, y0, t_span, n_steps)` | Fixed-step RK4 |
| `integrate_rk45(f, y0, t_span, rtol, atol)` | Adaptive RK45 |
| `integrate_euler(f, y0, t_span, n_steps)` | Forward Euler |

### Diffrax Functions

| Function | Description |
|----------|-------------|
| `integrate_diffrax(f, y0, t_span, solver, **kwargs)` | Direct diffrax integration |
| `integrate_stiff(f, y0, t_span, **kwargs)` | Stiff system integration |
| `integrate_dopri5(f, y0, t_span, **kwargs)` | Dormand-Prince 5(4) |
| `integrate_tsit5(f, y0, t_span, **kwargs)` | Tsitouras 5(4) |
| `list_diffrax_solvers()` | List available solvers |
| `check_diffrax_available()` | Check if diffrax installed |

### Classes

| Class | Description |
|-------|-------------|
| `DynamicUnit` | Protocol for dynamic units |
| `DynamicUnitBase` | Base class with utilities |
| `DynamicCSTR` | Dynamic CSTR reactor |
| `DynamicTank` | Storage tank with holdup |
| `DynamicFlowsheet` | Multi-unit flowsheet |
| `DAEUnit` | Protocol for DAE units |
| `DAEUnitBase` | Base class for DAE units |
| `DynamicFlashDrum` | Flash drum with VLE |

### State Classes

| Class | Description |
|-------|-------------|
| `StateVar` | Single state variable spec |
| `StateSpec` | Collection of states |
| `StateVector` | Runtime state access |
| `AlgebraicVar` | Algebraic variable spec |
| `AlgebraicSpec` | Collection of algebraic vars |

### Result Classes

| Class | Description |
|-------|-------------|
| `IntegrationResult` | ODE integration result |
| `Trajectory` | Time series of states |
| `IntegrationInfo` | Solver statistics |
| `DAEResult` | DAE integration result |
| `DynamicFlowsheetResult` | Flowsheet simulation result |
