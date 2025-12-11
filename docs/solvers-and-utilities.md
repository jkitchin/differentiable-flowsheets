# Solvers and Utilities

This document covers numerical solvers, uncertainty propagation, and utility functions in Difflow.

## Table of Contents

1. [Numerical Solvers](#numerical-solvers)
   - [Fixed-Point Iteration](#fixed-point-iteration)
   - [Newton-Raphson Solver](#newton-raphson-solver)
   - [Rachford-Rice Solver](#rachford-rice-solver)
   - [ODE Integration](#ode-integration)
2. [Uncertainty Propagation](#uncertainty-propagation)
   - [Linear Propagation](#linear-propagation)
   - [Monte Carlo Propagation](#monte-carlo-propagation)
   - [Sensitivity Analysis](#sensitivity-analysis)
   - [Sobol Indices](#sobol-indices)
3. [Implicit Differentiation](#implicit-differentiation)
4. [Utility Functions](#utility-functions)

---

## Numerical Solvers

**Location**: `difflow/solvers.py`

All solvers in Difflow are:
- Implemented using JAX primitives for automatic differentiation
- Support custom VJP rules for efficient backward passes
- Fully JIT-compilable for performance

### Fixed-Point Iteration

Solves equations of the form $x^* = f(x^*, \text{args})$.

```python
from difflow.solvers import fixed_point_solve

def solve_result(result):
    """Fixed-point iteration with damping."""
    x_final = fixed_point_solve(
        f=update_function,      # x_new = f(x_old, args)
        x0=initial_guess,       # Starting point
        args=additional_args,   # Passed to f
        tol=1e-8,               # Convergence tolerance
        max_iter=100,           # Maximum iterations
        damping=0.5             # Damping factor (0-1)
    )
    return x_final
```

**Algorithm**:

$$x^{(k+1)} = (1 - \alpha) x^{(k)} + \alpha \cdot f(x^{(k)}, \text{args})$$

Where $\alpha$ is the damping factor.

**Convergence Criterion**:

$$\|x^{(k+1)} - x^{(k)}\| < \epsilon$$

**Example**: Solving recycle loop

```python
def recycle_update(recycle_flow, args):
    feed, reactor_params = args

    # Mix fresh feed with recycle
    mixed = combine_streams(feed, make_stream({'A': recycle_flow}, T=350.0, P=101325.0))

    # Run reactor
    outlet, _ = reactor(mixed)

    # Flash and get recycle composition
    liquid, _, info = flash(outlet)

    # Return new recycle flow (converges when input = output)
    return liquid['F_A'] * 0.2  # 20% recycle

# Solve for steady-state recycle flow
recycle_ss = fixed_point_solve(
    f=recycle_update,
    x0=0.1,  # Initial guess
    args=(fresh_feed, reactor_params),
    tol=1e-6,
    damping=0.7
)
```

### Newton-Raphson Solver

Solves equations of the form $g(x^*, \text{args}) = 0$.

```python
from difflow.solvers import newton_solve

def solve_equation(x_solution):
    """Newton-Raphson solver."""
    x_solution = newton_solve(
        f=residual_function,    # g(x, args) = 0
        x0=initial_guess,
        args=additional_args,
        tol=1e-10,
        max_iter=50
    )
    return x_solution
```

**Algorithm**:

$$x^{(k+1)} = x^{(k)} - J^{-1} \cdot g(x^{(k)})$$

Where $J = \frac{\partial g}{\partial x}$ is the Jacobian, computed automatically using JAX.

**Features**:
- Automatic Jacobian computation via `jax.jacobian`
- Custom VJP for efficient backward differentiation
- Line search for improved robustness (optional)

**Example**: Bubble point temperature

```python
def bubble_residual(T, args):
    x, P, K_func = args
    K = K_func(T)
    # Bubble point: sum(x_i * K_i) = 1
    return jnp.sum(x * K) - 1.0

T_bubble = newton_solve(
    f=bubble_residual,
    x0=350.0,
    args=(liquid_composition, pressure, K_values_func),
    tol=1e-8
)
```

### Rachford-Rice Solver

Specialized solver for flash calculations.

```python
from difflow.solvers import rachford_rice, rachford_rice_compositions

# Solve for vapor fraction
V_frac = rachford_rice(
    z=feed_composition,     # Feed mole fractions (array)
    K=K_values,             # Equilibrium ratios (array)
    tol=1e-10,
    max_iter=50
)

# Get phase compositions from vapor fraction
x, y = rachford_rice_compositions(z, K, V_frac)
```

**Rachford-Rice Equation**:

$$f(V) = \sum_i \frac{z_i(K_i - 1)}{1 + V(K_i - 1)} = 0$$

**Algorithm**:
1. Bound vapor fraction: $V \in [V_{min}, V_{max}]$
   - $V_{min} = \max_i \frac{K_i z_i - 1}{K_i - 1}$ for $K_i > 1$
   - $V_{max} = \min_i \frac{1 - z_i}{1 - K_i}$ for $K_i < 1$
2. Newton iteration with bounds enforcement
3. Damping for stability near boundaries

**Phase Compositions**:

$$x_i = \frac{z_i}{1 + V(K_i - 1)}$$

$$y_i = K_i x_i = \frac{K_i z_i}{1 + V(K_i - 1)}$$

### ODE Integration

Used internally by PFR, fed-batch reactor, and other dynamic models.

```python
from jax import lax

def rk4_step(state, dt, derivative_fn, args):
    """Single RK4 step."""
    k1 = derivative_fn(state, args)
    k2 = derivative_fn(state + 0.5 * dt * k1, args)
    k3 = derivative_fn(state + 0.5 * dt * k2, args)
    k4 = derivative_fn(state + dt * k3, args)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_ode(y0, t_span, n_steps, derivative_fn, args):
    """Integrate ODE using RK4 with lax.scan."""
    dt = (t_span[1] - t_span[0]) / n_steps

    def scan_fn(carry, _):
        state, t = carry
        new_state = rk4_step(state, dt, derivative_fn, args)
        return (new_state, t + dt), new_state

    _, trajectory = lax.scan(scan_fn, (y0, t_span[0]), None, length=n_steps)
    return trajectory
```

**Why `lax.scan`?**
- Efficient memory usage (doesn't store intermediate Jacobians)
- Fully differentiable through the integration
- JIT-compilable

---

## Uncertainty Propagation

**Location**: `difflow/uncertainty.py`

### Linear Propagation

First-order Taylor expansion for uncertainty propagation.

```python
from difflow.uncertainty import linear_propagation

# Define model and uncertainties
def model(params):
    reactor_T, feed_flow = params
    inlet = make_stream({'A': feed_flow}, T=reactor_T, P=101325.0)
    outlet, info = reactor(inlet)
    return info['conversion']

nominal = jnp.array([350.0, 1.0])  # [T, F]
uncertainties = jnp.array([5.0, 0.05])  # Standard deviations

# Propagate uncertainty
mean, std = linear_propagation(model, nominal, uncertainties)
print(f"Conversion: {mean:.4f} +/- {std:.4f}")
```

**Theory**:

For $y = f(\mathbf{x})$ with $\mathbf{x} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$E[y] \approx f(\boldsymbol{\mu})$$

$$\text{Var}(y) \approx \mathbf{J} \boldsymbol{\Sigma} \mathbf{J}^T$$

Where $\mathbf{J} = \nabla f|_{\boldsymbol{\mu}}$ is the Jacobian.

**For uncorrelated inputs** ($\boldsymbol{\Sigma}$ is diagonal):

$$\sigma_y^2 \approx \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_{x_i}^2$$

**Advantages**:
- Fast (single gradient evaluation)
- Accurate for small uncertainties and linear systems

**Limitations**:
- First-order approximation
- May underestimate uncertainty for nonlinear systems

### Monte Carlo Propagation

Sampling-based uncertainty propagation.

```python
from difflow.uncertainty import monte_carlo_propagation

# Monte Carlo analysis
results = monte_carlo_propagation(
    model=model,
    nominal=nominal,
    uncertainties=uncertainties,
    n_samples=10000,
    distribution='normal'  # or 'uniform'
)

print(f"Mean: {results['mean']:.4f}")
print(f"Std: {results['std']:.4f}")
print(f"5th percentile: {results['p5']:.4f}")
print(f"95th percentile: {results['p95']:.4f}")
```

**Algorithm**:
1. Generate $N$ samples from input distribution
2. Evaluate model for each sample (vectorized with `vmap`)
3. Compute output statistics

**Implementation** (vectorized for efficiency):

```python
import jax.numpy as jnp
from jax import vmap, random

def monte_carlo_propagation(model, nominal, uncertainties, n_samples, key=None):
    if key is None:
        key = random.PRNGKey(0)

    # Generate samples
    samples = nominal + uncertainties * random.normal(key, shape=(n_samples, len(nominal)))

    # Vectorized model evaluation
    outputs = vmap(model)(samples)

    return {
        'mean': jnp.mean(outputs),
        'std': jnp.std(outputs),
        'p5': jnp.percentile(outputs, 5),
        'p95': jnp.percentile(outputs, 95),
        'samples': outputs
    }
```

**Advantages**:
- Accurate for nonlinear systems
- Provides full distribution, not just mean/variance
- Handles non-Gaussian distributions

### Sensitivity Analysis

Gradient-based sensitivity analysis.

```python
from difflow.uncertainty import sensitivity_analysis

# Compute sensitivities
sensitivities = sensitivity_analysis(
    model=model,
    nominal=nominal,
    uncertainties=uncertainties,
    param_names=['T_reactor', 'F_feed']
)

for name, sens in sensitivities.items():
    print(f"{name}: sensitivity = {sens['gradient']:.4f}, contribution = {sens['contribution']:.1%}")
```

**Sensitivity Metrics**:

1. **Gradient** (absolute sensitivity):
   $$S_i = \frac{\partial y}{\partial x_i}$$

2. **Normalized sensitivity** (dimensionless):
   $$S_i^* = \frac{\partial y}{\partial x_i} \cdot \frac{x_i}{y}$$

3. **Variance contribution**:
   $$C_i = \frac{S_i^2 \sigma_{x_i}^2}{\sum_j S_j^2 \sigma_{x_j}^2}$$

### Sobol Indices

Global sensitivity analysis using Sobol variance decomposition.

```python
from difflow.uncertainty import sobol_indices

# Compute Sobol indices
indices = sobol_indices(
    model=model,
    nominal=nominal,
    uncertainties=uncertainties,
    n_samples=10000,
    param_names=['T_reactor', 'F_feed']
)

for name, idx in indices.items():
    print(f"{name}: S1 = {idx['first_order']:.3f}, ST = {idx['total']:.3f}")
```

**Theory**:

Total variance decomposition:

$$\text{Var}(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + \ldots + V_{12\ldots n}$$

**First-order Sobol index** (main effect):

$$S_i = \frac{V_i}{\text{Var}(Y)} = \frac{\text{Var}_{X_i}[E_{X_{\sim i}}(Y|X_i)]}{\text{Var}(Y)}$$

**Total-order Sobol index** (main + interactions):

$$S_{Ti} = \frac{E_{X_{\sim i}}[\text{Var}_{X_i}(Y|X_{\sim i})]}{\text{Var}(Y)}$$

**Interpretation**:
- $S_i \approx S_{Ti}$: Parameter has mostly main effects
- $S_{Ti} \gg S_i$: Parameter has significant interactions
- $\sum_i S_{Ti} \approx 1$: Weak interactions
- $\sum_i S_{Ti} \gg 1$: Strong interactions

### Covariance Propagation

General covariance matrix propagation.

```python
from difflow.uncertainty import propagate_covariance

# Full covariance matrix (correlated inputs)
cov_input = jnp.array([
    [25.0, 2.0],   # Var(T) = 25, Cov(T,F) = 2
    [2.0, 0.01]    # Cov(F,T) = 2, Var(F) = 0.01
])

# Jacobian at nominal point
jacobian = jax.jacobian(model)(nominal)

# Propagate covariance
cov_output = propagate_covariance(jacobian, cov_input)
```

**Equation**:

$$\boldsymbol{\Sigma}_Y = \mathbf{J} \boldsymbol{\Sigma}_X \mathbf{J}^T$$

---

## Implicit Differentiation

Difflow uses implicit differentiation to compute gradients through iterative solvers.

### Theory

For a solution $x^* = f(x^*, \theta)$ (fixed-point) or $g(x^*, \theta) = 0$ (root-finding), the gradient w.r.t. parameters $\theta$ is:

**Fixed-point**:
$$\frac{dx^*}{d\theta} = \left(I - \frac{\partial f}{\partial x}\bigg|_{x^*}\right)^{-1} \frac{\partial f}{\partial \theta}\bigg|_{x^*}$$

**Root-finding**:
$$\frac{dx^*}{d\theta} = -\left(\frac{\partial g}{\partial x}\bigg|_{x^*}\right)^{-1} \frac{\partial g}{\partial \theta}\bigg|_{x^*}$$

### Implementation with Custom VJP

```python
from jax import custom_vjp

@custom_vjp
def fixed_point_solve(f, x0, args, tol, max_iter):
    # Forward pass: just run the iteration
    x = x0
    for _ in range(max_iter):
        x_new = f(x, args)
        if jnp.max(jnp.abs(x_new - x)) < tol:
            break
        x = x_new
    return x

def fixed_point_solve_fwd(f, x0, args, tol, max_iter):
    x_star = fixed_point_solve(f, x0, args, tol, max_iter)
    return x_star, (x_star, args)

def fixed_point_solve_bwd(res, g):
    x_star, args = res
    # Solve: (I - df/dx)^T v = g for v
    # Then: gradient w.r.t. args = (df/d_args)^T v
    ...
    return (None, None, args_grad, None, None)

fixed_point_solve.defvjp(fixed_point_solve_fwd, fixed_point_solve_bwd)
```

### Benefits

- **Memory efficient**: Only stores solution, not iteration history
- **Accurate gradients**: Uses implicit function theorem, not unrolling
- **Fast backward pass**: Single linear solve instead of backprop through iterations

---

## Utility Functions

### Numerical Helpers

```python
from difflow.utils import (
    safe_divide,
    safe_log,
    safe_sqrt,
    clip_positive,
    smooth_max,
    smooth_min
)

# Safe operations (avoid NaN/Inf)
x = safe_divide(a, b, default=0.0)  # Returns default if b ≈ 0
y = safe_log(x, min_val=1e-10)      # Clips x to avoid log(0)
z = safe_sqrt(x)                     # Clips x to avoid sqrt(negative)

# Smooth approximations (differentiable)
max_val = smooth_max(a, b, alpha=10.0)  # Softmax approximation
min_val = smooth_min(a, b, alpha=10.0)  # Softmin approximation
```

### Smooth Approximations

For optimization, smooth approximations of non-differentiable functions:

**Smooth maximum**:
$$\text{softmax}(a, b) = \frac{a e^{\alpha a} + b e^{\alpha b}}{e^{\alpha a} + e^{\alpha b}}$$

As $\alpha \to \infty$, approaches $\max(a, b)$.

**Smooth absolute value**:
$$|x|_\epsilon \approx \sqrt{x^2 + \epsilon^2}$$

**Smooth ReLU**:
$$\text{softplus}(x) = \frac{1}{\beta} \log(1 + e^{\beta x})$$

### Unit Conversions

```python
from difflow.utils import (
    celsius_to_kelvin,
    kelvin_to_celsius,
    bar_to_pascal,
    pascal_to_bar,
    psi_to_pascal,
    pascal_to_psi,
    kg_to_mol,
    mol_to_kg
)

# Temperature
T_K = celsius_to_kelvin(25.0)  # 298.15 K
T_C = kelvin_to_celsius(350.0)  # 76.85 °C

# Pressure
P_Pa = bar_to_pascal(10.0)      # 1,000,000 Pa
P_bar = pascal_to_bar(101325.0) # 1.01325 bar

# Mass/molar
n = kg_to_mol(1.0, MW=32.04)    # 31.21 mol (for methanol)
m = mol_to_kg(100.0, MW=32.04)  # 3.204 kg
```

### Thermodynamic Helpers

```python
from difflow.utils import (
    ideal_gas_density,
    ideal_gas_volume,
    reynolds_number,
    prandtl_number,
    nusselt_correlation
)

# Ideal gas calculations
rho = ideal_gas_density(T=300.0, P=101325.0, MW=28.97)  # kg/m³
V = ideal_gas_volume(n=1.0, T=300.0, P=101325.0)        # m³

# Dimensionless numbers
Re = reynolds_number(rho=1000, v=1.0, D=0.1, mu=0.001)
Pr = prandtl_number(Cp=4180, mu=0.001, k=0.6)
Nu = nusselt_correlation(Re=10000, Pr=7, correlation='dittus_boelter')
```

---

## Best Practices

### Solver Selection

| Problem Type | Recommended Solver |
|-------------|-------------------|
| Fixed-point (well-behaved) | `fixed_point_solve` with damping |
| Fixed-point (difficult) | Wegstein or Broyden acceleration |
| Root-finding | `newton_solve` |
| Flash calculation | `rachford_rice` |
| ODE integration | RK4 with `lax.scan` |

### Convergence Tips

1. **Good initial guess**: Use physical intuition or simpler model
2. **Appropriate tolerance**: 1e-6 to 1e-10 depending on application
3. **Damping**: Start with 0.3-0.5 for difficult problems
4. **Bounds**: Enforce physical constraints (positive concentrations, etc.)

### Uncertainty Analysis Workflow

```python
# 1. Define model
def process_model(params):
    T, P, F = params
    # ... process simulation ...
    return outputs

# 2. Identify uncertain parameters
nominal = jnp.array([350.0, 101325.0, 1.0])
uncertainties = jnp.array([10.0, 5000.0, 0.1])

# 3. Quick screening with linear propagation
mean, std = linear_propagation(process_model, nominal, uncertainties)

# 4. Identify important parameters with sensitivity analysis
sens = sensitivity_analysis(process_model, nominal, uncertainties)

# 5. Detailed analysis on key parameters with Monte Carlo
results = monte_carlo_propagation(process_model, nominal, uncertainties, n_samples=10000)

# 6. Global sensitivity with Sobol indices (if needed)
sobol = sobol_indices(process_model, nominal, uncertainties, n_samples=50000)
```

### Debugging Numerical Issues

```python
# Check for NaN/Inf
import jax.numpy as jnp

def check_numerics(x, name="value"):
    if jnp.any(jnp.isnan(x)):
        print(f"NaN detected in {name}")
    if jnp.any(jnp.isinf(x)):
        print(f"Inf detected in {name}")
    return x

# Monitor convergence
def solve_with_monitoring(f, x0, args, tol, max_iter):
    x = x0
    for i in range(max_iter):
        x_new = f(x, args)
        error = jnp.max(jnp.abs(x_new - x))
        print(f"Iter {i}: error = {error:.2e}")
        if error < tol:
            print(f"Converged in {i+1} iterations")
            return x_new
        x = x_new
    print("Warning: Did not converge")
    return x
```
