# Solvers and Utilities

This document covers numerical solvers, uncertainty propagation, and utility functions in Difflow.

## Table of Contents

1. [Numerical Solvers](#numerical-solvers)
   - [Fixed-Point Iteration](#fixed-point-iteration)
   - [Newton-Raphson Solver](#newton-raphson-solver)
   - [Rachford-Rice Solver](#rachford-rice-solver)
   - [ODE Integration](#ode-integration)
   - [Equation-Oriented Solver](eo-solver.md) — simultaneous solution of all unit equations
   - [External Solvers: pounce and discopt](external-solvers.md) — a flowsheet as a flat NLP or as an implicit residual block
2. [Uncertainty Propagation](#uncertainty-propagation)
   - [Linear Propagation](#linear-propagation)
   - [Monte Carlo Propagation](#monte-carlo-propagation)
   - [Sensitivity Analysis](#sensitivity-analysis)
   - [Sobol Indices](#sobol-indices)
3. [Implicit Differentiation](#implicit-differentiation)
4. [Utility Functions](#utility-functions)

---

## Numerical Solvers

**External Libraries**: Difflow uses [optimistix](https://docs.kidger.site/optimistix/) for fixed-point iteration and root-finding, and [diffrax](https://docs.kidger.site/diffrax/) for ODE integration.

All solvers in Difflow are:
- Implemented using JAX primitives for automatic differentiation
- Support implicit differentiation for efficient backward passes
- Fully JIT-compilable for performance

### Fixed-Point Iteration

Solves equations of the form $x^* = f(x^*, \text{args})$.

```python
import optimistix as optx

def solve_recycle(fresh_feed):
    """Fixed-point iteration for recycle streams."""

    def flowsheet_iteration(recycle, args):
        """Update function: recycle_out = f(recycle_in)"""
        # Mix fresh feed with recycle, run process, return new recycle
        ...
        return new_recycle

    # Create solver with tolerances
    solver = optx.FixedPointIteration(rtol=1e-8, atol=1e-8)

    # Solve for steady state
    solution = optx.fixed_point(
        fn=flowsheet_iteration,
        solver=solver,
        y0=initial_guess,
        args=fresh_feed,
        max_steps=100,
        throw=False  # Return solution even if not fully converged
    )
    return solution.value
```

**Algorithm**:

$$x^{(k+1)} = (1 - \alpha) x^{(k)} + \alpha \cdot f(x^{(k)}, \text{args})$$

Where $\alpha$ is the damping factor.

**Convergence Criterion**:

$$\|x^{(k+1)} - x^{(k)}\| < \epsilon$$

**Example**: Solving recycle loop

```python
import optimistix as optx

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
solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
solution = optx.fixed_point(
    fn=recycle_update,
    solver=solver,
    y0=0.1,  # Initial guess
    args=(fresh_feed, reactor_params),
    max_steps=100
)
recycle_ss = solution.value
```

### Newton-Raphson Solver

Solves equations of the form $g(x^*, \text{args}) = 0$.

```python
import optimistix as optx

def solve_equation(args):
    """Newton-Raphson root finding."""

    def residual_function(x, args):
        """Function whose root we seek: g(x, args) = 0"""
        # Return residual that should equal zero
        return ...

    # Create Newton solver
    solver = optx.Newton(rtol=1e-10, atol=1e-10)

    # Find root
    solution = optx.root_find(
        fn=residual_function,
        solver=solver,
        y0=initial_guess,
        args=args,
        max_steps=50
    )
    return solution.value
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
import optimistix as optx

def bubble_residual(T, args):
    x, P, K_func = args
    K = K_func(T)
    # Bubble point: sum(x_i * K_i) = 1
    return jnp.sum(x * K) - 1.0

solver = optx.Newton(rtol=1e-8, atol=1e-8)
solution = optx.root_find(
    fn=bubble_residual,
    solver=solver,
    y0=350.0,
    args=(liquid_composition, pressure, K_values_func),
    max_steps=50
)
T_bubble = solution.value
```

### Rachford-Rice Solver

Specialized solver for flash calculations. Uses optimistix's bisection method with automatic bounds.

```python
import optimistix as optx

def rachford_rice(psi, z, K):
    """Rachford-Rice equation: should equal zero at solution."""
    return jnp.sum(z * (K - 1) / (1 + psi * (K - 1)))

def solve_flash(z, K):
    """Solve for vapor fraction using bisection."""
    # Bisection bounds for vapor fraction
    psi_min = 1 / (1 - jnp.max(K)) + 1e-6
    psi_max = 1 / (1 - jnp.min(K)) - 1e-6
    psi_min = jnp.clip(psi_min, 0.0, None)
    psi_max = jnp.clip(psi_max, None, 1.0)

    solver = optx.Bisection(rtol=1e-10, atol=1e-10)
    solution = optx.root_find(
        fn=rachford_rice,
        solver=solver,
        y0=0.5,
        args=(z, K),
        options={"lower": psi_min, "upper": psi_max},
        max_steps=50
    )
    return solution.value

# Get phase compositions from vapor fraction
def phase_compositions(z, K, psi):
    x = z / (1 + psi * (K - 1))  # Liquid
    y = K * x                    # Vapor
    return x, y
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

Used internally by PFR, fed-batch reactor, and other dynamic models. Difflow uses [diffrax](https://docs.kidger.site/diffrax/) for ODE integration.

```python
import diffrax

def integrate_ode(y0, t_span, derivative_fn, args):
    """Integrate ODE using diffrax."""

    def vector_field(t, y, args):
        """dy/dt = f(t, y, args)"""
        return derivative_fn(y, args)

    # Define the ODE term
    term = diffrax.ODETerm(vector_field)

    # Choose a solver (adaptive)
    solver = diffrax.Tsit5()  # 5th order adaptive method

    # Configure step size controller
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Optionally save at specific points
    saveat = diffrax.SaveAt(ts=jnp.linspace(t_span[0], t_span[1], 101))

    # Solve
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.01,  # Initial step size
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller
    )
    return solution.ys  # Trajectory at saved points
```

**Why diffrax?**
- Adaptive step size control for efficiency and accuracy
- Fully differentiable through the integration (supports adjoint methods)
- JIT-compilable
- Multiple solvers available (Tsit5, Dopri5, Kvaerno5 for stiff problems)

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

### Implementation with Optimistix

Optimistix handles implicit differentiation automatically. When you use `optx.fixed_point()` or `optx.root_find()`, gradients are computed using the implicit function theorem rather than by backpropagating through the solver iterations.

```python
import optimistix as optx
import jax

def optimize_with_gradients(params):
    """Example showing automatic gradient computation through solver."""

    def my_fixed_point(x, args):
        # Fixed-point function that depends on params
        return some_function(x, args, params)

    solver = optx.FixedPointIteration(rtol=1e-8, atol=1e-8)
    solution = optx.fixed_point(
        fn=my_fixed_point,
        solver=solver,
        y0=initial_guess,
        args=args,
        max_steps=100
    )

    # The solution is differentiable w.r.t. params
    return loss_function(solution.value)

# Gradients computed via implicit differentiation
grad_fn = jax.grad(optimize_with_gradients)
gradients = grad_fn(params)
```

**Note on Numerical Challenges**: Implicit differentiation requires inverting a matrix $(I - \partial f/\partial x)$ at the solution. This can become singular or ill-conditioned when:
- The system is near a bifurcation point
- Reactions go to very high conversion (near 100%)
- The system is at a phase boundary
- Recycle ratios are extreme

If gradient computation fails while the forward solve succeeds, consider using finite differences for gradients or reformulating the problem.

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
| Fixed-point (well-behaved) | `optx.FixedPointIteration` |
| Fixed-point (difficult) | Increase max_steps, adjust tolerances |
| Root-finding | `optx.Newton` or `optx.Bisection` (bounded) |
| Flash calculation | `optx.Bisection` with Rachford-Rice bounds |
| ODE integration | `diffrax.Tsit5` (adaptive), `diffrax.Kvaerno5` (stiff) |

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
