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
import jax.numpy as jnp
import optimistix as optx

fresh_feed = 1.0          # mol/s of A
reactor_params = {'conversion': 0.6, 'recycle_fraction': 0.9}

def recycle_update(recycle_flow, args):
    feed, p = args

    # Mix fresh feed with recycle
    mixed = feed + recycle_flow

    # Reactor: unconverted A leaves, a fraction of it is recycled
    unconverted = mixed * (1.0 - p['conversion'])

    # Return new recycle flow (converges when input = output)
    return p['recycle_fraction'] * unconverted

# Solve for steady-state recycle flow
solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
solution = optx.fixed_point(
    fn=recycle_update,
    solver=solver,
    y0=jnp.asarray(0.1),  # Initial guess
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
import jax.numpy as jnp
import optimistix as optx

# Mole fractions, pressure and a simple K-value model (illustrative constants)
liquid_composition = jnp.array([0.4, 0.6])
pressure = 101325.0
A, B = jnp.array([10.8, 11.5]), jnp.array([2800.0, 3700.0])

def K_values_func(T):
    return jnp.exp(A - B / T) * 1e5 / pressure

def bubble_residual(T, args):
    x, P, K_func = args
    K = K_func(T)
    # Bubble point: sum(x_i * K_i) = 1
    return jnp.sum(x * K) - 1.0

solver = optx.Newton(rtol=1e-8, atol=1e-8)
solution = optx.root_find(
    fn=bubble_residual,
    solver=solver,
    y0=jnp.asarray(350.0),
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
import jax
import jax.numpy as jnp
from difflow.uncertainty import linear_propagation

# Define model and uncertainties: models take a dict of parameters
def model(params):
    # First-order reaction in a CSTR: conversion = k*tau / (1 + k*tau)
    k = 1e5 * jnp.exp(-6000.0 / params['T'])
    tau = 10.0 / params['F']
    return k * tau / (1.0 + k * tau)

nominal = {'T': 350.0, 'F': 1.0}
uncertainties = {'T': 5.0, 'F': 0.05}  # 1-sigma

# Propagate uncertainty (returns output, std and a sensitivity dict)
mean, std, info = linear_propagation(model, nominal, uncertainties)
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
mean, std, info = monte_carlo_propagation(
    model=model,
    nominal_params=nominal,
    uncertainties=uncertainties,
    n_samples=10000,
    distribution='normal'  # or 'uniform'
)

print(f"Mean: {float(mean):.4f}")
print(f"Std: {float(std):.4f}")
print(f"2.5th percentile: {info['p2.5']:.4f}")
print(f"97.5th percentile: {info['p97.5']:.4f}")
```

**Algorithm**:
1. Generate $N$ samples from input distribution
2. Evaluate model for each sample (vectorized with `vmap`)
3. Compute output statistics

**Implementation** (vectorized for efficiency):

```python
import jax.numpy as jnp
from jax import vmap, random

def monte_carlo_sketch(model, nominal, uncertainties, n_samples, key=None):
    if key is None:
        key = random.PRNGKey(0)

    # Generate samples
    samples = nominal + uncertainties * random.normal(key, shape=(n_samples, len(nominal)))

    # Vectorized model evaluation (model takes an array here)
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
    nominal_params=nominal,
    param_ranges={'T': (340.0, 360.0), 'F': (0.8, 1.2)},  # optional; default +/-20%
)

for name, sens in sensitivities.items():
    print(f"{name}: gradient = {float(sens['gradient']):.4f}, elasticity = {float(sens['elasticity']):.4f}")
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
    param_bounds={'T': (340.0, 360.0), 'F': (0.8, 1.2)},
    n_samples=1024,
)

for name, idx in indices.items():
    print(f"{name}: S1 = {idx['S1']:.3f} (normalized {idx['S1_normalized']:.3f})")
```

`sobol_indices` returns first-order indices only (`S1`, `S1_normalized`,
`bounds`); the total-order index below is the definition, not something the
function computes. For production-grade total-order indices use a dedicated
package such as SALib.

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

# Propagate covariance (the Jacobian is computed internally)
output, cov_output, jacobian = propagate_covariance(
    model, nominal, cov_input, param_order=['T', 'F'])
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
import jax.numpy as jnp

def optimize_with_gradients(params):
    """Example showing automatic gradient computation through solver."""

    def my_fixed_point(x, args):
        # Fixed-point function that depends on params
        return params['a'] * jnp.cos(x) + args

    solver = optx.FixedPointIteration(rtol=1e-8, atol=1e-8)
    solution = optx.fixed_point(
        fn=my_fixed_point,
        solver=solver,
        y0=jnp.asarray(0.5),
        args=0.1,
        max_steps=100
    )

    # The solution is differentiable w.r.t. params
    return solution.value ** 2

# Gradients computed via implicit differentiation
grad_fn = jax.grad(optimize_with_gradients)
params = {'a': jnp.asarray(0.4)}
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
import jax.numpy as jnp
from difflow.numerics import (
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_exp,
    smooth_max,
    smooth_min,
    smooth_clamp,
)

a, b = jnp.array(2.0), jnp.array(0.0)

# Safe operations (avoid NaN/Inf)
x = safe_divide(a, b)               # Uses eps (1e-10) in place of a zero denominator
y = safe_log(jnp.array(0.0))        # Clips x to avoid log(0)
z = safe_sqrt(jnp.array(-1.0))      # Clips x to avoid sqrt(negative)

# Smooth approximations (differentiable)
max_val = smooth_max(a, b, alpha=10.0)  # Log-sum-exp approximation
min_val = smooth_min(a, b, alpha=10.0)  # Log-sum-exp approximation
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

Unit conversions are deliberately not wrapped in helper functions: difflow
works in SI throughout (K, Pa, mol, kg, J), and thermodynamic properties come
from `difflow.thermo` and `difflow.eos`.

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
import jax.numpy as jnp
from difflow.uncertainty import (
    linear_propagation, sensitivity_analysis, monte_carlo_propagation, sobol_indices,
)

# 1. Define model
def process_model(params):
    # ... process simulation ...
    return params['F'] * jnp.exp(-1000.0 / params['T']) * (params['P'] / 101325.0)

# 2. Identify uncertain parameters
nominal = {'T': 350.0, 'P': 101325.0, 'F': 1.0}
uncertainties = {'T': 10.0, 'P': 5000.0, 'F': 0.1}

# 3. Quick screening with linear propagation
mean, std, info = linear_propagation(process_model, nominal, uncertainties)

# 4. Identify important parameters with sensitivity analysis
sens = sensitivity_analysis(process_model, nominal)

# 5. Detailed analysis on key parameters with Monte Carlo
mc_mean, mc_std, mc_info = monte_carlo_propagation(
    process_model, nominal, uncertainties, n_samples=10000)

# 6. Global sensitivity with Sobol indices (if needed)
bounds = {k: (v - 2 * uncertainties[k], v + 2 * uncertainties[k]) for k, v in nominal.items()}
sobol = sobol_indices(process_model, bounds, n_samples=1024)
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
