# Optimize a Flowsheet

Set up gradient-based optimization for a flowsheet or unit operation.

## Arguments
- $ARGUMENTS: Description of optimization goal (optional)

## Instructions

Help the user set up an optimization problem:

1. **Identify the objective**: What should be minimized/maximized?
   - Conversion, yield, purity
   - Cost, profit, NPV
   - Energy consumption
   - Environmental metrics

2. **Identify decision variables**: What can be adjusted?
   - Operating conditions (T, P, flow rates)
   - Design parameters (volumes, areas)
   - Feed compositions

3. **Identify constraints**: What limits apply?
   - Physical bounds (T > 0, 0 < X < 1)
   - Equipment limits
   - Safety constraints

4. **Set up optimization** using JAX + optax:

```python
import jax
import jax.numpy as jnp
import optax
from difflow import ...

# Define objective function
def objective(params):
    # Run flowsheet simulation
    result = simulate(params)
    # Return scalar to minimize
    return -result['profit']  # Negative for maximization

# Gradient function
grad_obj = jax.grad(objective)

# Optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(initial_params)

# Optimization loop
params = initial_params
for i in range(n_iterations):
    grads = grad_obj(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if i % 10 == 0:
        print(f"Iter {i}: obj = {objective(params):.4f}")
```

5. **Handle constraints** with:
   - Penalty methods: Add penalty term to objective
   - Projection: Clip parameters to bounds after each update
   - Barrier methods: Add log-barrier for inequality constraints

Provide a complete, runnable optimization script tailored to the user's problem.
