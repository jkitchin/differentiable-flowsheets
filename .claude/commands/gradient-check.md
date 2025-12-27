# Check Gradients

Verify that gradients are computed correctly for a unit operation or function.

## Arguments
- $ARGUMENTS: The unit or function to check (e.g., "CSTR", "PFR", "Flash")

## Instructions

Create a test script that:

1. Imports the specified unit from difflow
2. Creates a simple test case with known inputs
3. Uses `jax.test_util.check_grads()` to verify gradients match finite differences
4. Reports pass/fail status

Example gradient check pattern:
```python
import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from difflow import <Unit>, Stream

# Create test inputs
# ...

# Define function to check
def fn_to_check(params):
    # ... use unit with params
    return scalar_output

# Check gradients
try:
    check_grads(fn_to_check, (test_params,), order=1, modes=['rev'])
    print("✅ Gradient check passed!")
except AssertionError as e:
    print(f"❌ Gradient check failed: {e}")
```

Run the check and report results.
