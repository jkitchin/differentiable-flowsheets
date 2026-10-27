---
name: difflow-gradient-check
description: Verifies that JAX gradients through a difflow unit operation or flowsheet
  match finite differences, and tracks down NaN or wrong gradients. Use when the user
  asks to check, test or debug gradients, derivatives, sensitivities or jax.grad of a
  difflow unit (CSTR, PFR, Flash, distillation, ...) or flowsheet, or sees NaN gradients.
license: MIT
compatibility: Requires Python 3.11+ and difflow.
---

# Check difflow gradients

The argument (if any) names the unit or function, e.g. `CSTR`.

1. Look up the unit and its parameters (`help(difflow.<Unit>Params)`); do not
   guess field names.
2. Write a scalar function of the variable(s) to differentiate: a parameter,
   a feed flow, `T` or `P`. Rebuild the `Params` inside the function so the
   tracer reaches the unit.
3. Compare reverse mode to finite differences, then forward mode:

```python
import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from difflow import StoichiometricReactor, StoichiometricReactorParams, make_stream

def f(X):
    rx = StoichiometricReactor(StoichiometricReactorParams(
        stoich={"CO": -1, "H2": -2, "MeOH": 1}, key="CO", conversion=X))
    out, _ = rx(make_stream({"CO": 100.0, "H2": 200.0}, T=500.0, P=5e6))
    return out["F_MeOH"]

x0 = jnp.array(0.25)
print("grad:", jax.grad(f)(x0))
check_grads(f, (x0,), order=1, modes=["rev"])
check_grads(f, (x0,), order=1, modes=["fwd"])
print("gradient check passed")
```

4. Report the gradient value, pass or fail, and the tolerance used.

## If it fails

- **NaN**: `jax.config.update("jax_debug_nans", True)` and rerun to find the
  operation. Usual causes: `log(0)`, `sqrt` of a negative, `0/0`, dividing by
  an empty flow.
- **Wrong but finite**: an inner solve not converged (check `info["converged"]`),
  a `jnp.clip`/`jnp.where` kink at the evaluation point, or a Python branch on a
  concrete value that froze the derivative. Move `x0` off any kink.
- **Mismatch only at large step**: pass `eps=` to `check_grads` or loosen
  `atol`/`rtol` and say so.
- **Through a recycle**: `jax.jacobian` routes the solve through the
  optimistix fixed-point path; make sure the recycle converges tightly
  (`tol_basis="error"`).
