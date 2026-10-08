# Equation-Oriented (EO) Solver

## Overview

The EO solver is an alternative to the default sequential modular (SM) solver for solving flowsheets with recycle loops. Instead of evaluating units one-by-one and iterating on tear streams, the EO solver assembles all unit equations and connectivity constraints into a single nonlinear system $F(x) = 0$ and solves simultaneously using Newton's method.

## Mathematical Formulation

### State Vector

The state vector $x$ contains variables for all non-feed streams:

$$x = [x_1 | x_2 | \ldots | x_M]$$

where each stream contributes $N+2$ variables ($N$ species flows + temperature + pressure):

$$x_i = [F_{i,1}, F_{i,2}, \ldots, F_{i,N}, T_i, P_i]$$

Feed streams are treated as parameters, not unknowns.

### Residual Assembly

Each unit operation provides an `eo_residuals(inlets, outlets)` method that returns a vector of residuals. For the system to be square, the total number of residuals must equal the total number of unknowns.

**Example — CSTR (isothermal):**
- Material balance: $F_{out,i} - F_{in,i} - V \sum_j \nu_{ij} r_j = 0$ for each species
- Temperature: $T_{out} - T_{spec} = 0$
- Pressure: $P_{out} - P_{in} = 0$

**Example — Flash separator:**
- Material balance: $F_{in,i} - F_{liq,i} - F_{vap,i} = 0$
- Phase equilibrium: $F_{vap,i} L_{total} - K_i F_{liq,i} V_{total} = 0$
- Temperature and pressure specifications for both outlet phases

**Example — EOSFlash (cubic EOS):** the same balances and T/P rows, with the
phase-split rows chosen by a Michelsen stability test on the *feed* (the one
the sequential flash uses, held off the tape with `stop_gradient`):
- two-phase: fugacity equality $x_i \hat\varphi_i^L(x) - y_i \hat\varphi_i^V(y) = 0$, with $x = F_{liq}/L$, $y = F_{vap}/V$
- liquid only: $F_{vap,i} = 0$; vapor only: $F_{liq,i} = 0$

The choice depends only on the inlet, so the residual is smooth within a phase
regime and its Jacobian stays regular when one phase is absent.

**Example — EnthalpyCounterCurrentHX:** per side, flows and pressure pass
through and the energy row is the enthalpy balance with the duty eliminated,
$H_h(T_{h,in}) - H_h(T_{h,out}) - UA\,\mathrm{LMTD} = 0$ and
$H_c(T_{c,out}) - H_c(T_{c,in}) - UA\,\mathrm{LMTD} = 0$, using the thermo's
`stream_enthalpy_flash`. A non-isothermal CSTR's energy row is likewise its
enthalpy balance, $H_{out} - H_{in} + V\sum_j r_j \Delta H_j + H_{mix} - Q = 0$.

### Newton's Method

The system is solved using `optimistix.root_find` with a Newton solver. JAX computes the Jacobian automatically via automatic differentiation. The implicit function theorem provides gradients through the converged solution.

## API Reference

### `Flowsheet.solve_eo()`

<!-- doc-test: skip: signature listing, not runnable code -->
```python
def solve_eo(
    self,
    initial_guess: dict[str, Stream] | None = None,
    use_sm_init: bool = True,
    tol: float = 1e-8,
    max_steps: int = 100,
) -> dict[str, Stream]
```

Solve the flowsheet using the EO approach. Returns a dictionary of all streams. This method is JAX-traceable and can be used inside `jax.grad`.

**Parameters:**
- `initial_guess`: Initial values for unknown streams
- `use_sm_init`: If True and no initial_guess, run SM solver first for a good starting point
- `tol`: Convergence tolerance
- `max_steps`: Maximum Newton iterations

### `EOSolver`

```python
from difflow import EOSolver, Flowsheet, Heater, HeaterParams, Unit, make_stream

flowsheet = Flowsheet(species_order=["A", "B"])
flowsheet.add_feed("feed", make_stream({"A": 5.0, "B": 5.0}, T=300.0, P=101325.0))
flowsheet.add_unit(Unit("heater", Heater(HeaterParams(T_out=400.0)),
                        ["feed"], ["hot_out"]))

solver = EOSolver(flowsheet)
result = solver.solve(use_sm_init=True, tol=1e-8)
```

Direct access to the EO solver with convergence diagnostics.

**Methods:**
- `solve()` → `EOSolveResult` — Full solve with diagnostics (not JAX-traceable)
- `solve_streams()` → `dict[str, Stream]` — JAX-traceable solve

### `EOSolveResult`

<!-- doc-test: skip: dataclass field listing, not runnable code -->
```python
@dataclass
class EOSolveResult:
    streams: dict[str, Stream]
    converged: bool
    residual_norm: float
    n_iterations: int
    wall_time: float
```

### `EOStateLayout`

```python
from difflow import EOStateLayout, make_stream

streams_dict = {
    "s1": make_stream({"A": 1.0, "B": 2.0}, T=300.0, P=101325.0),
    "s2": make_stream({"A": 0.5, "B": 0.5}, T=350.0, P=101325.0),
}
layout = EOStateLayout(species_order=["A", "B"], stream_names=["s1", "s2"])
x = layout.pack(streams_dict)
streams = layout.unpack(x)
```

Manages mapping between flat state vector and named streams.

### `solve_residual_system` (#196)

Some models already *are* a residual and do not need a `Flowsheet` built around
them -- a counter-current equilibrium section, for instance. This is the
section-scope entry point for those:

```python
import jax.numpy as jnp
from difflow.eo_solver import solve_residual_system

def residual_fn(z, args):          # solve z**2 = a for each component
    return z**2 - args

z0 = jnp.array([1.0, 1.0])
args = jnp.array([2.0, 9.0])

z, residual_norm, feasible = solve_residual_system(
    residual_fn,          # (z, args) -> r, same shape as z; JAX-traceable
    z0,                   # initial guess; scale it well, Newton is local
    args,                 # any pytree, differentiable
    rtol=1e-12, atol=1e-12,
    max_steps=200,
)
```

It is one `optimistix.root_find`, so:

- the reverse-mode tape is constant size rather than proportional to stages
  times iterations -- optimistix differentiates the converged solution
  implicitly, it does not tape the iteration;
- the Jacobian `dr/dz` is an ordinary `jax.jacobian` of `residual_fn`, which is
  the object the linearization, back-off and estimation layers want;
- a recycle tear is just another row of `r`.

**Soft failure.** Nothing is raised. One cannot raise from inside `vmap` or
`scan`, so failure comes back as a value: `residual_norm` and `feasible` are
traced arrays a caller branches on with `jnp.where`, and a non-converged `z` is
still returned because the converged members of a batch have to come back too.

**Tolerance.** `rtol`/`atol` default to `1e-12`, far below any outer flowsheet
tolerance. Keep it that way: a loosely converged inner solve gives an
implicit-function gradient that is exact for the solution manifold but
inconsistent with the value the code actually returned, and the resulting
finite-difference disagreement is very hard to diagnose afterwards.

First user: the REE mass-action closure, `difflow_ree.equilibrium.mass_action`
(see [REE unit operations](unit-operations-ree.md)).

## Comparison: SM vs EO

| Aspect | Sequential Modular | Equation-Oriented |
|--------|-------------------|-------------------|
| Convergence | Linear (fixed-point) | Quadratic (Newton) |
| Iterations | Many for tight recycles | Few near solution |
| Per-iteration cost | Low (one unit eval) | High (full Jacobian) |
| Initialization | Tolerant of poor guesses | Needs reasonable guess |
| Best for | Simple, loosely coupled | Tightly coupled, optimization |

The two are not exclusive: `solve_eo(use_sm_init=True)` (the default) runs the
sequential solver first and hands its result to Newton as the starting point,
which is how the EO route gets a guess good enough to converge from. See
[The equation-oriented route](convergence.md#the-equation-oriented-route), and
[Convergence and Initialization](convergence.md) for tear guesses, the
acceleration methods and the traced fallback.

## Adding EO Support to New Units

To add EO support to a new unit operation, implement the `eo_residuals` method:

```python
class MyUnit:
    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        **kwargs,
    ) -> Array:
        """Return flat array of residuals.

        Number of residuals must equal the number of outlet
        stream variables this unit produces.
        """
        inlet = inlets[0]
        outlet = outlets[0]

        # Material balance residuals
        mat_resid = [...]

        # Energy/temperature residual
        T_resid = [...]

        # Pressure residual
        P_resid = [...]

        return jnp.concatenate(mat_resid + T_resid + P_resid)
```

**Requirements:**
- Residuals must be zero at the correct solution
- Number of residuals = number of outlet stream variables (N_species + 2 per outlet)
- All computations must use JAX operations (`jnp`, not `np`)
- No Python control flow on traced values

If a unit does not implement `eo_residuals`, the EO solver falls back to running the unit forward and computing the difference between computed and current outlet values.
