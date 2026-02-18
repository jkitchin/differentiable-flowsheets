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

### Newton's Method

The system is solved using `optimistix.root_find` with a Newton solver. JAX computes the Jacobian automatically via automatic differentiation. The implicit function theorem provides gradients through the converged solution.

## API Reference

### `Flowsheet.solve_eo()`

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
solver = EOSolver(flowsheet)
result = solver.solve(use_sm_init=True, tol=1e-8)
```

Direct access to the EO solver with convergence diagnostics.

**Methods:**
- `solve()` → `EOSolveResult` — Full solve with diagnostics (not JAX-traceable)
- `solve_streams()` → `dict[str, Stream]` — JAX-traceable solve

### `EOSolveResult`

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
layout = EOStateLayout(species_order=["A", "B"], stream_names=["s1", "s2"])
x = layout.pack(streams_dict)
streams = layout.unpack(x)
```

Manages mapping between flat state vector and named streams.

## Comparison: SM vs EO

| Aspect | Sequential Modular | Equation-Oriented |
|--------|-------------------|-------------------|
| Convergence | Linear (fixed-point) | Quadratic (Newton) |
| Iterations | Many for tight recycles | Few near solution |
| Per-iteration cost | Low (one unit eval) | High (full Jacobian) |
| Initialization | Tolerant of poor guesses | Needs reasonable guess |
| Best for | Simple, loosely coupled | Tightly coupled, optimization |

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
