"""Diffrax backend for advanced ODE/SDE integration.

This module provides integration with diffrax, a JAX-based library for
numerical differential equation solvers. Diffrax offers:

- Adaptive step-size control with error estimation
- Implicit solvers for stiff systems
- Continuous-time interpolation
- Event handling
- SDE solvers (Brownian motion, etc.)

This backend is optional - diffrax must be installed separately:
    pip install diffrax

Example Usage
-------------

>>> from difflow.dynamic import integrate, DynamicCSTR
>>> from difflow.streams import make_stream
>>>
>>> # Use diffrax with default Dopri5 solver
>>> result = integrate(
...     f, y0, t_span=(0, 100),
...     method="diffrax",  # or "diffrax:dopri5"
... )
>>>
>>> # Use specific diffrax solver
>>> result = integrate(
...     f, y0, t_span,
...     method="diffrax:kvaerno5",  # Implicit solver for stiff systems
...     rtol=1e-6, atol=1e-8,
... )
>>>
>>> # Direct access to diffrax solvers
>>> from difflow.dynamic.diffrax_backend import integrate_diffrax
>>> result = integrate_diffrax(
...     unit, inputs, t_span,
...     solver="tsit5",
...     saveat=jnp.linspace(0, 100, 101),
... )

Available Solvers
-----------------

Explicit (non-stiff):
- dopri5: Dormand-Prince 5(4) - good general-purpose choice
- dopri8: Dormand-Prince 8(7) - higher accuracy
- tsit5: Tsitouras 5(4) - efficient, recommended for most problems
- heun: Heun's method (2nd order)
- euler: Forward Euler (1st order)
- midpoint: Midpoint method (2nd order)

Implicit (stiff):
- kvaerno3: 3rd order implicit
- kvaerno4: 4th order implicit
- kvaerno5: 5th order implicit - good for stiff problems

Symplectic (Hamiltonian systems):
- leapfrog_midpoint: Symplectic leapfrog

Semi-implicit:
- semi_implicit_euler: For certain DAE-like structures
"""

from typing import Callable, Any, Literal
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

from difflow.dynamic.integrators import (
    IntegrationResult,
    Trajectory,
    IntegrationInfo,
    DerivativesFn,
)

# Check if diffrax is available
try:
    import diffrax
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False
    diffrax = None


# Solver name mapping
DIFFRAX_SOLVERS = {
    # Explicit adaptive
    "dopri5": "Dopri5",
    "dopri8": "Dopri8",
    "tsit5": "Tsit5",
    "bosh3": "Bosh3",
    # Explicit fixed
    "euler": "Euler",
    "heun": "Heun",
    "midpoint": "Midpoint",
    "ralston": "Ralston",
    # Implicit (for stiff systems)
    "kvaerno3": "Kvaerno3",
    "kvaerno4": "Kvaerno4",
    "kvaerno5": "Kvaerno5",
    "implicit_euler": "ImplicitEuler",
    # Semi-implicit
    "semi_implicit_euler": "SemiImplicitEuler",
    # Symplectic
    "leapfrog_midpoint": "LeapfrogMidpoint",
}

# Default solver for different problem types
DEFAULT_SOLVER = "tsit5"
DEFAULT_STIFF_SOLVER = "kvaerno5"


def _get_solver(name: str) -> Any:
    """Get diffrax solver class by name."""
    if not HAS_DIFFRAX:
        raise ImportError(
            "diffrax is required for this backend. Install with: pip install diffrax"
        )

    name_lower = name.lower()
    if name_lower not in DIFFRAX_SOLVERS:
        available = ", ".join(sorted(DIFFRAX_SOLVERS.keys()))
        raise ValueError(
            f"Unknown diffrax solver: {name}. Available: {available}"
        )

    solver_class_name = DIFFRAX_SOLVERS[name_lower]
    return getattr(diffrax, solver_class_name)()


def _get_stepsize_controller(
    rtol: float = 1e-5,
    atol: float = 1e-7,
    solver_name: str = "tsit5",
    dt0: float | None = None,
) -> Any:
    """Get appropriate stepsize controller for solver."""
    if not HAS_DIFFRAX:
        raise ImportError("diffrax is required")

    # Fixed-step solvers (no error estimate)
    fixed_step_solvers = {"euler", "heun", "midpoint", "ralston", "leapfrog_midpoint"}

    # Implicit solvers need different controller settings
    implicit_solvers = {"kvaerno3", "kvaerno4", "kvaerno5", "implicit_euler"}

    if solver_name.lower() in fixed_step_solvers:
        # Use constant step size for non-adaptive solvers
        return diffrax.ConstantStepSize()
    elif solver_name.lower() in implicit_solvers:
        return diffrax.PIDController(
            rtol=rtol,
            atol=atol,
            pcoeff=0.4,
            icoeff=0.3,
            dcoeff=0.0,
        )
    else:
        return diffrax.PIDController(rtol=rtol, atol=atol)


def integrate_diffrax(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    solver: str = DEFAULT_SOLVER,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    dt0: float | None = None,
    max_steps: int = 16**4,
    saveat: Array | None = None,
    dense: bool = False,
    bounds: tuple[Array, Array] | None = None,
    **kwargs,
) -> IntegrationResult:
    """Integrate ODE using diffrax.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial state array
        t_span: (t_start, t_end) time interval
        solver: Solver name (see DIFFRAX_SOLVERS)
        rtol: Relative tolerance
        atol: Absolute tolerance
        dt0: Initial step size (auto if None)
        max_steps: Maximum number of steps
        saveat: Time points to save solution (default: just endpoints)
        dense: Whether to use dense output interpolation
        bounds: Optional (lower, upper) arrays for post-integration state
            clipping. Applied to the trajectory and final state.
        **kwargs: Additional arguments passed to diffeqsolve

    Returns:
        IntegrationResult with final state and trajectory
    """
    if not HAS_DIFFRAX:
        raise ImportError(
            "diffrax is required for this backend. Install with: pip install diffrax"
        )

    t0, t1 = t_span
    y0 = jnp.asarray(y0)

    # Create ODE term
    def vector_field(t, y, args):
        return f(t, y)

    term = diffrax.ODETerm(vector_field)

    # Get solver
    solver_obj = _get_solver(solver)

    # Stepsize controller
    stepsize_controller = _get_stepsize_controller(rtol, atol, solver)

    # Initial step size
    if dt0 is None:
        dt0 = (t1 - t0) / 100.0

    # Save points
    if saveat is not None:
        saveat_obj = diffrax.SaveAt(ts=saveat)
    elif dense:
        saveat_obj = diffrax.SaveAt(dense=True)
    else:
        # Save at regular intervals for trajectory
        n_save = min(101, max_steps)
        ts = jnp.linspace(t0, t1, n_save)
        saveat_obj = diffrax.SaveAt(ts=ts)

    # Solve
    solution = diffrax.diffeqsolve(
        term,
        solver_obj,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat_obj,
        max_steps=max_steps,
        **kwargs,
    )

    # Extract results
    y_final = solution.ys[-1]
    ys = solution.ys

    # Apply bounds clipping if requested
    if bounds is not None:
        ys = jnp.clip(ys, bounds[0], bounds[1])
        y_final = ys[-1]

    # Build trajectory
    trajectory = Trajectory(
        t=solution.ts,
        y=ys,
    )

    # Build info
    stats = solution.stats
    info = IntegrationInfo(
        n_steps=int(stats.get("num_steps", 0)),
        n_eval=int(stats.get("num_accepted_steps", 0) + stats.get("num_rejected_steps", 0)),
        success=solution.result == diffrax.RESULTS.successful,
        message=f"diffrax:{solver} - {solution.result}",
    )

    return IntegrationResult(
        y_final=y_final,
        trajectory=trajectory,
        info=info,
    )


def integrate_diffrax_unit(
    unit,  # DynamicUnit
    inputs: dict,
    t_span: tuple[float, float],
    y0: Array | None = None,
    solver: str = DEFAULT_SOLVER,
    **kwargs,
) -> IntegrationResult:
    """Integrate a DynamicUnit using diffrax.

    Convenience wrapper that handles unit initialization and
    creates the appropriate derivative function.

    Args:
        unit: DynamicUnit to integrate
        inputs: Dictionary of inlet streams
        t_span: Time interval
        y0: Initial state (uses unit.initial_state if None)
        solver: Diffrax solver name
        **kwargs: Additional arguments for integrate_diffrax

    Returns:
        IntegrationResult
    """
    if y0 is None:
        y0 = unit.initial_state(inputs)

    def f(t, y):
        return unit.derivatives(t, y, inputs)

    return integrate_diffrax(f, y0, t_span, solver=solver, **kwargs)


def integrate_stiff(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    rtol: float = 1e-5,
    atol: float = 1e-7,
    **kwargs,
) -> IntegrationResult:
    """Integrate stiff ODE using implicit solver.

    Uses Kvaerno5 implicit solver by default, which is suitable
    for stiff systems (e.g., reactions with very different time scales).

    Args:
        f: Derivative function
        y0: Initial state
        t_span: Time interval
        rtol: Relative tolerance
        atol: Absolute tolerance
        **kwargs: Additional arguments

    Returns:
        IntegrationResult
    """
    return integrate_diffrax(
        f, y0, t_span,
        solver=DEFAULT_STIFF_SOLVER,
        rtol=rtol,
        atol=atol,
        **kwargs,
    )


# Convenience functions for common solver choices

def integrate_dopri5(f, y0, t_span, **kwargs) -> IntegrationResult:
    """Integrate using Dormand-Prince 5(4) method."""
    return integrate_diffrax(f, y0, t_span, solver="dopri5", **kwargs)


def integrate_tsit5(f, y0, t_span, **kwargs) -> IntegrationResult:
    """Integrate using Tsitouras 5(4) method (recommended)."""
    return integrate_diffrax(f, y0, t_span, solver="tsit5", **kwargs)


def integrate_kvaerno5(f, y0, t_span, **kwargs) -> IntegrationResult:
    """Integrate using Kvaerno5 implicit method (for stiff systems)."""
    return integrate_diffrax(f, y0, t_span, solver="kvaerno5", **kwargs)


# List available solvers
def list_diffrax_solvers() -> list[str]:
    """List available diffrax solvers."""
    return sorted(DIFFRAX_SOLVERS.keys())


def check_diffrax_available() -> bool:
    """Check if diffrax is installed."""
    return HAS_DIFFRAX
