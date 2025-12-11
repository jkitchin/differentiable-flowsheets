"""Unified ODE integrators for dynamic simulation.

This module provides differentiable ODE integrators that work with the
DynamicUnit protocol. All integrators are JAX-compatible and support
automatic differentiation through the solution.

Available integrators:
- RK4: Fixed-step 4th order Runge-Kutta (simple, fast)
- RK45: Adaptive step Runge-Kutta-Fehlberg (accurate, robust)
- Euler: First-order explicit (for testing/debugging)

All integrators return:
- Final state array
- Solution trajectory (time, states)
- Integration info (steps taken, etc.)

Example:
    >>> result = integrate(
    ...     derivatives_fn=unit.derivatives,
    ...     y0=initial_state,
    ...     t_span=(0.0, 100.0),
    ...     method="RK45",
    ... )
    >>> result.y_final  # Final state
    >>> result.trajectory.t  # Time points
    >>> result.trajectory.y  # State history
"""

from typing import Callable, NamedTuple, Literal, Any
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array, lax


# Type for derivative function: f(t, y, *args) -> dy/dt
DerivativesFn = Callable[[Array, Array], Array]


class Trajectory(NamedTuple):
    """Solution trajectory from integration.

    Attributes:
        t: Time points array, shape (n_steps + 1,)
        y: State arrays at each time, shape (n_steps + 1, n_states)
    """
    t: Array
    y: Array


class IntegrationInfo(NamedTuple):
    """Information about the integration process.

    Attributes:
        n_steps: Number of steps taken
        n_eval: Number of derivative evaluations
        success: Whether integration completed successfully
        message: Status message
    """
    n_steps: int
    n_eval: int
    success: bool
    message: str


@dataclass
class IntegrationResult:
    """Complete result from ODE integration.

    Attributes:
        y_final: Final state array
        trajectory: Time and state history
        info: Integration statistics
    """
    y_final: Array
    trajectory: Trajectory
    info: IntegrationInfo


# =============================================================================
# Fixed-Step RK4 Integrator
# =============================================================================

def rk4_step(
    f: DerivativesFn,
    t: Array,
    y: Array,
    dt: Array,
) -> Array:
    """Single RK4 step.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        State at t + dt
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)

    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    n_steps: int = 100,
    save_trajectory: bool = True,
) -> IntegrationResult:
    """Integrate ODE using fixed-step RK4.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial state
        t_span: (t_start, t_end)
        n_steps: Number of integration steps
        save_trajectory: Whether to save intermediate states

    Returns:
        IntegrationResult with final state and trajectory
    """
    t0, t_final = t_span
    dt = (t_final - t0) / n_steps

    t0 = jnp.asarray(t0)
    dt = jnp.asarray(dt)
    y0 = jnp.asarray(y0)

    def step_fn(carry, _):
        t, y = carry
        y_new = rk4_step(f, t, y, dt)
        t_new = t + dt
        return (t_new, y_new), (t_new, y_new)

    (t_final, y_final), (t_history, y_history) = lax.scan(
        step_fn, (t0, y0), None, length=n_steps
    )

    # Prepend initial state to trajectory
    t_traj = jnp.concatenate([jnp.array([t0]), t_history])
    y_traj = jnp.vstack([y0, y_history])

    return IntegrationResult(
        y_final=y_final,
        trajectory=Trajectory(t_traj, y_traj),
        info=IntegrationInfo(
            n_steps=n_steps,
            n_eval=4 * n_steps,
            success=True,
            message="RK4 integration complete",
        ),
    )


# =============================================================================
# Adaptive RK45 Integrator (Runge-Kutta-Fehlberg)
# =============================================================================

# Butcher tableau for RK45 (Fehlberg method)
# 4th order solution uses b coefficients
# 5th order solution uses b* coefficients
# Error estimate: y5 - y4

_RK45_A = jnp.array([
    [0, 0, 0, 0, 0, 0],
    [1/4, 0, 0, 0, 0, 0],
    [3/32, 9/32, 0, 0, 0, 0],
    [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
    [439/216, -8, 3680/513, -845/4104, 0, 0],
    [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0],
])

_RK45_B4 = jnp.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
_RK45_B5 = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
_RK45_C = jnp.array([0, 1/4, 3/8, 12/13, 1, 1/2])


def rk45_step(
    f: DerivativesFn,
    t: Array,
    y: Array,
    dt: Array,
) -> tuple[Array, Array, Array]:
    """Single RK45 step with error estimate.

    Args:
        f: Derivative function
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        (y4, y5, error): 4th order result, 5th order result, error estimate
    """
    # Compute stages
    k1 = f(t, y)
    k2 = f(t + _RK45_C[1]*dt, y + dt * _RK45_A[1, 0] * k1)
    k3 = f(t + _RK45_C[2]*dt, y + dt * (_RK45_A[2, 0]*k1 + _RK45_A[2, 1]*k2))
    k4 = f(t + _RK45_C[3]*dt, y + dt * (_RK45_A[3, 0]*k1 + _RK45_A[3, 1]*k2 + _RK45_A[3, 2]*k3))
    k5 = f(t + _RK45_C[4]*dt, y + dt * (_RK45_A[4, 0]*k1 + _RK45_A[4, 1]*k2 + _RK45_A[4, 2]*k3 + _RK45_A[4, 3]*k4))
    k6 = f(t + _RK45_C[5]*dt, y + dt * (_RK45_A[5, 0]*k1 + _RK45_A[5, 1]*k2 + _RK45_A[5, 2]*k3 + _RK45_A[5, 3]*k4 + _RK45_A[5, 4]*k5))

    # Stack stages for matrix multiply
    K = jnp.stack([k1, k2, k3, k4, k5, k6], axis=0)  # (6, n_states)

    # 4th and 5th order solutions
    y4 = y + dt * jnp.einsum('i,i...->...', _RK45_B4, K)
    y5 = y + dt * jnp.einsum('i,i...->...', _RK45_B5, K)

    # Error estimate (norm of difference)
    error = jnp.max(jnp.abs(y5 - y4))

    return y4, y5, error


def integrate_rk45(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 10000,
    initial_step: float | None = None,
    min_step: float = 1e-12,
    max_step: float | None = None,
) -> IntegrationResult:
    """Integrate ODE using adaptive RK45.

    Uses step size control based on local error estimate.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial state
        t_span: (t_start, t_end)
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_steps: Maximum number of steps
        initial_step: Initial step size (auto if None)
        min_step: Minimum step size
        max_step: Maximum step size (defaults to span/10)

    Returns:
        IntegrationResult with final state and trajectory
    """
    t0, t_final = t_span
    t0 = jnp.asarray(t0)
    t_final = jnp.asarray(t_final)
    y0 = jnp.asarray(y0)

    span = t_final - t0
    if max_step is None:
        max_step = span / 10.0

    if initial_step is None:
        # Estimate initial step from derivative magnitude
        dy0 = f(t0, y0)
        scale = atol + rtol * jnp.abs(y0)
        d0 = jnp.max(jnp.abs(dy0) / scale)
        initial_step = jnp.where(d0 > 1e-10, 0.01 / d0, span / 100.0)
        initial_step = jnp.clip(initial_step, min_step, max_step)

    dt0 = jnp.asarray(initial_step)

    # State for scan: (t, y, dt, n_steps, t_history, y_history)
    # We'll use a fixed-size buffer and track how many steps we've taken

    # Pre-allocate history buffers
    t_buffer = jnp.zeros(max_steps + 1)
    y_buffer = jnp.zeros((max_steps + 1, y0.shape[0]))
    t_buffer = t_buffer.at[0].set(t0)
    y_buffer = y_buffer.at[0].set(y0)

    def cond_fn(state):
        t, y, dt, step_idx, t_buf, y_buf, n_evals = state
        return (t < t_final) & (step_idx < max_steps)

    def body_fn(state):
        t, y, dt, step_idx, t_buf, y_buf, n_evals = state

        # Limit step to not overshoot
        dt = jnp.minimum(dt, t_final - t)

        # Take RK45 step
        y4, y5, error = rk45_step(f, t, y, dt)

        # Compute tolerance
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y5))
        error_ratio = error / jnp.max(scale)

        # Accept step if error is acceptable
        accept = error_ratio <= 1.0

        # New state (use y5 - the 5th order solution - if accepted)
        t_new = jnp.where(accept, t + dt, t)
        y_new = jnp.where(accept, y5, y)

        # Compute new step size using standard formula
        # dt_new = dt * (tol / error)^(1/5) with safety factor
        safety = 0.9
        factor_min = 0.2
        factor_max = 5.0

        factor = safety * jnp.power(1.0 / (error_ratio + 1e-10), 0.2)
        factor = jnp.clip(factor, factor_min, factor_max)
        dt_new = dt * factor
        dt_new = jnp.clip(dt_new, min_step, max_step)

        # Update history if step accepted
        new_idx = jnp.where(accept, step_idx + 1, step_idx)
        t_buf = jnp.where(
            accept,
            t_buf.at[step_idx + 1].set(t_new),
            t_buf
        )
        y_buf = jnp.where(
            accept,
            y_buf.at[step_idx + 1].set(y_new),
            y_buf
        )

        # Count evaluations (6 per attempt)
        n_evals_new = n_evals + 6

        return (t_new, y_new, dt_new, new_idx, t_buf, y_buf, n_evals_new)

    # Run integration
    init_state = (t0, y0, dt0, jnp.array(0), t_buffer, y_buffer, jnp.array(0))
    final_state = lax.while_loop(cond_fn, body_fn, init_state)

    t_end, y_final, dt_final, n_steps, t_history, y_history, n_evals = final_state

    # Trim history to actual steps taken
    # Note: n_steps is the index of the last filled entry
    n_steps_int = int(n_steps) + 1  # Convert to Python int for slicing
    trajectory = Trajectory(
        t=t_history[:n_steps_int + 1],
        y=y_history[:n_steps_int + 1],
    )

    return IntegrationResult(
        y_final=y_final,
        trajectory=trajectory,
        info=IntegrationInfo(
            n_steps=int(n_steps),
            n_eval=int(n_evals),
            success=True,
            message="RK45 integration complete",
        ),
    )


# =============================================================================
# Simple Euler Integrator (for testing)
# =============================================================================

def integrate_euler(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    n_steps: int = 1000,
) -> IntegrationResult:
    """Integrate ODE using explicit Euler method.

    Primarily for testing - use RK4 or RK45 for real work.

    Args:
        f: Derivative function
        y0: Initial state
        t_span: (t_start, t_end)
        n_steps: Number of steps

    Returns:
        IntegrationResult
    """
    t0, t_final = t_span
    dt = (t_final - t0) / n_steps

    t0 = jnp.asarray(t0)
    dt = jnp.asarray(dt)
    y0 = jnp.asarray(y0)

    def step_fn(carry, _):
        t, y = carry
        dy = f(t, y)
        y_new = y + dt * dy
        t_new = t + dt
        return (t_new, y_new), (t_new, y_new)

    (t_final, y_final), (t_history, y_history) = lax.scan(
        step_fn, (t0, y0), None, length=n_steps
    )

    t_traj = jnp.concatenate([jnp.array([t0]), t_history])
    y_traj = jnp.vstack([y0, y_history])

    return IntegrationResult(
        y_final=y_final,
        trajectory=Trajectory(t_traj, y_traj),
        info=IntegrationInfo(
            n_steps=n_steps,
            n_eval=n_steps,
            success=True,
            message="Euler integration complete",
        ),
    )


# =============================================================================
# Unified Interface
# =============================================================================

Method = Literal["RK4", "RK45", "Euler"]


def integrate(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    method: Method = "RK4",
    **kwargs,
) -> IntegrationResult:
    """Unified interface for ODE integration.

    Selects the appropriate integrator based on method parameter.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial state array
        t_span: (t_start, t_end) time interval
        method: Integration method ("RK4", "RK45", "Euler")
        **kwargs: Method-specific arguments

    Returns:
        IntegrationResult with final state, trajectory, and info

    Example:
        >>> def harmonic(t, y):
        ...     return jnp.array([y[1], -y[0]])
        >>> result = integrate(harmonic, jnp.array([1.0, 0.0]), (0, 10), "RK4")
        >>> result.y_final
    """
    if method == "RK4":
        n_steps = kwargs.get("n_steps", 100)
        return integrate_rk4(f, y0, t_span, n_steps=n_steps)

    elif method == "RK45":
        return integrate_rk45(
            f, y0, t_span,
            rtol=kwargs.get("rtol", 1e-6),
            atol=kwargs.get("atol", 1e-8),
            max_steps=kwargs.get("max_steps", 10000),
            initial_step=kwargs.get("initial_step"),
            min_step=kwargs.get("min_step", 1e-12),
            max_step=kwargs.get("max_step"),
        )

    elif method == "Euler":
        n_steps = kwargs.get("n_steps", 1000)
        return integrate_euler(f, y0, t_span, n_steps=n_steps)

    else:
        raise ValueError(f"Unknown integration method: {method}")


# =============================================================================
# Integration with DynamicUnit
# =============================================================================

def integrate_unit(
    unit,  # DynamicUnit
    inputs: dict,  # dict[str, Stream]
    t_span: tuple[float, float],
    y0: Array | None = None,
    method: Method = "RK4",
    **kwargs,
) -> IntegrationResult:
    """Integrate a DynamicUnit over time.

    Convenience function that wraps the unit's derivatives method
    and handles initialization.

    Args:
        unit: DynamicUnit to integrate
        inputs: Dictionary of inlet streams
        t_span: Time interval
        y0: Initial state (uses unit.initial_state if None)
        method: Integration method
        **kwargs: Method-specific arguments

    Returns:
        IntegrationResult
    """
    if y0 is None:
        y0 = unit.initial_state(inputs)

    def f(t, y):
        return unit.derivatives(t, y, inputs)

    return integrate(f, y0, t_span, method, **kwargs)


# =============================================================================
# Gradient-through-integration utilities
# =============================================================================

def integrate_with_grad(
    f: DerivativesFn,
    y0: Array,
    t_span: tuple[float, float],
    method: Method = "RK4",
    **kwargs,
) -> tuple[IntegrationResult, Callable]:
    """Integrate ODE and return function to compute gradients.

    This uses checkpointing to reduce memory usage for gradient
    computation through long integration trajectories.

    Args:
        f: Derivative function
        y0: Initial state
        t_span: Time interval
        method: Integration method
        **kwargs: Method-specific arguments

    Returns:
        (result, grad_fn): Integration result and gradient function

    The grad_fn takes a cotangent for y_final and returns gradient w.r.t. y0.
    """
    # Use JAX's remat for checkpointing
    @partial(jax.remat, prevent_cse=False)
    def integrate_fn(y0_):
        result = integrate(f, y0_, t_span, method, **kwargs)
        return result.y_final

    result = integrate(f, y0, t_span, method, **kwargs)

    # VJP function
    _, vjp_fn = jax.vjp(integrate_fn, y0)

    return result, vjp_fn


def sensitivity_analysis(
    f: Callable[[Array, Array, Array], Array],  # f(t, y, params) -> dy/dt
    y0: Array,
    params: Array,
    t_span: tuple[float, float],
    method: Method = "RK4",
    **kwargs,
) -> tuple[IntegrationResult, Array]:
    """Compute solution and sensitivity of final state to parameters.

    Uses forward-mode AD to compute dy_final/d_params.

    Args:
        f: Derivative function with explicit params: f(t, y, params) -> dy/dt
        y0: Initial state
        params: Parameter array
        t_span: Time interval
        method: Integration method
        **kwargs: Method-specific arguments

    Returns:
        (result, sensitivity): Integration result and Jacobian dy_final/d_params
    """
    def integrate_fn(p):
        f_with_params = lambda t, y: f(t, y, p)
        result = integrate(f_with_params, y0, t_span, method, **kwargs)
        return result.y_final

    # Compute Jacobian using forward-mode AD
    result = integrate(lambda t, y: f(t, y, params), y0, t_span, method, **kwargs)
    jacobian = jax.jacfwd(integrate_fn)(params)

    return result, jacobian
