"""Differentiable solvers for difflow.

This module provides solvers that work with JAX automatic differentiation:
- Fixed-point iteration with implicit differentiation
- Newton-Raphson solver
- Rachford-Rice solver for flash calculations

All solvers use implicit differentiation to compute gradients through
the converged solution, which is more accurate and efficient than
differentiating through the iteration steps.
"""

from typing import Callable, Any
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array, lax


def fixed_point_solve(
    f: Callable[[Array, Any], Array],
    x0: Array,
    args: Any = (),
    tol: float = 1e-8,
    max_iter: int = 100,
    damping: float = 1.0,
) -> Array:
    """Solve x = f(x, args) using fixed-point iteration.

    Uses unrolled iteration which is automatically differentiable through JAX.
    For better gradient accuracy with many iterations, consider using
    jax.checkpoint to reduce memory usage.

    Args:
        f: Function such that the solution satisfies x* = f(x*, args)
        x0: Initial guess
        args: Additional arguments passed to f (can be pytrees)
        tol: Convergence tolerance (not used, kept for API compatibility)
        max_iter: Number of iterations
        damping: Damping factor (0 < damping <= 1). x_new = (1-damping)*x + damping*f(x)

    Returns:
        Solution after max_iter iterations
    """
    def step(x, _):
        x_new = f(x, args)
        x_next = (1 - damping) * x + damping * x_new
        return x_next, None

    x_final, _ = lax.scan(step, x0, None, length=max_iter)
    return x_final


def newton_solve(
    f: Callable[[Array, Any], Array],
    x0: Array,
    args: Any = (),
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Array:
    """Solve f(x, args) = 0 using Newton-Raphson with implicit differentiation.

    Args:
        f: Function to find roots of, f(x, args) = 0
        x0: Initial guess
        args: Additional arguments passed to f
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Solution x* such that f(x*, args) ≈ 0
    """
    return _newton_impl(f, x0, args, tol, max_iter)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _newton_impl(
    f: Callable[[Array, Any], Array],
    x0: Array,
    args: Any,
    tol: float,
    max_iter: int,
) -> Array:
    """Newton-Raphson implementation."""

    def cond_fn(state):
        x, err, i = state
        return (err > tol) & (i < max_iter)

    def body_fn(state):
        x, _, i = state
        fx = f(x, args)

        # Compute Jacobian
        J = jax.jacfwd(lambda z: f(z, args))(x)

        # Newton step: solve J @ dx = -f
        # For scalar or small systems, direct solve is fine
        if x.ndim == 0:
            dx = -fx / J
        else:
            dx = jnp.linalg.solve(J, -fx)

        x_new = x + dx
        err = jnp.max(jnp.abs(fx))
        return x_new, err, i + 1

    x_final, _, _ = lax.while_loop(
        cond_fn,
        body_fn,
        (x0, jnp.inf, 0),
    )

    return x_final


def _newton_fwd(f, tol, max_iter, x0, args):
    x_star = _newton_impl(f, x0, args, tol, max_iter)
    return x_star, (x_star, args)


def _newton_bwd(f, tol, max_iter, res, g):
    """Backward pass for Newton solver using implicit function theorem.

    At f(x*, args) = 0, by implicit function theorem:
        df/dx @ dx* + df/dargs @ dargs = 0
        dx*/dargs = -(df/dx)^{-1} @ df/dargs

    For VJP:
        dL/dargs = -(df/dargs)^T @ (df/dx)^{-T} @ dL/dx*
    """
    x_star, args = res

    # Compute Jacobian at solution
    J = jax.jacfwd(lambda x: f(x, args))(x_star)

    # Solve J^T @ u = g for u
    if x_star.ndim == 0:
        u = g / J
    else:
        u = jnp.linalg.solve(J.T, g)

    # Compute dL/dargs = -(df/dargs)^T @ u
    _, vjp_fn = jax.vjp(lambda a: f(x_star, a), args)
    d_args = vjp_fn(-u)[0]

    d_x0 = jnp.zeros_like(x_star)
    return d_x0, d_args


_newton_impl.defvjp(_newton_fwd, _newton_bwd)


def rachford_rice(
    z: Array,
    K: Array,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Array:
    """Solve Rachford-Rice equation for vapor fraction.

    The Rachford-Rice equation is:
        sum_i z_i * (K_i - 1) / (1 + V * (K_i - 1)) = 0

    where:
        z_i = feed mole fraction of component i
        K_i = equilibrium ratio (y_i/x_i)
        V = vapor fraction (moles vapor / total moles)

    Args:
        z: Feed mole fractions (array)
        K: K-values for each component (array)
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Vapor fraction V in [0, 1]
    """
    # Rachford-Rice function and its derivative
    def rr_func(V, K_):
        return jnp.sum(z * (K_ - 1) / (1 + V * (K_ - 1)))

    def rr_deriv(V, K_):
        return -jnp.sum(z * (K_ - 1)**2 / (1 + V * (K_ - 1))**2)

    # Simple Newton iteration for scalar problem
    def newton_step(state, _):
        V, K_ = state
        f = rr_func(V, K_)
        df = rr_deriv(V, K_)
        # Damped Newton step with bounds
        dV = -f / df
        V_new = V + 0.5 * dV  # Damping
        V_new = jnp.clip(V_new, 0.001, 0.999)
        return (V_new, K_), None

    # Initial guess
    V0 = jnp.array(0.5)

    # Run fixed iterations (simpler than while loop for tracing)
    (V_solution, _), _ = lax.scan(newton_step, (V0, K), None, length=max_iter)

    # Clip to physical bounds
    return jnp.clip(V_solution, 0.0, 1.0)


def rachford_rice_compositions(
    z: Array,
    K: Array,
    V: Array,
) -> tuple[Array, Array]:
    """Calculate liquid and vapor compositions from Rachford-Rice solution.

    Args:
        z: Feed mole fractions
        K: K-values
        V: Vapor fraction from rachford_rice()

    Returns:
        (x, y): Liquid and vapor mole fractions
    """
    x = z / (1 + V * (K - 1))
    y = K * x
    return x, y
