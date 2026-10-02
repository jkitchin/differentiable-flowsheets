"""Newton with an implicit-function gradient, for equations that run a diffrax integration.

``optimistix``'s Newton forms its Jacobian in forward mode, and a ``diffrax``
solve differentiated with ``RecursiveCheckpointAdjoint`` (the reactor's
default) supports only reverse mode -- it is a ``custom_vjp``. So the
equations that wrap the reactor (the recycle tear, the quench balance, a
WABT or product-sulfur target) are solved here instead: damped Newton with
the Jacobian assembled row by row from vector-Jacobian products
(``jax.vjp`` once, then one pullback per unknown), inside
``lax.while_loop`` on stop-gradient inputs; then one Newton step with that
Jacobian frozen at the solution::

    x = x* - J(x*)^-1 f(x*, theta)

whose value is ``x*`` (to round-off) and whose derivative is exactly the
implicit-function one, ``dx/dtheta = -J^-1 df/dtheta`` -- the same device
:class:`~difflow_refinery.vacuum.column.StageColumn` uses. Reverse-mode
gradients of anything computed from ``x`` are therefore exact and never
differentiate the iterations. (Forward mode through it is unavailable
for the same reason the Newton loop needs it: the diffrax adjoint.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class NewtonSolution:
    """Attributes: value (the root, carrying implicit gradients), residual (max |f|), steps, converged."""

    value: Array
    residual: Array
    steps: Array
    converged: Array


jax.tree_util.register_dataclass(NewtonSolution, data_fields=["value", "residual", "steps", "converged"],
                                 meta_fields=[])


def _f_and_jac(F, x):
    r, pull = jax.vjp(F, x)
    eye = jnp.eye(r.size, dtype=r.dtype)
    J = jax.vmap(lambda e: pull(e)[0])(eye)
    return r, J


def newton_solve(f: Callable, x0: Array, args=None, tol: float = 1e-11, max_steps: int = 30,
                 max_step: float | None = None, jac: str = "rev",
                 f_iter: Callable | None = None) -> NewtonSolution:
    """Solve ``f(x, args) = 0`` for a vector ``x`` (see the module docstring).

    Args:
        f: Residual, ``(n,) -> (n,)``; should be well scaled (O(1) entries).
        x0: Start.
        args: Pytree the root depends on; gradients flow to it.
        tol: Stop when ``max |f| < tol``.
        max_steps: Iteration limit.
        max_step: If set, each step is shortened so that no entry of ``x``
            moves by more than this.
        jac: ``"rev"`` (Jacobian from VJPs, for reverse-only functions) or
            ``"fwd"`` (``jax.jacfwd``).
        f_iter: The residual the iterations use, if different from ``f``
            (same values; e.g. built with a forward-mode diffrax adjoint so
            ``jac="fwd"`` can differentiate it). ``f`` itself is used for the
            final implicit step, so the gradient follows ``f``.
    """
    x0 = jax.lax.stop_gradient(jnp.asarray(x0, dtype=float))
    a_s = jax.lax.stop_gradient(args)
    fi = f if f_iter is None else f_iter
    F = lambda x: fi(x, a_s)
    n = x0.size

    def cond(s):
        return ~s[4]

    def body(s):
        x, _, it, _, _ = s
        if jac == "fwd":
            r, J = F(x), jax.jacfwd(F)(x)
        else:
            r, J = _f_and_jac(F, x)
        rn = jnp.max(jnp.abs(r))
        done = (rn < tol) | (it >= max_steps) | ~jnp.isfinite(rn)
        dx = jnp.linalg.solve(J, -r)
        dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
        if max_step is not None:
            dx = dx * jnp.minimum(1.0, max_step / jnp.maximum(jnp.max(jnp.abs(dx)), 1e-300))
        xn = jnp.where(done, x, x + dx)
        return xn, J, it + jnp.where(done, 0, 1), rn, done

    x, J, it, rn, _ = jax.lax.while_loop(cond, body, (x0, jnp.eye(n), 0, jnp.asarray(jnp.inf), False))
    x, J = jax.lax.stop_gradient(x), jax.lax.stop_gradient(J)
    value = x - jnp.linalg.solve(J, f(x, args))
    return NewtonSolution(value=value, residual=rn, steps=it, converged=rn < tol)


def newton_scalar(f: Callable, x0, args=None, tol: float = 1e-11, max_steps: int = 30,
                  max_step: float | None = None, jac: str = "rev") -> NewtonSolution:
    """:func:`newton_solve` for a scalar unknown (``f(x, args) -> scalar``)."""
    sol = newton_solve(lambda v, a: jnp.reshape(f(v[0], a), (1,)), jnp.reshape(jnp.asarray(x0, dtype=float), (1,)),
                       args, tol=tol, max_steps=max_steps, max_step=max_step, jac=jac)
    return NewtonSolution(value=sol.value[0], residual=sol.residual, steps=sol.steps, converged=sol.converged)


__all__ = ["NewtonSolution", "newton_solve", "newton_scalar"]
