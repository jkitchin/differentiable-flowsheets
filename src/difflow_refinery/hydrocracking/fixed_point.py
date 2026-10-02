"""A fixed-point solve with an implicit (adjoint) reverse-mode gradient, for the UCO recycle.

The unconverted-oil recycle of a hydrocracker is a tear on a stream that
carries every UCO cut's molecules and attributes -- a couple of hundred
unknowns. Newton on it would need a Jacobian of that size through the
reactor integrations (one forward-mode tangent per unknown), which costs
more than the loop is worth: the UCO loop is a contraction (its gain is the
recycle fraction times the share of recycled oil that survives a pass,
typically 0.2-0.6), so plain successive substitution converges in tens of
passes.

The derivative is the implicit-function one, never the iterations. For
``z* = G(z*, theta)``, a cotangent ``v`` on ``z*`` pulls back to
``w^T dG/dtheta`` with ``w`` the solution of the adjoint fixed point ``w =
v + (dG/dz)^T w`` -- solved by the same substitution (it converges at the
same rate as the forward loop), each pass one vector-Jacobian product of
``G``. That needs only reverse mode, which is what the reactor's
checkpointed diffrax adjoint supports. (The classic construction; see e.g.
Christianson, B., "Reverse accumulation and attractive fixed points",
Optim. Methods Softw. 3(4), 311-326 (1994), doi:10.1080/10556789408805572
-- citation details unverified.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FixedPointSolution:
    """Attributes: value (carries the implicit gradient), residual (max scaled step), steps, converged."""

    value: Array
    residual: Array
    steps: Array
    converged: Array


jax.tree_util.register_dataclass(FixedPointSolution, data_fields=["value", "residual", "steps", "converged"],
                                 meta_fields=[])


def fixed_point(G: Callable, z0: Array, args, scale: Array, tol: float = 1e-12, max_steps: int = 80,
                G_iter: Callable | None = None, adjoint_tol: float = 1e-12, adjoint_max_steps: int = 200,
                damping: float = 1.0) -> FixedPointSolution:
    """Solve ``z = G(z, args)`` by successive substitution; reverse-mode gradient by the adjoint fixed point.

    Args:
        G: The map, a pure function of ``(z, args)`` (no traced closures).
        z0: Start (treated as a constant).
        args: Pytree of float arrays the solution depends on.
        scale: Per-entry scale of ``z`` for the convergence test ``max |dz|/scale < tol``.
        G_iter: The map the forward iterations use (same values, e.g. built with
            a forward-mode diffrax adjoint); ``G`` is what the gradient follows.
        adjoint_tol, adjoint_max_steps: Controls of the adjoint substitution.
        damping: Relaxation factor of the forward substitution (1: plain).
    """
    Gi = G if G_iter is None else G_iter
    z0 = jax.lax.stop_gradient(jnp.asarray(z0, dtype=float))
    scale = jax.lax.stop_gradient(scale)

    def solve(a):
        def cond(s):
            return (s[2] >= tol) & (s[1] < max_steps)

        def body(s):
            z, it, _ = s
            zn = Gi(z, a)
            err = jnp.max(jnp.abs(zn - z) / scale)
            err = jnp.where(jnp.isfinite(err), err, jnp.inf)
            return z + damping * (zn - z), it + 1, err

        z, it, err = jax.lax.while_loop(cond, body, (z0, 0, jnp.asarray(jnp.inf)))
        return z, it, err

    @jax.custom_vjp
    def fp(a):
        z, it, err = solve(a)
        return z, err, jnp.asarray(it, dtype=float)

    def fwd(a):
        z, it, err = solve(a)
        return (z, err, jnp.asarray(it, dtype=float)), (z, a)

    def bwd(res, cts):
        z, a = res
        v = cts[0]
        _, pull = jax.vjp(G, z, a)

        def cond(s):
            return (s[2] >= adjoint_tol) & (s[1] < adjoint_max_steps)

        def body(s):
            w, it, _ = s
            wn = v + pull(w)[0]
            err = jnp.max(jnp.abs(wn - w)) / jnp.maximum(jnp.max(jnp.abs(wn)), 1e-300)
            return wn, it + 1, err

        w, _, _ = jax.lax.while_loop(cond, body, (v, 0, jnp.asarray(jnp.inf)))
        return (pull(w)[1],)

    fp.defvjp(fwd, bwd)
    value, err, it = fp(args)
    err, it = jax.lax.stop_gradient(err), jax.lax.stop_gradient(it)
    return FixedPointSolution(value=value, residual=err, steps=it, converged=err < tol)


__all__ = ["FixedPointSolution", "fixed_point"]
