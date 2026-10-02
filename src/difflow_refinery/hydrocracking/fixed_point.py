"""A fixed-point solve with an implicit (adjoint) reverse-mode gradient, for the UCO recycle.

The unconverted-oil recycle of a hydrocracker is a tear on a stream that
carries every UCO cut's molecules and attributes -- a couple of hundred
unknowns. Newton on it would need a Jacobian of that size through the reactor
integrations (one forward-mode tangent per unknown), which costs more than
the loop is worth. Instead:

* **Forward**: Anderson acceleration of the substitution ``z <- G(z)``
  (Walker & Ni 2011, the "type II" form): with the last ``m`` iterates ``z_i``,
  their images ``g_i = G(z_i)`` and scaled residuals ``f_i = (g_i - z_i)/max(s, |g_i|)``,
  the next iterate is ``sum alpha_i g_i`` with ``alpha`` minimising
  ``||sum alpha_i f_i||`` subject to ``sum alpha_i = 1`` (solved from
  ``(F F^T + lambda I) a = 1``, ``alpha = a / sum a``). Plain substitution on
  the default unit contracts at about 0.8 per pass; Anderson takes it to the
  tolerance in a few tens of passes fewer.
* **Gradient**: the implicit-function one, never the iterations. For ``z* =
  G(z*, theta)`` a cotangent ``v`` on ``z*`` pulls back to ``w^T dG/dtheta``
  with ``(I - (dG/dz)^T) w = v``, solved by GMRES on the scaled system
  (``y = s w``), each operator application one vector-Jacobian product of ``G``.
  That needs only reverse mode -- what the reactor's checkpointed diffrax
  adjoint supports.

References:
    Walker, H.F. and Ni, P., "Anderson acceleration for fixed-point
        iterations", SIAM J. Numer. Anal. 49(4), 1715-1735 (2011),
        doi:10.1137/10078356X (citation details recalled, unverified).
    Christianson, B., "Reverse accumulation and attractive fixed points",
        Optim. Methods Softw. 3(4), 311-326 (1994),
        doi:10.1080/10556789408805572 (unverified) -- the adjoint of a
        fixed-point iteration.
    Saad, Y. and Schultz, M.H., "GMRES: a generalized minimal residual
        algorithm for solving nonsymmetric linear systems", SIAM J. Sci. Stat.
        Comput. 7(3), 856-869 (1986), doi:10.1137/0907058 (unverified).
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
                G_iter: Callable | None = None, memory: int = 5, adjoint_tol: float = 1e-12,
                adjoint_restart: int = 40, adjoint_maxiter: int = 4) -> FixedPointSolution:
    """Solve ``z = G(z, args)`` (Anderson-accelerated substitution); reverse-mode gradient by the adjoint.

    Args:
        G: The map, a pure function of ``(z, args)`` (no traced closures).
        z0: Start (treated as a constant).
        args: Pytree of float arrays the solution depends on.
        scale: Per-entry floor scale of ``z`` (finite, positive): the convergence
            test is ``max |G(z) - z| / max(scale, |G(z)|) < tol`` (relative where an
            entry is larger than its scale -- the reactor integration's own
            tolerance sets a relative noise floor), and the adjoint is solved on
            the scaled system.
        G_iter: The map the forward iterations use (same values, e.g. built
            with a forward-mode diffrax adjoint); ``G`` is what the gradient follows.
        memory: Anderson depth ``m`` (1: plain substitution).
        adjoint_tol, adjoint_restart, adjoint_maxiter: GMRES controls of the adjoint.
    """
    Gi = G if G_iter is None else G_iter
    z0 = jax.lax.stop_gradient(jnp.asarray(z0, dtype=float))
    scale = jax.lax.stop_gradient(jnp.asarray(scale, dtype=float))
    n = z0.size
    m = int(memory)

    def solve(a):
        def cond(s):
            return (s[5] >= tol) & (s[4] < max_steps)

        def body(s):
            z, Gb, Fb, valid, it, _ = s
            g = Gi(z, a)
            f = (g - z) / jnp.maximum(scale, jnp.abs(g))
            err = jnp.max(jnp.abs(f))
            err = jnp.where(jnp.isfinite(err), err, jnp.inf)
            k = it % m
            Gb = Gb.at[k].set(g)
            Fb = Fb.at[k].set(f)
            valid = valid.at[k].set(True)
            M = Fb @ Fb.T
            lam = 1e-12 * jnp.max(jnp.diag(M)) + 1e-300
            M = jnp.where(valid[:, None] & valid[None, :], M, 0.0) + jnp.diag(jnp.where(valid, lam, 1.0))
            rhs = jnp.where(valid, 1.0, 0.0)
            aa = jnp.linalg.solve(M, rhs)
            alpha = aa / jnp.sum(aa)
            zn = alpha @ Gb
            zn = jnp.where(jnp.all(jnp.isfinite(zn)), zn, g)
            return zn, Gb, Fb, valid, it + 1, err

        init = (z0, jnp.zeros((m, n)), jnp.zeros((m, n)), jnp.zeros(m, dtype=bool), 0, jnp.asarray(jnp.inf))
        z, _, _, _, it, err = jax.lax.while_loop(cond, body, init)
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

        def A(y):
            return y - scale * pull(y / scale)[0]

        y, _ = jax.scipy.sparse.linalg.gmres(A, scale * v, tol=adjoint_tol, atol=0.0,
                                             restart=adjoint_restart, maxiter=adjoint_maxiter,
                                             solve_method="batched")
        return (pull(y / scale)[1],)

    fp.defvjp(fwd, bwd)
    value, err, it = fp(args)
    err, it = jax.lax.stop_gradient(err), jax.lax.stop_gradient(it)
    return FixedPointSolution(value=value, residual=err, steps=it, converged=err < tol)


__all__ = ["FixedPointSolution", "fixed_point"]
