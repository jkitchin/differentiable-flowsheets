"""A damped Newton iteration in ``lax.while_loop``, and the implicit step after it."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def damped_newton(fun, z0, max_iter: int, tol: float, max_step: float, jac=None):
    """Solve ``fun(z) = 0`` from ``z0``; returns ``(z, norm, iterations)``.

    Each step is clipped so no entry moves more than ``max_step``, then
    halved until the merit ``|r|^2`` falls (or eight halvings). A step whose
    residual is not finite counts as an increase. ``jac`` defaults to
    ``jax.jacfwd(fun)``. Everything is concrete in the caller's sense: call
    it on stop-gradiented arguments.
    """
    jac = jax.jacfwd(fun) if jac is None else jac

    def merit(r):
        m = jnp.sum(r * r)
        return jnp.where(jnp.isfinite(m), m, jnp.inf)

    def body(c):
        z, r, k = c
        J = jac(z)
        dz = -jnp.linalg.solve(J, r)
        dz = jnp.where(jnp.isfinite(dz), dz, 0.0)
        big = jnp.max(jnp.abs(dz))
        dz = dz * jnp.minimum(1.0, max_step / jnp.maximum(big, 1e-300))
        m0 = merit(r)

        def bt_cond(s):
            a, zt, rt, i = s
            return (merit(rt) > (1.0 - 1e-4 * a) * m0) & (i < 8)

        def bt_body(s):
            a, _, _, i = s
            a = 0.5 * a
            zt = z + a * dz
            return a, zt, fun(zt), i + 1

        z1 = z + dz
        _, zt, rt, _ = jax.lax.while_loop(bt_cond, bt_body, (jnp.asarray(1.0), z1, fun(z1), 0))
        return zt, rt, k + 1

    def cond(c):
        _, r, k = c
        n = jnp.max(jnp.abs(r))
        return (k < max_iter) & ~(n <= tol)

    r0 = fun(z0)
    z, r, k = jax.lax.while_loop(cond, body, (z0, r0, 0))
    return z, jnp.max(jnp.abs(r)), k


def implicit_step(residual, z1, args, frozen):
    """The converged ``z1`` with the implicit derivative attached.

    ``residual(z, args)``; ``frozen`` is ``stop_gradient(args)``. The value
    is ``z1``; the derivative with respect to ``args`` is ``-J^-1 dr/dargs``.
    """
    z1 = jax.lax.stop_gradient(z1)
    J = jax.lax.stop_gradient(jax.jacfwd(residual)(z1, frozen))
    dz = jnp.linalg.solve(J, residual(z1, args))
    return z1 - (dz - jax.lax.stop_gradient(dz))
