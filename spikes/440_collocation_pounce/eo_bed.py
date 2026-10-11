"""Spike (#440): one adiabatic reforming bed discretised by Radau collocation.

Same physics as ``difflow_refinery.reforming.reactor.integrate_bed`` (same
rate model, same enthalpy basis, normalised bed coordinate w in [0, 1],
state x = F / F_in_total). The temperature is an ALGEBRAIC unknown at every
collocation point, tied to the state by the conserved enthalpy, instead of a
Newton solve inside the right-hand side.

Unknowns per point: 29 species fractions, coke, T  (31).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from difflow.dynamic.collocation import Collocation, collocation_residual
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.reactor import PA_PER_BAR, _rate_model

jax.config.update("jax_enable_x64", True)

NS = sp.N_SPECIES
NV = NS + 2          # x (29), coke, T
H_SCALE = th.R * 773.15   # J/mol: scale of the enthalpy residual


def bed_residual(Y, F_in, T_in, P, W, kin, mesh: Collocation, iso_fraction=0.5):
    """Residual of one bed, ``(NE, K, 31)``; ``Y`` is ``(NE, K, 31)`` = [x, coke, T]."""
    rm = _rate_model(iso_fraction)
    F0 = jnp.sum(F_in)
    P_bar = P / PA_PER_BAR
    h_in = th.total_enthalpy(F_in / F0, T_in, P)

    def f(t, z):
        x, T = z[:NS], z[NS + 1]
        y = x / jnp.sum(x)
        p = y * P_bar
        r = rm.rates(kin, p, T, P_bar)
        return jnp.concatenate([(W / F0) * (r @ rm.nu_j), jnp.atleast_1d(W * rm.coke_rate(kin, p, T)),
                                jnp.zeros(1)])

    y0 = jnp.concatenate([F_in / F0, jnp.zeros(2)])
    r = collocation_residual(f, y0, Y, mesh)[..., : NS + 1]   # T row is algebraic
    H_res = (jax.vmap(jax.vmap(lambda z: th.total_enthalpy(z[:NS], z[NS + 1], P)))(Y) - h_in) / H_SCALE
    return jnp.concatenate([r, H_res[..., None]], axis=-1)


def outlet(Y, F_in):
    F0 = jnp.sum(F_in)
    z = Y[-1, -1]
    return F0 * z[:NS], z[NS + 1], z[NS]


def initial_guess(F_in, T_in, mesh):
    z = jnp.concatenate([F_in / jnp.sum(F_in), jnp.zeros(1), jnp.asarray(T_in)[None]])
    return jnp.broadcast_to(z, (mesh.n_elements, mesh.K, NV)) * 1.0


def make_newton(mesh, kin):
    """A jitted damped Newton for one mesh; args ``(Y0, F, T, P, W)`` so it compiles once per mesh."""
    shape = (mesh.n_elements, mesh.K, NV)

    def res(y, F, T, P, W):
        return bed_residual(y.reshape(shape), F, T, P, W, kin, mesh).ravel()

    flat = jax.jit(res)
    jac = jax.jit(jax.jacfwd(res))

    def solve(Y0, F, T, P, W, tol=1e-10, max_iter=60):
        y = Y0.ravel()
        r = flat(y, F, T, P, W)
        for it in range(max_iter):
            nr = float(jnp.max(jnp.abs(r)))
            if nr < tol:
                return y.reshape(shape), it, nr
            dy = jnp.linalg.solve(jac(y, F, T, P, W), -r)
            a = 1.0
            while a > 1e-6:
                yn = y + a * dy
                rn = flat(yn, F, T, P, W)
                if jnp.all(jnp.isfinite(rn)) and float(jnp.max(jnp.abs(rn))) < (1 - 1e-4 * a) * nr:
                    break
                a *= 0.5
            y, r = yn, rn
        return y.reshape(shape), max_iter, float(jnp.max(jnp.abs(r)))

    solve.flat, solve.jac = flat, jac
    return solve
