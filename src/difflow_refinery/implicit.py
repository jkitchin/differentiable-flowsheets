"""Implicit-function derivatives at a converged loop, from one trace of its model.

A refinery loop (an air-demand controller, a recycle tear, a solvent loop) is
converged on stop-gradient inputs, and its solution is then handed to
:func:`solution`, which returns the model's outputs there with the derivative
the implicit function theorem gives. The model is traced ONCE for the value
and, when differentiated, once more as a linearisation; the loop itself is
never differentiated.

The older pattern, one Newton step ``x* - J^-1 g(x*)`` with ``J`` from a
separate ``jacfwd`` and the outputs from a further call, traced the model three
or four times, each trace repeating every solve nested inside it.

:func:`split_params` and :func:`merge_params` pull the arrays out of a nest of
frozen parameter dataclasses, so that a parameter held as a tracer (a bed
inlet temperature being differentiated, say) is passed in explicitly rather
than closed over, and the rest of the dataclass can key a ``jax.jit`` cache.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class _Slot:
    """Where :func:`split_params` took out an array."""

    index: int


def split_params(obj):
    """``(skeleton, arrays)``: ``obj`` with every JAX/NumPy array replaced by a
    :class:`_Slot`, and the arrays in order. Walks dataclasses, tuples,
    lists and dicts; Python scalars, strings and None stay in the skeleton."""
    arrays = []

    def go(o):
        if isinstance(o, (jax.Array, np.ndarray)):
            arrays.append(jnp.asarray(o))
            return _Slot(len(arrays) - 1)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.replace(o, **{f.name: go(getattr(o, f.name))
                                             for f in dataclasses.fields(o) if f.init})
        if isinstance(o, tuple):
            return tuple(go(v) for v in o)
        if isinstance(o, list):
            return [go(v) for v in o]
        if isinstance(o, dict):
            return {k: go(v) for k, v in o.items()}
        return o

    return go(obj), arrays


def merge_params(skeleton, arrays):
    """Inverse of :func:`split_params`."""

    def go(o):
        if isinstance(o, _Slot):
            return arrays[o.index]
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.replace(o, **{f.name: go(getattr(o, f.name))
                                             for f in dataclasses.fields(o) if f.init})
        if isinstance(o, tuple):
            return tuple(go(v) for v in o)
        if isinstance(o, list):
            return [go(v) for v in o]
        if isinstance(o, dict):
            return {k: go(v) for k, v in o.items()}
        return o

    return go(skeleton)


def hashable(*objs) -> bool:
    """True when every object can key a ``jax.jit`` static-argument cache."""
    try:
        for o in objs:
            hash(o)
    except TypeError:
        return False
    return True


def solution(G, x_star, theta, *, fixed_point: bool, jac: bool = False):
    """The outputs of ``G`` at a converged ``x_star``, with implicit derivatives.

    Args:
        G: ``G(x, theta) -> (g, aux)``. For ``fixed_point`` the solution has
            ``g(x*) = x*``; otherwise ``g(x*) = 0``. ``G`` must take every
            array it depends on through ``theta``: it must not close over a
            tracer.
        x_star: The converged solution, shape ``(n,)`` (its value is used
            as is; no gradient flows through it).
        theta: Pytree of float arrays: what the solution depends on.
        fixed_point: Fixed-point (``g = x``) or root (``g = 0``) form.
        jac: Also return ``dg/dx`` at the solution (no derivative of its own).

    Returns:
        ``(x, aux)`` or ``(x, aux, J)``. ``x`` equals ``x_star`` and its
        derivative is ``dx/dtheta = (I - J)^-1 dg/dtheta`` (fixed point) or
        ``-J^-1 dg/dtheta`` (root); ``aux``'s derivative is the total one.
    """
    n = x_star.shape[0]

    @jax.custom_jvp
    def sol(x, th):
        if jac:
            J, (_, aux) = jax.jacfwd(lambda z: (lambda g_aux: (g_aux[0], g_aux))(G(z, th)),
                                     has_aux=True)(x)
            return x, aux, J
        return x, G(x, th)[1]

    @sol.defjvp
    def sol_jvp(primals, tangents):
        x, th = primals
        _, dth = tangents
        (_, aux), lin = jax.linearize(G, x, th)
        zth = jax.tree_util.tree_map(jnp.zeros_like, th)
        J = jax.vmap(lambda e: lin(e, zth)[0], out_axes=1)(jnp.eye(n, dtype=x.dtype))
        g_th = lin(jnp.zeros_like(x), dth)[0]
        if fixed_point:
            dx = jnp.linalg.solve(jnp.eye(n, dtype=x.dtype) - J, g_th)
        else:
            dx = -jnp.linalg.solve(J, g_th)
        _, daux = lin(dx, dth)
        if jac:
            return (x, aux, J), (dx, daux, jnp.zeros_like(J))
        return (x, aux), (dx, daux)

    x_star = jax.lax.stop_gradient(jnp.asarray(x_star, dtype=float))
    return sol(x_star, theta)
