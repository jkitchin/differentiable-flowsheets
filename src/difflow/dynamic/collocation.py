"""Orthogonal collocation on finite elements (Radau IIA), as residuals.

A SPIKE helper (issue #440), deliberately not exported from
``difflow.dynamic`` yet. It discretises ``dy/dt = f(t, y)`` on ``[t0, t1]``
into algebraic equations so an ODE can sit inside one equation-oriented
system (an NLP or a square residual) with the rest of a flowsheet, instead
of being integrated by an adaptive solver inside a nonlinear solve.

The mesh is ``NE`` elements with edges ``edges`` (``NE + 1`` increasing
values, so a graded mesh is allowed: a stiff layer at the inlet wants small
elements there). On each element there are ``K`` collocation points, the
right-Radau points of ``[0, 1]`` (roots of ``P_K - P_{K-1}`` shifted to the
unit interval; the last one is exactly ``1``). Because the last point is the
element end, the state is continuous across elements by construction and
the unknowns are only the values ``Y[e, j]`` at the collocation points.
Radau IIA collocation is stiffly accurate and L-stable, order ``2K - 1`` at
the element ends, order ``K`` in between.

On element ``e`` of width ``h`` the polynomial through ``y_start`` (the
previous element's end value, or ``y0``) and the ``K`` unknowns satisfies::

    D @ [y_start; Y[e]]  -  h * f(t_ej, Y[e, j])  =  0        (K, ny)

with ``D`` the ``(K, K + 1)`` differentiation matrix of the Lagrange basis on
``[0, tau_1 .. tau_K]``, evaluated at the collocation points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "Collocation",
    "radau_nodes",
    "differentiation_matrix",
    "graded_edges",
    "collocation_residual",
    "radau_weights",
    "interpolate",
]


def radau_nodes(K: int) -> np.ndarray:
    """The ``K`` right-Radau points on ``(0, 1]`` (the last is exactly 1).

    Args:
        K: Number of points per element, ``K >= 1`` (``K = 1`` is implicit Euler).

    Returns:
        Increasing ``(K,)`` array.

    Example:
        >>> radau_nodes(2)  # doctest: +ELLIPSIS
        array([0.33333333, 1.        ])
    """
    if K < 1:
        raise ValueError("K must be at least 1")
    if K == 1:
        return np.array([1.0])
    from numpy.polynomial import legendre as L

    c = np.zeros(K + 1)
    c[K] = 1.0
    c[K - 1] = -1.0  # right Radau on [-1, 1]: roots of P_K - P_{K-1} (includes +1)
    r = np.sort(np.real(L.legroots(c)))
    r[-1] = 1.0
    return 0.5 * (r + 1.0)


def differentiation_matrix(tau: np.ndarray) -> np.ndarray:
    """``(K, K+1)`` matrix: derivative at each collocation point of the Lagrange
    polynomial through ``[0, tau]``, in terms of the ``K + 1`` nodal values."""
    nodes = np.concatenate([[0.0], np.asarray(tau, dtype=float)])
    n = nodes.size
    D = np.zeros((n - 1, n))
    for i in range(1, n):
        for j in range(n):
            if j == i:
                D[i - 1, j] = sum(1.0 / (nodes[i] - nodes[m]) for m in range(n) if m != i)
            else:
                num = np.prod([nodes[i] - nodes[m] for m in range(n) if m not in (i, j)])
                den = np.prod([nodes[j] - nodes[m] for m in range(n) if m != j])
                D[i - 1, j] = num / den
    return D


def radau_weights(tau: np.ndarray) -> np.ndarray:
    """Quadrature weights ``w`` on ``[0, 1]`` for the ``K`` points (exact to degree ``2K-2``)."""
    tau = np.asarray(tau, dtype=float)
    K = tau.size
    V = np.vander(tau, K, increasing=True).T  # V[p, j] = tau_j^p
    moments = 1.0 / (np.arange(K) + 1.0)
    return np.linalg.solve(V, moments)


def graded_edges(n_elements: int, grading: float = 1.0, t0: float = 0.0, t1: float = 1.0) -> np.ndarray:
    """Element edges on ``[t0, t1]``, each element ``grading`` times wider than the last.

    ``grading = 1`` is uniform; ``grading > 1`` refines toward ``t0`` (the
    inlet of a bed, where the fast relaxation is).
    """
    if grading == 1.0:
        s = np.linspace(0.0, 1.0, n_elements + 1)
    else:
        w = grading ** np.arange(n_elements)
        s = np.concatenate([[0.0], np.cumsum(w) / np.sum(w)])
    return t0 + (t1 - t0) * s


@dataclass(frozen=True)
class Collocation:
    """A finite-element Radau mesh.

    Attributes:
        K: Collocation points per element.
        edges: ``(NE + 1,)`` element edges.
    """

    K: int
    edges: tuple

    @classmethod
    def make(cls, n_elements: int, K: int = 3, grading: float = 1.0, t0: float = 0.0,
             t1: float = 1.0) -> "Collocation":
        return cls(K, tuple(float(e) for e in graded_edges(n_elements, grading, t0, t1)))

    @property
    def n_elements(self) -> int:
        return len(self.edges) - 1

    @property
    def n_points(self) -> int:
        return self.n_elements * self.K

    @property
    def tau(self) -> np.ndarray:
        return radau_nodes(self.K)

    @property
    def t(self) -> np.ndarray:
        """``(NE, K)`` collocation times."""
        e = np.asarray(self.edges)
        return e[:-1, None] + np.diff(e)[:, None] * self.tau[None, :]

    def residual(self, f: Callable, y0: Array, Y: Array, args=()) -> Array:
        return collocation_residual(f, y0, Y, self, args)


def collocation_residual(f: Callable, y0: Array, Y: Array, mesh: Collocation, args=()) -> Array:
    """Residual of the collocation equations, shape ``(NE, K, ny)``.

    Args:
        f: ``f(t, y, *args) -> dy/dt`` of shape ``(ny,)``; must be vmappable.
        y0: ``(ny,)`` initial state.
        Y: ``(NE, K, ny)`` unknown values at the collocation points.
        mesh: The :class:`Collocation` mesh.
        args: Extra arguments passed to ``f``.

    Returns:
        ``(NE, K, ny)``: zero when ``Y`` is the collocation solution.
    """
    D = jnp.asarray(differentiation_matrix(mesh.tau))
    e = np.asarray(mesh.edges)
    h = jnp.asarray(np.diff(e))                       # (NE,)
    t = jnp.asarray(mesh.t)                            # (NE, K)
    y0 = jnp.asarray(y0)
    starts = jnp.concatenate([y0[None, :], Y[:-1, -1, :]], axis=0)   # (NE, ny)
    Z = jnp.concatenate([starts[:, None, :], Y], axis=1)             # (NE, K+1, ny)
    dZ = jnp.einsum("jl,eln->ejn", D, Z)
    F = jax.vmap(jax.vmap(lambda tt, yy: f(tt, yy, *args)))(t, Y)
    return dZ - h[:, None, None] * F


def interpolate(y0: Array, Y: Array, mesh: Collocation, s: Array) -> Array:
    """Value of the collocation polynomial at times ``s`` (each in ``[t0, t1]``).

    ``s`` is a Python/NumPy array (the element of each ``s`` is chosen on the host).
    """
    e = np.asarray(mesh.edges)
    tau0 = np.concatenate([[0.0], mesh.tau])
    y0 = jnp.asarray(y0)
    starts = jnp.concatenate([y0[None, :], Y[:-1, -1, :]], axis=0)
    Z = jnp.concatenate([starts[:, None, :], Y], axis=1)
    out = []
    for si in np.asarray(s, dtype=float):
        k = int(min(max(np.searchsorted(e, si, side="right") - 1, 0), mesh.n_elements - 1))
        u = (si - e[k]) / (e[k + 1] - e[k])
        lag = np.array([np.prod([(u - tau0[m]) / (tau0[j] - tau0[m]) for m in range(tau0.size) if m != j])
                        for j in range(tau0.size)])
        out.append(jnp.asarray(lag) @ Z[k])
    return jnp.stack(out)
