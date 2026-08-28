"""Piecewise-linear blocks: where batching earns its keep.

A single delta vector is valid only inside a trust region.  When a block's
response to one lever is strongly curved over the whole operating range — an
allocation that saturates, a recovery that runs into a pinch — a piecewise
model captures the whole range at once and turns the plan into a MILP instead
of a sequence of trust-region LPs.

Building one is a batching problem, not a new modelling problem: ``vmap`` the
block over every breakpoint in a single call, and ``vmap`` its Jacobian too.
That is one dispatch for the whole curve.

Formulation
-----------
For the distinguished variable ``u_j`` with breakpoints ``g_0 < ... < g_K``::

    u_j  = sum_k lambda_k g_k
    y    = sum_k lambda_k y(g_k)  +  Jbar . (u_other - c_other)
    sum_k lambda_k = 1,  lambda >= 0,  at most two adjacent lambda nonzero

The adjacency is the SOS2 condition.  It is emitted natively for Pyomo and as
the standard binary formulation for SciPy, so the same model solves either
way.

Scope
-----
The response to ``u_j`` is piecewise-linear and *exact* at the breakpoints;
the response to the other inputs uses a single Jacobian ``Jbar`` taken at the
centre.  That is exact when the block is separable in ``u_j`` and a
first-order approximation otherwise — an approximation of the *cross* terms
only.  The alternative would multiply ``lambda_k`` by ``u_m``, which is
bilinear, and bilinear terms are nonconvex no matter how good the unit model
is.  That boundary is the same one that makes pooling and blending out of
scope for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.block import Block
from difflow.planning.linearize import jacobian_fn


@dataclass
class PiecewiseSpec:
    """Request a piecewise-linear model of one block variable.

    Attributes:
        block: Block name.
        variable: The distinguished input, by bare or qualified name.
        n_points: Number of breakpoints, spread over the variable's bounds.
        breakpoints: Explicit breakpoints, overriding ``n_points``.  Must be
            strictly increasing.
    """

    block: str
    variable: str
    n_points: int = 9
    breakpoints: Sequence[float] | None = None

    def __post_init__(self):
        if self.breakpoints is not None:
            g = np.asarray(self.breakpoints, dtype=float)
            if g.ndim != 1 or g.size < 2:
                raise ValueError("breakpoints must be a 1-D array of length >= 2")
            if np.any(np.diff(g) <= 0):
                raise ValueError("breakpoints must be strictly increasing")
            self.breakpoints = g
        elif self.n_points < 2:
            raise ValueError("n_points must be at least 2")


@dataclass
class PiecewiseData:
    """A sampled piecewise-linear model of one block.

    Attributes:
        block: Block name.
        variable: Bare name of the distinguished input.
        index: Its position in ``block.u_names``.
        breakpoints: Grid values, shape ``(K,)``.
        y: Block outputs at each breakpoint, shape ``(K, n_y)``.
        J: Block Jacobians at each breakpoint, shape ``(K, n_y, n_u)``.
        center: The input vector the off-grid variables were held at.
        cross_jacobian: ``Jbar``, shape ``(n_y, n_u)``, used for the inputs
            other than the distinguished one.
        n_evals: Model evaluations charged — one batched ``vmap`` call, but
            ``K`` evaluations' worth of work.
    """

    block: str
    variable: str
    index: int
    breakpoints: np.ndarray
    y: np.ndarray
    J: np.ndarray
    center: np.ndarray
    cross_jacobian: np.ndarray
    n_evals: int = 0

    @property
    def n_points(self) -> int:
        return int(self.breakpoints.shape[0])

    def predict(self, u: Array) -> np.ndarray:
        """Evaluate the piecewise model at ``u`` (linear interpolation)."""
        u = np.asarray(u, dtype=float)
        g = self.breakpoints
        xj = float(np.clip(u[self.index], g[0], g[-1]))
        k = int(np.clip(np.searchsorted(g, xj) - 1, 0, len(g) - 2))
        w = (xj - g[k]) / (g[k + 1] - g[k])
        y = (1 - w) * self.y[k] + w * self.y[k + 1]
        du = u - self.center
        du[self.index] = 0.0
        return y + self.cross_jacobian @ du

    def __repr__(self) -> str:
        return (f"PiecewiseData(block={self.block!r}, "
                f"variable={self.variable!r}, n_points={self.n_points})")


def sample_piecewise(block: Block, spec: PiecewiseSpec,
                     center: Array | None = None,
                     theta: Mapping[str, Any] | None = None,
                     cross_jacobian: str = "center") -> PiecewiseData:
    """Sample a block across a grid in one variable, in one batched call.

    Args:
        block: The block to sample.
        spec: Which variable to sample and how finely.
        center: Values held for the other inputs.  Defaults to ``block.u0``.
        theta: Parameter override.
        cross_jacobian: ``"center"`` uses the Jacobian at ``center`` for the
            other inputs; ``"mean"`` averages it over the grid.

    Returns:
        A :class:`PiecewiseData`.

    Example:
        >>> data = sample_piecewise(blk, PiecewiseSpec("sep", "recovery", 11))
        >>> data.y.shape
        (11, 5)
    """
    j = block.u_index(spec.variable)
    center = (block.u0 if center is None
              else jnp.atleast_1d(jnp.asarray(center, dtype=float)))

    if spec.breakpoints is not None:
        grid = jnp.asarray(spec.breakpoints, dtype=float)
    else:
        lo, hi = float(block.lb[j]), float(block.ub[j])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"variable {spec.variable!r} of block {block.name!r} is "
                "unbounded; give explicit breakpoints for a piecewise model")
        grid = jnp.linspace(lo, hi, spec.n_points)

    U = jnp.broadcast_to(center, (grid.shape[0], block.n_u)).at[:, j].set(grid)

    def f(u):
        return block.evaluate(u, theta)

    jac, _ = jacobian_fn(f, block.n_u, block.n_y, block.ad_mode)
    # One batched call for the values and one for the Jacobians: the whole
    # curve costs two dispatches, not 2K.
    Y = jax.vmap(f)(U)
    JJ = jax.vmap(jac)(U)

    if cross_jacobian == "center":
        Jbar = np.asarray(jac(center))
    elif cross_jacobian == "mean":
        Jbar = np.asarray(jnp.mean(JJ, axis=0))
    else:
        raise ValueError(
            f"cross_jacobian must be 'center' or 'mean', got "
            f"{cross_jacobian!r}")

    return PiecewiseData(
        block=block.name, variable=block.u_names[j], index=j,
        breakpoints=np.asarray(grid), y=np.asarray(Y), J=np.asarray(JJ),
        center=np.asarray(center), cross_jacobian=Jbar,
        n_evals=int(grid.shape[0]))


def sos2_rows(n_points: int) -> list[tuple[list[int], float]]:
    """Adjacency pattern of the SOS2 binary formulation.

    Returns the ``(lambda index, interval indices)`` pairs that the standard
    convex-combination formulation constrains; used by the LP assembler.
    """
    rows = []
    for k in range(n_points):
        intervals = [i for i in (k - 1, k) if 0 <= i <= n_points - 2]
        rows.append((intervals, 1.0))
    return rows
