"""Steady-state gain matrices from AD.

``G = dy/du`` and ``G_d = dy/dd`` are one :func:`jax.jacobian` call each on a
converged flowsheet.  That is the whole reason this module can exist inside a
design loop: the flowsheet's flash, recycle and unit solves are already
implicitly differentiated, so the reduced input-output sensitivity comes back
exact and at a cost independent of the number of inputs, where a
sequential-modular simulator would need ``2n`` perturbed re-solves and would
return a matrix contaminated by its own convergence tolerance.

The AD mode is chosen by shape via
:func:`difflow.planning.linearize.choose_ad_mode` — reverse when there are
fewer outputs than inputs, forward otherwise — the same rule the planning
module uses, and for the same reason.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.operability import Scaling, gain_matrix
    >>> def plant(u):
    ...     return jnp.array([2.0 * u[0] + u[1], u[0] + 3.0 * u[1]])
    >>> gain_matrix(plant, jnp.array([1.0, 1.0]))
    Array([[2., 1.],
           [1., 3.]], dtype=float64)
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array

from difflow.planning.linearize import jacobian_fn
from difflow.operability.scaling import Scaling

__all__ = ["gain_matrix", "disturbance_gain"]


def _as_vec(x: Any) -> Array:
    return jnp.atleast_1d(jnp.asarray(x, dtype=float))


def _as_mat(x: Any) -> Array:
    return jnp.atleast_2d(jnp.asarray(x, dtype=float))


def _jacobian(f: Callable[[Array], Array], x0: Array, mode: str) -> Array:
    """Jacobian of ``f`` at ``x0``, shaped ``(n_y, n_x)``."""
    y0 = _as_vec(f(x0))
    jac, _ = jacobian_fn(f, n_u=int(x0.shape[0]), n_y=int(y0.shape[0]),
                         mode=mode)
    J = jnp.asarray(jac(x0), dtype=float)
    return J.reshape(int(y0.shape[0]), int(x0.shape[0]))


def gain_matrix(model: Callable[..., Array] | Array,
                u0: Any = None, d0: Any = None, *,
                scaling: Scaling | None = None,
                mode: str = "auto") -> Array:
    """Steady-state gain matrix ``G = dy/du``.

    Args:
        model: Either a callable — ``fn(u) -> y``, or ``fn(u, d) -> y`` when
            ``d0`` is given — or an already-computed gain matrix, in which
            case only ``scaling`` is applied.  The callable must be pure and
            JAX-traceable; a difflow flowsheet with inner solves is.
        u0: Operating point for the manipulated variables.  Required when
            ``model`` is a callable.
        d0: Operating point for the disturbances.  When given, ``model`` is
            called as ``fn(u, d0)`` and the differentiation is still with
            respect to ``u`` only.
        scaling: :class:`~difflow.operability.scaling.Scaling` to apply.
            When ``None`` the raw gain in engineering units is returned; that
            is a perfectly good Jacobian but *not* something to take singular
            values of.
        mode: AD mode, ``"auto"``, ``"rev"`` or ``"fwd"``.

    Returns:
        Array of shape ``(n_y, n_u)``.  Scaled when ``scaling`` is given.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import Scaling, gain_matrix
        >>> plant = lambda u: jnp.array([u[0] - 0.5 * u[1]])
        >>> gain_matrix(plant, jnp.array([1.0, 1.0]),
        ...             scaling=Scaling(u_span=[2.0, 4.0], y_span=[1.0]))
        Array([[ 2., -2.]], dtype=float64)
    """
    if callable(model):
        if u0 is None:
            raise ValueError("gain_matrix needs u0 when model is a callable")
        u0 = _as_vec(u0)
        if d0 is None:
            f = lambda u: _as_vec(model(u))            # noqa: E731
        else:
            d_fixed = _as_vec(d0)
            f = lambda u: _as_vec(model(u, d_fixed))   # noqa: E731
        G = _jacobian(f, u0, mode)
    else:
        G = _as_mat(model)
    return G if scaling is None else scaling.scale_gain(G)


def disturbance_gain(model: Callable[..., Array] | Array,
                     u0: Any = None, d0: Any = None, *,
                     scaling: Scaling | None = None,
                     mode: str = "auto") -> Array:
    """Disturbance gain matrix ``G_d = dy/dd``.

    Scaled, this is the most directly useful matrix in the package: entry
    ``[i, k]`` is how many *allowable control errors* of output ``i`` a
    full-size excursion of disturbance ``k`` produces.  An entry with
    magnitude below 1 is a disturbance the plant absorbs on its own; above 1
    is one control has to reject, and the rest of the screen asks whether the
    available inputs can.

    Args:
        model: Either a callable ``fn(u, d) -> y`` (differentiated with
            respect to ``d`` at ``(u0, d0)``) or an already-computed
            ``dy/dd`` matrix, in which case only ``scaling`` is applied.
        u0: Manipulated-variable operating point, held fixed.
        d0: Disturbance operating point, the point of differentiation.
        scaling: :class:`~difflow.operability.scaling.Scaling` carrying a
            ``d_span``.  Without it the returned matrix is in engineering
            units and cannot be compared against 1.
        mode: AD mode, ``"auto"``, ``"rev"`` or ``"fwd"``.

    Returns:
        Array of shape ``(n_y, n_d)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import Scaling, disturbance_gain
        >>> plant = lambda u, d: jnp.array([u[0] + 3.0 * d[0]])
        >>> sc = Scaling(u_span=[1.0], y_span=[0.5], d_span=[2.0])
        >>> disturbance_gain(plant, jnp.array([1.0]), jnp.array([0.0]),
        ...                  scaling=sc)
        Array([[12.]], dtype=float64)
    """
    if callable(model):
        if u0 is None or d0 is None:
            raise ValueError(
                "disturbance_gain needs both u0 and d0 when model is a "
                "callable — dy/dd is evaluated at an operating point")
        u_fixed = _as_vec(u0)
        d0 = _as_vec(d0)
        f = lambda d: _as_vec(model(u_fixed, d))       # noqa: E731
        Gd = _jacobian(f, d0, mode)
    else:
        Gd = _as_mat(model)
    return Gd if scaling is None else scaling.scale_disturbance(Gd)
