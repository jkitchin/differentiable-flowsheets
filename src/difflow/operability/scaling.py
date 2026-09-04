"""Scaling: the part of a controllability screen users get wrong.

Every steady-state operability measure in this package except the RGA is a
statement about *magnitudes*, and a magnitude in mixed engineering units is
not a magnitude at all.  A gain of ``1e5 K/(mol/s)`` and a gain of
``0.02 mol/mol per fraction`` cannot be compared, ranked, or given to an SVD,
yet an unscaled ``jax.jacobian`` hands you exactly that matrix and the SVD
will happily return a number for it.

The convention here is Skogestad and Postlethwaite's (*Multivariable Feedback
Control*, 2nd ed., section 1.4 and chapter 6).  Three spans are declared by
the engineer, not inferred:

``u_span``
    The largest change in each manipulated variable that is actually
    available — valve fully closed to fully open, the usable turndown of a
    duty, the width of a setpoint's allowed range.

``y_span``
    The largest *acceptable control error* in each controlled variable.  Not
    its operating value, and not its measurement noise: the deviation at
    which someone would say the loop had failed.

``d_span``
    The largest expected excursion of each disturbance.

With ``Du = diag(u_span)``, ``De = diag(y_span)`` and ``Dd = diag(d_span)``,

.. math::

    \\tilde G = D_e^{-1} G D_u, \\qquad \\tilde G_d = D_e^{-1} G_d D_d

and every entry of both is dimensionless with the *same* meaning: "how many
allowable control errors of output *i* does a full move of input (or
disturbance) *j* produce".  That is what makes the number 1 the threshold that
all the rules of thumb are stated against — ``sigma_min(G~) > 1`` says the
inputs can cover the required output range in the plant's worst direction,
and ``|Gd~| > 1`` says a disturbance drives an output past what is acceptable
and must therefore be rejected by control rather than tolerated.

Example:
    >>> import jax.numpy as jnp
    >>> sc = Scaling(u_span=[10.0, 0.2], y_span=[2.0, 0.01])
    >>> sc.scale_gain(jnp.array([[0.5, 3.0], [0.001, 0.02]]))
    Array([[2.5, 0.3],
           [1. , 0.4]], dtype=float64)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

__all__ = ["OperabilityWarning", "Scaling"]


class OperabilityWarning(UserWarning):
    """An operability metric was requested in a way that makes it misleading.

    Almost always this means an unscaled gain matrix: the singular values,
    condition number and disturbance gains of a matrix in mixed engineering
    units measure the unit system, not the plant.  See :class:`Scaling`.
    """


def _is_concrete(x: Any) -> bool:
    """True when ``x`` holds actual numbers rather than a JAX tracer."""
    return not isinstance(x, jax.core.Tracer)


def _as_span(values: Sequence[float] | Array | float | None, what: str
             ) -> Array | None:
    """Coerce a span argument to a 1-D float array, validating when concrete."""
    if values is None:
        return None
    arr = jnp.atleast_1d(jnp.asarray(values, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{what} must be 1-D, got shape {tuple(arr.shape)}")
    if _is_concrete(arr):
        a = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{what} must be finite, got {a}")
        if np.any(a <= 0):
            raise ValueError(
                f"{what} must be strictly positive — it divides or multiplies "
                f"the gain matrix. Got {a}. A span of zero says the variable "
                "may not move at all, which means it is not a variable.")
    return arr


@dataclass
class Scaling(ParamsMixin):
    """Spans that make gain matrices dimensionless and comparable.

    Attributes:
        u_span: Largest available move in each manipulated variable, length
            ``n_u``.  Strictly positive.
        y_span: Largest *acceptable control error* in each controlled
            variable, length ``n_y``.  Strictly positive.
        d_span: Largest expected excursion of each disturbance, length
            ``n_d``, or ``None`` when no disturbances are declared.
        explicit: ``True`` when the spans are real engineering judgements.
            :meth:`unscaled` sets it ``False``, which makes every report
            built from it carry a caveat rather than silently reading as a
            scaled result.
        note: Free text carried into reports, e.g. where the spans came from.

    Example:
        >>> sc = Scaling(u_span=[20.0, 0.5], y_span=[1.0, 0.02],
        ...              d_span=[5.0])
        >>> sc.n_u, sc.n_y, sc.n_d
        (2, 2, 1)
    """

    u_span: Any
    y_span: Any
    d_span: Any = None
    explicit: bool = True
    note: str = ""

    def __post_init__(self):
        self.u_span = _as_span(self.u_span, "u_span")
        self.y_span = _as_span(self.y_span, "y_span")
        self.d_span = _as_span(self.d_span, "d_span")
        if self.u_span is None or self.y_span is None:
            raise ValueError("Scaling requires both u_span and y_span")

    # -- shape --------------------------------------------------------------
    @property
    def n_u(self) -> int:
        """Number of manipulated variables."""
        return int(self.u_span.shape[0])

    @property
    def n_y(self) -> int:
        """Number of controlled variables."""
        return int(self.y_span.shape[0])

    @property
    def n_d(self) -> int:
        """Number of disturbances, 0 when none are declared."""
        return 0 if self.d_span is None else int(self.d_span.shape[0])

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_bounds(cls, u_lb: Any, u_ub: Any, y_tol: Any,
                    d_lb: Any = None, d_ub: Any = None,
                    note: str = "") -> "Scaling":
        """Build spans from operating bounds and an output tolerance.

        Args:
            u_lb: Lower bounds on the manipulated variables.
            u_ub: Upper bounds on the manipulated variables.
            y_tol: Largest acceptable control error per controlled variable.
            d_lb: Lower bounds on the disturbances, optional.
            d_ub: Upper bounds on the disturbances, optional.
            note: Provenance note carried into reports.

        Returns:
            A :class:`Scaling` with ``u_span = u_ub - u_lb`` and, when
            disturbance bounds are given, ``d_span = d_ub - d_lb``.

        Note:
            The *full* bound range is the honest input span only when the
            controller may really use all of it.  If the plant normally sits
            at one end, halve it: a controllability claim made against travel
            the plant never has is not a claim about the plant.

        Example:
            >>> Scaling.from_bounds([0.0, 300.0], [1.0, 400.0],
            ...                     y_tol=[0.01, 2.0]).u_span
            Array([  1., 100.], dtype=float64)
        """
        u_lb = jnp.atleast_1d(jnp.asarray(u_lb, dtype=float))
        u_ub = jnp.atleast_1d(jnp.asarray(u_ub, dtype=float))
        d_span = None
        if d_lb is not None and d_ub is not None:
            d_span = (jnp.atleast_1d(jnp.asarray(d_ub, dtype=float))
                      - jnp.atleast_1d(jnp.asarray(d_lb, dtype=float)))
        return cls(u_span=u_ub - u_lb, y_span=y_tol, d_span=d_span,
                   note=note or "spans from operating bounds")

    @classmethod
    def from_block(cls, block, y_tol: Any, d_span: Any = None) -> "Scaling":
        """Build spans from a :class:`~difflow.planning.block.Block`.

        The block already declares ``lb``/``ub`` on its inputs, which is the
        same information ``u_span`` wants.  Output tolerances are not
        something a planning block knows, so ``y_tol`` must still be given.

        Args:
            block: A planning block whose ``u_names`` are the manipulated
                variables and ``y_names`` the controlled variables.
            y_tol: Largest acceptable control error per block output.
            d_span: Disturbance spans, optional.

        Returns:
            A :class:`Scaling`.

        Raises:
            ValueError: If any input of the block is unbounded, since an
                infinite span carries no information.
        """
        lb = np.asarray(block.lb, dtype=float)
        ub = np.asarray(block.ub, dtype=float)
        span = ub - lb
        if not np.all(np.isfinite(span)):
            bad = [n for n, s in zip(block.u_names, span) if not np.isfinite(s)]
            raise ValueError(
                f"Block {block.name!r} has unbounded input(s) {bad}; an "
                "infinite u_span cannot scale a gain. Give finite bounds, or "
                "build the Scaling directly with the move each lever actually "
                "has available.")
        return cls(u_span=span, y_span=y_tol, d_span=d_span,
                   note=f"spans from block {block.name!r} bounds")

    @classmethod
    def unscaled(cls, n_u: int, n_y: int, n_d: int | None = None,
                 note: str = "") -> "Scaling":
        """Unit spans — a deliberate, recorded refusal to scale.

        Use this only when the variables are *already* dimensionless and
        comparable (a gain matrix taken from a textbook, or one you scaled
        yourself).  Reports built from it are marked unscaled and carry a
        caveat finding, because a singular value of a matrix in mixed units
        describes the units.

        Args:
            n_u: Number of manipulated variables.
            n_y: Number of controlled variables.
            n_d: Number of disturbances, optional.
            note: Why scaling was skipped.

        Returns:
            A :class:`Scaling` of ones with ``explicit=False``.
        """
        return cls(u_span=jnp.ones(n_u), y_span=jnp.ones(n_y),
                   d_span=None if n_d is None else jnp.ones(n_d),
                   explicit=False, note=note or "unit spans (not scaled)")

    # -- use ----------------------------------------------------------------
    def scale_gain(self, G: Array) -> Array:
        """Return ``De^-1 G Du``.

        Args:
            G: Raw gain matrix ``dy/du``, shape ``(n_y, n_u)``.

        Returns:
            The dimensionless gain matrix.

        Raises:
            ValueError: If ``G`` does not match ``(n_y, n_u)``.
        """
        G = jnp.atleast_2d(jnp.asarray(G, dtype=float))
        self._check_shape(G, self.n_y, self.n_u, "gain matrix")
        return G * self.u_span[None, :] / self.y_span[:, None]

    def scale_disturbance(self, Gd: Array) -> Array:
        """Return ``De^-1 Gd Dd``.

        Args:
            Gd: Raw disturbance gain ``dy/dd``, shape ``(n_y, n_d)``.

        Returns:
            The dimensionless disturbance gain matrix.

        Raises:
            ValueError: If no ``d_span`` was declared, or shapes disagree.
        """
        if self.d_span is None:
            raise ValueError(
                "this Scaling declares no d_span, so a disturbance gain "
                "cannot be made dimensionless. Give d_span = the largest "
                "excursion you expect of each disturbance.")
        Gd = jnp.atleast_2d(jnp.asarray(Gd, dtype=float))
        self._check_shape(Gd, self.n_y, self.n_d, "disturbance gain")
        return Gd * self.d_span[None, :] / self.y_span[:, None]

    def unscale_gain(self, G_scaled: Array) -> Array:
        """Invert :meth:`scale_gain`, returning engineering units."""
        G_scaled = jnp.atleast_2d(jnp.asarray(G_scaled, dtype=float))
        self._check_shape(G_scaled, self.n_y, self.n_u, "scaled gain matrix")
        return G_scaled * self.y_span[:, None] / self.u_span[None, :]

    def caveat(self) -> str | None:
        """The one-line warning to attach to results, or ``None`` if scaled."""
        if self.explicit:
            return None
        return ("gains are UNSCALED (unit spans): singular values, condition "
                "numbers and disturbance gains below describe the choice of "
                "engineering units as much as the plant, and none of them may "
                "be compared against 1. Declare u_span/y_span/d_span.")

    def warn_if_unscaled(self, stacklevel: int = 3) -> None:
        """Emit an :class:`OperabilityWarning` when the spans are unit spans."""
        msg = self.caveat()
        if msg is not None:
            warnings.warn(msg, OperabilityWarning, stacklevel=stacklevel)

    def _check_shape(self, M: Array, n_rows: int, n_cols: int,
                     what: str) -> None:
        if M.shape != (n_rows, n_cols):
            raise ValueError(
                f"{what} has shape {tuple(M.shape)} but this Scaling is for "
                f"{(n_rows, n_cols)}. Scaling spans and gain matrices must "
                "agree, or the dimensionless numbers are nonsense.")

    def __repr__(self) -> str:
        tag = "" if self.explicit else ", UNSCALED"
        return (f"Scaling(n_u={self.n_u}, n_y={self.n_y}, n_d={self.n_d}"
                f"{tag})")


def _scaling_flatten(sc: Scaling):
    return (sc.u_span, sc.y_span, sc.d_span), (sc.explicit, sc.note)


def _scaling_unflatten(aux, children):
    u_span, y_span, d_span = children
    explicit, note = aux
    obj = Scaling.__new__(Scaling)
    object.__setattr__(obj, "u_span", u_span)
    object.__setattr__(obj, "y_span", y_span)
    object.__setattr__(obj, "d_span", d_span)
    object.__setattr__(obj, "explicit", explicit)
    object.__setattr__(obj, "note", note)
    return obj


# Registered by hand rather than via ParamsMixin.register_as_pytree because
# that helper packs its aux data into a dict, which jit cannot hash.
jax.tree_util.register_pytree_node(Scaling, _scaling_flatten,
                                   _scaling_unflatten)
