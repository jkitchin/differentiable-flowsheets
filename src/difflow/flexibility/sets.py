"""The two data objects a flexibility question needs: a set and a recourse.

An :class:`UncertaintySet` is a box around a nominal parameter vector, with
independent deviations up and down so that an asymmetric envelope -- an ore
liquor that can be much leaner than nominal but only slightly richer -- is
expressible without inventing a fictitious symmetric range.

A :class:`ControlSpec` is the box the *recourse* variables live in.  The
distinction between the two is the whole point of flexibility analysis: the
parameters in the uncertainty set are handed to you, the controls are yours to
re-optimize once you see them.

Note:
    Deviations are *data*, not traced values: the vertex enumeration branches
    on which coordinates actually vary, which has to be a static decision.
    The *scaling* of the set, on the other hand, is traceable, which is what
    lets :func:`~difflow.flexibility.index.flexibility_index` bisect on it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

#: Enumerating more vertices than this is almost certainly a mistake; use
#: ``method="continuous"`` or :func:`difflow.flexibility.expected_feasibility`.
MAX_VERTICES = 1 << 14


def _as_array(x, n: int | None = None, name: str = "value") -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if n is not None and a.size == 1 and n != 1:
        a = np.repeat(a, n)
    if n is not None and a.size != n:
        raise ValueError(f"{name} has length {a.size}, expected {n}")
    return a


@dataclass
class UncertaintySet(ParamsMixin):
    """A box of uncertain parameters, ``T``, around a nominal point.

    The set at scaling ``delta`` is

    .. math:: T(\\delta) = \\{\\theta : \\theta^N - \\delta\\,\\Delta^- \\le
              \\theta \\le \\theta^N + \\delta\\,\\Delta^+\\}

    so ``delta = 1`` is the envelope you actually expect and the flexibility
    index is the largest ``delta`` a design survives.

    Attributes:
        nominal: Nominal parameter values, ``theta^N``.
        lower: Downward deviations ``Delta^-``, as positive magnitudes.
        upper: Upward deviations ``Delta^+``, as positive magnitudes.
        names: Optional parameter names, used only in reports.

    Example:
        >>> T = UncertaintySet(nominal=[1.0, 300.0],
        ...                    lower=[0.2, 10.0], upper=[0.1, 10.0],
        ...                    names=["feed_Ce", "T_feed"])
        >>> T.vertices().shape
        (4, 2)
        >>> T.n
        2
    """

    nominal: Array
    lower: Array
    upper: Array
    names: tuple[str, ...] | None = None

    def __post_init__(self):
        nom = _as_array(self.nominal, name="nominal")
        n = nom.size
        lo = np.abs(_as_array(self.lower, n, "lower"))
        up = np.abs(_as_array(self.upper, n, "upper"))
        if self.names is None:
            self.names = tuple(f"theta{i}" for i in range(n))
        else:
            self.names = tuple(self.names)
            if len(self.names) != n:
                raise ValueError(
                    f"names has length {len(self.names)}, expected {n}")
        object.__setattr__(self, "_nominal_np", nom)
        object.__setattr__(self, "_lower_np", lo)
        object.__setattr__(self, "_upper_np", up)
        self.nominal = jnp.asarray(nom)
        self.lower = jnp.asarray(lo)
        self.upper = jnp.asarray(up)

    @property
    def n(self) -> int:
        """Number of uncertain parameters."""
        return int(self._nominal_np.size)

    @property
    def varying(self) -> np.ndarray:
        """Indices of the coordinates that actually move."""
        return np.flatnonzero((self._lower_np > 0) | (self._upper_np > 0))

    @property
    def n_vertices(self) -> int:
        """Number of distinct vertices, ``2 ** (number of varying axes)``."""
        return 1 << int(self.varying.size)

    def signs(self) -> np.ndarray:
        """The ``(n_vertices, n)`` matrix of vertex directions.

        Entry ``[v, i]`` is ``+1`` if vertex ``v`` takes the upward deviation
        in coordinate ``i``, ``-1`` if downward, and ``0`` for a coordinate
        that does not vary.

        Returns:
            An integer array.

        Raises:
            ValueError: If the enumeration would exceed :data:`MAX_VERTICES`.
        """
        active = self.varying
        if self.n_vertices > MAX_VERTICES:
            raise ValueError(
                f"{self.n_vertices} vertices from {active.size} varying "
                f"parameters exceeds MAX_VERTICES={MAX_VERTICES}. Vertex "
                "enumeration is exponential; use method='continuous' or "
                "expected_feasibility() for a set this size.")
        out = np.zeros((self.n_vertices, self.n), dtype=int)
        for v, combo in enumerate(itertools.product((-1, 1), repeat=active.size)):
            out[v, active] = combo
        return out

    def vertices(self, scale: float | Array = 1.0) -> Array:
        """Vertices of the scaled box.

        Args:
            scale: The scaling ``delta``.  May be a traced value.

        Returns:
            A ``(n_vertices, n)`` array of parameter realizations.
        """
        s = self.signs()
        dev = jnp.where(jnp.asarray(s) > 0, self.upper[None, :],
                        -self.lower[None, :])
        return self.nominal[None, :] + jnp.asarray(scale) * jnp.asarray(s != 0) * dev

    def bounds(self, scale: float | Array = 1.0) -> tuple[Array, Array]:
        """Lower and upper corner of the scaled box.

        Args:
            scale: The scaling ``delta``.

        Returns:
            ``(lo, hi)``, each of shape ``(n,)``.
        """
        s = jnp.asarray(scale)
        return self.nominal - s * self.lower, self.nominal + s * self.upper

    def contains(self, theta: Array, scale: float | Array = 1.0) -> Array:
        """Whether ``theta`` lies in the scaled box.

        Args:
            theta: A parameter realization.
            scale: The scaling ``delta``.

        Returns:
            A boolean scalar.
        """
        lo, hi = self.bounds(scale)
        t = jnp.asarray(theta)
        return jnp.all((t >= lo) & (t <= hi))

    def label(self, theta) -> str:
        """A ``name=value`` rendering of a realization, for reports."""
        t = np.atleast_1d(np.asarray(theta, dtype=float))
        return ", ".join(f"{n}={v:.6g}" for n, v in zip(self.names, t))

    def describe(self) -> str:
        """A table of the set, one row per parameter."""
        lines = [f"UncertaintySet: {self.n} parameters, "
                 f"{self.n_vertices} vertices",
                 f"  {'parameter':<20s}{'nominal':>12s}{'-Delta':>12s}"
                 f"{'+Delta':>12s}"]
        for i, nm in enumerate(self.names):
            lines.append(f"  {nm:<20s}{self._nominal_np[i]:12.5g}"
                         f"{self._lower_np[i]:12.5g}{self._upper_np[i]:12.5g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"UncertaintySet(n={self.n}, vertices={self.n_vertices}, "
                f"names={list(self.names)})")


@dataclass
class ControlSpec(ParamsMixin):
    """The recourse variables and the box they may move in.

    These are the degrees of freedom that get re-optimized *after* the
    uncertain parameters are revealed.  Everything a plant operator can still
    turn once the feed arrives belongs here; everything fixed at design time
    belongs in ``d``.

    Attributes:
        lower: Lower bounds, one per control.
        upper: Upper bounds, one per control.
        names: Optional control names, used only in reports.
        start: Optional starting point for the inner solve.  Defaults to the
            box midpoint.

    Note:
        Bounds must be finite.  An unbounded recourse variable makes the
        inner minimization ill-posed as often as not, and giving it a wide
        but finite box is both honest and numerically better behaved.

    Example:
        >>> u = ControlSpec(lower=[0.0], upper=[10.0], names=["reflux"])
        >>> u.n
        1
    """

    lower: Array
    upper: Array
    names: tuple[str, ...] | None = None
    start: Array | None = None

    def __post_init__(self):
        lo = _as_array(self.lower, name="lower")
        up = _as_array(self.upper, lo.size, "upper")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(up))):
            raise ValueError(
                "control bounds must be finite; an unbounded recourse "
                "variable makes the inner minimization ill-posed. Use a wide "
                "but finite box.")
        if np.any(up < lo):
            bad = int(np.argmax(lo - up))
            raise ValueError(
                f"control {bad} has upper bound {up[bad]:g} below lower "
                f"bound {lo[bad]:g}")
        if self.names is None:
            self.names = tuple(f"u{i}" for i in range(lo.size))
        else:
            self.names = tuple(self.names)
            if len(self.names) != lo.size:
                raise ValueError(
                    f"names has length {len(self.names)}, expected {lo.size}")
        object.__setattr__(self, "_lower_np", lo)
        object.__setattr__(self, "_upper_np", up)
        self.lower = jnp.asarray(lo)
        self.upper = jnp.asarray(up)
        if self.start is not None:
            self.start = jnp.asarray(_as_array(self.start, lo.size, "start"))

    @property
    def n(self) -> int:
        """Number of control variables."""
        return int(self._lower_np.size)

    def starts(self, n_starts: int = 3) -> Array:
        """Deterministic multi-start points inside the box.

        The inner problem ``min_u max_j f_j`` is not convex in general, so a
        single start can land on a local minimum and report a design as less
        flexible than it is.  These starts are fixed rather than random so
        that a reported index is reproducible.

        Args:
            n_starts: How many starts to return, at least one.

        Returns:
            A ``(n_starts, n)`` array.
        """
        lo, up = self._lower_np, self._upper_np
        span = up - lo
        frac = np.linspace(0.5, 0.5, 1) if n_starts <= 1 else np.concatenate(
            [[0.5], np.linspace(0.1, 0.9, max(n_starts - 1, 1))])
        pts = [lo + f * span for f in frac[:max(n_starts, 1)]]
        if self.start is not None:
            pts[0] = np.asarray(self.start, dtype=float)
        return jnp.asarray(np.stack(pts))

    def describe(self) -> str:
        """A table of the controls, one row per variable."""
        lines = [f"ControlSpec: {self.n} recourse variables",
                 f"  {'control':<20s}{'lower':>12s}{'upper':>12s}"]
        for i, nm in enumerate(self.names):
            lines.append(f"  {nm:<20s}{self._lower_np[i]:12.5g}"
                         f"{self._upper_np[i]:12.5g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ControlSpec(n={self.n}, names={list(self.names)})"


#: The recourse-free control spec: no variable may respond to the parameters.
NO_CONTROLS = ControlSpec(lower=np.zeros(0), upper=np.zeros(0), names=())


def as_uncertainty_set(obj) -> UncertaintySet:
    """Coerce a mapping or an :class:`UncertaintySet` to an set.

    Args:
        obj: An :class:`UncertaintySet`, or a mapping
            ``{name: (nominal, minus, plus)}`` / ``{name: (nominal, pm)}``.

    Returns:
        An :class:`UncertaintySet`.

    Example:
        >>> T = as_uncertainty_set({"C": (1.0, 0.2), "T": (300.0, 5.0, 10.0)})
        >>> T.names
        ('C', 'T')
    """
    if isinstance(obj, UncertaintySet):
        return obj
    names, nom, lo, up = [], [], [], []
    for k, v in obj.items():
        v = np.atleast_1d(np.asarray(v, dtype=float))
        if v.size == 2:
            n0, d_lo, d_up = v[0], v[1], v[1]
        elif v.size == 3:
            n0, d_lo, d_up = v
        else:
            raise ValueError(
                f"parameter {k!r} must be (nominal, pm) or "
                f"(nominal, minus, plus), got {v}")
        names.append(k)
        nom.append(n0)
        lo.append(d_lo)
        up.append(d_up)
    return UncertaintySet(nominal=nom, lower=lo, upper=up, names=tuple(names))


def as_control_spec(obj) -> ControlSpec:
    """Coerce a mapping, ``None``, or a :class:`ControlSpec` to a spec.

    Args:
        obj: A :class:`ControlSpec`, ``None`` (meaning no recourse), or a
            mapping ``{name: (lower, upper)}``.

    Returns:
        A :class:`ControlSpec`.
    """
    if obj is None:
        return NO_CONTROLS
    if isinstance(obj, ControlSpec):
        return obj
    names, lo, up = [], [], []
    for k, v in obj.items():
        v = np.atleast_1d(np.asarray(v, dtype=float))
        if v.size != 2:
            raise ValueError(
                f"control {k!r} must be (lower, upper), got {v}")
        names.append(k)
        lo.append(v[0])
        up.append(v[1])
    return ControlSpec(lower=lo, upper=up, names=tuple(names))
