"""Crude assays and their characterisation into pseudo-components.

A crude is too many molecules to list, so a simulator describes it by its true
boiling point (TBP) curve -- the cumulative fraction distilled against
temperature in a column of many stages at high reflux -- plus a gravity, and
cuts that curve into narrow boiling ranges. Each range becomes one
*pseudo-component*, a fictitious species with the cut's average boiling point
and specific gravity, whose remaining properties come from
:mod:`difflow_refinery.correlations`.

The two steps here:

* :class:`Assay` holds the data as measured: TBP points, a bulk gravity or a
  gravity curve, and optionally a light-ends analysis (C1-C6 as real species).
* :func:`characterize` cuts it and returns a :class:`Characterization`, whose
  arrays -- boiling points, gravities, critical constants, fractions -- are
  ``jax`` arrays and differentiable with respect to the assay data. A gradient
  of a product yield with respect to a TBP point is therefore one ``jax.grad``,
  and an assay's measurement uncertainty can be pushed through a flowsheet
  with :mod:`difflow.uncertainty`.

:meth:`Characterization.species_data` and :meth:`Characterization.thermo`
turn the result into the ``SpeciesData`` / ``CriticalProperties`` that
difflow's :class:`~difflow.thermo.IdealThermo` and
:class:`~difflow.thermo.CubicThermo` take. Those classes hold Python floats,
so that step needs concrete values and is not traceable; it is the boundary
between the assay and a column built from it.

What this is *not* is an assay library. Curated assays are proprietary data
(see ``docs/pims-integration.md``); this module characterises the curve the
caller brings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import ndtr, ndtri

from difflow.database import get_critical_props, get_species_data
from difflow.eos import CriticalProperties
from difflow.thermo import SpeciesData

from difflow_refinery import correlations as corr

Basis = Literal["volume", "mass"]

#: Standard-condition (60 F) liquid specific gravities of the light ends,
#: GPA 2145. Methane's 0.3 is the conventional value -- it has no liquid at
#: 60 F -- and only matters for a volume-basis light-ends analysis.
LIGHT_END_SG: dict[str, float] = {
    "methane": 0.3,
    "ethane": 0.35584,
    "propane": 0.50736,
    "isobutane": 0.56293,
    "n_butane": 0.58407,
    "isopentane": 0.62470,
    "n_pentane": 0.63086,
    "n_hexane": 0.66404,
    "water": 1.0,
}

#: Default cut widths (K) and the temperature (K) up to which each applies:
#: 20 C cuts to 400 C, 40 C cuts to 600 C, 100 C cuts beyond. Narrow where
#: the products are cut from each other, wide in the residue, where no cut
#: point falls.
DEFAULT_CUT_WIDTHS: tuple[tuple[float, float], ...] = (
    (673.15, 20.0),
    (873.15, 40.0),
    (np.inf, 100.0),
)

#: Default cut widths with a :class:`HeavyEnd`: 20 C cuts to 400 C, then
#: 25 C cuts to the heavy end's ``T_max`` -- the vacuum column cuts LVGO,
#: HVGO and slop from each other between 340 and 600 C, and the narrow grid
#: has to reach past its cut points.
DEFAULT_HEAVY_CUT_WIDTHS: tuple[tuple[float, float], ...] = (
    (673.15, 20.0),
    (np.inf, 25.0),
)

#: Contaminants an assay can carry, the bulk-assay field each is given by,
#: and the factor from that field's unit to a mass fraction.
CONTAMINANTS: dict[str, tuple[str, float]] = {
    "sulfur": ("sulfur_wt", 1e-2),
    "nitrogen": ("nitrogen_wppm", 1e-6),
    "ccr": ("ccr_wt", 1e-2),
    "nickel_vanadium": ("nickel_vanadium_wppm", 1e-6),
    "asphaltenes": ("asphaltenes_wt", 1e-2),
}

# Default distribution of each contaminant over boiling point: a logistic in
# TBP, (centre, width) in K, scaled so the crude matches its bulk value.
# Sulfur rises gradually through the distillates, nitrogen later, CCR and
# metals almost only in the residue -- the shape every crude assay shows,
# with no particular crude's numbers in it. A measured curve replaces it.
_CONTAMINANT_SHAPES: dict[str, tuple[float, float]] = {
    "sulfur": (673.15, 80.0),
    "nitrogen": (773.15, 70.0),
    "ccr": (923.15, 50.0),
    "nickel_vanadium": (1033.15, 30.0),
    "asphaltenes": (1013.15, 35.0),
}
# No pseudo-component is more than this mass fraction contaminant (it would
# otherwise be possible for a narrow residue lump to be scaled past 1).
_CONTAMINANT_CAP = 0.9

# Gauss-Legendre points per cut for the mean boiling point on the heavy-end
# curve, which has no closed-form antiderivative. A cut is 20-25 C wide and
# the curve is a C1 cubic in a smooth transform: eight points integrate it
# to well under 1e-6 K in Tb.
_GAUSS_POINTS = 8

# The cut-boundary inversion: bisection to round-off, then one Newton step
# for the derivative (see _invert).
_BISECTION_STEPS = 60

# Vapour-pressure (Antoine) fit: window in K, intersected with
# 0.35 < T/Tc < 0.95, and the number of points it is fitted on.
DEFAULT_ANTOINE_WINDOW = (290.0, 750.0)
_ANTOINE_TR = (0.35, 0.95)
_ANTOINE_POINTS = 40
_ANTOINE_P_MIN = 10.0  # Pa


# =============================================================================
# The assay
# =============================================================================


@dataclass(frozen=True)
class HeavyEnd:
    """How an assay's TBP curve is carried past its last point.

    A TBP curve stops where distillation does -- about 565 C at the end of
    ASTM D5236 -- and a vacuum column needs pseudo-components to 750-800 C.
    With a ``HeavyEnd`` the curve is interpolated and extended on a
    probability scale, ``z = Phi^-1(x)``, which is close to linear in T for
    crude oils (a log-normal-like boiling distribution): a monotone C1
    cubic in ``z`` through the data, continued by the least-squares line
    through the last ``n_tail`` points. Pseudo-components are cut up to
    ``T_max``; everything above it is one *residue lump* whose boiling
    point, molecular weight and (optionally) gravity are set directly,
    because no ``(Tb, SG)`` correlation has a root or any validity there
    (Twu's n-alkane reference fails at about 820 C TBP).

    Attributes:
        T_max: Top of the pseudo-component range (K); above it, the lump.
        residue_Tb: Equivalent normal boiling point of the lump (K).
        residue_mw: Molecular weight of the lump (g/mol).
        residue_sg: Specific gravity of the lump; ``None`` follows the
            crude's Watson K (or the end of the gravity curve).
        n_tail: Points the extension line is fitted through.
        widths: Default cut widths, as :data:`DEFAULT_CUT_WIDTHS`.
        lump_span: The lump's notional TBP range, in widths of the last cut
            above ``T_max``; only product TBP points inside the lump read it.
    """

    T_max: float = 1073.15
    residue_Tb: float | Array = 1223.15
    residue_mw: float | Array = 1500.0
    residue_sg: float | Array | None = None
    n_tail: int = 3
    widths: tuple[tuple[float, float], ...] = DEFAULT_HEAVY_CUT_WIDTHS
    lump_span: float = 4.0


@dataclass(frozen=True)
class Assay:
    """A crude assay: TBP curve, gravity, and optionally light ends.

    Attributes:
        tbp_percent: Cumulative percent distilled (0-100) at each TBP point,
            strictly increasing, starting at 0 (the initial boiling point)
            and ending at 100 (the final boiling point). A curve that stops
            short of 100 % has to be extrapolated before it can be cut, and
            that is a modelling decision left to the caller rather than made
            here silently.
        tbp_T: TBP temperature (K) at each point, strictly increasing.
        basis: Whether the percentages are by ``"volume"`` (liquid volume at
            60 F, the usual assay basis) or by ``"mass"``. Applies to the
            gravity curve and the light ends too.
        sg: Bulk specific gravity (60 F / 60 F) of the whole crude. With no
            ``sg_curve``, each cut's gravity follows from a Watson K constant
            across the crude, chosen so the cuts recombine to this gravity.
        api: Bulk API gravity -- the alternative to ``sg``. Give one.
        sg_curve: ``(mid_percent, SG)`` pairs: the gravity of the material
            at each cumulative percent. Each cut takes the average over its
            span (linear between points, flat beyond the ends). Overrides
            ``sg``/``api`` for the cut gravities.
        light_ends: ``{species: percent of the whole crude}`` for light ends
            measured as real species (difflow database names: ``methane``,
            ``ethane``, ``propane``, ``isobutane``, ``n_butane``,
            ``isopentane``, ``n_pentane``, ...). They replace the bottom of
            the TBP curve: pseudo-components start where the curve reaches
            their total.
        name: A label for reports.
        heavy_end: Extend the curve past its last point and close it with a
            residue lump (:class:`HeavyEnd`). Then ``tbp_percent`` must lie
            strictly inside (0, 100), the curve being open at both ends; the
            material below the first point joins the first cut. ``None``
            (the default) needs a curve from 0 to 100 and changes nothing.
        sulfur_wt, nitrogen_wppm, ccr_wt, nickel_vanadium_wppm,
            asphaltenes_wt: Bulk contaminants of the whole crude (sulfur wt%,
            nitrogen wppm, Conradson carbon wt%, Ni+V wppm, C7 asphaltenes
            wt%). Each is spread over the pseudo-components with a default
            boiling-point shape scaled to the bulk value. ``None`` leaves it
            at zero. Light ends carry none.
        sulfur_curve, nitrogen_curve, ccr_curve: ``(T, value)`` -- a measured
            curve, T (K) at cut mid-points and the value in the bulk field's
            unit. Interpolated at each cut's Tb (flat beyond the ends) and
            used as is, not rescaled to the bulk number.
    """

    tbp_percent: Sequence[float] | Array
    tbp_T: Sequence[float] | Array
    basis: Basis = "volume"
    sg: float | Array | None = None
    api: float | Array | None = None
    sg_curve: tuple[Sequence[float], Sequence[float]] | None = None
    light_ends: Mapping[str, float | Array] = field(default_factory=dict)
    name: str = "crude"
    heavy_end: HeavyEnd | None = None
    sulfur_wt: float | Array | None = None
    nitrogen_wppm: float | Array | None = None
    ccr_wt: float | Array | None = None
    nickel_vanadium_wppm: float | Array | None = None
    asphaltenes_wt: float | Array | None = None
    sulfur_curve: tuple[Sequence[float], Sequence[float]] | None = None
    nitrogen_curve: tuple[Sequence[float], Sequence[float]] | None = None
    ccr_curve: tuple[Sequence[float], Sequence[float]] | None = None

    def __post_init__(self):
        if self.basis not in ("volume", "mass"):
            raise ValueError(f"basis must be 'volume' or 'mass', not {self.basis!r}")
        pct = np.asarray(self.tbp_percent, dtype=float) if not _traced(self.tbp_percent) else None
        T = np.asarray(self.tbp_T, dtype=float) if not _traced(self.tbp_T) else None
        if pct is not None:
            if pct.ndim != 1 or pct.size < 2:
                raise ValueError("tbp_percent needs at least two points")
            if self.heavy_end is not None:
                if pct[0] <= 0.0 or pct[-1] >= 100.0:
                    raise ValueError(
                        "with a heavy_end, tbp_percent must lie strictly inside (0, 100): "
                        "the curve is extended on a probability scale, where 0 and 100 "
                        "are at infinity. Give the initial boiling point as light ends "
                        "or drop it; the material below the first point joins the first cut"
                    )
                if pct.size < max(self.heavy_end.n_tail, 2):
                    raise ValueError(f"a heavy_end needs at least {self.heavy_end.n_tail} TBP points")
            elif abs(pct[0]) > 1e-9 or abs(pct[-1] - 100.0) > 1e-9:
                raise ValueError(
                    f"tbp_percent must run from 0 to 100 (got {pct[0]} to {pct[-1]}); "
                    "extrapolate the curve to its initial and final boiling points first"
                )
            if np.any(np.diff(pct) <= 0):
                raise ValueError("tbp_percent must be strictly increasing")
        if T is not None:
            if pct is not None and T.shape != pct.shape:
                raise ValueError("tbp_T and tbp_percent must have the same length")
            if np.any(np.diff(T) <= 0):
                raise ValueError("tbp_T must be strictly increasing")
            if self.heavy_end is not None and T[-1] >= self.heavy_end.T_max:
                raise ValueError(
                    f"heavy_end.T_max ({self.heavy_end.T_max} K) must be above the "
                    f"last TBP point ({T[-1]} K)")
        if self.sg_curve is None and (self.sg is None) == (self.api is None):
            raise ValueError("give exactly one of sg or api (or an sg_curve)")
        for species in self.light_ends:
            if species not in LIGHT_END_SG:
                raise ValueError(
                    f"no standard liquid gravity for light end {species!r}; "
                    f"known: {', '.join(LIGHT_END_SG)}"
                )

    def with_tbp_point(self, index: int, T: float | Array) -> "Assay":
        """A copy with one TBP temperature (K) replaced -- for sensitivities."""
        return dataclasses.replace(
            self, tbp_T=jnp.asarray(self.tbp_T, dtype=float).at[index].set(T))

    @property
    def bulk_sg(self) -> Array | None:
        """The bulk specific gravity given, from ``sg`` or ``api``."""
        if self.sg is not None:
            return jnp.asarray(self.sg, dtype=float)
        if self.api is not None:
            return corr.sg_from_api(jnp.asarray(self.api, dtype=float))
        return None


def _light_end_mw(name: str) -> float:
    from difflow_refinery.thermo import LIGHT_ENDS, WATER_MW

    return WATER_MW if name == "water" else LIGHT_ENDS[name][0]


def _traced(x) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(x))


# =============================================================================
# Monotone interpolation of the TBP curve
# =============================================================================


def _pchip_slopes(x: Array, y: Array) -> Array:
    """Fritsch-Carlson slopes (the scheme scipy's ``PchipInterpolator`` uses).

    For a strictly increasing curve every secant is positive and the interior
    slope is a weighted harmonic mean of its neighbours -- smooth in the data,
    which is what keeps a cut fraction differentiable in the TBP points.
    """
    h = jnp.diff(x)
    delta = jnp.diff(y) / h
    if x.shape[0] == 2:
        return jnp.stack([delta[0], delta[0]])
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    same = delta[:-1] * delta[1:] > 0
    safe_prev = jnp.where(same, delta[:-1], 1.0)
    safe_next = jnp.where(same, delta[1:], 1.0)
    interior = jnp.where(same, (w1 + w2) / (w1 / safe_prev + w2 / safe_next), 0.0)

    def _end(h0, h1, d0, d1):
        d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        d = jnp.where(jnp.sign(d) != jnp.sign(d0), 0.0, d)
        return jnp.where((jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(d) > 3.0 * jnp.abs(d0)),
                         3.0 * d0, d)

    first = _end(h[0], h[1], delta[0], delta[1])
    last = _end(h[-1], h[-2], delta[-1], delta[-2])
    return jnp.concatenate([first[None], interior, last[None]])


@dataclass(frozen=True)
class _Pchip:
    """Monotone cubic through ``(x, y)``, with its exact antiderivative."""

    x: Array
    y: Array
    d: Array

    @classmethod
    def fit(cls, x, y):
        x = jnp.asarray(x, dtype=float)
        y = jnp.asarray(y, dtype=float)
        return cls(x, y, _pchip_slopes(x, y))

    def _segment(self, t):
        k = jnp.clip(jnp.searchsorted(self.x, t, side="right") - 1, 0, self.x.shape[0] - 2)
        h = self.x[k + 1] - self.x[k]
        return k, h, (t - self.x[k]) / h

    def __call__(self, t: Array) -> Array:
        k, h, s = self._segment(t)
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return (h00 * self.y[k] + h10 * h * self.d[k]
                + h01 * self.y[k + 1] + h11 * h * self.d[k + 1])

    def derivative(self, t: Array) -> Array:
        k, h, s = self._segment(t)
        return ((6 * s**2 - 6 * s) * self.y[k] / h + (3 * s**2 - 4 * s + 1) * self.d[k]
                + (-6 * s**2 + 6 * s) * self.y[k + 1] / h + (3 * s**2 - 2 * s) * self.d[k + 1])

    def integral(self, t: Array) -> Array:
        """``integral(x[0] -> t)`` of the interpolant, exactly."""
        h = jnp.diff(self.x)
        full = h * ((self.y[:-1] + self.y[1:]) / 2.0 + h * (self.d[:-1] - self.d[1:]) / 12.0)
        cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(full)])
        k, hk, s = self._segment(t)
        H00 = s - s**3 + s**4 / 2
        H10 = s**2 / 2 - 2 * s**3 / 3 + s**4 / 4
        H01 = s**3 - s**4 / 2
        H11 = -(s**3) / 3 + s**4 / 4
        part = hk * (H00 * self.y[k] + H10 * hk * self.d[k]
                     + H01 * self.y[k + 1] + H11 * hk * self.d[k + 1])
        return cumulative[k] + part

    def invert(self, target: Array) -> Array:
        """The ``t`` with ``self(t) = target``, for an increasing interpolant.

        Bisection finds it to round-off with the gradient stopped; one Newton
        step from there then carries the implicit-function derivative
        ``dt = -(d self)/self'`` -- exact at the root, so the result is
        differentiable in both the data and the target.
        """
        lo0 = jax.lax.stop_gradient(self.x[0])
        hi0 = jax.lax.stop_gradient(self.x[-1])
        frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, self)
        tgt = jax.lax.stop_gradient(target)

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            below = frozen(mid) < tgt
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        lo, hi = jax.lax.fori_loop(0, _BISECTION_STEPS, body,
                                   (jnp.full_like(tgt, lo0), jnp.full_like(tgt, hi0)))
        t0 = 0.5 * (lo + hi)
        return t0 - (self(t0) - target) / self.derivative(t0)


jax.tree_util.register_dataclass(_Pchip, data_fields=["x", "y", "d"], meta_fields=[])


def _linear_integral(xp: Array, fp: Array, t: Array) -> Array:
    """``integral(xp[0] -> t)`` of the linear interpolant, flat beyond the ends."""
    seg = jnp.diff(xp) * (fp[:-1] + fp[1:]) / 2.0
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    tc = jnp.clip(t, xp[0], xp[-1])
    k = jnp.clip(jnp.searchsorted(xp, tc, side="right") - 1, 0, xp.shape[0] - 2)
    s = tc - xp[k]
    slope = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k])
    inside = cumulative[k] + fp[k] * s + 0.5 * slope * s**2
    return inside + fp[0] * jnp.minimum(t - xp[0], 0.0) + fp[-1] * jnp.maximum(t - xp[-1], 0.0)


# =============================================================================
# The heavy-end curve: monotone C1 cubic on a probability scale
# =============================================================================


@dataclass(frozen=True)
class _ProbabilityCurve:
    """``x(T) = Phi(z(T))``: the TBP curve with a :class:`HeavyEnd`.

    ``z`` is a cubic Hermite through ``(T_i, Phi^-1(x_i))`` with
    Fritsch-Butland weighted-harmonic interior slopes, and straight lines
    beyond the data whose slopes -- least squares through the first/last
    ``n_tail`` points, capped at Fritsch-Carlson's ``3 delta`` so the end
    intervals stay monotone -- are also the Hermite end slopes. So the curve
    is C1 everywhere, including at every data point: a cut boundary placed
    on an assay point (the default grids do that) then has a single-valued
    derivative with respect to it, where a curve with a kink there has two
    and a central difference averages them.
    """

    T: Array
    z: Array
    d: Array

    @classmethod
    def fit(cls, T, x, n_tail: int):
        T = jnp.asarray(T, dtype=float)
        z = ndtri(jnp.asarray(x, dtype=float))

        def ls_slope(t, v):
            tm, vm = jnp.mean(t), jnp.mean(v)
            return jnp.sum((t - tm) * (v - vm)) / jnp.sum((t - tm) ** 2)

        h = jnp.diff(T)
        delta = jnp.diff(z) / h
        w1 = 2.0 * h[1:] + h[:-1]
        w2 = h[1:] + 2.0 * h[:-1]
        interior = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
        lo = jnp.minimum(ls_slope(T[:n_tail], z[:n_tail]), 3.0 * delta[0])
        hi = jnp.minimum(ls_slope(T[-n_tail:], z[-n_tail:]), 3.0 * delta[-1])
        return cls(T, z, jnp.concatenate([lo[None], interior, hi[None]]))

    def zeta(self, t: Array) -> Array:
        T, z, d = self.T, self.z, self.d
        k = jnp.clip(jnp.searchsorted(T, t, side="right") - 1, 0, T.shape[0] - 2)
        h = T[k + 1] - T[k]
        s = (t - T[k]) / h
        inside = ((2 * s**3 - 3 * s**2 + 1) * z[k] + (s**3 - 2 * s**2 + s) * h * d[k]
                  + (-2 * s**3 + 3 * s**2) * z[k + 1] + (s**3 - s**2) * h * d[k + 1])
        return jnp.where(t > T[-1], z[-1] + d[-1] * (t - T[-1]),
                         jnp.where(t < T[0], z[0] + d[0] * (t - T[0]), inside))

    def __call__(self, t: Array) -> Array:
        return ndtr(self.zeta(t))

    def invert(self, x: Array) -> Array:
        """The ``T`` with ``x(T) = x``: Newton on ``z`` from the piecewise-
        linear inverse, unrolled so it differentiates like the forward curve."""
        target = ndtri(jnp.asarray(x, dtype=float))
        T, z, d = self.T, self.z, self.d
        t = jnp.where(target > z[-1], T[-1] + (target - z[-1]) / d[-1],
                      jnp.where(target < z[0], T[0] + (target - z[0]) / d[0],
                                jnp.interp(target, z, T)))
        for _ in range(8):
            val, slope = jax.jvp(self.zeta, (t,), (jnp.ones_like(t),))
            t = t - (val - target) / slope
        return t

    def integral_between(self, a: Array, b: Array) -> Array:
        """``integral(a -> b) x(T) dT`` per interval, by Gauss-Legendre."""
        nodes, weights = np.polynomial.legendre.leggauss(_GAUSS_POINTS)
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        t = mid[..., None] + half[..., None] * nodes
        return half * jnp.sum(weights * self(t), axis=-1)


jax.tree_util.register_dataclass(_ProbabilityCurve, data_fields=["T", "z", "d"], meta_fields=[])


def tbp_curve(assay: "Assay"):
    """The assay's cumulative fraction distilled as a function of T (K).

    The monotone cubic the characterisation integrates: PCHIP in ``x``
    without a heavy end, the probability-scale curve of :class:`HeavyEnd`
    with one. Callable on arrays, and with an ``invert(x)`` method.
    """
    x = jnp.asarray(assay.tbp_percent, dtype=float) / 100.0
    if assay.heavy_end is None:
        return _Pchip.fit(assay.tbp_T, x)
    return _ProbabilityCurve.fit(assay.tbp_T, x, assay.heavy_end.n_tail)


# =============================================================================
# The characterisation
# =============================================================================


@dataclass(frozen=True)
class Characterization:
    """A crude as light ends plus pseudo-components.

    Pseudo-component arrays (``Tb`` ... ``cp_ig_coeffs``) have one entry per
    cut. Whole-crude arrays (``*_fraction``, ``component_MW``,
    ``component_SG``) follow :attr:`names`: the light ends first, in the
    order the assay listed them, then the cuts from lightest to heaviest.

    Every array is a ``jax`` array and differentiable with respect to the
    assay data. Registered as a pytree, so it can be returned from a
    ``jax.jit``-ed or differentiated function.

    Attributes:
        pseudo_names: Species names of the cuts (``pc01``, ``pc02``, ...).
        light_names: Species names of the light ends.
        method: The critical-property correlation used.
        cut_edges: Boundaries of the cuts (K); ``n_cuts + 1`` entries.
        Tb: Normal boiling point of each cut (K): the average of the TBP curve
            over the cut, on the assay's basis.
        SG: Specific gravity of each cut.
        MW, Tc, Pc, omega: Molecular weight (g/mol), critical temperature (K)
            and pressure (Pa), and the acentric factor (for an EOS).
        omega_vp: The acentric factor that puts the Lee-Kesler vapour
            pressure at exactly one atmosphere at ``Tb``. Equal to ``omega``
            below ``Tb/Tc = 0.8``; above it ``omega`` is Kesler-Lee's, which
            does not, and a pseudo-component whose vapour pressure missed its
            own boiling point would be a contradiction.
        Kw: Watson characterisation factor of each cut.
        hvap_nb: Heat of vaporisation at ``Tb`` (J/mol), Riedel.
        cp_liquid_coeffs, cp_ig_coeffs: ``(n_cuts, 4)`` cubic Cp coefficients
            (J/mol/K, T in K), liquid (Kesler-Lee) and ideal-gas
            (Watson-Nelson).
        component_MW, component_SG: MW and liquid SG of every component.
        volume_fraction, mass_fraction, mole_fraction: Composition of the
            whole crude.
        sulfur, nitrogen, ccr, nickel_vanadium, asphaltenes: Mass fraction
            of each contaminant in every component (zero for light ends, and
            for an assay that gives no bulk value or curve for it).
        residue_lump: Whether the last pseudo-component is a
            :class:`HeavyEnd` residue lump. Its ``Tb``, ``MW`` (and ``SG``
            if given) are set directly; its ``Tc``, ``Pc`` and ``omega`` are
            the correlation's values at ``T_max`` carried to its ``Tb`` at
            constant ``Tb/Tc`` -- placeholders that keep an EOS defined, not
            properties anyone measured. Its upper cut edge is notional.
    """

    pseudo_names: tuple[str, ...]
    light_names: tuple[str, ...]
    method: str
    cut_edges: Array
    Tb: Array
    SG: Array
    MW: Array
    Tc: Array
    Pc: Array
    omega: Array
    omega_vp: Array
    Kw: Array
    hvap_nb: Array
    cp_liquid_coeffs: Array
    cp_ig_coeffs: Array
    component_MW: Array
    component_SG: Array
    volume_fraction: Array
    mass_fraction: Array
    mole_fraction: Array
    sulfur: Array = None
    nitrogen: Array = None
    ccr: Array = None
    nickel_vanadium: Array = None
    asphaltenes: Array = None
    residue_lump: bool = False

    def __post_init__(self):
        # Contaminants default to zero, so a Characterization built by hand
        # (or by an older caller) is still complete.
        n = len(self.light_names) + len(self.pseudo_names)
        for key in CONTAMINANTS:
            if getattr(self, key) is None:
                object.__setattr__(self, key, jnp.zeros(n))

    @property
    def names(self) -> tuple[str, ...]:
        """Every component: light ends, then cuts."""
        return self.light_names + self.pseudo_names

    @property
    def n_cuts(self) -> int:
        return len(self.pseudo_names)

    @property
    def bulk_sg(self) -> Array:
        """Specific gravity of the whole crude (ideal mixing by volume)."""
        return jnp.sum(self.volume_fraction * self.component_SG)

    @property
    def bulk_api(self) -> Array:
        return corr.api_from_sg(self.bulk_sg)

    @property
    def bulk_MW(self) -> Array:
        """Number-average molecular weight of the whole crude (g/mol)."""
        return 1.0 / jnp.sum(self.mass_fraction / self.component_MW)

    def pseudo_components(self):
        """The pseudo-components as the vacuum column's property table.

        A :class:`~difflow_refinery.vacuum.PseudoComponents` over
        :attr:`pseudo_names` (the light ends are not in it: a vacuum column
        passes them, and anything else it has no properties for, to its
        overhead). ``T_lo``/``T_hi`` are the cut edges, so product TBP
        points read the same curve the cuts were taken from. Traceable.
        """
        from difflow_refinery.vacuum.assay import PseudoComponents

        k = len(self.light_names)
        return PseudoComponents(
            Tb=self.Tb, SG=self.SG, MW=self.MW, Kw=self.Kw, Tc=self.Tc, Pc=self.Pc,
            omega=self.omega, T_lo=self.cut_edges[:-1], T_hi=self.cut_edges[1:],
            names=self.pseudo_names,
            **{key: getattr(self, key)[k:] for key in CONTAMINANTS},
        )

    def mass_flows(self, total: Array | float) -> Array:
        """Mass flow of every component (``names`` order) for ``total`` kg/s."""
        return jnp.asarray(total, dtype=float) * self.mass_fraction

    def vapor_pressure(self, T: Array | float) -> Array:
        """Lee-Kesler vapour pressure of every cut at ``T`` (Pa)."""
        return corr.vapor_pressure(jnp.asarray(T, dtype=float), self.Tc, self.Pc, self.omega_vp)

    def flows(self, total: Array | float, basis: Literal["mass", "mole"] = "mass") -> dict[str, Array]:
        """Molar flows (mol/s) of every component for a crude feed.

        Args:
            total: The feed rate: kg/s for ``basis="mass"``, mol/s for
                ``basis="mole"``.
            basis: What ``total`` measures.
        """
        if basis == "mass":
            moles = jnp.asarray(total, dtype=float) * 1000.0 / self.bulk_MW
        elif basis == "mole":
            moles = jnp.asarray(total, dtype=float)
        else:
            raise ValueError(f"basis must be 'mass' or 'mole', not {basis!r}")
        return {n: moles * self.mole_fraction[i] for i, n in enumerate(self.names)}

    def stream(self, total: Array | float, T: Array | float, P: Array | float,
               basis: Literal["mass", "mole"] = "mass"):
        """A difflow feed stream of this crude (see :meth:`flows`)."""
        from difflow.streams import make_stream

        return make_stream(self.flows(total, basis), T, P)

    # ------------------------------------------------------------------
    # Concrete export to difflow's thermo classes
    # ------------------------------------------------------------------

    def species_data(
        self,
        cp: Literal["liquid", "ideal_gas"] = "liquid",
        antoine_window: tuple[float, float] = DEFAULT_ANTOINE_WINDOW,
    ) -> dict[str, SpeciesData]:
        """``SpeciesData`` for every component, for :class:`~difflow.thermo.IdealThermo`.

        Needs concrete values: call it outside ``jit``/``grad``.

        Args:
            cp: Which heat capacity goes in ``Cp_coeffs``. difflow's two
                thermo packages read that field differently --
                ``IdealThermo`` takes it as the *liquid* Cp and adds Watson's
                heat of vaporisation for a vapour, ``CubicThermo`` takes it as
                the *ideal-gas* Cp and adds the EOS departure for either phase.
                :meth:`thermo` picks the right one. Light ends always come from
                difflow's database as they are.
            antoine_window: Temperature range (K) the Antoine equation is
                fitted to the Lee-Kesler vapour pressure over (see
                :func:`fit_antoine` for how each cut narrows or extends it).
                Recorded as each cut's ``T_antoine_min``/``T_antoine_max``.
        """
        if cp not in ("liquid", "ideal_gas"):
            raise ValueError(f"cp must be 'liquid' or 'ideal_gas', not {cp!r}")
        out = {name: get_species_data(name) for name in self.light_names}
        arr = {k: np.asarray(getattr(self, k)) for k in (
            "Tb", "MW", "Tc", "Pc", "omega_vp", "hvap_nb", "cp_liquid_coeffs", "cp_ig_coeffs")}
        cp_key = "cp_liquid_coeffs" if cp == "liquid" else "cp_ig_coeffs"
        for i, name in enumerate(self.pseudo_names):
            Tb, Tc, Pc, w = arr["Tb"][i], arr["Tc"][i], arr["Pc"][i], arr["omega_vp"][i]
            (A, B, C), (T_lo, T_hi) = fit_antoine(Tc, Pc, w, antoine_window, Tb=Tb)
            out[name] = SpeciesData(
                name=name,
                MW=float(arr["MW"][i]),
                Cp_coeffs=tuple(float(c) for c in arr[cp_key][i]),
                Hvap_coeffs=_watson(float(arr["hvap_nb"][i]), float(Tc), float(Tb)),
                antoine_coeffs=(A, B, C),
                Hf=0.0,
                Cp_vapor_coeffs=tuple(float(c) for c in arr["cp_ig_coeffs"][i]),
                T_antoine_min=T_lo,
                T_antoine_max=T_hi,
            )
        return out

    def critical_properties(self) -> dict[str, CriticalProperties]:
        """``CriticalProperties`` of every component, for a cubic EOS.

        Needs concrete values: call it outside ``jit``/``grad``.
        """
        out = {name: get_critical_props(name) for name in self.light_names}
        for i, name in enumerate(self.pseudo_names):
            out[name] = CriticalProperties(
                name=name, Tc=float(self.Tc[i]), Pc=float(self.Pc[i]),
                omega=float(self.omega[i]), MW=float(self.MW[i]),
            )
        return out

    def thermo(self, eos: Literal["ideal", "pr", "srk"] = "ideal",
               antoine_window: tuple[float, float] = DEFAULT_ANTOINE_WINDOW):
        """A difflow thermo package for this crude.

        Args:
            eos: ``"ideal"`` for :class:`~difflow.thermo.IdealThermo`
                (Raoult K-values from the fitted vapour pressures), ``"pr"``
                or ``"srk"`` for :class:`~difflow.thermo.CubicThermo` over
                Peng-Robinson or SRK.
            antoine_window: See :meth:`species_data`.
        """
        from difflow.eos import PengRobinson, SRK
        from difflow.thermo import CubicThermo, IdealThermo

        if eos == "ideal":
            return IdealThermo(self.species_data("liquid", antoine_window))
        if eos not in ("pr", "srk"):
            raise ValueError(f"eos must be 'ideal', 'pr' or 'srk', not {eos!r}")
        cls = PengRobinson if eos == "pr" else SRK
        return CubicThermo(IdealThermo(self.species_data("ideal_gas", antoine_window)),
                           cls(self.critical_properties()))

    def table(self) -> str:
        """The cuts as a fixed-width text table."""
        rows = [f"{'name':<10} {'NBP C':>7} {'SG':>6} {'MW':>7} {'Tc K':>7} "
                f"{'Pc bar':>7} {'omega':>6} {'Kw':>6} {'vol %':>6} {'wt %':>6}"]
        k = len(self.light_names)
        for i, name in enumerate(self.light_names):
            rows.append(f"{name:<10} {'':>7} {float(self.component_SG[i]):6.4f} "
                        f"{float(self.component_MW[i]):7.2f} {'':>7} {'':>7} {'':>6} {'':>6} "
                        f"{100 * float(self.volume_fraction[i]):6.2f} "
                        f"{100 * float(self.mass_fraction[i]):6.2f}")
        for i, name in enumerate(self.pseudo_names):
            rows.append(
                f"{name:<10} {float(self.Tb[i]) - 273.15:7.1f} {float(self.SG[i]):6.4f} "
                f"{float(self.MW[i]):7.1f} {float(self.Tc[i]):7.1f} {float(self.Pc[i]) / 1e5:7.2f} "
                f"{float(self.omega[i]):6.3f} {float(self.Kw[i]):6.2f} "
                f"{100 * float(self.volume_fraction[k + i]):6.2f} "
                f"{100 * float(self.mass_fraction[k + i]):6.2f}")
        return "\n".join(rows)


jax.tree_util.register_dataclass(
    Characterization,
    data_fields=["cut_edges", "Tb", "SG", "MW", "Tc", "Pc", "omega", "omega_vp", "Kw",
                 "hvap_nb", "cp_liquid_coeffs", "cp_ig_coeffs", "component_MW",
                 "component_SG", "volume_fraction", "mass_fraction", "mole_fraction",
                 *CONTAMINANTS],
    meta_fields=["pseudo_names", "light_names", "method", "residue_lump"],
)


def _watson(hvap_nb: float, Tc: float, Tb: float, n: float = 0.38) -> tuple[float, float, float]:
    """``SpeciesData.Hvap_coeffs`` ``(A, n, Tc)`` through ``hvap_nb`` at ``Tb``.

    ``Hvap = A (1 - T/Tc)^n``, so ``A`` is the value at 0 K, not at ``Tb``.
    """
    return (hvap_nb / (1.0 - Tb / Tc) ** n, n, Tc)


def default_cut_points(assay: Assay, widths=DEFAULT_CUT_WIDTHS) -> tuple[float, ...]:
    """Interior cut boundaries (K) at round Celsius temperatures.

    Boundaries fall on multiples of the local width (20 C to 400 C, 40 C to
    600 C, 100 C beyond), between the start of the pseudo-component range
    and the final boiling point. A boundary closer than half a width to
    either end is dropped rather than leave a sliver of a cut.

    With a :class:`HeavyEnd` the range ends at its ``T_max`` instead of the
    final boiling point, and ``widths`` defaults to the heavy end's.

    Needs a concrete assay: the number of cuts is the shape of every array
    that follows, so it cannot depend on a traced value. Under ``jax.grad``
    pass ``cut_points`` to :func:`characterize` explicitly.
    """
    if _traced((assay.tbp_percent, assay.tbp_T, dict(assay.light_ends))):
        raise ValueError(
            "the default cut points need concrete assay data: the number of "
            "cuts fixes the shape of everything downstream. Pass cut_points= "
            "explicitly when differentiating with respect to the assay "
            "(default_cut_points(concrete_assay) gives the usual ones)."
        )
    x_le = sum(float(v) for v in assay.light_ends.values()) / 100.0
    if assay.heavy_end is None:
        curve = _Pchip.fit(assay.tbp_T, np.asarray(assay.tbp_percent) / 100.0)
        T_start = float(curve.invert(jnp.asarray(x_le))) if x_le > 0 else float(np.asarray(assay.tbp_T)[0])
        T_end = float(np.asarray(assay.tbp_T)[-1])
    else:
        T_start = float(_heavy_start(assay, tbp_curve(assay), x_le))
        T_end = float(assay.heavy_end.T_max)
        if widths is DEFAULT_CUT_WIDTHS:
            widths = assay.heavy_end.widths

    points = []
    T_C = 0.0
    while True:
        limit, width = next((lim, w) for lim, w in widths if T_C + 273.15 < lim)
        T_C = (np.floor(T_C / width) + 1) * width
        T = T_C + 273.15
        if T >= T_end - width / 2:
            break
        if T > T_start + width / 2:
            points.append(T)
    return tuple(points)


def _heavy_start(assay, curve, x_le):
    """Where the pseudo-components start on an open-ended curve: where it
    reaches the light-ends total, or its first point if that is lower (the
    material in between then joins the first cut)."""
    x0 = jnp.asarray(assay.tbp_percent, dtype=float)[0] / 100.0
    return curve.invert(jnp.maximum(x_le, x0))


def _contaminants(assay, Tb, mass, k):
    """Per-component contaminant mass fractions (light ends first, zero).

    A measured curve is interpolated at each cut's Tb; otherwise the default
    shape is scaled so the whole crude has the bulk value, then capped.
    """
    n = Tb.shape[0]
    out = {}
    for key, (field_name, unit) in CONTAMINANTS.items():
        curve = getattr(assay, f"{key}_curve", None)
        bulk = getattr(assay, field_name)
        if curve is not None:
            T_pts, values = (jnp.asarray(a, dtype=float) for a in curve)
            cut = jnp.interp(Tb, T_pts, values) * unit
        elif bulk is not None:
            centre, width = _CONTAMINANT_SHAPES[key]
            shape = jax.nn.sigmoid((Tb - centre) / width)
            scale = jnp.asarray(bulk, dtype=float) * unit / jnp.sum(mass[k:] * shape)
            cut = jnp.minimum(shape * scale, _CONTAMINANT_CAP)
        else:
            cut = jnp.zeros(n)
        out[key] = jnp.concatenate([jnp.zeros(k), cut])
    return out


def characterize(
    assay: Assay,
    cut_points: Sequence[float] | None = None,
    method: str | None = None,
    prefix: str = "pc",
) -> Characterization:
    """Cut an assay into pseudo-components and estimate their properties.

    Each cut takes the average of the TBP curve over its span as its boiling
    point (on the assay's basis) and the average of the gravity curve -- or,
    without one, the gravity that a crude-wide Watson K gives at that boiling
    point, with K chosen so the whole crude recombines to its bulk gravity.

    Differentiable with respect to the assay's numbers (TBP temperatures and
    percentages, gravity, light-end fractions) when ``cut_points`` is given.

    Args:
        assay: The data.
        cut_points: Interior cut boundaries (K). The outer boundaries are
            where the curve reaches the light-ends total, and the final
            boiling point. Default: :func:`default_cut_points`.
        method: Critical-property correlation, one of
            :data:`~difflow_refinery.correlations.CRITICAL_METHODS`. Default
            ``"twu"`` -- the crude unit's historical coding, whose numbers
            are pinned -- or ``"twu_1984"``, Twu as published, for an assay
            with a :class:`HeavyEnd` (see :mod:`~difflow_refinery.correlations`).
        prefix: Pseudo-component names are ``f"{prefix}{i:02d}"``, from 1;
            a residue lump is ``f"{prefix}resid"``.

    With a :class:`HeavyEnd` the cuts run from where the curve reaches the
    light ends to ``T_max`` and are followed by the residue lump; the curve
    is the probability-scale one and each cut's Tb is still the mean
    temperature over the cut (by Gauss-Legendre rather than in closed form).
    The Watson K that fits the bulk gravity is taken over the lump too.
    Contaminants given on the assay are distributed in either case.

    Returns:
        A :class:`Characterization`.
    """
    if method is None:
        method = "twu" if assay.heavy_end is None else "twu_1984"
    if cut_points is None:
        cut_points = default_cut_points(assay)
    points = jnp.asarray(cut_points, dtype=float).reshape(-1)
    if assay.heavy_end is not None:
        return _characterize_heavy(assay, points, method, prefix)

    curve = _Pchip.fit(assay.tbp_T, jnp.asarray(assay.tbp_percent, dtype=float) / 100.0)
    light_names = tuple(assay.light_ends)
    le_frac = jnp.asarray([assay.light_ends[n] for n in light_names], dtype=float).reshape(-1) / 100.0
    x_le = jnp.sum(le_frac)
    T_start = jnp.where(x_le > 0, curve.invert(x_le), curve.x[0])

    if not _traced((points, T_start, curve.x)):
        pts = np.asarray(points)
        if np.any(np.diff(pts) <= 0):
            raise ValueError("cut_points must be strictly increasing")
        if pts.size and (pts[0] <= float(T_start) or pts[-1] >= float(curve.x[-1])):
            raise ValueError(
                f"cut_points must lie strictly between {float(T_start):.1f} K (where the "
                f"pseudo-components start) and the final boiling point {float(curve.x[-1]):.1f} K"
            )

    edges = jnp.concatenate([T_start[None], points, curve.x[-1:]])
    x_edges = curve(edges).at[0].set(x_le).at[-1].set(1.0)
    frac = jnp.diff(x_edges)
    integral = curve.integral(edges)
    # Mean of T over the cut, by parts: int T dx = [T x] - int x dT.
    Tb = (edges[1:] * x_edges[1:] - edges[:-1] * x_edges[:-1] - jnp.diff(integral)) / frac

    le_sg = jnp.asarray([LIGHT_END_SG[n] for n in light_names], dtype=float).reshape(-1)
    le_mw = jnp.asarray([_light_end_mw(n) for n in light_names], dtype=float).reshape(-1)

    if assay.sg_curve is not None:
        mid, sgs = (jnp.asarray(a, dtype=float) / s for a, s in zip(assay.sg_curve, (100.0, 1.0)))
        SG = jnp.diff(_linear_integral(mid, sgs, x_edges)) / frac
    else:
        bulk = assay.bulk_sg
        cube = (1.8 * Tb) ** (1.0 / 3.0)
        if assay.basis == "volume":
            # sum(v SG) = SG_bulk, SG_i = cube_i / Kw
            Kw = jnp.sum(frac * cube) / (bulk - jnp.sum(le_frac * le_sg))
        else:
            # sum(w / SG) = 1 / SG_bulk (volumes add)
            Kw = (1.0 / bulk - jnp.sum(le_frac / le_sg)) / jnp.sum(frac / cube)
        SG = cube / Kw

    MW, Tc, Pc = corr.critical_properties(Tb, SG, method)
    omega = corr.acentric_factor(Tb, Tc, Pc, SG)
    Tbr = Tb / Tc
    omega_vp = (-jnp.log(Pc / corr.P_ATM) - corr._lk_f0(Tbr)) / corr._lk_f1(Tbr)

    comp_sg = jnp.concatenate([le_sg, SG])
    comp_mw = jnp.concatenate([le_mw, MW])
    basis_frac = jnp.concatenate([le_frac, frac])
    if assay.basis == "volume":
        vol = basis_frac
        mass = vol * comp_sg / jnp.sum(vol * comp_sg)
    else:
        mass = basis_frac
        vol = (mass / comp_sg) / jnp.sum(mass / comp_sg)
    mole = (mass / comp_mw) / jnp.sum(mass / comp_mw)

    n_cuts = int(edges.shape[0]) - 1
    k = len(light_names)
    has_contaminants = any(getattr(assay, f) is not None for f, _ in CONTAMINANTS.values()) or any(
        getattr(assay, f"{c}_curve") is not None for c in ("sulfur", "nitrogen", "ccr"))
    contaminants = _contaminants(assay, Tb, mass, k) if has_contaminants else {}
    return Characterization(
        pseudo_names=tuple(f"{prefix}{i + 1:02d}" for i in range(n_cuts)),
        light_names=light_names,
        method=method,
        cut_edges=edges,
        Tb=Tb,
        SG=SG,
        MW=MW,
        Tc=Tc,
        Pc=Pc,
        omega=omega,
        omega_vp=omega_vp,
        Kw=corr.watson_k(Tb, SG),
        hvap_nb=corr.hvap_at_tb(Tb, Tc, Pc),
        cp_liquid_coeffs=corr.cp_liquid_coeffs(Tb, SG, MW),
        cp_ig_coeffs=corr.cp_ideal_gas_coeffs(Tb, SG, MW),
        component_MW=comp_mw,
        component_SG=comp_sg,
        volume_fraction=vol,
        mass_fraction=mass,
        mole_fraction=mole,
        **contaminants,
    )


def _characterize_heavy(assay: Assay, points: Array, method: str, prefix: str) -> Characterization:
    """:func:`characterize` for an assay with a :class:`HeavyEnd`."""
    he = assay.heavy_end
    curve = tbp_curve(assay)
    light_names = tuple(assay.light_ends)
    le_frac = jnp.asarray([assay.light_ends[n] for n in light_names], dtype=float).reshape(-1) / 100.0
    x_le = jnp.sum(le_frac)
    T_start = _heavy_start(assay, curve, x_le)
    T_max = jnp.asarray(he.T_max, dtype=float)

    if not _traced((points, T_start)):
        pts = np.asarray(points)
        if np.any(np.diff(pts) <= 0):
            raise ValueError("cut_points must be strictly increasing")
        if pts.size and (pts[0] <= float(T_start) or pts[-1] >= he.T_max):
            raise ValueError(
                f"cut_points must lie strictly between {float(T_start):.1f} K (where the "
                f"pseudo-components start) and heavy_end.T_max = {he.T_max:.1f} K")

    # Cuts up to T_max, then the lump from x(T_max) to 1.
    edges = jnp.concatenate([T_start[None], points, T_max[None]])
    x_edges = curve(edges).at[0].set(x_le)
    frac_cut = jnp.diff(x_edges)
    # Mean T over the cut, by parts. With x_le below the first data point
    # the curve is entered at that point, so the difference sits at T_start.
    Tb_cut = (edges[1:] * x_edges[1:] - edges[:-1] * x_edges[:-1]
              - curve.integral_between(edges[:-1], edges[1:])) / frac_cut
    frac_res = 1.0 - x_edges[-1]
    frac = jnp.concatenate([frac_cut, frac_res[None]])
    res_Tb = jnp.asarray(he.residue_Tb, dtype=float)
    Tb_all = jnp.concatenate([Tb_cut, res_Tb[None]])

    le_sg = jnp.asarray([LIGHT_END_SG[n] for n in light_names], dtype=float).reshape(-1)
    le_mw = jnp.asarray([_light_end_mw(n) for n in light_names], dtype=float).reshape(-1)
    cube = (1.8 * Tb_all) ** (1.0 / 3.0)
    fixed_res = he.residue_sg is not None
    res_sg = jnp.asarray(he.residue_sg, dtype=float) if fixed_res else None

    if assay.sg_curve is not None:
        mid, sgs = (jnp.asarray(a, dtype=float) / s for a, s in zip(assay.sg_curve, (100.0, 1.0)))
        x_all = jnp.concatenate([x_edges, jnp.ones(1)])
        SG = jnp.diff(_linear_integral(mid, sgs, x_all)) / frac
        if fixed_res:
            SG = SG.at[-1].set(res_sg)
    else:
        bulk = assay.bulk_sg
        # The lump is part of the balance: with its SG free it follows the
        # same Kw; with it fixed, it is a known term like the light ends.
        n_free = Tb_all.shape[0] - (1 if fixed_res else 0)
        f_free, c_free = frac[:n_free], cube[:n_free]
        if assay.basis == "volume":
            known = jnp.sum(le_frac * le_sg) + (frac_res * res_sg if fixed_res else 0.0)
            Kw = jnp.sum(f_free * c_free) / (bulk - known)
        else:
            known = jnp.sum(le_frac / le_sg) + (frac_res / res_sg if fixed_res else 0.0)
            Kw = (1.0 / bulk - known) / jnp.sum(f_free / c_free)
        SG = cube / Kw
        if fixed_res:
            SG = SG.at[-1].set(res_sg)

    SG_cut = SG[:-1]
    MW_cut, Tc_cut, Pc_cut = corr.critical_properties(Tb_cut, SG_cut, method)
    omega_cut = corr.acentric_factor(Tb_cut, Tc_cut, Pc_cut, SG_cut)
    # The lump: the correlation at T_max, at the last cut's Watson K, carried
    # to the lump's Tb at constant Tb/Tc (Pc and omega stay at the edge).
    sg_edge = (1.8 * T_max) ** (1.0 / 3.0) / corr.watson_k(Tb_cut[-1], SG_cut[-1])
    _, Tc_edge, Pc_edge = corr.critical_properties(T_max, sg_edge, method)
    omega_edge = corr.acentric_factor(T_max, Tc_edge, Pc_edge, sg_edge)

    def cat(a, b):
        return jnp.concatenate([a, jnp.atleast_1d(b)])

    MW = cat(MW_cut, jnp.asarray(he.residue_mw, dtype=float))
    Tc = cat(Tc_cut, Tc_edge * res_Tb / T_max)
    Pc = cat(Pc_cut, Pc_edge)
    omega = cat(omega_cut, omega_edge)
    Tbr = Tb_all / Tc
    omega_vp = (-jnp.log(Pc / corr.P_ATM) - corr._lk_f0(Tbr)) / corr._lk_f1(Tbr)

    comp_sg = jnp.concatenate([le_sg, SG])
    comp_mw = jnp.concatenate([le_mw, MW])
    basis_frac = jnp.concatenate([le_frac, frac])
    if assay.basis == "volume":
        vol = basis_frac
        mass = vol * comp_sg / jnp.sum(vol * comp_sg)
    else:
        mass = basis_frac
        vol = (mass / comp_sg) / jnp.sum(mass / comp_sg)
    mole = (mass / comp_mw) / jnp.sum(mass / comp_mw)

    k = len(light_names)
    top = T_max + he.lump_span * (T_max - edges[-2])
    n_cuts = int(edges.shape[0]) - 1
    return Characterization(
        pseudo_names=tuple(f"{prefix}{i + 1:02d}" for i in range(n_cuts)) + (f"{prefix}resid",),
        light_names=light_names,
        method=method,
        cut_edges=cat(edges, top),
        Tb=Tb_all,
        SG=SG,
        MW=MW,
        Tc=Tc,
        Pc=Pc,
        omega=omega,
        omega_vp=omega_vp,
        Kw=corr.watson_k(Tb_all, SG),
        hvap_nb=corr.hvap_at_tb(Tb_all, Tc, Pc),
        cp_liquid_coeffs=corr.cp_liquid_coeffs(Tb_all, SG, MW),
        cp_ig_coeffs=corr.cp_ideal_gas_coeffs(Tb_all, SG, MW),
        component_MW=comp_mw,
        component_SG=comp_sg,
        volume_fraction=vol,
        mass_fraction=mass,
        mole_fraction=mole,
        residue_lump=True,
        **_contaminants(assay, Tb_all, mass, k),
    )


def fit_antoine(
    Tc: float, Pc: float, omega: float,
    window: tuple[float, float] = DEFAULT_ANTOINE_WINDOW,
    Tb: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Antoine coefficients fitted to the Lee-Kesler vapour pressure.

    difflow's ``SpeciesData`` carries vapour pressure as Antoine
    ``log10(P/Pa) = A - B/(T + C)``, so a pseudo-component's Lee-Kesler curve
    is fitted to that form by least squares in ``log10 P`` over ``window``
    -- extended up to ``Tb`` if given, and starting no lower than where the
    vapour pressure reaches 10 Pa -- intersected with ``0.35 < T/Tc < 0.95``.
    Concrete (numpy/scipy), not traceable.

    Returns:
        ``((A, B, C), (T_min, T_max))``: the coefficients and the range they
        were fitted over.
    """
    from scipy.optimize import least_squares

    def log10_psat(T):
        Tr = np.asarray(T) / Tc
        f0 = 5.92714 - 6.09648 / Tr - 1.28862 * np.log(Tr) + 0.169347 * Tr**6
        f1 = 15.2518 - 15.6875 / Tr - 13.4721 * np.log(Tr) + 0.43577 * Tr**6
        return np.log10(Pc) + (f0 + omega * f1) / np.log(10.0)

    # Start where the vapour pressure becomes worth having: below
    # _ANTOINE_P_MIN a component's K-value is negligible, and fitting the
    # three-parameter form across many decades of it costs accuracy where the
    # component does distil. End at the window or at Tb, whichever is higher,
    # so a heavy cut's fit always reaches its own boiling point.
    T_lo = max(window[0], _ANTOINE_TR[0] * Tc)
    T_hi = min(max(window[1], Tb if Tb is not None else window[1]), _ANTOINE_TR[1] * Tc)
    if log10_psat(T_lo) < np.log10(_ANTOINE_P_MIN) < log10_psat(T_hi):
        from scipy.optimize import brentq

        T_lo = brentq(lambda t: log10_psat(t) - np.log10(_ANTOINE_P_MIN), T_lo, T_hi)
    if T_hi - T_lo < 20.0:
        # A cut whose critical point sits at the edge of the window: fit over
        # the reduced-temperature range alone rather than over a sliver.
        T_lo, T_hi = _ANTOINE_TR[0] * Tc, _ANTOINE_TR[1] * Tc
    T = np.linspace(T_lo, T_hi, _ANTOINE_POINTS)
    target = log10_psat(T)

    # Clausius-Clapeyron (C = 0) through the end points as the start.
    B0 = (target[0] - target[-1]) / (1.0 / T[-1] - 1.0 / T[0])
    A0 = target[0] + B0 / T[0]

    def resid(p):
        A, B, C = p
        return A - B / (T + C) - target

    sol = least_squares(resid, [A0, B0, 0.0], bounds=([-np.inf, 0.0, -0.9 * T_lo], np.inf),
                        x_scale=[1.0, B0, 10.0])
    return tuple(float(v) for v in sol.x), (float(T_lo), float(T_hi))
