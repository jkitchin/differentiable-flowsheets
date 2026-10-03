"""Alkylation yield, octane and acid correlations of Sauer, Colville & Burwick.

Sauer, Colville & Burwick (1964) regressed the yield, octane and acid
consumption of one 1960s sulfuric-acid alkylation plant on its operating
variables; Bracken & McCormick (1968, Ch. 4) restated the regressions as an
optimisation problem, and the GAMS Model Library carries that problem as
``process.gms`` (SEQ=20, "Alkylation Process Optimization"). The equations
and coefficients below are transcribed from ``process.gms`` as published by
GAMS (checked against GAMS's own GAMSPy translation of it,
``GAMS-dev/gamspy-examples``, ``models/process/process.py``), in the units
of that model:

====================  =========================================================
``olefin``            olefin feed, bbl/d
``isor``              isobutane recycle, bbl/d
``isom``              isobutane makeup, bbl/d
``alkylate``          alkylate yield, bbl/d
``acid``              acid addition rate, 1000 lb/d
``strength``          acid strength, wt% H2SO4
``octane``            motor octane number
``ratio``             external isobutane-to-olefin ratio, ``(isor + isom) /
                      olefin`` by volume (``process.gms`` labels it "isobutane
                      makeup to olefin ratio", but defines it as this)
``dilute``            acid dilution factor
``f4``                F-4 performance number
====================  =========================================================

The equations (``process.gms`` equation names in brackets)::

    alkylate = olefin (1.12 + 0.13167 ratio - 0.00667 ratio^2)        [yield1]
    alkylate = olefin + isom - 0.22 alkylate                           [makeup]
    acid     = alkylate dilute strength / (98 - strength) / 1000       [sdef]
    octane   = 86.35 + 1.098 ratio - 0.038 ratio^2 - 0.325 (89 - strength)
                                                                       [motor]
    ratio    = (isor + isom) / olefin                                  [drat]
    dilute   = 35.82 - 0.222 f4                                        [ddil]
    f4       = -133 + 3 octane                                         [df4]
    profit   = 0.063 alkylate octane - 5.04 olefin - 0.035 isor
               - 10 acid - 3.36 isom                                   [dprofit]

``rproc`` is the same model with each regression allowed a +/-10 % error
(a range variable multiplying its left-hand side, bounded to [0.9, 1.1]) --
Bracken & McCormick's device for regression uncertainty.

These are class (b) correlations in the issue's taxonomy: regressed on one
plant, illustrative and not predictive for a modern unit. Refit them to your
own unit's data (:mod:`difflow.estimation`) before planning with them. The
functions take a :class:`SauerCorrelation` so that a refitted coefficient set
replaces the published one wholesale.

References:
    Sauer, R.N., Colville, A.R., Burwick, C.W. (1964), "Computer points the
    way to more profits", Hydrocarbon Processing 43(3), 84 (volume, issue
    and page unverified -- the issue's citation; no copy could be opened).

    Bracken, J., McCormick, G.P. (1968), Selected Applications of Nonlinear
    Programming, John Wiley & Sons, New York, Chapter 4 (as cited by
    ``process.gms``; the book itself was not opened).

    GAMS Model Library, ``process.gms``, "Alkylation Process Optimization"
    (SEQ=20); the reference optimum of the ``process`` model, profit
    1161.3366 $/d, is the primal bound MINLPLib lists for its instance
    ``process`` (-1161.33660200, minimisation sign).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin


@dataclass(repr=False)
class SauerCorrelation(ParamsMixin):
    """Coefficients of the Sauer-Colville-Burwick correlations.

    The defaults are ``process.gms``'s. The yield and octane correlations are
    in the external isobutane/olefin volume ratio ``r`` and the acid strength
    ``S`` (wt%).

    Attributes:
        y0: Yield intercept, vol alkylate / vol olefin (1.12).
        y1: Yield coefficient of ``r`` (0.13167).
        y2: Yield coefficient of ``r^2``, subtracted (0.00667).
        m0: Motor octane intercept (86.35).
        m1: Motor octane coefficient of ``r`` (1.098).
        m2: Motor octane coefficient of ``r^2``, subtracted (0.038).
        mS: Motor octane coefficient of the acid-strength deficit
            ``89 - S``, subtracted (0.325).
        S_ref: Acid strength at which the strength term vanishes (89 wt%).
        f0: F-4 performance number intercept (-133).
        f1: F-4 performance number per motor octane number (3).
        d0: Acid dilution factor intercept (35.82).
        d1: Acid dilution factor per F-4 number, subtracted (0.222).
        S_max: The ``98`` in the acid equation: the strength (wt%) of the
            makeup acid the spent acid is diluted from (``98 - S`` is the
            strength lost per pound).
        shrink: The ``0.22`` in the isobutane makeup volume balance
            ``alkylate = olefin + isom - shrink * alkylate``.
    """

    y0: float = 1.12
    y1: float = 0.13167
    y2: float = 0.00667
    m0: float = 86.35
    m1: float = 1.098
    m2: float = 0.038
    mS: float = 0.325
    S_ref: float = 89.0
    f0: float = -133.0
    f1: float = 3.0
    d0: float = 35.82
    d1: float = 0.222
    S_max: float = 98.0
    shrink: float = 0.22


#: The published coefficient set (``process.gms``).
SAUER_1964 = SauerCorrelation()


def alkylate_yield(r: Array, c: SauerCorrelation = SAUER_1964) -> Array:
    """Alkylate volume per volume of olefin, ``y0 + y1 r - y2 r^2`` [yield1]."""
    return c.y0 + c.y1 * r - c.y2 * r**2


def max_alkylate_yield(c: SauerCorrelation = SAUER_1964) -> float:
    """The vertex of :func:`alkylate_yield`: its largest value, ``y0 + y1^2/(4 y2)``."""
    return c.y0 + c.y1**2 / (4.0 * c.y2)


def motor_octane(r: Array, strength: Array, c: SauerCorrelation = SAUER_1964) -> Array:
    """Alkylate motor octane number [motor].

    ``m0 + m1 r - m2 r^2 - mS (S_ref - strength)``.
    """
    return c.m0 + c.m1 * r - c.m2 * r**2 - c.mS * (c.S_ref - strength)


def f4_performance(mon: Array, c: SauerCorrelation = SAUER_1964) -> Array:
    """F-4 performance number from the motor octane number [df4]."""
    return c.f0 + c.f1 * mon


def acid_dilution(f4: Array, c: SauerCorrelation = SAUER_1964) -> Array:
    """Acid dilution factor from the F-4 performance number [ddil]."""
    return c.d0 - c.d1 * f4


def acid_per_alkylate(mon: Array, strength: Array, c: SauerCorrelation = SAUER_1964) -> Array:
    """Acid consumption, lb of makeup acid per bbl of alkylate [sdef, ddil, df4].

    ``dilute * S / (S_max - S)``, with ``dilute`` from the octane through the
    F-4 performance number. ``process.gms`` writes the plant total in
    1000 lb/d: ``acid = alkylate * this / 1000``.
    """
    dilute = acid_dilution(f4_performance(mon, c), c)
    return dilute * strength / (c.S_max - strength)


def isobutane_makeup(alkylate: Array, olefin: Array, c: SauerCorrelation = SAUER_1964) -> Array:
    """Isobutane makeup (bbl/d) the volume balance needs [makeup].

    ``isom = (1 + shrink) alkylate - olefin``.
    """
    return (1.0 + c.shrink) * alkylate - olefin


# =============================================================================
# The process.gms optimisation problem
# =============================================================================

#: ``process.gms``'s variable order, bounds and starting point.
PROCESS_VARIABLES: tuple[str, ...] = (
    "olefin", "isor", "acid", "alkylate", "isom", "strength", "octane",
    "ratio", "dilute", "f4",
)
PROCESS_BOUNDS: dict[str, tuple[float, float]] = {
    "olefin": (10.0, 2000.0), "isor": (0.0, 16000.0), "acid": (0.0, 120.0),
    "alkylate": (0.0, 5000.0), "isom": (0.0, 2000.0), "strength": (85.0, 93.0),
    "octane": (90.0, 95.0), "ratio": (3.0, 12.0), "dilute": (1.2, 4.0),
    "f4": (145.0, 162.0),
}
PROCESS_START: dict[str, float] = {
    "olefin": 1745.0, "isor": 12000.0, "acid": 110.0, "alkylate": 3048.0,
    "isom": 1974.0, "strength": 89.2, "octane": 92.8, "ratio": 8.0,
    "dilute": 3.6, "f4": 145.0,
}
#: ``process.gms`` prices: alkylate value per octane-barrel, olefin feed,
#: isobutane recycle (handling), acid, isobutane makeup ($ per bbl or per
#: 1000 lb).
PROCESS_PRICES: dict[str, float] = {
    "alkylate_octane": 0.063, "olefin": 5.04, "isor": 0.035, "acid": 10.0,
    "isom": 3.36,
}
#: The ranged model's +/-10 % regression bands.
RANGE_BOUNDS = (0.9, 1.1)
#: The ``process`` model's optimum (profit, $/d): MINLPLib's primal bound for
#: its ``process`` instance, -1161.33660200 in the minimisation sign.
PROCESS_OPTIMUM_PROFIT = 1161.33660200

_SCALE = np.array([1000.0, 10000.0, 100.0, 1000.0, 1000.0, 10.0, 10.0, 1.0, 1.0, 100.0])


def process_profit(v: Mapping[str, Array], prices: Mapping[str, float] = PROCESS_PRICES) -> Array:
    """``process.gms``'s profit ($/d) of a point ``{variable: value}`` [dprofit]."""
    p = prices
    return (p["alkylate_octane"] * v["alkylate"] * v["octane"]
            - p["olefin"] * v["olefin"] - p["isor"] * v["isor"]
            - p["acid"] * v["acid"] - p["isom"] * v["isom"])


def process_residuals(v: Mapping[str, Array], c: SauerCorrelation = SAUER_1964,
                      ranges: Mapping[str, Array] | None = None) -> Array:
    """The seven ``process.gms`` equality rows, ``lhs - rhs`` in their own units.

    ``ranges`` (``rangey``, ``rangem``, ``ranged``, ``rangef``) multiply the
    left-hand sides of the four regressions, as in ``rproc``; omitted, they
    are 1 (the ``process`` model).
    """
    rg = {"rangey": 1.0, "rangem": 1.0, "ranged": 1.0, "rangef": 1.0}
    if ranges is not None:
        rg.update(ranges)
    r, S = v["ratio"], v["strength"]
    return jnp.stack([
        rg["rangey"] * v["alkylate"] - v["olefin"] * alkylate_yield(r, c),
        v["alkylate"] - (v["olefin"] + v["isom"] - c.shrink * v["alkylate"]),
        v["acid"] - v["alkylate"] * v["dilute"] * S / (c.S_max - S) / 1000.0,
        rg["rangem"] * v["octane"] - motor_octane(r, S, c),
        r - (v["isor"] + v["isom"]) / v["olefin"],
        rg["ranged"] * v["dilute"] - (c.d0 - c.d1 * v["f4"]),
        rg["rangef"] * v["f4"] - f4_performance(v["octane"], c),
    ])


@dataclass
class ProcessSolution:
    """The solution of the ``process.gms`` problem.

    Attributes:
        profit: Optimal profit ($/d).
        values: ``{variable: value}`` at the optimum (and the range variables
            for the ranged model).
        converged: The interior-point solver's verdict.
        ranged: Whether this is the ``rproc`` (ranged) model.
        residual: Max-norm of the scaled equality rows at the solution.
        result: The :class:`difflow_power.ipm.IPMResult`.
    """

    profit: float
    values: dict[str, float]
    converged: bool
    ranged: bool
    residual: float
    result: object = field(repr=False, default=None)


def solve_process_gms(ranged: bool = False, c: SauerCorrelation = SAUER_1964,
                      prices: Mapping[str, float] = PROCESS_PRICES,
                      start: Mapping[str, float] | None = None) -> ProcessSolution:
    """Solve Bracken & McCormick's alkylation problem, as in ``process.gms``.

    Gradient-based: the primal-dual interior-point NLP solver of
    :mod:`difflow_power.ipm` (written in JAX; exact Hessians from
    ``jax.hessian``), on variables scaled to order one and rows scaled to
    comparable size -- the scaling changes neither the feasible set nor the
    optimum. Starts from ``process.gms``'s initial levels.

    Args:
        ranged: Solve ``rproc`` (each regression +/-10 %) instead of
            ``process``.
        c: The correlation coefficients.
        prices: The profit coefficients.
        start: Starting levels; default ``process.gms``'s.

    Returns:
        A :class:`ProcessSolution`.
    """
    from difflow_power.ipm import NLP, solve_nlp

    names = list(PROCESS_VARIABLES)
    scale = list(_SCALE)
    lo = [PROCESS_BOUNDS[n][0] for n in names]
    hi = [PROCESS_BOUNDS[n][1] for n in names]
    x0 = [dict(PROCESS_START, **(start or {}))[n] for n in names]
    range_names = ["rangey", "rangem", "ranged", "rangef"] if ranged else []
    for _ in range_names:
        scale.append(1.0)
        lo.append(RANGE_BOUNDS[0])
        hi.append(RANGE_BOUNDS[1])
        x0.append(1.0)
    scale = jnp.asarray(scale)
    lo = jnp.asarray(lo) / scale
    hi = jnp.asarray(hi) / scale
    x0 = jnp.asarray(x0) / scale
    row_scale = jnp.array([1000.0, 1000.0, 100.0, 10.0, 1.0, 1.0, 100.0])

    def unpack(y):
        x = y * scale
        v = dict(zip(names, x[:len(names)]))
        rg = dict(zip(range_names, x[len(names):]))
        return v, rg

    def objective(y, _):
        v, _rg = unpack(y)
        return -process_profit(v, prices) / 1000.0

    def equalities(y, _):
        v, rg = unpack(y)
        return process_residuals(v, c, rg) / row_scale

    def inequalities(y, _):
        return jnp.concatenate([lo - y, y - hi])

    n = int(x0.shape[0])
    nlp = NLP(objective=objective, n=n, m_eq=7, m_in=2 * n,
              equalities=equalities, inequalities=inequalities)
    res = solve_nlp(nlp, x0)
    v, rg = unpack(res.x)
    values = {k: float(x) for k, x in {**v, **rg}.items()}
    resid = float(jnp.max(jnp.abs(equalities(res.x, None))))
    return ProcessSolution(profit=float(process_profit(v, prices)), values=values,
                           converged=bool(res.converged), ranged=ranged,
                           residual=resid, result=res)


__all__ = [
    "PROCESS_BOUNDS", "PROCESS_OPTIMUM_PROFIT", "PROCESS_PRICES", "PROCESS_START",
    "PROCESS_VARIABLES", "RANGE_BOUNDS", "SAUER_1964", "ProcessSolution",
    "SauerCorrelation", "acid_dilution", "acid_per_alkylate", "alkylate_yield",
    "f4_performance", "isobutane_makeup", "max_alkylate_yield", "motor_octane",
    "process_profit", "process_residuals", "solve_process_gms",
]
