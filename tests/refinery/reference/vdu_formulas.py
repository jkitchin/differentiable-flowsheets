"""The vacuum column's property model, written from the published equations.

The vacuum column (``difflow_refinery.vacuum``) has its own property model,
not the crude column's: Raoult K-values over **Maxwell-Bonnell** vapour
pressures with the Watson-K correction, liquid enthalpy from **Kesler-Lee's**
liquid heat capacity, vapour enthalpy one Clausius-Clapeyron heat of
vaporisation above it, and steam as a vapour-only ideal gas (Shomate). This
module is that model transcribed again, from the sources rather than from
difflow, over a pluggable math namespace (:data:`NUMPY`, :func:`pyomo_math`)
so the same text evaluates on floats and builds the IPOPT column's
constraints. Nothing here imports difflow.

Where the transcription is deliberately *not* difflow's text:

* **Rankine.** Maxwell-Bonnell and Kesler-Lee are published in degrees
  Rankine (Riazi, ASTM MNL50 (2005), Eqs. 7.22-7.26 and 7.49); difflow
  writes them in kelvin with the coefficients converted. Here they are in
  the published units and converted at the boundary, so a slip in either
  conversion is a disagreement.
* **The Watson-K correction as unknowns.** It shifts the boiling point by
  an amount that depends on the vapour pressure it produces, so it is a
  fixed point. difflow applies four passes from ``P = 760 mmHg``
  (``maxwell_bonnell_psat(n_iter=4)``) and differentiates the unrolled
  passes by AD. The reference poses each pass, and its T-derivative by the
  chain rule written out, as unknowns and constraints of the column --
  the same arithmetic, a different formulation -- and can instead pose the
  exact fixed point and its implicit derivative ``g_T / (1 - g_lp)``
  (:func:`mb_log10p` ``n_pass=None``). Four passes are not converged for
  the heaviest cuts far below their boiling point, so the generator solves
  the column both ways and reports what the truncation costs.
* **Enthalpy datums.** Hydrocarbon liquid enthalpy is the Kesler-Lee
  antiderivative with no reference subtracted, and steam is NIST's Shomate
  form with its own ``F - H`` constants; difflow puts both datums at
  298.15 K. In a column without reaction every component that enters leaves,
  so a per-component datum cancels from every balance and every duty; the
  two models agreeing with different datums is a check that it does.

What is the *same model* on purpose, as in :mod:`.formulas` for the crude
column: the join between Maxwell-Bonnell's three pressure ranges. The paper's
branches are discontinuous (by about 1% where they meet), which a Newton
solve and a derivative cannot use, so difflow blends them over a narrow
logistic and moves the upper join to ``Q = 1/748.1`` (exactly the corrected
boiling point). The reference uses the same blend, because layer 3 tests the
column; :func:`mb_log10_mmhg_published` is the paper's piecewise form, and
the generator re-solves the column on it to measure what the blend costs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

R = 8.314462618  # J/mol/K (CODATA 2018)
LN10 = math.log(10.0)
MMHG = 133.322368  # Pa
BTU_PER_LB = 2326.0  # J/kg (exact, from the International Table Btu)
MW_WATER = 18.01528

NUMPY = SimpleNamespace(exp=np.exp, log=np.log, where=np.where)


def pyomo_math():
    """``exp``, ``log`` and a ``where`` that builds a Pyomo ``Expr_if``."""
    import pyomo.environ as pyo

    def where(cond, a, b):
        return pyo.Expr_if(IF=cond, THEN=a, ELSE=b)

    return SimpleNamespace(exp=pyo.exp, log=pyo.log, where=where)


# -----------------------------------------------------------------------------
# Maxwell-Bonnell (1957), as given in Riazi MNL50 Eqs. 7.22-7.26, Rankine
# -----------------------------------------------------------------------------

#: The three branches, ``log10 P(mmHg) = (a Q + b) / (c Q + d)``:
#: Eq. 7.22 (P < 2 mmHg, Q > 0.0022), 7.23 (2-760 mmHg) and 7.24 (P > 760).
MB_LOW = (3000.538, -6.761560, 43.0, -0.987672)
MB_MID = (2663.129, -5.994296, 95.76, -0.972546)
MB_HIGH = (2770.085, -6.412631, 36.0, -0.989679)
#: The published break points in Q.
Q_PUB_LOW, Q_PUB_HIGH = 0.0022, 0.0013
#: The blend the column uses (see module docstring): lower join at the
#: published 0.0022, upper join where T is the corrected normal boiling point.
Q_JOIN_LOW, Q_JOIN_HIGH, Q_WIDTH = 0.0022, 1.0 / 748.1, 1.5e-5
#: Beyond this many widths the logistic is 0 or 1 to 1e-200; the cut keeps
#: ``exp`` from overflowing at IPOPT's trial points.
_Z_CUT = 500.0


def mb_q(Tb_corr_K, T_K):
    """Maxwell-Bonnell's Q from the corrected normal boiling point, Eq. 7.25,
    ``Q = (Tb'/T - 0.0002867 Tb') / (748.1 - 0.2145 Tb')`` with both in Rankine."""
    TbR, TR = 1.8 * Tb_corr_K, 1.8 * T_K
    return (TbR / TR - 0.0002867 * TbR) / (748.1 - 0.2145 * TbR)


def _branch(Q, c):
    a, b, cc, d = c
    return (a * Q + b) / (cc * Q + d)


def _dbranch(Q, c):
    a, b, cc, d = c
    return (a * d - b * cc) / (cc * Q + d) ** 2


def _logistic(m, z):
    """``1 / (1 + exp(z))``, cut off before ``exp`` overflows."""
    return m.where(z > _Z_CUT, 0.0, m.where(z < -_Z_CUT, 1.0, 1.0 / (1.0 + m.exp(z))))


def mb_log10_mmhg(m, Q):
    """``log10 P`` (mmHg) on the blended branches, and its derivative in Q."""
    lo, mid, hi = _branch(Q, MB_LOW), _branch(Q, MB_MID), _branch(Q, MB_HIGH)
    dlo, dmid, dhi = _dbranch(Q, MB_LOW), _dbranch(Q, MB_MID), _dbranch(Q, MB_HIGH)
    w_lo = _logistic(m, -(Q - Q_JOIN_LOW) / Q_WIDTH)  # -> 1 above the low join
    w_hi = _logistic(m, (Q - Q_JOIN_HIGH) / Q_WIDTH)  # -> 1 below the high join
    dw_lo = w_lo * (1.0 - w_lo) / Q_WIDTH
    dw_hi = -w_hi * (1.0 - w_hi) / Q_WIDTH
    core = w_lo * lo + (1.0 - w_lo) * mid
    dcore = dw_lo * (lo - mid) + w_lo * dlo + (1.0 - w_lo) * dmid
    val = w_hi * hi + (1.0 - w_hi) * core
    dval = dw_hi * (hi - core) + w_hi * dhi + (1.0 - w_hi) * dcore
    return val, dval


def mb_log10_mmhg_piecewise(m, Q):
    """The paper's piecewise form and its derivative: discontinuous at the
    joins (by about 1 %), smooth everywhere else."""
    def pick(lo, mid, hi):
        return m.where(Q > Q_PUB_LOW, lo, m.where(Q >= Q_PUB_HIGH, mid, hi))

    return (pick(_branch(Q, MB_LOW), _branch(Q, MB_MID), _branch(Q, MB_HIGH)),
            pick(_dbranch(Q, MB_LOW), _dbranch(Q, MB_MID), _dbranch(Q, MB_HIGH)))


def mb_log10_mmhg_published(Q):
    """The paper's piecewise form on floats (value only)."""
    return mb_log10_mmhg_piecewise(NUMPY, np.asarray(Q, dtype=float))[0]


def watson_f(Tb_K):
    """MNL50 Eq. 7.26's ``f``: 0 below Tb = 200 F, 1 above 400 F, linear between."""
    TbF = 1.8 * np.asarray(Tb_K, dtype=float) - 459.67
    return np.clip((TbF - 200.0) / 200.0, 0.0, 1.0)


def corrected_tb(Tb_K, Kw, f, log10p_mmhg):
    """``Tb' = Tb - dTb`` (K), Eq. 7.26: ``dTb = 2.5 f (Kw - 12) log10(P/760)``
    in Rankine."""
    return Tb_K - 2.5 * f * (Kw - 12.0) * (log10p_mmhg - math.log10(760.0)) / 1.8


def mb_pass(m, lp, T, Tb, Kw, f, published=False):
    """One pass of the Watson-K correction: ``g(T, lp) = log10 psat(T; Tb'(lp))``.

    Returns ``(g, dg/dT, dg/dlp)``, ``lp`` and ``g`` in log10 mmHg. The
    vapour pressure is the fixed point ``lp = g(T, lp)``; difflow takes four
    passes from ``lp = log10 760`` (``n_pass=4`` below), the exact fixed point
    is ``n_pass=None``. Both are posed in the column, the first as one
    unknown per pass, the second as one unknown and one constraint
    ``lp = g``. ``published`` takes the paper's unblended branches.
    """
    Tbc = corrected_tb(Tb, Kw, f, lp)
    TbR, TR = 1.8 * Tbc, 1.8 * T
    den = 748.1 - 0.2145 * TbR
    Q = (TbR / TR - 0.0002867 * TbR) / den
    g, dQ = (mb_log10_mmhg_piecewise if published else mb_log10_mmhg)(m, Q)
    dQ_dTR = -TbR / (TR ** 2 * den)
    dQ_dTbR = 748.1 * (1.0 / TR - 0.0002867) / den ** 2
    dTbR_dlp = -2.5 * f * (Kw - 12.0)
    return g, dQ * dQ_dTR * 1.8, dQ * dQ_dTbR * dTbR_dlp


#: difflow's number of fixed-point passes (``maxwell_bonnell_psat(n_iter=4)``),
#: the reference's default; ``None`` solves the fixed point exactly.
N_PASS = 4
LP_START = math.log10(760.0)


def mb_log10p(T, Tb, Kw, n_pass=N_PASS, published=False):
    """``log10 psat`` (mmHg) and its T-derivative (1/K) on floats.

    ``n_pass`` passes of the Watson-K correction from ``log10 760``, the
    derivative carried through them by the chain rule; ``n_pass=None`` is
    the exact fixed point (iterated to round-off) and its implicit
    derivative ``g_T / (1 - g_lp)``. ``published`` uses the paper's
    piecewise branches.
    """
    T = np.asarray(T, dtype=float)
    Tb = np.asarray(Tb, dtype=float)
    Kw = np.asarray(Kw, dtype=float)
    f = watson_f(Tb)
    shape = np.broadcast_shapes(T.shape, Tb.shape)
    lp, d = np.full(shape, LP_START), np.zeros(shape)
    for k in range(n_pass or 200):
        g, gT, glp = mb_pass(NUMPY, lp, T, Tb, Kw, f, published)
        if n_pass is None:
            if np.max(np.abs(g - lp)) < 1e-12:
                return g, gT / (1.0 - glp)
        lp, d = g, gT + glp * d
    if n_pass is None:
        raise RuntimeError("Maxwell-Bonnell fixed point did not converge")
    return lp, d


# -----------------------------------------------------------------------------
# Kesler-Lee (1976) liquid heat capacity, MNL50 Eq. 7.49: Btu/(lb F), T in R
# -----------------------------------------------------------------------------


def kesler_lee_coefficients(SG, Kw):
    A1 = -1.17126 + (0.023722 + 0.024907 * SG) * Kw + (1.14982 - 0.046535 * Kw) / SG
    A2 = (1.0 + 0.82463 * Kw) * (1.12172 - 0.27634 / SG) * 1e-4
    A3 = -(1.0 + 0.82463 * Kw) * (2.9027 - 0.70958 / SG) * 1e-8
    return A1, A2, A3


def h_liquid(T_K, SG, Kw, MW):
    """Liquid enthalpy (J/mol): ``integral Cp dT`` in Btu/lb with T in Rankine,
    datum 0 R (see module docstring for why a datum may be anything)."""
    A1, A2, A3 = kesler_lee_coefficients(SG, Kw)
    TR = 1.8 * T_K
    return (A1 * TR + A2 * TR ** 2 / 2.0 + A3 * TR ** 3 / 3.0) * BTU_PER_LB * MW / 1000.0


def dh_vap(T_K, dlog10p_dT):
    """Heat of vaporisation (J/mol), Clausius-Clapeyron on the same psat:
    ``R T^2 dln psat/dT``."""
    return R * T_K ** 2 * LN10 * dlog10p_dT


#: NIST Chemistry WebBook, water vapour, Shomate 500-1700 K (Chase 1998):
#: A..H, ``H - H298 = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F - H`` kJ/mol.
SHOMATE_WATER = dict(A=30.09200, B=6.832514, C=6.793435, D=-2.534480, E=0.082139,
                     F=-250.8810, H=-241.8264)


def h_steam(T_K):
    """Steam enthalpy (J/mol), NIST's Shomate form with its own constants."""
    s = SHOMATE_WATER
    t = T_K / 1000.0
    return 1000.0 * (s["A"] * t + s["B"] * t ** 2 / 2 + s["C"] * t ** 3 / 3
                     + s["D"] * t ** 4 / 4 - s["E"] / t + s["F"] - s["H"])


# -----------------------------------------------------------------------------
# On floats: K-values, enthalpies, flash, product inspection
# -----------------------------------------------------------------------------


class Props:
    """The model on numpy arrays, over a component table ``comp`` (lists)."""

    def __init__(self, comp: dict, n_pass=N_PASS, published=False):
        self.Tb = np.asarray(comp["Tb"], dtype=float)
        self.SG = np.asarray(comp["SG"], dtype=float)
        self.MW = np.asarray(comp["MW"], dtype=float)
        self.Kw = np.asarray(comp["Kw"], dtype=float)
        self.f = watson_f(self.Tb)
        self.n_pass = n_pass
        self.published = published

    def log10p(self, T):
        return mb_log10p(T, self.Tb, self.Kw, self.n_pass, self.published)[0]

    def psat(self, T):
        """Vapour pressure (Pa)."""
        return 10.0 ** self.log10p(T) * MMHG

    def K(self, T, P):
        return self.psat(T) / P

    def hL(self, T):
        return h_liquid(T, self.SG, self.Kw, self.MW)

    def dhvap(self, T):
        return dh_vap(T, mb_log10p(T, self.Tb, self.Kw, self.n_pass, self.published)[1])

    def hV(self, T):
        return self.hL(T) + self.dhvap(T)


def rachford_rice(z, K):
    """Vapour fraction of an ideal flash; 0 or 1 outside the two-phase range."""
    from scipy.optimize import brentq

    z, K = np.asarray(z), np.asarray(K)
    if np.sum(z * K) <= 1.0:
        return 0.0
    if np.sum(z / K) <= 1.0:
        return 1.0

    def g(b):
        return np.sum(z * (K - 1) / (1 + b * (K - 1)))

    lo = max(0.0, np.max(1 / (1 - K[K > 1])) if np.any(K > 1) else 0.0) + 1e-14
    hi = min(1.0, np.min(1 / (1 - K[K < 1])) if np.any(K < 1) else 1.0) - 1e-14
    return brentq(g, max(lo, 0.0), min(hi, 1.0), xtol=1e-15)


def product_inspection(comp: dict, mol, percents=(5, 10, 50, 90, 95)) -> dict:
    """Mass rate (kg/s), SG (volume-additive) and TBP points of a product.

    The TBP curve treats each cut as distilling uniformly between its edges
    ``T_lo`` and ``T_hi``, the residue lump between its own; the curve is
    then read by linear interpolation of cumulative mass fraction.
    """
    MW, SG = np.asarray(comp["MW"]), np.asarray(comp["SG"])
    mass = np.asarray(mol) * MW / 1000.0
    total = float(mass.sum())
    w = mass / total
    edges = np.concatenate([[comp["T_lo"][0]], comp["T_hi"]])
    cum = np.concatenate([[0.0], np.cumsum(w)])
    return {"mass": total, "SG": float(1.0 / np.sum(w / SG)),
            "TBP": {str(p): float(np.interp(p / 100.0, cum, edges)) for p in percents}}
