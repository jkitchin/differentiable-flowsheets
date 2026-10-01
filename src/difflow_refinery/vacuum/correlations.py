"""Property correlations for petroleum pseudocomponents.

Everything here is a pure ``jax.numpy`` function of a pseudocomponent's
normal boiling point ``Tb`` (K) and specific gravity ``SG`` (60/60 F), so a
derivative taken with respect to an assay's TBP curve or bulk gravity
passes straight through the characterization.

What is here and why it was chosen for a *vacuum* column:

- :func:`twu_critical_properties` -- Twu (1984) Tc, Pc, Vc and molecular
  weight. Twu's n-alkane reference reaches higher carbon numbers than
  Riazi-Daubert and Lee-Kesler, so it is the one that stays sane past
  565 C TBP. :func:`riazi_daubert_mw` is kept as a cross-check (it is only
  fitted to Tb of about 850 K).
- :func:`lee_kesler_acentric` -- Kesler-Lee (1976) acentric factor, both
  branches. Reported, not used in the VLE.
- :func:`maxwell_bonnell_psat` -- Maxwell-Bonnell (1957) vapor pressure,
  which is the correlation ASTM D1160 / D2892 use to convert a boiling point
  measured under vacuum to its atmospheric equivalent. It is built from a
  boiling point, so it is anchored where an assay is measured, and it is
  fitted down to a fraction of a mmHg. Corresponding-states vapor pressures
  (Lee-Kesler with Twu criticals) drift at the reduced temperatures of a
  VDU flash zone (Tr of about 0.6 for the heavy end), which is the range
  the issue asks to stay accurate in.
- :func:`kesler_lee_liquid_cp` -- liquid heat capacity of a petroleum
  fraction. The vapor enthalpy is the liquid's plus the heat of
  vaporization from the Clausius-Clapeyron slope of the *same*
  Maxwell-Bonnell curve (:mod:`difflow_refinery.vacuum.thermo`), so the energy
  balance and the K-values cannot disagree about volatility.

References:
    Twu, C. H. (1984) Fluid Phase Equilib. 16, 137-150.
    Kesler, M. G.; Lee, B. I. (1976) Hydrocarbon Process. 55(3), 153-158.
    Maxwell, J. B.; Bonnell, L. S. (1957) Ind. Eng. Chem. 49, 1187-1196.
    Riazi, M. R. (2005) Characterization and Properties of Petroleum
    Fractions, ASTM MNL50, chapters 2, 4 and 7.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: mmHg -> Pa
MMHG = 133.322368
#: 1 atm in mmHg, the Maxwell-Bonnell reference pressure
ATM_MMHG = 760.0
#: Btu/(lb R) -> J/(kg K)
BTU_LB_R = 4186.8
R_GAS = 8.314462618


def watson_k(Tb, SG):
    """Watson characterization factor ``Kw = (1.8 Tb)^(1/3) / SG`` (Tb in K)."""
    return (1.8 * Tb) ** (1.0 / 3.0) / SG


def sg_from_watson_k(Tb, Kw):
    """Specific gravity of a cut with boiling point ``Tb`` (K) at a given Kw."""
    return (1.8 * Tb) ** (1.0 / 3.0) / Kw


def api_from_sg(SG):
    """API gravity from specific gravity (60/60 F)."""
    return 141.5 / SG - 131.5


def sg_from_api(api):
    """Specific gravity (60/60 F) from API gravity."""
    return 141.5 / (api + 131.5)


# ---------------------------------------------------------------------------
# Molecular weight and critical properties
# ---------------------------------------------------------------------------


def riazi_daubert_mw(Tb, SG):
    """Molecular weight, extended Riazi-Daubert (1987), Tb in K.

    Riazi MNL50 Eq. 2.51. Fitted for M 70-700 (Tb up to about 850 K); used
    here as a cross-check on :func:`twu_critical_properties`.
    """
    return (42.965 * jnp.exp(2.097e-4 * Tb - 7.78712 * SG
                             + 2.08476e-3 * Tb * SG)
            * Tb ** 1.26007 * SG ** 4.98308)


def _twu_alkane_mw(Tb_R):
    """Molecular weight of the n-alkane boiling at ``Tb_R`` (Rankine).

    Twu's reference relation gives Tb as a function of ``theta = ln M``;
    it is inverted here by Newton's method from Twu's own starting guess.
    Unrolled, so it differentiates like any other expression.
    """
    theta = jnp.log(Tb_R / (10.44 - 0.0052 * Tb_R))
    for _ in range(12):
        e = jnp.exp(5.71419 + 2.71579 * theta - 0.286590 * theta ** 2
                    - 39.8544 / theta - 0.122488 / theta ** 2)
        f = e - 24.7522 * theta + 35.3155 * theta ** 2 - Tb_R
        df = (e * (2.71579 - 0.57318 * theta + 39.8544 / theta ** 2
                   + 0.244976 / theta ** 3)
              - 24.7522 + 70.631 * theta)
        theta = theta - f / df
    return jnp.exp(theta)


def twu_critical_properties(Tb, SG):
    """Twu (1984) critical properties and molecular weight.

    Args:
        Tb: normal boiling point (K)
        SG: specific gravity (60/60 F)

    Returns:
        dict with ``Tc`` (K), ``Pc`` (Pa), ``Vc`` (m^3/mol) and ``MW``
        (g/mol), each broadcast to the shape of the inputs.
    """
    Tb_R = 1.8 * Tb
    Tc0 = Tb_R / (0.533272 + 0.191017e-3 * Tb_R + 0.779681e-7 * Tb_R ** 2
                  - 0.284376e-10 * Tb_R ** 3 + 0.959468e28 / Tb_R ** 13)
    a = 1.0 - Tb_R / Tc0
    Pc0 = (3.83354 + 1.19629 * a ** 0.5 + 34.8888 * a + 36.1952 * a ** 2
           + 104.193 * a ** 4) ** 2
    Vc0 = (1.0 - (0.419869 - 0.505839 * a - 1.56436 * a ** 3
                  - 9481.70 * a ** 14)) ** (-8)
    SG0 = 0.843593 - 0.128624 * a - 3.36159 * a ** 3 - 13749.5 * a ** 12
    M0 = _twu_alkane_mw(Tb_R)
    rt = Tb_R ** 0.5

    dT = jnp.exp(5.0 * (SG0 - SG)) - 1.0
    fT = dT * (-0.362456 / rt + (0.0398285 - 0.948125 / rt) * dT)
    Tc = Tc0 * ((1 + 2 * fT) / (1 - 2 * fT)) ** 2

    dV = jnp.exp(4.0 * (SG0 ** 2 - SG ** 2)) - 1.0
    fV = dV * (0.466590 / rt + (-0.182421 + 3.01721 / rt) * dV)
    Vc = Vc0 * ((1 + 2 * fV) / (1 - 2 * fV)) ** 2

    dP = jnp.exp(0.5 * (SG0 - SG)) - 1.0
    fP = dP * ((2.53262 - 46.1955 / rt - 0.00127885 * Tb_R)
               + (-11.4277 + 252.140 / rt + 0.00230535 * Tb_R) * dP)
    Pc = Pc0 * (Tc / Tc0) * (Vc0 / Vc) * ((1 + 2 * fP) / (1 - 2 * fP)) ** 2

    dM = jnp.exp(5.0 * (SG0 - SG)) - 1.0
    x = jnp.abs(0.0123420 - 0.328086 / rt)
    fM = dM * (x + (-0.0175691 + 0.193168 / rt) * dM)
    MW = jnp.exp(jnp.log(M0) * ((1 + 2 * fM) / (1 - 2 * fM)) ** 2)

    return {
        "Tc": Tc / 1.8,
        "Pc": Pc * 6894.757,               # psia -> Pa
        "Vc": Vc * 0.0283168466 / 453.59237,  # ft3/lbmol -> m3/mol
        "MW": MW,
    }


def lee_kesler_acentric(Tb, Tc, Pc, Kw):
    """Kesler-Lee (1976) acentric factor.

    The ``Tb/Tc > 0.8`` branch (Kesler-Lee's own, in Kw) is the one heavy
    fractions use; the vapor-pressure branch below it is Lee-Kesler's.

    Args:
        Tb, Tc: normal boiling and critical temperature (K)
        Pc: critical pressure (Pa)
        Kw: Watson characterization factor
    """
    Tbr = Tb / Tc
    lnPbr = jnp.log(Pc / 101325.0)
    low = ((-lnPbr - 5.92714 + 6.09648 / Tbr + 1.28862 * jnp.log(Tbr)
            - 0.169347 * Tbr ** 6)
           / (15.2518 - 15.6875 / Tbr - 13.4721 * jnp.log(Tbr)
              + 0.43577 * Tbr ** 6))
    high = (-7.904 + 0.1352 * Kw - 0.007465 * Kw ** 2 + 8.359 * Tbr
            + (1.408 - 0.01063 * Kw) / Tbr)
    return jnp.where(Tbr > 0.8, high, low)


# ---------------------------------------------------------------------------
# Vapor pressure
# ---------------------------------------------------------------------------

# Maxwell-Bonnell's three pressure ranges, joined by a narrow logistic
# blend in Q so the derivative is continuous, which the Newton solve needs.
# The published break points (Q = 0.0022 and 0.0013) are where neighbouring
# branches agree to about 1%; the upper join is moved to Q = 1/748.1, which
# is exactly where T equals the (Kw-corrected) normal boiling point. There
# the two branches give 755 and 763 mmHg, so the blend gives psat(Tb) within
# 0.3% of an atmosphere -- 0.03 K in boiling point.
_Q_LOW = 0.0022
_Q_ATM = 1.0 / 748.1
_Q_WIDTH = 1.5e-5


def _mb_log10_mmhg(Q):
    lo = (3000.538 * Q - 6.761560) / (43.0 * Q - 0.987672)   # P < 2 mmHg
    mid = (2663.129 * Q - 5.994296) / (95.76 * Q - 0.972546)  # 2-760 mmHg
    hi = (2770.085 * Q - 6.412631) / (36.0 * Q - 0.989679)   # P > 760 mmHg
    w_lo = 1.0 / (1.0 + jnp.exp(-(Q - _Q_LOW) / _Q_WIDTH))
    w_hi = 1.0 / (1.0 + jnp.exp((Q - _Q_ATM) / _Q_WIDTH))
    core = w_lo * lo + (1.0 - w_lo) * mid
    return w_hi * hi + (1.0 - w_hi) * core


def maxwell_bonnell_psat(T, Tb, Kw=12.0, correct_kw=True, n_iter=4):
    """Vapor pressure (Pa) of a petroleum fraction, Maxwell-Bonnell.

    Riazi MNL50 Eqs. 7.22-7.26, in kelvin. The Watson-K correction shifts the
    normal boiling point by ``1.3889 F (Kw - 12) log10(P/760)`` K, which
    depends on the answer, so it is applied by fixed-point iteration (it
    contracts fast: the shift is a few kelvin per decade of pressure).

    Args:
        T: temperature (K)
        Tb: normal boiling point (K)
        Kw: Watson characterization factor (12 means no correction)
        correct_kw: apply the Kw correction (the API procedure does; it is
            often dropped when Kw is not known)
        n_iter: fixed-point passes for the correction

    Returns:
        Vapor pressure in Pa.
    """
    T = jnp.asarray(T)
    # F = 0 below 367 K, ramps to 1 at 478 K (200-400 F); smooth clip
    F = jnp.clip(0.009 * Tb - 3.2985, 0.0, 1.0)
    log10p = jnp.zeros(jnp.broadcast_shapes(jnp.shape(T), jnp.shape(Tb)))
    log10p = log10p + jnp.log10(ATM_MMHG)
    n = n_iter if correct_kw else 1
    for _ in range(n):
        if correct_kw:
            dTb = 1.3889 * F * (Kw - 12.0) * (log10p - jnp.log10(ATM_MMHG))
        else:
            dTb = 0.0
        Tbp = Tb - dTb
        Q = (Tbp / T - 0.00051606 * Tbp) / (748.1 - 0.3861 * Tbp)
        log10p = _mb_log10_mmhg(Q)
    return 10.0 ** log10p * MMHG


def maxwell_bonnell_boiling_point(P, Tb, Kw=12.0, correct_kw=True):
    """Inverse of :func:`maxwell_bonnell_psat`: the temperature (K) at which
    a fraction of normal boiling point ``Tb`` boils at pressure ``P`` (Pa).

    This is the D1160 conversion run backwards. Newton on ``log psat`` in
    ``1/T``, which is nearly linear, from a Clausius-Clapeyron start.
    """
    lnP = jnp.log(P)
    inv = 1.0 / Tb - (lnP - jnp.log(101325.0)) / (10.0 * Tb)  # rough start

    def g(inv):
        return jnp.log(maxwell_bonnell_psat(1.0 / inv, Tb, Kw, correct_kw)) - lnP

    for _ in range(20):
        val, slope = jax.jvp(g, (inv,), (jnp.ones_like(inv),))
        inv = inv - val / slope
    return 1.0 / inv


def lee_kesler_psat(T, Tc, Pc, omega):
    """Lee-Kesler (1975) vapor pressure (Pa), for comparison only."""
    Tr = T / Tc
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * jnp.log(Tr) + 0.169347 * Tr ** 6
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * jnp.log(Tr) + 0.43577 * Tr ** 6
    return Pc * jnp.exp(f0 + omega * f1)


def fit_antoine(Tb, Kw=12.0, T_range=(400.0, 700.0), n=40):
    """Fit ``ln P = A - B / (T + C)`` to Maxwell-Bonnell over ``T_range``.

    For handing a pseudocomponent to a simulator that only takes Antoine
    constants. Least squares in ``ln P`` by Gauss-Newton from a
    two-parameter start. Returns ``(A, B, C)`` with P in Pa and T in K, and
    the max absolute ``ln P`` error over the fitted range, so the caller can
    see whether three constants were enough -- they usually are to 1-2% over
    a decade of pressure, and not over five.
    """
    T = jnp.linspace(T_range[0], T_range[1], n)
    y = jnp.log(maxwell_bonnell_psat(T, Tb, Kw))
    # start: C = 0, linear fit of y on 1/T
    X = jnp.stack([jnp.ones_like(T), -1.0 / T], axis=1)
    A, B = jnp.linalg.lstsq(X, y)[0]
    p = jnp.array([A, B, 0.0])
    for _ in range(30):
        A, B, C = p
        r = A - B / (T + C) - y
        Jm = jnp.stack([jnp.ones_like(T), -1.0 / (T + C),
                        B / (T + C) ** 2], axis=1)
        p = p - jnp.linalg.lstsq(Jm, r)[0]
    A, B, C = p
    err = jnp.max(jnp.abs(A - B / (T + C) - y))
    return (A, B, C), err


# ---------------------------------------------------------------------------
# Heat capacity
# ---------------------------------------------------------------------------


def kesler_lee_liquid_cp_coefficients(SG, Kw):
    """Coefficients of Kesler-Lee's liquid Cp, Btu/(lb R), T in Rankine."""
    A1 = (-1.17126 + (0.023722 + 0.024907 * SG) * Kw
          + (1.14982 - 0.046535 * Kw) / SG)
    A2 = 1e-4 * (1.0 + 0.82463 * Kw) * (1.12172 - 0.27634 / SG)
    A3 = -1e-8 * (1.0 + 0.82463 * Kw) * (2.9027 - 0.70958 / SG)
    return A1, A2, A3


def kesler_lee_liquid_cp(T, SG, Kw):
    """Liquid heat capacity of a petroleum fraction, J/(kg K)."""
    A1, A2, A3 = kesler_lee_liquid_cp_coefficients(SG, Kw)
    TR = 1.8 * T
    return BTU_LB_R * (A1 + A2 * TR + A3 * TR ** 2)


def kesler_lee_liquid_enthalpy(T, SG, Kw, T_ref=298.15):
    """Liquid enthalpy relative to liquid at ``T_ref``, J/kg."""
    A1, A2, A3 = kesler_lee_liquid_cp_coefficients(SG, Kw)

    def antideriv(t):
        return A1 * t + A2 * 1.8 * t ** 2 / 2.0 + A3 * 3.24 * t ** 3 / 3.0

    return BTU_LB_R * (antideriv(T) - antideriv(T_ref))
