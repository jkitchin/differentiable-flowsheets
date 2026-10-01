"""Property correlations for petroleum pseudo-components.

A pseudo-component is a narrow boiling cut of a crude, and all that is known
about it is its normal boiling point ``Tb`` and its specific gravity ``SG``
(60 F / 60 F). Everything else a simulator needs -- molecular weight, critical
constants, acentric factor, vapour pressure, heat capacity, heat of
vaporisation -- comes from correlations in ``(Tb, SG)`` fitted to pure
hydrocarbons. This module holds those correlations as plain ``jax.numpy``
functions, so a property is differentiable with respect to the assay that
produced it.

Every function takes and returns SI units (K, Pa, J/mol, g/mol) whatever units
the correlation was published in; the conversions live inside, next to the
published constants, so the constants can be checked against their source.

Accuracy, measured against 13 pure hydrocarbons (n-C5 to n-C20, two
naphthenes, three aromatics; ``tests/refinery/test_correlations.py``)::

    method               MW     Tc     Pc     (average absolute % deviation)
    twu_1984             0.5    0.4    1.8
    twu                  2.4    0.4    1.8
    riazi_daubert_1987   2.4    0.6    3.1
    riazi_daubert_1980   4.1    0.7    3.3
    lee_kesler           5.3    0.6    4.0

``"twu_1984"`` is Twu's correlation as published. ``"twu"`` -- the crude
unit's default since it was written, kept bit-for-bit so its solved numbers
do not move under the planning and validation work built on them -- differs
in two constants of the molecular-weight perturbation: it has the Kelvin-form
``0.244541`` and ``0.143979`` (Twu's Rankine constants ``0.328086`` and
``0.193168`` divided by ``sqrt(1.8)``) against ``sqrt(Tb)`` in *Rankine*,
which shrinks the SG correction by ``sqrt(1.8)``. The n-alkanes are its
reference, where the perturbation vanishes, so they do not see it; aromatics
do (benzene -7.5 %, naphthalene -10 %, phenanthrene -15 % in MW, against
-0.5 %, +0.5 % and -3 % for ``"twu_1984"``). Earlier versions of this
docstring called that "the published behaviour"; it is not. Two independent
codings of the 1984 paper agree with ``"twu_1984"`` (pychemqt's
``lib/petro.py``, ``prop_Twu`` Eqs. 21-22, and sim21's ``data/twu.py``).
Tc, Pc and Vc are the same in both. The heavy end and the vacuum column use
``"twu_1984"``; switching the crude unit's default is left to a change that
can re-baseline the numbers pinned in ``tests/refinery/test_cdu_baseline.py``.

What else is here, and where it is used:

- :func:`acentric_factor` blends Lee-Kesler into Kesler-Lee over
  ``Tb/Tc = 0.8 +/- 0.01`` (differentiable everywhere);
  :func:`lee_kesler_acentric` is the literal hard switch both papers
  prescribe. Away from the switch they are the same function.
- :func:`vapor_pressure` is Lee-Kesler corresponding states (the crude
  unit's VLE). :func:`maxwell_bonnell_psat` is Maxwell-Bonnell (1957), the
  ASTM D1160 vacuum-to-atmospheric conversion, anchored at the boiling point
  and fitted to a fraction of a mmHg: the vacuum column's VLE, where
  corresponding states drifts at ``Tr ~ 0.6``.
- :func:`cp_liquid_coeffs` and :func:`kesler_lee_liquid_cp` are the same
  Kesler-Lee correlation, per mole as cubic coefficients and per kg.
- :func:`riazi_daubert_1980` (blending pool), :func:`riazi_daubert_1987`
  and :func:`edmister_omega` are kept as public alternatives.

References:
    Riazi, M.R. and Daubert, T.E., Hydrocarbon Processing 59(3), 115 (1980).
    Riazi, M.R. and Daubert, T.E., Ind. Eng. Chem. Res. 26, 755 (1987).
    Kesler, M.G. and Lee, B.I., Hydrocarbon Processing 55(3), 153 (1976).
    Lee, B.I. and Kesler, M.G., AIChE J. 21, 510 (1975).
    Twu, C.H., Fluid Phase Equilibria 16, 137 (1984).
    Maxwell, J.B. and Bonnell, L.S., Ind. Eng. Chem. 49, 1187 (1957).
    Edmister, W.C., Petroleum Refiner 37(4), 173 (1958).
    Riedel, L., Chem.-Ing.-Tech. 26, 679 (1954).
    Watson, K.M. and Nelson, E.F., Ind. Eng. Chem. 25, 880 (1933).
    Riazi, M.R., Characterization and Properties of Petroleum Fractions,
        ASTM MNL50 (2005) -- collects all of the above.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

#: Gas constant (J/mol/K).
R_GAS = 8.314462618
#: One standard atmosphere (Pa).
P_ATM = 101325.0
_PSIA = 6894.757293168  # Pa per psia
_BTU_PER_LB_F = 4186.8  # J/kg/K per Btu/lb/F, i.e. J/g/K x 1000
_FT3_PER_LBMOL = 0.0283168466 / 453.59237  # m3/mol per ft3/lbmol
#: mmHg in Pa.
MMHG = 133.322368
#: One atmosphere in mmHg, Maxwell-Bonnell's reference pressure.
ATM_MMHG = 760.0
#: Btu/(lb R) in J/(kg K).
BTU_LB_R = _BTU_PER_LB_F

#: Methods :func:`critical_properties` accepts.
CRITICAL_METHODS = ("twu", "twu_1984", "riazi_daubert_1987", "riazi_daubert_1980", "lee_kesler")

# Twu's reference-alkane molecular weight is implicit in Tb. Newton from his
# own starting estimate converges to round-off in under six steps across
# Tb = 250-1100 K; the extra steps are free and cover a poor start.
_TWU_NEWTON_STEPS = 12

# The acentric factor switches from Lee-Kesler to Kesler-Lee at Tbr = 0.8,
# where the two agree to about 0.01. The switch is blended over this
# half-width so that omega -- and a gradient through it -- stays smooth.
_OMEGA_BLEND_WIDTH = 0.01


def watson_k(Tb: Array, SG: Array) -> Array:
    """Watson (UOP) characterisation factor ``Kw = (1.8 Tb)^(1/3) / SG``.

    About 12.5 for paraffinic stocks, 11-12 for naphthenic and 10 for
    aromatic ones.

    Args:
        Tb: Normal boiling point (K).
        SG: Specific gravity, 60 F / 60 F.
    """
    return (1.8 * Tb) ** (1.0 / 3.0) / SG


def sg_from_api(api: Array) -> Array:
    """Specific gravity from API gravity: ``SG = 141.5 / (API + 131.5)``."""
    return 141.5 / (api + 131.5)


def api_from_sg(SG: Array) -> Array:
    """API gravity from specific gravity: ``API = 141.5 / SG - 131.5``."""
    return 141.5 / SG - 131.5


def sg_from_watson_k(Tb: Array, Kw: Array) -> Array:
    """Specific gravity of a cut boiling at ``Tb`` (K) at a given Watson K."""
    return (1.8 * Tb) ** (1.0 / 3.0) / Kw


# =============================================================================
# Molecular weight and critical constants
# =============================================================================


def _riazi_daubert_1980(Tb, SG):
    R = 1.8 * Tb
    MW = 4.5673e-5 * R**2.1962 * SG**-1.0164
    Tc = 24.2787 * R**0.58848 * SG**0.3596 / 1.8
    Pc = 3.12281e9 * R**-2.3125 * SG**2.3201 * _PSIA
    return MW, Tc, Pc


def _riazi_daubert_1987(Tb, SG):
    # The extended form, theta = a Tb^b SG^c exp(d Tb + e SG + f Tb SG), with
    # Tb in K and Pc in bar (API Technical Data Book 2B2.1 / 4D3.1 / 4D4.1).
    MW = (42.965 * jnp.exp(2.097e-4 * Tb - 7.78712 * SG + 2.08476e-3 * Tb * SG)
          * Tb**1.26007 * SG**4.98308)
    Tc = (9.5233 * jnp.exp(-9.314e-4 * Tb - 0.544442 * SG + 6.4791e-4 * Tb * SG)
          * Tb**0.81067 * SG**0.53691)
    Pc = (3.1958e5 * jnp.exp(-8.505e-3 * Tb - 4.8014 * SG + 5.749e-3 * Tb * SG)
          * Tb**-0.4844 * SG**4.0846) * 1e5
    return MW, Tc, Pc


def _lee_kesler(Tb, SG):
    R = 1.8 * Tb
    Tc = 341.7 + 811.0 * SG + (0.4244 + 0.1174 * SG) * R + (0.4669 - 3.2623 * SG) * 1e5 / R
    lnPc = (8.3634 - 0.0566 / SG
            - (0.24244 + 2.2898 / SG + 0.11857 / SG**2) * 1e-3 * R
            + (1.4685 + 3.648 / SG + 0.47227 / SG**2) * 1e-7 * R**2
            - (0.42019 + 1.6977 / SG**2) * 1e-10 * R**3)
    MW = (-12272.6 + 9486.4 * SG + (4.6523 - 3.3287 * SG) * R
          + (1.0 - 0.77084 * SG - 0.02058 * SG**2) * (1.3437 - 720.79 / R) * 1e7 / R
          + (1.0 - 0.80882 * SG + 0.02226 * SG**2) * (1.8828 - 181.98 / R) * 1e12 / R**3)
    return MW, Tc / 1.8, jnp.exp(lnPc) * _PSIA


def riazi_daubert_1980(Tb: Array, SG: Array) -> tuple[Array, Array, Array]:
    """Riazi-Daubert (1980) ``(MW, Tc, Pc)`` in g/mol, K and Pa.

    ``theta = a Tb^b SG^c`` with Tb in Rankine (Riazi MNL50 Eqs. 2.38, 2.40);
    intended for MW of about 70-300. The blending pool's default.
    """
    return _riazi_daubert_1980(jnp.asarray(Tb, dtype=float), jnp.asarray(SG, dtype=float))


def riazi_daubert_1987(Tb: Array, SG: Array) -> tuple[Array, Array, Array]:
    """Extended Riazi-Daubert (1987) ``(MW, Tc, Pc)`` in g/mol, K and Pa.

    Riazi MNL50 Eq. 2.51 for MW; fitted for MW 70-700 (Tb to about 850 K).
    """
    return _riazi_daubert_1987(jnp.asarray(Tb, dtype=float), jnp.asarray(SG, dtype=float))


def _twu_tb_of_theta(theta):
    """Twu's n-alkane boiling point (R) as a function of theta = ln(MW)."""
    return (jnp.exp(5.71419 + 2.71579 * theta - 0.286590 * theta**2
                    - 39.8544 / theta - 0.122488 / theta**2)
            - 24.7522 * theta + 35.3155 * theta**2)


def _twu_core(Tb, SG, mw_a, mw_b):
    R = 1.8 * Tb
    # n-alkane reference with the same boiling point.
    Tc0 = R / (0.533272 + 0.191017e-3 * R + 0.779681e-7 * R**2
               - 0.284376e-10 * R**3 + 0.959468e28 / R**13)
    a = 1.0 - R / Tc0
    Pc0 = (3.83354 + 1.19629 * a**0.5 + 34.8888 * a + 36.1952 * a**2 + 104.193 * a**4) ** 2
    Vc0 = (1.0 - (0.419869 - 0.505839 * a - 1.56436 * a**3 - 9481.70 * a**14)) ** -8
    SG0 = 0.843593 - 0.128624 * a - 3.36159 * a**3 - 13749.5 * a**12

    # MW0 solves _twu_tb_of_theta(ln MW0) = R. Unrolled Newton, so the
    # derivative is the implicit one to round-off.
    theta = jnp.log(R / (10.44 - 0.0052 * R))
    for _ in range(_TWU_NEWTON_STEPS):
        f, df = jax.jvp(_twu_tb_of_theta, (theta,), (jnp.ones_like(theta),))
        theta = theta - (f - R) / df
    MW0 = jnp.exp(theta)

    # Perturbation from the reference in specific gravity.
    s = jnp.sqrt(R)

    def _ratio(f):
        return ((1.0 + 2.0 * f) / (1.0 - 2.0 * f)) ** 2

    dT = jnp.exp(5.0 * (SG0 - SG)) - 1.0
    Tc = Tc0 * _ratio(dT * (-0.362456 / s + (0.0398285 - 0.948125 / s) * dT))

    dV = jnp.exp(4.0 * (SG0**2 - SG**2)) - 1.0
    Vc = Vc0 * _ratio(dV * (0.466590 / s + (-0.182421 + 3.01721 / s) * dV))

    dP = jnp.exp(0.5 * (SG0 - SG)) - 1.0
    fP = dP * ((2.53262 - 46.1955 / s - 0.00127885 * R)
               + (-11.4277 + 252.140 / s + 0.00230535 * R) * dP)
    Pc = Pc0 * (Tc / Tc0) * (Vc0 / Vc) * _ratio(fP)

    dM = jnp.exp(5.0 * (SG0 - SG)) - 1.0
    x = jnp.abs(0.012342 - mw_a / s)
    MW = jnp.exp(jnp.log(MW0) * _ratio(dM * (x + (-0.0175691 + mw_b / s) * dM)))
    # Vc: ft3/lbmol -> m3/mol.
    return MW, Tc / 1.8, Pc * _PSIA, Vc * _FT3_PER_LBMOL


def _twu(Tb, SG):
    # The crude unit's original coding: Kelvin-form MW constants against
    # sqrt(Tb in Rankine). Kept bit-for-bit; see the module docstring.
    return _twu_core(Tb, SG, 0.244541, 0.143979)[:3]


def _twu_1984(Tb, SG):
    # Twu (1984) Eqs. 21-22 as published (T in Rankine).
    return _twu_core(Tb, SG, 0.328086, 0.193168)[:3]


def twu_critical_properties(Tb: Array, SG: Array) -> dict[str, Array]:
    """Twu (1984) critical properties, critical volume and molecular weight.

    Args:
        Tb: Normal boiling point (K).
        SG: Specific gravity, 60 F / 60 F.

    Returns:
        dict with ``Tc`` (K), ``Pc`` (Pa), ``Vc`` (m^3/mol) and ``MW``
        (g/mol), broadcast to the shape of the inputs. The published
        constants (method ``"twu_1984"``), which the heavy end uses.
    """
    MW, Tc, Pc, Vc = _twu_core(jnp.asarray(Tb, dtype=float), jnp.asarray(SG, dtype=float),
                               0.328086, 0.193168)
    return {"Tc": Tc, "Pc": Pc, "Vc": Vc, "MW": MW}


_METHODS = {
    "twu": _twu,
    "twu_1984": _twu_1984,
    "riazi_daubert_1987": _riazi_daubert_1987,
    "riazi_daubert_1980": _riazi_daubert_1980,
    "lee_kesler": _lee_kesler,
}


def critical_properties(Tb: Array, SG: Array, method: str = "twu") -> tuple[Array, Array, Array]:
    """Molecular weight, critical temperature and critical pressure.

    Args:
        Tb: Normal boiling point (K).
        SG: Specific gravity, 60 F / 60 F.
        method: One of :data:`CRITICAL_METHODS`. ``"twu_1984"`` is the most
            accurate on the reference set. ``"twu"`` (the default, kept for
            the crude unit's pinned numbers) is the same except for the
            aromatics' molecular weight; see the module docstring.

    Returns:
        ``(MW, Tc, Pc)`` in g/mol, K and Pa.

    Example:
        >>> MW, Tc, Pc = critical_properties(447.3, 0.7342)   # n-decane
        >>> round(float(MW)), round(float(Tc)), round(float(Pc) / 1e5, 1)
        (142, 619, 21.2)
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose one of {', '.join(CRITICAL_METHODS)}"
        ) from None
    return fn(jnp.asarray(Tb, dtype=float), jnp.asarray(SG, dtype=float))


# =============================================================================
# Acentric factor and vapour pressure
# =============================================================================


def _lk_f0(Tr):
    return 5.92714 - 6.09648 / Tr - 1.28862 * jnp.log(Tr) + 0.169347 * Tr**6


def _lk_f1(Tr):
    return 15.2518 - 15.6875 / Tr - 13.4721 * jnp.log(Tr) + 0.43577 * Tr**6


def acentric_factor(Tb: Array, Tc: Array, Pc: Array, SG: Array) -> Array:
    """Acentric factor of a petroleum fraction.

    Lee-Kesler's vapour-pressure inversion below ``Tb/Tc = 0.8`` -- the
    acentric factor that makes :func:`vapor_pressure` pass through one
    atmosphere at ``Tb`` -- and Kesler-Lee's ``(Kw, Tbr)`` correlation above
    it, as both papers prescribe. The switch is blended over
    ``0.8 +/- 0.01`` (the two agree to about 0.01 there), so that omega has a
    derivative everywhere.

    Args:
        Tb: Normal boiling point (K).
        Tc: Critical temperature (K).
        Pc: Critical pressure (Pa).
        SG: Specific gravity, 60 F / 60 F.
    """
    Tbr = Tb / Tc
    low = (-jnp.log(Pc / P_ATM) - _lk_f0(Tbr)) / _lk_f1(Tbr)
    Kw = watson_k(Tb, SG)
    high = -7.904 + 0.1352 * Kw - 0.007465 * Kw**2 + 8.359 * Tbr + (1.408 - 0.01063 * Kw) / Tbr
    weight = jax.nn.sigmoid((Tbr - 0.8) / _OMEGA_BLEND_WIDTH * 4.0)
    return (1.0 - weight) * low + weight * high


def lee_kesler_acentric(Tb: Array, Tc: Array, Pc: Array, Kw: Array) -> Array:
    """Acentric factor with the literal switch at ``Tb/Tc = 0.8``.

    The same two branches as :func:`acentric_factor` -- Lee-Kesler's
    vapour-pressure inversion below, Kesler-Lee's ``(Kw, Tbr)`` correlation
    above -- switched by a hard ``where`` as Kesler and Lee prescribe, and
    taking ``Kw`` directly. Used where a cut's Kw is fitted rather than
    implied by its own ``(Tb, SG)``; the derivative jumps at the switch.
    """
    Tbr = Tb / Tc
    low = (-jnp.log(Pc / P_ATM) - _lk_f0(Tbr)) / _lk_f1(Tbr)
    high = -7.904 + 0.1352 * Kw - 0.007465 * Kw**2 + 8.359 * Tbr + (1.408 - 0.01063 * Kw) / Tbr
    return jnp.where(Tbr > 0.8, high, low)


def edmister_omega(Tb: Array, Tc: Array, Pc: Array) -> Array:
    """Acentric factor from Edmister (1958).

    ``omega = 3/7 log10(Pc / 1 atm) / (Tc/Tb - 1) - 1``.
    """
    return 3.0 / 7.0 * jnp.log10(Pc / P_ATM) / (Tc / Tb - 1.0) - 1.0


def vapor_pressure(T: Array, Tc: Array, Pc: Array, omega: Array) -> Array:
    """Lee-Kesler vapour pressure (Pa): ``ln Pr = f0(Tr) + omega f1(Tr)``.

    Meaningful for ``0.3 < Tr < 1``; the pseudo-components of a crude spend a
    column well inside that.
    """
    Tr = T / Tc
    return Pc * jnp.exp(_lk_f0(Tr) + omega * _lk_f1(Tr))


lee_kesler_psat = vapor_pressure
"""Alias of :func:`vapor_pressure` under the name the blending pool uses."""


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


def maxwell_bonnell_antoine(Tb, Kw=12.0, T_range=(400.0, 700.0), n=40):
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


# =============================================================================
# Calorimetric properties
# =============================================================================


def hvap_at_tb(Tb: Array, Tc: Array, Pc: Array) -> Array:
    """Heat of vaporisation at the normal boiling point (J/mol), Riedel (1954).

    ``dHvb = 1.093 R Tc Tbr (ln Pc[bar] - 1.013) / (0.930 - Tbr)``. Within 1 %
    for light hydrocarbons and 4 % at C10-C16 on the reference set.
    """
    Tbr = Tb / Tc
    return 1.093 * R_GAS * Tc * Tbr * (jnp.log(Pc / 1e5) - 1.013) / (0.930 - Tbr)


def cp_liquid_coeffs(Tb: Array, SG: Array, MW: Array) -> Array:
    """Liquid heat capacity, Kesler-Lee (1976), as cubic coefficients in K.

    The correlation is quadratic in temperature, so it converts exactly to
    difflow's ``Cp = a + b T + c T^2 + d T^3`` (J/mol/K) with ``d = 0``.
    Within 2-8 % of measured liquid Cp at 298 K on the reference set.

    Returns:
        Array ``(..., 4)`` of ``(a, b, c, d)``.
    """
    Kw = watson_k(Tb, SG)
    A1 = -1.17126 + (0.023722 + 0.024907 * SG) * Kw + (1.14982 - 0.046535 * Kw) / SG
    A2 = 1e-4 * (1.0 + 0.82463 * Kw) * (1.12172 - 0.27634 / SG)
    A3 = -1e-8 * (1.0 + 0.82463 * Kw) * (2.9027 - 0.70958 / SG)
    # Btu/lb/F with T in Rankine -> J/mol/K with T in K: T_R = 1.8 T_K.
    scale = _BTU_PER_LB_F * MW / 1000.0
    return jnp.stack([A1 * scale, A2 * 1.8 * scale, A3 * 1.8**2 * scale,
                      jnp.zeros_like(A1)], axis=-1)


def cp_ideal_gas_coeffs(Tb: Array, SG: Array, MW: Array) -> Array:
    """Ideal-gas heat capacity, Watson-Nelson (1933), as cubic coefficients in K.

    ``Cp/[Btu/lb/F] = (0.0450 Kw - 0.233) + (0.440 + 0.0177 Kw) 1e-3 t
    - 0.1520e-6 t^2`` with ``t`` in degrees Fahrenheit -- quadratic, so it too
    converts exactly to cubic coefficients (J/mol/K, T in K). Within 3 % of
    the ideal-gas Cp of n-alkanes and toluene from 300 to 600 K and 4 % at
    800 K.

    Returns:
        Array ``(..., 4)`` of ``(a, b, c, d)``.
    """
    Kw = watson_k(Tb, SG)
    c0 = 0.0450 * Kw - 0.233
    c1 = (0.440 + 0.0177 * Kw) * 1e-3
    c2 = -0.1520e-6 * jnp.ones_like(Kw)
    # t_F = 1.8 T - 459.67; expand c0 + c1 t + c2 t^2 in powers of T.
    off = -459.67
    a = c0 + c1 * off + c2 * off**2
    b = (c1 + 2.0 * c2 * off) * 1.8
    c = c2 * 1.8**2
    scale = _BTU_PER_LB_F * MW / 1000.0
    return jnp.stack([a * scale, b * scale, c * scale, jnp.zeros_like(a)], axis=-1)


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
