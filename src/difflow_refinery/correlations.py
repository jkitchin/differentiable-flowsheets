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
    twu                  2.4    0.4    1.8
    riazi_daubert_1987   2.4    0.6    3.1
    riazi_daubert_1980   4.1    0.7    3.3
    lee_kesler           5.3    0.6    4.0

Twu reproduces the n-alkanes almost exactly -- his correlation is a
perturbation about an n-alkane reference -- and is least accurate for
aromatics' molecular weight (-7 %). That is the published behaviour of each
method, not a defect of this implementation.

References:
    Riazi, M.R. and Daubert, T.E., Hydrocarbon Processing 59(3), 115 (1980).
    Riazi, M.R. and Daubert, T.E., Ind. Eng. Chem. Res. 26, 755 (1987).
    Kesler, M.G. and Lee, B.I., Hydrocarbon Processing 55(3), 153 (1976).
    Lee, B.I. and Kesler, M.G., AIChE J. 21, 510 (1975).
    Twu, C.H., Fluid Phase Equilibria 16, 137 (1984).
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

#: Methods :func:`critical_properties` accepts.
CRITICAL_METHODS = ("twu", "riazi_daubert_1987", "riazi_daubert_1980", "lee_kesler")

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


def _twu_tb_of_theta(theta):
    """Twu's n-alkane boiling point (R) as a function of theta = ln(MW)."""
    return (jnp.exp(5.71419 + 2.71579 * theta - 0.286590 * theta**2
                    - 39.8544 / theta - 0.122488 / theta**2)
            - 24.7522 * theta + 35.3155 * theta**2)


def _twu(Tb, SG):
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
    x = jnp.abs(0.012342 - 0.244541 / s)
    MW = jnp.exp(jnp.log(MW0) * _ratio(dM * (x + (-0.0175691 + 0.143979 / s) * dM)))
    return MW, Tc / 1.8, Pc * _PSIA


_METHODS = {
    "twu": _twu,
    "riazi_daubert_1987": _riazi_daubert_1987,
    "riazi_daubert_1980": _riazi_daubert_1980,
    "lee_kesler": _lee_kesler,
}


def critical_properties(Tb: Array, SG: Array, method: str = "twu") -> tuple[Array, Array, Array]:
    """Molecular weight, critical temperature and critical pressure.

    Args:
        Tb: Normal boiling point (K).
        SG: Specific gravity, 60 F / 60 F.
        method: One of :data:`CRITICAL_METHODS`. ``"twu"`` (the default) is
            the most accurate of the four on the reference set, and the one
            commercial simulators default to for crude.

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


def vapor_pressure(T: Array, Tc: Array, Pc: Array, omega: Array) -> Array:
    """Lee-Kesler vapour pressure (Pa): ``ln Pr = f0(Tr) + omega f1(Tr)``.

    Meaningful for ``0.3 < Tr < 1``; the pseudo-components of a crude spend a
    column well inside that.
    """
    Tr = T / Tc
    return Pc * jnp.exp(_lk_f0(Tr) + omega * _lk_f1(Tr))


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
