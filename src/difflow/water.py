"""Water and steam properties shared by evaporators and humid-air models.

Pure JAX, SI units (K, Pa, J/kg), differentiable. Everything here used to live
in :mod:`difflow.units.evaporator`, which re-exports it unchanged, so the old
import path keeps working.

What is in here and how far to trust it
---------------------------------------
``water_saturation_pressure(T)`` / ``water_saturation_temperature(P)``
    IAPWS-IF97 region 4 (Wagner & Pruss), formally valid 273.15-647 K and
    611 Pa-22 MPa. Below 273.15 K the same formula is a smooth extrapolation
    to **supercooled liquid**; it is used (unchecked against supercooled-water
    data) down to about 250 K, which only matters when you pass
    ``over_ice=False`` to :func:`water_vapor_pressure`.
``water_sublimation_pressure(T)``
    Saturation pressure over **ice Ih** from the IAPWS 2008 sublimation
    formula (Wagner, Riethmann, Feistel & Harvey 2011), valid 190-273.16 K.
    The three coefficient pairs were typed from memory (flagged unverified);
    they are checked in the tests for self-consistency (it must return the
    triple-point pressure 611.657 Pa at 273.16 K, which it can only do if
    ``a1 + a2 + a3 = 0``), for the Clausius-Clapeyron slope against the heat
    of sublimation, and against ``p_ice(-10 C) = 259.9 Pa`` and
    ``p_ice(-20 C) = 103.2 Pa``, which are values the author is confident of.
``water_vapor_pressure(T, over_ice=None)``
    Saturation pressure in the sense a psychrometrician needs: over liquid
    above the triple point and over ice below it, joined by a C2 smooth blend
    over +-0.25 K so there is no Python ``if`` on a traced T and no kink in
    the derivative (the two branches cross at the triple point).
``water_latent_heat(T)`` / ``water_liquid_enthalpy(T)`` / ``water_vapor_enthalpy(T)``
    Fits to eight steam-table points between 25 and 200 C (about 0.1 %),
    **unchanged** in 20-200 C. Below 20 C the raw fit drifts (at 0.01 C it
    gives h_fg = 2493.7 kJ/kg against 2500.9 and h_f = -0.19 kJ/kg against 0),
    so a correction that is zero at 20 C and exactly restores the triple-point
    values (h_fg = 2500.9 kJ/kg, h_f = 0, both recalled, flagged unverified)
    is blended in below 20 C with a C2 smoother-step. Between 0 and 20 C the
    result agrees with recalled steam-table values (2489.1 kJ/kg at 5 C,
    2477.2 at 10 C) to about 0.1 %. **Below 0 C these are extrapolations**:
    use them only for supercooled-liquid bookkeeping.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

#: 0 C in K; the liquid enthalpy reference (h = 0 for saturated liquid at 0 C)
T_REF = 273.15
#: Triple point of water (K) and its vapor pressure (Pa), IAPWS
T_TRIPLE = 273.16
P_TRIPLE = 611.657
#: Critical temperature of water (K), IAPWS
T_CRIT_WATER = 647.096
#: Specific gas constant of steam, J/kg/K
R_STEAM = 461.5
#: Latent heat of vaporization at the triple point (J/kg): the familiar
#: 2500.9 kJ/kg (2501 in the ASHRAE form). Recalled, unverified.
H_FG_TRIPLE = 2500.9e3
#: Latent heat of fusion of ice at 0 C (J/kg), 333.5 kJ/kg. Recalled, unverified.
H_FUSION = 333.5e3

# IAPWS-IF97 region-4 saturation-line coefficients (Wagner & Pruss 1993;
# IAPWS-IF97 Eq. 29-31). Checked numerically in tests (round trip, and
# Psat(100 C) = 101.418 kPa).
_N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
)

# Saturated-liquid enthalpy h_f(t) = c3 t^3 + c2 t^2 + c1 t + c0, t in C,
# J/kg, and latent heat h_fg = a (1 - T/Tc)^(b + c T/Tc). Both are least-squares
# fits to eight saturated-steam-table points (25, 50, 75, 100, 120, 150, 180,
# 200 C) recalled from a standard steam table (Cengel-type, IAPWS-95 values),
# NOT regenerated from IAPWS here. Max fit residuals over those points: h_f
# 0.04 %, h_fg 0.09 %. Valid 20-200 C; below 20 C see ``_low_t_weight``.
_HF_COEFFS = (4.53143312e-03, -6.69677099e-01, 4.21481377e03, -2.36660783e02)
_HFG_COEFFS = (2.92757368e06, 2.64717681e-01, 6.57306406e-02)

# IAPWS 2008 sublimation-pressure formula (Wagner et al., J. Phys. Chem. Ref.
# Data 40, 023103 (2011)): ln(p/p_t) = theta^-1 sum a_i theta^b_i,
# theta = T / T_t. Typed from memory; see the module notes for the checks.
_ICE_A = (-0.212144006e2, 0.273203819e2, -0.610598130e1)
_ICE_B = (0.333333333e-2, 0.120666667e1, 0.170333333e1)

#: Half-width (K) of the liquid/ice blend around the triple point
ICE_BLEND_HALF_WIDTH = 0.25
#: Upper edge (K) of the below-20-C correction to the h_f / h_fg fits
_LOW_T_EDGE = 293.15


def _smootherstep(t: Array) -> Array:
    """C2 smoother-step: 0 at ``t <= 0``, 1 at ``t >= 1``."""
    t = jnp.clip(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def water_saturation_pressure(T: Array | float) -> Array:
    """Saturation pressure of water (Pa) at temperature ``T`` (K).

    IAPWS-IF97 region-4 equation (Wagner & Pruss), 273.15-647 K. Below
    273.15 K this is the (unchecked) extrapolation to supercooled liquid;
    use :func:`water_vapor_pressure` for the pressure over ice.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    th = T + _N[8] / (T - _N[9])
    A = th**2 + _N[0] * th + _N[1]
    B = _N[2] * th**2 + _N[3] * th + _N[4]
    C = _N[5] * th**2 + _N[6] * th + _N[7]
    return (2.0 * C / (-B + jnp.sqrt(B * B - 4.0 * A * C))) ** 4 * 1.0e6


def water_saturation_temperature(P: Array | float) -> Array:
    """Saturation temperature of water (K) at pressure ``P`` (Pa).

    Closed-form inverse of :func:`water_saturation_pressure` (IAPWS-IF97
    backward equation), 611 Pa to 22 MPa.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    b = (P / 1.0e6) ** 0.25
    E = b * b + _N[2] * b + _N[5]
    F = _N[0] * b * b + _N[3] * b + _N[6]
    G = _N[1] * b * b + _N[4] * b + _N[7]
    D = 2.0 * G / (-F - jnp.sqrt(F * F - 4.0 * E * G))
    return 0.5 * (_N[9] + D - jnp.sqrt((_N[9] + D) ** 2 - 4.0 * (_N[8] + _N[9] * D)))


def water_sublimation_pressure(T: Array | float) -> Array:
    """Saturation (sublimation) pressure of water over ice (Pa) at ``T`` (K).

    IAPWS 2008 formula, 190-273.16 K; above the triple point it is the
    (metastable) continuation of the same curve. Coefficients from memory,
    checked in the tests (see the module notes).
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    th = T / T_TRIPLE
    s = sum(a * th**b for a, b in zip(_ICE_A, _ICE_B))
    return P_TRIPLE * jnp.exp(s / th)


def ice_weight(T: Array | float) -> Array:
    """Weight of the ice branch (1 well below the triple point, 0 above), C2 smooth."""
    T = jnp.asarray(T, dtype=jnp.float64)
    return 1.0 - _smootherstep((T - (T_TRIPLE - ICE_BLEND_HALF_WIDTH)) / (2.0 * ICE_BLEND_HALF_WIDTH))


def water_vapor_pressure(T: Array | float, over_ice: bool | None = None) -> Array:
    """Equilibrium vapor pressure of water (Pa) as a psychrometrician means it.

    Args:
        T: Temperature (K).
        over_ice: ``None`` (default) uses liquid above the triple point and
            ice below it, blended smoothly over +-0.25 K; ``False`` always
            uses liquid (supercooled below 273.15 K, an extrapolation);
            ``True`` always uses ice.

    Returns:
        Saturation pressure (Pa).
    """
    p_l = water_saturation_pressure(T)
    if over_ice is False:
        return p_l
    p_i = water_sublimation_pressure(T)
    if over_ice is True:
        return p_i
    w = ice_weight(T)
    # blend in ln p, so both branches stay positive and the blend is smooth
    return jnp.exp(w * jnp.log(p_i) + (1.0 - w) * jnp.log(p_l))


def _low_t_weight(T: Array) -> Array:
    """1 at the triple point and below, 0 at 20 C and above (C2)."""
    return 1.0 - _smootherstep((T - T_TRIPLE) / (_LOW_T_EDGE - T_TRIPLE))


def water_latent_heat(T: Array | float) -> Array:
    """Latent heat of vaporization of water (J/kg) at saturation temperature ``T`` (K).

    Fit of the form ``a (1 - Tr)^(b + c Tr)`` to steam-table points (about
    0.1 %, 20-200 C, unchanged from the original evaporator correlation).
    Below 20 C a smooth correction restores the triple-point value
    2500.9 kJ/kg; below 0 C the result is an extrapolation.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    Tr = T / T_CRIT_WATER
    a, b, c = _HFG_COEFFS
    raw = a * (1.0 - Tr) ** (b + c * Tr)
    raw_tp = a * (1.0 - T_TRIPLE / T_CRIT_WATER) ** (b + c * T_TRIPLE / T_CRIT_WATER)
    return raw + (H_FG_TRIPLE - raw_tp) * _low_t_weight(T)


def water_liquid_enthalpy(T: Array | float) -> Array:
    """Saturated-liquid enthalpy of water (J/kg), zero at 0 C.

    Cubic fit, valid 20-200 C (unchanged); a smooth correction below 20 C
    makes it exactly zero at the triple point. Below 0 C an extrapolation.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    c3, c2, c1, c0 = _HF_COEFFS

    def poly(t):
        return ((c3 * t + c2) * t + c1) * t + c0

    raw_tp = poly(T_TRIPLE - T_REF)
    return poly(T - T_REF) - raw_tp * _low_t_weight(T)


def water_vapor_enthalpy(T: Array | float) -> Array:
    """Saturated-vapor enthalpy of water (J/kg) at saturation temperature ``T`` (K)."""
    return water_liquid_enthalpy(T) + water_latent_heat(T)
