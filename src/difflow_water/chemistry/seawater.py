"""Seawater properties as functions of temperature and salinity.

Every correlation here is transcribed from the primary source and its
coefficients were read against it (status "content verified" in
``docs/unit-operations-water.md``):

- [S1] Sharqawy, Lienhard & Zubair, *Desalin. Water Treat.* 16 (2010)
  354-380, doi:10.5004/dwt.2010.1079, read from the authors' corrected
  version (MIT; it carries the five published corrections, none of which
  touches Eqs. 8, 22, 23 or 49).
- [S4] TEOS-10 manual (IOC Manuals and Guides 56, 2010) for the molar mass
  of sea salt, ``M_S = 31.4038218 g/mol``, which it takes from Millero et
  al. (2008) [S3].

The osmotic pressure is ``pi = phi * m * rho_w * R * T`` with ``m`` the
molality of sea salt on the ``M_S`` basis (so ``m`` already counts every
ion and no van't Hoff factor appears). That form is the dilute
approximation of ``pi = -(R T / V_w) ln a_w`` with ``ln a_w = -phi m M_w``
and ``V_w = M_w / rho_w``; it is the form WaterTAP's seawater property
package codes (citing Nayar et al. 2016 [S2], Eq. 48, not read here:
second-hand, unverified), which keeps the two codes comparable.

Units: T in K, salinity ``S`` as a mass fraction (kg salt / kg seawater),
densities kg/m^3, viscosities Pa s, pressures Pa. The correlations use
Celsius internally (ITS-90; Eqs. 8, 22, 23 and 49 are written in ``t``,
not ``t68``).

Validity (from [S1]):

========================  =================  ==================
Property                  t (C)              S (kg/kg)
========================  =================  ==================
density, Eq. 8            0 - 180            0 - 0.16
viscosity, Eqs. 22-23     0 - 180            0 - 0.15
osmotic coeff., Eq. 49    0 - 200            0.010 - 0.120
specific heat, Eq. 9      0 - 180            0 - 0.18
enthalpy, Eq. 43          10 - 120           0 - 0.12
BPE, Eq. 36               0 - 200            0 - 0.12
latent heat, Eq. 54       0 - 200            (pure water)
vapor pressure, Eq. 53    0 - 200            (pure water)
========================  =================  ==================

Thermal properties (for evaporators and preheaters; added in phase 3):

- specific heat [S1] Eq. 9 (content verified; ``t68`` from Eq. 4, ``S_P``
  taken as ``1000 S``, as WaterTAP does), boiling point elevation Eq. 36,
  latent heat Eqs. 37 + 54 (content verified), saturated-water enthalpy
  Eq. 55 (content verified);
- seawater enthalpy [S1] Eq. 43 (content verified at its printed 4-digit
  precision); the coefficients coded are the 6-digit values of Nayar et al.
  (2016) [S2] Eq. 25 as WaterTAP codes them, plus [S2]'s pressure term
  (Eq. 26) -- [S2] not read here: second-hand, unverified. They agree with
  Eq. 43 to its printed digits;
- vapor pressure of pure water [S1] Eq. 53 (Hyland & Wexler, content
  verified at its printed precision; the coded coefficients are the full
  ones WaterTAP takes from [S2] Eq. 5, unverified) and of seawater
  ``p_w exp(B1 s + B2 s^2)`` ([S2] Eq. 6 via WaterTAP, s in g/kg;
  unverified);
- water vapor enthalpy ``h_w(T, P) + h_fg(T)``, WaterTAP's water property
  package convention (an ideal-gas vapor whose enthalpy is that of
  saturated vapor at T).

Outside these ranges the functions still return a number (they are
polynomials); :func:`validity` reports whether a state is inside them.
Eq. 49 gives ``phi = 0.8945`` at ``S = 0`` rather than 1, so a permeate
osmotic pressure computed with it is an extrapolation. It is small, and
WaterTAP does the same.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

R_GAS = 8.314462618  # J/(mol K)
MW_WATER = 18.01528e-3  # kg/mol
MW_SEA_SALT = 31.4038218e-3  # kg/mol, [S4] (from Millero et al. 2008 [S3])

# [S1] Eq. 8 (density). Pure-water part a1..a5, salinity part b1..b5.
_RHO_A = (9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8)
_RHO_B = (8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5)

# [S1] Eq. 22 (seawater viscosity factors A and B) and Eq. 23 (pure water).
_MU_A = (1.541, 1.998e-2, -9.52e-5)
_MU_B = (7.974, -7.561e-2, 4.724e-4)

# [S1] Eq. 49 (osmotic coefficient), a1..a10.
_PHI = (8.9453e-1, 4.1561e-4, -4.6262e-6, 2.2211e-11, -1.1445e-1,
        -1.4783e-3, -1.3526e-8, 7.0132, 5.696e-2, -2.8624e-4)

# [S1] Eq. 9 (specific heat, kJ/(kg K) with T68 in K and S_P in g/kg):
# A, B, C, D each = c0 + c1 S_P + c2 S_P^2
_CP = ((5.328, -9.76e-2, 4.04e-4), (-6.913e-3, 7.351e-4, -3.15e-6),
       (9.6e-6, -1.927e-6, 8.23e-9), (2.5e-9, 1.666e-9, -7.125e-12))

# [S1] Eq. 55 (saturated pure-water enthalpy, J/kg, t in C)
_HW = (141.355, 4202.07, -0.535, 0.004)

# [S1] Eq. 43 (seawater enthalpy) a1..a10 at the 6-digit precision of [S2]
# Eq. 25 as coded in WaterTAP (unverified; equal to Eq. 43 to its digits)
_HSW = (-2.34825e4, 3.15183e5, 2.80269e6, -1.44606e7, 7.82607e3, -4.41733e1,
        2.1394e-1, -1.99108e4, 2.77846e4, 9.72801e1)

# [S2] Eq. 26 pressure term (J/kg per MPa above 0.101325 MPa), via WaterTAP
# (unverified): water part A1..A4, salinity part A5..A8 (times S in g/kg)
_HP_W = (996.7767, -3.2406, 0.0127, -4.7723e-5)
_HP_S = (-1.1748, 0.01169, -2.6185e-5, 7.0661e-8)

# [S1] Eq. 54 (latent heat of pure water, J/kg, t in C)
_HFG = (2.501e6, -2.369e3, 2.678e-1, -8.103e-3, -2.079e-5)

# [S1] Eq. 53 (pure-water vapor pressure, Pa; Hyland & Wexler) a1..a6 at the
# full precision WaterTAP takes from [S2] Eq. 5 (unverified; Eq. 53 prints
# -5800, 1.391, -4.846e-2, 4.176e-5, -1.445e-8, 6.545)
_PSAT_W = (-5.8002206e3, 1.3914993, -4.8640239e-2, 4.1764768e-5, -1.4452093e-8,
           6.5459673)
# [S2] Eq. 6 (seawater / pure-water vapor pressure), s in g/kg, via WaterTAP
_PSAT_S = (-4.5818e-4, -2.0443e-6)

# [S1] Eq. 36 (boiling point elevation, K; S in kg/kg, t in C)
_BPE_A = (17.95, 2.823e-1, -4.584e-4)
_BPE_B = (6.56, 5.267e-2, 1.536e-4)

P_REF = 101325.0  # Pa, reference pressure of the enthalpy correlations

#: [S1] validity ranges: (t_min C, t_max C, S_min, S_max)
RANGES = {
    "density": (0.0, 180.0, 0.0, 0.16),
    "viscosity": (0.0, 180.0, 0.0, 0.15),
    "osmotic_coefficient": (0.0, 200.0, 0.010, 0.120),
    "specific_heat": (0.0, 180.0, 0.0, 0.18),
    "enthalpy": (10.0, 120.0, 0.0, 0.12),
    "boiling_point_elevation": (0.0, 200.0, 0.0, 0.12),
}


def _celsius(T):
    return jnp.asarray(T, dtype=jnp.float64) - 273.15


def density_water(T) -> Array:
    """Pure-water density, the S = 0 part of [S1] Eq. 8 (kg/m^3).

    Args:
        T: Temperature (K).

    Returns:
        Density (kg/m^3).

    Example:
        >>> round(float(density_water(298.15)), 1)
        996.9
    """
    t = _celsius(T)
    a1, a2, a3, a4, a5 = _RHO_A
    return a1 + a2 * t + a3 * t**2 + a4 * t**3 + a5 * t**4


def density(T, S) -> Array:
    """Seawater density, [S1] Eq. 8 (kg/m^3).

    Args:
        T: Temperature (K).
        S: Salinity, mass fraction of sea salt (kg/kg).

    Returns:
        Density (kg/m^3). Stated accuracy +-0.1 %.

    Example:
        >>> round(float(density(298.15, 0.035)), 1)
        1023.6
    """
    t = _celsius(T)
    S = jnp.asarray(S, dtype=jnp.float64)
    b1, b2, b3, b4, b5 = _RHO_B
    return density_water(T) + b1 * S + b2 * S * t + b3 * S * t**2 + b4 * S * t**3 + b5 * S**2 * t**2


def viscosity_water(T) -> Array:
    """Pure-water dynamic viscosity, [S1] Eq. 23 (Pa s).

    Args:
        T: Temperature (K).

    Returns:
        Viscosity (Pa s).

    Example:
        >>> round(float(viscosity_water(298.15)) * 1e3, 3)
        0.89
    """
    t = _celsius(T)
    return 4.2844e-5 + 1.0 / (0.157 * (t + 64.993) ** 2 - 91.296)


def viscosity(T, S) -> Array:
    """Seawater dynamic viscosity, [S1] Eq. 22 (Pa s).

    Args:
        T: Temperature (K).
        S: Salinity (kg/kg).

    Returns:
        Viscosity (Pa s). Stated accuracy +-1.5 %.
    """
    t = _celsius(T)
    S = jnp.asarray(S, dtype=jnp.float64)
    A = _MU_A[0] + _MU_A[1] * t + _MU_A[2] * t**2
    B = _MU_B[0] + _MU_B[1] * t + _MU_B[2] * t**2
    return viscosity_water(T) * (1.0 + A * S + B * S**2)


def osmotic_coefficient(T, S) -> Array:
    """Seawater osmotic coefficient, [S1] Eq. 49 (dimensionless).

    Args:
        T: Temperature (K).
        S: Salinity (kg/kg).

    Returns:
        Osmotic coefficient. Stated accuracy +-1.4 %.

    Example:
        >>> round(float(osmotic_coefficient(298.15, 0.035)), 3)
        0.907
    """
    t = _celsius(T)
    S = jnp.asarray(S, dtype=jnp.float64)
    a = _PHI
    return (a[0] + a[1] * t + a[2] * t**2 + a[3] * t**4 + a[4] * S + a[5] * S * t
            + a[6] * S * t**3 + a[7] * S**2 + a[8] * S**2 * t + a[9] * S**2 * t**2)


def molality(S) -> Array:
    """Molality of sea salt on the ``M_S`` basis (mol/kg water).

    Args:
        S: Salinity (kg/kg).

    Returns:
        ``S / ((1 - S) M_S)`` (mol per kg of water).
    """
    S = jnp.asarray(S, dtype=jnp.float64)
    return S / ((1.0 - S) * MW_SEA_SALT)


def osmotic_pressure(T, S) -> Array:
    """Seawater osmotic pressure ``phi m rho_w R T`` (Pa).

    ``phi`` from [S1] Eq. 49, ``rho_w`` from [S1] Eq. 8 at S = 0, ``m``
    from :func:`molality`. See the module docstring for the form.

    Args:
        T: Temperature (K).
        S: Salinity (kg/kg).

    Returns:
        Osmotic pressure (Pa).

    Example:
        >>> round(float(osmotic_pressure(298.15, 0.035)) / 1e5, 1)
        25.9
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    return osmotic_coefficient(T, S) * molality(S) * density_water(T) * R_GAS * T


def salinity_from_mass_concentration(C, T, n_iter: int = 6) -> Array:
    """Salinity S from the mass concentration ``C = S rho(T, S)`` (kg/m^3).

    Newton on ``S rho(T, S) - C`` from ``S = C / rho_w``; unrolled, so it is
    differentiable and jit-friendly. Six steps reach machine precision for
    ``C`` up to the Eq. 8 range.

    Args:
        C: Mass concentration of sea salt (kg/m^3 of seawater).
        T: Temperature (K).
        n_iter: Newton steps.

    Returns:
        Salinity (kg/kg).
    """
    C = jnp.asarray(C, dtype=jnp.float64)
    t = _celsius(T)
    b1, b2, b3, b4, b5 = _RHO_B
    rho_w = density_water(T)
    lin = b1 + b2 * t + b3 * t**2 + b4 * t**3  # d rho / dS at S = 0
    quad = b5 * t**2
    S = C / rho_w
    for _ in range(n_iter):
        f = S * (rho_w + lin * S + quad * S**2) - C
        df = rho_w + 2.0 * lin * S + 3.0 * quad * S**2
        S = S - f / df
    return S


def validity(T, S) -> dict[str, bool]:
    """Whether a concrete state lies inside each correlation's range.

    Args:
        T: Temperature (K), concrete.
        S: Salinity (kg/kg), concrete.

    Returns:
        ``{property: inside}`` for each entry of :data:`RANGES`.
    """
    t = float(T) - 273.15
    s = float(S)
    return {k: (lo <= t <= hi) and (smin <= s <= smax)
            for k, (lo, hi, smin, smax) in RANGES.items()}


# --------------------------------------------------------------------------
# Thermal properties (phase 3)
# --------------------------------------------------------------------------

def specific_heat(T, S) -> Array:
    """Seawater specific heat at constant pressure, [S1] Eq. 9 (J/(kg K)).

    Args:
        T: Temperature (K, ITS-90; converted to T68 by [S1] Eq. 4).
        S: Salinity (kg/kg); used as ``S_P = 1000 S`` (as WaterTAP does).

    Returns:
        Specific heat (J/(kg K)). Stated accuracy +-0.28 %.

    Example:
        >>> round(float(specific_heat(298.15, 0.035)))
        4001
    """
    T68 = (jnp.asarray(T, dtype=jnp.float64) - 0.00025 * 273.15) / (1.0 - 0.00025)
    sp = 1000.0 * jnp.asarray(S, dtype=jnp.float64)
    A, B, C, D = (c0 + c1 * sp + c2 * sp**2 for c0, c1, c2 in _CP)
    return 1000.0 * (A + B * T68 + C * T68**2 + D * T68**3)


def enthalpy_water(T, P=P_REF) -> Array:
    """Pure (liquid) water specific enthalpy (J/kg): [S1] Eq. 55 + [S2] pressure term.

    Args:
        T: Temperature (K).
        P: Pressure (Pa).

    Returns:
        Specific enthalpy (J/kg), zero near 0 C.
    """
    t = _celsius(T)
    c1, c2, c3, c4 = _HW
    a1, a2, a3, a4 = _HP_W
    dP = (jnp.asarray(P, dtype=jnp.float64) - P_REF) / 1e6
    return c1 + c2 * t + c3 * t**2 + c4 * t**3 + dP * (a1 + a2 * t + a3 * t**2 + a4 * t**3)


def enthalpy(T, S, P=P_REF) -> Array:
    """Seawater specific enthalpy, [S1] Eq. 43 with [S2]'s pressure term (J/kg).

    ``h = h_w - S (a1 + a2 S + ... + a10 S t^2) + dP (A1 + ... + S_g (A5 + ...))``
    with ``dP`` in MPa above 1 atm and ``S_g`` in g/kg (WaterTAP's form).

    Args:
        T: Temperature (K).
        S: Salinity (kg/kg).
        P: Pressure (Pa).

    Returns:
        Specific enthalpy (J/kg of seawater).

    Example:
        >>> round(float(enthalpy(298.15, 0.035)))
        99766
    """
    t = _celsius(T)
    S = jnp.asarray(S, dtype=jnp.float64)
    a = _HSW
    h0 = enthalpy_water(T, P_REF) - S * (
        a[0] + a[1] * S + a[2] * S**2 + a[3] * S**3 + a[4] * t + a[5] * t**2
        + a[6] * t**3 + a[7] * S * t + a[8] * S**2 * t + a[9] * S * t**2)
    a1, a2, a3, a4 = _HP_W
    a5, a6, a7, a8 = _HP_S
    dP = (jnp.asarray(P, dtype=jnp.float64) - P_REF) / 1e6
    return h0 + dP * (a1 + a2 * t + a3 * t**2 + a4 * t**3
                      + 1000.0 * S * (a5 + a6 * t + a7 * t**2 + a8 * t**3))


def latent_heat_water(T) -> Array:
    """Latent heat of vaporization of pure water, [S1] Eq. 54 (J/kg)."""
    t = _celsius(T)
    c = _HFG
    return c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4


def latent_heat(T, S) -> Array:
    """Latent heat of seawater per kg of seawater, [S1] Eq. 37 (J/kg)."""
    return latent_heat_water(T) * (1.0 - jnp.asarray(S, dtype=jnp.float64))


def vapor_enthalpy(T, P=P_REF) -> Array:
    """Specific enthalpy of water vapor at T (J/kg): ``h_w(T, P) + h_fg(T)``.

    WaterTAP's water property package convention: the vapor is treated as an
    ideal gas whose enthalpy is that of saturated vapor at the same
    temperature (superheat changes it only through T).
    """
    return enthalpy_water(T, P) + latent_heat_water(T)


def vapor_pressure_water(T) -> Array:
    """Vapor pressure of pure water, [S1] Eq. 53 (Hyland & Wexler; Pa).

    Example:
        >>> round(float(vapor_pressure_water(373.124)))
        101325
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    a1, a2, a3, a4, a5, a6 = _PSAT_W
    return jnp.exp(a1 / T + a2 + a3 * T + a4 * T**2 + a5 * T**3 + a6 * jnp.log(T))


def vapor_pressure(T, S) -> Array:
    """Vapor pressure of seawater, ``p_w(T) exp(B1 s + B2 s^2)`` ([S2] Eq. 6; Pa).

    Args:
        T: Temperature (K).
        S: Salinity (kg/kg).
    """
    s = 1000.0 * jnp.asarray(S, dtype=jnp.float64)
    b1, b2 = _PSAT_S
    return vapor_pressure_water(T) * jnp.exp(b1 * s + b2 * s**2)


def boiling_point_elevation(T, S) -> Array:
    """Boiling point elevation of seawater, [S1] Eq. 36 (K).

    Args:
        T: Temperature (K).
        S: Salinity (kg/kg).

    Returns:
        BPE (K). Stated accuracy +-0.018 K against Bromley et al.

    Example:
        >>> round(float(boiling_point_elevation(373.15, 0.07)), 3)
        1.139
    """
    t = _celsius(T)
    S = jnp.asarray(S, dtype=jnp.float64)
    A = _BPE_A[0] + _BPE_A[1] * t + _BPE_A[2] * t**2
    B = _BPE_B[0] + _BPE_B[1] * t + _BPE_B[2] * t**2
    return A * S**2 + B * S


def saturation_temperature(P, S=0.0, T0=350.0, n_iter: int = 8) -> Array:
    """Temperature (K) at which seawater of salinity S boils at P (Newton on
    :func:`vapor_pressure` in ``ln p``; differentiable).
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T0, dtype=jnp.float64) + 0.0 * P
    for _ in range(n_iter):
        f = jnp.log(vapor_pressure(T, S)) - jnp.log(P)
        # d ln p / dT from the Hyland-Wexler form
        a1, _, a3, a4, a5, a6 = _PSAT_W
        df = -a1 / T**2 + a3 + 2 * a4 * T + 3 * a5 * T**2 + a6 / T
        T = T - f / df
    return T
