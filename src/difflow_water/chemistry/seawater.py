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
========================  =================  ==================

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

#: [S1] validity ranges: (t_min C, t_max C, S_min, S_max)
RANGES = {
    "density": (0.0, 180.0, 0.0, 0.16),
    "viscosity": (0.0, 180.0, 0.0, 0.15),
    "osmotic_coefficient": (0.0, 200.0, 0.010, 0.120),
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
