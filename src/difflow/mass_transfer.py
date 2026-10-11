"""Mass transfer: diffusivities, film theory, Sherwood correlations, Onda, HTU/NTU.

Pure functions, JAX, SI units, float64, differentiable. Nothing here is a
palette unit; the generic absorber and the column-hardware models call these
functions. Properties are *inputs*: core has no gas viscosity or surface
tension model, so :func:`onda_wetted_area` and friends take ``mu_G`` and
``sigma`` as required arguments (no hidden defaults such as a 0.072 N/m).

Contents
--------
* Diffusivities: :func:`fuller_diffusivity`, :func:`wilke_chang_diffusivity`,
  :func:`stokes_einstein_diffusivity`, tables :data:`FULLER_VOLUMES`,
  :data:`FULLER_ATOMIC_VOLUMES`, :data:`LE_BAS_INCREMENTS`,
  :data:`ASSOCIATION_FACTORS`, helpers :func:`diffusion_volume`,
  :func:`molar_volume_nbp`
* Film theory: :func:`log_mean`, :func:`equimolar_flux`,
  :func:`stagnant_film_flux`, :func:`mass_transfer_coefficient_conversions`,
  :func:`k_x_from_k_L`, :func:`k_L_from_k_x`, :func:`schmidt`,
  :func:`sherwood`, :func:`film_coefficient`
* Sherwood correlations: :func:`ranz_marshall`, :func:`frossling`,
  :func:`linton_sherwood`, :func:`gilliland_sherwood`,
  :func:`pipe_sherwood`, :func:`flat_plate_sherwood`,
  :func:`chilton_colburn_jD`, :func:`sherwood_from_heat_transfer`
* Two-resistance theory: :func:`overall_gas_coefficient`,
  :func:`overall_liquid_coefficient`, :func:`overall_K_G`,
  :func:`resistance_fractions`
* Packed columns: :class:`PackingData`, :data:`PACKINGS`,
  :func:`onda_wetted_area`, :func:`onda_gas_coefficient`,
  :func:`onda_liquid_coefficient`, :func:`htu_og`, :func:`htu_ol`,
  :func:`htu_og_from_films`, :func:`ntu_og_colburn`, :func:`ntu_og_integral`,
  :func:`hetp_from_htu`

Units
-----
Everything public is SI (K, Pa, Pa s, m, m^2/s, mol). Fuller and Wilke-Chang
are published in cm^2/s, atm, cP and cm^3/mol; they are implemented in those
units internally and converted at the boundary. Two inputs keep the published
unit: Wilke-Chang's solute molar volume ``V_A`` is in cm^3/mol and molar
masses are in g/mol (= kg/kmol); Fuller's diffusion volumes are the
dimensionless table values.

Provenance and verification
---------------------------
Correlations are implemented from the formulas as published (Fuller,
Schettler & Giddings 1966; Wilke & Chang 1955; Onda, Takeuchi & Okumoto
1968; Colburn 1939), written from memory of the sources, not copied from a
scan. What was and was not checked is stated in ``tests/test_mass_transfer.py``
and in the docs; in short: the water vapour-air diffusivity at 25 C agrees
with the data value (about 2.5e-5 m^2/s) and the Colburn NTU with its own
quadrature, but **no published worked example is reproduced** (no textbook is
available to verify numbers against). The tables (:data:`FULLER_VOLUMES`,
:data:`LE_BAS_INCREMENTS`, :data:`ASSOCIATION_FACTORS`, :data:`PACKINGS`) are
transcribed from memory and flagged as unverified where it matters.

Example:
    >>> from difflow.mass_transfer import fuller_diffusivity, FULLER_VOLUMES
    >>> D = fuller_diffusivity(298.15, 101325.0, 18.015, 28.96,
    ...                        FULLER_VOLUMES["H2O"], FULLER_VOLUMES["air"])
    >>> 2.3e-5 < float(D) < 2.7e-5      # water vapour in air, ~2.5e-5 m^2/s
    True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow import heat_transfer
from difflow.heat_transfer import _smooth_weight

_f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)  # noqa: E731

#: Universal gas constant (J/mol/K).
R_GAS = 8.314462618
#: Boltzmann constant (J/K).
K_BOLTZMANN = 1.380649e-23
#: Standard gravity (m/s^2).
GRAVITY = 9.80665
_ATM = 101325.0


# =============================================================================
# Small smooth helpers
# =============================================================================


def _log1p_over(z: Array) -> Array:
    """``ln(1 + z)/z`` with the removable singularity at ``z = 0`` handled smoothly.

    A Taylor series (to ``z^5``) is used for ``|z| < 1e-3`` where it is exact to
    machine precision; it joins the closed form with matching value and
    derivatives to ~1e-16, so gradients are continuous. Both branches are
    evaluated on safe arguments, so no NaN reaches ``jax.grad``.
    """
    z = _f64(z)
    small = jnp.abs(z) < 1e-3
    zs = jnp.where(small, 1.0, z)
    direct = jnp.log1p(zs) / zs
    series = 1.0 - z / 2.0 + z**2 / 3.0 - z**3 / 4.0 + z**4 / 5.0 - z**5 / 6.0
    return jnp.where(small, series, direct)


def log_mean(a: Array | float, b: Array | float) -> Array:
    """Logarithmic mean ``(a - b)/ln(a/b)`` of two positive numbers, smooth at ``a = b``.

    Used for the inert-gas log mean ``(1 - y)_lm`` of stagnant-film diffusion.
    With ``x = (a - b)/(a + b)`` it equals ``(a + b)/2 * x/atanh(x)``; for
    ``|x| < 0.05`` a series in ``x^2`` (through ``x^8``, error ``< 1e-13``) is
    used instead, so the function and its gradients are finite and continuous
    when the arguments coincide.

    Args:
        a: First positive value.
        b: Second positive value.

    Returns:
        The log mean (same units as ``a``).

    Example:
        >>> float(log_mean(2.0, 2.0))
        2.0
    """
    a = _f64(a)
    b = _f64(b)
    s = a + b
    x = (a - b) / s
    small = jnp.abs(x) < 0.05
    xs = jnp.where(small, 0.5, x)
    direct = jnp.where(small, 1.0, xs / jnp.arctanh(xs))
    x2 = x * x
    # x/atanh(x) = 1 - x^2/3 - 4 x^4/45 - 44 x^6/945 - 428 x^8/14175 ...
    series = 1.0 - x2 / 3.0 - 4.0 * x2**2 / 45.0 - 44.0 * x2**3 / 945.0 - 428.0 * x2**4 / 14175.0
    return 0.5 * s * jnp.where(small, series, direct)


# =============================================================================
# Diffusion-volume and molar-volume tables (FROM MEMORY; see docstrings)
# =============================================================================

#: Atomic diffusion volumes for Fuller-Schettler-Giddings (dimensionless).
#: Source believed to be Fuller, Ensley & Giddings (1969) as tabulated in
#: Poling, Prausnitz & O'Connell, *Properties of Gases and Liquids* 5th ed.,
#: Table 11-1. Transcribed from memory; C, H, O, N, Cl, S are the ones I am
#: confident in. Aromatic/heterocyclic ring: ``-18.3``.
FULLER_ATOMIC_VOLUMES: dict[str, float] = {
    "C": 15.9,
    "H": 2.31,
    "O": 6.11,
    "N": 4.54,
    "F": 14.7,
    "Cl": 21.0,
    "Br": 21.9,
    "I": 29.8,
    "S": 22.9,
    "ring": -18.3,
}

#: Diffusion volumes of simple molecules (dimensionless), same source and
#: caveat as :data:`FULLER_ATOMIC_VOLUMES` (PPO'C 5th ed. Table 11-1; older
#: editions list air 20.1 and H2O 12.7, which change D by < 1%).
FULLER_VOLUMES: dict[str, float] = {
    "He": 2.67,
    "Ne": 5.98,
    "Ar": 16.2,
    "Kr": 24.5,
    "Xe": 32.7,
    "H2": 6.12,
    "D2": 6.84,
    "N2": 18.5,
    "O2": 16.3,
    "air": 19.7,
    "CO": 18.0,
    "CO2": 26.9,
    "N2O": 35.9,
    "NH3": 20.7,
    "H2O": 13.1,
    "SF6": 71.3,
    "Cl2": 38.4,
    "Br2": 69.0,
    "SO2": 41.8,
}

#: Le Bas atomic increments to the molar volume at the normal boiling point
#: (cm^3/mol). Source believed to be Le Bas (1915) as tabulated in Reid,
#: Prausnitz & Poling and Treybal; **from memory, unverified**. ``O`` is the
#: value for alcohols, aldehydes and ketones; other oxygen environments
#: differ (``O_ether`` 9.1, ``O_acid`` 12.0). ``N_prim`` / ``N_sec`` are
#: nitrogen in primary / secondary amines.
LE_BAS_INCREMENTS: dict[str, float] = {
    "C": 14.8,
    "H": 3.7,
    "O": 7.4,
    "O_ether": 9.1,
    "O_acid": 12.0,
    "N": 15.6,
    "N_prim": 10.5,
    "N_sec": 12.0,
    "F": 8.7,
    "Cl": 21.6,
    "Br": 27.0,
    "I": 37.0,
    "S": 25.6,
    "ring3": -6.0,
    "ring4": -8.5,
    "ring5": -11.5,
    "ring6": -15.0,
    "naphthalene": -30.0,
    "anthracene": -47.5,
}

#: Molar volumes at the normal boiling point of common dissolved gases and
#: liquids (cm^3/mol), as tabulated in Treybal / Welty (believed); **from
#: memory, unverified**. Prefer :func:`molar_volume_nbp` for a molecule not
#: listed.
MOLAR_VOLUMES_NBP: dict[str, float] = {
    "O2": 25.6,
    "N2": 31.2,
    "CO2": 34.0,
    "NH3": 25.8,
    "H2": 14.3,
    "H2O": 18.9,
    "air": 29.9,
}

#: Wilke-Chang association factors of the *solvent* ``phi_B``. Source: Wilke &
#: Chang (1955), as quoted in every textbook (water 2.6, methanol 1.9,
#: ethanol 1.5, unassociated solvents 1.0). High confidence.
ASSOCIATION_FACTORS: dict[str, float] = {
    "water": 2.6,
    "methanol": 1.9,
    "ethanol": 1.5,
    "unassociated": 1.0,
    "benzene": 1.0,
    "ether": 1.0,
    "heptane": 1.0,
}


def diffusion_volume(species_or_atoms: str | Mapping[str, float]) -> float:
    """Fuller diffusion volume of a molecule.

    Args:
        species_or_atoms: A key of :data:`FULLER_VOLUMES` (``"H2O"``, ``"CO2"``,
            ``"air"`` ...) or an atom-count mapping such as
            ``{"C": 4, "H": 10, "O": 1}``, summed with
            :data:`FULLER_ATOMIC_VOLUMES` (use ``"ring": 1`` for each aromatic
            or heterocyclic ring).

    Returns:
        The diffusion volume (dimensionless).

    Raises:
        KeyError: Unknown species or atom.

    Example:
        >>> round(diffusion_volume({"C": 2, "H": 6, "O": 1}), 2)   # ethanol
        51.77
    """
    if isinstance(species_or_atoms, str):
        return FULLER_VOLUMES[species_or_atoms]
    return float(sum(FULLER_ATOMIC_VOLUMES[a] * n for a, n in species_or_atoms.items()))


def molar_volume_nbp(increments: Mapping[str, float]) -> float:
    """Molar volume at the normal boiling point by Le Bas additive increments (cm^3/mol).

    Args:
        increments: Mapping from a key of :data:`LE_BAS_INCREMENTS` to a count,
            e.g. ethanol ``{"C": 2, "H": 6, "O": 1}`` -> 59.2 cm^3/mol; add a
            ring term (``"ring6": 1``) for cyclic molecules.

    Returns:
        ``V_A`` in cm^3/mol, the input of :func:`wilke_chang_diffusivity`.

    Example:
        >>> round(molar_volume_nbp({"C": 2, "H": 6, "O": 1}), 1)
        59.2
    """
    return float(sum(LE_BAS_INCREMENTS[k] * n for k, n in increments.items()))


# =============================================================================
# Diffusivities
# =============================================================================


def fuller_diffusivity(
    T: Array | float,
    P: Array | float,
    MW_A: Array | float,
    MW_B: Array | float,
    V_A: Array | float,
    V_B: Array | float,
) -> Array:
    """Binary gas diffusivity, Fuller-Schettler-Giddings (1966).

    ``D_AB [cm^2/s] = 1e-3 T^1.75 (1/M_A + 1/M_B)^(1/2) /
    (P [atm] (V_A^(1/3) + V_B^(1/3))^2)`` with ``T`` in K, ``M`` in g/mol.
    Implemented in those units and converted to SI at the boundary. Typical
    accuracy 5-10% (non-polar and moderately polar gases, 1 atm up to a few
    atm; degrades at high pressure). Scales as ``T^1.75 / P``.

    Args:
        T: Temperature (K).
        P: Pressure (Pa).
        MW_A: Molar mass of A (g/mol, or kg/kmol).
        MW_B: Molar mass of B (g/mol).
        V_A: Diffusion volume of A (see :data:`FULLER_VOLUMES`,
            :func:`diffusion_volume`).
        V_B: Diffusion volume of B.

    Returns:
        ``D_AB`` (m^2/s).

    Example:
        Water vapour in air, 25 C, 1 atm (experimental about 2.5e-5 m^2/s):

        >>> D = fuller_diffusivity(298.15, 101325.0, 18.015, 28.96, 13.1, 19.7)
        >>> round(float(D) * 1e5, 2)
        2.51
    """
    T = _f64(T)
    P_atm = _f64(P) / _ATM
    MW_A = _f64(MW_A)
    MW_B = _f64(MW_B)
    s = _f64(V_A) ** (1.0 / 3.0) + _f64(V_B) ** (1.0 / 3.0)
    D_cm2 = 1.0e-3 * T**1.75 * jnp.sqrt(1.0 / MW_A + 1.0 / MW_B) / (P_atm * s**2)
    return D_cm2 * 1.0e-4


def wilke_chang_diffusivity(
    T: Array | float,
    mu_B: Array | float,
    MW_B: Array | float,
    V_A: Array | float,
    phi_B: Array | float = 1.0,
) -> Array:
    """Dilute liquid diffusivity, Wilke-Chang (1955).

    ``D_AB [cm^2/s] = 7.4e-8 (phi_B M_B)^(1/2) T / (mu_B [cP] V_A^0.6)`` with
    ``V_A`` the solute molar volume at its normal boiling point in cm^3/mol.
    Converted to SI at the boundary. Typical error 10-20% for dilute organic
    solutes in water (it tends to over-predict small polar solutes);
    ``phi_B`` has **no default for water**: pass ``2.6`` (see
    :data:`ASSOCIATION_FACTORS`); the default 1.0 is for an unassociated
    solvent.

    Args:
        T: Temperature (K).
        mu_B: Solvent viscosity (Pa s), e.g. from
            :func:`difflow.liquid_properties.liquid_viscosity`.
        MW_B: Solvent molar mass (g/mol).
        V_A: Solute molar volume at the normal boiling point (cm^3/mol); see
            :func:`molar_volume_nbp`, :data:`MOLAR_VOLUMES_NBP`.
        phi_B: Solvent association factor (-).

    Returns:
        ``D_AB`` (m^2/s).

    Example:
        Ethanol (``V_A`` = 59.2) in water at 25 C (experimental about
        1.2e-9; Wilke-Chang over-predicts by roughly 20%):

        >>> D = wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, 59.2, 2.6)
        >>> round(float(D) * 1e9, 2)
        1.47
    """
    mu_cP = _f64(mu_B) * 1.0e3
    D_cm2 = 7.4e-8 * jnp.sqrt(_f64(phi_B) * _f64(MW_B)) * _f64(T) / (mu_cP * _f64(V_A) ** 0.6)
    return D_cm2 * 1.0e-4


def stokes_einstein_diffusivity(T: Array | float, mu: Array | float, r: Array | float) -> Array:
    """Stokes-Einstein diffusivity ``D = k_B T/(6 pi mu r)`` of a sphere of radius ``r``.

    Valid for a large solute (``r`` much bigger than a solvent molecule:
    proteins, colloids, ions with a hydrodynamic radius); it under-predicts D
    of small molecules. Same law as ``difflow_ree.kinetics.diffusion_coefficient``.

    Args:
        T: Temperature (K).
        mu: Solvent viscosity (Pa s).
        r: Hydrodynamic radius (m).

    Returns:
        ``D`` (m^2/s).

    Example:
        >>> D = stokes_einstein_diffusivity(298.15, 0.89e-3, 2.0e-10)
        >>> 1e-9 < float(D) < 2e-9
        True
    """
    return K_BOLTZMANN * _f64(T) / (6.0 * math.pi * _f64(mu) * _f64(r))


# =============================================================================
# Fick's law, film theory, coefficient conversions
# =============================================================================


def schmidt(mu: Array | float, rho: Array | float, D: Array | float) -> Array:
    """Schmidt number ``Sc = mu/(rho D)``."""
    return _f64(mu) / (_f64(rho) * _f64(D))


def sherwood(k: Array | float, L: Array | float, D: Array | float) -> Array:
    """Sherwood number ``Sh = k L/D`` (``k`` a concentration-driven coefficient, m/s)."""
    return _f64(k) * _f64(L) / _f64(D)


def film_coefficient(Sh: Array | float, D: Array | float, L: Array | float) -> Array:
    """Mass-transfer coefficient ``k_c = Sh D/L`` (m/s) from a Sherwood number."""
    return _f64(Sh) * _f64(D) / _f64(L)


def equimolar_flux(
    D: Array | float, c: Array | float, y_A1: Array | float, y_A2: Array | float, delta: Array | float
) -> Array:
    """Flux for equimolar counter-diffusion: ``N_A = D c (y_A1 - y_A2)/delta``.

    Args:
        D: Diffusivity (m^2/s).
        c: Total molar concentration (mol/m^3), ``P/(R T)`` for a gas.
        y_A1: Mole fraction at the start of the film (``z = 0``).
        y_A2: Mole fraction at the end (``z = delta``).
        delta: Film thickness (m).

    Returns:
        Molar flux from 1 to 2 (mol/m^2/s).
    """
    return _f64(D) * _f64(c) * (_f64(y_A1) - _f64(y_A2)) / _f64(delta)


def stagnant_film_flux(
    D: Array | float, c: Array | float, y_A1: Array | float, y_A2: Array | float, delta: Array | float
) -> Array:
    """Flux of A through a stagnant film of inert B (Stefan flow).

    ``N_A = (D c/delta) (y_A1 - y_A2)/(1 - y)_lm`` with the log mean of the
    inert mole fractions ``(1 - y_A2)`` and ``(1 - y_A1)`` from
    :func:`log_mean` (smooth when the two coincide). Tends to
    :func:`equimolar_flux` as ``y -> 0`` and exceeds it by the bulk-flow
    factor ``1/(1 - y)_lm`` otherwise.

    Args:
        D: Diffusivity (m^2/s).
        c: Total molar concentration (mol/m^3).
        y_A1: Mole fraction of A at ``z = 0`` (the interface, for evaporation).
        y_A2: Mole fraction at ``z = delta``.
        delta: Film (diffusion path) length (m).

    Returns:
        Molar flux of A in the ``+z`` direction (mol/m^2/s).

    Example:
        Evaporation of water at 25 C into dry air through a 0.1 m film
        (D = 2.5e-5 m^2/s):

        >>> from difflow.units.evaporator import water_saturation_pressure
        >>> y1 = float(water_saturation_pressure(298.15)) / 101325.0
        >>> N = stagnant_film_flux(2.5e-5, 101325.0 / (8.314 * 298.15), y1, 0.0, 0.1)
        >>> 2e-4 < float(N) < 4e-4
        True
    """
    y1 = _f64(y_A1)
    y2 = _f64(y_A2)
    inert_lm = log_mean(1.0 - y2, 1.0 - y1)
    return _f64(D) * _f64(c) / _f64(delta) * (y1 - y2) / inert_lm


def mass_transfer_coefficient_conversions(
    k: Array | float, frm: str, to: str, T: Array | float, P: Array | float
) -> Array:
    """Convert a gas-phase coefficient between the ``k_c``, ``k_G`` and ``k_y`` bases.

    ``k_y = k_G P = k_c P/(R T)`` for an ideal gas:

    * ``"c"``: ``k_c`` in m/s (concentration driving force);
    * ``"G"``: ``k_G`` in mol/m^2/s/Pa (partial-pressure driving force);
    * ``"y"``: ``k_y`` in mol/m^2/s (mole-fraction driving force).

    For the liquid side use :func:`k_x_from_k_L` / :func:`k_L_from_k_x`.

    Args:
        k: Coefficient in basis ``frm``.
        frm: ``"c"``, ``"G"`` or ``"y"``.
        to: ``"c"``, ``"G"`` or ``"y"``.
        T: Temperature (K).
        P: Total pressure (Pa).

    Returns:
        The coefficient in basis ``to``.

    Raises:
        ValueError: Unknown basis name.
    """
    T = _f64(T)
    P = _f64(P)
    k = _f64(k)
    c_tot = P / (R_GAS * T)
    to_y = {"c": c_tot, "G": P, "y": 1.0}
    if frm not in to_y or to not in to_y:
        raise ValueError("bases must be 'c', 'G' or 'y'")
    return k * to_y[frm] / to_y[to]


def k_x_from_k_L(k_L: Array | float, c_total: Array | float) -> Array:
    """Liquid-phase ``k_x = k_L c`` (mol/m^2/s) from ``k_L`` (m/s), ``c`` = total mol/m^3."""
    return _f64(k_L) * _f64(c_total)


def k_L_from_k_x(k_x: Array | float, c_total: Array | float) -> Array:
    """Liquid-phase ``k_L = k_x/c`` (m/s)."""
    return _f64(k_x) / _f64(c_total)


# =============================================================================
# Sherwood correlations
# =============================================================================


def ranz_marshall(Re: Array | float, Sc: Array | float) -> Array:
    """Ranz-Marshall (1952) sphere: ``Sh = 2 + 0.6 Re^(1/2) Sc^(1/3)``.

    ``Re = rho v d/mu`` on the sphere diameter, ``Sh = k d/D``. Valid for
    ``2 < Re < 800`` and ``0.6 < Sc < 2.5`` (gases), commonly used well beyond
    for liquids with ``Re < 1e4`` at reduced accuracy. Tends to the
    stagnant-medium limit ``Sh = 2`` as ``Re -> 0`` (the conduction/diffusion
    limit), where the ``Re^(1/2)`` term has a finite value but an infinite
    slope: its gradient at exactly ``Re = 0`` is guarded.

    Args:
        Re: Particle Reynolds number.
        Sc: Schmidt number.

    Returns:
        Sherwood number (-).
    """
    Re = jnp.maximum(_f64(Re), 1e-300)
    return 2.0 + 0.6 * jnp.sqrt(Re) * _f64(Sc) ** (1.0 / 3.0)


def frossling(Re: Array | float, Sc: Array | float) -> Array:
    """Frossling (1938) sphere: ``Sh = 2 + 0.552 Re^(1/2) Sc^(1/3)``.

    Same form and range as :func:`ranz_marshall` with the coefficient 0.552.
    """
    Re = jnp.maximum(_f64(Re), 1e-300)
    return 2.0 + 0.552 * jnp.sqrt(Re) * _f64(Sc) ** (1.0 / 3.0)


def linton_sherwood(Re: Array | float, Sc: Array | float) -> Array:
    """Linton-Sherwood (1950) turbulent pipe flow: ``Sh = 0.023 Re^0.83 Sc^(1/3)``.

    ``Re = rho v D/mu`` on the diameter; valid ``2000 < Re < 7e4``,
    ``1 < Sc < 1000`` (liquids). ``Sh = k D/D_AB``.
    """
    return 0.023 * _f64(Re) ** 0.83 * _f64(Sc) ** (1.0 / 3.0)


def gilliland_sherwood(Re: Array | float, Sc: Array | float) -> Array:
    """Gilliland-Sherwood (1934) turbulent pipe flow: ``Sh = 0.023 Re^0.83 Sc^0.44``.

    Valid ``2000 < Re < 35000``, ``0.6 < Sc < 2.5`` (gases in wetted-wall
    columns).
    """
    return 0.023 * _f64(Re) ** 0.83 * _f64(Sc) ** 0.44


#: Fully developed laminar Sherwood number in a circular tube, uniform wall
#: concentration (the mass analogue of Nu = 3.66).
SH_LAMINAR_TUBE = heat_transfer.NU_LAMINAR_TW


def pipe_sherwood(Re: Array | float, Sc: Array | float, method: str = "linton") -> Array:
    """Tube Sherwood number, laminar to turbulent, smoothly blended.

    ``Sh = 3.66`` (fully developed laminar, uniform wall concentration) below
    ``Re = 2300``, the turbulent correlation (``method="linton"`` or
    ``"gilliland"``) above ``Re = 4000``, joined by the C2 weight
    :func:`difflow.heat_transfer._smooth_weight`. In the blend zone the value
    is an interpolation, not a measurement. Entrance effects are not included.

    Args:
        Re: Reynolds number on the diameter.
        Sc: Schmidt number.
        method: ``"linton"`` or ``"gilliland"``.

    Returns:
        Sherwood number (-).

    Raises:
        ValueError: Unknown ``method``.
    """
    if method == "linton":
        turb = linton_sherwood
    elif method == "gilliland":
        turb = gilliland_sherwood
    else:
        raise ValueError("method must be 'linton' or 'gilliland'")
    Re = jnp.maximum(_f64(Re), 1e-12)
    w = _smooth_weight(Re, heat_transfer.RE_LAMINAR, heat_transfer.RE_TURBULENT)
    return (1.0 - w) * SH_LAMINAR_TUBE + w * turb(jnp.maximum(Re, 1.0), Sc)


def flat_plate_sherwood(Re_L: Array | float, Sc: Array | float, regime: str = "mixed") -> Array:
    """Mean Sherwood number over a flat plate: the mass analogue of ``flat_plate_nusselt``.

    ``"laminar"``: ``0.664 Re_L^(1/2) Sc^(1/3)``; ``"turbulent"``:
    ``0.037 Re_L^(4/5) Sc^(1/3)``; ``"mixed"``: ``(0.037 Re_L^(4/5) - 871)
    Sc^(1/3)`` with a smooth laminar join. Delegates to
    :func:`difflow.heat_transfer.flat_plate_nusselt` with ``Pr -> Sc`` (the
    standard heat/mass analogy; valid for ``Sc >= 0.6`` and low mass-transfer
    rates).
    """
    return heat_transfer.flat_plate_nusselt(Re_L, Sc, regime)


def chilton_colburn_jD(Nu: Array | float, Re: Array | float, Pr: Array | float) -> Array:
    """Colburn j-factor of a heat-transfer result: ``j = Nu/(Re Pr^(1/3))``.

    The Chilton-Colburn analogy sets ``j_D = j_H``; then
    ``Sh = j Re Sc^(1/3)`` (see :func:`sherwood_from_heat_transfer`). Valid for
    ``0.6 < Sc < 2500`` and ``0.6 < Pr < 100`` and no form drag.
    """
    return _f64(Nu) / (_f64(Re) * _f64(Pr) ** (1.0 / 3.0))


def sherwood_from_heat_transfer(
    Nu: Array | float, Re: Array | float, Pr: Array | float, Sc: Array | float
) -> Array:
    """Sherwood number from a Nusselt number by the Chilton-Colburn analogy.

    ``Sh = Nu (Sc/Pr)^(1/3)`` (``j_D = j_H`` at equal ``Re``).
    """
    j = chilton_colburn_jD(Nu, Re, Pr)
    return j * _f64(Re) * _f64(Sc) ** (1.0 / 3.0)


# =============================================================================
# Two-resistance theory
# =============================================================================


def overall_gas_coefficient(
    k_y: Array | float, k_x: Array | float, m: Array | float, E: Array | float = 1.0
) -> Array:
    """Overall gas-phase coefficient: ``1/K_y = 1/k_y + m/(E k_x)``.

    ``m = dy*/dx`` is the slope of the equilibrium line (dimensionless, in
    mole-fraction units); ``E >= 1`` is an optional chemical-enhancement
    factor multiplying ``k_x`` (reactive absorption).

    Returns:
        ``K_y`` (same units as ``k_y``, mol/m^2/s).
    """
    return 1.0 / (1.0 / _f64(k_y) + _f64(m) / (_f64(E) * _f64(k_x)))


def overall_liquid_coefficient(
    k_y: Array | float, k_x: Array | float, m: Array | float, E: Array | float = 1.0
) -> Array:
    """Overall liquid-phase coefficient: ``1/K_x = 1/(m k_y) + 1/(E k_x)``."""
    return 1.0 / (1.0 / (_f64(m) * _f64(k_y)) + 1.0 / (_f64(E) * _f64(k_x)))


def overall_K_G(
    k_G: Array | float, k_L: Array | float, H: Array | float, E: Array | float = 1.0
) -> Array:
    """Overall ``K_G`` (mol/m^2/s/Pa) from film coefficients and Henry's constant.

    ``1/K_G = 1/k_G + H/(E k_L)`` with ``p_A* = H c_A``, ``H`` in Pa m^3/mol,
    ``k_L`` in m/s. (Equivalent to ``1/K_G = 1/k_G + 1/(H' E k_L)`` with
    ``H' = 1/H`` the solubility in mol/m^3/Pa.)
    """
    return 1.0 / (1.0 / _f64(k_G) + _f64(H) / (_f64(E) * _f64(k_L)))


def resistance_fractions(
    k_y: Array | float, k_x: Array | float, m: Array | float, E: Array | float = 1.0
) -> tuple[Array, Array]:
    """Fractions of the overall resistance in the gas and liquid films.

    ``(1/k_y)/(1/K_y)`` and ``(m/(E k_x))/(1/K_y)``; they sum to 1. Close to 1
    for the gas fraction means gas-film control (very soluble solute such as
    NH3 in water, small ``m``); close to 0 means liquid-film control (sparingly
    soluble solute such as CO2 in water, large ``m``).

    Returns:
        ``(gas_fraction, liquid_fraction)``.
    """
    Rg = 1.0 / _f64(k_y)
    Rl = _f64(m) / (_f64(E) * _f64(k_x))
    tot = Rg + Rl
    return Rg / tot, Rl / tot


# =============================================================================
# Packings
# =============================================================================


@dataclass(frozen=True)
class PackingData:
    """Random-packing record shared by the Onda and column-hardware models.

    Attributes:
        name: Identifier.
        d_p: Nominal packing size (m).
        a_p: Dry specific surface area (m^2/m^3).
        eps: Void fraction (-).
        sigma_c: Critical surface tension of the packing material (N/m).
        F_p: Packing factor for pressure-drop/flooding charts (1/m), or ``None``
            if not recorded.
        source: Provenance note.
    """

    name: str
    d_p: float
    a_p: float
    eps: float
    sigma_c: float
    F_p: float | None = None
    source: str = "from memory, unverified"


#: Critical surface tension of packing materials (N/m). Onda et al. (1968) as
#: quoted by Treybal / Seader (believed): ceramic 61, steel 75, plastic
#: (polyethylene 33, PVC 40), carbon 56, glass 73, paraffin 20 mN/m. From memory.
SIGMA_C: dict[str, float] = {
    "ceramic": 0.061,
    "steel": 0.075,
    "polyethylene": 0.033,
    "pvc": 0.040,
    "carbon": 0.056,
    "glass": 0.073,
    "paraffin": 0.020,
}

#: A small set of common random packings. **All values are from memory
#: (Billet-Schultes 1999 for ``a_p``/``eps`` of Pall rings and Raschig rings,
#: Perry's Handbook for packing factors) and unverified**; check against a
#: manufacturer's data sheet before design use. Build your own
#: :class:`PackingData` for anything else.
PACKINGS: dict[str, PackingData] = {
    "raschig_ceramic_25": PackingData("raschig_ceramic_25", 0.025, 190.0, 0.68, SIGMA_C["ceramic"], 510.0),
    "berl_ceramic_25": PackingData("berl_ceramic_25", 0.025, 260.0, 0.68, SIGMA_C["ceramic"], 360.0),
    "pall_metal_25": PackingData("pall_metal_25", 0.025, 223.5, 0.954, SIGMA_C["steel"], 183.0),
    "pall_metal_50": PackingData("pall_metal_50", 0.050, 112.6, 0.951, SIGMA_C["steel"], 89.0),
    "pall_plastic_25": PackingData("pall_plastic_25", 0.025, 225.0, 0.90, SIGMA_C["polyethylene"], 180.0),
    # Added in #429 (column hardware). From memory, unverified: Intalox ceramic a_p/eps
    # and packing factors as in Treybal's table (98, 52, 40 ft^-1 -> 320, 170, 131 1/m);
    # Pall 38 mm after Billet-Schultes / Perry (about 131 1/m).
    "intalox_ceramic_25": PackingData("intalox_ceramic_25", 0.025, 255.0, 0.73, SIGMA_C["ceramic"], 320.0),
    "intalox_ceramic_38": PackingData("intalox_ceramic_38", 0.038, 195.0, 0.75, SIGMA_C["ceramic"], 170.0),
    "intalox_ceramic_50": PackingData("intalox_ceramic_50", 0.050, 118.0, 0.76, SIGMA_C["ceramic"], 131.0),
    "pall_metal_38": PackingData("pall_metal_38", 0.038, 139.5, 0.959, SIGMA_C["steel"], 131.0),
    # Structured sheet-metal packing: d_p = 0 (not defined; the Onda random-packing
    # functions do not apply). F_p ~ 66 1/m (20 1/ft) from memory, unverified.
    "mellapak_250y": PackingData("mellapak_250y", 0.0, 250.0, 0.95, SIGMA_C["steel"], 66.0),
}


def _packing(
    packing: PackingData | None, a_p, d_p, sigma_c, need_sigma_c: bool = True, need_d_p: bool = True
):
    """Resolve ``(a_p, d_p, sigma_c)`` from a record and/or explicit overrides."""
    if packing is not None:
        a_p = packing.a_p if a_p is None else a_p
        d_p = packing.d_p if d_p is None else d_p
        sigma_c = packing.sigma_c if sigma_c is None else sigma_c
    if a_p is None:
        raise ValueError("give `packing` or a_p")
    if need_d_p and d_p is None:
        raise ValueError("give `packing` or d_p")
    if need_sigma_c and sigma_c is None:
        raise ValueError("give `packing` or sigma_c")
    return (
        _f64(a_p),
        None if d_p is None else _f64(d_p),
        None if sigma_c is None else _f64(sigma_c),
    )


# =============================================================================
# Onda (1968) packed-column correlations
# =============================================================================


def onda_wetted_area(
    u_L: Array | float,
    rho_L: Array | float,
    mu_L: Array | float,
    sigma: Array | float,
    packing: PackingData | None = None,
    *,
    a_p: Array | float | None = None,
    sigma_c: Array | float | None = None,
) -> Array:
    """Wetted (interfacial) area of random packing, Onda, Takeuchi & Okumoto (1968).

    ``a_w/a_p = 1 - exp(-1.45 (sigma_c/sigma)^0.75 Re_L^0.1 Fr_L^-0.05
    We_L^0.2)`` with ``Re_L = L/(a_p mu_L)``, ``Fr_L = L^2 a_p/(rho_L^2 g)``
    and ``We_L = L^2/(rho_L sigma a_p)`` for the liquid mass flux
    ``L = rho_L u_L`` (kg/m^2/s). Note the three groups use the *dry* area
    ``a_p``. Reported scatter about +/-20%; fitted for ``sigma_c/sigma`` below
    about 2 (water on ceramic/steel/plastic). Unchanged by #427 relative to
    the old ``difflow_cc`` function, which already agreed with the published
    form.

    Args:
        u_L: Superficial liquid velocity (m/s).
        rho_L: Liquid density (kg/m^3).
        mu_L: Liquid viscosity (Pa s).
        sigma: Liquid surface tension (N/m). Required.
        packing: :class:`PackingData` supplying ``a_p`` and ``sigma_c``.
        a_p: Dry packing area (m^2/m^3); overrides ``packing``.
        sigma_c: Critical surface tension of the packing (N/m); overrides
            ``packing``.

    Returns:
        ``a_w`` (m^2/m^3); strictly below ``a_p``.
    """
    a_p, _, sigma_c = _packing(packing, a_p, None, sigma_c, need_d_p=False)
    u_L = _f64(u_L)
    rho_L = _f64(rho_L)
    mu_L = _f64(mu_L)
    sigma = _f64(sigma)
    Re = rho_L * u_L / (a_p * mu_L)
    Fr = u_L**2 * a_p / GRAVITY
    We = rho_L * u_L**2 / (sigma * a_p)
    expo = -1.45 * (sigma_c / sigma) ** 0.75 * Re**0.1 * Fr**-0.05 * We**0.2
    return a_p * (1.0 - jnp.exp(expo))


#: Packing size (m) at which Onda's gas-film constant changes from 2.00 to 5.23.
ONDA_SIZE_SWITCH = 0.015


def onda_gas_coefficient(
    u_G: Array | float,
    rho_G: Array | float,
    mu_G: Array | float,
    D_G: Array | float,
    packing: PackingData | None = None,
    *,
    a_p: Array | float | None = None,
    d_p: Array | float | None = None,
    smooth: bool = True,
) -> Array:
    """Gas-film coefficient ``k_c`` (m/s) of random packing, Onda et al. (1968).

    Published form: ``k_G R T/(a_p D_G) = C (G/(a_p mu_G))^0.7 (mu_G/(rho_G
    D_G))^(1/3) (a_p d_p)^(-2)`` with ``C = 5.23`` for ``d_p >= 15 mm`` and
    ``C = 2.00`` for ``d_p < 15 mm``, ``G = rho_G u_G`` (kg/m^2/s). The
    left-hand side is ``k_c = k_G R T`` (m/s, concentration driving force), so

        ``k_c = C D_G a_p^(-1) d_p^(-2) Re_G^0.7 Sc_G^(1/3)``  with
        ``Re_G = G/(a_p mu_G)`` (dry area ``a_p``).

    Convert with :func:`mass_transfer_coefficient_conversions` to ``k_G``
    (mol/m^2/s/Pa) or ``k_y``. (The previous ``difflow_cc`` function returned
    ``5.23 (D_G/d_p) Re^0.7 Sc^(1/3)``, too large by a factor ``a_p d_p``, and
    never used 2.00.) The usual caveat of the Onda correlation applies: ~+/-20%
    for random packings, and ``k_G a_w`` (not ``k_G`` alone) is what is
    correlated well with the wetted area.

    Args:
        u_G: Superficial gas velocity (m/s).
        rho_G: Gas density (kg/m^3).
        mu_G: Gas viscosity (Pa s). Required (core has no gas-viscosity model).
        D_G: Gas diffusivity of the solute (m^2/s), e.g. :func:`fuller_diffusivity`.
        packing: :class:`PackingData` supplying ``a_p`` and ``d_p``.
        a_p: Dry packing area (m^2/m^3); overrides ``packing``.
        d_p: Nominal packing size (m); overrides ``packing``.
        smooth: If True (default) the constant switches 2.00 -> 5.23 through a
            C2 blend between 12 and 18 mm (so ``d_p`` is differentiable); if
            False it is the published hard step at 15 mm.

    Returns:
        ``k_c`` (m/s).
    """
    a_p, d_p, _ = _packing(packing, a_p, d_p, None, need_sigma_c=False)
    rho_G = _f64(rho_G)
    mu_G = _f64(mu_G)
    D_G = _f64(D_G)
    Re = rho_G * _f64(u_G) / (a_p * mu_G)
    Sc = mu_G / (rho_G * D_G)
    if smooth:
        w = _smooth_weight(d_p, 0.012, 0.018)
        C = 2.00 + (5.23 - 2.00) * w
    else:
        C = jnp.where(d_p >= ONDA_SIZE_SWITCH, 5.23, 2.00)
    return C * D_G / (a_p * d_p**2) * Re**0.7 * Sc ** (1.0 / 3.0)


def onda_liquid_coefficient(
    u_L: Array | float,
    rho_L: Array | float,
    mu_L: Array | float,
    D_L: Array | float,
    sigma: Array | float,
    packing: PackingData | None = None,
    *,
    a_p: Array | float | None = None,
    d_p: Array | float | None = None,
    sigma_c: Array | float | None = None,
    a_w: Array | float | None = None,
) -> Array:
    """Liquid-film coefficient ``k_L`` (m/s) of random packing, Onda et al. (1968).

    Published form: ``k_L (rho_L/(mu_L g))^(1/3) = 0.0051 (L/(a_w mu_L))^(2/3)
    (mu_L/(rho_L D_L))^(-1/2) (a_p d_p)^0.4`` with ``L = rho_L u_L`` and the
    liquid Reynolds number on the **wetted** area ``a_w``
    (:func:`onda_wetted_area`; pass ``a_w`` to reuse a value). Solving,
    ``k_L = 0.0051 Re_w^(2/3) Sc^(-1/2) (a_p d_p)^0.4 (mu_L g/rho_L)^(1/3)``.
    Order of magnitude for water in 25-50 mm packing: 1e-4 m/s.

    Corrections relative to the old ``difflow_cc.kinetics`` function, which
    contradicted its own docstring: the viscous-length factor is
    ``(mu_L g/rho_L)^(1/3)`` (it used ``(rho_L g/mu_L)^(1/3)``, a factor ~1e4
    too large: 0.49 m/s instead of 7.7e-5 m/s for water at u_L = 0.01 m/s),
    and ``Re`` is on ``a_w``, not ``a_p``.

    Args:
        u_L: Superficial liquid velocity (m/s).
        rho_L: Liquid density (kg/m^3).
        mu_L: Liquid viscosity (Pa s).
        D_L: Solute diffusivity in the liquid (m^2/s), e.g.
            :func:`wilke_chang_diffusivity`.
        sigma: Liquid surface tension (N/m). Required.
        packing: :class:`PackingData` supplying ``a_p``, ``d_p``, ``sigma_c``.
        a_p: Dry packing area (m^2/m^3); overrides ``packing``.
        d_p: Nominal packing size (m); overrides ``packing``.
        sigma_c: Critical surface tension (N/m); overrides ``packing``.
        a_w: Wetted area (m^2/m^3); computed by :func:`onda_wetted_area` if None.

    Returns:
        ``k_L`` (m/s).
    """
    a_p, d_p, sigma_c = _packing(packing, a_p, d_p, sigma_c, need_sigma_c=a_w is None)
    rho_L = _f64(rho_L)
    mu_L = _f64(mu_L)
    if a_w is None:
        a_w = onda_wetted_area(u_L, rho_L, mu_L, sigma, a_p=a_p, sigma_c=sigma_c)
    a_w = _f64(a_w)
    Re_w = rho_L * _f64(u_L) / (a_w * mu_L)
    Sc = mu_L / (rho_L * _f64(D_L))
    return 0.0051 * Re_w ** (2.0 / 3.0) * Sc**-0.5 * (a_p * d_p) ** 0.4 * (mu_L * GRAVITY / rho_L) ** (1.0 / 3.0)


# =============================================================================
# HTU / NTU / HETP
# =============================================================================


def htu_og(G_m: Array | float, K_y: Array | float, a: Array | float) -> Array:
    """Overall gas-phase transfer unit height ``H_OG = G_m/(K_y a)`` (m).

    Args:
        G_m: Gas molar flux (mol/m^2/s).
        K_y: Overall gas coefficient (mol/m^2/s).
        a: Interfacial area per volume (m^2/m^3).
    """
    return _f64(G_m) / (_f64(K_y) * _f64(a))


def htu_ol(L_m: Array | float, K_x: Array | float, a: Array | float) -> Array:
    """Overall liquid-phase transfer unit height ``H_OL = L_m/(K_x a)`` (m)."""
    return _f64(L_m) / (_f64(K_x) * _f64(a))


def htu_og_from_films(H_G: Array | float, H_L: Array | float, lam: Array | float) -> Array:
    """``H_OG = H_G + lambda H_L`` with stripping factor ``lambda = m G_m/L_m``.

    ``H_G = G_m/(k_y a)`` and ``H_L = L_m/(k_x a)`` are the film heights.
    """
    return _f64(H_G) + _f64(lam) * _f64(H_L)


def ntu_og_colburn(
    y_in: Array | float,
    y_out: Array | float,
    x_in: Array | float,
    m: Array | float,
    lam: Array | float,
) -> Array:
    """Colburn's analytic ``N_OG`` for straight operating and equilibrium lines.

    ``N_OG = ln[(1 - lam)(y_in - m x_in)/(y_out - m x_in) + lam]/(1 - lam)``
    with ``lam = m G/L`` (dilute absorption: ``y*= m x``, operating line
    through ``(x_in, y_out)``). Written as ``u ln(1+eps u)/(eps u)`` with
    ``eps = 1 - lam`` and ``u = (y_in - y_out)/(y_out - m x_in)``, so
    ``lam -> 1`` (where the limit is ``u``) and ``lam = 0`` (constant ``y*``,
    ``N = ln(1 + u)``) are smooth with finite gradients.

    Args:
        y_in: Inlet (bottom) gas mole fraction.
        y_out: Outlet (top) gas mole fraction.
        x_in: Inlet (top) liquid mole fraction.
        m: Equilibrium slope ``y* = m x``.
        lam: Stripping factor ``m G/L``.

    Returns:
        ``N_OG`` (-). Not clipped: a negative or NaN result flags an
        infeasible spec (``y_out <= m x_in``, i.e. below the pinch).
    """
    y_in = _f64(y_in)
    y_out = _f64(y_out)
    u = (y_in - y_out) / (y_out - _f64(m) * _f64(x_in))
    eps = 1.0 - _f64(lam)
    return u * _log1p_over(eps * u)


_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def ntu_og_integral(
    y_in: Array | float,
    y_out: Array | float,
    y_star_fn: Callable[[Array], Array],
    n: int = 64,
    concentrated: bool = False,
) -> Array:
    """``N_OG = integral dy/(y - y*)`` by fixed Gauss-Legendre quadrature, for curved equilibrium.

    ``N_OG = int_{y_out}^{y_in} dy/(y - y*(y))``; with ``concentrated=True``
    the integrand is ``(1 - y)_lm/((1 - y)(y - y*))`` with
    ``(1 - y)_lm`` the log mean of ``1 - y`` and ``1 - y*`` (concentrated
    absorption, ``N_OG`` defined on ``H_OG = G'/(K_y a (1-y)_lm)``-type
    bases). The ``n``-point grid is fixed (differentiable, jit-able). Exact to
    quadrature error for a smooth integrand; check convergence by doubling
    ``n`` if ``y*`` is nearly tangent to the operating line.

    Args:
        y_in: Inlet gas mole fraction.
        y_out: Outlet gas mole fraction.
        y_star_fn: Function ``y -> y*`` giving the equilibrium gas composition
            in equilibrium with the liquid at the operating-line point where
            the gas is ``y`` (compose the operating line into it).
        n: Number of Gauss-Legendre nodes.
        concentrated: Use the ``(1-y)_lm`` correction.

    Returns:
        ``N_OG`` (-).
    """
    y_in = _f64(y_in)
    y_out = _f64(y_out)
    xg, wg = _gauss_legendre(n)
    half = 0.5 * (y_in - y_out)
    y = 0.5 * (y_in + y_out) + half * jnp.asarray(xg)
    ys = y_star_fn(y)
    f = 1.0 / (y - ys)
    if concentrated:
        f = f * log_mean(1.0 - ys, 1.0 - y) / (1.0 - y)
    return half * jnp.sum(jnp.asarray(wg) * f)


def hetp_from_htu(H_OG: Array | float, lam: Array | float) -> Array:
    """``HETP = H_OG ln(lam)/(lam - 1)``, continuous through ``lam = 1``.

    Computed as ``H_OG ln(1+d)/d`` with ``d = lam - 1`` (series near ``d = 0``,
    limit ``HETP = H_OG``). Gradient-safe at ``lam = 1``.

    Args:
        H_OG: Overall transfer-unit height (m).
        lam: Stripping factor ``m G/L``.
    """
    return _f64(H_OG) * _log1p_over(_f64(lam) - 1.0)
