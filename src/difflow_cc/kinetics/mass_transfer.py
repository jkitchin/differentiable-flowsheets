"""Mass transfer correlations for gas-liquid contactors.

This module provides correlations for mass transfer coefficients
in packed columns used for amine-based CO2 absorption.

All correlations are JAX-compatible for automatic differentiation.

The Onda functions (:func:`gas_film_coefficient`,
:func:`liquid_film_coefficient`, :func:`interfacial_area`) and the HTU/NTU
helpers are **deprecated thin wrappers** on :mod:`difflow.mass_transfer`
(``onda_gas_coefficient``, ``onda_liquid_coefficient``, ``onda_wetted_area``,
``htu_og``, ``ntu_og_colburn``). Their values changed in #427: the old
implementations disagreed with the published Onda (1968) form (see the core
docstrings). ``overall_mass_transfer`` is unchanged.

References:
    Onda K et al. (1968). Mass transfer coefficients between gas
        and liquid phases in packed columns. J Chem Eng Japan 1:56-62.
    Rocha JA et al. (1996). Distillation columns containing structured
        packings: A comprehensive model for their performance.
        Ind Eng Chem Res 35:1660-1667.
    Billet R, Schultes M (1999). Prediction of mass transfer columns
        with dumped and arranged packings. Trans IChemE 77:498-504.
"""

__all__ = [
    "gas_film_coefficient",
    "liquid_film_coefficient",
    "overall_mass_transfer",
    "interfacial_area",
]

import warnings

import jax.numpy as jnp
from jax import Array

from difflow import mass_transfer as _mt
from difflow.numerics import safe_divide


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"difflow_cc.kinetics.{old} is deprecated; use difflow.mass_transfer.{new}. "
        "Values differ from earlier versions (corrected Onda 1968 form).",
        DeprecationWarning,
        stacklevel=3,
    )


def gas_film_coefficient(
    u_G: Array | float,
    d_p: Array | float,
    mu_G: Array | float,
    rho_G: Array | float,
    D_G: Array | float,
    a_p: Array | float,
) -> Array:
    """Deprecated: use ``difflow.mass_transfer.onda_gas_coefficient`` (corrected).

    Gas-side coefficient ``k_c`` (m/s).

    Uses Onda correlation (1968):
        k_G * R * T / (a_p * D_G) = 5.23 * (G / (a_p * mu_G))^0.7
                                    * (mu_G / (rho_G * D_G))^(1/3)
                                    * (a_p * d_p)^-2
    (C = 2.00 for d_p < 15 mm); see the core docstring.

    Args:
        u_G: Superficial gas velocity (m/s)
        d_p: Packing nominal diameter (m)
        mu_G: Gas viscosity (Pa·s)
        rho_G: Gas density (kg/m³)
        D_G: Gas diffusivity (m²/s)
        a_p: Packing specific area (m²/m³)

    Returns:
        Gas-side mass transfer coefficient k_G (m/s)

    References:
        Onda K et al. (1968). J Chem Eng Japan 1:56.
    """
    _deprecated("gas_film_coefficient", "onda_gas_coefficient")
    return _mt.onda_gas_coefficient(u_G, rho_G, mu_G, D_G, a_p=a_p, d_p=d_p)


def liquid_film_coefficient(
    u_L: Array | float,
    d_p: Array | float,
    mu_L: Array | float,
    rho_L: Array | float,
    D_L: Array | float,
    a_p: Array | float,
    sigma: Array | float = 0.072,
    sigma_c: Array | float = 0.075,
) -> Array:
    """Deprecated: use ``difflow.mass_transfer.onda_liquid_coefficient`` (corrected).

    Liquid-side coefficient. ``sigma`` / ``sigma_c`` keep their old hidden
    defaults here only for backward compatibility.

    Uses Onda correlation (1968):
        k_L * (rho_L / (mu_L * g))^(1/3) = 0.0051 * (L / (a_w * mu_L))^(2/3)
                                          * (mu_L / (rho_L * D_L))^(-1/2)
                                          * (a_p * d_p)^0.4

    Args:
        u_L: Superficial liquid velocity (m/s)
        d_p: Packing nominal diameter (m)
        mu_L: Liquid viscosity (Pa·s)
        rho_L: Liquid density (kg/m³)
        D_L: Liquid diffusivity (m²/s)
        a_p: Packing specific area (m²/m³)
        sigma: Liquid surface tension (N/m)
        sigma_c: Critical surface tension of packing (N/m)

    Returns:
        Liquid-side mass transfer coefficient k_L (m/s)

    References:
        Onda K et al. (1968). J Chem Eng Japan 1:56.
    """
    _deprecated("liquid_film_coefficient", "onda_liquid_coefficient")
    return _mt.onda_liquid_coefficient(
        u_L, rho_L, mu_L, D_L, sigma, a_p=a_p, d_p=d_p, sigma_c=sigma_c
    )


def interfacial_area(
    u_L: Array | float,
    u_G: Array | float,
    rho_L: Array | float,
    mu_L: Array | float,
    sigma: Array | float,
    a_p: Array | float,
    sigma_c: Array | float = 0.075,
) -> Array:
    """Deprecated: use ``difflow.mass_transfer.onda_wetted_area``.

    Effective gas-liquid interfacial area.

    Uses Onda correlation (1968):
        a_w / a_p = 1 - exp(-1.45 * (sigma_c/sigma)^0.75
                          * Re_L^0.1 * Fr_L^-0.05 * We_L^0.2)

    Args:
        u_L: Superficial liquid velocity (m/s)
        u_G: Superficial gas velocity (m/s)
        rho_L: Liquid density (kg/m³)
        mu_L: Liquid viscosity (Pa·s)
        sigma: Liquid surface tension (N/m)
        a_p: Packing specific area (m²/m³)
        sigma_c: Critical surface tension of packing (N/m)

    Returns:
        Effective interfacial area a_w (m²/m³)

    References:
        Onda K et al. (1968). J Chem Eng Japan 1:56.
    """
    _deprecated("interfacial_area", "onda_wetted_area")
    return _mt.onda_wetted_area(u_L, rho_L, mu_L, sigma, a_p=a_p, sigma_c=sigma_c)


def overall_mass_transfer(
    k_G: Array | float,
    k_L: Array | float,
    E: Array | float,
    H: Array | float,
    P: Array | float,
) -> Array:
    """Calculate overall mass transfer coefficient.

    Two-film model with enhancement:
        1/K_G = 1/k_G + 1/(H * E * k_L)

    Args:
        k_G: Gas-side coefficient (m/s)
        k_L: Liquid-side coefficient (m/s)
        E: Enhancement factor
        H: Henry's constant (Pa·m³/mol)
        P: Total pressure (Pa)

    Returns:
        Overall mass transfer coefficient K_G (m/s)

    References:
        Danckwerts PV (1970). Gas-Liquid Reactions. McGraw-Hill.
    """
    k_G = jnp.asarray(k_G)
    k_L = jnp.asarray(k_L)
    E = jnp.asarray(E)
    H = jnp.asarray(H)
    P = jnp.asarray(P)

    # Resistances in series
    # Two-film model: 1/K_G = 1/k_G + 1/(H * E * k_L)
    R_G = safe_divide(1.0, k_G)
    R_L = safe_divide(1.0, H * E * k_L)

    K_G = safe_divide(1.0, R_G + R_L)

    return K_G


def height_of_transfer_unit(
    u_G: Array | float,
    K_G: Array | float,
    a_w: Array | float,
) -> Array:
    """Deprecated: use ``difflow.mass_transfer.htu_og``.

    Height of a transfer unit (HTU).

    HTU_G = u_G / (K_G * a_w)

    Args:
        u_G: Superficial gas velocity (m/s)
        K_G: Overall mass transfer coefficient (m/s)
        a_w: Effective interfacial area (m²/m³)

    Returns:
        Height of transfer unit (m)

    Notes:
        Column height = HTU * NTU
        where NTU is number of transfer units from driving force integration.
    """
    _deprecated("height_of_transfer_unit", "htu_og")
    return _mt.htu_og(u_G, K_G, a_w)


def number_of_transfer_units(
    y_in: Array | float,
    y_out: Array | float,
    y_eq_avg: Array | float,
) -> Array:
    """Deprecated: use ``difflow.mass_transfer.ntu_og_colburn``.

    Number of transfer units (NTU), constant equilibrium value.

    For dilute systems with linear equilibrium:
        NTU = ln((y_in - y_eq)/(y_out - y_eq))

    Args:
        y_in: Inlet gas mole fraction
        y_out: Outlet gas mole fraction
        y_eq_avg: Average equilibrium mole fraction

    Returns:
        Number of transfer units

    Notes:
        For more accurate calculation, integrate the driving force
        along the column. This simplified version assumes constant
        equilibrium (valid for low conversion or high L/G).
    """
    _deprecated("number_of_transfer_units", "ntu_og_colburn")
    y_in = jnp.asarray(y_in)
    y_out = jnp.asarray(y_out)
    y_eq_avg = jnp.asarray(y_eq_avg)
    # Constant equilibrium value y* = y_eq_avg is the lam = 0 Colburn limit
    # (m = 1, x_in = y*); the driving forces keep their historical floor.
    y_in = y_eq_avg + jnp.maximum(y_in - y_eq_avg, 1e-10)
    y_out = y_eq_avg + jnp.maximum(y_out - y_eq_avg, 1e-10)
    NTU = _mt.ntu_og_colburn(y_in, y_out, y_eq_avg, 1.0, 0.0)
    return jnp.maximum(NTU, 0.0)

