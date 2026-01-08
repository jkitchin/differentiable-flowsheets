"""Mass transfer correlations for gas-liquid contactors.

This module provides correlations for mass transfer coefficients
in packed columns used for amine-based CO2 absorption.

All correlations are JAX-compatible for automatic differentiation.

References:
    Onda K et al. (1968). Mass transfer coefficients between gas
        and liquid phases in packed columns. J Chem Eng Japan 1:56-62.
    Rocha JA et al. (1996). Distillation columns containing structured
        packings: A comprehensive model for their performance.
        Ind Eng Chem Res 35:1660-1667.
    Billet R, Schultes M (1999). Prediction of mass transfer columns
        with dumped and arranged packings. Trans IChemE 77:498-504.
"""

import jax.numpy as jnp
from jax import Array


# Constants
g = 9.81  # m/s²


def gas_film_coefficient(
    u_G: Array | float,
    d_p: Array | float,
    mu_G: Array | float,
    rho_G: Array | float,
    D_G: Array | float,
    a_p: Array | float,
) -> Array:
    """Calculate gas-side mass transfer coefficient.

    Uses Onda correlation (1968):
        k_G * R * T / (a_p * D_G) = 5.23 * (G / (a_p * mu_G))^0.7
                                    * (mu_G / (rho_G * D_G))^(1/3)
                                    * (a_p * d_p)^-2

    Simplified form for packed columns:
        k_G = C * (u_G / a_p)^0.7 * (D_G / d_p)^0.67

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
    u_G = jnp.asarray(u_G)
    d_p = jnp.asarray(d_p)
    mu_G = jnp.asarray(mu_G)
    rho_G = jnp.asarray(rho_G)
    D_G = jnp.asarray(D_G)
    a_p = jnp.asarray(a_p)

    # Reynolds number
    Re_G = rho_G * u_G / (a_p * mu_G)

    # Schmidt number
    Sc_G = mu_G / (rho_G * D_G)

    # Onda correlation
    k_G = 5.23 * (D_G / d_p) * jnp.power(Re_G, 0.7) * jnp.power(Sc_G, 1/3)

    return k_G


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
    """Calculate liquid-side mass transfer coefficient.

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
    u_L = jnp.asarray(u_L)
    d_p = jnp.asarray(d_p)
    mu_L = jnp.asarray(mu_L)
    rho_L = jnp.asarray(rho_L)
    D_L = jnp.asarray(D_L)
    a_p = jnp.asarray(a_p)
    sigma = jnp.asarray(sigma)
    sigma_c = jnp.asarray(sigma_c)

    # Reynolds number
    Re_L = rho_L * u_L / (a_p * mu_L)

    # Schmidt number
    Sc_L = mu_L / (rho_L * D_L)

    # Froude number
    Fr_L = u_L**2 * a_p / g

    # Weber number
    We_L = rho_L * u_L**2 / (sigma * a_p)

    # Characteristic length
    L_char = jnp.power(mu_L**2 / (rho_L**2 * g), 1/3)

    # Onda correlation
    k_L = 0.0051 * jnp.power(Re_L, 2/3) * jnp.power(Sc_L, -0.5) * \
          jnp.power(a_p * d_p, 0.4) * jnp.power(rho_L * g / mu_L, 1/3)

    return k_L


def interfacial_area(
    u_L: Array | float,
    u_G: Array | float,
    rho_L: Array | float,
    mu_L: Array | float,
    sigma: Array | float,
    a_p: Array | float,
    sigma_c: Array | float = 0.075,
) -> Array:
    """Calculate effective gas-liquid interfacial area.

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
    u_L = jnp.asarray(u_L)
    rho_L = jnp.asarray(rho_L)
    mu_L = jnp.asarray(mu_L)
    sigma = jnp.asarray(sigma)
    a_p = jnp.asarray(a_p)
    sigma_c = jnp.asarray(sigma_c)

    # Dimensionless numbers
    Re_L = rho_L * u_L / (a_p * mu_L)
    Fr_L = u_L**2 * a_p / g
    We_L = rho_L * u_L**2 / (sigma * a_p)

    # Onda correlation
    ratio = sigma_c / sigma
    exponent = -1.45 * jnp.power(ratio, 0.75) * \
               jnp.power(Re_L, 0.1) * jnp.power(Fr_L, -0.05) * jnp.power(We_L, 0.2)

    a_w = a_p * (1 - jnp.exp(exponent))

    return a_w


def overall_mass_transfer(
    k_G: Array | float,
    k_L: Array | float,
    E: Array | float,
    H: Array | float,
    P: Array | float,
) -> Array:
    """Calculate overall mass transfer coefficient.

    Two-film model with enhancement:
        1/K_G = 1/k_G + H/(E * k_L * P)

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
    R_G = 1 / (k_G + 1e-10)
    R_L = H / (E * k_L * P + 1e-10)

    K_G = 1 / (R_G + R_L)

    return K_G


def height_of_transfer_unit(
    u_G: Array | float,
    K_G: Array | float,
    a_w: Array | float,
) -> Array:
    """Calculate height of a transfer unit (HTU).

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
    u_G = jnp.asarray(u_G)
    K_G = jnp.asarray(K_G)
    a_w = jnp.asarray(a_w)

    HTU = u_G / (K_G * a_w + 1e-10)

    return HTU


def number_of_transfer_units(
    y_in: Array | float,
    y_out: Array | float,
    y_eq_avg: Array | float,
) -> Array:
    """Calculate number of transfer units (NTU).

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
    y_in = jnp.asarray(y_in)
    y_out = jnp.asarray(y_out)
    y_eq_avg = jnp.asarray(y_eq_avg)

    # Log-mean driving force
    df_in = y_in - y_eq_avg
    df_out = y_out - y_eq_avg

    # Avoid log of negative numbers
    df_in = jnp.maximum(df_in, 1e-10)
    df_out = jnp.maximum(df_out, 1e-10)

    NTU = jnp.log(df_in / df_out)

    return jnp.maximum(NTU, 0.0)
