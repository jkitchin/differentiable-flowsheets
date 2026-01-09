"""CO2 solubility and diffusivity correlations for amine solutions.

This module provides transport property correlations needed for
rate-based mass transfer calculations in amine absorption systems.

All correlations are JAX-compatible for automatic differentiation.

References:
    Versteeg GF, Van Swaaij WPM (1988). Solubility and diffusivity
        of acid gases (CO2, N2O) in aqueous alkanolamine solutions.
        J Chem Eng Data 33:29-34.
    Snijder ED et al. (1993). Diffusion coefficients of several
        aqueous alkanolamine solutions. J Chem Eng Data 38:475-480.
    Haimour N, Sandall OC (1984). Absorption of carbon dioxide into
        aqueous methyldiethanolamine. Chem Eng Sci 39:1791-1796.
"""

__all__ = [
    "co2_physical_solubility",
    "diffusivity_co2_amine",
]

import jax.numpy as jnp
from jax import Array

from difflow_cc.database import get_solvent

# Gas constant
R = 8.314  # J/(mol*K)


def co2_physical_solubility(
    T: Array | float,
    solvent: str = "H2O",
    C_amine: Array | float = 0.0
) -> Array:
    """Calculate physical solubility of CO2 (Henry's constant).

    For pure water:
        ln(H_CO2/Pa) = -6789.04/T - 11.4519*ln(T) + 94.4914

    For amine solutions, uses N2O analogy with correction factor.

    Args:
        T: Temperature (K)
        solvent: Solvent name ('H2O' for pure water, or amine name)
        C_amine: Amine concentration (mol/m³), only used if solvent is H2O

    Returns:
        Henry's constant H (Pa·m³/mol)

    References:
        Versteeg GF, Van Swaaij WPM (1988). J Chem Eng Data 33:29.
    """
    T = jnp.asarray(T)

    if solvent == "H2O" or C_amine == 0.0:
        # Pure water correlation
        ln_H = -6789.04 / T - 11.4519 * jnp.log(T) + 94.4914
    else:
        # Amine solution - use N2O analogy
        # H_CO2,amine = H_CO2,water * (H_N2O,amine / H_N2O,water)
        # Simplified: reduction factor based on amine concentration
        s = get_solvent(solvent)
        ln_H_water = -6789.04 / T - 11.4519 * jnp.log(T) + 94.4914

        # Correction for amine presence (reduces solubility slightly)
        # Based on correlations from Versteeg & Van Swaaij
        C_amine = jnp.asarray(C_amine)
        correction = 1.0 + 0.1 * C_amine / 1000  # Empirical
        ln_H = ln_H_water + jnp.log(correction)

    return jnp.exp(ln_H)


def diffusivity_co2_water(T: Array | float) -> Array:
    """Diffusivity of CO2 in pure water.

    D_CO2 = D0 * exp(-Ed/(R*T))

    D0 = 2.35e-6 m²/s
    Ed = 2119 K (17620 J/mol)

    Args:
        T: Temperature (K)

    Returns:
        Diffusivity (m²/s)

    References:
        Versteeg GF, Van Swaaij WPM (1988). J Chem Eng Data 33:29.
    """
    T = jnp.asarray(T)
    D0 = 2.35e-6  # m²/s
    Ed = 17620.0  # J/mol
    return D0 * jnp.exp(-Ed / (R * T))


def diffusivity_co2_amine(
    T: Array | float,
    solvent: str,
    C_amine: Array | float = 5000.0
) -> Array:
    """Diffusivity of CO2 in aqueous amine solution.

    Accounts for viscosity increase due to amine using
    Stokes-Einstein relationship:

        D_amine = D_water * (mu_water / mu_amine)^0.8

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)

    Returns:
        Diffusivity (m²/s)

    References:
        Snijder ED et al. (1993). J Chem Eng Data 38:475.
        Uses modified Stokes-Einstein with exponent 0.8.
    """
    T = jnp.asarray(T)
    C_amine = jnp.asarray(C_amine)

    # CO2 diffusivity in water
    D_water = diffusivity_co2_water(T)

    # Get solvent viscosity parameters
    s = get_solvent(solvent)

    # Viscosity ratio
    # mu_water ~ 0.89 cP at 25°C
    # mu_amine from Arrhenius: mu = A * exp(B/T)
    mu_water = 0.001 * jnp.exp(1800 / T - 4.5)  # Approximate, Pa·s
    mu_amine = s.viscosity_A * jnp.exp(s.viscosity_B / T) * 0.001  # Pa·s

    # Concentration effect - viscosity increases with concentration
    # Approximate: mu ~ mu_pure * (1 + k*C)
    C_ref = 5000.0  # Reference concentration
    k_visc = 0.3  # Empirical
    visc_factor = 1.0 + k_visc * C_amine / C_ref
    mu_amine = mu_amine * visc_factor

    # Modified Stokes-Einstein
    D_amine = D_water * jnp.power(mu_water / mu_amine, 0.8)

    return D_amine


def diffusivity_amine(
    T: Array | float,
    solvent: str,
    C_amine: Array | float = 5000.0
) -> Array:
    """Diffusivity of amine in aqueous solution.

    D_amine ~ 0.6 * D_CO2 (typical ratio)

    More accurate correlations exist but this provides
    reasonable estimates for mass transfer calculations.

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)

    Returns:
        Diffusivity (m²/s)

    References:
        Snijder ED et al. (1993). J Chem Eng Data 38:475.
    """
    D_CO2 = diffusivity_co2_amine(T, solvent, C_amine)
    return 0.6 * D_CO2


def viscosity_amine_solution(
    T: Array | float,
    solvent: str,
    C_amine: Array | float = 5000.0,
    loading: Array | float = 0.0
) -> Array:
    """Viscosity of loaded amine solution.

    Viscosity increases with:
    - Lower temperature
    - Higher amine concentration
    - Higher CO2 loading

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)
        loading: CO2 loading (mol CO2 / mol amine)

    Returns:
        Dynamic viscosity (Pa·s)

    References:
        Weiland RH et al. (1998). Density and viscosity of some
            partially carbonated aqueous alkanolamine solutions.
            J Chem Eng Data 43:378-382.
    """
    T = jnp.asarray(T)
    C_amine = jnp.asarray(C_amine)
    loading = jnp.asarray(loading)

    s = get_solvent(solvent)

    # Base viscosity from Arrhenius
    mu_base = s.viscosity_A * jnp.exp(s.viscosity_B / T) * 0.001  # Pa·s

    # Concentration effect
    C_ref = 5000.0
    k_conc = 0.3
    conc_factor = 1.0 + k_conc * C_amine / C_ref

    # Loading effect (viscosity increases with loading)
    k_load = 0.5
    load_factor = 1.0 + k_load * loading / s.loading_capacity

    return mu_base * conc_factor * load_factor


def density_amine_solution(
    T: Array | float,
    solvent: str,
    C_amine: Array | float = 5000.0,
    loading: Array | float = 0.0
) -> Array:
    """Density of loaded amine solution.

    Simple mixing rule with loading correction.

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)
        loading: CO2 loading (mol CO2 / mol amine)

    Returns:
        Density (kg/m³)

    References:
        Weiland RH et al. (1998). J Chem Eng Data 43:378.
    """
    T = jnp.asarray(T)
    C_amine = jnp.asarray(C_amine)
    loading = jnp.asarray(loading)

    s = get_solvent(solvent)

    # Pure water density (kg/m³)
    rho_water = 1000 * (1 - 0.0002 * (T - 293.15))

    # Amine contribution
    # mass_amine = C_amine * MW / 1000 (kg/m³)
    mass_amine = C_amine * s.MW / 1000

    # Approximate density by mass-weighted average
    rho_amine = s.density  # kg/m³
    w_amine = mass_amine / (mass_amine + rho_water)
    rho_base = w_amine * rho_amine + (1 - w_amine) * rho_water

    # Loading increases density (absorbed CO2)
    # Each mol CO2 adds 44 g
    CO2_mass = loading * C_amine * 44 / 1000  # kg/m³
    rho_loaded = rho_base + CO2_mass

    return rho_loaded
