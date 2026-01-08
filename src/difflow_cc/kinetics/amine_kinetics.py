"""Reaction kinetics for amine-CO2 absorption.

This module provides kinetic rate expressions for CO2 absorption
into amine solutions. The models are based on the zwitterion
mechanism for primary/secondary amines and base-catalyzed
hydration for tertiary amines.

All functions are JAX-compatible for automatic differentiation.

References:
    Caplow M (1968). Kinetics of carbamate formation and breakdown.
        J Am Chem Soc 90:6795-6803.
    Versteeg GF et al. (1996). On the kinetics between CO2 and
        alkanolamines both in aqueous and non-aqueous solutions.
        Chem Eng Sci 51:3181-3195.
    Bishnoi S, Rochelle GT (2000). Absorption of CO2 into aqueous
        piperazine: reaction kinetics, mass transfer and solubility.
        Chem Eng Sci 55:5531-5543.
"""

import jax.numpy as jnp
from jax import Array

from difflow_cc.database import get_solvent
from difflow_cc.equilibrium.solubility import diffusivity_co2_amine

# Gas constant
R = 8.314  # J/(mol*K)


def reaction_rate_constant(
    T: Array | float,
    solvent: str,
) -> Array:
    """Calculate second-order reaction rate constant k2.

    For the zwitterion mechanism:
        CO2 + RNH2 -> RNH2+COO-  (rate-determining step)

    k2 = A * exp(-Ea / (R*T))

    Args:
        T: Temperature (K)
        solvent: Amine solvent name

    Returns:
        Second-order rate constant k2 (L/(mol·s))

    References:
        Versteeg GF et al. (1996). Chem Eng Sci 51:3181.
    """
    T = jnp.asarray(T)
    s = get_solvent(solvent)
    kinetics = s.kinetics

    # Convert to float to avoid JAX tracing issues with dict access
    A = float(kinetics["A"])  # Pre-exponential factor
    Ea = float(kinetics["Ea"])  # Activation energy (J/mol)

    k2 = A * jnp.exp(-Ea / (R * T))
    return k2


def pseudo_first_order_rate(
    T: Array | float,
    solvent: str,
    C_amine: Array | float,
) -> Array:
    """Calculate pseudo-first-order rate constant.

    For excess amine conditions:
        r = k1 * [CO2]

    where k1 = k2 * [Amine]

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)

    Returns:
        Pseudo-first-order rate constant k1 (1/s)
    """
    T = jnp.asarray(T)
    C_amine = jnp.asarray(C_amine)

    k2 = reaction_rate_constant(T, solvent)

    # Convert C_amine from mol/m³ to mol/L for consistency
    C_amine_L = C_amine / 1000

    k1 = k2 * C_amine_L
    return k1


def hatta_number(
    T: Array | float,
    solvent: str,
    C_amine: Array | float,
    kL: Array | float,
) -> Array:
    """Calculate Hatta number for gas-liquid reaction.

    Ha = sqrt(k1 * D_CO2) / kL

    The Hatta number characterizes the enhancement of mass
    transfer due to chemical reaction:
    - Ha < 0.3: Slow reaction (kinetic regime)
    - 0.3 < Ha < 3: Intermediate regime
    - Ha > 3: Fast reaction (diffusion-controlled)
    - Ha >> 3: Instantaneous reaction

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)
        kL: Liquid-side mass transfer coefficient (m/s)

    Returns:
        Hatta number (dimensionless)

    References:
        Danckwerts PV (1970). Gas-Liquid Reactions. McGraw-Hill.
    """
    T = jnp.asarray(T)
    kL = jnp.asarray(kL)

    k1 = pseudo_first_order_rate(T, solvent, C_amine)
    D_CO2 = diffusivity_co2_amine(T, solvent, C_amine)

    Ha = jnp.sqrt(k1 * D_CO2) / (kL + 1e-10)
    return Ha


def enhancement_factor(
    T: Array | float,
    solvent: str,
    C_amine: Array | float,
    kL: Array | float,
    regime: str = "auto",
) -> Array:
    """Calculate enhancement factor for mass transfer.

    The enhancement factor E accounts for the increase in
    absorption rate due to chemical reaction.

    For pseudo-first-order regime (Ha > 3):
        E = Ha / tanh(Ha) ≈ Ha

    For instantaneous regime (E_inf):
        E_inf = 1 + D_amine/D_CO2 * C_amine/(C_CO2_i * nu)

    Args:
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)
        kL: Liquid-side mass transfer coefficient (m/s)
        regime: 'auto', 'pseudo_first_order', or 'instantaneous'

    Returns:
        Enhancement factor E (dimensionless)

    References:
        van Swaaij WPM, Versteeg GF (1992). Mass transfer accompanied
            with complex reversible chemical reactions in gas-liquid
            systems. Chem Eng Sci 47:3181-3195.
    """
    T = jnp.asarray(T)

    Ha = hatta_number(T, solvent, C_amine, kL)

    if regime == "pseudo_first_order" or regime == "auto":
        # E = Ha / tanh(Ha) for fast reaction
        # For large Ha, E ≈ Ha
        E = jnp.where(
            Ha > 0.1,
            Ha / jnp.tanh(Ha + 1e-10),
            1.0 + Ha**2 / 3  # Taylor expansion for small Ha
        )
    else:
        # Instantaneous regime (simplified)
        # E_inf = 1 + D_Am/D_CO2 * C_Am / (C_CO2 * nu)
        # Approximate with large value for fast reactions
        E = jnp.maximum(Ha, 1.0)

    return E


def overall_reaction_rate(
    P_CO2: Array | float,
    T: Array | float,
    solvent: str,
    C_amine: Array | float,
    loading: Array | float = 0.0,
) -> Array:
    """Calculate overall CO2 absorption rate.

    Rate includes both forward reaction and equilibrium effects.

    r = k_ov * (C_CO2 - C_CO2_eq)

    Args:
        P_CO2: CO2 partial pressure (Pa)
        T: Temperature (K)
        solvent: Amine solvent name
        C_amine: Amine concentration (mol/m³)
        loading: Current CO2 loading (mol CO2/mol amine)

    Returns:
        Absorption rate (mol/(m³·s))

    Notes:
        This is a simplified rate expression. For rigorous modeling,
        the full speciation and equilibrium should be considered.
    """
    T = jnp.asarray(T)
    P_CO2 = jnp.asarray(P_CO2)
    C_amine = jnp.asarray(C_amine)
    loading = jnp.asarray(loading)

    # Physical solubility
    from difflow_cc.equilibrium.solubility import co2_physical_solubility
    H = co2_physical_solubility(T, solvent, C_amine)
    C_CO2 = P_CO2 / H  # mol/m³

    # Free amine (approximate)
    s = get_solvent(solvent)
    alpha_max = s.loading_capacity
    free_amine_fraction = 1.0 - loading / alpha_max
    C_amine_free = C_amine * free_amine_fraction

    # Reaction rate
    k2 = reaction_rate_constant(T, solvent)
    C_amine_L = C_amine_free / 1000  # mol/L
    C_CO2_L = C_CO2 / 1000  # mol/L

    # Second-order rate
    r = k2 * C_CO2_L * C_amine_L * 1000  # mol/(m³·s)

    return r
