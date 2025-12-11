"""Extractant loading and saturation models.

When the organic phase approaches saturation with extracted REE,
the effective distribution coefficient decreases.

Models:
- Langmuir isotherm for extractant capacity
- Loading correction factors for D values
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array


# =============================================================================
# Loading Isotherm Models
# =============================================================================

@dataclass
class LoadingIsotherm:
    """Extractant loading isotherm model.

    Models the relationship between aqueous REE concentration
    and organic phase loading.

    Attributes:
        max_loading: Maximum REE loading capacity (mol REE / mol extractant)
        K_L: Langmuir constant (L/mol)
        extractant_conc: Extractant concentration (M)
    """
    max_loading: float = 0.33  # Typical for acidic extractants (1 REE per 3 HA)
    K_L: float = 10.0  # Langmuir constant
    extractant_conc: float = 0.5  # M

    @property
    def max_ree_conc(self) -> float:
        """Maximum REE concentration in organic (M)."""
        return self.max_loading * self.extractant_conc

    def loading(self, c_aq: Array | float) -> Array:
        """Calculate organic phase loading (Langmuir isotherm).

        q = q_max * K_L * c / (1 + K_L * c)

        Args:
            c_aq: Aqueous phase REE concentration (M)

        Returns:
            Organic phase REE concentration (M)
        """
        c_aq = jnp.asarray(c_aq)
        q_max = self.max_ree_conc
        return q_max * self.K_L * c_aq / (1 + self.K_L * c_aq)

    def loading_fraction(self, c_org: Array | float) -> Array:
        """Calculate fraction of extractant capacity used.

        Args:
            c_org: Organic phase REE concentration (M)

        Returns:
            Loading fraction (0 to 1)
        """
        c_org = jnp.asarray(c_org)
        return c_org / self.max_ree_conc

    def apparent_D(
        self,
        D_infinite: Array | float,
        c_org: Array | float,
    ) -> Array:
        """Calculate apparent D accounting for loading.

        At high loading, effective D decreases due to
        reduced free extractant concentration.

        D_app = D_inf * (1 - theta)^n

        where theta = loading fraction, n = stoichiometry

        Args:
            D_infinite: D at infinite dilution
            c_org: Current organic phase REE concentration (M)

        Returns:
            Apparent distribution coefficient
        """
        D_infinite = jnp.asarray(D_infinite)
        theta = self.loading_fraction(c_org)
        # Stoichiometry effect: each REE binds ~3 extractant molecules
        n = 3.0
        return D_infinite * jnp.power(jnp.maximum(1 - theta, 0.01), n)


def langmuir_loading(
    c_aq: Array | float,
    q_max: float,
    K_L: float,
) -> Array:
    """Langmuir isotherm for extractant loading.

    q = q_max * K_L * c / (1 + K_L * c)

    Args:
        c_aq: Aqueous phase concentration (M)
        q_max: Maximum loading capacity (M in organic)
        K_L: Langmuir constant (L/mol)

    Returns:
        Organic phase concentration at equilibrium
    """
    c_aq = jnp.asarray(c_aq)
    return q_max * K_L * c_aq / (1 + K_L * c_aq)


def freundlich_loading(
    c_aq: Array | float,
    K_F: float,
    n: float,
) -> Array:
    """Freundlich isotherm (empirical).

    q = K_F * c^(1/n)

    Args:
        c_aq: Aqueous phase concentration (M)
        K_F: Freundlich constant
        n: Freundlich exponent (n > 1 for favorable isotherm)

    Returns:
        Organic phase concentration
    """
    c_aq = jnp.asarray(c_aq)
    return K_F * jnp.power(c_aq, 1/n)


def langmuir_freundlich_loading(
    c_aq: Array | float,
    q_max: float,
    K_LF: float,
    n: float,
) -> Array:
    """Langmuir-Freundlich (Sips) isotherm.

    q = q_max * (K_LF * c)^n / (1 + (K_LF * c)^n)

    Combines features of both models.

    Args:
        c_aq: Aqueous phase concentration (M)
        q_max: Maximum loading capacity (M)
        K_LF: Langmuir-Freundlich constant
        n: Heterogeneity parameter

    Returns:
        Organic phase concentration
    """
    c_aq = jnp.asarray(c_aq)
    Kc_n = jnp.power(K_LF * c_aq, n)
    return q_max * Kc_n / (1 + Kc_n)


# =============================================================================
# Loading Correction for Multi-Component Systems
# =============================================================================

def loading_correction(
    D_values: dict[str, Array],
    c_org: dict[str, Array | float],
    isotherm: LoadingIsotherm,
) -> dict[str, Array]:
    """Apply loading correction to D values for all elements.

    In multi-component systems, total loading affects all D values.

    Args:
        D_values: Dictionary of infinite-dilution D values
        c_org: Current organic concentrations for each element
        isotherm: Loading isotherm model

    Returns:
        Corrected D values accounting for total loading
    """
    # Calculate total loading fraction
    total_c_org = sum(jnp.asarray(c) for c in c_org.values())
    theta_total = isotherm.loading_fraction(total_c_org)

    # Apply correction to all elements
    # D_app = D_inf * (1 - theta)^n
    n = 3.0  # Stoichiometry
    correction = jnp.power(jnp.maximum(1 - theta_total, 0.01), n)

    return {elem: D * correction for elem, D in D_values.items()}


def competitive_langmuir(
    c_aq: dict[str, Array | float],
    K_L: dict[str, float],
    q_max: float,
) -> dict[str, Array]:
    """Competitive Langmuir isotherm for multi-component system.

    q_i = q_max * K_i * c_i / (1 + sum_j(K_j * c_j))

    Args:
        c_aq: Aqueous concentrations for each species
        K_L: Langmuir constants for each species
        q_max: Total maximum capacity (shared)

    Returns:
        Organic phase concentrations for each species
    """
    # Calculate denominator: 1 + sum(K_j * c_j)
    denominator = 1.0
    for species, c in c_aq.items():
        denominator = denominator + K_L[species] * jnp.asarray(c)

    # Calculate loading for each species
    q_org = {}
    for species, c in c_aq.items():
        q_org[species] = q_max * K_L[species] * jnp.asarray(c) / denominator

    return q_org


# =============================================================================
# Extractant Capacity Data
# =============================================================================

# Typical maximum loading capacities (mol REE per mol extractant)
EXTRACTANT_CAPACITIES = {
    "D2EHPA": {
        "stoichiometry": 3,  # 3 extractant molecules per REE
        "max_loading": 0.33,  # mol REE / mol extractant
        "typical_K_L": {
            "La": 5.0,
            "Ce": 8.0,
            "Pr": 12.0,
            "Nd": 15.0,
            "Sm": 30.0,
            "Eu": 40.0,
            "Gd": 50.0,
            "Tb": 80.0,
            "Dy": 100.0,
            "Y": 70.0,
        },
    },
    "PC88A": {
        "stoichiometry": 3,
        "max_loading": 0.33,
        "typical_K_L": {
            "La": 3.0,
            "Ce": 5.0,
            "Pr": 8.0,
            "Nd": 12.0,
            "Sm": 25.0,
            "Eu": 35.0,
            "Gd": 45.0,
            "Tb": 70.0,
            "Dy": 90.0,
            "Y": 60.0,
        },
    },
    "Cyanex272": {
        "stoichiometry": 3,
        "max_loading": 0.33,
        "typical_K_L": {
            "La": 2.0,
            "Ce": 3.0,
            "Pr": 5.0,
            "Nd": 8.0,
            "Sm": 15.0,
            "Eu": 20.0,
            "Gd": 25.0,
            "Tb": 40.0,
            "Dy": 50.0,
            "Y": 35.0,
        },
    },
    "TBP": {
        "stoichiometry": 3,
        "max_loading": 0.33,
        "typical_K_L": {
            "La": 2.0,
            "Ce": 2.5,
            "Pr": 3.0,
            "Nd": 3.5,
            "Sm": 5.0,
            "Eu": 6.0,
            "Gd": 7.0,
            "Tb": 9.0,
            "Dy": 10.0,
            "Y": 8.0,
        },
    },
}


def get_loading_isotherm(
    extractant: str,
    concentration: float = 0.5,
) -> LoadingIsotherm:
    """Create loading isotherm for specified extractant.

    Args:
        extractant: Extractant name
        concentration: Extractant concentration (M)

    Returns:
        LoadingIsotherm instance
    """
    if extractant not in EXTRACTANT_CAPACITIES:
        raise ValueError(f"Unknown extractant: {extractant}")

    data = EXTRACTANT_CAPACITIES[extractant]
    # Use average K_L across elements
    avg_K_L = sum(data["typical_K_L"].values()) / len(data["typical_K_L"])

    return LoadingIsotherm(
        max_loading=data["max_loading"],
        K_L=avg_K_L,
        extractant_conc=concentration,
    )


def get_competitive_K_L(extractant: str) -> dict[str, float]:
    """Get Langmuir constants for competitive adsorption.

    Args:
        extractant: Extractant name

    Returns:
        Dictionary of K_L values for each REE
    """
    if extractant not in EXTRACTANT_CAPACITIES:
        raise ValueError(f"Unknown extractant: {extractant}")
    return EXTRACTANT_CAPACITIES[extractant]["typical_K_L"].copy()
