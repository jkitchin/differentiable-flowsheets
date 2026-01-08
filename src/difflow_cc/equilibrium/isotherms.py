"""Adsorption isotherm models for CO2 capture.

This module provides JAX-compatible implementations of common
adsorption isotherm models used in pressure swing adsorption (PSA),
temperature swing adsorption (TSA), and related processes.

All functions use JAX numpy for automatic differentiation compatibility.

Models implemented:
- Langmuir: Single-site adsorption
- Sips (Langmuir-Freundlich): Heterogeneous surfaces
- Toth: Asymmetric quasi-Gaussian distribution
- Dual-site Langmuir: Two distinct adsorption sites

References:
    Do DD (1998). Adsorption Analysis: Equilibria and Kinetics.
        Imperial College Press.
    Ruthven DM (1984). Principles of Adsorption and Adsorption Processes.
        Wiley-Interscience.
    Cavenati S et al. (2004). J Chem Eng Data 49:1095.
        Zeolite 13X isotherm parameters.
"""

from typing import Literal
import jax.numpy as jnp
from jax import Array

from difflow_cc.database import get_adsorbent, Adsorbent

# Gas constant
R = 8.314  # J/(mol*K)


# =============================================================================
# Basic Isotherm Models (at fixed temperature)
# =============================================================================

def langmuir(P: Array | float, q_sat: Array | float, b: Array | float) -> Array:
    """Langmuir isotherm.

    q = q_sat * b * P / (1 + b * P)

    Assumes single-site adsorption with no lateral interactions.

    Args:
        P: Pressure (Pa)
        q_sat: Saturation capacity (mol/kg)
        b: Affinity constant (1/Pa)

    Returns:
        Loading q (mol/kg)

    References:
        Langmuir I (1918). J Am Chem Soc 40:1361.
    """
    P = jnp.asarray(P)
    q_sat = jnp.asarray(q_sat)
    b = jnp.asarray(b)
    return q_sat * b * P / (1.0 + b * P)


def sips(
    P: Array | float,
    q_sat: Array | float,
    b: Array | float,
    n: Array | float
) -> Array:
    """Sips (Langmuir-Freundlich) isotherm.

    q = q_sat * (b * P)^n / (1 + (b * P)^n)

    Accounts for surface heterogeneity through parameter n.
    n < 1: heterogeneous surface
    n = 1: reduces to Langmuir

    Args:
        P: Pressure (Pa)
        q_sat: Saturation capacity (mol/kg)
        b: Affinity constant (1/Pa)
        n: Heterogeneity parameter (0 < n <= 1)

    Returns:
        Loading q (mol/kg)

    References:
        Sips R (1948). J Chem Phys 16:490.
    """
    P = jnp.asarray(P)
    q_sat = jnp.asarray(q_sat)
    b = jnp.asarray(b)
    n = jnp.asarray(n)

    bP_n = jnp.power(b * P, n)
    return q_sat * bP_n / (1.0 + bP_n)


def toth(
    P: Array | float,
    q_sat: Array | float,
    b: Array | float,
    t: Array | float
) -> Array:
    """Toth isotherm.

    q = q_sat * b * P / (1 + (b * P)^t)^(1/t)

    Asymmetric quasi-Gaussian energy distribution.
    t < 1: more heterogeneous than Langmuir
    t = 1: reduces to Langmuir

    Args:
        P: Pressure (Pa)
        q_sat: Saturation capacity (mol/kg)
        b: Affinity constant (1/Pa)
        t: Heterogeneity parameter (0 < t <= 1)

    Returns:
        Loading q (mol/kg)

    References:
        Toth J (1971). Acta Chim Acad Sci Hung 69:311.
    """
    P = jnp.asarray(P)
    q_sat = jnp.asarray(q_sat)
    b = jnp.asarray(b)
    t = jnp.asarray(t)

    bP_t = jnp.power(b * P, t)
    return q_sat * b * P / jnp.power(1.0 + bP_t, 1.0 / t)


def dual_site_langmuir(
    P: Array | float,
    q1: Array | float,
    b1: Array | float,
    q2: Array | float,
    b2: Array | float
) -> Array:
    """Dual-site Langmuir isotherm.

    q = q1 * b1 * P / (1 + b1 * P) + q2 * b2 * P / (1 + b2 * P)

    Models adsorbents with two distinct types of adsorption sites.
    Common for MOFs with open metal sites.

    Args:
        P: Pressure (Pa)
        q1: Saturation capacity of site 1 (mol/kg)
        b1: Affinity constant of site 1 (1/Pa)
        q2: Saturation capacity of site 2 (mol/kg)
        b2: Affinity constant of site 2 (1/Pa)

    Returns:
        Loading q (mol/kg)

    References:
        Rege SU, Yang RT (1997). Ind Eng Chem Res 36:5358.
    """
    P = jnp.asarray(P)
    q1 = jnp.asarray(q1)
    b1 = jnp.asarray(b1)
    q2 = jnp.asarray(q2)
    b2 = jnp.asarray(b2)

    site1 = q1 * b1 * P / (1.0 + b1 * P)
    site2 = q2 * b2 * P / (1.0 + b2 * P)
    return site1 + site2


# =============================================================================
# Temperature-Dependent Isotherm Models
# =============================================================================

def langmuir_T(
    P: Array | float,
    T: Array | float,
    q_sat: Array | float,
    b0: Array | float,
    Q: Array | float
) -> Array:
    """Temperature-dependent Langmuir isotherm.

    q = q_sat * b(T) * P / (1 + b(T) * P)

    where b(T) = b0 * exp(Q / (R * T))

    Args:
        P: Pressure (Pa)
        T: Temperature (K)
        q_sat: Saturation capacity (mol/kg)
        b0: Pre-exponential affinity (1/Pa)
        Q: Isosteric heat of adsorption (J/mol)

    Returns:
        Loading q (mol/kg)

    Notes:
        The temperature dependence follows from van't Hoff equation.
        Q is typically positive for exothermic adsorption.
    """
    P = jnp.asarray(P)
    T = jnp.asarray(T)
    q_sat = jnp.asarray(q_sat)
    b0 = jnp.asarray(b0)
    Q = jnp.asarray(Q)

    b = b0 * jnp.exp(Q / (R * T))
    return langmuir(P, q_sat, b)


def sips_T(
    P: Array | float,
    T: Array | float,
    q_sat: Array | float,
    b0: Array | float,
    Q: Array | float,
    n: Array | float
) -> Array:
    """Temperature-dependent Sips isotherm.

    Args:
        P: Pressure (Pa)
        T: Temperature (K)
        q_sat: Saturation capacity (mol/kg)
        b0: Pre-exponential affinity (1/Pa)
        Q: Isosteric heat of adsorption (J/mol)
        n: Heterogeneity parameter

    Returns:
        Loading q (mol/kg)
    """
    P = jnp.asarray(P)
    T = jnp.asarray(T)
    q_sat = jnp.asarray(q_sat)
    b0 = jnp.asarray(b0)
    Q = jnp.asarray(Q)
    n = jnp.asarray(n)

    b = b0 * jnp.exp(Q / (R * T))
    return sips(P, q_sat, b, n)


def toth_T(
    P: Array | float,
    T: Array | float,
    q_sat: Array | float,
    b0: Array | float,
    Q: Array | float,
    t0: Array | float,
    alpha: Array | float = 0.0
) -> Array:
    """Temperature-dependent Toth isotherm.

    t(T) = t0 + alpha / T  (optional temperature dependence of t)

    Args:
        P: Pressure (Pa)
        T: Temperature (K)
        q_sat: Saturation capacity (mol/kg)
        b0: Pre-exponential affinity (1/Pa)
        Q: Isosteric heat of adsorption (J/mol)
        t0: Base heterogeneity parameter
        alpha: Temperature coefficient for t (default 0)

    Returns:
        Loading q (mol/kg)

    References:
        Cavenati S et al. (2004). J Chem Eng Data 49:1095.
    """
    P = jnp.asarray(P)
    T = jnp.asarray(T)
    q_sat = jnp.asarray(q_sat)
    b0 = jnp.asarray(b0)
    Q = jnp.asarray(Q)
    t0 = jnp.asarray(t0)
    alpha = jnp.asarray(alpha)

    b = b0 * jnp.exp(Q / (R * T))
    t = t0 + alpha / T
    return toth(P, q_sat, b, t)


def dual_site_langmuir_T(
    P: Array | float,
    T: Array | float,
    q1: Array | float,
    b1_0: Array | float,
    Q1: Array | float,
    q2: Array | float,
    b2_0: Array | float,
    Q2: Array | float
) -> Array:
    """Temperature-dependent dual-site Langmuir isotherm.

    Args:
        P: Pressure (Pa)
        T: Temperature (K)
        q1: Saturation capacity of site 1 (mol/kg)
        b1_0: Pre-exponential affinity of site 1 (1/Pa)
        Q1: Heat of adsorption on site 1 (J/mol)
        q2: Saturation capacity of site 2 (mol/kg)
        b2_0: Pre-exponential affinity of site 2 (1/Pa)
        Q2: Heat of adsorption on site 2 (J/mol)

    Returns:
        Loading q (mol/kg)

    References:
        Caskey SR et al. (2008). J Am Chem Soc 130:10870.
            Used for Mg-MOF-74.
    """
    P = jnp.asarray(P)
    T = jnp.asarray(T)

    b1 = b1_0 * jnp.exp(Q1 / (R * T))
    b2 = b2_0 * jnp.exp(Q2 / (R * T))
    return dual_site_langmuir(P, q1, b1, q2, b2)


# =============================================================================
# Isotherm Class
# =============================================================================

class Isotherm:
    """Unified isotherm interface for any model type.

    Loads parameters from the database and provides a unified
    interface for calculating loading at any T, P.

    Example:
        >>> iso = Isotherm('Zeolite_13X', 'CO2')
        >>> q = iso(P=101325, T=298.15)  # mol/kg
        >>> dq_dP = jax.grad(iso)(P, T)  # Sensitivity to pressure
    """

    def __init__(self, adsorbent: str, species: str = "CO2"):
        """Initialize isotherm from database.

        Args:
            adsorbent: Adsorbent name (e.g., 'Zeolite_13X')
            species: Species name (default 'CO2')
        """
        self.adsorbent_name = adsorbent
        self.species = species

        ads = get_adsorbent(adsorbent)
        if species not in ads.isotherms:
            raise ValueError(
                f"No {species} isotherm data for {adsorbent}. "
                f"Available: {list(ads.isotherms.keys())}"
            )

        iso = ads.isotherms[species]
        self.model = iso.model
        self.params = {k: jnp.array(v) for k, v in iso.params.items()}

    def __call__(self, P: Array | float, T: Array | float) -> Array:
        """Calculate loading at given pressure and temperature.

        Args:
            P: Pressure (Pa)
            T: Temperature (K)

        Returns:
            Loading q (mol/kg)
        """
        p = self.params

        if self.model == "langmuir":
            return langmuir_T(
                P, T,
                q_sat=p["q_sat"],
                b0=p["b0"],
                Q=p["Q"]
            )
        elif self.model == "sips":
            return sips_T(
                P, T,
                q_sat=p["q_sat"],
                b0=p["b0"],
                Q=p["Q"],
                n=p["n"]
            )
        elif self.model == "toth":
            return toth_T(
                P, T,
                q_sat=p["q_sat"],
                b0=p["b0"],
                Q=p["Q"],
                t0=p["t0"],
                alpha=p.get("alpha", jnp.array(0.0))
            )
        elif self.model == "dual_site_langmuir":
            return dual_site_langmuir_T(
                P, T,
                q1=p["q1"],
                b1_0=p["b1_0"],
                Q1=p["Q1"],
                q2=p["q2"],
                b2_0=p["b2_0"],
                Q2=p["Q2"]
            )
        else:
            raise ValueError(f"Unknown isotherm model: {self.model}")

    def isosteric_heat(self, q: Array | float, T: Array | float) -> Array:
        """Calculate isosteric heat of adsorption at given loading.

        Uses Clausius-Clapeyron equation:
            Q_st = -R * (d ln P / d (1/T))_q

        For Langmuir-type isotherms, Q_st is approximately constant.

        Args:
            q: Loading (mol/kg)
            T: Temperature (K)

        Returns:
            Isosteric heat (J/mol)
        """
        p = self.params

        if self.model in ("langmuir", "toth", "sips"):
            # For these models, Q is the isosteric heat
            return p["Q"]
        elif self.model == "dual_site_langmuir":
            # Weighted average based on site occupancy
            # This is an approximation
            q_sat = p["q1"] + p["q2"]
            f1 = p["q1"] / q_sat
            f2 = p["q2"] / q_sat
            return f1 * p["Q1"] + f2 * p["Q2"]
        else:
            return p.get("Q", jnp.array(0.0))


def get_isotherm(adsorbent: str, species: str = "CO2") -> Isotherm:
    """Get isotherm object for adsorbent and species.

    Convenience function to create Isotherm from database.

    Args:
        adsorbent: Adsorbent name
        species: Species name (default 'CO2')

    Returns:
        Isotherm object
    """
    return Isotherm(adsorbent, species)


# =============================================================================
# Working Capacity Functions
# =============================================================================

def working_capacity_PSA(
    isotherm: Isotherm,
    P_ads: Array | float,
    P_des: Array | float,
    T: Array | float
) -> Array:
    """Calculate working capacity for PSA cycle.

    Working capacity = q(P_ads, T) - q(P_des, T)

    Args:
        isotherm: Isotherm object
        P_ads: Adsorption pressure (Pa)
        P_des: Desorption pressure (Pa)
        T: Operating temperature (K)

    Returns:
        Working capacity (mol/kg)

    Example:
        >>> iso = get_isotherm('Zeolite_13X', 'CO2')
        >>> wc = working_capacity_PSA(iso, P_ads=101325, P_des=10000, T=298)
    """
    q_ads = isotherm(P_ads, T)
    q_des = isotherm(P_des, T)
    return q_ads - q_des


def working_capacity_TSA(
    isotherm: Isotherm,
    P: Array | float,
    T_ads: Array | float,
    T_des: Array | float
) -> Array:
    """Calculate working capacity for TSA cycle.

    Working capacity = q(P, T_ads) - q(P, T_des)

    Args:
        isotherm: Isotherm object
        P: Operating pressure (Pa)
        T_ads: Adsorption temperature (K)
        T_des: Desorption temperature (K)

    Returns:
        Working capacity (mol/kg)

    Example:
        >>> iso = get_isotherm('Mg_MOF_74', 'CO2')
        >>> wc = working_capacity_TSA(iso, P=15000, T_ads=298, T_des=353)
    """
    q_ads = isotherm(P, T_ads)
    q_des = isotherm(P, T_des)
    return q_ads - q_des
