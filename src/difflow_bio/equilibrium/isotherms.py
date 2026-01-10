"""Binding isotherm models for chromatography.

This module provides JAX-compatible implementations of common
binding isotherm models used in protein chromatography for
biopharmaceutical purification.

All functions use JAX numpy for automatic differentiation compatibility.

Models implemented:
- Langmuir: Single-site binding (affinity, ion exchange)
- Langmuir competitive: Multi-component competitive binding
- Steric mass action (SMA): Ion exchange with steric effects
- Linear: Simple partitioning (SEC, analytical separations)

References:
    Langmuir I (1916). J Am Chem Soc 38:2221.
    Brooks CA, Cramer SM (1992). AIChE J 38:1969.
    Carta G, Jungbauer A (2010). Protein Chromatography. Wiley-VCH.
    Guiochon G et al. (2006). Fundamentals of Preparative and
        Nonlinear Chromatography. Elsevier.
"""

__all__ = [
    "langmuir_binding",
    "langmuir_competitive",
    "steric_mass_action",
    "linear_partition",
    "langmuir_ph_dependent",
    "dynamic_binding_capacity",
    "breakthrough_curve",
    "column_efficiency",
    "van_deemter",
    "BindingIsotherm",
    "get_binding_isotherm",
]

import jax.numpy as jnp
from jax import Array

from difflow_bio.database import get_resin, Resin


# =============================================================================
# Basic Isotherm Models
# =============================================================================

def langmuir_binding(
    C: Array | float,
    q_max: Array | float,
    K_d: Array | float,
) -> Array:
    """Langmuir binding isotherm.

    q = q_max * C / (K_d + C)

    Models single-site binding with no lateral interactions.
    Commonly used for Protein A affinity chromatography.

    Args:
        C: Solute concentration in mobile phase (g/L)
        q_max: Maximum binding capacity (g/L resin)
        K_d: Dissociation constant (g/L)

    Returns:
        Bound concentration q (g/L resin)

    Example:
        >>> q = langmuir_binding(C=1.0, q_max=35.0, K_d=0.1)
        >>> # At 1 g/L feed, binding is near saturation for high affinity
    """
    C = jnp.asarray(C)
    q_max = jnp.asarray(q_max)
    K_d = jnp.asarray(K_d)
    return q_max * C / (K_d + C)


def langmuir_competitive(
    C: dict[str, Array | float],
    q_max: dict[str, Array | float],
    K_d: dict[str, Array | float],
) -> dict[str, Array]:
    """Competitive Langmuir isotherm for multi-component systems.

    q_i = q_max_i * C_i / K_d_i / (1 + sum_j(C_j / K_d_j))

    Models competitive binding where components compete for the same sites.
    Important for impurity co-elution in chromatography.

    Args:
        C: Dict of species -> concentration (g/L)
        q_max: Dict of species -> max capacity (g/L)
        K_d: Dict of species -> dissociation constant (g/L)

    Returns:
        Dict of species -> bound concentration (g/L)

    Example:
        >>> q = langmuir_competitive(
        ...     C={"mAb": 1.0, "HCP": 0.01},
        ...     q_max={"mAb": 35.0, "HCP": 5.0},
        ...     K_d={"mAb": 0.1, "HCP": 1.0},
        ... )
    """
    species = list(C.keys())

    # Calculate denominator: 1 + sum(C_j / K_d_j)
    denom = jnp.array(1.0)
    for s in species:
        c = jnp.asarray(C[s])
        kd = jnp.asarray(K_d[s])
        denom = denom + c / kd

    # Calculate bound concentration for each species
    q = {}
    for s in species:
        c = jnp.asarray(C[s])
        qm = jnp.asarray(q_max[s])
        kd = jnp.asarray(K_d[s])
        q[s] = qm * (c / kd) / denom

    return q


def steric_mass_action(
    C: Array | float,
    C_salt: Array | float,
    q_max: Array | float,
    K_eq: Array | float,
    nu: Array | float,
    sigma: Array | float,
) -> Array:
    """Steric mass action (SMA) model for ion exchange.

    q = q_max * K_eq * (C / C_salt^nu) / (1 + K_eq * (C / C_salt^nu))

    With steric factor correction for surface area shielding.

    The SMA model accounts for:
    - Ion exchange stoichiometry (nu = characteristic charge)
    - Steric shielding of binding sites (sigma = steric factor)
    - Salt concentration effects

    Args:
        C: Protein concentration (g/L or mol/L)
        C_salt: Salt concentration (mol/L)
        q_max: Maximum binding capacity (g/L or mol/L)
        K_eq: Equilibrium constant
        nu: Characteristic charge (stoichiometric coefficient)
        sigma: Steric factor (shielded sites per bound protein)

    Returns:
        Bound concentration q

    References:
        Brooks CA, Cramer SM (1992). AIChE J 38:1969.

    Notes:
        For mAbs (MW ~150 kDa), typical values are:
        - nu: 4-8 (characteristic charge)
        - sigma: 20-60 (steric factor)
    """
    C = jnp.asarray(C)
    C_salt = jnp.asarray(C_salt)
    q_max = jnp.asarray(q_max)
    K_eq = jnp.asarray(K_eq)
    nu = jnp.asarray(nu)
    sigma = jnp.asarray(sigma)

    # Effective binding term
    binding_term = K_eq * C / jnp.power(C_salt, nu)

    # Langmuir-like form with SMA
    q = q_max * binding_term / (1.0 + (1.0 + sigma) * binding_term)

    return q


def linear_partition(
    C: Array | float,
    K: Array | float,
) -> Array:
    """Linear partition isotherm.

    q = K * C

    Simple linear partitioning, used for:
    - Size exclusion chromatography (SEC)
    - Dilute conditions (Henry's law region)
    - Analytical chromatography

    Args:
        C: Solute concentration (g/L)
        K: Partition coefficient (L solution / L resin)

    Returns:
        Bound concentration q (g/L resin)
    """
    C = jnp.asarray(C)
    K = jnp.asarray(K)
    return K * C


def langmuir_ph_dependent(
    C: Array | float,
    pH: Array | float,
    q_max: Array | float,
    K_d_ref: Array | float,
    pH_ref: Array | float = 7.0,
    dpKa: Array | float = 1.0,
) -> Array:
    """pH-dependent Langmuir isotherm.

    K_d(pH) = K_d_ref * 10^((pH - pH_ref) / dpKa)

    Models pH-dependent binding strength, important for:
    - Protein A elution (low pH)
    - Ion exchange capacity vs pH

    Args:
        C: Solute concentration (g/L)
        pH: Solution pH
        q_max: Maximum capacity (g/L)
        K_d_ref: K_d at reference pH (g/L)
        pH_ref: Reference pH (default 7.0)
        dpKa: pH sensitivity parameter

    Returns:
        Bound concentration q (g/L)

    Notes:
        For Protein A at low pH (3-4), K_d increases dramatically,
        causing elution of bound mAb.
    """
    C = jnp.asarray(C)
    pH = jnp.asarray(pH)
    q_max = jnp.asarray(q_max)
    K_d_ref = jnp.asarray(K_d_ref)
    pH_ref = jnp.asarray(pH_ref)
    dpKa = jnp.asarray(dpKa)

    # pH-dependent dissociation constant
    K_d = K_d_ref * jnp.power(10.0, (pH - pH_ref) / dpKa)

    return langmuir_binding(C, q_max, K_d)


# =============================================================================
# Dynamic Binding and Column Models
# =============================================================================

def dynamic_binding_capacity(
    q_max: Array | float,
    C_feed: Array | float,
    K_d: Array | float,
    residence_time: Array | float,
    k_ads: Array | float,
    breakthrough: Array | float = 0.1,
) -> Array:
    """Estimate dynamic binding capacity (DBC).

    DBC accounts for mass transfer limitations that reduce
    the usable capacity compared to equilibrium (static) capacity.

    Args:
        q_max: Static binding capacity (g/L)
        C_feed: Feed concentration (g/L)
        K_d: Dissociation constant (g/L)
        residence_time: Column residence time (min)
        k_ads: Adsorption rate constant (1/min)
        breakthrough: Acceptable breakthrough fraction (default 0.1 = 10%)

    Returns:
        Dynamic binding capacity (g/L)

    Notes:
        DBC is typically 60-90% of static capacity depending on
        flow rate and mass transfer.
    """
    q_max = jnp.asarray(q_max)
    C_feed = jnp.asarray(C_feed)
    K_d = jnp.asarray(K_d)
    residence_time = jnp.asarray(residence_time)
    k_ads = jnp.asarray(k_ads)
    breakthrough = jnp.asarray(breakthrough)

    # Equilibrium capacity at feed concentration
    q_eq = langmuir_binding(C_feed, q_max, K_d)

    # Mass transfer efficiency factor
    efficiency = 1.0 - jnp.exp(-k_ads * residence_time)

    # Breakthrough correction (less capacity used to limit breakthrough)
    bt_factor = 1.0 - breakthrough

    return q_eq * efficiency * bt_factor


def breakthrough_curve(
    t: Array | float,
    t_b: Array | float,
    sigma: Array | float,
) -> Array:
    """Sigmoid breakthrough curve model.

    C/C0 = 1 / (1 + exp(-(t - t_b) / sigma))

    Simplified model for column breakthrough behavior.

    Args:
        t: Time or volume (same units as t_b)
        t_b: Breakthrough time/volume (at C/C0 = 0.5)
        sigma: Curve steepness (smaller = sharper)

    Returns:
        Normalized effluent concentration C/C0

    Notes:
        Real breakthrough curves are often more complex,
        but this captures the essential S-shape.
    """
    t = jnp.asarray(t)
    t_b = jnp.asarray(t_b)
    sigma = jnp.asarray(sigma)

    return 1.0 / (1.0 + jnp.exp(-(t - t_b) / sigma))


def column_efficiency(
    t_R: Array | float,
    w: Array | float,
) -> Array:
    """Calculate column efficiency (theoretical plates).

    N = 16 * (t_R / w)^2

    Where t_R is retention time and w is peak width at base.

    Args:
        t_R: Retention time
        w: Peak width at base (4 sigma)

    Returns:
        Number of theoretical plates N

    Notes:
        For analytical columns, N > 10,000 is typical.
        For preparative columns, N > 1,000 is acceptable.
    """
    t_R = jnp.asarray(t_R)
    w = jnp.asarray(w)
    return 16.0 * (t_R / w) ** 2


def van_deemter(
    u: Array | float,
    A: Array | float,
    B: Array | float,
    C: Array | float,
) -> Array:
    """Van Deemter equation for HETP.

    HETP = A + B/u + C*u

    Describes the dependence of plate height on linear velocity.

    Args:
        u: Linear velocity (cm/s or cm/min)
        A: Eddy diffusion term (cm)
        B: Molecular diffusion term (cm^2/s)
        C: Mass transfer resistance term (s)

    Returns:
        Height equivalent to theoretical plate (HETP) in cm

    Notes:
        Optimal velocity minimizes HETP:
        u_opt = sqrt(B/C)
    """
    u = jnp.asarray(u)
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    C = jnp.asarray(C)

    return A + B / u + C * u


# =============================================================================
# Binding Isotherm Class
# =============================================================================

class BindingIsotherm:
    """Unified binding isotherm interface.

    Loads resin parameters from database and provides a unified
    interface for calculating binding at any concentration.

    Example:
        >>> iso = BindingIsotherm('MabSelect_SuRe')
        >>> q = iso(C=1.0)  # g/L bound
        >>> dq_dC = jax.grad(iso)(C)  # Sensitivity
    """

    def __init__(
        self,
        resin: str,
        model: str = "langmuir",
        **kwargs,
    ):
        """Initialize binding isotherm from database or parameters.

        Args:
            resin: Resin name (e.g., 'MabSelect_SuRe')
            model: Model type ('langmuir', 'sma', 'linear')
            **kwargs: Override parameters (q_max, K_d, etc.)
        """
        self.resin_name = resin
        self.model = model

        resin_obj = get_resin(resin)
        self.params = {
            "q_max": jnp.array(kwargs.get("q_max", resin_obj.q_max)),
            "K_d": jnp.array(kwargs.get("K_d", resin_obj.K_d)),
        }

        # Add SMA params if specified
        if model == "sma":
            self.params["nu"] = jnp.array(kwargs.get("nu", 6.0))
            self.params["sigma"] = jnp.array(kwargs.get("sigma", 40.0))
            self.params["K_eq"] = jnp.array(kwargs.get("K_eq", 1.0))

    def __call__(
        self,
        C: Array | float,
        C_salt: Array | float = None,
    ) -> Array:
        """Calculate bound concentration.

        Args:
            C: Mobile phase concentration (g/L)
            C_salt: Salt concentration for SMA model (mol/L)

        Returns:
            Bound concentration q (g/L)
        """
        p = self.params

        if self.model == "langmuir":
            return langmuir_binding(C, p["q_max"], p["K_d"])
        elif self.model == "sma":
            if C_salt is None:
                raise ValueError("SMA model requires C_salt")
            return steric_mass_action(
                C, C_salt, p["q_max"], p["K_eq"], p["nu"], p["sigma"]
            )
        elif self.model == "linear":
            # K_d is used as partition coefficient
            return linear_partition(C, p["K_d"])
        else:
            raise ValueError(f"Unknown model: {self.model}")


def get_binding_isotherm(
    resin: str,
    model: str = "langmuir",
    **kwargs,
) -> BindingIsotherm:
    """Get binding isotherm for a resin.

    Convenience function to create BindingIsotherm from database.

    Args:
        resin: Resin name
        model: Model type (default 'langmuir')
        **kwargs: Override parameters

    Returns:
        BindingIsotherm object

    Example:
        >>> iso = get_binding_isotherm('MabSelect_SuRe')
        >>> q = iso(C=2.0)
    """
    return BindingIsotherm(resin, model, **kwargs)
