"""Membrane aging models for gas separation.

Membrane performance degrades over time due to:
1. Physical aging (free volume relaxation)
2. Plasticization (high CO2 pressure)
3. Fouling (particulates, condensables)
4. Chemical degradation

References:
    Baker RW (2012). Membrane Technology and Applications,
        3rd ed. Wiley. Chapter 8.
    Rowe BW et al. (2010). Physical aging of ultrathin
        glassy polymer films tracked by gas permeability.
        Polymer 51:3784-3792.
    Wessling M et al. (2001). Plasticization of gas separation
        membranes. Gas Sep Purif 5:222-228.
"""

__all__ = [
    "MembraneAgingParams",
    "physical_aging",
    "plasticization",
    "fouling_rate",
    "membrane_lifetime",
]

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide, safe_log

import jax.numpy as jnp
from jax import Array


@dataclass(repr=False)
class MembraneAgingParams(ParamsMixin):
    """Parameters for membrane aging modeling.

    Attributes:
        membrane_type: 'glassy', 'rubbery', 'facilitated', 'ceramic'
        thickness_um: Membrane thickness (μm)
        T_operating: Operating temperature (K)
        P_CO2_feed: CO2 partial pressure in feed (Pa)
        humidity: Feed relative humidity
        particulate_load: Particulate loading (mg/m³)
        initial_permeance: Initial CO2 permeance (GPU)
    """
    membrane_type: str = "glassy"
    thickness_um: float = 0.1  # Thin film composite
    T_operating: float = 298.15
    P_CO2_feed: float = 15000.0  # 0.15 bar
    humidity: float = 0.5
    particulate_load: float = 1.0  # mg/m³
    initial_permeance: float = 1000.0  # GPU


# Material-specific aging parameters
AGING_PARAMS = {
    "glassy": {
        "aging_rate": 0.05,  # % per year of log(perm) decline
        "plasticization_pressure": 1000000,  # Pa (onset)
        "plasticization_factor": 0.1,
        "fouling_k": 0.01,  # m³/mg/yr
    },
    "rubbery": {
        "aging_rate": 0.01,  # Very slow
        "plasticization_pressure": 10000000,  # High
        "plasticization_factor": 0.02,
        "fouling_k": 0.02,
    },
    "facilitated": {
        "aging_rate": 0.10,  # Carrier degradation
        "plasticization_pressure": 500000,
        "plasticization_factor": 0.05,
        "fouling_k": 0.05,
    },
    "ceramic": {
        "aging_rate": 0.005,  # Very stable
        "plasticization_pressure": 100000000,  # N/A
        "plasticization_factor": 0.0,
        "fouling_k": 0.005,
    },
}


def physical_aging(
    time_years: Array | float,
    params: MembraneAgingParams,
) -> Array:
    """Calculate permeance decline from physical aging.

    Glassy polymers undergo free volume relaxation,
    causing permeability to decrease over time.

    For thin films: log(P) = log(P0) - k * t^0.5

    Args:
        time_years: Operating time (years)
        params: Membrane parameters

    Returns:
        Permeance ratio (P/P0)
    """
    time_years = jnp.asarray(time_years)

    aging = AGING_PARAMS.get(params.membrane_type, AGING_PARAMS["glassy"])
    k = aging["aging_rate"]

    # Thickness effect: thinner films age faster
    thickness_factor = 1.0 / (params.thickness_um + 0.01)

    # Temperature effect: higher T = slower aging
    T_factor = jnp.exp(-(params.T_operating - 298.15) / 50)

    # Aging follows sqrt(time) for free volume relaxation
    decay = k * thickness_factor * T_factor * jnp.sqrt(time_years + 0.01)

    permeance_ratio = jnp.exp(-decay)
    permeance_ratio = jnp.clip(permeance_ratio, 0.5, 1.0)

    return permeance_ratio


def plasticization(
    P_CO2: Array | float,
    params: MembraneAgingParams,
) -> dict:
    """Calculate plasticization effects on membrane performance.

    High CO2 pressure causes polymer swelling, which:
    - Increases permeability
    - Decreases selectivity

    Args:
        P_CO2: CO2 partial pressure (Pa)
        params: Membrane parameters

    Returns:
        Dict with permeability and selectivity factors
    """
    P_CO2 = jnp.asarray(P_CO2)

    aging = AGING_PARAMS.get(params.membrane_type, AGING_PARAMS["glassy"])
    P_onset = aging["plasticization_pressure"]
    factor = aging["plasticization_factor"]

    # Plasticization onset
    excess_pressure = jnp.maximum(0, P_CO2 - P_onset)

    # Permeability increase (bad for process)
    perm_increase = 1.0 + factor * jnp.log(1 + excess_pressure / P_onset)

    # Selectivity decrease
    selectivity_decrease = 1.0 / (1.0 + factor * 0.5 * excess_pressure / P_onset)

    # Reversibility: plasticization is partially reversible
    # but repeated cycling causes permanent change
    permanent_factor = 0.1  # 10% of change becomes permanent

    return {
        "permeability_factor": perm_increase,
        "selectivity_factor": selectivity_decrease,
        "is_plasticized": P_CO2 > P_onset,
        "excess_pressure_Pa": excess_pressure,
        "permanent_damage_factor": 1.0 + permanent_factor * (perm_increase - 1),
    }


def fouling_rate(
    time_years: Array | float,
    params: MembraneAgingParams,
) -> Array:
    """Calculate permeance decline from fouling.

    Fouling sources:
    - Particulates
    - Condensable organics
    - Water (for some membranes)

    Args:
        time_years: Operating time (years)
        params: Membrane parameters

    Returns:
        Permeance ratio due to fouling
    """
    time_years = jnp.asarray(time_years)

    aging = AGING_PARAMS.get(params.membrane_type, AGING_PARAMS["glassy"])
    k_foul = aging["fouling_k"]

    # Fouling rate proportional to particulate loading
    fouling = k_foul * params.particulate_load * time_years

    # Humidity effect on fouling
    humidity_factor = 1.0 + params.humidity

    permeance_ratio = jnp.exp(-fouling * humidity_factor)
    permeance_ratio = jnp.clip(permeance_ratio, 0.3, 1.0)

    return permeance_ratio


def total_degradation(
    time_years: Array | float,
    params: MembraneAgingParams,
) -> dict:
    """Calculate total membrane degradation.

    Combines physical aging, plasticization damage, and fouling.

    Args:
        time_years: Operating time (years)
        params: Membrane parameters

    Returns:
        Complete degradation analysis
    """
    time_years = jnp.asarray(time_years)

    # Individual contributions
    f_aging = physical_aging(time_years, params)
    f_fouling = fouling_rate(time_years, params)
    plast = plasticization(params.P_CO2_feed, params)

    # Permanent plasticization damage (cumulative)
    f_plast = 1.0 / plast["permanent_damage_factor"]

    # Combined effect (multiplicative)
    f_total = f_aging * f_fouling * f_plast

    # Current performance
    current_permeance = params.initial_permeance * f_total

    return {
        "permeance_fraction": f_total,
        "current_permeance_GPU": current_permeance,
        "aging_contribution": f_aging,
        "fouling_contribution": f_fouling,
        "plasticization_contribution": f_plast,
        "permeance_loss_percent": (1 - f_total) * 100,
    }


def membrane_lifetime(
    params: MembraneAgingParams,
    min_permeance_fraction: float = 0.7,
) -> Array:
    """Estimate membrane lifetime.

    Lifetime is when permeance drops below threshold.

    Args:
        params: Membrane parameters
        min_permeance_fraction: Minimum acceptable permeance

    Returns:
        Estimated lifetime (years)
    """
    # Get approximate decay rate at 1 year
    deg_1yr = total_degradation(1.0, params)
    f_1yr = deg_1yr["permeance_fraction"]

    # Effective decay constant (assuming exponential)
    k_eff = -safe_log(f_1yr)

    # Time to reach minimum
    lifetime = safe_divide(-safe_log(min_permeance_fraction), k_eff)
    lifetime = jnp.clip(lifetime, 1, 20)  # 1-20 years

    return lifetime


def cleaning_effectiveness(
    fouling_fraction: Array | float,
    cleaning_type: str = "chemical",
) -> Array:
    """Estimate cleaning effectiveness for fouled membranes.

    Args:
        fouling_fraction: Current fouling contribution to degradation
        cleaning_type: 'backflush', 'chemical', 'thermal'

    Returns:
        Restored fraction of fouling
    """
    fouling_fraction = jnp.asarray(fouling_fraction)

    effectiveness = {
        "backflush": 0.3,  # Removes loose deposits
        "chemical": 0.7,  # Removes most organic fouling
        "thermal": 0.5,  # Risk of damage
    }

    eff = effectiveness.get(cleaning_type, 0.5)

    # Restoration (fouling fraction closer to 1.0)
    restored = fouling_fraction + eff * (1.0 - fouling_fraction)

    return restored


def replacement_schedule(
    params: MembraneAgingParams,
    target_availability: float = 0.95,
    module_cost: float = 100.0,  # $/m²
) -> dict:
    """Optimize membrane replacement schedule.

    Trade-off between replacement cost and performance loss.

    Args:
        params: Membrane parameters
        target_availability: Target average performance
        module_cost: Membrane module cost ($/m²)

    Returns:
        Optimal replacement schedule
    """
    # Lifetime to 70% performance
    lifetime = membrane_lifetime(params, 0.70)

    # Average performance over lifetime
    # Integral of degradation / time
    avg_performance = 0.85  # (1.0 + 0.70) / 2, assuming linear degradation to 70% threshold

    # If target > avg, need more frequent replacement
    if target_availability > avg_performance:
        replacement_interval = lifetime * avg_performance / target_availability
    else:
        replacement_interval = lifetime

    # Annual cost
    annual_cost = module_cost / replacement_interval

    return {
        "replacement_interval_years": replacement_interval,
        "membrane_lifetime_years": lifetime,
        "average_performance": avg_performance,
        "annual_replacement_cost_per_m2": annual_cost,
        "n_replacements_over_25yr": 25 / replacement_interval,
    }
