"""Adsorbent degradation models for cyclic adsorption processes.

Adsorbent degradation affects:
- CO2 working capacity
- Cycle time and productivity
- Replacement costs

Degradation mechanisms:
1. Thermal cycling stress
2. Hydrothermal stability (steam, humidity)
3. Chemical poisoning (SOx, NOx)
4. Mechanical attrition

References:
    Choi S et al. (2009). Adsorbent materials for carbon dioxide
        capture from large anthropogenic point sources.
        ChemSusChem 2:796-854.
    Sayari A et al. (2011). Flue gas treatment via CO2 adsorption.
        Chem Eng J 171:760-774.
    Hedin N et al. (2010). Adsorbents for the post-combustion
        capture of CO2 using rapid temperature swing or vacuum
        swing adsorption. Appl Energy 87:1977-1982.
"""

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin

import jax.numpy as jnp
from jax import Array


@dataclass(repr=False)
class AdsorbentDegradationParams(ParamsMixin):
    """Parameters for adsorbent degradation modeling.

    Attributes:
        material_type: 'zeolite', 'MOF', 'carbon', 'amine_silica'
        T_adsorption: Adsorption temperature (K)
        T_desorption: Desorption temperature (K)
        humidity: Relative humidity in feed
        SO2_ppm: SO2 concentration (ppm)
        NOx_ppm: NOx concentration (ppm)
        cycles_per_day: Number of cycles per day
        initial_capacity: Initial CO2 capacity (mol/kg)
    """
    material_type: str = "zeolite"
    T_adsorption: float = 298.15
    T_desorption: float = 423.15
    humidity: float = 0.5
    SO2_ppm: float = 0.0  # After FGD
    NOx_ppm: float = 50.0
    cycles_per_day: float = 48.0  # 30-min cycles
    initial_capacity: float = 4.0  # mol/kg


# Stability parameters by material type
STABILITY_PARAMS = {
    "zeolite": {
        "thermal_stability": 0.9999,  # Very stable per cycle
        "hydrothermal_k": 1e-6,  # Low sensitivity
        "SO2_sensitivity": 0.01,  # ppm^-1 per 1000 cycles
        "max_T": 673,  # K
    },
    "MOF": {
        "thermal_stability": 0.9995,
        "hydrothermal_k": 1e-4,  # More sensitive
        "SO2_sensitivity": 0.1,
        "max_T": 523,
    },
    "carbon": {
        "thermal_stability": 0.9998,
        "hydrothermal_k": 5e-6,
        "SO2_sensitivity": 0.001,  # Resistant
        "max_T": 573,
    },
    "amine_silica": {
        "thermal_stability": 0.999,
        "hydrothermal_k": 5e-5,
        "SO2_sensitivity": 0.5,  # Very sensitive
        "max_T": 393,  # Limited by amine stability
    },
}


def thermal_cycling_degradation(
    n_cycles: Array | float,
    params: AdsorbentDegradationParams,
) -> Array:
    """Calculate capacity loss from thermal cycling.

    Repeated heating/cooling causes:
    - Crystal structure stress
    - Loss of functional groups (amines)
    - Pore collapse in some materials

    Model: q/q0 = f^n where f is stability per cycle

    Args:
        n_cycles: Number of thermal cycles
        params: Degradation parameters

    Returns:
        Fraction of initial capacity remaining
    """
    n_cycles = jnp.asarray(n_cycles)

    stability = STABILITY_PARAMS.get(
        params.material_type, STABILITY_PARAMS["zeolite"]
    )

    # Base stability factor
    f_base = stability["thermal_stability"]

    # Temperature swing effect (larger swing = more stress)
    T_swing = params.T_desorption - params.T_adsorption
    T_swing_norm = T_swing / 100  # Normalize to 100 K swing

    # Adjusted stability
    f = f_base ** T_swing_norm

    # Check temperature limits
    max_T = stability["max_T"]
    if params.T_desorption > max_T:
        # Severe degradation above max T
        overheat_factor = (params.T_desorption - max_T) / 50
        f = f * jnp.exp(-overheat_factor)

    # Capacity fraction remaining
    capacity_fraction = jnp.power(f, n_cycles)

    return capacity_fraction


def hydrothermal_degradation(
    time_hours: Array | float,
    params: AdsorbentDegradationParams,
) -> Array:
    """Calculate capacity loss from humidity/steam exposure.

    Water can:
    - Compete for adsorption sites
    - Cause framework degradation (MOFs)
    - Leach amine from supported materials

    Model: q/q0 = exp(-k * RH * t)

    Args:
        time_hours: Operating time (hours)
        params: Degradation parameters

    Returns:
        Fraction of initial capacity remaining
    """
    time_hours = jnp.asarray(time_hours)

    stability = STABILITY_PARAMS.get(
        params.material_type, STABILITY_PARAMS["zeolite"]
    )

    k = stability["hydrothermal_k"]
    RH = params.humidity

    # Temperature accelerates hydrothermal degradation
    T_factor = jnp.exp((params.T_desorption - 373.15) / 50)

    capacity_fraction = jnp.exp(-k * RH * T_factor * time_hours)

    return capacity_fraction


def poisoning_degradation(
    time_hours: Array | float,
    params: AdsorbentDegradationParams,
) -> Array:
    """Calculate capacity loss from chemical poisoning.

    SO2 and NOx can irreversibly bind to adsorption sites,
    especially for amine-functionalized materials.

    Args:
        time_hours: Operating time (hours)
        params: Degradation parameters

    Returns:
        Fraction of initial capacity remaining
    """
    time_hours = jnp.asarray(time_hours)

    stability = STABILITY_PARAMS.get(
        params.material_type, STABILITY_PARAMS["zeolite"]
    )

    # SO2 poisoning (dominant for amines)
    k_SO2 = stability["SO2_sensitivity"]
    SO2_effect = k_SO2 * params.SO2_ppm * time_hours / 1000

    # NOx effect (smaller)
    k_NOx = k_SO2 * 0.1
    NOx_effect = k_NOx * params.NOx_ppm * time_hours / 1000

    capacity_fraction = jnp.exp(-(SO2_effect + NOx_effect))
    capacity_fraction = jnp.clip(capacity_fraction, 0.1, 1.0)

    return capacity_fraction


def capacity_fade(
    operating_hours: Array | float,
    params: AdsorbentDegradationParams,
) -> dict:
    """Calculate total capacity fade from all mechanisms.

    Args:
        operating_hours: Total operating hours
        params: Degradation parameters

    Returns:
        Dict with capacity fade breakdown
    """
    operating_hours = jnp.asarray(operating_hours)

    # Number of cycles
    n_cycles = operating_hours / 24 * params.cycles_per_day

    # Individual contributions
    f_thermal = thermal_cycling_degradation(n_cycles, params)
    f_hydro = hydrothermal_degradation(operating_hours, params)
    f_poison = poisoning_degradation(operating_hours, params)

    # Combined (multiplicative)
    f_total = f_thermal * f_hydro * f_poison

    # Current capacity
    q_current = params.initial_capacity * f_total

    return {
        "capacity_fraction": f_total,
        "current_capacity_mol_kg": q_current,
        "thermal_contribution": f_thermal,
        "hydrothermal_contribution": f_hydro,
        "poisoning_contribution": f_poison,
        "n_cycles": n_cycles,
        "capacity_loss_percent": (1 - f_total) * 100,
    }


def adsorbent_lifetime(
    params: AdsorbentDegradationParams,
    min_capacity_fraction: float = 0.7,  # 70% of initial
) -> Array:
    """Estimate adsorbent lifetime.

    Lifetime is when capacity drops below threshold.

    Args:
        params: Degradation parameters
        min_capacity_fraction: Minimum acceptable capacity

    Returns:
        Estimated lifetime (hours)
    """
    # Binary search for lifetime
    # Simplified: assume exponential decay

    # Get approximate decay rate
    test_hours = 1000.0
    fade = capacity_fade(test_hours, params)
    f_test = float(fade["capacity_fraction"])

    # Effective decay constant
    k_eff = -jnp.log(f_test + 1e-10) / test_hours

    # Time to reach minimum
    lifetime = -jnp.log(min_capacity_fraction) / (k_eff + 1e-10)
    lifetime = jnp.clip(lifetime, 1000, 50000)  # 1000 hr to 50000 hr

    return lifetime


def regeneration_optimization(
    params: AdsorbentDegradationParams,
) -> dict:
    """Optimize regeneration conditions to minimize degradation.

    Trade-off: Higher T = better regeneration but more degradation.

    Args:
        params: Degradation parameters

    Returns:
        Optimal regeneration conditions
    """
    stability = STABILITY_PARAMS.get(
        params.material_type, STABILITY_PARAMS["zeolite"]
    )

    max_T = stability["max_T"]

    # Optimal T is below max but high enough for good working capacity
    T_opt = min(params.T_desorption, max_T - 20)

    # If using TVSA, lower T can be compensated by vacuum
    # Recommend P_des for equivalent driving force
    if params.T_desorption > max_T:
        P_recommended = 10000.0  # 0.1 bar vacuum
        T_recommended = max_T - 20
    else:
        P_recommended = 101325.0  # Atmospheric
        T_recommended = params.T_desorption

    # Expected lifetime at optimal vs current conditions
    params_opt = AdsorbentDegradationParams(
        material_type=params.material_type,
        T_adsorption=params.T_adsorption,
        T_desorption=T_recommended,
        humidity=params.humidity,
        SO2_ppm=params.SO2_ppm,
        NOx_ppm=params.NOx_ppm,
        cycles_per_day=params.cycles_per_day,
        initial_capacity=params.initial_capacity,
    )

    lifetime_current = adsorbent_lifetime(params)
    lifetime_optimal = adsorbent_lifetime(params_opt)

    return {
        "T_desorption_optimal_K": T_recommended,
        "P_desorption_optimal_Pa": P_recommended,
        "max_T_K": max_T,
        "lifetime_current_hr": lifetime_current,
        "lifetime_optimal_hr": lifetime_optimal,
        "lifetime_improvement_factor": lifetime_optimal / (lifetime_current + 1),
    }
