"""Degradation models for carbon capture materials.

This module provides models for:
- Amine solvent degradation (oxidative, thermal, CO2-induced)
- Adsorbent degradation (thermal, hydrothermal, poisoning)
- Membrane aging

All models are JAX-compatible.
"""

from difflow_cc.degradation.amine_degradation import (
    AmineDegradationParams,
    oxidative_degradation_rate,
    thermal_degradation_rate,
    co2_induced_degradation_rate,
    total_amine_loss,
    degradation_products,
    solvent_lifetime,
    reclaimer_requirements,
)

from difflow_cc.degradation.adsorbent_degradation import (
    AdsorbentDegradationParams,
    thermal_cycling_degradation,
    hydrothermal_degradation,
    capacity_fade,
    adsorbent_lifetime,
    regeneration_optimization,
)

from difflow_cc.degradation.membrane_aging import (
    MembraneAgingParams,
    physical_aging,
    plasticization,
    fouling_rate,
    membrane_lifetime,
)

__all__ = [
    # Amine
    "AmineDegradationParams",
    "oxidative_degradation_rate",
    "thermal_degradation_rate",
    "co2_induced_degradation_rate",
    "total_amine_loss",
    "degradation_products",
    "solvent_lifetime",
    "reclaimer_requirements",
    # Adsorbent
    "AdsorbentDegradationParams",
    "thermal_cycling_degradation",
    "hydrothermal_degradation",
    "capacity_fade",
    "adsorbent_lifetime",
    "regeneration_optimization",
    # Membrane
    "MembraneAgingParams",
    "physical_aging",
    "plasticization",
    "fouling_rate",
    "membrane_lifetime",
]
