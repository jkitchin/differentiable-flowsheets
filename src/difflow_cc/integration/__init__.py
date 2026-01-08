"""Process integration for carbon capture with power plants.

This module provides models for:
- Steam extraction from power plant
- Efficiency penalty calculations
- Heat integration opportunities
- Flue gas specifications

All functions are JAX-compatible.
"""

from difflow_cc.integration.power_plant import (
    PowerPlantParams,
    PowerPlantIntegration,
    steam_extraction_penalty,
    compression_penalty,
    auxiliary_power,
    net_efficiency_with_capture,
    flue_gas_composition,
    flue_gas_flow_rate,
)

from difflow_cc.integration.steam_cycle import (
    SteamCycleParams,
    steam_properties,
    extraction_pressure_options,
    steam_flow_for_duty,
    crossover_extraction,
    letdown_valve,
)

__all__ = [
    # Power plant
    "PowerPlantParams",
    "PowerPlantIntegration",
    "steam_extraction_penalty",
    "compression_penalty",
    "auxiliary_power",
    "net_efficiency_with_capture",
    "flue_gas_composition",
    "flue_gas_flow_rate",
    # Steam cycle
    "SteamCycleParams",
    "steam_properties",
    "extraction_pressure_options",
    "steam_flow_for_duty",
    "crossover_extraction",
    "letdown_valve",
]
