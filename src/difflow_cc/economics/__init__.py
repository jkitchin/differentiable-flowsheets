"""Economics module for carbon capture cost analysis.

This module provides:
- Capital cost correlations (CAPEX)
- Operating cost calculations (OPEX)
- Levelized cost of CO2 capture
- Sensitivity analysis utilities

All functions are JAX-compatible for gradient-based optimization.
"""

from difflow_cc.economics.capex import (
    absorber_cost,
    stripper_cost,
    heat_exchanger_cost,
    compressor_cost,
    membrane_module_cost,
    adsorber_vessel_cost,
    total_equipment_cost,
    installed_cost,
    CapexParams,
    SCALING_EXPONENTS,
    VALID_EXPONENT_RANGE,
    validate_scaling_exponent,
)

from difflow_cc.economics.opex import (
    steam_cost,
    electricity_cost,
    cooling_water_cost,
    solvent_makeup_cost,
    membrane_replacement_cost,
    adsorbent_replacement_cost,
    labor_cost,
    maintenance_cost,
    total_operating_cost,
    OpexParams,
)

from difflow_cc.economics.levelized_cost import (
    levelized_cost_capture,
    cost_of_co2_avoided,
    net_present_value,
    internal_rate_return,
    payback_period,
    EconomicParams,
    CaptureSystemCost,
)

__all__ = [
    # CAPEX
    "absorber_cost",
    "stripper_cost",
    "heat_exchanger_cost",
    "compressor_cost",
    "membrane_module_cost",
    "adsorber_vessel_cost",
    "total_equipment_cost",
    "installed_cost",
    "CapexParams",
    "SCALING_EXPONENTS",
    "VALID_EXPONENT_RANGE",
    "validate_scaling_exponent",
    # OPEX
    "steam_cost",
    "electricity_cost",
    "cooling_water_cost",
    "solvent_makeup_cost",
    "membrane_replacement_cost",
    "adsorbent_replacement_cost",
    "labor_cost",
    "maintenance_cost",
    "total_operating_cost",
    "OpexParams",
    # Levelized cost
    "levelized_cost_capture",
    "cost_of_co2_avoided",
    "net_present_value",
    "internal_rate_return",
    "payback_period",
    "EconomicParams",
    "CaptureSystemCost",
]
