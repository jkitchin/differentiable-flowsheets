"""Economics module for biopharmaceutical manufacturing.

Provides cost estimation for:
- Capital expenditures (CAPEX): equipment, facilities
- Operating expenditures (OPEX): consumables, labor, utilities
- Profitability analysis: NPV, IRR, cost per gram

References:
    Farid SS et al. (2007). Biotechnol Prog 23:3.
        Economic modeling of bioprocesses.
    Pollock J et al. (2013). Biotechnol Bioeng 110:206.
        DSP cost benchmarking.
"""

from difflow_bio.economics.costs import (
    # Cost dataclasses
    ConsumableCosts,
    EquipmentCosts,
    OperatingCosts,
    # CAPEX functions
    estimate_bioreactor_capex,
    estimate_chromatography_capex,
    estimate_filtration_capex,
    estimate_facility_capex,
    estimate_total_capex,
    # OPEX functions
    estimate_resin_cost,
    estimate_membrane_cost,
    estimate_media_cost,
    estimate_labor_cost,
    estimate_utilities_cost,
    estimate_total_opex,
    # Analysis functions
    calculate_cogs,
    calculate_profit,
    cost_per_gram,
)

__all__ = [
    # Dataclasses
    "ConsumableCosts",
    "EquipmentCosts",
    "OperatingCosts",
    # CAPEX
    "estimate_bioreactor_capex",
    "estimate_chromatography_capex",
    "estimate_filtration_capex",
    "estimate_facility_capex",
    "estimate_total_capex",
    # OPEX
    "estimate_resin_cost",
    "estimate_membrane_cost",
    "estimate_media_cost",
    "estimate_labor_cost",
    "estimate_utilities_cost",
    "estimate_total_opex",
    # Analysis
    "calculate_cogs",
    "calculate_profit",
    "cost_per_gram",
]
