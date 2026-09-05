"""Economics module for REE separation processes.

Provides cost estimation and economic analysis specific to
rare earth element separation:
- REE product pricing
- Reagent costs (extractants, acids, bases)
- Equipment sizing and capital costs
- Operating costs
- Profitability analysis
- Saponification reagent duty: kg base per kg REO, effluent loads (#197)
"""

from difflow_ree.economics.saponification import (
    # Saponification reagent and effluent metrics (#197)
    BaseReagent,
    BASE_REAGENTS,
    BASE_PRICES_USD_KG,
    DEFAULT_BASE_FOR_COUNTER_ION,
    SaponificationDuty,
    base_for_counter_ion,
    get_base,
    ree_oxide_mass_flow,
    base_per_ree_oxide,
    stoichiometric_base_per_ree_oxide,
    nitrogen_per_ree_oxide,
    dissolved_salt_per_ree_oxide,
    saponification_duty,
    compare_counter_ions,
)
from difflow_ree.economics.costs import (
    REEPricing,
    ReagentCosts,
    OperatingCosts,
    estimate_capex,
    estimate_opex,
    calculate_revenue,
    calculate_profit,
    minimum_selling_price,
)

__all__ = [
    "REEPricing",
    "ReagentCosts",
    "OperatingCosts",
    "estimate_capex",
    "estimate_opex",
    "calculate_revenue",
    "calculate_profit",
    "minimum_selling_price",
    # Saponification reagent and effluent metrics (#197)
    "BaseReagent",
    "BASE_REAGENTS",
    "BASE_PRICES_USD_KG",
    "DEFAULT_BASE_FOR_COUNTER_ION",
    "SaponificationDuty",
    "base_for_counter_ion",
    "get_base",
    "ree_oxide_mass_flow",
    "base_per_ree_oxide",
    "stoichiometric_base_per_ree_oxide",
    "nitrogen_per_ree_oxide",
    "dissolved_salt_per_ree_oxide",
    "saponification_duty",
    "compare_counter_ions",
]
