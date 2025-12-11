"""Economics module for REE separation processes.

Provides cost estimation and economic analysis specific to
rare earth element separation:
- REE product pricing
- Reagent costs (extractants, acids, bases)
- Equipment sizing and capital costs
- Operating costs
- Profitability analysis
"""

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
]
