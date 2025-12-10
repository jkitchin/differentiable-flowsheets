"""Differentiable Technoeconomic Analysis (TEA) for process flowsheets.

This module provides comprehensive economic modeling capabilities for chemical
process design optimization. All functions are JAX-compatible for automatic
differentiation, enabling gradient-based optimization of process designs.

Submodules
----------
capital : Equipment cost correlations and installation factors
opex : Operating cost models (raw materials, labor, overhead)
utilities : Utility cost calculations (steam, cooling, electricity)
profitability : NPV, IRR, MSP, payback analysis
indices : Cost indices (CEPCI) for cost escalation

Example
-------
>>> import difflow.economics as econ
>>>
>>> # Equipment costs
>>> reactor_cost = econ.reactor_cost(jnp.array(10.0), "cstr_jacketed")
>>> installed_cost = econ.installed_cost(reactor_cost, lang_factor=4.74)
>>>
>>> # Operating costs
>>> utility_cost = econ.steam_cost_from_duty(jnp.array(1e6), "medium_pressure")
>>>
>>> # Profitability
>>> npv = econ.npv(cash_flows, discount_rate=0.10, initial_investment=1e6)
>>> msp = econ.minimum_selling_price(total_cost, annual_production)
"""

# Cost indices
from .indices import (
    CEPCIData,
    CEPCI_HISTORICAL,
    DEFAULT_BASE_YEAR,
    DEFAULT_CURRENT_YEAR,
    get_cepci,
    escalate_cost,
    escalate_cost_ratio,
    inflation_factor,
    inflation_factor_continuous,
    CEPCI_RATIOS,
)

# Capital costs
from .capital import (
    # Core types
    CostParams,
    InstallationFactors,
    CapitalInvestment,
    # Cost parameter databases
    REACTOR_COSTS,
    VESSEL_COSTS,
    HEAT_EXCHANGER_COSTS,
    COLUMN_COSTS,
    PUMP_COSTS,
    COMPRESSOR_COSTS,
    SEPARATOR_COSTS,
    INSTALLATION_FACTORS,
    LANG_FACTORS,
    MATERIAL_FACTORS,
    # Core functions
    equipment_cost,
    equipment_cost_continuous,
    # Equipment-specific functions
    reactor_cost,
    vessel_cost,
    heat_exchanger_cost,
    pump_cost,
    compressor_cost,
    column_cost,
    separator_cost,
    # Installation costs
    installed_cost,
    installed_cost_detailed,
    bare_module_cost,
    # Correction factors
    pressure_factor_vessel,
    pressure_factor_hx,
    # Total capital investment
    total_capital_investment,
    total_capital_investment_detailed,
)

# Utility costs
from .utilities import (
    # Types
    UtilityPrices,
    UtilityConsumption,
    DEFAULT_PRICES,
    # Steam
    steam_cost_from_duty,
    steam_flowrate_from_duty,
    # Cooling
    cooling_water_cost,
    cooling_water_flowrate,
    refrigeration_cost,
    refrigeration_cost_continuous,
    # Electricity
    electricity_cost,
    electricity_cost_per_second,
    pump_electricity_cost,
    compressor_electricity_cost,
    # Fuels
    fuel_cost,
    # Water
    process_water_cost,
    wastewater_cost,
    # Combined
    total_utility_cost,
    utility_cost_from_heat_duties,
)

# Operating costs
from .opex import (
    # Types
    RawMaterial,
    LaborRates,
    OverheadFactors,
    OperatingSchedule,
    OperatingCostBreakdown,
    # Constants
    HOURS_PER_YEAR,
    SECONDS_PER_YEAR,
    DEFAULT_SCHEDULE,
    DEFAULT_LABOR_RATES,
    DEFAULT_OVERHEAD,
    # Raw materials
    raw_material_cost,
    raw_material_cost_molar,
    total_raw_material_cost,
    annual_raw_material_cost,
    # Labor
    operating_labor_cost,
    labor_cost_from_equipment,
    # Overhead
    maintenance_cost,
    insurance_taxes_cost,
    plant_overhead_cost,
    # Total OPEX
    calculate_opex,
    simple_opex,
    opex_per_unit_product,
    opex_from_fci_quick,
    com_from_correlations,
)

# Profitability analysis
from .profitability import (
    # Types
    FinancialParams,
    CashFlowResult,
    DEFAULT_FINANCIAL,
    MACRS_SCHEDULES,
    # Time value of money
    present_value,
    future_value,
    discount_factor,
    capital_recovery_factor,
    present_value_factor,
    # NPV
    npv,
    npv_with_construction,
    npv_gradient,
    # IRR
    irr,
    irr_approx,
    # Payback
    simple_payback,
    discounted_payback,
    # ROI
    roi,
    average_roi,
    # MSP
    minimum_selling_price,
    msp_with_target_roi,
    msp_with_npv_zero,
    # Annualized costs
    annualized_cost,
    equivalent_annual_cost,
    # Cash flow analysis
    generate_cash_flows,
    full_cash_flow_analysis,
    npv_sensitivity,
    # Optimization objectives
    profit_objective,
    npv_objective,
)

__all__ = [
    # Indices
    "CEPCIData",
    "CEPCI_HISTORICAL",
    "DEFAULT_BASE_YEAR",
    "DEFAULT_CURRENT_YEAR",
    "get_cepci",
    "escalate_cost",
    "escalate_cost_ratio",
    "inflation_factor",
    "inflation_factor_continuous",
    "CEPCI_RATIOS",
    # Capital - Types
    "CostParams",
    "InstallationFactors",
    "CapitalInvestment",
    # Capital - Databases
    "REACTOR_COSTS",
    "VESSEL_COSTS",
    "HEAT_EXCHANGER_COSTS",
    "COLUMN_COSTS",
    "PUMP_COSTS",
    "COMPRESSOR_COSTS",
    "SEPARATOR_COSTS",
    "INSTALLATION_FACTORS",
    "LANG_FACTORS",
    "MATERIAL_FACTORS",
    # Capital - Functions
    "equipment_cost",
    "equipment_cost_continuous",
    "reactor_cost",
    "vessel_cost",
    "heat_exchanger_cost",
    "pump_cost",
    "compressor_cost",
    "column_cost",
    "separator_cost",
    "installed_cost",
    "installed_cost_detailed",
    "bare_module_cost",
    "pressure_factor_vessel",
    "pressure_factor_hx",
    "total_capital_investment",
    "total_capital_investment_detailed",
    # Utilities - Types
    "UtilityPrices",
    "UtilityConsumption",
    "DEFAULT_PRICES",
    # Utilities - Functions
    "steam_cost_from_duty",
    "steam_flowrate_from_duty",
    "cooling_water_cost",
    "cooling_water_flowrate",
    "refrigeration_cost",
    "refrigeration_cost_continuous",
    "electricity_cost",
    "electricity_cost_per_second",
    "pump_electricity_cost",
    "compressor_electricity_cost",
    "fuel_cost",
    "process_water_cost",
    "wastewater_cost",
    "total_utility_cost",
    "utility_cost_from_heat_duties",
    # OPEX - Types
    "RawMaterial",
    "LaborRates",
    "OverheadFactors",
    "OperatingSchedule",
    "OperatingCostBreakdown",
    # OPEX - Constants
    "HOURS_PER_YEAR",
    "SECONDS_PER_YEAR",
    "DEFAULT_SCHEDULE",
    "DEFAULT_LABOR_RATES",
    "DEFAULT_OVERHEAD",
    # OPEX - Functions
    "raw_material_cost",
    "raw_material_cost_molar",
    "total_raw_material_cost",
    "annual_raw_material_cost",
    "operating_labor_cost",
    "labor_cost_from_equipment",
    "maintenance_cost",
    "insurance_taxes_cost",
    "plant_overhead_cost",
    "calculate_opex",
    "simple_opex",
    "opex_per_unit_product",
    "opex_from_fci_quick",
    "com_from_correlations",
    # Profitability - Types
    "FinancialParams",
    "CashFlowResult",
    "DEFAULT_FINANCIAL",
    "MACRS_SCHEDULES",
    # Profitability - Functions
    "present_value",
    "future_value",
    "discount_factor",
    "capital_recovery_factor",
    "present_value_factor",
    "npv",
    "npv_with_construction",
    "npv_gradient",
    "irr",
    "irr_approx",
    "simple_payback",
    "discounted_payback",
    "roi",
    "average_roi",
    "minimum_selling_price",
    "msp_with_target_roi",
    "msp_with_npv_zero",
    "annualized_cost",
    "equivalent_annual_cost",
    "generate_cash_flows",
    "full_cash_flow_analysis",
    "npv_sensitivity",
    "profit_objective",
    "npv_objective",
]
