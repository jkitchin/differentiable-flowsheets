"""Levelized cost calculations for carbon capture.

Provides comprehensive cost metrics for comparing capture technologies.

References:
    Rubin ES et al. (2015). The cost of CO2 capture and storage.
        Int J Greenh Gas Control 40:378-400.
    GCCSI (2021). Technology Readiness and Costs of CCS.
"""

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin

import jax.numpy as jnp
from jax import Array

from difflow_cc.economics.capex import (
    installed_cost,
    annualized_capital_cost,
    CapexParams,
)
from difflow_cc.economics.opex import (
    total_operating_cost,
    specific_operating_cost,
    OpexParams,
)


@dataclass
class EconomicParams(ParamsMixin):
    """Economic analysis parameters.

    Attributes:
        discount_rate: Real discount rate (fraction)
        lifetime: Project lifetime (years)
        construction_time: Construction period (years)
        capacity_factor: Annual availability
        carbon_price: CO2 credit/tax price ($/tonne)
        electricity_price: Grid electricity ($/kWh)
        inflation_rate: Annual inflation (fraction)
    """
    discount_rate: float = 0.08
    lifetime: int = 25
    construction_time: int = 3
    capacity_factor: float = 0.85
    carbon_price: float = 50.0  # $/tonne CO2
    electricity_price: float = 0.06  # $/kWh
    inflation_rate: float = 0.02


@dataclass
class CaptureSystemCost:
    """Complete cost breakdown for a capture system.

    All costs in USD unless otherwise noted.
    """
    # Capital
    equipment_cost: float | Array = 0.0
    installed_cost: float | Array = 0.0
    total_overnight_cost: float | Array = 0.0
    annualized_capex: float | Array = 0.0

    # Operating
    utilities_cost: float | Array = 0.0
    consumables_cost: float | Array = 0.0
    fixed_cost: float | Array = 0.0
    total_opex: float | Array = 0.0

    # Performance
    CO2_captured_tpy: float | Array = 0.0  # tonne/yr
    capture_rate: float | Array = 0.0  # fraction
    energy_penalty: float | Array = 0.0  # fraction

    # Levelized costs
    capex_per_tonne: float | Array = 0.0
    opex_per_tonne: float | Array = 0.0
    total_cost_per_tonne: float | Array = 0.0
    cost_of_CO2_avoided: float | Array = 0.0


# =============================================================================
# Levelized Cost Calculations
# =============================================================================

def levelized_cost_capture(
    capital_cost: Array | float,
    annual_opex: Array | float,
    CO2_captured: Array | float,
    params: EconomicParams | None = None,
) -> dict:
    """Calculate levelized cost of CO2 capture.

    LCOC = (Annualized CAPEX + OPEX) / CO2 captured

    Args:
        capital_cost: Total overnight capital (USD)
        annual_opex: Annual operating cost (USD/yr)
        CO2_captured: CO2 capture rate (mol/s)
        params: Economic parameters

    Returns:
        Dict with cost breakdown ($/tonne CO2)
    """
    if params is None:
        params = EconomicParams()

    capital_cost = jnp.asarray(capital_cost)
    annual_opex = jnp.asarray(annual_opex)
    CO2_captured = jnp.asarray(CO2_captured)

    # Annualized capital
    ann_capex = annualized_capital_cost(
        capital_cost,
        params.discount_rate,
        params.lifetime,
    )

    # CO2 captured annually (tonne/yr)
    hours_per_year = 8760 * params.capacity_factor
    CO2_tonne_yr = CO2_captured * 44.0 * 3600 * hours_per_year / 1e6

    # Specific costs
    capex_per_tonne = ann_capex / (CO2_tonne_yr + 1e-10)
    opex_per_tonne = annual_opex / (CO2_tonne_yr + 1e-10)
    total_per_tonne = capex_per_tonne + opex_per_tonne

    return {
        "annualized_capex": ann_capex,
        "annual_opex": annual_opex,
        "CO2_captured_tpy": CO2_tonne_yr,
        "capex_per_tonne": capex_per_tonne,
        "opex_per_tonne": opex_per_tonne,
        "total_cost_per_tonne": total_per_tonne,
    }


def cost_of_co2_avoided(
    capture_cost_per_tonne: Array | float,
    reference_emissions: Array | float,
    capture_emissions: Array | float,
    reference_energy: Array | float,
    capture_energy: Array | float,
    electricity_price: float = 0.06,
) -> Array:
    """Calculate cost of CO2 avoided.

    Accounts for additional emissions from energy use.

    Cost_avoided = (Cost_capture - Cost_reference) / (E_reference - E_capture)

    where E = emissions per unit output

    Args:
        capture_cost_per_tonne: Capture cost ($/tonne CO2 captured)
        reference_emissions: Reference plant emissions (tonne CO2/MWh)
        capture_emissions: Capture plant emissions (tonne CO2/MWh)
        reference_energy: Reference plant output (MWh/yr)
        capture_energy: Capture plant output (MWh/yr, after energy penalty)
        electricity_price: Value of electricity ($/kWh)

    Returns:
        Cost of CO2 avoided ($/tonne)
    """
    capture_cost_per_tonne = jnp.asarray(capture_cost_per_tonne)
    reference_emissions = jnp.asarray(reference_emissions)
    capture_emissions = jnp.asarray(capture_emissions)

    # Emissions avoided per unit reference output
    emissions_avoided = reference_emissions - capture_emissions

    # Cost of avoided (simplified)
    cost_avoided = capture_cost_per_tonne / (emissions_avoided / reference_emissions + 1e-10)

    return cost_avoided


def energy_penalty_cost(
    base_power: Array | float,
    capture_power: Array | float,
    electricity_price: float = 0.06,
    hours_per_year: float = 7446,
) -> Array:
    """Cost of energy penalty from capture.

    Args:
        base_power: Power without capture (W)
        capture_power: Net power with capture (W)
        electricity_price: Electricity price ($/kWh)
        hours_per_year: Operating hours

    Returns:
        Annual cost of lost power ($/yr)
    """
    base_power = jnp.asarray(base_power)
    capture_power = jnp.asarray(capture_power)

    power_loss = base_power - capture_power  # W
    energy_loss = power_loss / 1000 * hours_per_year  # kWh/yr

    cost = energy_loss * electricity_price
    return cost


# =============================================================================
# Financial Metrics
# =============================================================================

def net_present_value(
    capital_cost: Array | float,
    annual_cash_flows: Array,
    discount_rate: float = 0.08,
) -> Array:
    """Calculate net present value.

    NPV = -C0 + sum(CFt / (1+r)^t)

    Args:
        capital_cost: Initial investment (USD)
        annual_cash_flows: Array of annual net cash flows (USD/yr)
        discount_rate: Discount rate (fraction)

    Returns:
        NPV (USD)
    """
    capital_cost = jnp.asarray(capital_cost)
    annual_cash_flows = jnp.asarray(annual_cash_flows)

    n_years = len(annual_cash_flows)
    years = jnp.arange(1, n_years + 1)

    discount_factors = 1 / jnp.power(1 + discount_rate, years)
    pv_cash_flows = jnp.sum(annual_cash_flows * discount_factors)

    npv = -capital_cost + pv_cash_flows
    return npv


def internal_rate_return(
    capital_cost: Array | float,
    annual_cash_flow: Array | float,
    lifetime: int = 25,
) -> Array:
    """Estimate internal rate of return.

    For constant annual cash flow, solves:
    C0 = CF * (1 - (1+IRR)^-n) / IRR

    Args:
        capital_cost: Initial investment (USD)
        annual_cash_flow: Constant annual cash flow (USD/yr)
        lifetime: Project lifetime (years)

    Returns:
        IRR estimate (fraction)
    """
    capital_cost = jnp.asarray(capital_cost)
    annual_cash_flow = jnp.asarray(annual_cash_flow)

    # Simple payback ratio
    payback_factor = capital_cost / (annual_cash_flow + 1e-10)

    # Approximate IRR using annuity formula inversion
    # For n=25, IRR ≈ 1/payback - 0.02 (rough approximation)
    irr_approx = 1 / (payback_factor + 1e-10) - 0.02
    irr = jnp.clip(irr_approx, -0.5, 1.0)

    return irr


def payback_period(
    capital_cost: Array | float,
    annual_cash_flow: Array | float,
) -> Array:
    """Simple payback period.

    Args:
        capital_cost: Initial investment (USD)
        annual_cash_flow: Annual net cash flow (USD/yr)

    Returns:
        Payback period (years)
    """
    capital_cost = jnp.asarray(capital_cost)
    annual_cash_flow = jnp.asarray(annual_cash_flow)

    payback = capital_cost / (annual_cash_flow + 1e-10)
    return payback


# =============================================================================
# Complete Analysis
# =============================================================================

def complete_cost_analysis(
    # Process performance
    CO2_captured_mol_s: Array | float,
    steam_duty_W: Array | float,
    electricity_W: Array | float,
    cooling_duty_W: Array | float,
    # Equipment
    equipment_costs: dict[str, Array | float],
    # Configuration
    membrane_area: Array | float = 0.0,
    adsorbent_mass: Array | float = 0.0,
    n_operators: int = 4,
    # Parameters
    capex_params: CapexParams | None = None,
    opex_params: OpexParams | None = None,
    econ_params: EconomicParams | None = None,
) -> CaptureSystemCost:
    """Complete techno-economic analysis.

    Args:
        CO2_captured_mol_s: CO2 capture rate (mol/s)
        steam_duty_W: Reboiler steam duty (W)
        electricity_W: Total power consumption (W)
        cooling_duty_W: Total cooling duty (W)
        equipment_costs: Dict of equipment costs
        membrane_area: Membrane area if applicable (m²)
        adsorbent_mass: Adsorbent mass if applicable (kg)
        n_operators: Operators per shift
        capex_params: CAPEX parameters
        opex_params: OPEX parameters
        econ_params: Economic parameters

    Returns:
        CaptureSystemCost with full breakdown
    """
    if capex_params is None:
        capex_params = CapexParams()
    if opex_params is None:
        opex_params = OpexParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Total equipment cost
    total_equip = sum(jnp.asarray(c) for c in equipment_costs.values())

    # Installed cost
    installed = installed_cost(total_equip, capex_params)

    # Annualized capital
    ann_capex = annualized_capital_cost(
        installed["total_overnight_cost"],
        econ_params.discount_rate,
        econ_params.lifetime,
    )

    # Operating costs
    opex = total_operating_cost(
        steam_duty=steam_duty_W,
        electricity=electricity_W,
        cooling_duty=cooling_duty_W,
        CO2_captured=CO2_captured_mol_s,
        capital_cost=installed["total_overnight_cost"],
        membrane_area=membrane_area,
        adsorbent_mass=adsorbent_mass,
        n_operators=n_operators,
        params=opex_params,
    )

    # CO2 captured annually
    hours_per_year = 8760 * econ_params.capacity_factor
    CO2_tonne_yr = float(CO2_captured_mol_s) * 44.0 * 3600 * hours_per_year / 1e6

    # Specific costs
    capex_per_tonne = ann_capex / (CO2_tonne_yr + 1e-10)
    opex_per_tonne = opex["total_opex"] / (CO2_tonne_yr + 1e-10)
    total_per_tonne = capex_per_tonne + opex_per_tonne

    return CaptureSystemCost(
        equipment_cost=total_equip,
        installed_cost=installed["bare_erected_cost"],
        total_overnight_cost=installed["total_overnight_cost"],
        annualized_capex=ann_capex,
        utilities_cost=opex["utilities_total"],
        consumables_cost=opex["consumables_total"],
        fixed_cost=opex["fixed_total"],
        total_opex=opex["total_opex"],
        CO2_captured_tpy=CO2_tonne_yr,
        capex_per_tonne=capex_per_tonne,
        opex_per_tonne=opex_per_tonne,
        total_cost_per_tonne=total_per_tonne,
    )
