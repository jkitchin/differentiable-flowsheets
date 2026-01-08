"""Operating cost calculations for carbon capture.

Includes utilities, consumables, labor, and maintenance.

All functions are JAX-compatible for optimization.

References:
    NETL (2019). Cost and Performance Baseline for Fossil
        Energy Plants. DOE/NETL-2019/1946.
    IEAGHG (2019). Towards Zero Emissions CCS in Power
        Plants Using Higher Capture Rates or Biomass.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass
class OpexParams:
    """Operating cost parameters.

    Attributes:
        steam_price: Steam cost ($/GJ)
        electricity_price: Electricity cost ($/kWh)
        cooling_water_price: Cooling water cost ($/m³)
        solvent_price: Amine solvent cost ($/kg)
        adsorbent_price: Adsorbent cost ($/kg)
        membrane_price: Membrane module cost ($/m²)
        labor_rate: Operator labor rate ($/hr)
        capacity_factor: Annual capacity factor
        operating_hours: Annual operating hours (if not using capacity_factor)
    """
    steam_price: float = 15.0  # $/GJ
    electricity_price: float = 0.06  # $/kWh
    cooling_water_price: float = 0.05  # $/m³
    solvent_price: float = 2.0  # $/kg (MEA)
    adsorbent_price: float = 5.0  # $/kg
    membrane_price: float = 50.0  # $/m²
    labor_rate: float = 50.0  # $/hr
    capacity_factor: float = 0.85
    operating_hours: float | None = None

    @property
    def hours_per_year(self) -> float:
        if self.operating_hours is not None:
            return self.operating_hours
        return 8760 * self.capacity_factor


# =============================================================================
# Utility Costs
# =============================================================================

def steam_cost(
    duty: Array | float,
    params: OpexParams | None = None,
) -> Array:
    """Annual steam cost.

    Args:
        duty: Steam consumption rate (W thermal)
        params: Operating cost parameters

    Returns:
        Annual cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    duty = jnp.asarray(duty)

    # Convert W to GJ/yr
    # W * 3600 s/hr * hours/yr / 1e9 = GJ/yr
    energy_GJ_yr = duty * 3600 * params.hours_per_year / 1e9

    cost = energy_GJ_yr * params.steam_price
    return cost


def electricity_cost(
    power: Array | float,
    params: OpexParams | None = None,
) -> Array:
    """Annual electricity cost.

    Args:
        power: Power consumption (W)
        params: Operating cost parameters

    Returns:
        Annual cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    power = jnp.asarray(power)

    # Convert W to kWh/yr
    energy_kWh_yr = power / 1000 * params.hours_per_year

    cost = energy_kWh_yr * params.electricity_price
    return cost


def cooling_water_cost(
    duty: Array | float,
    delta_T: float = 10.0,
    params: OpexParams | None = None,
) -> Array:
    """Annual cooling water cost.

    Args:
        duty: Cooling duty (W thermal)
        delta_T: Temperature rise in cooling water (K)
        params: Operating cost parameters

    Returns:
        Annual cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    duty = jnp.asarray(duty)

    # Water flow rate: Q = m_dot * Cp * dT
    # m_dot = Q / (Cp * dT), Cp_water = 4186 J/(kg·K)
    Cp_water = 4186.0
    m_dot = duty / (Cp_water * delta_T)  # kg/s

    # Volume flow (rho = 1000 kg/m³)
    V_dot = m_dot / 1000  # m³/s

    # Annual volume
    V_yr = V_dot * 3600 * params.hours_per_year  # m³/yr

    cost = V_yr * params.cooling_water_price
    return cost


# =============================================================================
# Consumables
# =============================================================================

def solvent_makeup_cost(
    CO2_captured: Array | float,
    loss_rate: float = 1.5,  # kg solvent / tonne CO2
    params: OpexParams | None = None,
) -> Array:
    """Annual solvent makeup cost.

    Solvent losses occur due to:
    - Volatilization (entrainment)
    - Degradation (oxidative, thermal)
    - Reclaimer losses

    Args:
        CO2_captured: CO2 capture rate (mol/s or kg/s depending on context)
        loss_rate: Solvent loss rate (kg solvent / tonne CO2)
        params: Operating cost parameters

    Returns:
        Annual cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    CO2_captured = jnp.asarray(CO2_captured)

    # Assume CO2_captured is in mol/s, convert to tonne/yr
    # mol/s * 44 g/mol * 3600 s/hr * hours/yr / 1e6 = tonne/yr
    CO2_tonne_yr = CO2_captured * 44.0 * 3600 * params.hours_per_year / 1e6

    # Solvent makeup
    solvent_kg_yr = CO2_tonne_yr * loss_rate

    cost = solvent_kg_yr * params.solvent_price
    return cost


def membrane_replacement_cost(
    area: Array | float,
    lifetime_years: float = 5.0,
    params: OpexParams | None = None,
) -> Array:
    """Annual membrane replacement cost.

    Args:
        area: Total membrane area (m²)
        lifetime_years: Membrane lifetime (years)
        params: Operating cost parameters

    Returns:
        Annualized replacement cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    area = jnp.asarray(area)

    # Total replacement cost
    replacement_cost = area * params.membrane_price

    # Annualized
    annual_cost = replacement_cost / lifetime_years

    return annual_cost


def adsorbent_replacement_cost(
    mass: Array | float,
    lifetime_years: float = 3.0,
    params: OpexParams | None = None,
) -> Array:
    """Annual adsorbent replacement cost.

    Args:
        mass: Total adsorbent mass (kg)
        lifetime_years: Adsorbent lifetime (years)
        params: Operating cost parameters

    Returns:
        Annualized replacement cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    mass = jnp.asarray(mass)

    # Total replacement cost
    replacement_cost = mass * params.adsorbent_price

    # Annualized
    annual_cost = replacement_cost / lifetime_years

    return annual_cost


# =============================================================================
# Labor and Maintenance
# =============================================================================

def labor_cost(
    n_operators: int = 4,
    shifts: int = 4,  # 4 shifts for 24/7 operation
    params: OpexParams | None = None,
) -> Array:
    """Annual labor cost.

    Args:
        n_operators: Operators per shift
        shifts: Number of shifts
        params: Operating cost parameters

    Returns:
        Annual cost ($/yr)
    """
    if params is None:
        params = OpexParams()

    # Annual hours per position (includes benefits multiplier)
    hours_per_position = 2000  # ~40 hr/week
    benefits_multiplier = 1.4

    total_positions = n_operators * shifts
    cost = total_positions * hours_per_position * params.labor_rate * benefits_multiplier

    return jnp.asarray(cost)


def maintenance_cost(
    capital_cost: Array | float,
    maintenance_fraction: float = 0.025,
) -> Array:
    """Annual maintenance cost.

    Typically 2-4% of capital cost.

    Args:
        capital_cost: Total installed capital (USD)
        maintenance_fraction: Maintenance as fraction of capital

    Returns:
        Annual cost ($/yr)
    """
    capital_cost = jnp.asarray(capital_cost)
    return capital_cost * maintenance_fraction


def insurance_and_taxes(
    capital_cost: Array | float,
    fraction: float = 0.02,
) -> Array:
    """Annual insurance and property taxes.

    Args:
        capital_cost: Total installed capital (USD)
        fraction: Annual rate (fraction of capital)

    Returns:
        Annual cost ($/yr)
    """
    capital_cost = jnp.asarray(capital_cost)
    return capital_cost * fraction


# =============================================================================
# Aggregation
# =============================================================================

def total_operating_cost(
    steam_duty: Array | float = 0.0,
    electricity: Array | float = 0.0,
    cooling_duty: Array | float = 0.0,
    CO2_captured: Array | float = 0.0,
    capital_cost: Array | float = 0.0,
    membrane_area: Array | float = 0.0,
    adsorbent_mass: Array | float = 0.0,
    n_operators: int = 4,
    params: OpexParams | None = None,
) -> dict:
    """Calculate total annual operating cost.

    Args:
        steam_duty: Reboiler steam duty (W)
        electricity: Total power consumption (W)
        cooling_duty: Total cooling duty (W)
        CO2_captured: CO2 capture rate (mol/s)
        capital_cost: Total installed capital (USD)
        membrane_area: Membrane area if applicable (m²)
        adsorbent_mass: Adsorbent mass if applicable (kg)
        n_operators: Operators per shift
        params: Cost parameters

    Returns:
        Dict with cost breakdown
    """
    if params is None:
        params = OpexParams()

    # Utilities
    cost_steam = steam_cost(steam_duty, params)
    cost_elec = electricity_cost(electricity, params)
    cost_cooling = cooling_water_cost(cooling_duty, 10.0, params)
    utilities = cost_steam + cost_elec + cost_cooling

    # Consumables
    cost_solvent = solvent_makeup_cost(CO2_captured, 1.5, params)
    cost_membrane = membrane_replacement_cost(membrane_area, 5.0, params)
    cost_adsorbent = adsorbent_replacement_cost(adsorbent_mass, 3.0, params)
    consumables = cost_solvent + cost_membrane + cost_adsorbent

    # Fixed costs
    cost_labor = labor_cost(n_operators, 4, params)
    cost_maint = maintenance_cost(capital_cost, 0.025)
    cost_insur = insurance_and_taxes(capital_cost, 0.02)
    fixed = cost_labor + cost_maint + cost_insur

    # Total
    total = utilities + consumables + fixed

    return {
        "steam": cost_steam,
        "electricity": cost_elec,
        "cooling_water": cost_cooling,
        "utilities_total": utilities,
        "solvent_makeup": cost_solvent,
        "membrane_replacement": cost_membrane,
        "adsorbent_replacement": cost_adsorbent,
        "consumables_total": consumables,
        "labor": cost_labor,
        "maintenance": cost_maint,
        "insurance_taxes": cost_insur,
        "fixed_total": fixed,
        "total_opex": total,
    }


def specific_operating_cost(
    total_opex: Array | float,
    CO2_captured: Array | float,
    params: OpexParams | None = None,
) -> Array:
    """Operating cost per tonne CO2 captured.

    Args:
        total_opex: Total annual operating cost ($/yr)
        CO2_captured: CO2 capture rate (mol/s)
        params: Operating parameters for hours

    Returns:
        Specific OPEX ($/tonne CO2)
    """
    if params is None:
        params = OpexParams()

    total_opex = jnp.asarray(total_opex)
    CO2_captured = jnp.asarray(CO2_captured)

    # CO2 captured annually (tonne/yr)
    CO2_tonne_yr = CO2_captured * 44.0 * 3600 * params.hours_per_year / 1e6

    specific = total_opex / (CO2_tonne_yr + 1e-10)
    return specific
