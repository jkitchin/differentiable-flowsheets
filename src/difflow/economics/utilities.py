"""Utility cost models for process economics.

This module provides differentiable utility cost calculations for common
process utilities: steam, cooling water, electricity, fuels, etc.

All functions are JAX-compatible for gradient-based optimization.
"""

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple
from dataclasses import dataclass, field


# =============================================================================
# Utility Price Data
# =============================================================================

@dataclass
class UtilityPrices:
    """Standard utility prices (2024 US Gulf Coast basis).

    All prices in $/GJ unless otherwise noted.
    """
    # Steam (by pressure level)
    steam_high_pressure: float = 14.05  # 4.1 MPa, 254°C, $/GJ
    steam_medium_pressure: float = 11.80  # 1.1 MPa, 184°C, $/GJ
    steam_low_pressure: float = 9.50  # 0.34 MPa, 138°C, $/GJ

    # Cooling utilities
    cooling_water: float = 0.35  # $/GJ (ΔT = 10°C typical)
    chilled_water: float = 4.50  # $/GJ (5°C supply)
    refrigeration_moderate: float = 8.00  # $/GJ (-20°C)
    refrigeration_low: float = 13.50  # $/GJ (-50°C)
    cryogenic: float = 35.00  # $/GJ (< -100°C)

    # Electricity
    electricity: float = 0.07  # $/kWh

    # Fuels
    natural_gas: float = 4.50  # $/GJ (LHV basis)
    fuel_oil: float = 6.00  # $/GJ
    coal: float = 2.50  # $/GJ

    # Water and waste
    process_water: float = 0.50  # $/m³
    boiler_feed_water: float = 2.50  # $/m³ (treated)
    wastewater_treatment: float = 0.80  # $/m³

    # Compressed air and inerts
    compressed_air: float = 0.03  # $/Nm³
    nitrogen: float = 0.08  # $/Nm³
    oxygen: float = 0.12  # $/Nm³


# Default prices instance
DEFAULT_PRICES = UtilityPrices()


# =============================================================================
# Physical Constants for Utility Calculations
# =============================================================================

# Steam enthalpies (kJ/kg) at various conditions
STEAM_ENTHALPY = {
    "high_pressure": {"vapor": 2800.0, "liquid": 1087.0},  # 4.1 MPa
    "medium_pressure": {"vapor": 2780.0, "liquid": 781.0},  # 1.1 MPa
    "low_pressure": {"vapor": 2730.0, "liquid": 561.0},  # 0.34 MPa
}

# Specific heat capacities
WATER_CP = 4.18  # kJ/(kg·K)
AIR_CP = 1.005  # kJ/(kg·K)


# =============================================================================
# Steam Cost Functions
# =============================================================================

def steam_cost_from_duty(
    duty: Array,
    steam_level: str = "medium_pressure",
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate steam cost from heat duty.

    Args:
        duty: Heat duty (W or J/s, positive = heating required)
        steam_level: "high_pressure", "medium_pressure", or "low_pressure"
        prices: Utility prices (uses defaults if None)

    Returns:
        Steam cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    # Price mapping
    price_map = {
        "high_pressure": prices.steam_high_pressure,
        "medium_pressure": prices.steam_medium_pressure,
        "low_pressure": prices.steam_low_pressure,
    }

    if steam_level not in price_map:
        raise ValueError(f"Unknown steam level: {steam_level}")

    price_per_gj = price_map[steam_level]

    # Convert W to GJ/s: 1 W = 1e-9 GJ/s
    duty_gj_per_s = jnp.maximum(duty, 0.0) * 1e-9

    return duty_gj_per_s * price_per_gj


def steam_flowrate_from_duty(
    duty: Array,
    steam_level: str = "medium_pressure",
) -> Array:
    """Calculate steam mass flowrate from heat duty.

    Args:
        duty: Heat duty (W)
        steam_level: Steam pressure level

    Returns:
        Steam flowrate (kg/s)
    """
    if steam_level not in STEAM_ENTHALPY:
        raise ValueError(f"Unknown steam level: {steam_level}")

    enthalpy = STEAM_ENTHALPY[steam_level]
    delta_h = (enthalpy["vapor"] - enthalpy["liquid"]) * 1000  # J/kg

    # m_dot = Q / delta_h
    return jnp.maximum(duty, 0.0) / delta_h


# =============================================================================
# Cooling Utility Cost Functions
# =============================================================================

def cooling_water_cost(
    duty: Array,
    delta_T: Array = jnp.array(10.0),
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate cooling water cost from heat duty.

    Args:
        duty: Cooling duty (W, positive = cooling required)
        delta_T: Temperature rise of cooling water (K)
        prices: Utility prices

    Returns:
        Cooling water cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    # Convert W to GJ/s
    duty_gj_per_s = jnp.maximum(duty, 0.0) * 1e-9

    return duty_gj_per_s * prices.cooling_water


def cooling_water_flowrate(
    duty: Array,
    delta_T: Array = jnp.array(10.0),
) -> Array:
    """Calculate cooling water mass flowrate.

    Args:
        duty: Cooling duty (W)
        delta_T: Temperature rise (K)

    Returns:
        Cooling water flowrate (kg/s)
    """
    # Q = m_dot * Cp * delta_T
    # m_dot = Q / (Cp * delta_T)
    return jnp.maximum(duty, 0.0) / (WATER_CP * 1000 * delta_T)


def refrigeration_cost(
    duty: Array,
    temperature_level: float = -20.0,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate refrigeration cost based on temperature level.

    Args:
        duty: Refrigeration duty (W)
        temperature_level: Temperature (°C)
        prices: Utility prices

    Returns:
        Refrigeration cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    # Select price based on temperature
    if temperature_level >= 5.0:
        price = prices.chilled_water
    elif temperature_level >= -20.0:
        price = prices.refrigeration_moderate
    elif temperature_level >= -50.0:
        price = prices.refrigeration_low
    else:
        price = prices.cryogenic

    duty_gj_per_s = jnp.maximum(duty, 0.0) * 1e-9
    return duty_gj_per_s * price


def refrigeration_cost_continuous(
    duty: Array,
    temperature: Array,
    prices: UtilityPrices | None = None,
) -> Array:
    """Differentiable refrigeration cost with continuous temperature dependence.

    Uses smooth interpolation between price levels for gradient-based optimization.

    Args:
        duty: Refrigeration duty (W)
        temperature: Temperature level (°C)
        prices: Utility prices

    Returns:
        Refrigeration cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    duty_gj_per_s = jnp.maximum(duty, 0.0) * 1e-9

    # Smooth sigmoid transitions between price levels
    def sigmoid(x, center, width=5.0):
        return 1.0 / (1.0 + jnp.exp(-(x - center) / width))

    # Price increases as temperature decreases
    # T > 5: chilled water
    # -20 < T < 5: moderate refrigeration
    # -50 < T < -20: low temperature refrigeration
    # T < -50: cryogenic

    w1 = sigmoid(temperature, 5.0)  # 1 if T > 5
    w2 = sigmoid(temperature, -20.0) * (1 - w1)  # between -20 and 5
    w3 = sigmoid(temperature, -50.0) * (1 - w1 - w2)  # between -50 and -20
    w4 = 1 - w1 - w2 - w3  # below -50

    price = (
        w1 * prices.chilled_water +
        w2 * prices.refrigeration_moderate +
        w3 * prices.refrigeration_low +
        w4 * prices.cryogenic
    )

    return duty_gj_per_s * price


# =============================================================================
# Electricity Cost Functions
# =============================================================================

def electricity_cost(
    power: Array,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate electricity cost.

    Args:
        power: Power consumption (kW)
        prices: Utility prices

    Returns:
        Electricity cost ($/h)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    return jnp.maximum(power, 0.0) * prices.electricity


def electricity_cost_per_second(
    power: Array,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate electricity cost per second.

    Args:
        power: Power consumption (W)
        prices: Utility prices

    Returns:
        Electricity cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    # Convert W to kWh/s: 1 W = 1/3600000 kWh/s = 1/3.6e6 kWh/s
    power_kwh_per_s = jnp.maximum(power, 0.0) / 3.6e6

    return power_kwh_per_s * prices.electricity


def pump_electricity_cost(
    flowrate: Array,
    head: Array,
    efficiency: float = 0.70,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate pump electricity cost.

    Args:
        flowrate: Volumetric flowrate (m³/s)
        head: Pump head (m)
        efficiency: Pump efficiency (0-1)
        prices: Utility prices

    Returns:
        Electricity cost ($/s)
    """
    # Power = (rho * g * Q * H) / eta
    # Assuming water: rho = 1000 kg/m³, g = 9.81 m/s²
    power_w = (1000.0 * 9.81 * flowrate * head) / efficiency

    return electricity_cost_per_second(power_w, prices)


def compressor_electricity_cost(
    flowrate: Array,
    pressure_ratio: Array,
    inlet_temperature: Array = jnp.array(298.15),
    efficiency: float = 0.75,
    gamma: float = 1.4,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate compressor electricity cost (isentropic compression).

    Args:
        flowrate: Molar flowrate (mol/s)
        pressure_ratio: Outlet/inlet pressure ratio
        inlet_temperature: Inlet temperature (K)
        efficiency: Isentropic efficiency
        gamma: Heat capacity ratio (Cp/Cv)
        prices: Utility prices

    Returns:
        Electricity cost ($/s)
    """
    R = 8.314  # J/(mol·K)

    # Isentropic work: W = (gamma/(gamma-1)) * R * T1 * ((P2/P1)^((gamma-1)/gamma) - 1)
    exp = (gamma - 1) / gamma
    work_ideal = (gamma / (gamma - 1)) * R * inlet_temperature * (
        jnp.power(pressure_ratio, exp) - 1.0
    )

    # Actual power
    power_w = flowrate * work_ideal / efficiency

    return electricity_cost_per_second(power_w, prices)


# =============================================================================
# Fuel Cost Functions
# =============================================================================

def fuel_cost(
    duty: Array,
    fuel_type: str = "natural_gas",
    efficiency: float = 0.85,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate fuel cost for fired heating.

    Args:
        duty: Heat duty (W)
        fuel_type: "natural_gas", "fuel_oil", or "coal"
        efficiency: Furnace/boiler efficiency
        prices: Utility prices

    Returns:
        Fuel cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    price_map = {
        "natural_gas": prices.natural_gas,
        "fuel_oil": prices.fuel_oil,
        "coal": prices.coal,
    }

    if fuel_type not in price_map:
        raise ValueError(f"Unknown fuel type: {fuel_type}")

    # Fuel consumption = duty / efficiency
    fuel_gj_per_s = jnp.maximum(duty, 0.0) * 1e-9 / efficiency

    return fuel_gj_per_s * price_map[fuel_type]


# =============================================================================
# Water and Waste Treatment Cost Functions
# =============================================================================

def process_water_cost(
    flowrate: Array,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate process water cost.

    Args:
        flowrate: Volumetric flowrate (m³/s)
        prices: Utility prices

    Returns:
        Water cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    return jnp.maximum(flowrate, 0.0) * prices.process_water


def wastewater_cost(
    flowrate: Array,
    prices: UtilityPrices | None = None,
) -> Array:
    """Calculate wastewater treatment cost.

    Args:
        flowrate: Volumetric flowrate (m³/s)
        prices: Utility prices

    Returns:
        Wastewater cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    return jnp.maximum(flowrate, 0.0) * prices.wastewater_treatment


# =============================================================================
# Combined Utility Cost Calculator
# =============================================================================

@dataclass
class UtilityConsumption:
    """Container for utility consumption data."""
    heating_duty: float = 0.0  # W
    cooling_duty: float = 0.0  # W
    electricity: float = 0.0  # kW
    process_water: float = 0.0  # m³/s
    wastewater: float = 0.0  # m³/s
    steam_level: str = "medium_pressure"
    cooling_type: str = "cooling_water"  # or "refrigeration"
    cooling_temperature: float = 25.0  # °C for refrigeration


def total_utility_cost(
    consumption: UtilityConsumption,
    prices: UtilityPrices | None = None,
) -> float:
    """Calculate total utility cost from consumption data.

    Args:
        consumption: Utility consumption breakdown
        prices: Utility prices

    Returns:
        Total utility cost ($/s)
    """
    if prices is None:
        prices = DEFAULT_PRICES

    cost = 0.0

    # Heating (steam)
    if consumption.heating_duty > 0:
        cost += float(steam_cost_from_duty(
            jnp.array(consumption.heating_duty),
            consumption.steam_level,
            prices
        ))

    # Cooling
    if consumption.cooling_duty > 0:
        if consumption.cooling_type == "cooling_water":
            cost += float(cooling_water_cost(
                jnp.array(consumption.cooling_duty),
                prices=prices
            ))
        else:
            cost += float(refrigeration_cost(
                jnp.array(consumption.cooling_duty),
                consumption.cooling_temperature,
                prices
            ))

    # Electricity
    if consumption.electricity > 0:
        cost += float(electricity_cost(
            jnp.array(consumption.electricity),
            prices
        )) / 3600  # Convert $/h to $/s

    # Water
    if consumption.process_water > 0:
        cost += float(process_water_cost(
            jnp.array(consumption.process_water),
            prices
        ))

    if consumption.wastewater > 0:
        cost += float(wastewater_cost(
            jnp.array(consumption.wastewater),
            prices
        ))

    return cost


def utility_cost_from_heat_duties(
    heating_duty: Array,
    cooling_duty: Array,
    steam_level: str = "medium_pressure",
    prices: UtilityPrices | None = None,
) -> Array:
    """Simple utility cost from heating and cooling duties.

    Differentiable function for optimization.

    Args:
        heating_duty: Total heating duty (W)
        cooling_duty: Total cooling duty (W, positive)
        steam_level: Steam pressure level for heating
        prices: Utility prices

    Returns:
        Total utility cost ($/s)
    """
    heating_cost = steam_cost_from_duty(heating_duty, steam_level, prices)
    cooling_cost = cooling_water_cost(cooling_duty, prices=prices)

    return heating_cost + cooling_cost
