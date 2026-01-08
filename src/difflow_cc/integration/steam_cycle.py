"""Steam cycle integration for amine reboiler.

Models steam extraction from power plant turbines
for solvent regeneration.

References:
    Lucquiaud M, Gibbins J (2011). On the integration of
        CO2 capture with coal-fired power plants.
        Chem Eng Res Des 89:1553-1571.
"""

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin

import jax.numpy as jnp
from jax import Array


@dataclass
class SteamCycleParams(ParamsMixin):
    """Steam cycle parameters.

    Attributes:
        hp_inlet_pressure: HP turbine inlet pressure (MPa)
        hp_inlet_temperature: HP turbine inlet temperature (K)
        ip_inlet_pressure: IP turbine inlet pressure (MPa)
        lp_inlet_pressure: LP turbine inlet pressure (MPa)
        condenser_pressure: Condenser pressure (kPa)
        turbine_efficiency: Isentropic efficiency
    """
    hp_inlet_pressure: float = 24.1  # MPa (supercritical)
    hp_inlet_temperature: float = 866.0  # K (593°C)
    ip_inlet_pressure: float = 4.0  # MPa
    lp_inlet_pressure: float = 0.8  # MPa
    condenser_pressure: float = 5.0  # kPa
    turbine_efficiency: float = 0.90


def steam_properties(
    pressure: Array | float,
    temperature: Array | float | None = None,
    quality: float | None = None,
) -> dict:
    """Simplified steam properties.

    Uses correlations for saturated and superheated steam.
    For accurate properties, use CoolProp or IAPWS.

    Args:
        pressure: Pressure (MPa)
        temperature: Temperature (K) for superheated
        quality: Steam quality (0-1) for wet steam

    Returns:
        Dict with h, s, T_sat
    """
    pressure = jnp.asarray(pressure)

    # Saturation temperature (simplified correlation)
    # T_sat = 100 + 73 * P^0.25 for P in MPa (approximate)
    T_sat = 373.15 + 73 * jnp.power(pressure, 0.25)

    # Saturation enthalpies (kJ/kg)
    # Simplified correlations
    h_f = 417 + 300 * jnp.log(pressure + 0.01)  # Liquid
    h_fg = 2258 - 200 * jnp.log(pressure + 0.01)  # Vaporization
    h_g = h_f + h_fg  # Vapor

    if quality is not None:
        # Wet steam
        h = h_f + quality * h_fg
        T = T_sat
    elif temperature is not None:
        temperature = jnp.asarray(temperature)
        # Superheated steam
        # h = h_g + Cp * (T - T_sat)
        Cp = 2.0  # kJ/(kg·K) approximate
        superheat = jnp.maximum(0, temperature - T_sat)
        h = h_g + Cp * superheat
        T = temperature
    else:
        # Saturated vapor
        h = h_g
        T = T_sat

    return {
        "h": h,  # kJ/kg
        "T": T,  # K
        "T_sat": T_sat,
        "h_f": h_f,
        "h_g": h_g,
    }


def extraction_pressure_options(
    reboiler_temperature: Array | float,
    approach: float = 10.0,
) -> Array:
    """Determine extraction pressure for reboiler duty.

    Steam extraction pressure must provide adequate driving
    force for heat transfer to reboiler.

    Args:
        reboiler_temperature: Reboiler temperature (K)
        approach: Minimum temperature approach (K)

    Returns:
        Required steam saturation pressure (MPa)
    """
    reboiler_temperature = jnp.asarray(reboiler_temperature)

    # Required steam temperature
    T_steam = reboiler_temperature + approach

    # Invert saturation temperature correlation
    # T_sat = 373.15 + 73 * P^0.25
    # P = ((T_sat - 373.15) / 73)^4
    P_sat = jnp.power((T_steam - 373.15) / 73, 4)
    P_sat = jnp.clip(P_sat, 0.1, 5.0)

    return P_sat


def steam_flow_for_duty(
    duty: Array | float,
    extraction_pressure: Array | float,
    condensate_subcooling: float = 10.0,
) -> Array:
    """Calculate steam flow rate for reboiler duty.

    Args:
        duty: Reboiler heat duty (W)
        extraction_pressure: Steam extraction pressure (MPa)
        condensate_subcooling: Condensate subcooling (K)

    Returns:
        Steam mass flow rate (kg/s)
    """
    duty = jnp.asarray(duty)
    extraction_pressure = jnp.asarray(extraction_pressure)

    # Steam properties at extraction
    props = steam_properties(extraction_pressure)
    h_steam = props["h"]  # kJ/kg

    # Condensate enthalpy (subcooled)
    T_cond = props["T_sat"] - condensate_subcooling
    h_cond = props["h_f"] - 4.18 * condensate_subcooling  # kJ/kg

    # Available heat per kg steam
    delta_h = h_steam - h_cond  # kJ/kg

    # Steam flow
    m_steam = duty / (delta_h * 1000)  # kg/s

    return m_steam


def crossover_extraction(
    duty: Array | float,
    params: SteamCycleParams,
) -> dict:
    """Model extraction from IP/LP crossover.

    Common extraction point for amine reboiler (0.3-0.5 MPa).

    Args:
        duty: Reboiler duty (W)
        params: Steam cycle parameters

    Returns:
        Extraction details including work impact
    """
    duty = jnp.asarray(duty)

    # Crossover pressure (between IP and LP)
    P_crossover = params.lp_inlet_pressure  # MPa

    # Steam flow needed
    m_steam = steam_flow_for_duty(duty, P_crossover)

    # Work lost = m_steam * (h_crossover - h_condenser)
    h_crossover = steam_properties(P_crossover)["h"]

    # Condenser enthalpy (wet steam at condenser pressure)
    P_cond = params.condenser_pressure / 1000  # MPa
    h_cond = steam_properties(P_cond, quality=0.90)["h"]

    # Isentropic work lost in LP turbine
    delta_h_is = h_crossover - h_cond
    eta = params.turbine_efficiency
    work_lost = m_steam * delta_h_is * eta * 1000  # W

    return {
        "extraction_pressure_MPa": P_crossover,
        "steam_flow_kg_s": m_steam,
        "h_extraction_kJ_kg": h_crossover,
        "work_lost_W": work_lost,
        "work_lost_MW": work_lost / 1e6,
        "specific_work_kJ_kg_steam": delta_h_is * eta,
    }


def letdown_valve(
    inlet_pressure: Array | float,
    outlet_pressure: Array | float,
    mass_flow: Array | float,
) -> dict:
    """Model pressure letdown through valve.

    Isenthalpic expansion (h = constant).

    Args:
        inlet_pressure: Inlet pressure (MPa)
        outlet_pressure: Outlet pressure (MPa)
        mass_flow: Steam flow rate (kg/s)

    Returns:
        Outlet conditions
    """
    inlet_pressure = jnp.asarray(inlet_pressure)
    outlet_pressure = jnp.asarray(outlet_pressure)

    # Inlet enthalpy
    h_in = steam_properties(inlet_pressure)["h"]

    # Outlet: same enthalpy, different pressure
    # Find outlet temperature (superheated)
    T_sat_out = steam_properties(outlet_pressure)["T_sat"]
    h_g_out = steam_properties(outlet_pressure)["h_g"]

    # If h_in > h_g_out, steam is superheated
    # T_out = T_sat + (h_in - h_g) / Cp
    Cp = 2.0  # kJ/(kg·K)
    superheat = (h_in - h_g_out) / Cp
    T_out = T_sat_out + jnp.maximum(0, superheat)

    return {
        "h": h_in,  # unchanged
        "T_out": T_out,
        "superheat": jnp.maximum(0, superheat),
        "P_out": outlet_pressure,
    }


def optimal_extraction_pressure(
    reboiler_temp: Array | float,
    params: SteamCycleParams,
) -> dict:
    """Find optimal steam extraction point.

    Trade-off between:
    - Higher pressure: more superheat, less efficient for heating
    - Lower pressure: better match to reboiler, more turbine work lost

    Args:
        reboiler_temp: Reboiler temperature (K)
        params: Steam cycle parameters

    Returns:
        Optimal extraction conditions
    """
    reboiler_temp = jnp.asarray(reboiler_temp)

    # Minimum pressure for adequate driving force
    P_min = extraction_pressure_options(reboiler_temp, approach=10.0)

    # Maximum practical pressure (IP exhaust)
    P_max = params.lp_inlet_pressure

    # Optimal is usually just above minimum for max efficiency
    P_opt = P_min * 1.1
    P_opt = jnp.clip(P_opt, P_min, P_max)

    # Properties at optimal point
    props = steam_properties(P_opt)

    return {
        "optimal_pressure_MPa": P_opt,
        "minimum_pressure_MPa": P_min,
        "steam_temperature_K": props["T"],
        "saturation_temp_K": props["T_sat"],
        "steam_enthalpy_kJ_kg": props["h"],
    }


def heat_integration_potential(
    rich_solvent_temp: Array | float,
    lean_solvent_temp: Array | float,
    solvent_flow: Array | float,
    Cp_solvent: float = 3500.0,  # J/(kg·K)
) -> dict:
    """Assess heat integration potential.

    Calculate heat recovery from:
    - Lean/rich cross-exchanger
    - Reboiler condensate
    - Stripper overhead

    Args:
        rich_solvent_temp: Rich solvent from absorber (K)
        lean_solvent_temp: Lean solvent from stripper (K)
        solvent_flow: Solvent mass flow (kg/s)
        Cp_solvent: Solvent heat capacity (J/(kg·K))

    Returns:
        Heat integration analysis
    """
    T_rich = jnp.asarray(rich_solvent_temp)
    T_lean = jnp.asarray(lean_solvent_temp)
    m_solvent = jnp.asarray(solvent_flow)

    # Maximum heat recovery in cross-exchanger
    Q_max = m_solvent * Cp_solvent * (T_lean - T_rich)

    # Realistic recovery (85% effectiveness)
    effectiveness = 0.85
    Q_recovered = effectiveness * Q_max

    # Remaining heating duty
    Q_remaining = Q_max - Q_recovered

    return {
        "Q_max_recovery_W": Q_max,
        "Q_actual_recovery_W": Q_recovered,
        "Q_external_heating_W": Q_remaining,
        "effectiveness": effectiveness,
        "T_rich_heated_K": T_rich + Q_recovered / (m_solvent * Cp_solvent),
    }
