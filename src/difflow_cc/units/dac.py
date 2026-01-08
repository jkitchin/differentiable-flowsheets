"""Direct Air Capture (DAC) unit operations.

This module provides models for direct air capture systems:
- Solid sorbent systems (TSA/TVSA)
- Liquid solvent systems (aqueous amine, hydroxide)
- Contactor designs

DAC faces unique challenges:
- Very low CO2 concentration (~420 ppm)
- Large air volumes required
- High energy demand per tonne CO2

All models are JAX-compatible for optimization.

References:
    Sanz-Pérez ES et al. (2016). Direct Capture of CO2 from
        Ambient Air. Chem Rev 116:11840-11876.
    Fasihi M et al. (2019). Techno-economic assessment of
        CO2 direct air capture plants. J Clean Prod 224:957-980.
    Keith DW et al. (2018). A Process for Capturing CO2 from
        the Atmosphere. Joule 2:1573-1594.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow_cc.database import get_adsorbent


# Constants
CO2_AMBIENT = 420e-6  # 420 ppm
MW_CO2 = 44.01  # g/mol
R = 8.314  # J/(mol·K)


@dataclass
class DACParams(ParamsMixin):
    """Parameters for DAC unit.

    Attributes:
        technology: 'solid_sorbent' or 'liquid_solvent'
        sorbent: Sorbent material name
        contactor_type: 'packed_bed', 'monolith', 'spray_tower'
        air_velocity: Superficial air velocity (m/s)
        bed_length: Contactor length in flow direction (m)
        cross_section: Contactor cross-sectional area (m²)
        n_units: Number of parallel contactor units
        T_adsorption: Adsorption temperature (K)
        T_desorption: Desorption/regeneration temperature (K)
        P_desorption: Desorption pressure for TVSA (Pa)
        cycle_time_ads: Adsorption time (s)
        cycle_time_des: Desorption time (s)
        ambient_humidity: Relative humidity (0-1)
    """
    technology: str = "solid_sorbent"
    sorbent: str = "PEI_Silica"
    contactor_type: str = "packed_bed"
    air_velocity: float | Array = 1.5  # m/s
    bed_length: float | Array = 0.5  # m
    cross_section: float | Array = 100.0  # m²
    n_units: int = 4  # For continuous operation
    T_adsorption: float | Array = 298.15  # K
    T_desorption: float | Array = 373.15  # K (100°C for amine sorbents)
    P_desorption: float | Array = 101325.0  # Pa (1 atm for TSA)
    cycle_time_ads: float = 1800.0  # 30 min
    cycle_time_des: float = 900.0  # 15 min
    ambient_humidity: float = 0.5


@dataclass
class LiquidDACParams(ParamsMixin):
    """Parameters for liquid solvent DAC.

    Attributes:
        solvent: Solvent type ('KOH', 'NaOH')
        n_contactors: Number of air contactors
        contactor_diameter: Contactor diameter (m)
        contactor_height: Contactor height (m)
        air_velocity: Air velocity in contactor (m/s)
        L_G_ratio: Liquid to gas ratio (kg/kg)
        calciner_temperature: Calciner temperature (K)
    """
    solvent: str = "KOH"
    n_contactors: int = 100
    contactor_diameter: float | Array = 10.0  # m
    contactor_height: float | Array = 8.0  # m
    air_velocity: float | Array = 1.5  # m/s
    L_G_ratio: float = 2.0  # kg liquid/kg air
    calciner_temperature: float | Array = 1173.15  # K (900°C)


# =============================================================================
# Solid Sorbent DAC
# =============================================================================

class SolidSorbentDAC:
    """Solid sorbent direct air capture unit.

    Uses amine-functionalized adsorbents with temperature
    or temperature-vacuum swing for regeneration.

    Typical sorbents:
    - PEI on silica (polyethylenimine)
    - TEPA on alumina
    - Amine-functionalized MOFs

    Example:
        >>> params = DACParams(
        ...     sorbent='PEI_Silica',
        ...     T_desorption=373.15,
        ...     n_units=4,
        ... )
        >>> dac = SolidSorbentDAC(params)
        >>> co2_out, info = dac(ambient_air)
    """

    def __init__(self, params: DACParams):
        self.params = params
        self._sorbent = get_adsorbent(params.sorbent)

    def __call__(
        self,
        ambient_air: Stream | None = None,
        T_ambient: float | Array = 298.15,
        P_ambient: float | Array = 101325.0,
    ) -> tuple[Stream, dict]:
        """Capture CO2 from ambient air.

        Args:
            ambient_air: Optional air stream (if None, uses ambient)
            T_ambient: Ambient temperature (K)
            P_ambient: Ambient pressure (Pa)

        Returns:
            co2_product: Captured CO2 stream
            info: Performance metrics
        """
        p = self.params

        T_ambient = jnp.asarray(T_ambient)
        P_ambient = jnp.asarray(P_ambient)

        # Air flow per contactor
        air_velocity = jnp.asarray(p.air_velocity)
        cross_section = jnp.asarray(p.cross_section)
        V_air = air_velocity * cross_section  # m³/s per unit

        # Molar air flow (ideal gas)
        n_air = V_air * P_ambient / (R * T_ambient)  # mol/s

        # CO2 in air
        n_CO2_in = n_air * CO2_AMBIENT  # mol/s

        # Sorbent properties
        sorbent = self._sorbent

        # Get isotherm if available
        try:
            isotherm = sorbent.isotherms.get("CO2")
            if isotherm:
                # Calculate loadings
                P_CO2_ads = CO2_AMBIENT * P_ambient
                P_CO2_des = CO2_AMBIENT * jnp.asarray(p.P_desorption) * 0.1  # Lower in desorption

                # Simplified working capacity
                T_ads = jnp.asarray(p.T_adsorption)
                T_des = jnp.asarray(p.T_desorption)

                # Temperature effect on capacity (simplified)
                q_ads = sorbent.CO2_capacity * jnp.exp(-0.02 * (T_ads - 298.15))
                q_des = sorbent.CO2_capacity * jnp.exp(-0.02 * (T_des - 298.15)) * 0.2

                working_capacity = q_ads - q_des
            else:
                # Use nominal capacity
                working_capacity = sorbent.CO2_capacity * 0.3  # 30% working capacity
        except Exception:
            working_capacity = 1.0  # mol/kg fallback

        working_capacity = jnp.asarray(working_capacity)

        # Sorbent mass per unit
        bed_length = jnp.asarray(p.bed_length)
        bed_volume = cross_section * bed_length  # m³
        bulk_density = 500.0  # kg/m³ (typical for supported amines)
        sorbent_mass = bed_volume * bulk_density  # kg per unit

        # CO2 captured per cycle
        CO2_per_cycle = sorbent_mass * working_capacity  # mol per unit

        # Cycle time
        cycle_time = p.cycle_time_ads + p.cycle_time_des  # s

        # Capture rate per unit (time-averaged)
        capture_rate_unit = CO2_per_cycle / cycle_time  # mol/s per unit

        # Total capture with multiple units
        # With n units cycling, average capture ≈ n * rate * (t_ads / cycle_time)
        duty_fraction = p.cycle_time_ads / cycle_time
        total_capture = p.n_units * capture_rate_unit * duty_fraction  # mol/s

        # Capture efficiency (fraction of CO2 in processed air)
        air_processed = p.n_units * n_air * duty_fraction
        CO2_available = air_processed * CO2_AMBIENT
        capture_efficiency = total_capture / (CO2_available + 1e-10)
        capture_efficiency = jnp.clip(capture_efficiency, 0.0, 0.95)

        # Energy requirements
        # Heat of adsorption
        delta_H_ads = jnp.asarray(sorbent.heat_of_adsorption) * 1000  # J/mol

        # Sensible heat for sorbent
        Cp_sorbent = 1000.0  # J/(kg·K)
        T_swing = jnp.asarray(p.T_desorption) - jnp.asarray(p.T_adsorption)
        Q_sensible = sorbent_mass * Cp_sorbent * T_swing / cycle_time  # W per unit

        # Desorption heat (per mol CO2)
        Q_desorption = capture_rate_unit * delta_H_ads  # W per unit

        # Total thermal (all units)
        Q_thermal_total = p.n_units * (Q_sensible + Q_desorption)  # W

        # Fan power (pressure drop)
        dP = 500.0  # Pa (typical for packed bed)
        fan_power_unit = V_air * dP / 0.7  # W per unit (70% efficiency)
        fan_power_total = p.n_units * fan_power_unit * duty_fraction

        # Vacuum power (if TVSA)
        if p.P_desorption < 50000:  # Vacuum
            P_vac = jnp.asarray(p.P_desorption)
            vacuum_ratio = P_ambient / P_vac
            # Vacuum pump work (isothermal compression)
            V_desorb = capture_rate_unit * R * jnp.asarray(p.T_desorption) / P_vac
            W_vacuum = V_desorb * P_vac * jnp.log(vacuum_ratio) / 0.6  # 60% efficiency
            vacuum_power = p.n_units * W_vacuum
        else:
            vacuum_power = 0.0

        # Total electrical
        electrical_total = fan_power_total + vacuum_power

        # Specific energy (GJ/tonne CO2)
        CO2_mass_rate = total_capture * MW_CO2 / 1000  # kg/s
        specific_thermal = Q_thermal_total / (CO2_mass_rate * 1e9 + 1e-10)  # GJ/tonne
        specific_electrical = electrical_total / (CO2_mass_rate * 1e9 + 1e-10)  # GJ/tonne

        # Create CO2 product stream
        co2_flows = {"CO2": total_capture}
        co2_product = make_stream(co2_flows, p.T_desorption, p.P_desorption)

        info = {
            "CO2_captured_mol_s": total_capture,
            "CO2_captured_kg_s": CO2_mass_rate,
            "CO2_captured_tonne_yr": CO2_mass_rate * 3600 * 8760 / 1000,
            "capture_efficiency": capture_efficiency,
            "working_capacity": working_capacity,
            "sorbent_mass_per_unit": sorbent_mass,
            "total_sorbent_mass": sorbent_mass * p.n_units,
            "air_processed_m3_s": V_air * p.n_units * duty_fraction,
            "Q_thermal_W": Q_thermal_total,
            "Q_sensible_W": Q_sensible * p.n_units,
            "Q_desorption_W": Q_desorption * p.n_units,
            "P_fan_W": fan_power_total,
            "P_vacuum_W": vacuum_power,
            "P_electrical_W": electrical_total,
            "specific_thermal_GJ_tonne": specific_thermal,
            "specific_electrical_GJ_tonne": specific_electrical,
            "specific_total_GJ_tonne": specific_thermal + specific_electrical * 2.5,
            "cycle_time_s": cycle_time,
            "n_units": p.n_units,
        }

        return co2_product, info


# =============================================================================
# Liquid Solvent DAC
# =============================================================================

class LiquidSolventDAC:
    """Liquid solvent direct air capture (e.g., Carbon Engineering design).

    Uses aqueous KOH or NaOH to capture CO2, forming carbonate.
    Regeneration via calcium caustic recovery loop:
    1. Absorber: CO2 + 2KOH → K2CO3 + H2O
    2. Pellet reactor: K2CO3 + Ca(OH)2 → CaCO3 + 2KOH
    3. Calciner: CaCO3 → CaO + CO2 (high T)
    4. Slaker: CaO + H2O → Ca(OH)2

    Example:
        >>> params = LiquidDACParams(
        ...     solvent='KOH',
        ...     n_contactors=100,
        ... )
        >>> dac = LiquidSolventDAC(params)
        >>> co2_out, info = dac()
    """

    def __init__(self, params: LiquidDACParams):
        self.params = params

    def __call__(
        self,
        T_ambient: float | Array = 298.15,
        P_ambient: float | Array = 101325.0,
        humidity: float = 0.5,
    ) -> tuple[Stream, dict]:
        """Capture CO2 from ambient air.

        Args:
            T_ambient: Ambient temperature (K)
            P_ambient: Ambient pressure (Pa)
            humidity: Relative humidity

        Returns:
            co2_product: Captured CO2 stream
            info: Performance metrics
        """
        p = self.params

        T_ambient = jnp.asarray(T_ambient)
        P_ambient = jnp.asarray(P_ambient)

        # Air flow per contactor
        diameter = jnp.asarray(p.contactor_diameter)
        A_contactor = jnp.pi * (diameter / 2) ** 2
        V_air_unit = jnp.asarray(p.air_velocity) * A_contactor  # m³/s

        # Total air flow
        V_air_total = V_air_unit * p.n_contactors

        # Molar air flow
        n_air = V_air_total * P_ambient / (R * T_ambient)  # mol/s
        n_CO2_in = n_air * CO2_AMBIENT  # mol/s

        # Capture efficiency (typically 70-80% for liquid systems)
        # Limited by mass transfer and equilibrium
        capture_eff = 0.75
        n_CO2_captured = n_CO2_in * capture_eff

        # Energy requirements
        # 1. Fan power
        dP_contactor = 100.0  # Pa (spray tower, low pressure drop)
        fan_power = V_air_total * dP_contactor / 0.7  # W

        # 2. Liquid circulation
        L_G_mass = p.L_G_ratio
        rho_air = P_ambient * 0.029 / (R * T_ambient)  # kg/m³
        m_air = V_air_total * rho_air  # kg/s
        m_liquid = m_air * L_G_mass  # kg/s
        pump_head = jnp.asarray(p.contactor_height) + 10.0  # m
        pump_power = 1000 * 9.81 * pump_head * (m_liquid / 1000) / 0.7  # W

        # 3. Calciner (high temperature thermal)
        # CaCO3 → CaO + CO2, ΔH = 178 kJ/mol
        delta_H_calcine = 178000.0  # J/mol CO2
        Q_calciner = n_CO2_captured * delta_H_calcine  # W

        # 4. Slaker heat recovery (exothermic)
        # CaO + H2O → Ca(OH)2, ΔH = -65 kJ/mol
        delta_H_slaker = -65000.0  # J/mol
        Q_slaker = n_CO2_captured * delta_H_slaker  # W (negative = released)

        # 5. Sensible heat losses (simplified)
        Q_sensible_loss = 0.1 * Q_calciner  # 10% heat loss

        # Net thermal demand
        Q_thermal_net = Q_calciner + Q_slaker + Q_sensible_loss

        # Electrical (fans + pumps + auxiliaries)
        P_electrical = fan_power + pump_power
        P_auxiliaries = 0.2 * P_electrical  # 20% for other equipment
        P_electrical_total = P_electrical + P_auxiliaries

        # High-temperature heat for calciner (natural gas or electric)
        # If using oxy-fuel calciner, need ASU power
        asu_power = n_CO2_captured * 0.3 * 1e6 / 1000  # ~300 kWh/tonne O2

        # Specific energy
        CO2_mass_rate = n_CO2_captured * MW_CO2 / 1000  # kg/s

        specific_thermal = Q_thermal_net / (CO2_mass_rate * 1e9 + 1e-10)  # GJ/tonne
        specific_electrical = (P_electrical_total + asu_power) / (CO2_mass_rate * 1e9 + 1e-10)

        # CO2 product (from calciner, high purity)
        T_calciner = jnp.asarray(p.calciner_temperature)
        co2_flows = {"CO2": n_CO2_captured}
        co2_product = make_stream(co2_flows, T_calciner, P_ambient)

        info = {
            "CO2_captured_mol_s": n_CO2_captured,
            "CO2_captured_kg_s": CO2_mass_rate,
            "CO2_captured_tonne_yr": CO2_mass_rate * 3600 * 8760 / 1000,
            "capture_efficiency": capture_eff,
            "air_processed_m3_s": V_air_total,
            "liquid_circulation_kg_s": m_liquid,
            "Q_calciner_W": Q_calciner,
            "Q_slaker_W": Q_slaker,
            "Q_thermal_net_W": Q_thermal_net,
            "P_fan_W": fan_power,
            "P_pump_W": pump_power,
            "P_asu_W": asu_power,
            "P_electrical_W": P_electrical_total + asu_power,
            "specific_thermal_GJ_tonne": specific_thermal,
            "specific_electrical_GJ_tonne": specific_electrical,
            "specific_total_GJ_tonne": specific_thermal + specific_electrical * 2.5,
            "n_contactors": p.n_contactors,
        }

        return co2_product, info


# =============================================================================
# DAC Cost Estimates
# =============================================================================

def dac_cost_estimate(
    capacity_tonne_yr: Array | float,
    technology: str = "solid_sorbent",
    nth_plant: int = 1,
) -> dict:
    """Estimate DAC capital and operating costs.

    Uses learning curve projections from literature.

    Args:
        capacity_tonne_yr: Annual capture capacity (tonne CO2/yr)
        technology: 'solid_sorbent' or 'liquid_solvent'
        nth_plant: Plant number for learning curve

    Returns:
        Cost estimates
    """
    capacity = jnp.asarray(capacity_tonne_yr)

    # Base costs (2020 USD) for 1 Mt/yr plant
    if technology == "solid_sorbent":
        capex_base = 1200.0  # $/tonne/yr capacity (FOAK)
        opex_base = 200.0  # $/tonne (dominated by energy)
        learning_rate = 0.15  # 15% cost reduction per doubling
    else:  # liquid_solvent
        capex_base = 800.0  # $/tonne/yr capacity
        opex_base = 150.0  # $/tonne
        learning_rate = 0.10

    # Learning curve adjustment
    # Cost_n = Cost_1 * n^(-b) where b = log2(1 - learning_rate)
    b = jnp.log(1 - learning_rate) / jnp.log(2)
    learning_factor = jnp.power(nth_plant, b)

    capex_per_capacity = capex_base * learning_factor
    opex_per_tonne = opex_base * learning_factor

    # Total capital
    total_capex = capex_per_capacity * capacity

    # Annual operating
    annual_opex = opex_per_tonne * capacity

    # Levelized cost (25 yr, 8% discount)
    crf = 0.08 * (1.08) ** 25 / ((1.08) ** 25 - 1)
    annual_capex = total_capex * crf

    levelized_cost = (annual_capex + annual_opex) / (capacity + 1e-10)

    return {
        "total_capex_USD": total_capex,
        "capex_per_capacity_USD": capex_per_capacity,
        "annual_opex_USD": annual_opex,
        "opex_per_tonne_USD": opex_per_tonne,
        "levelized_cost_USD_tonne": levelized_cost,
        "learning_factor": learning_factor,
        "nth_plant": nth_plant,
    }
