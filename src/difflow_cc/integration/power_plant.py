"""Power plant integration for carbon capture.

This module models the integration of post-combustion capture
with coal and natural gas power plants, including:
- Steam extraction for solvent regeneration
- Compression power for CO2
- Auxiliary power for pumps, fans, etc.
- Overall efficiency penalty

References:
    IEAGHG (2019). Towards Zero Emissions CCS.
    NETL (2019). Cost and Performance Baseline.
    Lucquiaud M, Gibbins J (2011). On the integration of CO2
        capture with coal-fired power plants. Chem Eng Res Des.
"""

__all__ = [
    "PowerPlantParams",
    "PowerPlantIntegration",
    "steam_extraction_penalty",
    "compression_penalty",
    "auxiliary_power",
    "net_efficiency_with_capture",
    "flue_gas_composition",
    "flue_gas_flow_rate",
]

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from typing import Literal

import jax.numpy as jnp
from jax import Array


@dataclass(repr=False)
class PowerPlantParams(ParamsMixin):
    """Power plant parameters.

    Attributes:
        plant_type: 'coal_subcritical', 'coal_supercritical', 'coal_usc', 'ngcc'
        gross_power: Gross power output (MW)
        net_efficiency: Net plant efficiency (LHV basis, fraction)
        fuel_carbon_content: Carbon content of fuel (kg C/kg fuel)
        fuel_heating_value: Fuel LHV (MJ/kg)
        flue_gas_CO2_fraction: CO2 mole fraction in flue gas
        flue_gas_temperature: Flue gas temperature at capture inlet (K)
        auxiliary_fraction: Auxiliary power as fraction of gross
    """
    plant_type: str = "coal_supercritical"
    gross_power: float | Array = 500.0  # MW
    net_efficiency: float | Array = 0.40  # LHV basis
    fuel_carbon_content: float = 0.70  # kg C/kg fuel (coal)
    fuel_heating_value: float = 25.0  # MJ/kg (coal)
    flue_gas_CO2_fraction: float = 0.13  # mol/mol
    flue_gas_temperature: float = 323.15  # K (50°C after FGD)
    auxiliary_fraction: float = 0.07  # 7% of gross


# Typical plant specifications
PLANT_SPECS = {
    "coal_subcritical": {
        "net_efficiency": 0.35,
        "flue_gas_CO2": 0.14,
        "steam_conditions": (16.5, 811),  # MPa, K
    },
    "coal_supercritical": {
        "net_efficiency": 0.40,
        "flue_gas_CO2": 0.13,
        "steam_conditions": (24.1, 866),  # MPa, K
    },
    "coal_usc": {
        "net_efficiency": 0.44,
        "flue_gas_CO2": 0.13,
        "steam_conditions": (27.6, 893),  # MPa, K
    },
    "ngcc": {
        "net_efficiency": 0.55,
        "flue_gas_CO2": 0.04,
        "steam_conditions": (10.0, 811),  # MPa, K
    },
}


# =============================================================================
# Flue Gas Calculations
# =============================================================================

def flue_gas_composition(
    plant_type: str = "coal_supercritical",
) -> dict[str, float]:
    """Typical flue gas composition by plant type.

    Args:
        plant_type: Type of power plant

    Returns:
        Dict of component mole fractions
    """
    compositions = {
        "coal_subcritical": {
            "CO2": 0.14,
            "H2O": 0.08,
            "O2": 0.04,
            "N2": 0.74,
            "SO2": 0.0001,  # After FGD
        },
        "coal_supercritical": {
            "CO2": 0.13,
            "H2O": 0.08,
            "O2": 0.04,
            "N2": 0.75,
            "SO2": 0.0001,
        },
        "coal_usc": {
            "CO2": 0.13,
            "H2O": 0.08,
            "O2": 0.04,
            "N2": 0.75,
            "SO2": 0.00005,
        },
        "ngcc": {
            "CO2": 0.04,
            "H2O": 0.08,
            "O2": 0.12,
            "N2": 0.76,
            "SO2": 0.0,
        },
    }
    return compositions.get(plant_type, compositions["coal_supercritical"])


def flue_gas_flow_rate(
    params: PowerPlantParams,
) -> Array:
    """Calculate flue gas flow rate from plant parameters.

    Args:
        params: Power plant parameters

    Returns:
        Flue gas molar flow rate (mol/s)
    """
    gross_power = jnp.asarray(params.gross_power) * 1e6  # W
    efficiency = jnp.asarray(params.net_efficiency)

    # Fuel energy input
    fuel_power = gross_power / efficiency  # W

    # Fuel mass flow
    fuel_flow = fuel_power / (params.fuel_heating_value * 1e6)  # kg/s

    # Carbon flow
    carbon_flow = fuel_flow * params.fuel_carbon_content  # kg C/s

    # CO2 flow (mol/s)
    CO2_flow = carbon_flow / 0.012  # mol C/s = mol CO2/s

    # Total flue gas flow
    y_CO2 = params.flue_gas_CO2_fraction
    total_flow = CO2_flow / y_CO2  # mol/s

    return total_flow


# =============================================================================
# Energy Penalty Calculations
# =============================================================================

def steam_extraction_penalty(
    steam_duty: Array | float,
    extraction_pressure: float = 0.4,  # MPa
    gross_power: Array | float = 500.0,  # MW
    plant_type: str = "coal_supercritical",
) -> Array:
    """Calculate power loss from steam extraction.

    Steam extraction reduces turbine output. The penalty depends
    on extraction pressure - lower pressure = more penalty.

    Penalty = Q_steam * (h_extraction - h_condensate) / η_Carnot_remaining

    Simplified: Penalty ≈ Q_steam * α

    where α depends on extraction point.

    Args:
        steam_duty: Reboiler duty (W)
        extraction_pressure: Steam extraction pressure (MPa)
        gross_power: Gross plant power (MW)
        plant_type: Type of power plant

    Returns:
        Power loss (MW)
    """
    steam_duty = jnp.asarray(steam_duty)
    gross_power = jnp.asarray(gross_power)

    # Equivalent work factor (α)
    # From literature correlations
    # α = work lost per unit heat extracted
    # Typical values: 0.15-0.25 depending on extraction point

    # Lower extraction pressure = higher α (more valuable steam)
    # P_ext = 0.3-0.5 MPa typical for 120-150°C reboiler

    # Correlation from Lucquiaud & Gibbins (2011)
    P_ext = extraction_pressure
    alpha = 0.28 - 0.06 * jnp.log10(P_ext / 0.1 + 0.01)
    alpha = jnp.clip(alpha, 0.15, 0.30)

    # Power loss
    power_loss = steam_duty * alpha / 1e6  # MW

    return power_loss


def compression_penalty(
    compression_power: Array | float,
) -> Array:
    """Power consumption for CO2 compression.

    Args:
        compression_power: Compressor power (W)

    Returns:
        Power penalty (MW)
    """
    compression_power = jnp.asarray(compression_power)
    return compression_power / 1e6  # MW


def auxiliary_power(
    solvent_flow: Array | float = 0.0,
    gas_flow: Array | float = 0.0,
    cooling_duty: Array | float = 0.0,
) -> Array:
    """Auxiliary power for capture plant.

    Includes:
    - Solvent pumps
    - Blowers/fans
    - Cooling water pumps

    Args:
        solvent_flow: Solvent circulation rate (m³/s)
        gas_flow: Flue gas flow rate (m³/s)
        cooling_duty: Cooling water duty (W)

    Returns:
        Auxiliary power (MW)
    """
    solvent_flow = jnp.asarray(solvent_flow)
    gas_flow = jnp.asarray(gas_flow)
    cooling_duty = jnp.asarray(cooling_duty)

    # Solvent pumps (rich pump has larger head)
    # P = ρgHQ/η, assume H=50m, η=0.7
    pump_power = 1000 * 9.81 * 50 * solvent_flow / 0.7 / 1e6  # MW

    # Flue gas blower (ΔP ≈ 10 kPa across absorber)
    blower_power = gas_flow * 10000 / 0.7 / 1e6  # MW

    # Cooling water pumps
    # Estimate from cooling duty
    cw_pump_power = cooling_duty * 0.01 / 1e6  # MW, ~1% of duty

    total = pump_power + blower_power + cw_pump_power
    return total


def net_efficiency_with_capture(
    params: PowerPlantParams,
    steam_penalty: Array | float,
    compression_penalty: Array | float,
    auxiliary_penalty: Array | float,
) -> Array:
    """Calculate net plant efficiency with capture.

    Args:
        params: Power plant parameters
        steam_penalty: Power loss from steam extraction (MW)
        compression_penalty: CO2 compression power (MW)
        auxiliary_penalty: Auxiliary power for capture (MW)

    Returns:
        Net efficiency with capture (fraction)
    """
    gross_power = jnp.asarray(params.gross_power)
    base_efficiency = jnp.asarray(params.net_efficiency)

    # Base net power
    base_net = gross_power * (1 - params.auxiliary_fraction)

    # Total penalty
    total_penalty = steam_penalty + compression_penalty + auxiliary_penalty

    # New net power
    new_net = base_net - total_penalty

    # Fuel input unchanged
    fuel_power = gross_power / base_efficiency

    # New efficiency
    new_efficiency = new_net / fuel_power

    return new_efficiency


def efficiency_penalty(
    params: PowerPlantParams,
    steam_duty: Array | float,
    compression_power: Array | float,
    auxiliary_power_total: Array | float,
) -> dict:
    """Calculate complete efficiency penalty breakdown.

    Args:
        params: Power plant parameters
        steam_duty: Reboiler steam duty (W)
        compression_power: Compression power (W)
        auxiliary_power_total: Total auxiliary power (W)

    Returns:
        Dict with penalty breakdown
    """
    gross = jnp.asarray(params.gross_power)
    base_eff = jnp.asarray(params.net_efficiency)

    # Individual penalties
    steam_pen = steam_extraction_penalty(steam_duty, 0.4, gross)
    comp_pen = compression_penalty(compression_power)
    aux_pen = auxiliary_power_total / 1e6

    total_pen = steam_pen + comp_pen + aux_pen

    # Efficiency with capture
    eff_capture = net_efficiency_with_capture(
        params, steam_pen, comp_pen, aux_pen
    )

    # Points lost
    efficiency_loss = base_eff - eff_capture

    return {
        "steam_penalty_MW": steam_pen,
        "compression_penalty_MW": comp_pen,
        "auxiliary_penalty_MW": aux_pen,
        "total_penalty_MW": total_pen,
        "base_efficiency": base_eff,
        "efficiency_with_capture": eff_capture,
        "efficiency_points_lost": efficiency_loss,
        "relative_penalty": total_pen / gross,
    }


# =============================================================================
# Complete Integration
# =============================================================================

class PowerPlantIntegration:
    """Complete power plant integration model.

    Models the full integration of capture with power plant,
    including all energy penalties.

    Example:
        >>> params = PowerPlantParams(
        ...     plant_type='coal_supercritical',
        ...     gross_power=500.0,
        ... )
        >>> integration = PowerPlantIntegration(params)
        >>> results = integration.analyze(
        ...     steam_duty=150e6,  # 150 MW thermal
        ...     compression_power=30e6,  # 30 MW
        ... )
    """

    def __init__(self, params: PowerPlantParams):
        self.params = params

    def flue_gas_rate(self) -> Array:
        """Get flue gas flow rate."""
        return flue_gas_flow_rate(self.params)

    def flue_gas_comp(self) -> dict:
        """Get flue gas composition."""
        return flue_gas_composition(self.params.plant_type)

    def analyze(
        self,
        steam_duty: Array | float,
        compression_power: Array | float,
        solvent_flow: Array | float = 0.0,
        gas_flow_m3s: Array | float = 0.0,
        cooling_duty: Array | float = 0.0,
    ) -> dict:
        """Analyze complete energy penalty.

        Args:
            steam_duty: Reboiler duty (W)
            compression_power: Compression power (W)
            solvent_flow: Solvent circulation (m³/s)
            gas_flow_m3s: Flue gas volumetric flow (m³/s)
            cooling_duty: Cooling duty (W)

        Returns:
            Complete analysis results
        """
        # Auxiliary power
        aux_power = auxiliary_power(solvent_flow, gas_flow_m3s, cooling_duty)

        # Full penalty analysis
        penalty = efficiency_penalty(
            self.params,
            steam_duty,
            compression_power,
            aux_power * 1e6,
        )

        # Add CO2 rates
        flue_rate = self.flue_gas_rate()
        y_CO2 = self.params.flue_gas_CO2_fraction
        CO2_rate = flue_rate * y_CO2  # mol/s
        CO2_mass = CO2_rate * 44.0 / 1000  # kg/s

        penalty["flue_gas_flow_mol_s"] = flue_rate
        penalty["CO2_in_flue_gas_mol_s"] = CO2_rate
        penalty["CO2_in_flue_gas_kg_s"] = CO2_mass

        return penalty

    def specific_energy(
        self,
        steam_duty: Array | float,
        compression_power: Array | float,
        capture_rate: Array | float = 0.90,
    ) -> Array:
        """Calculate specific energy penalty.

        Args:
            steam_duty: Reboiler duty (W)
            compression_power: Compression power (W)
            capture_rate: CO2 capture efficiency (fraction)

        Returns:
            Specific energy (GJ/tonne CO2)
        """
        steam_duty = jnp.asarray(steam_duty)
        compression_power = jnp.asarray(compression_power)
        capture_rate = jnp.asarray(capture_rate)

        # CO2 captured
        CO2_rate = self.flue_gas_rate() * self.params.flue_gas_CO2_fraction
        CO2_captured = CO2_rate * capture_rate * 44.0 / 1000  # kg/s

        # Total equivalent energy (steam + compression)
        # Steam: direct thermal
        # Compression: multiply by 2.5 for primary energy equivalent
        total_energy = steam_duty + compression_power * 2.5  # W

        # Specific (GJ/tonne)
        specific = safe_divide(total_energy, CO2_captured * 1000)  # J/kg = GJ/tonne

        return specific
