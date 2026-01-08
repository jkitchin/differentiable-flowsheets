"""Amine solvent degradation models.

Amine degradation is a major operational concern in CO2 capture:
- Increases solvent makeup costs
- Produces corrosive and volatile products
- Reduces capture efficiency over time

Degradation pathways:
1. Oxidative degradation (O2 in flue gas)
2. Thermal degradation (high stripper temperatures)
3. CO2-induced degradation (carbamate polymerization)

References:
    Lepaumier H et al. (2009). Degradation of MDEA solutions
        under various conditions. Energy Procedia 1:461-468.
    Sexton AJ, Rochelle GT (2011). Reaction products from the
        oxidative degradation of monoethanolamine.
        Ind Eng Chem Res 50:667-673.
    Davis J, Rochelle G (2009). Thermal degradation of
        monoethanolamine at stripper conditions.
        Energy Procedia 1:327-333.
"""

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin

import jax.numpy as jnp
from jax import Array


R = 8.314  # J/(mol·K)


@dataclass
class AmineDegradationParams(ParamsMixin):
    """Parameters for amine degradation modeling.

    Attributes:
        solvent: Amine type ('MEA', 'DEA', 'MDEA', 'PZ', 'AMP')
        concentration: Amine concentration (wt%)
        T_absorber: Absorber temperature (K)
        T_stripper: Stripper temperature (K)
        O2_concentration: O2 in flue gas (mol/mol)
        CO2_loading: Average CO2 loading (mol/mol)
        Fe_concentration: Iron ion concentration (ppm)
        Cu_concentration: Copper ion concentration (ppm)
        operating_hours: Annual operating hours
    """
    solvent: str = "MEA"
    concentration: float = 30.0  # wt%
    T_absorber: float = 313.15  # K (40°C)
    T_stripper: float = 393.15  # K (120°C)
    O2_concentration: float = 0.05  # 5% O2
    CO2_loading: float = 0.35  # mol/mol
    Fe_concentration: float = 1.0  # ppm
    Cu_concentration: float = 0.0  # ppm
    operating_hours: float = 8000.0  # hr/yr


# Kinetic parameters for degradation (Arrhenius form)
# k = A * exp(-Ea/RT)
DEGRADATION_KINETICS = {
    "MEA": {
        "oxidative": {"A": 1.0e8, "Ea": 65000},  # 1/s, J/mol
        "thermal": {"A": 1.0e12, "Ea": 130000},
        "CO2_induced": {"A": 1.0e6, "Ea": 80000},
    },
    "DEA": {
        "oxidative": {"A": 5.0e7, "Ea": 60000},
        "thermal": {"A": 5.0e11, "Ea": 125000},
        "CO2_induced": {"A": 5.0e5, "Ea": 75000},
    },
    "MDEA": {
        "oxidative": {"A": 1.0e7, "Ea": 55000},  # More stable
        "thermal": {"A": 1.0e10, "Ea": 140000},  # Very stable thermally
        "CO2_induced": {"A": 1.0e4, "Ea": 70000},  # Low (no carbamate)
    },
    "PZ": {
        "oxidative": {"A": 2.0e7, "Ea": 58000},
        "thermal": {"A": 2.0e13, "Ea": 145000},  # Very stable
        "CO2_induced": {"A": 1.0e5, "Ea": 85000},
    },
    "AMP": {
        "oxidative": {"A": 3.0e7, "Ea": 62000},
        "thermal": {"A": 5.0e11, "Ea": 135000},
        "CO2_induced": {"A": 2.0e4, "Ea": 72000},  # Low (hindered)
    },
}


def oxidative_degradation_rate(
    T: Array | float,
    params: AmineDegradationParams,
) -> Array:
    """Calculate oxidative degradation rate.

    Oxidative degradation occurs in the absorber due to O2
    in the flue gas. Metal ions (Fe, Cu) catalyze the reaction.

    Rate = k_ox * [Amine] * [O2] * (1 + k_Fe*[Fe] + k_Cu*[Cu])

    Args:
        T: Temperature (K)
        params: Degradation parameters

    Returns:
        Degradation rate (mol amine / L / s)
    """
    T = jnp.asarray(T)

    # Get kinetics
    kinetics = DEGRADATION_KINETICS.get(params.solvent, DEGRADATION_KINETICS["MEA"])
    A = kinetics["oxidative"]["A"]
    Ea = kinetics["oxidative"]["Ea"]

    # Rate constant
    k = A * jnp.exp(-Ea / (R * T))

    # Amine concentration (approximate mol/L from wt%)
    # Assume density ~1000 kg/m³, MW_MEA = 61 g/mol
    MW = {"MEA": 61, "DEA": 105, "MDEA": 119, "PZ": 86, "AMP": 89}
    mw = MW.get(params.solvent, 61)
    C_amine = params.concentration / 100 * 1000 / mw  # mol/L

    # O2 concentration
    C_O2 = params.O2_concentration

    # Metal ion catalysis factor
    k_Fe = 10.0  # Catalytic factor for Fe
    k_Cu = 100.0  # Cu is more active
    catalyst_factor = 1.0 + k_Fe * params.Fe_concentration / 1e6 + \
                           k_Cu * params.Cu_concentration / 1e6

    # Degradation rate
    rate = k * C_amine * C_O2 * catalyst_factor  # mol/L/s

    return rate


def thermal_degradation_rate(
    T: Array | float,
    params: AmineDegradationParams,
) -> Array:
    """Calculate thermal degradation rate.

    Thermal degradation occurs in the stripper at high
    temperatures, especially with high CO2 loading.

    For MEA, the main pathway is carbamate polymerization:
    2 MEA + CO2 → HEEDA + H2O

    Args:
        T: Temperature (K)
        params: Degradation parameters

    Returns:
        Degradation rate (mol amine / L / s)
    """
    T = jnp.asarray(T)

    kinetics = DEGRADATION_KINETICS.get(params.solvent, DEGRADATION_KINETICS["MEA"])
    A = kinetics["thermal"]["A"]
    Ea = kinetics["thermal"]["Ea"]

    k = A * jnp.exp(-Ea / (R * T))

    MW = {"MEA": 61, "DEA": 105, "MDEA": 119, "PZ": 86, "AMP": 89}
    mw = MW.get(params.solvent, 61)
    C_amine = params.concentration / 100 * 1000 / mw

    # Loading effect (higher loading = more degradation for primary amines)
    loading_factor = 1.0 + 2.0 * params.CO2_loading

    rate = k * C_amine ** 2 * loading_factor  # Second order in amine

    return rate


def co2_induced_degradation_rate(
    T: Array | float,
    params: AmineDegradationParams,
) -> Array:
    """Calculate CO2-induced degradation rate.

    CO2 reacts with carbamate to form degradation products.
    More significant for primary/secondary amines.

    Args:
        T: Temperature (K)
        params: Degradation parameters

    Returns:
        Degradation rate (mol amine / L / s)
    """
    T = jnp.asarray(T)

    kinetics = DEGRADATION_KINETICS.get(params.solvent, DEGRADATION_KINETICS["MEA"])
    A = kinetics["CO2_induced"]["A"]
    Ea = kinetics["CO2_induced"]["Ea"]

    k = A * jnp.exp(-Ea / (R * T))

    MW = {"MEA": 61, "DEA": 105, "MDEA": 119, "PZ": 86, "AMP": 89}
    mw = MW.get(params.solvent, 61)
    C_amine = params.concentration / 100 * 1000 / mw

    # Loading directly affects this pathway
    rate = k * C_amine * params.CO2_loading

    return rate


def total_amine_loss(
    params: AmineDegradationParams,
) -> dict:
    """Calculate total annual amine loss from all pathways.

    Args:
        params: Degradation parameters

    Returns:
        Dict with loss breakdown (kg/yr for 1000 m³ inventory)
    """
    # Rates at operating temperatures
    r_ox = oxidative_degradation_rate(params.T_absorber, params)
    r_th = thermal_degradation_rate(params.T_stripper, params)
    r_co2 = co2_induced_degradation_rate(params.T_stripper, params)

    # Convert to annual loss per m³ solvent
    # rate (mol/L/s) * 1000 L/m³ * 3600 s/hr * operating_hours
    MW = {"MEA": 61, "DEA": 105, "MDEA": 119, "PZ": 86, "AMP": 89}
    mw = MW.get(params.solvent, 61)

    # Assume time split: 70% in absorber, 30% in stripper
    t_abs = 0.7 * params.operating_hours * 3600  # seconds
    t_str = 0.3 * params.operating_hours * 3600

    loss_ox = float(r_ox) * 1000 * t_abs * mw / 1000  # kg per m³ inventory
    loss_th = float(r_th) * 1000 * t_str * mw / 1000
    loss_co2 = float(r_co2) * 1000 * t_str * mw / 1000

    total = loss_ox + loss_th + loss_co2

    return {
        "oxidative_kg_m3_yr": loss_ox,
        "thermal_kg_m3_yr": loss_th,
        "CO2_induced_kg_m3_yr": loss_co2,
        "total_kg_m3_yr": total,
        "total_fraction_yr": total / (params.concentration / 100 * 1000),
    }


def degradation_products(
    params: AmineDegradationParams,
) -> dict:
    """Estimate degradation product formation.

    Main products by solvent:
    - MEA: HEEDA, HEIA, OZD, formate, acetate, ammonia
    - DEA: THEED, BHEP
    - MDEA: More stable, less products

    Args:
        params: Degradation parameters

    Returns:
        Dict of product concentrations (mol/L after 1000 hr)
    """
    # Simplified product distribution
    loss = total_amine_loss(params)

    # Product yields (mol product / mol amine degraded)
    if params.solvent == "MEA":
        products = {
            "HEEDA": 0.3,  # Hydroxyethyl ethylenediamine
            "HEIA": 0.1,  # Hydroxyethyl imidazolidone
            "OZD": 0.1,  # Oxazolidone
            "formate": 0.2,
            "acetate": 0.15,
            "ammonia": 0.15,
        }
    elif params.solvent == "DEA":
        products = {
            "THEED": 0.4,
            "BHEP": 0.2,
            "formate": 0.2,
            "ammonia": 0.2,
        }
    else:
        products = {
            "formate": 0.5,
            "ammonia": 0.3,
            "others": 0.2,
        }

    MW = {"MEA": 61, "DEA": 105, "MDEA": 119, "PZ": 86, "AMP": 89}
    mw = MW.get(params.solvent, 61)

    # Convert to concentrations
    mol_degraded = loss["total_kg_m3_yr"] * 1000 / mw  # mol/m³/yr = mmol/L/yr
    mol_degraded /= 1000  # mol/L/yr

    result = {}
    for product, yield_frac in products.items():
        result[product] = mol_degraded * yield_frac  # mol/L/yr

    return result


def solvent_lifetime(
    params: AmineDegradationParams,
    max_degradation: float = 0.20,  # 20% capacity loss
) -> Array:
    """Estimate solvent lifetime before major reclamation.

    Args:
        params: Degradation parameters
        max_degradation: Maximum acceptable degradation fraction

    Returns:
        Estimated lifetime (years)
    """
    loss = total_amine_loss(params)
    annual_fraction = loss["total_fraction_yr"]

    lifetime = max_degradation / (annual_fraction + 1e-10)
    lifetime = jnp.clip(lifetime, 0.5, 10.0)

    return jnp.asarray(lifetime)


def reclaimer_requirements(
    params: AmineDegradationParams,
    solvent_inventory: float = 100.0,  # m³
) -> dict:
    """Calculate thermal reclaimer requirements.

    Thermal reclaiming removes heat-stable salts and
    high-boiling degradation products.

    Args:
        params: Degradation parameters
        solvent_inventory: Total solvent volume (m³)

    Returns:
        Reclaimer specifications
    """
    loss = total_amine_loss(params)

    # Reclaimer typically processes 1-3% of circulation per day
    # to maintain steady-state degradation product concentration

    degradation_kg_yr = loss["total_kg_m3_yr"] * solvent_inventory

    # Reclaimer removes ~95% of products
    reclaimer_efficiency = 0.95

    # Reclaimer feed to remove all products
    feed_kg_yr = degradation_kg_yr / reclaimer_efficiency

    # Typical reclaimer processes 0.5-1% of inventory per day
    # Calculate minimum reclaimer capacity
    daily_rate = feed_kg_yr / 365  # kg/day

    # Energy requirement (~1000 kJ/kg evaporated)
    energy_per_kg = 1000  # kJ/kg
    thermal_duty = daily_rate * energy_per_kg / (24 * 3600) * 1000  # W

    return {
        "degradation_products_kg_yr": degradation_kg_yr,
        "reclaimer_feed_kg_yr": feed_kg_yr,
        "reclaimer_daily_rate_kg_day": daily_rate,
        "reclaimer_thermal_duty_W": thermal_duty,
        "waste_generated_kg_yr": degradation_kg_yr * 1.5,  # With water
        "makeup_required_kg_yr": degradation_kg_yr,
    }
