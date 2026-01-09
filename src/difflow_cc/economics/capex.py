"""Capital cost correlations for carbon capture equipment.

Equipment cost correlations are based on literature data and
scaled using the six-tenths rule where applicable.

All costs are in USD (2020 basis). Use CEPCI indices for
year-to-year adjustments.

References:
    Rubin ES et al. (2015). The cost of CO2 capture and storage.
        Int J Greenh Gas Control 40:378-400.
    NETL (2019). Cost and Performance Baseline for Fossil Energy
        Plants Volume 1: Bituminous Coal and Natural Gas.
    Towler G, Sinnott R (2013). Chemical Engineering Design,
        2nd ed. Butterworth-Heinemann.
"""

from dataclasses import dataclass
from difflow.params_mixin import ParamsMixin
from typing import Literal

import jax.numpy as jnp
from jax import Array


# Cost index for scaling (CEPCI)
CEPCI_2020 = 596.2
CEPCI_2015 = 556.8
CEPCI_2010 = 550.8


@dataclass(repr=False)
class CapexParams(ParamsMixin):
    """Parameters for capital cost estimation.

    Attributes:
        cepci: Chemical Engineering Plant Cost Index
        location_factor: Geographic location multiplier
        contingency: Contingency factor (fraction)
        owner_cost: Owner's cost (fraction of TPC)
    """
    cepci: float = CEPCI_2020
    location_factor: float = 1.0  # US Gulf Coast = 1.0
    contingency: float = 0.15  # 15%
    owner_cost: float = 0.05  # 5%


# =============================================================================
# Equipment Cost Correlations
# =============================================================================

def absorber_cost(
    diameter: Array | float,
    height: Array | float,
    material: str = "carbon_steel",
    packing: str = "structured",
) -> Array:
    """Absorber column capital cost.

    Based on column shell + packing/internals.

    Args:
        diameter: Column diameter (m)
        height: Packing height (m)
        material: Construction material
        packing: Packing type ('structured', 'random', 'trays')

    Returns:
        Equipment cost (USD)
    """
    diameter = jnp.asarray(diameter)
    height = jnp.asarray(height)

    # Column volume
    volume = jnp.pi * (diameter / 2) ** 2 * height

    # Base cost correlation (from Towler & Sinnott)
    # Shell cost: C = a + b * S^n where S is surface area
    surface_area = jnp.pi * diameter * height  # m²

    # Material factors
    material_factors = {
        "carbon_steel": 1.0,
        "stainless_304": 1.8,
        "stainless_316": 2.1,
        "alloy_20": 3.5,
    }
    fm = material_factors.get(material, 1.0)

    # Shell cost (2020 USD)
    shell_cost = fm * (17400 + 79 * surface_area ** 0.85)

    # Packing cost
    packing_costs = {
        "structured": 3000,  # $/m³ (Mellapak-type)
        "random": 800,  # $/m³ (Pall rings)
        "trays": 1500,  # $/m² (sieve trays)
    }
    packing_unit_cost = packing_costs.get(packing, 2000)

    if packing == "trays":
        n_trays = height / 0.6  # Assume 0.6 m tray spacing
        tray_area = jnp.pi * (diameter / 2) ** 2
        packing_cost = packing_unit_cost * n_trays * tray_area
    else:
        packing_cost = packing_unit_cost * volume

    total = shell_cost + packing_cost
    return total


def stripper_cost(
    diameter: Array | float,
    height: Array | float,
    reboiler_duty: Array | float,
    condenser_duty: Array | float,
    material: str = "stainless_304",
) -> Array:
    """Stripper column with reboiler and condenser.

    Args:
        diameter: Column diameter (m)
        height: Packing height (m)
        reboiler_duty: Reboiler heat duty (W)
        condenser_duty: Condenser heat duty (W)
        material: Construction material

    Returns:
        Equipment cost (USD)
    """
    diameter = jnp.asarray(diameter)
    height = jnp.asarray(height)
    reboiler_duty = jnp.asarray(reboiler_duty)
    condenser_duty = jnp.asarray(condenser_duty)

    # Column cost (similar to absorber but SS for corrosion)
    column_cost = absorber_cost(diameter, height, material, "structured")

    # Reboiler cost (kettle type)
    # Area based on U = 1000 W/m²/K, LMTD = 30 K
    U_reb = 1000.0
    LMTD_reb = 30.0
    A_reb = reboiler_duty / (U_reb * LMTD_reb)
    reboiler_cost = 15000 * jnp.power(A_reb, 0.65)

    # Condenser cost (shell and tube)
    U_cond = 800.0
    LMTD_cond = 20.0
    A_cond = condenser_duty / (U_cond * LMTD_cond)
    condenser_cost = 10000 * jnp.power(A_cond, 0.65)

    total = column_cost + reboiler_cost + condenser_cost
    return total


def heat_exchanger_cost(
    area: Array | float,
    hx_type: str = "shell_tube",
    material: str = "carbon_steel",
    pressure: float = 1000000.0,  # Pa
) -> Array:
    """Heat exchanger capital cost.

    Args:
        area: Heat transfer area (m²)
        hx_type: 'shell_tube', 'plate', 'spiral'
        material: Construction material
        pressure: Design pressure (Pa)

    Returns:
        Equipment cost (USD)
    """
    area = jnp.asarray(area)

    # Base cost by type
    base_costs = {
        "shell_tube": (10000, 88, 0.68),  # (a, b, n) in C = a + b*A^n
        "plate": (5000, 150, 0.60),
        "spiral": (12000, 95, 0.70),
    }
    a, b, n = base_costs.get(hx_type, (10000, 88, 0.68))

    # Material factor
    material_factors = {
        "carbon_steel": 1.0,
        "stainless_304": 1.7,
        "stainless_316": 2.0,
        "titanium": 4.5,
    }
    fm = material_factors.get(material, 1.0)

    # Pressure factor (for P > 10 bar)
    P_barg = pressure / 100000 - 1
    fp = jnp.where(P_barg > 10, 0.9 + 0.1 * P_barg / 10, 1.0)

    cost = fm * fp * (a + b * jnp.power(area, n))
    return cost


def compressor_cost(
    power: Array | float,
    comp_type: str = "centrifugal",
    driver: str = "electric",
) -> Array:
    """Compressor capital cost.

    Args:
        power: Shaft power (W)
        comp_type: 'centrifugal', 'reciprocating', 'screw'
        driver: 'electric', 'steam_turbine', 'gas_turbine'

    Returns:
        Equipment cost (USD)
    """
    power = jnp.asarray(power)
    power_kW = power / 1000

    # Base cost by type
    type_factors = {
        "centrifugal": 1.0,
        "reciprocating": 1.3,
        "screw": 0.9,
    }
    ft = type_factors.get(comp_type, 1.0)

    # Driver cost multiplier
    driver_factors = {
        "electric": 1.0,
        "steam_turbine": 1.4,
        "gas_turbine": 1.6,
    }
    fd = driver_factors.get(driver, 1.0)

    # Cost correlation
    # C = 580000 * (kW / 1000)^0.82 for centrifugal
    cost = ft * fd * 580000 * jnp.power(power_kW / 1000 + 0.1, 0.82)

    return cost


def membrane_module_cost(
    area: Array | float,
    membrane_type: str = "polymeric",
) -> Array:
    """Membrane module capital cost.

    Args:
        area: Membrane area (m²)
        membrane_type: 'polymeric', 'ceramic', 'facilitated'

    Returns:
        Equipment cost (USD)
    """
    area = jnp.asarray(area)

    # Cost per unit area by type
    area_costs = {
        "polymeric": 50,  # $/m²
        "ceramic": 500,  # $/m²
        "facilitated": 200,  # $/m²
        "mixed_matrix": 100,  # $/m²
    }
    cost_per_m2 = area_costs.get(membrane_type, 100)

    # Module housing adds ~30%
    module_factor = 1.3

    cost = module_factor * cost_per_m2 * area
    return cost


def adsorber_vessel_cost(
    diameter: Array | float,
    length: Array | float,
    n_beds: int = 2,
    adsorbent_mass: Array | float = 1000.0,
    adsorbent_cost: float = 5.0,  # $/kg
) -> Array:
    """Adsorption vessel capital cost.

    Args:
        diameter: Vessel diameter (m)
        length: Vessel length (m)
        n_beds: Number of parallel beds
        adsorbent_mass: Mass of adsorbent per bed (kg)
        adsorbent_cost: Adsorbent cost ($/kg)

    Returns:
        Equipment cost (USD)
    """
    diameter = jnp.asarray(diameter)
    length = jnp.asarray(length)
    adsorbent_mass = jnp.asarray(adsorbent_mass)

    # Pressure vessel cost
    # Based on weight correlation
    thickness = 0.01 * diameter  # Approximate wall thickness
    weight = jnp.pi * diameter * length * thickness * 7800  # kg steel

    vessel_cost = 17000 * jnp.power(weight / 1000, 0.62)

    # Internals (bed supports, distributors)
    internals_cost = 0.15 * vessel_cost

    # Adsorbent initial charge
    adsorbent_cost_total = adsorbent_mass * adsorbent_cost

    # Total for n beds
    cost = n_beds * (vessel_cost + internals_cost + adsorbent_cost_total)
    return cost


def blower_cost(
    flow_rate: Array | float,
    pressure_rise: Array | float,
) -> Array:
    """Blower/fan capital cost for DAC or low-pressure applications.

    Args:
        flow_rate: Volumetric flow (m³/s)
        pressure_rise: Pressure rise (Pa)

    Returns:
        Equipment cost (USD)
    """
    flow_rate = jnp.asarray(flow_rate)
    pressure_rise = jnp.asarray(pressure_rise)

    # Power requirement (with efficiency)
    eta_blower = 0.7
    power = flow_rate * pressure_rise / eta_blower  # W

    # Cost correlation
    cost = 15000 * jnp.power(power / 10000 + 0.1, 0.6)
    return cost


# =============================================================================
# Aggregation Functions
# =============================================================================

def total_equipment_cost(
    equipment_costs: dict[str, Array | float],
) -> Array:
    """Sum all equipment costs.

    Args:
        equipment_costs: Dict of equipment name to cost

    Returns:
        Total equipment cost (USD)
    """
    total = jnp.asarray(0.0)
    for name, cost in equipment_costs.items():
        total = total + jnp.asarray(cost)
    return total


def installed_cost(
    equipment_cost: Array | float,
    params: CapexParams | None = None,
) -> dict:
    """Calculate total installed cost from equipment cost.

    Uses factorial method with installation factors.

    Args:
        equipment_cost: Purchased equipment cost (USD)
        params: Cost parameters

    Returns:
        Dict with cost breakdown
    """
    if params is None:
        params = CapexParams()

    equipment_cost = jnp.asarray(equipment_cost)

    # Installation factors (Lang-type)
    factors = {
        "equipment": 1.0,
        "installation": 0.45,
        "instrumentation": 0.15,
        "piping": 0.40,
        "electrical": 0.10,
        "buildings": 0.10,
        "site_prep": 0.05,
        "service_facilities": 0.15,
    }

    # Direct costs
    direct_cost = equipment_cost * sum(factors.values())

    # Indirect costs
    engineering = 0.15 * direct_cost
    construction = 0.10 * direct_cost
    indirect_cost = engineering + construction

    # Total plant cost before contingency
    bare_erected_cost = direct_cost + indirect_cost

    # Apply factors
    contingency = params.contingency * bare_erected_cost
    total_plant_cost = bare_erected_cost + contingency

    # Location adjustment
    location_adjusted = total_plant_cost * params.location_factor

    # Owner's costs
    owner_cost = params.owner_cost * location_adjusted
    total_overnight_cost = location_adjusted + owner_cost

    return {
        "equipment_cost": equipment_cost,
        "direct_cost": direct_cost,
        "indirect_cost": indirect_cost,
        "bare_erected_cost": bare_erected_cost,
        "contingency": contingency,
        "total_plant_cost": total_plant_cost,
        "location_factor": params.location_factor,
        "owner_cost": owner_cost,
        "total_overnight_cost": total_overnight_cost,
    }


def annualized_capital_cost(
    total_capital: Array | float,
    discount_rate: float = 0.08,
    lifetime: int = 25,
) -> Array:
    """Convert capital cost to annual payment.

    Uses capital recovery factor (CRF).

    Args:
        total_capital: Total overnight cost (USD)
        discount_rate: Discount rate (fraction)
        lifetime: Project lifetime (years)

    Returns:
        Annualized capital cost (USD/yr)
    """
    total_capital = jnp.asarray(total_capital)
    r = discount_rate
    n = lifetime

    # Capital recovery factor
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)

    return total_capital * crf
