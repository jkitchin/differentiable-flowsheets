"""Capital cost correlations for carbon capture equipment.

Column shells and packing, adsorber vessels, heat exchangers, reboilers
and compressors use the sourced purchased-cost correlations of
:mod:`difflow.economics.capital` (Towler & Sinnott; Turton et al.; Seider
et al.). A unit larger than its correlation's range is costed as equal units
in parallel. Blowers are Seider et al.'s fans, material factors Towler &
Sinnott's, driver factors Seider et al.'s, and the polymeric membrane price
Zhai & Rubin's. Values with no cited source are marked as such where they
are defined.

All costs are in USD (2020 basis). Use CEPCI indices for
year-to-year adjustments.

References:
    Rubin ES et al. (2015). The cost of CO2 capture and storage.
        Int J Greenh Gas Control 40:378-400.
    NETL (2019). Cost and Performance Baseline for Fossil Energy
        Plants Volume 1: Bituminous Coal and Natural Gas.
    Towler G, Sinnott R (2013). Chemical Engineering Design,
        2nd ed. Butterworth-Heinemann.
    Towler G, Sinnott R (2022). Chemical Engineering Design, 3rd ed.
        Elsevier (material cost factors).
    Seider WD et al. (2016). Product and Process Design Principles, 4th ed.
        Wiley, Sec. 16.5 (fans, compressor drivers, packing).
    Zhai H, Rubin ES (2019). Advanced membranes and learning scale required
        for cost-effective post-combustion carbon capture. iScience.
        doi:10.1016/j.isci.2019.03.006.
"""

__all__ = [
    "absorber_cost",
    "stripper_cost",
    "heat_exchanger_cost",
    "compressor_cost",
    "membrane_module_cost",
    "adsorber_vessel_cost",
    "MATERIAL_FACTORS",
    "DRIVER_FACTORS",
    "MEMBRANE_PRICES",
    "total_equipment_cost",
    "installed_cost",
    "CapexParams",
    "SCALING_EXPONENTS",
    "VALID_EXPONENT_RANGE",
    "validate_scaling_exponent",
]

from dataclasses import dataclass
from difflow.economics.capital import (
    COLUMN_COSTS,
    COMPRESSOR_COSTS,
    HEAT_EXCHANGER_COSTS,
    CostParams,
    _check_size_range,
    equipment_cost,
)
from difflow.params_mixin import ParamsMixin
from typing import Literal

import jax.numpy as jnp
from jax import Array


# Cost index for scaling (CEPCI)
CEPCI_2020 = 596.2
CEPCI_2015 = 556.8
CEPCI_2010 = 550.8

# Cost basis year of this module's results.
_BASIS_YEAR = 2020

# Towler & Sinnott 2e Table 7.2, vertical carbon-steel pressure vessel, with S
# the shell mass in kg (openpytea: vertical_cs_press_vessel_towler_2010).
_VERTICAL_VESSEL = CostParams(
    a=11600.0, b=34.0, n=0.85, S_min=160.0, S_max=250000.0,
    S_units="kg (shell mass)", base_year=2010,
    source="Towler & Sinnott 2e Table 7.2: vertical carbon-steel pressure vessel",
)
_STEEL_DENSITY = 7850.0  # kg/m³, carbon steel

# Materials cost factors relative to carbon steel, Towler & Sinnott 3e (2022),
# as transcribed by openpytea 3.2.1. alloy_20 and titanium are not in that
# table; their factors have no cited source.
MATERIAL_FACTORS = {
    "carbon_steel": 1.0,
    "aluminum": 1.07,
    "stainless_304": 1.3,
    "stainless_316": 1.3,
    "stainless_321": 1.5,
    "hastelloy_c": 1.55,
    "monel": 1.65,
    "nickel": 1.7,
    "inconel": 1.7,
    "alloy_20": 3.5,   # no cited source
    "titanium": 4.5,   # no cited source
}

# Compressor driver factors relative to an electric motor: Seider et al. 4e
# Sec. 16.5 prices each compressor with an electric motor, steam turbine or gas
# turbine drive; the turbine rows are the motor row times these factors.
DRIVER_FACTORS = {
    "electric": 1.0,
    "steam_turbine": 1.15,
    "gas_turbine": 1.25,
}

# Packing and trays: entries of difflow.economics.capital.COLUMN_COSTS.
# "random" is Turton's loose packing (purchased cost, per m³); "structured" is
# Seider's corrugated-sheet stainless packing, an INSTALLED cost per m³ that
# Seider calls very approximate; "trays" is Turton's sieve tray, per tray on
# its area (m²).
_PACKING_TYPES = {
    "structured": "packing_structured",
    "random": "packing_random",
    "trays": "sieve_tray",
}

# Seider et al. 4e Sec. 16.5, centrifugal backward-curved fan with an electric
# motor, 2013: ln C = 11.4152 - 1.3805 ln Q + 0.1139 (ln Q)**2, Q in ACFM
# (1000-100,000), for a head up to 4 in H2O; written here in log10 form. A
# higher head multiplies it by _FAN_HEAD_FACTORS (openpytea 3.2.1 rows).
_LN10 = 2.302585092994046
_FAN_BACKWARD_CURVED = CostParams(
    a=11.4152 / _LN10, b=-1.3805, n=0.1139 * _LN10, S_min=1000.0, S_max=100000.0,
    S_units="ACFM", base_year=2013, form="log_quadratic",
    source="Seider et al. 4e Sec. 16.5: centrifugal backward-curved fan, with motor",
)
_M3S_TO_ACFM = 2118.880003
_PA_PER_IN_H2O = 249.08891
# Head factor by head (in H2O): 1.0 up to 4, 1.15 for 5-8, 1.30 for 9-15,
# 1.45 for 16-30, 1.55 for 31-40. Interpolated linearly across the gaps
# between Seider's bins (4-5, 8-9, 15-16, 30-31) so the cost stays continuous.
_FAN_HEAD_KNOTS = (4.0, 5.0, 8.0, 9.0, 15.0, 16.0, 30.0, 31.0)
_FAN_HEAD_FACTORS = (1.0, 1.15, 1.15, 1.30, 1.30, 1.45, 1.45, 1.55)
# Only S_max is checked (a head below 4 in H2O takes the base cost).
_FAN_HEAD_RANGE = CostParams(
    a=0.0, b=0.0, n=0.0, S_min=0.0, S_max=40.0, S_units="in H2O (fan head)",
    base_year=2013, source="Seider et al. 4e Sec. 16.5: fan head factors",
)

# Installed membrane price ($/m²). Polymeric: Zhai & Rubin (2019) -- installed
# module prices of $27-80/m² are widely assumed for a mature (nth-of-a-kind)
# plant, with $50/m² the usual value; they take $200/m² for a first-of-a-kind
# plant, and commercial gas-separation membranes cost up to several hundred
# $/m². The other types have no cited source.
MEMBRANE_PRICES = {
    "polymeric": 50.0,
    "ceramic": 650.0,       # no cited source
    "facilitated": 260.0,   # no cited source
    "mixed_matrix": 130.0,  # no cited source
}


def _lookup(table: dict, key: str, what: str):
    """``table[key]``, or ValueError naming the valid keys."""
    if key not in table:
        raise ValueError(f"Unknown {what} {key!r}. Available: {sorted(table)}")
    return table[key]


# hx_type -> entry of difflow.economics.capital.HEAT_EXCHANGER_COSTS
_HX_TYPES = {
    "shell_tube": "shell_tube_floating",
    "plate": "plate_frame",
    "spiral": "spiral_plate",
}


def _shell_mass(diameter: Array, length: Array, thickness: Array) -> Array:
    """Steel mass (kg) of a cylindrical shell and two heads, pi rho D (L + 0.8 D) t."""
    return jnp.pi * _STEEL_DENSITY * diameter * (length + 0.8 * diameter) * thickness


def _parallel_units_cost(size: Array, params: CostParams) -> Array:
    """Purchased cost (2020 USD) of ``size``, split into parallel units above ``S_max``.

    The unit count ``size / S_max`` is not rounded up, so the cost stays
    differentiable; above ``S_max`` it is linear in ``size``. A zero size
    costs nothing.
    """
    present = size > 0
    safe = jnp.where(present, size, params.S_max)  # keeps 0/0 out of the gradient
    per_unit = jnp.minimum(safe, params.S_max)
    cost = safe / per_unit * equipment_cost(per_unit, params, target_year=_BASIS_YEAR)
    return jnp.where(present, cost, 0.0)


# Scaling exponents of the correlations below: ``n`` of a Towler & Sinnott
# power law, or the local exponent at 100 m² of a Turton log-quadratic (which
# varies with size). Above a correlation's range, parallel units make the cost
# linear (exponent 1). Exposed so callers can validate them, mirroring
# difflow.economics.capital.validate_cost_params.
SCALING_EXPONENTS = {
    "absorber": 0.85,            # column shell vs. shell mass
    "stripper_reboiler": 0.9,    # kettle reboiler vs. area
    "stripper_condenser": 0.42,  # floating-head exchanger vs. area, at 100 m²
    "heat_exchanger_shell_tube": 0.42,  # floating head, at 100 m²
    "heat_exchanger_plate": 0.46,       # flat plate, at 100 m²
    "compressor": 0.6,           # centrifugal compressor vs. driver power
    "membrane_module": 1.0,      # linear in area (modular)
    "adsorber_vessel": 0.85,     # vessel vs. shell mass
    "blower": 0.72,              # backward-curved fan vs. flow, at 10,000 ACFM
}

# Typical valid range for equipment cost-scaling exponents (matches the core
# economics module, difflow.economics.capital.VALID_EXPONENT_RANGE).
VALID_EXPONENT_RANGE = (0.3, 1.2)


def validate_scaling_exponent(equipment_type: str) -> dict:
    """Validate the power-law scaling exponent for an equipment type.

    Analogue of ``difflow.economics.capital.validate_cost_params`` for the
    carbon-capture CAPEX correlations, which embed type-specific exponents
    rather than a single six-tenths default.

    Args:
        equipment_type: Key into :data:`SCALING_EXPONENTS` (e.g. ``"compressor"``,
            ``"heat_exchanger_shell_tube"``).

    Returns:
        Dict with ``equipment_type``, ``exponent``, ``exponent_valid`` (whether
        the exponent is within :data:`VALID_EXPONENT_RANGE`), and
        ``valid_range``.

    Raises:
        KeyError: If ``equipment_type`` is not a known correlation.
    """
    if equipment_type not in SCALING_EXPONENTS:
        raise KeyError(
            f"Unknown equipment_type {equipment_type!r}. "
            f"Available: {sorted(SCALING_EXPONENTS)}"
        )
    n = SCALING_EXPONENTS[equipment_type]
    return {
        "equipment_type": equipment_type,
        "exponent": n,
        "exponent_valid": VALID_EXPONENT_RANGE[0] <= n <= VALID_EXPONENT_RANGE[1],
        "valid_range": VALID_EXPONENT_RANGE,
    }


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
    wall_thickness: Array | float = 0.01,
) -> Array:
    """Absorber column capital cost.

    Based on column shell + packing/internals. The shell is Towler &
    Sinnott's vertical carbon-steel pressure vessel (Table 7.2, January 2010
    basis), ``C = 11600 + 34 S**0.85`` with ``S`` the shell mass in kg
    (valid 160-250,000 kg), times the material factor and escalated to 2020
    by CEPCI. The shell mass, heads included, is
    ``pi rho D (H + 0.8 D) t`` (the usual vessel-weight estimate). Packing
    is costed on the column volume (Turton loose packing for ``random``;
    Seider's installed, very approximate structured-packing price) and trays
    at 0.6 m spacing on the column cross-section (Turton sieve tray).

    Args:
        diameter: Column diameter (m)
        height: Packing height (m)
        material: Construction material (key of :data:`MATERIAL_FACTORS`)
        packing: Packing type ('structured', 'random', 'trays')
        wall_thickness: Shell wall thickness (m), including the corrosion
            allowance. The 10 mm default suits a near-ambient-pressure column
            a few metres across; set it from the mechanical design otherwise.

    Returns:
        Equipment cost (USD)

    Raises:
        ValueError: Unknown ``material`` or ``packing``.
    """
    diameter = jnp.asarray(diameter)
    height = jnp.asarray(height)
    fm = _lookup(MATERIAL_FACTORS, material, "material")
    packing_params = COLUMN_COSTS[_lookup(_PACKING_TYPES, packing, "packing")]

    # Column volume
    volume = jnp.pi * (diameter / 2) ** 2 * height

    shell_mass = _shell_mass(diameter, height, wall_thickness)

    # Shell cost: Towler & Sinnott vertical CS vessel, escalated to 2020
    shell_cost = fm * equipment_cost(shell_mass, _VERTICAL_VESSEL, target_year=_BASIS_YEAR)

    # Packing cost
    if packing == "trays":
        n_trays = height / 0.6  # Assume 0.6 m tray spacing
        tray_area = jnp.pi * (diameter / 2) ** 2
        packing_cost = n_trays * _parallel_units_cost(tray_area, packing_params)
    else:
        packing_cost = _parallel_units_cost(volume, packing_params)

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

    The column is :func:`absorber_cost`. The reboiler is Towler & Sinnott's
    U-tube kettle reboiler and the condenser Turton's floating-head
    exchanger, on areas from fixed U and LMTD; an area above the
    correlation's range is costed as parallel units.

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
    reboiler_cost = _parallel_units_cost(A_reb, HEAT_EXCHANGER_COSTS["kettle_reboiler"])

    # Condenser cost (shell and tube)
    U_cond = 800.0
    LMTD_cond = 20.0
    A_cond = condenser_duty / (U_cond * LMTD_cond)
    condenser_cost = _parallel_units_cost(A_cond, HEAT_EXCHANGER_COSTS["shell_tube_floating"])

    total = column_cost + reboiler_cost + condenser_cost
    return total


def heat_exchanger_cost(
    area: Array | float,
    hx_type: str = "shell_tube",
    material: str = "carbon_steel",
    pressure: float = 1000000.0,  # Pa
) -> Array:
    """Heat exchanger capital cost.

    The base cost is Turton et al.'s floating-head (``shell_tube``), flat
    plate (``plate``) or spiral plate (``spiral``) correlation from
    :data:`difflow.economics.capital.HEAT_EXCHANGER_COSTS`, escalated to
    2020. An area above the correlation's range (1000 m², or 100 m² for
    spiral) is costed as parallel units.

    Args:
        area: Heat transfer area (m²)
        hx_type: 'shell_tube', 'plate', 'spiral'
        material: Construction material (key of :data:`MATERIAL_FACTORS`)
        pressure: Design pressure (Pa)

    Returns:
        Equipment cost (USD)

    Raises:
        ValueError: Unknown ``hx_type`` or ``material``.
    """
    area = jnp.asarray(area)
    hx_params = HEAT_EXCHANGER_COSTS[_lookup(_HX_TYPES, hx_type, "hx_type")]
    fm = _lookup(MATERIAL_FACTORS, material, "material")

    # Pressure factor (for P > 10 bar)
    P_barg = pressure / 100000 - 1
    fp = jnp.where(P_barg > 10, 0.9 + 0.1 * P_barg / 10, 1.0)

    cost = fm * fp * _parallel_units_cost(area, hx_params)
    return cost


def compressor_cost(
    power: Array | float,
    comp_type: str = "centrifugal",
    driver: str = "electric",
) -> Array:
    """Compressor capital cost.

    The base cost is :data:`difflow.economics.capital.COMPRESSOR_COSTS`
    (Towler & Sinnott centrifugal and reciprocating, on driver power; Turton
    rotary for ``screw``, on fluid power; the shaft power is used for
    both), escalated to 2020, times Seider et al.'s driver factor
    (:data:`DRIVER_FACTORS`). A power above the correlation's range is
    costed as parallel machines.

    Args:
        power: Shaft power (W)
        comp_type: 'centrifugal', 'reciprocating', 'screw'
        driver: 'electric', 'steam_turbine', 'gas_turbine'

    Returns:
        Equipment cost (USD)

    Raises:
        ValueError: Unknown ``comp_type`` or ``driver``.
    """
    power = jnp.asarray(power)
    power_kW = power / 1000
    comp_params = _lookup(COMPRESSOR_COSTS, comp_type, "comp_type")
    fd = _lookup(DRIVER_FACTORS, driver, "driver")

    cost = fd * _parallel_units_cost(power_kW, comp_params)

    return cost


def membrane_module_cost(
    area: Array | float,
    membrane_type: str = "polymeric",
    price_per_m2: Array | float | None = None,
) -> Array:
    """Membrane module capital cost.

    Installed price per m² times area. The polymeric default, $50/m², is the
    mature-technology value widely assumed in the literature (Zhai & Rubin
    2019); a first-of-a-kind plant is nearer $200/m², which ``price_per_m2``
    sets. The other types' prices have no cited source.

    Args:
        area: Membrane area (m²)
        membrane_type: Key of :data:`MEMBRANE_PRICES` ('polymeric',
            'ceramic', 'facilitated', 'mixed_matrix')
        price_per_m2: Installed price ($/m²); overrides ``membrane_type``

    Returns:
        Equipment cost (USD)

    Raises:
        ValueError: Unknown ``membrane_type``.
    """
    area = jnp.asarray(area)
    if price_per_m2 is None:
        price_per_m2 = _lookup(MEMBRANE_PRICES, membrane_type, "membrane_type")

    cost = price_per_m2 * area
    return cost


def adsorber_vessel_cost(
    diameter: Array | float,
    length: Array | float,
    n_beds: int = 2,
    adsorbent_mass: Array | float = 1000.0,
    adsorbent_cost: float = 5.0,  # $/kg
) -> Array:
    """Adsorption vessel capital cost.

    Each bed is Towler & Sinnott's vertical carbon-steel pressure vessel on
    its shell mass (wall 1% of the diameter, two heads), escalated to 2020.

    Args:
        diameter: Vessel diameter (m)
        length: Vessel length (m)
        n_beds: Number of parallel beds
        adsorbent_mass: Mass of adsorbent per bed (kg)
        adsorbent_cost: Adsorbent cost ($/kg). The $5/kg default is near
            Seider et al.'s molecular sieve price, $85/ft³ of bulk volume
            (about $4.6/kg at 650 kg/m³)

    Returns:
        Equipment cost (USD)
    """
    diameter = jnp.asarray(diameter)
    length = jnp.asarray(length)
    adsorbent_mass = jnp.asarray(adsorbent_mass)

    # Pressure vessel: Towler & Sinnott vertical CS vessel on shell mass
    thickness = 0.01 * diameter  # Approximate wall thickness
    weight = _shell_mass(diameter, length, thickness)

    vessel_cost = equipment_cost(weight, _VERTICAL_VESSEL, target_year=_BASIS_YEAR)

    # Internals (bed supports, distributors); 15% has no cited source
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

    Seider et al.'s centrifugal backward-curved fan with its electric motor
    (2013, escalated to 2020) on the inlet flow, times Seider's head factor
    for ``pressure_rise`` (1.0 up to 4 in H2O, 1.55 at 31-40 in H2O, about
    10 kPa). A flow above 100,000 ACFM (47 m³/s) is costed as parallel fans.
    Above 40 in H2O a fan is the wrong machine; use :func:`compressor_cost`.

    Args:
        flow_rate: Volumetric flow at the fan inlet (m³/s)
        pressure_rise: Pressure rise (Pa)

    Returns:
        Equipment cost (USD)

    Warns:
        CostRangeWarning: A concrete ``pressure_rise`` above 40 in H2O
        (9.96 kPa); the head factor is held at 1.55.
    """
    flow_rate = jnp.asarray(flow_rate)
    head = jnp.asarray(pressure_rise) / _PA_PER_IN_H2O  # in H2O
    _check_size_range(head, _FAN_HEAD_RANGE)

    f_head = jnp.interp(head, jnp.asarray(_FAN_HEAD_KNOTS), jnp.asarray(_FAN_HEAD_FACTORS))
    cost = f_head * _parallel_units_cost(flow_rate * _M3S_TO_ACFM, _FAN_BACKWARD_CURVED)
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
