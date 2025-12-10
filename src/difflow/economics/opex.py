"""Operating cost (OPEX) models for process economics.

This module provides comprehensive operating cost calculations including:
- Raw materials
- Utilities
- Labor
- Maintenance
- Overhead and administrative costs

All functions are JAX-compatible for gradient-based optimization.
"""

import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass, field
from typing import NamedTuple

from .utilities import UtilityConsumption, total_utility_cost, UtilityPrices


# =============================================================================
# Operating Hours and Capacity
# =============================================================================

# Standard operating hours per year
HOURS_PER_YEAR = 8000.0  # Typical for continuous chemical plant
SECONDS_PER_YEAR = HOURS_PER_YEAR * 3600.0


@dataclass
class OperatingSchedule:
    """Plant operating schedule parameters."""
    hours_per_year: float = 8000.0  # Total operating hours
    shifts_per_day: int = 3  # Number of shifts
    days_per_week: float = 7.0  # Operating days per week
    weeks_per_year: float = 50.0  # Operating weeks per year
    stream_factor: float = 0.91  # Fraction of time at full capacity

    @property
    def seconds_per_year(self) -> float:
        return self.hours_per_year * 3600.0


DEFAULT_SCHEDULE = OperatingSchedule()


# =============================================================================
# Raw Material Costs
# =============================================================================

@dataclass
class RawMaterial:
    """Raw material specification."""
    name: str
    price: float  # $/kg
    consumption_rate: float = 0.0  # kg/s at full capacity
    molecular_weight: float = 1.0  # g/mol (for molar flow conversion)


def raw_material_cost(
    flowrate: Array,
    price: Array,
) -> Array:
    """Calculate raw material cost.

    Args:
        flowrate: Mass flowrate (kg/s)
        price: Material price ($/kg)

    Returns:
        Cost rate ($/s)
    """
    return jnp.maximum(flowrate, 0.0) * price


def raw_material_cost_molar(
    molar_flowrate: Array,
    price: Array,
    molecular_weight: Array,
) -> Array:
    """Calculate raw material cost from molar flowrate.

    Args:
        molar_flowrate: Molar flowrate (mol/s)
        price: Material price ($/kg)
        molecular_weight: Molecular weight (g/mol)

    Returns:
        Cost rate ($/s)
    """
    mass_flowrate = molar_flowrate * molecular_weight / 1000.0  # kg/s
    return raw_material_cost(mass_flowrate, price)


def total_raw_material_cost(
    materials: dict[str, tuple[Array, Array]],
) -> Array:
    """Calculate total raw material cost.

    Args:
        materials: Dict of material_name -> (flowrate_kg_s, price_per_kg)

    Returns:
        Total cost rate ($/s)
    """
    total = jnp.array(0.0)
    for name, (flowrate, price) in materials.items():
        total = total + raw_material_cost(flowrate, price)
    return total


def annual_raw_material_cost(
    materials: dict[str, tuple[Array, Array]],
    schedule: OperatingSchedule | None = None,
) -> Array:
    """Calculate annual raw material cost.

    Args:
        materials: Dict of material_name -> (flowrate_kg_s, price_per_kg)
        schedule: Operating schedule

    Returns:
        Annual cost ($)
    """
    if schedule is None:
        schedule = DEFAULT_SCHEDULE

    cost_per_second = total_raw_material_cost(materials)
    return cost_per_second * schedule.seconds_per_year * schedule.stream_factor


# =============================================================================
# Labor Costs
# =============================================================================

@dataclass
class LaborRates:
    """Labor cost parameters."""
    operator_salary: float = 65000.0  # $/year per operator
    supervisor_salary: float = 90000.0  # $/year per supervisor
    overhead_factor: float = 1.4  # Benefits, taxes, etc. multiplier
    operators_per_shift: int = 4  # Operators per shift
    shifts_per_day: int = 3
    supervisor_ratio: float = 0.2  # Supervisors per operator


DEFAULT_LABOR_RATES = LaborRates()


def operating_labor_cost(
    n_operators_per_shift: int = 4,
    n_shifts: int = 3,
    rates: LaborRates | None = None,
) -> float:
    """Calculate annual operating labor cost.

    Args:
        n_operators_per_shift: Operators required per shift
        n_shifts: Number of shifts per day
        rates: Labor rate parameters

    Returns:
        Annual labor cost ($)
    """
    if rates is None:
        rates = DEFAULT_LABOR_RATES

    # Total operators needed (including relief for 24/7 operation)
    # 4.5 shift coverage factor for continuous operation
    shift_coverage = 4.5 if n_shifts == 3 else 3.0
    total_operators = n_operators_per_shift * shift_coverage

    operator_cost = total_operators * rates.operator_salary * rates.overhead_factor

    # Supervisors
    n_supervisors = int(total_operators * rates.supervisor_ratio + 0.5)
    supervisor_cost = n_supervisors * rates.supervisor_salary * rates.overhead_factor

    return operator_cost + supervisor_cost


def labor_cost_from_equipment(
    n_major_equipment: int,
    process_complexity: str = "medium",
    rates: LaborRates | None = None,
) -> float:
    """Estimate labor cost from equipment count.

    Uses correlation: Operators/shift = (6.29 + 0.23*P)^0.5
    Where P = number of processing steps requiring operator attention

    Args:
        n_major_equipment: Number of major equipment items
        process_complexity: "simple", "medium", or "complex"
        rates: Labor rate parameters

    Returns:
        Annual labor cost ($)
    """
    # Complexity factor
    complexity_map = {"simple": 0.8, "medium": 1.0, "complex": 1.3}
    factor = complexity_map.get(process_complexity, 1.0)

    # Correlation from literature
    P = n_major_equipment * factor
    operators_per_shift = (6.29 + 0.23 * P) ** 0.5
    operators_per_shift = max(2, int(operators_per_shift + 0.5))

    return operating_labor_cost(
        n_operators_per_shift=operators_per_shift,
        rates=rates
    )


# =============================================================================
# Maintenance, Insurance, and Overhead
# =============================================================================

@dataclass
class OverheadFactors:
    """Overhead cost factors as fractions of relevant bases."""
    # Fractions of Fixed Capital Investment (FCI)
    maintenance: float = 0.04  # 4% of FCI
    property_taxes: float = 0.02  # 2% of FCI
    insurance: float = 0.01  # 1% of FCI

    # Fractions of Operating Labor
    supervision: float = 0.25  # 25% of operating labor
    laboratory: float = 0.10  # 10% of operating labor
    plant_overhead: float = 0.60  # 60% of operating labor

    # Fractions of Total Manufacturing Cost
    general_admin: float = 0.05  # 5% of manufacturing cost
    distribution_selling: float = 0.05  # 5% of manufacturing cost
    research_dev: float = 0.03  # 3% of manufacturing cost


DEFAULT_OVERHEAD = OverheadFactors()


def maintenance_cost(
    fixed_capital: Array,
    factors: OverheadFactors | None = None,
) -> Array:
    """Calculate annual maintenance cost.

    Args:
        fixed_capital: Fixed capital investment ($)
        factors: Overhead factors

    Returns:
        Annual maintenance cost ($)
    """
    if factors is None:
        factors = DEFAULT_OVERHEAD
    return fixed_capital * factors.maintenance


def insurance_taxes_cost(
    fixed_capital: Array,
    factors: OverheadFactors | None = None,
) -> Array:
    """Calculate annual property taxes and insurance.

    Args:
        fixed_capital: Fixed capital investment ($)
        factors: Overhead factors

    Returns:
        Annual taxes and insurance ($)
    """
    if factors is None:
        factors = DEFAULT_OVERHEAD
    return fixed_capital * (factors.property_taxes + factors.insurance)


def plant_overhead_cost(
    operating_labor: Array,
    factors: OverheadFactors | None = None,
) -> Array:
    """Calculate plant overhead costs.

    Includes supervision, laboratory, and general plant overhead.

    Args:
        operating_labor: Annual operating labor cost ($)
        factors: Overhead factors

    Returns:
        Annual plant overhead cost ($)
    """
    if factors is None:
        factors = DEFAULT_OVERHEAD

    return operating_labor * (
        factors.supervision + factors.laboratory + factors.plant_overhead
    )


# =============================================================================
# Total Operating Cost Calculator
# =============================================================================

@dataclass
class OperatingCostBreakdown:
    """Detailed operating cost breakdown."""
    raw_materials: float = 0.0
    utilities: float = 0.0
    operating_labor: float = 0.0
    maintenance: float = 0.0
    plant_overhead: float = 0.0
    taxes_insurance: float = 0.0
    general_expenses: float = 0.0

    @property
    def variable_costs(self) -> float:
        """Variable operating costs (depend on production rate)."""
        return self.raw_materials + self.utilities

    @property
    def fixed_costs(self) -> float:
        """Fixed operating costs (independent of production rate)."""
        return (
            self.operating_labor + self.maintenance +
            self.plant_overhead + self.taxes_insurance
        )

    @property
    def manufacturing_cost(self) -> float:
        """Total manufacturing cost (COM)."""
        return self.variable_costs + self.fixed_costs

    @property
    def total_production_cost(self) -> float:
        """Total production cost (TPC) including general expenses."""
        return self.manufacturing_cost + self.general_expenses


def calculate_opex(
    raw_material_cost: float,
    utility_cost: float,
    fixed_capital: float,
    operating_labor: float | None = None,
    n_operators_per_shift: int = 4,
    factors: OverheadFactors | None = None,
    labor_rates: LaborRates | None = None,
) -> OperatingCostBreakdown:
    """Calculate comprehensive operating cost breakdown.

    Args:
        raw_material_cost: Annual raw material cost ($)
        utility_cost: Annual utility cost ($)
        fixed_capital: Fixed capital investment ($)
        operating_labor: Annual operating labor cost (if known)
        n_operators_per_shift: Operators per shift (if labor not specified)
        factors: Overhead factors
        labor_rates: Labor rates

    Returns:
        OperatingCostBreakdown with all components
    """
    if factors is None:
        factors = DEFAULT_OVERHEAD

    # Calculate labor if not provided
    if operating_labor is None:
        operating_labor = operating_labor_cost(
            n_operators_per_shift=n_operators_per_shift,
            rates=labor_rates
        )

    # Capital-dependent costs
    maint = float(maintenance_cost(jnp.array(fixed_capital), factors))
    taxes_ins = float(insurance_taxes_cost(jnp.array(fixed_capital), factors))

    # Labor-dependent costs
    overhead = float(plant_overhead_cost(jnp.array(operating_labor), factors))

    # Create breakdown
    breakdown = OperatingCostBreakdown(
        raw_materials=raw_material_cost,
        utilities=utility_cost,
        operating_labor=operating_labor,
        maintenance=maint,
        plant_overhead=overhead,
        taxes_insurance=taxes_ins,
    )

    # General expenses (fraction of manufacturing cost)
    general = breakdown.manufacturing_cost * (
        factors.general_admin + factors.distribution_selling + factors.research_dev
    )
    breakdown.general_expenses = general

    return breakdown


def simple_opex(
    raw_materials: Array,
    utilities: Array,
    fixed_capital: Array,
    labor_factor: float = 0.02,  # Labor as fraction of FCI
    maintenance_factor: float = 0.04,
    overhead_factor: float = 0.02,
) -> Array:
    """Simplified total operating cost (JAX-differentiable).

    OPEX = raw_materials + utilities + (labor + maintenance + overhead) * FCI

    Args:
        raw_materials: Annual raw material cost ($)
        utilities: Annual utility cost ($)
        fixed_capital: Fixed capital investment ($)
        labor_factor: Labor as fraction of FCI
        maintenance_factor: Maintenance as fraction of FCI
        overhead_factor: Overhead as fraction of FCI

    Returns:
        Total annual operating cost ($)
    """
    variable = raw_materials + utilities
    fixed = fixed_capital * (labor_factor + maintenance_factor + overhead_factor)
    return variable + fixed


def opex_per_unit_product(
    total_opex: Array,
    annual_production: Array,
) -> Array:
    """Calculate operating cost per unit of product.

    Args:
        total_opex: Total annual operating cost ($)
        annual_production: Annual production rate (kg or mol)

    Returns:
        Operating cost per unit ($/kg or $/mol)
    """
    return total_opex / jnp.maximum(annual_production, 1e-10)


# =============================================================================
# Quick Estimation Methods
# =============================================================================

def opex_from_fci_quick(
    fixed_capital: Array,
    raw_material_fraction: float = 0.50,
    utility_fraction: float = 0.10,
) -> Array:
    """Quick OPEX estimate from fixed capital investment.

    Typical ranges:
    - Raw materials: 30-60% of TPC
    - Utilities: 5-15% of TPC
    - Labor + overhead + maintenance: ~8-12% of FCI

    This uses: OPEX ~ FCI * (raw_material + utility + 0.10)

    For a rough estimate when detailed flows aren't available.

    Args:
        fixed_capital: Fixed capital investment ($)
        raw_material_fraction: Raw material cost as fraction of FCI
        utility_fraction: Utility cost as fraction of FCI

    Returns:
        Estimated annual operating cost ($)
    """
    # Fixed costs typically ~10% of FCI
    fixed_factor = 0.10

    return fixed_capital * (raw_material_fraction + utility_fraction + fixed_factor)


def com_from_correlations(
    fixed_capital: Array,
    utility_cost: Array,
    raw_material_cost: Array,
    waste_treatment_cost: Array = jnp.array(0.0),
    n_operators_per_shift: int = 4,
) -> Array:
    """Cost of Manufacturing (COM) from Turton correlations.

    COM = 0.18*FCI + 2.73*COL + 1.23*(CUT + CWT + CRM)

    Where:
    - FCI = Fixed capital investment
    - COL = Cost of operating labor
    - CUT = Utility cost
    - CWT = Waste treatment cost
    - CRM = Raw material cost

    Args:
        fixed_capital: Fixed capital investment ($)
        utility_cost: Annual utility cost ($)
        raw_material_cost: Annual raw material cost ($)
        waste_treatment_cost: Annual waste treatment cost ($)
        n_operators_per_shift: Number of operators per shift

    Returns:
        Annual cost of manufacturing ($)
    """
    col = operating_labor_cost(n_operators_per_shift)

    com = (
        0.18 * fixed_capital +
        2.73 * col +
        1.23 * (utility_cost + waste_treatment_cost + raw_material_cost)
    )

    return com
