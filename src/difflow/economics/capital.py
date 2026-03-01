"""Capital cost estimation for process equipment.

This module provides differentiable cost correlations for common chemical
process equipment. All functions are JAX-compatible for gradient-based
optimization.

Cost correlations follow the general form:
    C = a + b * S^n

Where:
    C = purchased equipment cost ($)
    S = size/capacity parameter
    a, b, n = correlation constants

Reference: Turton et al., "Analysis, Synthesis, and Design of Chemical Processes"
           Seider et al., "Product and Process Design Principles"
"""

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple, Optional
from dataclasses import dataclass, replace, fields, asdict as dc_asdict

from .indices import escalate_cost, DEFAULT_BASE_YEAR, DEFAULT_CURRENT_YEAR


# =============================================================================
# Equipment Cost Correlation Parameters
# =============================================================================

class CostParams(NamedTuple):
    """Parameters for power-law cost correlation: C = a + b * S^n"""
    a: float  # Fixed cost component ($)
    b: float  # Scaling coefficient
    n: float  # Scaling exponent (typically 0.5-0.8)
    S_min: float  # Minimum size for correlation validity
    S_max: float  # Maximum size for correlation validity
    S_units: str  # Units of size parameter
    base_year: int  # Year of cost basis


# Equipment cost parameters (2019 basis from Turton et al.)
# Format: (a, b, n, S_min, S_max, units, year)

REACTOR_COSTS = {
    "cstr_jacketed": CostParams(
        a=17400, b=79.0, n=0.85,
        S_min=0.1, S_max=100.0, S_units="m³", base_year=2019
    ),
    "cstr_coil": CostParams(
        a=14000, b=68.0, n=0.85,
        S_min=0.1, S_max=100.0, S_units="m³", base_year=2019
    ),
    "pfr_tube": CostParams(
        a=3000, b=1200.0, n=0.65,
        S_min=0.01, S_max=10.0, S_units="m³", base_year=2019
    ),
    "batch_reactor": CostParams(
        a=25000, b=95.0, n=0.85,
        S_min=0.5, S_max=50.0, S_units="m³", base_year=2019
    ),
}

VESSEL_COSTS = {
    "pressure_vessel_vertical": CostParams(
        a=8000, b=380.0, n=0.72,
        S_min=0.1, S_max=200.0, S_units="m³", base_year=2019
    ),
    "pressure_vessel_horizontal": CostParams(
        a=7000, b=350.0, n=0.72,
        S_min=0.1, S_max=200.0, S_units="m³", base_year=2019
    ),
    "storage_tank_atmospheric": CostParams(
        a=5000, b=180.0, n=0.65,
        S_min=1.0, S_max=10000.0, S_units="m³", base_year=2019
    ),
    "flash_drum": CostParams(
        a=6500, b=320.0, n=0.70,
        S_min=0.1, S_max=50.0, S_units="m³", base_year=2019
    ),
}

HEAT_EXCHANGER_COSTS = {
    "shell_tube_floating": CostParams(
        a=11000, b=340.0, n=0.60,
        S_min=10.0, S_max=1000.0, S_units="m²", base_year=2019
    ),
    "shell_tube_fixed": CostParams(
        a=8500, b=280.0, n=0.60,
        S_min=10.0, S_max=1000.0, S_units="m²", base_year=2019
    ),
    "shell_tube_utube": CostParams(
        a=9000, b=300.0, n=0.60,
        S_min=10.0, S_max=1000.0, S_units="m²", base_year=2019
    ),
    "double_pipe": CostParams(
        a=1500, b=120.0, n=0.65,
        S_min=1.0, S_max=50.0, S_units="m²", base_year=2019
    ),
    "plate_frame": CostParams(
        a=3000, b=80.0, n=0.70,
        S_min=5.0, S_max=500.0, S_units="m²", base_year=2019
    ),
    "air_cooler": CostParams(
        a=15000, b=180.0, n=0.65,
        S_min=20.0, S_max=2000.0, S_units="m²", base_year=2019
    ),
}

COLUMN_COSTS = {
    "tray_column_shell": CostParams(
        a=15000, b=68.0, n=0.85,
        S_min=0.5, S_max=100.0, S_units="m³", base_year=2019
    ),
    "packed_column_shell": CostParams(
        a=12000, b=58.0, n=0.85,
        S_min=0.5, S_max=100.0, S_units="m³", base_year=2019
    ),
    "sieve_tray": CostParams(
        a=200, b=450.0, n=0.60,
        S_min=0.5, S_max=5.0, S_units="m² (per tray)", base_year=2019
    ),
    "valve_tray": CostParams(
        a=300, b=500.0, n=0.60,
        S_min=0.5, S_max=5.0, S_units="m² (per tray)", base_year=2019
    ),
    "packing_random": CostParams(
        a=0, b=800.0, n=1.0,
        S_min=0.1, S_max=100.0, S_units="m³", base_year=2019
    ),
    "packing_structured": CostParams(
        a=0, b=3000.0, n=1.0,
        S_min=0.1, S_max=100.0, S_units="m³", base_year=2019
    ),
}

PUMP_COSTS = {
    "centrifugal_single": CostParams(
        a=3500, b=320.0, n=0.55,
        S_min=0.5, S_max=100.0, S_units="kW", base_year=2019
    ),
    "centrifugal_multistage": CostParams(
        a=6000, b=410.0, n=0.55,
        S_min=1.0, S_max=500.0, S_units="kW", base_year=2019
    ),
    "reciprocating": CostParams(
        a=8000, b=680.0, n=0.50,
        S_min=1.0, S_max=200.0, S_units="kW", base_year=2019
    ),
    "gear": CostParams(
        a=2500, b=250.0, n=0.60,
        S_min=0.1, S_max=50.0, S_units="kW", base_year=2019
    ),
}

COMPRESSOR_COSTS = {
    "centrifugal": CostParams(
        a=50000, b=1800.0, n=0.65,
        S_min=100.0, S_max=10000.0, S_units="kW", base_year=2019
    ),
    "reciprocating": CostParams(
        a=25000, b=2200.0, n=0.60,
        S_min=10.0, S_max=1000.0, S_units="kW", base_year=2019
    ),
    "screw": CostParams(
        a=15000, b=1400.0, n=0.65,
        S_min=50.0, S_max=3000.0, S_units="kW", base_year=2019
    ),
}

SEPARATOR_COSTS = {
    "mixer_settler": CostParams(
        a=8000, b=250.0, n=0.75,
        S_min=0.1, S_max=50.0, S_units="m³", base_year=2019
    ),
    "centrifuge_disk": CostParams(
        a=30000, b=15000.0, n=0.45,
        S_min=0.01, S_max=5.0, S_units="m³/s", base_year=2019
    ),
    "filter_rotary": CostParams(
        a=20000, b=5000.0, n=0.60,
        S_min=1.0, S_max=100.0, S_units="m²", base_year=2019
    ),
    "extraction_column": CostParams(
        a=18000, b=72.0, n=0.85,
        S_min=0.5, S_max=100.0, S_units="m³", base_year=2019
    ),
}


# =============================================================================
# Core Cost Functions (JAX-differentiable)
# =============================================================================

def equipment_cost(
    size: Array,
    params: CostParams,
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate purchased equipment cost using power-law correlation.

    C = (a + b * S^n) * (CEPCI_target / CEPCI_base)

    Args:
        size: Equipment size/capacity parameter
        params: Cost correlation parameters
        target_year: Year for cost escalation

    Returns:
        Purchased equipment cost in target year dollars
    """
    base_cost = params.a + params.b * jnp.power(size, params.n)
    return escalate_cost(base_cost, params.base_year, target_year)


def equipment_cost_continuous(
    size: Array,
    a: Array,
    b: Array,
    n: Array,
    cepci_ratio: Array = jnp.array(1.0),
) -> Array:
    """Fully differentiable equipment cost (for optimization).

    All parameters can be JAX arrays for gradient computation.

    Args:
        size: Equipment size/capacity
        a: Fixed cost component
        b: Scaling coefficient
        n: Scaling exponent
        cepci_ratio: Cost escalation ratio (CEPCI_target/CEPCI_base)

    Returns:
        Equipment cost
    """
    return (a + b * jnp.power(size, n)) * cepci_ratio


# =============================================================================
# Specific Equipment Cost Functions
# =============================================================================

def reactor_cost(
    volume: Array,
    reactor_type: str = "cstr_jacketed",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate reactor purchased equipment cost.

    Args:
        volume: Reactor volume (m³)
        reactor_type: Type of reactor (see REACTOR_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if reactor_type not in REACTOR_COSTS:
        raise ValueError(f"Unknown reactor type: {reactor_type}. "
                        f"Available: {list(REACTOR_COSTS.keys())}")
    return equipment_cost(volume, REACTOR_COSTS[reactor_type], target_year)


def vessel_cost(
    volume: Array,
    vessel_type: str = "pressure_vessel_vertical",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate vessel purchased equipment cost.

    Args:
        volume: Vessel volume (m³)
        vessel_type: Type of vessel (see VESSEL_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if vessel_type not in VESSEL_COSTS:
        raise ValueError(f"Unknown vessel type: {vessel_type}. "
                        f"Available: {list(VESSEL_COSTS.keys())}")
    return equipment_cost(volume, VESSEL_COSTS[vessel_type], target_year)


def heat_exchanger_cost(
    area: Array,
    hx_type: str = "shell_tube_floating",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate heat exchanger purchased equipment cost.

    Args:
        area: Heat transfer area (m²)
        hx_type: Type of heat exchanger (see HEAT_EXCHANGER_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if hx_type not in HEAT_EXCHANGER_COSTS:
        raise ValueError(f"Unknown heat exchanger type: {hx_type}. "
                        f"Available: {list(HEAT_EXCHANGER_COSTS.keys())}")
    return equipment_cost(area, HEAT_EXCHANGER_COSTS[hx_type], target_year)


def pump_cost(
    power: Array,
    pump_type: str = "centrifugal_single",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate pump purchased equipment cost.

    Args:
        power: Pump power (kW)
        pump_type: Type of pump (see PUMP_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if pump_type not in PUMP_COSTS:
        raise ValueError(f"Unknown pump type: {pump_type}. "
                        f"Available: {list(PUMP_COSTS.keys())}")
    return equipment_cost(power, PUMP_COSTS[pump_type], target_year)


def compressor_cost(
    power: Array,
    compressor_type: str = "centrifugal",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate compressor purchased equipment cost.

    Args:
        power: Compressor power (kW)
        compressor_type: Type of compressor (see COMPRESSOR_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if compressor_type not in COMPRESSOR_COSTS:
        raise ValueError(f"Unknown compressor type: {compressor_type}. "
                        f"Available: {list(COMPRESSOR_COSTS.keys())}")
    return equipment_cost(power, COMPRESSOR_COSTS[compressor_type], target_year)


def column_cost(
    volume: Array,
    column_type: str = "tray_column_shell",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate column shell purchased equipment cost.

    Args:
        volume: Column shell volume (m³)
        column_type: Type of column (see COLUMN_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if column_type not in COLUMN_COSTS:
        raise ValueError(f"Unknown column type: {column_type}. "
                        f"Available: {list(COLUMN_COSTS.keys())}")
    return equipment_cost(volume, COLUMN_COSTS[column_type], target_year)


def separator_cost(
    size: Array,
    separator_type: str = "mixer_settler",
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate separator purchased equipment cost.

    Args:
        size: Size parameter (m³ for settlers, m²/s for centrifuges, etc.)
        separator_type: Type of separator (see SEPARATOR_COSTS)
        target_year: Cost basis year

    Returns:
        Purchased equipment cost ($)
    """
    if separator_type not in SEPARATOR_COSTS:
        raise ValueError(f"Unknown separator type: {separator_type}. "
                        f"Available: {list(SEPARATOR_COSTS.keys())}")
    return equipment_cost(size, SEPARATOR_COSTS[separator_type], target_year)


# =============================================================================
# Installation Factors and Total Installed Cost
# =============================================================================

@dataclass
class InstallationFactors:
    """Factors for converting purchased equipment cost to installed cost.

    Total installed cost = C_equipment * (1 + sum of factors)
    """
    piping: float = 0.35
    instrumentation: float = 0.20
    electrical: float = 0.12
    buildings: float = 0.15
    yard_improvements: float = 0.05
    service_facilities: float = 0.15
    engineering: float = 0.10
    construction: float = 0.10
    contingency: float = 0.15

    @property
    def total_factor(self) -> float:
        """Total installation factor (Lang-type)."""
        return 1.0 + (
            self.piping + self.instrumentation + self.electrical +
            self.buildings + self.yard_improvements + self.service_facilities +
            self.engineering + self.construction + self.contingency
        )

    def update(self, **kwargs) -> "InstallationFactors":
        """Return a new InstallationFactors with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., piping=0.40, contingency=0.20)

        Returns:
            New InstallationFactors with updated fields
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        """Get parameter value by name for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists in the params."""
        return key in {f.name for f in fields(self)}

    def keys(self):
        """Return field names for dict-like iteration."""
        return (f.name for f in fields(self))

    def values(self):
        """Return field values for dict-like iteration.

        Returns:
            Iterator over field values
        """
        return (getattr(self, f.name) for f in fields(self))

    def items(self):
        """Return (name, value) pairs for dict-like iteration.

        Returns:
            Iterator over (field_name, value) tuples
        """
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def __iter__(self):
        """Iterate over field names (like dict)."""
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        """Return number of fields."""
        return len(fields(self))

    def asdict(self) -> dict:
        """Convert params to a dictionary."""
        return dc_asdict(self)

    def __repr__(self) -> str:
        """Concise string representation."""
        def fmt(v):
            if v is None:
                return "None"
            if callable(v) and hasattr(v, '__name__'):
                return v.__name__
            if hasattr(v, 'shape'):
                if v.ndim == 0:
                    return f"{float(v):.4g}"
                return f"Array{list(v.shape)}"
            if isinstance(v, dict):
                items = ", ".join(f"{k}: {fmt(val)}" for k, val in v.items())
                return "{" + items + "}"
            if isinstance(v, (list, tuple)) and len(v) > 5:
                return f"{type(v).__name__}[{len(v)}]"
            return repr(v)
        items = ", ".join(f"{f.name}={fmt(getattr(self, f.name))}" for f in fields(self))
        return f"{self.__class__.__name__}({items})"


# Pre-defined installation factors by plant type
INSTALLATION_FACTORS = {
    "fluid_processing": InstallationFactors(
        piping=0.66, instrumentation=0.30, electrical=0.11,
        buildings=0.18, yard_improvements=0.10, service_facilities=0.55,
        engineering=0.32, construction=0.34, contingency=0.37
    ),
    "solids_processing": InstallationFactors(
        piping=0.31, instrumentation=0.18, electrical=0.11,
        buildings=0.29, yard_improvements=0.10, service_facilities=0.55,
        engineering=0.32, construction=0.34, contingency=0.37
    ),
    "mixed_processing": InstallationFactors(
        piping=0.45, instrumentation=0.25, electrical=0.11,
        buildings=0.22, yard_improvements=0.10, service_facilities=0.55,
        engineering=0.32, construction=0.34, contingency=0.37
    ),
}

# Classic Lang factors
LANG_FACTORS = {
    "fluid_processing": 4.74,
    "solids_processing": 3.63,
    "mixed_processing": 4.22,
}


def installed_cost(
    purchased_cost: Array,
    lang_factor: float = 4.74,
) -> Array:
    """Calculate total installed cost using Lang factor method.

    Installed cost = Purchased cost * Lang factor

    Args:
        purchased_cost: Purchased equipment cost ($)
        lang_factor: Lang factor (default 4.74 for fluid processing)

    Returns:
        Total installed cost ($)
    """
    return purchased_cost * lang_factor


def installed_cost_detailed(
    purchased_cost: Array,
    factors: InstallationFactors | None = None,
) -> Array:
    """Calculate installed cost using detailed factor method.

    Args:
        purchased_cost: Purchased equipment cost ($)
        factors: Installation factors (default: fluid processing)

    Returns:
        Total installed cost ($)
    """
    if factors is None:
        factors = INSTALLATION_FACTORS["fluid_processing"]
    return purchased_cost * factors.total_factor


def bare_module_cost(
    purchased_cost: Array,
    material_factor: float = 1.0,
    pressure_factor: float = 1.0,
    B1: float = 1.89,
    B2: float = 1.35,
) -> Array:
    """Calculate bare module cost (Guthrie method).

    C_BM = C_p * (B_1 + B_2 * F_M * F_P)

    Default B1, B2 values are for heat exchangers (Turton et al.)

    Args:
        purchased_cost: Purchased equipment cost ($)
        material_factor: Material of construction factor (F_M)
        pressure_factor: Pressure factor (F_P)
        B1: Bare module factor constant 1 (default 1.89)
        B2: Bare module factor constant 2 (default 1.35)

    Returns:
        Bare module cost ($)
    """
    return purchased_cost * (B1 + B2 * material_factor * pressure_factor)


# =============================================================================
# Material and Pressure Factors
# =============================================================================

MATERIAL_FACTORS = {
    "carbon_steel": 1.0,
    "stainless_304": 1.8,
    "stainless_316": 2.1,
    "stainless_317": 2.5,
    "monel": 3.5,
    "inconel": 3.9,
    "nickel": 5.4,
    "hastelloy_c": 4.0,
    "titanium": 7.0,
    "copper": 1.4,
    "brass": 1.3,
    "aluminum": 1.2,
    "fiberglass": 1.5,
    "rubber_lined": 1.6,
    "glass_lined": 4.0,
    "teflon_lined": 2.5,
}


def pressure_factor_vessel(
    pressure: Array,
    diameter: Array,
) -> Array:
    """Calculate pressure factor for vessels.

    Based on ASME pressure vessel code thickness requirements.

    Args:
        pressure: Design pressure (barg)
        diameter: Vessel diameter (m)

    Returns:
        Pressure factor (multiplicative)
    """
    # Simplified correlation
    # F_P = 1.0 for P < 5 barg
    # F_P increases with pressure and diameter
    P_normalized = jnp.maximum(pressure - 5.0, 0.0) / 100.0
    D_factor = 1.0 + 0.1 * jnp.maximum(diameter - 1.0, 0.0)
    return 1.0 + P_normalized * D_factor


def pressure_factor_hx(pressure: Array) -> Array:
    """Calculate pressure factor for heat exchangers.

    Args:
        pressure: Shell-side design pressure (barg)

    Returns:
        Pressure factor (multiplicative)
    """
    # Correlation from Turton with smooth transition at 5 barg
    log_P = jnp.log10(jnp.maximum(pressure, 0.1))
    F_P_high = 0.9803 + 0.018 * log_P + 0.0017 * log_P**2
    # Smooth transition at 5 barg using sigmoid blend
    blend = 1.0 / (1.0 + jnp.exp(-2.0 * (pressure - 5.0)))
    return 1.0 * (1.0 - blend) + F_P_high * blend


# =============================================================================
# Total Capital Investment
# =============================================================================

@dataclass
class CapitalInvestment:
    """Breakdown of total capital investment."""
    purchased_equipment: float
    installed_equipment: float
    total_direct_costs: float  # Installed + piping + instrumentation + etc.
    total_indirect_costs: float  # Engineering + construction + contingency
    fixed_capital_investment: float  # Direct + indirect
    working_capital: float
    total_capital_investment: float


def total_capital_investment(
    purchased_equipment_cost: Array,
    lang_factor: float = 4.74,
    working_capital_fraction: float = 0.15,
) -> Array:
    """Calculate total capital investment (TCI).

    TCI = FCI + Working Capital
    FCI = Purchased Equipment Cost * Lang Factor
    Working Capital = fraction of FCI

    Args:
        purchased_equipment_cost: Total purchased equipment cost ($)
        lang_factor: Installation factor
        working_capital_fraction: Working capital as fraction of FCI

    Returns:
        Total capital investment ($)
    """
    fci = purchased_equipment_cost * lang_factor
    working_capital = fci * working_capital_fraction
    return fci + working_capital


def total_capital_investment_detailed(
    equipment_costs: dict[str, Array],
    factors: InstallationFactors | None = None,
    working_capital_fraction: float = 0.15,
) -> CapitalInvestment:
    """Detailed capital investment breakdown.

    Args:
        equipment_costs: Dict of equipment name -> purchased cost
        factors: Installation factors (default: fluid processing)
        working_capital_fraction: Working capital as fraction of FCI

    Returns:
        CapitalInvestment breakdown
    """
    if factors is None:
        factors = INSTALLATION_FACTORS["fluid_processing"]

    total_purchased = sum(float(c) for c in equipment_costs.values())

    # Direct costs
    direct_factor = 1.0 + factors.piping + factors.instrumentation + \
                    factors.electrical + factors.buildings + \
                    factors.yard_improvements + factors.service_facilities
    total_direct = total_purchased * direct_factor

    # Indirect costs
    indirect_factor = factors.engineering + factors.construction + factors.contingency
    total_indirect = total_purchased * indirect_factor

    fci = total_direct + total_indirect
    working_capital = fci * working_capital_fraction
    tci = fci + working_capital

    return CapitalInvestment(
        purchased_equipment=total_purchased,
        installed_equipment=total_purchased * factors.total_factor,
        total_direct_costs=total_direct,
        total_indirect_costs=total_indirect,
        fixed_capital_investment=fci,
        working_capital=working_capital,
        total_capital_investment=tci,
    )
