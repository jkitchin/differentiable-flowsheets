"""Capital cost estimation for process equipment.

This module provides differentiable purchased-cost correlations for common
chemical process equipment. All functions are JAX-compatible for
gradient-based optimization.

Each table entry is a :class:`CostParams` in one of two published forms:

- ``"log_quadratic"`` (Turton et al.)::

      log10 C = K1 + K2 log10 S + K3 (log10 S)**2     (a, b, n hold K1, K2, K3)

- ``"power_law"`` (Towler & Sinnott; Seider et al.)::

      C = a + b S**n

``C`` is the purchased cost in ``base_year`` dollars, escalated by CEPCI.
These are carbon-steel, near-ambient-pressure costs. Use
:func:`bare_module_cost` and the pressure factors for anything else.

References:
    Turton, Shaeiwitz, Bhattacharyya and Whiting, *Analysis, Synthesis, and
    Design of Chemical Processes*, 5th ed. (2018), Table A.1 (2001 basis).
    Towler and Sinnott, *Chemical Engineering Design*, 2nd ed. (2012),
    Table 7.2 (January 2010 US Gulf Coast basis).
    Seider et al., *Product and Process Design Principles*, 4th ed. (2016),
    Section 16.5 (2013 basis).
The coefficients were checked against openpytea's transcription of these
tables (``cost_correlations.csv``, openpytea.readthedocs.io).
"""

import warnings

import jax.numpy as jnp
import numpy as np
from jax import Array
from typing import NamedTuple, Optional
from dataclasses import dataclass, replace, fields, asdict as dc_asdict

from .indices import escalate_cost, DEFAULT_BASE_YEAR, DEFAULT_CURRENT_YEAR


class CostRangeWarning(UserWarning):
    """A cost correlation was evaluated outside its stated size range.

    Every :class:`CostParams` stores the size range its source fitted
    (``S_min``..``S_max``). Outside it the power law or log-quadratic is an
    extrapolation, and the log-quadratic form in particular can bend the
    wrong way. :func:`equipment_cost` raises this warning for a concrete size
    outside the range; the cost is still returned (and is differentiable).
    Under ``jit``/``grad`` the size is traced and no check is made.
    """


def _check_size_range(size, params) -> None:
    """Warn (CostRangeWarning) if a concrete ``size`` is outside the range."""
    if params.S_min is None or params.S_max is None:
        return
    try:
        s = np.asarray(size, dtype=float)
    except Exception:  # traced value: nothing to check
        return
    if s.size == 0:
        return
    lo, hi = float(np.min(s)), float(np.max(s))
    if lo < params.S_min or hi > params.S_max:
        bad = lo if lo < params.S_min else hi
        src = f" ({params.source})" if params.source else ""
        warnings.warn(
            f"size {bad:g} {params.S_units} is outside the correlation's valid "
            f"range [{params.S_min:g}, {params.S_max:g}] {params.S_units}{src}; "
            "the cost is an extrapolation",
            CostRangeWarning,
            stacklevel=3,
        )


# =============================================================================
# Equipment Cost Correlation Parameters
# =============================================================================

class CostParams(NamedTuple):
    """One purchased-cost correlation.

    For ``form="power_law"``, ``C = a + b * S**n``. For
    ``form="log_quadratic"``, ``log10 C = a + b log10 S + n (log10 S)**2``,
    where a, b and n are Turton's K1, K2 and K3. ``C`` is in ``base_year``
    dollars.
    """
    a: float  # Fixed cost ($), or K1
    b: float  # Scaling coefficient, or K2
    n: float  # Scaling exponent, or K3
    S_min: float  # Minimum size for correlation validity
    S_max: float  # Maximum size for correlation validity
    S_units: str  # Units of size parameter
    base_year: int  # Year of cost basis
    form: str = "power_law"  # "power_law" | "log_quadratic"
    source: str = ""  # Citation for the coefficients

    def base_cost(self, size: Array) -> Array:
        """Purchased cost in ``base_year`` dollars (differentiable in size)."""
        if self.form == "log_quadratic":
            return equipment_cost_log_quadratic(size, self.a, self.b, self.n)
        return self.a + self.b * jnp.power(size, self.n)

    def exponent(self, size: Array) -> Array:
        """Local scaling exponent d ln C / d ln S at ``size``."""
        if self.form == "log_quadratic":
            return self.b + 2.0 * self.n * jnp.log10(size)
        return self.b * self.n * jnp.power(size, self.n) / self.base_cost(size)


TURTON = "Turton et al. 5e Table A.1"
TOWLER = "Towler & Sinnott 2e Table 7.2"
SEIDER = "Seider et al. 4e Sec. 16.5"

REACTOR_COSTS = {
    "cstr_jacketed": CostParams(
        a=4.1052, b=0.532, n=-0.0005, S_min=0.1, S_max=35.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': jacketed agitated reactor',
    ),
    "cstr_coil": CostParams(
        a=4.1052, b=0.532, n=-0.0005, S_min=0.1, S_max=35.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': jacketed agitated reactor (no separate coil row)',
    ),
    "pfr_tube": CostParams(
        a=3.3496, b=0.7235, n=0.0025, S_min=5.0, S_max=45.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': jacketed non-agitated reactor (proxy for a tubular reactor)',
    ),
    "batch_reactor": CostParams(
        a=4.5587, b=0.2986, n=0.002, S_min=1.0, S_max=15.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': autoclave',
    ),
}

VESSEL_COSTS = {
    "pressure_vessel_vertical": CostParams(
        a=3.4974, b=0.4485, n=0.1074, S_min=0.3, S_max=520.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': vertical process vessel',
    ),
    "pressure_vessel_horizontal": CostParams(
        a=3.5565, b=0.3776, n=0.0905, S_min=0.1, S_max=628.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': horizontal process vessel',
    ),
    "storage_tank_atmospheric": CostParams(
        a=4.8509, b=-0.3973, n=0.1445, S_min=90.0, S_max=30000.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': API fixed-roof tank',
    ),
    "flash_drum": CostParams(
        a=3.4974, b=0.4485, n=0.1074, S_min=0.3, S_max=520.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': vertical process vessel',
    ),
}

HEAT_EXCHANGER_COSTS = {
    "shell_tube_floating": CostParams(
        a=4.8306, b=-0.8509, n=0.3187, S_min=10.0, S_max=1000.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': floating head',
    ),
    "shell_tube_fixed": CostParams(
        a=4.3247, b=-0.303, n=0.1634, S_min=10.0, S_max=1000.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': fixed tube sheet',
    ),
    "shell_tube_utube": CostParams(
        a=4.1884, b=-0.2503, n=0.1974, S_min=10.0, S_max=1000.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': U-tube',
    ),
    "double_pipe": CostParams(
        a=3.3444, b=0.2745, n=-0.0472, S_min=1.0, S_max=10.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': double pipe',
    ),
    "plate_frame": CostParams(
        a=4.6656, b=-0.1557, n=0.1547, S_min=10.0, S_max=1000.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': flat plate',
    ),
    "air_cooler": CostParams(
        a=4.0336, b=0.2341, n=0.0497, S_min=10.0, S_max=10000.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': air cooler',
    ),
}

COLUMN_COSTS = {
    "tray_column_shell": CostParams(
        a=3.4974, b=0.4485, n=0.1074, S_min=0.3, S_max=520.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': tower (tray and packed)',
    ),
    "packed_column_shell": CostParams(
        a=3.4974, b=0.4485, n=0.1074, S_min=0.3, S_max=520.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': tower (tray and packed)',
    ),
    "sieve_tray": CostParams(
        a=2.9949, b=0.4465, n=0.3961, S_min=0.07, S_max=12.3, S_units="m² (per tray)",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': sieve trays',
    ),
    "valve_tray": CostParams(
        a=3.3322, b=0.4838, n=0.3434, S_min=0.7, S_max=10.5, S_units="m² (per tray)",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': valve trays',
    ),
    "packing_random": CostParams(
        a=2.4493, b=0.9744, n=0.0055, S_min=0.03, S_max=628.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': loose packing',
    ),
    "packing_structured": CostParams(
        a=0.0, b=10065.0, n=1.0, S_min=0.1, S_max=100.0, S_units="m³",
        base_year=2013,
        source=SEIDER + ': corrugated-sheet stainless, $285/ft³ INSTALLED, very approximate',
    ),
}

PUMP_COSTS = {
    "centrifugal_single": CostParams(
        a=3.3892, b=0.0536, n=0.1538, S_min=1.0, S_max=300.0, S_units="kW",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': centrifugal pump',
    ),
    "centrifugal_multistage": CostParams(
        a=3.3892, b=0.0536, n=0.1538, S_min=1.0, S_max=300.0, S_units="kW",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': centrifugal pump (one row covers all centrifugal pumps)',
    ),
    "reciprocating": CostParams(
        a=3.8696, b=0.3161, n=0.122, S_min=0.1, S_max=200.0, S_units="kW",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': reciprocating pump',
    ),
    "gear": CostParams(
        a=3.4771, b=0.135, n=0.1438, S_min=1.0, S_max=100.0, S_units="kW",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': positive displacement pump',
    ),
}

COMPRESSOR_COSTS = {
    "centrifugal": CostParams(
        a=580000.0, b=20000.0, n=0.6, S_min=75.0, S_max=30000.0, S_units="kW",
        base_year=2010,
        source=TOWLER + ': centrifugal compressor, driver power',
    ),
    "reciprocating": CostParams(
        a=260000.0, b=2700.0, n=0.75, S_min=93.0, S_max=16800.0, S_units="kW",
        base_year=2010,
        source=TOWLER + ': reciprocating compressor, driver power',
    ),
    "screw": CostParams(
        a=5.0355, b=-1.8002, n=0.8253, S_min=18.0, S_max=950.0, S_units="kW",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': rotary compressor, fluid power',
    ),
}

SEPARATOR_COSTS = {
    "mixer_settler": CostParams(
        a=4.7116, b=0.4479, n=0.0004, S_min=0.04, S_max=60.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': mixer/settler',
    ),
    "centrifuge_disk": CostParams(
        a=57000.0, b=480000.0, n=0.7, S_min=0.26, S_max=0.49, S_units="m (bowl diameter)",
        base_year=2010,
        source=TOWLER + ': high-speed disk centrifuge',
    ),
    "filter_rotary": CostParams(
        a=4.8123, b=0.2858, n=0.042, S_min=0.9, S_max=300.0, S_units="m²",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': disc and drum filter',
    ),
    "extraction_column": CostParams(
        a=3.4974, b=0.4485, n=0.1074, S_min=0.3, S_max=520.0, S_units="m³",
        base_year=2001, form="log_quadratic",
        source=TURTON + ': tower shell (column internals not included)',
    ),
}


# =============================================================================
# Core Cost Functions (JAX-differentiable)
# =============================================================================

VALID_EXPONENT_RANGE = (0.3, 1.2)


def validate_cost_params(params: CostParams, size: float | Array) -> dict:
    """Validate cost correlation parameters and sizing.

    Args:
        params: Cost correlation parameters
        size: Equipment size/capacity

    Returns:
        Dictionary with validation flags:
        - 'exponent_valid': Whether the scaling exponent is in the typical
          [0.3, 1.2] range
        - 'size_in_range': Whether size is within S_min to S_max
        - 'exponent': The scaling exponent: ``n`` for a power law, the local
          exponent ``K2 + 2 K3 log10 S`` for a log-quadratic correlation
        - 'size_range': (S_min, S_max)
    """
    size_val = float(size) if hasattr(size, '__float__') else size
    if params.form == "log_quadratic":
        exponent = float(params.exponent(size_val))
    else:
        exponent = params.n
    return {
        'exponent_valid': VALID_EXPONENT_RANGE[0] <= exponent <= VALID_EXPONENT_RANGE[1],
        'size_in_range': params.S_min <= size_val <= params.S_max,
        'exponent': exponent,
        'size_range': (params.S_min, params.S_max),
    }


def equipment_cost(
    size: Array,
    params: CostParams,
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Calculate purchased equipment cost from a cost correlation.

    C = C_base(S) * (CEPCI_target / CEPCI_base), where C_base is the
    power-law or log-quadratic form named by ``params.form``.

    Args:
        size: Equipment size/capacity parameter
        params: Cost correlation parameters
        target_year: Year for cost escalation

    Returns:
        Purchased equipment cost in target year dollars

    Warns:
        CostRangeWarning: if a concrete ``size`` lies outside
        ``[params.S_min, params.S_max]``. The extrapolated cost is still
        returned; traced sizes (under ``jit``/``grad``) are not checked.
    """
    _check_size_range(size, params)
    return escalate_cost(params.base_cost(size), params.base_year, target_year)


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


def equipment_cost_log_quadratic(
    size: Array,
    K1: Array,
    K2: Array,
    K3: Array,
    cepci_ratio: Array = jnp.array(1.0),
) -> Array:
    """Turton's purchased-cost form, differentiable in every argument.

    log10 C = K1 + K2 log10 S + K3 (log10 S)**2

    Args:
        size: Equipment size/capacity
        K1, K2, K3: Coefficients (Turton et al., Table A.1)
        cepci_ratio: Cost escalation ratio (CEPCI_target/CEPCI_base)

    Returns:
        Equipment cost

    Example:
        Floating-head exchanger, 100 m2, in 2001 dollars:

        >>> round(float(equipment_cost_log_quadratic(100.0, 4.8306, -0.8509, 0.3187)))
        25328
    """
    x = jnp.log10(size)
    return jnp.power(10.0, K1 + K2 * x + K3 * x * x) * cepci_ratio


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
        power: Pump shaft power (kW)
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
        power: Compressor power (kW): driver power for centrifugal and
            reciprocating, fluid power for screw (see each entry's ``source``)
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
        volume: Column shell volume (m³); tray area per tray (m²) for
            ``sieve_tray``/``valve_tray``
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
        size: Size parameter: m³ for a mixer-settler or extraction column,
            bowl diameter in m for a disk centrifuge, filter area in m² for a
            rotary filter (see ``S_units`` on each entry)
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


# =============================================================================
# Installed piping cost per unit length (economic pipe diameter)
# =============================================================================

#: Diameter exponents of carbon-steel pipe cost per unit length:
#: ``(n below 1 in, n above 1 in)``. Cost per length grows about as the first
#: power of the diameter up to 1 in and as the 1.5 power above it -- the
#: classical assumption behind the economic-pipe-diameter formulas of
#: Genereaux (1937), Ind. Eng. Chem. 29, 385, and Peters, Timmerhaus and
#: West, *Plant Design and Economics for Chemical Engineers*, 5e (2003),
#: Ch. 12 (whose turbulent optimum ``D_opt ~ q^0.45 rho^0.13`` follows from
#: ``n = 1.5``).
PIPE_COST_EXPONENTS = (1.0, 1.5)

#: The diameter (m) where the exponent changes, 1 in.
PIPE_COST_BREAK_DIAMETER = 0.0254


def _pipe_size_factor(D, n_small, n_large, D_break):
    x = jnp.asarray(D, dtype=jnp.float64) / D_break
    return jnp.where(x < 1.0, x**n_small, x**n_large)


def pipe_cost_per_length(
    D: Array | float,
    C_ref: Array | float,
    D_ref: Array | float,
    base_year: int,
    target_year: int = DEFAULT_CURRENT_YEAR,
    exponents: tuple[float, float] = PIPE_COST_EXPONENTS,
    D_break: float = PIPE_COST_BREAK_DIAMETER,
) -> Array:
    """Installed cost of carbon-steel pipe per metre as a function of diameter.

    ``C(D) = C_ref g(D)/g(D_ref) x CEPCI(target)/CEPCI(base)`` with the
    piecewise power law ``g(D) = (D/1 in)^n``, ``n = 1.0`` below 1 in and
    ``1.5`` above (:data:`PIPE_COST_EXPONENTS`, Genereaux 1937; Peters,
    Timmerhaus and West 5e Ch. 12). ``g`` is continuous at 1 in, and ``C`` is
    differentiable in ``D`` (except at the 1-in break), so it can sit in an
    economic-pipe-diameter objective next to the pumping cost.

    **The cost level is an input.** ``C_ref`` is the installed cost per metre
    (pipe, fittings share, supports, erection -- whatever your basis
    includes) of one reference size in ``base_year`` dollars, from a vendor
    quote, a company cost database or your textbook's table. difflow does
    not bundle an absolute $/m figure because none could be verified against
    its source; only the diameter scaling and the CEPCI escalation
    (:func:`difflow.economics.escalate_cost`) are supplied here.

    Args:
        D: Inside diameter (m).
        C_ref: Installed cost per metre ($/m, ``base_year`` dollars) at ``D_ref``.
        D_ref: Reference inside diameter (m).
        base_year: CEPCI year of ``C_ref``.
        target_year: CEPCI year to escalate to.
        exponents: ``(n below D_break, n above D_break)``.
        D_break: Diameter where the exponent changes (m), default 1 in.

    Returns:
        Installed cost per metre ($/m, ``target_year`` dollars).

    Example:
        >>> # a quote of $150/m for 2-in pipe in 2019 -> 4-in pipe, same year
        >>> c = pipe_cost_per_length(0.1023, 150.0, 0.0525, 2019, 2019)
        >>> round(float(c / 150.0), 4)      # (102.3/52.5)^1.5
        2.72
    """
    n_small, n_large = exponents
    ratio = (_pipe_size_factor(D, n_small, n_large, D_break)
             / _pipe_size_factor(D_ref, n_small, n_large, D_break))
    return escalate_cost(jnp.asarray(C_ref, dtype=jnp.float64) * ratio, base_year, target_year)


def pipe_cost_nps(
    nps,
    schedule="40",
    *,
    C_ref: float,
    nps_ref,
    base_year: int,
    schedule_ref="40",
    target_year: int = DEFAULT_CURRENT_YEAR,
    length: Array | float = 1.0,
    exponents: tuple[float, float] = PIPE_COST_EXPONENTS,
) -> Array:
    """Installed carbon-steel pipe cost for a nominal size and schedule.

    The diameter dependence is :func:`pipe_cost_per_length` on the inside
    diameters of ``nps`` and ``nps_ref`` in the reference schedule; the
    schedule dependence is the ratio of metal mass per metre at the same
    NPS, ``W(nps, schedule) / W(nps, schedule_ref)``
    (:func:`difflow.fluids.pipe_weight_per_length`, the ASME B36.10M
    plain-end weight). Scaling by metal mass is an approximation: it treats
    the whole installed cost as proportional to the steel, while labour for
    erection and welding grows more slowly with wall thickness, so a heavier
    schedule is somewhat over-costed.

    Args:
        nps: Nominal pipe size (in), e.g. ``4`` or ``"1-1/2"``.
        schedule: Schedule of the pipe being costed.
        C_ref: Installed cost per metre ($/m, ``base_year`` dollars) of
            ``nps_ref`` in ``schedule_ref``. User-supplied, see
            :func:`pipe_cost_per_length`.
        nps_ref: Nominal size of the reference cost.
        base_year: CEPCI year of ``C_ref``.
        schedule_ref: Schedule of the reference cost (default 40).
        target_year: CEPCI year to escalate to.
        length: Pipe length (m); the result is the cost of this length.
        exponents: Diameter exponents (below, above 1 in).

    Returns:
        Installed cost ($, ``target_year`` dollars) of ``length`` metres.

    Raises:
        KeyError: A size/schedule pair not in ASME B36.10M (e.g. 22-in Sch 40).

    Example:
        >>> c40 = pipe_cost_nps(4, "40", C_ref=200.0, nps_ref=4, base_year=2019, target_year=2019)
        >>> c80 = pipe_cost_nps(4, "80", C_ref=200.0, nps_ref=4, base_year=2019, target_year=2019)
        >>> round(float(c40), 6), round(float(c80 / c40), 3)   # 4-in Sch 80 / Sch 40 steel mass
        (200.0, 1.389)
    """
    from difflow.fluids import pipe_inside_diameter, pipe_weight_per_length

    D = pipe_inside_diameter(nps, schedule_ref)
    D_ref = pipe_inside_diameter(nps_ref, schedule_ref)
    mass_ratio = pipe_weight_per_length(nps, schedule) / pipe_weight_per_length(nps, schedule_ref)
    per_m = pipe_cost_per_length(D, C_ref, D_ref, base_year, target_year, exponents)
    return per_m * mass_ratio * jnp.asarray(length, dtype=jnp.float64)
