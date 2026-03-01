"""Cost indices for equipment cost escalation.

This module provides Chemical Engineering Plant Cost Index (CEPCI) values
and utilities for escalating equipment costs across time periods.

All functions are JAX-compatible for automatic differentiation.
"""

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple


class CEPCIData(NamedTuple):
    """CEPCI index data for a given year."""
    year: int
    index: float
    equipment: float  # Equipment sub-index
    construction_labor: float
    buildings: float
    engineering: float


# Historical CEPCI values (Chemical Engineering magazine)
# Source: Chemical Engineering Plant Cost Index
CEPCI_HISTORICAL = {
    2000: CEPCIData(2000, 394.1, 374.9, 353.2, 353.9, 352.6),
    2001: CEPCIData(2001, 394.3, 374.6, 363.5, 357.7, 355.2),
    2002: CEPCIData(2002, 395.6, 374.1, 371.2, 363.5, 359.5),
    2003: CEPCIData(2003, 402.0, 379.9, 382.2, 370.1, 364.7),
    2004: CEPCIData(2004, 444.2, 426.4, 400.1, 379.7, 377.7),
    2005: CEPCIData(2005, 468.2, 449.4, 424.1, 394.1, 391.3),
    2006: CEPCIData(2006, 499.6, 480.4, 450.1, 408.1, 404.8),
    2007: CEPCIData(2007, 525.4, 505.2, 478.3, 424.3, 420.3),
    2008: CEPCIData(2008, 575.4, 559.5, 512.7, 445.5, 442.8),
    2009: CEPCIData(2009, 521.9, 499.7, 502.5, 456.2, 453.3),
    2010: CEPCIData(2010, 550.8, 532.3, 513.6, 461.4, 459.2),
    2011: CEPCIData(2011, 585.7, 567.5, 542.2, 475.1, 471.4),
    2012: CEPCIData(2012, 584.6, 569.4, 538.4, 481.8, 477.6),
    2013: CEPCIData(2013, 567.3, 551.2, 532.6, 489.4, 485.5),
    2014: CEPCIData(2014, 576.1, 560.4, 538.5, 494.3, 491.2),
    2015: CEPCIData(2015, 556.8, 539.6, 531.2, 497.7, 494.6),
    2016: CEPCIData(2016, 541.7, 523.8, 525.3, 501.2, 498.2),
    2017: CEPCIData(2017, 567.5, 551.0, 539.8, 508.9, 505.7),
    2018: CEPCIData(2018, 603.1, 587.4, 568.1, 520.3, 517.1),
    2019: CEPCIData(2019, 607.5, 591.2, 576.3, 527.4, 524.1),
    2020: CEPCIData(2020, 596.2, 578.5, 582.4, 533.5, 530.2),
    2021: CEPCIData(2021, 708.0, 695.2, 640.8, 548.1, 544.7),
    2022: CEPCIData(2022, 816.0, 802.5, 712.3, 578.4, 574.5),
    2023: CEPCIData(2023, 797.9, 782.1, 728.5, 598.2, 594.1),
    2024: CEPCIData(2024, 800.0, 784.0, 735.0, 605.0, 601.0),  # Estimated
    2025: CEPCIData(2025, 810.0, 794.0, 742.0, 612.0, 608.0),  # Estimated
    2026: CEPCIData(2026, 820.0, 804.0, 749.0, 619.0, 615.0),  # Estimated
}

# Default base year for cost correlations (common in literature)
DEFAULT_BASE_YEAR = 2019
DEFAULT_CURRENT_YEAR = 2026


def cepci_available_years() -> list[int]:
    """Return list of years with CEPCI data.

    Returns:
        Sorted list of available years
    """
    return sorted(CEPCI_HISTORICAL.keys())


def estimate_cepci(year: int) -> float:
    """Estimate CEPCI for a year beyond available data.

    Uses linear extrapolation from the last 3 data points.

    Args:
        year: Target year

    Returns:
        Estimated CEPCI index value
    """
    if year in CEPCI_HISTORICAL:
        return CEPCI_HISTORICAL[year].index
    available = sorted(CEPCI_HISTORICAL.keys())
    # Use last 3 years for linear extrapolation
    last_years = available[-3:]
    last_values = [CEPCI_HISTORICAL[y].index for y in last_years]
    # Simple linear fit: slope from last 3 points
    n = len(last_years)
    mean_y = sum(last_years) / n
    mean_v = sum(last_values) / n
    num = sum((last_years[i] - mean_y) * (last_values[i] - mean_v) for i in range(n))
    den = sum((last_years[i] - mean_y) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    intercept = mean_v - slope * mean_y
    return slope * year + intercept


def get_cepci(year: int) -> float:
    """Get CEPCI index for a given year.

    Args:
        year: Calendar year (2000-2026 available)

    Returns:
        CEPCI index value

    Raises:
        ValueError: If year not in database
    """
    if year not in CEPCI_HISTORICAL:
        available = sorted(CEPCI_HISTORICAL.keys())
        raise ValueError(
            f"CEPCI data not available for year {year}. "
            f"Available years: {available[0]}-{available[-1]}"
        )
    return CEPCI_HISTORICAL[year].index


def escalate_cost(
    base_cost: Array,
    base_year: int = DEFAULT_BASE_YEAR,
    target_year: int = DEFAULT_CURRENT_YEAR,
) -> Array:
    """Escalate equipment cost from base year to target year using CEPCI.

    Cost_target = Cost_base * (CEPCI_target / CEPCI_base)

    Args:
        base_cost: Cost in base year dollars
        base_year: Year of base cost
        target_year: Year to escalate to

    Returns:
        Cost escalated to target year dollars
    """
    cepci_base = get_cepci(base_year)
    cepci_target = get_cepci(target_year)
    return base_cost * (cepci_target / cepci_base)


def escalate_cost_ratio(
    base_cost: Array,
    cepci_ratio: Array,
) -> Array:
    """Escalate cost using a provided CEPCI ratio.

    This version accepts the ratio directly for JAX differentiation.

    Args:
        base_cost: Cost in base year dollars
        cepci_ratio: Ratio of (CEPCI_target / CEPCI_base)

    Returns:
        Escalated cost
    """
    return base_cost * cepci_ratio


def inflation_factor(
    base_year: int,
    target_year: int,
    annual_rate: float = 0.025,
) -> float:
    """Calculate inflation factor between years.

    Factor = (1 + rate)^(target - base)

    Args:
        base_year: Starting year
        target_year: Ending year
        annual_rate: Annual inflation rate (default 2.5%)

    Returns:
        Multiplicative inflation factor
    """
    years = target_year - base_year
    return (1 + annual_rate) ** years


def inflation_factor_continuous(
    years: Array,
    annual_rate: Array,
) -> Array:
    """JAX-differentiable continuous inflation factor.

    Args:
        years: Number of years (can be fractional)
        annual_rate: Annual inflation rate

    Returns:
        Multiplicative inflation factor
    """
    return jnp.power(1.0 + annual_rate, years)


# Common cost index ratios (pre-computed for efficiency)
CEPCI_RATIOS = {
    (2019, 2026): get_cepci(2026) / get_cepci(2019),
    (2019, 2024): get_cepci(2024) / get_cepci(2019),
    (2018, 2026): get_cepci(2026) / get_cepci(2018),
    (2018, 2024): get_cepci(2024) / get_cepci(2018),
    (2015, 2026): get_cepci(2026) / get_cepci(2015),
    (2015, 2024): get_cepci(2024) / get_cepci(2015),
    (2010, 2026): get_cepci(2026) / get_cepci(2010),
    (2010, 2024): get_cepci(2024) / get_cepci(2010),
    (2000, 2026): get_cepci(2026) / get_cepci(2000),
    (2000, 2024): get_cepci(2024) / get_cepci(2000),
}
