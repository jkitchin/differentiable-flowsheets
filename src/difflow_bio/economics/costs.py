"""Cost estimation for biopharmaceutical manufacturing.

Provides economic analysis including:
- Capital cost (CAPEX) estimation
- Operating cost (OPEX) estimation
- Cost of goods sold (COGS)
- Profitability metrics

All functions support JAX arrays for differentiable optimization.

References:
    Farid SS et al. (2007). Biotechnol Prog 23:3.
    Pollock J et al. (2013). Biotechnol Bioeng 110:206.
    Kelley B (2009). Biotechnol Prog 25:1512.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow_bio.database import get_resin, get_bio_membrane


# =============================================================================
# Cost Data Structures
# =============================================================================

@dataclass
class ConsumableCosts:
    """Consumable cost data for biopharmaceutical manufacturing.

    Attributes:
        resins: Chromatography resin costs (USD/L)
        membranes: Membrane costs (USD/m2)
        media: Cell culture media costs (USD/L)
        buffers: Buffer costs (USD/L)
        bags: Single-use bag costs (USD per bag)
    """
    resins: dict[str, float] = field(default_factory=lambda: {
        "MabSelect_SuRe": 15000.0,
        "MabSelect_PrismA": 20000.0,
        "Protein_A_Sepharose": 12000.0,
        "SP_Sepharose_FF": 3000.0,
        "Capto_S_ImpAct": 5000.0,
        "Q_Sepharose_FF": 3000.0,
        "Capto_Q": 4500.0,
        "Superdex_200": 8000.0,
    })

    membranes: dict[str, float] = field(default_factory=lambda: {
        "Pellicon_3_30kDa": 400.0,
        "Pellicon_3_10kDa": 400.0,
        "Biomax_50kDa": 350.0,
        "Kvick_30kDa": 380.0,
        "Pellicon_3_0.45um": 300.0,
        "Pellicon_3_0.22um": 320.0,
        "Viresolve_Pro": 2000.0,
        "Planova_20N": 2500.0,
    })

    media: dict[str, float] = field(default_factory=lambda: {
        "CHO_chemically_defined": 80.0,  # USD/L
        "CHO_serum_free": 50.0,
        "HEK293_media": 100.0,
        "feed_concentrate": 200.0,
    })

    buffers: dict[str, float] = field(default_factory=lambda: {
        "PBS": 5.0,
        "equilibration": 8.0,
        "elution": 15.0,
        "wash": 6.0,
        "CIP_NaOH": 3.0,
        "WFI": 0.5,
    })

    bags: dict[str, float] = field(default_factory=lambda: {
        "50L": 150.0,
        "200L": 400.0,
        "500L": 800.0,
        "1000L": 1500.0,
        "2000L": 2500.0,
    })


@dataclass
class EquipmentCosts:
    """Equipment cost data for biopharmaceutical manufacturing.

    Base costs at reference scale, to be adjusted with scaling exponent.

    Attributes:
        bioreactors: Bioreactor costs by type and scale
        chromatography: Chromatography system costs
        filtration: Filtration system costs
        utilities: Utility equipment costs
    """
    bioreactors: dict[str, float] = field(default_factory=lambda: {
        # Stainless steel bioreactors (USD)
        "500L_SS": 500000.0,
        "1000L_SS": 800000.0,
        "2000L_SS": 1200000.0,
        "5000L_SS": 2500000.0,
        "10000L_SS": 4000000.0,
        "15000L_SS": 5500000.0,
        # Single-use bioreactors (USD)
        "50L_SU": 50000.0,
        "200L_SU": 80000.0,
        "500L_SU": 150000.0,
        "1000L_SU": 250000.0,
        "2000L_SU": 400000.0,
    })

    chromatography: dict[str, float] = field(default_factory=lambda: {
        # AKTA systems (USD)
        "pilot_10L_column": 150000.0,
        "pilot_50L_column": 300000.0,
        "process_100L_column": 500000.0,
        "process_500L_column": 1000000.0,
    })

    filtration: dict[str, float] = field(default_factory=lambda: {
        # TFF systems (USD)
        "TFF_pilot_1m2": 50000.0,
        "TFF_process_10m2": 150000.0,
        "TFF_process_50m2": 400000.0,
        # Depth filtration
        "depth_filter_train": 30000.0,
        # Virus filtration
        "VF_skid": 100000.0,
    })


@dataclass
class OperatingCosts:
    """Operating cost parameters.

    Attributes:
        labor_rate: Labor cost (USD/hour/FTE)
        electricity_rate: Electricity cost (USD/kWh)
        steam_rate: Steam cost (USD/kg)
        wfi_rate: Water for injection cost (USD/L)
        waste_disposal_rate: Waste disposal cost (USD/L)
        maintenance_factor: Annual maintenance as fraction of CAPEX
    """
    labor_rate: float = 75.0  # USD/hour (loaded)
    electricity_rate: float = 0.10  # USD/kWh
    steam_rate: float = 0.02  # USD/kg
    wfi_rate: float = 0.50  # USD/L
    waste_disposal_rate: float = 2.0  # USD/L
    maintenance_factor: float = 0.05  # 5% of CAPEX


# =============================================================================
# Capital Cost Estimation
# =============================================================================

def estimate_bioreactor_capex(
    volume_L: float,
    bioreactor_type: str = "stainless_steel",
    n_reactors: int = 1,
) -> float:
    """Estimate bioreactor capital cost.

    Uses power law scaling from reference sizes.

    Args:
        volume_L: Bioreactor volume (L)
        bioreactor_type: 'stainless_steel' or 'single_use'
        n_reactors: Number of bioreactors

    Returns:
        Capital cost (USD)
    """
    # Scaling parameters
    scale_exp = 0.6  # Economies of scale exponent

    if bioreactor_type == "single_use":
        # Reference: 1000L SU = $250,000
        ref_volume = 1000.0
        ref_cost = 250000.0
    else:  # stainless_steel
        # Reference: 2000L SS = $1,200,000
        ref_volume = 2000.0
        ref_cost = 1200000.0

    cost = ref_cost * (volume_L / ref_volume) ** scale_exp
    return cost * n_reactors


def estimate_chromatography_capex(
    column_volume_L: float,
    n_columns: int = 1,
    include_skid: bool = True,
) -> float:
    """Estimate chromatography capital cost.

    Includes column and associated equipment.

    Args:
        column_volume_L: Column volume (L)
        n_columns: Number of columns
        include_skid: Include chromatography skid cost

    Returns:
        Capital cost (USD)
    """
    scale_exp = 0.5

    # Reference: 50L column system = $300,000
    ref_volume = 50.0
    ref_cost = 300000.0

    column_cost = ref_cost * (column_volume_L / ref_volume) ** scale_exp

    if include_skid:
        # Skid cost scales with throughput
        skid_cost = 200000.0 * (column_volume_L / ref_volume) ** 0.4
        column_cost += skid_cost

    return column_cost * n_columns


def estimate_filtration_capex(
    membrane_area_m2: float,
    filtration_type: str = "tff",
) -> float:
    """Estimate filtration capital cost.

    Args:
        membrane_area_m2: Membrane area (m2)
        filtration_type: 'tff', 'depth', 'virus'

    Returns:
        Capital cost (USD)
    """
    scale_exp = 0.5

    if filtration_type == "tff":
        # Reference: 10 m2 TFF = $150,000
        ref_area = 10.0
        ref_cost = 150000.0
    elif filtration_type == "virus":
        # Reference: 1 m2 VF = $100,000 (skid)
        ref_area = 1.0
        ref_cost = 100000.0
    else:  # depth
        ref_area = 1.0
        ref_cost = 30000.0

    return ref_cost * (membrane_area_m2 / ref_area) ** scale_exp


def estimate_facility_capex(
    production_scale: str = "clinical",
    include_utilities: bool = True,
) -> float:
    """Estimate facility capital cost.

    Args:
        production_scale: 'clinical', 'commercial_small', 'commercial_large'
        include_utilities: Include utility infrastructure

    Returns:
        Capital cost (USD)
    """
    facility_costs = {
        "clinical": 20_000_000.0,  # $20M
        "commercial_small": 100_000_000.0,  # $100M
        "commercial_large": 300_000_000.0,  # $300M
    }

    base_cost = facility_costs.get(production_scale, 50_000_000.0)

    if include_utilities:
        # Add 20% for utilities infrastructure
        base_cost *= 1.2

    return base_cost


def estimate_total_capex(
    bioreactor_volume_L: float,
    n_bioreactors: int = 1,
    column_volume_L: float = 10.0,
    n_chromatography_steps: int = 3,
    tff_area_m2: float = 10.0,
    production_scale: str = "clinical",
) -> dict[str, float]:
    """Estimate total capital cost for mAb facility.

    Args:
        bioreactor_volume_L: Bioreactor volume (L)
        n_bioreactors: Number of bioreactors
        column_volume_L: Average column volume (L)
        n_chromatography_steps: Number of chromatography steps
        tff_area_m2: TFF membrane area (m2)
        production_scale: Facility scale

    Returns:
        Dictionary of CAPEX components
    """
    bioreactor = estimate_bioreactor_capex(bioreactor_volume_L, n_reactors=n_bioreactors)
    chromatography = estimate_chromatography_capex(column_volume_L, n_chromatography_steps)
    tff = estimate_filtration_capex(tff_area_m2, "tff")
    vf = estimate_filtration_capex(1.0, "virus")
    facility = estimate_facility_capex(production_scale)

    total = bioreactor + chromatography + tff + vf + facility

    return {
        "bioreactors": bioreactor,
        "chromatography": chromatography,
        "tff": tff,
        "virus_filtration": vf,
        "facility": facility,
        "total": total,
    }


# =============================================================================
# Operating Cost Estimation
# =============================================================================

def estimate_resin_cost(
    resin_name: str,
    column_volume_L: float,
    n_cycles: int,
    resin_lifetime_cycles: int = 200,
) -> float:
    """Estimate annual resin cost.

    Args:
        resin_name: Resin name from database
        column_volume_L: Column volume (L)
        n_cycles: Annual number of cycles
        resin_lifetime_cycles: Expected resin lifetime (cycles)

    Returns:
        Annual resin cost (USD/year)
    """
    resin = get_resin(resin_name)
    cost_per_L = resin.cost_usd_L

    # Resin replacement per year
    replacements_per_year = n_cycles / resin_lifetime_cycles

    return cost_per_L * column_volume_L * replacements_per_year


def estimate_membrane_cost(
    membrane_name: str,
    area_m2: float,
    n_cycles: int,
    membrane_lifetime_cycles: int = 100,
) -> float:
    """Estimate annual membrane cost.

    Args:
        membrane_name: Membrane name from database
        area_m2: Membrane area (m2)
        n_cycles: Annual number of cycles
        membrane_lifetime_cycles: Expected lifetime (cycles)

    Returns:
        Annual membrane cost (USD/year)
    """
    membrane = get_bio_membrane(membrane_name)
    cost_per_m2 = membrane.cost_usd_m2

    # Membrane replacement per year
    replacements_per_year = n_cycles / membrane_lifetime_cycles

    return cost_per_m2 * area_m2 * replacements_per_year


def estimate_media_cost(
    bioreactor_volume_L: float,
    n_batches: int,
    media_cost_per_L: float = 80.0,
    feed_volume_fraction: float = 0.3,
    feed_cost_per_L: float = 200.0,
) -> float:
    """Estimate annual media and feed cost.

    Args:
        bioreactor_volume_L: Bioreactor working volume (L)
        n_batches: Annual number of batches
        media_cost_per_L: Basal media cost (USD/L)
        feed_volume_fraction: Feed volume as fraction of bioreactor
        feed_cost_per_L: Feed concentrate cost (USD/L)

    Returns:
        Annual media cost (USD/year)
    """
    # Basal media
    basal = media_cost_per_L * bioreactor_volume_L * n_batches

    # Feed
    feed_volume = bioreactor_volume_L * feed_volume_fraction
    feed = feed_cost_per_L * feed_volume * n_batches

    return basal + feed


def estimate_labor_cost(
    n_operators: int = 10,
    n_shifts: int = 3,
    hours_per_year: float = 2000,
    labor_rate: float = 75.0,
) -> float:
    """Estimate annual labor cost.

    Args:
        n_operators: Operators per shift
        n_shifts: Number of shifts (3 for 24/7)
        hours_per_year: Hours per operator per year
        labor_rate: Loaded labor rate (USD/hour)

    Returns:
        Annual labor cost (USD/year)
    """
    return n_operators * n_shifts * hours_per_year * labor_rate


def estimate_utilities_cost(
    bioreactor_volume_L: float,
    n_batches: int,
    electricity_rate: float = 0.10,
    wfi_rate: float = 0.50,
) -> float:
    """Estimate annual utilities cost.

    Args:
        bioreactor_volume_L: Bioreactor volume (L)
        n_batches: Annual batches
        electricity_rate: USD/kWh
        wfi_rate: USD/L

    Returns:
        Annual utilities cost (USD/year)
    """
    # Rough estimates based on production scale
    # Electricity: ~0.5 kWh/L bioreactor/batch
    electricity = 0.5 * bioreactor_volume_L * n_batches * electricity_rate

    # WFI: ~10x bioreactor volume per batch
    wfi = 10.0 * bioreactor_volume_L * n_batches * wfi_rate

    return electricity + wfi


def estimate_total_opex(
    annual_batches: int,
    bioreactor_volume_L: float,
    column_volume_L: float = 10.0,
    n_chromatography_steps: int = 3,
    tff_area_m2: float = 10.0,
    capex: float = 0.0,
) -> dict[str, float]:
    """Estimate total annual operating cost.

    Args:
        annual_batches: Number of batches per year
        bioreactor_volume_L: Bioreactor volume (L)
        column_volume_L: Average column volume (L)
        n_chromatography_steps: Number of chromatography steps
        tff_area_m2: TFF membrane area (m2)
        capex: Total CAPEX for maintenance calculation

    Returns:
        Dictionary of OPEX components
    """
    # Consumables
    # Assume Protein A is the capture resin
    resin = estimate_resin_cost(
        "MabSelect_SuRe",
        column_volume_L,
        annual_batches,
    ) * n_chromatography_steps

    membrane = estimate_membrane_cost(
        "Pellicon_3_30kDa",
        tff_area_m2,
        annual_batches,
    )

    media = estimate_media_cost(bioreactor_volume_L, annual_batches)

    # Labor
    labor = estimate_labor_cost()

    # Utilities
    utilities = estimate_utilities_cost(bioreactor_volume_L, annual_batches)

    # Maintenance (5% of CAPEX)
    maintenance = capex * 0.05 if capex > 0 else 0.0

    total = resin + membrane + media + labor + utilities + maintenance

    return {
        "consumables_resin": resin,
        "consumables_membrane": membrane,
        "consumables_media": media,
        "labor": labor,
        "utilities": utilities,
        "maintenance": maintenance,
        "total": total,
    }


# =============================================================================
# Profitability Analysis
# =============================================================================

def calculate_cogs(
    opex: float,
    annual_production_g: float,
) -> float:
    """Calculate cost of goods sold per gram.

    Args:
        opex: Annual operating cost (USD/year)
        annual_production_g: Annual production (g/year)

    Returns:
        COGS (USD/g)
    """
    return opex / annual_production_g


def cost_per_gram(
    bioreactor_volume_L: float,
    titer_g_L: float,
    annual_batches: int,
    yield_overall: float = 0.7,
    capex: float = 0.0,
    depreciation_years: int = 10,
) -> dict[str, float]:
    """Calculate cost per gram of product.

    Args:
        bioreactor_volume_L: Bioreactor volume (L)
        titer_g_L: Product titer (g/L)
        annual_batches: Batches per year
        yield_overall: Overall DSP yield (0-1)
        capex: Total CAPEX (for depreciation)
        depreciation_years: Depreciation period

    Returns:
        Dictionary with cost breakdown per gram
    """
    # Annual production
    harvest_per_batch = bioreactor_volume_L * titer_g_L
    product_per_batch = harvest_per_batch * yield_overall
    annual_production = product_per_batch * annual_batches

    # Get OPEX estimate
    opex = estimate_total_opex(
        annual_batches,
        bioreactor_volume_L,
        capex=capex,
    )

    # COGS
    cogs = opex["total"] / annual_production

    # Add depreciation
    depreciation_per_year = capex / depreciation_years if capex > 0 else 0.0
    depreciation_per_g = depreciation_per_year / annual_production

    total_per_g = cogs + depreciation_per_g

    return {
        "annual_production_g": annual_production,
        "opex_per_g": cogs,
        "depreciation_per_g": depreciation_per_g,
        "total_per_g": total_per_g,
    }


def calculate_profit(
    revenue: float,
    opex: float,
    capex: float,
    tax_rate: float = 0.25,
    depreciation_years: int = 10,
) -> dict[str, float]:
    """Calculate profitability metrics.

    Args:
        revenue: Annual revenue (USD/year)
        opex: Annual operating cost (USD/year)
        capex: Total capital cost (USD)
        tax_rate: Corporate tax rate
        depreciation_years: Depreciation period

    Returns:
        Dictionary of profitability metrics
    """
    # EBITDA
    ebitda = revenue - opex

    # Depreciation (straight-line)
    depreciation = capex / depreciation_years

    # EBIT
    ebit = ebitda - depreciation

    # Taxes
    taxes = max(0.0, ebit * tax_rate)

    # Net income
    net_income = ebit - taxes

    # Cash flow (add back depreciation)
    cash_flow = net_income + depreciation

    # Simple payback
    payback = capex / cash_flow if cash_flow > 0 else float('inf')

    # ROI
    roi = net_income / capex if capex > 0 else 0.0

    return {
        "revenue": revenue,
        "opex": opex,
        "ebitda": ebitda,
        "depreciation": depreciation,
        "ebit": ebit,
        "taxes": taxes,
        "net_income": net_income,
        "cash_flow": cash_flow,
        "payback_years": payback,
        "roi": roi,
    }
