"""Cost estimation for REE separation processes.

Provides economic analysis including:
- REE product pricing (volatile market)
- Reagent and utility costs
- Capital and operating cost estimation
- Profitability metrics

All functions support JAX arrays for differentiable optimization.
"""

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow_ree.database import get_ree_database, get_extractant_database


# =============================================================================
# REE Product Pricing
# =============================================================================

@dataclass
class REEPricing:
    """REE product pricing model.

    REE prices are highly volatile and depend on:
    - Purity (higher purity = premium)
    - Form (oxide, metal, compounds)
    - Market conditions

    Attributes:
        base_prices: Base prices in USD/kg (oxide basis)
        purity_premium: Premium factor for high purity
        form_factors: Price multipliers for different forms
    """
    base_prices: dict[str, float] = field(default_factory=lambda: {
        "La": 5.0,
        "Ce": 2.0,
        "Pr": 85.0,
        "Nd": 120.0,
        "Sm": 15.0,
        "Eu": 35.0,
        "Gd": 55.0,
        "Tb": 1500.0,
        "Dy": 450.0,
        "Y": 35.0,
    })

    purity_premium: dict[str, float] = field(default_factory=lambda: {
        "99%": 1.0,
        "99.9%": 1.3,
        "99.99%": 2.0,
        "99.999%": 5.0,
    })

    form_factors: dict[str, float] = field(default_factory=lambda: {
        "oxide": 1.0,
        "metal": 3.0,
        "chloride": 0.9,
        "nitrate": 0.85,
        "carbonate": 0.8,
    })

    def get_price(
        self,
        element: str,
        purity: str = "99%",
        form: str = "oxide",
    ) -> float:
        """Get price for specific REE product.

        Args:
            element: REE symbol
            purity: Purity grade
            form: Product form

        Returns:
            Price in USD/kg
        """
        base = self.base_prices.get(element, 50.0)
        purity_mult = self.purity_premium.get(purity, 1.0)
        form_mult = self.form_factors.get(form, 1.0)
        return base * purity_mult * form_mult

    def get_price_array(
        self,
        elements: list[str],
        purity: str = "99%",
        form: str = "oxide",
    ) -> Array:
        """Get prices as JAX array.

        Args:
            elements: List of element symbols
            purity: Purity grade
            form: Product form

        Returns:
            JAX array of prices
        """
        prices = [self.get_price(e, purity, form) for e in elements]
        return jnp.array(prices)

    def update_prices(self, new_prices: dict[str, float]):
        """Update base prices (for market scenarios).

        Args:
            new_prices: Dictionary of new prices
        """
        self.base_prices.update(new_prices)


# =============================================================================
# Reagent Costs
# =============================================================================

@dataclass
class ReagentCosts:
    """Reagent cost data for REE processing.

    Attributes:
        extractants: Extractant costs (USD/kg)
        acids: Acid costs (USD/kg)
        bases: Base costs (USD/kg)
        precipitants: Precipitant costs (USD/kg)
        diluents: Diluent costs (USD/L)
    """
    extractants: dict[str, float] = field(default_factory=lambda: {
        "D2EHPA": 8.0,
        "PC88A": 12.0,
        "Cyanex272": 25.0,
        "TBP": 4.0,
    })

    acids: dict[str, float] = field(default_factory=lambda: {
        "HCl": 0.15,  # USD/kg (30% solution)
        "H2SO4": 0.10,
        "HNO3": 0.40,
    })

    bases: dict[str, float] = field(default_factory=lambda: {
        "NaOH": 0.50,
        "NH4OH": 0.35,
        "Na2CO3": 0.30,
    })

    precipitants: dict[str, float] = field(default_factory=lambda: {
        "oxalic_acid": 2.0,
        "Na2CO3": 0.30,
        "NH4OH": 0.35,
    })

    diluents: dict[str, float] = field(default_factory=lambda: {
        "kerosene": 1.0,
        "n-heptane": 2.5,
        "Shellsol_D70": 1.5,
    })

    def extractant_makeup_cost(
        self,
        extractant: str,
        loss_rate: float = 0.001,  # kg lost per kg REE processed
        ree_throughput: float = 1.0,  # kg REE/hour
    ) -> float:
        """Calculate extractant makeup cost.

        Args:
            extractant: Extractant name
            loss_rate: Extractant loss (kg/kg REE)
            ree_throughput: REE throughput (kg/hour)

        Returns:
            Makeup cost (USD/hour)
        """
        unit_cost = self.extractants.get(extractant, 10.0)
        return unit_cost * loss_rate * ree_throughput

    def acid_cost(
        self,
        acid: str,
        consumption: float,  # kg/hour
    ) -> float:
        """Calculate acid cost.

        Args:
            acid: Acid type
            consumption: Consumption rate (kg/hour)

        Returns:
            Cost (USD/hour)
        """
        unit_cost = self.acids.get(acid, 0.20)
        return unit_cost * consumption


# =============================================================================
# Operating Costs
# =============================================================================

@dataclass
class OperatingCosts:
    """Operating cost estimation.

    Attributes:
        labor_rate: Labor cost (USD/hour/worker)
        electricity_rate: Electricity cost (USD/kWh)
        steam_rate: Steam cost (USD/kg)
        cooling_water_rate: Cooling water cost (USD/m³)
        maintenance_factor: Maintenance as fraction of CAPEX
    """
    labor_rate: float = 35.0  # USD/hour
    electricity_rate: float = 0.08  # USD/kWh
    steam_rate: float = 0.02  # USD/kg
    cooling_water_rate: float = 0.05  # USD/m³
    maintenance_factor: float = 0.03  # 3% of CAPEX

    def labor_cost(
        self,
        n_operators: int,
        hours_per_year: float = 8000,
    ) -> float:
        """Calculate annual labor cost.

        Args:
            n_operators: Total number of operators (all shifts combined)
            hours_per_year: Operating hours per year

        Returns:
            Annual labor cost (USD/year)
        """
        # n_operators is total headcount across all shifts
        # (e.g., 6 total for a small REE plant with ~2 per shift)
        return self.labor_rate * n_operators * hours_per_year

    def utility_cost(
        self,
        electricity_kw: float,
        steam_kg_hr: float,
        cooling_m3_hr: float,
        hours_per_year: float = 8000,
    ) -> float:
        """Calculate annual utility cost.

        Args:
            electricity_kw: Power consumption (kW)
            steam_kg_hr: Steam consumption (kg/hr)
            cooling_m3_hr: Cooling water (m³/hr)
            hours_per_year: Operating hours

        Returns:
            Annual utility cost (USD/year)
        """
        elec = self.electricity_rate * electricity_kw * hours_per_year
        steam = self.steam_rate * steam_kg_hr * hours_per_year
        cooling = self.cooling_water_rate * cooling_m3_hr * hours_per_year
        return elec + steam + cooling


# =============================================================================
# Capital Cost Estimation
# =============================================================================

def estimate_capex(
    annual_ree_tonnes: float,
    n_stages_extraction: int,
    n_stages_scrubbing: int,
    n_stages_stripping: int,
    include_precipitation: bool = True,
    include_ce_removal: bool = False,
    year: int = 2024,
) -> dict[str, float]:
    """Estimate capital cost for REE separation plant.

    Uses factored estimation based on throughput and complexity.

    Args:
        annual_ree_tonnes: Annual REE production (tonnes/year)
        n_stages_extraction: Number of extraction stages
        n_stages_scrubbing: Number of scrubbing stages
        n_stages_stripping: Number of stripping stages
        include_precipitation: Include precipitation section
        include_ce_removal: Include Ce oxidation section
        year: Cost basis year (for CEPCI adjustment)

    Returns:
        Dictionary of capital cost components
    """
    # Base cost for mixer-settlers (USD per stage per tonne/year capacity)
    mixer_settler_base = 50000  # USD per stage

    # Scale factor (economies of scale)
    scale_exp = 0.6

    # Reference capacity
    ref_capacity = 1000  # tonnes/year

    # Scale factor
    scale = (annual_ree_tonnes / ref_capacity) ** scale_exp

    # Mixer-settler costs
    n_total_stages = n_stages_extraction + n_stages_scrubbing + n_stages_stripping
    mixer_settler_cost = mixer_settler_base * n_total_stages * scale

    # Tanks and vessels (20% of mixer-settlers)
    tanks_cost = mixer_settler_cost * 0.20

    # Pumps and piping (25% of mixer-settlers)
    pumps_piping_cost = mixer_settler_cost * 0.25

    # Instrumentation and controls (15% of equipment)
    equipment_subtotal = mixer_settler_cost + tanks_cost + pumps_piping_cost
    instrumentation_cost = equipment_subtotal * 0.15

    # Precipitation section
    if include_precipitation:
        precipitation_cost = 200000 * scale  # Reactors, filters, dryers
    else:
        precipitation_cost = 0.0

    # Ce removal section
    if include_ce_removal:
        ce_removal_cost = 150000 * scale
    else:
        ce_removal_cost = 0.0

    # Direct costs
    direct_cost = (
        equipment_subtotal +
        instrumentation_cost +
        precipitation_cost +
        ce_removal_cost
    )

    # Installation (40% of direct)
    installation_cost = direct_cost * 0.40

    # Indirect costs
    engineering_cost = direct_cost * 0.15
    contingency_cost = direct_cost * 0.20

    # Total CAPEX
    total_capex = direct_cost + installation_cost + engineering_cost + contingency_cost

    # CEPCI adjustment (base year 2020 = 596.2)
    cepci_2020 = 596.2
    cepci_current = {2020: 596.2, 2021: 708.0, 2022: 816.0, 2023: 800.0, 2024: 820.0}
    cepci_ratio = cepci_current.get(year, 820.0) / cepci_2020
    total_capex *= cepci_ratio

    return {
        "mixer_settlers": mixer_settler_cost * cepci_ratio,
        "tanks_vessels": tanks_cost * cepci_ratio,
        "pumps_piping": pumps_piping_cost * cepci_ratio,
        "instrumentation": instrumentation_cost * cepci_ratio,
        "precipitation": precipitation_cost * cepci_ratio,
        "ce_removal": ce_removal_cost * cepci_ratio,
        "installation": installation_cost * cepci_ratio,
        "engineering": engineering_cost * cepci_ratio,
        "contingency": contingency_cost * cepci_ratio,
        "total": total_capex,
    }


def estimate_opex(
    annual_ree_tonnes: float,
    capex: float,
    extractant: str = "D2EHPA",
    product_elements: list[str] = None,
) -> dict[str, float]:
    """Estimate annual operating cost.

    Args:
        annual_ree_tonnes: Annual REE production
        capex: Total capital cost
        extractant: Primary extractant
        product_elements: Elements being produced

    Returns:
        Dictionary of operating cost components
    """
    if product_elements is None:
        product_elements = ["Nd", "Dy"]

    reagents = ReagentCosts()
    opex = OperatingCosts()

    # Reagent costs (rough estimates based on throughput)
    # Extractant makeup: ~$2/kg REE
    extractant_cost = annual_ree_tonnes * 1000 * 2.0

    # Acid consumption: ~$1/kg REE
    acid_cost = annual_ree_tonnes * 1000 * 1.0

    # Base consumption: ~$0.5/kg REE
    base_cost = annual_ree_tonnes * 1000 * 0.5

    # Precipitant: ~$3/kg REE (oxalic acid)
    precipitant_cost = annual_ree_tonnes * 1000 * 3.0

    # Labor (estimate 6 operators per shift)
    labor_cost = opex.labor_cost(6, 8000)

    # Utilities (estimate based on throughput)
    electricity_kw = 50 + annual_ree_tonnes * 0.5  # kW
    steam_kg_hr = annual_ree_tonnes * 0.1  # kg/hr
    cooling_m3_hr = annual_ree_tonnes * 0.2  # m³/hr
    utility_cost = opex.utility_cost(electricity_kw, steam_kg_hr, cooling_m3_hr)

    # Maintenance (3% of CAPEX)
    maintenance_cost = capex * opex.maintenance_factor

    # Total OPEX
    total_opex = (
        extractant_cost +
        acid_cost +
        base_cost +
        precipitant_cost +
        labor_cost +
        utility_cost +
        maintenance_cost
    )

    return {
        "extractant": extractant_cost,
        "acid": acid_cost,
        "base": base_cost,
        "precipitant": precipitant_cost,
        "labor": labor_cost,
        "utilities": utility_cost,
        "maintenance": maintenance_cost,
        "total": total_opex,
    }


# =============================================================================
# Profitability Analysis
# =============================================================================

def calculate_revenue(
    product_flows: dict[str, float],  # kg/year per element
    pricing: REEPricing | None = None,
    purity: str = "99%",
    form: str = "oxide",
) -> dict[str, float]:
    """Calculate annual revenue from REE products.

    Args:
        product_flows: Annual production of each element (kg/year)
        pricing: REE pricing model
        purity: Product purity
        form: Product form

    Returns:
        Revenue breakdown by element and total
    """
    if pricing is None:
        pricing = REEPricing()

    revenue = {}
    total = 0.0

    for element, production in product_flows.items():
        price = pricing.get_price(element, purity, form)
        elem_revenue = production * price
        revenue[element] = elem_revenue
        total += elem_revenue

    revenue["total"] = total
    return revenue


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
    taxes = max(0, ebit * tax_rate)

    # Net income
    net_income = ebit - taxes

    # Cash flow (add back depreciation)
    cash_flow = net_income + depreciation

    # Simple payback
    payback = capex / cash_flow if cash_flow > 0 else float('inf')

    # ROI
    roi = net_income / capex if capex > 0 else 0

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


def minimum_selling_price(
    opex: float,
    capex: float,
    annual_production_kg: float,
    target_roi: float = 0.15,
    tax_rate: float = 0.25,
    depreciation_years: int = 10,
) -> float:
    """Calculate minimum selling price for target ROI.

    Args:
        opex: Annual operating cost (USD/year)
        capex: Total capital cost (USD)
        annual_production_kg: Annual production (kg/year)
        target_roi: Target return on investment
        tax_rate: Corporate tax rate
        depreciation_years: Depreciation period

    Returns:
        Minimum selling price (USD/kg)
    """
    # Target net income
    target_net = target_roi * capex

    # Required EBIT (before tax)
    required_ebit = target_net / (1 - tax_rate)

    # Required EBITDA
    depreciation = capex / depreciation_years
    required_ebitda = required_ebit + depreciation

    # Required revenue
    required_revenue = required_ebitda + opex

    # MSP
    msp = required_revenue / annual_production_kg

    return msp
