"""Profitability analysis for process economics.

This module provides differentiable profitability metrics including:
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Payback Period
- Minimum Selling Price (MSP)
- Return on Investment (ROI)
- Annualized costs

All functions are JAX-compatible for gradient-based optimization of
process designs to maximize profitability.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax import lax
from dataclasses import dataclass, replace, fields, asdict as dc_asdict
from typing import Callable, NamedTuple
import optimistix as optx


# =============================================================================
# Financial Parameters
# =============================================================================

@dataclass
class FinancialParams:
    """Financial parameters for profitability analysis."""
    discount_rate: float = 0.10  # 10% discount rate (WACC or hurdle rate)
    tax_rate: float = 0.21  # Corporate tax rate
    depreciation_years: int = 10  # MACRS depreciation period
    plant_life: int = 20  # Years of operation
    construction_years: int = 2  # Construction period
    salvage_fraction: float = 0.05  # Salvage value as fraction of FCI
    working_capital_fraction: float = 0.15  # Working capital as fraction of FCI
    inflation_rate: float = 0.02  # Annual inflation for revenues/costs

    def update(self, **kwargs) -> "FinancialParams":
        """Return a new FinancialParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., discount_rate=0.12, tax_rate=0.25)

        Returns:
            New FinancialParams with updated fields
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


DEFAULT_FINANCIAL = FinancialParams()


# MACRS depreciation schedules (percentage of depreciable capital per year)
MACRS_SCHEDULES = {
    5: jnp.array([0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576]),
    7: jnp.array([0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]),
    10: jnp.array([0.1000, 0.1800, 0.1440, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655, 0.0328]),
    15: jnp.array([0.0500, 0.0950, 0.0855, 0.0770, 0.0693, 0.0623, 0.0590, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0295]),
}


# =============================================================================
# Time Value of Money Functions
# =============================================================================

def present_value(
    future_value: Array,
    rate: Array,
    years: Array,
) -> Array:
    """Calculate present value of a future amount.

    PV = FV / (1 + r)^n

    Args:
        future_value: Future value ($)
        rate: Discount rate
        years: Number of years

    Returns:
        Present value ($)
    """
    return future_value / jnp.power(1.0 + rate, years)


def future_value(
    present_value: Array,
    rate: Array,
    years: Array,
) -> Array:
    """Calculate future value of a present amount.

    FV = PV * (1 + r)^n

    Args:
        present_value: Present value ($)
        rate: Growth/discount rate
        years: Number of years

    Returns:
        Future value ($)
    """
    return present_value * jnp.power(1.0 + rate, years)


def discount_factor(
    rate: Array,
    year: Array,
) -> Array:
    """Calculate discount factor for a given year.

    DF = 1 / (1 + r)^n

    Args:
        rate: Discount rate
        year: Year number (1 to N)

    Returns:
        Discount factor
    """
    return 1.0 / jnp.power(1.0 + rate, year)


def capital_recovery_factor(
    rate: Array,
    years: Array,
) -> Array:
    """Calculate capital recovery factor (annualization factor).

    CRF = r * (1 + r)^n / ((1 + r)^n - 1)

    Converts present value to equivalent annual payment.

    Args:
        rate: Interest rate
        years: Number of years

    Returns:
        Capital recovery factor
    """
    factor = jnp.power(1.0 + rate, years)
    return rate * factor / (factor - 1.0)


def present_value_factor(
    rate: Array,
    years: Array,
) -> Array:
    """Calculate present value factor for annuity.

    PVF = ((1 + r)^n - 1) / (r * (1 + r)^n)

    Converts annual payment to present value.

    Args:
        rate: Discount rate
        years: Number of years

    Returns:
        Present value factor
    """
    factor = jnp.power(1.0 + rate, years)
    return (factor - 1.0) / (rate * factor)


# =============================================================================
# Net Present Value (NPV)
# =============================================================================

def npv(
    cash_flows: Array,
    discount_rate: Array,
    initial_investment: Array = jnp.array(0.0),
) -> Array:
    """Calculate Net Present Value.

    NPV = -I_0 + sum(CF_t / (1 + r)^t)

    This is fully differentiable with respect to all inputs.

    Args:
        cash_flows: Array of annual cash flows (year 1 to N)
        discount_rate: Discount rate
        initial_investment: Initial investment at year 0

    Returns:
        Net Present Value ($)
    """
    years = jnp.arange(1, len(cash_flows) + 1)
    discount_factors = discount_factor(discount_rate, years)
    present_values = cash_flows * discount_factors
    return -initial_investment + jnp.sum(present_values)


def npv_with_construction(
    operating_cash_flows: Array,
    capital_investment: Array,
    discount_rate: Array,
    construction_years: int = 2,
    construction_schedule: Array | None = None,
) -> Array:
    """NPV including construction period capital expenditure.

    Args:
        operating_cash_flows: Annual operating cash flows during production
        capital_investment: Total capital investment
        discount_rate: Discount rate
        construction_years: Number of construction years
        construction_schedule: Fraction of capital spent each construction year
                              (default: equal distribution)

    Returns:
        NPV ($)
    """
    if construction_schedule is None:
        # Equal distribution during construction
        construction_schedule = jnp.ones(construction_years) / construction_years

    # Discount construction costs
    construction_years_arr = jnp.arange(1, construction_years + 1)
    construction_factors = discount_factor(discount_rate, construction_years_arr)
    pv_construction = jnp.sum(
        capital_investment * construction_schedule * construction_factors
    )

    # Discount operating cash flows (start after construction)
    n_years = len(operating_cash_flows)
    operating_years = jnp.arange(
        construction_years + 1,
        construction_years + n_years + 1
    )
    operating_factors = discount_factor(discount_rate, operating_years)
    pv_operating = jnp.sum(operating_cash_flows * operating_factors)

    return pv_operating - pv_construction


def npv_gradient(
    cash_flows: Array,
    discount_rate: Array,
    initial_investment: Array,
) -> tuple[Array, Array, Array]:
    """Calculate NPV and its gradients.

    Returns:
        (npv_value, d_npv/d_cash_flows, d_npv/d_rate, d_npv/d_investment)
    """
    npv_val = npv(cash_flows, discount_rate, initial_investment)

    grad_fn = jax.grad(lambda cf, r, i: npv(cf, r, i), argnums=(0, 1, 2))
    grad_cf, grad_r, grad_i = grad_fn(cash_flows, discount_rate, initial_investment)

    return npv_val, grad_cf, grad_r, grad_i


# =============================================================================
# Internal Rate of Return (IRR)
# =============================================================================

def irr_residual(
    rate: Array,
    cash_flows: Array,
    initial_investment: Array,
) -> Array:
    """Residual function for IRR calculation (NPV = 0)."""
    return npv(cash_flows, rate, initial_investment)


def irr(
    cash_flows: Array,
    initial_investment: Array,
    initial_guess: float = 0.10,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Array:
    """Calculate Internal Rate of Return using optimistix Newton solver.

    IRR is the discount rate at which NPV = 0.

    Note: IRR may not exist for all cash flow patterns, and multiple
    IRRs may exist for non-conventional cash flows.

    Args:
        cash_flows: Array of annual cash flows (year 1 to N)
        initial_investment: Initial investment at year 0
        initial_guess: Starting guess for IRR
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance

    Returns:
        Internal Rate of Return (as decimal, e.g., 0.15 for 15%)
    """
    rate0 = jnp.array(initial_guess)

    # Wrap residual for optimistix (expects (y, args) signature)
    def optx_residual(rate, args):
        cf, inv = args
        return irr_residual(rate, cf, inv)

    # Create Newton solver
    solver = optx.Newton(rtol=tol, atol=tol)

    # Solve using optimistix
    sol = optx.root_find(
        optx_residual,
        solver,
        rate0,
        args=(cash_flows, initial_investment),
        max_steps=max_iter,
        throw=False,
    )

    return sol.value


def irr_approx(
    cash_flows: Array,
    initial_investment: Array,
) -> Array:
    """Quick IRR approximation using linear interpolation.

    Less accurate but always differentiable.

    Args:
        cash_flows: Annual cash flows
        initial_investment: Initial investment

    Returns:
        Approximate IRR
    """
    # Try two rates and interpolate
    r1 = jnp.array(0.05)
    r2 = jnp.array(0.50)

    npv1 = npv(cash_flows, r1, initial_investment)
    npv2 = npv(cash_flows, r2, initial_investment)

    # Linear interpolation to find where NPV = 0
    irr_approx = r1 - npv1 * (r2 - r1) / (npv2 - npv1 + 1e-10)

    # Clamp to reasonable range
    return jnp.clip(irr_approx, 0.0, 1.0)


# =============================================================================
# Payback Period
# =============================================================================

def simple_payback(
    annual_cash_flow: Array,
    initial_investment: Array,
) -> Array:
    """Calculate simple payback period.

    Payback = Initial Investment / Annual Cash Flow

    Args:
        annual_cash_flow: Constant annual cash flow ($)
        initial_investment: Initial investment ($)

    Returns:
        Payback period (years)
    """
    return initial_investment / jnp.maximum(annual_cash_flow, 1e-10)


def discounted_payback(
    cash_flows: Array,
    initial_investment: Array,
    discount_rate: Array,
) -> Array:
    """Calculate discounted payback period.

    Finds the year when cumulative discounted cash flow equals investment.

    Args:
        cash_flows: Annual cash flows (year 1 to N)
        initial_investment: Initial investment
        discount_rate: Discount rate

    Returns:
        Discounted payback period (years, fractional)
    """
    years = jnp.arange(1, len(cash_flows) + 1)
    dcf = cash_flows * discount_factor(discount_rate, years)
    cumulative = jnp.cumsum(dcf)

    # Find crossover point using differentiable approximation
    # Use softmax-weighted average of years where cumulative > investment
    weights = jax.nn.sigmoid(10.0 * (cumulative - initial_investment))
    # Approximate payback as weighted average
    payback = jnp.sum(years * weights) / jnp.sum(weights + 1e-10)

    return jnp.minimum(payback, float(len(cash_flows)))


# =============================================================================
# Return on Investment (ROI)
# =============================================================================

def roi(
    annual_profit: Array,
    total_investment: Array,
) -> Array:
    """Calculate simple Return on Investment.

    ROI = Annual Profit / Total Investment * 100%

    Args:
        annual_profit: Annual net profit ($)
        total_investment: Total capital investment ($)

    Returns:
        ROI (as decimal, e.g., 0.20 for 20%)
    """
    return annual_profit / jnp.maximum(total_investment, 1e-10)


def average_roi(
    total_profit: Array,
    total_investment: Array,
    years: Array,
) -> Array:
    """Calculate average Return on Investment over project life.

    Args:
        total_profit: Total profit over project life ($)
        total_investment: Total capital investment ($)
        years: Project life (years)

    Returns:
        Average annual ROI (as decimal)
    """
    annual_avg_profit = total_profit / jnp.maximum(years, 1.0)
    return annual_avg_profit / jnp.maximum(total_investment, 1e-10)


# =============================================================================
# Minimum Selling Price (MSP)
# =============================================================================

def minimum_selling_price(
    total_annual_cost: Array,
    annual_production: Array,
) -> Array:
    """Calculate Minimum Selling Price (breakeven price).

    MSP = Total Annual Cost / Annual Production

    Args:
        total_annual_cost: Total annual production cost ($)
        annual_production: Annual production rate (kg, mol, or units)

    Returns:
        Minimum selling price ($/unit)
    """
    return total_annual_cost / jnp.maximum(annual_production, 1e-10)


def msp_with_target_roi(
    total_annual_cost: Array,
    annual_production: Array,
    total_investment: Array,
    target_roi: Array = jnp.array(0.15),
) -> Array:
    """Calculate selling price to achieve target ROI.

    Price = (Cost + Target_Profit) / Production
    Target_Profit = Total_Investment * Target_ROI

    Args:
        total_annual_cost: Total annual production cost ($)
        annual_production: Annual production rate
        total_investment: Total capital investment ($)
        target_roi: Target annual return on investment

    Returns:
        Required selling price ($/unit)
    """
    target_profit = total_investment * target_roi
    required_revenue = total_annual_cost + target_profit
    return required_revenue / jnp.maximum(annual_production, 1e-10)


def msp_with_npv_zero(
    opex: Array,
    capex: Array,
    annual_production: Array,
    discount_rate: Array,
    plant_life: int,
    working_capital_fraction: float = 0.15,
) -> Array:
    """Calculate MSP that gives NPV = 0 (true economic breakeven).

    This finds the price P such that:
    NPV = -CAPEX + sum((Revenue - OPEX) / (1+r)^t) + WC_recovery = 0

    Where Revenue = P * Production

    Args:
        opex: Annual operating cost (not including depreciation)
        capex: Capital investment
        annual_production: Annual production
        discount_rate: Discount rate
        plant_life: Years of operation
        working_capital_fraction: Working capital as fraction of CAPEX

    Returns:
        Minimum selling price for NPV = 0
    """
    # Present value factor for annuity
    pvf = present_value_factor(discount_rate, jnp.array(float(plant_life)))

    # Working capital recovery at end of project
    wc = capex * working_capital_fraction
    wc_pv = present_value(wc, discount_rate, jnp.array(float(plant_life)))

    # Total investment (CAPEX + WC - WC recovery)
    total_investment_pv = capex + wc - wc_pv

    # Required annual profit = Investment_PV / PVF
    required_annual_profit = total_investment_pv / pvf

    # MSP = (OPEX + Required_Profit) / Production
    msp = (opex + required_annual_profit) / jnp.maximum(annual_production, 1e-10)

    return msp


# =============================================================================
# Annualized Cost
# =============================================================================

def annualized_cost(
    capital_cost: Array,
    annual_opex: Array,
    discount_rate: Array,
    plant_life: Array,
) -> Array:
    """Calculate total annualized cost.

    TAC = CAPEX * CRF + OPEX

    Where CRF is the capital recovery factor.

    Args:
        capital_cost: Total capital investment ($)
        annual_opex: Annual operating cost ($)
        discount_rate: Discount rate
        plant_life: Plant life (years)

    Returns:
        Total annualized cost ($/year)
    """
    crf = capital_recovery_factor(discount_rate, plant_life)
    return capital_cost * crf + annual_opex


def equivalent_annual_cost(
    capital_cost: Array,
    operating_costs: Array,
    discount_rate: Array,
    plant_life: int,
) -> Array:
    """Calculate Equivalent Annual Cost for comparing alternatives.

    Args:
        capital_cost: Initial capital cost
        operating_costs: Array of annual operating costs
        discount_rate: Discount rate
        plant_life: Project life

    Returns:
        Equivalent annual cost ($/year)
    """
    # PV of operating costs
    years = jnp.arange(1, plant_life + 1)
    pv_opex = jnp.sum(operating_costs * discount_factor(discount_rate, years))

    # Total PV
    total_pv = capital_cost + pv_opex

    # Convert to equivalent annual cost
    crf = capital_recovery_factor(discount_rate, jnp.array(float(plant_life)))
    return total_pv * crf


# =============================================================================
# Cash Flow Analysis
# =============================================================================

@dataclass
class CashFlowResult:
    """Results from cash flow analysis."""
    years: Array
    revenue: Array
    operating_cost: Array
    depreciation: Array
    taxable_income: Array
    taxes: Array
    net_income: Array
    cash_flow: Array
    cumulative_cash_flow: Array
    npv: float
    irr: float
    payback: float


def generate_cash_flows(
    capital_investment: Array,
    annual_revenue: Array,
    annual_opex: Array,
    params: FinancialParams | None = None,
) -> Array:
    """Generate annual cash flows for a project.

    Cash Flow = (Revenue - OPEX) * (1 - tax) + Depreciation * tax

    Args:
        capital_investment: Total capital investment ($)
        annual_revenue: Annual revenue (constant or array)
        annual_opex: Annual operating cost (constant or array)
        params: Financial parameters

    Returns:
        Array of annual after-tax cash flows
    """
    if params is None:
        params = DEFAULT_FINANCIAL

    n_years = params.plant_life

    # Handle scalar or array inputs
    if isinstance(annual_revenue, (int, float)):
        revenue = jnp.ones(n_years) * annual_revenue
    else:
        revenue = annual_revenue

    if isinstance(annual_opex, (int, float)):
        opex = jnp.ones(n_years) * annual_opex
    else:
        opex = annual_opex

    # Depreciation (MACRS or straight-line)
    if params.depreciation_years in MACRS_SCHEDULES:
        macrs = MACRS_SCHEDULES[params.depreciation_years]
        depreciation = jnp.zeros(n_years)
        depreciation = depreciation.at[:len(macrs)].set(macrs * capital_investment)
    else:
        # Straight-line depreciation
        annual_dep = capital_investment / params.depreciation_years
        depreciation = jnp.zeros(n_years)
        depreciation = depreciation.at[:params.depreciation_years].set(annual_dep)

    # Taxable income and taxes
    taxable_income = revenue - opex - depreciation
    taxes = jnp.maximum(taxable_income * params.tax_rate, 0.0)

    # After-tax cash flow
    net_income = taxable_income - taxes
    cash_flow = net_income + depreciation  # Add back non-cash depreciation

    return cash_flow


def full_cash_flow_analysis(
    capital_investment: float,
    annual_revenue: float,
    annual_opex: float,
    params: FinancialParams | None = None,
) -> CashFlowResult:
    """Perform complete cash flow analysis.

    Args:
        capital_investment: Total capital investment ($)
        annual_revenue: Annual revenue ($)
        annual_opex: Annual operating cost ($)
        params: Financial parameters

    Returns:
        CashFlowResult with all metrics
    """
    if params is None:
        params = DEFAULT_FINANCIAL

    n_years = params.plant_life
    cap = jnp.array(capital_investment)
    rev = jnp.array(annual_revenue)
    opex = jnp.array(annual_opex)

    # Working capital
    working_capital = cap * params.working_capital_fraction

    # Generate operating cash flows
    cash_flows = generate_cash_flows(cap, rev, opex, params)

    # Add working capital recovery in final year
    cash_flows = cash_flows.at[-1].add(working_capital)

    # Add salvage value
    salvage = cap * params.salvage_fraction
    cash_flows = cash_flows.at[-1].add(salvage)

    # Total investment
    total_investment = cap + working_capital

    # Calculate metrics
    npv_val = float(npv(cash_flows, jnp.array(params.discount_rate), total_investment))
    irr_val = float(irr(cash_flows, total_investment))

    avg_cf = jnp.mean(cash_flows)
    payback_val = float(simple_payback(avg_cf, total_investment))

    # Build detailed results
    years = jnp.arange(1, n_years + 1)

    # Depreciation schedule
    if params.depreciation_years in MACRS_SCHEDULES:
        macrs = MACRS_SCHEDULES[params.depreciation_years]
        depreciation = jnp.zeros(n_years)
        depreciation = depreciation.at[:len(macrs)].set(macrs * cap)
    else:
        annual_dep = cap / params.depreciation_years
        depreciation = jnp.zeros(n_years)
        depreciation = depreciation.at[:params.depreciation_years].set(annual_dep)

    revenue_arr = jnp.ones(n_years) * rev
    opex_arr = jnp.ones(n_years) * opex

    taxable = revenue_arr - opex_arr - depreciation
    taxes = jnp.maximum(taxable * params.tax_rate, 0.0)
    net_income = taxable - taxes

    cumulative = jnp.cumsum(cash_flows) - total_investment

    return CashFlowResult(
        years=years,
        revenue=revenue_arr,
        operating_cost=opex_arr,
        depreciation=depreciation,
        taxable_income=taxable,
        taxes=taxes,
        net_income=net_income,
        cash_flow=cash_flows,
        cumulative_cash_flow=cumulative,
        npv=npv_val,
        irr=irr_val,
        payback=payback_val,
    )


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def npv_sensitivity(
    base_case: dict,
    parameter_ranges: dict[str, tuple[float, float]],
    n_points: int = 20,
) -> dict[str, tuple[Array, Array]]:
    """Calculate NPV sensitivity to key parameters.

    Args:
        base_case: Dict with keys: capital, revenue, opex, discount_rate, plant_life
        parameter_ranges: Dict mapping parameter names to (min, max) ranges
        n_points: Number of points to evaluate

    Returns:
        Dict mapping parameter names to (values, npv_values) arrays
    """
    results = {}

    for param, (pmin, pmax) in parameter_ranges.items():
        values = jnp.linspace(pmin, pmax, n_points)
        npv_values = []

        for val in values:
            case = base_case.copy()
            case[param] = float(val)

            cash_flows = generate_cash_flows(
                jnp.array(case["capital"]),
                jnp.array(case["revenue"]),
                jnp.array(case["opex"]),
            )
            npv_val = npv(
                cash_flows,
                jnp.array(case["discount_rate"]),
                jnp.array(case["capital"])
            )
            npv_values.append(float(npv_val))

        results[param] = (values, jnp.array(npv_values))

    return results


# =============================================================================
# Optimization Objectives
# =============================================================================

def profit_objective(
    revenue: Array,
    opex: Array,
    capex: Array,
    discount_rate: Array,
    plant_life: int,
) -> Array:
    """Objective function for profit maximization (negative for minimization).

    This is a simplified single-year profit suitable for gradient optimization.

    Args:
        revenue: Annual revenue
        opex: Annual operating cost
        capex: Capital investment
        discount_rate: Discount rate for annualization
        plant_life: Plant life in years

    Returns:
        Negative annual profit (for minimization)
    """
    crf = capital_recovery_factor(discount_rate, jnp.array(float(plant_life)))
    annualized_capex = capex * crf
    annual_profit = revenue - opex - annualized_capex
    return -annual_profit  # Negative for minimization


def npv_objective(
    revenue: Array,
    opex: Array,
    capex: Array,
    discount_rate: Array,
    plant_life: int,
    tax_rate: float = 0.21,
) -> Array:
    """NPV objective for optimization (negative for minimization).

    Args:
        revenue: Annual revenue
        opex: Annual operating cost
        capex: Capital investment
        discount_rate: Discount rate
        plant_life: Plant life in years
        tax_rate: Corporate tax rate

    Returns:
        Negative NPV (for minimization)
    """
    # Simple cash flow: (Revenue - OPEX) * (1 - tax) + Depreciation * tax
    depreciation = capex / plant_life
    taxable = revenue - opex - depreciation
    tax = jnp.maximum(taxable, 0.0) * tax_rate
    cash_flow = revenue - opex - tax

    # Create cash flow array
    cash_flows = jnp.ones(plant_life) * cash_flow

    npv_val = npv(cash_flows, discount_rate, capex)
    return -npv_val  # Negative for minimization
