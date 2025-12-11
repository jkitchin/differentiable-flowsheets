"""Example: Technoeconomic Analysis with Differentiable Flowsheets.

This example demonstrates comprehensive technoeconomic analysis (TEA)
integrated with process simulation. Key features:

1. Equipment sizing and capital cost estimation
2. Operating cost calculation (utilities, materials, labor)
3. Profitability analysis (NPV, IRR, MSP)
4. Gradient-based optimization for profit maximization
5. Sensitivity analysis

All calculations are JAX-differentiable for optimization.
"""

import jax
import jax.numpy as jnp
from jax import Array

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Process simulation imports
from difflow.streams import make_stream, get_flows
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.cstr import CSTR, CSTRParams

# Economics imports
import difflow.economics as econ


# =============================================================================
# Process Setup: A → B reaction in CSTR
# =============================================================================

species_data = {
    "A": SpeciesData(
        name="A", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
        Hvap_coeffs=(35000.0, 0.38, 500.0), antoine_coeffs=(10.0, 3000.0, -50.0),
        Hf=0.0,
    ),
    "B": SpeciesData(
        name="B", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
        Hvap_coeffs=(30000.0, 0.38, 450.0), antoine_coeffs=(10.0, 2800.0, -40.0),
        Hf=-50000.0,  # Exothermic reaction
    ),
}

thermo = IdealThermo(species_data)
species_order = ["A", "B"]
stoichiometry = jnp.array([[-1.0], [+1.0]])


def rate_function(C: dict[str, Array], T: Array, params: dict) -> Array:
    """First-order reaction: A → B with Arrhenius kinetics."""
    k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
    return jnp.array([k * C["A"]])


# =============================================================================
# Integrated Process + Economics Model
# =============================================================================

class ProcessEconomics:
    """Integrated process simulation with economics.

    This class combines the reactor model with comprehensive TEA.
    All methods are JAX-differentiable.
    """

    def __init__(
        self,
        feed_rate: float = 10.0,  # mol/s of A
        feed_temp: float = 300.0,  # K
        kinetic_params: dict | None = None,
        prices: dict | None = None,
    ):
        self.feed_rate = feed_rate
        self.feed_temp = feed_temp
        self.kinetic_params = kinetic_params or {
            "A": jnp.array(1e6),
            "Ea": jnp.array(50000.0),
        }

        # Economic parameters
        self.prices = prices or {
            "A": 50.0,  # $/kmol raw material A
            "B": 200.0,  # $/kmol product B
        }

        # Operating parameters
        self.hours_per_year = 8000.0
        self.seconds_per_year = self.hours_per_year * 3600.0

    def simulate(self, V: Array, T: Array) -> tuple[dict, dict]:
        """Run reactor simulation.

        Args:
            V: Reactor volume (m³)
            T: Operating temperature (K)

        Returns:
            (outlet_stream, info_dict)
        """
        params = CSTRParams(
            V=V,
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params=self.kinetic_params,
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream(
            {"A": self.feed_rate, "B": 0.0},
            T=self.feed_temp,
            P=101325.0
        )

        outlet, info = cstr(inlet, T_spec=T)
        return outlet, info

    def capital_cost(self, V: Array) -> dict[str, Array]:
        """Calculate capital costs.

        Args:
            V: Reactor volume (m³)

        Returns:
            Dict of capital cost components
        """
        # Reactor cost (jacketed CSTR)
        reactor_purchased = econ.reactor_cost(V, "cstr_jacketed")

        # Heat exchanger for cooling (estimate area from duty)
        # Assume U = 500 W/(m²·K), LMTD = 30 K
        # Area will be calculated based on duty

        # Installed costs
        reactor_installed = econ.installed_cost(reactor_purchased, lang_factor=4.74)

        # Add auxiliary equipment (pumps, piping, etc.) as 30% of reactor
        auxiliary = reactor_installed * 0.30

        total_installed = reactor_installed + auxiliary

        # Working capital
        working_capital = total_installed * 0.15

        return {
            "reactor_purchased": reactor_purchased,
            "reactor_installed": reactor_installed,
            "auxiliary": auxiliary,
            "total_installed": total_installed,
            "working_capital": working_capital,
            "total_capital": total_installed + working_capital,
        }

    def operating_cost(self, V: Array, T: Array, outlet: dict, info: dict) -> dict[str, Array]:
        """Calculate operating costs.

        Args:
            V: Reactor volume
            T: Temperature
            outlet: Outlet stream
            info: Simulation info dict

        Returns:
            Dict of operating cost components ($/year)
        """
        # Raw material cost
        F_A_in = self.feed_rate  # mol/s
        raw_material_per_s = F_A_in * self.prices["A"] / 1000  # $/s (price per kmol)
        raw_material_annual = raw_material_per_s * self.seconds_per_year

        # Utility cost (cooling for exothermic reaction)
        Q = jnp.abs(info["Q"])  # Heat duty (W)
        utility_per_s = econ.cooling_water_cost(Q)
        utility_annual = utility_per_s * self.seconds_per_year

        # Labor and overhead (from FCI)
        capex = self.capital_cost(V)
        fci = capex["total_installed"]

        # Simplified: labor = 2% of FCI, maintenance = 4%, overhead = 2%
        labor_annual = fci * 0.02
        maintenance_annual = fci * 0.04
        overhead_annual = fci * 0.02

        total_opex = (
            raw_material_annual + utility_annual +
            labor_annual + maintenance_annual + overhead_annual
        )

        return {
            "raw_materials": raw_material_annual,
            "utilities": utility_annual,
            "labor": labor_annual,
            "maintenance": maintenance_annual,
            "overhead": overhead_annual,
            "total_opex": total_opex,
        }

    def revenue(self, outlet: dict) -> dict[str, Array]:
        """Calculate annual revenue.

        Args:
            outlet: Outlet stream

        Returns:
            Dict of revenue components ($/year)
        """
        F_B_out = outlet["F_B"]  # mol/s
        revenue_per_s = F_B_out * self.prices["B"] / 1000  # $/s
        revenue_annual = revenue_per_s * self.seconds_per_year

        return {
            "product_revenue": revenue_annual,
        }

    def annual_profit(self, V: Array, T: Array) -> Array:
        """Calculate annual profit (differentiable).

        Args:
            V: Reactor volume (m³)
            T: Operating temperature (K)

        Returns:
            Annual profit ($/year)
        """
        outlet, info = self.simulate(V, T)

        capex = self.capital_cost(V)
        opex = self.operating_cost(V, T, outlet, info)
        rev = self.revenue(outlet)

        # Annualized capital cost (10% discount rate, 20 year life)
        crf = econ.capital_recovery_factor(jnp.array(0.10), jnp.array(20.0))
        annual_capex = capex["total_capital"] * crf

        profit = rev["product_revenue"] - opex["total_opex"] - annual_capex

        return profit

    def npv(self, V: Array, T: Array, discount_rate: float = 0.10, plant_life: int = 20) -> Array:
        """Calculate Net Present Value.

        Args:
            V: Reactor volume
            T: Temperature
            discount_rate: Discount rate
            plant_life: Years

        Returns:
            NPV ($)
        """
        outlet, info = self.simulate(V, T)

        capex = self.capital_cost(V)
        opex = self.operating_cost(V, T, outlet, info)
        rev = self.revenue(outlet)

        # Annual cash flow (simplified: no taxes)
        annual_cf = rev["product_revenue"] - opex["total_opex"]
        cash_flows = jnp.ones(plant_life) * annual_cf

        # Add working capital recovery in final year
        cash_flows = cash_flows.at[-1].add(capex["working_capital"])

        return econ.npv(cash_flows, jnp.array(discount_rate), capex["total_capital"])

    def minimum_selling_price(self, V: Array, T: Array) -> Array:
        """Calculate minimum selling price for breakeven.

        Args:
            V: Reactor volume
            T: Temperature

        Returns:
            MSP ($/kmol of B)
        """
        outlet, info = self.simulate(V, T)

        capex = self.capital_cost(V)
        opex = self.operating_cost(V, T, outlet, info)

        # Annualized capital
        crf = econ.capital_recovery_factor(jnp.array(0.10), jnp.array(20.0))
        annual_capex = capex["total_capital"] * crf

        # Total annual cost (excluding product revenue, including raw material at cost)
        # Adjust: remove raw material cost since it's in OPEX
        total_annual_cost = opex["total_opex"] + annual_capex

        # Annual production of B
        F_B_out = outlet["F_B"]  # mol/s
        annual_production = F_B_out * self.seconds_per_year / 1000  # kmol/year

        return total_annual_cost / jnp.maximum(annual_production, 1e-10)


# =============================================================================
# Demo Functions
# =============================================================================

def demo_basic_tea():
    """Demonstrate basic technoeconomic analysis."""
    print("\n" + "=" * 70)
    print("1. BASIC TECHNOECONOMIC ANALYSIS")
    print("=" * 70)

    pe = ProcessEconomics()

    # Base case: V = 2 m³, T = 400 K
    V = jnp.array(2.0)
    T = jnp.array(400.0)

    print(f"\nBase Case: V = {float(V):.1f} m³, T = {float(T):.0f} K")
    print("-" * 50)

    # Simulate
    outlet, info = pe.simulate(V, T)
    conversion = float(info["conversion"]["A"]) * 100
    print(f"Conversion: {conversion:.1f}%")
    print(f"Heat duty: {float(info['Q'])/1000:.1f} kW (cooling)")

    # Capital costs
    capex = pe.capital_cost(V)
    print(f"\nCapital Costs:")
    print(f"  Reactor (purchased):  ${float(capex['reactor_purchased']):>12,.0f}")
    print(f"  Reactor (installed):  ${float(capex['reactor_installed']):>12,.0f}")
    print(f"  Auxiliary equipment:  ${float(capex['auxiliary']):>12,.0f}")
    print(f"  Working capital:      ${float(capex['working_capital']):>12,.0f}")
    print(f"  Total Capital:        ${float(capex['total_capital']):>12,.0f}")

    # Operating costs
    opex = pe.operating_cost(V, T, outlet, info)
    print(f"\nOperating Costs ($/year):")
    print(f"  Raw materials:        ${float(opex['raw_materials']):>12,.0f}")
    print(f"  Utilities:            ${float(opex['utilities']):>12,.0f}")
    print(f"  Labor:                ${float(opex['labor']):>12,.0f}")
    print(f"  Maintenance:          ${float(opex['maintenance']):>12,.0f}")
    print(f"  Overhead:             ${float(opex['overhead']):>12,.0f}")
    print(f"  Total OPEX:           ${float(opex['total_opex']):>12,.0f}")

    # Revenue
    rev = pe.revenue(outlet)
    print(f"\nRevenue ($/year):")
    print(f"  Product B sales:      ${float(rev['product_revenue']):>12,.0f}")

    # Profitability
    profit = pe.annual_profit(V, T)
    npv_val = pe.npv(V, T)
    msp = pe.minimum_selling_price(V, T)

    print(f"\nProfitability Metrics:")
    print(f"  Annual Profit:        ${float(profit):>12,.0f}/year")
    print(f"  NPV (10%, 20yr):      ${float(npv_val):>12,.0f}")
    print(f"  MSP (breakeven):      ${float(msp):>12.2f}/kmol")


def demo_gradient_optimization():
    """Demonstrate gradient-based optimization for profit maximization."""
    print("\n" + "=" * 70)
    print("2. GRADIENT-BASED PROFIT OPTIMIZATION")
    print("=" * 70)

    pe = ProcessEconomics()

    # Objective: maximize profit (minimize negative profit)
    def neg_profit(params: Array) -> Array:
        V, T = params[0], params[1]
        return -pe.annual_profit(V, T)

    # Gradient function
    grad_fn = jax.grad(neg_profit)

    # Initial guess
    x = jnp.array([1.0, 350.0])
    print(f"\nInitial: V = {x[0]:.2f} m³, T = {x[1]:.0f} K")
    print(f"Initial profit: ${-float(neg_profit(x)):,.0f}/year")

    # Simple gradient descent with Adam-like adaptive learning
    learning_rate = jnp.array([0.02, 1.0])  # Different rates for V and T

    print("\nOptimizing...")
    for i in range(100):
        grad = grad_fn(x)
        x = x - learning_rate * grad

        # Apply bounds
        x = jnp.clip(x, jnp.array([0.1, 300.0]), jnp.array([10.0, 500.0]))

        if (i + 1) % 25 == 0:
            profit = -neg_profit(x)
            print(f"  Iter {i+1:3d}: V = {x[0]:.2f} m³, T = {x[1]:.0f} K, "
                  f"Profit = ${float(profit):,.0f}/year")

    V_opt, T_opt = float(x[0]), float(x[1])

    # Final results
    print(f"\nOptimal Design:")
    print(f"  Volume: {V_opt:.2f} m³")
    print(f"  Temperature: {T_opt:.0f} K")

    outlet, info = pe.simulate(jnp.array(V_opt), jnp.array(T_opt))
    print(f"  Conversion: {float(info['conversion']['A'])*100:.1f}%")
    print(f"  Annual Profit: ${float(pe.annual_profit(jnp.array(V_opt), jnp.array(T_opt))):,.0f}")
    print(f"  NPV: ${float(pe.npv(jnp.array(V_opt), jnp.array(T_opt))):,.0f}")


def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis using gradients."""
    print("\n" + "=" * 70)
    print("3. SENSITIVITY ANALYSIS")
    print("=" * 70)

    pe = ProcessEconomics()

    # Base case
    V = jnp.array(2.0)
    T = jnp.array(400.0)

    print(f"\nBase Case: V = {float(V):.1f} m³, T = {float(T):.0f} K")

    # Calculate gradients of profit w.r.t. design variables
    def profit_fn(V, T):
        return pe.annual_profit(V, T)

    grad_V = jax.grad(profit_fn, argnums=0)(V, T)
    grad_T = jax.grad(profit_fn, argnums=1)(V, T)

    print(f"\nProfit Sensitivities ($/year per unit change):")
    print(f"  ∂Profit/∂V = ${float(grad_V):,.0f} per m³")
    print(f"  ∂Profit/∂T = ${float(grad_T):,.0f} per K")

    # NPV sensitivities
    def npv_fn(V, T):
        return pe.npv(V, T)

    grad_npv_V = jax.grad(npv_fn, argnums=0)(V, T)
    grad_npv_T = jax.grad(npv_fn, argnums=1)(V, T)

    print(f"\nNPV Sensitivities:")
    print(f"  ∂NPV/∂V = ${float(grad_npv_V):,.0f} per m³")
    print(f"  ∂NPV/∂T = ${float(grad_npv_T):,.0f} per K")

    # Sensitivity to economic parameters
    print("\n" + "-" * 50)
    print("Sensitivity to Product Price:")

    base_profit = profit_fn(V, T)
    for price_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        pe_temp = ProcessEconomics(prices={"A": 50.0, "B": 200.0 * price_mult})
        profit = pe_temp.annual_profit(V, T)
        change = (float(profit) - float(base_profit)) / float(base_profit) * 100
        print(f"  B price = ${200*price_mult:.0f}/kmol: "
              f"Profit = ${float(profit):>10,.0f} ({change:+.1f}%)")


def demo_equipment_costs():
    """Demonstrate equipment cost correlations."""
    print("\n" + "=" * 70)
    print("4. EQUIPMENT COST CORRELATIONS")
    print("=" * 70)

    print("\nReactor Costs (2024 $, installed with Lang factor 4.74):")
    print("-" * 50)
    for V in [0.5, 1.0, 2.0, 5.0, 10.0]:
        purchased = econ.reactor_cost(jnp.array(V), "cstr_jacketed")
        installed = econ.installed_cost(purchased)
        print(f"  V = {V:5.1f} m³: Purchased = ${float(purchased):>10,.0f}, "
              f"Installed = ${float(installed):>12,.0f}")

    print("\nHeat Exchanger Costs (Shell & Tube, Floating Head):")
    print("-" * 50)
    for A in [10, 50, 100, 500, 1000]:
        purchased = econ.heat_exchanger_cost(jnp.array(float(A)), "shell_tube_floating")
        installed = econ.installed_cost(purchased)
        print(f"  A = {A:5d} m²: Purchased = ${float(purchased):>10,.0f}, "
              f"Installed = ${float(installed):>12,.0f}")

    print("\nMaterial Factors:")
    print("-" * 50)
    base_cost = econ.reactor_cost(jnp.array(2.0), "cstr_jacketed")
    for material, factor in [
        ("Carbon Steel", 1.0),
        ("Stainless 304", 1.8),
        ("Stainless 316", 2.1),
        ("Hastelloy C", 4.0),
        ("Titanium", 7.0),
    ]:
        adjusted = base_cost * factor
        print(f"  {material:15s}: ${float(adjusted):>12,.0f} (factor = {factor})")


def demo_profitability_metrics():
    """Demonstrate profitability metrics."""
    print("\n" + "=" * 70)
    print("5. PROFITABILITY ANALYSIS")
    print("=" * 70)

    pe = ProcessEconomics()
    V = jnp.array(2.0)
    T = jnp.array(400.0)

    outlet, info = pe.simulate(V, T)
    capex = pe.capital_cost(V)
    opex = pe.operating_cost(V, T, outlet, info)
    rev = pe.revenue(outlet)

    # Cash flow analysis
    total_investment = float(capex["total_capital"])
    annual_revenue = float(rev["product_revenue"])
    annual_opex = float(opex["total_opex"])

    print(f"\nProject Parameters:")
    print(f"  Total Investment: ${total_investment:,.0f}")
    print(f"  Annual Revenue: ${annual_revenue:,.0f}")
    print(f"  Annual OPEX: ${annual_opex:,.0f}")
    print(f"  Annual Cash Flow: ${annual_revenue - annual_opex:,.0f}")

    # Generate cash flows with depreciation effects
    params = econ.FinancialParams(
        discount_rate=0.10,
        tax_rate=0.21,
        depreciation_years=10,
        plant_life=20,
    )

    result = econ.full_cash_flow_analysis(
        capital_investment=total_investment,
        annual_revenue=annual_revenue,
        annual_opex=annual_opex,
        params=params,
    )

    print(f"\nProfitability Metrics:")
    print(f"  NPV (10%):       ${result.npv:>12,.0f}")
    print(f"  IRR:             {result.irr*100:>12.1f}%")
    print(f"  Payback Period:  {result.payback:>12.1f} years")

    # MSP calculation
    msp = pe.minimum_selling_price(V, T)
    print(f"  MSP (breakeven): ${float(msp):>12.2f}/kmol")

    # ROI
    annual_profit = annual_revenue - annual_opex
    roi_val = annual_profit / total_investment * 100
    print(f"  Simple ROI:      {roi_val:>12.1f}%")

    print("\nCash Flow Summary (first 10 years):")
    print("-" * 50)
    print("  Year    Revenue        OPEX   Depreciation    Cash Flow")
    for i in range(min(10, len(result.years))):
        print(f"  {int(result.years[i]):4d}  {float(result.revenue[i]):>10,.0f}  "
              f"{float(result.operating_cost[i]):>10,.0f}  "
              f"{float(result.depreciation[i]):>10,.0f}  "
              f"{float(result.cash_flow[i]):>10,.0f}")


def demo_comparison():
    """Compare different process alternatives."""
    print("\n" + "=" * 70)
    print("6. PROCESS ALTERNATIVES COMPARISON")
    print("=" * 70)

    alternatives = [
        ("Conservative", 1.0, 350.0),
        ("Moderate", 2.0, 400.0),
        ("Aggressive", 4.0, 450.0),
    ]

    print("\nComparing reactor designs:")
    print("-" * 80)
    print(f"{'Design':12s} {'V(m³)':>6s} {'T(K)':>6s} {'Conv(%)':>8s} "
          f"{'CAPEX($M)':>10s} {'OPEX($M/y)':>11s} {'Profit($M/y)':>12s} {'NPV($M)':>10s}")
    print("-" * 80)

    pe = ProcessEconomics()

    for name, V, T in alternatives:
        V_arr = jnp.array(V)
        T_arr = jnp.array(T)

        outlet, info = pe.simulate(V_arr, T_arr)
        capex = pe.capital_cost(V_arr)
        opex = pe.operating_cost(V_arr, T_arr, outlet, info)

        conv = float(info["conversion"]["A"]) * 100
        total_capex = float(capex["total_capital"]) / 1e6
        total_opex = float(opex["total_opex"]) / 1e6
        profit = float(pe.annual_profit(V_arr, T_arr)) / 1e6
        npv_val = float(pe.npv(V_arr, T_arr)) / 1e6

        print(f"{name:12s} {V:>6.1f} {T:>6.0f} {conv:>8.1f} "
              f"{total_capex:>10.3f} {total_opex:>11.3f} {profit:>12.3f} {npv_val:>10.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("TECHNOECONOMIC ANALYSIS WITH DIFFERENTIABLE FLOWSHEETS")
    print("=" * 70)
    print("\nThis example demonstrates comprehensive TEA integrated with")
    print("JAX-based process simulation for gradient-based optimization.")

    demo_basic_tea()
    demo_gradient_optimization()
    demo_sensitivity_analysis()
    demo_equipment_costs()
    demo_profitability_metrics()
    demo_comparison()

    print("\n" + "=" * 70)
    print("All technoeconomic analysis examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
