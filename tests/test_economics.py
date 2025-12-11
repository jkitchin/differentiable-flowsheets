"""Tests for the economics module.

Tests cover:
- Cost indices and escalation
- Capital cost correlations
- Utility costs
- Operating costs
- Profitability metrics
- JAX differentiability
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow import economics as econ


class TestIndices:
    """Tests for cost indices module."""

    def test_cepci_values_exist(self):
        """Check CEPCI values are available."""
        for year in [2010, 2015, 2019, 2020, 2024]:
            cepci = econ.get_cepci(year)
            assert cepci > 0
            assert 300 < cepci < 1000  # Reasonable range

    def test_cepci_invalid_year(self):
        """Check error for invalid year."""
        with pytest.raises(ValueError):
            econ.get_cepci(1950)

    def test_escalate_cost(self):
        """Test cost escalation."""
        base_cost = jnp.array(100000.0)

        # Costs should increase from older to newer years
        cost_2010 = econ.escalate_cost(base_cost, base_year=2019, target_year=2010)
        cost_2024 = econ.escalate_cost(base_cost, base_year=2019, target_year=2024)

        assert float(cost_2010) < float(base_cost)
        assert float(cost_2024) > float(base_cost)

    def test_inflation_factor(self):
        """Test inflation factor calculation."""
        factor = econ.inflation_factor(2020, 2025, annual_rate=0.02)
        expected = (1.02) ** 5
        assert abs(factor - expected) < 1e-10


class TestCapitalCosts:
    """Tests for capital cost module."""

    def test_reactor_cost_basic(self):
        """Test basic reactor cost calculation."""
        V = jnp.array(2.0)
        cost = econ.reactor_cost(V, "cstr_jacketed")

        assert float(cost) > 0
        assert float(cost) > 10000  # Reasonable minimum

    def test_reactor_cost_scaling(self):
        """Test that larger reactors cost more."""
        V1 = jnp.array(1.0)
        V2 = jnp.array(5.0)

        cost1 = econ.reactor_cost(V1, "cstr_jacketed")
        cost2 = econ.reactor_cost(V2, "cstr_jacketed")

        assert float(cost2) > float(cost1)

    def test_reactor_cost_scaling_sublinear(self):
        """Test economy of scale (sublinear scaling)."""
        V1 = jnp.array(1.0)
        V2 = jnp.array(10.0)

        cost1 = econ.reactor_cost(V1, "cstr_jacketed")
        cost2 = econ.reactor_cost(V2, "cstr_jacketed")

        # Cost ratio should be less than volume ratio (economy of scale)
        cost_ratio = float(cost2) / float(cost1)
        volume_ratio = 10.0

        assert cost_ratio < volume_ratio

    def test_installed_cost(self):
        """Test installed cost calculation."""
        purchased = jnp.array(100000.0)
        installed = econ.installed_cost(purchased, lang_factor=4.74)

        expected = 100000.0 * 4.74
        assert abs(float(installed) - expected) < 0.01

    def test_equipment_types(self):
        """Test that all equipment types work."""
        V = jnp.array(2.0)
        A = jnp.array(50.0)
        P = jnp.array(10.0)

        # Reactors
        for rtype in econ.REACTOR_COSTS.keys():
            cost = econ.reactor_cost(V, rtype)
            assert float(cost) > 0

        # Heat exchangers
        for hxtype in econ.HEAT_EXCHANGER_COSTS.keys():
            cost = econ.heat_exchanger_cost(A, hxtype)
            assert float(cost) > 0

        # Pumps
        for ptype in econ.PUMP_COSTS.keys():
            cost = econ.pump_cost(P, ptype)
            assert float(cost) > 0

    def test_total_capital_investment(self):
        """Test total capital investment calculation."""
        purchased = jnp.array(500000.0)
        tci = econ.total_capital_investment(purchased, lang_factor=4.74, working_capital_fraction=0.15)

        fci = 500000.0 * 4.74
        wc = fci * 0.15
        expected = fci + wc

        assert abs(float(tci) - expected) < 0.01

    def test_material_factors(self):
        """Test material factors are reasonable."""
        assert econ.MATERIAL_FACTORS["carbon_steel"] == 1.0
        assert econ.MATERIAL_FACTORS["stainless_316"] > 1.0
        assert econ.MATERIAL_FACTORS["titanium"] > econ.MATERIAL_FACTORS["stainless_316"]


class TestUtilityCosts:
    """Tests for utility costs module."""

    def test_steam_cost(self):
        """Test steam cost calculation."""
        duty = jnp.array(1e6)  # 1 MW
        cost = econ.steam_cost_from_duty(duty, "medium_pressure")

        assert float(cost) > 0

    def test_cooling_water_cost(self):
        """Test cooling water cost."""
        duty = jnp.array(1e6)  # 1 MW
        cost = econ.cooling_water_cost(duty)

        assert float(cost) > 0
        # Cooling water should be cheaper than steam
        steam_cost = econ.steam_cost_from_duty(duty, "low_pressure")
        assert float(cost) < float(steam_cost)

    def test_electricity_cost(self):
        """Test electricity cost calculation."""
        power = jnp.array(100.0)  # 100 kW
        cost = econ.electricity_cost(power)

        # 100 kW * $0.07/kWh = $7/h
        expected = 100.0 * 0.07
        assert abs(float(cost) - expected) < 0.01

    def test_refrigeration_costs_increase_with_lower_temp(self):
        """Test that refrigeration costs more at lower temperatures."""
        duty = jnp.array(1e6)

        cost_chilled = econ.refrigeration_cost(duty, temperature_level=5.0)
        cost_moderate = econ.refrigeration_cost(duty, temperature_level=-20.0)
        cost_low = econ.refrigeration_cost(duty, temperature_level=-50.0)

        assert float(cost_moderate) > float(cost_chilled)
        assert float(cost_low) > float(cost_moderate)

    def test_zero_duty_zero_cost(self):
        """Test that zero duty gives zero cost."""
        assert float(econ.steam_cost_from_duty(jnp.array(0.0))) == 0.0
        assert float(econ.cooling_water_cost(jnp.array(0.0))) == 0.0


class TestOperatingCosts:
    """Tests for operating cost module."""

    def test_raw_material_cost(self):
        """Test raw material cost calculation."""
        flowrate = jnp.array(1.0)  # 1 kg/s
        price = jnp.array(10.0)  # $10/kg

        cost = econ.raw_material_cost(flowrate, price)
        assert float(cost) == 10.0  # $/s

    def test_operating_labor_cost(self):
        """Test operating labor cost."""
        labor = econ.operating_labor_cost(n_operators_per_shift=4, n_shifts=3)

        # Should be positive and reasonable
        assert labor > 0
        assert labor < 5e6  # Less than $5M/year

    def test_maintenance_cost(self):
        """Test maintenance cost calculation."""
        fci = jnp.array(1e6)
        maint = econ.maintenance_cost(fci)

        # Default is 4% of FCI
        expected = 1e6 * 0.04
        assert abs(float(maint) - expected) < 0.01

    def test_simple_opex(self):
        """Test simplified OPEX calculation."""
        opex = econ.simple_opex(
            raw_materials=jnp.array(1e6),
            utilities=jnp.array(0.5e6),
            fixed_capital=jnp.array(10e6),
        )

        # Variable: 1.0 + 0.5 = 1.5M
        # Fixed: 10M * (0.02 + 0.04 + 0.02) = 0.8M
        # Total: 2.3M
        expected = 1.5e6 + 10e6 * 0.08
        assert abs(float(opex) - expected) < 1.0


class TestProfitability:
    """Tests for profitability metrics."""

    def test_present_value(self):
        """Test present value calculation."""
        fv = jnp.array(1000.0)
        rate = jnp.array(0.10)
        years = jnp.array(1.0)

        pv = econ.present_value(fv, rate, years)
        expected = 1000.0 / 1.10
        assert abs(float(pv) - expected) < 0.01

    def test_capital_recovery_factor(self):
        """Test capital recovery factor."""
        rate = jnp.array(0.10)
        years = jnp.array(20.0)

        crf = econ.capital_recovery_factor(rate, years)

        # Known value for 10%, 20 years
        expected = 0.1175  # approximately
        assert abs(float(crf) - expected) < 0.01

    def test_npv_basic(self):
        """Test basic NPV calculation."""
        cash_flows = jnp.array([100.0, 100.0, 100.0])
        rate = jnp.array(0.10)
        investment = jnp.array(200.0)

        npv_val = econ.npv(cash_flows, rate, investment)

        # PV of cash flows = 100/1.1 + 100/1.21 + 100/1.331 = 248.69
        # NPV = 248.69 - 200 = 48.69
        assert float(npv_val) > 0
        assert abs(float(npv_val) - 48.69) < 1.0

    def test_npv_negative_for_bad_project(self):
        """Test NPV is negative for unprofitable project."""
        cash_flows = jnp.array([10.0, 10.0, 10.0])  # Small cash flows
        rate = jnp.array(0.10)
        investment = jnp.array(1000.0)  # Large investment

        npv_val = econ.npv(cash_flows, rate, investment)
        assert float(npv_val) < 0

    def test_irr_basic(self):
        """Test IRR calculation."""
        # Project: invest 100, get 50 each year for 3 years
        cash_flows = jnp.array([50.0, 50.0, 50.0])
        investment = jnp.array(100.0)

        irr_val = econ.irr(cash_flows, investment)

        # IRR should be positive and reasonable
        assert 0.0 < float(irr_val) < 1.0

        # Verify: NPV at IRR should be ~0
        npv_at_irr = econ.npv(cash_flows, irr_val, investment)
        assert abs(float(npv_at_irr)) < 1.0

    def test_simple_payback(self):
        """Test simple payback calculation."""
        annual_cf = jnp.array(100.0)
        investment = jnp.array(250.0)

        payback = econ.simple_payback(annual_cf, investment)
        assert float(payback) == 2.5

    def test_minimum_selling_price(self):
        """Test MSP calculation."""
        cost = jnp.array(1000.0)
        production = jnp.array(100.0)

        msp = econ.minimum_selling_price(cost, production)
        assert float(msp) == 10.0

    def test_annualized_cost(self):
        """Test annualized cost calculation."""
        capex = jnp.array(1e6)
        opex = jnp.array(100000.0)
        rate = jnp.array(0.10)
        life = jnp.array(20.0)

        tac = econ.annualized_cost(capex, opex, rate, life)

        # CRF ~= 0.1175
        # TAC = 1e6 * 0.1175 + 100000 = 217500
        assert abs(float(tac) - 217500) < 1000


class TestDifferentiability:
    """Tests for JAX differentiability."""

    def test_reactor_cost_differentiable(self):
        """Test reactor cost is differentiable."""
        def cost_fn(V):
            return econ.reactor_cost(V, "cstr_jacketed")

        grad_fn = jax.grad(cost_fn)
        V = jnp.array(2.0)

        grad = grad_fn(V)
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # Cost increases with volume

    def test_npv_differentiable(self):
        """Test NPV is differentiable with respect to cash flows."""
        def npv_fn(cf):
            cash_flows = jnp.ones(10) * cf
            return econ.npv(cash_flows, jnp.array(0.10), jnp.array(1000.0))

        grad_fn = jax.grad(npv_fn)
        cf = jnp.array(200.0)

        grad = grad_fn(cf)
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # NPV increases with cash flow

    def test_utility_cost_differentiable(self):
        """Test utility costs are differentiable."""
        def cost_fn(duty):
            return econ.steam_cost_from_duty(duty)

        grad_fn = jax.grad(cost_fn)
        duty = jnp.array(1e6)

        grad = grad_fn(duty)
        assert jnp.isfinite(grad)
        assert float(grad) > 0

    def test_annualized_cost_differentiable(self):
        """Test annualized cost is differentiable."""
        def tac_fn(capex):
            return econ.annualized_cost(
                capex,
                jnp.array(100000.0),
                jnp.array(0.10),
                jnp.array(20.0)
            )

        grad_fn = jax.grad(tac_fn)
        capex = jnp.array(1e6)

        grad = grad_fn(capex)
        assert jnp.isfinite(grad)
        # Gradient should be approximately equal to CRF
        crf = float(econ.capital_recovery_factor(jnp.array(0.10), jnp.array(20.0)))
        assert abs(float(grad) - crf) < 0.01

    def test_equipment_cost_jit(self):
        """Test equipment cost can be JIT compiled."""
        @jax.jit
        def cost_fn(V):
            reactor = econ.reactor_cost(V, "cstr_jacketed")
            return econ.installed_cost(reactor)

        V = jnp.array(2.0)
        cost = cost_fn(V)
        assert jnp.isfinite(cost)

    def test_profitability_vmap(self):
        """Test profitability metrics work with vmap."""
        def npv_fn(investment):
            cash_flows = jnp.ones(10) * 100.0
            return econ.npv(cash_flows, jnp.array(0.10), investment)

        investments = jnp.array([500.0, 600.0, 700.0, 800.0])
        npvs = jax.vmap(npv_fn)(investments)

        assert len(npvs) == 4
        # NPV should decrease with investment
        assert float(npvs[0]) > float(npvs[-1])


class TestCashFlowAnalysis:
    """Tests for cash flow analysis."""

    def test_generate_cash_flows(self):
        """Test cash flow generation."""
        cash_flows = econ.generate_cash_flows(
            capital_investment=jnp.array(1e6),
            annual_revenue=jnp.array(500000.0),
            annual_opex=jnp.array(200000.0),
        )

        assert len(cash_flows) == 20  # Default plant life
        assert all(jnp.isfinite(cash_flows))

    def test_full_cash_flow_analysis(self):
        """Test full cash flow analysis."""
        result = econ.full_cash_flow_analysis(
            capital_investment=1e6,
            annual_revenue=400000.0,
            annual_opex=150000.0,
        )

        assert hasattr(result, "npv")
        assert hasattr(result, "irr")
        assert hasattr(result, "payback")
        assert hasattr(result, "cash_flow")

        # Basic sanity checks
        assert result.npv != 0  # NPV should be calculated
        assert 0 < result.irr < 1  # IRR should be a reasonable percentage


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_complete_economic_analysis(self):
        """Test a complete economic analysis workflow."""
        # Equipment sizing
        reactor_volume = jnp.array(5.0)  # m³
        hx_area = jnp.array(100.0)  # m²

        # Capital costs
        reactor_cost = econ.reactor_cost(reactor_volume, "cstr_jacketed")
        hx_cost = econ.heat_exchanger_cost(hx_area, "shell_tube_floating")
        total_purchased = reactor_cost + hx_cost
        total_installed = econ.installed_cost(total_purchased)
        tci = econ.total_capital_investment(total_purchased)

        # Operating costs
        heating = econ.steam_cost_from_duty(jnp.array(500000.0))  # 500 kW heating
        cooling = econ.cooling_water_cost(jnp.array(300000.0))  # 300 kW cooling
        utility_annual = (heating + cooling) * 8000 * 3600

        raw_materials = econ.raw_material_cost(jnp.array(1.0), jnp.array(50.0))  # 1 kg/s at $50/kg
        rm_annual = raw_materials * 8000 * 3600

        opex_annual = utility_annual + rm_annual + total_installed * 0.08  # 8% fixed costs

        # Revenue
        production = jnp.array(0.9)  # 0.9 kg/s product
        price = jnp.array(200.0)  # $200/kg
        revenue_annual = production * price * 8000 * 3600

        # Profitability
        annual_profit = revenue_annual - opex_annual
        cash_flows = jnp.ones(20) * annual_profit

        npv_val = econ.npv(cash_flows, jnp.array(0.10), tci)

        # All values should be finite and reasonable
        assert jnp.isfinite(tci)
        assert jnp.isfinite(opex_annual)
        assert jnp.isfinite(revenue_annual)
        assert jnp.isfinite(npv_val)

        print(f"\nIntegration Test Results:")
        print(f"  Total Capital: ${float(tci):,.0f}")
        print(f"  Annual OPEX: ${float(opex_annual):,.0f}")
        print(f"  Annual Revenue: ${float(revenue_annual):,.0f}")
        print(f"  Annual Profit: ${float(annual_profit):,.0f}")
        print(f"  NPV: ${float(npv_val):,.0f}")
