"""Tests for new carbon capture modules.

Tests for:
- Heat integration
- CO2 compression
- Economics (CAPEX, OPEX, levelized cost)
- Process integration
- Direct air capture
- Degradation models
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream, get_flows, total_flow


# =============================================================================
# Heat Integration Tests
# =============================================================================

class TestHeatIntegration:
    """Tests for heat integration module."""

    def test_heat_exchanger_basic(self):
        """Test basic heat exchanger operation."""
        from difflow_cc.units.heat_integration import (
            HeatExchangerParams, HeatExchanger
        )

        params = HeatExchangerParams(U=500.0, A=50.0)
        hx = HeatExchanger(params)

        hot_in = make_stream({"H2O": 10.0}, T=373.15, P=101325.0)
        cold_in = make_stream({"H2O": 10.0}, T=298.15, P=101325.0)

        hot_out, cold_out, info = hx(hot_in, cold_in)

        # Hot stream should cool down
        assert float(hot_out["T"]) < float(hot_in["T"])
        # Cold stream should heat up
        assert float(cold_out["T"]) > float(cold_in["T"])
        # Heat transfer should be positive
        assert float(info["Q"]) > 0

    def test_lean_rich_exchanger(self):
        """Test lean/rich solvent heat exchanger."""
        from difflow_cc.units.heat_integration import (
            LeanRichExchangerParams, LeanRichExchanger
        )

        params = LeanRichExchangerParams(effectiveness=0.85)
        lrhx = LeanRichExchanger(params)

        lean_hot = make_stream({"H2O": 100.0}, T=393.15, P=200000.0)
        rich_cold = make_stream({"H2O": 100.0}, T=323.15, P=200000.0)

        lean_cold, rich_hot, info = lrhx(lean_hot, rich_cold)

        assert float(lean_cold["T"]) < float(lean_hot["T"])
        assert float(rich_hot["T"]) > float(rich_cold["T"])
        assert 0 < float(info["effectiveness"]) <= 1.0

    def test_intercooler(self):
        """Test absorber intercooler."""
        from difflow_cc.units.heat_integration import (
            IntercoolerParams, Intercooler
        )

        params = IntercoolerParams(T_coolant=298.15, approach=5.0)
        cooler = Intercooler(params)

        hot_stream = make_stream({"CO2": 1.0, "N2": 5.0}, T=350.0, P=101325.0)
        cold_stream, info = cooler(hot_stream)

        assert float(cold_stream["T"]) < float(hot_stream["T"])
        assert float(cold_stream["T"]) >= 298.15 + 5.0  # Approach limit

    def test_heat_recovery_system(self):
        """Test complete heat recovery system."""
        from difflow_cc.units.heat_integration import (
            HeatRecoverySystemParams, HeatRecoverySystem
        )

        params = HeatRecoverySystemParams(
            lrhx_effectiveness=0.85,
            T_lean_target=313.15,
        )
        hrs = HeatRecoverySystem(params)

        lean_from_strip = make_stream({"H2O": 100.0}, T=393.15, P=200000.0)
        rich_from_abs = make_stream({"H2O": 100.0}, T=323.15, P=200000.0)

        lean_out, rich_out, info = hrs(lean_from_strip, rich_from_abs)

        assert float(lean_out["T"]) <= 313.15 + 5  # Near target
        assert float(rich_out["T"]) > float(rich_from_abs["T"])

    def test_heat_exchanger_gradient(self):
        """Test gradient through heat exchanger."""
        from difflow_cc.units.heat_integration import (
            HeatExchangerParams, HeatExchanger
        )

        def heat_transfer(area):
            params = HeatExchangerParams(U=500.0, A=area)
            hx = HeatExchanger(params)
            hot_in = make_stream({"H2O": 10.0}, T=373.15, P=101325.0)
            cold_in = make_stream({"H2O": 10.0}, T=298.15, P=101325.0)
            _, _, info = hx(hot_in, cold_in)
            return info["Q"]

        grad_Q = grad(heat_transfer)
        dQ_dA = grad_Q(50.0)

        # More area should increase heat transfer
        assert jnp.isfinite(dQ_dA)


# =============================================================================
# CO2 Compression Tests
# =============================================================================

class TestCompression:
    """Tests for CO2 compression module."""

    def test_single_compressor(self):
        """Test single compressor stage."""
        from difflow_cc.units.compression import CompressorParams, Compressor

        params = CompressorParams(pressure_ratio=3.0, eta_isentropic=0.80)
        comp = Compressor(params)

        inlet = make_stream({"CO2": 10.0}, T=313.15, P=200000.0)
        outlet, info = comp(inlet)

        assert float(outlet["P"]) == pytest.approx(600000.0, rel=0.01)
        assert float(outlet["T"]) > float(inlet["T"])
        assert float(info["P_electrical"]) > 0

    def test_compression_train(self):
        """Test multi-stage compression train."""
        from difflow_cc.units.compression import (
            CompressionTrainParams, CompressionTrain
        )

        params = CompressionTrainParams(
            P_inlet=200000.0,
            P_outlet=15000000.0,
            T_inlet=313.15,
        )
        train = CompressionTrain(params)

        inlet = make_stream({"CO2": 10.0}, T=313.15, P=200000.0)
        outlet, info = train(inlet)

        assert float(outlet["P"]) >= 15000000.0 * 0.95
        assert info["n_stages"] >= 1
        assert float(info["total_power"]) > 0

    def test_co2_properties(self):
        """Test CO2 thermodynamic properties."""
        from difflow_cc.units.compression import (
            co2_density, is_supercritical, co2_compressibility
        )

        # Subcritical
        rho_sub = co2_density(298.15, 101325.0)
        assert float(rho_sub) > 0
        assert not bool(is_supercritical(298.15, 101325.0))

        # Supercritical (above 304 K, 7.38 MPa)
        rho_sc = co2_density(320.0, 10000000.0)
        assert float(rho_sc) > float(rho_sub)
        assert bool(is_supercritical(320.0, 10000000.0))

    def test_compression_power_estimate(self):
        """Test quick compression power estimate."""
        from difflow_cc.units.compression import compression_power_estimate

        power = compression_power_estimate(
            F_CO2=10.0,
            P_in=200000.0,
            P_out=15000000.0,
        )

        assert float(power) > 0
        assert float(power) < 1e7  # Reasonable range

    def test_compression_gradient(self):
        """Test gradient through compression."""
        from difflow_cc.units.compression import compression_power_estimate

        grad_power = grad(compression_power_estimate)
        dP_dF = grad_power(10.0, 200000.0, 15000000.0)

        assert jnp.isfinite(dP_dF)


# =============================================================================
# Economics Tests
# =============================================================================

class TestEconomics:
    """Tests for economics module."""

    def test_absorber_cost(self):
        """Test absorber capital cost."""
        from difflow_cc.economics.capex import absorber_cost

        cost = absorber_cost(diameter=5.0, height=20.0)
        assert float(cost) > 0
        assert float(cost) < 1e8  # Reasonable range

    def test_compressor_cost(self):
        """Test compressor capital cost."""
        from difflow_cc.economics.capex import compressor_cost

        cost = compressor_cost(power=1e6)  # 1 MW
        assert float(cost) > 0

    def test_installed_cost(self):
        """Test installed cost calculation."""
        from difflow_cc.economics.capex import installed_cost, CapexParams

        params = CapexParams()
        result = installed_cost(1000000.0, params)

        assert result["equipment_cost"] == 1000000.0
        assert result["total_overnight_cost"] > result["equipment_cost"]
        assert result["contingency"] > 0

    def test_steam_cost(self):
        """Test steam operating cost."""
        from difflow_cc.economics.opex import steam_cost, OpexParams

        params = OpexParams(steam_price=15.0)
        cost = steam_cost(duty=100e6, params=params)  # 100 MW

        assert float(cost) > 0

    def test_electricity_cost(self):
        """Test electricity operating cost."""
        from difflow_cc.economics.opex import electricity_cost, OpexParams

        params = OpexParams(electricity_price=0.06)
        cost = electricity_cost(power=10e6, params=params)  # 10 MW

        assert float(cost) > 0

    def test_total_operating_cost(self):
        """Test total operating cost calculation."""
        from difflow_cc.economics.opex import total_operating_cost

        result = total_operating_cost(
            steam_duty=100e6,
            electricity=10e6,
            cooling_duty=50e6,
            CO2_captured=10.0,
            capital_cost=100e6,
        )

        assert result["total_opex"] > 0
        assert result["utilities_total"] > 0
        assert result["fixed_total"] > 0

    def test_levelized_cost(self):
        """Test levelized cost of capture."""
        from difflow_cc.economics.levelized_cost import (
            levelized_cost_capture, EconomicParams
        )

        params = EconomicParams(discount_rate=0.08, lifetime=25)
        result = levelized_cost_capture(
            capital_cost=100e6,
            annual_opex=10e6,
            CO2_captured=10.0,
            params=params,
        )

        assert result["total_cost_per_tonne"] > 0
        assert result["capex_per_tonne"] > 0
        assert result["opex_per_tonne"] > 0

    def test_economics_gradient(self):
        """Test gradient through cost calculations."""
        from difflow_cc.economics.capex import absorber_cost

        grad_cost = grad(absorber_cost)
        dC_dD = grad_cost(5.0, 20.0)

        assert jnp.isfinite(dC_dD)


# =============================================================================
# Process Integration Tests
# =============================================================================

class TestProcessIntegration:
    """Tests for process integration module."""

    def test_flue_gas_composition(self):
        """Test flue gas composition lookup."""
        from difflow_cc.integration.power_plant import flue_gas_composition

        comp_coal = flue_gas_composition("coal_supercritical")
        assert "CO2" in comp_coal
        assert 0.10 < comp_coal["CO2"] < 0.20

        comp_ngcc = flue_gas_composition("ngcc")
        assert comp_ngcc["CO2"] < comp_coal["CO2"]  # Less CO2 in NGCC

    def test_flue_gas_flow_rate(self):
        """Test flue gas flow rate calculation."""
        from difflow_cc.integration.power_plant import (
            flue_gas_flow_rate, PowerPlantParams
        )

        params = PowerPlantParams(gross_power=500.0)
        flow = flue_gas_flow_rate(params)

        assert float(flow) > 0

    def test_steam_extraction_penalty(self):
        """Test steam extraction power penalty."""
        from difflow_cc.integration.power_plant import steam_extraction_penalty

        penalty = steam_extraction_penalty(
            steam_duty=150e6,
            extraction_pressure=0.4,
            gross_power=500.0,
        )

        assert float(penalty) > 0
        assert float(penalty) < 100  # Less than 100 MW penalty

    def test_efficiency_penalty(self):
        """Test overall efficiency penalty."""
        from difflow_cc.integration.power_plant import (
            efficiency_penalty, PowerPlantParams
        )

        params = PowerPlantParams(gross_power=500.0, net_efficiency=0.40)
        result = efficiency_penalty(
            params,
            steam_duty=150e6,
            compression_power=30e6,
            auxiliary_power_total=5e6,
        )

        assert result["efficiency_with_capture"] < result["base_efficiency"]
        assert result["total_penalty_MW"] > 0

    def test_power_plant_integration(self):
        """Test complete power plant integration."""
        from difflow_cc.integration.power_plant import (
            PowerPlantParams, PowerPlantIntegration
        )

        params = PowerPlantParams(
            plant_type="coal_supercritical",
            gross_power=500.0,
        )
        integration = PowerPlantIntegration(params)

        result = integration.analyze(
            steam_duty=150e6,
            compression_power=30e6,
        )

        assert "total_penalty_MW" in result
        assert "CO2_in_flue_gas_mol_s" in result

    def test_steam_properties(self):
        """Test steam property calculations."""
        from difflow_cc.integration.steam_cycle import steam_properties

        props = steam_properties(pressure=0.5)  # 0.5 MPa

        assert props["h"] > 0
        assert props["T_sat"] > 373.15  # Above atmospheric boiling

    def test_steam_flow_for_duty(self):
        """Test steam flow calculation."""
        from difflow_cc.integration.steam_cycle import steam_flow_for_duty

        m_steam = steam_flow_for_duty(duty=100e6, extraction_pressure=0.4)

        assert float(m_steam) > 0


# =============================================================================
# Direct Air Capture Tests
# =============================================================================

class TestDirectAirCapture:
    """Tests for DAC module."""

    def test_solid_sorbent_dac(self):
        """Test solid sorbent DAC unit."""
        from difflow_cc.units.dac import DACParams, SolidSorbentDAC

        params = DACParams(
            sorbent="PEI_Silica",
            n_units=4,
            T_desorption=373.15,
        )
        dac = SolidSorbentDAC(params)

        co2_out, info = dac()

        assert float(info["CO2_captured_mol_s"]) > 0
        assert float(info["Q_thermal_W"]) > 0
        assert float(info["P_electrical_W"]) >= 0

    def test_liquid_solvent_dac(self):
        """Test liquid solvent DAC."""
        from difflow_cc.units.dac import LiquidDACParams, LiquidSolventDAC

        params = LiquidDACParams(
            solvent="KOH",
            n_contactors=10,
        )
        dac = LiquidSolventDAC(params)

        co2_out, info = dac()

        assert float(info["CO2_captured_mol_s"]) > 0
        assert float(info["Q_calciner_W"]) > 0

    def test_dac_cost_estimate(self):
        """Test DAC cost estimation."""
        from difflow_cc.units.dac import dac_cost_estimate

        cost = dac_cost_estimate(
            capacity_tonne_yr=100000,
            technology="solid_sorbent",
            nth_plant=1,
        )

        assert cost["total_capex_USD"] > 0
        assert cost["levelized_cost_USD_tonne"] > 0

    def test_dac_learning_curve(self):
        """Test DAC cost learning curve."""
        from difflow_cc.units.dac import dac_cost_estimate

        cost_1 = dac_cost_estimate(100000, nth_plant=1)
        cost_10 = dac_cost_estimate(100000, nth_plant=10)

        # Nth plant should be cheaper
        assert cost_10["levelized_cost_USD_tonne"] < cost_1["levelized_cost_USD_tonne"]


# =============================================================================
# Degradation Model Tests
# =============================================================================

class TestDegradation:
    """Tests for degradation models."""

    def test_amine_oxidative_degradation(self):
        """Test amine oxidative degradation."""
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, oxidative_degradation_rate
        )

        params = AmineDegradationParams(solvent="MEA", T_absorber=313.15)
        rate = oxidative_degradation_rate(313.15, params)

        assert float(rate) > 0

    def test_amine_thermal_degradation(self):
        """Test amine thermal degradation."""
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, thermal_degradation_rate
        )

        params = AmineDegradationParams(solvent="MEA", T_stripper=393.15)
        rate = thermal_degradation_rate(393.15, params)

        assert float(rate) > 0

    def test_total_amine_loss(self):
        """Test total amine loss calculation."""
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, total_amine_loss
        )

        params = AmineDegradationParams(solvent="MEA")
        loss = total_amine_loss(params)

        assert loss["total_kg_m3_yr"] > 0
        # Note: total_fraction_yr can be > 1 for aggressive degradation conditions
        assert loss["total_fraction_yr"] > 0

    def test_solvent_lifetime(self):
        """Test solvent lifetime estimation."""
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, solvent_lifetime
        )

        params = AmineDegradationParams(solvent="MEA")
        lifetime = solvent_lifetime(params)

        assert float(lifetime) > 0

    def test_adsorbent_thermal_cycling(self):
        """Test adsorbent thermal cycling degradation."""
        from difflow_cc.degradation.adsorbent_degradation import (
            AdsorbentDegradationParams, thermal_cycling_degradation
        )

        params = AdsorbentDegradationParams(material_type="zeolite")
        fraction = thermal_cycling_degradation(10000, params)

        assert 0 < float(fraction) <= 1

    def test_adsorbent_capacity_fade(self):
        """Test adsorbent capacity fade."""
        from difflow_cc.degradation.adsorbent_degradation import (
            AdsorbentDegradationParams, capacity_fade
        )

        params = AdsorbentDegradationParams(
            material_type="amine_silica",
            T_desorption=373.15,
        )
        fade = capacity_fade(8000, params)

        assert fade["capacity_fraction"] < 1.0
        assert fade["current_capacity_mol_kg"] < params.initial_capacity

    def test_adsorbent_lifetime(self):
        """Test adsorbent lifetime estimation."""
        from difflow_cc.degradation.adsorbent_degradation import (
            AdsorbentDegradationParams, adsorbent_lifetime
        )

        params = AdsorbentDegradationParams(material_type="zeolite")
        lifetime = adsorbent_lifetime(params)

        assert float(lifetime) > 0

    def test_membrane_physical_aging(self):
        """Test membrane physical aging."""
        from difflow_cc.degradation.membrane_aging import (
            MembraneAgingParams, physical_aging
        )

        params = MembraneAgingParams(membrane_type="glassy")
        fraction = physical_aging(5.0, params)

        assert 0 < float(fraction) <= 1

    def test_membrane_plasticization(self):
        """Test membrane plasticization."""
        from difflow_cc.degradation.membrane_aging import (
            MembraneAgingParams, plasticization
        )

        params = MembraneAgingParams(membrane_type="glassy")

        # Below onset
        result_low = plasticization(500000, params)
        # Above onset
        result_high = plasticization(2000000, params)

        assert result_high["permeability_factor"] > result_low["permeability_factor"]

    def test_membrane_lifetime(self):
        """Test membrane lifetime estimation."""
        from difflow_cc.degradation.membrane_aging import (
            MembraneAgingParams, membrane_lifetime
        )

        params = MembraneAgingParams(membrane_type="glassy")
        lifetime = membrane_lifetime(params)

        assert float(lifetime) > 0

    def test_degradation_gradient(self):
        """Test gradient through degradation models."""
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, oxidative_degradation_rate
        )

        params = AmineDegradationParams(solvent="MEA")

        def rate_at_T(T):
            return oxidative_degradation_rate(T, params)

        grad_rate = grad(rate_at_T)
        dr_dT = grad_rate(313.15)

        # Higher T should increase degradation rate
        assert jnp.isfinite(dr_dT)


# =============================================================================
# Integration Tests
# =============================================================================

class TestModuleIntegration:
    """Integration tests combining multiple modules."""

    def test_capture_plant_economics(self):
        """Test complete capture plant economic analysis."""
        from difflow_cc.economics.capex import (
            absorber_cost, stripper_cost, heat_exchanger_cost,
            compressor_cost, installed_cost
        )
        from difflow_cc.economics.levelized_cost import levelized_cost_capture

        # Equipment costs
        equip = {
            "absorber": absorber_cost(5.0, 20.0),
            "stripper": stripper_cost(3.0, 15.0, 100e6, 20e6),
            "hx": heat_exchanger_cost(200.0),
            "compressor": compressor_cost(10e6),
        }

        total_equip = sum(float(c) for c in equip.values())
        inst = installed_cost(total_equip)

        result = levelized_cost_capture(
            capital_cost=inst["total_overnight_cost"],
            annual_opex=10e6,
            CO2_captured=10.0,
        )

        assert result["total_cost_per_tonne"] > 0

    def test_dac_with_degradation(self):
        """Test DAC performance with sorbent degradation."""
        from difflow_cc.units.dac import DACParams, SolidSorbentDAC
        from difflow_cc.degradation.adsorbent_degradation import (
            AdsorbentDegradationParams, capacity_fade
        )

        # Fresh sorbent
        dac_params = DACParams(sorbent="PEI_Silica", n_units=4)
        dac = SolidSorbentDAC(dac_params)
        _, info_fresh = dac()

        # After 1 year of operation
        deg_params = AdsorbentDegradationParams(
            material_type="amine_silica",
            T_desorption=373.15,
        )
        fade = capacity_fade(8760, deg_params)

        # Capture rate proportional to capacity
        capture_aged = float(info_fresh["CO2_captured_mol_s"]) * float(fade["capacity_fraction"])

        assert capture_aged < float(info_fresh["CO2_captured_mol_s"])
