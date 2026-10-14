"""Smoke tests for the carbon capture support modules: heat integration,
CO2 compression, economics, power-plant integration, direct air capture
and degradation."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream  # noqa: E402


def _finite(x):
    return bool(jnp.all(jnp.isfinite(jnp.asarray(x))))


def _hot_cold():
    return (make_stream({"H2O": 10.0}, T=373.15, P=101325.0),
            make_stream({"H2O": 10.0}, T=298.15, P=101325.0))


class TestHeatIntegration:
    def test_heat_exchanger(self):
        from difflow_cc.units.heat_integration import HeatExchanger, HeatExchangerParams
        hot_out, cold_out, info = HeatExchanger(HeatExchangerParams(U=500.0, A=50.0))(
            *_hot_cold())
        assert _finite(info["Q"]) and _finite(hot_out["T"]) and _finite(cold_out["T"])

    def test_lean_rich_and_recovery_system(self):
        from difflow_cc.units.heat_integration import (
            HeatRecoverySystem, HeatRecoverySystemParams,
            LeanRichExchanger, LeanRichExchangerParams)
        lean = make_stream({"H2O": 100.0}, T=393.15, P=2e5)
        rich = make_stream({"H2O": 100.0}, T=323.15, P=2e5)
        for unit in (LeanRichExchanger(LeanRichExchangerParams(effectiveness=0.85)),
                     HeatRecoverySystem(HeatRecoverySystemParams(
                         lrhx_effectiveness=0.85, T_lean_target=313.15))):
            lean_out, rich_out, _ = unit(lean, rich)
            assert _finite(lean_out["T"]) and _finite(rich_out["T"])

    def test_intercooler(self):
        from difflow_cc.units.heat_integration import Intercooler, IntercoolerParams
        out, _ = Intercooler(IntercoolerParams(T_coolant=298.15, approach=5.0))(
            make_stream({"CO2": 1.0, "N2": 5.0}, T=350.0, P=101325.0))
        assert _finite(out["T"])

    def test_grad(self):
        from difflow_cc.units.heat_integration import HeatExchanger, HeatExchangerParams

        def Q(area):
            return HeatExchanger(HeatExchangerParams(U=500.0, A=area))(*_hot_cold())[2]["Q"]
        assert _finite(jax.grad(Q)(50.0))


class TestCompression:
    def test_compressor_and_train(self):
        from difflow_cc.units.compression import (
            CompressionTrain, CompressionTrainParams, Compressor, CompressorParams)
        inlet = make_stream({"CO2": 10.0}, T=313.15, P=2e5)
        out, info = Compressor(CompressorParams(pressure_ratio=3.0))(inlet)
        assert _finite(out["P"]) and _finite(info["P_electrical"])
        out, info = CompressionTrain(CompressionTrainParams(P_inlet=2e5, P_outlet=1.5e7))(inlet)
        assert _finite(out["P"]) and _finite(info["total_power"])

    def test_functions_and_grad(self):
        from difflow_cc.units.compression import (
            co2_density, compression_power_estimate, is_supercritical)
        assert _finite(co2_density(320.0, 1e7))
        assert bool(is_supercritical(320.0, 1e7))
        assert _finite(jax.grad(compression_power_estimate)(10.0, 2e5, 1.5e7))

    def test_inconsistent_P_inlet_rejected(self):
        from difflow_cc import CompressionTrain, CompressionTrainParams
        with pytest.raises(ValueError, match="does not match"):
            CompressionTrain(CompressionTrainParams(P_inlet=2e5))(
                make_stream({"CO2": 1.0}, T=313.15, P=101325.0))


class TestEconomics:
    def test_capex_opex_levelized(self):
        from difflow_cc.economics.capex import (
            absorber_cost, compressor_cost, heat_exchanger_cost, installed_cost,
            stripper_cost)
        from difflow_cc.economics.levelized_cost import levelized_cost_capture
        from difflow_cc.economics.opex import total_operating_cost
        equip = sum(float(c) for c in (
            absorber_cost(5.0, 20.0), stripper_cost(3.0, 15.0, 100e6, 20e6),
            heat_exchanger_cost(200.0), compressor_cost(10e6)))
        inst = installed_cost(equip)
        opex = total_operating_cost(steam_duty=100e6, electricity=10e6,
                                    cooling_duty=50e6, CO2_captured=10.0,
                                    capital_cost=100e6)
        lc = levelized_cost_capture(capital_cost=inst["total_overnight_cost"],
                                    annual_opex=float(opex["total_opex"]),
                                    CO2_captured=10.0)
        assert _finite(lc["total_cost_per_tonne"])

    def test_grad(self):
        from difflow_cc.economics.capex import absorber_cost
        assert _finite(jax.grad(absorber_cost)(5.0, 20.0))

    def test_unknown_equipment_raises(self):
        from difflow_cc.economics import validate_scaling_exponent
        with pytest.raises(KeyError):
            validate_scaling_exponent("nonexistent_unit")


class TestProcessIntegration:
    def test_power_plant(self):
        from difflow_cc.integration.power_plant import (
            PowerPlantIntegration, PowerPlantParams, flue_gas_composition)
        assert "CO2" in flue_gas_composition("coal_supercritical")
        result = PowerPlantIntegration(PowerPlantParams(
            plant_type="coal_supercritical", gross_power=500.0)).analyze(
            steam_duty=150e6, compression_power=30e6)
        assert _finite(result["total_penalty_MW"])

    def test_steam_cycle(self):
        from difflow_cc.integration.steam_cycle import steam_flow_for_duty, steam_properties
        assert _finite(steam_properties(pressure=0.5)["h"])
        assert _finite(steam_flow_for_duty(duty=100e6, extraction_pressure=0.4))


class TestDirectAirCapture:
    def test_units_and_cost(self):
        from difflow_cc.units.dac import (
            DACParams, LiquidDACParams, LiquidSolventDAC, SolidSorbentDAC,
            dac_cost_estimate)
        _, info = SolidSorbentDAC(DACParams(sorbent="PEI_Silica", n_units=4))()
        assert _finite(info["CO2_captured_mol_s"])
        _, info = LiquidSolventDAC(LiquidDACParams(solvent="KOH", n_contactors=10))()
        assert _finite(info["CO2_captured_mol_s"])
        assert _finite(dac_cost_estimate(100000)["levelized_cost_USD_tonne"])

    def test_liquid_grad(self):
        from difflow_cc.units.dac import LiquidDACParams, LiquidSolventDAC

        def cap(h):
            return LiquidSolventDAC(LiquidDACParams(contactor_height=h))()[1][
                "CO2_captured_mol_s"]
        assert _finite(jax.grad(cap)(8.0))

    def test_desorption_colder_than_adsorption_rejected(self):
        from difflow_cc.units.dac import DACParams
        with pytest.raises(ValueError, match="T_desorption"):
            DACParams(T_desorption=290.0)


class TestDegradation:
    def test_amine(self):
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, solvent_lifetime, total_amine_loss)
        p = AmineDegradationParams(solvent="MEA")
        assert _finite(total_amine_loss(p)["total_fraction_yr"])
        assert _finite(solvent_lifetime(p))

    def test_adsorbent(self):
        from difflow_cc.degradation.adsorbent_degradation import (
            AdsorbentDegradationParams, adsorbent_lifetime, capacity_fade)
        p = AdsorbentDegradationParams(material_type="amine_silica", T_desorption=373.15)
        assert _finite(capacity_fade(8000, p)["capacity_fraction"])
        assert _finite(adsorbent_lifetime(p))

    def test_membrane(self):
        from difflow_cc.degradation.membrane_aging import (
            MembraneAgingParams, membrane_lifetime, plasticization)
        p = MembraneAgingParams(membrane_type="glassy")
        assert _finite(plasticization(2e6, p)["permeability_factor"])
        assert _finite(membrane_lifetime(p))

    def test_grad(self):
        from difflow_cc.degradation.amine_degradation import (
            AmineDegradationParams, oxidative_degradation_rate)
        p = AmineDegradationParams(solvent="MEA")
        assert _finite(jax.grad(lambda T: oxidative_degradation_rate(T, p))(313.15))
