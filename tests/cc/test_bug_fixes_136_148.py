"""Tests for bug fixes #136, #137, #146, #147, #148 in difflow_cc.

Bug #136: Compression specific_power unit conversion (kWh/tonne)
Bug #137: DAC specific energy off by 1000x (GJ/tonne)
Bug #146: compression_power_estimate ad hoc formula
Bug #147: Heat exchanger Q_hot != Q_cold after approach constraints
Bug #148: Two-film mass transfer model incompatible resistance dimensions
"""

import pytest
import jax.numpy as jnp

from difflow.streams import make_stream, total_flow


class TestBug136_CompressionSpecificPower:
    """Bug #136: specific_power should be in kWh/tonne CO2."""

    def test_specific_power_units(self):
        """Check specific_power is in a reasonable range (50-150 kWh/tonne)."""
        from difflow_cc.units.compression import CompressionTrain, CompressionTrainParams

        params = CompressionTrainParams(
            P_inlet=200_000.0,      # 2 bar (typical capture outlet)
            P_outlet=15_000_000.0,  # 150 bar (pipeline)
            eta_isentropic=0.80,
            n_stages=4,
        )
        train = CompressionTrain(params)

        # 1 kmol/s CO2
        co2_stream = make_stream({"CO2": 1000.0}, 313.15, 200_000.0)
        outlet, info = train(co2_stream)

        sp = float(info["specific_power_kWh_tonne"])
        # Literature values for CO2 compression 2 bar -> 150 bar: ~80-130 kWh/tonne
        assert 30 < sp < 300, f"specific_power = {sp} kWh/tonne, expected 30-300"

    def test_specific_power_manual_calculation(self):
        """Verify unit conversion: W / (kg/s) / 3600 = kWh/tonne."""
        from difflow_cc.units.compression import CompressionTrain, CompressionTrainParams

        params = CompressionTrainParams(
            P_inlet=200_000.0,
            P_outlet=15_000_000.0,
            n_stages=4,
        )
        train = CompressionTrain(params)

        co2_stream = make_stream({"CO2": 100.0}, 313.15, 200_000.0)
        outlet, info = train(co2_stream)

        total_power = float(info["total_power"])  # W
        F = 100.0  # mol/s
        MW_CO2 = 44.01 / 1000  # kg/mol
        m_dot = F * MW_CO2  # kg/s

        # Manual: J/kg / 3600 = kWh/tonne
        expected = (total_power / m_dot) / 3600
        actual = float(info["specific_power_kWh_tonne"])

        assert abs(actual - expected) / expected < 1e-6, (
            f"specific_power mismatch: got {actual}, expected {expected}"
        )


class TestBug137_DACSpecificEnergy:
    """Bug #137: DAC specific_thermal should be ~5-10 GJ/tonne, not 0.005-0.01."""

    def test_solid_sorbent_dac_specific_thermal(self):
        """Solid sorbent DAC specific thermal energy should be ~3-15 GJ/tonne."""
        from difflow_cc.units.dac import SolidSorbentDAC, DACParams

        params = DACParams(
            sorbent="PEI_Silica",
            n_units=4,
            T_desorption=373.15,
        )
        dac = SolidSorbentDAC(params)
        co2_out, info = dac()

        specific_thermal = float(info["specific_thermal_GJ_tonne"])
        # Literature: solid sorbent DAC typically 5-10 GJ/tonne thermal
        assert 1.0 < specific_thermal < 50.0, (
            f"specific_thermal = {specific_thermal} GJ/tonne, expected 1-50"
        )

    def test_solid_sorbent_dac_manual_conversion(self):
        """Verify unit conversion: W / (kg/s) / 1e6 = GJ/tonne."""
        from difflow_cc.units.dac import SolidSorbentDAC, DACParams

        params = DACParams(
            sorbent="PEI_Silica",
            n_units=4,
            T_desorption=373.15,
        )
        dac = SolidSorbentDAC(params)
        co2_out, info = dac()

        Q = float(info["Q_thermal_W"])        # W = J/s
        m = float(info["CO2_captured_kg_s"])   # kg/s

        # Manual: J/kg / 1e6 = GJ/tonne (since 1 GJ = 1e9 J, 1 tonne = 1000 kg)
        expected = (Q / m) / 1e6
        actual = float(info["specific_thermal_GJ_tonne"])

        assert abs(actual - expected) / (expected + 1e-30) < 1e-6, (
            f"specific_thermal mismatch: got {actual}, expected {expected}"
        )

    def test_liquid_solvent_dac_specific_thermal(self):
        """Liquid solvent DAC specific thermal energy should be reasonable."""
        from difflow_cc.units.dac import LiquidSolventDAC, LiquidDACParams

        params = LiquidDACParams(solvent="KOH", n_contactors=10)
        dac = LiquidSolventDAC(params)
        co2_out, info = dac()

        specific_thermal = float(info["specific_thermal_GJ_tonne"])
        # Liquid DAC uses calciner at ~900C, so thermal is high (~5-10 GJ/tonne)
        assert 1.0 < specific_thermal < 30.0, (
            f"specific_thermal = {specific_thermal} GJ/tonne, expected 1-30"
        )


class TestBug146_CompressionPowerEstimate:
    """Bug #146: compression_power_estimate should use proper multi-stage model."""

    def test_single_stage_matches_basic_formula(self):
        """For 1 stage, W = F * R * T / k * (ratio^k - 1) / eta."""
        from difflow_cc.units.compression import compression_power_estimate

        F = 100.0  # mol/s
        P_in = 500_000.0  # 5 bar
        # Choose ratio < 3 so n_stages = ceil(log(ratio)/log(3)) = 1
        P_out = 1_000_000.0  # 10 bar, ratio = 2
        T_in = 313.15
        eta = 0.80
        R = 8.314

        power = float(compression_power_estimate(F, P_in, P_out, T_in, eta))

        # Manual single-stage calculation
        ratio = P_out / P_in
        gamma = 1.3
        k = (gamma - 1) / gamma
        W_per_mol = R * T_in / k * (ratio**k - 1)
        expected = F * W_per_mol / eta

        assert abs(power - expected) / expected < 0.01, (
            f"Single-stage power mismatch: got {power}, expected {expected}"
        )

    def test_multistage_less_than_single_stage(self):
        """Multi-stage with intercooling should use less work than single-stage."""
        from difflow_cc.units.compression import compression_power_estimate

        F = 100.0  # mol/s
        P_in = 100_000.0   # 1 bar
        P_out = 10_000_000.0  # 100 bar, ratio = 100
        T_in = 313.15
        eta = 0.80
        R = 8.314

        power = float(compression_power_estimate(F, P_in, P_out, T_in, eta))

        # Single-stage (no intercooling) reference
        ratio = P_out / P_in
        gamma = 1.3
        k = (gamma - 1) / gamma
        W_single = F * R * T_in / k * (ratio**k - 1) / eta

        # Multi-stage with intercooling should be significantly less
        assert power < W_single, (
            f"Multi-stage power ({power:.0f} W) should be less than "
            f"single-stage ({W_single:.0f} W)"
        )

    def test_reasonable_specific_power(self):
        """Check compression power gives reasonable kWh/tonne."""
        from difflow_cc.units.compression import compression_power_estimate

        F = 100.0  # mol/s
        P_in = 200_000.0    # 2 bar
        P_out = 15_000_000.0  # 150 bar
        T_in = 313.15

        power = float(compression_power_estimate(F, P_in, P_out, T_in, eta=0.80))

        # Convert to kWh/tonne
        MW_CO2 = 44.01 / 1000  # kg/mol
        m_dot = F * MW_CO2  # kg/s
        sp = (power / m_dot) / 3600  # kWh/tonne

        assert 50 < sp < 250, f"specific power = {sp:.1f} kWh/tonne, expected 50-250"


class TestBug147_HeatExchangerEnergyBalance:
    """Bug #147: Q_hot should equal Q_cold after approach constraints."""

    def test_energy_balance_symmetric(self):
        """With equal flows and Cp, Q_hot == Q_cold exactly."""
        from difflow_cc.units.heat_integration import HeatExchanger, HeatExchangerParams

        params = HeatExchangerParams(
            U=500.0,
            A=100.0,
            min_approach=10.0,
        )
        hx = HeatExchanger(params)

        hot_in = make_stream({"H2O": 10.0}, 400.0, 101325.0)
        cold_in = make_stream({"H2O": 10.0}, 300.0, 101325.0)

        hot_out, cold_out, info = hx(hot_in, cold_in, Cp_hot=75.0, Cp_cold=75.0)

        Q = float(info["Q"])
        T_hot_in = float(info["T_hot_in"])
        T_hot_out = float(info["T_hot_out"])
        T_cold_in = float(info["T_cold_in"])
        T_cold_out = float(info["T_cold_out"])

        F_hot = float(total_flow(hot_in))
        F_cold = float(total_flow(cold_in))

        Q_hot = F_hot * 75.0 * (T_hot_in - T_hot_out)
        Q_cold = F_cold * 75.0 * (T_cold_out - T_cold_in)

        assert abs(Q_hot - Q_cold) < 1e-6, (
            f"Energy imbalance: Q_hot={Q_hot:.4f}, Q_cold={Q_cold:.4f}"
        )

    def test_energy_balance_asymmetric(self):
        """With unequal flows, Q_hot should still equal Q_cold."""
        from difflow_cc.units.heat_integration import HeatExchanger, HeatExchangerParams

        params = HeatExchangerParams(
            U=500.0,
            A=50.0,
            min_approach=15.0,
        )
        hx = HeatExchanger(params)

        hot_in = make_stream({"H2O": 20.0}, 450.0, 101325.0)
        cold_in = make_stream({"H2O": 10.0}, 300.0, 101325.0)

        hot_out, cold_out, info = hx(hot_in, cold_in, Cp_hot=75.0, Cp_cold=75.0)

        T_hot_in = float(info["T_hot_in"])
        T_hot_out = float(info["T_hot_out"])
        T_cold_in = float(info["T_cold_in"])
        T_cold_out = float(info["T_cold_out"])

        Q_hot = 20.0 * 75.0 * (T_hot_in - T_hot_out)
        Q_cold = 10.0 * 75.0 * (T_cold_out - T_cold_in)

        assert abs(Q_hot - Q_cold) < 1e-6, (
            f"Energy imbalance: Q_hot={Q_hot:.4f}, Q_cold={Q_cold:.4f}"
        )

    def test_approach_temperature_respected(self):
        """Minimum approach temperature should be maintained."""
        from difflow_cc.units.heat_integration import HeatExchanger, HeatExchangerParams

        min_approach = 10.0
        params = HeatExchangerParams(
            U=1000.0,  # Very high U*A to force approach constraint
            A=1000.0,
            min_approach=min_approach,
        )
        hx = HeatExchanger(params)

        hot_in = make_stream({"H2O": 10.0}, 350.0, 101325.0)
        cold_in = make_stream({"H2O": 10.0}, 300.0, 101325.0)

        hot_out, cold_out, info = hx(hot_in, cold_in)

        T_hot_out = float(info["T_hot_out"])
        T_cold_out = float(info["T_cold_out"])
        T_hot_in = float(info["T_hot_in"])
        T_cold_in = float(info["T_cold_in"])

        # Counter-current approach temps
        approach_hot_end = T_hot_in - T_cold_out
        approach_cold_end = T_hot_out - T_cold_in

        assert approach_hot_end >= min_approach - 0.01, (
            f"Hot end approach {approach_hot_end:.2f} < {min_approach}"
        )
        assert approach_cold_end >= min_approach - 0.01, (
            f"Cold end approach {approach_cold_end:.2f} < {min_approach}"
        )


class TestBug148_MassTransferResistance:
    """Bug #148: Two-film model should have consistent resistance dimensions."""

    def test_overall_less_than_individual(self):
        """K_G should be less than k_G (series resistance reduces overall)."""
        from difflow_cc.kinetics.mass_transfer import overall_mass_transfer

        k_G = 0.01   # m/s
        k_L = 1e-4   # m/s
        E = 10.0
        H = 3400.0   # Pa*m^3/mol (CO2 in water ~3400 at 25C)
        P = 101325.0

        K_G = float(overall_mass_transfer(k_G, k_L, E, H, P))

        assert K_G > 0, f"K_G should be positive, got {K_G}"
        assert K_G < k_G, (
            f"K_G ({K_G}) should be less than k_G ({k_G}) due to liquid resistance"
        )

    def test_no_liquid_resistance(self):
        """With very large H*E*k_L, K_G should approach k_G."""
        from difflow_cc.kinetics.mass_transfer import overall_mass_transfer

        k_G = 0.01
        k_L = 1.0     # Very fast liquid side
        E = 1000.0    # Very high enhancement
        H = 1e6       # Very large H
        P = 101325.0

        K_G = float(overall_mass_transfer(k_G, k_L, E, H, P))

        # With negligible liquid resistance, K_G ≈ k_G
        assert abs(K_G - k_G) / k_G < 0.01, (
            f"K_G ({K_G}) should approach k_G ({k_G}) when liquid resistance is negligible"
        )

    def test_enhancement_increases_transfer(self):
        """Higher enhancement factor should increase K_G."""
        from difflow_cc.kinetics.mass_transfer import overall_mass_transfer

        k_G = 0.01
        k_L = 1e-4
        H = 3400.0
        P = 101325.0

        K_G_low = float(overall_mass_transfer(k_G, k_L, E=1.0, H=H, P=P))
        K_G_high = float(overall_mass_transfer(k_G, k_L, E=100.0, H=H, P=P))

        assert K_G_high > K_G_low, (
            f"Higher E should give higher K_G: E=1 -> {K_G_low}, E=100 -> {K_G_high}"
        )

    def test_P_parameter_accepted(self):
        """P parameter should still be accepted (backward compatibility)."""
        from difflow_cc.kinetics.mass_transfer import overall_mass_transfer

        # Should not raise an error (P is still a parameter even if unused in new formula)
        K_G = overall_mass_transfer(k_G=0.01, k_L=1e-4, E=10.0, H=3400.0, P=101325.0)
        assert float(K_G) > 0
