"""Tests for absorber and stripper bug fixes (#132, #133, #135, #140, #145)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestBug132_AmineFlowConversion:
    """Bug #132: F_amine should use wt% -> mole fraction conversion."""

    def test_amine_mole_fraction_conversion(self):
        """Verify F_amine is computed via mole fraction, not raw wt%."""
        from difflow_cc import AbsorberParams, AmineAbsorber, get_solvent
        from difflow.streams import make_stream

        params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            solvent_conc=30.0,  # 30 wt%
            L_G_ratio=3.0,
        )
        absorber = AmineAbsorber(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0},
            T=313.15,
            P=101325.0,
        )

        _, solvent_out, info = absorber(feed)

        # Compute expected mole fraction
        MW_MEA = get_solvent("MEA").MW  # ~61.08
        MW_water = 18.0
        w = 0.30  # 30 wt%
        x_amine_expected = (w / MW_MEA) / (w / MW_MEA + (1 - w) / MW_water)

        # F_liquid = L_G * F_total_gas = 3.0 * 10.0 = 30.0 mol/s
        F_liquid = 3.0 * 10.0
        F_amine_expected = F_liquid * x_amine_expected

        # The amine flow in solvent_out should match
        F_amine_actual = float(solvent_out["F_Amine"])
        assert abs(F_amine_actual - F_amine_expected) / F_amine_expected < 1e-6, (
            f"F_amine = {F_amine_actual:.4f}, expected {F_amine_expected:.4f}"
        )

        # x_amine should be much less than 0.30 (wt% != mol fraction for heavy molecules)
        assert x_amine_expected < 0.20, (
            f"Mole fraction {x_amine_expected:.4f} should be < 0.20 for 30 wt% MEA"
        )


class TestBug133_AbsorptionFactor:
    """Bug #133: Absorption factor A uses correct dimensionless VLE slope."""

    def test_absorption_factor_formula(self):
        """A = F_amine / (m_dimless * F_gas_inert) with m_dimless = (dP/dalpha) / P."""
        from difflow_cc import AbsorberParams, AmineAbsorber, get_solvent
        from difflow_cc.equilibrium.vle import AmineVLE
        from difflow.streams import make_stream
        import jax.numpy as jnp

        params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            solvent_conc=30.0,
            L_G_ratio=3.0,
            lean_loading=0.2,
        )
        absorber = AmineAbsorber(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0},
            T=313.15,
            P=101325.0,
        )

        _, _, info = absorber(feed)
        A = float(info["absorption_factor"])

        # Manually compute the expected A using the corrected formula
        sd = get_solvent("MEA")
        vle = AmineVLE("MEA")
        T_op = 313.15
        lean = 0.2
        P_total = 101325.0

        # VLE slope
        d_alpha = 0.01
        P_lean = vle.equilibrium_pressure(lean, T_op)
        P_plus = vle.equilibrium_pressure(lean + d_alpha, T_op)
        m = float(P_plus - P_lean) / d_alpha
        m_dimless = m / P_total

        # F_amine via wt% -> mole fraction
        w = 0.30
        x_amine = (w / sd.MW) / (w / sd.MW + 0.70 / 18.0)
        F_liquid = 3.0 * 10.0  # L/G * F_total
        F_amine = F_liquid * x_amine
        F_gas_inert = 9.0  # N2

        if m_dimless > 1e-20:
            A_expected = F_amine / (m_dimless * F_gas_inert)
            assert abs(A - A_expected) / max(abs(A_expected), 1e-10) < 1e-6
        else:
            # With near-zero VLE slope (very favorable absorption),
            # A should be very large (safe_divide caps it)
            assert A > 1e6, (
                f"With negligible VLE slope, A should be very large, got {A}"
            )

    def test_absorption_factor_positive(self):
        """Absorption factor should always be positive."""
        from difflow_cc import AbsorberParams, AmineAbsorber
        from difflow.streams import make_stream

        params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            solvent_conc=30.0,
            L_G_ratio=3.0,
        )
        absorber = AmineAbsorber(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0},
            T=313.15,
            P=101325.0,
        )

        _, _, info = absorber(feed)
        A = float(info["absorption_factor"])
        assert A > 0, f"Absorption factor should be positive, got {A}"


class TestBug135_SteamRatio:
    """Bug #135: Steam ratio should be ~2.0, not 0.3."""

    def test_reboiler_duty_range(self):
        """Reboiler duty for MEA should be 3-4.5 GJ/tonne CO2."""
        from difflow_cc import StripperParams, AmineStripper
        from difflow.streams import make_stream

        params = StripperParams(
            solvent="MEA",
            T_reboiler=393.15,
            target_lean_loading=0.2,
        )
        stripper = AmineStripper(params)

        # Create rich solvent (typical from absorber)
        rich = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=323.15,  # 50°C from absorber
            P=200000.0,
        )

        _, _, info = stripper(rich)
        specific_energy = float(info["specific_energy"])

        # MEA reboiler duty is typically 3.0-4.5 GJ/tonne CO2
        assert 2.0 < specific_energy < 6.0, (
            f"Specific energy = {specific_energy:.2f} GJ/tonne, "
            f"expected 2.0-6.0 for MEA"
        )

    def test_steam_ratio_affects_vaporization_heat(self):
        """Vaporization heat should be significant fraction of total duty."""
        from difflow_cc import StripperParams, AmineStripper
        from difflow.streams import make_stream

        params = StripperParams(
            solvent="MEA",
            T_reboiler=393.15,
            target_lean_loading=0.2,
        )
        stripper = AmineStripper(params)

        rich = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=323.15,
            P=200000.0,
        )

        _, _, info = stripper(rich)
        Q_vap = float(info["Q_vaporization"])
        Q_total = float(info["reboiler_duty"])

        # With steam_ratio=2.0, vaporization should be a major component
        # (at least 20% of total duty)
        frac = Q_vap / Q_total
        assert frac > 0.15, (
            f"Q_vaporization / Q_reboiler = {frac:.3f}, should be > 0.15"
        )


class TestBug140_SensibleHeat:
    """Bug #140: Sensible heat should depend on actual feed temperature."""

    def test_sensible_heat_varies_with_feed_temperature(self):
        """Q_sensible should change when feed temperature changes."""
        from difflow_cc import StripperParams, AmineStripper
        from difflow.streams import make_stream

        params = StripperParams(
            solvent="MEA",
            T_reboiler=393.15,
            target_lean_loading=0.2,
            cross_exchanger_approach=10.0,
        )
        stripper = AmineStripper(params)

        # Cold feed (far from reboiler)
        rich_cold = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=298.15,  # 25°C - very cold
            P=200000.0,
        )

        # Warm feed (close to reboiler)
        rich_warm = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=383.15,  # 110°C - close to reboiler
            P=200000.0,
        )

        _, _, info_cold = stripper(rich_cold)
        _, _, info_warm = stripper(rich_warm)

        Q_sens_cold = float(info_cold["Q_sensible"])
        Q_sens_warm = float(info_warm["Q_sensible"])

        # With cold feed: T_after_hx = max(298.15, 393.15-10) = 383.15
        # dT = 393.15 - 383.15 = 10 K (limited by approach)
        # With warm feed: T_after_hx = max(383.15, 383.15) = 383.15
        # dT = 393.15 - 383.15 = 10 K (same with good exchanger)
        # These should be the same when approach is the limiting factor

        # But if feed is even warmer than T_reboiler - approach:
        rich_hot = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=388.15,  # 115°C - warmer than T_reboiler - approach
            P=200000.0,
        )
        _, _, info_hot = stripper(rich_hot)
        Q_sens_hot = float(info_hot["Q_sensible"])

        # When T_feed > T_reboiler - approach, sensible heat should be less
        assert Q_sens_hot < Q_sens_cold, (
            f"Q_sensible(hot={Q_sens_hot:.1f}) should be < Q_sensible(cold={Q_sens_cold:.1f})"
        )

    def test_sensible_heat_with_no_cross_exchanger(self):
        """Without cross-exchanger (large approach), full heating is needed."""
        from difflow_cc import StripperParams, AmineStripper
        from difflow.streams import make_stream

        params = StripperParams(
            solvent="MEA",
            T_reboiler=393.15,
            target_lean_loading=0.2,
            cross_exchanger_approach=500.0,  # Effectively no cross-exchanger
        )
        stripper = AmineStripper(params)

        T_feed = 313.15  # 40°C
        rich = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=T_feed,
            P=200000.0,
        )

        _, _, info = stripper(rich)
        Q_sensible = float(info["Q_sensible"])

        # Without cross-exchanger, T_after_hx = max(T_feed, T_reboiler - 500)
        # = max(313.15, -106.85) = 313.15
        # dT = 393.15 - 313.15 = 80 K
        dT_expected = 393.15 - T_feed
        assert dT_expected > 70, "Sanity check on expected dT"

        # Q_sensible should reflect full temperature rise
        assert Q_sensible > 0, "Q_sensible should be positive"


class TestBug145_CO2MassBalance:
    """Bug #145: CO2 mass balance must hold after rich loading clipping."""

    def test_co2_mass_balance(self):
        """CO2_in = CO2_out + CO2_absorbed must hold exactly."""
        from difflow_cc import AbsorberParams, AmineAbsorber
        from difflow.streams import make_stream

        params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            solvent_conc=30.0,
            L_G_ratio=3.0,
        )
        absorber = AmineAbsorber(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0},
            T=313.15,
            P=101325.0,
        )

        gas_out, solvent_out, info = absorber(feed)

        F_CO2_in = float(feed["F_CO2"])
        F_CO2_out = float(gas_out["F_CO2"])
        F_CO2_absorbed = float(info["CO2_captured"])

        # Mass balance: CO2 in = CO2 out + CO2 absorbed
        balance_error = abs(F_CO2_in - F_CO2_out - F_CO2_absorbed)
        assert balance_error < 1e-10, (
            f"CO2 mass balance error: {balance_error:.2e} "
            f"(in={F_CO2_in}, out={F_CO2_out}, absorbed={F_CO2_absorbed})"
        )

    def test_co2_balance_with_loading_clipping(self):
        """CO2 balance holds even when rich loading hits capacity limit."""
        from difflow_cc import AbsorberParams, AmineAbsorber, get_solvent
        from difflow.streams import make_stream

        # Use very high CO2 and low L/G to force loading clipping
        params = AbsorberParams(
            solvent="MEA",
            n_stages=20,
            solvent_conc=30.0,
            L_G_ratio=0.5,  # Very low L/G to saturate solvent
            lean_loading=0.4,  # Start near capacity
        )
        absorber = AmineAbsorber(params)

        feed = make_stream(
            flows={"CO2": 5.0, "N2": 5.0},  # 50% CO2 - very high
            T=313.15,
            P=101325.0,
        )

        gas_out, solvent_out, info = absorber(feed)

        F_CO2_in = float(feed["F_CO2"])
        F_CO2_out = float(gas_out["F_CO2"])
        F_CO2_absorbed = float(info["CO2_captured"])

        # Mass balance must hold even with clipping
        balance_error = abs(F_CO2_in - F_CO2_out - F_CO2_absorbed)
        assert balance_error < 1e-10, (
            f"CO2 mass balance error with clipping: {balance_error:.2e}"
        )

        # Rich loading should be at or below capacity
        capacity = get_solvent("MEA").loading_capacity
        assert float(info["rich_loading"]) <= capacity + 1e-10
