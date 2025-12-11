"""Tests for heat exchanger unit operations."""

import pytest
import jax
import jax.numpy as jnp

from difflow import make_stream
from difflow.units.heat_exchanger import (
    Heater,
    HeaterParams,
    Cooler,
    CoolerParams,
    CounterCurrentHX,
    CoCurrentHX,
    HeatExchangerParams,
    log_mean_temperature_difference,
    effectiveness_counter_current,
    effectiveness_co_current,
    design_heat_exchanger,
    size_heat_exchanger,
)


class TestLMTD:
    """Tests for log mean temperature difference."""

    def test_lmtd_basic(self):
        """Test LMTD calculation."""
        # Counter-current example: hot 400→350, cold 300→350
        dT1 = jnp.array(50.0)  # 400 - 350
        dT2 = jnp.array(50.0)  # 350 - 300
        lmtd = log_mean_temperature_difference(dT1, dT2)
        # When dT1 = dT2, LMTD = dT
        assert float(lmtd) == pytest.approx(50.0, rel=0.01)

    def test_lmtd_unequal(self):
        """Test LMTD with unequal temperature differences."""
        dT1 = jnp.array(100.0)
        dT2 = jnp.array(50.0)
        lmtd = log_mean_temperature_difference(dT1, dT2)
        # LMTD = (100 - 50) / ln(100/50) = 50 / ln(2) ≈ 72.1
        expected = 50.0 / jnp.log(2.0)
        assert float(lmtd) == pytest.approx(float(expected), rel=0.01)

    def test_lmtd_differentiable(self):
        """Test that LMTD is differentiable."""
        def lmtd_fn(dT1):
            return log_mean_temperature_difference(dT1, jnp.array(50.0))

        grad = jax.grad(lmtd_fn)(jnp.array(100.0))
        assert jnp.isfinite(grad)


class TestEffectiveness:
    """Tests for effectiveness-NTU correlations."""

    def test_counter_current_low_ntu(self):
        """Test counter-current effectiveness at low NTU."""
        eps = effectiveness_counter_current(jnp.array(0.5), jnp.array(0.5))
        # At NTU=0.5, Cr=0.5, effectiveness should be moderate
        assert 0.3 < float(eps) < 0.6

    def test_counter_current_high_ntu(self):
        """Test counter-current approaches 1 at high NTU."""
        eps = effectiveness_counter_current(jnp.array(10.0), jnp.array(0.5))
        assert float(eps) > 0.95

    def test_co_current_limit(self):
        """Test co-current has lower limit than counter-current."""
        NTU = jnp.array(10.0)
        Cr = jnp.array(0.5)
        eps_counter = effectiveness_counter_current(NTU, Cr)
        eps_co = effectiveness_co_current(NTU, Cr)
        # Counter-current always higher
        assert float(eps_counter) > float(eps_co)

    def test_effectiveness_differentiable(self):
        """Test effectiveness is differentiable."""
        def eps_fn(NTU):
            return effectiveness_counter_current(NTU, jnp.array(0.5))

        grad = jax.grad(eps_fn)(jnp.array(1.0))
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # Effectiveness increases with NTU


class TestHeater:
    """Tests for single-stream heater."""

    def test_heater_duty_mode(self):
        """Test heater with specified duty."""
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        heater = Heater(HeaterParams(duty=7500.0, Cp=75.0))
        outlet, info = heater(stream)

        # Q = m * Cp * dT => dT = Q / (m * Cp) = 7500 / (10 * 75) = 10 K
        assert float(outlet["T"]) == pytest.approx(310.0, rel=0.01)
        assert float(info["Q"]) == pytest.approx(7500.0, rel=0.01)

    def test_heater_temperature_mode(self):
        """Test heater with specified outlet temperature."""
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        heater = Heater(HeaterParams(T_out=350.0, Cp=75.0))
        outlet, info = heater(stream)

        assert float(outlet["T"]) == pytest.approx(350.0, rel=0.01)
        # Q = m * Cp * dT = 10 * 75 * 50 = 37500 W
        assert float(info["Q"]) == pytest.approx(37500.0, rel=0.01)

    def test_heater_rating_mode(self):
        """Test heater rating with UA and utility temperature."""
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        heater = Heater(HeaterParams(UA=500.0, T_utility=400.0, Cp=75.0))
        outlet, info = heater(stream)

        # Should heat up but not reach utility temperature
        assert float(outlet["T"]) > 300.0
        assert float(outlet["T"]) < 400.0
        assert float(info["Q"]) > 0

    def test_heater_differentiable(self):
        """Test heater is differentiable."""
        def outlet_T(duty):
            stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
            heater = Heater(HeaterParams(Cp=75.0))
            outlet, _ = heater(stream, duty=duty)
            return outlet["T"]

        grad = jax.grad(outlet_T)(jnp.array(5000.0))
        assert jnp.isfinite(grad)
        # dT/dQ = 1/(m*Cp) = 1/(10*75) = 0.00133
        assert float(grad) == pytest.approx(1.0 / (10.0 * 75.0), rel=0.01)


class TestCooler:
    """Tests for single-stream cooler."""

    def test_cooler_duty_mode(self):
        """Test cooler with specified duty."""
        stream = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cooler = Cooler(CoolerParams(duty=7500.0, Cp=75.0))
        outlet, info = cooler(stream)

        # Temperature should decrease
        assert float(outlet["T"]) == pytest.approx(390.0, rel=0.01)
        assert float(info["Q"]) == pytest.approx(7500.0, rel=0.01)

    def test_cooler_temperature_mode(self):
        """Test cooler with specified outlet temperature."""
        stream = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cooler = Cooler(CoolerParams(T_out=350.0, Cp=75.0))
        outlet, info = cooler(stream)

        assert float(outlet["T"]) == pytest.approx(350.0, rel=0.01)
        # Q = 10 * 75 * 50 = 37500 W
        assert float(info["Q"]) == pytest.approx(37500.0, rel=0.01)


class TestCounterCurrentHX:
    """Tests for counter-current heat exchanger."""

    def test_counter_current_basic(self):
        """Test basic counter-current heat exchanger."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        hx = CounterCurrentHX(HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0))
        hot_out, cold_out, info = hx(hot_in, cold_in)

        # Heat should transfer from hot to cold
        assert float(hot_out["T"]) < 400.0
        assert float(cold_out["T"]) > 300.0

        # Energy balance: Q_hot = Q_cold
        Q_hot = 10.0 * 75.0 * (400.0 - float(hot_out["T"]))
        Q_cold = 10.0 * 75.0 * (float(cold_out["T"]) - 300.0)
        assert Q_hot == pytest.approx(Q_cold, rel=0.01)
        assert float(info["Q"]) == pytest.approx(Q_hot, rel=0.01)

    def test_counter_current_approach(self):
        """Test temperature approach in counter-current HX."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        # High UA should give small approach
        hx = CounterCurrentHX(HeatExchangerParams(UA=5000.0, Cp_hot=75.0, Cp_cold=75.0))
        hot_out, cold_out, info = hx(hot_in, cold_in)

        # With balanced flow (same mCp), temperatures should cross-over
        # For counter-current, outlet temps approach each other
        assert float(info["approach"]) > 0  # No temperature cross

    def test_counter_current_differentiable(self):
        """Test counter-current HX is differentiable."""
        def heat_duty(UA):
            hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
            cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)
            hx = CounterCurrentHX(HeatExchangerParams(Cp_hot=75.0, Cp_cold=75.0))
            _, _, info = hx(hot_in, cold_in, UA=UA)
            return info["Q"]

        grad = jax.grad(heat_duty)(jnp.array(1000.0))
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # More UA = more heat transfer


class TestCoCurrentHX:
    """Tests for co-current heat exchanger."""

    def test_co_current_basic(self):
        """Test basic co-current heat exchanger."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        hx = CoCurrentHX(HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0))
        hot_out, cold_out, info = hx(hot_in, cold_in)

        # Heat should transfer
        assert float(hot_out["T"]) < 400.0
        assert float(cold_out["T"]) > 300.0

    def test_co_current_limit(self):
        """Test co-current approaches common temperature."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        # Very high UA
        hx = CoCurrentHX(HeatExchangerParams(UA=10000.0, Cp_hot=75.0, Cp_cold=75.0))
        hot_out, cold_out, info = hx(hot_in, cold_in)

        # With balanced flow, both approach (400+300)/2 = 350 K
        assert float(hot_out["T"]) == pytest.approx(350.0, rel=0.05)
        assert float(cold_out["T"]) == pytest.approx(350.0, rel=0.05)

    def test_co_vs_counter_current(self):
        """Test counter-current transfers more heat than co-current."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        UA = 1500.0
        hx_counter = CounterCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0))
        hx_co = CoCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0))

        _, _, info_counter = hx_counter(hot_in, cold_in)
        _, _, info_co = hx_co(hot_in, cold_in)

        # Counter-current is more effective
        assert float(info_counter["Q"]) >= float(info_co["Q"])
        assert float(info_counter["effectiveness"]) >= float(info_co["effectiveness"])


class TestDesignFunctions:
    """Tests for heat exchanger design functions."""

    def test_design_heat_exchanger(self):
        """Test LMTD-based design."""
        result = design_heat_exchanger(
            Q=jnp.array(50000.0),  # 50 kW
            T_hot_in=jnp.array(400.0),
            T_hot_out=jnp.array(350.0),
            T_cold_in=jnp.array(300.0),
            T_cold_out=jnp.array(330.0),
            U=jnp.array(500.0),  # W/m²·K
            flow_config="counter_current",
        )

        assert float(result["A"]) > 0
        assert float(result["LMTD"]) > 0
        assert float(result["UA"]) > 0

        # Verify Q = UA * LMTD
        Q_check = float(result["UA"]) * float(result["LMTD"])
        assert Q_check == pytest.approx(50000.0, rel=0.01)

    def test_size_heat_exchanger(self):
        """Test effectiveness-NTU based sizing."""
        result = size_heat_exchanger(
            Q=jnp.array(30000.0),  # 30 kW
            T_hot_in=jnp.array(400.0),
            T_cold_in=jnp.array(300.0),
            C_hot=jnp.array(750.0),  # W/K (10 mol/s * 75 J/mol·K)
            C_cold=jnp.array(750.0),
            U=jnp.array(500.0),
            flow_config="counter_current",
        )

        assert float(result["A"]) > 0
        assert float(result["NTU"]) > 0
        assert 0 < float(result["effectiveness"]) < 1

    def test_design_differentiable(self):
        """Test design functions are differentiable."""
        def area_fn(Q):
            result = design_heat_exchanger(
                Q=Q,
                T_hot_in=jnp.array(400.0),
                T_hot_out=jnp.array(350.0),
                T_cold_in=jnp.array(300.0),
                T_cold_out=jnp.array(330.0),
                U=jnp.array(500.0),
            )
            return result["A"]

        grad = jax.grad(area_fn)(jnp.array(50000.0))
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # More duty needs more area


class TestIntegration:
    """Integration tests for heat exchangers."""

    def test_heater_cooler_round_trip(self):
        """Test heating then cooling returns to original temperature."""
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)

        heater = Heater(HeaterParams(T_out=400.0, Cp=75.0))
        hot_stream, _ = heater(stream)

        cooler = Cooler(CoolerParams(T_out=300.0, Cp=75.0))
        final_stream, _ = cooler(hot_stream)

        assert float(final_stream["T"]) == pytest.approx(300.0, rel=0.01)

    def test_process_to_process_recovery(self):
        """Test heat recovery between process streams."""
        # Hot product stream (waste heat)
        hot_product = make_stream({"product": 5.0}, T=450.0, P=101325.0)
        # Cold feed stream (needs heating)
        cold_feed = make_stream({"feed": 5.0}, T=300.0, P=101325.0)

        # Heat exchanger for heat recovery
        hx = CounterCurrentHX(HeatExchangerParams(UA=500.0, Cp_hot=80.0, Cp_cold=80.0))
        hot_out, cold_out, info = hx(hot_product, cold_feed)

        # Energy should be conserved
        Q_hot = 5.0 * 80.0 * (450.0 - float(hot_out["T"]))
        Q_cold = 5.0 * 80.0 * (float(cold_out["T"]) - 300.0)
        assert Q_hot == pytest.approx(Q_cold, rel=0.01)

        # Preheat should save utility
        utility_without_recovery = 5.0 * 80.0 * (450.0 - 300.0)
        utility_with_recovery = 5.0 * 80.0 * (450.0 - float(cold_out["T"]))
        assert utility_with_recovery < utility_without_recovery
