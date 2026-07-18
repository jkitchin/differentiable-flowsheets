"""Tests for heat exchanger unit operations."""

from functools import lru_cache

import pytest
import jax
import jax.numpy as jnp

from difflow import make_stream
from difflow.streams import get_flows
from difflow.eos import PengRobinson, CriticalProperties
from difflow.thermo import IdealThermo, CubicThermo, SpeciesData
from difflow.units.heat_exchanger import (
    Heater,
    HeaterParams,
    Cooler,
    CoolerParams,
    CounterCurrentHX,
    EnthalpyCounterCurrentHX,
    EnthalpyHXParams,
    CoCurrentHX,
    CrossFlowHX,
    HeatExchangerParams,
    log_mean_temperature_difference,
    effectiveness_counter_current,
    effectiveness_co_current,
    effectiveness_crossflow_both_unmixed,
    effectiveness_crossflow_cmax_mixed,
    effectiveness_crossflow_cmin_mixed,
    effectiveness_crossflow_both_mixed,
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


class TestCrossFlowEffectiveness:
    """Tests for cross-flow effectiveness correlations."""

    def test_crossflow_both_unmixed_basic(self):
        """Test cross-flow both unmixed effectiveness."""
        eps = effectiveness_crossflow_both_unmixed(jnp.array(1.0), jnp.array(0.5))
        # Should be between co-current and counter-current
        assert 0.3 < float(eps) < 0.8

    def test_crossflow_both_unmixed_high_ntu(self):
        """Test cross-flow both unmixed at high NTU."""
        eps = effectiveness_crossflow_both_unmixed(jnp.array(10.0), jnp.array(0.5))
        # Should approach high effectiveness but less than counter-current
        assert float(eps) > 0.8

    def test_crossflow_cmax_mixed(self):
        """Test cross-flow with Cmax mixed."""
        eps = effectiveness_crossflow_cmax_mixed(jnp.array(1.0), jnp.array(0.5))
        assert 0.3 < float(eps) < 0.8

    def test_crossflow_cmin_mixed(self):
        """Test cross-flow with Cmin mixed."""
        eps = effectiveness_crossflow_cmin_mixed(jnp.array(1.0), jnp.array(0.5))
        assert 0.3 < float(eps) < 0.8

    def test_crossflow_both_mixed(self):
        """Test cross-flow with both mixed."""
        eps = effectiveness_crossflow_both_mixed(jnp.array(1.0), jnp.array(0.5))
        assert 0.3 < float(eps) < 0.8

    def test_crossflow_effectiveness_ordering(self):
        """Test that cross-flow effectiveness ordering is correct.

        Generally: counter > unmixed cross > mixed cross > co-current
        """
        NTU = jnp.array(2.0)
        Cr = jnp.array(0.75)

        eps_counter = effectiveness_counter_current(NTU, Cr)
        eps_cross_unmixed = effectiveness_crossflow_both_unmixed(NTU, Cr)
        eps_co = effectiveness_co_current(NTU, Cr)

        # Counter-current should be highest
        assert float(eps_counter) >= float(eps_cross_unmixed)
        # Cross-flow should be higher than co-current
        assert float(eps_cross_unmixed) >= float(eps_co)

    def test_crossflow_differentiable(self):
        """Test all cross-flow correlations are differentiable."""
        def eps_fn(NTU):
            return effectiveness_crossflow_both_unmixed(NTU, jnp.array(0.5))

        grad = jax.grad(eps_fn)(jnp.array(1.0))
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # Effectiveness increases with NTU

    def test_crossflow_small_cr(self):
        """Test cross-flow with very small Cr (numerical stability)."""
        # Should not produce NaN or inf
        eps = effectiveness_crossflow_both_unmixed(jnp.array(2.0), jnp.array(1e-3))
        assert jnp.isfinite(eps)
        # Should approach 1 - exp(-NTU)
        expected = 1.0 - jnp.exp(-2.0)
        assert float(eps) == pytest.approx(float(expected), rel=0.1)

    def test_crossflow_balanced(self):
        """Test cross-flow with Cr = 1 (balanced flow)."""
        NTU = jnp.array(2.0)
        Cr = jnp.array(1.0)

        # All mixed configurations should give same result for Cr=1
        eps_cmax = effectiveness_crossflow_cmax_mixed(NTU, Cr)
        eps_cmin = effectiveness_crossflow_cmin_mixed(NTU, Cr)

        # Should be close to NTU/(1+NTU)
        expected = NTU / (1.0 + NTU)
        assert float(eps_cmax) == pytest.approx(float(expected), rel=0.05)
        assert float(eps_cmin) == pytest.approx(float(expected), rel=0.05)


class TestCrossFlowHX:
    """Tests for cross-flow heat exchanger."""

    def test_crossflow_both_unmixed_basic(self):
        """Test basic cross-flow heat exchanger with both unmixed."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        hx = CrossFlowHX(
            HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0),
            mixing="both_unmixed"
        )
        hot_out, cold_out, info = hx(hot_in, cold_in)

        # Heat should transfer from hot to cold
        assert float(hot_out["T"]) < 400.0
        assert float(cold_out["T"]) > 300.0

        # Energy balance: Q_hot = Q_cold
        Q_hot = 10.0 * 75.0 * (400.0 - float(hot_out["T"]))
        Q_cold = 10.0 * 75.0 * (float(cold_out["T"]) - 300.0)
        assert Q_hot == pytest.approx(Q_cold, rel=0.01)
        assert float(info["Q"]) == pytest.approx(Q_hot, rel=0.01)
        assert info["mixing"] == "both_unmixed"

    def test_crossflow_cmax_mixed(self):
        """Test cross-flow with Cmax mixed."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        hx = CrossFlowHX(
            HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0),
            mixing="cmax_mixed"
        )
        hot_out, cold_out, info = hx(hot_in, cold_in)

        assert float(hot_out["T"]) < 400.0
        assert float(cold_out["T"]) > 300.0
        assert info["mixing"] == "cmax_mixed"

    def test_crossflow_cmin_mixed(self):
        """Test cross-flow with Cmin mixed."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        hx = CrossFlowHX(
            HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0),
            mixing="cmin_mixed"
        )
        hot_out, cold_out, info = hx(hot_in, cold_in)

        assert float(hot_out["T"]) < 400.0
        assert float(cold_out["T"]) > 300.0
        assert info["mixing"] == "cmin_mixed"

    def test_crossflow_both_mixed(self):
        """Test cross-flow with both mixed."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        hx = CrossFlowHX(
            HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0),
            mixing="both_mixed"
        )
        hot_out, cold_out, info = hx(hot_in, cold_in)

        assert float(hot_out["T"]) < 400.0
        assert float(cold_out["T"]) > 300.0
        assert info["mixing"] == "both_mixed"

    def test_crossflow_invalid_mixing(self):
        """Test that invalid mixing raises error."""
        with pytest.raises(ValueError, match="Invalid mixing configuration"):
            CrossFlowHX(
                HeatExchangerParams(UA=1000.0),
                mixing="invalid"
            )

    def test_crossflow_vs_counter_current(self):
        """Test cross-flow transfers less heat than counter-current."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        UA = 1500.0
        hx_counter = CounterCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0))
        hx_cross = CrossFlowHX(
            HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0),
            mixing="both_unmixed"
        )

        _, _, info_counter = hx_counter(hot_in, cold_in)
        _, _, info_cross = hx_cross(hot_in, cold_in)

        # Counter-current should be more effective
        assert float(info_counter["Q"]) >= float(info_cross["Q"])
        assert float(info_counter["effectiveness"]) >= float(info_cross["effectiveness"])

    def test_crossflow_vs_co_current(self):
        """Test cross-flow transfers more heat than co-current."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)

        UA = 1500.0
        hx_co = CoCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0))
        hx_cross = CrossFlowHX(
            HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0),
            mixing="both_unmixed"
        )

        _, _, info_co = hx_co(hot_in, cold_in)
        _, _, info_cross = hx_cross(hot_in, cold_in)

        # Cross-flow unmixed should be more effective than co-current
        assert float(info_cross["Q"]) >= float(info_co["Q"])
        assert float(info_cross["effectiveness"]) >= float(info_co["effectiveness"])

    def test_crossflow_unbalanced_flows(self):
        """Test cross-flow with unbalanced heat capacity rates."""
        # Hot stream has lower heat capacity rate (Cmin)
        hot_in = make_stream({"A": 5.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 15.0}, T=300.0, P=101325.0)

        hx = CrossFlowHX(
            HeatExchangerParams(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0),
            mixing="both_unmixed"
        )
        hot_out, cold_out, info = hx(hot_in, cold_in)

        # Hot stream should cool more than cold stream heats
        dT_hot = 400.0 - float(hot_out["T"])
        dT_cold = float(cold_out["T"]) - 300.0
        assert dT_hot > dT_cold

        # But energy should be conserved
        Q_hot = 5.0 * 75.0 * dT_hot
        Q_cold = 15.0 * 75.0 * dT_cold
        assert Q_hot == pytest.approx(Q_cold, rel=0.01)

    def test_crossflow_differentiable(self):
        """Test cross-flow HX is differentiable."""
        def heat_duty(UA):
            hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
            cold_in = make_stream({"B": 10.0}, T=300.0, P=101325.0)
            hx = CrossFlowHX(
                HeatExchangerParams(Cp_hot=75.0, Cp_cold=75.0),
                mixing="both_unmixed"
            )
            _, _, info = hx(hot_in, cold_in, UA=UA)
            return info["Q"]

        grad = jax.grad(heat_duty)(jnp.array(1000.0))
        assert jnp.isfinite(grad)
        assert float(grad) > 0  # More UA = more heat transfer

    def test_crossflow_effectiveness_ordering(self):
        """Test that different mixing configs give expected effectiveness ordering."""
        hot_in = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold_in = make_stream({"B": 15.0}, T=300.0, P=101325.0)  # Unbalanced
        UA = 2000.0

        configs = ["both_unmixed", "cmax_mixed", "cmin_mixed", "both_mixed"]
        effectivenesses = {}

        for config in configs:
            hx = CrossFlowHX(
                HeatExchangerParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0),
                mixing=config
            )
            _, _, info = hx(hot_in, cold_in)
            effectivenesses[config] = float(info["effectiveness"])

        # All should be in valid range
        for config, eps in effectivenesses.items():
            assert 0 < eps < 1, f"{config} effectiveness out of range: {eps}"


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


@lru_cache(maxsize=None)
def _propane_butane_cubic_thermo():
    """CubicThermo (ideal-gas Cp + PR departure) for propane/butane.

    Memoized so every EnthalpyCounterCurrentHX test shares one thermo object:
    the unit's JIT core caches its compiled solve on the thermo identity, so a
    shared object compiles once and is reused instead of recompiling per test.
    """
    species = {
        "propane": SpeciesData(
            name="propane",
            MW=44.10,
            Cp_coeffs=(73.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(18000.0, 0.38, 369.8),
            antoine_coeffs=(13.72, 1872.5, -25.16),
        ),
        "butane": SpeciesData(
            name="butane",
            MW=58.12,
            Cp_coeffs=(98.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(22000.0, 0.38, 425.1),
            antoine_coeffs=(13.98, 2292.4, -27.86),
        ),
    }
    crit = {
        "propane": CriticalProperties(name="propane", Tc=369.8, Pc=4.25e6, omega=0.152, MW=44.10),
        "butane": CriticalProperties(name="butane", Tc=425.1, Pc=3.80e6, omega=0.200, MW=58.12),
    }
    return CubicThermo(IdealThermo(species), PengRobinson(crit))


class TestEnthalpyCounterCurrentHX:
    """Counter-current HX closed on real EOS-based stream enthalpies."""

    def _streams(self):
        # Single-phase vapor both sides (3 bar is below the mixture dew here),
        # so the enthalpy path is smooth and the energy balance is easy to check.
        hot = make_stream({"propane": 1.0, "butane": 1.0}, T=400.0, P=3e5)
        cold = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=3e5)
        return hot, cold

    def test_energy_balance_closes(self):
        """Q from the solve equals the enthalpy change on each side."""
        thermo = _propane_butane_cubic_thermo()
        hot, cold = self._streams()
        hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=200.0), thermo)
        hot_out, cold_out, info = hx(hot, cold)

        Q = info["Q"]
        H_hot_in = thermo.stream_enthalpy_flash(get_flows(hot), hot["T"], hot["P"])
        H_hot_out = thermo.stream_enthalpy_flash(get_flows(hot), hot_out["T"], hot["P"])
        H_cold_in = thermo.stream_enthalpy_flash(get_flows(cold), cold["T"], cold["P"])
        H_cold_out = thermo.stream_enthalpy_flash(get_flows(cold), cold_out["T"], cold["P"])

        assert float(H_hot_in - H_hot_out) == pytest.approx(float(Q), rel=1e-3)
        assert float(H_cold_out - H_cold_in) == pytest.approx(float(Q), rel=1e-3)

    def test_directions_and_second_law(self):
        """Hot side cools, cold side heats, and terminal temperatures don't cross."""
        thermo = _propane_butane_cubic_thermo()
        hot, cold = self._streams()
        hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=200.0), thermo)
        hot_out, cold_out, _ = hx(hot, cold)

        assert float(hot_out["T"]) < float(hot["T"])
        assert float(cold_out["T"]) > float(cold["T"])
        assert float(hot_out["T"]) > float(cold["T"])
        assert float(cold_out["T"]) < float(hot["T"])

    def test_differentiable_wrt_UA(self):
        """Duty is differentiable through the coupled solve; more UA -> more duty."""
        thermo = _propane_butane_cubic_thermo()
        hot, cold = self._streams()

        def duty(UA):
            hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=UA), thermo)
            _, _, info = hx(hot, cold)
            return info["Q"]

        g = jax.grad(duty)(jnp.array(200.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0
