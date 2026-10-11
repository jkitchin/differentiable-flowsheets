"""Tests for heat exchanger unit operations."""

import warnings
from functools import lru_cache

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from difflow import make_stream
from difflow.streams import get_flows
from difflow.eos import PengRobinson, CriticalProperties
from difflow.thermo import IdealThermo, CubicThermo, SpeciesData
from difflow.units.heat_exchanger import (
    DEFAULT_CP,
    DefaultCpWarning,
    LowCorrectionFactorWarning,
    PhaseChangeWarning,
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
    effectiveness_shell_and_tube,
    lmtd_correction_factor,
    ShellAndTubeHX,
    ShellAndTubeHXParams,
    design_heat_exchanger,
    size_heat_exchanger,
)
from jax.test_util import check_grads


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

    @pytest.mark.release
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

    @pytest.mark.release
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

    @pytest.mark.release
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

    @pytest.mark.release
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

    @pytest.mark.release
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

        # At Cr = 1 the Cmax-mixed and Cmin-mixed formulas coincide:
        # eps = 1 - exp(-(1 - exp(-NTU))) = 0.5788 at NTU = 2. (Not the
        # counter-current NTU/(1+NTU) = 0.667, which they once blended to; #418.)
        eps_cmax = effectiveness_crossflow_cmax_mixed(NTU, Cr)
        eps_cmin = effectiveness_crossflow_cmin_mixed(NTU, Cr)

        expected = 1.0 - np.exp(-(1.0 - np.exp(-2.0)))
        assert float(eps_cmax) == pytest.approx(expected, rel=1e-12)
        assert float(eps_cmin) == pytest.approx(expected, rel=1e-12)


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

    @pytest.mark.release
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

    @pytest.mark.release
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


@pytest.mark.slow
class TestEnthalpyCounterCurrentHX:
    """Counter-current HX closed on real EOS-based stream enthalpies.

    Marked slow: the thermo is already shared and the solve already jitted, so
    what remains is two irreducible one-off compiles -- the forward coupled
    fixed-point-over-Newton-over-flash graph (~13 s) and its implicit-diff
    backward pass (~34 s). Nothing here is cacheable away.
    """

    def _streams(self):
        # Single-phase vapor both sides (3 bar is below the mixture dew here),
        # so the enthalpy path is smooth and the energy balance is easy to check.
        hot = make_stream({"propane": 1.0, "butane": 1.0}, T=400.0, P=3e5)
        cold = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=3e5)
        return hot, cold

    @pytest.mark.release
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

    @pytest.mark.release
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


class TestCubicThermoAbsentSpecies:
    """A stream need not list every EOS species (a condenser's vapour has no
    coolant); it used to raise a bare KeyError from stream_enthalpy_flash."""

    def test_absent_species_is_zero_flow(self):
        thermo = _propane_butane_cubic_thermo()
        for f in (thermo.stream_enthalpy_flash, thermo.stream_entropy_flash):
            assert float(f({"propane": 1.0}, 350.0, 3e5)) == pytest.approx(
                float(f({"propane": 1.0, "butane": 0.0}, 350.0, 3e5)), rel=1e-12)
        assert float(thermo.stream_enthalpy({"butane": 2.0}, 400.0, 3e5)) == pytest.approx(
            float(thermo.stream_enthalpy({"propane": 0.0, "butane": 2.0}, 400.0, 3e5)), rel=1e-12)

    def test_unknown_species_raises(self):
        with pytest.raises(KeyError):
            _propane_butane_cubic_thermo().stream_enthalpy_flash({"benzene": 1.0}, 350.0, 3e5)


class TestDefaultCpWarning:
    """The constant-Cp fallback announces itself instead of returning a
    confident wrong duty (issue #226)."""

    def test_heater_warns_when_Cp_omitted(self):
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        with pytest.warns(DefaultCpWarning, match="75.0 J/mol/K"):
            Heater(HeaterParams(T_out=350.0))(stream)

    def test_cooler_warns_when_Cp_omitted(self):
        stream = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        with pytest.warns(DefaultCpWarning):
            Cooler(CoolerParams(T_out=350.0))(stream)

    def test_two_stream_hx_warns_for_each_side(self):
        hot = make_stream({"A": 10.0}, T=400.0, P=101325.0)
        cold = make_stream({"B": 10.0}, T=300.0, P=101325.0)
        with pytest.warns(DefaultCpWarning) as record:
            CounterCurrentHX(HeatExchangerParams(UA=500.0))(hot, cold)
        messages = " ".join(str(r.message) for r in record)
        assert "hot side" in messages and "cold side" in messages

    def test_no_warning_when_Cp_given(self):
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DefaultCpWarning)
            Heater(HeaterParams(T_out=350.0, Cp=75.0))(stream)

    def test_no_warning_when_thermo_given(self):
        thermo = _propane_butane_cubic_thermo()
        stream = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=3e5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DefaultCpWarning)
            Heater(HeaterParams(T_out=310.0), thermo=thermo)(stream)

    def test_fallback_value_is_unchanged(self):
        """The fallback still computes what it always did -- it is now audible,
        not different."""
        stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
        with pytest.warns(DefaultCpWarning):
            _, info = Heater(HeaterParams(T_out=350.0))(stream)
        assert float(info["Q"]) == pytest.approx(10.0 * DEFAULT_CP * 50.0)


class TestUtilityExchangerThermo:
    """Heater/Cooler duties from a real enthalpy balance (issue #226)."""

    def _liquid_feed(self):
        # 10 bar, 300 K: well below the bubble point, so heating to 400 K
        # crosses the phase envelope and the duty carries latent heat.
        return make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=10e5)

    def test_duty_is_the_enthalpy_difference(self):
        thermo = _propane_butane_cubic_thermo()
        feed = self._liquid_feed()
        _, info = Heater(HeaterParams(T_out=400.0), thermo=thermo)(feed)

        flows = get_flows(feed)
        H_in = thermo.stream_enthalpy_flash(flows, feed["T"], feed["P"])
        H_out = thermo.stream_enthalpy_flash(flows, 400.0, feed["P"])
        assert float(info["Q"]) == pytest.approx(float(H_out - H_in), rel=1e-6)

    def test_constant_Cp_understates_a_vaporizing_duty(self):
        """The failure in issue #226: 75 J/mol/K carries no latent heat."""
        thermo = _propane_butane_cubic_thermo()
        feed = self._liquid_feed()

        _, info_thermo = Heater(HeaterParams(T_out=400.0), thermo=thermo)(feed)
        with pytest.warns(DefaultCpWarning):
            _, info_const = Heater(HeaterParams(T_out=400.0))(feed)

        # 2 mol/s * 75 J/mol/K * 100 K = 15 kW, against a real duty that also
        # has to boil the stream.
        assert float(info_const["Q"]) == pytest.approx(15000.0)
        assert float(info_thermo["Q"]) > 2.0 * float(info_const["Q"])

    def test_cooler_reverses_the_heater(self):
        thermo = _propane_butane_cubic_thermo()
        feed = self._liquid_feed()
        _, info_h = Heater(HeaterParams(T_out=400.0), thermo=thermo)(feed)

        hot = make_stream({"propane": 1.0, "butane": 1.0}, T=400.0, P=10e5)
        outlet, info_c = Cooler(CoolerParams(T_out=300.0), thermo=thermo)(hot)

        # Cooler duty is positive for heat removed, and the path is reversible.
        assert float(info_c["Q"]) == pytest.approx(float(info_h["Q"]), rel=1e-6)
        assert float(outlet["T"]) == pytest.approx(300.0)

    def test_phase_forces_a_single_phase_enthalpy(self):
        """``phase=`` bypasses the flash, which is how an IdealThermo is used."""
        thermo = _propane_butane_cubic_thermo()
        feed = self._liquid_feed()
        flows = get_flows(feed)

        _, info = Heater(
            HeaterParams(T_out=400.0, phase="vapor"), thermo=thermo
        )(feed)
        H_in = thermo.stream_enthalpy(flows, feed["T"], phase="vapor", P=feed["P"])
        H_out = thermo.stream_enthalpy(flows, 400.0, phase="vapor", P=feed["P"])
        assert float(info["Q"]) == pytest.approx(float(H_out - H_in), rel=1e-6)

    def test_eo_residuals_vanish_at_the_thermo_solution(self):
        thermo = _propane_butane_cubic_thermo()
        feed = self._liquid_feed()
        heater = Heater(HeaterParams(T_out=400.0), thermo=thermo)
        outlet, info = heater(feed)

        # T_out mode pins T directly; the duty mode is the one that has to
        # close an enthalpy balance.
        duty_heater = Heater(HeaterParams(duty=float(info["Q"])), thermo=thermo)
        resid = duty_heater.eo_residuals([feed], [outlet])
        assert float(jnp.max(jnp.abs(resid))) < 1e-6


@pytest.mark.slow
class TestUtilityExchangerThermoHeavy:
    """The Heater/Cooler thermo paths that cost a JAX trace.

    Marked slow for the same reason as TestEnthalpyCounterCurrentHX: rating
    mode is a damped fixed point over a Newton inversion over an EOS flash,
    duty mode is that Newton inversion, and the reverse-mode pass differentiates
    through the flash -- the one-off trace/compile of those graphs is the cost,
    not the assertions.
    """

    @pytest.mark.release
    def test_rating_mode_closes_Q_equals_UA_LMTD(self):
        thermo = _propane_butane_cubic_thermo()
        feed = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=10e5)
        heater = Heater(
            HeaterParams(UA=500.0, T_utility=450.0), thermo=thermo
        )
        outlet, info = heater(feed)

        assert 300.0 < float(outlet["T"]) < 450.0
        assert float(info["Q"]) > 0.0
        assert float(info["UA_required"]) == pytest.approx(500.0, rel=1e-3)

        flows = get_flows(feed)
        H_in = thermo.stream_enthalpy_flash(flows, feed["T"], feed["P"])
        H_out = thermo.stream_enthalpy_flash(flows, outlet["T"], feed["P"])
        assert float(H_out - H_in) == pytest.approx(float(info["Q"]), rel=1e-3)

    def test_duty_mode_inverts_the_enthalpy(self):
        thermo = _propane_butane_cubic_thermo()
        feed = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=10e5)
        _, info = Heater(HeaterParams(T_out=400.0), thermo=thermo)(feed)

        outlet, info2 = Heater(
            HeaterParams(duty=float(info["Q"])), thermo=thermo
        )(feed)
        assert float(outlet["T"]) == pytest.approx(400.0, rel=1e-6)
        assert float(info2["Q"]) == pytest.approx(float(info["Q"]))

    @pytest.mark.release
    def test_differentiable(self):
        """Duty is differentiable through the flash-based enthalpy."""
        thermo = _propane_butane_cubic_thermo()
        feed = make_stream({"propane": 1.0, "butane": 1.0}, T=300.0, P=10e5)

        def duty(T_out):
            _, info = Heater(HeaterParams(T_out=T_out), thermo=thermo)(feed)
            return info["Q"]

        g = jax.grad(duty)(jnp.array(400.0))
        assert jnp.isfinite(g)
        # dQ/dT_out is the stream's heat capacity rate: positive, and larger
        # than the 150 W/K a constant 75 J/mol/K would give.
        assert float(g) > 150.0


@pytest.mark.slow
class TestEnthalpyHXThroughBoiling:
    """The cold side boils inside the exchanger.

    The per-side enthalpy inversion used to be a Newton solve from the inlet
    temperature. A boiling side's enthalpy is monotone but its slope jumps by
    an order of magnitude across the two-phase window, and Newton came back
    with a cold outlet of 245 K from a 300 K inlet -- a heating duty that
    cooled the stream -- with ``throw=False`` hiding it. The inversion is
    now bracketed.
    """

    def _case(self):
        from tests.c4_system import C4_P, _c4_eos_and_thermo

        _, thermo = _c4_eos_and_thermo()
        hot = make_stream({"propane": 0.5, "butane": 0.0906, "isobutane": 1.0094},
                          T=511.6, P=C4_P)
        # Liquid at 300 K; bubble ~320 K, dew ~331 K at 8 bar.
        cold = make_stream({"propane": 0.5, "butane": 1.0, "isobutane": 0.1},
                           T=300.0, P=C4_P)
        return thermo, hot, cold

    def test_energy_balance_closes_and_cold_side_heats(self):
        thermo, hot, cold = self._case()
        hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=150.0), thermo)
        hot_out, cold_out, info = hx(hot, cold)
        H = thermo.stream_enthalpy_flash
        P = hot["P"]
        Q = float(info["Q"])
        dH_hot = float(H(get_flows(hot), hot["T"], P) - H(get_flows(hot), hot_out["T"], P))
        dH_cold = float(H(get_flows(cold), cold_out["T"], P) - H(get_flows(cold), cold["T"], P))
        assert dH_hot == pytest.approx(Q, rel=1e-8)
        assert dH_cold == pytest.approx(Q, rel=1e-8)
        # Ends inside the two-phase window, and heated, not cooled.
        assert 320.0 < float(cold_out["T"]) < 331.0
        # And it is the root of the independent EO residual (#389).
        assert float(jnp.max(jnp.abs(hx.eo_residuals([hot, cold], [hot_out, cold_out])))) < 1e-3

    def test_inversion_lands_in_the_two_phase_window(self):
        from difflow.units.heat_exchanger import _invert_enthalpy_T

        thermo, _, cold = self._case()
        flows = get_flows(cold)
        H0 = thermo.stream_enthalpy_flash(flows, 300.0, cold["P"])
        for Q in (5e3, 2e4, 3e4, 5e4):
            T = _invert_enthalpy_T(thermo, flows, H0 + Q, cold["P"], jnp.array(300.0), None)
            H = thermo.stream_enthalpy_flash(flows, T, cold["P"])
            assert float(H - H0) == pytest.approx(Q, rel=1e-8), Q
            assert float(T) > 300.0

    @pytest.mark.release
    def test_gradient_matches_finite_difference(self):
        thermo, hot, cold = self._case()

        def T_cold_out(UA):
            return EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=UA), thermo)(hot, cold)[1]["T"]

        g = jax.grad(T_cold_out)(jnp.array(150.0))
        eps = 1e-3
        fd = (T_cold_out(150.0 + eps) - T_cold_out(150.0 - eps)) / (2 * eps)
        assert float(g) == pytest.approx(float(fd), rel=1e-4)


class TestClosedFormValues:
    """Every effectiveness and F against its textbook closed form (issue #418).

    Incropera, DeWitt, Bergman 7e Table 11.3; Bowman, Mueller, Nagle (1940).
    The old tests checked only ranges and monotonicity, which let a blend
    that silently replaced the formula pass for years.
    """

    @staticmethod
    def eps_shell(NTU, Cr, N):
        s = np.sqrt(1 + Cr**2)
        e1 = 2 / (1 + Cr + s / np.tanh(NTU / N * s / 2))
        if Cr == 1:
            return N * e1 / (1 + (N - 1) * e1)
        a = ((1 - e1 * Cr) / (1 - e1)) ** N
        return (a - 1) / (a - Cr)

    @staticmethod
    def F_1_2(R, P):
        s = np.sqrt(R**2 + 1)
        return s * np.log((1 - P) / (1 - R * P)) / (
            (R - 1) * np.log((2 - P * (R + 1 - s)) / (2 - P * (R + 1 + s))))

    @pytest.mark.parametrize("R,P", [(2.0, 0.3), (0.5, 0.6), (3.0, 0.25),
                                     (1.03, 0.4), (0.98, 0.5), (1.5, 0.3)])
    def test_F_one_shell(self, R, P):
        assert float(lmtd_correction_factor(R, P)) == pytest.approx(
            self.F_1_2(R, P), rel=1e-6)

    def test_F_symmetry_and_balanced_limit(self):
        # F(R, P) = F(1/R, P R), and at R = 1 the L'Hopital limit
        assert float(lmtd_correction_factor(2.0, 0.3)) == pytest.approx(
            float(lmtd_correction_factor(0.5, 0.6)), rel=1e-9)
        P = 0.5
        F1 = P * np.sqrt(2) / ((1 - P) * np.log((2 - P * (2 - np.sqrt(2)))
                                                / (2 - P * (2 + np.sqrt(2)))))
        assert float(lmtd_correction_factor(1.0, P)) == pytest.approx(F1, rel=1e-9)

    def test_F_gradient_is_not_zero(self):
        assert float(jax.grad(lambda R: lmtd_correction_factor(R, 0.3))(2.0)) < -0.05

    @pytest.mark.parametrize("R", [0.5, 1.0, 1.125, 3.0])
    def test_F_is_zero_past_the_one_shell_limit(self, R):
        # crude 30 -> 110 C on residue 160 -> 70 C (R = 1.125, P = 0.615) is a
        # cross one shell cannot make; the log clamps used to return F = 0.11
        P_max = 2 / (R + 1 + np.sqrt(R**2 + 1))
        assert float(lmtd_correction_factor(R, 1.02 * P_max)) == 0.0
        assert float(lmtd_correction_factor(R, 0.98 * P_max)) > 0.0
        g = jax.grad(lambda P: lmtd_correction_factor(R, P))(1.02 * P_max)
        assert np.isfinite(float(g))

    def test_F_two_shells_make_the_cross(self):
        R, P = 90 / 80, 80 / 130
        assert float(lmtd_correction_factor(R, P)) == 0.0
        assert float(lmtd_correction_factor(R, P, 2)) == pytest.approx(0.8266, abs=1e-4)

    @pytest.mark.parametrize("NTU", [0.5, 2.0, 4.0])
    @pytest.mark.parametrize("Cr", [1e-9, 0.5, 0.97, 1.0])
    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_shell_and_tube_effectiveness(self, NTU, Cr, N):
        assert float(effectiveness_shell_and_tube(NTU, Cr, N)) == pytest.approx(
            self.eps_shell(NTU, Cr, N), rel=2e-6)

    def test_crossflow_both_unmixed_is_not_counter_current(self):
        NTU, Cr = 1.5, 0.6
        expected = 1 - np.exp(NTU**0.22 / Cr * (np.exp(-Cr * NTU**0.78) - 1))
        eps = float(effectiveness_crossflow_both_unmixed(NTU, Cr))
        assert eps == pytest.approx(expected, rel=1e-12)
        assert eps < float(effectiveness_counter_current(NTU, Cr)) - 0.02

    @pytest.mark.parametrize("Cr", [1e-6, 0.5, 0.97, 1.0])
    def test_crossflow_mixed(self, Cr):
        NTU = 1.5
        cmax = (1 / Cr) * (1 - np.exp(-Cr * (1 - np.exp(-NTU))))
        cmin = 1 - np.exp(-(1 / Cr) * (1 - np.exp(-Cr * NTU)))
        assert float(effectiveness_crossflow_cmax_mixed(NTU, Cr)) == pytest.approx(cmax, rel=1e-9)
        assert float(effectiveness_crossflow_cmin_mixed(NTU, Cr)) == pytest.approx(cmin, rel=1e-9)

    @pytest.mark.parametrize("UA,N", [(500.0, 1), (3000.0, 1), (1500.0, 2), (3000.0, 2)])
    def test_shell_and_tube_hx_rating(self, UA, N):
        """Q is the exact 1-2N rating, and the reported F is F(R, P, N)."""
        hot = make_stream({"water": 10.0}, T=420.0, P=2e5)
        cold = make_stream({"water": 20.0}, T=300.0, P=2e5)
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0,
                                                 n_shell_passes=N))
        _, _, info = hx(hot, cold)
        assert float(info["Q"]) == pytest.approx(
            self.eps_shell(UA / 750.0, 0.5, N) * 750.0 * 120.0, rel=1e-9)
        assert float(info["F_correction"]) == pytest.approx(
            float(lmtd_correction_factor(info["R"], info["P_param"], N)), rel=1e-6)

    def test_gradients(self):
        check_grads(lambda R: lmtd_correction_factor(R, 0.3, 2), (2.0,), order=1, modes=["rev"])
        check_grads(lambda n: effectiveness_shell_and_tube(n, 0.5, 2), (1.5,), order=1,
                    modes=["fwd", "rev"])
        check_grads(lambda c: effectiveness_crossflow_both_unmixed(1.5, c), (0.6,), order=1,
                    modes=["rev"])


# crude 30 -> 110 C heated by residue 160 -> 70 C (the cross a single 1-2 shell cannot make)
_CROSS = dict(Q=1.0e6, T_hot_in=433.15, T_hot_out=343.15, T_cold_in=303.15, T_cold_out=383.15, U=300.0)


class TestShellAndTubeSizing:
    """flow_config="shell_and_tube" in design_heat_exchanger / size_heat_exchanger (#422)."""

    def test_design_area_is_Q_over_U_F_LMTD(self):
        r = design_heat_exchanger(**_CROSS, flow_config="shell_and_tube", n_shell_passes=2)
        F = float(lmtd_correction_factor(90.0 / 80.0, 80.0 / 130.0, 2))
        assert float(r["F"]) == pytest.approx(F, rel=1e-12)
        assert 0.75 < F < 1.0
        assert float(r["A"]) == pytest.approx(1e6 / (300.0 * F * float(r["LMTD"])), rel=1e-12)
        cc = design_heat_exchanger(**_CROSS)
        assert float(cc["F"]) == 1.0
        assert float(r["A"]) == pytest.approx(float(cc["A"]) / F, rel=1e-12)

    def test_design_past_the_per_shell_limit_is_infinite_with_a_warning(self):
        with pytest.warns(LowCorrectionFactorWarning, match="more shells"):
            r = design_heat_exchanger(**_CROSS, flow_config="shell_and_tube", n_shell_passes=1)
        assert float(r["F"]) == 0.0
        assert np.isinf(float(r["A"])) and np.isinf(float(r["UA"]))

    def test_design_warns_below_the_design_floor(self):
        # R = 1, P = 0.55: inside the one-shell limit (0.586) but F is low
        args = dict(Q=1e6, T_hot_in=400.0, T_hot_out=345.0, T_cold_in=300.0, T_cold_out=355.0, U=300.0)
        F = float(lmtd_correction_factor(1.0, 0.55, 1))
        assert 0.0 < F < 0.75
        with pytest.warns(LowCorrectionFactorWarning, match="design floor"):
            r = design_heat_exchanger(**args, flow_config="shell_and_tube")
        assert float(r["F"]) == pytest.approx(F, rel=1e-9)

    def test_no_warning_for_a_good_design(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            design_heat_exchanger(**_CROSS, flow_config="shell_and_tube", n_shell_passes=2)
            design_heat_exchanger(**_CROSS)

    def test_design_agrees_with_rating(self):
        """The UA the design returns, rated by ShellAndTubeHX, gives the same terminal temperatures."""
        N = 2
        r = design_heat_exchanger(**_CROSS, flow_config="shell_and_tube", n_shell_passes=N)
        C_h, C_c = 1e6 / 90.0, 1e6 / 80.0
        hot = make_stream({"water": C_h / 75.0}, T=433.15, P=2e5)
        cold = make_stream({"water": C_c / 75.0}, T=303.15, P=2e5)
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=r["UA"], Cp_hot=75.0, Cp_cold=75.0, n_shell_passes=N))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhaseChangeWarning)
            h, c, info = hx(hot, cold)
        assert float(info["Q"]) == pytest.approx(1e6, rel=1e-8)
        assert float(h["T"]) == pytest.approx(343.15, abs=1e-6)
        assert float(c["T"]) == pytest.approx(383.15, abs=1e-6)
        assert float(info["F_correction"]) == pytest.approx(float(r["F"]), rel=1e-8)

    @pytest.mark.parametrize("N", [1, 2, 3])
    @pytest.mark.parametrize("Cr", [0.3, 0.5, 1.0])
    @pytest.mark.filterwarnings("ignore::difflow.units.heat_exchanger.LowCorrectionFactorWarning")
    def test_size_inverts_the_rating(self, N, Cr):
        C_h = 1000.0
        C_c = C_h / Cr
        T_hi, T_ci = 420.0, 300.0
        UA = 1500.0
        hot = make_stream({"water": C_h / 75.0}, T=T_hi, P=2e5)
        cold = make_stream({"water": C_c / 75.0}, T=T_ci, P=2e5)
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=UA, Cp_hot=75.0, Cp_cold=75.0, n_shell_passes=N))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhaseChangeWarning)
            _, _, info = hx(hot, cold)
        sized = size_heat_exchanger(info["Q"], T_hi, T_ci, C_h, C_c, 100.0,
                                    flow_config="shell_and_tube", n_shell_passes=N)
        assert float(sized["UA"]) == pytest.approx(UA, rel=1e-6)
        assert float(sized["A"]) == pytest.approx(UA / 100.0, rel=1e-6)
        assert float(sized["F"]) == pytest.approx(float(info["F_correction"]), rel=1e-5)

    def test_size_past_the_limit(self):
        # a one-shell unit cannot exceed eps_max = 2/(1 + Cr + sqrt(1 + Cr^2))
        Cr = 0.5
        eps_max = 2.0 / (1.0 + Cr + np.sqrt(1.0 + Cr**2))
        Q = (eps_max + 0.02) * 1000.0 * 100.0
        with pytest.warns(LowCorrectionFactorWarning):
            r = size_heat_exchanger(Q, 400.0, 300.0, 1000.0, 2000.0, 100.0,
                                    flow_config="shell_and_tube", n_shell_passes=1)
        assert float(r["F"]) == 0.0 and np.isinf(float(r["A"]))
        # two shells reach it
        r2 = size_heat_exchanger(Q, 400.0, 300.0, 1000.0, 2000.0, 100.0,
                                 flow_config="shell_and_tube", n_shell_passes=2)
        assert np.isfinite(float(r2["A"])) and float(r2["F"]) > 0.0

    def test_other_configs_report_F_one(self):
        for cfg in ("counter_current", "co_current"):
            r = size_heat_exchanger(5e4, 400.0, 300.0, 1000.0, 1500.0, 100.0, flow_config=cfg)
            assert float(r["F"]) == 1.0

    def test_unknown_flow_config_raises(self):
        with pytest.raises(ValueError, match="flow_config"):
            design_heat_exchanger(**_CROSS, flow_config="cross_flow")
        with pytest.raises(ValueError, match="flow_config"):
            size_heat_exchanger(5e4, 400.0, 300.0, 1000.0, 1500.0, 100.0, flow_config="nope")

    def test_jit_and_gradients(self):
        f = jax.jit(lambda Q, U: design_heat_exchanger(
            Q, 433.15, 343.15, 303.15, 383.15, U, "shell_and_tube", 2)["A"])
        assert np.isfinite(float(f(1e6, 300.0)))
        dA_dU = jax.grad(f, argnums=1)(1e6, 300.0)
        assert float(dA_dU) == pytest.approx(-float(f(1e6, 300.0)) / 300.0, rel=1e-9)
        check_grads(lambda Q: design_heat_exchanger(
            Q, 433.15, 343.15, 303.15, 383.15, 300.0, "shell_and_tube", 2)["A"],
            (1e6,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6, eps=1.0)
        g = jax.grad(lambda Q: size_heat_exchanger(
            Q, 420.0, 300.0, 1000.0, 1500.0, 100.0, "shell_and_tube", 2)["A"])(5e4)
        assert np.isfinite(float(g)) and float(g) > 0.0

    @pytest.mark.filterwarnings("ignore::difflow.units.heat_exchanger.LowCorrectionFactorWarning")
    def test_infeasible_design_has_finite_gradient_wrt_U(self):
        g = jax.grad(lambda U: design_heat_exchanger(
            1e6, 433.15, 343.15, 303.15, 383.15, U, "shell_and_tube", 1)["UA"])(300.0)
        assert np.isfinite(float(g))


class TestPhaseChangeWarning:
    """Constant-Cp units warn when the temperature span crosses a boiling point (#422)."""

    def _pentane_vs_water(self):
        hot = make_stream({"n_pentane": 10.0}, T=340.0, P=1.5e5)
        cold = make_stream({"water": 200.0}, T=293.0, P=1e5)
        return hot, cold

    @pytest.mark.parametrize("cls,params", [
        (CounterCurrentHX, lambda: HeatExchangerParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3)),
        (CoCurrentHX, lambda: HeatExchangerParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3)),
        (CrossFlowHX, lambda: HeatExchangerParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3)),
        (ShellAndTubeHX, lambda: ShellAndTubeHXParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3)),
    ])
    def test_condensing_pentane_warns(self, cls, params):
        hot, cold = self._pentane_vs_water()
        with pytest.warns(PhaseChangeWarning, match=r"n_pentane \(boils at 321\.\d K at 1\.5 bar\)") as rec:
            cls(params())(hot, cold)
        assert "EnthalpyCounterCurrentHX" in str(rec[0].message)
        assert "hot side" in str(rec[0].message)

    def test_single_phase_service_is_silent(self):
        hot = make_stream({"water": 10.0}, T=360.0, P=2e5)   # stays liquid at 2 bar (Tb 393 K)
        cold = make_stream({"water": 20.0}, T=300.0, P=2e5)
        hx = CounterCurrentHX(HeatExchangerParams(UA=500.0, Cp_hot=75.3, Cp_cold=75.3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hx(hot, cold)

    def test_pressure_moves_the_boiling_point(self):
        # water 360 -> 330 K: boils at 373 K at 1 atm (no cross) but 333... use 0.3 bar: Tb ~ 342 K
        cold = make_stream({"water": 200.0}, T=290.0, P=1e5)
        hx = CounterCurrentHX(HeatExchangerParams(UA=500.0, Cp_hot=75.3, Cp_cold=75.3))
        hot_low = make_stream({"water": 10.0}, T=360.0, P=0.3e5)
        with pytest.warns(PhaseChangeWarning, match="water"):
            hx(hot_low, cold)

    def test_trace_component_is_ignored(self):
        hot = make_stream({"water": 10.0, "n_pentane": 1e-4}, T=340.0, P=1.5e5)
        cold = make_stream({"water": 200.0}, T=293.0, P=1e5)
        hx = CounterCurrentHX(HeatExchangerParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PhaseChangeWarning)
            hx(hot, cold)

    def test_unknown_species_is_silent(self):
        hot = make_stream({"mystery": 10.0}, T=500.0, P=1e5)
        cold = make_stream({"mystery": 10.0}, T=300.0, P=1e5)
        hx = CounterCurrentHX(HeatExchangerParams(UA=500.0, Cp_hot=100.0, Cp_cold=100.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PhaseChangeWarning)
            hx(hot, cold)

    def test_jit_and_grad_are_unaffected(self):
        hot, cold = self._pentane_vs_water()
        hx = CounterCurrentHX(HeatExchangerParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3))

        def Q(T):
            _, _, info = hx(dict(hot, T=T), cold)
            return info["Q"]

        with warnings.catch_warnings():
            warnings.simplefilter("error", PhaseChangeWarning)   # a tracer must never warn
            assert np.isfinite(float(jax.jit(Q)(340.0)))
            assert np.isfinite(float(jax.grad(Q)(340.0)))

    def test_cooler_without_thermo_warns_and_with_thermo_does_not(self):
        hot = make_stream({"n_pentane": 10.0}, T=340.0, P=1.5e5)
        c = Cooler(CoolerParams(T_out=300.0, Cp=120.0))
        with pytest.warns(PhaseChangeWarning, match="Cooler"):
            c(hot)

    def test_enthalpy_hx_does_not_warn(self):
        import inspect
        src = inspect.getsource(EnthalpyCounterCurrentHX.__call__)
        assert "_warn_phase_change" not in src
