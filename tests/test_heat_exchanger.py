"""Smoke tests for heat exchanger unit operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import make_stream
from difflow.units.heat_exchanger import (
    CoCurrentHX,
    Cooler,
    CoolerParams,
    CounterCurrentHX,
    CrossFlowHX,
    DefaultCpWarning,
    HeaterParams,
    Heater,
    HeatExchangerParams,
    LowCorrectionFactorWarning,
    PhaseChangeWarning,
    ShellAndTubeHX,
    ShellAndTubeHXParams,
    design_heat_exchanger,
    effectiveness_co_current,
    effectiveness_counter_current,
    effectiveness_crossflow_both_mixed,
    effectiveness_crossflow_both_unmixed,
    effectiveness_crossflow_cmax_mixed,
    effectiveness_crossflow_cmin_mixed,
    effectiveness_shell_and_tube,
    lmtd_correction_factor,
    log_mean_temperature_difference,
    size_heat_exchanger,
)

_PARAMS = dict(UA=1000.0, Cp_hot=75.0, Cp_cold=75.0)
_CROSS = dict(Q=1.0e6, T_hot_in=433.15, T_hot_out=343.15, T_cold_in=303.15, T_cold_out=383.15, U=300.0)


def _streams():
    return (make_stream({"A": 10.0}, T=400.0, P=101325.0),
            make_stream({"B": 10.0}, T=300.0, P=101325.0))


def test_correlations_are_finite():
    NTU, Cr = jnp.array(1.0), jnp.array(0.5)
    for f in (effectiveness_counter_current, effectiveness_co_current,
              effectiveness_crossflow_both_unmixed, effectiveness_crossflow_cmax_mixed,
              effectiveness_crossflow_cmin_mixed, effectiveness_crossflow_both_mixed,
              effectiveness_shell_and_tube):
        assert np.isfinite(float(f(NTU, Cr)))
    assert np.isfinite(float(log_mean_temperature_difference(50.0, 30.0)))
    assert np.isfinite(float(lmtd_correction_factor(2.0, 0.3)))


@pytest.mark.parametrize("params", [
    HeaterParams(duty=7500.0, Cp=75.0),
    HeaterParams(T_out=350.0, Cp=75.0),
    HeaterParams(UA=500.0, T_utility=400.0, Cp=75.0),
], ids=["duty", "T_out", "UA"])
def test_heater_modes(params):
    outlet, info = Heater(params)(make_stream({"A": 10.0}, T=300.0, P=101325.0))
    assert np.isfinite(float(outlet["T"])) and np.isfinite(float(info["Q"]))


def test_cooler_smoke():
    outlet, info = Cooler(CoolerParams(T_out=350.0, Cp=75.0))(
        make_stream({"A": 10.0}, T=400.0, P=101325.0))
    assert np.isfinite(float(outlet["T"])) and np.isfinite(float(info["Q"]))


@pytest.mark.parametrize("make", [
    lambda: CounterCurrentHX(HeatExchangerParams(**_PARAMS)),
    lambda: CoCurrentHX(HeatExchangerParams(**_PARAMS)),
    lambda: CrossFlowHX(HeatExchangerParams(**_PARAMS), mixing="both_unmixed"),
    lambda: ShellAndTubeHX(ShellAndTubeHXParams(**_PARAMS)),
], ids=["counter", "co", "cross", "shell"])
def test_two_stream_hx_smoke(make):
    hot_out, cold_out, info = make()(*_streams())
    assert np.isfinite(float(hot_out["T"])) and np.isfinite(float(cold_out["T"]))
    assert np.isfinite(float(info["Q"]))


def test_crossflow_invalid_mixing():
    with pytest.raises(ValueError, match="Invalid mixing configuration"):
        CrossFlowHX(HeatExchangerParams(**_PARAMS), mixing="invalid")


def test_design_and_size_smoke():
    d = design_heat_exchanger(**_CROSS, flow_config="shell_and_tube", n_shell_passes=2)
    assert {"A", "LMTD", "UA", "F"} <= set(d)
    assert np.isfinite(float(d["A"]))
    s = size_heat_exchanger(5e4, 400.0, 300.0, 1000.0, 1500.0, 100.0, flow_config="counter_current")
    assert {"A", "NTU", "effectiveness", "F"} <= set(s)
    assert np.isfinite(float(s["A"]))


def test_unknown_flow_config_raises():
    with pytest.raises(ValueError, match="flow_config"):
        design_heat_exchanger(**_CROSS, flow_config="cross_flow")


def test_heater_grad_is_finite():
    stream = make_stream({"A": 10.0}, T=300.0, P=101325.0)
    heater = Heater(HeaterParams(Cp=75.0))
    g = jax.grad(lambda duty: heater(stream, duty=duty)[0]["T"])(jnp.array(5000.0))
    assert np.isfinite(float(g))


def test_default_cp_warning():
    with pytest.warns(DefaultCpWarning):
        Heater(HeaterParams(T_out=350.0))(make_stream({"A": 10.0}, T=300.0, P=101325.0))


def test_phase_change_warning():
    hot = make_stream({"n_pentane": 10.0}, T=340.0, P=1.5e5)
    cold = make_stream({"water": 200.0}, T=293.0, P=1e5)
    with pytest.warns(PhaseChangeWarning):
        CounterCurrentHX(HeatExchangerParams(UA=2e4, Cp_hot=120.0, Cp_cold=75.3))(hot, cold)


def test_low_correction_factor_warning():
    with pytest.warns(LowCorrectionFactorWarning):
        design_heat_exchanger(**_CROSS, flow_config="shell_and_tube", n_shell_passes=1)
