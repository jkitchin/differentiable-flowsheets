"""Smoke tests for difflow.psychrometrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import psychrometrics as ps
from difflow.psychrometrics import (
    adiabatic_humidify,
    air_stream_from_state,
    cool_dehumidify,
    dew_point,
    mix_air,
    psychrometric_chart_data,
    sensible_heat,
    state,
    state_from_air_stream,
    wet_bulb_temperature,
)

jax.config.update("jax_enable_x64", True)


def _finite(s):
    return all(bool(jnp.all(jnp.isfinite(s[k]))) for k in ps.STATE_KEYS)


def test_state_smoke():
    s = state(T=303.15, RH=0.5)
    assert set(ps.STATE_KEYS) <= set(s) and _finite(s)
    assert np.isfinite(float(wet_bulb_temperature(303.15, s["W"])))
    assert np.isfinite(float(dew_point(s["W"])))


def test_state_validation():
    with pytest.raises(ValueError, match="not independent"):
        state(W=0.01, T_dp=285.0)
    with pytest.raises(ValueError, match="exactly two"):
        state(T=300.0)


def test_processes_smoke():
    a, b = state(T=303.15, RH=0.6), state(T=283.15, RH=0.5)
    assert _finite(mix_air([a, b], [1.0, 2.0]))
    assert _finite(sensible_heat(a, 20e3, m_dry_air=2.0))
    out, info = cool_dehumidify(a, 281.15, bypass_factor=0.15)
    assert _finite(out) and np.isfinite(float(info["Q_total"]))
    assert _finite(adiabatic_humidify(a, effectiveness=0.8))
    with pytest.raises(ValueError):
        adiabatic_humidify(a, RH_out=0.9, mu_out=0.9)


def test_air_stream_round_trip():
    s = state(T=305.0, RH=0.45)
    stream = air_stream_from_state(s, 12.5)
    assert set(stream) == {"F_H2O", "F_air", "T", "P"}
    assert float(state_from_air_stream(stream)["W"]) == pytest.approx(float(s["W"]), rel=1e-9)
    empty = state_from_air_stream({"F_H2O": 0.0, "F_air": 0.0, "T": 300.0, "P": 101325.0})
    assert bool(jnp.isfinite(empty["h"]))


def test_grad_finite():
    g = jax.grad(lambda T: state(T=T, RH=0.5)["T_wb"])(300.0)
    assert jnp.isfinite(g)


def test_chart_data_and_plot():
    d = psychrometric_chart_data()
    assert {"saturation", "rh", "wb", "h", "v"} <= set(d)
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from difflow.visualization import plot_psychrometric_chart
    ax = plot_psychrometric_chart(d)
    assert ax.get_xlabel().startswith("dry-bulb")
