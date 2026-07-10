"""Tests for difflow_gas.physics."""

import jax.numpy as jnp
import pytest

from difflow_gas.physics import (
    DEFAULT_TEMP_K,
    EPS_FLOW,
    compressor_power,
    kg_s_to_knm3h,
    knm3h_to_kg_s,
    nikuradse_friction,
    resistor_xi,
    smoothed_power_w,
    specific_gas_constant,
    weymouth_beta,
)


def test_specific_gas_constant():
    # R_s = R / M; methane-ish M = 16 kg/kmol
    assert specific_gas_constant(16.0) == pytest.approx(8314.462618 / 16.0)


def test_flow_conversion_roundtrip():
    q = knm3h_to_kg_s(300.0, 0.785)
    assert kg_s_to_knm3h(q, 0.785) == pytest.approx(300.0)
    # 300 kNm3/h of 0.785 kg/m3 gas = 300 * 1000/3600 * 0.785 kg/s
    assert q == pytest.approx(300.0 * 1000.0 / 3600.0 * 0.785)


def test_nikuradse_friction_decreases_with_smoothness():
    rough = nikuradse_friction(0.5, 1e-3)
    smooth = nikuradse_friction(0.5, 1e-5)
    assert smooth < rough
    # golden regression value (L = D/k = 5000)
    assert nikuradse_friction(0.5, 1e-4) == pytest.approx(
        0.01372452402130078, rel=1e-12
    )


def test_weymouth_beta_golden_and_scaling():
    beta = weymouth_beta(1000.0, 0.5, 1e-4, 283.15, 18.0)
    # golden regression value pinning the formula
    # beta = 16 lambda L R_s T z / (pi^2 D^5)
    assert beta == pytest.approx(83808537.60334712, rel=1e-12)
    # linear in length, inverse fifth power in diameter
    assert weymouth_beta(2000.0, 0.5, 1e-4, 283.15, 18.0) == pytest.approx(
        2 * beta
    )
    ratio = beta / weymouth_beta(1000.0, 1.0, 1e-4, 283.15, 18.0)
    # D^-5 plus the (weak) friction-factor change with D/k
    lam_ratio = nikuradse_friction(0.5, 1e-4) / nikuradse_friction(1.0, 1e-4)
    assert ratio == pytest.approx(32.0 * lam_ratio, rel=1e-12)


def test_resistor_xi_golden():
    assert resistor_xi(5.0, 0.5, 283.15, 18.0) == pytest.approx(
        7633100.560834964, rel=1e-12
    )


def test_compressor_power_zero_at_unity_ratio():
    assert compressor_power(100.0, 1.0) == 0.0
    assert float(smoothed_power_w(100.0, 1.0)) == 0.0


def test_compressor_power_positive_and_increasing():
    w1 = compressor_power(100.0, 1.2, DEFAULT_TEMP_K)
    w2 = compressor_power(100.0, 1.4, DEFAULT_TEMP_K)
    assert 0 < w1 < w2


def test_smoothed_power_matches_plain_away_from_zero():
    w_plain = compressor_power(50.0, 1.3)
    w_smooth = float(smoothed_power_w(50.0, 1.3))
    # smoothing error is O(eps^2 / q)
    assert w_smooth == pytest.approx(w_plain, rel=1e-9)


def test_smoothed_power_uses_absolute_flow():
    # reverse flow through a boosting station still costs |q|-based power
    w_neg = float(smoothed_power_w(-50.0, 1.3))
    w_pos = float(smoothed_power_w(50.0, 1.3))
    assert w_neg == pytest.approx(w_pos)
    # and is C^1 at q = 0 with value ~ eps * (...)
    w0 = float(smoothed_power_w(0.0, 1.3))
    assert 0 < w0 < float(smoothed_power_w(EPS_FLOW * 10, 1.3))


def test_smoothed_power_is_differentiable_at_zero():
    import jax

    g = jax.grad(lambda q: smoothed_power_w(q, 1.3))(0.0)
    assert jnp.isfinite(g)
    assert float(g) == pytest.approx(0.0, abs=1e-12)  # even function
