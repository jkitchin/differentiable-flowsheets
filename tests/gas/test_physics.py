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


@pytest.mark.release
def test_smoothed_power_is_differentiable_at_zero():
    import jax

    g = jax.grad(lambda q: smoothed_power_w(q, 1.3))(0.0)
    assert jnp.isfinite(g)
    assert float(g) == pytest.approx(0.0, abs=1e-12)  # even function


# ---------------------------------------------------------------------------
# isothermal_pipe_beta (G05-5): jnp, differentiable, honest friction options
# ---------------------------------------------------------------------------

from difflow_gas.physics import isothermal_pipe_beta, weymouth_friction  # noqa: E402


def test_weymouth_beta_is_the_isothermal_pipe_beta_alias():
    assert weymouth_beta is isothermal_pipe_beta
    assert isothermal_pipe_beta(1000.0, 0.5, 1e-4, 283.15, 18.0) == pytest.approx(
        83808537.60334712, rel=1e-12
    )


def test_pipe_beta_differentiable_in_length_diameter_roughness():
    import jax
    from jax.test_util import check_grads

    def f(L_km, D, eps_um):  # O(1) arguments for the finite differences
        return isothermal_pipe_beta(L_km * 1e3, D, eps_um * 1e-6, 288.15, 16.04, 0.851) / 1e6

    check_grads(f, (50.0, 0.6, 45.0), order=1, modes=["rev"], eps=1e-4)
    g = jax.grad(f, argnums=(0, 1, 2))(50.0, 0.6, 45.0)
    assert float(g[0]) > 0 and float(g[1]) < 0 and float(g[2]) > 0


def test_friction_laws_book_oracles():
    """wex05-1: 0.6 m, 45 um, 121.54 kg/s methane, mu 1.2e-5 Pa s."""
    from difflow_gas.physics import R_UNIVERSAL

    pref = 16.0 * 50e3 * (R_UNIVERSAL / 16.04) * 288.15 * 0.851 / (jnp.pi**2 * 0.6**5)
    nik = isothermal_pipe_beta(50e3, 0.6, 45e-6, 288.15, 16.04, 0.851)
    assert float(nik / pref) == pytest.approx(0.011347, abs=2e-6)
    cb = isothermal_pipe_beta(50e3, 0.6, 45e-6, 288.15, 16.04, 0.851,
                              friction="colebrook", mass_flow_kg_s=121.54,
                              viscosity_pa_s=1.2e-5)
    assert float(cb / pref) == pytest.approx(0.011461, abs=2e-6)
    wy = isothermal_pipe_beta(50e3, 0.6, 45e-6, 288.15, 16.04, 0.851,
                              friction="weymouth")
    # 0.032 / d_in^(1/3), d_in = 0.6/0.0254 = 23.62 in
    assert float(wy / pref) == pytest.approx(0.032 / (0.6 / 0.0254) ** (1 / 3), rel=1e-12)
    assert float(weymouth_friction(0.6)) == pytest.approx(0.011153, abs=1e-6)


def test_pipe_beta_rejects_bad_friction_inputs():
    with pytest.raises(ValueError):
        isothermal_pipe_beta(1e3, 0.5, 1e-4, friction="panhandle")
    with pytest.raises(ValueError):
        isothermal_pipe_beta(1e3, 0.5, 1e-4, friction="colebrook")
