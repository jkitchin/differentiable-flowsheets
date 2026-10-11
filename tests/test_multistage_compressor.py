"""MultistageCompressor and stage_pressures (issue 442, Ch. 5)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import (
    Compressor, CompressorParams, CubicThermo, IdealThermo, MultistageCompressor,
    MultistageCompressorParams, PengRobinson, make_stream,
)
from difflow.compressible import optimal_stage_ratios, stage_pressures
from difflow.database import get_critical_props, get_species_data

jax.config.update("jax_enable_x64", True)

NAMES = ["methane", "ethane"]


@pytest.fixture(scope="module")
def thermo():
    return CubicThermo(IdealThermo({c: get_species_data(c) for c in NAMES}),
                       PengRobinson({c: get_critical_props(c) for c in NAMES}))


@pytest.fixture(scope="module")
def feed():
    return make_stream({"methane": 0.9, "ethane": 0.1}, T=313.15, P=1e5)


# ---------------------------------------------------------------------------
# stage_pressures
# ---------------------------------------------------------------------------

def test_stage_pressures_without_loss_is_the_equal_ratio_split():
    Ps, Pd = stage_pressures(1e5, 110e5, 4)
    np.testing.assert_allclose(np.asarray(Pd / Ps), np.asarray(optimal_stage_ratios(1e5, 110e5, 4)),
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(Ps[1:]), np.asarray(Pd[:-1]), rtol=1e-12)


def test_stage_pressures_with_loss():
    dP = 0.3e5
    Ps, Pd = stage_pressures(1e5, 30e5, 3, dP)
    r = Pd / Ps
    np.testing.assert_allclose(np.asarray(r), float(r[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(Ps[1:]), np.asarray(Pd[:-1] - dP), rtol=1e-12)
    assert float(Pd[-1]) == pytest.approx(30e5, rel=1e-12)
    assert float(r[0]) > 30 ** (1 / 3)  # the loss is made up by a higher ratio


def test_stage_pressures_gradients():
    f = lambda Pout, dP: stage_pressures(1.0, Pout, 3, dP)[1][0]  # noqa: E731
    check_grads(f, (30.0, 0.2), order=1, modes=["rev"], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        stage_pressures(1e5, 30e5, 0)


# ---------------------------------------------------------------------------
# MultistageCompressor
# ---------------------------------------------------------------------------

def test_train_balances(thermo, feed):
    params = MultistageCompressorParams(P_out=30e5, n_stages=3, intercooler_dP=0.2e5)
    out, info = MultistageCompressor(params, thermo)(feed)
    assert bool(info["converged"])
    assert float(out["P"]) == pytest.approx(30e5)
    Ps, Pd = stage_pressures(1e5, 30e5, 3, 0.2e5)
    np.testing.assert_allclose(np.asarray(info["discharge_pressures"]), np.asarray(Pd))
    # First law over the train: W + Q = H_out - H_in.
    flows = {"methane": 0.9, "ethane": 0.1}
    H_in = thermo.stream_enthalpy_flash(flows, 313.15, 1e5)
    H_out = thermo.stream_enthalpy_flash(flows, out["T"], out["P"])
    assert float(info["W"] + info["Q"]) == pytest.approx(float(H_out - H_in), rel=1e-6)
    # Intercooling saves work against one stage across the whole ratio.
    _, one = Compressor(CompressorParams(P_out=30e5), thermo)(feed)
    assert float(info["W"]) < 0.85 * float(one["W"])
    assert np.all(np.asarray(info["intercooler_vapor_fraction"]) == 1.0)


def test_one_stage_train_is_the_compressor(thermo, feed):
    _, one = Compressor(CompressorParams(P_out=8e5), thermo)(feed)
    out, info = MultistageCompressor(
        MultistageCompressorParams(P_out=8e5, n_stages=1), thermo)(feed)
    assert float(info["W"]) == pytest.approx(float(one["W"]), rel=1e-10)
    assert float(out["T"]) == pytest.approx(float(one["T_out"]), rel=1e-10)


def test_aftercooler_and_given_pressures(thermo, feed):
    params = MultistageCompressorParams(P_out=0.0, discharge_pressures=(3e5, 9e5),
                                        intercooler_dP=0.1e5, aftercool=True)
    out, info = MultistageCompressor(params, thermo)(feed)
    assert float(out["T"]) == pytest.approx(313.15)
    assert float(out["P"]) == pytest.approx(8.9e5)
    np.testing.assert_allclose(np.asarray(info["suction_pressures"]), [1e5, 2.9e5])
    assert info["intercooler_Q"].shape == (2,)


def test_work_is_differentiable_in_the_split(thermo, feed):
    # Near-ideal gas, no loss: the equal split is close to the minimum, so the
    # derivative of the total work in the middle pressure is small against
    # its derivative in the final pressure.
    def W(P_mid, P_out):
        params = MultistageCompressorParams(P_out=0.0, discharge_pressures=(P_mid * 1e5, P_out * 1e5))
        return MultistageCompressor(params, thermo)(feed)[1]["W"] / 1e3

    P_mid = float(np.sqrt(9.0))
    g_mid, g_out = jax.grad(W, argnums=(0, 1))(P_mid, 9.0)
    assert float(g_out) > 0
    assert abs(float(g_mid)) < 0.05 * float(g_out) * 9.0 / P_mid
    check_grads(W, (P_mid, 9.0), order=1, modes=["rev"], atol=1e-3, rtol=1e-3)
