"""Tests for the rotameter (variable-area meter) model in difflow.fluids."""

import math

import jax
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow.fluids import GRAVITY, rotameter_float_dp, rotameter_flow, rotameter_height

jax.config.update("jax_enable_x64", True)

# A stainless float in a glass tube: 20 mm float, 4 cm^3, taper 0.05 m/m
GEOM = dict(D_float=0.02, V_float=4e-6, rho_float=8000.0, C_R=0.72, taper=0.05)


def _areas(h, D_float, taper, D0=None):
    D0 = D_float if D0 is None else D0
    Dt = D0 + taper * h
    A_t = math.pi / 4 * Dt**2
    A_f = math.pi / 4 * D_float**2
    return A_t, A_f, A_t - A_f


@pytest.mark.parametrize("h", [0.02, 0.1, 0.25])
@pytest.mark.parametrize("rho", [700.0, 1000.0, 1500.0])
def test_force_balance_identity(h, rho):
    """The orifice-equation pressure drop at the computed Q times A_f equals V_f (rho_f - rho) g."""
    Q = float(rotameter_flow(h, rho=rho, **GEOM))
    A_t, A_f, A_a = _areas(h, GEOM["D_float"], GEOM["taper"])
    dP = 0.5 * rho * (Q / (GEOM["C_R"] * A_a)) ** 2 * (1.0 - (A_a / A_t) ** 2)
    np.testing.assert_allclose(dP * A_f, GEOM["V_float"] * (GEOM["rho_float"] - rho) * GRAVITY, rtol=1e-12)
    np.testing.assert_allclose(float(rotameter_float_dp(GEOM["V_float"], GEOM["rho_float"], rho, A_f)),
                               dP, rtol=1e-12)


def test_hand_value():
    h, rho = 0.1, 1000.0
    A_t, A_f, A_a = _areas(h, 0.02, 0.05)
    Q = 0.72 * A_a / math.sqrt(1 - (A_a / A_t) ** 2) * math.sqrt(2 * GRAVITY * 4e-6 * 7000.0 / (1000.0 * A_f))
    np.testing.assert_allclose(float(rotameter_flow(h, rho=rho, **GEOM)), Q, rtol=1e-13)


def test_zero_at_seat_and_increasing():
    hs = np.linspace(0.0, 0.3, 13)
    Q = np.asarray(rotameter_flow(hs, rho=1000.0, **GEOM))
    assert Q[0] == 0.0
    assert np.all(np.diff(Q) > 0)


@pytest.mark.parametrize("h", [0.005, 0.05, 0.2, 0.3])
def test_height_inverts_flow(h):
    Q = rotameter_flow(h, rho=998.0, **GEOM)
    np.testing.assert_allclose(float(rotameter_height(Q, rho=998.0, **GEOM)), h, rtol=1e-10)


def test_tube_offset_diameter():
    kw = dict(GEOM, D_tube_0=0.021)
    Q = rotameter_flow(0.0, rho=1000.0, **kw)
    assert float(Q) > 0.0  # annulus already open at h = 0
    np.testing.assert_allclose(float(rotameter_height(Q, rho=1000.0, **kw)), 0.0, atol=1e-10)


def test_density_correction():
    """Mass flow at fixed float height scales as sqrt(rho (rho_f - rho))."""
    h = 0.15
    m1 = 800.0 * float(rotameter_flow(h, rho=800.0, **GEOM))
    m2 = 1200.0 * float(rotameter_flow(h, rho=1200.0, **GEOM))
    np.testing.assert_allclose(m1 / m2, math.sqrt(800 * 7200 / (1200 * 6800)), rtol=1e-12)


def test_gradients():
    check_grads(lambda h, rho: rotameter_flow(h, rho=rho, **GEOM) * 1e4, (0.12, 1000.0), order=1,
                modes=["rev", "fwd"], eps=1e-6, rtol=1e-4)
    check_grads(lambda Q, rho: rotameter_height(Q, rho=rho, **GEOM), (2e-4, 1000.0), order=1,
                modes=["rev", "fwd"], eps=1e-8, rtol=1e-4)
    # inverse-function derivative: dh/dQ = 1 / (dQ/dh)
    h0 = 0.12
    Q0 = rotameter_flow(h0, rho=1000.0, **GEOM)
    dQdh = jax.grad(lambda h: rotameter_flow(h, rho=1000.0, **GEOM))(h0)
    dhdQ = jax.grad(lambda Q: rotameter_height(Q, rho=1000.0, **GEOM))(Q0)
    np.testing.assert_allclose(float(dQdh) * float(dhdQ), 1.0, rtol=1e-10)
