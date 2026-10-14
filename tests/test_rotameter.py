"""Smoke tests for the rotameter (variable-area meter) model in difflow.fluids."""

import jax
import numpy as np

from difflow.fluids import rotameter_flow, rotameter_height

jax.config.update("jax_enable_x64", True)

GEOM = dict(D_float=0.02, V_float=4e-6, rho_float=8000.0, C_R=0.72, taper=0.05)


def test_flow_and_height_smoke():
    Q = rotameter_flow(0.1, rho=1000.0, **GEOM)
    assert np.isfinite(float(Q)) and float(Q) > 0.0
    h = rotameter_height(Q, rho=1000.0, **GEOM)
    np.testing.assert_allclose(float(h), 0.1, rtol=1e-8)


def test_grad_finite():
    g = jax.grad(lambda h: rotameter_flow(h, rho=1000.0, **GEOM))(0.12)
    assert np.isfinite(float(g))
