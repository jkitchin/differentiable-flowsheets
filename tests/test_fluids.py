"""Smoke tests for difflow.fluids: friction factors, fittings and flow meters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.fluids import (
    FITTINGS,
    darcy_pressure_drop,
    equivalent_length,
    fitting_K,
    friction_factor,
    fully_rough_friction_factor,
    hydraulic_diameter,
    minor_loss,
    orifice_dp,
    orifice_flow,
    reynolds_number,
    venturi_dp,
    venturi_flow,
)


def test_basic_functions_finite():
    for v in (reynolds_number(1000.0, 2.0, 0.05, 1e-3),
              hydraulic_diameter(0.01, 0.3),
              darcy_pressure_drop(0.02, 100.0, 0.05, 1000.0, 2.0),
              minor_loss(2.0, 1000.0, 3.0),
              fully_rough_friction_factor(1e-3)):
        assert np.isfinite(float(v))


def test_friction_factor_methods_and_grad():
    Re = jnp.array([1e3, 3e3, 1e5])
    for method in ("colebrook", "churchill", "haaland", "swamee_jain"):
        f = jax.jit(friction_factor, static_argnums=2)(Re, 1e-4, method)
        assert f.shape == (3,) and jnp.all(jnp.isfinite(f))
    g = jax.grad(friction_factor, argnums=(0, 1))(1e5, 0.0)
    assert all(np.isfinite(float(x)) for x in g)
    with pytest.raises(ValueError):
        friction_factor(1e5, 0.0, "blasius")


def test_fittings():
    assert FITTINGS
    assert float(fitting_K("exit")) == 1.0
    K = fitting_K("elbow_90_standard", f_T=0.02)
    assert np.isfinite(float(equivalent_length(K, 0.05, 0.02)))
    with pytest.raises(ValueError):
        fitting_K("gate_valve")
    with pytest.raises(KeyError):
        fitting_K("nonexistent")


def test_flow_meters():
    D, d, rho = 0.1, 0.05, 998.0
    for fl, dp in ((orifice_flow, orifice_dp), (venturi_flow, venturi_dp)):
        Q = fl(1e4, D, d, rho)
        assert float(dp(Q, D, d, rho)) == pytest.approx(1e4, rel=1e-10)
    assert np.isfinite(float(jax.grad(lambda x: orifice_flow(x, D, d, rho))(1e4)))
