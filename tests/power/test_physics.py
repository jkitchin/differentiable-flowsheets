"""Smoke tests for difflow_power.physics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power import physics as ph


def test_physics_functions_smoke():
    yff, yft, ytf, ytt = ph.branch_admittances(0.01, 0.1, 0.05, 0.95, 0.1)
    v = jnp.asarray([1.0 + 0j, 0.98 * jnp.exp(-0.05j)])
    s_f, s_t = ph.branch_power_flows(v, [0], [1], yff, yft, ytf, ytt)
    assert bool(jnp.all(jnp.isfinite(s_f))) and bool(jnp.all(jnp.isfinite(s_t)))
    assert np.isfinite(float(ph.apparent_power_squared(s_f[0])))
    assert np.isfinite(float(ph.polynomial_cost(100.0, (0.11, 5.0, 150.0))))
    assert np.isfinite(float(ph.marginal_cost(100.0, (0.11, 5.0, 150.0))))
    assert np.isfinite(float(ph.line_charging_pu(100.0, 12.0, 230.0, 100.0, 60.0)))
    z = ph.ohms_to_pu(52.9, 230.0, 100.0)
    assert ph.pu_to_ohms(z, 230.0, 100.0) == pytest.approx(52.9)
    assert ph.pu_to_mw(ph.mw_to_pu(250.0)) == pytest.approx(250.0)


def test_ybus_shape():
    net = dp.cases.case3()
    assert net.ybus().shape == (3, 3)


def test_admittance_grad_is_finite():
    g = jax.grad(lambda x: jnp.imag(ph.branch_admittances(0.01, x, 0.05)[0]))(0.1)
    assert np.isfinite(float(g))
