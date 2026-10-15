"""Smoke tests for the non-Newtonian friction factors in difflow.fluids."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.fluids import (
    bingham_friction_factor,
    bingham_pressure_drop,
    hedstrom_number,
    metzner_reed_reynolds,
    power_law_friction_factor,
    power_law_pressure_drop,
)

jax.config.update("jax_enable_x64", True)


def test_power_law_smoke():
    assert np.isfinite(float(metzner_reed_reynolds(998.0, 1.3, 0.05, 1.0e-3, 0.7)))
    f = power_law_friction_factor(jnp.array([500.0, 3e4]), 0.6)
    assert np.all(np.isfinite(np.asarray(f)))
    assert np.isfinite(float(power_law_pressure_drop(3e-3, 50.0, 0.05, 1050.0, 0.05, 0.7)))


def test_bingham_smoke():
    assert np.isfinite(float(hedstrom_number(1200.0, 8.0, 0.05, 0.05)))
    f = bingham_friction_factor(jnp.array([1.0, 10.0, 1e3, 1e5, 1e7]), 1e4)
    assert np.all(np.isfinite(np.asarray(f)))
    assert np.isfinite(float(bingham_pressure_drop(5e-4, 20.0, 0.05, 1200.0, 8.0, 0.05)))
    with pytest.raises(ValueError):
        bingham_friction_factor(1e3, 1e3, "darby1981")


def test_grad_finite():
    assert np.isfinite(float(jax.grad(lambda R: power_law_friction_factor(R, 0.6))(3e4)))
    assert np.isfinite(float(jax.grad(lambda R: bingham_friction_factor(R, 1e5))(5e3)))
