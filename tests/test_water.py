"""Smoke tests for difflow.water: the shared water/steam correlations."""

import jax
import jax.numpy as jnp

from difflow.water import (
    water_latent_heat,
    water_liquid_enthalpy,
    water_sublimation_pressure,
    water_vapor_enthalpy,
    water_vapor_pressure,
)


def test_correlations_finite():
    T = jnp.array([263.15, 300.0, 400.0])
    for f in (water_latent_heat, water_liquid_enthalpy, water_vapor_enthalpy,
              water_vapor_pressure, water_sublimation_pressure):
        assert bool(jnp.all(jnp.isfinite(f(T))))


def test_vapor_pressure_grad_finite():
    assert jnp.isfinite(jax.grad(water_vapor_pressure)(273.16))
