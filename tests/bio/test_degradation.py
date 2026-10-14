"""Smoke tests for protein degradation kinetics."""

import jax
import jax.numpy as jnp

from difflow_bio import aggregate_fraction, stretched_exponential_fraction, lumry_eyring_fraction

jax.config.update("jax_enable_x64", True)


def test_degradation_models_run_and_differentiate():
    assert jnp.isfinite(aggregate_fraction(5.0, 0.1))
    assert jnp.isfinite(stretched_exponential_fraction(5.0, 0.1, beta=0.7))
    assert jnp.isfinite(lumry_eyring_fraction(5.0, 0.3, 0.5))
    assert jnp.isfinite(lumry_eyring_fraction(5.0, 0.3, 0.3))  # equal rates: no 0/0
    assert jnp.isfinite(jax.grad(lambda k: lumry_eyring_fraction(5.0, 0.3, k))(0.5))
