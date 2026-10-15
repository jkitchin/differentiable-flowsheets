"""Dense-gas viscosity/conductivity and mixture rules (issue 442, Ch. 5)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.gas_properties import (
    dense_gas_conductivity, dense_gas_viscosity, gas_mixture_transport,
    mixture_conductivity, mixture_viscosity,
)

jax.config.update("jax_enable_x64", True)

AIR = (["nitrogen", "oxygen", "argon"], [0.7812, 0.2096, 0.0092])


def test_dense_gas_smoke_and_grad():
    assert np.isfinite(float(dense_gas_viscosity("methane", 288.15, 50e5)))
    assert np.isfinite(float(dense_gas_conductivity("nitrogen", 300.0, 50e5)))
    g = jax.grad(lambda P: dense_gas_viscosity("methane", 300.0, P))(50e5)
    assert np.isfinite(float(g))


def test_mixture_rules_smoke():
    assert np.isfinite(float(mixture_viscosity(*AIR, 300.0)))
    assert np.isfinite(float(mixture_viscosity(*AIR, 300.0, method="herning_zipperer")))
    assert np.isfinite(float(mixture_conductivity(*AIR, 300.0)))
    g = jax.grad(lambda y1: dense_gas_conductivity(
        ["methane", "nitrogen"], 300.0, 50e5, y=jnp.stack([y1, 1 - y1])))(0.8)
    assert np.isfinite(float(g))


def test_bad_inputs():
    with pytest.raises(ValueError):
        mixture_viscosity(*AIR, 300.0, method="nope")
    with pytest.raises(ValueError):
        dense_gas_viscosity(["methane", "nitrogen"], 300.0, 1e6)  # no y


def test_mixture_transport_smoke():
    r = gas_mixture_transport(["methane", "nitrogen", "carbon_dioxide"],
                              [0.9, 0.05, 0.05], 288.15, 70e5)
    for k in ("Z", "mu", "k", "rho", "MW"):
        assert np.isfinite(float(r[k]))
