"""Smoke tests for bioreactor unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    ContinuousBioreactor,
    FedBatchBioreactor,
    BioreactorParams,
    FedBatchParams,
    monod_kinetics,
    product_inhibition_kinetics,
)
from difflow_bio.units.bioreactors import call_kinetics, get_kinetic_arity
from difflow import make_stream

jax.config.update("jax_enable_x64", True)


def _chemostat():
    return ContinuousBioreactor(BioreactorParams(
        V=10.0, Y_xs=0.5, kinetic_fn=monod_kinetics,
        kinetic_params={"mu_max": 0.4, "K_s": 0.5}, k_d=0.01, alpha=0.1,
        species_order=["cells", "substrate", "product"],
    ))


def _feed():
    return make_stream({"cells": 0.0, "substrate": 40.0, "product": 0.0},
                       T=310.0, P=101325.0)


def test_chemostat_runs():
    outlet, info = _chemostat()(_feed(), D=0.2)
    for k in ("X", "S", "P", "mu"):
        assert jnp.isfinite(info[k])
    assert all(jnp.isfinite(v) for v in outlet.values())


def test_chemostat_grad_is_finite():
    g = jax.grad(lambda D: _chemostat()(_feed(), D=D)[1]["X"] * D)(0.2)
    assert jnp.isfinite(g)


def test_fedbatch_runs():
    fb = FedBatchBioreactor(FedBatchParams(
        V0=5.0, Y_xs=0.5, kinetic_fn=monod_kinetics,
        kinetic_params={"mu_max": 0.4, "K_s": 0.5}, alpha=0.1,
    ))
    _, info = fb(X0=0.5, S0=20.0, P0=0.0, t_final=10.0, n_steps=50, solver="rk4")
    assert jnp.isfinite(info["X_final"]) and jnp.isfinite(info["P_final"])


def test_product_inhibition_gets_product_not_cells():
    """Regression (#93): call_kinetics passed X where P belongs."""
    params = {"mu_max": 0.5, "K_s": 0.1, "P_max": 50.0, "n": 1.0}
    S, X, P = jnp.array(10.0), jnp.array(5.0), jnp.array(25.0)
    arity = get_kinetic_arity(product_inhibition_kinetics)
    mu = call_kinetics(product_inhibition_kinetics, arity, S, X, P, params)
    assert float(mu) == pytest.approx(float(product_inhibition_kinetics(S, P, params)))
