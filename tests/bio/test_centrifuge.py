"""Smoke tests for centrifuge unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import Centrifuge, CentrifugeParams, DiscStackCentrifuge, DiscStackParams
from difflow import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


def _feed():
    return make_stream({"cells": 100.0, "substrate": 10.0, "product": 50.0},
                       T=300.0, P=101325.0)


def test_centrifuge_runs_and_closes():
    c = Centrifuge(CentrifugeParams(sigma=5000.0, efficiency=0.7, cell_species="cells"))
    conc, clar, info = c(_feed(), Q=1e-4, concentrate_fraction=0.1)
    assert jnp.isfinite(info["cell_recovery"])
    for s, f in get_flows(_feed()).items():
        assert float(get_flows(conc)[s] + get_flows(clar)[s]) == pytest.approx(float(f), rel=1e-8)


def test_centrifuge_grad_is_finite():
    def f(sigma):
        conc, _, _ = Centrifuge(CentrifugeParams(sigma=sigma, efficiency=0.9))(
            _feed(), Q=1e-3, concentrate_fraction=0.1)
        return conc["F_cells"]
    assert jnp.isfinite(jax.grad(f)(100.0))


def test_disc_stack_runs():
    c = DiscStackCentrifuge(DiscStackParams(n_discs=100, r_outer=0.1, r_inner=0.05,
                                            rpm=6000.0, efficiency=0.8))
    conc, clar, info = c(_feed(), Q=1e-4, concentrate_fraction=0.1)
    assert jnp.isfinite(conc["F_cells"]) and jnp.isfinite(clar["F_cells"])
