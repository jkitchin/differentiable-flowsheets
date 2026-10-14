"""Smoke tests for installed piping cost (difflow.economics)."""

import jax
import numpy as np
import pytest

import difflow.economics as econ
from difflow.fluids import pipe_weight_per_length

jax.config.update("jax_enable_x64", True)


def test_pipe_cost_smoke_and_grad():
    c = econ.pipe_cost_per_length(0.1, 150.0, 0.0525, 2010, 2022)
    assert np.isfinite(float(c))
    g = jax.grad(lambda D: econ.pipe_cost_per_length(D, 150.0, 0.0525, 2019, 2019))(0.1)
    assert np.isfinite(float(g))
    assert np.isfinite(pipe_weight_per_length(4, "40"))


def test_nps_cost_smoke_and_validation():
    kw = dict(C_ref=200.0, nps_ref=2, base_year=2019, target_year=2019)
    assert np.isfinite(float(econ.pipe_cost_nps(4, "80", length=50.0, **kw)))
    with pytest.raises(KeyError):
        econ.pipe_cost_nps(22, "40", **kw)
