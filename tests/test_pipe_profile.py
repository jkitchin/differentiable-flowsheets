"""Smoke tests for the elevation profile (cavitation over a crest) of Pipe."""

import warnings

import jax
import numpy as np
import pytest

from difflow import Pipe, PipeCavitationWarning, PipeParams, make_stream

jax.config.update("jax_enable_x64", True)

RHO, MU, MW = 998.2, 1.002e-3, 18.015
CREST = [(0.0, 0.0), (100.0, 32.0), (200.0, 0.0)]


def water(P=3e5):
    return make_stream({"W": 0.003 * RHO / (MW * 1e-3)}, T=293.15, P=P)


def pipe(**kw):
    base = dict(L=200.0, D=0.0525, rho=RHO, mu=MU, MW=MW, P_vap=2339.0)
    base.update(kw)
    return Pipe(PipeParams(**base))


def test_profile_smoke():
    out, info = pipe(profile=[(0.0, 0.0), (100.0, 5.0), (200.0, 0.0)])(water())
    assert np.isfinite(float(info["P_min"]))
    assert info["P_profile"].shape == (3,)


def test_crest_cavitation_warns():
    with pytest.warns(PipeCavitationWarning):
        _, info = pipe(profile=CREST)(water())
    assert bool(info["P_min_below_vapour"])


def test_no_warning_under_jit():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipeCavitationWarning)
        jax.jit(lambda P: pipe(profile=CREST)(water(P=P))[1]["P_min"])(3e5)


def test_profile_validation():
    with pytest.raises(ValueError, match="at least two"):
        pipe(profile=[(0.0, 0.0)])(water())
    with pytest.raises(ValueError, match="disagrees"):
        pipe(profile=[(0.0, 0.0), (200.0, 8.0)], dz=5.0)(water())


def test_grad_finite():
    g = jax.grad(lambda P: pipe(profile=CREST)(water(P=P))[1]["P_min"])(3e5)
    assert np.isfinite(float(g))
