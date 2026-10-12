"""Tests for the elevation profile (cavitation over a crest) of difflow.units.pipe.Pipe."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import Pipe, PipeCavitationWarning, PipeParams, make_stream
from difflow.fluids import GRAVITY
from difflow.units.pump import system_curve

jax.config.update("jax_enable_x64", True)

RHO, MU, MW = 998.2, 1.002e-3, 18.015
P_VAP = 2339.0  # water at 20 C


def water(Q=0.003, P=3e5):
    return make_stream({"W": Q * RHO / (MW * 1e-3)}, T=293.15, P=P)


def pipe(**kw):
    base = dict(L=200.0, D=0.0525, rho=RHO, mu=MU, MW=MW, P_vap=P_VAP)
    base.update(kw)
    return Pipe(PipeParams(**base))


CREST = [(0.0, 0.0), (60.0, 10.0), (100.0, 25.0), (150.0, 12.0), (200.0, 0.0)]


def test_no_profile_is_unchanged():
    out, info = pipe()(water())
    assert "P_min" not in info and "P_profile" not in info
    out_flat, info_flat = pipe(profile=[(0.0, 0.0), (200.0, 0.0)])(water())
    np.testing.assert_allclose(float(out_flat["P"]), float(out["P"]), rtol=1e-14)
    np.testing.assert_allclose(float(info_flat["P_min"]), float(out["P"]), rtol=1e-14)


def test_crest_pressure_by_hand():
    fit = {"elbow_90_standard": 2, "gate_valve": 1}
    p = pipe(profile=CREST, fittings=fit)
    out, info = p(water())
    dPf, dPm = float(info["dP_friction"]), float(info["dP_minor"])
    expected = 3e5 - dPm - dPf * 100.0 / 200.0 - RHO * GRAVITY * 25.0
    np.testing.assert_allclose(float(info["P_min"]), expected, rtol=1e-12)
    assert float(info["s_P_min"]) == 100.0 and float(info["z_P_min"]) == 25.0
    # the outlet is the last point of the profile and dz comes from the profile (0 here)
    np.testing.assert_allclose(float(info["P_profile"][-1]), float(out["P"]), rtol=1e-14)
    np.testing.assert_allclose(float(out["P"]), 3e5 - dPf - dPm, rtol=1e-14)
    np.testing.assert_allclose(float(info["cavitation_margin"]), expected - P_VAP, rtol=1e-12)
    np.testing.assert_allclose(float(info["cavitation_margin_head"]), (expected - P_VAP) / (RHO * GRAVITY),
                               rtol=1e-12)
    prof = p.pressure_profile(water())
    np.testing.assert_allclose(np.asarray(prof["P"]), np.asarray(info["P_profile"]), rtol=1e-14)


def test_crest_cavitates_while_outlet_is_fine():
    """A line over a 32 m crest from 3 bar: healthy outlet, vapour at the crest."""
    crest = [(0.0, 0.0), (100.0, 32.0), (200.0, 0.0)]
    p = pipe(profile=crest)
    with pytest.warns(PipeCavitationWarning, match="s = 100"):
        out, info = p(water())
    assert not bool(info["P_out_below_vapour"])
    assert float(out["P"]) > 2e5
    assert bool(info["P_min_below_vapour"])
    assert float(info["cavitation_margin"]) < 0.0
    # the same line without the profile is not flagged at all
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipeCavitationWarning)
        pipe()(water())


def test_safe_crest_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipeCavitationWarning)
        out, info = pipe(profile=CREST)(water())
    assert not bool(info["P_min_below_vapour"])


def test_no_warning_under_jit():
    crest = [(0.0, 0.0), (100.0, 32.0), (200.0, 0.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipeCavitationWarning)
        flag = jax.jit(lambda P: pipe(profile=crest)(water(P=P))[1]["P_min_below_vapour"])(3e5)
    assert bool(flag)


def test_profile_sets_elevation_gain():
    uphill = [(0.0, 0.0), (50.0, 3.0), (200.0, 8.0)]
    p = pipe(profile=uphill)
    np.testing.assert_allclose(float(p.elevation_gain()), 8.0)
    out, info = p(water())
    out_dz, info_dz = pipe(dz=8.0)(water())
    np.testing.assert_allclose(float(out["P"]), float(out_dz["P"]), rtol=1e-14)
    np.testing.assert_allclose(float(info["dP_static"]), RHO * GRAVITY * 8.0, rtol=1e-14)
    # consistent dz is accepted, an inconsistent one is an error
    pipe(profile=uphill, dz=8.0)(water())
    with pytest.raises(ValueError, match="disagrees"):
        pipe(profile=uphill, dz=5.0)(water())
    # inverse hydraulics and the pump system curve count the profile's gain
    Q = pipe(profile=uphill).flow_at_dp(float(info["dP"]))
    np.testing.assert_allclose(float(Q), 0.003, rtol=1e-8)
    sc = system_curve(0.0, pipe(profile=uphill))
    sc_dz = system_curve(0.0, pipe(dz=8.0))
    np.testing.assert_allclose(float(sc.head(0.003)), float(sc_dz.head(0.003)), rtol=1e-12)


def test_bad_profile_shape():
    with pytest.raises(ValueError, match="at least two"):
        pipe(profile=[(0.0, 0.0)])(water())


@pytest.mark.slow
def test_gradient_of_minimum_pressure():
    def P_min(D, P_in):
        return pipe(D=D, profile=CREST)(water(P=P_in))[1]["P_min"] / 1e5

    check_grads(P_min, (0.0525, 3e5), order=1, modes=["rev", "fwd"], eps=1e-6, rtol=1e-4)
    g = jax.grad(P_min, argnums=1)(0.0525, 3e5)
    np.testing.assert_allclose(float(g), 1e-5, rtol=1e-12)  # dP_min/dP_in = 1
    # and with respect to a crest elevation
    def margin(z):
        prof = jnp.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]).at[1, 1].set(z)
        return pipe(profile=prof)(water())[1]["cavitation_margin"]
    np.testing.assert_allclose(float(jax.grad(margin)(25.0)), -RHO * GRAVITY, rtol=1e-12)
