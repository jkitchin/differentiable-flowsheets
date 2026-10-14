"""Smoke, gradient and validation tests for the CoolingTower unit and Merkel functions."""

import jax
import numpy as np
import pytest

from difflow import CoolingTower, CoolingTowerParams, make_stream
from difflow.psychrometrics import MW_WATER, saturation_enthalpy
from difflow.units.cooling_tower import maximum_L_over_G, merkel_number

K = 273.15
T_IN, T_OUT, T_WB = 35.0 + K, 27.0 + K, 22.0 + K
L = 10e6 / (4186.0 * 8.0)
H_AIN = float(saturation_enthalpy(T_WB))


def water(T=T_IN, flow_kg=L, P_=3e5):
    return make_stream({"H2O": flow_kg / MW_WATER, "air": 0.0}, T, P_)


def tower(**kw):
    base = dict(mode="design", L_over_G=1.2, T_out=T_OUT)
    base.update(kw)
    return CoolingTower(CoolingTowerParams(**base))


def test_merkel_number_smoke_and_grad():
    assert np.isfinite(float(merkel_number(T_IN, T_OUT, H_AIN, 1.2)))
    g = jax.grad(lambda lg: merkel_number(T_IN, T_OUT, H_AIN, lg))(1.2)
    assert np.isfinite(float(g))
    assert np.isfinite(float(maximum_L_over_G(T_IN, T_OUT, H_AIN)))


def test_zero_range_is_masked_not_nan():
    assert float(merkel_number(T_IN, T_IN, H_AIN, 1.0)) == np.inf


def test_unknown_quadrature():
    with pytest.raises(ValueError):
        merkel_number(T_IN, T_OUT, H_AIN, 1.0, quadrature="simpson")


def test_design_mode_smoke():
    wo, ao, info = tower()(water(), T_wb=T_WB)
    assert bool(info["feasible"])
    for k in ("KaV_L", "Q", "L", "G", "range", "approach"):
        assert np.isfinite(float(info[k]))
    assert np.isfinite(float(wo["T"])) and np.isfinite(float(ao["T"]))


def test_rating_mode_smoke():
    info = tower(mode="rating", KaV_L=1.4)(water(), T_wb=T_WB)[2]
    assert np.isfinite(float(info["T_out"]))


def test_wet_bulb_above_the_requested_outlet_is_infeasible():
    _, _, info = tower(T_out=T_WB - 1.0)(water(), T_wb=T_WB)
    assert not bool(info["feasible"])


def test_call_validation():
    with pytest.raises(ValueError, match="air_in or T_wb"):
        tower()(water())
    with pytest.raises(ValueError, match="L_over_G"):
        tower(L_over_G=None)(water(), T_wb=T_WB)
    with pytest.raises(ValueError, match="T_out or approach"):
        tower(T_out=None)(water(), T_wb=T_WB)
    with pytest.raises(ValueError, match="KaV_L"):
        tower(mode="rating")(water(), T_wb=T_WB)


def test_params_validation():
    with pytest.raises(ValueError):
        CoolingTowerParams(mode="sizing")
    with pytest.raises(ValueError):
        CoolingTowerParams(quadrature="simpson")
