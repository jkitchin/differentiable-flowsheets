"""Thermal desalination (MVC, MED) and the thermal seawater properties: code checks."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw
from difflow_water.chemistry import seawater as sw

jax.config.update("jax_enable_x64", True)


def _mvc_feed():
    return dw.make_seawater_stream(flow_mass=44.444, salinity=0.1)


def test_thermal_properties_finite_and_differentiable():
    T, S = 348.15, 0.07
    for f in (sw.specific_heat, sw.enthalpy, sw.latent_heat, sw.vapor_pressure,
              sw.boiling_point_elevation):
        assert jnp.isfinite(f(T, S))
        assert jnp.isfinite(jnp.array(jax.grad(f, argnums=(0, 1))(T, S))).all()
    assert jnp.isfinite(sw.vapor_enthalpy(T, 3e4))
    assert jnp.isfinite(jax.grad(sw.saturation_temperature)(3e4))


def test_mvc_runs():
    out = dw.mvc_performance(_mvc_feed())
    assert bool(out["converged"]) and bool(out["feasible"])
    for k in ("evaporator_area", "compressor_work", "SEC", "LCOW", "pressure_ratio"):
        assert jnp.isfinite(out[k]), k
    assert set(out["capital"]) == {"evaporator", "hx_distillate", "hx_brine", "compressor",
                                   "pump_feed", "pump_brine", "pump_distillate", "mixer_feed"}


def test_mvc_fixed_pressure_ratio_mode_runs():
    p = dw.MVCParams(mode="fixed_pressure_ratio", pressure_ratio=1.58)
    out = dw.mvc_performance(_mvc_feed(), p)
    assert bool(out["converged"]) and jnp.isfinite(out["condensate_subcooling"])


def test_mvc_gradient_finite():
    feed = _mvc_feed()

    def f(x):
        return dw.mvc_performance(feed, dw.MVCParams(T_brine=x[0], A_hx_brine=x[1]))["LCOW"]

    g = jax.grad(f)(jnp.array([348.15, 175.0]))
    assert jnp.isfinite(g).all()


def test_mvc_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        dw.mvc_performance(_mvc_feed(), dw.MVCParams(mode="bogus"))


def test_mvc_unit_outlets():
    dist, brine, info = dw.MVCDesalinator(dw.MVCParams())(_mvc_feed())
    assert {"F_H2O", "F_TDS", "T", "P"} <= set(dist) and {"F_H2O", "F_TDS"} <= set(brine)
    assert jnp.isfinite(info["LCOW"]) and "capital" not in info


def test_med_runs_and_differentiates():
    feed = dw.make_seawater_stream(flow_mass=20.0, salinity=0.035)
    dist, brine, info = dw.MEDDesalinator(dw.MEDParams(n_effects=4))(feed)
    assert info["D"].shape == (4,) and jnp.isfinite(info["GOR"])
    assert jnp.isfinite(dist["F_H2O"]) and jnp.isfinite(brine["F_TDS"])
    g = jax.grad(lambda s: dw.med_performance(feed, dw.MEDParams(n_effects=4, steam_flow=s))
                 ["distillate_mass"])(1.0)
    assert jnp.isfinite(g)


def test_med_rejects_zero_effects():
    with pytest.raises(ValueError, match="n_effects"):
        dw.MEDDesalinator(dw.MEDParams(n_effects=0))
