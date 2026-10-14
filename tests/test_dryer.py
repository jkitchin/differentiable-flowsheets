"""Smoke tests for the drying library and the Dryer unit (issue #432)."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    Dryer,
    DryerParams,
    constant_drying_rate,
    dry_basis,
    drying_periods,
    drying_rate_curve,
    drying_time,
    equilibrium_moisture,
    free_moisture,
    heat_transfer_coefficient_tray,
    solids_moisture,
    water_activity,
    wet_basis,
)
from difflow.psychrometrics import MW_WATER, air_stream_from_state, state

K0 = 273.15
X1, X2, XC, XE, RC, LSA = 0.35, 0.10, 0.20, 0.05, 2.78e-4, 25.0


def make_air(T_C=100.0, W=0.01, G=10.0):
    return air_stream_from_state(state(T=T_C + K0, W=W), G)


def make_wet(m_s=0.5, X=0.30, T=300.0):
    return {"F_solid": m_s, "F_H2O": m_s * X / MW_WATER, "T": T, "P": 101325.0}


def test_moisture_bookkeeping():
    assert float(dry_basis(0.2)) == pytest.approx(0.25)
    assert float(wet_basis(0.25)) == pytest.approx(0.2)
    assert float(free_moisture(0.3, 0.05)) == pytest.approx(0.25)
    s = {"F_solid": 2.0, "F_H2O": 0.5 / MW_WATER}
    assert float(solids_moisture(s)) == pytest.approx(0.25)


def test_isotherms():
    p = {"X_m": 0.07, "C": 12.0, "K": 0.85}
    X = equilibrium_moisture(0.4, 300.0, "gab", p)
    assert jnp.isfinite(X)
    assert float(water_activity(X, 300.0, "gab", p)) == pytest.approx(0.4, rel=1e-6)
    assert jnp.isfinite(equilibrium_moisture(0.4, 300.0, "bet", {"X_m": 0.06, "C": 9.0}))
    tab = {"a_w": [0.0, 0.5, 1.0], "X": [0.0, 0.1, 0.4]}
    assert float(equilibrium_moisture(0.5, 300.0, "table", tab)) == pytest.approx(0.1)
    assert float(equilibrium_moisture(0.5, 300.0, lambda a, T: 0.2 * a)) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        equilibrium_moisture(0.5, 300.0, "gab")


def test_constant_rate():
    assert jnp.isfinite(constant_drying_rate(65.0 + K0, 0.01, 30.0))
    assert jnp.isfinite(constant_drying_rate(60.0 + K0, 0.02, None, k_W=0.05))
    assert jnp.isfinite(heat_transfer_coefficient_tray(1.5))
    with pytest.raises(ValueError):
        heat_transfer_coefficient_tray(1.5, "diagonal")


def test_drying_time():
    t = drying_time(X1, X2, XC, XE, RC, LSA)
    t_c, t_f = drying_periods(X1, X2, XC, XE, RC, LSA)
    assert float(t) == pytest.approx(float(t_c + t_f), rel=1e-6)
    assert jnp.all(jnp.isfinite(drying_rate_curve(jnp.array([0.4, 0.12]), XC, XE, RC)))
    g = jax.grad(drying_time, argnums=(0, 1, 2, 3))(X1, X2, XC, XE, RC, LSA)
    assert all(jnp.isfinite(x) for x in g)
    with pytest.raises(ValueError):
        drying_time(X1, X2, XC, XE, RC, LSA, falling="diffusion")


def test_continuous_dryer():
    params = DryerParams(mode="continuous", X_c=0.15, X_eq=0.02, product_moisture=0.05,
                         h=60.0, solids_per_area=10.0, max_solids_T=340.0)
    dry, air_out, info = Dryer(params)(make_wet(), make_air())
    assert jnp.isfinite(dry["T"]) and jnp.isfinite(air_out["T"])
    assert abs(float(info["water_balance_residual"])) < 1e-10
    with pytest.raises(ValueError):
        Dryer(params.update(product_moisture=None))(make_wet(), make_air())


def test_continuous_dryer_gradient():
    params = DryerParams(mode="continuous", X_c=0.15, X_eq=0.02, product_moisture=0.05, h=60.0)

    def T_out(G):
        return Dryer(params)(make_wet(), make_air(G=G))[1]["T"]

    assert jnp.isfinite(jax.grad(T_out)(10.0))


def test_batch_dryer():
    params = DryerParams(mode="batch", X_c=0.20, X_eq=0.05, product_moisture=0.10,
                         air_mass_velocity=1.0, load=12.5, area=0.5, max_solids_T=330.0)
    dry, _, info = Dryer(params)(make_wet(0.01, 0.35, 295.0), make_air(70.0, 0.01, 0.5))
    assert jnp.isfinite(info["drying_time"]) and float(info["drying_time"]) > 0
    assert float(solids_moisture(dry)) == pytest.approx(0.10, rel=1e-8)
