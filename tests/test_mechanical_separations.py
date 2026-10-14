"""Smoke tests for the mechanical-separation units: Settler, CakeFilter, Cyclone."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import CakeFilter, CakeFilterParams, Cyclone, CycloneParams, Settler, SettlerParams
from difflow.units.mechanical import filtrate_volume, optimal_cycle, ruth_constant_pressure

RHO_S, RHO_L, MU = 2650.0, 998.0, 1.0e-3
D_BINS = [2e-6, 5e-6, 10e-6, 20e-6, 50e-6]
W_BINS = [0.1, 0.2, 0.3, 0.3, 0.1]


def slurry():
    return {"F_solid": 2.0, "F_H2O": 40.0 * 1000.0 / 18.015, "F_NaCl": 3.0, "T": 300.0, "P": 2e5}


def thick_feed():
    return {"F_solid": 5.0, "F_H2O": 50.0 * 1000.0 / 18.015, "T": 300.0, "P": 1e5}


def dusty_gas():
    return {"F_air": 101325.0 * 0.85 / (8.314462618 * 300.0), "F_solid": 0.05,
            "T": 300.0, "P": 101325.0}


def cf_params(**kw):
    base = dict(delta_P=3e5, alpha=1e11, R_m=1e10, cake_moisture=0.4, cycle_time=3600.0,
                downtime=600.0, wash_ratio=0.1)
    base.update(kw)
    return CakeFilterParams(**base)


def test_ruth_library():
    V = jnp.array([0.01, 0.3, 2.0])
    np.testing.assert_allclose(filtrate_volume(ruth_constant_pressure(V, 2e3, 5e2), 2e3, 5e2),
                               V, rtol=1e-10)
    assert np.isfinite(float(jax.grad(filtrate_volume)(0.0, 3.0, 1.0)))
    assert np.isfinite(float(optimal_cycle(12.0, 3.0, 900.0)["t_f"]))


@pytest.mark.parametrize("mode", ["plate", "rotary"])
def test_cake_filter_smoke(mode):
    feed = slurry()
    if mode == "rotary":
        p = CakeFilterParams(delta_P=6e4, alpha=5e10, R_m=2e10, cake_moisture=0.35,
                             mode="rotary", cycle_time=120.0, submergence=0.3)
    else:
        p = cf_params()
    cake, filt, info = CakeFilter(p)(feed)
    for k in ("F_solid", "F_H2O", "F_NaCl"):
        assert float(cake[k] + filt[k]) == pytest.approx(float(feed[k]), rel=1e-10)
    assert np.isfinite(float(info["area"]))


def test_cake_filter_validation():
    with pytest.raises(ValueError):
        cf_params(alpha=None)


def test_settler_smoke_routes():
    t = np.linspace(0.0, 3000.0, 40)
    z = 0.1 + 0.4 * np.exp(-t / 600.0)
    cd = np.linspace(15.0, 150.0, 30)
    routes = [
        dict(d_p=25e-6, richardson_zaki_n=4.65, viscosity=MU),
        dict(method="coe_clevenger", C_data=list(cd), v_data=list(2e-4 * np.exp(-0.012 * cd))),
        dict(method="kynch", t_data=list(t), z_data=list(z), batch_C0=20.0),
        dict(method="overflow_rate", d_p=20e-6, viscosity=MU, d_bins=D_BINS, mass_fractions=W_BINS),
    ]
    feed = thick_feed()
    for kw in routes:
        p = SettlerParams(underflow_concentration=200.0, rho_p=RHO_S, liquid_density=RHO_L,
                          n_grid=60, **kw)
        over, under, info = Settler(p)(feed)
        assert np.isfinite(float(info["area"])), kw
        for k in ("F_solid", "F_H2O"):
            assert float(over[k] + under[k]) == pytest.approx(feed[k], rel=1e-8)


def test_settler_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        SettlerParams(underflow_concentration=200.0, C_data=[80.0, 60.0, 40.0],
                      v_data=[1e-5, 5e-5, 2e-4])
    with pytest.raises(ValueError):
        SettlerParams(underflow_concentration=100.0, method="foo")


def test_cyclone_smoke_and_grad():
    p = CycloneParams(D=1.0, rho_p=2000.0, d_bins=D_BINS, mass_fractions=W_BINS)
    gas = dusty_gas()
    cl, so, info = Cyclone(p)(gas)
    for k in ("F_air", "F_solid"):
        assert float(cl[k] + so[k]) == pytest.approx(gas[k], rel=1e-10)
    assert 0.0 < float(info["overall_efficiency"]) < 1.0

    def eff(D):
        return Cyclone(CycloneParams(D=D, rho_p=2000.0, d_bins=D_BINS,
                                     mass_fractions=W_BINS))(gas)[2]["overall_efficiency"]
    assert np.isfinite(float(jax.grad(eff)(1.0)))
    with pytest.raises(ValueError):
        CycloneParams(D=1.0, rho_p=2000.0, d_bins=D_BINS, mass_fractions=W_BINS, geometry="foo")


def test_cyclone_in_a_flowsheet():
    from difflow.flowsheet import Flowsheet, Unit

    fs = Flowsheet(species_order=["air", "solid"])
    fs.add_feed("gas", {"F_air": 35.0, "F_solid": 0.1, "T": 300.0, "P": 101325.0})
    fs.add_unit(Unit("cyc", Cyclone(CycloneParams(D=1.0, rho_p=2000.0, d_bins=D_BINS,
                                                  mass_fractions=W_BINS)),
                     ["gas"], ["clean", "dust"]))
    out = fs.solve()
    assert float(out["clean"]["F_solid"] + out["dust"]["F_solid"]) == pytest.approx(0.1, rel=1e-10)
