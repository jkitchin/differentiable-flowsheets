"""SWRO + pressure exchanger plant: the function and the Flowsheet, and the LCOW."""

import jax
import jax.numpy as jnp

import difflow_water as dw

FEED = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035)


def _ro(area=50.0):
    return dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=area, n_steps=10)


def test_lcow_runs_and_differentiates():
    def f(area, work):
        units = {"RO": dw.cost_ro(area, dw.CostingParams()),
                 "P1": dw.cost_hp_pump(work, dw.CostingParams())}
        return dw.lcow(units, work, 4e-4)["LCOW"]
    assert jnp.isfinite(f(50.0, 3e3))
    assert all(jnp.isfinite(g) for g in jax.grad(f, argnums=(0, 1))(50.0, 3e3))


def test_swro_performance_runs():
    r = dw.swro_performance(FEED, _ro(), 65e5)
    for k in ("LCOW", "SEC", "recovery", "salinity_permeate", "work_P1", "work_P2"):
        assert jnp.isfinite(r[k]), k


def test_swro_lcow_gradient_is_finite():
    g = jax.grad(lambda P, A: dw.swro_performance(FEED, _ro(A), P)["LCOW"],
                 argnums=(0, 1))(65e5, 50.0)
    assert all(jnp.isfinite(x) for x in g)


def test_swro_with_mixing_runs():
    r = dw.swro_performance(FEED, _ro(), 65e5, px=dw.PressureExchangerParams(mixing=0.03),
                            n_passes=3)
    assert jnp.isfinite(r["LCOW"])


def test_swro_flowsheet_wiring_solves():
    """API contract: the Flowsheet form solves and produces every named stream."""
    fs = dw.build_swro_flowsheet(_ro(), 65e5)
    fs.add_feed("seawater", FEED)
    out = fs.solve()
    for name in ("permeate", "brine", "ro_feed", "px_hp", "px_bypass"):
        assert jnp.isfinite(out[name]["F_H2O"]), name
