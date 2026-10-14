"""The saturated gas plant (#312): smoke, validation and API tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.gasplant import (
    AmineTreater,
    AmineTreaterParams,
    GasCompressor,
    GasCompressorParams,
    GasPlantColumn,
    GasPlantColumnParams,
    absorber_deethanizer,
    debutanizer,
    gas_components,
    lpg_quality,
)

jax.config.update("jax_enable_x64", True)

SMALL = ["propane", "isobutane", "n_butane", "isopentane", "n_pentane", "n_hexane"]
SMALL_FEED = {"F_propane": 10.0, "F_isobutane": 8.0, "F_n_butane": 15.0, "F_isopentane": 12.0,
              "F_n_pentane": 15.0, "F_n_hexane": 40.0, "T": 370.0, "P": 12e5}


def _flows(stream, names):
    return np.array([float(stream.get(f"F_{n}", 0.0)) for n in names])


@pytest.fixture(scope="module")
def small():
    c = gas_components(SMALL)
    col = GasPlantColumn(debutanizer(c, n_trays=20, feed_tray=10, naphtha_rvp=80e3))
    return c, col


def test_debutanizer_smoke_and_grad(small):
    """One compile: the column solves, closes its balance, and has a finite gradient."""
    c, col = small
    th0 = col.theta(SMALL_FEED)

    def duty(T):
        res = col.solve_theta(dict(th0, feeds={"feed": dict(th0["feeds"]["feed"], T=T)}))
        return res["outputs"]["reboiler.duty"], res

    (q, res), g = jax.value_and_grad(duty, has_aux=True)(jnp.asarray(370.0))
    assert bool(res["converged"])
    assert np.isfinite(float(q)) and np.isfinite(float(g))
    assert abs(float(res["outputs"]["energy_balance"])) < 1e-6 * float(q)


def test_feed_validation(small):
    col = small[1]
    with pytest.raises(ValueError, match="not components"):
        col.theta(dict(SMALL_FEED, F_methane=1.0))
    with pytest.raises(ValueError, match="expected 1 feed"):
        col.theta(SMALL_FEED, SMALL_FEED)


def test_params_validation():
    c = gas_components(SMALL)
    with pytest.raises(ValueError, match="reboiler.duty has no value"):
        GasPlantColumn(GasPlantColumnParams(components=c, n_trays=5, feed_trays={"feed": 3},
                                            draw_rates={"distillate": 1.0}))
    with pytest.raises(ValueError, match="keys"):
        GasPlantColumn(GasPlantColumnParams(components=c, n_trays=5, feed_trays={"feed": 3},
                                            reboiler_duty=1e6, draw_rates={"distillate": 1.0},
                                            tray_efficiency=None))
    with pytest.raises(ValueError, match="grade"):
        lpg_quality(jnp.ones(c.n), c, "HD-6")


def test_factories_check_their_specs():
    c = gas_components(["methane", "ethane", "propane", "n_butane"])
    with pytest.raises(ValueError, match="exactly one"):
        absorber_deethanizer(c, c2_in_bottoms=0.01, bottoms_T=400.0)
    with pytest.raises(ValueError, match="naphtha_rvp or c4"):
        debutanizer(c, c4_in_naphtha=None)


def test_registered():
    names = []

    class Reg:
        def register(self, name, **kw):
            names.append(name)

    dr.register(Reg())
    assert {"GasPlantColumn", "WetGasCompressor", "AmineTreater"} <= set(names)


def test_amine_smoke():
    sweet, acid, _ = AmineTreater(AmineTreaterParams())(
        {"F_methane": 10.0, "F_hydrogen_sulfide": 0.5, "T": 310.0, "P": 14e5})
    assert float(sweet["F_hydrogen_sulfide"] + acid["F_hydrogen_sulfide"]) == pytest.approx(0.5)


def test_compressor_smoke():
    c = gas_components(["hydrogen", "methane", "ethane", "propane", "n_butane", "n_pentane"])
    comp = GasCompressor(GasCompressorParams(c, outlet_P=14e5, n_stages=2))
    gas, cond, info = comp({"F_hydrogen": 5.0, "F_methane": 30.0, "F_ethane": 15.0,
                            "F_propane": 15.0, "F_n_butane": 10.0, "F_n_pentane": 5.0,
                            "T": 313.15, "P": 1.5e5})
    assert np.isfinite(float(info["power"])) and float(info["power"]) > 0
    assert float(gas["P"]) == pytest.approx(14e5)


def test_solve_T_grad_is_not_zero():
    """Regression: the Newton slope was grad(stop_gradient(fn)) -- zero."""
    from difflow_refinery.gasplant.units import _solve_T

    fn = lambda T, a: a * T ** 3 + jnp.log(T)  # noqa: E731
    target = 2.0 * 350.0 ** 3 + np.log(350.0)
    assert float(_solve_T(lambda t: fn(t, 2.0), target, 300.0)) == pytest.approx(350.0, rel=1e-13)
    g = jax.grad(lambda a: _solve_T(lambda t: fn(t, a), target, 300.0))(2.0)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_log_mmatrix_solve_resolves_tiny_flows():
    from difflow_refinery.gasplant.column import _log_mmatrix_solve

    N = 12
    S = np.eye(N)[None].copy()
    S[0, np.arange(N - 1), np.arange(1, N)] = -np.exp(-100.0)
    fb = np.zeros((1, N))
    fb[0, -1] = 1.0
    got = np.asarray(_log_mmatrix_solve(jnp.asarray(S), jnp.asarray(fb)))[0]
    np.testing.assert_allclose(got, -100.0 * np.arange(N - 1, -1, -1), atol=1e-9)
