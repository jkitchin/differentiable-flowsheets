"""Smoke tests for the hydroprocessing building blocks and the hydrotreater (#306).

The full hydrotreater solve compiles for about a minute, so it is not run
here; one bed and one flash on the real diesel layout cover the reactor path.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydroprocessing.reactor import integrate_bed
from difflow_refinery.hydroprocessing.separator import flash_components, pr_flash, rachford_rice
from difflow_refinery.hydroprocessing.solve import newton_solve
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut
from difflow_refinery.hydrotreating.feed import select_cuts

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]


def test_layout_requires_carbon_and_hydrogen():
    with pytest.raises(ValueError, match="required"):
        Layout(("hydrogen",), ("pc01",), ("S",), ("S",))


def test_rachford_rice_negative_flash():
    z = jnp.asarray([0.6, 0.3, 0.1])
    K = jnp.asarray([8.0, 3.0, 0.9])
    V = rachford_rice(z, K)
    assert float(jnp.sum(z * (K - 1) / (1 + V * (K - 1)))) == pytest.approx(0.0, abs=1e-12)
    assert float(V) > 1.0


def test_newton_solve_grad_finite():
    f = lambda x, a: x**3 + a * x - 3.0
    g = jax.grad(lambda a: newton_solve(f, jnp.asarray([1.0]), a).value[0])(2.0)
    assert np.isfinite(float(g))


def test_fractionator_shares_telescope_to_one():
    from difflow_refinery.hydrotreating.fractionator import split_shares
    s = split_shares(jnp.linspace(300.0, 700.0, 41), jnp.asarray([420.0, 520.0, 600.0]), 8.0)
    assert s.shape == (41, 4)
    np.testing.assert_allclose(np.asarray(s.sum(axis=1)), 1.0, rtol=0, atol=1e-15)


@pytest.fixture(scope="module")
def diesel_unit():
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, 230 + 273.15, 370 + 273.15))
    return feed, Hydrotreater(char, feed, HydrotreaterParams())


def test_diesel_flash_and_bed_smoke(diesel_unit):
    feed, unit = diesel_unit
    th = unit.theta(feed)
    comps, kin, lay = unit._comps(th), unit._kin_params(th["kin"]), unit.layout
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(700.0),
                jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    r = pr_flash(323.15, 47e5, flash_components(th["oil"] + gas, lay, comps), comps)
    assert float(r.residual) < 1e-10 and 0.0 < float(r.beta) < 1.0
    b = integrate_bed(unit.kinetics, kin, lay, comps, th["oil"] + gas, 613.15, 50e5, 67000.0)
    assert np.isfinite(float(b.T_out)) and float(b.T_out) > 613.15   # exothermic


def test_compressor_with_an_empty_loop(diesel_unit):
    # purge = 1 leaves no recycle gas; the tear Newton's derivative through the
    # compressor was then a non-finite linear solve.
    from difflow_refinery.hydroprocessing.recycle import compress
    feed, unit = diesel_unit
    lay, comps = unit.layout, unit._comps(unit.theta(feed))

    def f(s):
        gas = Flows(s * jnp.ones(lay.n_gas), jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        return jnp.stack(compress(gas, lay, comps, 323.15, 45e5, 50e5, 0.75))

    assert float(f(0.0)[1]) == 0.0
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(f)(0.0))))
