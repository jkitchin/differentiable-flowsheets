"""Hydrocracker (#307): smoke tests of the kinetics, solver pieces and one cracking bed.

The whole-unit solve compiles for minutes, so it is not run here; one
pretreat bed plus one cracking bed on the real VGO layout cover the reactor path.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import difflow_refinery as dr
from difflow_refinery.hydrocracking import kinetics as hck
from difflow_refinery.hydrocracking.fixed_point import fixed_point, gmres
from difflow_refinery.hydrocracking.unit import Hydrocracker, HydrocrackerParams, bmci
from difflow_refinery.hydroprocessing.layout import Flows
from difflow_refinery.hydroprocessing.reactor import integrate_bed
from difflow_refinery.hydrotreating.feed import select_cuts, straight_run_cut

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]
VGO_RANGE = (370 + 273.15, 560 + 273.15)


def test_correlations_finite():
    pr = hck.product_properties(jnp.asarray([400.0, 500.0, 600.0]), 12.1)
    assert all(np.all(np.isfinite(np.asarray(pr[k]))) for k in ("SG", "MW", "H", "n_C"))
    assert np.isfinite(float(hck.yield_distribution(0.5, 1.0, 0.2, 0.02)))
    assert np.isfinite(float(bmci(353.24, 0.8844)))


def test_gmres_solves_a_nonsymmetric_system():
    rng = np.random.default_rng(0)
    M = jnp.asarray(np.eye(60) - 0.08 * rng.standard_normal((60, 60)))
    b = jnp.asarray(rng.standard_normal(60))
    x = gmres(lambda v: M @ v, b, k=20, restarts=3)
    np.testing.assert_allclose(np.asarray(M @ x), np.asarray(b), atol=1e-10)


def test_fixed_point_grad_finite():
    A = jnp.asarray([[0.3, 0.1], [0.05, 0.2]])

    def G(z, a):
        return a + A @ jnp.tanh(z)

    J = jax.grad(lambda a: jnp.sum(fixed_point(G, jnp.zeros(2), a, scale=jnp.ones(2),
                                               tol=1e-14).value ** 2))(jnp.asarray([0.5, -0.2]))
    assert np.all(np.isfinite(np.asarray(J)))


def test_pretreat_and_cracking_bed_smoke():
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, *VGO_RANGE))
    unit = Hydrocracker(char, feed, HydrocrackerParams())
    th = unit.theta(feed)
    comps, cst, lay, oil = unit._comps(th), unit._crack_state(th), unit.layout, th["oil"]
    Q = float(unit._feed_volume(th))
    h2 = 1200.0 * Q * 44.615
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(h2),
                jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    W = Q * 3600 / 1.5 * 800
    r = unit.pretreat_reactor(oil, gas.scale(0.5), [648.15], 340.0, 150e5, [0.5 * W, 0.5 * W],
                              comps, th["pre"], quench=(0.12,))
    assert np.all(np.isfinite(np.asarray(r.outlet.attr)))
    b = integrate_bed(unit.kinetics, cst, lay, comps, r.outlet + gas.scale(0.1), 643.15, 150e5, 0.1 * W)
    assert np.isfinite(float(b.T_out)) and float(b.T_out) > 643.15
    e_in, e_out = (r.outlet + gas.scale(0.1)).elements(lay), b.outlet.elements(lay)
    np.testing.assert_allclose(np.asarray(e_out), np.asarray(e_in), rtol=1e-10)


def test_makeup_is_data():
    # the makeup composition is a theta entry, not a shape: a hydrogen loop
    # (hydrogen.close_hydrotreater_loop) re-solves the unit at a new purity
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, *VGO_RANGE))
    unit = Hydrocracker(char, feed, HydrocrackerParams())
    p = dataclasses.replace(unit.params, makeup={"hydrogen": 0.999, "methane": 0.001})
    unit._jit = lambda th: th                     # check the guard and theta, not the solve
    th = unit.solve(feed, params=p, warn=False)
    assert float(th["makeup_y"][unit.layout.gas_index("hydrogen")]) == 0.999
