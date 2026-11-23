"""Swing-cut crude unit block (#531), checked by hand.

Run from the repository root::

    PYTHONPATH=src:. python -m pytest tests/reference/swing_cut_check.py -v

Not part of the suite: the CrudeUnit generator solves the column once per
swing plus once (~20 s), and the benchmark checks need the Du et al. data
(``du2025.fetch()``; skipped without it).

1. With the published ``y``, ``FQcut`` and ``phi``, the block reproduces both
   benchmark crude units' outlet flows and properties at the published
   solution.
2. The swing-fraction Jacobian matches central differences.
3. Generated from the rigorous CrudeUnit (the CDU tests' assay), cut yields
   sum to 1 and SG and sulfur rise from light to heavy cut.
4. Generated from an assay at TBP cut points, the Jacobian with respect to
   the cut points matches central differences.
5. Constant versus varying swing-cut properties: the product property error
   across each swing range is printed (the docs quote it).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay
from difflow_refinery import column as cc
from difflow_refinery.benchmarks import du2025
from difflow_refinery.swing_cut import (cut_table_from_assays, cut_table_from_crude_unit,
                                        swing_cut_block, swing_property_error)

jax.config.update("jax_enable_x64", True)
CASE = du2025.find_case(1)
needs_data = pytest.mark.skipif(CASE is None, reason="benchmark data not fetched")


@pytest.fixture(scope="module")
def bench():
    from difflow_refinery.benchmarks.du2025.crude import cut_table, published_rates
    data = du2025.load_case(CASE)
    path = CASE / "solution_baron.json"
    sol = du2025.load_solution(path if path.exists() else CASE / "solution_baron.gdx")
    out = {}
    for u in data.elements("UCDU"):
        t, phi = cut_table(data, u)
        out[u] = (t, phi, published_rates(t, u, sol))
    return data, sol, out


@needs_data
def test_reproduces_published_crude_units(bench):
    data, sol, units = bench
    SQ, FIX = set(data.sets["SQ"]), set(data.sets["FIX"])
    for u, (t, phi, r) in units.items():
        F, q = swing_cut_block(t, jnp.asarray(r), jnp.asarray(phi))
        for i, p in enumerate(t.products):
            assert float(F[i]) == pytest.approx(sol["FVO"].get((p, "1"), 0.0), abs=1e-9)
            for k, v in q.items():
                if (p, k) in SQ and (p, k) not in FIX and float(F[i]) > 1e-9:
                    assert float(v[i]) == pytest.approx(sol["FQ"][(p, k, "1")], rel=1e-9)


@needs_data
def test_phi_jacobian_fd(bench):
    _, _, units = bench
    for t, phi, r in units.values():
        f = lambda ph: jnp.concatenate([swing_cut_block(t, jnp.asarray(r), ph, varying=True)[0],
                                        swing_cut_block(t, jnp.asarray(r), ph, varying=True)[1][t.spg]])
        J = np.asarray(jax.jacfwd(f)(jnp.asarray(phi)))
        h = 1e-6
        for k in range(len(phi)):
            e = jnp.zeros(len(phi)).at[k].set(h)
            fd = (np.asarray(f(jnp.asarray(phi) + e)) - np.asarray(f(jnp.asarray(phi) - e))) / (2 * h)
            np.testing.assert_allclose(J[:, k], fd, rtol=1e-5, atol=1e-7)


@needs_data
def test_report_swing_property_error(bench):
    _, _, units = bench
    for u, (t, phi, r) in units.items():
        for k, (p, c) in enumerate(t.swings):
            _, e_sg = swing_property_error(t, jnp.asarray(r), jnp.asarray(phi), k, t.spg)
            _, e_s = swing_property_error(t, jnp.asarray(r), jnp.asarray(phi), k, "Q0")
            print(f"{u} swing {c} -> {p}: max |dSG| {np.abs(e_sg).max():.4f}, "
                  f"max |dS wt%| {np.abs(e_s).max():.4f}")


def test_crude_unit_generator():
    params = cc.CrudeColumnParams(n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5,
                                  P_bottom=1.6e5,
                                  specs=(cc.product_rate("naphtha", 0.004), cc.overflash(0.05)))
    unit = dr.CrudeUnit(Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85,
                              sulfur_wt=1.5, name="test"), params)
    t = cut_table_from_crude_unit(unit, 10_000.0, 480.0, 4e5, [("naphtha", "residue", 0.02)])
    assert t.yields.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(t.props["SG"][0]) > 0) and np.all(np.diff(t.props["S_wt"][0]) > 0)


def test_assay_generator_cut_point_jacobian():
    a = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85, sulfur_wt=1.5)
    cuts = ("n", "s", "k", "r")
    args = (cuts, ("N", "K", "R"), {"N": "n", "K": "k", "R": "r"}, (("N", "s"), ("K", "s")))
    cp = jnp.array([430.0, 450.0, 520.0])
    f = lambda c: cut_table_from_assays([a], c, *args, as_arrays=True)[0][0]
    J = np.asarray(jax.jacfwd(f)(cp))
    h = 1e-3
    for k in range(3):
        e = jnp.zeros(3).at[k].set(h)
        fd = (np.asarray(f(cp + e)) - np.asarray(f(cp - e))) / (2 * h)
        np.testing.assert_allclose(J[:, k], fd, rtol=1e-4, atol=1e-9)
