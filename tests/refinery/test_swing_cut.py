"""Contract tests for the swing-cut crude unit block (#531). Fast: no column
solves here -- the CrudeUnit generator is checked by hand in
tests/reference/swing_cut_check.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.swing_cut import SwingCutTable, cut_table_from_assays, swing_cut_block

jax.config.update("jax_enable_x64", True)


def _table():
    return SwingCutTable(
        cuts=("a", "s", "b"), products=("A", "B"), own={"A": "a", "B": "b"},
        swings=(("A", "s"), ("B", "s")), crudes=("c1", "c2"),
        yields=np.array([[0.4, 0.2, 0.4], [0.3, 0.1, 0.6]]),
        props={"SG": np.array([[0.7, 0.8, 0.9], [0.72, 0.82, 0.95]]),
               "S": np.array([[0.1, 0.5, 2.0], [0.2, 0.6, 3.0]])})


def test_block_smoke():
    t = _table()
    F, q = swing_cut_block(t, jnp.array([10.0, 5.0]), jnp.array([0.4, 0.6]))
    assert float(jnp.sum(F)) == pytest.approx(15.0)
    assert set(q) == {"SG", "S"} and all(np.all(np.isfinite(np.asarray(v))) for v in q.values())


def test_block_grad_finite():
    t = _table()
    g = jax.grad(lambda ph: swing_cut_block(t, jnp.array([10.0, 5.0]), ph, varying=True)[1]["S"][0])(
        jnp.array([0.4, 0.6]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_varying_matches_constant_at_ends():
    t = _table()
    for ph in ([1.0, 0.0], [0.0, 1.0]):
        _, a = swing_cut_block(t, jnp.array([10.0, 5.0]), jnp.array(ph))
        _, b = swing_cut_block(t, jnp.array([10.0, 5.0]), jnp.array(ph), varying=True)
        for k in a:
            np.testing.assert_allclose(np.asarray(a[k]), np.asarray(b[k]), atol=1e-12)


def test_bad_structure_raises():
    with pytest.raises(ValueError, match="unknown product or cut"):
        SwingCutTable(("a",), ("A",), {"A": "x"}, (), ("c",), np.ones((1, 1)),
                      {"SG": np.ones((1, 1))})


def test_varying_unknown_swing_raises():
    with pytest.raises(ValueError, match="not swing cuts"):
        swing_cut_block(_table(), jnp.array([1.0, 1.0]), jnp.array([0.5, 0.5]), varying=["a"])


def test_cut_points_length_raises():
    from difflow_refinery import Assay
    a = Assay([0, 50, 100], [300.0, 500.0, 800.0], sg=0.85)
    with pytest.raises(ValueError, match="cut points"):
        cut_table_from_assays([a], [400.0], ("a", "s", "b"), ("A", "B"),
                              {"A": "a", "B": "b"}, (("A", "s"), ("B", "s")))


def test_cut_table_from_assay_smoke():
    from difflow_refinery import Assay
    a = Assay([0, 50, 100], [300.0, 500.0, 800.0], sg=0.85, sulfur_wt=1.0)
    t = cut_table_from_assays([a], [430.0, 450.0], ("a", "s", "b"), ("A", "B"),
                              {"A": "a", "B": "b"}, (("A", "s"), ("B", "s")))
    assert t.yields.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(t.props["SG"])) and np.all(np.isfinite(t.props["S_wt"]))
