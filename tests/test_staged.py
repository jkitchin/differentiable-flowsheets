"""Tests for :mod:`difflow.staged` (McCabe-Thiele stepper, pinches, Kremser):
smoke, API contracts, input validation and crash/NaN regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import staged as S

jax.config.update("jax_enable_x64", True)

AL, Z, XD, XB = 2.5, 0.4, 0.95, 0.05


def _mt(R=2.2, **kw):
    c = S.constant_alpha(AL)
    return S.mccabe_thiele(c, S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0),
                           XD, XB, q_line=S.q_line(1.0, Z), **kw)


def test_mccabe_thiele_smoke():
    res = _mt()
    for k in ("N", "feed_stage", "N_rect", "N_strip", "pinched", "capped"):
        assert np.isfinite(float(res[k])), k
    assert np.all(np.isfinite(np.asarray(res["x"])))


def test_pinch_and_kremser_helpers_smoke():
    c = S.constant_alpha(AL)
    vals = [
        S.minimum_reflux(c, Z, 1.0, XD),
        S.total_reflux_stages(c, XD, XB),
        S.minimum_stages(AL, XD, XB),
        S.minimum_solvent_ratio(S.linear_equilibrium(2.0), 0.1, 0.001, 0.0),
        S.kremser_stages(1.5, 0.99),
        S.kremser_fraction(1.0, 3.0),
    ]
    assert all(np.isfinite(float(v)) for v in vals)


def test_grad_N_and_rmin_finite():
    gN = jax.grad(lambda R: _mt(R)["N"])(2.2)
    gR = jax.grad(lambda a: S.minimum_reflux(S.constant_alpha(a), Z, 1.0, XD))(AL)
    gK = jax.grad(lambda A: S.kremser_stages(A, 0.9))(1.0)   # the A = 1 branch point
    assert np.isfinite(float(gN)) and np.isfinite(float(gR)) and np.isfinite(float(gK))


def test_bad_arguments():
    c = S.constant_alpha(AL)
    rect, strip = S.rectifying_line(2.0, XD), S.stripping_line(2.0, XD, XB, Z, 1.0)
    with pytest.raises(ValueError):
        S.mccabe_thiele(c, rect, strip, XD, XB)
    with pytest.raises(ValueError):
        S.mccabe_thiele(c, rect, None, XD, XB, fraction="nope")
    with pytest.raises(ValueError, match="exactly one"):
        S.mccabe_thiele(c, None, None, XD, XB, lines=[rect, strip])
    with pytest.raises(ValueError, match="switches"):
        S.mccabe_thiele(c, None, None, XD, XB, lines=[rect, strip], switch_x=[0.4, 0.3])


def test_sections_smoke():
    c = S.constant_alpha(AL)
    rect, strip = S.rectifying_line(2.0, XD), S.stripping_line(2.0, XD, XB, Z, 1.0)
    res = S.mccabe_thiele(c, None, None, XD, XB, lines=[rect, strip], switch_x=[Z])
    assert np.isfinite(float(res["N"]))


def test_below_minimum_reflux_does_not_return_nan():
    R = 0.97 * float(S.minimum_reflux(S.constant_alpha(AL), Z, 1.0, XD))
    res = _mt(R, n_max=60)
    assert not np.any(np.isnan(np.asarray(res["x"])))


def test_capped_warns_and_flag_survives_jit():
    with pytest.warns(RuntimeWarning, match="n_max"):
        assert bool(_mt(n_max=5)["capped"])
    assert bool(jax.jit(lambda: _mt(n_max=5)["capped"])())


def test_solve_monotone_nan_at_one_end_of_bracket():
    """Regression: a NaN at ``lo`` sent the root (and its gradient) to the wrong end."""
    root = lambda p: S._solve_monotone(  # noqa: E731
        lambda x: jnp.sqrt(x - 300.0) - jnp.sqrt(p - 300.0), 200.0, 674.0)
    assert float(root(350.0)) == pytest.approx(350.0, abs=1e-9)
    assert np.isfinite(float(jax.grad(root)(350.0)))


def test_reexports():
    import difflow_ree
    from difflow.units import lle

    assert lle.kremser_stages is S.kremser_stages
    assert difflow_ree.kremser_fraction is S.kremser_fraction


def test_plot_mccabe_thiele_runs():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from difflow.visualization import plot_mccabe_thiele

    c = S.constant_alpha(AL)
    top, bot, ql = S.rectifying_line(2.2, XD), S.stripping_line(2.2, XD, XB, Z, 1.0), S.q_line(1.0, Z)
    res = S.mccabe_thiele(c, top, bot, XD, XB, q_line=ql)
    assert len(plot_mccabe_thiele(res, c, top_line=top, bottom_line=bot, q_line=ql).lines) >= 5
