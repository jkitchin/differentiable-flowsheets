"""on_nonconvergence on Flowsheet.solve (issue #249)."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import ConvergenceError, ConvergenceWarning
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream


class Recycled:
    """Tear map ``x -> F + k sqrt(x + 0.1)``: converges slowly, never in 3 steps."""

    def __call__(self, feed, tear):
        return make_stream({"A": feed["F_A"] + 0.9 * jnp.sqrt(tear["F_A"] + 0.1)},
                           feed["T"], feed["P"])


def _slow_loop() -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", Recycled(), ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 0.0}, 300.0, 1e5)}


def _solve(fs, **kwargs):
    kwargs.setdefault("tear_initial", GUESS)
    kwargs.setdefault("tol", 1e-12)
    kwargs.setdefault("max_iter", 3)
    return fs.solve(**kwargs)


def test_nonconverged_warns_raises_or_ignores():
    fs = _slow_loop()
    with pytest.warns(ConvergenceWarning):
        _solve(fs)
    assert fs.last_solve_converged is False

    with pytest.raises(ConvergenceError):
        _solve(_slow_loop(), on_nonconvergence="raise")

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        streams = _solve(_slow_loop(), on_nonconvergence="ignore")
    assert "loop_out" in streams


def test_silent_when_converged():
    fs = _slow_loop()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        _solve(fs, tol=1e-8, max_iter=500)
    assert fs.last_solve_converged is True


def test_grad_through_short_solve_does_not_raise():
    def objective(feed_flow):
        fs = _slow_loop()
        fs.add_feed("feed", make_stream({"A": feed_flow}, 300.0, 1e5))
        return _solve(fs, on_nonconvergence="raise")["loop_out"]["F_A"]

    with pytest.warns(ConvergenceWarning):
        value = jax.grad(objective)(1.0)
    assert jnp.isfinite(value)


def test_unknown_option_is_rejected():
    with pytest.raises(ValueError, match="on_nonconvergence"):
        _solve(_slow_loop(), on_nonconvergence="warm")
