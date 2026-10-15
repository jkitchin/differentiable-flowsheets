"""``damping`` on Flowsheet.solve (it used to be silently ignored)."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import ConvergenceWarning
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream


class Overshoot:
    """``g(x) = 3 F - 2 x``: plain substitution diverges, damping < 2/3 converges."""

    def __call__(self, feed, tear):
        return make_stream({"A": 3.0 * feed["F_A"] - 2.0 * tear["F_A"]},
                           feed["T"], feed["P"])


class Scaled:
    """``g(x) = F + k x / (1 + k)`` with ``k`` in a closure."""

    def __init__(self, k):
        self.k = k

    def __call__(self, feed, tear):
        return make_stream({"A": feed["F_A"] + self.k * tear["F_A"] / (1.0 + self.k)},
                           feed["T"], feed["P"])


def _loop(operation) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", operation, ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 0.0}, 300.0, 1e5)}


def test_damping_is_read():
    fs = _loop(Overshoot())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fs.solve(acceleration="none", tear_initial=GUESS, damping=0.3,
                 tol=1e-10, max_iter=200)
    assert fs.last_solve_converged is True


def test_nonpositive_damping_is_refused():
    with pytest.raises(ValueError, match="damping must be positive"):
        _loop(Overshoot()).solve(damping=0.0)


def test_closure_tracer_gradient_is_finite():
    """Regression: a tracer hidden in a closure gave a NaN gradient."""
    def objective(k):
        return _loop(Scaled(k)).solve(acceleration="anderson", tol=1e-12,
                                      tear_initial=GUESS)["loop_out"]["F_A"]

    assert jnp.isfinite(jax.grad(objective)(2.0))
