"""Traced Anderson/Wegstein: a requested acceleration survives jit/grad (#440)."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream

GUESS = {"tear": make_stream({"A": 0.0}, 300.0, 1e5)}


class Slow:
    """``g(x) = F + k x / (1 + k)``, ``x* = F (1 + k)``."""

    def __init__(self, k):
        self.k = k

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + self.k * tear["F_A"] / (1.0 + self.k)},
            feed["T"], feed["P"],
        )


def _solve_at(op, F, box=None, **kw):
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": F}, 300.0, 1e5))
    fs.add_unit(Unit("loop", op, ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    if box is not None:
        box["fs"] = fs
    return fs.solve(tol=1e-12, tear_initial=GUESS, error_probe=0, **kw)["loop_out"]["F_A"]


def test_traced_acceleration_is_used_and_solves():
    box = {}
    out = jax.jit(lambda F: _solve_at(Slow(9.0), F, box, acceleration="anderson"))(1.0)
    assert float(out) == pytest.approx(10.0, rel=1e-8)
    assert box["fs"].last_solve_method == "fixed_point (traced)"
    assert box["fs"].last_solve_acceleration == "anderson"


def test_grad_through_closed_over_tracer_is_finite():
    g = jax.grad(lambda k: _solve_at(Slow(k), 1.0, acceleration="anderson"))(9.0)
    assert jax.numpy.isfinite(g)
