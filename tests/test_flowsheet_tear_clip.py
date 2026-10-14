"""The clip_negative_flows option on Flowsheet.solve (issues #164, #263)."""

import jax.numpy as jnp
import pytest

from difflow.flowsheet import ConvergenceWarning, Flowsheet, Unit
from difflow.streams import make_stream


class AffineRecycle:
    """out_flow = offset + 0.5 * in_flow; fixed point 2 * offset."""

    def __init__(self, offset: float):
        self.offset = offset

    def __call__(self, inlet):
        return make_stream({"A": self.offset + 0.5 * inlet["F_A"]},
                           inlet["T"], inlet["P"])


def _recycle_flowsheet(offset: float) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", AffineRecycle(offset), ["tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 1.0}, 300.0, 1e5)}


def test_negative_fixed_point_reachable_without_clip():
    fs = _recycle_flowsheet(offset=-2.0)
    fs.solve(tear_initial=GUESS, tol=1e-10, max_iter=200,
             acceleration="anderson", clip_negative_flows=False)
    assert fs.last_solve_converged is True


def test_default_clip_warns_and_names_the_flag():
    fs = _recycle_flowsheet(offset=-2.0)
    with pytest.warns(ConvergenceWarning, match="clip_negative_flows=False"):
        fs.solve(tear_initial=GUESS, tol=1e-10, max_iter=50, acceleration="anderson")
    assert fs.last_solve_clip_active > 0


def test_clip_touches_only_flows():
    fs = Flowsheet(species_order=["A"])
    clipped = fs._clip_flows(jnp.array([-1.0, -5.0, -7.0]), fs._tear_flow_mask(1))
    assert list(map(float, clipped)) == [0.0, -5.0, -7.0]
