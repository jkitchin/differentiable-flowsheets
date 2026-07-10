"""Tests for the clip_negative_flows option on Flowsheet.solve (issue #164).

The accelerated tear solvers (Wegstein, Anderson) historically clipped
the packed tear vector at zero on every iteration. That is a safeguard
for molar flows in chemical flowsheets, but flowsheets with signed
flows (e.g. bidirectional flows in gas networks) can have a legitimate
negative fixed point, which the clip makes unreachable. ``solve`` now
exposes ``clip_negative_flows`` (default True, the historical
behavior), and the clip only ever touches flow entries, never T or P.
"""

import jax.numpy as jnp
import pytest

from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream


class AffineRecycle:
    """out_flow = offset + slope * in_flow, at the inlet's T and P.

    With |slope| < 1 the tear map is contractive with fixed point
    offset / (1 - slope), which is negative for negative offset.
    """

    def __init__(self, offset: float, slope: float):
        self.offset = offset
        self.slope = slope

    def __call__(self, inlet):
        return make_stream(
            {"A": self.offset + self.slope * inlet["F_A"]},
            inlet["T"],
            inlet["P"],
        )


def _recycle_flowsheet(offset: float, slope: float = 0.5) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(
        Unit("loop", AffineRecycle(offset, slope), ["tear"], ["loop_out"])
    )
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 1.0}, 300.0, 1e5)}

# Wegstein's accelerated step previously could not converge these
# contractive maps because its update formula was inconsistent with its
# q = s/(s-1) factor (issue #166). That formula is now fixed
# (wegstein_acceleration uses x_new = q*x + (1-q)*g), so the Wegstein
# cases converge and are no longer expected failures.


@pytest.mark.parametrize("acceleration", ["wegstein", "anderson"])
def test_negative_fixed_point_reachable_without_clip(acceleration):
    """clip_negative_flows=False converges to a negative tear flow."""
    fs = _recycle_flowsheet(offset=-2.0)  # fixed point: -4.0
    streams = fs.solve(
        tear_initial=GUESS,
        tol=1e-10,
        max_iter=200,
        acceleration=acceleration,
        clip_negative_flows=False,
    )
    assert float(streams["loop_out"]["F_A"]) == pytest.approx(-4.0, abs=1e-8)


@pytest.mark.parametrize("acceleration", ["wegstein", "anderson"])
def test_default_clip_blocks_negative_fixed_point(acceleration):
    """The historical default cannot reach a negative fixed point.

    This pins the default behavior: the iterate is projected back to
    zero each step, so the returned tear is the clipped value, not the
    fixed point. If this test ever fails because the default changed,
    that is a deliberate API decision, not an accident.
    """
    fs = _recycle_flowsheet(offset=-2.0)
    streams = fs.solve(
        tear_initial=GUESS, tol=1e-10, max_iter=50, acceleration=acceleration
    )
    assert float(streams["loop_out"]["F_A"]) != pytest.approx(-4.0, abs=1e-3)


@pytest.mark.parametrize("acceleration", ["wegstein", "anderson"])
@pytest.mark.parametrize("clip", [True, False])
def test_positive_fixed_point_unaffected_by_option(acceleration, clip):
    """For ordinary non-negative flows the option changes nothing."""
    fs = _recycle_flowsheet(offset=2.0)  # fixed point: +4.0
    streams = fs.solve(
        tear_initial=GUESS,
        tol=1e-10,
        max_iter=200,
        acceleration=acceleration,
        clip_negative_flows=clip,
    )
    assert float(streams["loop_out"]["F_A"]) == pytest.approx(4.0, abs=1e-8)


def test_flow_mask_marks_only_flows():
    fs = Flowsheet(species_order=["A", "B"])
    mask = fs._tear_flow_mask(2)
    # per stream: [F_A, F_B, T, P]
    expected = jnp.array([True, True, False, False] * 2)
    assert bool(jnp.all(mask == expected))


def test_clip_leaves_temperature_and_pressure_alone():
    fs = Flowsheet(species_order=["A"])
    mask = fs._tear_flow_mask(1)
    packed = jnp.array([-1.0, -5.0, -7.0])  # F_A, T, P (nonphysical T, P)
    clipped = fs._clip_flows(packed, mask)
    assert clipped[0] == 0.0        # flow clipped
    assert clipped[1] == -5.0       # T untouched
    assert clipped[2] == -7.0       # P untouched
