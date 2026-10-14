"""Tests for the clip_negative_flows option on Flowsheet.solve (issue #164).

The accelerated tear solvers (Wegstein, Anderson) historically clipped
the packed tear vector at zero on every iteration. That is a safeguard
for molar flows in chemical flowsheets, but flowsheets with signed
flows (e.g. bidirectional flows in gas networks) can have a legitimate
negative fixed point, which the clip makes unreachable. ``solve`` now
exposes ``clip_negative_flows`` (default True, the historical
behavior), and the clip only ever touches flow entries, never T or P.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow.flowsheet import ConvergenceWarning, Flowsheet, Unit
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
        tear_initial=GUESS, tol=1e-10, max_iter=50, acceleration=acceleration,
        # The clip is what makes the fixed point unreachable, so this solve
        # never converges by construction (#249); the warning is expected
        # here and would only be noise.
        on_nonconvergence="ignore",
    )
    assert fs.last_solve_converged is False
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


# ---------------------------------------------------------------------------
# Which solver paths honour the flag, and why (issue #263)
# ---------------------------------------------------------------------------
#
# ``clip_negative_flows`` binds on the accelerated paths only. #263 asked
# whether that asymmetry should be closed by clipping everywhere or nowhere;
# the answer recorded here is neither. The projection safeguards an
# *extrapolated* guess, and the unaccelerated path extrapolates nothing, so
# clipping there would clip ``g`` itself -- which invents fixed points and,
# because that path is the one ``jax.grad`` runs on, corrupts gradients.
# ``test_clipping_inside_the_map_would_break_the_gradient`` measures the
# second cost rather than asserting it.


class ClippedMapFlowsheet(Flowsheet):
    """What "clip on the unaccelerated path too" would actually mean.

    The projection moves from the proposed iterate onto the packed tear
    vector itself, which is where the unaccelerated path would have to put
    it: that path has no proposal of its own to project.
    """

    def _streams_to_array(self, streams):
        arr = super()._streams_to_array(streams)
        mask = self._tear_flow_mask(len(streams))
        return jnp.where(mask, jnp.maximum(arr, 0.0), arr)


def _trace_species_loop(cls, p):
    """A recycle whose second species converges to exactly zero at ``p = 0``.

    ``F_A`` settles at 2 and ``F_B`` at ``2 p``. Nothing is contrived about
    the zero: a species absent from a recycle is an everyday flowsheet, and
    ``p`` is the rate at which it starts to appear.
    """
    def loop(feed, tear):
        return make_stream(
            {"A": feed["F_A"] + 0.5 * tear["F_A"],
             "B": p + 0.5 * tear["F_B"]},
            feed["T"], feed["P"],
        )

    fs = cls(["A", "B"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", loop, ["feed", "tear"], ["out"]))
    fs.add_recycle("out", "tear")
    return fs


def _trace_species_outlet(cls):
    def f(p):
        fs = _trace_species_loop(cls, p)
        streams = fs.solve(tol=1e-12, max_iter=200, acceleration="none",
                           use_initialization=False)
        return streams["out"]["F_B"]
    return f


def test_clipping_inside_the_map_would_break_the_gradient():
    """Why the unaccelerated path does not clip (#263).

    ``F_B`` converges to exactly zero, so a clip on that path would sit on
    its own kink, where ``jnp.maximum(x, 0)`` has a derivative of one half.
    The forward answer is right either way and only the gradient is wrong,
    which is the kind of failure nobody notices.
    """
    exact = 2.0  # d/dp of 2 p

    shipped = jax.grad(_trace_species_outlet(Flowsheet))(0.0)
    assert float(shipped) == pytest.approx(exact, rel=1e-6)

    clipped = jax.grad(_trace_species_outlet(ClippedMapFlowsheet))(0.0)
    assert float(clipped) != pytest.approx(exact, rel=1e-3)
    assert float(clipped) == pytest.approx(4.0 / 3.0, rel=1e-6)


def test_the_unaccelerated_path_ignores_the_flag():
    """Pins the decision, so the asymmetry is on the record and not a bug.

    Both settings give the same answer on the same map, including the
    negative fixed point that the accelerated paths cannot reach by
    default.
    """
    results = []
    for clip in (True, False):
        fs = _recycle_flowsheet(offset=-2.0)
        streams = fs.solve(tear_initial=GUESS, tol=1e-10, max_iter=200,
                           acceleration="none", clip_negative_flows=clip)
        assert fs.last_solve_clip_active == 0
        results.append(float(streams["loop_out"]["F_A"]))

    assert results[0] == pytest.approx(-4.0, abs=1e-8)
    assert results[0] == results[1]


@pytest.mark.parametrize("acceleration", ["wegstein", "anderson"])
def test_a_clipped_signed_tear_says_so_when_it_fails(acceleration):
    """The failure this flag causes used to be silent (#263)."""
    fs = _recycle_flowsheet(offset=-2.0)
    with pytest.warns(ConvergenceWarning) as record:
        fs.solve(tear_initial=GUESS, tol=1e-10, max_iter=50,
                 acceleration=acceleration)

    assert fs.last_solve_converged is False
    assert fs.last_solve_clip_active > 0
    message = str(record[0].message)
    assert "clip_negative_flows=False" in message
    assert "negative-flow clip" in message


@pytest.mark.parametrize("acceleration", ["wegstein", "anderson"])
def test_an_ordinary_flowsheet_never_reports_the_clip(acceleration):
    """A counter that fired on every solve would say nothing."""
    fs = _recycle_flowsheet(offset=2.0)
    fs.solve(tear_initial=GUESS, tol=1e-10, max_iter=200,
             acceleration=acceleration)
    assert fs.last_solve_converged is True
    assert fs.last_solve_clip_active == 0
