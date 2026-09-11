"""Tests for on_nonconvergence on Flowsheet.solve (issue #249).

A recycle solve that ran out of iterations used to reach the same
``return`` as a converged one: the last iterate came back looking like any
other result, and the caller had to know to go read ``last_solve_converged``
afterwards. The material balance around the loop is off by the tear
residual, which for a limit-cycling loop is not small.

``solve`` now warns by default, and lets the caller escalate to an
exception or silence it. Under tracing the residual is not concrete, so
there is no verdict to report and the solve stays silent.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import ConvergenceError, ConvergenceWarning
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream

ACCELERATIONS = ["none", "wegstein", "anderson"]


class Recycled:
    """One unit closing a loop with the tear map ``x -> F + k sqrt(x + 0.1)``.

    Deliberately nonlinear: Anderson acceleration is exact on an affine
    map and would converge a linear loop within two iterations, which is
    no use for testing what happens when a solve runs out of them. This
    map converges under all three methods given a few hundred iterations
    and under none of them given three.

    T and P come from the feed rather than the tear on purpose. Passing
    the tear's own T straight through would make that row of the tear map
    the identity, and ``I - dg/dx`` -- the matrix the implicit-diff
    backward pass inverts -- exactly singular, which is a property of this
    toy rather than of anything being tested here.
    """

    def __init__(self, slope: float):
        self.slope = slope

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + self.slope * jnp.sqrt(tear["F_A"] + 0.1)},
            feed["T"],
            feed["P"],
        )


def _slow_loop(slope: float = 0.9) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", Recycled(slope), ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 0.0}, 300.0, 1e5)}


def _solve(fs, **kwargs):
    kwargs.setdefault("tear_initial", GUESS)
    kwargs.setdefault("tol", 1e-12)
    kwargs.setdefault("max_iter", 3)
    return fs.solve(**kwargs)


class TestNonConverged:
    """A solve that ran out of iterations is not silent."""

    @pytest.mark.parametrize("acceleration", ACCELERATIONS)
    def test_warns_by_default(self, acceleration):
        fs = _slow_loop()
        with pytest.warns(ConvergenceWarning):
            _solve(fs, acceleration=acceleration)
        assert fs.last_solve_converged is False

    @pytest.mark.parametrize("acceleration", ACCELERATIONS)
    def test_raise_escalates(self, acceleration):
        fs = _slow_loop()
        with pytest.raises(ConvergenceError):
            _solve(fs, acceleration=acceleration, on_nonconvergence="raise")

    @pytest.mark.parametrize("acceleration", ACCELERATIONS)
    def test_ignore_is_the_historical_behaviour(self, acceleration):
        """Silent, and still returns the last iterate."""
        fs = _slow_loop()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            streams = _solve(fs, acceleration=acceleration,
                             on_nonconvergence="ignore")
        assert "loop_out" in streams
        assert fs.last_solve_converged is False

    def test_message_names_what_is_needed_to_act(self):
        """Tear streams, residual, tolerance, iteration count, method."""
        fs = _slow_loop()
        with pytest.raises(ConvergenceError) as exc:
            _solve(fs, acceleration="anderson", max_iter=4,
                   on_nonconvergence="raise")
        message = str(exc.value)
        assert "tear" in message               # the tear stream's name
        assert "4" in message                  # the iteration cap
        assert "1.000e-12" in message          # the tolerance asked for
        assert "anderson" in message           # the method that ran
        assert f"{fs.last_solve_residual:.3e}" in message

    def test_diagnostics_survive_every_option(self):
        """last_solve_* is recorded the same way whatever is chosen."""
        fs = _slow_loop()
        with pytest.warns(ConvergenceWarning):
            _solve(fs, acceleration="anderson")
        warned = (fs.last_solve_residual, fs.last_solve_iterations,
                  fs.last_solve_tear_streams, fs.last_solve_tol)

        fs = _slow_loop()
        _solve(fs, acceleration="anderson", on_nonconvergence="ignore")
        ignored = (fs.last_solve_residual, fs.last_solve_iterations,
                   fs.last_solve_tear_streams, fs.last_solve_tol)

        assert warned == ignored

    def test_warning_points_at_the_callers_line(self):
        """stacklevel lands on the solve() call, not inside flowsheet.py."""
        fs = _slow_loop()
        with pytest.warns(ConvergenceWarning) as record:
            fs.solve(tear_initial=GUESS, tol=1e-12, max_iter=3)
        assert record[0].filename == __file__


class TestConverged:
    """A solve that met its tolerance says nothing."""

    @pytest.mark.parametrize("acceleration", ACCELERATIONS)
    def test_silent_when_converged(self, acceleration):
        fs = _slow_loop()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            streams = _solve(fs, acceleration=acceleration, tol=1e-8,
                             max_iter=500)
        assert fs.last_solve_converged is True
        # x = 1 + 0.9 sqrt(x + 0.1)
        x = float(streams["loop_out"]["F_A"])
        assert x == pytest.approx(1.0 + 0.9 * (x + 0.1) ** 0.5, abs=1e-6)

    def test_recycle_free_flowsheet_is_silent(self):
        fs = Flowsheet(species_order=["A"])
        fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
        fs.add_unit(
            Unit("pass", lambda s: make_stream({"A": s["F_A"]}, s["T"], s["P"]),
                 ["feed"], ["out"])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fs.solve()
        assert fs.last_solve_converged is True


class TestUnderTracing:
    """No concrete residual means no verdict -- and no guessing."""

    def test_grad_through_a_short_solve_is_silent(self):
        """The traced path routes to optimistix and cannot judge itself.

        ``max_iter`` here is far too small to converge, which is exactly
        the case that must not raise: a ConvergenceError inside a
        gradient would make the flowsheet undifferentiable.
        """
        def objective(feed_flow):
            fs = _slow_loop()
            fs.add_feed("feed", make_stream({"A": feed_flow}, 300.0, 1e5))
            streams = fs.solve(tear_initial=GUESS, tol=1e-12, max_iter=3,
                               on_nonconvergence="raise")
            return streams["loop_out"]["F_A"]

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            value = jax.grad(objective)(1.0)
        assert jnp.isfinite(value)

    def test_verdict_is_none_not_false_under_tracing(self):
        fs = _slow_loop()

        def objective(feed_flow):
            fs.add_feed("feed", make_stream({"A": feed_flow}, 300.0, 1e5))
            return fs.solve(tear_initial=GUESS, tol=1e-12,
                            max_iter=3)["loop_out"]["F_A"]

        jax.grad(objective)(1.0)
        assert fs.last_solve_converged is None


class TestValidation:
    def test_unknown_option_is_rejected(self):
        fs = _slow_loop()
        with pytest.raises(ValueError, match="on_nonconvergence"):
            _solve(fs, on_nonconvergence="warm")

    def test_rejected_before_any_solving_happens(self):
        """A typo must not cost a 100-iteration solve first."""
        fs = _slow_loop()
        with pytest.raises(ValueError):
            fs.solve(on_nonconvergence="shout")
        assert fs.last_solve_converged is None
