"""``tol`` tests a step, and the answer is somewhere else (issue #264).

``Flowsheet.solve(tol=...)`` tests the max-norm of ``g(x) - x``, the step
between successive tear iterates. The error in the answer is
``(I - M)^-1 (g(x) - x)``, and on a loop of gain ``g`` those differ by
``1/(1 - g)``: at 0.97 a solve that stops inside ``1e-8`` is 3e-7 out and
reports convergence, meaning it.

``solve`` now measures that error instead of leaving the caller to guess at
it: ``last_solve_gain`` and ``last_solve_error_estimate`` after every
converged solve, a :class:`TearToleranceWarning` when the step test was met
and the tolerance was not, and ``tol_basis="error"`` to act on it.
"""

import warnings

import jax
import pytest

from difflow.flowsheet import Flowsheet, TearToleranceWarning, Unit
from difflow.streams import make_stream


class AffineLoop:
    """``out = offset + slope * in``: fixed point ``offset / (1 - slope)``.

    ``slope`` is the loop gain, so the error a tear tolerance leaves behind
    is ``1 / (1 - slope)`` times the step -- known in closed form, which is
    what makes this the right instrument for testing an estimate of it.
    """

    def __init__(self, offset: float, slope: float):
        self.offset, self.slope = offset, slope

    def __call__(self, inlet):
        return make_stream(
            {"A": self.offset + self.slope * inlet["F_A"]},
            inlet["T"], inlet["P"],
        )


def _loop(slope: float, offset: float = 1.0) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", AffineLoop(offset, slope), ["tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 1.0}, 300.0, 1e5)}


def _solve(fs, **kwargs):
    kwargs.setdefault("tear_initial", GUESS)
    kwargs.setdefault("tol", 1e-8)
    kwargs.setdefault("max_iter", 400)
    kwargs.setdefault("acceleration", "wegstein")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TearToleranceWarning)
        return fs.solve(**kwargs)


class TestTheMeasurement:
    def test_the_gain_is_measured_not_assumed(self):
        fs = _loop(slope=0.97)
        _solve(fs)
        assert fs.last_solve_gain == pytest.approx(0.97, abs=1e-6)

    def test_the_error_estimate_matches_the_closed_form(self):
        """``error = residual / (1 - gain)``, against a fixed point of 33.3."""
        fs = _loop(slope=0.97)
        streams = _solve(fs)

        exact = 1.0 / (1.0 - 0.97)
        true_error = abs(float(streams["loop_out"]["F_A"]) - exact)
        assert fs.last_solve_error_estimate == pytest.approx(true_error, rel=0.05)
        # And it is the thing tol is not: three decades above the step.
        assert fs.last_solve_error_estimate > 30 * fs.last_solve_residual

    def test_an_oscillating_loop_is_closer_than_its_step_suggests(self):
        """A negative gain contracts the error; a norm-only ratio would not see it.

        ``1/(1 - gain)`` is 1/2 at ``gain = -1``, so the answer sits inside
        the step rather than outside it. An estimator built on magnitudes
        would report the error on the wrong side of the residual and warn
        about a solve that is better than it claims.
        """
        fs = _loop(slope=-0.9)
        _solve(fs, acceleration="none")
        assert fs.last_solve_gain < 0
        assert fs.last_solve_error_estimate < fs.last_solve_residual

    def test_a_residual_at_round_off_reports_no_gain(self):
        """Nothing to measure, and a ratio of noise lands near one.

        Near one is exactly where ``1/(1 - gain)`` blows up, so an
        estimator that did not stop here would invent its loudest alarms on
        its best solves.
        """
        fs = _loop(slope=0.2)
        _solve(fs, tol=1e-14, acceleration="anderson")
        assert fs.last_solve_residual < 1e-13
        assert fs.last_solve_gain is None
        assert fs.last_solve_error_estimate == 0.0

    def test_error_probe_zero_measures_nothing(self):
        fs = _loop(slope=0.97)
        _solve(fs, error_probe=0)
        assert fs.last_solve_gain is None
        assert fs.last_solve_error_estimate is None

    def test_the_probe_does_not_run_under_grad(self):
        """A traced solve has no concrete residual and no verdict to probe.

        The gradient has to come back exact: the probe is extra flowsheet
        passes taken after the fact, and if any of it leaked into the
        traced solve it would show up here.
        """
        def converged_flow(feed_flow):
            def loop(feed, tear):
                return make_stream({"A": feed["F_A"] + 0.5 * tear["F_A"]},
                                   feed["T"], feed["P"])

            fs = Flowsheet(["A"], default_flow=1.0)
            fs.add_feed("feed", make_stream({"A": feed_flow}, 300.0, 1e5))
            fs.add_unit(Unit("loop", loop, ["feed", "tear"], ["loop_out"]))
            fs.add_recycle("loop_out", "tear")
            streams = fs.solve(tear_initial=GUESS, tol=1e-12, max_iter=200)
            return streams["loop_out"]["F_A"]

        # d/d feed of feed / (1 - 0.5)
        assert float(jax.grad(converged_flow)(1.0)) == pytest.approx(2.0, rel=1e-6)


class TestTheWarning:
    def test_a_high_gain_solve_says_it_is_outside_its_tolerance(self):
        fs = _loop(slope=0.97)
        with pytest.warns(TearToleranceWarning) as record:
            fs.solve(tear_initial=GUESS, tol=1e-8, max_iter=400,
                     acceleration="wegstein")
        assert fs.last_solve_converged is True
        message = str(record[0].message)
        assert "step" in message
        assert 'tol_basis="error"' in message

    def test_an_ordinary_loop_is_quiet(self):
        """A warning that fires on every solve is read as noise.

        A gain of 0.5 doubles the step into the error, and the step test's
        own slack is of that order -- so this is not news, and saying it
        would spend the warning that the 0.97 case needs.
        """
        fs = _loop(slope=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", TearToleranceWarning)
            fs.solve(tear_initial=GUESS, tol=1e-8, max_iter=400,
                     acceleration="wegstein")

    def test_a_solve_that_did_not_converge_is_left_to_its_own_warning(self):
        fs = _loop(slope=0.999)
        with warnings.catch_warnings():
            warnings.simplefilter("error", TearToleranceWarning)
            fs.solve(tear_initial=GUESS, tol=1e-12, max_iter=3,
                     acceleration="none", on_nonconvergence="ignore")
        assert fs.last_solve_converged is False
        assert fs.last_solve_error_estimate is None


class TestToleranceBasis:
    def test_error_basis_reaches_the_tolerance_the_step_basis_reports(self):
        step = _loop(slope=0.97)
        _solve(step, tol=1e-8)

        error = _loop(slope=0.97)
        _solve(error, tol=1e-8, tol_basis="error")

        assert step.last_solve_error_estimate > 1e-8
        assert error.last_solve_error_estimate <= 1e-8
        # It is bought with iterations, which is why it is opt-in.
        assert error.last_solve_iterations > step.last_solve_iterations

    def test_error_basis_lands_closer_to_the_answer(self):
        exact = 1.0 / (1.0 - 0.97)
        out = {}
        for basis in ("step", "error"):
            fs = _loop(slope=0.97)
            streams = _solve(fs, tol=1e-8, tol_basis=basis)
            out[basis] = abs(float(streams["loop_out"]["F_A"]) - exact)
        assert out["error"] < out["step"] / 10

    def test_error_basis_costs_nothing_on_a_loop_that_is_already_accurate(self):
        plain = _loop(slope=0.3)
        _solve(plain, tol=1e-8)
        strict = _loop(slope=0.3)
        _solve(strict, tol=1e-8, tol_basis="error")
        assert strict.last_solve_iterations == plain.last_solve_iterations

    def test_error_basis_without_a_probe_is_refused(self):
        fs = _loop(slope=0.5)
        with pytest.raises(ValueError, match="error_probe"):
            fs.solve(tear_initial=GUESS, tol_basis="error", error_probe=0)

    def test_an_unknown_basis_is_refused(self):
        fs = _loop(slope=0.5)
        with pytest.raises(ValueError, match="tol_basis"):
            fs.solve(tear_initial=GUESS, tol_basis="absolute")


def test_a_flowsheet_without_recycles_reports_no_error():
    fs = Flowsheet(species_order=["A"])
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("pass", lambda s: s, ["feed"], ["out"]))
    fs.solve()
    assert fs.last_solve_gain is None
    assert fs.last_solve_error_estimate == 0.0
