"""``compare_solvers`` used to report the SM solve as always converged.

The function predates #249, which gave ``Flowsheet`` its own verdict
(``last_solve_converged``, ``last_solve_iterations``, ``last_solve_residual``).
Before that there was nothing to read, so it hard-coded the dataclass default
``sm_converged=True`` and reported ``sm_iterations=max_iter`` with a comment
saying SM does not report iteration counts.

Both are now wrong rather than merely approximate: a diverged SM solve looked
like a converged one that happened to disagree with EO, which is exactly the
comparison the function exists to make.
"""

import warnings

import pytest

from difflow.benchmarks import compare_solvers
from difflow.flowsheet import ConvergenceWarning, Flowsheet, Unit
from difflow.streams import get_flows, make_stream


class Divergent:
    """``g(x) = 6F - 2x``: eigenvalue -2, so substitution never settles."""

    def __call__(self, feed, tear):
        return make_stream({"A": 6.0 * feed["F_A"] - 2.0 * tear["F_A"]},
                           feed["T"], feed["P"])


class Contractive:
    """``g(x) = 0.5 (F + x)``: converges from anywhere."""

    def __call__(self, feed, tear):
        return make_stream({"A": 0.5 * (feed["F_A"] + tear["F_A"])},
                           feed["T"], feed["P"])


def _loop(operation) -> Flowsheet:
    fs = Flowsheet(["A"], default_flow=0.01)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("loop", operation, ["feed", "tear"], ["loop_out"]))
    fs.add_unit(Unit(
        "out", lambda s: make_stream(get_flows(s), s["T"], s["P"]),
        ["loop_out"], ["product"]))
    fs.add_recycle("loop_out", "tear")
    return fs


def _compare(fs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compare_solvers(fs, **kwargs)


class TestSMVerdictIsReported:

    def test_a_diverged_sm_solve_is_not_reported_as_converged(self):
        """The regression: this used to come back True regardless."""
        result = _compare(_loop(Divergent()), max_iter=20,
                          sm_acceleration="none")
        assert result.sm_converged is False

    def test_a_converged_sm_solve_is_reported_as_converged(self):
        result = _compare(_loop(Contractive()), max_iter=200,
                          sm_acceleration="none")
        assert result.sm_converged is True

    def test_sm_iterations_is_the_real_count_not_the_cap(self):
        """It was hard-coded to max_iter, so it always equalled the cap."""
        result = _compare(_loop(Contractive()), max_iter=200,
                          sm_acceleration="none")
        assert result.sm_iterations is not None
        assert 0 < result.sm_iterations < 200

    def test_the_residual_comes_back_too(self):
        result = _compare(_loop(Contractive()), tol=1e-8, max_iter=200,
                          sm_acceleration="none")
        assert result.sm_residual is not None
        assert result.sm_residual <= 1e-8

    def test_a_diverged_solve_still_warns(self):
        """#249 made a failed solve say so; comparing is no reason to mute it.

        The verdict is returned *as well*, so a caller can test
        ``sm_converged`` rather than catch the warning --- but the warning
        is not the thing that got traded away for it.
        """
        with pytest.warns(ConvergenceWarning):
            result = compare_solvers(_loop(Divergent()), max_iter=20,
                                     sm_acceleration="none")
        assert result.sm_converged is False
