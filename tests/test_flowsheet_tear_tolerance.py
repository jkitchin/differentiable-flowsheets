"""last_solve_gain, last_solve_error_estimate, TearToleranceWarning, tol_basis (#264)."""

import math
import warnings

import pytest

from difflow.flowsheet import Flowsheet, TearToleranceWarning, Unit
from difflow.streams import make_stream


class AffineLoop:
    def __init__(self, slope: float):
        self.slope = slope

    def __call__(self, inlet):
        return make_stream({"A": 1.0 + self.slope * inlet["F_A"]}, inlet["T"], inlet["P"])


def _loop(slope: float) -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", AffineLoop(slope), ["tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


GUESS = {"tear": make_stream({"A": 1.0}, 300.0, 1e5)}
KW = dict(tear_initial=GUESS, tol=1e-8, max_iter=400, acceleration="wegstein")


def test_high_gain_solve_warns_and_reports_error():
    fs = _loop(slope=0.97)
    with pytest.warns(TearToleranceWarning, match='tol_basis="error"'):
        fs.solve(**KW)
    assert math.isfinite(fs.last_solve_gain)
    assert math.isfinite(fs.last_solve_error_estimate)


def test_error_basis_runs():
    fs = _loop(slope=0.97)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TearToleranceWarning)
        fs.solve(tol_basis="error", **KW)
    assert fs.last_solve_converged is True


def test_error_probe_zero_measures_nothing():
    fs = _loop(slope=0.5)
    fs.solve(error_probe=0, **KW)
    assert fs.last_solve_gain is None and fs.last_solve_error_estimate is None


def test_tol_basis_validation():
    with pytest.raises(ValueError, match="error_probe"):
        _loop(0.5).solve(tear_initial=GUESS, tol_basis="error", error_probe=0)
    with pytest.raises(ValueError, match="tol_basis"):
        _loop(0.5).solve(tear_initial=GUESS, tol_basis="absolute")
