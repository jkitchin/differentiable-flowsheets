"""Smoke tests for the countercurrent Leacher and the washing helpers."""

import jax
import numpy as np
import pytest

from difflow import make_stream
from difflow.units.leaching import (
    Leacher, LeacherParams, tabulate_underflow, washing_recovery, washing_stages,
)

T, P = 300.0, 1e5
Y = [0.0, 0.1, 0.2, 0.4]
S = [1.5, 1.8, 2.2, 3.0]


def streams(V=150.0):
    feed = make_stream({"pulp": 100.0, "sugar": 20.0, "water": 180.0}, T, P)
    solv = make_stream({"water": V}, T, P)
    return feed, solv


def prm(**kw):
    base = dict(inert_species="pulp", solute="sugar", solvent="water")
    base.update(kw)
    return LeacherParams(**base)


def _balance(feed, solv, under, over):
    for sp in ("pulp", "sugar", "water"):
        fin = float(feed.get(f"F_{sp}", 0.0)) + float(solv.get(f"F_{sp}", 0.0))
        fout = float(under.get(f"F_{sp}", 0.0)) + float(over.get(f"F_{sp}", 0.0))
        assert abs(fin - fout) < 1e-8


def test_constant_underflow_smoke():
    feed, solv = streams()
    under, over, info = Leacher(prm(n_stages=3.0, underflow=1.5))(feed, solv)
    assert np.isfinite(float(info["recovery"]))
    _balance(feed, solv, under, over)


def test_variable_underflow_smoke():
    feed, solv = streams()
    under, over, info = Leacher(prm(n_stages=3, underflow_model="variable",
                                    underflow_y=Y, underflow_s=S))(feed, solv)
    assert np.isfinite(float(info["recovery"]))
    _balance(feed, solv, under, over)


def test_stage_design_smoke():
    feed, solv = streams()
    _, _, i = Leacher(prm(underflow=1.5, recovery=0.8, design="stages"))(feed, solv)
    assert np.isfinite(float(i["n_stages"]))


def test_washing_helpers_run():
    assert np.isfinite(float(washing_recovery(0.9, 3.0)))
    assert np.isfinite(float(washing_stages(0.9, 0.4)))
    ys, ss = tabulate_underflow(lambda y: 1.5 + 3.0 * y, 0.4, 5)
    assert len(ys) == len(ss) == 5


def test_validation():
    with pytest.raises(ValueError):
        prm(design="stages")                          # needs a recovery
    with pytest.raises(ValueError):
        prm(underflow_model="variable")               # needs tables
    with pytest.raises(ValueError):
        LeacherParams("a", "a", "b")
    with pytest.raises(ValueError):
        prm(underflow_model="nope")
    feed, solv = streams()
    with pytest.raises(ValueError, match="integer"):
        Leacher(prm(n_stages=3.5, underflow_model="variable", underflow_y=Y,
                    underflow_s=S))(feed, solv)


def test_grad_finite():
    feed, _ = streams()
    f = lambda v: Leacher(prm(n_stages=3.0, underflow=1.5))(  # noqa: E731
        feed, make_stream({"water": v}, T, P))[2]["recovery"]
    assert np.isfinite(float(jax.grad(f)(150.0)))
