"""The isomerization unit with its deisohexanizer recycle (#311).

One paraffinic feed through a converged recycle (reactor, separator,
stabilizer and a deisohexanizer column), and the implicit gradient through
that recycle against central differences. The unit is solved once and shared
by every test in the file (the compile of a pass is most of the cost), on a
shortened column: 12 trays instead of 40 (the feed and side-draw trays scale
with it). The benzene-rich feed and the 40-tray column are not run here (the
reactor pass on the benzene-rich feed is in ``test_isomerization.py``); when
the unit was built the implicit gradients on both feeds agreed with central
differences to 4.4e-6 relative (#311).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow_refinery.isomerization import (
    OUTPUT_NAMES,
    IsomerizationHydrogenWarning,
    IsomerizationUnit,
    IsomerizationUnitParams,
    constructed_feed,
)

jax.config.update("jax_enable_x64", True)

SHORT_DIH = dict(dih_trays=12, dih_feed_tray=5, dih_side_tray=11)


@pytest.fixture(scope="module")
def solved():
    feed = constructed_feed("paraffinic", 10.0)
    dih = IsomerizationUnit(IsomerizationUnitParams(configuration="dih", **SHORT_DIH))
    with warnings.catch_warnings():
        warnings.simplefilter("error", IsomerizationHydrogenWarning)
        _, _, info = dih(feed)
    return dih, feed, info


def test_the_recycle_converges(solved):
    loop = solved[2]["loop"]
    assert loop["converged"], loop
    assert loop["closure"] < 1e-8


def test_every_column_converges(solved):
    for col in ("separator", "dih"):
        assert bool(solved[2][col]["converged"]), col


def test_the_whole_unit_closes(solved):
    unit, feed, info = solved
    b = unit.balances(feed, {"streams": info["streams"], "info": info})
    assert set(b) >= {"mass", "C5", "C6"}
    for k, v in b.items():
        assert abs(float(v)) < 1e-8, k


def test_the_side_draw_is_what_was_asked(solved):
    """The column meets its draw in kg/s on the species database's molar
    masses; ``recycle_mass`` is reported on the formula masses of
    :mod:`~difflow_refinery.isomerization.thermochem`. The two agree to
    within 0.05 g/mol (checked when the components are built), about 1e-5
    relative on a hexane, which is the whole of the difference here."""
    unit, info = solved[0], solved[2]
    assert float(info["outputs"]["recycle_mass"]) == pytest.approx(
        unit.params.dih_side_draw, rel=1e-4)
    assert float(info["outputs"]["dih_duty"]) > 1e6


def test_the_implicit_gradient_matches_central_differences(solved):
    """One ``jax.jacfwd`` through the converged recycle (the tear Jacobian at
    the solution, then the implicit function theorem in
    :meth:`IsomerizationUnit.outputs`) against central differences of
    warm-started re-solves, for T_in on RON, volume yield and DIH
    duty."""
    unit, feed, info = solved
    guess = info["streams"]["side_draw"]          # the converged tear: every solve warm-starts
    steps = (0.05,)                                        # T_in (K)
    outputs = ("RON", "yield_volume", "dih_duty")
    base = jnp.array([unit.params.T_in])

    J = jax.jacfwd(lambda v: unit.outputs(feed, v[0], unit.params.reactor.LHSV,
                                       recycle_guess=guess))(base)

    def f(v):
        return unit.outputs(feed, v[0], unit.params.reactor.LHSV, recycle_guess=guess)

    for j, h in enumerate(steps):
        e = jnp.zeros(1).at[j].set(h)
        fd = (f(base + e) - f(base - e)) / (2 * h)
        for n in outputs:
            i = OUTPUT_NAMES.index(n)
            a, b = float(J[i, j]), float(fd[i])
            assert b != 0.0, n
            assert abs(a - b) <= 1e-5 * abs(b), (j, n, a, b)
