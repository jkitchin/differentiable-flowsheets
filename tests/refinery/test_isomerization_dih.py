"""The isomerization unit with its deisohexanizer recycle, on both feeds (#311).

``release`` and ``slow`` throughout: each feed is a converged recycle
(eight or nine Anderson iterations of reactor, separator, stabilizer and a
40-tray DIH column), one and a half to three minutes apiece. A file of its own so that
``--dist loadfile`` puts it on a worker apart from the gradients in
``test_isomerization_dih_gradients.py``.
"""

from __future__ import annotations

import warnings

import jax
import pytest

from difflow_refinery.isomerization import (
    OUTPUT_NAMES,
    IsomerizationHydrogenWarning,
    IsomerizationUnit,
    IsomerizationUnitParams,
    constructed_feed,
)

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.release, pytest.mark.slow]

FEEDS = ("paraffinic", "benzene_rich")


@pytest.fixture(scope="module", params=FEEDS)
def solved(request):
    kind = request.param
    feed = constructed_feed(kind, 10.0)
    dih = IsomerizationUnit(IsomerizationUnitParams(configuration="dih"))
    once = IsomerizationUnit(IsomerizationUnitParams())
    with warnings.catch_warnings():
        warnings.simplefilter("error", IsomerizationHydrogenWarning)
        _, _, info = dih(feed)
        ron_once = float(once.outputs(feed, once.params.T_in, once.params.reactor.LHSV)[
            OUTPUT_NAMES.index("RON")])
    return kind, dih, feed, info, ron_once


def test_the_recycle_converges(solved):
    loop = solved[3]["loop"]
    assert loop["converged"], loop
    assert loop["closure"] < 1e-8


def test_every_column_converges(solved):
    info = solved[3]
    for col in ("separator", "dih"):
        assert bool(info[col]["converged"]), col


def test_the_whole_unit_closes(solved):
    _, unit, feed, info, _ = solved
    b = unit.balances(feed, {"streams": info["streams"], "info": info})
    assert set(b) >= {"mass", "C5", "C6"}
    for k, v in b.items():
        assert abs(float(v)) < 1e-8, k


def test_the_recycle_raises_the_octane(solved):
    """Sending the methylpentanes and n-hexane back for another pass is the
    point of a DIH. The gain is modest: about 0.6 RON on the paraffinic feed
    and 1.2 on the benzene-rich one at the default conditions."""
    info, ron_once = solved[3], solved[4]
    assert float(info["outputs"]["RON"]) > ron_once


def test_the_side_draw_is_what_was_asked(solved):
    """The column meets its draw in kg/s on the species database's molar
    masses; ``recycle_mass`` is reported on the formula masses of
    :mod:`~difflow_refinery.isomerization.thermochem`. The two agree to
    within 0.05 g/mol (checked when the components are built), about 1e-5
    relative on a hexane, which is the whole of the difference here."""
    unit, info = solved[1], solved[3]
    assert float(info["outputs"]["recycle_mass"]) == pytest.approx(
        unit.params.dih_side_draw, rel=1e-4)


def test_the_dih_has_a_duty(solved):
    assert float(solved[3]["outputs"]["dih_duty"]) > 1e6
