"""Tearing a stream that does not carry every species.

Plugin units routinely return only the species their outlet holds: the
aqueous scrub liquor of a rare-earth solvent extraction circuit has no
extractant or diluent in it. Tearing such a stream used to fail inside
the tear packing with ``KeyError: 'F_<species>'``. A species a stream
omits is zero flow, so the loop below has a plain answer: half of A and
B goes round, so the recycle carries 1 mol/s of each and no S.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, make_stream  # noqa: E402
from difflow.flowsheet import Unit  # noqa: E402

SPECIES = ["A", "B", "S"]


def mix(feed, rec):
    return make_stream(
        {s: feed.get(f"F_{s}", 0.0) + rec.get(f"F_{s}", 0.0) for s in SPECIES},
        feed["T"], feed["P"],
    )


def split_without_solvent(mixed):
    """Half of A and B back to the mixer, in a stream with no S at all."""
    back = make_stream({"A": 0.5 * mixed["F_A"], "B": 0.5 * mixed["F_B"]},
                       mixed["T"], mixed["P"])
    product = make_stream({s: mixed[f"F_{s}"] - back.get(f"F_{s}", 0.0)
                           for s in SPECIES}, mixed["T"], mixed["P"])
    return product, back


def build():
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 1.0, "S": 2.0}, 300.0, 1e5))
    fs.add_unit(Unit("mix", mix, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("split", split_without_solvent, ["mixed"],
                     ["product", "back"]))
    fs.add_recycle("back", "rec")
    return fs


@pytest.mark.parametrize("acceleration", ["anderson", "wegstein", "none"])
def test_a_tear_missing_a_species_solves(acceleration):
    streams = build().solve(acceleration=acceleration, tol=1e-12, max_iter=200)
    assert float(streams["back"]["F_A"]) == pytest.approx(1.0, abs=1e-9)
    assert float(streams["back"]["F_B"]) == pytest.approx(1.0, abs=1e-9)
    assert "F_S" not in streams["back"]
    assert float(streams["product"]["F_S"]) == pytest.approx(2.0)


def test_the_missing_species_packs_as_zero():
    fs = build()
    packed = fs._streams_to_array(
        {"back": make_stream({"A": 0.5, "B": 0.25}, 300.0, 1e5)})
    assert [float(v) for v in packed] == [0.5, 0.25, 0.0, 300.0, 1e5]


def test_a_partial_outlet_is_a_usable_tear_guess():
    """It used to be thrown away for the flowsheet default."""
    fs = build()
    assert fs._usable_tear(make_stream({"A": 0.5}, 300.0, 1e5))
    assert not fs._usable_tear({"F_A": 0.5, "F_B": 0.5, "F_S": 0.0})
