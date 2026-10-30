"""Tests for difflow.workspace.layout: it must terminate on a recycle and be stable."""

from difflow import Flowsheet, Unit, make_stream
from difflow.workspace.layout import auto_layout, unit_columns
from difflow.units import Mixer, Splitter

SPECIES = ["A", "B"]


def _recycle() -> Flowsheet:
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(SPECIES), ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("split", Splitter(SPECIES), ["mixed"],
                     ["product", "purge"], params={"split_frac": [0.5, 0.5]}))
    fs.add_recycle("purge", "recycle")
    return fs


def test_unit_columns():
    assert unit_columns(["a", "b", "c"], {"b": ["a"], "c": ["b"]}) == {"a": 0, "b": 1, "c": 2}
    col = unit_columns(["a", "b"], {"a": ["b"], "b": ["a"]})   # a cycle terminates
    assert set(col) == {"a", "b"}


def test_auto_layout():
    assert auto_layout(Flowsheet(species_order=SPECIES)) == {}
    fs = _recycle()
    pos = auto_layout(fs)
    assert set(pos) == {"feed:feed", "mix", "split"}
    assert pos["feed:feed"][0] < pos["mix"][0] < pos["split"][0]
    assert auto_layout(fs) == pos     # stable


def test_the_layout_round_trips_through_the_view_block():
    from difflow import serialize

    fs = _recycle()
    fs.view["nodes"] = {k: {"x": x, "y": y} for k, (x, y) in auto_layout(fs).items()}
    back = serialize.from_dict(serialize.to_dict(fs))
    assert back.view["nodes"]["mix"] == fs.view["nodes"]["mix"]
