"""Tests for difflow.gui.layout.

A canvas needs coordinates and a flowsheet carries none, so the first
thing that happens when a flowsheet is opened is that these functions
invent them. Two properties matter and neither is obvious: the layout
must terminate on a recycle (the graph is cyclic, and a longest path on
a cyclic graph is not a thing), and it must be stable, or reopening a
file shuffles the diagram under the user.
"""

import pytest

from difflow import Flowsheet, Unit, make_stream
from difflow.gui.layout import COL_W, MARGIN, ROW_H, auto_layout, unit_columns
from difflow.units import Mixer, Splitter

SPECIES = ["A", "B"]


def _chain() -> Flowsheet:
    """feed -> mix -> split -> (product, purge)."""
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(SPECIES), ["feed"], ["mixed"]))
    fs.add_unit(Unit("split", Splitter(SPECIES), ["mixed"],
                     ["product", "purge"], params={"split_frac": [0.5, 0.5]}))
    return fs


def _recycle() -> Flowsheet:
    """The same chain with the purge sent back to the mixer."""
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(SPECIES), ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("split", Splitter(SPECIES), ["mixed"],
                     ["product", "purge"], params={"split_frac": [0.5, 0.5]}))
    fs.add_recycle("purge", "recycle")
    return fs


class TestUnitColumns:
    def test_a_chain_is_one_column_per_step(self):
        col = unit_columns(["a", "b", "c"], {"b": ["a"], "c": ["b"]})
        assert col == {"a": 0, "b": 1, "c": 2}

    def test_a_join_sits_past_its_deepest_predecessor(self):
        col = unit_columns(["a", "b", "c", "d"],
                           {"b": ["a"], "c": ["b"], "d": ["a", "c"]})
        assert col["d"] == 3, "longest path, not shortest"

    def test_a_cycle_terminates(self):
        """A longest path on a cycle is not a thing; the pass cap bounds it.

        auto_layout never feeds a cycle in --- it drops the recycle edges
        first --- but the relaxation must still return rather than run away
        if something else does.
        """
        col = unit_columns(["a", "b"], {"a": ["b"], "b": ["a"]})
        assert set(col) == {"a", "b"}
        assert all(v <= 2 * len(col) for v in col.values())

    def test_a_unit_with_no_predecessors_is_column_zero(self):
        assert unit_columns(["a"], {}) == {"a": 0}


class TestAutoLayout:
    def test_an_empty_flowsheet_lays_out_to_nothing(self):
        assert auto_layout(Flowsheet(species_order=SPECIES)) == {}

    def test_every_node_gets_a_position(self):
        pos = auto_layout(_chain())
        assert set(pos) == {"feed:feed", "mix", "split",
                            "product:product", "product:purge"}

    def test_it_reads_left_to_right(self):
        pos = auto_layout(_chain())
        assert pos["feed:feed"][0] < pos["mix"][0] < pos["split"][0]
        assert pos["split"][0] < pos["product:product"][0]

    def test_two_outlets_of_one_unit_stack(self):
        pos = auto_layout(_chain())
        assert pos["product:product"][0] == pos["product:purge"][0]
        assert pos["product:product"][1] != pos["product:purge"][1]

    def test_the_first_slot_sits_at_the_margin(self):
        assert auto_layout(_chain())["feed:feed"] == (float(MARGIN), float(MARGIN))

    def test_the_spacing_is_the_declared_geometry(self):
        pos = auto_layout(_chain())
        assert pos["mix"][0] - pos["feed:feed"][0] == float(COL_W)
        assert (pos["product:purge"][1] - pos["product:product"][1]) == float(ROW_H)

    def test_a_recycle_does_not_push_anything_right(self):
        """A recycle is an arrow going back, not another column."""
        plain, looped = auto_layout(_chain()), auto_layout(_recycle())
        assert looped["mix"][0] == plain["mix"][0]
        assert looped["split"][0] == plain["split"][0]

    def test_a_recycled_stream_is_not_a_product(self):
        pos = auto_layout(_recycle())
        assert "product:purge" not in pos
        assert "product:product" in pos

    def test_it_is_stable(self):
        """Reopening a file must not shuffle the diagram."""
        fs = _recycle()
        assert auto_layout(fs) == auto_layout(fs)

    def test_the_geometry_is_a_parameter(self):
        pos = auto_layout(_chain(), col_w=10, row_h=5, margin=0)
        assert pos["feed:feed"] == (0.0, 0.0)
        assert pos["mix"][0] == 10.0

    def test_the_keys_are_the_flowsheet_vocabulary(self):
        """Same shape as _apply_params keys: a feed is prefixed, a unit is not."""
        pos = auto_layout(_chain())
        assert "mix" in pos and "feed:mix" not in pos
        assert "feed:feed" in pos and "feed" not in pos

    def test_the_result_drops_into_the_view_block(self):
        from difflow import serialize

        fs = _chain()
        fs.view["nodes"] = {k: {"x": x, "y": y}
                            for k, (x, y) in auto_layout(fs).items()}
        back = serialize.from_dict(serialize.to_dict(fs))
        assert back.view["nodes"]["mix"]["x"] == fs.view["nodes"]["mix"]["x"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
