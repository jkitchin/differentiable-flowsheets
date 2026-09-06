"""Incremental edits: one unit, one wire, one position.

The whole point of these routes is that they change part of a flowsheet
and leave the rest alone, so most of what is worth asserting is about
what did *not* move: the other units' objects, the live thermo, the
recycles that had nothing to do with the edit.
"""

import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    Flowsheet,
    IdealThermo,
    Mixer,
    Unit,
    get_species_data,
    make_stream,
    mass_action_kinetics,
    serialize,
)
from difflow.gui import FlowsheetSession, edit

SPECIES = ["water", "ethanol"]


@pytest.fixture(scope="module")
def thermo():
    return IdealThermo({n: get_species_data(n) for n in SPECIES})


def kinetics():
    return mass_action_kinetics([{
        "equation": "water -> ethanol",
        "reactants": {"water": 1.0}, "products": {"ethanol": 1.0},
        "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},
    }], SPECIES)


def build(thermo):
    """feed -> mixer -> reactor -> flash, with vap recycled to the mixer."""
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                    T=350.0, P=101325.0))
    fs.add_unit(Unit("mixer", Mixer(SPECIES), ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=1.0, molar_density=1000.0, **kinetics().params_kwargs()
    )), ["mixed"], ["rx"]))
    fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                     ["rx"], ["liq", "vap"]))
    fs.add_recycle("vap", "recycle")
    return fs


@pytest.fixture
def session(thermo):
    return FlowsheetSession(build(thermo))


class TestGraphReading:
    def test_reaches_follows_arcs(self, thermo):
        fs = build(thermo)
        assert edit.reaches(fs, "mixer", "flash")
        assert edit.reaches(fs, "mixer", "reactor")

    def test_reaches_does_not_follow_a_tear(self, thermo):
        """A recycle is a tear, not an arc.

        Walking it would make every unit in an existing loop reach every
        other, and the next genuinely new loop would be wired as an
        ordinary arc -- a flowsheet that solves in one pass and is wrong.
        """
        fs = build(thermo)
        assert not edit.reaches(fs, "flash", "mixer")

    def test_unique_leaves_a_free_name_alone(self):
        assert edit.unique("s", {"a"}) == "s"
        assert edit.unique("s", {"s"}) == "s2"
        assert edit.unique("s", {"s", "s2", "s3"}) == "s4"

    def test_stream_names_include_the_ends_of_a_recycle(self, thermo):
        names = edit.stream_names(build(thermo))
        assert {"feed", "recycle", "mixed", "rx", "liq", "vap"} <= names

    def test_an_unknown_unit_names_the_ones_that_exist(self, thermo):
        with pytest.raises(edit.EditError, match="mixer, reactor, flash"):
            edit.unit(build(thermo), "nope")


class TestPatchUnit:
    def test_a_parameter_changes(self, session):
        assert session.patch_unit("reactor", {"params": {"V": 3.0}})["ok"]
        reactor = edit.unit(session.flowsheet, "reactor")
        assert float(reactor.operation.params.V) == 3.0

    def test_the_other_units_are_not_rebuilt(self, session):
        """The reason to have this route at all."""
        before = {u.name: id(u.operation) for u in session.flowsheet.units}
        session.patch_unit("reactor", {"params": {"V": 3.0}})
        after = {u.name: id(u.operation) for u in session.flowsheet.units}
        assert after["mixer"] == before["mixer"]
        assert after["flash"] == before["flash"]
        assert after["reactor"] != before["reactor"]

    def test_a_live_thermo_survives_by_identity(self, session, thermo):
        """Not by round-tripping through JSON, which some kinds cannot."""
        session.patch_unit("flash", {"params": {"T": 360.0}})
        assert edit.unit(session.flowsheet, "flash").operation.thermo is thermo

    def test_an_unknown_field_is_refused_by_name(self, session):
        answer = session.patch_unit("reactor", {"params": {"nope": 1.0}})
        assert answer["ok"] is False
        assert "nope" in answer["error"]

    def test_an_unknown_change_key_is_refused_rather_than_ignored(self, session):
        answer = session.patch_unit("reactor", {"colour": "red"})
        assert answer["ok"] is False and "colour" in answer["error"]

    def test_a_rename_moves_the_canvas_position_with_it(self, session):
        session.set_layout({"reactor": {"x": 10, "y": 20}})
        assert session.patch_unit("reactor", {"name": "kettle"})["ok"]
        nodes = session.flowsheet.view["nodes"]
        assert "reactor" not in nodes
        assert nodes["kettle"] == {"x": 10.0, "y": 20.0}

    def test_a_rename_onto_an_existing_name_is_refused(self, session):
        answer = session.patch_unit("reactor", {"name": "flash"})
        assert answer["ok"] is False and "already" in answer["error"]

    def test_a_position_alone_does_not_touch_the_model(self, session):
        before = serialize.to_dict(session.flowsheet)["units"]
        session.patch_unit("reactor", {"position": {"x": 1, "y": 2}})
        assert serialize.to_dict(session.flowsheet)["units"] == before

    def test_a_bad_position_is_refused(self, session):
        answer = session.patch_unit("reactor", {"position": {"x": "left"}})
        assert answer["ok"] is False and "position" in answer["error"]

    def test_an_edited_flowsheet_still_serializes(self, session, thermo):
        """The edit path goes through the file path, so this must hold."""
        session.patch_unit("reactor", {"params": {"V": 2.5}})
        back = serialize.from_dict(serialize.to_dict(session.flowsheet),
                                   extras={"flash": {"thermo": thermo}})
        assert float(edit.unit(back, "reactor").operation.params.V) == 2.5


class TestAddAndRemove:
    def test_a_unit_arrives_unwired(self, session):
        answer = session.add_unit("Mixer", position={"x": 5, "y": 6})
        assert answer["ok"], answer
        added = edit.unit(session.flowsheet, answer["name"])
        assert added.inlet_names and added.outlet_names
        # nothing produces its inlets and nothing reads its outlets
        made = edit.producers(session.flowsheet)
        assert all(made[s] == added.name for s in added.outlet_names)
        assert not any(s in made for s in added.inlet_names)

    def test_its_streams_do_not_collide(self, session):
        first = session.add_unit("Mixer")["name"]
        second = session.add_unit("Mixer")["name"]
        assert first != second
        a = set(edit.unit(session.flowsheet, first).inlet_names)
        b = set(edit.unit(session.flowsheet, second).inlet_names)
        assert not a & b

    def test_a_unit_that_needs_a_thermo_says_so(self, session):
        answer = session.add_unit("Flash")
        assert answer["ok"] is False
        assert "thermo" in answer["error"]

    def test_an_unregistered_operation_is_refused(self, session):
        answer = session.add_unit("Teleporter")
        assert answer["ok"] is False and "registered" in answer["error"]

    def test_removing_a_unit_drops_the_recycle_that_needed_it(self, session):
        """Left behind, it would tear a stream nothing produces."""
        assert session.flowsheet.recycles == {"vap": "recycle"}
        answer = session.remove_unit("flash")
        assert answer["ok"] and answer["recycles_dropped"] == {"vap": "recycle"}
        assert session.flowsheet.recycles == {}

    def test_removing_a_unit_leaves_the_others(self, session):
        session.remove_unit("flash")
        assert [u.name for u in session.flowsheet.units] == ["mixer", "reactor"]

    def test_removing_a_unit_forgets_its_position(self, session):
        session.set_layout({"flash": {"x": 1, "y": 2}, "mixer": {"x": 3, "y": 4}})
        session.remove_unit("flash")
        assert set(session.flowsheet.view["nodes"]) == {"mixer"}


class TestWiring:
    def test_wiring_a_dangling_inlet_renames_it(self, session):
        name = session.add_unit("Mixer")["name"]
        inlet = edit.unit(session.flowsheet, name).inlet_names[0]
        answer = session.connect("flash", "liq", name, inlet)
        assert answer == {"ok": True, "kind": "arc", "stream": "liq"}
        assert "liq" in edit.unit(session.flowsheet, name).inlet_names

    def test_a_wire_that_closes_a_loop_becomes_a_recycle(self, session):
        """difflow spells a loop as a tear, never as an arc."""
        session.remove_unit("flash")
        # reactor -> mixer would close mixer -> reactor
        free = edit.unit(session.flowsheet, "mixer").inlet_names[1]
        answer = session.connect("reactor", "rx", "mixer", free)
        assert answer["kind"] == "recycle"
        assert session.flowsheet.recycles["rx"] == free

    def test_an_occupied_inlet_names_what_holds_it(self, session):
        answer = session.connect("flash", "liq", "reactor", "mixed")
        assert answer["ok"] is False and "mixer" in answer["error"]

    def test_a_feed_inlet_is_refused(self, session):
        answer = session.connect("flash", "liq", "mixer", "feed")
        assert answer["ok"] is False and "feed" in answer["error"]

    def test_a_port_that_does_not_exist_is_refused(self, session):
        answer = session.connect("flash", "steam", "mixer", "recycle")
        assert answer["ok"] is False and "steam" in answer["error"]

    def test_disconnecting_an_arc_leaves_a_free_port(self, session):
        answer = session.disconnect("mixer", "mixed", "reactor", "mixed")
        assert answer["ok"] and answer["kind"] == "arc"
        reactor = edit.unit(session.flowsheet, "reactor")
        assert reactor.inlet_names != ["mixed"]
        assert len(reactor.inlet_names) == 1, "the port stays, only the wire goes"
        assert reactor.inlet_names[0] not in edit.producers(session.flowsheet)

    def test_disconnecting_a_recycle_removes_the_tear(self, session):
        answer = session.disconnect("flash", "vap", "mixer", "recycle")
        assert answer["ok"] and answer["kind"] == "recycle"
        assert session.flowsheet.recycles == {}
        assert "recycle" in edit.unit(session.flowsheet, "mixer").inlet_names

    def test_disconnecting_what_is_not_connected_is_refused(self, session):
        answer = session.disconnect("flash", "liq", "reactor", "mixed")
        assert answer["ok"] is False

    def test_a_wire_and_back_round_trips(self, session):
        before = [list(u.inlet_names) for u in session.flowsheet.units]
        session.disconnect("mixer", "mixed", "reactor", "mixed")
        freed = edit.unit(session.flowsheet, "reactor").inlet_names[0]
        session.connect("mixer", "mixed", "reactor", freed)
        assert [list(u.inlet_names) for u in session.flowsheet.units] == before


class TestLayout:
    def test_positions_are_adopted(self, session):
        assert session.set_layout({"mixer": {"x": 1, "y": 2}})["ok"]
        assert session.flowsheet.view["nodes"]["mixer"] == {"x": 1.0, "y": 2.0}

    def test_positions_reach_the_file(self, session, tmp_path, thermo):
        session.set_layout({"mixer": {"x": 7, "y": 8}})
        session.path = tmp_path / "plant.json"
        assert session.save()["ok"]
        back = serialize.load(session.path, extras={"flash": {"thermo": thermo}})
        assert back.view["nodes"]["mixer"] == {"x": 7.0, "y": 8.0}

    def test_a_layout_does_not_solve_or_rebuild(self, session):
        before = [id(u.operation) for u in session.flowsheet.units]
        session.set_layout({"mixer": {"x": 1, "y": 2}})
        assert [id(u.operation) for u in session.flowsheet.units] == before

    def test_nonsense_is_refused(self, session):
        assert session.set_layout({"mixer": [1, 2]})["ok"] is False
        assert session.set_layout("nope")["ok"] is False


class TestEditedFlowsheetStillSolves:
    def test_after_a_parameter_change(self, session):
        session.patch_unit("reactor", {"params": {"V": 2.0}})
        assert session.solve()["ok"]

    def test_after_a_disconnect_and_reconnect(self, session):
        session.disconnect("mixer", "mixed", "reactor", "mixed")
        freed = edit.unit(session.flowsheet, "reactor").inlet_names[0]
        session.connect("mixer", "mixed", "reactor", freed)
        assert session.solve()["ok"]

    def test_no_flowsheet_is_a_refusal_not_a_crash(self):
        empty = FlowsheetSession()
        for answer in (empty.patch_unit("x", {}), empty.add_unit("Mixer"),
                       empty.remove_unit("x"), empty.set_layout({}),
                       empty.connect("a", "b", "c", "d")):
            assert answer == {"ok": False, "error": "no flowsheet loaded"}
