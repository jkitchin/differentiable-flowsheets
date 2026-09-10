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
from difflow.incomplete import Incomplete, IncompleteUnitError

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
        """It says so on the canvas now, not in a refusal."""
        answer = session.add_unit("Flash")
        assert answer["ok"] is True and answer["pending"] is True
        assert answer["needs"] == ["thermo"]
        assert "thermo" in answer["hint"]

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


class TestFeeds:
    """What a stream carries, and what happens when nothing does.

    A feed is the one part of a flowsheet that is typed rather than
    drawn, which makes it the one part where a refusal has to say what
    was wrong with the number. It is also the part a from-scratch
    flowsheet was missing entirely: everything else about building is a
    gesture, so there was no verb for a feed and a flowsheet made on the
    canvas could be drawn and never solved.
    """

    def test_the_declared_feeds_and_the_waiting_inlets_are_both_reported(
        self, session
    ):
        """`unfed` is what the canvas draws with nothing behind it.

        A recycle destination is not in it: nothing produces `recycle`
        either, and calling it unfed would claim the flowsheet has an
        inlet it does not have.
        """
        assert session.feeds() == {"ok": True, "feeds": ["feed"], "unfed": []}
        session.remove_feed("feed")
        assert session.feeds() == {"ok": True, "feeds": [], "unfed": ["feed"]}

    def test_one_field_at_a_time_leaves_the_others_alone(self, session):
        """Editing the temperature must not zero the flows."""
        answer = session.set_feed("feed", {"T": 360.0})
        assert answer["ok"]
        assert answer["T"] == 360.0
        assert answer["P"] == pytest.approx(101325.0)
        assert answer["flows"] == pytest.approx({"water": 1.0, "ethanol": 0.1})
        stream = session.flowsheet.feeds["feed"]
        assert float(stream["T"]) == 360.0
        assert float(stream["F_water"]) == pytest.approx(1.0)

    def test_a_stream_a_unit_makes_cannot_also_be_fed(self, session):
        """Two sources for one stream, and the solver would pick one."""
        answer = session.set_feed("mixed", {})
        assert answer["ok"] is False
        assert "made by 'mixer'" in answer["error"]
        assert "mixed" not in session.flowsheet.feeds

    def test_a_stream_nothing_reads_cannot_be_fed_either(self, session):
        """A feed into thin air is a typo, not a flowsheet."""
        answer = session.set_feed("nope", {})
        assert answer["ok"] is False
        assert "nothing reads 'nope'" in answer["error"]

    def test_a_feed_needs_the_species_before_it_can_carry_anything(self):
        empty = FlowsheetSession()
        answer = empty.set_feed("anything", {})
        assert answer["ok"] is False
        assert "species" in answer["error"]

    @pytest.mark.parametrize(
        "spec, why",
        [
            ({"T": "hot"}, "T must be a number"),
            ({"P": "1 bar"}, "P must be a number"),
            ({"flows": [1.0, 0.1]}, "flows must be a mapping"),
            ({"flows": {"argon": 1.0}}, "argon is not one of the species"),
            ({"flows": {"water": "some"}}, "water must be a number"),
            ({"flows": {"water": -1.0}}, "a flow cannot be negative"),
            ({"T": 0.0}, "temperature and pressure are absolute"),
            ({"P": -1.0}, "temperature and pressure are absolute"),
        ],
    )
    def test_a_number_that_is_not_one_is_named(self, session, spec, why):
        answer = session.set_feed("feed", spec)
        assert answer["ok"] is False
        assert why in answer["error"], answer
        # And the feed it refused to change is untouched.
        assert float(session.flowsheet.feeds["feed"]["T"]) == 350.0

    def test_a_zero_flow_is_a_real_flow(self, session):
        """A species absent from a feed, which is not the same as a typo."""
        answer = session.set_feed("feed", {"flows": {"ethanol": 0.0}})
        assert answer["ok"]
        assert float(session.flowsheet.feeds["feed"]["F_ethanol"]) == 0.0

    def test_which_stream_has_to_be_said(self, session):
        for name in (None, "", "   ", 7):
            answer = session.set_feed(name, {})
            assert answer["ok"] is False
            assert "which stream" in answer["error"]

    def test_removing_a_feed_that_is_not_there_lists_the_ones_that_are(
        self, session
    ):
        answer = session.remove_feed("mixed")
        assert answer["ok"] is False
        assert "no feed called 'mixed'" in answer["error"]
        assert "have: feed" in answer["error"]

    def test_an_unfed_inlet_is_reported_as_a_missing_feed_not_a_key_error(
        self, session
    ):
        """`KeyError: 'feed'` names the stream and nothing else.

        Which is the least useful place to be told: the name is on the
        canvas already, and what is missing is a feed on it. The
        translation only applies to an inlet that is actually unfed, so
        an unrelated `KeyError` still travels as itself.
        """
        session.remove_feed("feed")
        answer = session.solve()
        assert answer["ok"] is False
        assert "nothing feeds 'feed'" in answer["error"]
        assert "give it a feed" in answer["error"]

    def test_a_feed_survives_the_round_trip_through_the_document(self, session):
        session.set_feed("feed", {"T": 333.0, "P": 2.0e5,
                                  "flows": {"water": 4.0, "ethanol": 0.25}})
        doc = session.document()["flowsheet"]
        assert doc["feeds"]["feed"] == pytest.approx(
            {"T": 333.0, "P": 2.0e5, "F_water": 4.0, "F_ethanol": 0.25}
        )
        again = FlowsheetSession(serialize.from_dict(doc))
        assert again.feeds() == {"ok": True, "feeds": ["feed"], "unfed": []}


class TestEditedFlowsheetStillSolves:
    def test_after_a_parameter_change(self, session):
        session.patch_unit("reactor", {"params": {"V": 2.0}})
        assert session.solve()["ok"]

    def test_after_a_disconnect_and_reconnect(self, session):
        session.disconnect("mixer", "mixed", "reactor", "mixed")
        freed = edit.unit(session.flowsheet, "reactor").inlet_names[0]
        session.connect("mixer", "mixed", "reactor", freed)
        assert session.solve()["ok"]

    def test_an_empty_flowsheet_refuses_by_name_rather_than_crashing(self):
        """Every verb answers over a session opened with no file.

        It used to answer "no flowsheet loaded" to all of them, because
        there was none --- which is what made dragging a unit onto a fresh
        canvas do nothing at all. Now the flowsheet exists and empty, so
        each refusal is about the thing actually missing: the unit that is
        not there, or the species the streams would be indexed by.
        """
        empty = FlowsheetSession()
        for answer, expected in (
            (empty.patch_unit("x", {}), "no unit called 'x'"),
            (empty.remove_unit("x"), "no unit called 'x'"),
            (empty.connect("a", "b", "c", "d"), "no unit called 'a'"),
        ):
            assert answer["ok"] is False
            assert expected in answer["error"], answer

        # A drop is the exception: it is not refused any more, it is
        # parked. The unit lands as a pending node that says what it is
        # waiting for --- here, the species the streams are indexed by.
        parked = empty.add_unit("Mixer")
        assert parked["ok"] is True and parked["pending"] is True
        assert "needs the species" in parked["hint"], parked

        # Placing nothing is a request that succeeded at nothing.
        assert empty.set_layout({}) == {"ok": True, "nodes": 0}


THERMO_SOURCE = (
    "from difflow import IdealThermo, get_species_data\n"
    "thermo = IdealThermo({n: get_species_data(n) for n in "
    "['water', 'ethanol']})\n"
)


class TestPendingUnits:
    """A drop that cannot be built yet lands anyway, in red.

    Refusing it threw away the only thing the user actually said --- I
    want a Flash, here --- and left them to reconstruct it after writing
    the thermo. So the drop is kept as a *pending* node: it says what it
    is waiting for, and the two edits that can answer it (the code
    context, the species) finish it in place.

    It goes on the flowsheet, as a real `Unit` whose operation is an
    `Incomplete` stand-in. Held off the flowsheet it had no ports, and a
    port is what a wire lands on --- so the node was there and inert,
    and a dropped Compressor could not be connected to anything. The
    stand-in carries the ports the operation declares and raises if
    anything asks it to compute, so the ways out are all closed: a solve
    refuses by name, and `session.pending` is the note about what each
    one is still waiting for.
    """

    def test_a_blocked_drop_lands_on_the_flowsheet_with_its_ports(self, session):
        """On it, so it can be wired now rather than after the thermo."""
        answer = session.add_unit("Flash", position={"x": 10, "y": 20})
        assert answer["pending"] is True
        assert answer["name"] == "flash2", "the built flash keeps its name"
        assert "flash2" in [u.name for u in session.flowsheet.units]
        assert session.flowsheet.view["nodes"]["flash2"] == {"x": 10.0, "y": 20.0}, (
            "its coordinate is in the view with everybody else's")
        assert [e["name"] for e in session.pending_units()] == ["flash2"]

        unit = next(u for u in session.flowsheet.units if u.name == "flash2")
        assert isinstance(unit.operation, Incomplete)
        assert unit.inlet_names and unit.outlet_names, (
            "the ports are what a wire lands on")
        with pytest.raises(IncompleteUnitError):
            unit.operation()

    def test_the_document_carries_it_inside_the_flowsheet(self, session):
        """Inside, and it has to survive the trip to disk and back.

        `document()["flowsheet"]` is what gets written out and read back
        by `serialize`, so an unfinished unit that did not round-trip
        would be a file that loses the wiring the user had already done.
        It reloads with no extras, because there is nothing to supply
        yet --- that is the whole point of it being unfinished.
        """
        session.add_unit("Flash", position={"x": 10, "y": 20})
        document = session.document()
        assert [e["name"] for e in document["pending"]] == ["flash2"]
        assert document["flowsheet"]["view"]["nodes"]["flash2"] == {
            "x": 10.0, "y": 20.0}, "its coordinate is in the view like anyone's"

        written = [u for u in document["flowsheet"]["units"]
                   if u["name"] == "flash2"]
        assert written and written[0]["incomplete"]["needs"] == ["thermo"]

        reloaded = serialize.from_dict(document["flowsheet"])
        again = next(u for u in reloaded.units if u.name == "flash2")
        assert isinstance(again.operation, Incomplete)
        assert again.operation.operation == "Flash"

    def test_a_pending_name_is_taken(self, session):
        """Two drops of the same blocked unit are two nodes.

        If pending names did not count as taken, both would be called
        `flash-2` --- and the second would either overwrite the first or
        collide with it at promotion.
        """
        first = session.add_unit("Flash")["name"]
        second = session.add_unit("Flash")["name"]
        assert first != second
        assert sorted(session.pending) == sorted([first, second])

    def test_the_code_context_builds_it_where_it_was_dropped(self, session):
        session.add_unit("Flash", position={"x": 10, "y": 20})
        answer = session.set_code_context(THERMO_SOURCE)
        assert answer["ok"] and answer["promoted"] == ["flash2"]
        assert answer["pending"] == []
        assert "flash2" in [u.name for u in session.flowsheet.units]
        assert session.flowsheet.view["nodes"]["flash2"] == {"x": 10.0, "y": 20.0}
        assert session.pending_units() == []

    def test_the_species_build_what_was_waiting_on_them(self):
        """The empty-canvas case: nearly the whole palette is blocked."""
        empty = FlowsheetSession()
        parked = empty.add_unit("Mixer", position={"x": 3, "y": 4})
        assert parked["pending"] and "species" in parked["hint"]

        answer = empty.set_species(SPECIES)
        assert answer["promoted"] == ["mixer"] and answer["pending"] == []
        assert [u.name for u in empty.flowsheet.units] == ["mixer"]
        assert empty.flowsheet.view["nodes"]["mixer"] == {"x": 3.0, "y": 4.0}

    def test_an_answer_that_only_half_answers_leaves_it_parked(self):
        """And re-asks, so the hint is about what is missing *now*."""
        empty = FlowsheetSession()
        empty.add_unit("Flash")
        assert sorted(empty.pending["flash"]["needs"]) == [
            "species_order", "thermo"]

        answer = empty.set_species(SPECIES)
        assert answer["promoted"] == [] and answer["pending"] == ["flash"]
        assert empty.pending["flash"]["needs"] == ["thermo"]
        assert "species" not in empty.pending["flash"]["hint"]

    def test_a_pending_node_is_deleted_like_any_other(self, session):
        """It is a unit, so there is no second delete path to get wrong."""
        name = session.add_unit("Flash")["name"]
        answer = session.remove_unit(name)
        assert answer["ok"] and answer["name"] == name
        assert name not in [u.name for u in session.flowsheet.units]
        assert session.pending_units() == [], "and the note goes with it"

    def test_a_solve_refuses_while_one_is_unfinished(self, session):
        """The price of the node being in the model: it has to say no.

        Up front and by name, rather than raising from somewhere deep in
        the solve --- or, worse, appearing to have worked.
        """
        session.add_unit("Flash")
        answer = session.solve()
        assert answer["ok"] is False
        assert answer["pending"] == ["flash2"]
        assert "flash2" in answer["error"] and "thermo" in answer["error"]

    def test_loading_a_file_forgets_them(self, session):
        """They belong to the canvas that was open, not to the next one."""
        session.add_unit("Flash")
        session.replace(serialize.to_dict(Flowsheet(species_order=SPECIES)))
        assert session.pending_units() == []

    def test_the_boilerplate_answers_the_recorded_needs(self, session):
        name = session.add_unit("Flash")["name"]
        answer = session.boilerplate("Flash", name=name)
        assert answer["ok"] and answer["needs"] == ["thermo"]
        assert "thermo = IdealThermo" in answer["source"]
        assert 'SPECIES = ["water", "ethanol"]' in answer["source"], (
            "the flowsheet's own species, not the fallback")
        assert answer["merged"] == answer["source"], "the context was empty"

    def test_the_boilerplate_is_appended_to_what_is_already_there(self, session):
        session.set_code_context("solvent = 'MEA'\n")
        answer = session.boilerplate("Flash")
        assert answer["merged"].startswith("solvent = 'MEA'")
        assert "thermo = IdealThermo" in answer["merged"]

    def test_the_boilerplate_closes_the_loop_it_promises(self, session):
        """Write it, apply it, and the red node has to become a unit."""
        name = session.add_unit("Flash", position={"x": 7, "y": 8})["name"]
        session.set_code_context(session.boilerplate("Flash", name=name)["merged"])
        assert name in [u.name for u in session.flowsheet.units]
        assert session.flowsheet.view["nodes"][name] == {"x": 7.0, "y": 8.0}

    def test_an_unregistered_operation_has_no_boilerplate(self, session):
        answer = session.boilerplate("Teleporter")
        assert answer["ok"] is False and "registered" in answer["error"]
