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

    def test_an_unknown_unit_names_the_ones_that_exist(self, thermo):
        with pytest.raises(edit.EditError, match="mixer, reactor, flash"):
            edit.unit(build(thermo), "nope")


class TestPatchUnit:
    def test_an_empty_name_is_refused_not_ignored(self, session):
        answer = session.patch_unit("reactor", {"name": ""})
        assert answer["ok"] is False and "empty" in answer["error"]

    def test_a_parameter_changes(self, session):
        assert session.patch_unit("reactor", {"params": {"V": 3.0}})["ok"]
        reactor = edit.unit(session.flowsheet, "reactor")
        assert float(reactor.operation.params.V) == 3.0

    def test_an_unknown_field_is_refused_by_name(self, session):
        answer = session.patch_unit("reactor", {"params": {"nope": 1.0}})
        assert answer["ok"] is False
        assert "nope" in answer["error"]

    def test_text_in_a_number_is_refused_and_the_value_kept(self, session):
        """It used to be stored, and every later solve failed in JAX with
        an error naming neither the unit nor the field."""
        answer = session.patch_unit("reactor", {"params": {"V": "abc"}})
        assert answer["ok"] is False
        assert "CSTRParams.V" in answer["error"] and "abc" in answer["error"]
        reactor = edit.unit(session.flowsheet, "reactor")
        assert float(reactor.operation.params.V) == 1.0

    def test_an_unknown_change_key_is_refused_rather_than_ignored(self, session):
        answer = session.patch_unit("reactor", {"colour": "red"})
        assert answer["ok"] is False and "colour" in answer["error"]

    def test_a_name_the_canvas_keys_feeds_by_is_refused(self, session):
        answer = session.patch_unit("reactor", {"name": "feed:feed"})
        assert answer["ok"] is False and "feeds and products" in answer["error"]
        assert "reactor" in [u.name for u in session.flowsheet.units]

    def test_a_rename_onto_an_existing_name_is_refused(self, session):
        answer = session.patch_unit("reactor", {"name": "flash"})
        assert answer["ok"] is False and "already" in answer["error"]

    def test_a_bad_position_is_refused(self, session):
        answer = session.patch_unit("reactor", {"position": {"x": "left"}})
        assert answer["ok"] is False and "position" in answer["error"]

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

    def test_an_unregistered_operation_is_refused(self, session):
        answer = session.add_unit("Teleporter")
        assert answer["ok"] is False and "registered" in answer["error"]

    def test_removing_a_unit_drops_the_recycle_that_needed_it(self, session):
        """Left behind, it would tear a stream nothing produces."""
        assert session.flowsheet.recycles == {"vap": "recycle"}
        answer = session.remove_unit("flash")
        assert answer["ok"] and answer["recycles_dropped"] == {"vap": "recycle"}
        assert session.flowsheet.recycles == {}

    def test_removing_a_unit_keeps_the_feed_only_it_read(self, session):
        """The feed is the part that took typing; a replacement takes it."""
        feed = next(iter(session.flowsheet.feeds))
        spec = session.flowsheet.feeds[feed]
        reader = next(u.name for u in session.flowsheet.units
                      if feed in u.inlet_names)
        session.set_layout({f"feed:{feed}": {"x": 1, "y": 2}})
        answer = session.remove_unit(reader)
        assert answer["ok"] and answer["feeds_unread"] == [feed]
        assert session.flowsheet.feeds[feed] is spec
        assert f"feed:{feed}" in session.flowsheet.view["nodes"]

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

    def test_disconnecting_what_is_not_connected_is_refused(self, session):
        answer = session.disconnect("flash", "liq", "reactor", "mixed")
        assert answer["ok"] is False

class TestLayout:
    def test_positions_are_adopted(self, session):
        assert session.set_layout({"mixer": {"x": 1, "y": 2}})["ok"]
        assert session.flowsheet.view["nodes"]["mixer"] == {"x": 1.0, "y": 2.0}

    def test_nonsense_is_refused(self, session):
        assert session.set_layout({"mixer": [1, 2]})["ok"] is False
        assert session.set_layout("nope")["ok"] is False

    def test_a_non_finite_position_is_refused(self, session):
        before = dict(session.flowsheet.view.get("nodes", {}))
        answer = session.set_layout({"mixer": {"x": 1, "y": 2},
                                     "reactor": {"x": float("nan"), "y": 0}})
        assert answer["ok"] is False and "finite" in answer["error"]
        # and nothing was half applied
        assert session.flowsheet.view.get("nodes", {}) == before

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

    def test_a_stream_a_unit_makes_cannot_also_be_fed(self, session):
        """Two sources for one stream, and the solver would pick one."""
        answer = session.set_feed("mixed", {})
        assert answer["ok"] is False
        assert "made by 'mixer'" in answer["error"]
        assert "mixed" not in session.flowsheet.feeds

    def test_a_feed_needs_the_species_before_it_can_carry_anything(self):
        empty = FlowsheetSession()
        answer = empty.set_feed("anything", {})
        assert answer["ok"] is False
        assert "species" in answer["error"]

    @pytest.mark.parametrize(
        "spec, why",
        [
            ({"T": "hot"}, "T must be a number"),
            ({"flows": [1.0, 0.1]}, "flows must be a mapping"),
            ({"flows": {"argon": 1.0}}, "argon is not one of the species"),
            ({"flows": {"water": -1.0}}, "a flow cannot be negative"),
            ({"T": 0.0}, "temperature and pressure are absolute"),
            ({"T": float("nan")}, "T must be a finite number"),
        ],
    )
    def test_a_number_that_is_not_one_is_named(self, session, spec, why):
        answer = session.set_feed("feed", spec)
        assert answer["ok"] is False
        assert why in answer["error"], answer
        # And the feed it refused to change is untouched.
        assert float(session.flowsheet.feeds["feed"]["T"]) == 350.0

class TestEditedFlowsheetStillSolves:
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


class TestCallParameters:
    """Arguments a unit takes each time it runs, not when it is built.

    A Splitter's ``split_frac`` is an argument of ``__call__``; difflow keeps
    it on ``Unit.params``. A palette Splitter that could not be given one
    could never solve.
    """

    def test_they_are_set_and_removed_through_patch_unit(self, session):
        name = session.add_unit("Splitter")["name"]
        answer = session.patch_unit(name, {"call_params": {"split_frac": 0.7}})
        assert answer["ok"], answer
        assert edit.unit(session.flowsheet, name).params == {"split_frac": 0.7}
        assert session.patch_unit(
            name, {"call_params": {"split_frac": None}})["ok"]
        assert edit.unit(session.flowsheet, name).params == {}

    def test_an_unknown_one_is_refused(self, session):
        name = session.add_unit("Splitter")["name"]
        answer = session.patch_unit(name, {"call_params": {"spilt_frac": 0.7}})
        assert answer["ok"] is False
        assert "spilt_frac" in answer["error"]
        assert edit.unit(session.flowsheet, name).params == {}

    def test_a_missing_required_one_stops_the_solve_and_says_where(
            self, session):
        name = session.add_unit("Splitter")["name"]
        session.connect("flash", "liq", name,
                        edit.unit(session.flowsheet, name).inlet_names[0])
        answer = session.solve()
        assert answer["ok"] is False
        assert name in answer["error"] and "split_frac" in answer["error"]
        assert "call parameters" in answer["error"]


class TestSolveAudit:
    """A solve that returns is not yet a solve that is right.

    A CSTR with ``V = -1`` solved, and the editor said "solved", with more
    mass leaving than entering.
    """

    @pytest.fixture
    def ester(self):
        s = FlowsheetSession()
        assert s.open_example("02_reactor_flash")["ok"]
        return s

    def test_a_negative_volume_is_reported_not_passed(self, ester):
        assert ester.patch_unit("reactor", {"params": {"V": -1.0}})["ok"]
        answer = ester.solve()
        assert answer["ok"]   # it ran; the audit is what says it is wrong
        assert any("mass is not conserved" in w
                   for w in answer["audit"]["warnings"]), answer["audit"]

class TestUnsavedWork:
    """Quitting, or choosing an example, used to throw edits away unasked."""

    @pytest.fixture
    def ester(self):
        s = FlowsheetSession()
        assert s.open_example("02_reactor_flash")["ok"]
        return s

    def test_an_edit_marks_it(self, ester):
        answer = ester.patch_unit("reactor", {"params": {"V": 0.7}})
        assert answer.get("ok", True), answer
        assert ester.document()["dirty"] is True

    def test_saving_clears_it(self, ester, tmp_path):
        ester.patch_unit("reactor", {"params": {"V": 0.7}})
        ester.path = tmp_path / "plant.json"
        assert ester.save()["ok"]
        assert ester.document()["dirty"] is False

class TestFiles:
    """Save As, Open and New: an example used to be unsaveable."""

    @pytest.fixture
    def ester(self):
        s = FlowsheetSession()
        assert s.open_example("02_reactor_flash")["ok"]
        return s

    def test_save_as_gives_an_example_a_file(self, ester, tmp_path):
        answer = ester.save(str(tmp_path / "ester"))
        assert answer["ok"], answer
        assert answer["path"].endswith("ester.json")
        assert ester.path == tmp_path / "ester.json"
        # and plain Save now goes there
        ester.patch_unit("reactor", {"params": {"V": 0.7}})
        assert ester.save()["ok"]
        assert FlowsheetSession(path=ester.path).flowsheet is not None

    def test_save_as_will_not_write_over_a_file_unasked(self, ester, tmp_path):
        other = tmp_path / "other.json"
        other.write_text("{}")
        answer = ester.save(str(other))
        assert not answer["ok"] and answer["exists"]
        assert other.read_text() == "{}"
        assert ester.save(str(other), overwrite=True)["ok"]
        assert other.read_text() != "{}"

    def test_a_bad_file_leaves_the_flowsheet_alone(self, ester, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        before = ester.flowsheet
        answer = ester.open_file(str(bad))
        assert not answer["ok"] and "bad.json" in answer["error"]
        assert ester.flowsheet is before
        assert not ester.open_file(str(tmp_path / "missing.json"))["ok"]

    def test_an_unfinished_unit_survives_a_save_and_open(self, tmp_path):
        s = FlowsheetSession()
        s.set_species(["water", "ethanol"])
        answer = s.add_unit("CSTR")
        assert answer.get("pending"), answer
        s.save(str(tmp_path / "half.json"))
        reopened = FlowsheetSession()
        assert reopened.open_file(str(tmp_path / "half.json"))["ok"]
        assert [p["name"] for p in reopened.pending_units()] == [answer["name"]]
        assert [p["name"] for p in
                FlowsheetSession(path=tmp_path / "half.json").pending_units()
                ] == [answer["name"]]


class TestUndo:
    """There was no way back from a deleted unit but retyping it."""

    @pytest.fixture
    def ester(self):
        s = FlowsheetSession()
        assert s.open_example("02_reactor_flash")["ok"]
        return s

    def V(self, s):
        return float(edit.unit(s.flowsheet, "reactor").operation.params.V)

    def test_undo_and_redo_a_parameter(self, ester):
        ester.patch_unit("reactor", {"params": {"V": 0.7}})
        ester.patch_unit("reactor", {"params": {"V": 0.9}})
        assert ester.undo()["ok"] and self.V(ester) == pytest.approx(0.7)
        assert ester.undo()["ok"] and self.V(ester) == pytest.approx(0.5)
        assert ester.document()["dirty"] is False
        answer = ester.redo()
        assert answer == {"ok": True, "undo": True, "redo": True}
        assert self.V(ester) == pytest.approx(0.7)

    def test_a_new_edit_drops_the_redo(self, ester):
        ester.patch_unit("reactor", {"params": {"V": 0.7}})
        ester.undo()
        ester.patch_unit("reactor", {"params": {"V": 0.8}})
        assert not ester.redo()["ok"]

    def test_history_is_bounded(self, ester):
        from difflow.gui.session import UNDO_DEPTH
        for i in range(UNDO_DEPTH + 5):
            ester.set_layout({"reactor": {"x": float(i), "y": 0.0}})
        assert len(ester._undo) == UNDO_DEPTH


class TestSolverOptions:
    """The recycle solver's options were fixed at their defaults.

    A loop that needed 150 iterations, a looser tolerance or Wegstein in
    place of Anderson could not be solved from the editor at all.
    """

    @pytest.fixture
    def recycle(self):
        s = FlowsheetSession()
        assert s.open_example("03_reactor_recycle")["ok"]
        return s

    def test_they_are_saved_with_the_file_and_undone(self, recycle):
        assert recycle.set_solver_options({"tol": 1e-6})["ok"]
        assert recycle.flowsheet.view["solver"] == {"tol": 1e-6}
        assert recycle.undo()["ok"]
        assert "solver" not in recycle.flowsheet.view
        assert recycle.solver_options()["tol"] == 1e-8

    @pytest.mark.parametrize("bad", [
        {"tol": 0}, {"acceleration": "newton"}, {"future_option": 3}, ["tol"],
    ])
    def test_a_bad_option_is_refused_and_nothing_is_kept(self, recycle, bad):
        good = {"acceleration": "wegstein"}
        answer = recycle.set_solver_options(
            {**good, **bad} if isinstance(bad, dict) else bad)
        assert not answer["ok"]
        assert "solver" not in recycle.flowsheet.view

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

    def test_the_species_build_what_was_waiting_on_them(self):
        """The empty-canvas case: nearly the whole palette is blocked."""
        empty = FlowsheetSession()
        parked = empty.add_unit("Mixer", position={"x": 3, "y": 4})
        assert parked["pending"] and "species" in parked["hint"]

        answer = empty.set_species(SPECIES)
        assert answer["promoted"] == ["mixer"] and answer["pending"] == []
        assert [u.name for u in empty.flowsheet.units] == ["mixer"]
        assert empty.flowsheet.view["nodes"]["mixer"] == {"x": 3.0, "y": 4.0}

    def test_a_species_name_that_would_split_its_keys_is_refused(self):
        """`<stream>.F_<species>` has to come apart at the right dot."""
        empty = FlowsheetSession()
        answer = empty.set_species(["water", "a.b"])
        assert answer["ok"] is False and "'a.b'" in answer["error"]
        assert empty.set_species(["water", "1_butanol"])["ok"]

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

    def test_loading_a_file_forgets_them(self, session):
        """They belong to the canvas that was open, not to the next one."""
        session.add_unit("Flash")
        session.replace(serialize.to_dict(Flowsheet(species_order=SPECIES)))
        assert session.pending_units() == []

    def test_an_unregistered_operation_has_no_boilerplate(self, session):
        answer = session.boilerplate("Teleporter")
        assert answer["ok"] is False and "registered" in answer["error"]


class TestReplace:
    def test_a_document_that_will_not_build_leaves_everything_as_it_was(
            self, session):
        """The code context used to be adopted before the build, so a
        failing document left the old flowsheet on the new names."""
        session.set_code_context("K = 2.0")
        before = session.flowsheet
        document = serialize.to_dict(build_plain())
        document["view"] = {"code_context": "OTHER = 1"}
        document["units"][0]["operation"] = "NoSuchOperation"
        answer = session.replace(document)
        assert answer["ok"] is False
        assert session.flowsheet is before
        assert sorted(session.bindings) == ["K"]

    def test_a_document_that_is_not_an_object_is_refused(self, session):
        assert session.replace(["not", "a", "flowsheet"])["ok"] is False

    def test_an_unfinished_unit_in_the_document_arrives_pending(self, session):
        fs = build_plain()
        fs.add_unit(Unit("todo", Incomplete("CSTR", needs=["rate_fn"]),
                         ["liq"], ["out"]))
        assert session.replace(serialize.to_dict(fs))["ok"]
        assert "todo" in session.pending


def build_plain():
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                    T=350.0, P=101325.0))
    fs.add_unit(Unit("mixer", Mixer(SPECIES), ["feed"], ["liq"]))
    return fs
