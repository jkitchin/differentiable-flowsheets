"""Tests for difflow.gui.

The editor is a thin shell over serialize, codegen and catalog, so the
tests exercise the shell: that every route answers, that an edit made in
the browser reaches the model and changes the answer, and that a bad
edit is reported rather than taking the server down.

The two load-bearing tests are `test_an_edit_changes_the_solution` --- an
editor whose edits do not reach the model is worse than none --- and
`TestNonFiniteFloats`, because `mass_action_kinetics` puts `inf` in
`K_eq` for every irreversible reaction and Python's `json` writes that
as `Infinity`, which the browser refuses to parse.
"""

import errno
import json
import pathlib
import shutil
import socket
import subprocess
import threading
import urllib.error
import urllib.request

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    Flowsheet,
    Heater,
    HeaterParams,
    IdealThermo,
    Mixer,
    Unit,
    get_species_data,
    gui,
    make_stream,
    mass_action_kinetics,
    serialize,
)
from difflow.gui import (
    FlowsheetSession,
    _json_restore,
    _json_safe,
    console,
    doclinks,
    edit,
    make_server,
    server,
)

SPECIES = ["water", "ethanol"]


@pytest.fixture(scope="module")
def thermo():
    return IdealThermo({n: get_species_data(n) for n in SPECIES})


def build_flowsheet(thermo, V=1.0):
    """A reactor with a data-built rate law, then a flash."""
    kin = mass_action_kinetics([{
        "equation": "water -> ethanol",
        "reactants": {"water": 1.0}, "products": {"ethanol": 1.0},
        "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},
    }], SPECIES)
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream(
        {"water": 1.0, "ethanol": 0.1}, T=350.0, P=101325.0
    ))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=V, molar_density=1000.0, **kin.params_kwargs()
    )), ["feed"], ["rx"]))
    fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                     ["rx"], ["liq", "vap"]))
    return fs


class Client:
    """A live server on an ephemeral port, plus the verbs it takes."""

    def __init__(self, session):
        self.session = session
        self.server = make_server(session, port=0)
        self.base = f"http://{gui.HOST}:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def get(self, path, headers=None):
        request = urllib.request.Request(self.base + path,
                                         headers=headers or {})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def get_json(self, path):
        status, body = self.get(path)
        return status, json.loads(body)

    def send(self, verb, path, body=None, headers=None):
        """A mutating request, carrying the token the page would carry."""
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json",
                     gui.TOKEN_HEADER: self.server.token, **(headers or {})},
            method=verb,
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def post(self, path, body=None, headers=None):
        return self.send("POST", path, body, headers)

    def patch(self, path, body=None, headers=None):
        return self.send("PATCH", path, body, headers)

    def delete(self, path, body=None, headers=None):
        return self.send("DELETE", path, body, headers)


@pytest.fixture
def client(thermo):
    live = Client(FlowsheetSession(build_flowsheet(thermo)))
    yield live
    live.close()


# =============================================================================
# Routes
# =============================================================================


class TestRoutes:
    def test_the_flowsheet_is_served_as_the_serialize_document(self, client):
        status, doc = client.get_json("/api/flowsheet")
        assert status == 200
        assert [u["name"] for u in doc["flowsheet"]["units"]] == ["reactor", "flash"]
        assert doc["flowsheet"]["format_version"] == serialize.version_for(doc["flowsheet"])

    def test_the_served_document_carries_canvas_positions(self, client):
        """A canvas needs coordinates and a flowsheet carries none."""
        status, doc = client.get_json("/api/flowsheet")
        assert status == 200
        nodes = doc["flowsheet"]["view"]["nodes"]
        assert {"reactor", "flash", "feed:feed"} <= set(nodes)
        assert nodes["reactor"]["x"] < nodes["flash"]["x"], "left to right"

    def test_the_assistant_brief_is_served(self, client):
        """The one route whose request is a question, not a resource."""
        status, pack = client.get_json(
            "/api/context?kind=block&name=reactor&q=what+is+the+volume")
        assert status == 200 and pack["ok"]
        assert "Operation: CSTR" in pack["prompt"]
        assert pack["prompt"].endswith("what is the volume")

    def test_an_unknown_brief_is_refused_in_the_answer(self, client):
        status, pack = client.get_json("/api/context?kind=tarot")
        assert status == 200, "a bad question is an answer, not a 4xx"
        assert pack["ok"] is False and "tarot" in pack["error"]

    def test_the_key_the_server_holds_is_reported_before_it_is_used(
            self, client, monkeypatch):
        """So "no key here" is a sentence in the settings, not a failure."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        status, payload = client.get_json("/api/assistant")
        assert status == 200 and payload["ok"] and payload["configured"] is False

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
        _, payload = client.get_json("/api/assistant")
        assert payload["configured"] is True

    def test_forwarding_without_a_key_is_refused_in_the_answer(
            self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        status, payload = client.post(
            "/api/assistant", {"messages": [{"role": "user", "content": "hi"}]})
        assert status == 200, "a missing key is an answer, not a 4xx"
        assert payload["ok"] is False
        assert "ANTHROPIC_API_KEY" in payload["error"]

    def test_the_python_export_is_served(self, client):
        status, payload = client.get_json("/api/code")
        assert status == 200
        assert payload["error"] is None
        assert "mass_action_kinetics" in payload["source"]

    def test_solve_returns_every_stream(self, client):
        status, payload = client.post("/api/solve")
        assert status == 200 and payload["ok"]
        assert set(payload["streams"]) == {"feed", "rx", "liq", "vap"}
        assert isinstance(payload["streams"]["rx"]["F_ethanol"], float)

    def test_favicon_is_answered(self, client):
        """Otherwise every page load logs a 404 in the console."""
        assert client.get("/favicon.ico")[0] == 200

    def test_an_unknown_route_is_a_404(self, client):
        assert client.get("/api/nope")[0] == 404
        assert client.post("/api/nope")[0] == 404


# =============================================================================
# Static assets
# =============================================================================


class TestStatic:
    """The page is files on disk now, so the tree is a route."""

    def test_the_page_comes_from_the_static_directory(self, client):
        """The built file, plus the one thing the server adds: the token."""
        on_disk = (gui.STATIC / "index.html").read_text(encoding="utf-8")
        assert gui.page() == on_disk
        served = client.get("/")[1].decode("utf-8")
        tag = f'<meta name="{gui.TOKEN_META}" content="{client.server.token}">'
        assert tag in served
        # One line added inside <head>; the rest is the file, byte for byte.
        assert served.replace(f"  {tag}\n  ", "", 1) == on_disk

    def test_the_built_bundle_is_served(self, client):
        """The committed build output is what the page loads."""
        status, body = client.get("/app.js")
        assert status == 200 and len(body) > 1000
        assert b"<title>difflow editor</title>" in client.get("/")[1]

    def test_the_classic_editor_is_still_reachable(self, client):
        """It stays until the canvas covers what it does."""
        status, body = client.get("/classic")
        assert status == 200
        assert b'id="palette"' in body

    def test_a_type_that_the_build_never_emits_is_a_404(self, client):
        """An allow-list, so a stray file in static/ is not a route."""
        stray = gui.STATIC / "_probe.txt"
        stray.write_text("not for the browser")
        try:
            assert client.get("/_probe.txt")[0] == 404
        finally:
            stray.unlink()

    def test_a_traversal_cannot_escape_the_static_directory(self, client):
        """The server is reachable from a browser; the tree is not the disk."""
        for path in ("/../server.py", "/../../difflow/flowsheet.py",
                     "/..%2fserver.py", "/static/../../__init__.py"):
            assert client.get(path)[0] == 404, path

# =============================================================================
# Editing
# =============================================================================


class TestEditing:
    def test_the_edited_flowsheet_is_what_the_session_holds(self, client):
        _, doc = client.get_json("/api/flowsheet")
        doc["flowsheet"]["units"][0]["params"]["V"] = 3.0
        client.post("/api/flowsheet", doc["flowsheet"])
        assert float(client.session.flowsheet.units[0].operation.params.V) == 3.0

    def test_a_bad_edit_is_reported_and_the_server_survives(self, client):
        status, payload = client.post("/api/flowsheet", {"units": "not a flowsheet"})
        assert status == 400 and not payload["ok"]
        assert "error" in payload
        # the previous model is untouched and still solvable
        assert client.post("/api/solve")[1]["ok"]

    def test_malformed_json_is_reported(self, client):
        request = urllib.request.Request(
            client.base + "/api/flowsheet", data=b"{not json",
            headers={"Content-Type": "application/json",
                     gui.TOKEN_HEADER: client.server.token}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400
        assert "bad JSON" in json.loads(excinfo.value.read())["error"]

    @pytest.mark.parametrize("raw, expect", [
        (b"[1, 2]", "JSON object"),
        (b'{"a": "\xff"}', "not UTF-8"),
    ])
    def test_a_body_that_is_not_an_object_is_refused_as_such(self, client, raw, expect):
        """Said about the request, not as an AttributeError from a route."""
        request = urllib.request.Request(
            client.base + "/api/solve", data=raw,
            headers={"Content-Type": "application/json",
                     gui.TOKEN_HEADER: client.server.token}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400
        error = json.loads(excinfo.value.read())["error"]
        assert expect in error and "AttributeError" not in error

    def test_a_query_string_does_not_hide_a_route(self, client):
        status, payload = client.get_json("/api/flowsheet?v=2")
        assert status == 200 and "flowsheet" in payload
        status, _ = client.get("/?reload=1")
        assert status == 200


# =============================================================================
# Incremental routes
# =============================================================================


class TestIncrementalRoutes:
    """The verbs the canvas uses. What they do is covered in test_gui_edit."""

    def test_a_patch_changes_one_parameter(self, client):
        status, payload = client.patch("/api/unit/reactor", {"params": {"V": 5.0}})
        assert status == 200 and payload["ok"]
        assert float(client.session.flowsheet.units[0].operation.params.V) == 5.0

    def test_a_patch_to_a_unit_that_is_not_there_is_a_refusal(self, client):
        status, payload = client.patch("/api/unit/nope", {"params": {}})
        assert status == 200, "a wrong name is an answer about the flowsheet"
        assert payload["ok"] is False and "nope" in payload["error"]

    def test_a_unit_is_added_and_removed(self, client):
        status, added = client.post("/api/unit", {"operation": "Mixer"})
        assert status == 200 and added["ok"]
        names = [u.name for u in client.session.flowsheet.units]
        assert added["name"] in names
        assert client.delete(f"/api/unit/{added['name']}")[1]["ok"]
        assert added["name"] not in [u.name for u in client.session.flowsheet.units]

    def test_a_wire_is_made_and_broken(self, client):
        added = client.post("/api/unit", {"operation": "Mixer"})[1]
        wire = {"source": "flash", "outlet": "liq",
                "target": added["name"], "inlet": added["inlets"][0]}
        assert client.post("/api/connect", wire)[1] == {
            "ok": True, "kind": "arc", "stream": "liq"
        }
        wire["inlet"] = "liq"
        assert client.delete("/api/connect", wire)[1]["ok"]

    def test_half_a_wire_is_refused_by_name(self, client):
        status, payload = client.post("/api/connect", {"source": "flash"})
        assert status == 200 and payload["ok"] is False
        for missing in ("outlet", "target", "inlet"):
            assert missing in payload["error"]

    def test_positions_are_stored_without_a_rebuild(self, client):
        reactor = client.session.flowsheet.units[0].operation
        status, payload = client.post(
            "/api/layout", {"nodes": {"reactor": {"x": 40, "y": 12}}}
        )
        assert status == 200 and payload["ok"]
        assert client.session.flowsheet.view["nodes"]["reactor"] == {"x": 40.0, "y": 12.0}
        assert client.session.flowsheet.units[0].operation is reactor

    def test_an_unrouted_verb_is_a_404(self, client):
        assert client.patch("/api/nothing")[0] == 404
        assert client.delete("/api/nothing")[0] == 404

# =============================================================================
# Non-finite floats
# =============================================================================


class TestNonFiniteFloats:
    """JSON has no literal for these, and the browser rejects Python's."""

    def test_the_document_parses_under_browser_rules(self, client):
        """`JSON.parse` has no Infinity; Python's `json.loads` allows it."""
        def reject(token):
            raise AssertionError(f"bare {token} would break JSON.parse")

        status, body = client.get("/api/flowsheet")
        assert status == 200
        json.loads(body, parse_constant=reject)

    def test_the_round_trip_restores_the_value(self, client):
        _, doc = client.get_json("/api/flowsheet")
        status, payload = client.post("/api/flowsheet", doc["flowsheet"])
        assert status == 200 and payload["ok"]
        rate_params = client.session.flowsheet.units[0].operation.params.rate_params
        assert float(rate_params["K_eq"][0]) == float("inf")

    def test_json_safe_and_restore_are_inverse(self):
        value = {"a": [1.0, float("inf"), -float("inf")], "b": {"c": 2.0}}
        assert _json_safe(value) == {
            "a": [1.0, {"$float": "inf"}, {"$float": "-inf"}], "b": {"c": 2.0}}
        assert _json_restore(_json_safe(value)) == value

    def test_a_string_that_spells_a_non_finite_float_stays_a_string(self):
        """It used to be turned into one: a unit or species called NaN
        arrived as a float."""
        value = {"name": "NaN", "notes": ["Infinity", "-Infinity"]}
        assert _json_restore(value) == value

# =============================================================================
# Files
# =============================================================================


class TestFiles:
    def test_a_session_loads_a_flowsheet_from_a_path(self, thermo, tmp_path):
        path = tmp_path / "plant.json"
        serialize.save(build_flowsheet(thermo), path)

        session = FlowsheetSession(path=path)
        assert [u.name for u in session.flowsheet.units] == ["reactor", "flash"]

    def test_save_writes_a_readable_file(self, thermo, tmp_path):
        path = tmp_path / "plant.json"
        session = FlowsheetSession(build_flowsheet(thermo), path)
        assert session.save()["ok"]
        assert [u.name for u in serialize.load(path).units] == ["reactor", "flash"]

    def test_save_without_a_path_is_refused_not_raised(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        result = session.save()
        assert not result["ok"] and "Save As" in result["error"]

# =============================================================================
# Failures the browser has to be told about
# =============================================================================


class TestFailureReporting:
    def test_a_failing_solve_is_reported_not_raised(self, thermo):
        """The browser needs the message; a traceback at the socket is useless."""
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0}, T=350.0, P=101325.0))
        # a Heater takes one inlet; two is a modelling error, not a crash
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)),
                         ["feed", "recycle"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("liq", "recycle")

        result = FlowsheetSession(fs).solve()
        assert not result["ok"]
        assert result["error"], "a failure must carry a message"

    def test_an_unregistered_unit_makes_the_code_panel_report(self, thermo):
        class HomeMadeUnit:
            params = None

        fs = build_flowsheet(thermo)
        fs.add_unit(Unit("mystery", HomeMadeUnit(), ["rx"], ["out"]))
        assert "registry" in FlowsheetSession(fs).code()["error"]


# =============================================================================
# Recycles
# =============================================================================


class TestRecycles:
    def test_a_recycle_flowsheet_serves_edits_and_solves(self, thermo):
        """The diagram draws recycles, so the document has to carry them."""
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("mix", Mixer(SPECIES, thermo),
                         ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)),
                         ["mixed"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("vap", "recycle")

        live = Client(FlowsheetSession(fs))
        try:
            _, doc = live.get_json("/api/flowsheet")
            assert doc["flowsheet"]["recycles"] == {"vap": "recycle"}

            doc["flowsheet"]["units"][1]["params"]["T_out"] = 365.0
            assert live.post("/api/flowsheet", doc["flowsheet"])[1]["ok"]

            _, solved = live.post("/api/solve")
            assert solved["ok"]
            assert solved["streams"]["hot"]["T"] == pytest.approx(365.0, abs=1e-9)
        finally:
            live.close()


# =============================================================================
# Building: adding units from the palette, and naming streams
# =============================================================================


def heater_unit(name="heater", inlet="liq"):
    """Exactly the document the page's ``newUnit`` builds for a Heater.

    Written out rather than derived, so that a change to the page which
    stopped producing this shape would fail here.
    """
    return {"name": name, "operation": "Heater", "params": {},
            "constructor": {}, "extra_params": {},
            "inlets": [inlet], "outlets": [name + "_out"]}


class TestWhatBlocksADrop:
    """The palette's flag and what the adder does, which must agree.

    They did not. The catalog answered a question about the *class* ---
    could a form construct one --- and the adder answered a question
    about this session, and where the two differed the user got a
    traceback from the file-loading path offering "written by a
    different version of difflow" as the diagnosis of a unit dropped one
    second earlier.

    A blocked drop is no longer refused: it lands as a *pending* node
    that says what it is waiting for. So "blocks" now means "parks", and
    the flag must predict the parking exactly.
    """

    def test_a_binding_unblocks_the_unit_that_wanted_it(self, client):
        """The round trip the message promises has to actually work."""
        assert client.get_json("/api/catalog")[1]["Flash"]["needs"] == ["thermo"]
        assert client.post("/api/unit", {"operation": "Flash"})[1]["pending"]

        client.post("/api/code-context", {"source": THERMO_CONTEXT})

        assert client.get_json("/api/catalog")[1]["Flash"]["needs"] == []
        assert not client.post(
            "/api/unit", {"operation": "Flash"})[1].get("pending")

class TestPendingNodesOverHTTP:
    """The red node, from the browser's side.

    The routes are the contract the front end draws against: a drop that
    cannot be built answers 200 with ``pending``, the node is on the
    flowsheet with real ports so it can be wired straight away, the
    served document lists it under ``pending`` so the canvas can paint it
    red, and there is a route that writes the code it is waiting for.
    """

    def test_a_blocked_drop_answers_200_and_pending(self, client):
        status, answer = client.post("/api/unit", {"operation": "CSTR"})
        assert status == 200, "not being buildable is an answer, not a bad request"
        assert answer["ok"] is True and answer["pending"] is True
        assert answer["needs"] and answer["hint"]
        assert "error" not in answer

    def test_the_boilerplate_route_writes_what_it_is_waiting_for(self, client):
        name = client.post("/api/unit", {"operation": "Flash"})[1]["name"]
        status, answer = client.post(
            "/api/boilerplate", {"operation": "Flash", "name": name}
        )
        assert status == 200 and answer["ok"]
        assert answer["needs"] == ["thermo"]
        assert "thermo = IdealThermo" in answer["source"]
        assert answer["merged"].endswith(answer["source"])

    def test_writing_it_and_applying_it_builds_the_node(self, client):
        """The loop the button promises, over the wire."""
        name = client.post("/api/unit", {"operation": "Flash",
                                         "position": {"x": 7, "y": 8}})[1]["name"]
        _, written = client.post("/api/boilerplate",
                                 {"operation": "Flash", "name": name})
        _, applied = client.post("/api/code-context",
                                 {"source": written["merged"]})
        assert applied["ok"] and applied["promoted"] == [name]
        _, payload = client.get_json("/api/flowsheet")
        assert payload["pending"] == []
        assert name in [u["name"] for u in payload["flowsheet"]["units"]]
        assert payload["flowsheet"]["view"]["nodes"][name] == {"x": 7.0, "y": 8.0}

    def test_an_unregistered_operation_is_refused(self, client):
        _, answer = client.post("/api/boilerplate", {"operation": "Teleporter"})
        assert answer["ok"] is False and "registered" in answer["error"]

    def test_a_solve_refuses_while_one_is_unfinished(self, client):
        """By name, and saying what it is waiting for.

        The unit is on the flowsheet now, so nothing stops the solver
        reaching it; what it would reach is a stand-in with no model
        behind it. Better to say so at the top than to raise from the
        middle of a recycle loop.
        """
        client.post("/api/unit", {"operation": "CSTR", "name": "R1"})
        _, answer = client.post("/api/solve", {})
        assert answer["ok"] is False and answer["pending"] == ["R1"]
        assert "R1" in answer["error"] and "rate_fn" in answer["error"]


class TestAnEmptyEditor:
    """`difflow gui` with no file: a live canvas, waiting for species.

    The regression this class exists for: the session left
    ``self.flowsheet = None`` when it was opened without a path, and
    every edit route begins by refusing "no flowsheet loaded". Nothing
    said so. The palette filled in, the canvas drew its grid, and
    dragging a unit onto it did nothing at all --- no error, no node ---
    which reads as a broken drag rather than as an editor that has no
    flowsheet to edit.

    So an empty editor holds a real, empty ``Flowsheet``, and the one
    thing it is missing --- the species, which every stream array is
    indexed by --- is asked for by name.
    """

    @pytest.fixture
    def empty(self):
        live = Client(FlowsheetSession())
        yield live
        live.close()

    def test_an_editor_opened_with_no_file_still_has_a_flowsheet(self, empty):
        status, payload = empty.get_json("/api/flowsheet")
        assert status == 200
        assert payload["flowsheet"]["units"] == []
        assert payload["path"] == ""
        assert payload["species"] == []
        assert payload["editable"] is True

    def test_naming_the_species_unblocks_the_drop(self, empty):
        assert empty.post("/api/species", {"species": SPECIES})[1]["ok"]
        assert empty.get_json("/api/species")[1]["species"] == SPECIES
        answer = empty.post("/api/unit", {"operation": "Mixer"})[1]
        assert answer["ok"], answer
        assert [u.name for u in empty.session.flowsheet.units] == ["mixer"]

    def test_the_solver_options_route(self, empty):
        status, answer = empty.post("/api/solver", {"max_iter": 7})
        assert status == 200 and answer["solver"]["max_iter"] == 7
        flowsheet = empty.get_json("/api/flowsheet")[1]["flowsheet"]
        assert flowsheet["view"]["solver"] == {"max_iter": 7}
        assert not empty.post("/api/solver", {"tol": -1})[1]["ok"]

    @pytest.mark.parametrize(
        "names, why",
        [
            ("water", "must be a list"),
            (["water", "water"], "named twice"),
            (["water", "  "], "needs a name"),
        ],
    )
    def test_a_species_list_that_cannot_index_a_stream_is_refused(
        self, empty, names, why
    ):
        _, answer = empty.post("/api/species", {"species": names})
        assert answer["ok"] is False
        assert why in answer["error"]

    def test_the_list_freezes_once_a_unit_indexes_it(self, empty):
        """Re-ordering under a built unit would relabel its numbers.

        A ``Stream`` holds molar flows as an array indexed by
        ``species_order``. Swapping the order once a unit holds one turns
        a water flow into an ethanol flow with nothing on screen changing,
        which is the worst kind of wrong answer.
        """
        empty.post("/api/species", {"species": SPECIES})
        empty.post("/api/unit", {"operation": "Mixer"})
        assert empty.get_json("/api/species")[1]["editable"] is False
        _, answer = empty.post("/api/species", {"species": SPECIES[::-1]})
        assert answer["ok"] is False and "already has units" in answer["error"]
        assert list(empty.session.flowsheet.species_order) == SPECIES

    def test_a_solve_can_be_recorded_for_the_trace_panel(self, client):
        status, answer = client.post("/api/debug/record", {})
        assert status == 200 and answer["ok"]
        trace = answer["trace"]
        assert trace["protocol"] == "difflow-trace/1"
        assert trace["frames"][0]["checkpoint"] == "solve_start"
        assert trace["frames"][-1]["checkpoint"] == "terminated"

    def test_a_what_if_replays_the_recording(self, thermo):
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("mix", Mixer(SPECIES, thermo), ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)), ["mixed"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("vap", "recycle")
        session = FlowsheetSession(fs)
        assert "record a solve first" in session.debug_whatif(0, "recycle", "T", 1)["error"]

        trace = session.debug_record()["trace"]
        k = next(i for i, f in enumerate(trace["frames"]) if f["checkpoint"] == "iteration")
        answer = session.debug_whatif(k, "recycle", "T", 361.0)
        assert answer["ok"], answer
        assert answer["trace"]["result"]["diverged_at"] is None
        assert not session.debug_whatif(0, "recycle", "T", 1)["ok"]       # not an iteration
        assert not session.debug_whatif(k, "recycle", "T", "warm")["ok"]

        assert session.patch_unit("heat", {"params": {"T_out": 365.0}})["ok"]
        stale = session.debug_whatif(k, "recycle", "T", 361.0)
        assert not stale["ok"] and "changed since" in stale["error"]

    def test_a_flowsheet_built_from_nothing_can_be_fed_and_solved(self, empty):
        """Drop, wire, feed, solve --- the whole reported defect, over HTTP.

        The second half of the same bug. Once the drops worked, a
        from-scratch flowsheet could be drawn and still not solved: every
        other part of building is a gesture, and a feed is *data*, so
        there was no verb for one and ``solve`` answered
        ``KeyError: 'mixer_in'``.
        """
        empty.post("/api/species", {"species": SPECIES})
        assert empty.post("/api/unit", {"operation": "Mixer"})[1]["ok"]
        assert empty.post("/api/unit", {"operation": "Heater"})[1]["ok"]

        units = {u["name"]: u for u
                 in empty.get_json("/api/flowsheet")[1]["flowsheet"]["units"]}
        assert empty.post("/api/connect", {
            "source": "mixer", "outlet": units["mixer"]["outlets"][0],
            "target": "heater", "inlet": units["heater"]["inlets"][0],
        })[1]["ok"]

        # Every inlet was unfed; wiring the heater's fed that one. The
        # mixer keeps two, because a mixer arrives able to mix.
        _, waiting = empty.get_json("/api/feeds")
        assert waiting == {"ok": True, "feeds": [],
                           "unfed": ["mixer_in", "mixer_in2"]}

        # And the solver says which stream, and what to do about it,
        # rather than raising the name on its own.
        _, refused = empty.post("/api/solve", {})
        assert refused["ok"] is False
        assert "nothing feeds 'mixer_in'" in refused["error"]

        _, fed = empty.post("/api/feed", {"name": "mixer_in", "T": 320.0,
                                          "flows": {"water": 2.0}})
        assert fed["ok"], fed
        assert empty.post("/api/feed", {"name": "mixer_in2"})[1]["ok"]
        # A field left out keeps what it had: the pressure is the
        # flowsheet's default, and ethanol its default flow.
        assert fed["T"] == 320.0
        assert fed["P"] == empty.session.flowsheet.default_P
        assert fed["flows"] == {"water": 2.0,
                                "ethanol": empty.session.flowsheet.default_flow}

        empty.send("PATCH", "/api/unit/heater", {"params": {"T_out": 340.0}})
        _, solved = empty.post("/api/solve", {})
        assert solved["ok"], solved
        assert solved["streams"]["heater_out"]["T"] == pytest.approx(340.0)

    def test_a_feed_can_be_taken_back_off_a_stream(self, empty):
        """Which is how an inlet becomes wirable again.

        `connect` refuses to wire into a stream that is already fed
        rather than quietly dropping the feed, so undeclaring one has to
        be something the page can ask for.
        """
        empty.post("/api/species", {"species": SPECIES})
        empty.post("/api/unit", {"operation": "Mixer"})
        assert empty.post("/api/feed", {"name": "mixer_in"})[1]["ok"]
        assert empty.get_json("/api/feeds")[1]["feeds"] == ["mixer_in"]

        _, gone = empty.delete("/api/feed/mixer_in")
        assert gone == {"ok": True, "name": "mixer_in"}
        assert empty.get_json("/api/feeds")[1] == {
            "ok": True, "feeds": [], "unfed": ["mixer_in", "mixer_in2"]
        }
        assert empty.delete("/api/feed/mixer_in")[1]["ok"] is False


class TestBuilding:
    def test_a_unit_added_from_the_palette_reaches_the_model(self, client):
        _, doc = client.get_json("/api/flowsheet")
        doc["flowsheet"]["units"].append(heater_unit())
        status, payload = client.post("/api/flowsheet", doc["flowsheet"])
        assert status == 200 and payload["ok"]
        assert [u.name for u in client.session.flowsheet.units] == [
            "reactor", "flash", "heater"
        ]

    def test_a_dangling_inlet_is_reported_rather_than_raised(self, client):
        """The page catches this first, but the socket must survive it."""
        _, doc = client.get_json("/api/flowsheet")
        doc["flowsheet"]["units"][1]["inlets"] = ["ghost"]
        assert client.post("/api/flowsheet", doc["flowsheet"])[1]["ok"]

        status, solved = client.post("/api/solve")
        assert status == 200 and not solved["ok"]
        assert "ghost" in solved["error"]


# =============================================================================
# The code context: the Python a flowsheet carries
# =============================================================================


THERMO_CONTEXT = (
    "from difflow import IdealThermo, get_species_data\n"
    "thermo = IdealThermo({n: get_species_data(n) for n in "
    "['water', 'ethanol']})\n"
)


class TestCodeContext:
    """Half the catalog needs an object, and no form supplies one."""

    def test_a_snippet_defines_names(self, client):
        status, payload = client.post("/api/code-context",
                                      {"source": THERMO_CONTEXT})
        assert status == 200 and payload["ok"]
        assert "thermo" in payload["names"]
        assert client.get_json("/api/code-context")[1]["names"] == payload["names"]

    def test_a_syntax_error_is_an_answer_and_not_a_traceback(self, client):
        status, payload = client.post("/api/code-context", {"source": "x = (\n"})
        assert status == 200, "a snippet that will not compile is about the file"
        assert payload["ok"] is False
        assert "SyntaxError" in payload["error"] and "line 1" in payload["error"]

    def test_a_snippet_that_raises_names_the_line(self, client):
        source = "a = 1\nb = 2\nraise ValueError('no good')\n"
        _, payload = client.post("/api/code-context", {"source": source})
        assert payload["ok"] is False
        assert payload["error"] == "line 3: ValueError: no good"

    def test_a_unit_from_the_context_is_stored_as_a_reference(self, client):
        """Not inlined: the document points at the name the snippet binds."""
        client.post("/api/code-context", {"source": THERMO_CONTEXT})
        name = client.post("/api/unit", {"operation": "Flash"})[1]["name"]
        _, doc = client.get_json("/api/flowsheet")
        unit = next(u for u in doc["flowsheet"]["units"] if u["name"] == name)
        assert unit["constructor"] == {"thermo": {"$ref": "thermo"}}

    def test_the_exported_script_carries_the_snippet(self, client):
        """What is exported has to be what ran."""
        client.post("/api/code-context", {"source": THERMO_CONTEXT})
        name = client.post("/api/unit", {"operation": "Flash"})[1]["name"]
        _, payload = client.get_json("/api/code")
        assert payload["error"] is None
        assert "code context" in payload["source"]
        assert THERMO_CONTEXT.strip() in payload["source"]
        assert "Flash(FlashParams(species_order=['water', 'ethanol']), thermo)" \
            in payload["source"], "the reference is emitted by name, not inlined"
        assert payload["source"].index("thermo = IdealThermo") < \
            payload["source"].index(f"'{name}'"), "defined before it is used"

    def test_the_context_survives_a_save_and_reopen(self, client, tmp_path):
        client.post("/api/code-context", {"source": THERMO_CONTEXT})
        added = client.post("/api/unit", {"operation": "Flash"})[1]["name"]
        client.session.path = tmp_path / "plant.json"
        assert client.post("/api/save")[1]["ok"]

        # a new session asks before running the file's code; trusted, it opens
        reopened = FlowsheetSession(path=tmp_path / "plant.json", trust=True)
        assert reopened.code_context()["source"] == THERMO_CONTEXT
        assert added in [u.name for u in reopened.flowsheet.units], (
            "a $ref that cannot be resolved on load makes the file unopenable"
        )

    def test_a_context_that_is_not_a_string_is_refused(self, client):
        _, payload = client.post("/api/code-context", {"source": 3})
        assert payload["ok"] is False and "string" in payload["error"]


# =============================================================================
# Guarding an endpoint that runs Python
# =============================================================================


class TestSecurity:
    """The code context makes the server `exec`-capable; it must only
    answer the page it served."""

    def test_a_second_server_gets_a_different_token(self, client):
        other = Client(FlowsheetSession(build_flowsheet(None)))
        try:
            assert other.server.token != client.server.token
        finally:
            other.close()

    def test_a_mutating_request_without_the_token_is_refused(self, client):
        request = urllib.request.Request(
            client.base + "/api/code-context",
            data=json.dumps({"source": "import os\nos.environ['X'] = '1'\n"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 403
        assert gui.TOKEN_HEADER in json.loads(excinfo.value.read())["error"]
        assert client.get_json("/api/code-context")[1]["source"] == ""

    def test_a_wrong_token_is_refused(self, client):
        status, payload = client.post(
            "/api/solve", headers={gui.TOKEN_HEADER: "not-the-token"}
        )
        assert status == 403 and payload["ok"] is False

    def test_a_cross_origin_request_is_refused_even_with_the_token(self, client):
        """The whole point: a page on another origin cannot drive this one."""
        status, payload = client.post(
            "/api/solve", headers={"Origin": "http://evil.example"}
        )
        assert status == 403
        assert "evil.example" in payload["error"]

    def test_the_page_s_own_origin_is_accepted(self, client):
        status, payload = client.post("/api/solve",
                                      headers={"Origin": client.base})
        assert status == 200 and payload["ok"]

    def test_a_rebound_host_name_is_refused(self, client):
        """A DNS-rebinding request arrives carrying the attacker's name."""
        port = client.server.server_address[1]
        status, payload = client.post(
            "/api/solve", headers={"Host": f"attacker.example:{port}"}
        )
        assert status == 403 and "Host" in payload["error"]

    def test_reading_needs_no_token(self, client):
        """The guard is on writes; the page fetches the catalog before it
        has done anything."""
        assert client.get("/api/catalog")[0] == 200
        assert client.get("/api/flowsheet")[0] == 200
        assert client.get("/")[0] == 200

    def test_a_rebound_host_name_cannot_read_either(self, client):
        """Rebound, the hostile page is same-origin with the answer: "/"
        would hand it the token, and the API routes the model."""
        port = client.server.server_address[1]
        status, body = client.get("/",
                                  headers={"Host": f"attacker.example:{port}"})
        assert status == 403
        assert client.server.token.encode() not in body


# =============================================================================
# The page's own logic
# =============================================================================


class TestPageLogic:
    """Run the editor's model functions under node.

    Renaming a stream, seeding a new unit and spotting a dangling inlet
    all happen in the browser, and all of them would break the wiring
    silently if they were wrong --- the one thing the Python tests
    above cannot see. node is not a difflow dependency, so this skips
    when it is absent.
    """

    def test_the_canvas_model_functions_behave(self):
        """The wiring is what can be wrong without looking wrong.

        Runs the front end's own ``node --test`` suite over
        ``frontend/src/lib/model/``: the pure functions that turn a
        serialize document into nodes and edges. No build, no browser, and
        no npm install --- these modules import nothing.
        """
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed")

        frontend = pathlib.Path(gui.__file__).parent / "frontend"
        tests = sorted((frontend / "src" / "lib" / "model").glob("*.test.js"))
        if not tests:
            pytest.skip("the front-end source is not in this install")

        result = subprocess.run(
            [node, "--test", *[str(t) for t in tests]],
            capture_output=True, text=True, cwd=frontend,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestDocs:
    """`GET /api/docs/<op>` --- what the inspector shows about a unit.

    The catalog already carries the docstring; the route's job is to
    render it, to answer for any registered name, and to refuse an
    unregistered one in the same shape as every other refusal rather
    than with a traceback.
    """

    def test_a_unit_documents_itself(self, client):
        status, body = client.get_json("/api/docs/CSTR")
        assert status == 200
        assert body["ok"] is True
        assert body["operation"] == "CSTR"
        assert body["symbol"] == "CSTR"
        assert body["html"]
        assert body["equations"], "the CSTR declares its governing equations"
        assert body["assumptions"]
        assert body["numerical_method"]

    def test_an_unknown_operation_is_refused_not_raised(self, client):
        status, body = client.get_json("/api/docs/NotAUnit")
        assert status == 200, "a refusal is an answer, not a broken route"
        assert body["ok"] is False
        assert "NotAUnit" in body["error"]

class TestAnthropicForwarder:
    """`difflow.gui.assistant` --- the shaping and the refusals.

    Nothing here reaches the network: the API is stubbed. What is worth
    testing is that the OpenAI-shaped turns the page sends are
    translated correctly (the system turn is a field, not a message),
    and that every way this can fail comes back as an answer the panel
    can show rather than a traceback.
    """

    @staticmethod
    def _stub(monkeypatch, payload, *, capture=None):
        import io
        from difflow.gui import assistant as module

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(request, timeout=None):
            if capture is not None:
                capture["body"] = json.loads(request.data)
                capture["headers"] = dict(request.headers)
            return _Response(json.dumps(payload).encode())

        monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    def test_the_system_turn_becomes_the_system_field(self, monkeypatch):
        from difflow.gui import assistant

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        seen = {}
        self._stub(monkeypatch, {
            "content": [{"type": "text", "text": "the volume is 1.0 m^3"}],
            "model": "claude-sonnet-5", "usage": {"input_tokens": 900},
        }, capture=seen)

        answer = assistant.answer([
            {"role": "system", "content": "answer only from the brief"},
            {"role": "user", "content": "## Unit\nCSTR"},
        ])
        assert answer["ok"] and answer["text"] == "the volume is 1.0 m^3"
        assert seen["body"]["system"] == "answer only from the brief"
        assert seen["body"]["messages"] == [
            {"role": "user", "content": "## Unit\nCSTR"}
        ], "the system turn must not also be sent as a message"
        assert seen["headers"]["X-api-key"] == "sk-test"
        assert seen["headers"]["Anthropic-version"] == assistant.VERSION

    def test_an_http_error_is_reported_in_the_answer(self, monkeypatch):
        import io

        from difflow.gui import assistant

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        def urlopen(request, timeout=None):
            raise urllib.error.HTTPError(
                assistant.API, 429, "Too Many Requests", {},
                io.BytesIO(b'{"error": {"message": "rate limited"}}'))

        monkeypatch.setattr(assistant.urllib.request, "urlopen", urlopen)
        answer = assistant.answer([{"role": "user", "content": "q"}])
        assert answer["ok"] is False
        assert "429" in answer["error"] and "rate limited" in answer["error"]

    def test_an_oversized_brief_is_refused_before_it_is_paid_for(
            self, monkeypatch):
        from difflow.gui import assistant

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        def urlopen(request, timeout=None):
            raise AssertionError("must not reach the API")

        monkeypatch.setattr(assistant.urllib.request, "urlopen", urlopen)
        answer = assistant.answer(
            [{"role": "user", "content": "x" * (assistant.MAX_CHARS + 1)}])
        assert answer["ok"] is False and str(assistant.MAX_CHARS) in answer["error"]

    def test_a_brief_with_no_question_is_refused(self, monkeypatch):
        from difflow.gui import assistant

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        answer = assistant.answer([{"role": "system", "content": "rules"}])
        assert answer["ok"] is False and "no user turn" in answer["error"]


class TestDocsRendering:
    """`difflow.gui.docs` --- the rendering itself, without a server."""

    def test_a_link_cannot_carry_script(self):
        """The panel is put in with {@html}. Raw HTML is already off, but a
        reST link target is copied into href as written -- a plugin's
        docstring could otherwise run script in the editor's page."""
        from difflow.gui import docs

        if not docs.available():
            pytest.skip("docutils is not installed")
        for target in ("javascript:alert(1)", "JavaScript:alert(1)",
                       "data:text/html,x", "vbscript:x"):
            html, _ = docs.render(f"See `here <{target}>`_.")
            assert "href" not in html, target
            assert "here" in html, "the text of the link stays"
        html, _ = docs.render("See `the book <https://example.org/x>`_.")
        assert 'href="https://example.org/x"' in html
        html, _ = docs.render("A <img src=x onerror=alert(1)> tag.")
        assert "<img" not in html

    def test_an_image_is_not_fetched(self):
        """A docstring image would be a request the user never made."""
        from difflow.gui import docs

        if not docs.available():
            pytest.skip("docutils is not installed")
        for rst in (".. image:: https://t.example/p.png\n   :alt: a plot",
                    ".. figure:: //t.example/p.png\n\n   cap",
                    "x |a| y\n\n.. |a| image:: https://t.example/i.gif",
                    ".. image:: https://t.example/v.mp4"):
            html, _ = docs.render(rst)
            assert "<img" not in html and " src=" not in html, rst
        html, _ = docs.render(".. image:: https://t.example/p.png\n   :alt: a plot")
        assert "[a plot]" in html

    def test_an_svg_is_not_pasted_in_from_disk(self, tmp_path):
        """``:loading: embed`` would inline the file, fetches and all."""
        from difflow.gui import docs

        if not docs.available():
            pytest.skip("docutils is not installed")
        svg = tmp_path / "x.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                       '<image href="https://t.example/p.png"/></svg>')
        html, _ = docs.render(f".. image:: {svg}\n   :loading: embed")
        assert "<svg" not in html and "t.example" not in html

    def test_it_falls_back_to_text_without_docutils(self, monkeypatch):
        from difflow.gui import docs

        real = __import__

        def no_docutils(name, *args, **kwargs):
            if name.startswith("docutils"):
                raise ImportError("no docutils")
            return real(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_docutils)
        html, fmt = docs.render("A <script> & an ampersand.")
        assert fmt == "text"
        assert html.startswith("<pre>")
        assert "&lt;script&gt;" in html and "&amp;" in html


# =============================================================================
# Results and derivatives
# =============================================================================


class TestResults:
    """The solve reports how it solved, not only what it found."""

    def test_a_solve_carries_its_diagnostics(self, thermo):
        answer = FlowsheetSession(build_flowsheet(thermo)).solve()
        assert answer["ok"]
        # No recycle here, so this is the sequential path -- and saying so
        # is the point: the panel must be able to tell the two apart.
        assert answer["method"] == "direct"
        assert answer["converged"] is True
        assert answer["tear_streams"] == []
        assert answer["residual"] == 0.0
        assert answer["tol"] is not None
        assert answer["species"] == SPECIES

    def test_an_edit_makes_the_last_solve_stale(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"]
        assert session.levers()["solved"] is True
        assert session.patch_unit("reactor", {"params": {"V": 2.0}})["ok"]
        # The streams on screen describe the flowsheet as it was.
        assert session.levers()["solved"] is False
        assert session.levers()["outputs"] == []

class TestSensitivity:
    """The derivatives, which is what makes this difflow and not a form."""

    def test_a_lever_that_is_not_one_is_refused_by_name(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.sensitivity(lever="reactor.rate_fn")
        assert not answer["ok"] and "not a lever" in answer["error"]

    def test_asking_both_directions_at_once_is_refused(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert not session.sensitivity()["ok"]
        assert not session.sensitivity(lever="reactor.V",
                                       target="liq.T")["ok"]

    def test_the_routes_answer(self, thermo):
        live = Client(FlowsheetSession(build_flowsheet(thermo)))
        try:
            _, before = live.get_json("/api/levers")
            assert before["ok"] and before["solved"] is False
            assert any(item["key"] == "reactor.V" for item in before["levers"])

            assert live.post("/api/solve")[1]["ok"]
            _, after = live.get_json("/api/levers")
            assert after["solved"] is True and after["outputs"]

            _, forward = live.post("/api/sensitivity", {"lever": "reactor.V"})
            assert forward["ok"] and forward["mode"] == "forward"
        finally:
            live.close()


class TestExport:
    """The ways out of the editor: a script, a document, a drawing."""

    def test_a_flowsheet_with_nothing_in_it_says_so_rather_than_raising(self):
        """An empty string is not an SVG, and a blank download is not an answer."""
        assert FlowsheetSession(None).diagram() == {
            "ok": False, "error": "nothing to draw yet"}

    def test_the_export_routes_answer(self, thermo):
        live = Client(FlowsheetSession(build_flowsheet(thermo)))
        try:
            # All three exports are reads, so none of them needs the token.
            _, svg = live.get_json("/api/diagram")
            assert svg["ok"] and svg["svg"].startswith("<svg")
            _, code = live.get_json("/api/code")
            assert code["error"] is None and "Flowsheet(" in code["source"]
            _, doc = live.get_json("/api/flowsheet")
            assert doc["flowsheet"]["units"]
        finally:
            live.close()

class TestPlanning:
    """Delta vectors, which is what a planning system asks difflow for.

    The load-bearing test here is the finite-difference one: everything
    else in this file can be wrong in a way a user notices, and a
    Jacobian cannot -- it leaves as a table of numbers and is priced
    against by someone who never sees the flowsheet.
    """

    LEVERS = ["reactor.V", "feed:feed.total_flow"]
    OUTPUTS = ["liq.F_ethanol", "vap.total_flow"]

    def test_an_empty_pick_is_refused_with_a_sentence(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        assert session.linearize([], ["liq.F_ethanol"]) == {
            "ok": False, "error": "pick at least one lever and one output"}
        assert session.linearize(["reactor.V"], [])["ok"] is False

    def test_a_radius_that_is_not_a_positive_number_is_refused(self, thermo):
        """Not linearized into a trust region of NaN or inverted bounds."""
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        answer = session.linearize(["reactor.V"], ["liq.F_ethanol"], radius=float("nan"))
        assert answer["ok"] is False and "radius" in answer["error"]
        assert "planning" not in session.flowsheet.view

    def test_downloading_before_linearizing_says_so(self, thermo):
        assert FlowsheetSession(build_flowsheet(thermo)).linearization_files(
            "json") == {"ok": False, "error": "nothing linearized yet"}

    def test_the_route_is_guarded_because_it_writes(self, thermo):
        """It persists the selection, so it goes through the token check."""
        live = Client(FlowsheetSession(build_flowsheet(thermo)))
        try:
            status, payload = live.post(
                "/api/linearize", {"u": self.LEVERS, "y": self.OUTPUTS},
                headers={gui.TOKEN_HEADER: "wrong"})
            assert status == 403 and payload["ok"] is False

            live.post("/api/solve")
            status, answer = live.post(
                "/api/linearize", {"u": self.LEVERS, "y": self.OUTPUTS})
            assert status == 200 and answer["ok"], answer
            assert answer["table"].startswith("delta vectors for block")

            _, files = live.post("/api/linearize/files", {"format": "csv"})
            assert files["ok"] and files["files"]
        finally:
            live.close()

# =============================================================================
# The console
# =============================================================================


class TestConsole:
    """A Python prompt over the objects the editor is already holding.

    The thing worth testing is not that `1 + 1` is 2. It is that `fs`
    is the *same* flowsheet the canvas has --- a console over a copy
    would answer questions about a model nobody is looking at --- and
    that the session notices when a cell has moved it.
    """

    def test_the_route_is_guarded_like_every_other_exec(self, client):
        status, body = client.send("POST", "/api/console", {"source": "1"},
                                   headers={gui.TOKEN_HEADER: "wrong"})
        assert status == 403 and body["ok"] is False
        status, body = client.send("POST", "/api/console", {"source": "1 + 1"})
        assert status == 200
        assert body["outputs"][-1]["text"] == "2"

class TestConsoleFigures:
    """The seam plots will arrive through, exercised without matplotlib."""

    def test_a_broken_hook_does_not_eat_the_result(self):
        """A renderer that fails must not lose a number computed correctly."""
        def broken(ns):
            raise RuntimeError("no display")

        answer = console.Console(display_hooks=[broken]).run("6 * 7")
        assert answer["error"] is None
        assert answer["outputs"][0]["text"] == "42"
        assert "no display" in answer["outputs"][1]["text"]
        assert answer["outputs"][1]["stream"] == "stderr"

# =============================================================================
# Starting up
# =============================================================================


class TestPortInUse:
    """Bind failures are the one startup error a user meets by accident.

    Leaving them raw means a traceback out of ``socketserver`` whose only
    content is ``[Errno 48] Address already in use`` --- no port, no
    program name, no next move.
    """

    def taken_port(self):
        """A port with a live listener on it, released at teardown."""
        held = socket.socket()
        held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.bind((gui.HOST, 0))
        held.listen(1)
        return held, held.getsockname()[1]

    def test_a_taken_port_is_a_message_and_an_exit_code(self, capsys):
        held, port = self.taken_port()
        try:
            status = server.main(["--port", str(port), "--no-browser"])
        finally:
            held.close()

        assert status == 1, "a failed start must not report success"
        said = capsys.readouterr().err
        assert str(port) in said, "the reader needs to know which port"
        assert "--port" in said, "and how to get past it"

    def test_another_oserror_still_raises(self, monkeypatch):
        """Only the actionable one is swallowed; the rest keep their traceback."""
        def refuse(**kwargs):
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(server, "serve", refuse)
        with pytest.raises(OSError) as caught:
            server.main(["--no-browser"])
        assert caught.value.errno == errno.EACCES

# =============================================================================
# Renaming a stream, and the ports of a unit that has as many as it likes
# =============================================================================


class TestRenamingAStream:
    """The name is the wiring, which is why this is not a label edit.

    `connect` already renames an inlet -- that is *how* a wire is made
    in difflow -- so the mechanism was there and only the door was
    missing. The door has to be narrower than the mechanism: two of the
    renames `connect` performs on purpose are, asked for by a user,
    silently something other than a rename.
    """

    def test_an_intermediate_stream_moves_at_both_ends(self, client):
        """Or the graph quietly splits into two that each look fine."""
        assert client.patch("/api/stream/rx", {"name": "effluent"})[1]["ok"]
        fs = client.session.flowsheet
        assert edit.unit(fs, "reactor").outlet_names == ["effluent"]
        assert edit.unit(fs, "flash").inlet_names == ["effluent"]
        # And it still solves, which is the whole claim.
        assert client.post("/api/solve", {})[1]["ok"]

    def test_a_name_another_stream_has_is_refused(self, client):
        """Because that is a connection, and it should be drawn as one.

        `rename_stream` would do it and produce a flowsheet where the
        reactor's inlet and the flash's outlet are the same stream --
        which is a wire nobody drew.
        """
        _, answer = client.patch("/api/stream/feed", {"name": "rx"})
        assert answer["ok"] is False
        assert "already a stream" in answer["error"]
        assert "connection" in answer["error"]
        assert "feed" in client.session.flowsheet.feeds

    def test_a_name_that_is_not_an_identifier_is_refused(self, client):
        """It would survive here and fail in `codegen`, hours later."""
        for bad in ("two words", "3rd", "liq-out", ""):
            _, answer = client.patch("/api/stream/feed", {"name": bad})
            assert answer["ok"] is False, bad
        assert "feed" in client.session.flowsheet.feeds

    def test_renaming_a_stream_that_is_not_there_says_so(self, client):
        _, answer = client.patch("/api/stream/nope", {"name": "x"})
        assert answer["ok"] is False
        assert "no stream called 'nope'" in answer["error"]

class TestAUnitThatTakesAsManyInletsAsItIsGiven:
    """A mixer's inlet count is a property of the flowsheet, not the class.

    Which is why it is the canvas's to change, and why until now the
    only way to get a third inlet on a mixer was to write the JSON by
    hand.
    """

    @pytest.fixture
    def empty(self):
        live = Client(FlowsheetSession())
        live.post("/api/species", {"species": SPECIES})
        yield live
        live.close()

    def variadic_names(self):
        from difflow.catalog import catalog

        return {n for n, s in catalog().items() if s.to_dict()["ports"]["variadic"]}

    def test_an_inlet_is_added_and_removed(self, empty):
        empty.post("/api/unit", {"operation": "Mixer"})
        status, added = empty.post("/api/inlet", {"unit": "mixer"})
        assert status == 200
        assert added == {"ok": True, "kind": "inlet", "stream": "mixer_in3"}
        assert edit.unit(empty.session.flowsheet, "mixer").inlet_names == [
            "mixer_in", "mixer_in2", "mixer_in3"
        ]
        _, gone = empty.delete("/api/inlet", {"unit": "mixer",
                                              "stream": "mixer_in3"})
        assert gone == {"ok": True, "kind": "inlet", "stream": "mixer_in3"}
        assert edit.unit(empty.session.flowsheet, "mixer").inlet_names == [
            "mixer_in", "mixer_in2"
        ]

    def test_a_unit_with_fixed_ports_refuses_both_verbs(self, empty):
        empty.post("/api/unit", {"operation": "Heater"})
        _, refused = empty.post("/api/inlet", {"unit": "heater"})
        assert refused["ok"] is False
        assert "exactly 1 inlet" in refused["error"]
        _, no = empty.delete("/api/inlet", {"unit": "heater",
                                            "stream": "heater_in"})
        assert no["ok"] is False and "fixed set of inlets" in no["error"]

    def test_the_last_inlet_cannot_be_removed(self, empty):
        """A mixer with no inlets is not a smaller mixer."""
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.delete("/api/inlet", {"unit": "mixer", "stream": "mixer_in2"})
        _, refused = empty.delete("/api/inlet", {"unit": "mixer",
                                                 "stream": "mixer_in"})
        assert refused["ok"] is False
        assert "Delete the unit instead" in refused["error"]

    def test_a_fed_inlet_is_refused_rather_than_cascaded(self, empty):
        """Deleting the port would take the feed's conditions with it.

        Two edits, and undoing the first does not bring back the second:
        the temperature, pressure and per-species flows are gone. So the
        refusal names what is in the way.
        """
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.post("/api/feed", {"name": "mixer_in2", "T": 300.0})
        _, refused = empty.delete("/api/inlet", {"unit": "mixer",
                                                 "stream": "mixer_in2"})
        assert refused["ok"] is False and "is a feed" in refused["error"]
        assert "mixer_in2" in empty.session.flowsheet.feeds

    def test_a_wired_inlet_is_refused(self, empty):
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.post("/api/unit", {"operation": "Heater"})
        empty.post("/api/connect", {"source": "heater", "outlet": "heater_out",
                                    "target": "mixer", "inlet": "mixer_in2"})
        _, refused = empty.delete("/api/inlet", {"unit": "mixer",
                                                 "stream": "heater_out"})
        assert refused["ok"] is False
        assert "comes from 'heater'" in refused["error"]

# =============================================================================
# The editor's lifetime, and the links out to the book
# =============================================================================


class TestTheEditorStopsWithItsPage:
    """Closing the tab should give the port back.

    The editor is a Python process and a browser tab, and to the person
    using it they are one thing: the complaint that started this was
    that closing the tab left port 8756 held by a server nobody could
    see, so the next ``difflow gui`` refused to start. Nothing in HTTP
    says when a page has gone, so the page says so --- a ping while it
    is open, a farewell as it unloads --- and :class:`server.Lifetime`
    is what listens.
    """

    def clock(self):
        """A hand-wound monotonic clock, so no test waits out a grace."""
        now = [1000.0]
        return now, lambda: now[0]

    def test_a_page_that_stops_pinging_ages_out(self):
        now, clock = self.clock()
        life = server.Lifetime(grace=10, clock=clock)
        life.ping("tab-a")
        now[0] += 9
        assert not life.expired(), "still inside the grace"
        now[0] += 2
        assert life.expired()

    def test_the_farewell_is_what_makes_it_prompt(self):
        """Without it the port comes back a grace later; with it, a linger later."""
        now, clock = self.clock()
        life = server.Lifetime(grace=90, linger=5, clock=clock)
        life.ping("tab-a")
        assert not life.expired()
        life.bye("tab-a")
        now[0] += 6
        assert life.expired(), "no waiting out 90 seconds for a closed tab"

    def test_the_routes_reach_the_lifetime(self, client):
        """ping / bye / quit, over the wire, as the page sends them."""
        assert client.post("/api/ping", {"client": "tab-a"})[1]["ok"]
        assert not client.server.lifetime.expired()
        assert client.post("/api/bye", {"client": "tab-a"})[1]["ok"]
        life = client.server.lifetime
        assert not life.expired(), "a farewell alone stops nothing; it may be a reload"
        life.linger = 0
        assert life.expired()

        status, answer = client.post("/api/quit")
        assert status == 200, "answered before anything stops, or the page sees a drop"
        assert answer == {"ok": True, "stopped": True}
        assert client.server.lifetime.asked

    def test_stopping_the_editor_is_not_a_GET(self, client):
        """A GET that stops the server is a link another page could embed."""
        for path in ("/api/quit", "/api/ping", "/api/bye"):
            status, _ = client.get(path)
            assert status == 404, f"{path} answered a GET"
        assert not client.server.lifetime.asked

class TestTheHeaderLinks:
    """Where to read more, without leaving the editor to go and find it."""

    def test_about_carries_the_project_urls(self, client):
        status, about = client.get_json("/api/about")
        assert status == 200 and about["ok"]
        for key in ("repository", "documentation"):
            assert about["links"][key].startswith("https://"), key

class TestWhereTheBookTalksAboutAUnit:
    """Every palette entry offers the documentation for that unit.

    Resolved against ``static/docs-index.json`` rather than written down:
    a hand-kept table of 87 operations against prose that gets
    reorganised is wrong within a release, and wrong silently, because a
    link to a renamed heading still returns 200 and lands at the top of
    the page. :mod:`tests.test_doclinks` is where the resolution rules
    themselves are pinned; what is here is the editor's side of it.
    """

    def test_a_unit_with_its_own_section_lands_on_it(self):
        url = doclinks.url_for("CSTR")
        assert url.startswith(doclinks.BASE_URL)
        assert "unit-operations-chemical.html#" in url

    def test_a_name_the_book_never_uses_gets_no_link(self):
        """Reported as an absence rather than papered over with a home page."""
        assert doclinks.url_for("NotAnOperationAtAll") is None
        assert doclinks.url_for("") is None

    def test_a_missing_index_is_no_links_rather_than_no_editor(self, monkeypatch):
        """A source checkout that has never run the front-end build.

        Patched at :func:`docs_index.load` rather than at its ``INDEX``
        constant: the path is a default argument, bound once at
        definition, so rebinding the module attribute would leave the
        real index still being read and this test passing for no reason.
        """
        from difflow.gui import doclinks as dl

        monkeypatch.setattr(dl.docs_index, "load", lambda *a, **k: None)
        dl.resolve.cache_clear()
        try:
            assert dl.url_for("CSTR") is None
        finally:
            dl.resolve.cache_clear()


class TestSayingWhatHoldsThePort:
    """"In use" is half a sentence; the reader's next move needs the rest."""

    def test_the_message_survives_a_machine_with_neither_helper(self, monkeypatch):
        """It runs on the way to an error; it must not become one."""
        monkeypatch.setattr(server, "_run", lambda argv, timeout: "")
        said = server.port_in_use_message(8756, "difflow gui")
        assert "8756" in said and "--port" in said
        assert "pid" not in said, "nothing was found, so nothing is claimed"

    def test_ss_output_is_read_when_lsof_says_nothing(self, monkeypatch):
        """Linux without lsof. Parsed here rather than only in the wild."""
        sample = (
            'LISTEN 0 5 127.0.0.1:8756 0.0.0.0:* '
            'users:(("python3",pid=4242,fd=3))\n'
        )
        monkeypatch.setattr(
            server, "_run",
            lambda argv, timeout: sample if argv[0] == "ss" else "")
        found = server.listener_on(8756)
        assert found["pid"] == 4242
        assert found["name"] == "python3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
