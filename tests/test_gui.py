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
import math
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
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
    context,
    doclinks,
    edit,
    make_server,
    sensitivity,
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

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path) as response:
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
    def test_the_page_is_served(self, client):
        status, body = client.get("/")
        assert status == 200
        assert b"<title>difflow editor</title>" in body

    def test_the_catalog_lists_registered_operations(self, client):
        status, catalog = client.get_json("/api/catalog")
        assert status == 200
        assert "CSTR" in catalog and "Flash" in catalog
        assert catalog["Flash"]["ports"]["n_outlets"] == 2, (
            "the palette needs port arity to draw anything"
        )

    def test_the_flowsheet_is_served_as_the_serialize_document(self, client):
        status, doc = client.get_json("/api/flowsheet")
        assert status == 200
        assert [u["name"] for u in doc["flowsheet"]["units"]] == ["reactor", "flash"]
        assert doc["flowsheet"]["format_version"] == serialize.FORMAT_VERSION

    def test_the_served_document_carries_canvas_positions(self, client):
        """A canvas needs coordinates and a flowsheet carries none."""
        status, doc = client.get_json("/api/flowsheet")
        assert status == 200
        nodes = doc["flowsheet"]["view"]["nodes"]
        assert {"reactor", "flash", "feed:feed"} <= set(nodes)
        assert nodes["reactor"]["x"] < nodes["flash"]["x"], "left to right"

    def test_opening_a_flowsheet_does_not_give_it_a_view(self, client):
        """Positions are served, not adopted; the file is untouched on open."""
        client.get_json("/api/flowsheet")
        assert client.session.flowsheet.view == {}

    def test_positions_the_document_already_has_are_kept(self, client):
        client.session.flowsheet.view = {
            "nodes": {"reactor": {"x": 999.0, "y": 999.0}}
        }
        _, doc = client.get_json("/api/flowsheet")
        nodes = doc["flowsheet"]["view"]["nodes"]
        assert nodes["reactor"] == {"x": 999.0, "y": 999.0}, (
            "a hand-placed canvas must not be re-laid-out under the user"
        )
        assert "flash" in nodes, "but a node it never placed still needs one"

    def test_the_assistant_brief_is_served(self, client):
        """The one route whose request is a question, not a resource."""
        status, pack = client.get_json(
            "/api/context?kind=block&name=reactor&q=what+is+the+volume")
        assert status == 200 and pack["ok"]
        assert "Operation: CSTR" in pack["prompt"]
        assert pack["prompt"].endswith("what is the volume")

    def test_the_brief_defaults_to_the_whole_flowsheet(self, client):
        status, pack = client.get_json("/api/context")
        assert status == 200 and pack["ok"]
        assert pack["kind"] == "flowsheet"
        assert "reactor (CSTR)" in pack["prompt"]

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

    def test_forwarding_is_a_mutating_route(self, client, monkeypatch):
        """It spends money, so it goes through the token check."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
        status, payload = client.post(
            "/api/assistant", {"messages": []},
            headers={gui.TOKEN_HEADER: "wrong"})
        assert status == 403 and payload["ok"] is False

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

    def test_a_built_asset_is_served_with_its_content_type(self, client, tmp_path):
        asset = gui.STATIC / "_probe.js"
        asset.write_text("export const x = 1;\n")
        try:
            status, body = client.get("/_probe.js")
        finally:
            asset.unlink()
        assert status == 200
        assert body == b"export const x = 1;\n"

    def test_a_missing_asset_is_a_404(self, client):
        assert client.get("/nothing-was-ever-built-here.js")[0] == 404

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

    def test_the_module_entry_point_still_runs(self):
        """``python -m difflow.gui`` is the documented way in."""
        result = subprocess.run(
            [sys.executable, "-m", "difflow.gui", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--no-browser" in result.stdout


# =============================================================================
# Editing
# =============================================================================


class TestEditing:
    def test_an_edit_changes_the_solution(self, client):
        """An editor whose edits do not reach the model is worse than none."""
        _, before = client.post("/api/solve")
        _, doc = client.get_json("/api/flowsheet")

        doc["flowsheet"]["units"][0]["params"]["V"] = 5.0
        status, payload = client.post("/api/flowsheet", doc["flowsheet"])
        assert status == 200 and payload["ok"]

        _, after = client.post("/api/solve")
        assert after["ok"]
        assert after["streams"]["rx"]["F_ethanol"] > before["streams"]["rx"]["F_ethanol"], (
            "a five-fold larger reactor must convert more"
        )

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


# =============================================================================
# Incremental routes
# =============================================================================


class TestIncrementalRoutes:
    """The verbs the canvas uses. What they do is covered in test_gui_edit."""

    def test_a_patch_changes_one_parameter(self, client):
        status, payload = client.patch("/api/unit/reactor", {"params": {"V": 5.0}})
        assert status == 200 and payload["ok"]
        assert float(client.session.flowsheet.units[0].operation.params.V) == 5.0

    def test_a_patch_leaves_the_other_units_alone(self, client):
        flash = client.session.flowsheet.units[1].operation
        client.patch("/api/unit/reactor", {"params": {"V": 5.0}})
        assert client.session.flowsheet.units[1].operation is flash

    def test_a_patched_flowsheet_still_solves(self, client):
        before = client.post("/api/solve")[1]["streams"]["rx"]["F_ethanol"]
        client.patch("/api/unit/reactor", {"params": {"V": 5.0}})
        after = client.post("/api/solve")[1]["streams"]["rx"]["F_ethanol"]
        assert after > before, "a five-fold larger reactor must convert more"

    def test_a_patch_to_a_unit_that_is_not_there_is_a_refusal(self, client):
        status, payload = client.patch("/api/unit/nope", {"params": {}})
        assert status == 200, "a wrong name is an answer about the flowsheet"
        assert payload["ok"] is False and "nope" in payload["error"]

    def test_a_name_with_a_space_survives_the_url(self, client):
        client.patch("/api/unit/reactor", {"name": "hot reactor"})
        status, payload = client.patch("/api/unit/hot%20reactor",
                                       {"params": {"V": 2.0}})
        assert status == 200 and payload["ok"]

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

    def test_positions_come_back_in_the_document(self, client):
        client.post("/api/layout", {"nodes": {"reactor": {"x": 40, "y": 12}}})
        _, doc = client.get_json("/api/flowsheet")
        assert doc["flowsheet"]["view"]["nodes"]["reactor"] == {"x": 40, "y": 12}

    def test_one_moved_node_does_not_unplace_the_others(self, client):
        """Dragging one box must not scatter the rest of the flowsheet."""
        _, before = client.get_json("/api/flowsheet")
        placed = set(before["flowsheet"]["view"]["nodes"])
        client.post("/api/layout", {"nodes": {"reactor": {"x": 40, "y": 12}}})
        _, after = client.get_json("/api/flowsheet")
        assert set(after["flowsheet"]["view"]["nodes"]) == placed
        for name in placed - {"reactor"}:
            assert after["flowsheet"]["view"]["nodes"][name] == \
                before["flowsheet"]["view"]["nodes"][name]

    def test_an_unrouted_verb_is_a_404(self, client):
        assert client.patch("/api/nothing")[0] == 404
        assert client.delete("/api/nothing")[0] == 404

    def test_malformed_json_on_a_patch_is_reported(self, client):
        request = urllib.request.Request(
            client.base + "/api/unit/reactor", data=b"{not json",
            headers={"Content-Type": "application/json",
                     gui.TOKEN_HEADER: client.server.token}, method="PATCH",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400


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

    def test_an_irreversible_reaction_puts_inf_in_the_document(self, client):
        """Guards the premise: without inf present the test above is vacuous."""
        _, doc = client.get_json("/api/flowsheet")
        rate_params = doc["flowsheet"]["units"][0]["params"]["rate_params"]
        assert rate_params["K_eq"]["$array"] == ["Infinity"]

    def test_the_round_trip_restores_the_value(self, client):
        _, doc = client.get_json("/api/flowsheet")
        status, payload = client.post("/api/flowsheet", doc["flowsheet"])
        assert status == 200 and payload["ok"]
        rate_params = client.session.flowsheet.units[0].operation.params.rate_params
        assert float(rate_params["K_eq"][0]) == float("inf")

    def test_json_safe_and_restore_are_inverse(self):
        value = {"a": [1.0, float("inf"), -float("inf")], "b": {"c": 2.0}}
        assert _json_safe(value) == {"a": [1.0, "Infinity", "-Infinity"],
                                     "b": {"c": 2.0}}
        assert _json_restore(_json_safe(value)) == value

    def test_nan_survives_the_round_trip(self):
        restored = _json_restore(_json_safe({"x": float("nan")}))
        assert restored["x"] != restored["x"]

    def test_ordinary_strings_are_left_alone(self):
        assert _json_restore({"phase": "vapor"}) == {"phase": "vapor"}


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
        assert not result["ok"] and "path" in result["error"]

    def test_an_empty_session_is_an_empty_flowsheet(self):
        """Not `None`. See `TestAnEmptyEditor` for why.

        Everything a session can be asked still answers over one: it
        serializes, it emits code that runs, and solving nothing succeeds
        at nothing. The only verb that refuses is the diagram, because an
        empty SVG downloads as a file that opens onto nothing.
        """
        session = FlowsheetSession()
        document = session.document()
        assert document["flowsheet"]["units"] == []
        assert document["species"] == []
        assert session.solve() == {
            "ok": True, "streams": {}, "species": [], "converged": True,
            "iterations": 0, "method": "direct", "residual": 0.0,
            "tol": 1e-08, "tear_streams": [], "pending": [],
        }
        assert session.code()["error"] is None
        assert "Flowsheet(species_order=[]" in session.code()["source"]
        assert session.diagram() == {"ok": False, "error": "nothing to draw yet"}


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


class TestPalette:
    def test_the_catalog_says_what_a_form_can_build(self, client):
        """The palette dims what it cannot add; it needs to be told which."""
        _, catalog = client.get_json("/api/catalog")
        assert catalog["Heater"]["buildable"]
        assert not catalog["Flash"]["buildable"], "Flash needs a thermo object"
        assert catalog["Flash"]["constructor_extras"] == ["thermo"]
        assert not catalog["CSTR"]["buildable"], "CSTR needs a rate law"
        assert catalog["CSTR"]["constructor_extras"] == []

    def test_every_buildable_operation_declares_its_ports(self, client):
        """A unit whose arity is unknown cannot be given outlet names."""
        _, catalog = client.get_json("/api/catalog")
        for name, spec in catalog.items():
            if not spec["buildable"]:
                continue
            ports = spec["ports"]
            assert ports["variadic"] or ports["n_inlets"] is not None, name

    def test_every_entry_carries_where_to_read_about_it(self, client):
        """The palette offers a documentation link; the server resolves it.

        `tests/test_doclinks.py` is what holds the prose to having
        somewhere for each of these to point (#228); this is only that
        the catalog the page fetches actually carries the answer.
        """
        from difflow.gui import doclinks

        _, catalog = client.get_json("/api/catalog")
        missing = [n for n, spec in catalog.items() if not spec["docs_url"]]
        assert missing == []
        assert catalog["Heater"]["docs_url"] == doclinks.url_for("Heater")
        assert catalog["Heater"]["docs_url"].endswith(
            "unit-operations-chemical.html#heater")


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

    def test_the_catalog_names_the_fields_and_not_just_the_extras(self, client):
        """`constructor_extras` is empty for a CSTR; the rate law is not."""
        _, catalog = client.get_json("/api/catalog")
        assert catalog["CSTR"]["constructor_extras"] == []
        assert catalog["CSTR"]["needs"] == ["rate_fn", "stoich", "rate_params"]
        assert catalog["Flash"]["needs"] == ["thermo"]
        assert catalog["Heater"]["needs"] == []

    def test_a_required_string_blocks_a_drop_the_class_calls_buildable(self, client):
        """The regression. No callable, no constructor object, still undroppable.

        ``AbsorberParams.solvent`` is a ``str``, so ``is_buildable`` --- which
        looks for required *callables* and constructor arguments --- says
        yes, and nothing can invent a solvent name.
        """
        pytest.importorskip("difflow_cc")
        _, catalog = client.get_json("/api/catalog")
        spec = catalog.get("AmineAbsorber")
        if spec is None:
            pytest.skip("difflow_cc not registered")
        assert spec["needs"] == ["solvent"]
        assert spec["buildable"] is False

    def test_the_flag_and_the_parking_cannot_disagree(self, client):
        """The invariant, over every operation the catalog offers.

        A non-empty ``needs`` must mean the drop parks as a pending node,
        and an empty one must mean it is *answered* --- either it lands
        built, or the class refuses on its own terms with a message about
        the model. What it must never mean is a 400, a traceback, or a
        claim about difflow versions.

        Empty ``needs`` promises a clean answer rather than a successful
        one on purpose: a class can validate whatever it likes, and
        ``Transformer`` rejecting ``tap=1, shift=0`` as "this is a line"
        is the model being right. Nothing short of constructing one can
        know that in advance.

        Checked exhaustively rather than by sampling, because the cases
        that matter are the ones nobody thought to name --- a required
        ``str``, an annotation in quotes.
        """
        _, catalog = client.get_json("/api/catalog")
        wrong = []
        for name, spec in catalog.items():
            status, answer = client.post("/api/unit", {"operation": name})
            if status != 200:
                wrong.append((name, f"HTTP {status}"))
                continue
            if spec["needs"] and not answer.get("pending"):
                wrong.append((name, "flagged, but built anyway"))
            if not spec["needs"] and answer.get("pending"):
                wrong.append((name, "parked, but the palette said it was ready"))
            if not spec["needs"] and not answer["ok"]:
                if "different version" in (answer.get("error") or ""):
                    wrong.append((name, answer["error"]))
            if answer["ok"]:
                client.delete(f"/api/unit/{answer['name']}")
        assert not wrong

    def test_a_class_that_refuses_its_own_parameters_is_quoted(self, client):
        """`Transformer` explains itself better than any generic message."""
        _, catalog = client.get_json("/api/catalog")
        if "Transformer" not in catalog:
            pytest.skip("difflow_power not registered")
        assert catalog["Transformer"]["needs"] == []
        status, answer = client.post("/api/unit", {"operation": "Transformer"})
        assert status == 200, "the class refusing is an answer, not a bad request"
        assert answer["ok"] is False
        assert "this is a line" in answer["error"]

    def test_the_hint_names_the_field_and_blames_no_version(self, client):
        _, answer = client.post("/api/unit", {"operation": "CSTR"})
        assert answer["pending"] is True
        assert "rate_fn" in answer["needs"]
        assert "rate_fn" in answer["hint"]
        assert "different version of difflow" not in answer["hint"]
        assert "mass_action_kinetics" in answer["hint"], "offer the easy route"

    def test_the_hint_separates_code_from_data(self, client):
        """Calling a required `str` "code" tells the reader something false."""
        pytest.importorskip("difflow_cc")
        _, answer = client.post("/api/unit", {"operation": "AmineAbsorber"})
        if not answer.get("pending"):
            pytest.skip("difflow_cc not registered")
        assert "no way to guess" in answer["hint"]
        assert "code rather than data" not in answer["hint"]

    def test_a_binding_unblocks_the_unit_that_wanted_it(self, client):
        """The round trip the message promises has to actually work."""
        assert client.get_json("/api/catalog")[1]["Flash"]["needs"] == ["thermo"]
        assert client.post("/api/unit", {"operation": "Flash"})[1]["pending"]

        client.post("/api/code-context", {"source": THERMO_CONTEXT})

        assert client.get_json("/api/catalog")[1]["Flash"]["needs"] == []
        assert not client.post(
            "/api/unit", {"operation": "Flash"})[1].get("pending")

    def test_a_bare_value_in_the_context_counts_as_a_binding(self, client):
        """`solvent = "MEA"` is matched by field name, so the hint is true."""
        pytest.importorskip("difflow_cc")
        if "AmineAbsorber" not in client.get_json("/api/catalog")[1]:
            pytest.skip("difflow_cc not registered")
        client.post("/api/code-context", {"source": "solvent = 'MEA'\n"})
        assert client.get_json("/api/catalog")[1]["AmineAbsorber"]["needs"] == []
        assert client.post("/api/unit", {"operation": "AmineAbsorber"})[1]["ok"]


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

    def test_the_document_carries_the_parked_nodes(self, client):
        client.post("/api/unit", {"operation": "CSTR",
                                  "position": {"x": 10, "y": 20}})
        _, payload = client.get_json("/api/flowsheet")
        assert [p["name"] for p in payload["pending"]] == ["cstr"]
        # On the flowsheet, like any other node, and placed in the view
        # where any other node is placed. It has to be: a node the
        # flowsheet has never heard of has no ports, and a node with no
        # ports cannot be wired to the unit that feeds it.
        assert "cstr" in [u["name"] for u in payload["flowsheet"]["units"]]
        assert payload["flowsheet"]["view"]["nodes"]["cstr"] == {"x": 10.0,
                                                                "y": 20.0}
        [spec] = [u for u in payload["flowsheet"]["units"] if u["name"] == "cstr"]
        # And it is saved saying so, rather than as a CSTR that isn't one.
        assert spec["incomplete"]["needs"] == ["rate_fn", "stoich",
                                               "rate_params"]
        assert spec["inlets"] and spec["outlets"]

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

    def test_a_parked_node_is_deleted_like_any_other(self, client):
        """Like any other, now literally: one path, not two."""
        name = client.post("/api/unit", {"operation": "CSTR"})[1]["name"]
        status, answer = client.delete(f"/api/unit/{name}")
        assert status == 200 and answer["ok"]
        _, payload = client.get_json("/api/flowsheet")
        assert payload["pending"] == []
        assert name not in [u["name"] for u in payload["flowsheet"]["units"]]

    def test_the_boilerplate_route_needs_the_token(self, client):
        """It writes nothing, and it is still a POST.

        The answer carries the code context back inside ``merged``, so an
        unrelated page that could call it would be reading the session's
        source. A GET would be reachable from one.
        """
        status, payload = client.post(
            "/api/boilerplate", {"operation": "Flash"},
            headers={gui.TOKEN_HEADER: "wrong"},
        )
        assert status == 403 and payload["ok"] is False

    def test_an_unregistered_operation_is_refused(self, client):
        _, answer = client.post("/api/boilerplate", {"operation": "Teleporter"})
        assert answer["ok"] is False and "registered" in answer["error"]

    def test_an_unfinished_unit_can_be_wired_at_once(self, client):
        """The bug this whole arrangement exists to fix.

        A reactor cannot be built until its rate law exists, and the
        rate law is code the user writes later --- so for as long as the
        unfinished unit was held off the flowsheet it had no ports, and
        a compressor upstream of it had nothing to connect to. Wiring
        first and finishing afterwards is the order people work in.
        """
        name = client.post("/api/unit", {"operation": "CSTR"})[1]["name"]
        assert client.post("/api/connect", {
            "source": "flash", "outlet": "liq",
            "target": name, "inlet": f"{name}_in",
        })[1] == {"ok": True, "kind": "arc", "stream": "liq"}
        [unit] = [u for u in client.session.flowsheet.units if u.name == name]
        assert list(unit.inlet_names) == ["liq"]

    def test_finishing_it_keeps_the_wiring(self, client):
        """Built in place, not dropped again.

        A fresh drop would arrive with its own default port names and
        quietly detach whatever had been connected in the meantime.
        """
        name = client.post("/api/unit", {"operation": "Flash",
                                         "position": {"x": 3, "y": 4}})[1]["name"]
        assert client.post("/api/connect", {
            "source": "flash", "outlet": "vap",
            "target": name, "inlet": f"{name}_in",
        })[1]["ok"]
        _, written = client.post("/api/boilerplate",
                                 {"operation": "Flash", "name": name})
        _, applied = client.post("/api/code-context",
                                 {"source": written["merged"]})
        assert applied["promoted"] == [name]
        [unit] = [u for u in client.session.flowsheet.units if u.name == name]
        assert list(unit.inlet_names) == ["vap"], "the wire survived the build"
        assert type(unit.operation).__name__ == "Flash"

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


class TestANumberInTheConstructor:
    """A required number is a placeholder wherever the unit keeps it.

    ``known_params`` has always invented :data:`~difflow.gui.edit.PLACEHOLDER`
    for a required ``Params`` field with no default, so a ``CSTR`` drops
    with ``V = 1.0`` flagged for the user to correct. A unit that builds
    its own ``Params`` from plain arguments --- ``GasPipe(beta)``,
    ``Compressor(ratio)`` --- names an identical ``float`` one line away
    and was refused outright. Nothing chose that asymmetry; the
    placeholder path simply only ran down one of the two.

    What must not follow from fixing it: inventing an *object*, or
    handing the constructor the same number twice.
    """

    @pytest.fixture
    def catalog(self, client):
        return client.get_json("/api/catalog")[1]

    def test_a_plain_float_argument_no_longer_blocks_the_drop(self, client, catalog):
        """`GasPipe(beta)` keeps its number outside any Params difflow can find."""
        if "GasPipe" not in catalog:
            pytest.skip("difflow_gas not registered")
        assert catalog["GasPipe"]["needs"] == []
        status, added = client.post("/api/unit", {"operation": "GasPipe"})
        assert status == 200 and added["ok"], added.get("error")
        assert added["placeholders"] == ["beta"], (
            "an invented number has to be flagged, exactly like a Params field"
        )

    def test_a_number_named_twice_is_supplied_once(self, client, catalog):
        """`Compressor(ratio)` feeds `CompressorParams.ratio`.

        The constructor argument and the field are one number. Answering
        for it in both places hands the builder two values for `ratio`
        and the drop dies in the file-loading path.
        """
        if "Compressor" not in catalog:
            pytest.skip("difflow_gas not registered")
        assert catalog["Compressor"]["needs"] == []
        status, added = client.post("/api/unit", {"operation": "Compressor"})
        assert status == 200 and added["ok"], added.get("error")
        assert added["placeholders"] == ["ratio"], "named once, not twice"

    def test_an_int_argument_gets_an_int(self, client, catalog):
        """`direction` is +1 or -1; a `direction` of 1.0 is a different claim."""
        if "CompressorBoost" not in catalog:
            pytest.skip("difflow_gas not registered")
        status, added = client.post("/api/unit", {"operation": "CompressorBoost"})
        assert status == 200 and added["ok"], added.get("error")
        name = added["name"]
        _, doc = client.get_json("/api/flowsheet")
        unit = next(u for u in doc["flowsheet"]["units"] if u["name"] == name)
        assert unit["constructor"]["direction"] == 1
        assert not isinstance(unit["constructor"]["direction"], float)

    def test_a_tuple_of_numbers_is_still_refused(self, client, catalog):
        """`AffineFlow(signs)` has a length nothing here knows."""
        if "AffineFlow" not in catalog:
            pytest.skip("difflow_gas not registered")
        assert catalog["AffineFlow"]["needs"] == ["signs"], (
            "`const`, `T_k` and `P_pa` are plain numbers; `signs` is not"
        )

    def test_a_non_numeric_argument_is_still_refused(self, client, catalog):
        """The line held: only `_is_number` annotations get a value.

        A `thermo` is an object and a `solvent` is a name; neither has a
        plausible 1.0. `Mixer(species_order)` is deliberately absent from
        this list --- the flowsheet already knows its species order, so
        that one is answered rather than guessed.
        """
        assert catalog["Flash"]["needs"] == ["thermo"]
        if "GroupSeparator" in catalog:
            assert catalog["GroupSeparator"]["needs"] == ["elements"]
        if "LLEEquilibrium" in catalog:
            assert catalog["LLEEquilibrium"]["needs"] == [
                "solutes", "aqueous_carrier", "organic_carrier",
            ]

    def test_a_real_binding_outranks_the_placeholder(self, client, catalog):
        """A number the code context supplies must not be shadowed."""
        if "GasPipe" not in catalog:
            pytest.skip("difflow_gas not registered")
        client.post("/api/code-context", {"source": "beta = 12.5\n"})
        status, added = client.post("/api/unit", {"operation": "GasPipe"})
        assert status == 200 and added["ok"], added.get("error")
        assert "beta" not in added["placeholders"], (
            "a bound value is known, not guessed"
        )

    def test_everything_droppable_saves_and_loads_back(self, client, catalog):
        """The drop must not build a flowsheet the file format cannot hold.

        ``encoded_params`` takes the encoding route so that a parameter
        the editor accepts is one the file can carry. A constructor
        argument reaches the file by a different road, and unblocking
        nine units puts that road under load for the first time.
        """
        from difflow.serialize import from_json, to_json

        unsaveable = []
        for name, spec in catalog.items():
            if spec["needs"]:
                continue
            status, answer = client.post("/api/unit", {"operation": name})
            if status != 200 or not answer["ok"]:
                continue
            try:
                from_json(to_json(client.session.flowsheet))
            except Exception as exc:
                unsaveable.append((name, f"{type(exc).__name__}: {exc}"))
            client.delete(f"/api/unit/{answer['name']}")
        assert not unsaveable


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

    def test_a_drop_before_the_species_are_named_says_which_field(self, empty):
        """Not "no flowsheet loaded", and not silence.

        It lands as a pending node rather than being refused, so the
        sentence is the node's hint --- but it is the same sentence, and
        it still has to name the field and both ways of filling it in.
        """
        _, answer = empty.post("/api/unit", {"operation": "Mixer"})
        assert answer["pending"] is True
        assert "species" in answer["hint"]
        assert "header" in answer["hint"] and "code context" in answer["hint"]

    def test_naming_the_species_unblocks_the_drop(self, empty):
        assert empty.post("/api/species", {"species": SPECIES})[1]["ok"]
        assert empty.get_json("/api/species")[1]["species"] == SPECIES
        answer = empty.post("/api/unit", {"operation": "Mixer"})[1]
        assert answer["ok"], answer
        assert [u.name for u in empty.session.flowsheet.units] == ["mixer"]

    def test_the_code_context_can_name_them_instead(self, empty):
        """`SPECIES` as well as `species_order`, because the starter says so.

        The snippet the Code context panel opens with defines ``SPECIES``.
        A session that only looked for ``species_order`` left the reader
        applying the sample code they were handed and watching the drop
        get refused anyway.
        """
        _, answer = empty.post(
            "/api/code-context", {"source": 'SPECIES = ["water", "ethanol"]\n'}
        )
        assert answer["ok"] and answer["species"] == SPECIES
        assert empty.get_json("/api/flowsheet")[1]["species"] == SPECIES
        assert empty.post("/api/unit", {"operation": "Mixer"})[1]["ok"]

    def test_a_named_list_is_not_overwritten_by_the_code_context(self, empty):
        """Whoever typed in the header meant it."""
        empty.post("/api/species", {"species": ["a", "b"]})
        _, answer = empty.post(
            "/api/code-context", {"source": 'SPECIES = ["water", "ethanol"]\n'}
        )
        assert answer["ok"] and "species" not in answer
        assert empty.get_json("/api/species")[1]["species"] == ["a", "b"]

    @pytest.mark.parametrize(
        "names, why",
        [
            ("water", "must be a list"),
            (["water", "water"], "named twice"),
            (["water", "  "], "needs a name"),
            ([1, 2], "needs a name"),
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

    def test_the_palette_says_what_it_is_waiting_for(self, empty):
        """`needs species_order`, on every row, until the list exists.

        The palette answers `needs` against the session rather than
        against the class, which is what lets it be honest here: a Mixer
        needs nothing of its own and still cannot be built, and a row that
        looked droppable would send the reader to a refusal instead of to
        the field that fixes it. The name is the same one the code context
        would bind, so the two ways of answering it read alike.
        """
        _, catalog = empty.get_json("/api/catalog")
        assert catalog["Mixer"]["needs"] == ["species_order"]
        assert catalog["Mixer"]["buildable"] is False
        empty.post("/api/species", {"species": SPECIES})
        _, catalog = empty.get_json("/api/catalog")
        assert catalog["Mixer"]["needs"] == []
        assert catalog["Mixer"]["buildable"] is True
        assert empty.post("/api/unit", {"operation": "Mixer"})[1]["ok"]

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

    def test_the_defaults_come_back_so_they_can_be_edited(self, client):
        """The page reloads after adding; that is where the fields come from.

        A unit is added with no parameters at all, and the Params
        dataclass fills them in. If they were not written back the new
        unit would show an empty card and be uneditable.
        """
        _, doc = client.get_json("/api/flowsheet")
        doc["flowsheet"]["units"].append(heater_unit())
        client.post("/api/flowsheet", doc["flowsheet"])

        _, after = client.get_json("/api/flowsheet")
        assert set(after["flowsheet"]["units"][-1]["params"]) == {
            "duty", "T_out", "UA", "T_utility", "Cp", "phase"
        }

    def test_the_added_unit_then_solves(self, client):
        _, doc = client.get_json("/api/flowsheet")
        doc["flowsheet"]["units"].append(heater_unit())
        client.post("/api/flowsheet", doc["flowsheet"])

        _, doc = client.get_json("/api/flowsheet")
        doc["flowsheet"]["units"][-1]["params"]["T_out"] = 400.0
        assert client.post("/api/flowsheet", doc["flowsheet"])[1]["ok"]

        _, solved = client.post("/api/solve")
        assert solved["ok"]
        assert solved["streams"]["heater_out"]["T"] == pytest.approx(400.0)

    def test_renamed_streams_rewire_the_model(self, client):
        """What the page's renameStream produces has to solve unchanged."""
        _, before = client.post("/api/solve")
        _, doc = client.get_json("/api/flowsheet")
        units = doc["flowsheet"]["units"]
        units[0]["outlets"] = ["crude"]        # was rx
        units[1]["inlets"] = ["crude"]         # the consumer followed
        assert client.post("/api/flowsheet", doc["flowsheet"])[1]["ok"]

        _, after = client.post("/api/solve")
        assert after["ok"]
        assert "rx" not in after["streams"] and "crude" in after["streams"]
        assert after["streams"]["crude"]["F_ethanol"] == pytest.approx(
            before["streams"]["rx"]["F_ethanol"]
        )

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

KINETICS_CONTEXT = (
    "from difflow import mass_action_kinetics\n"
    "kin = mass_action_kinetics([{'equation': 'water -> ethanol',\n"
    "    'reactants': {'water': 1.0}, 'products': {'ethanol': 1.0},\n"
    "    'rate_params': {'A': 1.0e3, 'Ea': 40_000.0, 'n': 0.0}}],\n"
    "    ['water', 'ethanol'])\n"
)


class TestCodeContext:
    """Half the catalog needs an object, and no form supplies one."""

    def test_a_flowsheet_starts_with_no_context(self, client):
        status, context = client.get_json("/api/code-context")
        assert status == 200
        assert context == {"source": "", "names": [], "error": None}

    def test_a_snippet_defines_names(self, client):
        status, payload = client.post("/api/code-context",
                                      {"source": THERMO_CONTEXT})
        assert status == 200 and payload["ok"]
        assert "thermo" in payload["names"]
        assert client.get_json("/api/code-context")[1]["names"] == payload["names"]

    def test_imported_modules_are_not_offered_as_names(self, client):
        """`import jax` binds a module; nothing in a flowsheet refers to one."""
        client.post("/api/code-context", {"source": "import math\nx = math.pi\n"})
        assert client.get_json("/api/code-context")[1]["names"] == ["x"]

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

    def test_a_failing_snippet_does_not_cost_the_bindings(self, client):
        """A half-typed line must not unbuild the units that depend on one."""
        client.post("/api/code-context", {"source": THERMO_CONTEXT})
        client.post("/api/code-context", {"source": "thermo = (\n"})
        context = client.get_json("/api/code-context")[1]
        assert context["names"] == ["IdealThermo", "get_species_data", "thermo"]
        assert context["source"] == THERMO_CONTEXT

    def test_the_context_travels_in_the_document(self, client):
        client.post("/api/code-context", {"source": THERMO_CONTEXT})
        _, doc = client.get_json("/api/flowsheet")
        assert doc["flowsheet"]["view"]["code_context"] == THERMO_CONTEXT

    def test_an_empty_snippet_clears_it(self, client):
        client.post("/api/code-context", {"source": THERMO_CONTEXT})
        assert client.post("/api/code-context", {"source": "   "})[1]["ok"]
        assert client.get_json("/api/code-context")[1] == {
            "source": "", "names": [], "error": None
        }
        assert "code_context" not in client.session.flowsheet.view

    def test_a_flash_is_unbuildable_until_a_thermo_exists(self, client):
        status, parked = client.post("/api/unit", {"operation": "Flash"})
        assert status == 200 and parked["pending"] is True
        assert "code context" in parked["hint"], (
            "the node has to say where the missing object comes from"
        )

        # The context both promotes the node that was waiting and lets
        # the next drop build outright.
        _, applied = client.post("/api/code-context", {"source": THERMO_CONTEXT})
        assert applied["promoted"] == [parked["name"]]
        status, added = client.post("/api/unit", {"operation": "Flash"})
        assert status == 200 and added["ok"], added.get("error")
        assert len(added["outlets"]) == 2

    def test_a_reactor_becomes_placeable_from_a_declared_rate_law(self, client):
        """`mass_action_kinetics` is the declarative route, not a callable."""
        parked = client.post("/api/unit", {"operation": "CSTR"})[1]
        assert parked["pending"] is True
        client.post("/api/code-context", {"source": KINETICS_CONTEXT})
        status, added = client.post("/api/unit", {"operation": "CSTR"})
        assert status == 200 and added["ok"], added.get("error")
        assert added["placeholders"] == ["V"], (
            "a required number with no default is a placeholder, and says so"
        )

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

        reopened = FlowsheetSession(path=tmp_path / "plant.json")
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

    def test_the_page_carries_the_token(self, client):
        page = client.get("/")[1].decode("utf-8")
        assert f'content="{client.server.token}"' in page
        assert client.server.token, "minted per process, not a constant"

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

    def test_another_port_on_localhost_is_still_cross_origin(self, client):
        status, _ = client.post(
            "/api/solve", headers={"Origin": "http://127.0.0.1:1"}
        )
        assert status == 403

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

    def test_the_pages_model_functions_behave(self, tmp_path):
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed")

        classic = (gui.STATIC / "classic.html").read_text(encoding="utf-8")
        script = re.search(r"<script>(.*)</script>", classic, re.S)
        assert script, "the classic page must carry its script inline"
        page_js = tmp_path / "page.js"
        page_js.write_text(script.group(1))

        here = pathlib.Path(__file__).parent / "js"
        result = subprocess.run(
            [node, str(here / "checks.js")],
            env={**os.environ, "HARNESS": str(here / "harness.js"),
                 "PAGE_JS": str(page_js)},
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

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

    def test_docutils_renders_it_when_it_is_installed(self, client):
        from difflow.gui import docs

        _, body = client.get_json("/api/docs/CSTR")
        expected = "rst" if docs.available() else "text"
        assert body["format"] == expected
        if expected == "rst":
            # the paragraph is markup, not the escaped source
            assert "<p>" in body["html"]

    def test_an_unknown_operation_is_refused_not_raised(self, client):
        status, body = client.get_json("/api/docs/NotAUnit")
        assert status == 200, "a refusal is an answer, not a broken route"
        assert body["ok"] is False
        assert "NotAUnit" in body["error"]

    def test_a_name_with_a_query_string_still_resolves(self, client):
        _, body = client.get_json("/api/docs/CSTR?t=1")
        assert body["ok"] is True and body["operation"] == "CSTR"

    def test_the_route_needs_no_token(self, client):
        """Reads are readable; only the mutating routes carry the token."""
        status, _ = client.get("/api/docs/Mixer")
        assert status == 200

    def test_every_operation_answers(self, client):
        """The palette can ask about anything it lists."""
        _, catalog = client.get_json("/api/catalog")
        for name in catalog:
            _, body = client.get_json(f"/api/docs/{name}")
            assert body["ok"] is True, name
            assert body["html"], name

    def test_the_session_answers_without_a_socket(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.docs("Flash")["ok"] is True
        assert session.docs("nope")["ok"] is False


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

    def test_the_model_is_overridable_from_the_environment(self, monkeypatch):
        from difflow.gui import assistant

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("DIFFLOW_ASSISTANT_MODEL", "claude-haiku-4-5")
        seen = {}
        self._stub(monkeypatch, {"content": [{"type": "text", "text": "hi"}]},
                   capture=seen)
        assistant.answer([{"role": "user", "content": "q"}])
        assert seen["body"]["model"] == "claude-haiku-4-5"

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

    def test_an_empty_reply_says_why(self, monkeypatch):
        from difflow.gui import assistant

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        self._stub(monkeypatch, {"content": [], "stop_reason": "max_tokens"})
        answer = assistant.answer([{"role": "user", "content": "q"}])
        assert answer["ok"] is False and "max_tokens" in answer["error"]

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

    def test_empty_text_is_empty(self):
        from difflow.gui import docs

        assert docs.render("   ") == ("", "text")

    def test_a_sphinx_role_does_not_swallow_the_line(self):
        """Bare docutils does not know `:class:`, and an unknown role is
        an error that takes the whole paragraph with it."""
        from difflow.gui import docs

        if not docs.available():
            pytest.skip("docutils is not installed")
        html, fmt = docs.render("A :class:`~difflow.streams.Stream` goes in.")
        assert fmt == "rst"
        assert "Stream" in html and "goes in" in html
        assert "difflow.streams" not in html, "`~` abbreviates, as in Sphinx"

    def test_no_system_messages_reach_the_panel(self):
        """A few docstrings indent in ways docutils reads as a block
        quote. That is difflow's prose to fix, not a red box in the
        user's inspector."""
        from difflow.catalog import catalog
        from difflow.gui import docs

        if not docs.available():
            pytest.skip("docutils is not installed")
        for name, spec in catalog().items():
            html, fmt = docs.render(spec.doc)
            assert fmt == "rst", name
            assert "system-message" not in html, name

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

    def test_a_recycle_reports_its_tear_streams(self, thermo):
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("mix", Mixer(SPECIES, thermo),
                         ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES),
                                        thermo), ["mixed"], ["liq", "vap"]))
        fs.add_recycle("vap", "recycle")

        answer = FlowsheetSession(fs).solve()
        assert answer["ok"]
        assert answer["tear_streams"] == ["recycle"]
        assert answer["method"] != "direct"

    def test_an_edit_makes_the_last_solve_stale(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"]
        assert session.levers()["solved"] is True
        assert session.patch_unit("reactor", {"params": {"V": 2.0}})["ok"]
        # The streams on screen describe the flowsheet as it was.
        assert session.levers()["solved"] is False
        assert session.levers()["outputs"] == []

    def test_a_move_does_not(self, thermo):
        """Positions are not physics: dragging a node keeps the results."""
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"]
        assert session.set_layout({"reactor": {"x": 10, "y": 20}})["ok"]
        assert session.levers()["solved"] is True


class TestSensitivity:
    """The derivatives, which is what makes this difflow and not a form."""

    def test_levers_are_the_scalars_and_nothing_else(self, thermo):
        found = {item["key"]: item for item in
                 sensitivity.levers(build_flowsheet(thermo))}
        assert found["reactor.V"]["value"] == 1.0
        assert found["reactor.V"]["units"] == "m^3"
        # The rate function, the stoichiometry array and the species list
        # are on the same params object and are not levers.
        assert "reactor.rate_fn" not in found
        assert "reactor.stoich" not in found
        assert "reactor.species_order" not in found
        assert found["feed:feed.total_flow"]["value"] == pytest.approx(1.1)
        assert found["feed:feed.T"]["units"] == "K"

    def test_every_lever_is_a_key_apply_params_accepts(self, thermo):
        fs = build_flowsheet(thermo)
        for item in sensitivity.levers(fs):
            fs._apply_params({item["key"]: item["value"]})

    def test_outputs_name_the_solved_quantities(self, thermo):
        fs = build_flowsheet(thermo)
        keys = {o["key"] for o in sensitivity.outputs(fs.solve())}
        assert {"liq.total_flow", "liq.T", "liq.P", "liq.F_ethanol"} <= keys

    def test_forward_moves_every_stream_from_one_lever(self, thermo):
        answer = sensitivity.forward(build_flowsheet(thermo), "reactor.V")
        assert answer["mode"] == "forward" and answer["u0"] == 1.0
        # A bigger reactor converts more water to ethanol, and the two
        # derivatives are equal and opposite because the reaction is 1:1.
        liq = answer["streams"]["liq"]
        assert liq["F_ethanol"]["d"] > 0
        assert liq["F_water"]["d"] == pytest.approx(-liq["F_ethanol"]["d"])
        # Nothing upstream of the reactor can move.
        assert answer["streams"]["feed"]["F_water"]["d"] == 0.0

    @pytest.mark.release
    def test_forward_matches_a_finite_difference(self, thermo):
        """The decisive check: AD through the solve, against the real thing."""
        answer = sensitivity.forward(build_flowsheet(thermo), "reactor.V")
        h = 1e-4
        up = build_flowsheet(thermo, V=1.0 + h).solve()
        down = build_flowsheet(thermo, V=1.0 - h).solve()
        fd = (float(up["liq"]["F_ethanol"]) - float(down["liq"]["F_ethanol"])) / (2 * h)
        assert answer["streams"]["liq"]["F_ethanol"]["d"] == pytest.approx(fd, rel=1e-5)

    def test_reverse_ranks_the_levers_dimensionlessly(self, thermo):
        answer = sensitivity.reverse(build_flowsheet(thermo), "liq.F_ethanol")
        assert answer["mode"] == "reverse" and answer["y0"] > 0
        ranked = answer["levers"]
        relative = [abs(item["rel"]) for item in ranked if item["rel"] is not None]
        assert relative == sorted(relative, reverse=True)
        by_key = {item["key"]: item for item in ranked}
        # V and molar_density enter the rate as their product, so their
        # dimensionless sensitivities have to come out equal. A test that
        # only checked signs would not notice if one were scaled wrong.
        assert by_key["reactor.V"]["rel"] == pytest.approx(
            by_key["reactor.molar_density"]["rel"], rel=1e-9
        )

    def test_total_flow_differentiates_through_every_species(self, thermo):
        answer = sensitivity.reverse(build_flowsheet(thermo), "rx.total_flow")
        assert answer["y0"] == pytest.approx(1.1)
        # The reaction conserves moles, so the reactor cannot change the
        # total -- but the feed rate obviously can.
        by_key = {item["key"]: item for item in answer["levers"]}
        assert by_key["reactor.V"]["d"] == pytest.approx(0.0, abs=1e-9)
        assert by_key["feed:feed.total_flow"]["d"] == pytest.approx(1.0, rel=1e-6)

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
            _, reverse = live.post("/api/sensitivity",
                                   {"target": "liq.F_ethanol"})
            assert reverse["ok"] and reverse["mode"] == "reverse"
        finally:
            live.close()


class TestExport:
    """The ways out of the editor: a script, a document, a drawing."""

    def test_the_diagram_is_drawn_at_the_canvas_s_layout(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.diagram()
        assert answer["ok"] and answer["svg"].startswith("<svg")
        assert "reactor" in answer["svg"]

        assert session.set_layout({"reactor": {"x": 500, "y": 300}})["ok"]
        moved = session.diagram()["svg"]
        assert moved != answer["svg"] and "500" in moved

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

    @pytest.mark.release
    def test_the_jacobian_matches_central_differences(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"]
        answer = session.linearize(self.LEVERS, self.OUTPUTS, check=True)
        assert answer["ok"], answer
        assert answer["check"]["passed"], answer["check"]
        assert answer["check"]["max_rel_error"] < 1e-5

    def test_a_lever_the_flowsheet_moves_has_a_nonzero_column(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        answer = session.linearize(["reactor.V"], ["liq.F_ethanol"])
        vector = answer["delta_vectors"]["vectors"][0]
        assert vector["J"][0][0] > 0, "a bigger reactor makes more ethanol"
        assert vector["u0"] == [1.0]

    def test_units_travel_with_the_coefficients(self, thermo):
        """A Jacobian with unlabelled axes is not an export."""
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        answer = session.linearize(self.LEVERS, self.OUTPUTS)
        vector = answer["delta_vectors"]["vectors"][0]
        # `V` carries its units in CSTR.parameter_units, not in the
        # dataclass field metadata, which is where most of them live.
        assert vector["u_units"] == ["m^3", "mol/s"]
        assert vector["y_units"] == ["mol/s", "mol/s"]

    def test_an_unbounded_lever_gets_a_window_not_an_infinity(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        vector = session.linearize(
            ["reactor.V"], ["liq.F_ethanol"])["delta_vectors"]["vectors"][0]
        assert vector["lb"] == [0.0] and vector["ub"] == [2.0]
        assert all(map(math.isfinite, vector["tr_lo"] + vector["tr_hi"]))

    def test_bounds_given_are_the_bounds_used(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        answer = session.linearize(
            ["reactor.V"], ["liq.F_ethanol"],
            bounds={"reactor.V": {"lb": 0.5, "ub": 4.0}})
        vector = answer["delta_vectors"]["vectors"][0]
        assert vector["lb"] == [0.5] and vector["ub"] == [4.0]

    def test_the_selection_is_persisted_so_the_panel_reopens_on_it(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        session.linearize(self.LEVERS, self.OUTPUTS, radius=0.1,
                          bounds={"reactor.V": {"lb": 0.5}})
        saved = session.flowsheet.view["planning"]
        assert saved["u"] == self.LEVERS and saved["y"] == self.OUTPUTS
        assert saved["radius"] == 0.1
        assert saved["bounds"] == {"reactor.V": {"lb": 0.5}}

    def test_an_empty_pick_is_refused_with_a_sentence(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        assert session.linearize([], ["liq.F_ethanol"]) == {
            "ok": False, "error": "pick at least one lever and one output"}
        assert session.linearize(["reactor.V"], [])["ok"] is False

    def test_a_name_the_flowsheet_does_not_have_is_an_answer(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        answer = session.linearize(["reactor.V"], ["nowhere.total_flow"])
        assert answer["ok"] is False
        assert "nowhere" in answer["error"]

    def test_an_empty_session_answers_rather_than_raising(self):
        """A refusal about the names asked for, from an empty flowsheet.

        It no longer says "no flowsheet loaded", because there is one ---
        an editor opened with no file starts empty rather than inert. What
        matters here is unchanged: a request naming things that are not
        there comes back as a message and not as a traceback.
        """
        answer = FlowsheetSession(None).linearize(["a"], ["b"])
        assert answer["ok"] is False and answer["error"]

    def test_health_findings_travel_with_the_export(self, thermo):
        """A dead lever has to be visible downstream, not just locally."""
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        # Nothing upstream of the reactor can respond to its volume.
        answer = session.linearize(["reactor.V"], ["feed.total_flow"])
        kinds = {f["kind"] for f in answer["health"]}
        assert "dead_lever" in kinds
        assert answer["delta_vectors"]["health"] == answer["health"]

    def test_the_downloads_are_what_the_export_writers_write(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        session.linearize(self.LEVERS, self.OUTPUTS)

        manifest = session.linearization_files("json")
        assert manifest["ok"] and len(manifest["files"]) == 1
        # Never the flowsheet's own name: a download landing next to the
        # document under that name is a flowsheet overwritten.
        assert manifest["files"][0]["name"].endswith("_delta_vectors.json")
        loaded = json.loads(manifest["files"][0]["text"])
        assert loaded["vectors"][0]["u_names"]
        assert loaded["meta"]["source"] == "difflow.gui"

        tables = session.linearization_files("csv")
        names = {f["name"] for f in tables["files"]}
        assert "flowsheet_jacobian.csv" in names and "bounds.csv" in names

    def test_a_format_that_does_not_exist_is_refused(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        session.linearize(self.LEVERS, self.OUTPUTS)
        answer = session.linearization_files("mps")
        assert answer["ok"] is False and "mps" in answer["error"]

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

    def test_the_assistant_can_brief_on_the_linearization(self, thermo):
        """`context.pack(kind="planning")` reads what linearize stored."""
        session = FlowsheetSession(build_flowsheet(thermo))
        session.solve()
        assert context.pack(session, kind="planning")["ok"] is False

        session.linearize(self.LEVERS, self.OUTPUTS)
        pack = context.pack(session, kind="planning", question="what is this?")
        assert pack["ok"] and "reactor_V" in pack["prompt"]
        assert "trust radius" in pack["prompt"]


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

    def test_an_expression_comes_back_as_its_repr(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("6 * 7")
        assert answer["ok"] and answer["error"] is None
        assert answer["outputs"] == [
            {"kind": "value", "stream": "stdout", "text": "42"}]

    def test_statements_run_and_the_namespace_persists(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.console_run("import math\nradius = 2.0")
        answer = session.console_run("round(math.pi * radius ** 2, 3)")
        assert answer["outputs"][-1]["text"] == "12.566"
        # a module the user imported is a name they defined, and seeing
        # it is the confirmation the import took
        assert answer["names"] == ["math", "radius"]

    def test_print_and_the_value_both_arrive_in_order(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("print('working')\n5")
        assert [o["text"] for o in answer["outputs"]] == ["working\n", "5"]

    def test_a_traceback_is_the_answer_not_a_refusal(self, thermo):
        """`ok` is about the request; the cell failing is a result."""
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("1 / 0")
        assert answer["ok"] is True
        assert "ZeroDivisionError" in answer["error"]
        assert "1 / 0" in answer["error"], "the offending line, from linecache"
        assert "session.py" not in answer["error"], "server frames trimmed"

    def test_what_a_cell_printed_survives_the_exception(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("print('got this far')\nboom")
        assert answer["outputs"][0]["text"] == "got this far\n"
        assert "NameError" in answer["error"]

    def test_a_syntax_error_names_the_line(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("x = 1\ndef f(:")
        assert "SyntaxError" in answer["error"] and "line 2" in answer["error"]

    def test_fs_is_the_flowsheet_on_the_canvas(self, thermo):
        """Not a copy. This is the whole reason the panel exists."""
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("fs is session.flowsheet")
        assert answer["outputs"][-1]["text"] == "True"

    def test_the_live_names_are_rebound_every_cell(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.console_run("streams is None")["outputs"][-1]["text"] == "True"
        assert session.solve()["ok"]
        answer = session.console_run("sorted(streams)[0]")
        assert answer["error"] is None
        assert answer["outputs"][-1]["text"].strip("'\"") in session.streams

    def test_the_code_context_bindings_are_in_scope(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.set_code_context("greeting = 'hello'")
        assert session.console_run("greeting")["outputs"][-1]["text"] == "'hello'"

    def test_gradients_run_here_which_is_the_point(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run(
            "float(jax.grad(lambda V: fs._apply_params({'reactor.V': V})"
            ".solve()['liq']['F_ethanol'])(1.0))")
        assert answer["error"] is None, answer["error"]
        assert float(answer["outputs"][-1]["text"]) > 0

    def test_a_cell_that_edits_the_model_says_so(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"] and session.streams is not None
        answer = session.console_run(
            "unit = next(u for u in fs.units if u.name == 'reactor')\n"
            "unit.operation.params = unit.operation.params.update(V=2.0)")
        assert answer["error"] is None, answer["error"]
        assert answer["changed"] is True
        assert session.streams is None, "the cached solve describes the old model"

    def test_a_cell_that_only_reads_leaves_the_canvas_alone(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"]
        assert session.console_run("len(fs.units)")["changed"] is False
        assert session.streams is not None

    def test_reset_forgets_what_the_console_defined(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.console_run("keep = 1")
        session.console_reset()
        assert "NameError" in session.console_run("keep")["error"]
        assert session.console_run("fs is not None")["outputs"][-1]["text"] == "True"

    def test_an_empty_cell_is_not_a_round_trip(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.console_run("   \n  ") == {
            "ok": True, "outputs": [], "error": None,
            "changed": False, "names": []}

    def test_the_names_in_scope_are_advertised(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        session.set_code_context("greeting = 'hello'")
        names = session.console_names()
        assert set(names["live"]) == {"fs", "streams", "dvs", "session"}
        assert names["bindings"] == ["greeting"]
        assert names["defined"] == []

    def test_the_route_is_guarded_like_every_other_exec(self, client):
        status, body = client.send("POST", "/api/console", {"source": "1"},
                                   headers={gui.TOKEN_HEADER: "wrong"})
        assert status == 403 and body["ok"] is False
        status, body = client.send("POST", "/api/console", {"source": "1 + 1"})
        assert status == 200
        assert body["outputs"][-1]["text"] == "2"

    def test_output_is_capped(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        answer = session.console_run("print('x' * 500_000)")
        body = answer["outputs"][0]["text"]
        assert len(body) < console.MAX_OUTPUT + 100
        assert "truncated" in body


class TestConsoleFigures:
    """The seam plots will arrive through, exercised without matplotlib."""

    def test_a_display_hook_appends_to_the_cell(self):
        drawn = console.image(b"\x89PNG-not-really")
        shell = console.Console(display_hooks=[lambda ns: [drawn]])
        answer = shell.run("1")
        assert answer["outputs"] == [
            {"kind": "value", "stream": "stdout", "text": "1"}, drawn]
        assert drawn["kind"] == "image" and drawn["mime"] == "image/png"

    def test_a_hook_sees_the_namespace(self):
        seen = {}
        shell = console.Console(display_hooks=[lambda ns: seen.update(ns) or []])
        shell.run("marker = 7")
        assert seen["marker"] == 7

    def test_a_broken_hook_does_not_eat_the_result(self):
        """A renderer that fails must not lose a number computed correctly."""
        def broken(ns):
            raise RuntimeError("no display")

        answer = console.Console(display_hooks=[broken]).run("6 * 7")
        assert answer["error"] is None
        assert answer["outputs"][0]["text"] == "42"
        assert "no display" in answer["outputs"][1]["text"]
        assert answer["outputs"][1]["stream"] == "stderr"

    def test_an_image_survives_the_json_the_server_sends(self):
        drawn = console.image(b"\x89PNG\r\n\x1a\n binary \xff\xfe")
        assert json.loads(json.dumps(drawn)) == drawn


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

    def test_the_suggested_port_steps_over_a_run_of_busy_ones(self):
        """A suggestion that fails the same way is worse than none.

        The run has to be longer than one, or a naive ``port + 1`` passes
        this by luck: the next port is usually free anyway.
        """
        held, port = self.taken_port()
        # Held open for the length of the test. A socket that goes out of
        # scope is collected and its port freed, which quietly empties the
        # run this test is built on.
        holding = [held]
        busy = {port}
        try:
            for offset in (1, 2, 3):
                neighbour = socket.socket()
                neighbour.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    neighbour.bind((gui.HOST, port + offset))
                    neighbour.listen(1)
                except OSError:       # already someone else's; still busy
                    neighbour.close()
                else:
                    holding.append(neighbour)
                busy.add(port + offset)
            assert len(busy) > 1, "the point of the test is a run, not one port"

            suggested = server.free_port_near(port)
            assert suggested not in busy, (
                f"suggested {suggested}, which is in the busy run {sorted(busy)}")
            probe = socket.socket()
            try:
                probe.bind((gui.HOST, suggested))   # the point: this works
            finally:
                probe.close()
        finally:
            for sock in holding:
                sock.close()

    def test_the_message_names_the_way_in_that_was_used(self):
        """Two entry points; naming the other one sends the reader in a circle."""
        said = server.port_in_use_message(8756, "python -m difflow.gui")
        assert "python -m difflow.gui --port" in said
        assert "difflow gui --port" not in said

    def test_another_oserror_still_raises(self, monkeypatch):
        """Only the actionable one is swallowed; the rest keep their traceback."""
        def refuse(**kwargs):
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(server, "serve", refuse)
        with pytest.raises(OSError) as caught:
            server.main(["--no-browser"])
        assert caught.value.errno == errno.EACCES

    def test_reuse_address_is_already_on(self):
        """So the bind refusal is a live listener, not a lingering TIME_WAIT.

        Pinning this down because the tempting "fix" for Errno 48 is to set
        ``SO_REUSEADDR``, and it is set. If a future stdlib stops setting it
        the diagnosis above changes and this should be read again.
        """
        from http.server import ThreadingHTTPServer
        assert ThreadingHTTPServer.allow_reuse_address


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

    def test_a_feed_renames_everywhere_it_appears(self, client):
        status, answer = client.patch("/api/stream/feed", {"name": "charge"})
        assert status == 200 and answer == {
            "ok": True, "kind": "stream", "stream": "charge"
        }
        fs = client.session.flowsheet
        assert "charge" in fs.feeds and "feed" not in fs.feeds
        assert edit.unit(fs, "reactor").inlet_names == ["charge"]

    def test_an_intermediate_stream_moves_at_both_ends(self, client):
        """Or the graph quietly splits into two that each look fine."""
        assert client.patch("/api/stream/rx", {"name": "effluent"})[1]["ok"]
        fs = client.session.flowsheet
        assert edit.unit(fs, "reactor").outlet_names == ["effluent"]
        assert edit.unit(fs, "flash").inlet_names == ["effluent"]
        # And it still solves, which is the whole claim.
        assert client.post("/api/solve", {})[1]["ok"]

    def test_both_ends_of_a_recycle_follow(self, client):
        # `connect` will not wire into a stream that is already fed, so
        # the declared feed comes off first.
        assert client.delete("/api/feed/feed")[1]["ok"]
        assert client.post("/api/connect", {"source": "flash", "outlet": "vap",
                                            "target": "reactor",
                                            "inlet": "feed"})[1]["ok"]
        fs = client.session.flowsheet
        assert fs.recycles, "expected the wire back to be torn"
        assert client.patch("/api/stream/vap", {"name": "overhead"})[1]["ok"]
        assert "overhead" in fs.recycles
        assert "vap" not in fs.recycles

    def test_the_canvas_node_follows_the_feed(self, client):
        client.post("/api/layout", {"nodes": {"feed:feed": {"x": 5.0, "y": 6.0}}})
        assert client.patch("/api/stream/feed", {"name": "charge"})[1]["ok"]
        nodes = client.session.flowsheet.view["nodes"]
        assert nodes["feed:charge"] == {"x": 5.0, "y": 6.0}
        assert "feed:feed" not in nodes

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

    def test_renaming_to_the_same_name_is_a_no_op_and_not_an_error(self, client):
        _, answer = client.patch("/api/stream/feed", {"name": "feed"})
        assert answer == {"ok": True, "kind": "stream", "stream": "feed"}

    def test_a_url_encoded_stream_name_is_decoded(self, client):
        """The same path `PATCH /api/unit/` takes, so it decodes alike."""
        client.patch("/api/stream/feed", {"name": "feed"})
        status, answer = client.patch("/api/stream/rx%78", {"name": "z"})
        assert status == 200 and answer["ok"] is False
        assert "'rxx'" in answer["error"]

    def test_a_rename_needs_the_token(self, client):
        status, _ = client.patch("/api/stream/feed", {"name": "charge"},
                                 headers={gui.TOKEN_HEADER: "wrong"})
        assert status == 403
        assert "feed" in client.session.flowsheet.feeds


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

    def test_a_mixer_arrives_able_to_mix(self, empty):
        """One inlet is what it means to not be there: that is a pipe."""
        _, added = empty.post("/api/unit", {"operation": "Mixer"})
        assert added["inlets"] == ["mixer_in", "mixer_in2"]
        assert added["outlets"] == ["mixer_out"]

    def test_a_fixed_unit_still_gets_the_ports_it_declares(self, empty):
        _, added = empty.post("/api/unit", {"operation": "Heater"})
        assert added["inlets"] == ["heater_in"]

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

    def test_a_new_inlet_dangles(self, empty):
        """Same as a port on a unit just dropped: unfed, awaiting a feed."""
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.post("/api/inlet", {"unit": "mixer"})
        assert empty.get_json("/api/feeds")[1]["unfed"] == [
            "mixer_in", "mixer_in2", "mixer_in3"
        ]

    def test_a_new_inlet_does_not_collide_with_a_name_in_use(self, empty):
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.patch("/api/stream/mixer_in2", {"name": "mixer_in3"})
        _, added = empty.post("/api/inlet", {"unit": "mixer"})
        # `mixer_in32` would be the answer from bolting a digit onto a
        # taken base, and it reads as the thirty-second inlet.
        assert added["stream"] == "mixer_in4", added

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

    def test_a_recycle_destination_is_refused(self, empty):
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.post("/api/unit", {"operation": "Heater"})
        empty.post("/api/connect", {"source": "mixer", "outlet": "mixer_out",
                                    "target": "heater", "inlet": "heater_in"})
        empty.post("/api/connect", {"source": "heater", "outlet": "heater_out",
                                    "target": "mixer", "inlet": "mixer_in2"})
        assert empty.session.flowsheet.recycles, "expected a tear"
        stream = next(iter(empty.session.flowsheet.recycles.values()))
        _, refused = empty.delete("/api/inlet", {"unit": "mixer",
                                                 "stream": stream})
        assert refused["ok"] is False and "recycle" in refused["error"]

    def test_an_inlet_the_unit_does_not_have_lists_the_ones_it_does(self, empty):
        empty.post("/api/unit", {"operation": "Mixer"})
        _, refused = empty.delete("/api/inlet", {"unit": "mixer",
                                                 "stream": "nope"})
        assert refused["ok"] is False
        assert "mixer_in, mixer_in2" in refused["error"]

    def test_a_unit_that_is_not_there_says_so(self, empty):
        _, refused = empty.post("/api/inlet", {"unit": "nope"})
        assert refused["ok"] is False and "no unit called 'nope'" in refused["error"]

    def test_the_verbs_need_the_token(self, empty):
        empty.post("/api/unit", {"operation": "Mixer"})
        status, _ = empty.post("/api/inlet", {"unit": "mixer"},
                               headers={gui.TOKEN_HEADER: "wrong"})
        assert status == 403
        assert len(edit.unit(empty.session.flowsheet, "mixer").inlet_names) == 2

    def test_the_catalog_says_which_units_take_more(self, empty):
        """So the panel can offer the verb only where it works.

        The front end reads `ports.variadic` off the catalog to decide
        whether to draw `Add inlet`; if that flag and the server's
        refusal ever disagree, the button is a lie.
        """
        _, catalog = empty.get_json("/api/catalog")
        flagged = {n for n, e in catalog.items() if e["ports"]["variadic"]}
        assert flagged == self.variadic_names()
        assert "Mixer" in flagged and "Flash" not in flagged

    def test_every_flagged_unit_actually_takes_another_inlet(self, empty):
        """The button is drawn from the flag, so the flag has to be true."""
        accepted = []
        for name in sorted(self.variadic_names()):
            answer = empty.post("/api/unit", {"operation": name})[1]
            if not answer["ok"] or answer.get("pending"):
                continue        # needs something from the code context
            unit_name = answer["name"]
            added = empty.post("/api/inlet", {"unit": unit_name})[1]
            assert added["ok"], (name, added)
            assert added["stream"] in edit.unit(
                empty.session.flowsheet, unit_name).inlet_names
            accepted.append(name)
        assert "Mixer" in accepted

    def test_an_added_inlet_survives_the_round_trip(self, empty, tmp_path):
        """It is a port on a saved unit, not a decoration on the canvas."""
        from difflow import serialize

        empty.post("/api/unit", {"operation": "Mixer"})
        empty.post("/api/inlet", {"unit": "mixer"})
        path = tmp_path / "plant.json"
        serialize.save(empty.session.flowsheet, path)
        reloaded = serialize.load(path)
        assert edit.unit(reloaded, "mixer").inlet_names == [
            "mixer_in", "mixer_in2", "mixer_in3"
        ]

    def test_a_mixer_with_three_fed_inlets_solves(self, empty):
        """The point of all of it: three streams in, one out."""
        empty.post("/api/unit", {"operation": "Mixer"})
        empty.post("/api/inlet", {"unit": "mixer"})
        for i, stream in enumerate(("mixer_in", "mixer_in2", "mixer_in3")):
            empty.post("/api/feed", {"name": stream, "T": 300.0 + 10 * i,
                                     "flows": {"water": 1.0 + i}})
        _, solved = empty.post("/api/solve", {})
        assert solved["ok"], solved
        out = solved["streams"]["mixer_out"]
        assert out["F_water"] == pytest.approx(6.0)


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

    def test_nothing_expires_before_a_page_has_ever_checked_in(self):
        """``--no-browser``, then a coffee, then open it. It must be there.

        Also the invariant that keeps every other test in this file
        alive: they drive the routes directly and never pretend to be a
        page, and the server must not vanish underneath them.
        """
        now, clock = self.clock()
        life = server.Lifetime(grace=10, clock=clock)
        now[0] += 10_000
        assert not life.expired()

    def test_a_page_that_stops_pinging_ages_out(self):
        now, clock = self.clock()
        life = server.Lifetime(grace=10, clock=clock)
        life.ping("tab-a")
        now[0] += 9
        assert not life.expired(), "still inside the grace"
        now[0] += 2
        assert life.expired()

    def test_a_second_tab_keeps_the_editor_open(self):
        """Two tabs on one flowsheet is ordinary, and a counter gets it wrong.

        A reload increments before it decrements, and a tab that dies
        without unloading never decrements at all. Ids simply age out.
        """
        now, clock = self.clock()
        life = server.Lifetime(grace=10, clock=clock)
        life.ping("tab-a")
        life.ping("tab-b")
        life.bye("tab-a")
        now[0] += 5
        life.ping("tab-b")
        now[0] += 6
        assert not life.expired(), "tab-b is still there"
        life.bye("tab-b")
        now[0] += life.linger + 1
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

    def test_a_reload_is_not_a_closed_tab(self):
        """The bug this linger exists for: reloading the page killed the editor.

        A reload is a farewell and then a hello, and the hello cannot go
        out until the new document has fetched the bundle and mounted.
        Read the farewell as final and the server stops in that gap ---
        so the reloaded page loads against nothing and the user sees an
        editor that shows nothing when it starts.
        """
        now, clock = self.clock()
        life = server.Lifetime(grace=90, linger=5, clock=clock)
        life.ping("first-load")
        life.bye("first-load")              # pagehide, on the way to reloading

        now[0] += 0.4                       # fetch, parse, mount
        assert not life.expired(), "the watchdog looked between the two halves"
        life.ping("second-load")            # the new document checks in
        now[0] += 60
        life.ping("second-load")
        assert not life.expired(), "the reloaded page is here and being served"

    def test_the_linger_is_a_wait_and_not_a_reprieve(self):
        """Silence after the farewell still gives the port back, promptly."""
        now, clock = self.clock()
        life = server.Lifetime(grace=90, linger=5, clock=clock)
        life.ping("tab-a")
        life.bye("tab-a")
        now[0] += 4
        assert not life.expired(), "still inside the linger"
        now[0] += 2
        assert life.expired(), "and no waiting out the 90-second grace after it"

    def test_the_linger_runs_from_the_farewell_not_from_the_next_look(self):
        """The watchdog polls; the clock must not start when it happens to look."""
        now, clock = self.clock()
        life = server.Lifetime(grace=90, linger=5, clock=clock)
        life.ping("tab-a")
        life.bye("tab-a")
        now[0] += 5.5                        # a poll that arrived late
        assert life.expired(), "the wait became the linger plus the poll gap"

    def test_quit_does_not_wait_for_anyone(self):
        now, clock = self.clock()
        life = server.Lifetime(grace=10_000, clock=clock)
        life.ping("tab-a")
        life.quit()
        assert life.expired()
        assert life.asked, "serve() says which of the two endings it was"

    def test_a_ping_from_a_client_that_had_gone_brings_it_back(self):
        """A tab restored from the back/forward cache is a live tab again."""
        now, clock = self.clock()
        life = server.Lifetime(grace=10, linger=5, clock=clock)
        life.ping("tab-a")
        life.bye("tab-a")
        now[0] += 6
        assert life.expired()
        life.ping("tab-a")
        assert not life.expired()
        now[0] += 6
        assert not life.expired(), "the farewell before it must not still count"

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

    def test_a_page_from_nowhere_cannot_stop_the_editor(self, client):
        """The token guard covers these as it covers every other mutation."""
        request = urllib.request.Request(
            client.base + "/api/quit", data=b"{}", method="POST",
            headers={"Content-Type": "application/json"},   # no token
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 403
        assert not client.server.lifetime.asked

    def test_the_watcher_shuts_the_server_down(self):
        """End to end: nobody is watching, so the port comes back."""
        session = FlowsheetSession(None, None)
        life = server.Lifetime(grace=0.05, linger=0.05)
        srv = server.make_server(session, port=0, lifetime=life)
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        stop = threading.Event()
        try:
            server.watch(srv, life, poll=0.02, stop=stop)
            life.ping("tab-a")
            life.bye("tab-a")
            thread.join(timeout=5)
            assert not thread.is_alive(), "the watcher never called shutdown"
        finally:
            stop.set()
            srv.server_close()

        rebind = socket.socket()
        rebind.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            rebind.bind((gui.HOST, port))       # the whole point of the feature
        finally:
            rebind.close()

    def test_building_a_server_does_not_start_anything_that_can_stop_it(self):
        """``make_server`` is what the tests use; it must not grow a watchdog."""
        before = {t.name for t in threading.enumerate()}
        srv = server.make_server(FlowsheetSession(None, None), port=0)
        try:
            after = {t.name for t in threading.enumerate()}
            assert not [n for n in after - before if "watch" in n]
            assert srv.lifetime is not None, "the routes still need one"
        finally:
            srv.server_close()

    def test_the_page_and_the_server_agree_on_the_interval(self, client):
        """One agreement, written down once. The page reads its half here."""
        status, about = client.get_json("/api/about")
        assert status == 200
        assert about["heartbeat"] == server.HEARTBEAT_SECONDS
        assert server.IDLE_GRACE_SECONDS > 4 * server.HEARTBEAT_SECONDS, (
            "a background tab has its timers throttled to about one firing a "
            "minute; a tight grace shuts the editor down when the user looks "
            "at another tab"
        )


class TestTheHeaderLinks:
    """Where to read more, without leaving the editor to go and find it."""

    def test_about_carries_the_project_urls(self, client):
        status, about = client.get_json("/api/about")
        assert status == 200 and about["ok"]
        for key in ("repository", "documentation"):
            assert about["links"][key].startswith("https://"), key

    def test_the_urls_come_from_the_packaging_metadata(self):
        """So ``pyproject.toml`` stays the one place they are written down."""
        tomllib = pytest.importorskip("tomllib")   # 3.11+; the check, not the code

        declared = tomllib.loads(
            (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        )["project"]["urls"]
        found = server.links()
        assert found["repository"] == declared["Repository"]
        assert found["documentation"] == declared["Documentation"]

    def test_a_tree_with_no_metadata_still_has_links(self, monkeypatch):
        """Running from a source checkout is not a header without links."""
        import importlib.metadata

        def missing(_name):
            raise importlib.metadata.PackageNotFoundError("difflow")

        monkeypatch.setattr(importlib.metadata, "metadata", missing)
        assert server.links() == server.FALLBACK_LINKS


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

    def test_an_ordinary_word_is_matched_as_well_as_a_camel_case_one(self):
        """``Junction`` and ``Transformer`` are the names a heuristic misses.

        Guarding the choice to match against the name asked about rather
        than against a guess at what an operation name looks like.
        """
        assert doclinks.url_for("Transformer") is not None
        assert doclinks.url_for("Junction") is not None

    def test_a_name_the_book_never_uses_gets_no_link(self):
        """Reported as an absence rather than papered over with a home page."""
        assert doclinks.url_for("NotAnOperationAtAll") is None
        assert doclinks.url_for("") is None

    def test_a_unit_operations_page_beats_a_passing_mention_elsewhere(self):
        """A reader who clicked a compressor does not want the LP tutorial."""
        url = doclinks.url_for("Compressor")
        assert "unit-operations" in url

    def test_no_link_points_at_a_page_the_book_does_not_build(self):
        """A 404 is worse than no link, and cheap to rule out here."""
        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        toc = yaml.safe_load((root / "_toc.yml").read_text())
        built = set()

        def walk(node):
            if isinstance(node, dict):
                if "file" in node:
                    built.add(node["file"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(toc)
        session = FlowsheetSession(None, None)
        for name, entry in session.catalog().items():
            url = entry["docs_url"]
            if url is None:
                continue
            page = url[len(doclinks.BASE_URL):].split("#")[0]
            assert "docs/" + page.removesuffix(".html") in built, (
                f"{name} links to {page}, which _toc.yml does not build")

    def test_the_catalog_carries_the_link_for_every_operation(self):
        """All of them, now that #228 wrote the prose that was missing.

        The count used to be "all but five". It is not a count any more:
        an operation with nowhere to link to is a palette entry with
        nothing to read, and :mod:`tests.test_doclinks` refuses it at the
        source. This is the same line drawn where the editor stands.
        """
        session = FlowsheetSession(None, None)
        catalog = session.catalog()
        assert len(catalog) > 50
        assert not [n for n, e in catalog.items() if not e["docs_url"]], (
            "units with no documentation link: "
            f"{sorted(n for n, e in catalog.items() if not e['docs_url'])}")

    def test_the_inspector_gets_the_same_link_as_the_palette(self):
        session = FlowsheetSession(None, None)
        assert (session.docs("Flash")["docs_url"]
                == session.catalog()["Flash"]["docs_url"])

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

    def test_the_message_names_the_process_holding_the_port(self):
        held = socket.socket()
        held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.bind((gui.HOST, 0))
        held.listen(1)
        port = held.getsockname()[1]
        try:
            found = server.listener_on(port)
            if found is None:
                pytest.skip("no lsof or ss on this machine to ask")
            assert found["pid"] == os.getpid(), "we are the one holding it"
            assert found["mine"] is True
            said = server.port_in_use_message(port, "difflow gui")
            assert str(os.getpid()) in said
            assert f"kill {os.getpid()}" in said
        finally:
            held.close()

    def test_a_free_port_has_no_listener_to_name(self):
        probe = socket.socket()
        probe.bind((gui.HOST, 0))
        port = probe.getsockname()[1]
        probe.close()
        assert server.listener_on(port, timeout=1.0) is None

    def test_the_message_survives_a_machine_with_neither_helper(self, monkeypatch):
        """It runs on the way to an error; it must not become one."""
        monkeypatch.setattr(server, "_run", lambda argv, timeout: "")
        said = server.port_in_use_message(8756, "difflow gui")
        assert "8756" in said and "--port" in said
        assert "pid" not in said, "nothing was found, so nothing is claimed"

    def test_a_helper_that_hangs_or_is_missing_is_not_fatal(self, monkeypatch):
        def explode(*args, **kwargs):
            raise FileNotFoundError("lsof")

        monkeypatch.setattr(server.subprocess, "run", explode)
        assert server.listener_on(8756) is None

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
