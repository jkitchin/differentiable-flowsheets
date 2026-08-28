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

import json
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
from difflow.gui import FlowsheetSession, _json_restore, _json_safe, make_server

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
    """A live server on an ephemeral port, plus the two verbs it takes."""

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

    def post(self, path, body=None):
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


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
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400
        assert "bad JSON" in json.loads(excinfo.value.read())["error"]


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

    def test_an_empty_session_reports_rather_than_raises(self):
        session = FlowsheetSession()
        assert session.document()["flowsheet"] is None
        assert not session.solve()["ok"]
        assert session.code()["error"]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
