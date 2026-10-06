"""difflow.agent: the operations behind ``difflow mcp``, called directly.

The phase-1 bar for the agent tools is that the three editor examples can
be built from nothing but tool calls and solve to the same answer as the
shipped files. Those rebuilds are here, with the discovery calls an agent
uses to get there and the guard rails (timeouts, --no-exec, rollback).
"""

import json
import time
from pathlib import Path

import pytest

from difflow.agent import TOOLS, Workbench, jsonable

EXAMPLES = Path(__file__).parent.parent / "src" / "difflow" / "gui" / "examples"
ESTERIFICATION = ["acetic_acid", "ethanol", "ethyl_acetate", "water"]


def context_of(example: str) -> str:
    return json.loads((EXAMPLES / f"{example}.json").read_text())["view"]["code_context"]


def reference(example: str) -> dict:
    """The shipped example's solved streams."""
    wb = Workbench()
    assert wb.open_example(example)["ok"]
    assert wb.solve()["converged"]
    return wb.get_streams()["streams"]


def assert_same(mine: dict, ref: dict, rel=1e-9):
    assert set(mine) == set(ref)
    for key, value in ref.items():
        assert mine[key] == pytest.approx(value, rel=rel, abs=1e-12), key


# =============================================================================
# Discovery
# =============================================================================


class TestDiscovery:
    def test_operations_come_from_the_registry(self):
        found = Workbench().list_operations("flash")
        names = {op["name"] for op in found["operations"]}
        assert {"Flash", "EOSFlash"} <= names

    def test_a_plugin_filter_takes_the_short_name(self):
        ops = Workbench().list_operations(plugin="gas")["operations"]
        assert ops and all(op["plugin"] == "difflow_gas" for op in ops)

    def test_describe_says_what_is_missing_and_how_to_supply_it(self):
        """In an empty session a CSTR cannot be built: it needs a rate law."""
        spec = Workbench().describe_operation("CSTR")
        assert spec["ok"]
        assert {"rate_fn"} <= set(spec["needs"])
        assert "mass_action_kinetics" in spec["starter_code"]
        assert any(p["name"] == "V" for p in spec["parameters"])

    def test_an_unknown_operation_points_at_the_search(self):
        answer = Workbench().describe_operation("Reboiler9000")
        assert not answer["ok"] and "list_operations" in answer["error"]

    def test_libraries_that_are_not_operations_are_listed(self):
        """planning is deliberately not a palette operation; it is still found."""
        api = Workbench().list_api("difflow.planning")
        names = {n["name"] for n in api["names"]}
        assert {"Block", "linearize_block"} <= names
        assert api["submodules"]

    def test_a_plugin_library_is_walkable(self):
        api = Workbench().list_api("difflow_refinery")
        assert any(m["name"] == "difflow_refinery.reforming" for m in api["submodules"])

    def test_describe_api_lists_dataclass_fields(self):
        answer = Workbench().describe_api("difflow.CSTRParams")
        assert answer["kind"] == "dataclass"
        assert "molar_density" in {f["name"] for f in answer["fields"]}

    def test_only_difflow_is_importable(self):
        """Importing a module runs it; the tool is not a general importer."""
        for name in ("os", "subprocess.run"):
            assert not Workbench().describe_api(name)["ok"]
        assert not Workbench().list_api("os")["ok"]

    def test_species_report_what_data_they_have(self):
        found = Workbench().search_species("water")["species"]
        water = next(s for s in found if s["name"] == "water")
        assert water["MW"] == pytest.approx(18.015)
        assert water["critical"] and water["ideal_thermo"]

    def test_docs_search_returns_text_and_a_url(self):
        hits = Workbench().search_docs("negative flows tear clipping")["results"]
        assert hits and hits[0]["url"].startswith("https://")
        assert hits[0]["text"]

    def test_plugin_status(self):
        status = Workbench().plugin_status()
        assert status["errors"] == {}
        assert "refinery" in status["installed"]


# =============================================================================
# Building the editor examples from tool calls alone
# =============================================================================


class TestBuildFromScratch:
    def test_the_flash_drum(self):
        wb = Workbench()
        assert wb.set_code_context(context_of("01_flash_drum"))["ok"]
        added = wb.add_unit("Flash", "flash")
        # Defining `species` in the context is not naming the flowsheet's
        # species, and the reply says what to do about it.
        assert added["pending"] and "species" in added["hint"]
        assert wb.set_species(["water", "ethanol"])["promoted"] == ["flash"]
        assert wb.get_flowsheet()["unfed_inlets"] == ["flash_in"]
        assert wb.set_feed("flash_in", T=362.0, P=101325.0,
                           flows={"water": 1.0, "ethanol": 1.0})["ok"]
        answer = wb.solve()
        assert answer["converged"] and answer["audit"]["warnings"] == []
        ref = reference("01_flash_drum")
        mine = wb.get_streams()["streams"]
        assert_same(mine["flash_out"], ref["liquid"])
        assert_same(mine["flash_out2"], ref["vapor"])

    def test_the_reactor_and_flash(self):
        wb = Workbench()
        assert wb.set_code_context(context_of("02_reactor_flash"))["ok"]
        assert wb.set_species(ESTERIFICATION)["ok"]
        added = wb.add_unit("CSTR", "reactor",
                            params={"V": 0.5, "molar_density": 17000.0})
        assert added["ok"] and added["placeholders"] == []
        assert wb.add_unit("Flash", "flash")["ok"]
        assert wb.connect("reactor", "reactor_out", "flash", "flash_in")["kind"] == "arc"
        assert wb.set_feed("reactor_in", T=366.0, P=101325.0,
                           flows={"acetic_acid": 1.0, "ethanol": 1.0,
                                  "ethyl_acetate": 0.0, "water": 0.0})["ok"]
        answer = wb.solve()
        assert answer["converged"] and answer["warnings"] == []
        ref = reference("02_reactor_flash")
        mine = wb.get_streams()["streams"]
        assert_same(mine["flash_out2"], ref["vapor"])
        assert_same(mine["flash_out"], ref["liquid"])

    def test_the_reactor_recycle(self):
        wb = Workbench()
        assert wb.set_code_context(context_of("03_reactor_recycle"))["ok"]
        assert wb.set_species(ESTERIFICATION)["ok"]
        for op, name, params in [("Mixer", "mixer", None),
                                 ("CSTR", "reactor", {"V": 0.5, "molar_density": 17000.0}),
                                 ("Flash", "flash", None),
                                 ("Splitter", "splitter", None)]:
            assert wb.add_unit(op, name, params=params)["ok"]
        assert wb.update_unit("splitter", call_params={"split_frac": 0.7})["ok"]
        wires = [("mixer", "mixer_out", "reactor", "reactor_in"),
                 ("reactor", "reactor_out", "flash", "flash_in"),
                 ("flash", "flash_out", "splitter", "splitter_in")]
        for wire in wires:
            assert wb.connect(*wire)["kind"] == "arc"
        # Closing the loop makes the tear; nobody has to declare it.
        closing = wb.connect("splitter", "splitter_out", "mixer", "mixer_in2")
        assert closing["kind"] == "recycle"
        assert wb.set_feed("mixer_in", T=366.0, P=101325.0,
                           flows={"acetic_acid": 1.0, "ethanol": 1.0,
                                  "ethyl_acetate": 0.0, "water": 0.0})["ok"]
        answer = wb.solve()
        assert answer["converged"] and answer["audit"]["warnings"] == []
        assert 0 < answer["gain"] < 1
        ref = reference("03_reactor_recycle")
        mine = wb.get_streams()["streams"]
        assert_same(mine["flash_out2"], ref["vapor"], rel=1e-7)
        assert_same(mine["splitter_out2"], ref["purge"], rel=1e-7)

    def test_a_forgotten_density_is_reported_by_the_solve(self):
        """The mistake this test file's author made first: no molar_density.

        The CSTR falls back to water's density and warns; the warning used
        to go to stderr, and now comes back with the answer.
        """
        wb = Workbench()
        wb.set_code_context(context_of("02_reactor_flash"))
        wb.set_species(ESTERIFICATION)
        wb.add_unit("CSTR", "reactor", params={"V": 0.5})
        wb.set_feed("reactor_in", flows={"acetic_acid": 1.0, "ethanol": 1.0})
        categories = {w["category"] for w in wb.solve()["warnings"]}
        assert "CSTRDensityWarning" in categories


# =============================================================================
# Guard rails
# =============================================================================


class TestGuardRails:
    def test_bad_params_leave_no_unit_behind(self):
        wb = Workbench()
        wb.set_code_context(context_of("02_reactor_flash"))
        wb.set_species(ESTERIFICATION)
        answer = wb.add_unit("CSTR", "reactor", params={"V": "big"})
        assert not answer["ok"]
        assert wb.get_flowsheet()["units"] == []

    def test_streams_before_a_solve(self):
        wb = Workbench()
        wb.open_example("01_flash_drum")
        assert "solve" in wb.get_streams()["error"]

    def test_sessions_are_independent(self):
        wb = Workbench()
        assert wb.new_session("b")["ok"]
        assert wb.open_example("01_flash_drum", session="b")["ok"]
        assert wb.get_flowsheet()["units"] == []
        assert not wb.new_session("b")["ok"]
        assert not wb.solve(session="nope")["ok"]

    def test_no_exec_refuses_every_code_path(self):
        wb = Workbench(allow_exec=False)
        assert "disabled" in wb.run_python("1 + 1")["error"]
        assert "disabled" in wb.set_code_context("x = 1")["error"]
        assert "disabled" in wb.open_file("plant.py")["error"]

    def test_a_cell_past_its_timeout_returns_and_the_session_waits(self):
        wb = Workbench()
        wb.open_example("01_flash_drum")
        answer = wb.run_python("import time; time.sleep(1.5)", timeout=0.2)
        assert answer["timed_out"]
        busy = wb.solve()
        assert not busy["ok"] and "busy" in busy["error"]
        assert wb.list_sessions()["sessions"][0]["busy"]
        time.sleep(2.0)
        assert wb.run_python("1 + 1")["outputs"][-1]["text"] == "2"

    def test_results_are_json(self):
        """inf and nan are not JSON; the error estimate can be either."""
        value = jsonable({"a": float("inf"), "b": [float("nan"), 1.0], "c": {1, 2}})
        assert json.loads(json.dumps(value)) == {"a": "inf", "b": ["nan", 1.0],
                                                 "c": [1, 2]}

    def test_every_tool_is_a_workbench_method_with_a_docstring(self):
        for name, kind in TOOLS.items():
            assert kind in ("read", "edit", "exec")
            method = getattr(Workbench, name)
            assert (method.__doc__ or "").strip(), name
