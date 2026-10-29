"""difflow.agent: the operations behind ``difflow mcp``, called directly.

The phase-1 bar for the agent tools is that the three editor examples can
be built from nothing but tool calls and solve to the same answer as the
shipped files. Those rebuilds are here, with the discovery calls an agent
uses to get there and the guard rails (timeouts, --no-exec, rollback).
"""

import json
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


class TestLiveDebugger:
    """The debug_* tools: the session's live debugger, as an agent drives it."""

    @staticmethod
    def bench(middle=None):
        from difflow.gui.session import FlowsheetSession
        from tests.test_debug import loop

        wb = Workbench()
        wb.sessions["main"] = FlowsheetSession(loop(middle) if middle else loop())
        return wb

    def test_the_guide_is_the_contract_and_the_handshake(self):
        guide = Workbench().debug_guide()
        assert guide["hello"]["protocol"] == "difflow-dbg/1"
        assert "debug_wait" in guide["contract"]

    def test_start_step_and_finish(self):
        wb = self.bench()
        assert wb.debug_state()["state"] == "idle"
        first = wb.debug_start()
        assert (first["state"], first["event"]["checkpoint"]) == ("paused", "solve_start")
        stepped = wb.debug_command("step")
        assert stepped["event"]["next"]["checkpoint"] == "iteration"
        assert wb.debug_state()["event"]["checkpoint"] == "iteration"
        wb.debug_command("continue")
        end = wb.debug_command("continue")
        assert end["state"] == "finished" and end["event"]["next"]["ok"]
        assert wb.debug_stop() == {"ok": True, "state": "idle"}

    def test_on_error_stops_where_the_nan_is_born(self):
        from tests.test_debug import poisoned

        wb = self.bench(poisoned)
        stop = wb.debug_start(on_error=True)
        assert (stop["event"]["unit"], stop["event"]["trigger"]) == ("R1", "nonfinite")
        wb.debug_stop()

    def test_a_short_wait_says_running_and_wait_picks_it_up(self):
        from tests.test_debug import slow

        wb = self.bench(slow)
        wb.debug_start()
        running = wb.debug_command("continue", timeout=0.01)
        assert running["state"] == "running"
        assert wb.debug_pause()["event"]["ok"]
        paused = wb.debug_wait(timeout=60)
        assert paused["state"] == "paused" and paused["event"]["reason"] == "paused"
        wb.debug_stop()

    def test_a_missing_session_and_a_missing_debugger_are_refused(self):
        wb = Workbench()
        assert not wb.debug_start(session="nowhere")["ok"]
        assert "start one" in wb.debug_command("step")["error"]
        assert json.loads(json.dumps(wb.debug_state()))["state"] == "idle"

    def test_record_frame_and_what_if(self):
        wb = self.bench()
        recorded = wb.debug_record()
        assert recorded["ok"], recorded
        summary = recorded["summary"]
        k = summary["iterations"][0]["frame"]
        assert wb.sessions["main"].flowsheet.last_solve_history is None   # a copy was solved

        shown = wb.debug_trace_frame(k, streams=["recycle"])
        assert shown["frame"]["checkpoint"] == "iteration"
        assert set(shown["streams"]) == {"recycle"}

        replay = wb.debug_whatif(k, "recycle", "F_A", 5.0)
        assert replay["ok"], replay
        comparison = replay["comparison"]
        assert comparison["diverged_at"] is None and comparison["same_answer"]
        assert [e["value"] for e in comparison["edits"]] == [5.0]
        assert wb.debug_trace_frame(k, which="whatif")["frame"]["edits"]
        json.dumps(replay)

    def test_a_what_if_can_find_another_steady_state(self):
        from difflow.gui.session import FlowsheetSession
        from tests.test_debug import bistable

        wb = Workbench()
        wb.sessions["main"] = FlowsheetSession(bistable())
        wb.sessions["main"].set_solver_options({"use_initialization": False})
        k = wb.debug_record()["summary"]["iterations"][0]["frame"]
        comparison = wb.debug_whatif(k, "tear", "F_A", 0.8)["comparison"]
        assert comparison["same_answer"] is False
        assert comparison["largest_difference"]["where"] == "out.F_A"

    def test_record_and_replay_refusals(self):
        wb = self.bench()
        assert "debug_record" in wb.debug_trace_frame(0)["error"]
        assert "record a solve first" in wb.debug_whatif(0, "recycle", "F_A", 1.0)["error"]
        wb.debug_record()
        assert "debug_whatif" in wb.debug_trace_frame(0, which="whatif")["error"]
        assert not wb.debug_trace_frame(10_000)["ok"]
        assert "known:" in wb.debug_trace_frame(0, streams=["nope"])["error"]
        assert not wb.debug_whatif(0, "recycle", "F_A", 1.0)["ok"]      # not an iteration
