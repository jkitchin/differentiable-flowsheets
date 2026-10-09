"""Outlet roles, progress, prompts and the editor link (MCP polish)."""

import asyncio
import json
import os
import signal
import time
import urllib.request

import pytest

from difflow.agent import Workbench
from difflow.catalog import _returned_names, describe_operation


class TestOutletRoles:
    @pytest.mark.parametrize("operation, roles", [
        ("Flash", ["liquid", "vapor"]),
        ("ShortcutColumn", ["distillate", "bottoms"]),
        ("CounterCurrentHX", ["hot_outlet", "cold_outlet"]),
        ("ComponentSeparator", ["residue", "product"]),     # class docstring
        ("EnthalpyCounterCurrentHX", ["hot_outlet", "cold_outlet"]),  # inline
    ])
    def test_roles_come_from_the_docstrings(self, operation, roles):
        assert describe_operation(operation).ports.outlet_roles == roles

    def test_a_count_that_disagrees_is_not_trusted(self):
        assert _returned_names("Returns:\n    a: x\n    b: y\n    info: z") == ["a", "b"]
        assert describe_operation("Mixer").ports.outlet_roles is None or \
            len(describe_operation("Mixer").ports.outlet_roles) == 1

    def test_add_unit_and_the_summary_name_them(self):
        wb = Workbench()
        added = wb.add_unit("ShortcutColumn", "col")
        assert added["outlet_roles"] == {"col_out": "distillate", "col_out2": "bottoms"}
        wb.open_example("01_flash_drum")
        flash = wb.get_flowsheet()["units"][0]
        assert flash["outlet_roles"] == {"liquid": "liquid", "vapor": "vapor"}


def test_open_in_editor_starts_the_editor(tmp_path):
    wb = Workbench()
    wb.open_example("01_flash_drum")
    answer = wb.open_in_editor(path=str(tmp_path / "drum.json"), browser=False)
    try:
        assert answer["ok"] and (tmp_path / "drum.json").exists()
        for _ in range(60):
            try:
                assert urllib.request.urlopen(answer["url"], timeout=2).status == 200
                break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("the editor never answered")
    finally:
        os.killpg(answer["pid"], signal.SIGTERM)


def test_a_session_without_a_file_needs_a_path():
    assert "path" in Workbench().open_in_editor()["error"]


class TestServer:
    @pytest.fixture(autouse=True)
    def _mcp(self):
        pytest.importorskip(
            "mcp.server.mcpserver",
            reason="difflow.mcp is written against mcp 2 (MCPServer); the installed mcp is older or absent",
        )

    def test_long_tools_send_progress_and_hide_the_context(self, monkeypatch):
        from mcp import Client

        import difflow.mcp.server as srv

        monkeypatch.setattr(srv, "HEARTBEAT", 0.2)
        seen = []

        async def on_progress(progress, total, message):
            seen.append(message)

        async def go():
            async with Client(srv.build_server()) as client:
                tools = {t.name: t for t in (await client.list_tools()).tools}
                await client.call_tool("open_example", {"key": "01_flash_drum"})
                result = await client.call_tool(
                    "run_python", {"code": "import time; time.sleep(1); 7"},
                    progress_callback=on_progress)
                return tools, json.loads(result.content[0].text)

        tools, answer = asyncio.run(go())
        assert "ctx" not in tools["solve"].input_schema["properties"]
        assert answer["outputs"][-1]["text"] == "7"
        assert seen and "run_python: still running" in seen[0]

    def test_prompts(self):
        from mcp import Client

        from difflow.mcp import build_server

        async def go():
            async with Client(build_server()) as client:
                names = [p.name for p in (await client.list_prompts()).prompts]
                text = (await client.get_prompt("design_flowsheet",
                                                {"description": "a flash drum"}))
                return names, text.messages[0].content.text

        names, text = asyncio.run(go())
        assert names == ["design_flowsheet", "fix_convergence", "sensitivity_study"]
        assert "a flash drum" in text
