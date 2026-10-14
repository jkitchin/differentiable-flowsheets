"""Plugin agent support (MCP phase 4): the ``difflow.agent`` entry point.

Every installed plugin brings its own guide through the entry point, core
names none of them, and a plugin whose support fails to load is reported
rather than taking the server down. The power tools are checked against
MATPOWER's published answers, the same numbers tests/power asserts.
"""

import importlib.metadata

import pytest

from difflow.agent import Workbench, plugins
from difflow.agent.doctor import _symptom_findings

PLUGINS = {"bio", "cc", "gas", "power", "ree", "refinery"}


def test_every_plugin_has_a_guide():
    support, errors = plugins.load(force=True)
    assert errors == {}
    assert set(support) == PLUGINS
    for name, sup in support.items():
        assert sup.plugin == name and sup.summary and sup.solver_notes, name


def test_the_guide_tool():
    wb = Workbench()
    listing = wb.plugin_guide()
    assert set(listing["plugins"]) == PLUGINS
    assert listing["plugins"]["power"]["tools"] == ["power_flow", "power_opf"]
    gas = wb.plugin_guide("difflow_gas")
    assert "clip_negative_flows=False" in gas["solver_notes"]
    assert not wb.plugin_guide("nuclear")["ok"]


def test_plugin_symptoms_reach_diagnose():
    titles = [f.detail for f in _symptom_findings("HydrotreaterConvergenceWarning")]
    assert any("refinery unit's own solve" in t for t in titles)


def test_a_failing_plugin_is_reported_not_raised(monkeypatch):
    class Broken:
        name = "broken"
        value = "nowhere:support"

        def load(self):
            raise ImportError("no module named nowhere")

    real = importlib.metadata.entry_points

    def entry_points(group=None, **kw):
        found = list(real(group=group, **kw))
        return found + [Broken()] if group == plugins.GROUP else found

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    support, errors = plugins.load(force=True)
    assert "broken" in errors and PLUGINS <= set(support)
    monkeypatch.undo()
    plugins.load(force=True)


def test_a_bad_case_is_an_answer():
    answer = Workbench().plugin_tool("power_flow", case="case99")
    assert not answer["ok"] and "case9" in answer["error"]


def test_plugin_tools_are_served():
    pytest.importorskip("mcp")
    import asyncio

    from mcp import Client

    from difflow.mcp import build_server

    async def go():
        async with Client(build_server()) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    tools = asyncio.run(go())
    assert {"power_flow", "power_opf", "plugin_guide"} <= set(tools)
    assert tools["power_opf"].annotations.read_only_hint is True
    assert tools["power_opf"].input_schema["properties"]["kind"]["default"] == "ac"
