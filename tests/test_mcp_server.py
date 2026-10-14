"""``difflow mcp``: the agent tools as an MCP client sees them.

Driven through the SDK's in-process client, so what is tested is the
protocol view: the tool list, the schemas built from the signatures, the
annotations a client uses to decide what to ask before calling, and the
round trip of a call.
"""

import asyncio

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from difflow.agent import TOOLS  # noqa: E402
from difflow.mcp import build_server  # noqa: E402


def run(coro):
    return asyncio.run(coro)


async def tools_of(server) -> dict:
    async with Client(server) as client:
        return {t.name: t for t in (await client.list_tools()).tools}


def test_every_tool_is_served_with_its_kind():
    from difflow.agent import plugins

    tools = run(tools_of(build_server()))
    # The core tools plus whatever installed plugins contribute.
    assert set(tools) == set(TOOLS) | set(plugins.tools())
    for name, kind in TOOLS.items():
        hints = tools[name].annotations
        assert hints.read_only_hint is (kind == "read"), name
        assert hints.destructive_hint is (kind == "exec"), name
        assert tools[name].description, name


def test_no_exec_leaves_out_the_tools_that_run_python():
    tools = run(tools_of(build_server(allow_exec=False)))
    assert {"run_python", "set_code_context", "open_file"}.isdisjoint(tools)
    assert "solve" in tools


def test_schemas_come_from_the_signatures():
    schema = run(tools_of(build_server()))["add_unit"].input_schema
    assert schema["required"] == ["operation"]
    assert schema["properties"]["session"]["default"] == "main"


def test_a_missing_sdk_names_the_extra(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mcp)
    with pytest.raises(ImportError, match=r"difflow\[mcp\]"):
        build_server()
