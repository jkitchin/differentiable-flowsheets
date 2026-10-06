"""``difflow mcp``: the :mod:`difflow.agent` tools, served over MCP.

This file is the only part of difflow that knows the MCP SDK. The tools
themselves are :class:`difflow.agent.Workbench` methods; here they are
registered with their docstrings as descriptions and with annotations
from :data:`difflow.agent.TOOLS`, so a client can tell the read-only ones
from those that change a flowsheet or run Python, and ask before the
latter. Register it with Claude Code as::

    claude mcp add difflow -- difflow mcp

The SDK is optional (``pip install "difflow[mcp]"``) and imported only
when a server is built.
"""

from __future__ import annotations

import argparse
import functools

from difflow.agent import DEFAULT_TIMEOUT, TOOLS, Workbench

INSTRUCTIONS = """\
difflow is a JAX-based differentiable flowsheet simulator. A session holds
one flowsheet ("main" unless you name another).

Building: search operations with list_operations and read one with
describe_operation (parameters, ports, and what it still needs). Name the
species with set_species (check names with search_species). Units that need
objects (a thermo or EOS, a rate law) get them from the code context: set it
with set_code_context, using describe_operation's starter_code. add_unit
places a unit with its own stream names on every port; connect wires an
outlet stream to an inlet stream, and a connection that closes a loop
becomes a recycle. set_feed puts a feed on each unfed inlet (get_flowsheet
lists them). Ports are stream names, and a stream's name is its wiring.

Solving: solve reports `converged`, the loop `gain` and `error_estimate`,
captured `warnings` and an `audit` (mass balance, negative flows, NaN). An
answer is good only when it converged AND the audit is clean. Read values
with get_streams.

Libraries that are not unit operations (planning, stochastic, estimation,
economics, the refinery reformer and FCC) are found with list_api and
describe_api, and used through run_python, where `fs` is the live flowsheet.
Prefer the structured tools to run_python so diagnostics and undo are kept.
search_docs searches the difflow book.
"""


def build_server(workbench: Workbench | None = None, *, allow_exec: bool = True,
                 timeout: float = DEFAULT_TIMEOUT):
    """An MCP server over a :class:`Workbench`.

    Args:
        workbench: The sessions to serve; a new one when omitted.
        allow_exec: Register the tools that run Python (``run_python``,
            ``set_code_context`` and ``open_file``). Without them a
            ``.py`` flowsheet cannot be opened either.
        timeout: Default seconds before a solve or Python call returns.

    Raises:
        ImportError: If the MCP SDK is not installed, naming the extra.
    """
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise ImportError(
            "difflow mcp needs the MCP SDK: pip install \"difflow[mcp]\" "
            f"(the PyPI distribution is 'mcp'). ({exc})"
        ) from exc
    import difflow

    wb = workbench or Workbench(allow_exec=allow_exec, timeout=timeout)
    wb.allow_exec = allow_exec
    server = MCPServer(name="difflow", version=difflow.__version__,
                       instructions=INSTRUCTIONS, log_level="WARNING")
    for name, kind in TOOLS.items():
        if kind == "exec" and not allow_exec:
            continue
        method = getattr(wb, name)

        # A plain function with the method's signature and docstring: the
        # SDK builds the input schema from the one and the description
        # from the other.
        @functools.wraps(method)
        def tool(*args, __method=method, **kwargs):
            return __method(*args, **kwargs)

        server.tool(
            name=name,
            annotations=ToolAnnotations(
                readOnlyHint=kind == "read",
                destructiveHint=kind == "exec",
                idempotentHint=kind == "read",
                openWorldHint=False,
            ),
            structured_output=False,
        )(tool)
    return server


def main(argv: list[str] | None = None) -> int:
    """Run the server over stdio."""
    parser = argparse.ArgumentParser(
        prog="difflow mcp",
        description="Serve difflow's flowsheet tools to an MCP client over stdio.")
    parser.add_argument(
        "--no-exec", action="store_true",
        help="leave out the tools that run Python (run_python, "
             "set_code_context, open_file); use for shared or hosted clients")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"seconds before a solve or Python call returns and keeps "
             f"running in the background (default {DEFAULT_TIMEOUT:g})")
    args = parser.parse_args(argv)
    try:
        server = build_server(allow_exec=not args.no_exec, timeout=args.timeout)
    except ImportError as exc:
        parser.exit(2, f"difflow mcp: {exc}\n")
    server.run("stdio")
    return 0
