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
        tool = _with_progress(name, method) if _is_long(method) else \
            _plain(method)
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
    _register_plugin_tools(server, wb, allow_exec, ToolAnnotations)
    _register_prompts(server)
    return server


def _register_prompts(server) -> None:
    """Workflows a client can offer as one-click starting points."""

    @server.prompt(name="design_flowsheet",
                   description="Build and solve a flowsheet from a process description")
    def design_flowsheet(description: str) -> str:
        return (
            f"Build this process as a difflow flowsheet and solve it:\n\n{description}\n\n"
            "Find each operation with list_operations and read it with "
            "describe_operation before adding it; check species with "
            "search_species and set them first. Supply objects a unit needs "
            "(thermo, rate laws) through set_code_context using the starter "
            "code describe_operation returns. Wire outlets to inlets with "
            "connect (outlet_roles says which outlet is which), put a feed on "
            "every unfed inlet, then solve. Report the result only if it "
            "converged and its audit is clean; otherwise run diagnose.")

    @server.prompt(name="fix_convergence",
                   description="Find out why a flowsheet does not converge, and fix it")
    def fix_convergence(session: str = "main") -> str:
        return (
            f"The flowsheet in session {session!r} does not solve correctly. "
            "Run diagnose and read the findings in order. Then run converge "
            "(without apply) and compare the trials. Apply a numerics remedy "
            "with converge(apply=True) only if one passed; propose any problem "
            "remedy, a tear change or a model change to the user with the "
            "evidence rather than making it. If two trials disagree on the "
            "products, report the multiple steady states.")

    @server.prompt(name="sensitivity_study",
                   description="Rank what an output responds to, and by how much")
    def sensitivity_study(output: str, session: str = "main") -> str:
        return (
            f"In session {session!r}, study what {output!r} responds to. "
            "Solve and confirm convergence, define the output as a named "
            "quantity if it is reused, run sensitivity over all levers, and "
            "sweep the two levers with the largest elasticities over a "
            "sensible range to check the derivative holds beyond the point. "
            "Report elasticities with units and the convergence at every point.")


#: seconds between progress notifications during a long call
HEARTBEAT = 5.0


def _is_long(method) -> bool:
    """A tool that can run long is one that takes a timeout."""
    import inspect

    return "timeout" in inspect.signature(method).parameters


def _plain(method):
    """A function with the method's signature and docstring: the SDK builds
    the input schema from the one and the description from the other."""
    @functools.wraps(method)
    def tool(*args, __method=method, **kwargs):
        return __method(*args, **kwargs)

    return tool


def _with_progress(name: str, method):
    """An async tool that runs ``method`` on a worker thread and sends a
    progress notification every :data:`HEARTBEAT` seconds until it ends.

    A first solve of a large unit can compile for minutes; without these a
    client sees nothing and may give up on a call that is working. The
    notifications go out only when the client asked for them (sent a
    progress token).
    """
    import inspect
    import time
    import typing

    import anyio
    import anyio.to_thread
    from mcp.server.mcpserver import Context

    hints = typing.get_type_hints(method)

    async def tool(ctx: Context, **kwargs):
        done = anyio.Event()
        start = time.monotonic()

        async def heartbeat():
            while not done.is_set():
                with anyio.move_on_after(HEARTBEAT):
                    await done.wait()
                if not done.is_set():
                    elapsed = time.monotonic() - start
                    try:
                        await ctx.report_progress(
                            elapsed, None, f"{name}: still running ({elapsed:.0f} s)")
                    except Exception:  # noqa: BLE001 -- progress is best effort
                        pass

        async with anyio.create_task_group() as group:
            group.start_soon(heartbeat)
            try:
                result = await anyio.to_thread.run_sync(
                    functools.partial(method, **kwargs))
            finally:
                done.set()
        return result

    signature = inspect.signature(method)
    ctx_param = inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY,
                                  annotation=Context)
    tool.__name__ = name
    tool.__doc__ = method.__doc__
    tool.__signature__ = signature.replace(
        parameters=list(signature.parameters.values()) + [ctx_param])
    tool.__annotations__ = {**hints, "ctx": Context}
    return tool


def _register_plugin_tools(server, wb: Workbench, allow_exec: bool,
                           ToolAnnotations) -> None:
    """Serve each plugin's own tools as ``<plugin>_<name>``.

    A plugin tool is ``function(workbench, **arguments)``; what the client
    sees is the signature after ``workbench``, so the schema comes from the
    plugin's own type hints and the description from its docstring.
    """
    import inspect

    from difflow.agent import plugins

    for name, (_, tool) in sorted(plugins.tools().items()):
        if tool.kind == "exec" and not allow_exec:
            continue
        signature = inspect.signature(tool.function)
        params = list(signature.parameters.values())[1:]

        def call(*, __name=name, **kwargs):
            return wb.plugin_tool(__name, **kwargs)

        call.__name__ = name
        call.__doc__ = tool.function.__doc__
        call.__signature__ = signature.replace(parameters=params)
        call.__annotations__ = {k: v for k, v in tool.function.__annotations__.items()
                                if k != "workbench"}
        call.__module__ = tool.function.__module__
        server.tool(
            name=name,
            annotations=ToolAnnotations(
                readOnlyHint=tool.kind == "read",
                destructiveHint=tool.kind == "exec",
                idempotentHint=tool.kind == "read",
                openWorldHint=False,
            ),
            structured_output=False,
        )(call)


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
