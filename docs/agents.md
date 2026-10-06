# Agents: the `difflow mcp` server

`difflow mcp` serves difflow's flowsheet tools to an AI agent over the
[Model Context Protocol](https://modelcontextprotocol.io). An agent connected
to it can find unit operations, build and edit a flowsheet, solve it, read the
streams, and run Python in the same session, with the same engine the browser
editor uses.

## Installing and connecting

The server needs the MCP SDK, which is an optional extra:

```bash
pip install "difflow[mcp]"
```

Register it with Claude Code (or any MCP client that launches a command over
stdio):

```bash
claude mcp add difflow -- difflow mcp
```

Two options matter:

| Option | Effect |
|---|---|
| `--no-exec` | Leaves out the three tools that run Python: `run_python`, `set_code_context` and `open_file`. Use it for a shared or hosted client. |
| `--timeout S` | Seconds a solve or a Python call may run before the tool returns (default 300). The call keeps running in the background and its session refuses other calls until it ends. A first solve of a large unit can take minutes to compile. |

## What the tools are

Nothing in the tool layer is a hand-written list of units. Operations come
from the plugin registry and the catalog, so a newly installed plugin's units
are usable at once; libraries that are not unit operations (planning,
stochastic programming, the refinery reformer and FCC) are found by reading
each package's `__all__`.

| Group | Tools |
|---|---|
| Discovery | `plugin_status`, `list_operations`, `describe_operation`, `list_api`, `describe_api`, `search_species`, `search_docs`, `list_examples` |
| Sessions | `list_sessions`, `new_session`, `close_session`, `open_example`, `open_file`, `save`, `undo`, `redo`, `get_flowsheet` |
| Building | `set_species`, `set_code_context`, `add_unit`, `update_unit`, `remove_unit`, `connect`, `disconnect`, `set_feed`, `remove_feed` |
| Running | `set_solver_options`, `solve`, `get_streams` |
| Python | `run_python` |

Each tool is annotated as read-only, editing, or running code, so a client can
ask before the last kind. A session holds one flowsheet; every tool takes a
`session` argument (default `"main"`), so a base case and a variant can be
kept side by side.

## How a flowsheet is built

Ports are stream names, and a stream's name is its wiring. `add_unit` places a
unit with its own stream name on every port; `connect` joins one unit's outlet
stream to another's inlet stream, and a connection that closes a loop becomes
a recycle without being declared. `set_feed` puts a feed on an inlet nothing
else supplies, and `get_flowsheet` lists the inlets still waiting for one.

Units that need objects rather than numbers (a thermo or EOS object, a rate
law) get them from the *code context*, a Python snippet saved with the
flowsheet. `describe_operation` reports what a unit still needs in the current
session and returns starter code that supplies it. A unit added before its
needs are met is placed as *pending* and builds itself once they are.

## Reading a solve

`solve` returns more than streams, and an answer is good only when it
converged **and** its audit is clean:

- `converged`: the recycle verdict.
- `gain` and `error_estimate`: at loop gain $g$ the error is about $1/(1-g)$
  times the step `tol` tested (see {doc}`convergence`).
- `warnings`: what the solve warned about (a `TearToleranceWarning`, a
  `CSTRDensityWarning`, ...), returned instead of printed.
- `audit`: non-finite values, negative flows, unread feeds and the overall
  mass balance.

Every option of `Flowsheet.solve` that is a setting (tolerance and its basis,
acceleration, damping, tear choice, clipping) can be set with
`set_solver_options` or passed to `solve`, and is saved with the flowsheet.

## Using the tools without MCP

The tools are methods of `difflow.agent.Workbench`, which does not need the
SDK:

```python
from difflow.agent import Workbench

wb = Workbench()
wb.open_example("03_reactor_recycle")
answer = wb.solve()
answer["converged"], answer["gain"]
```
