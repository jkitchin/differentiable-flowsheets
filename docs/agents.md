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
| Discovery | `plugin_status`, `plugin_guide`, `list_operations`, `describe_operation`, `list_api`, `describe_api`, `search_species`, `search_docs`, `list_examples` |
| Sessions | `list_sessions`, `new_session`, `close_session`, `open_example`, `open_file`, `save`, `open_in_editor`, `undo`, `redo`, `get_flowsheet` |
| Building | `set_species`, `set_code_context`, `add_unit`, `update_unit`, `remove_unit`, `connect`, `disconnect`, `set_feed`, `remove_feed` |
| Running | `set_solver_options`, `solve`, `get_streams` |
| Diagnosis | `diagnose`, `converge`, `tear_analysis`, `trace_solve`, `get_unit_info` |
| Analysis | `levers`, `define_quantity`, `remove_quantity`, `list_quantities`, `evaluate`, `sensitivity`, `sweep`, `optimize`, `uncertainty`, `linearize`, `tea`, `report` |
| Python | `run_python` |

Each tool is annotated as read-only, editing, or running code, so a client can
ask before the last kind. A tool that can run long (a solve, a search, Python)
sends a progress notification every few seconds while it works, so a client
waiting on a first compile knows the call is alive. The server also offers
three prompts, `design_flowsheet`, `fix_convergence` and `sensitivity_study`,
as starting points for those workflows.

`open_in_editor` saves the flowsheet and opens it in the browser editor, so a
person can see what an agent built. The editor runs on the saved file in its
own process; its edits reach the agent's session when `open_file` reads the
file again. A session holds one flowsheet; every tool takes a
`session` argument (default `"main"`), so a base case and a variant can be
kept side by side.

## How a flowsheet is built

Ports are stream names, and a stream's name is its wiring. `add_unit` places a
unit with its own stream name on every port, and says what each outlet is
(`outlet_roles`: which is the vapor, which the distillate), read from the
unit's own documentation; `connect` joins one unit's outlet
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

## When a solve fails

`diagnose` checks what can be checked before solving (units still waiting on
something, unfed inlets), solves, and returns findings, most serious first:
a recycle that did not converge together with what its residual history
says (diverging, oscillating, creeping, stalled), an error estimate above
`tol`, clipping, a unit whose own inner solve did not close its balance,
tear-set problems, audit failures and captured warnings. Each finding names
the remedies that address it. The remedies and the symptom cards live in
`difflow.diagnostics`, the same table the editor's assistant reads.

`converge` tries those remedies on the live flowsheet: the stored settings
first, then Anderson, more iterations, Wegstein, damped substitution, an
error-based tolerance, unclipped tears and a cold start. A trial passes only
when it converged **and** is correct: finite, mass conserved, every unit's
inner solve closed, and the error not far above `tol`. Negative flows are
reported as a caveat rather than a failure, because a signed tear is negative
in the right answer. When two trials pass with different products, the
flowsheet has more than one steady state, and `converge` says so instead of
picking one.

By default `converge` changes nothing. With `apply=True` it keeps the first
passing remedy, but only a *numerics* one, which changes how the fixed point
is reached and not which one. A remedy that can change the answer (unclipped
tears, a cold start) is returned as a proposal to apply with
`set_solver_options`.

## Objectives as expressions

Analysis in difflow takes functions, and a function cannot cross an MCP
boundary, so the analysis tools take *expressions*: strings compiled to JAX
functions of the solved flowsheet, which differentiate exactly as the same
function written in Python would.

```text
vapor.F_ethyl_acetate / vapor.total_flow
20.0 * vapor.F_ethyl_acetate - 0.5 * reactor.V - 1.0 * feed.total_flow
```

An expression may use `<stream>.<quantity>` (`T`, `P`, `total_flow`,
`F_<species>`), `<unit>.<param>` (the value at the point evaluated, so it
follows a lever), named quantities, `+ - * / **`, numbers, `exp log log10 sqrt
abs min max`, and `econ.<function>` for the functions of `difflow.economics`.
It is parsed against a whitelist and never evaluated as Python.
`define_quantity` names an expression (`purity`, `revenue`, `capex`) for reuse
in others; quantities are saved with the flowsheet.

Levers are what `levers` lists: `"<unit>.<param>"` and
`"feed:<stream>.<field>"`. With them:

- `sensitivity` ranks every lever by its elasticity, d ln y / d ln u, from
  one reverse pass through the converged solve;
- `sweep` solves along one lever and reports outputs and convergence at each
  point;
- `optimize` minimizes or maximizes an expression over bounded levers with
  constraints, using SLSQP with exact gradients, and with `apply=True` writes
  the optimum into the flowsheet (undoably);
- `uncertainty` propagates independent normal uncertainty in levers to an
  output, linearly from the gradient with each lever's share of the variance,
  and by Monte Carlo when `samples` is set;
- `linearize` builds the delta vectors of an LP planning model
  (see {doc}`planning`), and `report` the flowsheet's self-documenting report.

Every result that rests on a solve carries that solve's convergence verdict.

`tea` prices the flowsheet's capital from cost bases the units declare
themselves: a unit class may name a correlation in
`difflow.economics.capital` and the parameter that sizes it (a CSTR is a
jacketed vessel sized by `V`). Units with no basis are listed as uncosted
rather than given a size nobody chose, and a size outside a correlation's
range is flagged. The result is also written as two named quantities,
`purchased_equipment` and `capex`, so an objective such as
`-revenue + 0.1 * capex` trades capital against the process with exact
gradients.

## What each plugin adds

A plugin registers its unit operations through the `difflow.plugins` entry
point and its agent support through `difflow.agent`, whose target returns a
`difflow.agent.plugins.AgentSupport`: a summary of what it models, notes on
its own solvers, symptom cards that `diagnose` matches alongside the core
ones, and extra tools, served as `<plugin>_<name>`. Core names no plugin, so a
new plugin brings its own support.

`plugin_guide` lists the guides and shows one. Of the installed plugins:

| Plugin | What its support adds |
|---|---|
| power | `power_flow` and `power_opf` (AC or DC) on the benchmark cases or a MATPOWER case as JSON, with LMPs checked against `jax.grad` of the optimal cost |
| gas | signed flows (`clip_negative_flows=False`), damping, and a card that reads negative flows as direction rather than error |
| refinery | library-only units and how to reach them, compile times, and cards keyed to each unit's convergence warning |
| bio, cc, ree | what each unit needs before it can be built (a growth model, a solvent, an extractant) and the plugin's limits |

```toml
[project.entry-points."difflow.agent"]
power = "difflow_power.agent:support"
```

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
