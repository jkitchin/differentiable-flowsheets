# Streams and Flowsheets

This document covers stream handling, flowsheet management, and recycle calculations in Difflow.

## Table of Contents

1. [Stream Representation](#stream-representation)
2. [Stream Operations](#stream-operations)
3. [Flowsheet Management](#flowsheet-management)
4. [Recycle Calculations](#recycle-calculations)
5. [Plugin System](#plugin-system)

---

## Stream Representation

**Location**: `difflow/streams.py`

### Stream Structure

In Difflow, streams are represented as Python dictionaries containing:
- Molar flows for each species (`F_species`)
- Temperature (`T` in Kelvin)
- Pressure (`P` in Pascal)

```python
# Example stream structure
stream = {
    'F_methane': 10.0,    # mol/s
    'F_ethane': 5.0,      # mol/s
    'F_propane': 2.0,     # mol/s
    'T': 300.0,           # K
    'P': 500000.0         # Pa
}
```

### Creating Streams

```python
from difflow.streams import make_stream

# Create a stream
stream = make_stream(
    flows={'methane': 10.0, 'ethane': 5.0, 'propane': 2.0},
    T=300.0,
    P=500000.0
)
```

**Parameters**:
- `flows`: Dictionary of species names to molar flow rates (mol/s)
- `T`: Temperature (K)
- `P`: Pressure (Pa)

**Returns**: Stream dictionary with `F_` prefix added to species names

### JAX Pytree Compatibility

Streams are designed as JAX pytrees, making them compatible with:
- `jax.grad()`: Automatic differentiation
- `jax.jit()`: Just-in-time compilation
- `jax.vmap()`: Vectorization
- `jax.lax.scan()`: Efficient loops

```python
import jax
import jax.numpy as jnp

# Streams work seamlessly with JAX transformations
@jax.jit
def process_stream(stream):
    total = stream['F_methane'] + stream['F_ethane'] + stream['F_propane']
    return total

# Gradient through stream operations
def objective(inlet_T):
    stream = make_stream({'A': 1.0}, T=inlet_T, P=101325.0)
    outlet, _ = reactor(stream)
    return outlet['F_B']

grad_fn = jax.grad(objective)
sensitivity = grad_fn(350.0)
```

---

## Stream Operations

### Basic Stream Functions

```python
from difflow.streams import (
    get_flows,
    get_species,
    get_flow_array,
    total_flow,
    mole_fractions,
    combine_streams,
    scale_stream
)
```

#### Get Molar Flows

```python
# Extract flows as dictionary (without F_ prefix)
flows = get_flows(stream)
# {'methane': 10.0, 'ethane': 5.0, 'propane': 2.0}

# Get list of species names
species = get_species(stream)
# ['methane', 'ethane', 'propane']

# Get flows as JAX array (in specified order)
flow_array = get_flow_array(stream, species_order=['methane', 'ethane', 'propane'])
# Array([10., 5., 2.])
```

#### Total Flow Rate

```python
total = total_flow(stream)  # 17.0 mol/s
```

**Equation**:
$$F_{total} = \sum_i F_i$$

#### Mole Fractions

```python
x = mole_fractions(stream)
# {'methane': 0.588, 'ethane': 0.294, 'propane': 0.118}
```

**Equation**:
$$x_i = \frac{F_i}{\sum_j F_j}$$

#### Combining Streams

```python
# Combine multiple streams (adiabatic mixing)
stream1 = make_stream({'A': 1.0, 'B': 0.5}, T=350.0, P=101325.0)
stream2 = make_stream({'B': 0.3, 'C': 0.2}, T=360.0, P=101325.0)

combined = combine_streams(stream1, stream2)
```

**Equations**:

Mass balance:
$$F_{i,out} = \sum_k F_{i,k}$$

Energy balance (simplified, equal Cp):
$$T_{out} = \frac{\sum_k F_k T_k}{\sum_k F_k}$$

Pressure (minimum):
$$P_{out} = \min(P_1, P_2, ...)$$

#### Scaling Streams

```python
# Scale all flows by a factor
scaled = scale_stream(stream, factor=0.5)  # 50% of original flows
```

**Equation**:
$$F_{i,out} = \alpha \cdot F_{i,in}$$

Temperature and pressure are preserved.

---

## Flowsheet Management

**Location**: `difflow/flowsheet.py`

### Flowsheet Class

The `Flowsheet` class manages sequential modular process simulation.

```python
from difflow.flowsheet import Flowsheet, Unit

class Flowsheet:
    def __init__(self):
        self.feeds = {}        # Feed streams
        self.units = []        # Unit operations
        self.recycles = []     # Recycle connections
        self.streams = {}      # Computed stream results

    def add_feed(self, name: str, stream: dict): ...
    def add_unit(self, unit: Unit): ...
    def add_recycle(self, source: str, dest: str): ...
    def solve(self, tol: float = 1e-6, max_iter: int = 100): ...
```

### Unit Definition

```python
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

@dataclass
class Unit:
    name: str                    # Unique identifier
    operation: Callable          # Unit operation function
    inlet_names: List[str]       # Names of inlet streams
    outlet_names: List[str]      # Names of outlet streams
    params: Dict[str, Any]       # Operating parameters
```

### Building a Flowsheet

```python
from difflow.flowsheet import Flowsheet, Unit
from difflow.units.cstr import CSTR
from difflow.units.flash import Flash, Mixer

# Initialize flowsheet
fs = Flowsheet()

# Add feed streams
fs.add_feed('fresh_feed', make_stream({'A': 1.0, 'B': 0.0}, T=350.0, P=101325.0))

# Add mixer (combines fresh feed and recycle)
fs.add_unit(Unit(
    name='mixer',
    operation=Mixer(thermo, species),
    inlet_names=['fresh_feed', 'recycle'],
    outlet_names=['reactor_feed'],
    params={}
))

# Add reactor
fs.add_unit(Unit(
    name='reactor',
    operation=CSTR(reactor_params, thermo, species),
    inlet_names=['reactor_feed'],
    outlet_names=['reactor_effluent'],
    params={'T_spec': 380.0}
))

# Add flash drum
fs.add_unit(Unit(
    name='flash',
    operation=Flash(thermo, species),
    inlet_names=['reactor_effluent'],
    outlet_names=['vapor_product', 'liquid'],
    params={}
))

# Add splitter for recycle
fs.add_unit(Unit(
    name='splitter',
    operation=Splitter(),
    inlet_names=['liquid'],
    outlet_names=['liquid_product', 'recycle'],
    params={'fractions': [0.8, 0.2]}
))

# Define recycle connection
fs.add_recycle(source='recycle', dest='mixer')

# Solve flowsheet
results = fs.solve(tol=1e-6, max_iter=50)
```

### Sequential Modular Solution

The flowsheet solver uses sequential modular approach:

1. **Tear streams**: Identify recycle streams to "tear"
2. **Initialize**: Set initial guesses for tear streams
3. **Sequential calculation**: Solve units in order
4. **Update tear streams**: Compare calculated vs assumed
5. **Iterate**: Repeat until convergence

```
┌─────────────────────────────────────────────────────┐
│                   Flowsheet                          │
│                                                      │
│  Fresh Feed ──┬──► Mixer ──► Reactor ──► Flash      │
│               │              │           │          │
│               │              │      ┌────┴────┐     │
│               │              │      ▼         ▼     │
│               │              │   Vapor    Liquid    │
│               │              │             │        │
│               │              │      ┌──────┴──────┐ │
│               │              │      ▼             ▼ │
│               │         (Tear Stream)         Product│
│               │              │                      │
│               └──────────────┘                      │
│                  Recycle                            │
└─────────────────────────────────────────────────────┘
```

---

## Recycle Calculations

### Direct Substitution

Default method: simple fixed-point iteration on tear streams.

$$\mathbf{x}^{(k+1)} = f(\mathbf{x}^{(k)})$$

Where $\mathbf{x}$ is the tear stream vector and $f$ is the flowsheet calculation.

### Wegstein Acceleration

Accelerated convergence using Wegstein method:

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \frac{q}{1-q}[\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}]$$

Where:
$$q = \frac{f(\mathbf{x}^{(k)}) - f(\mathbf{x}^{(k-1)})}{\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}}$$

Bounded: $q \in [-5, 0]$ for stability.

### Broyden's Method

Quasi-Newton method for challenging convergence:

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \mathbf{B}^{-1} \mathbf{g}(\mathbf{x}^{(k)})$$

Where $\mathbf{g}(\mathbf{x}) = \mathbf{x} - f(\mathbf{x})$ and $\mathbf{B}$ is updated using Broyden's formula.

### Convergence Parameters

```python
results = fs.solve(
    tol=1e-6,           # Convergence tolerance
    max_iter=100,       # Maximum iterations
    method='wegstein',  # 'direct', 'wegstein', or 'broyden'
    damping=0.5         # Damping factor for direct substitution
)
```

### Implicit Differentiation

The flowsheet solver uses implicit differentiation through the converged solution:

$$\frac{d\mathbf{y}}{d\mathbf{p}} = \left(\mathbf{I} - \frac{\partial f}{\partial \mathbf{x}}\right)^{-1} \frac{\partial f}{\partial \mathbf{p}}$$

This is implemented using JAX custom VJP rules, enabling gradient computation through the entire flowsheet solve.

```python
import jax

# Gradient through flowsheet
def flowsheet_objective(feed_T):
    fs.feeds['fresh_feed'] = make_stream({'A': 1.0}, T=feed_T, P=101325.0)
    results = fs.solve()
    return results['product']['F_B']

grad_fn = jax.grad(flowsheet_objective)
sensitivity = grad_fn(350.0)
```

---

## Plugin System

**Location**: `difflow/plugins.py`

### Plugin Architecture

Difflow uses a plugin system for extensibility:

```python
from difflow.plugins import OperationRegistry, UnitOperation

# Register a new operation
@OperationRegistry.register(
    name='my_reactor',
    category='reactors',
    description='Custom reactor model'
)
class MyReactor:
    def __call__(self, inlet, **params):
        # Custom reactor logic
        return outlet, info
```

### Protocol Definitions

```python
from typing import Protocol

class UnitOperation(Protocol):
    """Protocol for single-inlet unit operations."""
    def __call__(self, inlet: dict, **kwargs) -> tuple[dict, dict]:
        """
        Process inlet stream.

        Args:
            inlet: Input stream dictionary
            **kwargs: Operating parameters

        Returns:
            (outlet_stream, info_dict)
        """
        ...

class MultiInletOperation(Protocol):
    """Protocol for multi-inlet unit operations."""
    def __call__(self, inlets: list[dict], **kwargs) -> tuple[dict, dict]:
        ...
```

### Using the Registry

```python
from difflow.plugins import OperationRegistry

# List available operations
operations = OperationRegistry.list_operations()

# Get operation by name
CSTR = OperationRegistry.get('cstr')

# List operations by category
reactors = OperationRegistry.list_by_category('reactors')
```

### Loading Plugins

Plugins are discovered via Python entry points:

```toml
# pyproject.toml
[project.entry-points."difflow.plugins"]
bio = "difflow_bio:register"
```

```python
from difflow.plugins import load_plugins, discover_plugins

# Discover installed plugins
plugins = discover_plugins()

# Load all plugins
load_plugins()
```

### Bio Plugin

The `difflow_bio` package is automatically registered as a plugin:

```python
# After loading plugins, bio operations are available:
from difflow.plugins import OperationRegistry

bioreactor = OperationRegistry.get('continuous_bioreactor')
centrifuge = OperationRegistry.get('disc_stack_centrifuge')
protein_a = OperationRegistry.get('protein_a_chromatography')
```

---

## Saving and Loading Flowsheets

A flowsheet is otherwise only expressible as Python: the code that builds it *is* the model. `difflow.serialize` gives it a file format, so a flowsheet can be saved, diffed, sent to a service, or read by something that never imported the module that built it.

```python
from difflow import serialize

serialize.save(fs, "plant.json")
fs2 = serialize.load("plant.json")
```

The round trip preserves the answer, not just the shape — a reloaded flowsheet solves to bit-identical results. `to_json`/`from_json` and `to_dict`/`from_dict` are available if you want the text or the data rather than a file.

The format records `format_version` (checked on read against `SUPPORTED_VERSIONS`) and the difflow version that wrote it (for provenance only). difflow writes version 2 and reads 1 and 2, so a file written before the editor existed keeps opening.

Version 2 added an optional top-level `view` block, which round-trips through `Flowsheet.view` and is read by nothing numeric:

```python
fs.view = {"nodes": {"reactor": {"x": 260.0, "y": 40.0},
                     "feed:feed": {"x": 40.0, "y": 40.0}},
           "code_context": "thermo = IdealThermo(...)"}
```

It is where the editor keeps canvas positions, its code-context snippet and its planning selection, so a flowsheet laid out by hand opens the way it was left. Node keys follow the same vocabulary as `_apply_params`: a unit is its bare name, a feed is `feed:<stream>`. Nothing validates the contents, and `_apply_params` copies the block through, so a swept or optimized flowsheet keeps its layout.

### What it can and cannot write

Round-tripping goes through the [operation registry](#operation-catalog): a unit is written as the name it is registered under and rebuilt by looking that name up. An **unregistered** operation is refused, because nothing would know how to rebuild it.

Parameters are written when they are *data* — numbers, strings, arrays, lists, dicts, nested `Params` dataclasses, and NamedTuples such as `SpeciesData`. A parameter holding a **callable** is refused rather than dropped:

```
SerializationError: unit 'reactor' field 'rate_fn' holds a callable ('rate_fn'),
which cannot be written to a file. Build it from data instead ...
```

That is deliberate. A file that silently lost a reactor's rate law would reload into a different model that still looked plausible.

### Thermodynamics

About half the core units need a `thermo` or `eos` object in the constructor, not just a `Params`. `IdealThermo` is written and rebuilt automatically. Anything else is refused on write, and can be supplied on load instead:

```python
fs2 = serialize.load("plant.json", extras={"flash": {"thermo": my_thermo}})
```

`extras` also *overrides* a stored thermo, which is the way to reload a saved flowsheet against a different property package.

### Units that build their own `Params`

Most units are constructed as `Unit(Params(...), thermo)`, but a substantial minority take their numbers as plain constructor arguments and build the `Params` themselves — `Compressor(ratio)`, `FlowSplit(w)`, `GasPipe(beta)`, and most of the gas plugin. Those numbers are written under `params`, because that is where the built unit keeps them, and they are passed straight back to the constructor on load. Nothing extra is needed:

```python
fs = Flowsheet(species_order=["CH4"])
fs.add_unit(Unit("boost", Compressor(ratio=1.3), ["a"], ["b"]))
serialize.from_json(serialize.to_json(fs))   # ratio comes back as 1.3
```

An argument that is *not* data — `Mixer(species_order)`, a `thermo` — still travels under `constructor`, and the two channels compose: `CompressorBoost(ratio, direction)` carries `ratio` by the first road and `direction` by the second.

Only *required* constructor arguments are carried. An optional one left at its default is not written, so a unit that was built with a non-default optional argument comes back with the default; pass it on load with `extras=` when it matters.

---

## Generating Python

`serialize` gives a flowsheet a file format; `codegen` gives it a *source* form — the same model written as the code someone would have typed.

```python
from difflow import codegen

print(codegen.to_python(fs))
codegen.save_script(fs, "plant.py")
```

The two together close the loop. A graphical editor you can only enter is worse than none: the moment you want something the palette does not offer, you have to be able to drop into Python and keep going. Build in a GUI, export a script, edit it, and read the result back through `serialize`.

The output is laid out to be read and edited — imports, then thermodynamics, then kinetics, then the flowsheet:

```python
"""Flowsheet generated by difflow 0.1.0.

Edit freely --- this is ordinary difflow code.
"""

from difflow import CSTR, CSTRParams, Flowsheet, Unit, make_stream, mass_action_kinetics

kinetics_reactor = mass_action_kinetics(reactions=[...], species_order=['A', 'B'], reverse='error')

fs = Flowsheet(species_order=['A', 'B'], ...)
fs.add_feed('feed', make_stream({'A': 1.0}, T=350.0, P=101325.0))
fs.add_unit(Unit('reactor', CSTR(CSTRParams(**kinetics_reactor.params_kwargs(), V=1.0, ...)), ['feed'], ['out']))
```

A data-built rate law is **hoisted** into its own statement and splatted back rather than inlined, which keeps the reactor to one readable line and avoids repeating the arrays the factory derives anyway. Thermo is hoisted the same way, and written as a `get_species_data` expression when every species is in the database.

Running the generated script rebuilds a flowsheet that solves to bit-identical results.

`codegen` refuses exactly what `serialize` refuses, for the same reason: an unregistered operation, and a callable that does not record how it was built. A generated script that quietly dropped a rate law would still run, and would be a different model.

---

## Operation Catalog

The registry answers *what units exist*. `difflow.catalog` answers *what you can do with one*: how many streams go in and out, what parameters it takes, which are required, and which hold code rather than data.

```python
from difflow import catalog, describe_operation

spec = describe_operation("Flash")
spec.ports.inlets          # ['inlet']
spec.ports.n_outlets       # 2
spec.required_parameters() # ['T']
spec.equations             # LaTeX governing equations
spec.to_dict()             # JSON-serializable, for a UI or code generator
```

`catalog()` returns a schema for every registered operation, optionally filtered:

```python
reactors = catalog(category="reactors")
```

All of it is **derived by introspection**, not from a second hand-maintained table: parameters come from `dataclasses.fields` of the unit's `Params` class, ports from the `__call__` signature, and equations from the `equations` class attribute the units already carry. The catalog therefore cannot drift from the code, and an operation whose signature is unannotated is reported as *unknown* rather than guessed at — `Splitter` returns a bare `tuple`, so its `n_outlets` is `None`.

### Where parameter descriptions come from

Each parameter also carries the prose that documents it, so a schema can label a field rather than only name it:

```python
for p in describe_operation("Heater").parameters:
    print(p.name, "--", p.units, "--", p.description)
```

```
duty -- W -- Heat duty (W). Positive = heating.
T_out -- K -- Outlet temperature (K). Alternative to duty.
UA -- W/K -- Overall heat transfer coefficient × area (W/K). For rating.
T_utility -- K -- Utility temperature (K). For LMTD calculation.
Cp -- J/mol/K -- Heat capacity (J/mol·K). If None, uses thermo.
```

That text is not a second copy. `difflow.docstrings` reads it out of the `Params` class's own `Attributes:` section, which is where the project already writes it, and out of the comments beside the fields for the 35 that are documented there instead — `CSTRParams.eos` and `CSTRParams.outlet_volumetric_basis` among them. Populating `field(metadata={"description": ...})` on all 87 `Params` classes would have duplicated every description and then drifted from it.

The editor gets this for free: the inspector's parameter list already rendered `spec.description` as help text under each field, and the palette already rendered `spec.units` beside its name — both written against a schema that carried neither. 470 of the 487 catalogued parameters gain help text in the inspector with no front-end change.

Units are *not* read from the prose. They come from the class's `parameter_units` table (above), which names every numeric field and is guarded by a test; a parenthetical in a sentence is a weaker signal, since `(0-1)` and `(default 10)` sit in the same position as `(Pa)`, and a formula in a description will happily offer `-1/k` as "per kelvin".

Where a field needs to say something different from its docstring, its metadata still wins:

```python
P: float = field(default=101325.0, metadata={"description": "..."})
```

### Which operations are declarative

`is_declarative` marks the operations whose parameters are all data, so a form or a JSON file could supply them:

```python
[name for name, spec in catalog().items() if not spec.is_declarative]
```

The handful that are not are the reactors, and always because of the rate law — see [Declarative Kinetics](unit-operations-chemical.md) for building that from data instead.

### Core units and the registry

The core reactors, separators, columns and exchangers are registered when `difflow` is imported, so they appear in `catalog()` alongside the plugin units with no `load_plugins()` call needed.

One name is deliberately not the class name: `difflow_gas` registers a `Compressor`, and plugins load *after* the core, so the EOS-consistent `difflow.Compressor` is catalogued as **`EOSCompressor`**. Registering both under the bare name would silently drop one.

---

## The Local Editor

`difflow.gui` serves a single-page flowsheet editor on `localhost`, with the installed package doing the solving:

```bash
difflow                                   # an empty canvas, in a browser
difflow gui plant.json                    # ...on a flowsheet
difflow gui --port 9000 --no-browser
```

The editor is what a bare `difflow` does, because it is the one thing here that has nothing to print and everything to show. `python -m difflow.gui` is the same command for an environment where the console script is not on `PATH`.

or from Python, on a flowsheet you already have:

```python
from difflow import gui

gui.serve(fs, path="plant.json")
```

Two pages are served, and they are at different stages.

`/` is the **canvas**: the flowsheet as a node graph, laid out automatically, pan and zoom, feeds banked left and products right, recycle edges dashed and orange. Drag an operation off the palette to drop a unit; drag from a unit's outlet port to another unit's inlet to wire it; drag a box to move it; select and press Delete to remove. Every gesture is one request and then a redraw from the answer, so the picture cannot drift from the model.

Two things about wiring are worth knowing before you use it, because both are properties of difflow rather than of the editor. A **stream name is the wiring**: connecting an outlet to an inlet renames the inlet, it does not add an arc, so the downstream unit's port is called whatever the upstream unit's outlet is called. And a **loop is a tear, never an arc**: if the wire you draw would close a cycle, the editor records a recycle instead — the same thing `add_recycle` does — and the edge draws dashed with both stream names on it, because the two ends carry different names.

Feed and product nodes are drawn *from* the topology; they are stream names with nothing on one end rather than objects the flowsheet holds, so they are not draggable endpoints. A unit dropped from the palette arrives unwired with a dangling stream on each port, which is why it appears with feed-ish and product-ish stubs until you connect it.

**Code context** opens the Python the flowsheet carries (see below). Parameter editing, the docstring inspector and the results panel are being built onto the canvas; until they land, `/classic` is where you change a number.

`/classic` is the **form editor**, and it is the one that can currently change a model: a palette of every registered operation with its port arity, an SVG of the topology, editable parameter fields per unit, and a panel that switches between solved stream values and the generated Python. **Solve** re-solves the edited model; **Save** writes the JSON; **Python** shows what `codegen.to_python` would produce. It stays until the canvas covers what it does; each page links to the other in its header.

Everything on both pages is derived from `catalog()`, so plugin units appear with no extra work, and a field the catalog reports as holding code is listed as *set in code* rather than given a text box that could only reject what you type.

**The palette says what each unit is waiting for.** Over half the catalog cannot be dropped as it stands — a `Flash` wants a `thermo`, a `CSTR` wants a rate law, an `AmineAbsorber` wants the name of a solvent — and those entries are dimmed with the missing *field names* on the row: `needs thermo`, `needs rate_fn, stoich, rate_params`. The names are the point. "Unbuildable" tells you nothing you can act on, whereas `rate_fn` is the thing you go and write, and it is the same list the refusal quotes if you drag the unit anyway. They stay visible because a palette that cannot show a CSTR is not a palette; a filter hides them for anyone who would rather it did.

What each row needs is answered **against the code context as it stands**, not against the class, so a row un-dims the moment its binding appears — one `thermo = IdealThermo(...)` clears the flag on every unit that was waiting for one, and the page refetches the catalog when you apply the snippet. A bare value counts too: `solvent = "MEA"` is matched by field name, which is why the message can honestly tell you to go and write one.

That the flag and the refusal are the same computation (`gui.edit.unmet`) is the part worth insisting on. When they were two, the catalog answered a question about the *class* — could a form construct one — and the adder answered a question about this session. `AbsorberParams.solvent` is a required `str`: no callable, no constructor object, so the class read as buildable, and nothing on earth can invent a solvent name. The palette offered it, the drop failed, and the error came from the file-loading path, which offered *"the file may have been written by a different version of difflow"* as the diagnosis of a unit dropped one second earlier. A test now walks every operation and asserts the two agree.

A flag that is empty promises a clean *answer*, not a success. A class may still refuse on its own terms — `Transformer` rejects a unity tap with no phase shift, because that is a line and not a transformer — and nothing short of constructing one can know that in advance, so the class's own sentence is quoted rather than replaced.

The point is not to replace writing Python. It is to make the tedious parts quick — seeing the topology, changing one parameter and re-solving, checking what a unit expects — while leaving the door open in both directions: export a script, edit it, and read the result back through `serialize`. An editor you can only enter is worse than none.

It is stdlib only (`http.server`), binds to `127.0.0.1`, and is meant for a single local user. It is a development tool, not a hardened service — do not expose it to a network.

Because the code context runs Python, the server answers mutating requests only from the page it served. Three checks, all in `server.py`: a token minted per process, put into the page as a `<meta>` tag and required in an `X-Difflow-Token` header; an `Origin` that must be this exact host and port; and a `Host` that must be a loopback name, which is what a DNS-rebinding request cannot produce. A request that fails any of them gets a **403** — the one case where a refusal is not a 200, because it is a failure of the request rather than an answer about the flowsheet. Reads are not guarded: the page fetches the catalog before it has done anything. A client outside the browser — `curl`, or `npm run dev` proxying to this server — has to send the token too.

### The code context

Roughly half the catalog cannot be built from data alone. A `Flash` needs a `thermo`, a `PengRobinson` unit needs an `eos`, a reactor needs a rate law — objects, not numbers, and no form can supply one. So the flowsheet carries a snippet of Python in `view["code_context"]`, the session `exec`s it in a fresh namespace, and anything it defines can be referred to by name:

```python
from difflow import IdealThermo, get_species_data

thermo = IdealThermo({n: get_species_data(n) for n in ['water', 'ethanol']})
```

With that in the panel, `Flash` becomes droppable; with a `mass_action_kinetics(...)` call in it, so does `CSTR`. The reference is stored as `{"$ref": "thermo"}` in the unit's `constructor` block rather than inlined, and the same tag works anywhere a value goes:

```python
serialize.save(fs, "plant.json", refs={"thermo": thermo})
serialize.load("plant.json", refs={"thermo": thermo})
```

The scan is by **identity**, and it runs after the primitive branch: two `IdealThermo`s built the same way are two objects, and small ints and short strings are interned, so `x = 1` in the context must not turn every `1.0` in the file into a reference. A `$ref` the namespace cannot resolve is a `SerializationError` naming the missing name, not a silent `None`.

`codegen.to_python` emits the snippet as the script's preamble, ahead of everything that uses it, so what is exported is what ran. A snippet that fails to compile or raises is refused with the line number and *not stored* — the flowsheet keeps the last one that worked, so a half-typed line cannot unbuild the units already depending on it.

That the editor runs the Python a file carries is worth stating plainly: **opening a flowsheet in the GUI executes its code context**, exactly as running a script would. Treat a flowsheet JSON from someone else the way you would treat their `.py` file.

### For tests and embedding

`make_server` builds the server without starting it, which is what a test wants:

```python
from difflow.gui import FlowsheetSession, make_server

server = make_server(FlowsheetSession(fs), port=0)   # 0 => any free port
```

The routes are:

| Route | What it does |
|---|---|
| `GET /api/catalog` | every registered operation, its ports and its parameters |
| `GET /api/flowsheet` | the document, with `view.nodes` filled in |
| `GET /api/code` | `codegen.to_python` of the current model |
| `POST /api/flowsheet` | replace the whole model (load a file) |
| `POST /api/solve`, `/api/save` | solve; write the JSON |
| `PATCH /api/unit/<name>` | one unit's `params`, `name` or `position` |
| `POST` / `DELETE /api/unit[/<name>]` | add one from the palette; remove one |
| `POST` / `DELETE /api/connect` | wire or unwire, body `{source, outlet, target, inlet}` |
| `GET` / `POST /api/species` | the species list, and whether it can still be changed |
| `GET /api/feeds` | the declared feeds, and the inlets still waiting for one |
| `POST` / `DELETE /api/feed[/<name>]` | declare or change a feed, body `{name, T, P, flows}`; take one back off a stream |
| `POST /api/layout` | positions only, no rebuild and no solve |
| `GET` / `POST /api/code-context` | read the snippet and what it defines; set it, or say why it will not run |
| `GET /api/docs/<op>` | one operation's rendered docstring, equations, assumptions and references |
| `GET /api/levers` | what a derivative can be taken with respect to, and of |
| `GET /api/diagram` | the flowsheet drawn as one SVG, at the canvas's own layout |
| `POST /api/sensitivity` | one derivative sweep, forward or reverse |

A failed solve or a rejected edit comes back as `{"ok": false, "error": ...}` with a **200** rather than a traceback at the socket: it is an answer about the flowsheet, not a failure of the request, and a bad edit from the browser cannot take the server down. Only malformed JSON and an unrouted path get a 4xx.

The incremental routes exist because `POST /api/flowsheet` re-runs `serialize.from_dict` and re-instantiates every unit — wrong twice over for a canvas, since it costs a full reconstruction per keystroke and it drops any constructor object the file cannot carry. `PATCH` rebuilds only the unit you touched and hands its `thermo` back **by identity**, so a hand-built one survives an edit that JSON could not have round-tripped. A unit dropped from the palette takes `species_order` from the flowsheet and its constructor objects from the code context below; one whose `thermo` is nowhere to be found is refused with a message naming what is missing and where to define it. Removing a unit also drops any recycle naming its streams, which would otherwise tear a stream nothing produces and fail the solve somewhere far from the edit.

`difflow.gui` is a package rather than one module: `session.py` holds the flowsheet and everything that can be done to it, `server.py` holds the wire encoding and the routes, and `static/` holds the page as files on disk. `FlowsheetSession` needs no socket, so the interesting half — load, edit, solve, emit code — is usable and testable on its own:

```python
from difflow.gui import FlowsheetSession

session = FlowsheetSession(path="plant.json")
session.solve()["converged"]
session.code()["source"]
```

`GET /api/flowsheet` fills in `view.nodes` when the document has none, from `difflow.gui.layout.auto_layout` — a longest-path column assignment with feeds banked left and dangling products right, shared with `difflow.report.diagram` so a report and the canvas agree on where a unit sits. Recycle edges are left out of the path length, which is what makes a recycle draw as an arrow going back rather than as another column. The positions are served, not adopted: opening a file does not give it a `view`, and coordinates reach disk only when the user saves. Stored positions sit on top of the automatic ones rather than replacing them, so moving one box does not send every other node back to the browser unplaced.

Anything else under `static/` is served alongside the page, by an allow-list of the suffixes a front-end build emits (`.html`, `.js`, `.css`, `.json`, `.svg`, `.map`, `.woff2`, `.ico`); the page is re-read from disk on every request, so a rebuilt bundle appears on reload rather than on restart.

### Building the canvas

The canvas is Svelte plus [`@xyflow/svelte`](https://svelteflow.dev), and its source lives in `src/difflow/gui/frontend/`. The **build output is committed** to `src/difflow/gui/static/`, which is the whole point of the arrangement: `pip install difflow` never needs node, only changing the UI does.

```bash
make gui-build     # npm ci && npm run build, into static/
make gui-test      # the pure model functions, under bare node
make gui FLOWSHEET=plant.json
```

`npm run build` runs vite twice: once for the editor (`app.js`, `app.css`, `index.html`) and once for the frozen page `difflow.publish` inlines (`publish.js`, `publish.css`). Two builds rather than two entries, for the reason given under [Publishing a Model](#publishing-a-model).

A unit is drawn as its **PFD symbol** rather than as a labelled box: a distillation column as a tray stack, a heat exchanger as the circle with its zigzag, a pump as the circle with its volute, a CSTR as a vessel with an impeller. The symbols are hand-drawn SVG paths in `src/difflow/gui/frontend/src/lib/nodes/symbols.js`, mapped to operations **by name first and by catalog category second**, which is what lets a plugin's own unit — one this front end has never heard of — still draw as equipment: an unknown `reactor` gets the reactor symbol from its category. `tests/test_gui_symbols.py` joins the map to `difflow.catalog()` from the Python side and asserts that every one of the catalog's operations resolves to real equipment and none falls through to a plain block, so adding a unit operation and forgetting its symbol fails a test rather than shipping a rectangle. The same symbol is drawn on the palette row, so what is dragged looks like what lands.

Wires route orthogonally, ports are small grey squares, and two keys toggle the rest: `L` names every port beside its handle, `T` swaps to a dark palette. The dark palette is opt-in by a `data-theme` attribute on the root rather than by `prefers-color-scheme`, so a *report* opened in a dark browser keeps the light palette it was designed and screenshotted with.

### Starting from an empty canvas

`difflow gui` with no file opens on an empty flowsheet, and *empty* rather than absent: the session used to leave `self.flowsheet = None`, every edit route begins by refusing "no flowsheet loaded", and nothing said so — so the palette filled in, the canvas drew its grid, and dragging a unit onto it did nothing at all. Which reads as a broken drag rather than as an editor with no flowsheet to edit.

So the flowsheet exists from the first frame, and the one thing it is missing is asked for by name. Every stream in difflow is an array indexed by `species_order`, so until that list exists nothing can be built — and the palette says so on every row (`needs species_order`) rather than offering a drop that will be refused. The list can be typed into the header field or bound by the code context below (as `species_order`, or as `SPECIES`, which is what the starter snippet defines); whoever typed it in the header wins, and the code context does not overwrite it. It freezes once a unit indexes it: re-ordering under a built unit would turn a water flow into an ethanol flow with nothing on screen changing.

The other half of building from nothing is the **feed**, and it is the half that has no gesture. Dropping and wiring can describe a whole topology, but a feed is data — a temperature, a pressure and a flow per species — so a from-scratch flowsheet could be drawn and never solved, and `solve` answered `KeyError: 'mixer_in'`: the name of the stream, which was on the canvas already, and no hint that a feed was the thing missing. An inlet with nothing on the other end is now selectable, and the inspector gives it a form. `set_feed` fills any field left out from the feed that is already there, so editing a temperature does not zero the flows, and from the flowsheet's own `default_flow`/`default_T`/`default_P` when there is no feed yet — the same numbers `Flowsheet.solve` invents for a tear stream, so an untouched feed is not a new guess about the model. It refuses a stream a unit already produces (two sources for one stream, and the solver would silently use one of them), a stream nothing reads, an unknown species, a negative flow and a non-positive temperature or pressure, each by name. A blank box is a question and not a zero: zero is a real flow, and guessing which was meant would put a number nobody typed into the model. What remains unfed at solve time is reported as *that*, with what to do about it, and `GET /api/feeds` answers the same question ahead of time.

### The inspector

Selecting a unit shows what the code already knows about it. None of it is written twice: the unit classes have carried `symbol`, `equations`, `assumptions`, `references`, `parameter_symbols`, `parameter_units` and `numerical_method` since the report writer needed them, `difflow.report.metadata.get_metadata` reads them with docstring fallbacks, and `difflow.catalog` now goes through that same function. So `ParameterSpec.units` — declared from the start and always `None`, because no `Params` field uses `field(metadata={"units": ...})` — fills in from the class's own table for 147 of the 486 catalog parameters, and a field that *does* declare its own metadata still wins, being the more specific of the two.

`GET /api/docs/<op>` renders the class docstring with [docutils](https://docutils.sourceforge.io) when it is installed and returns the escaped source in a `<pre>` when it is not — an optional extra, never a hard dependency, and the panel says which of the two it is showing. Two accommodations make difflow's own docstrings render: nothing rewrites the Google-style sections, because `Args:` followed by an indented block already *is* a reStructuredText definition list; and the Sphinx roles the codebase uses (`` :class:`~difflow.streams.Stream` ``) are registered as literal text, since bare docutils treats an unknown role as an error that swallows the line. Messages are suppressed rather than rendered: a red box in the inspector would be about difflow's prose, not about the user's flowsheet.

Equations are rendered with a bundled [KaTeX](https://katex.org), to **MathML** rather than to KaTeX's own HTML. The HTML output is laid out against KaTeX's fonts and looks wrong without them, so taking it would mean committing twenty `.woff2` files and a stylesheet whose only job is positioning glyphs; MathML asks the browser to do that instead. All 214 equations in the catalog render.

Parameters are editable in place, through `PATCH /api/unit/<name>`. Not all of them: `serialize` writes a JAX array as `{"$array": ...}`, a rate law from `mass_action_kinetics` as `{"$callable": ...}`, and a code-context object as `{"$ref": ...}`, and none of those is something a text input can edit. Each is shown with *where its value came from* — `mass_action_kinetics(...)`, `array 2×1`, `thermo (code context)` — in place of an input, which is the same fact the old editor put in a "set in code:" line under the form, moved to the field it belongs to. The constructor objects get their own section for the same reason: a Flash's thermodynamics are not a `Params` field, and a panel that showed only parameters would not say where they came from.

### Results, and the derivatives that come with them

Solving fills a drawer under the canvas with three things, and the first two are what any flowsheet editor shows: a stream table, and the solve's own diagnostics. The table's columns are the union of the species across all streams, so a stream that never sees a component reads `0` rather than going missing, and a mole-fraction toggle divides by the total — leaving the cells of a zero-flow stream blank rather than printing `NaN`. The diagnostics are `last_solve_converged`, `last_solve_method`, `last_solve_iterations`, `last_solve_residual`, `last_solve_tol` and `last_solve_tear_streams`: `Flowsheet` has recorded how it solved, how far the tear residual came down, against what tolerance and on which streams for as long as the recycle solver has, and none of that was shown anywhere. It matters because a recycle that stopped at `max_iter` with a residual of 1e-3 returns numbers that look exactly like an answer. A solve that did not converge says so in red and still shows its numbers, because the residual and the tear list are exactly what one wants to see when it did not.

The third tab is the one no other flowsheet editor can offer, and it is the reason difflow is built on JAX. Ask for a derivative and the panel takes it — not by re-solving the flowsheet once per lever, but with a single AD pass:

- **Pin a lever** — "if I change the reactor volume, what moves?" — and it is one `jax.jvp`. One forward pass gives `dy/du` for *every* quantity in *every* stream at once, so the cost does not grow with how much you want to look at.
- **Pin an output** — "what moves the purge's benzene?" — and it is one `jax.value_and_grad`. One reverse pass gives `dy/du` for *every* lever at once, so the cost does not grow with how many knobs the flowsheet has.

That is `difflow.planning`'s `choose_ad_mode` rule surfaced in the UI: the direction of the pass follows from which end you pinned, and either question by finite differences would cost one solve per lever. The derivative goes through the recycle tear solve implicitly, so a flowsheet with a recycle is no more expensive to differentiate than one without.

Levers are found rather than declared. `difflow.gui.sensitivity.levers` walks the flowsheet and keeps every parameter whose **current value is a real scalar** — a Python `int` or `float`, or a 0-d array of float dtype. A `rate_fn` is a function, a `stoich` is an array, a `species_order` is a list and a `thermo` is an object, so all four fall out without a maintained exclusion list: they are ruled out by the same test the solver itself would apply. It falls out as very nearly the set the inspector greys out, arrived at independently and from the other direction. Feed streams are levers too, as `total_flow`, `T` and `P`; `x_<species>` is deliberately left out, because `_apply_params` rescales the other species to hold the total, which makes that derivative a composition swap rather than one knob. Every key the picker offers is a key `Flowsheet._apply_params` accepts, and therefore a key `planning.Block.from_flowsheet`'s `u` list takes — the GUI picker and the Python API name the same things.

Rankings are **relative**: `d ln y / d ln u`, which is the only comparison that means anything between a volume in m³ and a temperature in K. It is `None`, not infinity, when either base value is zero. The reverse view draws them as a tornado — plain SVG geometry computed in `model/results.js`, no plotting library, so the committed bundle stays at half a megabyte and `publish.py`'s self-contained page has nothing new to swallow.

Two smaller things make the panel honest. Edges carry their flow as a label once solved, and tint by relative sensitivity after a derivative — thickness by magnitude, colour by sign — through a CSS custom property, so `Canvas.svelte`'s stylesheet decides what "up" and "down" look like and the model layer never computes a colour. And any edit that is not a move drops the stored solve: results are about the flowsheet you solved, and a panel still showing the previous one is worse than an empty panel.

### Getting the model out again

An editor you can only enter is worse than none, so **Export** offers four files and none of them is a re-implementation of something difflow already writes:

| File | Where it comes from |
|---|---|
| `<name>.py` | `codegen.to_python`, code context and all — the script that reproduces this solve |
| `<name>.json` | the served document, `view.nodes` included, and it loads straight back |
| `<name>.svg` | `GET /api/diagram` |
| `<name>.png` | that same SVG rastered at 2x in the page, on a white ground |

The diagram is the part worth explaining. `difflow.report.diagram` already owned the only flowsheet-to-SVG drawer in the codebase, and the editor already shared its column algorithm through `difflow.gui.layout.unit_columns` — so rather than grow a second drawer, that one was parameterised: `topology_svg(units, recycles, positions=None)` is the drawing, `flowsheet_svg(report)` is the adapter a report uses, and `flowsheet_diagram(flowsheet, positions)` is the adapter the editor uses. A picture exported from the canvas and a picture in an HTML report are therefore the same picture, and `positions` is what makes the exported one a diagram of *your* flowsheet rather than of some flowsheet with the same topology: pass the canvas's own coordinates and the boxes land where they were dragged. The two callers key their nodes differently — the canvas uses a unit's bare name, because that is `_apply_params`'s vocabulary, while the report prefixes `unit:` — and `_diagram_keys` is the single place that translates. A position naming a node that no longer exists is ignored rather than raised on, because a stale layout is a normal thing to be holding.

The PNG goes through an `Image` and a `<canvas>`, with the SVG handed over as a **data** URL rather than a blob URL: a canvas that has drawn a blob-URL image counts as tainted and `toBlob` on it throws. Its size comes from the SVG's own `viewBox`, since the drawer writes `width="100%"` — right for a document that flows, useless for a raster. Naming, the XML declaration and the raster size are pure functions in `model/download.js` and are tested under node; the `Blob` and the anchor click stay in the component, where nothing can be quietly wrong.

The one thing that can go wrong with committed build output is drift — source edited, bundle not rebuilt — so CI reruns `gui-build` and fails if `static/` differs from the commit.

The functions that turn a serialized flowsheet into nodes and edges (`frontend/src/lib/model/graph.js`) and the ones that turn a canvas gesture into a request (`model/edit.js`) are plain JavaScript with no framework in them, and they are tested under bare `node --test`. That is where the testing effort goes on purpose: a wire attached to the wrong port, or a delete sent to the wrong endpoint, looks exactly like one that worked until the next reload. `tests/test_gui.py` runs those same files when node is on `PATH` and skips when it is not: node is a tool for building difflow, never a dependency of it.

One wrinkle worth knowing: JSON has no literal for the non-finite floats, and `JSON.parse` rejects the `Infinity` that Python's `json` writes. This is the *common* case, not an exotic one — `mass_action_kinetics` puts `inf` in `K_eq` for every irreversible reaction — so those values travel as the strings `"Infinity"`, `"-Infinity"` and `"NaN"`, and are restored on the way back.

### Delta vectors, for someone else's planning model

**Planning** is the panel that turns the open flowsheet into the object an LP planning system consumes: a Jacobian of chosen outputs against chosen levers, around the solved base case, with the bounds and the trust radius that say where it is valid. `difflow.planning` has computed that since it existed. What it never had was a way to say *which* levers, by pointing at them.

Check parameters to make them levers and stream quantities to make them outputs, press **Linearize**, and `POST /api/linearize` builds a `Block.from_flowsheet` over the live flowsheet, calls `linearize_block`, and returns a `DeltaVectorSet`. Lever keys are `_apply_params`'s own vocabulary — `reactor.V`, `feed:feed.total_flow` — which is why the same picker serves the sensitivity sweep and this: they are asking for derivatives of the same thing, one at a time versus all at once.

Four things about it are deliberate.

**The selection is part of the document.** It persists as `view.planning = {"u", "y", "bounds", "radius"}` alongside `view.nodes`, so a flowsheet reopens on the levers it was last linearized against. Choosing them is the work; recomputing the Jacobian is a second.

**A blank bound is not an infinity.** A lever with no bounds still needs a range for the trust region to mean anything, so an unbounded lever gets a symmetric window around its own base value. An infinite bound would make every scaled column zero and the health report would then flag the whole model as dead — a diagnostic failure caused by the diagnostic's own input.

**The finite-difference check is a button, not a default.** `check_delta_vectors` compares the AD Jacobian against central differences, which costs `2 n_u` extra solves against the one the Jacobian took. It is the check that decides whether a Jacobian should leave the building — a `clip`, a `minimum` or a `where` sitting on an active constraint gives an AD derivative that is perfectly correct and completely unlike the flowsheet's actual response — so the panel offers it prominently and states the number either way, but does not make everyone pay for it on every press.

**Health findings travel with the coefficients.** `check_delta_health` runs on every linearization and its findings ship inside the export, because a dead lever or a recycle loop gain near one is a fact about the numbers, and the person pricing against them downstream never sees this panel. The panel also renders an exactly-zero cell as `0` rather than `6.781e-21`: a structurally dead column is the one thing in that table a reader must not miss, and scientific notation hides it among the merely small.

Downloads are the JSON manifest and the CSV tables, and they are fetched from the server rather than rebuilt in the page — `difflow.planning.export`'s own writers produce them, so what a planning system reads is byte-for-byte what `difflow plan-export` writes. The name is `<flowsheet>_delta_vectors.json`, never `<flowsheet>.json`: a download landing next to the document under that name is a flowsheet overwritten by a Jacobian.

`.lp` and `.mps` are **not** offered here, and their absence is not an omission. Those are renderings of an assembled LP, and this panel poses none — no prices, no constraints, therefore no shadow prices. Run `DeltaBasePlanner` or `difflow plan-export` for those; the JSON this writes is what such a run consumes.

### What the assistant is told

The editor can put a small language model next to the canvas, and the model is the least interesting half of that. A 3B model knows nothing about difflow; the useful thing it can do is *read*. So the work is in the brief — and the brief is assembled from things difflow already computed and never showed anyone.

`GET /api/context` (or `session.context(...)` in Python, with no server at all) returns one of four:

| `kind` | What it contains |
|---|---|
| `block` | the catalog schema for the operation, its equations and assumptions, its docstring, and **this node's own parameter values with their units** |
| `flowsheet` | the units and how they are wired, the feeds, the recycles, the species, and the Python `codegen` would emit |
| `solve` | `last_solve_converged`, `method`, `iterations`, `residual`, `tol`, `tear_streams` — every one of them already recorded on `Flowsheet` and none of them previously reported — plus a troubleshooting card matched to the symptom |
| `planning` | a `DeltaVectorSet` read out as a table, the prices and constraints, the binding marginals, and the health findings |

Three decisions in there are worth stating.

**The brief is the product, not the answer.** Every pack comes back whole — sections, assembled prompt, token count, and a note for anything the budget dropped — and the panel shows it. When the local model is weak that is still the useful output: read it, or paste it into a stronger assistant. It is a report about the flowsheet that difflow could not previously produce.

**Retrieval is lexical and build-time.** `docs/` is 23 files and 720 KB, fixed at release. `difflow.gui.docs_index` splits it at headings — outside fenced code, since every chapter is full of Python and `# Build the flowsheet` in a fence is a comment, not a section — and writes `static/docs-index.json`. That file is committed and CI rebuilds it and fails on a difference, exactly like the JS bundle: an index that has silently stopped matching the prose it indexes is worse than no index. It lives under `static/` because `docs/` is not in the wheel and `static/` is, and it is built by a stdlib-only script so the CI job needs nothing installed.

Scoring is TF-IDF with length normalisation, and it runs in Python rather than in the browser. Where a dot product happens is not architecture; running it here makes it testable without a headless browser and keeps a 667 KB file off the wire on every question. What actually sharpens it is that **each pack searches on the question plus its own subject** — the operation name, the units in the flowsheet, the solver that ran. A user's question is four words long and two of them are "why" and "this"; the pack is what knows what "this" is.

**And where the question goes is stated, not assumed.** Three things can answer a brief, chosen in the panel:

| Provider | Where the brief goes |
|---|---|
| *No model* (the default) | nowhere. The brief is assembled and shown; copy it into an assistant of your choosing |
| *In this browser* | a small model over WebGPU, via WebLLM. The weights come from the MLC CDN once (1–2 GB) and are cached by the browser; nothing you type leaves the machine |
| *Local server* | an OpenAI-compatible base URL you give — Ollama, llama.cpp, vLLM |
| *Anthropic* | `POST /api/assistant`, which the difflow process forwards with a key from **its own environment** (`ANTHROPIC_API_KEY`). The page never holds the key, and the panel says up front if the server has none |

The footer names the active one in a sentence, in the accent colour when the answer is that the brief leaves the machine. The default sends nothing anywhere, and the WebLLM runtime is bundled rather than fetched from a CDN at run time — this page carries the CSRF token for a server that can `exec` Python, so it must not import third-party code over the network. That costs about 6 MB of committed build output, loaded only if someone selects that provider.

**The budget is a real constraint.** The default runtime is a 3B model with a 4096-token window, and a brief that overflows it is truncated *at the end* — where the question is. So a pack is fitted to `context.BUDGET` tokens, dropping whole low-priority sections rather than truncating any of them (half a parameter table is a table with parameters missing from it), never dropping the subject, and recording what went. A thin answer then has a visible cause.

### The console, where the flowsheet is a Python object again

The assistant answers a question in English. **Console** is the same question asked in Python, and it is the panel that admits the editor can never have a button for everything. It is a REPL running *in the difflow process* — every cell is `exec`'d in a namespace that is rebound before each run:

| Name | What it is |
|---|---|
| `fs` | the flowsheet on the canvas. Not a copy |
| `streams` | the last solve's streams, or `None` if nothing has solved |
| `dvs` | the last linearization, or `None` |
| `session` | the session itself — `session.solve()` keeps the panels in step |
| `difflow`, `jax`, `jnp` | imported for you |

plus whatever the code context defines, so a `thermo` object built there is a name at the prompt.

**That `fs` is live is the whole point.** A REPL over a deepcopy would answer questions about a model nobody is looking at. Edit a parameter in a cell and the server refingerprints the document, notices, drops the stale solve, and tells the page to redraw — the canvas follows the prompt. It is also what makes the panel difflow's rather than any editor's: `jax.grad` at the prompt differentiates through the recycle tear solve of the flowsheet you are looking at.

```python
>>> import jax
>>> f = lambda V: fs._apply_params({'reactor.V': V}).solve()['liq']['F_ethanol']
>>> float(jax.grad(f)(1.0))
0.249960277574744
```

Four things about it are deliberate.

**A traceback is an answer.** `console_run` returns `ok: True` with `error` set, which is the one place difflow's `{"ok": false, "error": ...}` convention is on purpose not followed: the cell raising *is* the normal outcome someone typing at a prompt asked about, and a panel that reported it as a failed request would be reporting on the wrong thing. The traceback is trimmed to start at the user's own frame — the server's stack above it is noise — and the offending line is shown, because cells are registered with `linecache` under `<console:n>` so Python can quote source that was never a file.

**The namespace is not the code context's.** `view["code_context"]` is part of the model: it is serialized, re-evaluated on load, and it is what `$ref` tags resolve against. If a cell's assignments landed there, typing `k = 0.5` would silently change what the saved file means. The console *reads* those bindings and writes to a layer of its own, and `names` lists what you defined — never the names that were injected for you.

**Figures are a hook.** v1 emits text, but the wire format (`{"kind", "mime", "data"}`) and the browser's renderer both already understand an image, so plots are a callable and nothing else:

```python
from difflow.gui.console import Console, image

def figures(namespace):
    import io, matplotlib.pyplot as plt
    out = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=110)
        out.append(image(buf.getvalue()))
    plt.close("all")
    return out

Console(display_hooks=[figures])
```

A hook that raises is reported where its output would have been, rather than taking the cell's result down with it. And the page has an explicit branch for a `kind` it does not know, so a server emitting something newer degrades to a legible placeholder instead of an empty box.

**It is not a sandbox, and says so.** The server already `exec`s browser-supplied Python for the code context; a console adds no capability that endpoint did not have, and the CSRF token plus the `Origin`/`Host` check are the whole protection either way. What it does add is a way to hang the single-threaded server with a cell that loops forever — there is no safe way to interrupt a running thread in Python, so the panel states that rather than offering a stop button that would not work.

---

## Publishing a Model

`difflow.publish` turns a flowsheet into a **self-contained HTML page** that anyone can open with nothing installed — no Python, no server, no network. It is the form a model needs for a paper's supplementary material or a project page.

```python
from difflow import publish, SweepAxis

publish(
    fs,
    axes=[
        SweepAxis("reactor.V", 0.5, 5.0, n=21, label="Reactor volume", units="m³"),
        SweepAxis("heater.T_out", 320.0, 400.0, n=9, label="Inlet temperature", units="K"),
    ],
    outputs={"conversion": lambda streams: 1 - streams["out"]["F_A"] / 1.0},
    path="model.html",
    title="Reactor sizing",
)
```

Axis keys are whatever `_apply_params` accepts — `"<unit>.<param>"` and `"feed:<stream>.<field>"` — so a feed rate or a feed temperature is an axis like any other.

The page carries the flowsheet itself, sliders for each axis, the outputs, and their sensitivities. This works because the solve is *pre-computed*: `sweep` evaluates the flowsheet on the grid with `jax.vmap`, takes gradients with `jax.grad`, and bakes the results into the page, which interpolates between them.

### The page is the editor, frozen

A published page is the editor's own front end with nothing editable and nothing to solve: the same `@xyflow` canvas, the same graph model, the same palette. Click a unit and it says what the unit is — description, parameters with units, ports, equations, assumptions, references — read from `difflow.catalog`, which reads the class. A published model therefore cannot describe a unit differently from the editor, and cannot go stale against the code it was published from.

That replaces what a published page used to be, which was a set of sliders belonging to nothing visible. Pass `topology=False` to `publish` to get that back.

Three details follow from *self-contained*, and they are the reason this is a second Vite build rather than a second entry point of the first:

- A build with two entries shares chunks between them, and a chunk is a file the page would have to fetch. `vite.publish.config.js` uses `build.lib` with `formats: ['iife']`, so `publish.js` is one file `publish.py` can inline verbatim.
- KaTeX is deliberately left out. A quarter megabyte of typesetting is a poor trade for a file that has to open in ten years, so equations are shown as their LaTeX source.
- The topology on the page is **not** `serialize.to_dict()`. That is a document meant to be loaded back, and it refuses a flowsheet holding a `rate_fn` or a thermo object — which is most interesting flowsheets, and no reason to publish a page without a picture. `publish._topology` builds a description meant for *looking at*: same node keys as the editor, every value reduced to something printable, and anything that cannot be shown named by its type (`<function>`, `<array (2, 1)>`) rather than dropped, because a parameter that vanishes reads as one the unit does not have.

`test_is_self_contained` is now two tests, because the bundle makes the old one impossible to satisfy: Svelte and `@xyflow` ship documentation URLs inside their error messages, and every SVG carries the `http://www.w3.org/2000/svg` namespace, which is a name that looks like an address. So one test forbids the constructs that actually load something (`<script src`, `<link `, `url(http`, `@import url(`), and the other walks every URL in the file against an allowlist — which is the stricter of the two, since a new URL has to be looked at before it can be added.

That is a deliberate trade, and its limits should be stated plainly. JAX has no WebAssembly build, so a browser cannot run the real solver; the published page is an interpolation of a grid, not a live model. It is exact at the grid points and only as good as the grid between them, and it can only vary what the axes name. When you need the real thing, use the local editor above, or the generated script.

The arithmetic the page runs — interpolation between solved points, and the exact derivatives recorded alongside them — lives in `frontend/src/lib/model/sweep.js` and is tested under bare `node --test` with everything else in `model/`. It used to be a string of JavaScript inside a Python template, where nothing could reach it.

`sweep` is available on its own when you want the grid as data rather than as a page:

```python
from difflow import sweep

result = sweep(fs, axes, outputs)
result.values["conversion"]      # shape (21, 9)
result.gradients["conversion"]   # d(conversion)/d(axis), same shape per axis
```

---

## Best Practices

### Stream Naming Conventions

```python
# Clear, descriptive names
'fresh_feed'           # Feed streams
'reactor_effluent'     # Unit outputs
'flash_vapor'          # Phase-specific
'recycle_to_mixer'     # Recycle streams
'product_A'            # Product streams
```

### Flowsheet Organization

```python
# Good: Logical unit ordering
fs.add_unit(mixer)      # 1. Combine feeds
fs.add_unit(preheater)  # 2. Preheat
fs.add_unit(reactor)    # 3. React
fs.add_unit(cooler)     # 4. Cool
fs.add_unit(separator)  # 5. Separate
fs.add_unit(splitter)   # 6. Split recycle

# Define recycles last
fs.add_recycle('recycle', 'mixer')
```

### Debugging Convergence Issues

```python
# Check individual units
for unit in fs.units:
    print(f"\n{unit.name}:")
    inlet = fs.streams.get(unit.inlet_names[0])
    outlet, info = unit.operation(inlet, **unit.params)
    print(f"  Inlet T: {inlet['T']:.1f} K")
    print(f"  Outlet T: {outlet['T']:.1f} K")

# Monitor recycle convergence
def callback(iteration, error, tear_streams):
    print(f"Iter {iteration}: error = {error:.2e}")

results = fs.solve(callback=callback)
```

### Memory Efficiency

```python
# Use JIT compilation for repeated evaluations
@jax.jit
def evaluate_flowsheet(feed_conditions):
    fs.feeds['fresh_feed'] = make_stream(feed_conditions)
    return fs.solve()

# Vectorize over multiple cases
cases = [{'A': 1.0, 'B': 0.0}, {'A': 0.8, 'B': 0.2}, ...]
results = jax.vmap(evaluate_flowsheet)(cases)
```
