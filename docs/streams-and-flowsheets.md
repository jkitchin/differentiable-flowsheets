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

The format records `format_version` (checked on read) and the difflow version that wrote it (for provenance only).

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
python -m difflow.gui plant.json          # opens a browser
python -m difflow.gui --port 9000 --no-browser
```

or from Python, on a flowsheet you already have:

```python
from difflow import gui

gui.serve(fs, path="plant.json")
```

The page shows a palette of every registered operation with its port arity, a diagram of the topology, editable parameter fields per unit, and a panel that switches between solved stream values and the generated Python. **Solve** re-solves the edited model; **Save** writes the JSON; **Python** shows what `codegen.to_python` would produce.

Everything on the page is derived from `catalog()`, so plugin units appear with no extra work, and a field the catalog reports as holding code is listed as *set in code* rather than given a text box that could only reject what you type.

The point is not to replace writing Python. It is to make the tedious parts quick — seeing the topology, changing one parameter and re-solving, checking what a unit expects — while leaving the door open in both directions: export a script, edit it, and read the result back through `serialize`. An editor you can only enter is worse than none.

It is stdlib only (`http.server`), binds to `127.0.0.1`, and is meant for a single local user. It is a development tool, not a hardened service — do not expose it to a network.

### For tests and embedding

`make_server` builds the server without starting it, which is what a test wants:

```python
from difflow.gui import FlowsheetSession, make_server

server = make_server(FlowsheetSession(fs), port=0)   # 0 => any free port
```

The routes are `GET /api/catalog`, `GET /api/flowsheet`, `GET /api/code`, and `POST /api/flowsheet`, `/api/solve`, `/api/save`. A failed solve or a rejected edit comes back as `{"ok": false, "error": ...}` rather than a traceback at the socket, so a bad edit from the browser cannot take the server down.

One wrinkle worth knowing: JSON has no literal for the non-finite floats, and `JSON.parse` rejects the `Infinity` that Python's `json` writes. This is the *common* case, not an exotic one — `mass_action_kinetics` puts `inf` in `K_eq` for every irreversible reaction — so those values travel as the strings `"Infinity"`, `"-Infinity"` and `"NaN"`, and are restored on the way back.

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

Axis keys use the same `"<unit>.<param>"` dot notation as `make_objective_fn`, so they name a **unit** parameter — feed conditions are not addressable this way. To vary a feed, put a `Heater` or a `Mixer` in front of it and sweep that.

The page carries sliders for each axis, the outputs, and their sensitivities. This works because the solve is *pre-computed*: `sweep` evaluates the flowsheet on the grid with `jax.vmap`, takes gradients with `jax.grad`, and bakes the results into the page, which interpolates between them in a few lines of JavaScript.

That is a deliberate trade, and its limits should be stated plainly. JAX has no WebAssembly build, so a browser cannot run the real solver; the published page is an interpolation of a grid, not a live model. It is exact at the grid points and only as good as the grid between them, and it can only vary what the axes name. When you need the real thing, use the local editor above, or the generated script.

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
