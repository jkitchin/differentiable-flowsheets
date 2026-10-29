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

# Gradient through stream operations: a first-order A -> B CSTR
from difflow.units.cstr import CSTR, CSTRParams

def rate_fn(C, T, rp):
    return jnp.array([rp['k0'] * jnp.exp(-rp['Ea'] / 8.314 * (1.0 / T - 1.0 / 350.0)) * C['A']])

reactor = CSTR(CSTRParams(V=1.0, rate_fn=rate_fn, stoich=jnp.array([[-1.0], [1.0]]),
                          rate_params={'k0': 0.1, 'Ea': 50000.0},
                          species_order=['A', 'B'], molar_density=10.0))

def objective(inlet_T):
    stream = make_stream({'A': 1.0, 'B': 0.0}, T=inlet_T, P=101325.0)
    outlet, _ = reactor(stream, T_spec=inlet_T)
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

<!-- doc-test: skip: signature listing, redefines the class -->
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

<!-- doc-test: skip: signature listing, redefines the class -->
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
import jax.numpy as jnp
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream
from difflow.thermo import IdealThermo
from difflow.database import get_species_data
from difflow.kinetics import mass_action_kinetics
from difflow.units.cstr import CSTR, CSTRParams
from difflow.units.flash import Flash, FlashParams, Mixer, Splitter

# Isomerisation n-hexane -> 2-methylpentane
species = ['n_hexane', '2_methylpentane']
thermo = IdealThermo({s: get_species_data(s) for s in species})

# Declarative mass-action kinetics (data, so the flowsheet can be serialized later)
kin = mass_action_kinetics(
    [{'reactants': {species[0]: 1.0}, 'products': {species[1]: 1.0},
      'rate_params': {'A': 0.05}}],            # k = 0.05 1/s
    species_order=species,
)
reactor_params = CSTRParams(V=1.0, molar_density=10.0, **kin.params_kwargs())
feed = make_stream({species[0]: 1.0, species[1]: 0.0}, T=340.0, P=101325.0)

mixer = Mixer(species, thermo)
reactor = CSTR(reactor_params, thermo)
flash = Flash(FlashParams(species_order=species), thermo)
splitter = Splitter(species)

# Initialize flowsheet
fs = Flowsheet(species_order=species)

# Add feed streams
fs.add_feed('fresh_feed', feed)

# Add mixer (combines fresh feed and recycle)
fs.add_unit(Unit(
    name='mixer',
    operation=mixer,
    inlet_names=['fresh_feed', 'recycle'],
    outlet_names=['reactor_feed'],
    params={}
))

# Add reactor
fs.add_unit(Unit(
    name='reactor',
    operation=reactor,
    inlet_names=['reactor_feed'],
    outlet_names=['reactor_effluent'],
    params={'T_spec': 340.0}
))

# Add flash drum (Flash returns liquid, then vapor)
fs.add_unit(Unit(
    name='flash',
    operation=flash,
    inlet_names=['reactor_effluent'],
    outlet_names=['liquid', 'vapor_product'],
    params={'T': 337.0}
))

# Add splitter for recycle
fs.add_unit(Unit(
    name='splitter',
    operation=splitter,
    inlet_names=['liquid'],
    outlet_names=['recycle', 'liquid_product'],
    params={'split_frac': 0.8}   # fraction to the first outlet
))

# Define recycle connection
fs.add_recycle(source='recycle', dest='recycle')   # same name: the splitter outlet feeds the mixer inlet

# Solve flowsheet
results = fs.solve(tol=1e-6, max_iter=50)
```

### Degrees of Freedom

`fs.degrees_of_freedom()` counts the flowsheet's variables, specifications and
equations from its topology, without solving anything. Every stream carries
`n_species + 2` variables (the flows, T and P); the feeds are specified; each
unit contributes the equations fixing its outlets from its inlets (the length
of its `eo_residuals` when it has them); a recycle whose source and destination
are named differently adds `dest = source`. A unit's parameters (a volume, a
conversion, a split fraction) are its design specifications and are listed per
unit rather than counted. A well-posed flowsheet has zero degrees of freedom:

```python
dof = fs.degrees_of_freedom()
print(dof)                      # variables, specified, equations, per unit
assert dof.degrees_of_freedom == 0 and dof.is_well_posed
dof.unknowns                    # the non-feed streams: the EO solver's unknowns
```

A positive count names an inlet nothing supplies (not a feed, not a unit outlet,
not a recycle destination); a negative one names a stream fixed twice (made by
two units, or a feed that a unit also makes). Both are listed in `dof.problems`.
The result is a `difflow.DegreesOfFreedom`.

### Sequential Modular Solution

The flowsheet solver uses sequential modular approach:

1. **Tear streams**: Identify recycle streams to "tear"
2. **Initialize**: Set initial guesses for tear streams (see
   [Where the initial guess comes from](convergence.md#where-the-initial-guess-comes-from))
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

### Choosing the Tear Streams

Where a loop is torn is not a formality. The tear is the fixed point the
solve iterates on, so its placement sets the spectral radius of the map ---
how fast the loop converges, and sometimes whether it converges at all.

By default that choice is yours and only yours: `add_recycle(source, dest)`
tears exactly where you said, and a loop closed in the topology but never
declared is not torn at all (the sequential pass hits an inlet nothing has
written yet and raises `KeyError`). `tear_analysis()` says what difflow
would have picked, without solving anything and without running a single
unit:

```python
print(fs.tear_analysis())
```

For a variant of the flowsheet above in which the splitter also sends a stream
`short` straight back to the mixer, the report reads:

```text
Tear analysis: 2 recycle loop(s)
  loop: mixer -> reactor -> splitter -> flash -> mixer
  loop: mixer -> reactor -> splitter -> mixer
  declared tears:  recycle
  heuristic would: recycle, mixed
  minimum would:   mixed
  NOT TORN: mixer -> reactor -> splitter -> mixer
  inlets read before they are written: short
```

That flowsheet has two loops and one `add_recycle`. The second loop, the
splitter's `short` stream back to the mixer, is closed in the topology and
torn nowhere --- which is why the units cannot run in the order they were
added, and why `solve()` raises `KeyError: 'short'` rather than iterating
on it. The heuristic keeps the declared tear and adds one for the loop that
has none; `minimum` notes that tearing `mixed` alone would break both.

The returned `TearAnalysis` carries the same thing as data --- `cycles`,
`declared`, `heuristic`, `minimum`, `uncovered`, `missing_inputs`,
`out_of_order`, and `torn`, which is `True` when the declared tears break
every loop that was found. The last three are the ones worth checking on a
flowsheet that will not run: `missing_inputs` names inlets nothing
supplies, and `out_of_order` names inlets read before they are written ---
a unit order that needs a tear it does not have.

### Tearing Automatically

`solve(tears="auto")` fills the gap when no recycle has been declared:

```python
auto_fs = Flowsheet(species)
auto_fs.add_feed("feed", feed)
auto_fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
auto_fs.add_unit(Unit("reactor", reactor, ["mixed"], ["rx_out"], {"T_spec": 340.0}))
auto_fs.add_unit(Unit("splitter", splitter, ["rx_out"], ["product", "recycle"], {"split_frac": 0.7}))

streams = auto_fs.solve(tears="auto")     # no add_recycle anywhere
auto_fs.last_solve_tear_streams           # -> ['mixed'], what it chose
```

Two strategies, and `tears=` names them: `"auto"` (`"heuristic"`) takes one
tear per loop, preferring a stream the user declared, then one lying on
several loops, then one leaving a mixing point --- where the stream is the
sum of everything entering it, so a guess that is wrong in composition is
still right in order of magnitude. `"minimum"` instead covers every loop
with as few tears as it can (greedily: an exact minimum tear set is
NP-hard).

Tearing somewhere other than where the units happen to be listed means the
units run in a different order, and `calculation_order` works it out: tear
`mixed` in the loop above and the reactor runs first, the mixer last. The
fixed point is the same one.

Two things it deliberately does not do. A declared recycle always wins ---
an automatic choice fills a gap rather than overruling a decision someone
made --- and the choice is not kept: `fs.recycles` and `fs.units` are
exactly what you declared once the solve returns, so nothing about a later
`solve()` changes. Inferring tears silently would change the answer for
flowsheets that already run, which is why the default stays `"declared"`.

The pieces are usable on their own, against any flowsheet:

```python
from difflow import (
    FlowsheetGraph, find_cycles, select_tear_streams, calculation_order,
)

graph = FlowsheetGraph.from_flowsheet(fs)   # units, streams, edges
find_cycles(graph)                          # elementary cycles, as unit names
tears = select_tear_streams(fs, method="minimum")
calculation_order(graph, tears)             # unit order once those are seeded
```

`find_cycles` enumerates *elementary* cycles, of which a densely recycled
graph can have exponentially many; `max_cycles` bounds the work and a
`CycleEnumerationWarning` says when it was hit.

Three methods iterate on the tear streams, selected with `acceleration`.
[Convergence and Initialization](convergence.md) covers all of this in
depth --- where the initial guess comes from, what each method costs, and
what to reach for when a loop will not close. The short version:

### Direct Substitution

`acceleration="none"`: fixed-point iteration on tear streams, run through
`optimistix`, taking a fraction `damping` of each step.

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}
+ \alpha\left(f(\mathbf{x}^{(k)}) - \mathbf{x}^{(k)}\right)$$

Where $\mathbf{x}$ is the tear stream vector, $f$ is the flowsheet calculation
and $\alpha$ is `damping` (1.0 by default, i.e. plain substitution). Damping
leaves the fixed point alone and makes the iteration contractive where $f$
overshoots. This is also the path a traced solve falls back to, because it is
the only one without a Python branch on the residual.

### Wegstein Acceleration

`acceleration="wegstein"`: estimates the tear map's slope from the last two
iterates and extrapolates.

$$\mathbf{x}^{(k+1)} = q\, \mathbf{x}^{(k)} + (1 - q)\, f(\mathbf{x}^{(k)}),
\qquad q = \frac{s}{s - 1},
\qquad s = \frac{f(\mathbf{x}^{(k)}) - f(\mathbf{x}^{(k-1)})}{\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}}$$

$q$ is the weight on the *old* iterate, bounded to $q \in [-5, 0]$ for
stability. The bound is what makes this the method for an *oscillating* loop.

### Anderson Acceleration

`acceleration="anderson"` (the default): keeps a history of `anderson_depth`
iterates and solves a small least-squares problem for the combination that
minimises the residual --- equivalent to GMRES on the fixed-point iteration,
and exact on an affine map.

### Convergence Parameters

```python
results = fs.solve(
    tol=1e-6,                 # Convergence tolerance
    max_iter=100,             # Maximum iterations
    acceleration='wegstein',  # 'none', 'wegstein' (default: 'anderson')
    anderson_depth=5,         # History depth, for acceleration='anderson'
    damping=1.0,              # Step fraction, for acceleration='none'
    clip_negative_flows=True, # False for signed tear flows (e.g. gas networks);
                              # binds on the accelerated paths only
    error_probe=2,            # extra passes measuring the error tol does not
    tol_basis='step',         # 'error' to make tol mean the answer's accuracy
)
```

### When the Solve Does Not Converge

A tear iteration that runs out of iterations returns anyway, and what it
returns is the last iterate: every stream present, every number plausible, and
the material balance around the loop off by the tear residual. For a loop that
is limit-cycling that error is not small.

`solve()` says so rather than leaving it to the caller to go looking:

```python
streams = fs.solve()                          # ConvergenceWarning if it didn't
streams = fs.solve(on_nonconvergence="raise") # ConvergenceError instead
streams = fs.solve(on_nonconvergence="ignore")# silent; check last_solve_* yourself
```

The message names the tear streams, the final residual, the tolerance, the
iteration count and the method used --- the same `last_solve_*` diagnostics the
report layer and the editor read, which stay available whichever option is
chosen:

```text
Recycle solve did not converge: tear stream(s) recycle reached a residual of
3.2e-03 after 100 of 100 iterations, against a tolerance of 1.0e-08
(method: anderson). The returned streams are the last iterate, not a solution
-- material balances around the loop are off by the tear residual. ...
```

`ConvergenceWarning` and `ConvergenceError` are exported from `difflow`, so the
usual `warnings.simplefilter("error", ConvergenceWarning)` turns every
non-converged solve in a script into a failure.

What to do about it --- a better tear guess, a different acceleration, the
equation-oriented solver --- is
[When a solve does not converge](convergence.md#when-a-solve-does-not-converge).

Under `jax.grad` or `jit` the residual is a tracer with no numeric value at
trace time, so `last_solve_converged` is `None`. That path is the optimistix
fixed point, whose implicit-differentiation rule assumes the loop converged; an
unconverged one gives a wrong gradient. The check therefore runs when the
computation does: a `ConvergenceWarning` fires as the gradient is evaluated
(a warning even with `on_nonconvergence="raise"`, since an exception would make
the flowsheet undifferentiable), and `on_nonconvergence="ignore"` turns it off. See
[The traced fallback](convergence.md#the-traced-fallback).

### Implicit Differentiation

The flowsheet solver uses implicit differentiation through the converged solution:

$$\frac{d\mathbf{y}}{d\mathbf{p}} = \left(\mathbf{I} - \frac{\partial f}{\partial \mathbf{x}}\right)^{-1} \frac{\partial f}{\partial \mathbf{p}}$$

This is implemented using JAX custom VJP rules, enabling gradient computation through the entire flowsheet solve.

```python
import jax

# Gradient through flowsheet
def flowsheet_objective(feed_flow):
    original = fs.feeds['fresh_feed']
    fs.feeds['fresh_feed'] = make_stream({species[0]: feed_flow, species[1]: 0.0},
                                         T=340.0, P=101325.0)
    try:
        results = fs.solve()
    finally:
        fs.feeds['fresh_feed'] = original   # do not leave a tracer in the flowsheet
    return results['liquid_product'][f'F_{species[1]}']

grad_fn = jax.grad(flowsheet_objective)
sensitivity = grad_fn(1.0)   # d(product B) / d(fresh feed flow)
```

### Objectives over unit parameters

`create_objective(fs, objective_fn, **solve_kwargs)` (also
`fs.make_objective_fn`) turns a flowsheet into a function of a dict of unit
parameters, `{"unit.field": value}`, that `jax.grad` can differentiate. A
field may name a key of the unit's `Unit.params` (the keyword arguments it is
called with, such as a splitter's `split_frac` or the reactor's `T_spec`
above) or a field of the operation's own `Params`. Extra keyword arguments go
to `fs.solve`, so a loop that needs more iterations under `jax.grad` (plain
substitution) can be given them.

```python
from difflow import create_objective

obj = create_objective(
    fs, lambda res: res['liquid_product'][f'F_{species[1]}'], max_iter=500,
)
p = {'splitter.split_frac': 0.8}
print(float(obj(p)), float(jax.grad(obj)(p)['splitter.split_frac']))
```

---

## Plugin System

**Location**: `difflow/plugins.py`

### Plugin Architecture

Difflow uses a plugin system for extensibility:

```python
from difflow.plugins import register_operation, UnitOperation

# Register a new operation (adds it to the global ``registry``)
@register_operation(
    name='my_reactor',
    category='reactors',
    description='Custom reactor model'
)
class MyReactor:
    def __call__(self, inlet, **params):
        # Custom reactor logic
        outlet, info = dict(inlet), {}
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
from difflow.plugins import registry

# List available operations (name -> OperationInfo)
operations = registry.list_operations()

# Get operation by name
CSTR = registry.get('CSTR')

# List operations by category
reactors = registry.list_operations(category='reactors')
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
from difflow.plugins import registry

bioreactor = registry.get('ContinuousBioreactor')
centrifuge = registry.get('DiscStackCentrifuge')
protein_a = registry.get('ProteinAChromatography')
```

---

## Saving and Loading Flowsheets

A flowsheet is otherwise only expressible as Python: the code that builds it *is* the model. `difflow.serialize` gives it a file format, so a model can be saved, diffed, sent to a service, or read by something that never imported the module that built it.

```python
from difflow import serialize

serialize.save(fs, "plant.json")
fs2 = serialize.load("plant.json")
```

The round trip preserves the answer, not just the shape — a reloaded flowsheet solves to bit-identical results, and its gradients are the same. `to_json`/`from_json` and `to_dict`/`from_dict` are available if you want the text or the data rather than a file.

`save`/`load` take any difflow model, not only a steady-state `Flowsheet`:

| Model | What is written |
|---|---|
| `Flowsheet`, and a subclass such as `GasNetworkFlowsheet` | feeds, units, wiring, recycles; the subclass is kept, since it may solve differently |
| `DynamicFlowsheet` | each unit as its class and constructor call, feeds (a constant stream or a function of time), wiring and connections |
| `PowerNetwork`, `RadialFeederFlowsheet` | buses, branches, generators and loads in insertion order; a feeder as its constructor call |

For a single object — a `Params`, a thermo, an equation of state, a `ReactionSet` — use `dumps_object`/`loads_object` or `save_object`/`load_object`, and `to_value`/`from_value` for the JSON-ready data:

```python
serialize.save_object(thermo, "thermo.json")
thermo = serialize.load_object("thermo.json")
```

### Format versions

Every file records `format_version` (checked on read against `SUPPORTED_VERSIONS`) and the difflow version that wrote it (for provenance only). difflow reads versions 1, 2 and 3, and writes the **lowest version that can hold the file**: a flowsheet using nothing newer than version 2 is still written as 2, so a release that reads only 2 keeps opening it. Version 3 is used only by files an older reader would *misread* rather than refuse — an equation of state, a `CubicThermo` or `ActivityThermo`, a function or class named by import path, a flowsheet subclass, a dynamic flowsheet or another model kind — and an older release refuses it by name.

Version 2 added an optional top-level `view` block, which round-trips through `Flowsheet.view` and is read by nothing numeric:

```python
fs.view = {"nodes": {"reactor": {"x": 260.0, "y": 40.0},
                     "feed:feed": {"x": 40.0, "y": 40.0}},
           "code_context": "thermo = IdealThermo(...)"}
```

It is where the editor keeps canvas positions, its code-context snippet and its planning selection, so a flowsheet laid out by hand opens the way it was left. Node keys follow the same vocabulary as `_apply_params`: a unit is its bare name, a feed is `feed:<stream>`. Nothing validates the contents, and `_apply_params` copies the block through, so a swept or optimized flowsheet keeps its layout.

### What it writes

Round-tripping goes through the [operation registry](#operation-catalog): a unit is written as the name it is registered under and rebuilt by looking that name up. A unit class that is not registered is written by its import path (`"mylab.units:Scale"`) — see [Your own code](#your-own-code) below.

Values are written as *data*:

- numbers, strings, arrays (including `inf`/`nan`, which plain JSON cannot hold), lists, dicts;
- dataclasses such as `Params`, and NamedTuples such as `SpeciesData` — a field declared as a tuple comes back a tuple, so a reloaded object compares equal to the one written;
- thermo and equations of state (below);
- a function: one built from data by a factory such as `mass_action_kinetics` is written as that call; a module-level function, difflow's (`kinetic_fn=monod_kinetics`) or yours, is written by its import path;
- any other object whose class either keeps every constructor argument under an attribute of the same name (`EquilibriumCurve`, `IdealGasSet`), or records its constructor call with `difflow.init_args.record_init_args` (the dynamic units, `RadialFeederFlowsheet`). An object rebuilt from its attributes is rebuilt once on write and must give back the same data, so a class whose attributes do not say how it was built is refused rather than written wrong.

A **lambda or closure** has no data form and no import path, and is refused rather than dropped:

```
SerializationError: unit 'reactor' field 'rate_fn' holds a callable ('<lambda>'),
which cannot be written to a file: a lambda, a closure or a function defined in a
script or notebook has no import path to be found again by. ...
```

That is deliberate. A file that silently lost a reactor's rate law would reload into a different model that still looked plausible. Pass such an object by name instead, with `refs=` on both save and load:

```python
serialize.save(fs, "plant.json", refs={"rate": rate})
fs2 = serialize.load("plant.json", refs={"rate": rate})
```

A value traced by `jax.grad`, `jit` or `vmap` has no number yet, and is refused with a message saying so.

### Your own code

A function or class defined at the top level of a module of yours — a rate law, a `Params` dataclass, a unit class, a `Flowsheet` subclass — is written by its import path. Loading it imports that module, which runs its code, so a file is only allowed to do that when you say so:

```python
fs = serialize.load("plant.json")               # SerializationError: ... pass trust=True
fs = serialize.load("plant.json", trust=True)   # imports mylab and rebuilds
```

Modules of difflow and its plugins are always allowed. Code defined in a script or a notebook (`__main__`) has no import path that would find it again in another session, so it is refused on write; move it into a module, or use `refs=`. The editor asks the same question before opening such a file, listing the modules (`serialize.untrusted_modules(document)` lists them for any document).

### Thermodynamics

About half the core units need a `thermo` or `eos` object in the constructor, not just a `Params`. Written and rebuilt automatically, wherever they appear (a constructor argument or a `Params` field):

- `IdealThermo`, as its species data;
- `CubicThermo(ideal, eos)` and `ActivityThermo(ideal, activity_model)`, as the `IdealThermo` and the object that corrects it;
- `PengRobinson` and `SRK`, as the `CriticalProperties` of each species and `k_ij` in the form it was given (a `{(i, j): value}` dict travels as `[i, j, value]` triples, since JSON keys cannot be tuples);
- any thermo that is a dataclass (`IdealGasThermo`, `GibbsThermo`).

A subclass of one of these is not written as its base, since that would drop whatever it adds. Any thermo the format cannot write can be supplied on load instead:

```python
my_thermo = thermo   # any property package
fs2 = serialize.load("plant.json", extras={"flash": {"thermo": my_thermo}})
```

`extras` also *overrides* a stored thermo, which is the way to reload a saved flowsheet against a different property package.

### Units that build their own `Params`

Most units are constructed as `Unit(Params(...), thermo)`, but a substantial minority take their numbers as plain constructor arguments and build the `Params` themselves — `Compressor(ratio)`, `FlowSplit(w)`, `GasPipe(beta)`, and most of the gas plugin. Those numbers are written under `params`, because that is where the built unit keeps them, and they are passed straight back to the constructor on load. Nothing extra is needed:

```python
from difflow_gas import Compressor

gas_fs = Flowsheet(species_order=["CH4"])
gas_fs.add_unit(Unit("boost", Compressor(ratio=1.3), ["a"], ["b"]))
serialize.from_json(serialize.to_json(gas_fs))   # ratio comes back as 1.3
```

An argument that is *not* data — `Mixer(species_order)`, a `thermo` — still travels under `constructor`, and the two channels compose: `CompressorBoost(ratio, direction)` carries `ratio` by the first road and `direction` by the second.

An *optional* constructor argument is written when it was set away from its default — `Mixer(species, thermo)`, `CSTR(params, thermo, mode="adiabatic")` — and left out when it was not. It is read off the unit by the attribute of the same name, so a unit that normalises a default it was given (`TFF` stores `rejection=None` as `{}`) writes that normalised value; it reloads to the same unit. Dropping it would be quietly wrong: `Mixer(species)` mixes temperatures without an enthalpy balance, so the reloaded flowsheet would solve to a different answer. An optional argument the format cannot hold (a lambda, an unsupported thermo) raises `SerializationWarning` rather than refusing the whole file, and the unit reloads with the default unless you pass it back with `extras=`. The editor keeps these arguments, too, when you change a parameter of the unit.

### Saved results

A solve's streams can be kept with the model, to inspect without solving again and to start the next solve where this one finished:

```python
streams = fs.solve()
serialize.save(fs, "plant.json", results=streams)

fs = serialize.load("plant.json")
fs.saved_results["product"]          # the saved stream
fs.saved_solve_info["converged"]     # what the solve reported
streams = fs.solve(tear_initial=serialize.warm_start(fs))
```

`warm_start` returns the saved value of every tear stream, ready for `tear_initial`. Nothing is warm-started unless you ask: a solve never silently starts from a file's numbers. An older reader ignores the `results` block, so saving results does not raise the format version.

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

<!-- doc-test: skip: illustrative generated-script output (contains `...` placeholders) -->
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

A data-built rate law is **hoisted** into its own statement and splatted back rather than inlined, which keeps the reactor to one readable line and avoids repeating the arrays the factory derives anyway. Thermo is hoisted the same way, and written as a `get_species_data` expression when every species is in the database *and* its data is the database's (an edited `SpeciesData` is written out in full); an equation of state likewise as `get_critical_props`.

Running the generated script rebuilds a flowsheet that solves to bit-identical results. A plugin unit is imported from its plugin (`from difflow_gas import GasPipe`), and your own functions and classes from your module, matched by class rather than by name, since some names are shared with core (the gas `Compressor` is not `difflow.Compressor`). Every example flowsheet bundled with the editor round-trips through its generated script.

`codegen` writes steady-state flowsheets, and refuses exactly what `serialize` refuses, for the same reason: a callable with no data form and no import path. A generated script that quietly dropped a rate law would still run, and would be a different model. A `DynamicFlowsheet` or power network is saved with `serialize.save`.

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
Cp -- J/mol/K -- Constant heat capacity (J/mol·K). Ignored when the unit is built with a ``thermo``; when there is neither, the unit falls back to DEFAULT_CP and warns (DefaultCpWarning).
phase -- - -- Force one phase ('liquid'/'vapor') for the thermo enthalpy. None (default) uses the thermo's two-phase flash enthalpy where it has one, so latent heat is carried through a partial vaporization.
```

That text is not a second copy. `difflow.docstrings` reads it out of the `Params` class's own `Attributes:` section, which is where the project already writes it, and out of the comments beside the fields for the 35 that are documented there instead — `CSTRParams.eos` and `CSTRParams.outlet_volumetric_basis` among them. Populating `field(metadata={"description": ...})` on all 87 `Params` classes would have duplicated every description and then drifted from it.

The editor gets this for free: the inspector's parameter list already rendered `spec.description` as help text under each field, and the palette already rendered `spec.units` beside its name — both written against a schema that carried neither. 470 of the 487 catalogued parameters gain help text in the inspector with no front-end change.

Units are *not* read from the prose. They come from the class's `parameter_units` table (above), which names every numeric field and is guarded by a test; a parenthetical in a sentence is a weaker signal, since `(0-1)` and `(default 10)` sit in the same position as `(Pa)`, and a formula in a description will happily offer `-1/k` as "per kelvin".

Where a field needs to say something different from its docstring, its metadata still wins:

<!-- doc-test: skip: dataclass-field fragment, not standalone -->
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

## The Browser Editor

difflow's browser editor --- a canvas to build a flowsheet from the
operation palette, an inspector with each unit's documentation, results and
sensitivities, planning, a Python console, an assistant, and the debugger's
**Debug** and **Trace** panels --- is an app that comes with the difflow
book. It installs as its own package and registers itself with difflow, so
with it installed:

```bash
difflow gui plant.json            # a saved flowsheet
difflow gui plant.py              # one a script builds
```

Without it, `difflow gui` says where it comes from, and nothing else in
difflow needs it. Its engine is open and here:
`difflow.workspace.FlowsheetSession` is a live flowsheet with every
operation on it --- edits, the code context, undo, solves, sensitivities,
the console, the [debugger](debugger.md) --- and it is what the
[agent tools](agents.md) drive. The flowsheet files the editor reads and
writes are `difflow.serialize` documents; `difflow.workspace.examples`
holds the example flowsheets.

## Parameter Sweeps

`sweep` evaluates a flowsheet across a grid of parameter values with
`jax.vmap`, and records every output and its exact derivative with respect
to every axis:

```python
from difflow import SweepAxis, sweep

axes = [
    SweepAxis("reactor.V", 0.5, 5.0, n=5, label="Reactor volume", units="m³"),
    SweepAxis("feed:fresh_feed.F_n_hexane", 0.5, 1.5, n=3, label="Feed rate", units="mol/s"),
]
outputs = {"conversion": lambda streams: 1 - (streams["liquid_product"]["F_n_hexane"]
                                              + streams["vapor_product"]["F_n_hexane"])
                                    / streams["fresh_feed"]["F_n_hexane"]}

result = sweep(fs, axes, outputs)
result.values["conversion"]                    # shape (5, 3)
result.gradients["conversion"]["reactor.V"]    # d conversion / d V at every point
```

Axis keys are whatever `_apply_params` accepts --- `"<unit>.<param>"` and
`"feed:<stream>.<field>"` --- so a feed rate or a feed temperature is an
axis like any other. (The book's app can turn a sweep into a self-contained
interactive page for a paper's supplementary material.)

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

<!-- doc-test: skip: ordering sketch with placeholder unit names -->
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
# Check individual units against the solved streams
streams = fs.solve()
for unit in fs.units:
    inlet = streams[unit.inlet_names[0]]
    outlet = streams[unit.outlet_names[0]]
    print(f"{unit.name}: {inlet['T']:.1f} K -> {outlet['T']:.1f} K")

# What the last recycle solve actually did
print(fs.last_solve_method)      # 'anderson', 'wegstein', 'none', ...
print(fs.last_solve_iterations)  # == max_iter if it ran out
print(fs.last_solve_residual, fs.last_solve_tol)
print(fs.last_solve_converged)   # False is a finding; None means "not judged"
```

The full decision tree is in
[Convergence and Initialization](convergence.md#when-a-solve-does-not-converge).

### Memory Efficiency

```python
# Use JIT compilation for repeated evaluations
@jax.jit
def evaluate_flowsheet(feed_flow):
    original = fs.feeds['fresh_feed']
    fs.feeds['fresh_feed'] = make_stream({species[0]: feed_flow, species[1]: 0.0},
                                         T=340.0, P=101325.0)
    try:
        streams = fs.solve()
    finally:
        fs.feeds['fresh_feed'] = original
    return streams['liquid_product'][f'F_{species[1]}']

# Vectorize over multiple cases
cases = jnp.array([0.8, 1.0, 1.2])
results = jax.vmap(evaluate_flowsheet)(cases)
```
