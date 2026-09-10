# Gas Transmission Network Unit Operations

This document provides comprehensive documentation for the `difflow_gas` plugin, which models steady-state gas transmission networks as sequential-modular differentiable flowsheets.

---

(gas-overview)=
## Overview

The `difflow_gas` plugin provides:

- **Physics**: squared-pressure Weymouth pipe law, resistors, Nikuradse friction, adiabatic compressor power (plain and NLP-smoothed), GasLib unit conversions
- **A parser-agnostic network model** (`GasNetwork`) covering pipes, compressor stations, open valves, control valves, resistors, and short pipes
- **A topology-driven sequential decomposition** (`decompose`): the tear set and unit schedule of a meshed network are computed from the graph, not derived by hand
- **Unit operations** for both computed and hand-built decompositions
- **`GasNetworkFlowsheet`**: signed-flow Anderson tear solving plus a damped, jit- and grad-safe fixed-point solve (implicit function theorem gradients)
- **A mechanical flowsheet builder** (`build_network_flowsheet`)
- **Equation-oriented residual verification** (`difflow_gas.verify`) usable across solution methods

All operations are fully differentiable using JAX: gradients of any stream quantity with respect to compressor ratios, control valve drops, slack pressure, or pipe coefficients are exact, through the converged tear iteration.

The plugin was extracted from benchmark studies comparing sequential-modular and equation-oriented (interior-point) solutions of GasLib-11 and GasLib-40; see the package README (`src/difflow_gas/README.md`) for the findings that shaped its defaults, and Schmidt et al., *Data* 2(4):40, 2017 (doi:10.3390/data2040040) for the GasLib instances.

---

(gas-installation)=
## Installation

The gas plugin is included as an optional dependency:

```bash
pip install difflow[gas]
```

Or install with all extras:

```bash
pip install difflow[all]
```

---

(gas-conventions)=
## Stream and unit conventions

Gas streams use one pseudo-species `"gas"` whose flow is **signed mass flow in kg/s**; a negative flow is flow against an arc's reference direction, which is routine in meshed networks. Pressures are in Pa internally (bar in the reporting helpers), temperatures in K.

```python
from difflow_gas import gas_stream

s = gas_stream(mass_flow_kg_s=25.0, T_k=283.15, P_pa=50e5)
```

Because tear flows are signed, gas flowsheets must be solved with `clip_negative_flows=False`; `GasNetworkFlowsheet.solve()` does this by default.

---

(gas-network-model)=
## The network model

```python
import difflow_gas as dg

net = dg.GasNetwork(
    arcs={
        "p1":  ("src", "a", "pipe"),
        "cs1": ("a", "b", "compressor"),
        "p2":  ("b", "c", "pipe"),
        "p3":  ("b", "d", "pipe"),
        "p4":  ("c", "d", "pipe"),          # closes a loop
    },
    beta={aid: dg.weymouth_beta(L, 0.6, 1e-4)
          for aid, L in [("p1", 20e3), ("p2", 40e3),
                         ("p3", 60e3), ("p4", 80e3)]},
    supply_kg_s={"src": 120.0, "c": -50.0, "d": -70.0},
)
```

Arc kinds and their pressure relations:

| kind            | relation                        | decision parameter | may close a loop |
|-----------------|---------------------------------|--------------------|------------------|
| `pipe`          | `p_f^2 - p_t^2 = beta q abs(q)` | none               | yes              |
| `resistor`      | same, with `xi`                 | none               | yes              |
| `compressor`    | `p_t = ratio * p_f`             | `ratio`            | no               |
| `valve` (open)  | `p_t = p_f`                     | none               | no               |
| `short_pipe`    | `p_t = p_f`                     | none               | no               |
| `control_valve` | `p_t = p_f - dp`                | `dp_pa`            | no               |

The constructor validates kinds, self-loops, parallel arcs (not yet supported), missing/nonpositive resistance coefficients, and that the nominations balance. Optional fields carry node pressure bounds and compressor limits for the verification helpers.

---

(gas-decomposition)=
## The computed decomposition

```python
dec = dg.decompose(net, root="src")
dec.chord_ids     # the tear arcs, one per independent loop
dec.balances      # leaf-to-root affine mass-balance schedule
```

`decompose` builds a spanning tree with the non-invertible arc kinds forced in-tree and the **most resistive pipe/resistor of each loop pushed out as the chord** (tear). Given the chord flows, all tree-arc flows follow from affine leaf-to-root balances; pressures propagate root-to-leaf from the slack node; each chord recomputes its flow from its end pressures, which is the tear update. Every loop must contain at least one pipe or resistor, or `decompose` raises.

The chord choice controls convergence: the tear-map slope of a chord is roughly `-sum(beta_e |q_e|)/(beta_c |q_c|)` over its loop's tree arcs, so resistive chords keep the spectral radius small.

---

(gas-solving)=
## Building and solving

```python
fs, dec = dg.build_network_flowsheet(
    net, root="src", p_slack_pa=60e5, ratios={"cs1": 1.3},
)

streams = fs.solve(tol=1e-8)                    # Anderson, eager
streams = fs.solve_differentiable(alpha=0.3)    # damped, jit/grad-safe

rep = dg.residual_report(streams, net, dec)
assert rep.ok
```

Gas tear maps typically have real negative eigenvalues with spectral radius above 1 (loop updates overshoot), so the differentiable path iterates the damped map `x + alpha (g(x) - x)`. For eigenvalues in `[-m, 0)` the iteration contracts for `alpha < 2/(1+m)`; the default `alpha = 0.3` is safe for `m` up to about 5 (GasLib-40 measured `m ~ 3.1`). Gradients come from the implicit function theorem at the fixed point, so they are exact regardless of `alpha` or the iteration count.

Decision parameters are addressable through difflow's dot notation:

```python
import jax

obj = fs.make_objective_fn(
    lambda s: dg.total_compressor_power_w(s, dec, net.gas_temp_k))
gradient = jax.grad(obj)({"cs_cs1.ratio": 1.3, "src_src.P_set": 60e5})
```

---

(gas-optimization)=
## Optimization guidance

Two findings from the GasLib benchmark studies that apply to any reduced-space optimization over this simulator:

1. **Pose pressure constraints in squared pressure (bar^2).** The network response to controls is nearly linear in p^2; in p, low-pressure constraints are so nonlinear near their bounds that SQP linearizations overshoot into the floored-gradient region and optimizations collapse to false optima.
2. **Mind the pressure floor.** Units floor squared pressures at `MIN_P_SQUARED` ((0.5 bar)^2) to survive unphysical tear transients; gradients vanish there. Squared-pressure constraints and feasible starting points keep optimizers out of that region.

`total_compressor_power_w` uses the smoothed `|q|` (`smoothed_power_w`, eps = 1e-4 kg/s) so reduced-space objectives are identical to standard equation-oriented NLP objectives, making cross-method optimum comparisons meaningful at tight tolerance.

---

(gas-unit-reference)=
## Unit operation reference

Fifteen small units, each one equation. They are listed together rather
than given a chapter apiece because that is how they are understood ---
as the equation set of a network decomposition --- but each has its own
anchor, so a link can land on the unit rather than on this heading.

Which ones you use follows from how the flowsheet was decomposed.
`build_network_flowsheet` uses the tree-propagation set; a hand-built
flowsheet that pushes pressures downstream from a known source uses the
forward-mode set.

### Tree-propagation units

Two inlets each: the parent node's stream (which carries the pressure)
and the arc's flow stream. The output is the child node's stream. These
are what `decompose` schedules.

(gas-op-pipepressure)=
`PipePressure(beta, direction)` --- squared-pressure Weymouth drop along
the arc: `p_child^2 = p_parent^2 ∓ beta q |q|`, with `direction=+1` when
the tree is traversed *with* the arc (the parent is the arc's `from`
node) and `-1` against it. Floored at `MIN_P_SQUARED`.

(gas-op-compressorboost)=
`CompressorBoost(ratio, direction)` --- a compressor station:
`p_child = ratio * p_parent` downstream (`direction=+1`), or
`p_parent / ratio` when the child is the station inlet. `ratio` is the
decision variable.

(gas-op-pressureequal)=
`PressureEqual()` --- open valves and short pipes, in either traversal
direction: `p_child = p_parent`, no parameters.

(gas-op-controlvalvedrop)=
`ControlValveDrop(dp_pa, direction)` --- a control valve's parametric
linear reduction, `p_child = p_parent ∓ dp`, floored at `MIN_P` so an
unphysical iterate cannot produce a nonpositive pressure. `dp_pa` is the
station's decision variable.

### Chord unit

(gas-op-pressuredrivenpipe)=
`PressureDrivenPipe(beta)` --- the inverse relation: *flow* from the two
end pressures, `q = sign(Δp²) sqrt(|Δp²| / beta)`. One per independent
loop; its computed flow is what the tear iteration updates against.

### Forward-mode units

Single inlet (or inlet plus specification), for hand-built flowsheets
that propagate state downstream.

(gas-op-gaspipe)=
`GasPipe(beta)` --- `p_out = sqrt(p_in^2 - beta q |q|)`, carrying the
signed flow through unchanged.

(gas-op-backpipe)=
`BackPipe(beta)` --- the same pipe read backwards:
`p_src = sqrt(p_node^2 + beta q |q|)`, for a flow-specified entry whose
pressure is an *output* of the solve rather than an input.

(gas-op-compressor)=
`Compressor(ratio)` --- fixed-ratio boost, `p_out = ratio * p_in`. (The
catalog name is `Compressor`; difflow's EOS-consistent compressor is
registered as `EOSCompressor`.)

(gas-op-openvalve)=
`OpenValve()` --- `p_out = p_in`.

### Topology and bookkeeping

(gas-op-sourcehead)=
`SourceHead(P_set)` --- pin the slack node's pressure. A nomination
scenario fixes boundary *flows*, so one node must supply the pressure
level; keeping it in a unit parameter rather than in the feed stream is
what makes it differentiable through `Flowsheet._apply_params`.

(gas-op-affineflow)=
`AffineFlow(const, signs, T_k, P_pa)` --- a tree arc's flow from the
node's local mass balance, `q = const + Σ signs_i q_i`, where `const` is
the node's nomination and the inlets are its child tree-arc flows and
incident chord tears. T and P on the output are placeholders: a flow
stream carries only flow.

(gas-op-flowsplit)=
`FlowSplit(w)` --- fixed draw: `w` to the first outlet, the remainder to
the second. For a demand branch whose flow the nomination fixes.

(gas-op-tearsplit)=
`TearSplit()` --- the same split with the flow taken from a second
(tear) inlet instead of a parameter. The entry point of a hand-built
recycle that closes a loop.

(gas-op-junction)=
`Junction()` --- a network node: flows add, and the pressure comes from
the **first** inlet. Unlike difflow's `combine_streams` (which takes the
minimum pressure), every arc at a gas node sees the same nodal pressure,
so the junction takes it from the designated pressure-defining branch;
at convergence all inlets agree, and the difference beforehand is a
residual rather than a modelling choice. The outlet temperature is the
flow-weighted mean.

(gas-op-flowminus)=
`FlowMinus()` --- `q = q_a - q_b`, with T and P from `a`. Tear-update
bookkeeping.

Parameters live in `ParamsMixin` dataclasses (`PipeParams`,
`CompressorParams`, `ControlValveParams`, `SourceHeadParams`, ...), so
`Flowsheet._apply_params` can rebind them functionally for
differentiation.

---

(gas-limitations)=
## Scope and roadmap

Implemented: the six arc kinds above, fixed balanced nominations, one slack node, isothermal steady state. Not yet: parallel arcs, closed/switchable valves (currently a topology edit), pressure-specified entries / multiple slacks, elevation terms, transients. See the package README for the ordered roadmap.
