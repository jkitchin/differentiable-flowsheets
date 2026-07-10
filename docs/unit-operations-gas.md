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

Tree-propagation units (used by the builder; two inlets, parent node stream and arc flow stream):

- `PipePressure(beta, direction)`: squared-pressure drop with (+1) or against (-1) the arc
- `CompressorBoost(ratio, direction)`: multiply or divide by the ratio
- `PressureEqual()`: valves and short pipes
- `ControlValveDrop(dp_pa, direction)`: parametric linear drop, floored at 0.5 bar

Chord unit:

- `PressureDrivenPipe(beta)`: signed flow from two end pressures, `q = sign(dp2) sqrt(|dp2|/beta)`

Forward-mode units (hand-built flowsheets):

- `GasPipe(beta)`, `BackPipe(beta)`, `Compressor(ratio)`, `OpenValve()`

Topology and bookkeeping:

- `SourceHead(P_set)`: pin the slack node pressure (differentiable parameter)
- `AffineFlow(const, signs, T_k, P_pa)`: tree-arc flow from a local mass balance
- `FlowSplit(w)`, `TearSplit()`, `Junction()`, `FlowMinus()`

Parameters live in `ParamsMixin` dataclasses (`PipeParams`, `CompressorParams`, `ControlValveParams`, `SourceHeadParams`, ...), so `Flowsheet._apply_params` can rebind them functionally for differentiation.

---

(gas-limitations)=
## Scope and roadmap

Implemented: the six arc kinds above, fixed balanced nominations, one slack node, isothermal steady state. Not yet: parallel arcs, closed/switchable valves (currently a topology edit), pressure-specified entries / multiple slacks, elevation terms, transients. See the package README for the ordered roadmap.
