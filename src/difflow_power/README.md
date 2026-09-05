# difflow_power — electrical grids as differentiable flowsheets

AC power flow, AC optimal power flow and grid control, built so that
every answer carries its derivatives.

```python
import difflow_power as dp

net = dp.cases.case9()                 # WSCC 9-bus benchmark
pf  = dp.solve_power_flow(net)         # Newton-Raphson
opf = dp.solve_acopf(net)              # interior-point AC-OPF

opf.cost          # 5296.69 $/h  -- MATPOWER's published optimum
opf.lmp_mw        # locational marginal prices, $/MWh
opf.binding()     # which limits hold the dispatch back, and their price
```

## Why this exists

Every number a grid study produces is really a question about a
derivative. *How much does another megawatt at this bus cost?* is
`d(cost)/d(load)`. *What does re-tapping this transformer do to the
voltage profile?* is `d(V)/d(tap)`. *Which line rating is worth
raising?* is a shadow price. Classical power-system tools answer these
with hand-derived sensitivity factors — generation shift factors, loss
factors, the reduced Jacobian — each derived, coded and validated
separately, and each able to drift out of step with the model it came
from.

Here the model is differentiable, so those factors are the derivatives
themselves, taken through the implicit function theorem at the
converged point. They are exact by construction, and they cannot
disagree with the model.

## What is modelled

| Unit | Model |
|---|---|
| **Bus** | complex voltage `V = Vm∠Va`, fixed shunt `Ysh = (Gs+jBs)/base`, kind slack / PV / PQ |
| **Branch** | one model for line, transformer and phase shifter: series `ys = 1/(r+jx)`, total charging `jb` halved per end, complex tap `t = τe^{jθ}` at the *from* end |
| **Generator** | injects `Pg+jQg`, boxed by `[Pmin,Pmax]` and `[Qmin,Qmax]`, cost `c₂P² + c₁P + c₀` $/h |
| **Load** | constant-power withdrawal `Pd+jQd` |

The branch block is

```
Yff = (ys + jb/2)/τ²     Yft = -ys/conj(t)
Ytf = -ys/t              Ytt =  ys + jb/2
```

`τ=1, θ=0` is a line; `b=0, τ≠1` a tap-changing transformer; `θ≠0` a
phase shifter. One unit, three devices, and no place for their
conventions to drift apart.

The only *equations* are nodal power balance — two real rows per bus —
plus one angle-reference row. Voltage limits, generator boxes, thermal
ratings and angle-difference limits are **bounds**, and live with the
optimiser that can act on them, not with the physics.

## How things are solved

**Power flow** — Newton-Raphson as an `optimistix` root find. The
`2n_bus + 1` physics rows are closed with `2n_gen − 1` setpoint rows
(the classical bus-type specification, written as equations rather than
by eliminating variables), giving a square system. Gradients come from
the implicit function theorem at the converged point, so a gradient
costs one linear solve however many Newton steps the forward pass took.

**AC-OPF** — a nonconvex NLP, solved by a primal-dual interior-point
method written in JAX (`difflow_power.ipm`), because there is no IPOPT
in JAX and calling out to one would end the differentiability that is
the point. Three things make it converge on a real case: inertia
correction on an equilibrated KKT matrix, fraction-to-boundary step
capping, and an ℓ1 merit line search. The barrier parameter follows
IPOPT's monotone schedule, judged on the subproblem rather than on
complementarity — tying it to `s·z` deadlocks on a degenerate problem.

**DC-OPF, PTDF, LODF** — the linearisation markets clear on, running
through the *same* interior-point solver (a QP is an NLP with a
constant Hessian), so a DC price and an AC price are directly
comparable.

**Radial feeders** — the backward/forward sweep, which is genuinely a
sequential-modular flowsheet solve: units in a topological schedule
with the voltage profile as the tear. It converges linearly (about five
passes per decade on the example feeder) rather than quadratically, and
is still much less work than Newton because a pass is O(n) with no
Jacobian formed. Exact, and the right method where Newton's decoupling
assumption fails.

## Validated against MATPOWER

Not a self-consistency check. A power-flow tool with the phase-shift
sign backwards or the charging susceptance halved twice converges
beautifully to the wrong numbers, so every benchmark result is asserted
against MATPOWER's published answer for the same case file:

| check | reference |
|---|---|
| `case9` power flow | `Pg = (71.955, 163, 85)` MW, `Va₂ = 9.6687°`, losses 4.9547 MW |
| `case14` power flow | `Pg₁ = 232.39` MW, losses 13.393 MW, all three taps and the bus-9 shunt exercised |
| `case5` AC-OPF | $17551.89/h |
| `case9` AC-OPF | $5296.69/h |
| `case14` AC-OPF | $8081.53/h |
| `case5` DC-OPF | $17479.90/h, LMPs (16.98, 26.38, 30.00, 39.94, 10.00) $/MWh |

Two further cross-checks have no MATPOWER counterpart and are worth
more: the LMPs read off the equality multipliers agree with
`jax.grad` of the optimal cost through the KKT system to ~1e-12 $/MWh,
and
the backward/forward sweep agrees with Newton to 1e-12 on every bus of
a feeder — two entirely different algorithms on the same equations.

## Layout

| module | contents |
|---|---|
| `physics.py` | branch admittances, per-unit, cost curves |
| `network.py` | `PowerNetwork` and its components, validation |
| `residuals.py` | the single definition of the equation set |
| `powerflow.py` | the setpoint closure and Newton |
| `ipm.py` | the JAX primal-dual interior-point NLP solver |
| `opf.py` | AC-OPF assembly, prices, sensitivities |
| `dc.py` | DC model, DC-OPF, PTDF, LODF, contingencies |
| `units/`, `flowsheet.py` | difflow unit operations and the feeder sweep |
| `verify.py` | residuals and limit violations, in engineering units |
| `sensitivity.py` | shift factors, loss factors, stability margin |
| `estimation.py` | state estimation over `difflow.reconciliation` |
| `plotting.py` | network schematics |
| `cases.py` | benchmark cases and a MATPOWER importer |

## Gotchas worth knowing

- **Thermal limits are posed on `|S|²`, never `|S|`.** The modulus has
  curvature going as `1/|S|`, and a lightly loaded branch is exactly
  where an early interior-point iterate sits.
- **The angle-reference row is not bookkeeping.** Without it the
  Jacobian is one rank short for purely structural reasons, and
  everything that inverts it fails on a perfectly well-posed network.
- **A converged power flow is not a feasible operating point.** It
  solves the equations and enforces no limit; `verify.operating_report`
  separates the two, and `PowerFlowResult.violations()` says where.
- **The linear algebra is dense.** Right up to a few hundred buses,
  wrong past a few thousand, where a sparse inertia-revealing `LDLᵀ` is
  what production solvers use.
- **Bus order is insertion order, not sorted.** Sorting numeric bus
  labels as strings would interleave `"10"` between `"1"` and `"2"` and
  make every printed vector unreadable.
