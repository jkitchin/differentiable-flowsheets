# Electrical Grid Unit Operations

This document provides comprehensive documentation for the `difflow_power` plugin, which models steady-state electrical transmission and distribution networks as differentiable flowsheets and solves AC optimal power flow problems.

---

(power-overview)=
## Overview

The `difflow_power` plugin provides:

- **Physics**: the pi-model branch that serves as line, transformer and phase shifter alike; per-unit conversions; polynomial generator cost curves
- **A parser-agnostic network model** (`PowerNetwork`) covering buses, branches, generators and loads, with a MATPOWER case-struct importer
- **One JAX-traceable definition of the equation set** (`power_flow_residuals`), consumed unchanged by the power flow, the OPF, the state estimator and the verifier
- **`solve_power_flow`**: Newton-Raphson via `optimistix`, with implicit-function-theorem gradients through the converged solution
- **`solve_acopf`**: the full nonconvex AC optimal power flow, solved by a primal-dual interior-point method written in JAX, with locational marginal prices from the multipliers and exact sensitivities from the KKT system
- **`solve_dcopf`, `ptdf`, `lodf`**: the linearised model markets clear on, plus contingency screening
- **`RadialFeederFlowsheet`**: the backward/forward sweep, the sequential-modular method distribution feeders actually use
- **Unit operations** that compose into a `difflow.Flowsheet`
- **State estimation** over `difflow.reconciliation`, which is the same computation as chemical data reconciliation
- **Verification** against the full equation set and every operating limit

All operations are fully differentiable using JAX: gradients of any solved quantity with respect to load, generator setpoints, line impedances, transformer taps or fuel prices are exact, through the converged solve.

Every benchmark result is asserted against MATPOWER's published answer for the same case file; see [Validation](power-validation) below.

---

(power-installation)=
## Installation

The power plugin is included as an optional dependency:

```bash
pip install difflow[power]
```

Or install with all extras:

```bash
pip install difflow[all]
```

---

(power-conventions)=
## Stream and unit conventions

Electrical streams use **two** pseudo-species, `"P"` and `"Q"`, whose flows are signed real and reactive power in per unit. The remaining two stream slots carry the complex voltage:

| stream key | electrical quantity |
|---|---|
| `F_P` | real power flow (pu), signed |
| `F_Q` | reactive power flow (pu), signed |
| `P` | **voltage magnitude** (pu) |
| `T` | **voltage angle** (radians) |

The `P` slot carrying voltage is not a pun. In a flowsheet that slot is the potential that drives flow through a resistance, and voltage is exactly that — the gas plugin puts pressure there for the same reason. The angle has no fluid analogue at all, so it takes the remaining slot; nothing downstream interprets it as a temperature.

```python
from difflow_power import power_stream, complex_power, complex_voltage

s = power_stream(p_pu=1.0, q_pu=0.3, vm_pu=1.02, va_rad=0.0)
complex_power(s)      # 1.0 + 0.3j
complex_voltage(s)    # 1.02 + 0j
```

Flows are **signed** along a branch's reference (from → to) direction, so flowsheets built from these streams must be solved with `clip_negative_flows=False`.

Units throughout: voltage and impedance in per unit on `base_mva`, angles in radians (degrees in reports), power in per unit internally and MW/MVAr in reports, cost in $/h with real power in MW.

---

(power-branch-model)=
## The branch model

One model covers transmission lines, transformers and phase shifters. The series admittance `ys = 1/(r + jx)` sits between two halves of the total charging susceptance `b`, and an ideal transformer with complex ratio `t = τ·exp(jθ)` sits at the **from** end:

```
Yff = (ys + j b/2) / τ²      Yft = -ys / conj(t)
Ytf = -ys / t                Ytt =  ys + j b/2
```

so that `[I_from; I_to] = [[Yff, Yft], [Ytf, Ytt]] · [V_from; V_to]`.

- `τ = 1, θ = 0` → a plain pi-model line
- `b = 0, τ ≠ 1` → a tap-changing transformer
- `θ ≠ 0` → a phase-shifting transformer

There is no separate unit operation for the three, and therefore no place for their conventions to drift apart. This is the same convention MATPOWER, PYPOWER, PowerModels and pandapower use, so a network built here compares row for row against those tools.

```python
from difflow_power import branch_admittances

yff, yft, ytf, ytt = branch_admittances(r=0.01, x=0.10, b=0.05)
```

MATPOWER's sentinels are normalised at construction: `tap = 0` means 1.0, `rateA = 0` means unlimited. Nothing downstream has to know that.

---

(power-equations)=
## The equation set

`difflow_power.residuals` is the single definition of a network's equations, and every other module is a consumer of it. The equations are

$$S_i^{\text{sched}} - V_i \overline{(Y_{\text{bus}} V)_i} = 0, \qquad \theta_{\text{slack}} - \theta_{\text{ref}} = 0$$

with $S^{\text{sched}} = (P_g - P_d) + j(Q_g - Q_d)$, split into real and imaginary parts: `2·n_bus` balance rows plus one reference row.

**The reference row is not bookkeeping.** The AC equations are invariant under adding a constant to every bus angle, so without a row pinning one angle the Jacobian is rank `2n - 1`, not `2n`, for purely structural reasons — and every downstream method that inverts it fails on a network that is perfectly well posed physically. (This is the same argument that makes `difflow_gas` carry boundary flows as state.)

**Limits are not equations.** Voltage limits, generator boxes, thermal ratings and angle-difference limits are inequalities and live with the optimiser that can act on them.

The state is packed by `PowerStateLayout`:

```python
from difflow_power import power_state_layout, power_flow_residuals
import jax

layout = power_state_layout(net)
x = layout.pack(vm, va, pg, qg)
r = power_flow_residuals(x, net, layout)
A = jax.jacobian(power_flow_residuals)(x, net, layout)   # constraint Jacobian
```

Optional blocks put demand, transformer taps, phase shifts or switched shunts into the state rather than treating them as parameters — which is how the same equation set serves a power flow (where they are known), an OPF (where they are decisions) and a state estimator (where they are unknowns).

---

(power-powerflow)=
## Power flow

A power flow closes the underdetermined system with **setpoints**; an OPF closes it with **cost**. The classical bus-type specification supplies `2·n_gen − 1` extra equations:

| bus / unit | what is specified |
|---|---|
| slack bus | voltage magnitude (and angle, from the reference row); its generators' MW is whatever balances the system |
| PV bus | voltage magnitude, and each generator's real power |
| PQ bus | nothing — but a generator on one has both P and Q fixed |
| several units on one bus | vars shared in proportion to reactive capability; MW likewise at the slack |

```python
import difflow_power as dp

net = dp.cases.case9()
res = dp.solve_power_flow(net)

res.converged          # True
res.pg_mw              # {'g1': 71.955, 'g2': 163.0, 'g3': 85.0}
res.vm, res.va_degrees
res.losses_mw          # 4.9547
res.branch_loading     # fraction of rating, per rated branch
res.violations()       # limits this operating point breaks
```

Newton-Raphson via `optimistix`, so gradients come from the implicit function theorem at the converged point rather than from unrolling the iteration: a gradient costs one linear solve however many Newton steps the forward pass took, and does not depend on the initial guess.

A converged power flow is **not** a feasible operating point. It solves the equations and enforces no limit, so it will happily return a generator past its var capability and a line at 140% of rating. `violations()` and `verify.operating_report` say where.

---

(power-acopf)=
## AC optimal power flow

$$\min_{V,\theta,P_g,Q_g} \sum_k c_k(P_{g,k})$$

subject to the power flow equations, `Vmin ≤ |V| ≤ Vmax`, the generator boxes, and `|S_f|² ≤ S̄²` at **both** ends of every rated branch.

```python
opf = dp.solve_acopf(net)

opf.cost           # 5296.69 $/h
opf.pg_mw          # the optimal dispatch
opf.lmp_mw         # locational marginal prices, $/MWh
opf.lmp_mvar       # reactive prices -- small, but not zero
opf.binding()      # binding constraints and their shadow prices
```

### Why the squared thermal limit

`|S| ≤ rate` and `|S|² ≤ rate²` describe the same set for a non-negative rating, but `|S|` has curvature going as `1/|S|`, and a lightly loaded branch is exactly where an early interior-point iterate sits. Both ends are limited because a lossy branch carries more at its sending end.

### Prices

The multiplier on a bus's real-power balance *is* its locational marginal price. With the balance written as `(Pg − Pd) − Pinj(V) = 0`, adding a MW at bus *i* perturbs row *i* by `−1`, so

```
LMP_i = -lambda_i / base_mva      $/MWh
```

At an uncongested solution every LMP equals the marginal cost of the marginal unit plus a small loss component. Where a rating binds they separate, and the spread is the congestion rent — the whole reason to run an AC-OPF rather than an economic dispatch.

`ACOPFResult.check_prices()` verifies the multipliers against `jax.grad` of the optimal cost with respect to load, computed independently through the KKT system. On `case9` the two agree to around 1e-12 $/MWh, which is solver precision on a price of $24.

### The solver

`difflow_power.ipm` is a primal-dual interior-point method written in JAX. There is no IPOPT in JAX, and calling out to one would end the differentiability that is the point of this framework. Slacks turn the inequalities into `h(x) + s = 0` with a log barrier, and each iteration is a Newton step on the perturbed KKT conditions, condensed to

$$\begin{bmatrix} W + J_h^T \Sigma J_h & J_g^T \\ J_g & 0 \end{bmatrix} \begin{bmatrix} dx \\ d\lambda \end{bmatrix} = \dots, \qquad \Sigma = S^{-1} Z$$

with `W` the exact Lagrangian Hessian from `jax.hessian`. Three things make it converge on a real nonconvex case:

- **Inertia correction.** At a minimum the KKT matrix has exactly `m_eq` negative eigenvalues; anywhere else the step points at a saddle. The count is taken after Ruiz equilibration — inertia is invariant under diagonal congruence, and `Σ = z/s` otherwise spans twelve decades near the solution and swamps any tolerance.
- **Fraction to boundary**, capping the step so the slacks stay interior.
- **An ℓ1 merit line search**, without which the full Newton step overshoots from a flat start on a congested case.

The barrier parameter follows IPOPT's monotone schedule, reduced once the *subproblem* is solved. Tying it to complementarity instead deadlocks on a degenerate problem: a rejected step leaves `s` and `z` unchanged, so `μ` stops moving and the iteration spins.

### Differentiating the optimum

The iteration is not itself differentiated — deliberately. At the solution the KKT system holds, so `differentiable_solution` re-solves it with `optimistix`, converging in one step from the converged point and carrying implicit-function-theorem gradients.

```python
opf.solution_sensitivity()   # d(state)/d(load)
opf.price_sensitivity()      # d(cost)/d(load), an independent LMP
opf.check_prices()           # the two must agree
```

The gradient is of the *barrier* solution at the final `μ`, which differs from the exact optimum by `O(μ)` — below any modelling error at the default tolerance, but the reason driving `μ` down matters for a sensitivity even when the primal answer already looks converged.

---

(power-dc)=
## DC model, PTDF and LODF

Three assumptions — negligible resistance, flat voltages, small angles — turn the AC equations into a linear model. It is a severe approximation and an indispensable one: it is what wholesale markets clear on, and what makes contingency screening over thousands of outages tractable.

```python
dc = dp.solve_dcopf(net)         # a convex QP, same interior-point solver
H  = dp.ptdf(net)                # (n_branch, n_bus) shift factors
L  = dp.lodf(net)                # (n_branch, n_branch) outage factors
after = dp.contingency_flows(net, base_flows)
```

`PTDF[l, b]` is the MW on branch `l` per MW injected at bus `b` and withdrawn at the reference. `LODF[l, k]` is the fraction of branch `k`'s pre-outage flow that lands on `l` when `k` trips, so every single-branch contingency is one matrix product rather than `n_branch` power flows.

A branch whose outage would **island** the network has an undefined column, returned as `nan` rather than as a large finite number — a screening loop must not mistake a disconnection for a manageable overload.

DC-OPF runs through the same interior-point solver as the AC problem (a QP is an NLP with a constant Hessian), so a DC price and an AC price are directly comparable, and their difference is a clean measure of what the linearisation costs. DC cost is systematically optimistic: the model has no losses, so nobody generates them.

---

(power-flowsheet)=
## Radial feeders: the sequential-modular sweep

Newton on the full system is the right method for a meshed transmission network and a poor one for a distribution feeder, which has a high R/X ratio (invalidating the decoupling Newton relies on) and is radial (making a far cheaper method available).

That method is the backward/forward sweep, and it is genuinely a sequential-modular flowsheet solve: units in a topological schedule with **the voltage profile** as the tear.

1. **Backward.** At each bus, KCL gives the current into its parent branch as what its children and shunt did not take. One pass, no matrix.
2. **Forward.** The branch relation inverts for the child voltage given the parent's. One pass from the fixed slack voltage.

```python
fs = dp.RadialFeederFlowsheet(dp.cases.radial_feeder())
streams = fs.solve()
streams["bus_n4"]["P"]      # voltage magnitude at bus n4
```

Iterating is a contraction, so it converges **linearly** — about 0.4 per pass on the example feeder, or five passes per decade: eight to 1e-4, eighteen to 1e-8, twenty-eight to floating-point. That is many more iterations than Newton's five, and still much less work, because a pass is O(n) with no Jacobian formed, factorised or differentiated.

It is exact, not an approximation: charging, bus shunts and taps all go through the same 2×2 admittance block the equation-oriented path uses. The sweep and Newton agree to 1e-12 on every bus.

`build_ladder_flowsheet` assembles a genuine `difflow.Flowsheet` for a non-branching feeder out of the unit operations, with the substation infeed as the single tear:

```
infeed (tear) -> SlackSource -> [SeriesBranch -> LoadDraw] x N
              -> LadderClose(end, infeed) -> infeed_next
recycle: infeed_next -> infeed
```

`LadderClose` is what makes the fixed point `leftover = 0` rather than `leftover = infeed`; the correction `infeed_next = infeed − leftover` is close to an exact Newton step and converges in about four Anderson iterations.

---

(power-units)=
## Unit operations

| unit | what it does |
|---|---|
| `SeriesBranch` | line/transformer, forward: to-end voltage and power from the from end |
| `BranchDrop` | voltage propagation at a known current (linear) |
| `BranchFlow` | equation-oriented: both end powers from both end voltages |
| `Transformer` | `SeriesBranch` that refuses to be a line |
| `SlackSource` | pin a feed to a regulated voltage |
| `LoadDraw` | constant-power demand |
| `ShuntDraw` | fixed shunt (capacitor bank or reactor) |
| `GeneratorInject` | injection with a polynomial cost curve |
| `BusNode` | sum powers, voltage from the first inlet |
| `PowerSplit` | divide a bus's outgoing power (a tear variable, not a physical parameter) |
| `LadderClose` | correct a feeder's infeed by the residual at its open end |

`SeriesBranch` inverts the branch relation and `BranchFlow` evaluates it directly, so they agree exactly — which is what makes either usable as a check on the other.

Constant-power load is where a power flow's difficulty actually lives. Constant *impedance* would make the whole system linear in `V`; constant *current*, linear in the phasor. Constant power gives `S = V·conj(YV)`, and that is the nonlinearity Newton spends its iterations on.

---

(power-sensitivity)=
## Sensitivities

Power systems have a long tradition of hand-derived sensitivity factors. Each is a derivative of a solved state, and each is one `jax.jacobian` call here — not a reimplementation of the classical formula, but the derivative itself, so it cannot drift out of step with the model.

```python
dp.demand_sensitivity(net)        # d(state)/d(load) -- includes voltages
dp.branch_flow_sensitivity(net)   # AC injection shift factors
dp.loss_sensitivity(net)          # marginal loss factors
dp.parameter_sensitivity(net, "tap")
dp.voltage_stability_margin(x, net)
```

These differentiate a **power flow**, where setpoints hold and the slack absorbs. The OPF counterparts, where the dispatch re-optimises, are on `ACOPFResult` and answer a different question.

`voltage_stability_margin` is the smallest singular value of the power flow Jacobian — the classical proximity-to-collapse index, which goes to zero at the nose of the P-V curve, where the solution ceases to exist rather than merely becoming poor.

---

(power-estimation)=
## State estimation

Power system state estimation and chemical process data reconciliation are the same computation: a weighted least-squares distance from noisy measurements, minimised subject to a model's equations, with bad data found by looking for a residual too large to be noise. So `difflow_power.estimation` is a thin layer over `difflow.reconciliation`, not a reimplementation.

```python
layout = dp.power_state_layout(net, demand_buses=net.bus_ids)
sigma  = dp.measurement_sigma(layout, overrides={"va_7": 0.001})   # a PMU
y      = dp.perturb(x_true, sigma, jax.random.PRNGKey(0))
est    = dp.estimate_state(net, y, sigma, layout)
```

Defaults reflect control-centre practice: voltage transducers good to a few tenths of a percent, generator output metered well, load the least reliable number in the system, and **angles unmeasured** — without PMUs they are, and inferring them is the estimator's job.

Demand belongs in the *state*, not the network: it is measured badly and is exactly what an estimator corrects, and measured loads never balance generation, so a `PowerNetwork` built from them would describe a state that cannot exist. Making them balance is precisely what the estimate does.

An unobservable network gives a singular normal-equation system; `reconcile` checks the structure first and raises rather than returning a plausible-looking answer, so placing one more meter is a diagnosable fix.

---

(power-validation)=
## Validation

A self-consistent power flow tool with the phase-shift sign backwards, or the charging susceptance halved twice, converges beautifully to the wrong numbers. So every benchmark result is asserted against MATPOWER's published answer for the same case file.

| check | reference |
|---|---|
| `case9` power flow | `Pg = (71.955, 163, 85)` MW, `Qg = (24.07, 14.46, −3.65)` MVAr, `Va₂ = 9.6687°`, losses 4.9547 MW |
| `case14` power flow | `Pg₁ = 232.39` MW, losses 13.393 MW |
| `case5` AC-OPF | $17551.89/h |
| `case9` AC-OPF | $5296.69/h, `Pg = (89.80, 134.32, 94.19)` MW |
| `case14` AC-OPF | $8081.53/h |
| `case5` DC-OPF | $17479.90/h, LMPs (16.98, 26.38, 30.00, 39.94, 10.00) $/MWh |

Two cross-checks have no MATPOWER counterpart and are worth more, because they compare two independent computations of the same quantity inside this package:

- LMPs read off the equality multipliers agree with `jax.grad` of the optimal cost through the KKT system to around 1e-12 $/MWh.
- The backward/forward sweep agrees with Newton to 1e-12 on every bus of a feeder — two entirely different algorithms on the same equations.

The equation set itself is checked in `tests/power/test_residuals.py` against an independent restatement in **polar** form, written from the textbook while the implementation works in complex rectangular form. Two algebraic routes to the same numbers is what makes the comparison worth anything.

---

(power-cases)=
## Benchmark cases

| case | description |
|---|---|
| `case3` | a hand-built 3-bus loop: the smallest network where power divides between two paths |
| `case5` | PJM 5-bus: linear costs and two binding ratings, the standard congestion/LMP demo |
| `case9` | WSCC 9-bus, 3 machines: the classic power flow and OPF benchmark |
| `case14` | IEEE 14-bus: three tap-changing transformers and a shunt capacitor |
| `radial_feeder` | a 7-bus 12.47 kV distribution feeder, radial, high R/X |

`from_matpower` imports any MATPOWER or PYPOWER case struct, handling the format's sentinels and unit conventions. Piecewise-linear cost curves (gencost model 1) are refused with a reason rather than silently fitted to a polynomial, which would misprice the dispatch.

---

(power-gotchas)=
## Gotchas

- **A converged power flow is not a feasible operating point.** Use `verify.operating_report` to separate "the equations hold" from "the limits hold".
- **The linear algebra is dense.** Right up to a few hundred buses, wrong past a few thousand, where a sparse inertia-revealing `LDLᵀ` is what production solvers use.
- **The feeder sweep unrolls into the traced graph**, since its schedule is a Python loop over a static topology. Fine for a few hundred buses; past a few thousand the trace itself becomes the cost and the equation-oriented Newton is the better tool again.
- **Bus order is insertion order, not sorted.** Sorting numeric labels as strings would interleave `"10"` between `"1"` and `"2"`.
- **Solve with `clip_negative_flows=False`** in any hand-built flowsheet: power flows are signed.
- **Loss factors are negative beside a scheduled generator.** Not a sign error: load added next to a unit exporting a fixed schedule is served locally instead of wheeled, so total transport falls.
