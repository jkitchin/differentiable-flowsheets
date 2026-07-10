# difflow_gas: Gas Transmission Networks as Differentiable Flowsheets

`difflow_gas` models steady-state gas transmission networks as
sequential-modular difflow flowsheets. Its distinguishing feature is
that the sequential decomposition of a meshed network (which units to
solve in which order, and where to tear) is **computed from the
topology**, not derived by hand, so a network with dozens of loops is
as easy to set up as a radial one. Because the flowsheet is built from
pure JAX unit operations and the tear iteration is solved with an
implicit-differentiation-friendly fixed-point solver, everything is
differentiable: gradients of any stream quantity with respect to any
parameter (compressor ratios, control valve drops, slack pressure,
pipe coefficients) are exact, through the converged recycle.

The plugin was extracted from benchmark studies on GasLib-11 and
GasLib-40 (Schmidt et al., *Data* 2017, doi:10.3390/data2040040) that
compared sequential-modular and equation-oriented solutions of the
same networks; the defaults here (damping, chord selection, constraint
formulation advice) encode what those studies measured. The studies
live in the `gaslib` repository alongside their reports.

## Contents

| module            | contents                                                          |
|-------------------|-------------------------------------------------------------------|
| `physics`         | Weymouth/resistor coefficients, friction, conversions, compressor power (plain and NLP-smoothed) |
| `streams`         | the single-pseudo-species signed-mass-flow stream convention      |
| `network`         | `GasNetwork` data model, `decompose()` spanning-tree schedule     |
| `units/`          | pipes, compressors, valves, control valves, topology units       |
| `flowsheet`       | `GasNetworkFlowsheet`: signed-flow Anderson solve + damped differentiable solve |
| `flowsheets`      | `build_network_flowsheet()`: Decomposition -> flowsheet, power helper |
| `verify`          | full equation-oriented residual checks, bounds, compressor margins |

## The model

Steady-state, isothermal, benchmark-standard physics:

- **Pipes** (and **resistors**): squared-pressure Weymouth law
  `p_from^2 - p_to^2 = beta q |q|` with `q` in kg/s (signed), `p` in
  Pa, and `beta` from `physics.weymouth_beta` (Nikuradse friction,
  constant compressibility) or `physics.resistor_xi`.
- **Compressor stations**: `p_out = ratio * p_in` with the ratio a
  differentiable decision parameter; adiabatic shaft power evaluated
  from the solved streams.
- **Valves (open)** and **short pipes**: pressure equality, flow free.
- **Control valves**: `p_out = p_in - dp` with the drop `dp` a
  differentiable decision parameter (the `|dp| <= dp_max` data bound
  belongs to the optimization layer).
- **Nominations**: every boundary flow fixed (a balanced scenario);
  one **slack node** supplies the pressure level. This matches the
  GasLib validation-of-nominations setting; freeing boundary flows is
  an optimization-layer concern.

Flows are **signed**: negative flow means flow against the arc's
reference direction, which is routine in meshed networks. This is why
`GasNetworkFlowsheet.solve()` passes `clip_negative_flows=False` to
the difflow tear solvers (difflow issue #164).

## The decomposition

`decompose(network, root)` computes the complete sequential schedule:

1. **Spanning tree.** Arcs whose pressure relation is not an
   invertible flow law (compressors, valves, control valves, short
   pipes) are forced in-tree. Among pipes and resistors, a minimum
   spanning tree on `beta` keeps the least resistive arcs, so the
   **most resistive arc of each independent loop becomes the chord**
   (the tear). That choice is deliberate: the tear-map slope of a
   chord is roughly `-sum(beta_e |q_e|) / (beta_c |q_c|)` over its
   loop's tree arcs, so resistive chords keep the spectral radius
   small.
2. **Flows.** With nominations fixed, the chord flows are the only
   unknowns (one per loop, `cycle rank = arcs - nodes + 1`). Given
   them, every tree-arc flow follows from leaf-to-root mass balances,
   affine with signs +-1 (`BalanceSpec`).
3. **Pressures.** Root-to-leaf propagation from the slack node: p^2
   drop across pipes/resistors, ratio across compressors, equality
   across valves/short pipes, parametric drop across control valves.
4. **Tear update.** Each chord recomputes its flow from its end
   pressures, `q = sign(dp2) sqrt(|dp2| / beta)`.

Every independent loop must therefore contain at least one pipe or
resistor; a loop of only compressors/valves has no pressure-driven
element to tear and is rejected with a clear error.

`build_network_flowsheet` translates the schedule mechanically into
difflow units (about `2 * nodes + loops` units), with neutral
zero-flow tear guesses attached.

## Solving and convergence

The tear map of a meshed gas network typically has **real negative
eigenvalues** (a loop flow guessed too high overshoots the opposing
pressure difference and the update overcorrects), often with spectral
radius above 1, so plain successive substitution diverges. Two solvers
are provided:

- `fs.solve(...)`: difflow's Anderson-accelerated iteration (robust,
  eager, not traceable by JAX). Signed flows preserved.
- `fs.solve_differentiable(alpha=0.3, ...)`: optimistix fixed-point
  iteration on the damped map `x + alpha (g(x) - x)`, which has the
  same fixed point but contracts; optimistix differentiates the
  converged solution via the implicit function theorem, so gradients
  are exact regardless of `alpha` or iteration count, and the whole
  solve can be `jax.jit`-compiled (GasLib-40: ~90 us per compiled
  solve, 62 iterations at alpha = 0.3).

Choosing `alpha`: for eigenvalues in `[-m, 0)` the iteration contracts
for `alpha < 2/(1 + m)`; the optimal scalar damping is about
`2/(2 + m)` (GasLib-40 measured m ~ 3.1, so alpha ~ 0.3-0.4). Measure
`m` with a finite-difference Jacobian of the tear map at a solved
state if iteration counts matter; the default 0.3 is safe up to
m ~ 5.

Convergence tolerances are ABSOLUTE on the packed tear vector
`[flow (kg/s), T (K), P (Pa)]`; because pressures are order 10^6 Pa, a
tolerance of 1e-6 is already near the floating-point floor and 1e-10
may never trigger. Verify convergence by residuals
(`verify.residual_report`), not by iteration counts.

## Verification

A sequential solve satisfies most equations by construction; the
meaningful check evaluates ALL equation-oriented residuals on the
solved state, exactly as a simultaneous NLP would pose them:

```python
rep = difflow_gas.residual_report(streams, network, dec,
                                  cv_drops_bar={"cv_1": 2.0})
assert rep.ok   # balances ~1e-9 kg/s, pipe laws ~1e-9 bar^2, ...
```

`residuals_from_values(p_bar, q, network)` accepts raw values, so the
same checker validates solutions from any other method (Pyomo,
equation-oriented interior-point solvers) for cross-method studies.

## Optimization guidance

Two hard-won lessons from the GasLib studies, encoded here as advice
because they will bite any reduced-space optimization over this
simulator:

1. **Pose pressure constraints in squared pressure.** The network's
   response to controls is nearly linear in p^2 (Weymouth drops are
   additive in p^2 and flows barely move), while in p the sqrt makes
   low-pressure constraints violently nonlinear near their bounds. On
   GasLib-40, SLSQP in p-space overshot into the region where the
   units' pressure floor zeroes all gradients and collapsed to a false
   optimum; in p^2-space the same problem is almost a QP and even
   finite-difference gradients found the optimum to 1e-12.
2. **Mind the pressure floor.** Unit operations floor squared
   pressures at `MIN_P_SQUARED` ((0.5 bar)^2) so tear iterations
   survive unphysical transients, but gradients vanish in the floored
   region. Squared-pressure constraints keep optimizers out of it;
   starting from a feasible operating point helps too.

For power minimization, `total_compressor_power_w` uses the smoothed
`|q|` (`physics.smoothed_power_w`, `eps = 1e-4 kg/s`) so the
reduced-space objective is identical to the standard equation-oriented
NLP objective and optima can be compared at tight tolerance.

## Example

```python
import jax
import difflow_gas as dg

net = dg.GasNetwork(
    arcs={
        "p1":  ("src", "a", "pipe"),
        "cs1": ("a", "b", "compressor"),
        "p2":  ("b", "c", "pipe"),
        "p3":  ("b", "d", "pipe"),
        "p4":  ("c", "d", "pipe"),          # closes a loop: the tear
    },
    beta={aid: dg.weymouth_beta(L, 0.6, 1e-4)
          for aid, L in [("p1", 20e3), ("p2", 40e3), ("p3", 60e3),
                         ("p4", 80e3)]},
    supply_kg_s={"src": 120.0, "c": -50.0, "d": -70.0},
)

fs, dec = dg.build_network_flowsheet(
    net, root="src", p_slack_pa=60e5, ratios={"cs1": 1.3},
)
streams = fs.solve(tol=1e-8)                       # Anderson
assert dg.residual_report(streams, net, dec).ok

# exact gradient of shaft power w.r.t. the ratio, through the tear
obj = fs.make_objective_fn(
    lambda s: dg.total_compressor_power_w(s, dec, net.gas_temp_k))
dW_dr = jax.grad(obj)({"cs_cs1.ratio": 1.3})
```

## Scope and roadmap

Implemented: pipes, resistors, compressor stations (fixed ratio),
open valves, control valves (parametric drop), short pipes; fixed
balanced nominations with one slack node; isothermal steady state.

Not yet implemented, in rough order of need:

- parallel arcs between one node pair (requires a multigraph spanning
  tree and composite chord updates),
- closed/switchable valves and compressor bypass (discrete states;
  currently a topology edit),
- free boundary flows / multiple slack nodes (pressure-specified
  entries),
- elevation terms in the pipe law, richer friction/compressibility,
- transients (difflow's `dynamic` machinery is the natural home).

## Testing

`pytest tests/gas/ -q` covers physics regression values, every unit
operation (including direction inverses and differentiability),
network validation errors, decomposition invariants (mass closure of
the affine schedule for arbitrary tears, chord selection,
determinism), end-to-end solves against closed-form loop splits,
signed-flow (negative tear) convergence, root-independence of the
solution, AD-vs-finite-difference gradients, jit compilation, and
plugin registration.
