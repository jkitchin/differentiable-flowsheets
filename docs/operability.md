# Operability and Controllability Screening

`difflow.operability` asks a question a steady-state flowsheet can answer but is
almost never asked of it: **do the manipulated variables have enough independent
influence on the controlled variables to hold them against the disturbances the
plant will actually see?**

Everything follows from two Jacobians of a *converged* flowsheet,

$$G = \frac{\partial y}{\partial u}, \qquad G_d = \frac{\partial y}{\partial d}$$

which are one `jax.jacobian` call each. Obtained the usual way — finite
differences through a sequential-modular simulator — they cost `2n` re-solves
and carry the simulator's convergence tolerance as noise. That expense is why
controllability screening is normally done on a linear model fitted *separately*
from the design model, after the design is frozen. With AD the gains are exact,
cost a constant multiple of one model evaluation, and are a differentiable
function of the design variables, so the screen can live **inside** the design
loop rather than after it.

## Table of Contents

1. [Quick start](#quick-start)
2. [Scaling comes first](#scaling-comes-first)
3. [The measures](#the-measures)
4. [Reading a screen](#reading-a-screen)
5. [On a real flowsheet](#on-a-real-flowsheet)
6. [Inside a design loop](#inside-a-design-loop)
7. [Non-square and singular plants](#non-square-and-singular-plants)
8. [What this does not tell you](#what-this-does-not-tell-you)
9. [API summary](#api-summary)
10. [References](#references)

---

## Quick start

```python
import jax.numpy as jnp
from difflow.operability import Scaling, screen

# The LV distillation column of Skogestad and Postlethwaite, Example 3.11.
def column(u, d):
    L, V = u
    return jnp.array([0.878 * L - 0.864 * V + 0.394 * d[0],
                      1.082 * L - 1.096 * V + 0.586 * d[0]])

sc = Scaling(u_span=[1.0, 1.0],        # available move in reflux and boilup
             y_span=[0.01, 0.01],      # 1% composition is the largest
                                       #   acceptable control error
             d_span=[0.2],             # the feed wanders by 20%
             note="LV column, 1% composition tolerance")

rep = screen(column, jnp.zeros(2), jnp.zeros(1), scaling=sc,
             u_names=["L", "V"], y_names=["x_D", "x_B"], d_names=["feed"])
print(rep.summary())
```

```text
operability screen: 2 outputs, 2 inputs, 1 disturbances
  scaling: LV column, 1% composition tolerance
  sigma_min(G) = 1.391   cond(G) = 141.7   rank(G) = 2/2
  RGA number  = 136.3   (pairing x_D-L, x_B-V)
  RGA:
                              L          V
    x_D                   35.07     -34.07
    x_B                  -34.07      35.07
  scaled disturbance gain (|.| > 1 needs control):
                           feed
    x_D                    7.88
    x_B                   11.72
  per disturbance: feed: gamma_d=11.7, max input move=0.643
  4 findings (0 error, 4 warning)
    [warning] directional: plant: condition number 142 > 10: ...
    [warning] rga_interaction: plant: largest paired relative gain is 35.1 > 5: ...
    [warning] disturbance_gain: feed: scaled disturbance gain 11.7 > 1: ...
    [warning] disturbance_direction: feed: disturbance condition number 11.7 > 10: ...
```

`lambda_11 = 35.1` and `cond(G) = 141.7` are the numbers quoted for this column
in the reference; `tests/test_operability.py` pins both.

## Scaling comes first

Every measure in this package except the RGA is a statement about *magnitudes*,
and a magnitude in mixed engineering units is not a magnitude at all. A gain of
`1e5 K/(mol/s)` and a gain of `0.02 mol/mol` cannot be compared, ranked, or
handed to an SVD — but `jax.jacobian` returns exactly that matrix, and the SVD
will happily return a number for it. **This is where these metrics are misused,
and it is why `screen` has no default scaling and will not run without one.**

`Scaling` takes three engineering judgements, in the convention of Skogestad and
Postlethwaite section 1.4:

| Span | Meaning | Common mistake |
|---|---|---|
| `u_span` | The largest change in each manipulated variable that is actually *available* — valve shut to open, usable duty turndown | Using the full design range when the plant sits at one end of it |
| `y_span` | The largest *acceptable control error* in each controlled variable | Using the operating value, or the measurement noise, instead of the deviation at which the loop has failed |
| `d_span` | The largest expected excursion of each disturbance | Using the nominal value rather than the excursion |

With `Du = diag(u_span)`, `De = diag(y_span)` and `Dd = diag(d_span)`,

$$\tilde G = D_e^{-1} G D_u, \qquad \tilde G_d = D_e^{-1} G_d D_d$$

and every entry of both is dimensionless with the *same* meaning: **how many
allowable control errors of output _i_ does a full move of input (or
disturbance) _j_ produce.** That is what makes `1` the threshold every rule of
thumb is stated against.

Three constructors:

```python
Scaling(u_span=[...], y_span=[...], d_span=[...])       # the honest one
Scaling.from_bounds(u_lb, u_ub, y_tol, d_lb, d_ub)      # from operating bounds
Scaling.from_block(planning_block, y_tol=[...])         # reuse a planning Block
Scaling.unscaled(n_u, n_y)                              # the recorded refusal
```

`Scaling.unscaled` is the explicit opt-out — use it only when the variables are
*already* dimensionless. It stamps the report `scaled=False`, prints a caveat
banner, adds an `unscaled` finding, and suppresses every threshold comparison
that would otherwise be meaningless. Calling a metric with no `Scaling` at all
raises an `OperabilityWarning`; `assume_scaled=True` is the silent, deliberate
way to say the matrix is already dimensionless.

## The measures

### Relative gain array

`rga(G)` returns `G * pinv(G).T` elementwise. Entry `[i, j]` is the ratio of the
open-loop gain from input `j` to output `i` to the gain that remains when every
*other* loop is perfectly controlled:

- **1** — the pairing is unaffected by the other loops.
- **0** — input `j` does nothing for output `i` once the others close.
- **large positive** — the loops fight each other and the pairing is very
  sensitive to model error (`|RGA| > 5` raises `rga_interaction`).
- **negative** — the gain *changes sign* when the other loops close. Pairing
  here gives a system that is unstable with all loops closed, with that loop
  alone, or whenever another loop saturates. Under integral control that is
  structural, not a tuning problem, and it is reported as an **error**.

The RGA is invariant to diagonal input and output scaling, so it is the one
measure here that may be read off a raw AD Jacobian. `negative_pairings`,
`rga_number` (distance from the pairing permutation) and `suggest_pairing`
(greedy, positive relative gains nearest 1) all read from it.

### Singular values

`min_singular_value(G, scaling)` is the smallest output move the inputs can
produce per unit input move, over all directions — the plant's *worst*
direction, not its typical one. In the scaling convention above the threshold
is 1: `sigma_min >= 1` says that in every direction the available inputs can
cover the range that has to be covered. Below 1 there is a direction in which
they cannot, and **no controller design recovers gain the steady state does not
have**.

`condition_number(G, scaling)` is `sigma_max / sigma_min` — how *directional*
the plant is. Above roughly 10 the plant responds strongly to some input
combinations and weakly to others; that is not necessarily hard to control, but
it is hard to control with decentralised loops and it is sensitive to model
error in the weak direction. Unlike the RGA it is **not** scaling-invariant,
which is a reason to state the scaling, not a reason to distrust the measure.

### Disturbances

`disturbance_gain(model, u0, d0, scaling=sc)` is the most directly useful matrix
in the package once scaled: entry `[i, k]` is how many allowable control errors
of output `i` a full-size excursion of disturbance `k` produces. Below 1 is a
disturbance the process absorbs on its own; above 1 is one control has to
reject.

Two further questions then matter, and neither is visible in the gain alone:

- `required_input_move(G, Gd)` = `pinv(G) @ Gd`. Entry `[j, k]` is the fraction
  of input `j`'s available range needed to cancel a full excursion of
  disturbance `k`. Greater than 1 means the plant **cannot** reject it,
  whatever the controller (`disturbance_infeasible`, an error).
- `disturbance_condition_number(G, Gd)` = `sigma_max(G) * ||pinv(G) y_d||` with
  `y_d` the unit vector along the disturbance's output direction. It measures
  *alignment* and lies between 1 (the disturbance pushes exactly where the plant
  is strongest) and `cond(G)` (exactly where it is weakest). A large `gamma_d`
  on a disturbance whose scaled gain is under 1 is harmless. A large `gamma_d`
  on one whose gain exceeds 1 is the combination that makes a design
  uncontrollable, and it is invisible to either measure by itself.

## Reading a screen

`screen` returns an `OperabilityReport`. Its numeric fields (`G`, `Gd`, `RGA`,
`svals`, `msv`, `cond`, `rga_num`, `rga_pairs`, `rank`, `dist_cond`,
`u_required`) are JAX arrays, and the class is a registered pytree. Its
*interpretation* is computed lazily and needs concrete values:

```python
rep.findings            # list[Finding], errors first
rep.errors, rep.warnings
rep.ok                  # True when nothing was flagged
rep.summary()           # the text above
rep.warn()              # re-emit every finding as an OperabilityWarning
rep.suggested_pairing() # [('x_D', 'L'), ('x_B', 'V')]
```

`Finding` is the same dataclass `difflow.planning.health` uses, and the
reporting style is deliberately the same: each finding names the measured value
*and* the remedy, and nothing raises during a solve.

| Finding | Severity | Trigger |
|---|---|---|
| `non_finite` | error | A non-finite entry in `G`; nothing else means anything |
| `singular` | error | `rank(G) < min(n_y, n_u)` — fewer independent directions than outputs |
| `underactuated` | error | Fewer inputs than outputs |
| `overactuated` | warning | More inputs than outputs; steady-state freedom left over |
| `rga_negative` | error | A paired relative gain is negative |
| `rga_interaction` | warning | Largest paired \|RGA\| > 5 |
| `weak_direction` | warning | Scaled `sigma_min` < 1 |
| `directional` | warning | `cond(G)` > 10 |
| `disturbance_infeasible` | error | Rejecting a disturbance needs more than an input's full range |
| `disturbance_gain` | warning | Scaled \|Gd\| > 1: must be rejected by control, not absorbed |
| `disturbance_direction` | warning | A significant disturbance with `gamma_d` > 10 |
| `unscaled` | warning | The report was built on unit spans |

## On a real flowsheet

`screen` takes any pure JAX callable `fn(u) -> y` or `fn(u, d) -> y`. A difflow
flowsheet qualifies: its flash, recycle and unit solves are implicitly
differentiated, so what comes back is the *converged* steady-state gain, not a
gain through one Newton iteration.

```python
def fn(u, d):
    inlet = make_stream({"Light": d[0], "Heavy": 0.1}, T=320.0, P=101325.0)
    reacted, _ = cstr(inlet, T_spec=u[0])          # inner steady-state solve
    liquid, vapor, _ = flash(reacted, T=u[1], P=101325.0)   # Rachford-Rice
    return jnp.array([get_flows(vapor)["Light"], get_flows(liquid)["Heavy"]])

sc = Scaling(u_span=[10.0, 10.0],     # 10 K of usable swing on each unit
             y_span=[0.2, 0.2],       # 0.2 mol/s of product flow
             d_span=[1.0])            # the feed wanders by 1 mol/s
rep = screen(fn, jnp.array([350.0, 380.0]), jnp.array([10.0]), scaling=sc,
             u_names=["T_reactor", "T_flash"],
             y_names=["light_vapor", "heavy_liquid"], d_names=["feed"])
```

```text
  sigma_min(G) = 0.9895   cond(G) = 9.973   rank(G) = 2/2
  RGA number  = 8.18   (pairing light_vapor-T_reactor, heavy_liquid-T_flash)
  RGA:
                      T_reactor    T_flash
    light_vapor          -1.045      2.045
    heavy_liquid          2.045     -1.045
  4 findings (3 error, 1 warning)
    [error] rga_negative: light_vapor-T_reactor: relative gain -1.05 is negative ...
    [error] rga_negative: heavy_liquid-T_flash: relative gain -1.05 is negative ...
    [error] disturbance_infeasible: feed: rejecting a full-size excursion of 'feed'
            needs 2.57 times the available range of 'T_flash' ...
    [warning] weak_direction: plant: sigma_min of the scaled gain is 0.989 < 1 ...
```

Two structural results, neither of which a steady-state simulation reports on
its own: the obvious pairing is *sign-reversed* under closed loop and must be
swapped, and a 1 mol/s feed excursion cannot be rejected with ±10 K on the two
temperatures. Both are conclusions about the flowsheet, not about a controller.

These gains are verified against central differences through the inner solves in
`tests/test_operability.py::test_gain_matrix_matches_central_differences` — the
AD Jacobian and a perturbation study agree to better than 1e-6 relative.

## Inside a design loop

This is the point of the module. `screen` is a pure function of the design
variables: two `jax.jacobian` calls and a handful of SVDs, with no Python
branching on the numbers. So it is `jit`-, `vmap`- and `grad`-safe, and a
controllability term can go straight into an economic objective:

```python
import jax

def objective(design):
    rep = screen(build_flowsheet(design), u0, d0, scaling=sc)
    penalty = jax.nn.relu(1.0 - rep.msv)          # want sigma_min >= 1
    return -profit(design) + w * penalty ** 2

grad_obj = jax.grad(objective)     # differentiates *through* the SVD
```

and a whole batch of candidate designs can be screened at once, because the
report is a pytree:

```python
reports = jax.jit(jax.vmap(lambda u: screen(fn, u, d0, scaling=sc)))(U)
reports.msv.shape        # (n_candidates,)
```

This makes it possible to ask whether the economically optimal structure is also
a controllable one — the central question of integrated design and control,
normally too expensive to pose.

Two practical cautions:

- Singular values are differentiable only where they are **distinct**, and the
  two families of measures fail differently there. `min_singular_value`,
  `condition_number` and `singular_values` use `compute_uv=False` and stay
  finite: at a crossing `sigma_min` has a kink and AD silently returns one
  arm's slope, so an optimiser stalls rather than diverges. Anything needing
  the singular *vectors* — `pinv`, and therefore `rga`, `required_input_move`
  and `disturbance_condition_number` — carries a `1 / (s_i^2 - s_j^2)` term and
  returns **`nan`** at an exactly repeated singular value. Prefer `msv` as the
  quantity you differentiate. A plant with two exactly equal gain directions is
  contrived; a symmetric test case reaches it.
- `rep.findings`, `rep.summary()` and `rep.ok` need Python branching on the
  values, so they raise inside `jit`/`vmap`. Pull `msv`, `cond`, `RGA` out of
  the trace and interpret them afterwards.

## Non-square and singular plants

Nothing here requires a square `G`; the pseudo-inverse gives the non-square RGA
of Chang and Yu. Exactly one of the two sum rules survives, and which one is
determined by the shape — row sums are `diag(G G+)` and column sums
`diag(G+ G)`:

| Shape | | |
|---|---|---|
| Square, nonsingular | rows sum to 1 | columns sum to 1 |
| Wide (`n_u > n_y`, more inputs) | **rows sum to 1** | columns do not |
| Tall (`n_y > n_u`, more outputs) | rows do not | **columns sum to 1** |

For a wide plant a small column entry means that input carries little of the
job, not that it is a bad pairing. For a tall plant a row summing to well under
1 names an output that no input combination really controls.

A rank-deficient `G` does not raise. It produces an RGA whose rows do not sum to
1 — which is itself the diagnosis — plus a `singular` error finding, and
`min_singular_value` goes to zero and `condition_number` to infinity, which are
the honest answers. The pseudo-inverse uses an explicit relative cutoff
(`RCOND`, `8 * eps` by default, exposed as an `rcond` argument everywhere) so an
exactly singular plant yields finite numbers rather than `nan`.

## What this does not tell you

**Steady state only.** A design that passes this screen can still be undone by
right-half-plane zeros, dead time, actuator dynamics or a bandwidth limit. What
the screen gives is the other half of the implication: a design that *fails* it
cannot be rescued by any controller, which is what makes it worth running first
and worth running early.

**Pairing suggestions are suggestions.** `suggest_pairing` is a greedy rule on
the steady-state RGA and ignores dynamics entirely — the standing limitation of
every steady-state pairing rule.

**One operating point.** `G` and `G_d` are local. A plant whose gains change
sign over the operating envelope needs the screen run at several points; that is
cheap here, and `vmap` is the way to do it.

**Thresholds are conventions.** `MSV_TOL = 1`, `COND_TOL = 10`, `RGA_TOL = 5`
and `GD_TOL = 1` are the usual rules of thumb, exported so they can be read and
argued with. Only the first and last are pinned by the scaling convention; the
middle two are judgement.

## API summary

| Object | Purpose |
|---|---|
| `Scaling` | The three spans that make every measure dimensionless |
| `Scaling.from_bounds`, `.from_block`, `.unscaled` | Constructors; the last is the recorded refusal to scale |
| `Scaling.scale_gain`, `.scale_disturbance`, `.unscale_gain` | Apply and invert the scaling |
| `gain_matrix` | `G = dy/du` by AD, mode chosen by shape |
| `disturbance_gain` | `G_d = dy/dd` by AD |
| `rga` | `G * pinv(G).T`, scaling-invariant |
| `rga_number` | Distance of the RGA from a pairing permutation |
| `negative_pairings` | Pairings whose relative gain is negative |
| `suggest_pairing` | Greedy pairing on positive relative gains near 1 |
| `min_singular_value`, `max_singular_value`, `singular_values` | The plant's worst and best directions |
| `condition_number` | Directionality of the scaled gain |
| `effective_rank`, `pinv` | Rank and pseudo-inverse with an explicit `rcond` |
| `required_input_move` | `pinv(G) @ Gd`: input range needed per disturbance |
| `disturbance_condition_number` | Alignment of each disturbance with the plant's strong direction |
| `screen` | All of the above in one traceable call |
| `OperabilityReport` | The result; pytree of arrays plus `.findings`, `.summary()`, `.ok` |
| `OperabilityWarning` | Raised when a measure is requested in a way that makes it misleading |
| `MSV_TOL`, `COND_TOL`, `RGA_TOL`, `GD_TOL`, `RCOND` | The thresholds, exported to be argued with |

See also `difflow.planning.health.check_delta_health`, which applies the same
reporting pattern to the delta vectors of a planning LP. The two are asking
related questions of the same Jacobians.

## References

- Bristol, E. H. *On a new measure of interaction for multivariable process
  control*. IEEE Trans. Automatic Control **11**(1), 133–134, 1966.
  [doi:10.1109/TAC.1966.1098266](https://doi.org/10.1109/TAC.1966.1098266)
- Morari, M. *Design of resilient processing plants — III. A general framework
  for the assessment of dynamic resilience*. Chem. Eng. Sci. **38**(11),
  1881–1891, 1983.
  [doi:10.1016/0009-2509(83)85044-1](https://doi.org/10.1016/0009-2509(83)85044-1)
  (Part I, which introduces the resilience framing, is Chem. Eng. Sci.
  **37**(2), 245–258, 1982,
  [doi:10.1016/0009-2509(82)80159-0](https://doi.org/10.1016/0009-2509(82)80159-0).)
- Skogestad, S. and Postlethwaite, I. *Multivariable Feedback Control: Analysis
  and Design*, 2nd ed. Wiley, 2005. Sections 1.4, 3.5, 6.10 and chapter 10.
- Chang, J.-W. and Yu, C.-C. *The relative gain for non-square multivariable
  systems*. Chem. Eng. Sci. **45**(5), 1309–1323, 1990.
  [doi:10.1016/0009-2509(90)87123-a](https://doi.org/10.1016/0009-2509(90)87123-a)
- Sakizlis, V., Perkins, J. D. and Pistikopoulos, E. N. *Recent advances in
  optimization-based simultaneous process and control design*. Comput. Chem.
  Eng. **28**(10), 2069–2086, 2004.
  [doi:10.1016/j.compchemeng.2004.03.018](https://doi.org/10.1016/j.compchemeng.2004.03.018)
- Yuan, Z., Chen, B., Sin, G. and Gani, R. *State-of-the-art and progress in the
  optimization-based simultaneous design and control for chemical processes*.
  AIChE Journal **58**(6), 1640–1659, 2012.
  [doi:10.1002/aic.13786](https://doi.org/10.1002/aic.13786)
