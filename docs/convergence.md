# Convergence and Initialization

Lack of convergence is the standard failure mode of a flowsheet simulator. A
recycle loop is a fixed point, a fixed point needs a starting guess, and a bad
guess turns a correct model into a solver that limit-cycles, overshoots into
negative flows, or returns an answer that looks like every other answer and is
off by the tear residual.

This page is the whole story in one place: where the initial guess comes from,
what the three acceleration methods cost, which knobs are tuning and which are
correctness, what changes under `jax.grad`, and what to reach for next when the
loop still will not close.

## Table of Contents

- [When a solve does not converge](#when-a-solve-does-not-converge)
- [Where the initial guess comes from](#where-the-initial-guess-comes-from)
- [Unit-level `initialize()`](#unit-level-initialize)
- [The three acceleration methods](#the-three-acceleration-methods)
- [The traced fallback](#the-traced-fallback)
- [The equation-oriented route](#the-equation-oriented-route)
- [Diagnostics](#diagnostics)
- [A worked diagnosis](#a-worked-diagnosis)
- [Tear selection](#tear-selection)
- [What the plugins add](#what-the-plugins-add)

---

## When a solve does not converge

`Flowsheet.solve()` warns rather than staying quiet
(see [When the Solve Does Not Converge](streams-and-flowsheets.md#when-the-solve-does-not-converge)),
so the first thing to do is read the message it already wrote: it names the tear
streams, the residual reached, the tolerance, the iteration count and the method
that ran. Then work down this list.

1. **Is it actually diverging, or just slow?** Compare
   `fs.last_solve_residual` against `fs.last_solve_tol`. A residual a factor of
   two or three above tolerance after `max_iter` iterations is a loop that needs
   more iterations; a residual of the same order as the flows themselves is a
   loop that is not converging at all. Raise `max_iter` only for the first.

2. **Give it a starting point.** `tear_initial` is the only guess you control
   directly, and a guess taken from a converged neighbouring case (a
   slightly different feed, a slightly different setpoint) is usually worth more
   than any change of method. See
   [Explicit guesses](#explicit-guesses) below.

3. **Change the method.** The default is Anderson. A loop that *oscillates*
   — the residual alternating in sign, flows overshooting and coming back — has
   a tear map with negative eigenvalues, and `acceleration="wegstein"` is the
   one that handles that directly: its acceleration factor is clipped into
   `[-5, 0]`, which damps toward the previous iterate exactly in that case. A
   loop that is merely slow and monotone is what Anderson is for.
   `acceleration="none"` is plain substitution through `optimistix` and is the
   honest baseline: if that diverges, the map itself is not contractive and no
   amount of acceleration will rescue it.

4. **Check `clip_negative_flows`.** If any tear flow is legitimately signed,
   the default clipping is not a safeguard but a bug, and the solve cannot
   converge. This is a correctness switch, not a tuning knob — see
   [Clipping negative flows is a correctness switch](#clipping-negative-flows-is-a-correctness-switch).

5. **Look for a unit that could not run.** A `TearInitializationWarning` during
   the solve says the initial pass over the units failed somewhere and that
   unit's outlets started from the flowsheet default. That is often the real
   cause: the loop is being started from a stream that is nothing like the
   answer. Fix it by giving that unit an [`initialize()`](#unit-level-initialize),
   or by passing `tear_initial` directly.

6. **Scale the default stream.** For a loop carrying trace species, the
   default tear guess of 0.01 mol/s per species is orders of magnitude too
   large. `Flowsheet(species_order, default_flow=1e-6)` moves it.

7. **Switch to the equation-oriented solver.** `fs.solve_eo()` assembles every
   unit's equations into one system and runs Newton on it. Tightly coupled
   loops that take hundreds of tear iterations often take a handful of Newton
   steps. By default it initializes *itself* by running the sequential solver
   first — see [the equation-oriented route](#the-equation-oriented-route).

8. **Make non-convergence loud.** In a script or a test, escalate so a bad
   solve cannot be mistaken for a good one:

   ```python
   streams = fs.solve(on_nonconvergence="raise")
   ```

---

## Where the initial guess comes from

The tear streams are the **destinations** of the recycles registered with
`fs.add_recycle(source, dest)`. A destination is an inlet, so nothing in the
flowsheet computes it before the iteration starts, and it has to be guessed.
`solve()` tries three sources in order.

### Explicit guesses

`tear_initial` takes a dict of `{destination_name: Stream}`. It reaches the
solver unmodified and wins over everything below:

```python
guess = make_stream({"A": 2.4}, 300.0, 1e5)
streams = fs.solve(tear_initial={"tear": guess})
```

The natural sources of a good guess are a converged solve of a nearby case
(continuation by hand), a hand calculation of the loop's material balance, or
the result of a looser solve:

```python
coarse = fs.solve(tol=1e-4, max_iter=200)
fine = fs.solve(tear_initial={"tear": coarse["loop_out"]}, tol=1e-10)
```

### One pass from the feeds

With `use_initialization=True` (the default) and no explicit guess, `solve()`
runs every unit once from the feeds with the recycles unknown, and takes the
value that pass puts on the *source* end of each recycle. One pass serves every
tear, however many recycles there are.

Units run in the order they were added, so ordering the flowsheet so that
material flows forward through the list makes this pass more useful. A unit
whose inlets are not available yet sees the flowsheet default stream.

A unit that **raises** on this pass is not necessarily broken: the pass runs
before any recycle is known, so a unit inside a loop sees 0.01 mol/s arriving,
and a root find or a Rachford-Rice solve can genuinely fail on an almost empty
stream. When that happens `solve()` falls back to the unit's
[`initialize()`](#unit-level-initialize). When there is no `initialize()` either,
the unit's outlets start from the flowsheet default and a
`TearInitializationWarning` says so, because a tear guess that is quietly worse
than it looks is the thing worth avoiding:

```python
from difflow.initialization import TearInitializationWarning
```

(It lives in `difflow.initialization`, not on the top-level package.)

### The flowsheet default stream

The last resort, and what is used when `use_initialization=False`: every species
at `default_flow`, at `default_T` and `default_P`. Those three are constructor
arguments:

```python
fs = Flowsheet(["A", "B"], default_flow=1e-6, default_T=350.0, default_P=2e5)
```

Under tracing the guessing pass is skipped entirely and the default stream is
used, because a best-effort pass that catches exceptions has no business running
over tracers — and the initial guess cannot affect a gradient that comes from
the converged solution anyway.

---

## Unit-level `initialize()`

`difflow.initialization.Initializable` is a runtime-checkable protocol: a unit
supports initialization if it has an `initialize(inlet, **kwargs)` method. The
flowsheet's only test is `isinstance(operation, Initializable)`, so nothing has
to be registered.

Three core units implement it, each with an analytic estimate that cannot fail
the way the real calculation can:

| Unit | Estimate |
|---|---|
| `CSTR` | residence time from the volumetric flow, then `X = k tau / (1 + k tau)` |
| `PFR` | the analytic plug-flow conversion for the same rate constant |
| `Flash` | a simplified Rachford-Rice from the thermo package's K-values, with a bubble/dew check for the fully liquid and fully vapor cases |

It is consulted only where the real call would have failed: the flowsheet
calls `__call__` first on its initialization pass and reaches for
`initialize()` when that raises. An initializer is a fallback, not a
pre-processing step.

The return value is a dict. The flowsheet reads only the outlet guesses out of
it, under either of two keys:

- `outlets`: the guesses as a **sequence, in the order `__call__` returns
  them**. That order is the only thing a caller holding `Unit.outlet_names` can
  match the guesses up by, so a multi-outlet unit has to provide it to be
  usable by the flowsheet.
- `outlet`: the singular spelling, accepted for a unit with one outlet, and
  meaning the same as a one-element `outlets`.

`states` and `info` may carry anything else; the flowsheet ignores them, and
they are for a caller that knows which unit it is talking to (`Flash` also
returns its guesses under `liquid` and `vapor`).

Writing one for a custom unit:

```python
from difflow.streams import make_stream


class MyReactor:
    def __call__(self, inlet):
        ...  # the real calculation, which may fail on an empty inlet

    def initialize(self, inlet, **kwargs):
        outlet = make_stream({"A": 0.5 * inlet["F_A"]}, inlet["T"], inlet["P"])
        return {"outlets": (outlet,), "outlet": outlet}
```

`**kwargs` receives the unit's `params` dict, so an initializer may read the
same specifications the call does. Keep it cheap and unconditional: its whole
value is that it answers where the real calculation would not.

---

## The three acceleration methods

All three iterate on the same packed tear vector, which is, per tear stream and
in `species_order`, `[F_1, ..., F_N, T, P]`, with the streams in sorted name
order.

### Direct substitution (`"none"`)

`x_{k+1} = g(x_k)`, run through `optimistix.FixedPointIteration`. No history, no
extrapolation, and the only path that is traceable — which is why it is also
what `jax.grad` falls back to. It converges linearly at a rate set by the
spectral radius of the tear map and diverges when that exceeds one.

Its convergence verdict is judged against the criterion `optimistix` actually
stops on — elementwise `|dx| < atol + rtol |x|`, with `tol` serving as both —
not against a bare `|dx| < tol`. The two differ by the magnitude of the tear:
with flows of order 1 and `tol=1e-8`, a solve `optimistix` calls successful
lands near `2e-8`. `last_solve_residual` alongside it stays the plain max-norm,
which is the number to compare against `tol` by eye.

### Wegstein

Estimates the tear map's slope from the last two iterates and extrapolates:

$$x^{(k+1)} = q\, x^{(k)} + (1 - q)\, g(x^{(k)}),
\qquad q = \frac{s}{s - 1},
\qquad s = \frac{g(x^{(k)}) - g(x^{(k-1)})}{x^{(k)} - x^{(k-1)}}$$

$q$ is the weight on the **old** iterate and $(1-q)$ the weight on the
substitution value. For a linear $g$ this cancels the map's slope exactly and
lands on the solution in one step.

$q$ is clipped into $[-5, 0]$. That bound is what makes Wegstein the right
choice for an oscillating loop: for $s < 0$ — the overshoot case — $q$ lands in
$(0, 1)$ and the update is damped toward the previous iterate, while for
$0 < s < 1$ it is negative and the update extrapolates past $g$.

Cost is one flowsheet pass per iteration plus two vectors of storage.

### Anderson

The default. Keeps a rolling history (`anderson_depth`, default 5) and solves a
small least-squares problem for the combination of past iterates that minimises
the residual — equivalent to GMRES on the fixed-point iteration. It is exact on
an affine map within `depth` steps and is the best general-purpose choice for a
loop that is slow but not oscillating.

Two implementation details matter to a user:

- The least-squares problem is solved with `jnp.linalg.lstsq` on the difference
  matrix directly, **not** through the normal equations. The normal equations
  square the condition number, which overflows to NaN for a badly scaled packed
  tear vector — gas transmission networks, whose `[F, T, P]` entries span
  1e1 kg/s to 1e6 Pa, have a normal-equation condition number of order
  `(P/F)^4`. Badly scaled tear vectors are therefore fine.
- If an accelerated step comes back non-finite (a degenerate history), the
  iteration silently falls back to plain substitution for that step rather than
  propagating NaN.

Cost is one flowsheet pass per iteration plus `depth` vectors of storage and a
small `lstsq`.

On a representative nonlinear one-loop test problem solved to `1e-10`:

| `acceleration` | iterations |
|---|---|
| `"none"` | 19 |
| `"wegstein"` | 5 |
| `"anderson"` | 8 |

Those numbers are problem-specific — Wegstein's scalar slope estimate is at its
best on a loop with one dominant mode, and Anderson pulls ahead as the tear
vector grows — but the order of magnitude is typical: acceleration is worth a
factor of a few, not a factor of a hundred.

### Clipping negative flows is a correctness switch

Both accelerated methods extrapolate, and an extrapolated molar flow can land
below zero on the way to a solution that is entirely positive. By default the
accelerated iterations therefore project the **flow** entries of the tear vector
onto `[0, inf)` after each step. Temperature and pressure are never touched.

This is right for a chemical flowsheet and wrong wherever a negative tear flow
is a legitimate answer. Gas transmission networks are the standing example:
flow against an arc's reference direction is routine, the fixed point genuinely
has negative components, and clipping means the iteration can never reach it.
`difflow_gas.GasNetworkFlowsheet` overrides the default to `False` for exactly
this reason.

```python
streams = fs.solve(clip_negative_flows=False)   # signed tear flows
```

If a solve with signed flows will not converge and nothing else explains it,
this is the first thing to check.

### Damping

For a tear map whose eigenvalue spectrum lies in `[-m, 0)` — the overshoot case
— the damped iteration

$$x^{(k+1)} = x^{(k)} + \alpha\left(g(x^{(k)}) - x^{(k)}\right)$$

has the same fixed point but contracts for $\alpha < 2/(1+m)$, with the optimal
scalar damping near $\alpha = 2/(2+m)$. An $\alpha$ of 0.3 is safe up to a
spectral radius of about 5. When in doubt, measure: a finite-difference
Jacobian of the tear map at a solved state gives $m$ directly. The analysis is
general even though it was written down for gas networks, where measured
spectra reach $m \approx 3.1$ (GasLib-40).

```{warning}
`Flowsheet.solve(damping=...)` is accepted and documented but **is not
currently applied**: the `acceleration="none"` path iterates the *undamped*
map through `optimistix`, and changing `damping` changes nothing. Verified by
inspection and by measurement — 1.0, 0.5, 0.3 and 0.05 all take the same
iteration count to the same residual.

Where a genuinely damped, traceable fixed point is needed today, the working
implementation is `difflow_gas.GasNetworkFlowsheet.solve_differentiable(alpha=...)`,
which iterates the damped map through `optimistix` and differentiates through
the converged solution. For an oscillating loop on a plain `Flowsheet`, use
`acceleration="wegstein"`, whose clipped $q$ damps in the same regime.
```

---

## The traced fallback

The accelerated solvers are Python loops that branch on the residual to stop
early, and a tracer cannot answer `residual < tol`. So when `solve()` sees a
tracer in any feed or unit parameter, it **replaces the requested acceleration
with the `optimistix` fixed point** and records that it did:

```python
def objective(feed_F):
    fs.feeds["feed"] = make_stream({"A": feed_F}, 300.0, 1e5)
    return fs.solve(tol=1e-10, max_iter=200)["loop_out"]["F_A"]

gradient = jax.grad(objective)(1.0)
fs.last_solve_method     # 'fixed_point (traced)'
fs.last_solve_converged  # None -- the residual was a tracer
```

Three consequences worth knowing before differentiating through a flowsheet:

- **The gradient is exact, and is not the gradient of the iteration.**
  `optimistix` carries an implicit-differentiation rule, so the derivative comes
  from the converged solution through the implicit function theorem, not from
  an unrolled loop. It does not depend on how many iterations ran.
- **You lose the acceleration.** A loop that needs Anderson to converge in 100
  iterations eagerly may need many more under `grad`. Raise `max_iter` on the
  differentiated path.
- **You lose the convergence verdict.** There is no concrete residual to judge,
  so `last_solve_converged` is `None` and the solve stays silent whatever
  `on_nonconvergence` says — guessing would either raise inside a gradient or
  warn on a solve that was fine. Check convergence eagerly, at the same
  parameter values, before trusting a gradient.
- **You lose the feed-propagation guess.** The initialization pass catches
  exceptions, which has no business running over tracers, so a traced solve
  starts from the flowsheet default stream unless you pass `tear_initial`.
  For a loop the default start is nowhere near, pass the eagerly converged
  tear in explicitly.

The same rule explains a related gotcha: never record a solve diagnostic with a
bare `float()` in code that may be traced. `flowsheet._concrete()` returns
`None` under tracing instead of raising.

---

## The equation-oriented route

`fs.solve_eo()` assembles every unit's `eo_residuals` into one system
$F(x) = 0$ and solves it with Newton via `optimistix`
(see [Equation-Oriented (EO) Solver](eo-solver.md)). It converges quadratically
near the solution, which is why a tightly coupled loop that takes hundreds of
tear iterations often takes a handful of Newton steps — and why it needs a
starting point inside the basin.

By default it gets one by **running the sequential solver first**:

```python
streams = fs.solve_eo(use_sm_init=True)   # the default
```

`use_sm_init` runs `fs.solve(tol=1e-6, max_iter=50, acceleration="anderson")`
and uses the result — converged or not — as the Newton start, keeping whichever
streams the EO state vector needs. A loose sequential solve is cheap and lands
close enough for Newton far more often than a feed-propagation guess does. If
the sequential solve raises, `solve_eo` falls back to feed propagation rather
than failing.

Two things follow:

- A non-converged sequential solve still makes a fine EO start. That is the
  point: SM is being used for its robustness, Newton for its accuracy.
- Because the sequential pre-solve swallows exceptions, a `solve_eo` that
  behaves as if it had a bad start may have had one. Run `fs.solve()` on its
  own to find out.

`use_sm_init=False` uses feed propagation instead, and an explicit
`initial_guess` (a dict of streams, like `tear_initial` but covering every
unknown stream) overrides both.

---

## Diagnostics

Every solve records what happened, whichever path ran and whether or not it
converged:

| Attribute | Meaning |
|---|---|
| `last_solve_converged` | `True` met the tolerance, `False` ran out of iterations, `None` the residual was a tracer and there is nothing to judge |
| `last_solve_residual` | final tear residual, as a plain max-norm of `g(x) - x` |
| `last_solve_tol` | the tolerance that was asked for |
| `last_solve_iterations` | iterations used; equals `max_iter` when it did not converge |
| `last_solve_method` | `"anderson"`, `"wegstein"`, `"none"`, `"fixed_point (traced)"`, or `"direct"` for a recycle-free sequential solve |
| `last_solve_tear_streams` | the tear (recycle destination) names |

The tri-state `last_solve_converged` is the one to test against `is False`
rather than falsy: `None` means "not judged", not "failed".

`ConvergenceWarning` and `ConvergenceError` are exported from `difflow`, so

```python
import warnings
from difflow import ConvergenceWarning

warnings.simplefilter("error", ConvergenceWarning)
```

turns every non-converged solve in a script into a failure.

---

## A worked diagnosis

A one-unit loop whose tear map is $F \mapsto F_{\text{feed}} + k\sqrt{F + 0.1}$,
deliberately given too few iterations:

```python
import warnings

import jax
import jax.numpy as jnp

from difflow import ConvergenceWarning
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream

jax.config.update("jax_enable_x64", True)


class Loop:
    def __init__(self, k):
        self.k = k

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + self.k * jnp.sqrt(tear["F_A"] + 0.1)},
            feed["T"],
            feed["P"],
        )


def build():
    fs = Flowsheet(species_order=["A"], default_flow=0.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", Loop(0.9), ["feed", "tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


fs = build()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    streams = fs.solve(max_iter=3)

assert fs.last_solve_converged is False
assert fs.last_solve_method == "anderson"
assert fs.last_solve_residual > fs.last_solve_tol   # 1.9e-2 against 1e-8
```

The residual is six orders above tolerance, so this is not a loop that needs a
few more iterations. A starting point taken from anywhere near the answer buys
four orders of magnitude at the *same* three iterations:

```python
import warnings

import jax
import jax.numpy as jnp

from difflow import ConvergenceWarning
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream

jax.config.update("jax_enable_x64", True)


class Loop:
    def __init__(self, k):
        self.k = k

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + self.k * jnp.sqrt(tear["F_A"] + 0.1)},
            feed["T"],
            feed["P"],
        )


fs = Flowsheet(species_order=["A"], default_flow=0.0)
fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
fs.add_unit(Unit("loop", Loop(0.9), ["feed", "tear"], ["loop_out"]))
fs.add_recycle("loop_out", "tear")

guess = {"tear": make_stream({"A": 2.4}, 300.0, 1e5)}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    fs.solve(tear_initial=guess, max_iter=3)

assert fs.last_solve_residual < 1e-4     # 8.3e-6, from 1.9e-2
```

And the gradient through the converged loop comes from the traced path, matching
a central difference on the eager one:

```python
import jax
import jax.numpy as jnp

from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream

jax.config.update("jax_enable_x64", True)


class Loop:
    def __init__(self, k):
        self.k = k

    def __call__(self, feed, tear):
        return make_stream(
            {"A": feed["F_A"] + self.k * jnp.sqrt(tear["F_A"] + 0.1)},
            feed["T"],
            feed["P"],
        )


fs = Flowsheet(species_order=["A"], default_flow=0.0)
fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
fs.add_unit(Unit("loop", Loop(0.9), ["feed", "tear"], ["loop_out"]))
fs.add_recycle("loop_out", "tear")


def objective(feed_F):
    fs.feeds["feed"] = make_stream({"A": feed_F}, 300.0, 1e5)
    return fs.solve(tol=1e-10, max_iter=200)["loop_out"]["F_A"]


gradient = jax.grad(objective)(1.0)
assert fs.last_solve_method == "fixed_point (traced)"
assert fs.last_solve_converged is None

h = 1e-6
finite_difference = (objective(1.0 + h) - objective(1.0 - h)) / (2 * h)
assert abs(gradient - finite_difference) < 1e-6
```

---

## Tear selection

difflow does **not** choose tear streams for you: the tears are whatever
`add_recycle` registered, one per recycle, and where you tear a loop is a
modelling decision with real consequences for how fast it converges.

`difflow.select_tear_streams` and `difflow.find_cycles` are analysis helpers for
making that decision, not part of the solve path:

```python
from difflow import find_cycles, select_tear_streams
from difflow.initialization import FlowsheetGraph

cycles = find_cycles(FlowsheetGraph.from_flowsheet(fs))
tears = select_tear_streams(fs, method="heuristic")   # or "minimum"
```

`"heuristic"` prefers a stream leaving a mixer, where the composition is
already known; `"minimum"` greedily picks the stream appearing in the most
remaining cycles, to tear as few streams as possible. The general rule is the
one the gas plugin states quantitatively: tear where the map is least
sensitive. A chord's tear-map slope there goes as
`-sum(beta_e |q_e|) / (beta_c |q_c|)` over its loop, so tearing the *most*
resistive element keeps the spectral radius small.

---

## What the plugins add

Two plugins solve initialization problems the core does not, and both are worth
reading even if you work in neither domain.

**Gas networks compute the decomposition from the topology.**
`difflow_gas.decompose(net, root=...)` builds a spanning tree with the
non-invertible arc kinds forced in-tree and the most resistive pipe of each
loop pushed out as the chord, then emits a leaf-to-root balance schedule. The
tears are derived rather than declared, and the chord choice is made on the
convergence criterion above. See
[The computed decomposition](unit-operations-gas.md#the-computed-decomposition)
and [Building and solving](unit-operations-gas.md#building-and-solving).

**REE mass-action sections initialize by continuation from a correlation.**
Mass-action equilibrium is exponentially nonlinear and loses Newton from a poor
start, so `difflow_ree` solves the existing L1 correlation first and uses its
Kremser profile as the starting point, then walks it into Newton's basin with
damped Newton, a trust region and damped Newton again —
`n_continuation_steps > 1` additionally ramps the rare-earth feed from dilute to
full strength. All of it runs under `stop_gradient`: it moves the starting point
and never the answer, which is the pattern to copy for any globalization in a
differentiable model. See
[Unknowns, equations and how they are solved](unit-operations-ree.md#unknowns-equations-and-how-they-are-solved).

---

## See also

- [Streams and Flowsheets](streams-and-flowsheets.md) — building flowsheets, recycles, implicit differentiation
- [Equation-Oriented (EO) Solver](eo-solver.md) — the simultaneous alternative
- [Solvers and Utilities](solvers-and-utilities.md) — the numerics underneath
- [Gas Transmission Network Unit Operations](unit-operations-gas.md) — computed decomposition, damped differentiable tear solve
