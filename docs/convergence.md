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
- [Measured pass rates](#measured-pass-rates)
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

3. **Damp it, or change the method.** The default is Anderson. A loop that
   *oscillates* — the residual alternating in sign, flows overshooting and
   coming back — has a tear map with negative eigenvalues, and there are two
   answers to that. `acceleration="none", damping=0.3` takes a short step
   toward the substitution value, which makes the iteration contractive where
   the undamped one is not; it is the one to reach for when the residual
   *rises*, and the only one that also works under `jax.grad`.
   `acceleration="wegstein"` gets there by a different route — its
   acceleration factor is clipped into `[-5, 0]`, which damps toward the
   previous iterate in exactly that regime — and needs no tuning. A loop that
   is merely slow and monotone is what Anderson is for. Undamped
   `acceleration="none"` is the honest baseline: if that diverges, the map
   itself is not contractive.

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

`x_{k+1} = x_k + \alpha (g(x_k) - x_k)`, run through
`optimistix.FixedPointIteration`, with $\alpha$ = `damping` = 1 by default. No
history, no extrapolation, and the only path that is traceable — which is why
it is also what `jax.grad` falls back to. Undamped it converges linearly at a
rate set by the spectral radius of the tear map and diverges when that exceeds
one; [damping](#damping) is what extends it to maps that overshoot.

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

`damping` takes a step only part of the way toward the substitution value:

$$x^{(k+1)} = x^{(k)} + \alpha\left(g(x^{(k)}) - x^{(k)}\right)$$

The fixed point is unchanged — at $g(x) = x$ the correction vanishes for any
$\alpha$ — but the iteration is contractive where plain substitution
overshoots. For a tear map whose eigenvalue spectrum lies in $[-m, 0)$ it
contracts for $\alpha < 2/(1+m)$, with the optimal scalar damping near
$\alpha = 2/(2+m)$. An $\alpha$ of 0.3 is safe up to a spectral radius of
about 5, which is why it is the usual first thing to try; when in doubt,
measure, since a finite-difference Jacobian of the tear map at a solved state
gives $m$ directly. The analysis is general even though it was written down
for gas networks, where measured spectra reach $m \approx 3.1$ (GasLib-40).

It applies to `acceleration="none"` — and therefore to the traced path, which
falls back to it. The default is `1.0`, which is plain undamped substitution:

```python
streams = fs.solve(acceleration="none", damping=0.3)
```

On a loop whose tear map has eigenvalue $-2$ — substitution triples the error
and flips its sign every step — the difference is not a matter of iteration
count:

| `damping` | outcome after 200 iterations |
|---|---|
| `1.0` | residual `4.8e+60`; diverged |
| `0.3` | residual `3.0e-12`; converged to the exact fixed point |

Three conventions keep this a knob on *how the solve gets there* rather than on
what it returns:

- **`damping` never moves the answer.** The fixed point is a property of the
  map. Converged solves at 1.0, 0.5, 0.3 and 0.1 agree to the tolerance.
- **`last_solve_residual` is the undamped residual** $|g(x) - x|$, not the
  $\alpha$-times-smaller step the solver actually took. A heavily damped solve
  would otherwise look converged by exactly the factor it was damped by.
- **`tol` still means `tol`.** `optimistix` stops on the step it takes, so the
  tolerance handed to it carries the damping factor. Without that, a solve at
  $\alpha = 0.3$ would stop three times short of the tolerance it was asked
  for — and the verdict above would then, correctly, call it non-converged.

Gradients are unaffected: `optimistix` differentiates the converged solution
through the implicit function theorem, so the derivative is exact whatever
$\alpha$ and however many iterations ran. That cuts both ways — on the
diverging run above, the *gradient* is still 1.0 while the *value* is `-6.4e+60`.
A gradient is only as trustworthy as the solve underneath it, and under tracing
nothing warns you.

`damping` must be positive; `0.0` makes the iteration the identity, under which
every point is a fixed point, and is refused.

```{note}
`difflow_gas.GasNetworkFlowsheet.solve_differentiable(alpha=...)` predates
this and does the same thing with separate `rtol`/`atol` and an optional stats
return. It is still the convenient entry point for gas networks, which also
need `clip_negative_flows=False` and the builder's tear guesses.
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

## Measured pass rates

How often any of this actually works is a measurement, and
`difflow.convergence` is where it is made. It holds a corpus of deliberately
hard flowsheets and runs each one across every acceleration and every
initialization strategy:

```python
from difflow.convergence import run_benchmark

report = run_benchmark()
print(report.as_text())
```

The full grid is 99 solves and takes several minutes, most of it JAX
compilation. Narrow it while iterating:

```python
report = run_benchmark(cases=["high_gain_recycle"], accelerations=("anderson",))
```

### Passing means converged *and* right

A solve reports two separate things, and the benchmark keeps them apart:

| | meaning |
|---|---|
| `Outcome.converged` | the solver's own verdict, `last_solve_converged` |
| `Outcome.correct` | the answer closes its material balance, or matches a known one |
| `Outcome.passed` | both |

They are not the same. `pass_rate` is 46.5% against a `convergence_rate` of
49.5%: three of the 99 solves report success and land somewhere that does not
audit. A benchmark that counted only the solver's flag would call those wins.

### What the corpus says today

```
99 solves  (tol=1e-08, max_iter=100)
pass rate         46.5%   (converged AND the answer audits)
convergence rate  49.5%   (the solver's own verdict)

by acceleration                    by initialization
  anderson      28/33   84.8%        default       15/33   45.5%
  wegstein       9/33   27.3%        unit          15/33   45.5%
  none           9/33   27.3%        feed          16/33   48.5%

mean iterations over passing solves: anderson=4.3, wegstein=18.9, none=41.0
```

**Every case in the corpus is solved by at least one acceleration.** Eleven
cases across the hard families — high loop gain, sharp splits, multi-loop,
phase-regime switching, near-pinch columns, trace-magnitude tears, signed
tears — and none of them defeats all three methods. That is the most useful
thing the benchmark says, and it is a negative result.

Four findings worth acting on:

**Acceleration dominates; initialization barely registers.** Anderson passes
84.8% where plain substitution passes 27.3%. Against that, the three starting
points are separated by a single solve. The first thing to change on a failing
recycle is `acceleration`, not the guess.

**The `initialize()` path (#247) changed no outcome.** `"unit"` — the
propagation pass that #247 wired up — scores exactly what `"default"` scores,
the 0.01 mol/s stream it replaced: 15/33 either way, unchanged from when the
corpus was six cases. It reliably saves an iteration or two and it flips no
verdict anywhere. It made the first step better without making a failing solve
succeed. If a recycle is failing, a better guess of the same kind is not the
fix.

**Anderson is not free, and `clip_negative_flows` is why, once.**
`phase_coupled_flash` fails under Anderson and converges under Wegstein in 28
iterations — Anderson walks off somewhere else rather than running out of road.
And `signed_tear` inverts the ranking outright: plain substitution solves it
and *both* accelerated methods fail. The reason is worth knowing —
`clip_negative_flows` defaults to `True` and is applied by the Wegstein and
Anderson paths but **not** by the unaccelerated one, so on a tear whose answer
is genuinely negative the two accelerated methods carry a projection the plain
one does not. Passing `clip_negative_flows=False` fixes Anderson on it
immediately (100 iterations without convergence → 13 with). This is the same
point `difflow_gas` already makes for signed flows; it applies to any tear
whose components can go negative.

**A disjunction does not need binaries here.** `regime_switch` is a unit
choosing between two linear branches with the answer exactly on the boundary,
expanding below the switch and contracting above it. Plain substitution can
only chatter across the join, and Wegstein stalls — but Anderson lands on it
in two iterations.

### What is *not* hard

Worth recording, because these are the families most often assumed difficult:

- **A rigorous MESH column in a recycle.** `column_recycle` — twelve stages,
  95% of the distillate returned — passes under every setting, at every reflux
  ratio tried from just above minimum to well above it.
- **Nonsmooth tear maps as such.** Both `regime_switch` and the kinked maps
  tried while building the corpus fall to Anderson quickly.
- **Tear dimension.** A tear with twelve components and a spectral radius of
  0.985 converges under Anderson as readily as one with three, provided the
  coupling is non-negative. Where high-dimensional cases did fail, the cause
  was the negative-flow clipping above, not the dimension.

### The trap the corpus exposes

`tol` is an **absolute** residual on the tear, and on a high-gain loop that is
much weaker than it looks. A step of $d$ on a loop of gain $g$ leaves an error
of about $d/(1-g)$, so at $g = 0.97$ the residual understates the remaining
error by a factor of 33.

`trace_recycle` is built on exactly that: gain 0.97 on a tear whose converged
value is about $3 \times 10^{-5}$. Wegstein stops with a residual inside
`1e-8`, reports convergence, and is about 1% out. It is right on its own terms
— the step really is that small — and wrong on yours.

On a loop where you know the gain is near one, tighten `tol` by roughly
$1/(1-g)$, or audit the answer rather than the residual.

For a multicomponent tear the factor is $\lVert (I-M)^{-1} \rVert$ rather than
$1/(1-\rho)$, and for a non-normal $M$ those are not close: `signed_tear` has
$\rho = 0.75$, which suggests a factor of 4, and actually leaves a relative
error of $1.2 \times 10^{-6}$ from a residual of $10^{-8}$ — a factor of about
450.

### Adding a case

A case is a name, a builder, and a sentence saying what makes it hard:

```python
from difflow.convergence import Case, run_case

case = Case(
    name="my_hard_loop",
    build=build_my_flowsheet,          # must return a FRESH Flowsheet
    difficulty="Why this one is hard, in a sentence.",
    tags=("recycle", "high-gain"),
)
run_case(case, acceleration="wegstein")
```

`build` has to return a new flowsheet each call — `solve` records its verdict
on the object, so a shared one carries one run's result into the next. The
default audit is the overall mole balance; a case whose reaction changes the
mole count, or whose answer is known analytically, passes its own `check`.

Keep at least one case in the corpus that is *meant* to pass. A benchmark
where everything fails cannot distinguish a hard corpus from a broken solver.
`two_phase_flash` and `column_recycle` are the controls, and both pass 9/9.

And check a new case is not *rigged* — that no initialization strategy starts
on the answer it is about to be scored against. `overshoot_loop` originally
used $g(x) = 3F - 2x$, whose fixed point is $x = F$, exactly the `"feed"`
guess; that strategy collected two passes on a case it had not solved.
`TestNoCaseIsRigged` checks every case in the corpus against this.

---

## Tear selection

Where you tear a loop is a modelling decision with real consequences: the tear
is the fixed point the solve iterates on, so its placement sets the spectral
radius of the map, and with it how fast the loop converges and sometimes
whether it converges at all.

By default difflow does not make that decision for you. The tears are whatever
`add_recycle` registered, one per recycle, and a loop closed in the topology
but never declared is not torn at all — inferring one silently would change
the answer for flowsheets that already run.

`fs.tear_analysis()` reports the loops, the declared tears, what each strategy
would pick instead, and any inlet the unit order cannot supply. It reads the
topology and runs no unit:

```python
print(fs.tear_analysis())
```

`solve(tears="auto")` opts in to the choice, but only when no recycle has been
declared; `"minimum"` names the other strategy. The selection is recorded in
`last_solve_tear_streams`, and neither it nor the unit sequence is kept, so
`fs.recycles` and `fs.units` are what you declared once the solve returns.
[Streams and Flowsheets](streams-and-flowsheets.md) covers both in full.

The pieces are also usable on their own:

```python
from difflow import (
    FlowsheetGraph, find_cycles, select_tear_streams, calculation_order,
)

graph = FlowsheetGraph.from_flowsheet(fs)
cycles = find_cycles(graph)                           # elementary cycles
tears = select_tear_streams(fs, method="heuristic")   # or "minimum"
calculation_order(graph, tears)                       # order once those are seeded
```

`"heuristic"` takes one tear per loop, preferring a stream leaving a mixing
point, where the stream is the sum of everything entering it so a guess wrong
in composition is still right in order of magnitude; `"minimum"` covers every
loop with as few tears as it can. The general rule is the one the gas plugin
states quantitatively: tear where the map is least sensitive. A chord's
tear-map slope there goes as `-sum(beta_e |q_e|) / (beta_c |q_c|)` over its
loop, so tearing the *most* resistive element keeps the spectral radius small.

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
