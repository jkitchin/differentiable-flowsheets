# Flexibility Analysis

`difflow.flexibility` answers one question about a finished design: **does it
still work when the feed does not?** — and, when the answer is "not quite", it
says which realization runs out first, which constraint binds there, and how
much of the shortfall is a bill controls can pay and how much is one they
cannot.

It is a module alongside `difflow.uncertainty` and `difflow.planning` — not a
`difflow.plugins` entry point. That registry is for unit operations.

## Table of Contents

1. [The two numbers](#the-two-numbers)
2. [Quick start](#quick-start)
3. [Writing the model](#writing-the-model)
4. [Vertex enumeration, and when it is exact](#vertex-enumeration-and-when-it-is-exact)
5. [The flexibility index, and the diagnosis that matters more](#the-flexibility-index-and-the-diagnosis-that-matters-more)
6. [Feed uncertainty and parameter uncertainty are different bills](#feed-uncertainty-and-parameter-uncertainty-are-different-bills)
7. [When the worst case is too conservative](#when-the-worst-case-is-too-conservative)
8. [Derivatives](#derivatives)
9. [Drawings](#drawings)
10. [Numerical notes](#numerical-notes)
11. [What this module is not](#what-this-module-is-not)
12. [API summary](#api-summary)

---

## The two numbers

An optimizer hands back a design that sits exactly on its binding
specification, because that is where the money is. A design placed exactly on a
purity floor spends roughly half its campaigns on the wrong side of it. The
question of how much room is left is not the same as the question of what the
optimum was, and it needs its own measurement.

The **feasibility function** of Halemane and Grossmann is

$$\psi(d) \;=\; \max_{\theta \in T}\ \min_u\ \max_j\ f_j(d, u, \theta)$$

where the constraints are written so that $f_j \le 0$ means "satisfied". Read
it from the inside out: for a given realization $\theta$ the controls $u$ are
re-optimized to make the worst constraint as small as they can; the outer
maximum then picks the realization that defeats them most thoroughly. The
design is feasible over the whole set exactly when $\psi(d) \le 0$, and the
value itself is the leftover residual in the units of the constraint that
binds.

The **flexibility index** of Swaney and Grossmann is the largest scaling of
that set the design survives:

$$F \;=\; \max\ \{\delta \ge 0 : \psi(d, \delta) \le 0\},
\qquad
T(\delta) = \{\theta : \theta^N - \delta\Delta^- \le \theta \le \theta^N + \delta\Delta^+\}.$$

$F \ge 1$ covers the envelope you were given. $F = 0.6$ covers 60% of it, and
the design will spend campaigns outside. $F = 2$ is margin that was paid for
and may not be needed.

## Quick start

One mixer-settler stage of a rare-earth circuit. The feed is an ore-derived
liquor whose concentration varies; the extractant flow is a control the
operator can still turn once today's liquor is assayed; the installed capacity
is the design.

```python
import jax.numpy as jnp
from difflow.flexibility import feasibility_function, flexibility_index

C_MAX = 0.05                      # raffinate purity floor

def stage(d, u, theta):
    c_feed, = theta
    S = u.sum()                   # extractant flow, the recourse variable
    c_raff = c_feed / (1.0 + S)   # single-stage extraction, K = 1
    return jnp.array([c_raff / C_MAX - 1.0,    # purity  (dimensionless)
                      S / d[0] - 1.0])         # capacity (dimensionless)

FEED = {"c_feed": (1.0, 0.3)}     # nominal 1.0, +/- 0.3
CTRL = {"S": (0.0, 60.0)}         # the extractant flow may move in here

res = feasibility_function(stage, [26.0], FEED, CTRL,
                           constraint_names=("purity", "capacity"))
print(res.summary())
```

```
psi = -0.0190433  ->  FEASIBLE over the set at scale 1  [vertex]
  critical realization (vertex 1): c_feed=1.3
  controls re-optimized to: S=25.5047
  binding constraint: purity
  constraint                       value         slack
  purity                       -0.019043      0.019043  <-- binding
  capacity                     -0.019049      0.019049
```

and the index:

```python
idx = flexibility_index(stage, [26.0], FEED, CTRL,
                        constraint_names=("purity", "capacity"))
print(idx.summary())
```

```
flexibility index F = 1.167  -- covers the stated envelope
  limited by vertex 1 [+]: c_feed=1.34999
  binding constraint: purity
  controls there: S=25.9998
  vertex    direction            limit
  1         +                    1.167  <-- binds
  0         -                        4
```

The answer is checkable by hand. Purity holds when $1 + S \ge c_{feed}/c_{max}$
and capacity when $S \le d$, so the design covers a feed of
$(d+1)\,c_{max} = 1.35$, which is $\delta = 0.35/0.3 = 1.1\overline{6}$ of the
stated envelope.

## Writing the model

```text
model_fn(d, u, theta) -> array of constraint values, feasible where <= 0
```

* `d` — the design. Fixed before the uncertainty is revealed.
* `u` — the **recourse** variables, re-optimized *after* it is revealed.
  Everything an operator can still turn once the feed arrives belongs here.
* `theta` — the uncertain parameters the controls are allowed to see.

The split between `d` and `u` is the whole content of a flexibility question.
Move a variable from `d` to `u` and the index goes up, because the design is
now allowed to respond; that difference *is* the value of the control.

Pass `controls=None` for a design with no recourse at all. `u` then arrives as
a length-zero array. Writing the model with `u.sum()` rather than `u[0]` lets
the same model be posed both ways, which makes the with/without comparison a
controlled experiment:

```python
free   = flexibility_index(stage, [26.0], FEED, {"S": (0.0, 60.0)})
frozen = flexibility_index(stage, [26.0], FEED, {"S": (22.309, 22.309)})
free.index      # 1.167  -- the extractant flow may follow the assay
frozen.index    # 0.552  -- pinned at its nominal optimum
```

Being allowed to turn one valve doubles the feed envelope this design covers.
That factor is the value of the control, in the only units that matter.

**Scale the constraints.** The inner problem is a maximum over rows, so a row
in $\mathrm{mol\,s^{-1}}$ and a row in mole fraction do not compare. Write each
constraint in relative form — `value / limit - 1` — as above.

Uncertainty sets and control boxes may be given as mappings, or built
explicitly:

```python
from difflow.flexibility import UncertaintySet, ControlSpec

T = UncertaintySet(nominal=[1.0, 320.0], lower=[0.3, 5.0], upper=[0.15, 10.0],
                   names=["c_feed", "T_feed"])        # asymmetric envelope
u = ControlSpec(lower=[0.0], upper=[60.0], names=["S"])
```

Deviations up and down are independent, so an envelope that can be much leaner
than nominal but only slightly richer is expressible without inventing a
fictitious symmetric range. Control bounds must be finite: an unbounded
recourse variable makes the inner minimization ill-posed as often as not, and a
wide-but-finite box is both honest and better behaved.

## Vertex enumeration, and when it is exact

`method="vertex"` (the default) evaluates the inner problem at every vertex of
the box and takes the largest. It is the right first implementation: simple,
`vmap`-parallel, and **exact whenever the critical realization is a vertex** —
which holds when each $f_j$ is jointly quasi-convex in $\theta$, the usual
situation for a process constraint monotone in a feed variable. Its cost is
$2^n$ inner solves; `MAX_VERTICES` refuses an enumeration that has run away.

It fails, structurally, when the critical realization is *interior* to the set.
`method="continuous"` is the fallback: a projected ascent over $\theta$ inside
the box, seeded at the best vertex, converging to a KKT point of the outer
maximization.

```python
# psi_true = max over theta in [-1, 1] of (0.5 - theta^2 - d) = +0.2 at theta = 0
model = lambda d, u, th: jnp.array([0.5 - th[0] ** 2 - d[0]])

feasibility_function(model, [0.3], {"th": (0.0, 1.0)}, None).psi
# -0.8   -- both vertices look fine

feasibility_function(model, [0.3], {"th": (0.0, 1.0)}, None,
                     method="continuous").psi
# +0.199...   -- the interior point that actually binds
```

Because it is seeded at the vertex answer it never reports *less* than the
vertex answer, and because it is a local ascent it is a lower bound on the true
$\psi$. Both facts run in the direction of honesty about which way the error
goes; neither makes it a global guarantee. If the critical point may be
interior and the stakes are high, use `expected_feasibility` as well — sampling
finds islands that neither a vertex scan nor a local ascent will.

## The flexibility index, and the diagnosis that matters more

The number on its own rarely changes a decision. `FlexibilityResult` therefore
carries:

| field | what it tells you |
|---|---|
| `index` | $F$ |
| `limited_by_vertex`, `direction()` | *which* direction of variability runs out first |
| `binding_constraint` | the spec that binds there |
| `critical_theta`, `controls` | the realization and the controls at $\delta = F$ |
| `vertex_limits` | how much room *every* vertex has |
| `slack_vertices()` | the directions with room to spare |
| `saturated` | `True` when every vertex survived `delta_max`, so $F$ is a lower bound |
| `nominal_feasible` | `False` means the design fails at nominal and $F = 0$ |

`vertex_limits` is the field that changes decisions. A single low entry among
otherwise comfortable ones names *one* direction of feed variability as the
problem, and that is a specification to renegotiate or a control to add — not a
column to make taller.

The method is a bisection on $\delta$ per vertex direction, which assumes the
inner value is nondecreasing in $\delta$ along each direction. That is the
standard assumption behind the vertex characterization of the index. A
constraint genuinely non-monotone in a parameter can hide a small infeasible
island inside a feasible outer set, and no bisection will find it.

## Feed uncertainty and parameter uncertainty are different bills

This is the distinction the module exists to make measurable, and it is the
reason `flexibility` sits next to `uncertainty` and `planning.backoff` rather
than replacing either.

**Feed uncertainty is answerable.** You learn today's liquor before you have to
run it, so the controls can be re-optimized against it. What that variability
costs is not the full swing of the constraint, it is whatever swing survives
after the controls have done their best — and that surviving part is exactly
what $\psi$ measures. The part recourse removes is the *recourse credit*.

**Parameter uncertainty is not answerable.** A distribution coefficient is
never revealed; no control move can be scheduled against a constant you do not
know. The whole propagated swing lands on the constraint and has to be bought
as margin in advance. That is **back-off**, sized as $\kappa\sigma$ with
$\sigma^2 = g^{\mathsf T}\Sigma_\theta\, g$ from
`difflow.uncertainty.propagate_covariance` — the same quantity
`difflow.planning.backoff` sizes for a plan, evaluated here at the critical
feed realization rather than at a planning point.

`uncertainty_penalties` charges both and prints them side by side. The model
takes both kinds explicitly, `model_fn(d, u, theta, phi)`:

```python
from difflow.flexibility import uncertainty_penalties

def stage_with_K(d, u, theta, phi):
    c_feed, = theta
    K, = phi                                    # the controls never see K
    S = u.sum()
    c_raff = c_feed / (1.0 + K * S)
    return jnp.array([c_raff / C_MAX - 1.0, S / d[0] - 1.0])

rep = uncertainty_penalties(stage_with_K, [26.0], FEED, CTRL,
                            parameters={"K": 1.0}, covariance=[[0.05 ** 2]],
                            constraint_names=("purity", "capacity"))
print(rep.summary())
```

```
uncertainty penalties (kappa = 2), feed set 2 vertices, 1 parameters
  psi over the feed set = -0.0190433 at c_feed=1.3
  constraint                nominal       feed   recourse      param     margin  dominant
  purity                     -0.142     0.1229     0.1345    0.09439     0.2173  feed
  capacity                   -0.142     0.1229    -0.1229          0     0.1229  feed
  feed penalty is bought down with controls and instruments: 0.24583 total
  parameter back-off is bought down with experiments: 0.094395 total
  verdict: INFEASIBLE once both penalties are charged
```

Read the last line against the index above. $F = 1.17$ says the design covers
the stated feed envelope with room to spare. A 5% uncertainty on $K$ sinks it
anyway, because the 0.094 of back-off that uncertainty demands is larger than
the 0.019 of margin the worst feed leaves. **The flexibility index alone would
have passed this design.**

That is why the two numbers are reported separately rather than added into one
"uncertainty allowance". They have different remedies and different purchase
orders: feed penalty is bought down with *controls and instrumentation*,
parameter back-off is bought down with *experiments*. `rep.dominant` names
which lever to pull first, per constraint.

The `recourse` column is what makes the first half quantitative. On the purity
row it is `+0.1345`: re-optimizing the extractant flow against the revealed
assay removes that much of the swing, leaving only the `0.1229` of feed penalty
in the column beside it. The `capacity` row shows the credit going the other
way, `-0.1229`, which is correct and worth understanding — recourse minimizes
the *worst* row, and it spends capacity slack to buy purity slack. Only the
maximum over rows is a promise; a per-row credit is bookkeeping.

The cleanest statement of the asymmetry is a controlled experiment on a model
small enough to check by hand. Write it with `u.sum()` so the same function can
be posed with and without a control:

```python
def toy(d, u, theta, phi):
    x = u.sum()
    return jnp.array([phi[0] * theta[0] - x - 1.0, x - d[0]])

kw = dict(parameters={"K": 1.0}, covariance=[[0.01]])      # sigma_K = 0.1
free   = uncertainty_penalties(toy, [2.0], {"feed": (1.0, 0.2)},
                               {"u": (0.0, 5.0)}, **kw)
frozen = uncertainty_penalties(toy, [2.0], {"feed": (1.0, 0.2)}, None, **kw)

free.feed_penalty[0], frozen.feed_penalty[0]    # 0.10   vs 0.20
free.backoff[0],      frozen.backoff[0]         # 0.24   vs 0.24
```

The control halves the feed penalty and leaves the back-off bit-for-bit
identical, because there is nothing to re-optimize against.

## When the worst case is too conservative

$\psi(d) \le 0$ is a guarantee over every point of the set, including the corner
where every parameter is simultaneously at its worst extreme. For ten
independent parameters that corner is one part in a thousand *if* each extreme
were a coin flip, and far less if the extremes are tails of real distributions.
Designing for it buys a guarantee nobody asked for at a capital cost somebody
has to pay.

`expected_feasibility` replaces the guarantee with a probability, using the
*same* inner solve at sampled realizations instead of at vertices:

```python
from difflow.flexibility import expected_feasibility

res = expected_feasibility(stage, [26.0], FEED, CTRL, n_samples=512,
                           distribution="uniform")
res.probability          # P(psi_sample <= 0), with res.standard_error
res.chance_margin(0.95)  # the 95% quantile; <= 0 iff P(feasible) >= 0.95
res.satisfies(0.95)
res.violation_rate       # per-constraint P(f_j > 0)
res.blame                # among failures, which constraint was worst
```

`blame` is the practical payoff and the reason not to collapse this to a single
number: a design that fails 4% of the time on one purity spec is a different
problem from one that fails 4% of the time spread over six constraints.

`distribution="normal"` draws a two-piece normal whose one-sigma half-widths
are the box deviations, so an asymmetric envelope stays asymmetric and about
68% of draws land inside the $\delta = 1$ box.

This is a Monte Carlo estimate and is exposed with its standard error rather
than as a bare probability. Use it to relax an over-conservative worst case,
not to certify a rare event.

## Derivatives

`feasibility_value` is the bare traceable scalar and is `jit`-, `grad`- and
`vmap`-safe:

```python
import jax
from difflow.flexibility import feasibility_value

psi = jax.jit(lambda d: feasibility_value(stage, d, FEED, CTRL))
g   = jax.grad(lambda d: feasibility_value(stage, d, FEED, CTRL))
```

Two things make those derivatives usable rather than merely defined.

**The maximum is smoothed only for the search, never for the answer.** The
inner descent direction comes from a log-sum-exp with an annealed temperature;
the value reported is the exact $\max_j f_j$ at the point the search lands on. A
smoothed value sits *below* the true maximum, which would make an infeasible
design look feasible. An imperfectly converged search reports a value that is
too *high*, which is the safe direction.

**The inner gradient is the multiplier-weighted one, not the active
constraint's.** At the minimum of a maximum the optimal $u$ sits on the kink
where two or more constraints are equal, and differentiating whichever row
`jnp.max` happened to select gives a number that is not the derivative at all —
in the textbook one-control, two-constraint case it is off by a factor of two,
or identically zero. The minimizer is wrapped in `stop_gradient` and the
sensitivity taken from the inner problem's own optimality conditions,

$$\frac{\partial}{\partial p}\min_u \max_j f_j(u,p)
= \sum_j \lambda_j \frac{\partial f_j}{\partial p}\bigg|_{u^*},
\qquad \sum_j \lambda_j = 1,\quad \sum_j \lambda_j \nabla_u f_j = 0,$$

with $\lambda$ recovered from a small least-squares problem at $u^*$. This
costs one Jacobian of the model rather than a reverse pass through a few
hundred solver steps, and it does not accumulate the solver's own error into
the derivative.

## Drawings

matplotlib is imported inside the functions, so importing the module costs
nothing in a headless run.

```python
from difflow.flexibility import draw_flexibility_region, draw_penalty_split

two = flexibility_index(
    lambda d, u, th: jnp.array([th[0] / (1.0 + u.sum()) / C_MAX - 1.0,
                                u.sum() / (d[0] * th[1]) - 1.0]),
    [26.0], {"c_feed": (1.0, 0.3), "avail": (1.0, 0.1)}, CTRL)
draw_flexibility_region(two, axes=(0, 1))   # stated vs covered envelope
draw_penalty_split(rep)                     # feed bar vs parameter bar
```

`draw_flexibility_region` draws the stated envelope, the envelope the design
actually covers, the binding vertex, and each other vertex annotated with its
own limit. It needs **two** parameters to plot and raises otherwise, because
that is what a plane holds honestly; with one parameter, read `summary()`.

## Numerical notes

* Everything runs in float64 (difflow sets `jax_enable_x64` on import).
* The inner minimax is projected Adam from several deterministic starts.
  Starts are fixed rather than random so a reported index is reproducible.
  Raise `SolverOptions(n_starts=..., steps=...)` for a stubborn inner problem.
* Expect the inner value to be accurate to roughly $10^{-5}$ of the constraint
  scale with defaults, and the multiplier-weighted derivative to a few parts in
  $10^{3}$. Do not read a verdict off a $\psi$ that is within solver noise of
  zero; that is what the margin, not the sign, is for.
* `flexibility_index` bisects to `delta_max` (default 4). If every vertex
  survives, `saturated` is `True` and the index is a lower bound — raise
  `delta_max` rather than reporting the number.
* `MAX_VERTICES` (16384) refuses a runaway enumeration. Past that, use
  `method="continuous"` or `expected_feasibility`.

## What this module is not

**Not a design optimizer.** This module *measures* a given design. Optimizing a
design subject to a flexibility constraint is a two-stage stochastic program,
and `difflow.planning` is where optimization lives.

**Not a mixed-integer formulation of $\psi$.** The active-set/MILP
characterizations of Grossmann and Floudas, and the KKT/complementarity and
bilevel reformulations of the max–min–max, are not implemented here. Where a
rigorous global $\psi$ is needed, the formulation machinery — bilevel
reformulation by KKT conditions or strong duality, complementarity handling by
Scholtes regularization / SOS1 / disjunctions, robust counterparts with affine
decision rules, and scenario/SAA/DRO stochastic programming — lives in
`discopt`, and posing a difflow flowsheet to it is tracked separately. What is
here is the vertex-enumeration route with a local continuous fallback, which is
exact on the common monotone case and honest about when it is not.

**Not a substitute for back-off.** Flexibility and back-off answer different
questions and neither subsumes the other; `uncertainty_penalties` exists
precisely to report them together without confusing them.

## API summary

```python
from difflow.flexibility import (
    # Sets and recourse
    UncertaintySet, ControlSpec, as_uncertainty_set, as_control_spec,
    NO_CONTROLS, MAX_VERTICES,
    # Feasibility function
    feasibility_function, feasibility_value, FeasibilityResult,
    inner_value, vertex_values, METHODS,
    # Flexibility index
    flexibility_index, vertex_limits, FlexibilityResult,
    # Stochastic counterpart
    expected_feasibility, sample_set, StochasticFeasibilityResult, DISTRIBUTIONS,
    # Feed vs parameter penalties
    uncertainty_penalties, PenaltyReport,
    # Inner solver
    minimax_value, minimax_controls, smooth_max, box_adam,
    SolverOptions, DEFAULT_OPTIONS,
    # Drawings
    draw_flexibility_region, draw_penalty_split,
)
```

Every result object carries `summary()` (the answer) and `describe()` (the
problem that was solved, then the answer), matching the reporting style of
`difflow.planning`.

## References

* K. P. Halemane and I. E. Grossmann, "Optimal process design under
  uncertainty", *AIChE J.* **29** (1983) 425.
  [doi:10.1002/aic.690290312](https://doi.org/10.1002/aic.690290312)
* R. E. Swaney and I. E. Grossmann, "An index for operational flexibility in
  chemical process design", *AIChE J.* **31** (1985) 621 (Part I) and 631
  (Part II).
  [doi:10.1002/aic.690310412](https://doi.org/10.1002/aic.690310412),
  [doi:10.1002/aic.690310413](https://doi.org/10.1002/aic.690310413)
* I. E. Grossmann, B. A. Calfa and P. Garcia-Herreros, "Evolution of concepts
  and models for quantifying resiliency and flexibility of chemical processes",
  *Comput. Chem. Eng.* **70** (2014) 22.
  [doi:10.1016/j.compchemeng.2013.12.013](https://doi.org/10.1016/j.compchemeng.2013.12.013)
