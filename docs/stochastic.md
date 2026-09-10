# Stochastic Programming (`difflow.stochastic`)

A design that is optimal at the nominal parameters is optimal at exactly one
point of a distribution nobody knows. `difflow.stochastic` replaces that point
with a sample and optimizes the resulting histogram — the **sample average
approximation** of a two-stage stochastic program — with the model, the
gradients and the scenario batching all coming from the same JAX code difflow
already runs.

```{contents}
:local:
:depth: 2
```

## When to reach for this, and when not to

Four modules in difflow answer questions about uncertainty, and they cost
wildly different amounts. Work down this list and stop at the first one that
answers your question.

| Question | Module | Cost |
|---|---|---|
| How much does my output move when the parameters do? | {doc}`difflow.uncertainty <solvers-and-utilities>` | one Jacobian |
| How much margin should I hold on a constraint? | `difflow.planning.backoff` | one Jacobian |
| Does the design I already have meet spec often enough? | `difflow.flexibility.expected_feasibility` | one sampled sweep |
| Is it feasible over the *whole* envelope, guaranteed? | {doc}`difflow.flexibility <flexibility>` | vertex enumeration |
| **What design should I build, given the distribution?** | **`difflow.stochastic`** | **an optimization per scenario set** |

Only the last row needs this module. It is the only one that *changes the
design*, and it is roughly two orders of magnitude more expensive than the
first two. The diagnostics below exist so you can find out quickly whether
that was worth paying.

## The split that is the whole model

```{math}
\min_{x}\ \rho\Big[\, F\big(x, u_s(\theta_s), \theta_s\big) \,\Big]
```

*First stage*, `x` — **here and now**. Decided once, before anything is
revealed, and lived with in every scenario. Number of stages, settler volume,
whether a scrub section exists. There is exactly one value shared by every
scenario, and that sharing **is** non-anticipativity: it is enforced
structurally, by there being one array, not by constraints that could be
written down wrongly.

*Second stage*, `u_s` — **wait and see**. Re-decided per scenario, once the
uncertain parameters are known. Operating pH, solvent-to-feed ratio, scrub
acid — the knobs an operator still has when today's assay comes back.

Getting the split wrong goes wrong in one direction far more often than the
other. Putting an operating variable in the first stage prices a plant that
cannot be adjusted, which understates its worth and over-designs it. Putting a
*design* variable in the second stage prices a plant that rebuilds itself
every shift. The second mistake is the dangerous one, because its answer looks
*better*.

### Which uncertainties admit recourse at all

Recourse is only real when the parameter is **observed** before the recourse
must be committed. This is the same distinction {doc}`flexibility` draws
between feed uncertainty and parameter uncertainty. A feed assay is measured on
arrival, so pH can genuinely be re-optimized against it. A distribution
coefficient is never revealed: it is a property of the chemistry, identical in
every campaign and unknown in all of them.

That does not make a stochastic program over distribution coefficients wrong,
but it does change what the recourse variable means. The two honest
formulations are:

1. **No recourse at all** (`recourse=None`) — the design is scored against the
   parameter distribution at a single fixed operating point. Right for an
   unobservable parameter and a plant with no feedback.
2. **Recourse on an observable proxy** — the operator does not see `D`, but does
   see the raffinate assay, and turns pH until the assay is on target. Then pH
   is genuinely second-stage, not because `D` was revealed but because its
   *consequence* was.

The module will not guess which you mean. State it.

## Quick start

```python
import difflow.stochastic as st

scen = st.ScenarioSet.from_covariance(
    ["a_Nd", "a_Dy"], mean, Sigma, n=256, seed=0)

prob = st.TwoStageProblem(
    model=circuit,                          # (x, u, theta) -> {name: value}
    first_stage={"n_stages": (2.0, 16.0),   # decided once
                 "solvent":  (3.0, 45.0)},
    recourse={"pH": (1.8, 2.8)},            # re-decided per campaign
    objective="profit", maximize=True,
    risk=("cvar", 0.9),
    constraints=[("purity", ">=", 0.80, 0.90)],
)

res = st.solve_saa(prob, scen)
print(res.summary())
print(st.bounds(prob, scen, result=res).summary())        # VSS and EVPI
print(st.check_scenario_health(prob, scen, res).summary())
```

The model is any pure JAX callable taking three dicts and returning a dict of
named scalars. A flowsheet solve qualifies.

## The sample

`ScenarioSet` is drawn once and reused for the whole solve. There is no
resample-per-iteration option, because there is no correct way to use one: an
SAA objective evaluated on a fresh sample at every iteration is not a function
but a function plus noise of the same order as the improvements the optimizer
is chasing, and a descent method on it stalls at a distance set by the noise
rather than by the tolerance. Fixing the sample makes the objective a genuine
deterministic function of `x` with an exact gradient.

Constructors, ordered by how much you actually know:

| Constructor | Use when |
|---|---|
| `ScenarioSet.from_samples(draws, names)` | you have draws — an MCMC posterior, campaign history, a vendor's lot record |
| `ScenarioSet.from_covariance(names, mean, Sigma, ...)` | you have a fit |
| `ScenarioSet.normal / .lognormal / .uniform` | you have a nominal value and a spread, independent |
| `ScenarioSet.from_uncertainty_set(T, ...)` | you have a flexibility envelope and want its stochastic counterpart |

**Prefer `from_covariance` whenever a fit exists.**
{doc}`difflow.estimation <parameter-estimation>` and
{doc}`difflow.reconciliation <data-reconciliation>` both hand back exactly the
`(mean, Sigma)` pair it wants, so the distribution the plant is designed
against is the one the data supports — correlations included. Correlation is
not a refinement here. Two distribution coefficients fitted to the same
isotherm are strongly correlated, and the difference of their logarithms is the
separation factor, which is the only thing the plant cares about. A diagonal
`Sigma` gets its variance wrong by a factor of a few, in whichever direction is
least convenient.

## What "best" means

| Risk measure | Meaning | Notes |
|---|---|---|
| `Expectation()` | risk neutral | right when many campaigns average out |
| `MeanStd(kappa)` | `E + kappa sd` | cheap and smooth, but penalizes upside too — not coherent |
| `CVaR(alpha)` | mean of the worst `1-alpha` tail | coherent, convex, sensitive to *how bad* the tail is |
| `WorstCase()` | max over the sample | the worst of `S` draws, not the worst of the distribution |

And the constraint forms:

| Constraint | Meaning |
|---|---|
| `("g", ">=", b)` → `Expected` | holds on average — right only for genuinely pooled quantities |
| `("g", ">=", b, alpha)` → `Chance` | holds with probability at least `alpha` |
| `Robust("g", ">=", b)` | holds in every scenario drawn |

A chance constraint is enforced through its CVaR surrogate,
`CVaR_alpha(residual) <= 0`, the standard convex **conservative** approximation
(Nemirovski and Shapiro). Satisfying it implies `P(satisfied) >= alpha`, but
not conversely — so a design this rejects may still be acceptable. The exact
empirical probability is printed beside it, and `summary()` says so explicitly
when a design fails the surrogate while meeting its probability. Counting
violations with an indicator instead is exact, non-convex, has zero gradient
almost everywhere, and turns the problem into a MINLP. That is deliberately
not implemented.

## Three implementation choices worth knowing about

These are not stylistic preferences. Each of them is the difference between a
converged answer and a confident wrong one.

### The Rockafellar–Uryasev auxiliary is not a decision variable

CVaR is written in the Rockafellar–Uryasev form

```{math}
\mathrm{CVaR}_\alpha(Z) = \min_t\ t + \frac{1}{1-\alpha}\,\mathbb{E}\,[(Z-t)_+]
```

whose textbook use hands `t` to the optimizer as one more variable. This module
does not. `t` lives on the scale of the objective, its useful range is not
known before the run, and a projected method takes steps proportional to the
width of the box you give it — so a generously-sized box for `t` makes the
auxiliary swing harder than the design variables and the search follows it
instead of the design. That failure is silent and looks like a converged answer
at a bound.

Instead `t` is recomputed at every iterate as the empirical value at risk — its
exact minimizer at fixed `Z` — and frozen with `stop_gradient`. By the envelope
theorem this is the **exact** gradient of CVaR, and the value reported is
exactly `min_t`. {doc}`difflow.flexibility <flexibility>` makes the same trade
for the same reason.

### An augmented Lagrangian, not a growing penalty

A plain penalty continuation fails here, and fails quietly. A projected Adam
step is normalized by the running second moment of the gradient, so one
excursion into a region where the penalty weight is 10⁴ inflates that second
moment by eight orders of magnitude and every later step shrinks to nothing.
The iterate freezes where the spike left it, reports itself feasible, and is
nowhere near the optimum. The augmented Lagrangian absorbs the constraint
multiplier into `lambda` instead of into the weight, so `rho` stays at its
modest starting value, gradients stay the size of the multiplier, and the
converged point is the exactly constrained one.
`tests/test_stochastic.py::TestSolve::test_a_growing_penalty_would_have_frozen_here`
is the regression test.

### Everything is rescaled, and the rescaling is exact

A profit in dollars per year and a purity in mole fraction differ by eight
orders of magnitude, and an unscaled penalty makes the whole problem one
constraint. Objective values and constraint residuals are affinely rescaled at
the starting point. This is not an approximation: every risk measure here
satisfies `rho(a + bZ) = a + b rho(Z)` for `b > 0` and every constraint form is
positively homogeneous in its residual, so the rescaled problem has the *same*
solution and the reported numbers unscale exactly.

Feasibility is always scored by re-evaluating the model at the candidate point,
never read off the multipliers — the same rule {doc}`planning` states for its
LP slacks.

## Was it worth solving?

Two classical numbers, and they answer different questions.

**Value of the stochastic solution** — *should I have bothered?*

```{math}
\mathrm{VSS} = \mathrm{EEV} - \mathrm{SP}
```

Solve the deterministic problem at the mean parameters, take the design it
produced, and run it — with recourse — against the real distribution. The gap
is what modelling the uncertainty bought. A VSS of zero is a completely
legitimate answer, and finding it early is the point: it says the mean-value
design was already right and you should stop paying for scenario runs.

**Expected value of perfect information** — *should I buy an instrument
instead?*

```{math}
\mathrm{EVPI} = \mathrm{SP} - \mathrm{WS}
```

EVPI compares against the unattainable plan that knows each realization in
advance, so it is an **upper bound on what any measurement, assay or online
analyzer can possibly be worth**. If EVPI is \$40k/yr, a \$200k analyzer cannot
pay back no matter how good it is. That is usually the most decision-relevant
number in the whole run.

```python
rep = st.bounds(prob, scen, result=res)
print(rep.summary())
```

Both obey `WS <= SP <= EEV` for a minimization. `BoundReport.ordered` says
whether that held; when it did not, a solve failed to converge and the gaps are
not meaningful — the report says so rather than reporting a negative EVPI as
though it meant something.

The wait-and-see bound carries the constraints across **unchanged**, which is
what makes it a bound: dropping non-anticipativity and changing nothing else
makes the two-stage feasible set a subset of it. Rewriting a chance constraint
as a per-scenario one instead would break that — it can make the relaxation
infeasible where the original was fine, since a chance constraint is precisely
the statement that some scenarios are allowed to fail.

**Recourse and information are substitutes.** A common and useful finding is
EVPI ≈ 0 with recourse and EVPI large without it: the operating loop already
absorbs what the measurement would have told you. Run the problem both ways
before buying an analyzer.

## Is the sample big enough?

`check_scenario_health` is the counterpart of `check_delta_health` in
{doc}`planning` — the things that go wrong quietly as a problem grows, each
with a number attached. It never raises and never re-solves.

* **Dead levers.** A first-stage variable whose SAA gradient is exactly zero
  cannot be moved by the optimizer, usually because a `clip`, `where` or
  `minimum` on an active path has flattened it. Identical failure to the
  delta-vector one, identical cause.
* **A tail with nothing in it.** `CVaR(0.99)` on 200 scenarios averages two of
  them. The number will be confident and meaningless. The sample size that
  matters is `(1 - alpha) S`, not `S`.
* **Saturated recourse.** A recourse variable pinned at a bound in most
  scenarios is not providing recourse — the credit the model took for it is not
  there. Usually the most actionable single number in a run: it names the knob
  to widen or the instrument to buy.
* **Sample sensitivity.** The risk value on half the sample against the whole
  sample, plus a bootstrap standard error. Differences between two designs
  smaller than that error are not real.

The formal version is `optimality_gap`, the Mak–Morton–Wood replication bound:
solve on several independent samples for a statistical lower bound, evaluate
the candidate on one large independent sample for an upper bound, and report
the gap with a confidence limit. It is the one place in the module that
deliberately redraws.

## Cost, and where it stops working

Scenarios are handled by `vmap`, which is the right answer up to the point
where the batch stops fitting in memory. The scenario dimension is nearly free
because {py:meth}`difflow.Flowsheet.solve` detects a tracer and swaps its
Anderson and Wegstein loops — Python loops that branch on the residual, and so
cannot be traced — for an optimistix fixed point carrying an
implicit-differentiation rule. A recycle solve therefore batches over scenarios
and differentiates through to the design, with the gradient coming from the
converged solution rather than from an unrolled iteration.

Two practical notes:

* Under `vmap` every scenario runs the same number of fixed-point iterations,
  with no early exit, so the cost is the worst scenario times `S`.
* The tear-solve tolerance is a floor on gradient accuracy. Tighten it before
  blaming the scenario count.

## Out of scope by design

No L-shaped or Benders decomposition, no scenario reduction, no multistage
scenario trees, no integer recourse, no distributionally robust formulations.
A decomposition belongs with a solver that can exploit it, and
{doc}`external-solvers` explains why an implicit flowsheet block cannot be
handed to an integer solver and still carry a certificate.

## Reference

- Rockafellar and Uryasev, *J. Risk* **2** (2000) 21,
  [doi:10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038) — CVaR.
- Nemirovski and Shapiro, *SIAM J. Optim.* **17** (2006) 969,
  [doi:10.1137/050622328](https://doi.org/10.1137/050622328) — CVaR as the
  convex conservative approximation of a chance constraint.
- Birge, *Oper. Res.* **30** (1982) 989,
  [doi:10.1287/opre.30.5.989](https://doi.org/10.1287/opre.30.5.989) — EVPI and
  VSS.
- Mak, Morton and Wood, *Oper. Res. Lett.* **24** (1999) 47,
  [doi:10.1016/S0167-6377(98)00054-6](https://doi.org/10.1016/S0167-6377(98)00054-6)
  — the replication gap bound.
- Birge and Louveaux, *Introduction to Stochastic Programming*, 2nd ed.,
  Springer (2011).

Example: `examples/32_stochastic_ree_separation.ipynb`.
Tests: `tests/test_stochastic.py`, `tests/ree/test_coefficient_overrides.py`.
