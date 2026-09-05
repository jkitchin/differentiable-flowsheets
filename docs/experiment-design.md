# Experiment Design and Identifiability

`difflow.estimation` fits parameters to experiments that have already been run.
This page covers the two questions that come *before* that:

1. **Can these parameters be told apart at all** from the measurements
   available? (`check_identifiability`)
2. **Which experiments should be run next**, and what confidence intervals
   would they buy? (`design_experiments`, `predicted_covariance`)

Both rest on the sensitivity matrix

$$ S_{kj} = \frac{\partial y_k}{\partial \theta_j} $$

which is one `jax.jacobian` call on a differentiable model — exact, and about
as cheap as a single model evaluation. Computing $S$ by finite differences is
what makes model-based design of experiments expensive elsewhere. The API
follows [Pyomo.DoE](https://doi.org/10.1002/aic.17813) (Wang & Dowling, *AIChE
J.* **68** (2022) e17813), which does the same thing for Pyomo models; see
Franceschini & Macchietto, *Chem. Eng. Sci.* **63** (2008) 4846,
[doi:10.1016/j.ces.2007.11.034](https://doi.org/10.1016/j.ces.2007.11.034) for
the wider method.

## Table of Contents

1. [The ordering is not optional](#the-ordering-is-not-optional)
2. [Candidate experiments](#candidate-experiments)
3. [Structural identifiability](#structural-identifiability)
4. [Fisher information](#fisher-information)
5. [Design criteria](#design-criteria)
6. [Choosing a run list](#choosing-a-run-list)
7. [Predicted confidence intervals](#predicted-confidence-intervals)
8. [Numerical notes](#numerical-notes)
9. [Worked example](#worked-example)
10. [Related tools: `discopt-doe`, and sensor placement](#related-tools-discopt-doe-and-sensor-placement)
11. [Limitations](#limitations)

> **difflow is not the only place design of experiments lives, and for many
> problems it is not the right one.** The `discopt-doe` plugin is a far larger
> DoE package — continuous design optimization, profile likelihood, model
> discrimination, classical and screening designs, Bayesian optimization. What
> is *here* is the narrow thing it cannot do: design against a model that only
> exists as a differentiable JAX function, such as a difflow flowsheet. See
> [Related tools](#related-tools-discopt-doe-and-sensor-placement) before you
> start.

---

## The ordering is not optional

**Check identifiability first.** If the sensitivity matrix is rank deficient —
if some direction $v$ in parameter space satisfies $Sv = 0$ — then moving the
parameters along $v$ changes no prediction. The Fisher information
$S^T\Sigma^{-1}S$ is singular, the covariance is infinite along $v$, every
design criterion is degenerate, and the fitted parameters you get back are
whatever the optimizer happened to stop at.

More data does not fix this. Additional experiments add rows to $S$, and every
new row is orthogonal to $v$ by the same structural argument, so the null space
survives any amount of data. The fixes are structural:

- **measure something else** — a new kind of measurement that responds to $v$,
- **reparameterize** on the identifiable combination (fit $k = AB$ rather than
  $A$ and $B$),
- **fix** one of the parameters from independent knowledge.

Because it is so easy to skip this step and then spend a month collecting data
that cannot possibly answer the question, `design_experiments` and
`predicted_covariance` run the rank test themselves and raise
`IdentifiabilityError` rather than returning a design built on a singular
matrix. You can pass `require_identifiable=False` to inspect a degenerate case
deliberately; you then get infinite standard errors, which is the honest answer.

```python
from difflow.estimation import (
    Experiment, check_identifiability, design_experiments, predicted_covariance,
)

report = check_identifiability(model_fn, theta, candidates)   # 1. structural
print(report.summary())
report.raise_if_unidentifiable()

design = design_experiments(model_fn, theta, candidates, n=8) # 2. which runs
ci = predicted_covariance(model_fn, theta, design.selected)   # 3. what they buy
# ... run the experiments, then Estimator.fit(), and iterate.
```

## Candidate experiments

`Experiment` doubles as the record of a run and as a *candidate* for a run that
has not happened yet. Everything the Fisher information needs — the inputs,
which outputs are measured, and their 1-sigma uncertainties — is known before
the run; only the measured values are not, and the FIM does not depend on them.

```python
pool = [
    Experiment.candidate({'T': T, 'pH': pH}, measures=['C_aq', 'C_org'],
                         uncertainties={'C_aq': 1e-4, 'C_org': 1e-4},
                         name=f'T{T:.0f}-pH{pH:.1f}')
    for T in (298.0, 318.0, 338.0)
    for pH in (1.0, 2.0, 3.0, 4.0)
]
```

`candidate` leaves `observed` empty and sets `measured`; `measured_names` falls
back to the keys of `observed` for a recorded experiment, so both kinds flow
through the same functions. `sigma_array` gives the 1-sigma vector, defaulting
to 1.0 for any output with no stated uncertainty.

## Structural identifiability

```python
report = check_identifiability(model_fn, theta, experiments, param_names=None,
                               rank_tol=None, scale='theta')
```

Linearizes the model at `theta` and takes the numerical rank of the (weighted,
column-scaled) sensitivity matrix by SVD. `IdentifiabilityReport` carries:

| field | meaning |
|---|---|
| `identifiable` | `rank == n_params` |
| `rank`, `n_params`, `n_obs` | the counts behind the verdict |
| `singular_values` | the full spectrum, so a marginal case can be looked at |
| `rank_tol`, `rank_gap` | the threshold used, and how clean the gap is |
| `condition_number` | $s_\max/s_\min$; large means *practically* unidentifiable |
| `unidentifiable` | parameters implicated in a null-space direction |
| `null_space`, `combinations` | the offending directions, and a readable rendering |
| `reason`, `summary()`, `raise_if_unidentifiable()` | diagnosis and enforcement |

Columns are scaled by $|\theta_j|$ by default, so the test is about *relative*
sensitivity and does not change when a parameter is re-expressed in different
units. That is the same equilibration
`difflow.reconciliation.structure.classify` applies before its rank test, and
the SVD-based rank routine is literally reused from there: observability of a
reconciliation problem and identifiability of a parameter set are the same
linear-algebra question asked of different Jacobians.

Two classic failures, both detected:

```python
def product(theta, exp):     # A and B only ever appear as A*B
    return {'y': theta['A'] * theta['B'] * exp.inputs['x']}

report = check_identifiability(product, {'A': 2.0, 'B': 3.0}, pool)
report.identifiable        # False
report.unidentifiable      # ['A', 'B']
report.combinations        # ['- 0.707*A + 0.707*B ~ 0']
```

A full-rank report with a huge `condition_number` is a different diagnosis:
the parameters are separable in principle, but only weakly, and *that* is the
case experiment design is for.

## Fisher information

$$ \mathrm{FIM}(\theta) = \sum_i S_i^T \Sigma_i^{-1} S_i $$

with $\Sigma_i$ the diagonal measurement-error covariance built from each
experiment's `uncertainties`.

```python
fim = fisher_information(model_fn, theta, experiments, prior_fim=None)
```

Two properties do all the work:

- **It does not depend on the measured values**, only on the conditions, the
  model and `theta`. So a campaign can be scored before it runs.
- **It is additive over experiments.** The FIM of any subset is the sum of the
  per-experiment contributions, which is what makes greedy selection and
  exchange cheap: each is computed once and then added and subtracted.

`prior_fim` folds in information already in hand — from previously run
experiments, or a Bayesian prior precision. `design_experiments` also accepts
`existing=[...]` experiments directly.

## Design criteria

`inv(FIM)` is the asymptotic parameter covariance, i.e. the confidence
ellipsoid. Each criterion is a different scalar summary of that ellipsoid:

| criterion | value | direction | geometry |
|---|---|---|---|
| `'D'` | $\log\det \mathrm{FIM}$ | maximize | shrink the ellipsoid's volume |
| `'A'` | $\mathrm{tr}(\mathrm{FIM}^{-1})$ | minimize | shrink the average axis (sum of variances) |
| `'E'` | $\lambda_{\min}(\mathrm{FIM})$ | maximize | shrink the longest axis (worst direction) |
| `'ME'` | $\lambda_{\max}/\lambda_{\min}$ | minimize | round the ellipsoid out (conditioning) |

`design_criterion(fim, criterion)` returns the conventional value, so `'D'` and
`'E'` are better when larger and `'A'` and `'ME'` better when smaller.

**D-optimality is the default**, because it is the only one of the four that is
invariant to rescaling the parameters. A- and E-optimality compare variances of
quantities with different units, so switching a rate constant from s⁻¹ to h⁻¹
can change the design they recommend. Use `'A'` when the parameters really are
commensurate and you care about the total variance; `'E'` when one poorly
determined direction is the whole problem; `'ME'` when correlated parameters
are making the fit ill-conditioned.

## Choosing a run list

```python
design = design_experiments(model_fn, theta, candidates, n=8, criterion='D',
                            method='greedy', replace=True,
                            existing=None, prior_fim=None)
print(design.summary())
```

- **Greedy** (default) repeatedly adds the candidate that most improves the
  criterion given everything already chosen. For D-optimality this is the
  standard construction and is usually within a few percent of the optimum.
- **`method='exchange'`** then runs Fedorov-style swaps: try replacing each
  selected run with each unselected one, keep the best improvement, repeat.
  It costs `n * n_candidates` evaluations per sweep and never returns a worse
  design than the greedy start.
- **`replace=True`** (default) allows replicates. Replicating an informative
  condition is often genuinely optimal — for a straight line the D-optimal
  design is half the runs at each end of the range, and nothing in between.

`DesignResult` carries `selected`, `indices`, `fim`, `covariance`,
`std_errors`, `criterion_value`, `criterion_history` (so the point of
diminishing returns is visible), the `identifiability` report, and `summary()`.

The design is **local**: it is computed at the current `theta`, and a different
`theta` can give a different design. That is not a defect of the method, it is
why the loop is *design → run → refit → design again*.

## Predicted confidence intervals

```python
ci = predicted_covariance(model_fn, theta, proposed_campaign, alpha=0.05)
ci.std_errors, ci.ci_lower, ci.ci_upper, ci.correlation
```

This returns the same `ConfidenceResult` type as
`Estimator.confidence_intervals`, so the intervals you predicted and the
intervals you achieved can be compared field by field. The one difference is
where the error model comes from: `fisher_confidence_intervals` estimates the
residual variance from data that exist, while `predicted_covariance` takes the
`uncertainties` declared on each experiment as the assumed error model — the
only thing available before the run.

The intervals are a linearization about `theta`. For a nonlinear model they are
exact only to the extent the model is locally linear over the interval, and they
are only as good as the `theta` used. They are, in the authors' experience,
still the right thing to put in front of an experimental collaborator: a ranked
run list with the confidence-interval shrinkage attached is a concrete,
checkable claim, and the check is one refit away.

## Numerical notes

- **Log-determinant via Cholesky**, never `det`. For an $n \times n$ FIM the
  determinant scales like the $n$-th power of the information and overflows or
  underflows long before its logarithm does. `log_det(fim)` equilibrates the
  matrix by its own diagonal, $M = D\tilde{M}D$ with
  $D = \mathrm{diag}(\sqrt{M_{ii}})$, and returns
  $2\sum\log d + 2\sum\log\mathrm{diag}(\mathrm{chol}(\tilde M))$. The
  equilibration is exact, and it keeps every logarithm on a sane scale when
  the parameters differ by ten orders of magnitude — a pre-exponential factor
  and an activation energy, say, which is the normal case, not a pathology.
- **A singular FIM is not an error.** `log_det` returns `-inf`,
  `trace(FIM^-1)` returns `+inf`, $\lambda_\min$ returns 0 and the condition
  number `+inf`. Those are the correct limits: infinite variance in some
  direction.
- **Cholesky success is not a rank test**, and neither is the relative size of
  its pivots. A FIM that is singular in exact arithmetic often factors anyway
  with a tiny pivot; $\begin{bmatrix}1 & 1-2\epsilon\\ 1-2\epsilon & 1\end{bmatrix}$
  factors cleanly and
  `slogdet` reports a perfectly finite $-35.35$. Worse, in a badly scaled FIM
  the pivots inherit the spread of the diagonal, so the small one is not small
  *relative to the largest* and a rank-deficient design is scored as merely
  mediocre. Singularity is therefore decided on the **spectrum**: an
  eigenvalue below $n\,\epsilon\,\lambda_\max$ is zero.
- **The two rank tolerances agree by construction.** `check_identifiability`
  calls a singular value of $S$ zero below $\sqrt{\epsilon}$ times the largest,
  following `difflow.reconciliation.structure`. The FIM's eigenvalues are the
  *squares* of those singular values, so the matching threshold here is
  $\epsilon$, not $\sqrt{\epsilon}$, and the two tests then flag exactly the
  same degenerate problems. Using $\sqrt{\epsilon}$ on the FIM would be an
  $\epsilon^{1/4}$ test on $S$: an informative but ill-conditioned design —
  an Arrhenius pair correlated to one part in $10^5$ — would be called
  singular, and greedy selection would walk away from the only pairs in the
  pool that determine both parameters.
- **Selection under a singular FIM.** Early in a greedy selection, fewer runs
  have been chosen than there are parameters, so the FIM is *always* singular
  and the criterion is `-inf` for every candidate. Selection therefore ranks on
  the pair `(rank, pseudo-value)`, where the pseudo-value applies the criterion
  to the nonzero eigenvalues only: information is first added in as many
  independent directions as possible, and only then optimized. Once the FIM is
  nonsingular this is exactly the criterion, so nothing changes for the picks
  that matter.
- **float64** throughout, as everywhere in difflow.
- `fisher_information`, `sensitivity_matrix` and `design_criterion` are
  `jit`- and `grad`-safe in `theta`, so the D-criterion can itself be the
  objective of a continuous design optimization over the input space.

## Worked example

A straight line, where the answer is known analytically — the D-optimal design
puts half the runs at each end of the range:

```python
import numpy as np
from difflow.estimation import Experiment, design_experiments, predicted_covariance

def model(theta, exp):
    return {'y': theta['a'] * exp.inputs['x'] + theta['b']}

pool = [Experiment.candidate({'x': float(x)}, ['y'], {'y': 0.5})
        for x in np.linspace(0.0, 10.0, 11)]

design = design_experiments(model, {'a': 2.0, 'b': 1.0}, pool, n=8)
sorted(e.inputs['x'] for e in design.selected)
# [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]

ci = predicted_covariance(model, {'a': 2.0, 'b': 1.0}, design.selected)
ci.std_errors      # what those eight runs would buy
```

Those eight runs buy `std_errors == {'a': 0.0354, 'b': 0.25}`. Eight runs bunched
in the middle of the range instead (`pool[2:10]`) give `0.0772` on the slope —
2.2 times worse, for the same eight runs and the same cost. The gap widens as
the pool widens, because the endpoints move apart and the middle does not.

The motivating case is rare-earth solvent extraction, where each equilibrium
experiment takes days, the design space (pH, temperature, extractant loading) is
continuous, and a mechanistic model already exists. Handing an experimental
collaborator a ranked run list with predicted confidence intervals attached is
exactly what this module is for.

## Related tools: `discopt-doe`, and sensor placement

### Where design of experiments actually lives

Most of the design-of-experiments capability in this ecosystem is **not** in
difflow. It is in [`discopt`](https://github.com/jkitchin/discopt) and its
`discopt-doe` plugin, and if your model can be written in `discopt`'s modeling
language then `discopt-doe` is the better tool by a wide margin.

`discopt.estimate.estimate_parameters` already does weighted least-squares
estimation and returns the parameters, the covariance **and** the Fisher
information — `_compute_estimation_fim` builds `FIM = J^T (N Σ⁻¹) J` by
`jax.jacobian`, with `N` the per-response replication counts taken from the
data. `discopt-doe` builds the design layer on top of that matrix.

| question | `difflow.estimation` | `discopt.doe` (the `discopt-doe` plugin) |
|---|---|---|
| what model can it take? | any JAX-differentiable `model_fn(theta, exp)` — including a difflow flowsheet with recycles | a `discopt.estimate.Experiment` whose `create_model` builds a symbolic `discopt.modeling` DAG (plus a sympy string-expression path and a numpy basis-row path for models linear in the parameters) |
| FIM | `fisher_information` (diagonal Σ, `prior_fim`) | `compute_fim` → `FIMResult` (diagonal Σ, `prior_fim`, autodiff or finite-difference Jacobian) |
| criteria | `'D'`, `'A'`, `'E'`, `'ME'` | `DesignCriterion.D_OPTIMAL / A_OPTIMAL / E_OPTIMAL / ME_OPTIMAL` |
| how a design is found | **pick `n` runs from a candidate list** — greedy, or Fedorov exchange | **optimize over a continuous design box** — random multi-start then `L-BFGS-B` (`SLSQP` with constraints): `optimal_experiment`, `batch_optimal_experiment` (greedy / joint / penalized, with a `min_distance` diversity penalty) |
| identifiability | `check_identifiability` — SVD rank test, null-space directions, condition number | `check_identifiability` (rank/condition) **and** `diagnose_identifiability` — the full Belsley–Kuh–Welsch bundle: condition indices, VIF, variance-decomposition proportions, Gutenkunst sloppy-model spectrum |
| estimability (which parameters to fit at all) | — | `estimability_rank` (Yao orthogonalization), `collinearity_index`, `d_optimal_subset` |
| profile likelihood | — | `profile_likelihood`, `profile_all` |
| model discrimination | — | `discriminate_design`, `discriminate_compound`, `sequential_discrimination`, `model_selection`, `likelihood_ratio_test`, `vuong_test` |
| sequential campaigns | write the loop yourself (`existing=`, `prior_fim=`) | `sequential_doe` — estimate, use the FIM as the next round's prior, design, run, repeat |
| classical / screening designs | — | `factorial_2level_design`, `fractional_factorial_design`, `central_composite_design`, `box_behnken_design`, `latin_hypercube_design`, Latin/Graeco-Latin squares, `anova_report` |
| mixture / constrained design | — | `DesignConstraint`, `sum_constraint`, simplex sampling and projection, Scheffé templates |
| active learning / BO | — | `optimize_round` (GP surrogate, expected improvement / confidence bound, Sobol candidates), Excel `Workbook` campaigns, a CLI and a GUI |
| criterion is itself `jit`/`grad`-safe in θ | yes | no — the design optimizer drives the FIM criterion with scipy's finite differences |
| exploring a grid of conditions | build the pool and read `criterion_history` | `explore_design_space` |

Neither package does **approximate (continuous-weight) design** with the
associated equivalence-theorem check, **Bayesian or minimax-robust design over a
parameter prior**, or a **non-diagonal measurement covariance** in the FIM.

### Which one to reach for

**Reach for `discopt-doe`** whenever the model can live in `discopt.modeling` —
which is most algebraic kinetics, thermodynamics and response-surface work. Also
reach for it, regardless of the model, when the design space is a continuous box
and you want the optimizer to *find* the conditions rather than pick from a grid
you discretized by hand, and for anything in the bottom two-thirds of that table:
profile likelihood, model discrimination, estimability, screening, ANOVA,
Bayesian optimization.

**Reach for `difflow.estimation`** when the model is a difflow flowsheet or any
other JAX callable — a recycle loop, an implicit solve, a `diffrax` integration —
that cannot be re-expressed symbolically; when the design space genuinely *is* a
finite list of runs (a plate layout, the temperatures the rig can hold, a set of
feed drums you already own), so candidate selection with replicates is the right
combinatorics and exchange is the right algorithm; or when you want to
differentiate the design criterion itself.

**You cannot hand a difflow flowsheet to `discopt-doe` today.** `discopt` models
are symbolic expression DAGs — JAX is a *backend* it lowers to
(`discopt.parametric.compile_expression`), not an entry point — and there is no
black-box function hook, so a JAX flowsheet has no way in. Building that bridge
is difflow issue #203; until it exists the two paths do not meet, and this page
is the difflow-native one.

Two practical notes:

- `discopt-doe` is a **separate distribution** (`pip install discopt-doe`) that
  installs into the `discopt.doe` namespace. It is *not* a difflow dependency and
  is not installed with difflow. Because it is a namespace package, `import
  discopt.doe` can succeed and give you an empty module when the plugin is
  absent; check by importing a name (`from discopt.doe import compute_fim`).
- Both packages export a `check_identifiability`. They are different functions
  with different signatures — difflow's takes a `model_fn` and a list of
  `Experiment`s, discopt's takes a `discopt` `Experiment` object. Import them by
  qualified module if you use both.

### Sensor placement is the other half of the question

`design_experiments` chooses **conditions**: which runs to make, at what
temperature and pH. The complementary question — **which quantity to measure** —
is answered in two other places, and it is worth knowing which one you are in:

- `difflow.reconciliation.design.sensor_value` and `sensor_ranking` rank
  candidate *sensors* on a plant by how much each would shrink the reconciled
  standard deviation of a target quantity. Same "what would this buy me before I
  buy it" logic, run through the reconciliation covariance rather than the FIM,
  and the decision variable is an instrument rather than a run. See
  [Data Reconciliation](data-reconciliation.md).
- On this page, the measurement-set question is `Experiment.candidate`'s
  `measures` argument. When `check_identifiability` fails, the fix is a *new kind
  of measurement*, and you test a proposed one by rerunning the check with it
  added to `measures` — see [Structural identifiability](#structural-identifiability)
  above, where measuring `z` as well as `y` turns an unidentifiable `A*B` model
  into an identifiable one. That is the cheapest useful thing in this module:
  it is a rank test, it costs one Jacobian, and it can save a campaign.

## Limitations

- **Candidate-pool selection only.** The design space must be discretized into
  candidates; there is no continuous optimization over the input space. The
  criterion is differentiable in `theta`, but the selection itself is
  combinatorial. (`jax.grad` through the criterion w.r.t. *inputs* is possible
  if the model is written to expose them, but no driver is provided.)
  `discopt.doe.optimal_experiment` optimizes over a continuous design box
  instead — use it when the conditions are continuous and the model can be
  written in `discopt.modeling`.
- **Local design.** Everything is evaluated at one `theta`; robust and
  Bayesian-average designs over a parameter distribution are not implemented.
  A cheap approximation is to design at several plausible `theta` values and
  take the runs the designs agree on.
- **Diagonal $\Sigma$.** Measurement errors are assumed independent, with the
  1-sigma values declared per output. Correlated measurement error is not
  supported.
- **Structural identifiability is tested numerically**, at one `theta`, by
  rank. That is local structural identifiability, not the global symbolic
  result a differential-algebra tool would give. In exchange it works on any
  model that JAX can differentiate, including a flowsheet with recycles.
- **No cost model.** Candidates are ranked purely by information, so a run list
  cannot yet trade information against the cost or duration of a run.
- **One model at a time, and no profile likelihood.** There is no model
  discrimination criterion (which experiment best tells two rival mechanisms
  apart) and no likelihood-profile confidence interval, so the intervals here
  are always the linearized ones. Both live in `discopt-doe`
  (`discriminate_design`, `profile_likelihood`) — see
  [Related tools](#related-tools-discopt-doe-and-sensor-placement).
- **No estimability ranking.** When the parameters are identifiable but badly
  correlated, this module tells you *that* (the condition number) but will not
  choose a subset to fit and fix the rest; `discopt.doe.estimability_rank` and
  `d_optimal_subset` will.
