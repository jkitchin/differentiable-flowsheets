# Spike #440: reformer as one collocation system, solved by pounce

Status: measurement spike, not a production change. Not in the book toctree
(`_config.yml` excludes `docs/spikes/**`). Code: `spikes/440_collocation_pounce/`
(scripts and logs' commands), `src/difflow/dynamic/collocation.py` (the one
reusable piece, with `tests/test_collocation.py`). The existing reforming code
is untouched.

**Question.** Can `CatalyticReformer` be one equation-oriented system (each
bed's ODE by Radau collocation on finite elements, separator / stabilizer /
recycle as algebraic equations), handed to pounce through `difflow.solvers`, with
lever derivatives from pounce's KKT implicit-function backward
(`pounce.jax.JaxProblem`), instead of diffrax inside a recycle fixed point with
a dense forward-mode Jacobian?

**Answer.** Yes, it works and is faster, but it is reverse-mode only and its
accuracy is set by a hand-tuned mesh. Recommendation at the end: optional
alternate solver, not the default.

All timings: 4-core Xeon shared with another agent's heavy JAX jobs, each
process pinned to 2 cores (`taskset -c 0,1`), venv with `pounce-solver[jax]`
0.12.0, JAX 0.11.2, float64. Repeats are 3 where stated; "best (spread)" is the
min and the range of the repeats. Expect the noise to be large: the same
`jit(jacrev)` measured 8.6 s at WAIT 500 C and 15.2 s at 505 C in one process.

**Reference point.** The numbers quoted in the issue (RON 95.79937555, net H2
144.03648238, Jacobian `[[0.630373],[1.15690654]]`) are the values at
**WAIT = 505 C** (the "5 C from the stored tear" point), not 500 C. At 500 C the
base reformer gives RON 92.50122625 and net H2 137.70503809. All comparisons
below are at 505 C, lean naphtha, default `ReformerParams`.

## What was built

* `difflow.dynamic.collocation`: right-Radau nodes, differentiation matrix,
  quadrature weights, graded element edges, `collocation_residual(f, y0, Y, mesh)`
  (continuity is automatic: the last Radau node is the element end). Tested
  against analytic ODEs (exp decay, oscillator: end-point order 5 for K = 3, a
  stiff problem with element width >> 1/1000, a graded mesh resolving an inlet
  layer, gradient w.r.t. a parameter). 14 tests pass.
* `eo_bed.py`: one bed, same rate model and enthalpy basis as `reactor.py`;
  31 unknowns per collocation point (29 fractions, coke, T); T is an algebraic
  unknown tied to the state by the conserved enthalpy (no Newton inside the rhs).
* `eo_reformer.py`: three beds + `EOSFlash.eo_residuals` (PR fugacity equality)
  for the separator + the H2/HC recycle closure; recycle `R`, separator liquid and
  vapour are unknowns; stabilizer, compressor and heater duties are explicit
  post-processing through the existing unit classes, so
  `ReformerResult.outputs()` is reused unchanged. Square system, zero objective,
  `p = [WAIT (C), feed rate (kg/s), H2/HC]`.

`as_nlp` could not be used: it needs every unit to expose `eo_residuals` and has
no place for a bed's internal collocation states, and the reactor, heater and
stabilizer have none. `f`, `g` and `Bounds` are built by hand and handed to
`difflow.solvers.pounce_problem` / `differentiable_problem`, which never let
pounce probe sparsity. Jacobian pattern: `asdex` (global, graph-derived).
Lagrangian-Hessian pattern: `asdex` raises `NotImplementedError: No handler for
primitive 'one_minus_square'` on this model, so the pattern is the repo's own
structural superset (outer products of the Jacobian row supports). Two further
settings were forced:

* `sparse=False` (dense forward Jacobian): `RESOURCE_EXHAUSTED`, 22.6 GB
  allocation. `sparse=True` with the supplied patterns works.
* exact Lagrangian Hessian (colored HVPs): `RESOURCE_EXHAUSTED`, 74.5 GB
  allocation. `hessian_approximation=limited-memory` works, and for a square
  system with a zero objective pounce converges in 4 iterations anyway.

## Stage 1: one bed, accuracy

Reference: `integrate_bed(rtol=1e-11)`, the three beds at the converged inlets
of the lean-naphtha reformer at WAIT 480, 500 and 530 C (9 beds). Metric: worst
relative error over species holding more than 0.1 % of the bed feed, worst
absolute error over the total feed, worst outlet temperature error. Tolerances
the refinery tests use: rel 1e-4 (long-bed equilibrium quotient), rtol 1e-5
(implicit gradients vs central differences), abs 1e-3 on RON
(`wait_for_ron`), 1e-8 on balances, rtol 1e-7 (block vs solve).

| mesh (K = 3, geometric grading g) | unknowns / bed | worst rel F | worst abs F/F0 | worst dT (K) |
|---|---|---|---|---|
| NE 4, g 2 | 372 | 4.0e-3 | 9.7e-5 | 5.1e-2 |
| NE 8, g 2 | 744 | 9.3e-4 | 1.0e-5 | 5.0e-3 |
| NE 12, g 2 | 1116 | 1.4e-4 | 3.7e-6 | 3.2e-3 |
| NE 12, g 1.8 | 1116 | 1.0e-4 (9.96e-5) | 1.8e-6 | 1.6e-3 |
| NE 16, g 1.5 | 1488 | 1.3e-4 | 1.7e-6 | 8.3e-4 |
| NE 20, g 1.4 | 1860 | no number: bed-1 Newton from the flat start returned NaN at all three WAITs | | |
| NE 8 / 12 / 16, g 3 | 744-1488 | 6.2e-4 (plateau) | 2.4e-5 | 2.1e-2 |

Smallest mesh inside 1e-4 on every significant species: NE 12, K 3, g 1.8 (1116
unknowns per bed, 9.96e-5, marginal). NE 16 / g 1.5 is just outside on the worst
species (1.28e-4, bed 1 at 530 C) although its answer-level error is smaller.
NE 20 / g 1.4 could not be assessed at bed level (see below). The error lives
in bed 1 (the inlet layer, relaxation rate of a few hundred per unit w) and, for
strongly graded meshes, in the coarse tail elements (g 3 plateaus because the last
element is huge). K = 4 and 5 and hybrid geometric-then-uniform meshes were not
better (answer-level errors of about +3e-3 RON, below). The collocation system also
needs a sensible start: damped Newton from a flat inlet profile took 6-31
iterations when it converged, and **failed** (60 iterations, or NaN) for bed 1 in
NE 6 g 2 at 530 C, in a hybrid NE 11 mesh at 480 C, in a K 4 hybrid mesh at 480
and 530 C, and for NE 20 g 1.4 at all three WAITs. NE 12 g 1.8 and NE 16 g 1.5
converged in all 9 beds.

Cost of the evaluations that matter (one bed, jit; compile = first call):

| | NE 12 | NE 16 |
|---|---|---|
| residual, warm | 0.5 ms | 0.4 ms |
| dense `jacfwd` (all tangents), warm / first | 1.27 s / 3.7 s | 1.99 s / 4.4 s |
| diffrax `integrate_bed` warm / first | 60 ms / 6.9 s | 61 ms / 6.5 s |
| diffrax bed `jacfwd` (29 inlet tangents) warm / first | 5.8 s / 21 s | 5.3 s / 22 s |

(The dense bed Jacobian is not how pounce uses it; it uses the colored sparse
one. It is shown because it is what a stand-alone bed Newton would pay.)

## Stage 2: whole flowsheet, solved

NE 12 / g 2: n = m = 3435 (3 x 1116 + 87), Jacobian nnz 305,660 (2.6 %),
Hessian superset nnz 443,241 (7.5 % of the lower triangle), KKT dimension 6870.
NE 16 / g 1.5: n = m = 4551, Jacobian nnz 405,056, Hessian superset 590,733.

pounce (`tol 1e-9`, L-BFGS Hessian, sparse colored AD): **4 iterations,
`Solve_Succeeded`**, `max|g|` ~1e-13, from either start below. Warm solve (NE 12
/ NE 16): 4.9-5.4 s / 5.8-6.4 s. First call (compile): 19-22 s. Patterns by
`asdex`: 22-29 s (once). Problem build: 7.6-12.6 s.

Starts: "warm" = the existing base solution (500 C) moved to 505 C; "cold" = one
sequential-modular pass from `initial_recycle` (diffrax beds + flash). Both
converge in 4 iterations. A **flat** start (every collocation point at the bed
inlet) did not converge: 94 iterations, CPU limit of 120 s hit, `max|g|` 68.

Products at 505 C against the existing solver (RON 95.79937555, H2 144.03648238),
all outputs of `ReformerResult.outputs()` compared:

| mesh (EO minus existing) | RON error | net H2 rel error | worst rel error, any output |
|---|---|---|---|
| NE 12, g 2 (3435) | -1.21e-3 | -1.9e-5 | 5.5e-5 (rx3.dT) |
| NE 16, g 2 (4551) | -1.21e-3 | -1.9e-5 | 5.5e-5 |
| NE 12, g 1.8 (3435) | -5.7e-4 | -9.8e-6 | not recorded |
| NE 14, g 1.7 (3993) | -3.7e-4 | -6.2e-6 | not recorded |
| NE 16, g 1.5 (4551) | -7.6e-5 | -2.1e-6 | not recorded |
| NE 20, g 1.4 (5667) | -4.1e-5 | -8.0e-7 | not recorded |
| K 3, geometric to 0.1 then uniform (3714) | +3.2e-3 | -3.2e-5 | not recorded |
| K 4, NE 10, g 1.6 (3807) | +3.0e-3 | -2.9e-5 | not recorded |
| K 5, NE 8, g 1.7 (3807) | +3.7e-3 | -3.6e-5 | not recorded |

So the **RON tolerance of 1e-3 needs NE >= 12 with g <= 1.8**, and the answer
converges as the mesh refines only if both the inlet layer and the tail are
resolved (NE 12 to 16 at g 2 changes nothing: the last element stays 0.5 wide).
The existing solver is not the source of the gap: its RON is identical to 8
digits at reactor `rtol` 1e-8, 1e-10 and 1e-12, and at the existing solution the
EO flash rows are satisfied to 1.4e-8 and the recycle closure to 5e-12.
Balances at the EO solution: mass, C, H about 2e-14, energy 7e-11, each
adiabatic bed 1e-15 (the "closes to round-off" invariant holds as a solver
tolerance rather than by construction).

## Stage 3: derivatives

`pounce.jax.JaxProblem` (`differentiable_problem`, `factor_reuse=True`), outputs
`[reformate.RON, h2.net_mol_s]`, levers `[WAIT, feed.rate, H2/HC]`.

WAIT column of the Jacobian at 505 C:

| source | dRON/dWAIT | dH2/dWAIT |
|---|---|---|
| existing implicit (baseline `jacfwd`) | 0.630373 | 1.15690654 |
| existing, central FD (h 0.02 C) | 0.630372964 | 1.15690635 |
| EO, `jacrev` through pounce, NE 12 g 2 | 0.630377133 | 1.15693606 |
| EO, central FD of the pounce solve, NE 12 g 2 | 0.630377074 | 1.15693578 |
| EO, `jacrev`, NE 16 g 1.5 | 0.63037947 | 1.1568954 |
| EO, KKT sensitivity (`solve_with_jacobian`), 500 C | 0.68552758 | 1.36149217 |

`jacrev` agrees with its own FD to 9e-8 and 2e-7 relative (the KKT backward is
right), and with the existing implicit Jacobian to 7e-6 / 2.5e-5 (NE 12, g 2) and
1e-5 / 1e-5 (NE 16, g 1.5): the residual gap is the discretisation. The other
columns at 500 C (NE 12, g 2): feed rate, H2 13.7702 vs existing FD 13.7705
(2e-5), RON 0 (exact, scale invariance); H2/HC, RON 1.48029 vs existing FD
1.48006 (1.6e-4) and H2 2.76474 vs 2.76404 (2.5e-4), so the H2/HC derivative is
the least accurate and was not rechecked on the finer mesh.

What composes (levers to products, `fun(p) = outputs(jp.solve(p, z0), p)`):

| transform | result |
|---|---|
| eager call, `jax.jit` | works (jit pounce solve is a host callback) |
| `jax.jacrev`, `jit(jacrev)`, `jax.grad`, `jit(grad)` | work |
| `jax.jacfwd`, `jax.jvp` | fail: `can't apply forward-mode autodiff (jvp) to a custom_vjp function` |
| `jax.vmap(fun)`, `vmap(jacrev)` | fail: `NotImplementedError: vmap is only supported for the pure_callback primitive when vmap_method is one of ...; Got vmap_method=None` |
| `jax.hessian` / second order | fails: `Pure callbacks do not support JVP` |
| `jp.vmap_solve(p_batch, x0)` | works (`lax.map`, sequential): 11.9 s for 2 points |
| `factor_reuse=False` (dense KKT backward) | `RESOURCE_EXHAUSTED`, 150 GB, at n+m = 9102 |
| `solve_with_jacobian` | works, 135 s for one lever column at n = 3435 |

`difflow.planning.Block` takes `ad_mode="rev"`, but `planning.piecewise` and
`planning.attribution` `vmap` the block, and `Chain(... modes="fwd")` /
`AD_MODES` in `difflow_refinery.plant` record the reformer as forward-only.

## Results table (this spike vs baseline)

Baseline = `origin/main` in this venv, 2 cores, `ReformerPlanningModel`, WAIT
lever, 505 C (reproduces the issue: first `jit(jacfwd)` 330 s, warm 43-46 s).
The other agent's linearize-and-solve fix is NOT in this baseline, and was not
measured.

| | baseline (diffrax + recycle) | spike NE 12 g 2 | spike NE 16 g 1.5 |
|---|---|---|---|
| unknowns | recycle tear 31 (beds inside diffrax) | 3435 | 4551 |
| primal eval, eager warm | 11.1-12.0 s | 4.9-5.4 s (pounce solve) | 5.8-6.4 s |
| primal eval, `jit` first / warm | 69.9 s / 16.5-17.4 s | 7.1 s / 5.8-5.9 s | not run |
| Jacobian first call (compile) | 330.1 s (`jacfwd`) | 23.9 s, but after an eager `jacrev` had warmed it (that first eager call: 77 s) | 41.4 s (fresh process, includes one solve) |
| Jacobian warm, best (spread) | 43.5 (43.5-45.5) s | 8.6 (8.6-15.4) s | 14.4 (14.4-19.5) s |
| peak RSS | 5.4 GB (`jacfwd` process) | 3.0 GB (solve only, with the existing solves) | 3.6 GB solve; 6.1 GB after `jacrev` |
| RON error vs existing | - | -1.2e-3 (fails 1e-3) | -7.6e-5 |
| transforms | `jacfwd` | `jit`, `jacrev`, `grad` | same |

The spread of the warm Jacobian is dominated by machine load, so the honest
statement is "about 3-5 times faster warm, about 8-14 times faster to first
call", not a precise factor (the NE 12 first-call figure is not a cold one). Memory is not improved. Per-cotangent cost of the
backward was only measured for 2 outputs; the default planning block has 15.

## Recommendation: adopt only as an optional alternate solver (not the default)

For: the compile wall (330 s) and the 17 s primal go away; the solve is 4 pounce
iterations; balances close; answer-level agreement 1e-5 to 1e-4 with a suitable mesh;
the `factor_reuse=True` backward ran at n + m = 9102.

Against (each measured above): reverse-mode only, no `vmap` or second order, so
the planning-block defaults (`ad_mode="fwd"`), `planning.piecewise`,
`planning.attribution` and the `plant.Chain` AD table do not carry over; a
sequential-modular pass (so diffrax) is still needed to start, a flat start
failed; accuracy depends on a hand-tuned graded mesh (RON error from -1.2e-3
to -4e-5 across meshes, and a mesh wrong for faster kinetics is silently wrong:
no error control); memory not better; two upstream gaps worked around (asdex
Hessian handler, exact Hessian memory).

What a production version needs: pounce-solver[jax] and asdex as optional
extras (`difflow.solvers._lazy.require`), `pytest.importorskip("pounce.jax")`
in CI and the nightly Full suite only; a `ReformerPlanningModel(...,
solver="collocation")` that builds the `EOReformer` once, warm-starts from
the base `z`, uses `ad_mode="rev"`, and keeps the diffrax path as default and as
the start generator; mesh selection by a convergence check against `integrate_bed`
at build time (stage-1 metric, refuse if above 1e-4); the sparse + L-BFGS settings
above as defaults with the OOM cases documented; reference data unchanged (the
EO answers differ from the stored ones by design and must never loosen the
existing tolerances: use a new, separately stated tolerance); docs in
`docs/unit-operations-refinery.md`. Invariants at risk (`reforming/CLAUDE.md`):
"beds use `diffrax.ForwardMode`, differentiate with `jacfwd`" (reverse-only
here), "one enthalpy basis, balances to round-off at any tolerance" (becomes a
pounce tolerance, observed 1e-15), "warm-start with `tear_initial`" (no tear; the
start is a full profile).

Decision gate: compare against the linearize-and-solve fix on the same four
numbers (primal jit, Jacobian first call, Jacobian warm, RSS). If that fix
brings the Jacobian to the same order, this is not worth its limitations.

## Not measured

Per-iteration breakdown of the 4 pounce iterations; Jacobian cost as a function
of the number of outputs; vmap-free batching beyond 2 points; other feeds
(rich naphtha), other levers' large steps, 4-reactor trains; the other agent's
fix; exact-Hessian runs on a machine with more than 15 GB; kvaerno-class stiff
kinetics; why `solve_with_jacobian` first killed a process silently in one run
(no OOM message in `dmesg`) and then worked in the next.
