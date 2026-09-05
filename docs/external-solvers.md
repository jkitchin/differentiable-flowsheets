# External solvers: pounce and discopt

`difflow.solvers` is a bridge, not a solver. It gives a difflow flowsheet two
shapes that external optimization packages already accept, so their capabilities
do not have to be reimplemented here:

| View | Function | Consumer |
|------|----------|----------|
| Flat NLP `f(x)`, `g(x)`, bounds | `as_nlp` | `pounce.jax.from_jax`, `pounce.jax.JaxProblem` |
| Residual `g(u, v) = 0` | `as_residual`, `residual_from_system` | `discopt.modeling.implicit` |

Both back ends are optional. They are imported lazily inside the functions that
need them, and the `ImportError` names the PyPI distribution — which for pounce
is `pounce-solver`, not `pounce`.

```
pip install difflow[solvers]      # asdex -- where the sparsity pattern comes from
pip install pounce-solver[jax]
pip install discopt
```

`asdex` is not a back end. It is how `as_nlp` derives the sparsity pattern the
next section is about, so it is the one thing here that is *not* optional in
practice: without it, `as_nlp` has only the coarse topology derivation to fall
back on, and it will tell you so.

---

## Read this first: sparsity is a promise pounce does not check

This is the single most important thing in the module, and it is the reason the
adapter exists rather than a three-line recipe in a notebook.

`pounce.jax.from_jax` and `pounce.jax.JaxProblem` detect the constraint-Jacobian
and Lagrangian-Hessian structure by **evaluating derivatives at random
`N(0, 1)` points** unless a pattern is supplied. Those points have nothing to do
with `x0` or with the variable bounds. For a process model, an `N(0, 1)` draw
means a temperature of about **−1.3 K** and a pressure of about **0.4 Pa**.

At that point an Arrhenius term `exp(-Ea / (R T))` overflows, a reactor's linear
solve goes singular, and the derivative comes back as `inf` or `nan`. The probe
records a nonzero where `abs(J[i, j]) > eps`, and **`nan > eps` is `False`** — so
every `nan` entry is recorded as a *structural zero*.

On the one-CSTR design problem in the test suite this is not subtle. The probed
pattern loses the entire reactor-volume column, the one column the optimizer has
to move:

```
probed pattern      topology pattern       derived (default)     true at x0
[0 0 1 0 1 0]       [1 1 1 1 1 1]          [1 0 1 0 1 0]         [1 0 1 0 1 0]
[0 0 1 1 1 0]       [1 1 1 1 1 1]          [1 0 1 1 1 0]         [1 0 1 1 1 0]
[0 1 0 0 0 0]       [1 1 1 1 1 1]          [0 1 0 0 1 0]         [0 1 0 0 1 0]
[0 0 0 0 0 1]       [1 1 1 1 1 1]          [0 0 0 0 0 1]         [0 0 0 0 0 1]
[0 0 0 1 0 0]       [0 0 0 1 0 0]          [0 0 0 1 0 0]         [0 0 0 1 0 0]
```

The probe is missing entries the model has; the topology pattern has entries
the model does not (safe, but it pays for them in the Hessian); the derived
pattern is the true structure, and — unlike the last column — it is the true
structure *everywhere*, not only at `x0`.

and the consequence is not a warning:

```
probed solve:     Infeasible_Problem_Detected   obj 0.030   (wrong)
patterned solve:  Solve_Succeeded               obj 0.066   (analytic optimum)
```

**Every adapter in `difflow.solvers` supplies `jac_pattern` and `hess_pattern`,
always.** There is no code path in `solve_with_pounce` or
`differentiable_problem` that reaches pounce with either pattern unset — a
`Bounds` arriving without one raises `SparsityDetectionError` rather than
quietly substituting a dense pattern. Dense is *valid*, and on a real
flowsheet it is unaffordable: the Hessian then costs `n` colors, and `n` grows
with the plant. Dense is a mode you can ask for, never one you get by default
and never one you get without being told.

### The contract on a supplied pattern

A pattern must be a **superset** of the true structure.

- **Extra entries are harmless.** They report a zero and may cost one extra
  color under `sparse=True`.
- **A missing entry is silently wrong.** On the dense path that derivative is
  dropped. Under `sparse=True` it is worse: the entry aliases into a
  same-colored reported entry and corrupts *that* one too. pounce never
  evaluates the model to check.

`as_nlp` keeps that contract three ways:

1. **By graph analysis — the default.** `asdex` propagates index sets through
   the jaxpr of `g`, so the pattern it returns holds for *every* input, not for
   the point it was traced at. The Lagrangian Hessian is obtained by tracing
   `lambda` as an argument, which makes the result the union over all
   multipliers rather than the pattern at one particular `lambda`. This is
   `sparsity="auto"` (falls back if the analysis fails) or `"global"` (raises
   instead of falling back).
2. **By construction — the fallback.** A unit's residual rows can only touch the
   stream variables of its own inlets and outlets, plus the decisions written
   into that unit. Everything downstream travels through stream variables, which
   are already columns of their own. That argument also makes the pattern valid
   everywhere, and it needs no analysis of the code — but it is coarse, and any
   row it cannot see inside (a `Spec` with a callable body, the objective) is a
   dense row whose outer product makes the *Hessian* dense. It says so, in a
   `RuntimeWarning` naming the rows. This is `sparsity="structural"`.
3. **By verification.** `validate_patterns` compares the pattern against the
   real AD Jacobian and Lagrangian Hessian and raises `SparsityPatternError` on
   any missing entry. `as_nlp(validate="auto")` — the default — checks every
   entry when `n * m` is under 40 000 and switches to sampled columns above it,
   where a sampled column is a JVP against a basis vector and so is *exactly*
   that column of `J`. It is never skipped. A point check is necessary, not
   sufficient — but it catches mistakes in the derivation.

   The Hessian half of that check uses **random multiplier vectors**, not
   `lambda = 1`. A unit's material balances share one reaction term whose
   stoichiometric coefficients sum to zero over the species, so weighting the
   rows equally cancels the nonlinearity exactly: the Lagrangian Hessian comes
   out identically zero and *any* pattern passes, including an empty one. (The
   symbolic union with `lambda` traced is unaffected — graph reachability does
   not cancel. This is the numerical check, and it is the one that can be
   fooled.)

### Choosing the source

| `sparsity=` | Where the pattern comes from | When it fails |
|-------------|------------------------------|---------------|
| `"auto"` (default) | graph analysis; topology if that fails, with a warning | never silently: `ImportError` if `asdex` is missing, `SparsityDetectionError` if neither derivation applies |
| `"global"` | graph analysis only | raises rather than falling back — use when you want to know |
| `"structural"` | flowsheet topology only | needs no `asdex`; warns about the rows it cannot see inside |
| `"dense"` | no derivation at all | never wrong, and quadratic |

Why this is not a micro-optimization — the `Heater -> CSTR` train from
`tests/test_solvers_bridge.py`, `N` stages:

| N | n | dense Jac | topology Jac | global Jac | dense Hess | topology Hess | global Hess |
|---|---|-----------|--------------|------------|------------|---------------|-------------|
| 1 | 9 | 81 | 53 | 17 | 45 | 45 | 4 |
| 2 | 18 | 306 | 121 | 36 | 171 | 171 | 8 |
| 4 | 36 | 1 188 | 257 | 74 | 666 | 666 | 16 |
| 8 | 72 | 4 680 | 529 | 150 | 2 628 | 2 628 | 32 |
| 16 | 144 | 18 576 | 1 073 | 302 | 10 440 | 10 440 | 64 |

The topology Hessian column is not a coincidence: it is the *dense* column,
exactly, at every size — with the objective's variables unnamed the topology
derivation gives it every column, and `cols(f) x cols(f)` is the whole
triangle. The derived Hessian is `4 N`: linear in the flowsheet, because that
is what the flowsheet is. Detection costs about a second at `N = 16`, once, at
build time.

`sparsity="dense"` remains available for when you doubt a derivation, and
`validate_patterns` is the cheaper way to settle the same doubt.

### Why the pattern can only come from the equation-oriented form

Sequential-modular calls (`cstr(inlet, T_spec=...)`) close their balance with an
inner `optimistix.Newton` inside `optx.root_find`. That is differentiable, but it
is opaque to *structural* analysis: the inner solve emits a `linear_solve`
primitive that `asdex` has no handler for, and a conservative fallback handler
then trips on a data-dependent `dynamic_slice` in the Newton while-loop.

The working path is `eo_residuals`, which is fully explicit and has no inner
solve. It is also the *better* formulation: the inner solve genuinely couples
every variable in its block, so the sequential form is **denser** than the
residual form. `eo_residuals` exposes structure that the inner solve hides.

One trap when you write the residual form by hand rather than through `as_nlp`:
`eo_residuals` returns `n_species` material balances **plus a T row and a P
row**. In the equation-oriented form those are real equations (`T_out` and
`P_out` are stream variables). But if you promote `T` to a decision variable
directly and hold `P` fixed, those last two rows collapse to `0 = 0`, and you
must drop them or the constraint Jacobian gets exactly-zero rows. `as_nlp` never
hits this because it keeps `T` and `P` as stream variables; the hand-written
train in `examples/27_pounce_optimization.ipynb` does hit it and slices
`r[:n_species]`.

---

## The flat NLP view

```python
from difflow.solvers import as_nlp, Decision, solve_with_pounce

f, g, bounds = as_nlp(
    flowsheet,
    decisions=[
        Decision("unit:reactor.params.V", lb=0.01, ub=5.0, x0=0.5),
        Decision("unit:reactor.T_spec",   lb=320.0, ub=420.0, x0=360.0),
    ],
    specs=[("product.F_B", ">=", 8.0)],
    objective=lambda streams, decisions: decisions["unit:reactor.params.V"],
)

x, info = solve_with_pounce(f, g, bounds, options={"tol": 1e-9})
```

The variable vector is

```
x = [ decisions | stream variables ]
```

where the stream block is `EOStateLayout`'s state vector — every non-feed stream
contributes `[F_s1, ..., F_sn, T, P]`. The constraint body is

```
g(x) = [ unit residuals (= 0) | specification bodies (in [lo, hi]) ]
```

and the unit residuals *are* the ones `flowsheet.solve_eo()` solves. Nothing is
re-derived, so a feasible point of the NLP is a converged flowsheet by
construction — there is no separate simulation step to keep in sync.

### The address grammar

Decisions and parameters name where their value goes:

| Address | Meaning |
|---------|---------|
| `unit:<unit>.<kwarg>` | an entry of `Unit.params`, i.e. a keyword passed to the unit call and to `eo_residuals` (`T_spec`, `volumetric_flow`, ...) |
| `unit:<unit>.params.<field>` | a field of the operation's `Params` dataclass (`V`, `UA`, ...), applied with `ParamsMixin.update` |
| `feed:<stream>.<key>` | `T`, `P` or `F_<species>` of a feed stream |

The `unit:` prefix may be dropped. For anything the grammar cannot reach, pass a
builder callable instead of a `Flowsheet`:

```python
def build(values):
    fs = make_flowsheet()
    ...                       # use values["my_name"] however you like
    return fs

f, g, bounds = as_nlp(build, [Decision("my_name", 0.0, 1.0, 0.5)], specs)
```

The topology must stay fixed across calls — only values may vary — because the
NLP layout is fixed at build time.

### Specifications

`("<stream>.<key>", op, value)` with `op` in `<=`, `>=`, `==`, or a full `Spec`
with a callable body `fn(streams, decisions) -> scalar`. Supplying
`Spec(variables=[...])` tightens that Jacobian row on the `sparsity="structural"`
path; omitting it makes the row dense there, which is safe but takes the
Lagrangian Hessian dense with it. The default `sparsity="auto"` reads a callable
body's structure off its jaxpr and does not need the hint.

### Post-optimal sensitivity

`info["mult_g"][i]` is the constraint multiplier `lambda_i`. pounce forms
`L = sigma f + lambda^T g`, so

```
d(objective*) / d(bound) = -lambda
```

`bound_sensitivities(info, bounds)` returns the engineering quantity — what one
more unit of product costs — already negated and keyed by constraint name. The
sign was measured, not assumed: on the CSTR problem a central difference of the
converged objective with respect to the bound agrees with `-mult_g` to seven
digits in **both** directions (`F_B >= 8` gives FD `+0.72494` against `mult_g
= -0.72494`; `F_A <= 2` gives FD `-0.72494` against `mult_g = +0.72494`). Both
are pinned in `tests/test_solvers_bridge.py`.

### Differentiating through the whole design problem

Declare `parameters=` and the objective and constraints become `f(x, p)` and
`g(x, p)`. `differentiable_problem` then builds a `pounce.jax.JaxProblem`, whose
`solve(p, x0)` is differentiable with respect to `p` by the implicit-function
rule on the KKT system:

```python
from difflow.solvers import Parameter, differentiable_problem

f, g, bd = as_nlp(fs, decisions, specs, objective=profit,
                  parameters=[Parameter("unit:reactor.T_spec", 350.0)])
jp = differentiable_problem(f, g, bd)

loss = lambda p: f(jp.solve(p, bd.x0), p)
jax.grad(loss)(bd.p0)          # d(optimal design)/d(parameter)
```

**Use `JaxProblem`, not `pounce.jax.solve`.** `pounce.jax.solve` is the other
differentiable entry point, but its signature has no `jac_pattern` /
`hess_pattern` arguments, so it always probes — see the section above for why
that is fatal here. `differentiable_problem` is the reason this module exists in
the shape it does.

---

## The residual view

```python
from difflow.solvers import as_residual

view = as_residual(flowsheet)     # u = feed streams, v = every other stream
view(view.u0, view.v0)            # -> residual vector, ~0 at the solution
view.n_unknowns, view.u_names, view.v_names
```

`as_residual` accepts:

- a **`Flowsheet`** — `u` is the flat vector of every feed's `[F..., T, P]`
  (feeds sorted by name), `v` is the EO state vector of every non-feed stream,
  and the residual is `EOSolver`'s own;
- a **`Unit`** or a bare unit operation with `eo_residuals` — `u` is its
  inlets, `v` its outlets (`inlets=` and `outlets=` give the nominal values and
  the starting guess);
- a **raw residual function** `residual_fn(z, args)` — difflow's section-scope
  convention, the shape `difflow.eo_solver.solve_residual_system` takes and the
  shape `difflow_ree.equilibrium.mass_action.make_section_residual` returns.
  Pass `z0=` and `args=`; this routes to `residual_from_system`.

```python
from difflow.solvers import residual_from_system

residual_fn, _ = make_section_residual(network, n_stages=4)
view = residual_from_system(residual_fn, z0=guess, args=args,
                            u_keys=["feed_totals"])
```

`u_keys` names the subset of `args` the outer model varies; everything else is
closed over at its nominal value. **The order of `u` is not the order of
`u_keys`** — `ravel_pytree` flattens with `jax.tree_util`, which sorts dict keys.
Read `view.u_names` and pass the model expressions in *that* order.

`ResidualView` is itself callable as `g(u, v)`, so it is a drop-in `residual`
argument for `discopt.modeling.implicit`.

### Units without an equation-oriented form

Only units with `eo_residuals` (CSTR, Flash, Mixer, Splitter, the heat
exchangers) can be exposed. `require_eo_residuals(fs)` refuses the rest up front
with the offending unit names. It has to: `EOSolver._build_residual_fn`'s
fallback branch for units without `eo_residuals` reads a bare name `feed_names`
that is a local of `EOSolver.__init__` and neither a closure cell nor a module
global, so it raises `NameError: name 'feed_names' is not defined` from inside a
JAX trace with no indication of which unit caused it.

---

## The restriction: a difflow block in discopt is local-NLP-only

`discopt.modeling.implicit(g, u_inputs, n_unknowns)` compiles `g(u, v) = 0` into
a differentiable inner Newton solve whose derivatives come from
`jax.lax.custom_root`. That supports higher-order AD, so a second-order NLP
solver's Hessian works through the node.

It rides on `dm.custom`, which produces a **`CustomCall`** — an opaque, AD-only
node. Every guarantee discopt offers is built on being able to *see* the
algebra, so a model containing one is restricted:

- **No global optimality certificate.** The solve reports `status="feasible"`
  with `bound` and `gap` both `None`. Which root the inner Newton lands in *is*
  the definition of `v`, so two starting points in the same box can legitimately
  give two different "optima".
- **Relaxation compilation and `.nl` export raise.** The Rust tape has no
  equivalent for a `CustomCall`
  (`UnsupportedForTape("CustomCall (dm.custom) has no tape equivalent")`), so
  the model falls back to the JAX evaluator. `Model.to_nl` on a model holding
  an indexed implicit node raises earlier still
  (`Cannot resolve indexed expression: custom:...[i]`) — the message varies,
  the outcome does not.
- **The solver raises if any integer or binary variable is present.** Spatial
  branch and bound has no valid node relaxation for an opaque callable, so
  discopt refuses rather than returning an unsound bound.

Verified against discopt's source (`discopt/solver.py`, the
`_model_contains_custom_call` → `_custom_call_reduced_admissible` →
`_is_pure_continuous` gate) and against a live solve, which raises:

```
ValueError: Model contains a dm.custom(...) AD-only user function that is
OUTSIDE the sound reduced-space (MCBox) scope, together with integer/binary
variables. Global branch-and-bound needs a valid node relaxation, ...
```

Current discopt has one refinement worth knowing: a `CustomCall` whose body
traces soundly through discopt's McCormick-box (MCBox) intrinsics **is** globally
relaxable, integers included. **A difflow residual never qualifies** —
`dm.implicit`'s forward is a `jax.lax.while_loop` Newton iteration over raw
`jnp` intrinsics, which is outside MCBox scope by construction — so for difflow
the strict restriction always applies.

### What this means in practice

**You cannot wrap a difflow flowsheet as a UDF and then put binaries around it in
one discopt model.** If you need integrality or a global bound over a flowsheet,
decompose:

- keep the binaries in a **master problem** and solve the flowsheet-shaped
  subproblem as an NLP with `difflow.solvers.pounce_bridge`; or
- **re-express the block in discopt's own algebraic language** (`dm.exp`,
  `dm.log`, ...), where a relaxation exists.
  `dm.implicit(..., formulation="full_space")` is that second option applied to
  this exact node — it lowers `v` to real variables and the residuals to real
  equality constraints, which keeps a certificate reachable. But it requires the
  residual to be written in discopt operators, so a JAX-traced difflow residual
  cannot be passed to it.

`as_implicit` enforces the integrality half at **build** time rather than letting
you discover it at solve time, after the model is already written:

```python
import discopt.modeling as dm
from difflow.solvers import as_implicit

m = dm.Model()
F_A = m.continuous("F_A_feed", lb=1.0, ub=20.0)
node, view = as_implicit(m, flowsheet, [F_A, 0.0, 320.0, 101325.0])
m.minimize((node[1] - 5.0) ** 2)
result = m.solve()          # status == "feasible"; bound and gap are None
```

`as_implicit` also defaults the inner Newton's `x0` to `view.v0` — the
flowsheet's own sequential-modular estimate. `dm.implicit`'s own default is a
vector of zeros, where a difflow residual is usually singular. And it checks that
`u_inputs` flattens to exactly `view.n_inputs` scalars, since `dm.implicit`
concatenates them in argument order.

`CUSTOMCALL_RESTRICTION` is the one-paragraph statement of all of the above, for
embedding in your own error messages and reports.

---

## API summary

**NLP view** (`difflow.solvers.nlp`)

| Name | Purpose |
|------|---------|
| `as_nlp(flowsheet, decisions, specs, ...)` | `(f, g, Bounds)` |
| `Decision`, `Parameter`, `Spec` | problem description |
| `Bounds` | boxes, `x0`, `p0`, names, both sparsity patterns, and `sparsity_source` |
| `require_eo_residuals` | refuse a flowsheet with no EO form |

**Sparsity** (`difflow.solvers.sparsity`)

| Name | Purpose |
|------|---------|
| `detect_patterns(f, g, x0, m)` | both patterns from the jaxpr, for a hand-built model |
| `detect_jacobian_pattern`, `detect_hessian_pattern` | one at a time |
| `validate_patterns` | the check pounce does not do (dense or sampled) |
| `dense_jacobian_pattern`, `dense_hessian_pattern` | always-valid supersets, never a default |
| `pattern_density` | fraction of dense, for reporting |
| `SparsityPatternError` | a pattern misses a real entry |
| `SparsityDetectionError` | no pattern could be derived, and none was invented |

**pounce** (`difflow.solvers.pounce_bridge`)

| Name | Purpose |
|------|---------|
| `solve_with_pounce(f, g, bounds, ...)` | one solve, patterns always supplied |
| `optimize_flowsheet(...)` | `as_nlp` + solve, returns `FlowsheetOptimum` |
| `differentiable_problem(...)` | a `JaxProblem`: build once, solve many, differentiable |
| `bound_sensitivities(info, bounds)` | `d(objective)/d(bound)` by constraint name |

**Residual view / discopt** (`difflow.solvers.residual`, `.discopt_bridge`)

| Name | Purpose |
|------|---------|
| `as_residual(unit_or_section, ...)` | `ResidualView`, callable as `g(u, v)` |
| `residual_from_system(residual_fn, z0, args, ...)` | wrap difflow's `r(z; args)` convention |
| `as_implicit(model, ..., u_inputs)` | add the block to a discopt model |
| `check_no_integrality`, `integer_variables` | the build-time gate |
| `DiscoptIntegralityError`, `CUSTOMCALL_RESTRICTION` | what it raises, and why |

## See also

- `examples/27_pounce_optimization.ipynb` — the same three-library split done by
  hand (difflow + asdex + pounce) on a four-stage reactor train, including the
  spy plots of the detected structure.
- [Equation-oriented solver](eo-solver.md) — where the residuals come from.
- Kitchin, POUNCE, doi:10.5281/zenodo.20387011.
- Kitchin, discopt, doi:10.5281/zenodo.19762815.
