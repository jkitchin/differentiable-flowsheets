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
pip install pounce-solver[jax]
pip install discopt
```

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
probed pattern      our structural pattern      true structure at x0
[0 0 1 0 1 0]       [1 1 1 1 1 1]               [1 0 1 0 1 0]
[0 0 1 1 1 0]       [1 1 1 1 1 1]               [1 0 1 1 1 0]
[0 1 0 0 0 0]       [1 1 1 1 1 1]               [0 1 0 0 1 0]
[0 0 0 0 0 1]       [1 1 1 1 1 1]               [0 0 0 0 0 1]
[0 0 0 1 0 0]       [0 0 0 1 0 0]               [0 0 0 1 0 0]
```

and the consequence is not a warning:

```
probed solve:     Infeasible_Problem_Detected   obj 0.030   (wrong)
patterned solve:  Solve_Succeeded               obj 0.066   (analytic optimum)
```

**Every adapter in `difflow.solvers` supplies `jac_pattern` and `hess_pattern`,
always.** There is no code path in `solve_with_pounce` or
`differentiable_problem` that reaches pounce with either pattern unset: if
`Bounds` carries structural patterns they are used, otherwise dense ones are
substituted. A dense pattern is trivially a valid superset; it costs a little
work and is never wrong.

### The contract on a supplied pattern

A pattern must be a **superset** of the true structure.

- **Extra entries are harmless.** They report a zero and may cost one extra
  color under `sparse=True`.
- **A missing entry is silently wrong.** On the dense path that derivative is
  dropped. Under `sparse=True` it is worse: the entry aliases into a
  same-colored reported entry and corrupts *that* one too. pounce never
  evaluates the model to check.

`as_nlp` keeps that contract two ways:

1. **By construction.** A unit's residual rows can only touch the stream
   variables of its own inlets and outlets, plus the decisions written into that
   unit. Everything downstream travels through stream variables, which are
   already columns of their own. That argument is what makes the pattern valid
   everywhere, not just at one point.
2. **By verification.** `validate_patterns` compares the pattern against a dense
   AD Jacobian and Lagrangian Hessian at `x0` and raises `SparsityPatternError`
   on any missing entry. `as_nlp(validate="auto")` runs it whenever `n * m` is
   under 40 000. A point check is necessary, not sufficient — but it catches
   mistakes in the derivation.

   The Hessian half of that check uses **random multiplier vectors**, not
   `lambda = 1`. A unit's material balances share one reaction term whose
   stoichiometric coefficients sum to zero over the species, so weighting the
   rows equally cancels the nonlinearity exactly: the Lagrangian Hessian comes
   out identically zero and *any* pattern passes, including an empty one. (A
   symbolic union at `lambda = 1`, as in `asdex.hessian_sparsity`, is fine —
   graph reachability does not cancel. This is a numerical check.)

If you ever doubt the derivation, `sparsity="dense"` is the safe answer.

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
`Spec(variables=[...])` tightens that Jacobian row; omitting it makes the row
dense, which is always safe.

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
| `Bounds` | boxes, `x0`, `p0`, names, and both sparsity patterns |
| `dense_jacobian_pattern`, `dense_hessian_pattern` | always-valid supersets |
| `validate_patterns` | the check pounce does not do |
| `SparsityPatternError` | raised by it |
| `require_eo_residuals` | refuse a flowsheet with no EO form |

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
