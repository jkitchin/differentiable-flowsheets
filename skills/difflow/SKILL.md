---
name: difflow
description: Builds, solves and differentiates chemical process flowsheets with difflow,
  the JAX differentiable flowsheet framework. Use when the user wants to simulate unit
  operations (CSTR, PFR, flash, distillation, heat exchangers, mixers, splitters,
  refinery, carbon capture, bioprocess, rare earth, gas or power networks), wire a
  flowsheet with recycles, take gradients or sensitivities with jax.grad, optimize a
  process, run technoeconomics, or fix a recycle that does not converge.
license: MIT
compatibility: Requires Python 3.11+ and difflow (pip install difflow); JAX float64.
metadata:
  source: https://github.com/jkitchin/differentiable-flowsheets
---

# difflow

difflow unit operations are pure JAX functions, so `jax.grad`, `jax.jacobian`
and `jax.jit` work through any unit, any flowsheet and its recycle solve.

## Streams and units

A stream is a dict of molar flows plus `T` (K) and `P` (Pa):

```python
from difflow import make_stream
inlet = make_stream({"A": 1.0, "B": 0.5}, T=350.0, P=101325.0)  # {'F_A', 'F_B', 'T', 'P'}
```

Every unit is `Unit(Params)(inlet) -> (outlet, info)`. `info` carries
diagnostics (`converged`, `conversion`, rates, ...); check it.

```python
import jax.numpy as jnp
from difflow import CSTR, CSTRParams

def rate_fn(c, T, p):  # -> one rate per reaction, mol/m^3/s
    return jnp.array([p["k"] * jnp.exp(-p["Ea"] / (8.314 * T)) * c["A"]])

params = CSTRParams(V=1.0, rate_fn=rate_fn,
                    stoich=jnp.array([[-1.0], [1.0]]),  # (n_species, n_reactions)
                    rate_params={"k": 1e3, "Ea": 4e4}, species_order=["A", "B"],
                    molar_density=55500.0)
outlet, info = CSTR(params)(inlet)
params.update(V=2.0)  # Params are dict-like and functional: returns a new object
```

A CSTR needs a molar density (`molar_density=`, or `eos=` with
`reaction_phase=`); without one it assumes water and raises
`CSTRDensityWarning`.

## Flowsheets and recycles

Streams are wired by name. A recycle is declared from the stream a unit
produces to the stream a unit reads:

```python
import jax
from difflow import (Flowsheet, Unit, make_stream, Mixer, Splitter,
                     StoichiometricReactor, StoichiometricReactorParams)

SP = ["A", "B"]

def build(X):
    fs = Flowsheet(species_order=SP)
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(SP), ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("rx", StoichiometricReactor(StoichiometricReactorParams(
        stoich={"A": -1, "B": 1}, key="A", conversion=X)), ["mixed"], ["effluent"]))
    fs.add_unit(Unit("split", Splitter(SP), ["effluent"], ["product", "purge"],
                     params={"split_frac": 0.7}))
    fs.add_recycle("purge", "recycle")
    return fs

streams = build(0.5).solve()                     # dict: stream name -> stream
print(streams["product"]["F_B"])                 # 5.882...
print(jax.grad(lambda X: build(X).solve()["product"]["F_B"])(0.5))  # 9.689...
```

- Tears are the user's choice. `fs.tear_analysis()` reports the loops and
  what an automatic choice would be; `fs.solve(tears="auto")` uses it.
- `fs.solve(acceleration="anderson" | "wegstein" | "none", tol=..., damping=...)`.
- A solve is good only when it converged **and** the answer is right
  (finite, mass balance closes). `tol` tests the step, not the error: at loop
  gain `g` the error is about `1/(1-g)` times larger. `fs.last_solve_gain`
  and `fs.last_solve_error_estimate` report it, and `TearToleranceWarning`
  fires when it matters; `solve(tol_basis="error")` tightens to the error.
- `fs.make_objective_fn(f)` turns a flowsheet into `objective({"unit.param": value})`
  for optimization; `fs.solve_eo()` solves equation-oriented.

## Finding units and APIs

Do not guess class names or parameters. Look them up:

```python
import difflow
[n for n in difflow.__all__ if "Flash" in n]
help(difflow.FlashParams)
```

Plugins add domains: `difflow_refinery`, `difflow_cc` (carbon capture),
`difflow_bio`, `difflow_ree` (rare earths), `difflow_gas`, `difflow_power`.
Many refinery models (reformer, hydrotreater, FCC) and `difflow.planning` /
`difflow.stochastic` are libraries called directly, not flowsheet units.

If the `difflow` MCP server is connected (`difflow mcp`, from
`pip install "difflow[mcp]"`), prefer its tools: `list_operations`,
`describe_operation`, `search_docs`, `add_unit`, `connect`, `solve`,
`diagnose`, `converge`, `sensitivity`, `optimize`, `tea`.

## JAX rules that bite

- Use `jax.numpy as jnp`, float64 (difflow enables it), functional updates
  (`x.at[i].set(v)`).
- No Python `if`/`float()` on traced values: use `jnp.where` / `lax.cond`.
  `TracerArrayConversionError` or `ConcretizationError` means one slipped in.
- NaN gradients come from `log(0)`, `sqrt(<0)`, `0/0` or dividing by an empty
  flow. Find them with `jax.config.update("jax_debug_nans", True)`.
- Verify gradients against finite differences with
  `jax.test_util.check_grads(f, (x,), order=1, modes=["rev"])`.
- The first call of a large unit compiles for a while; `jax.jit` a function
  you call repeatedly.

## When a recycle does not converge

1. `fs.tear_analysis()`: is every loop torn, and is the unit order able to
   supply every inlet?
2. Try `acceleration="anderson"`, then `"wegstein"`, then damped substitution
   (`acceleration="none", damping=0.5`), and more `max_iter`.
3. Give a better `tear_initial={stream: make_stream(...)}`.
4. A tear that is legitimately negative needs `clip_negative_flows=False`.

Full documentation: https://github.com/jkitchin/differentiable-flowsheets (the
`docs/` book, including `docs/convergence.md` and `docs/agents.md`).
