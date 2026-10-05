# CLAUDE.md - Project Guide for Claude Code

**difflow** is a JAX-based differentiable flowsheet framework for chemical
process simulation: automatic differentiation through unit operations for
gradient-based optimization, sensitivity analysis and technoeconomics.

Module-specific invariants live in a `CLAUDE.md` next to the code and load
when you work there: `src/difflow/{planning,stochastic,reconciliation}/`,
every plugin package (`src/difflow_{bio,ree,cc,gas,power,refinery}/`), and
the refinery subpackages (`vacuum/`, `reforming/`, `hydroprocessing/`,
`hydrotreating/`, `hydrocracking/`, `hydrogen/`, `residue/`, `preheat/`,
`gasplant/`, `fcc/`). Read the relevant one before changing a module -- most
say "do not weaken them" for a reason a regression test records.

## Commands

```bash
pip install -e ".[dev,examples,solvers]"

make test                 # parallel, skips `slow`
make test-all             # everything
pytest tests/ -v -n auto --dist loadfile   # loadfile keeps a module's JAX cache in one worker
pytest tests/test_cstr.py -v
make test-durations       # re-measure .test_durations (CI shards on it)
make convergence          # recycle-solver pass rate over the hard corpus
make book                 # Jupyter Book (+ the "Ask" index); make ask-check
make notebooks            # execute example notebooks
make gui-build            # rebuild the editor bundle + docs-index.json
```

Markers: `slow` is a cost label (deselected by `make test`); `release` marks
tests whose subject is the *answer* (physics, FD checks, published
benchmarks) -- deselected per commit, run by the nightly Full suite workflow.
CI shards by measured duration (`pytest-split`): 4 shards per commit, 12 for
the full suite (2 workers each: 4 heavy JAX files at once exhaust a 16 GB runner).

## Layout

```
src/difflow/           core: streams, thermo, eos, database, flowsheet,
                       units/, dynamic/, economics/, uncertainty, convergence,
                       planning/, stochastic/, reconciliation/, catalog,
                       docstrings, serialize, codegen, kinetics, publish,
                       gui/ (local browser editor; static/ is a committed build)
src/difflow_bio/       bioreactors, filtration, chromatography
src/difflow_ree/       rare earth solvent extraction
src/difflow_cc/        carbon capture (amine, membrane, adsorption, DAC)
src/difflow_gas/       gas transmission networks
src/difflow_power/     electrical grids (power flow, AC/DC-OPF)
src/difflow_refinery/  petroleum refining (assay -> CDU/VDU, conversion units,
                       gas plant, blending, H2 network, planning)
tests/                 pytest; plugin tests in tests/<plugin>/
examples/              Jupyter notebooks
docs/                  Jupyter Book source (Markdown)
_ext/                  Sphinx extension for the book's "Ask" assistant
```

## Key concepts

```python
from difflow import CSTR, CSTRParams, create_experiment_stream
import jax.numpy as jnp

inlet = create_experiment_stream(conditions={'T': 350.0, 'P': 101325.0},
                                 species=['A', 'B'], molar_flows=[1.0, 0.5])

def rate_fn(c, T, p):
    return p['k'] * jnp.exp(-p['Ea'] / (8.314 * T)) * c['A']

params = CSTRParams(V=1.0, rate_fn=rate_fn, stoich={'A': -1, 'B': 1},
                    molar_density=55500.0)
outlet = CSTR(params)(inlet)
params['V']; 'V' in params; params.update(V=2.0)   # ParamsMixin: dict-like, functional
```

- Everything is differentiable: `jax.grad`, `jacobian`, `jit` over any difflow function.
- CSTR concentrations need a molar density: `molar_density=`, or `eos=` +
  `reaction_phase=`. With neither it falls back to 55500 mol/m^3 (water) and
  raises `CSTRDensityWarning`.
- Every `Params` dataclass inherits `ParamsMixin`. Its `Attributes:` docstring
  is read by `difflow.docstrings` for the catalog (a field comment works too;
  descriptions go silently to None in a zipimport/frozen build). Units go in
  the unit class's `parameter_units`.
- Flowsheets: `fs.add_unit`, `fs.connect`, `fs.set_recycle`/`add_recycle`,
  `fs.solve(feed)`. Tears are the user's call by default;
  `fs.tear_analysis()` reports loops and what `select_tear_streams` would
  pick; `fs.solve(tears="auto")` picks them without changing `fs.recycles`.
- Under `jax.jacobian` a recycle solve routes to the optimistix fixed-point
  path (the Anderson/Wegstein loops are Python). Never record a diagnostic
  with bare `float()`; use `flowsheet._concrete()`.

## Recycle convergence (`difflow.convergence`; details in `docs/convergence.md`)

- Passing is `converged AND correct`; the solver's flag alone is not a pass.
- Every `Case.build` returns a FRESH flowsheet; no case may start on its answer
  (`TestNoCaseIsRigged`); keep a control case that passes; a raise is an
  outcome, not an error; every case states its `difficulty`.
- `tol` tests the STEP, not the error: at loop gain `g` the error is ~`1/(1-g)`
  larger. `solve` measures it (`last_solve_gain`, `last_solve_error_estimate`,
  `TearToleranceWarning`); `tol_basis="error"` tightens to the error.
- `clip_negative_flows` binds on Wegstein/Anderson only, never on plain
  substitution (it would put a kink in the map `optx.fixed_point` differentiates).
  A genuinely negative tear wants `clip_negative_flows=False`.

## Conventions

- `jax.numpy` as `jnp`; functional updates (`x.at[i].set(v)`); no Python
  control flow on traced values (`jnp.where`, `lax.cond`); register
  array-holding classes as PyTrees; float64.
- Type hints on public APIs; NumPy-style docstrings with Args/Returns/Example.
- Test the forward pass and the gradients (`jax.test_util.check_grads`).
- Reference data (IDAES/DWSIM JSON in `tests/refinery/reference/`) is
  REGENERATED when it goes stale, never loosened.

## Adding a unit operation

1. `src/difflow/units/` (or `dynamic/`, or a plugin package); a `Params`
   dataclass on `ParamsMixin`; `__call__(inlet) -> outlet`, JAX-only.
2. Tests in `tests/test_<unit>.py` (or `tests/<plugin>/`); an example notebook.
3. Plugins: export in the plugin's `__init__.py`/`__all__` and register it in
   `register()`.
4. Document it in `docs/unit-operations-*.md` under a heading naming it (`###`
   or shallower), or a MyST label `(op-<lowercase name>)=` before its table
   row, then run `python3 src/difflow/gui/docs_index.py`.
   `tests/test_doclinks.py` fails otherwise.

`difflow.planning`, `difflow.stochastic` and most refinery units (reformer,
hydrotreater, FCC, ...) are deliberately libraries, NOT palette operations;
check the module's CLAUDE.md before registering one.

## Debugging

- NaN gradients: `jax.config.update('jax_debug_nans', True)`; usual causes are
  `log(0)`, `sqrt(<0)`, `0/0`, `x/max(F, tiny)` on an empty flow (use a safe
  inverse).
- `TracerArrayConversionError` / `ConcretizationError`: Python control flow on
  a traced value.
- `jax.debug.print` inside `jit`; `check_grads(f, (x,), order=1, modes=['rev'])`.
- Slow compiles come from big nested solver graphs; `jit` a stage that holds a
  Python recycle; `jax.clear_caches()` when memory, not time, is the limit.

## Dependencies

JAX, diffrax, lineax, optimistix, PyYAML; dev: pytest, jupyter-book.
Optional: matplotlib, cantera, pyglenn (`difflow.pyglenn_import`), pythonnet
(`difflow.dwsim_import`), ipycytoscape/networkx.
