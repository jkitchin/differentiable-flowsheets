---
name: difflow-add-unit
description: Adds a new unit operation to the difflow codebase (core, dynamic or a
  plugin such as difflow_refinery or difflow_cc) following the project's conventions,
  with a ParamsMixin params dataclass, JAX-only __call__, tests with gradient checks,
  docs and catalog registration. Use when contributing to difflow and the user asks to
  add, implement or create a unit, reactor, column, separator or other unit operation.
license: MIT
compatibility: Run inside a clone of github.com/jkitchin/differentiable-flowsheets.
---

# Add a difflow unit operation

The argument (if any) is the unit's name, e.g. `Crystallizer`. Ask the user
for the physics (balances, equations, what is specified) before writing code.

## 0. Read the local rules first

Read `CLAUDE.md` at the repo root, and the `CLAUDE.md` next to the code you
are changing (every plugin package and many subpackages have one). Some
modules are deliberately libraries, not palette operations; their CLAUDE.md
says so. Do not weaken an invariant a CLAUDE.md records.

## 1. Where it goes

- Core steady-state: `src/difflow/units/<name>.py`, exported from
  `src/difflow/units/__init__.py` and `src/difflow/__init__.py` (`__all__` too).
- Dynamic: `src/difflow/dynamic/`.
- Plugin: `src/difflow_<plugin>/...`; export in the plugin's `__init__.py` /
  `__all__` and register it in the plugin's `register()`.

## 2. Shape of the code

Model it on a small existing unit, e.g. `src/difflow/units/stoichiometric.py`:

```python
from dataclasses import dataclass
import jax.numpy as jnp
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream


@dataclass(repr=False)
class MyUnitParams(ParamsMixin):
    """Parameters of a :class:`MyUnit`.

    Attributes:
        efficiency: Fractional efficiency, 0 to 1. May be a JAX tracer.
        pressure_drop: Pressure drop (Pa) across the unit.
    """

    efficiency: float
    pressure_drop: float = 0.0


class MyUnit:
    """One-line summary.

    ``MyUnit(params)(inlet) -> (outlet, info)``. Describe the model and
    every ``info`` key.

    Example:
        >>> ...
    """

    symbol = "My Unit"
    equations = [r"..."]
    assumptions = ["..."]
    references = ["..."]
    parameter_units = {"efficiency": "-", "pressure_drop": "Pa"}

    def __init__(self, params: MyUnitParams):
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = get_flows(inlet)
        ...
        return make_stream(out_flows, T, P), info
```

Rules:

- The `Attributes:` docstring feeds the catalog (`difflow.docstrings`); every
  field needs a description. Units go in `parameter_units`.
- JAX only in `__call__`: `jnp`, `jnp.where`/`lax.cond` instead of `if` on
  values, `x.at[i].set(v)`, float64. No bare `float()` of a traced value.
- Guard divisions by flows that can be zero with a safe inverse, not
  `x / max(F, tiny)` (NaN gradients).
- Validate concrete inputs in `__post_init__`, but let tracers through.
- Type hints and NumPy/Google-style docstrings with Args/Returns/Example.

## 3. Tests

`tests/test_<name>.py` (plugins: `tests/<plugin>/`):

- forward pass against a hand calculation or published value;
- mass (and energy, if modelled) balance closes;
- gradients: `jax.test_util.check_grads(f, (x,), order=1, modes=["rev"])`
  on the parameters and the feed;
- works inside a `Flowsheet` and under `jax.jit`.

Mark an expensive solve `@pytest.mark.slow`; per-commit tests must finish
within `DIFFLOW_TEST_BUDGET` (180 s). Run `pytest tests/test_<name>.py -v`.

## 4. Docs and catalog

1. Document it in `docs/unit-operations-*.md` under a heading naming it
   (`###` or shallower), or a MyST label `(op-<lowercase name>)=` before its
   table row.
2. Run `python3 src/difflow/gui/docs_index.py` to refresh the editor index.
3. `pytest tests/test_doclinks.py` must pass.
4. Add an example notebook in `examples/` if the unit is user-facing.
