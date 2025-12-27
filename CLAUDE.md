# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**difflow** is a JAX-based differentiable flowsheet framework for chemical process simulation. It enables automatic differentiation through chemical engineering unit operations for gradient-based optimization, sensitivity analysis, and technoeconomic modeling.

## Quick Start Commands

```bash
# Install in development mode
pip install -e ".[dev,examples]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cstr.py -v

# Run tests with coverage
pytest tests/ --cov=src/difflow

# Build documentation (Jupyter Book)
make book

# Execute all example notebooks
make notebooks
```

## Repository Structure

```
difflow/
├── src/
│   ├── difflow/           # Core package
│   │   ├── streams.py     # Stream representation
│   │   ├── thermo.py      # Thermodynamics (ideal)
│   │   ├── eos.py         # Equations of State (PR, SRK)
│   │   ├── database.py    # Species property database
│   │   ├── flowsheet.py   # Flowsheet with recycle solving
│   │   ├── uncertainty.py # Sensitivity & UQ
│   │   ├── units/         # Steady-state unit operations
│   │   ├── dynamic/       # Dynamic modeling (DAE)
│   │   ├── economics/     # Technoeconomic analysis
│   │   └── visualization/ # Flowsheet visualization
│   ├── difflow_bio/       # Bio manufacturing plugin
│   └── difflow_ree/       # Rare earth element plugin
├── tests/                 # pytest test files
├── examples/              # Jupyter notebook examples
├── jax-tutorials/         # JAX/autodiff tutorials
└── docs/                  # Documentation (Markdown)
```

## Key Concepts

### 1. Streams
Streams are JAX-compatible data structures representing material flows:
```python
from difflow import Stream, experiment_stream

# Create a stream
stream = experiment_stream(
    experiment={'T': 350.0, 'P': 101325.0},
    experiment_species=['A', 'B'],
    experiment_molar_flows=[1.0, 0.5]
)
```

### 2. Unit Operations
All units are differentiable and follow a consistent pattern:
```python
from difflow import CSTR, experiment_reaction

# Define reaction: A -> B, r = k*C_A
rxn = experiment_reaction(
    experiment_species=['A', 'B'],
    experiment_stoich=[-1, 1],
    experiment_rate_law=lambda experiment, experiment_experiment: experiment['k'] * experiment_experiment['A']
)

# Create and run CSTR
cstr = CSTR(experiment_experiment, experiment_reactions=[rxn])
outlet = cstr(inlet_stream)
```

### 3. Automatic Differentiation
Use JAX's `grad`, `jacobian`, `jit` with any difflow function:
```python
import jax
from jax import grad, jit

def experiment(experiment):
    outlet = cstr(inlet, experiment={'experiment_experiment': experiment})
    return outlet.experiment['experiment']  # Scalar output

# Gradient of experiment w.r.t. experiment
d_experiment_d_experiment = grad(experiment)(experiment_value)

# JIT compile for speed
fast_experiment = jit(experiment)
```

### 4. Flowsheets with Recycles
```python
from difflow import Flowsheet

fs = Flowsheet()
fs.add_experiment('cstr', cstr)
fs.add_experiment('flash', flash)
fs.connect('cstr', 'flash')
fs.set_experiment('flash', 'cstr', experiment_experiment=0.5)  # Recycle

result = fs.solve(feed_stream)
```

## Code Conventions

### JAX Compatibility
- All numerical operations use `jax.numpy` (imported as `jnp`)
- Use `@jit` decorator for performance-critical functions
- Avoid in-place operations; use functional updates: `x = x.at[i].set(v)`
- Register custom classes as PyTrees if they contain arrays

### Type Hints
- Use type hints for public APIs
- Common types: `Array` (jax array), `Scalar` (float), `Dict[str, Array]`

### Testing
- Tests use pytest
- Each module has corresponding `test_*.py`
- Test both forward pass and gradients where applicable
- Use `jax.test_util.check_grads()` for gradient verification

### Documentation
- Docstrings follow NumPy style
- Include Args, Returns, and Example sections
- Example notebooks in `examples/` demonstrate usage

## Common Development Tasks

### Adding a New Unit Operation

1. Create file in `src/difflow/units/` (steady-state) or `src/difflow/dynamic/` (dynamic)
2. Inherit from appropriate base class
3. Implement `__call__` method that takes inlet stream(s) and returns outlet stream(s)
4. Ensure all operations are JAX-compatible (use `jnp`, no Python loops over arrays)
5. Add tests in `tests/test_<unit>.py`
6. Add example usage in `examples/`

### Adding to a Plugin (bio, ree)

1. Add to appropriate plugin directory (`src/difflow_bio/` or `src/difflow_ree/`)
2. Export in plugin's `__init__.py`
3. Add tests in `tests/bio/` or `tests/ree/`
4. Register in `pyproject.toml` entry points if needed

### Debugging Gradients

```python
# Check for NaN gradients
jax.config.update('jax_debug_nans', True)

# Finite difference gradient check
from jax.test_util import check_grads
check_grads(my_function, (x,), order=1, modes=['rev'])

# Print inside JIT
jax.debug.print("value: {x}", x=value)
```

## Dependencies

**Core:**
- JAX (>=0.4.0) - Automatic differentiation
- diffrax (>=0.6.0) - ODE/DAE solvers
- lineax (>=0.0.7) - Linear solvers
- optimistix (>=0.0.6) - Root finding
- PyYAML (>=6.0) - Configuration

**Development:**
- pytest, pytest-cov - Testing
- jupyter-book - Documentation

**Optional:**
- matplotlib, jupyter - Examples
- cantera - Complex chemistry
- ipycytoscape, networkx - Visualization

## Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration, dependencies |
| `Makefile` | Build automation (test, book, notebooks) |
| `src/difflow/__init__.py` | Main API exports |
| `tests/` | All pytest tests |
| `examples/` | Usage examples (Jupyter notebooks) |
| `jax-tutorials/` | JAX autodiff tutorials |

## Performance Tips

1. **JIT compile** hot paths: `@jit` or `jit(fn)`
2. **Vectorize** with `vmap` instead of Python loops
3. **Use 64-bit floats** for numerical stability: `jax.config.update('jax_enable_x64', True)`
4. **Checkpoint** memory-heavy computations: `jax.checkpoint(fn)`
5. **Profile** with `jax.profiler` for bottlenecks

## Troubleshooting

### "TracerArrayConversionError"
- Cause: Using JAX arrays in Python control flow during tracing
- Fix: Use `jax.lax.cond`, `jax.lax.switch`, or `jax.lax.fori_loop`

### "ConcretizationError"
- Cause: Trying to use abstract array values concretely
- Fix: Avoid `if x > 0:` with traced values; use `jnp.where(x > 0, ...)`

### NaN in Gradients
- Enable debug: `jax.config.update('jax_debug_nans', True)`
- Common causes: `log(0)`, `sqrt(negative)`, `0/0`
- Fix: Add small epsilon, use `jnp.clip`, safe functions

### Slow Compilation
- Large functions take time to JIT compile (one-time cost)
- Consider breaking into smaller functions
- Check for Python loops that could be `vmap`/`scan`
