"""Unified dynamic modeling framework for difflow.

This module provides a unified paradigm for dynamic simulation of
process units, enabling both steady-state and transient analysis
within the same framework.

Key Components
--------------

**State Management** (state.py):
- StateVar: Single state variable specification
- StateSpec: Collection of states for a unit
- StateVector: Runtime container with named access

**Unit Protocols** (base.py):
- DynamicUnit: Protocol for all dynamic units
- DynamicUnitBase: Abstract base class with utilities
- DynamicCSTR: Example dynamic CSTR implementation
- DynamicTank: Example storage tank with holdup

**Integration** (integrators.py):
- integrate(): Unified ODE integration interface
- integrate_rk4(): Fixed-step 4th order Runge-Kutta
- integrate_rk45(): Adaptive step RK45
- integrate_unit(): Convenience wrapper for DynamicUnit

**Flowsheet** (flowsheet.py):
- DynamicFlowsheet: Connect multiple dynamic units
- DynamicFlowsheetResult: Result container with unit access
- DynamicUnitEntry: Unit registration in flowsheet

**DAE Support** (dae.py):
- DAEUnit: Protocol for units with algebraic constraints
- DAEUnitBase: Base class for DAE units
- AlgebraicVar/AlgebraicSpec: Algebraic state specification
- integrate_dae(): DAE integration with Newton solver
- DynamicFlashDrum: Example DAE unit with VLE equilibrium

Example Usage
-------------

Basic integration:

>>> from difflow.dynamic import integrate
>>> import jax.numpy as jnp
>>>
>>> def harmonic_oscillator(t, y):
...     x, v = y[0], y[1]
...     return jnp.array([v, -x])
>>>
>>> result = integrate(
...     harmonic_oscillator,
...     y0=jnp.array([1.0, 0.0]),
...     t_span=(0.0, 10.0),
...     method="RK4",
...     n_steps=100,
... )
>>> result.y_final  # Final [position, velocity]

Using DynamicUnit:

>>> from difflow.dynamic import DynamicCSTR, integrate_unit
>>> from difflow.streams import make_stream
>>>
>>> # Define reaction kinetics
>>> def rate_fn(C, T, params):
...     k = params["k"] * jnp.exp(-params["Ea"] / (8.314 * T))
...     return jnp.array([k * C["A"]])
>>>
>>> # Create dynamic CSTR
>>> cstr = DynamicCSTR(
...     volume=1.0,
...     rate_fn=rate_fn,
...     stoich=jnp.array([[-1], [1]]),  # A -> B
...     species_order=["A", "B"],
...     rate_params={"k": 1e6, "Ea": 50000},
... )
>>>
>>> # Create inlet stream
>>> inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
>>>
>>> # Integrate
>>> result = integrate_unit(
...     cstr,
...     inputs={"inlet": inlet},
...     t_span=(0.0, 1000.0),
...     method="RK4",
... )

State specification:

>>> from difflow.dynamic import StateSpec, StateVar, molar_states
>>>
>>> # Manual state spec
>>> spec = StateSpec([
...     StateVar("n_A", "moles", "mol", bounds=(0, None)),
...     StateVar("n_B", "moles", "mol", bounds=(0, None)),
...     StateVar("T", "temperature", "K"),
... ])
>>>
>>> # Using helper functions
>>> spec = molar_states(["A", "B", "C"]) + thermal_state()
"""

# State management
from difflow.dynamic.state import (
    # Core classes
    StateVar,
    StateSpec,
    StateVector,
    StateCategory,
    # Factory functions
    molar_states,
    concentration_states,
    thermal_state,
    volume_state,
    pressure_state,
    reactor_states,
)

# Base classes and protocols
from difflow.dynamic.base import (
    # Protocol
    DynamicUnit,
    # Base class
    DynamicUnitBase,
    # Implementations
    DynamicCSTR,
    DynamicTank,
    # Types
    Params,
    RateFunction,
)

# Integrators
from difflow.dynamic.integrators import (
    # Main interface
    integrate,
    integrate_unit,
    # Specific methods
    integrate_rk4,
    integrate_rk45,
    integrate_euler,
    # Individual steps
    rk4_step,
    rk45_step,
    # Results
    IntegrationResult,
    Trajectory,
    IntegrationInfo,
    # Events
    EventSpec,
    EventResult,
    detect_events,
    # Stiffness detection
    estimate_stiffness,
    # Gradient utilities
    integrate_with_grad,
    sensitivity_analysis,
    # Types
    Method,
    DerivativesFn,
)

# Dynamic flowsheet
from difflow.dynamic.flowsheet import (
    DynamicFlowsheet,
    DynamicFlowsheetResult,
    DynamicUnitEntry,
    Connection,
    FlowsheetState,
)

# Diffrax backend (optional)
try:
    from difflow.dynamic.diffrax_backend import (
        integrate_diffrax,
        integrate_diffrax_unit,
        integrate_stiff,
        integrate_dopri5,
        integrate_tsit5,
        integrate_kvaerno5,
        list_diffrax_solvers,
        check_diffrax_available,
        HAS_DIFFRAX,
        DIFFRAX_SOLVERS,
    )
except ImportError:
    # diffrax not installed - provide stubs
    HAS_DIFFRAX = False
    DIFFRAX_SOLVERS = {}
    integrate_diffrax = None
    integrate_diffrax_unit = None
    integrate_stiff = None
    integrate_dopri5 = None
    integrate_tsit5 = None
    integrate_kvaerno5 = None
    list_diffrax_solvers = lambda: []
    check_diffrax_available = lambda: False

# DAE support
from difflow.dynamic.dae import (
    # Algebraic specification
    AlgebraicVar,
    AlgebraicSpec,
    AlgebraicVector,
    # DAE protocol and base
    DAEUnit,
    DAEUnitBase,
    # DAE integration
    integrate_dae,
    dae_step_euler,
    dae_step_rk4,
    DAEResult,
    DAEMethod,
    # Newton solver
    newton_solve,
    solve_algebraic,
    # Utility functions
    vapor_fraction_algebraic,
    k_value_algebraic,
    pressure_algebraic,
    equilibrium_extent_algebraic,
    # Example units
    DynamicFlashDrum,
)

__all__ = [
    # State
    "StateVar",
    "StateSpec",
    "StateVector",
    "StateCategory",
    "molar_states",
    "concentration_states",
    "thermal_state",
    "volume_state",
    "pressure_state",
    "reactor_states",
    # Base
    "DynamicUnit",
    "DynamicUnitBase",
    "DynamicCSTR",
    "DynamicTank",
    "Params",
    "RateFunction",
    # Integrators
    "integrate",
    "integrate_unit",
    "integrate_rk4",
    "integrate_rk45",
    "integrate_euler",
    "rk4_step",
    "rk45_step",
    "IntegrationResult",
    "Trajectory",
    "IntegrationInfo",
    "EventSpec",
    "EventResult",
    "detect_events",
    "estimate_stiffness",
    "integrate_with_grad",
    "sensitivity_analysis",
    "Method",
    "DerivativesFn",
    # Flowsheet
    "DynamicFlowsheet",
    "DynamicFlowsheetResult",
    "DynamicUnitEntry",
    "Connection",
    "FlowsheetState",
    # DAE
    "AlgebraicVar",
    "AlgebraicSpec",
    "AlgebraicVector",
    "DAEUnit",
    "DAEUnitBase",
    "integrate_dae",
    "dae_step_euler",
    "dae_step_rk4",
    "DAEResult",
    "DAEMethod",
    "newton_solve",
    "solve_algebraic",
    "vapor_fraction_algebraic",
    "k_value_algebraic",
    "pressure_algebraic",
    "equilibrium_extent_algebraic",
    "DynamicFlashDrum",
    # Diffrax (optional)
    "HAS_DIFFRAX",
    "DIFFRAX_SOLVERS",
    "integrate_diffrax",
    "integrate_diffrax_unit",
    "integrate_stiff",
    "integrate_dopri5",
    "integrate_tsit5",
    "integrate_kvaerno5",
    "list_diffrax_solvers",
    "check_diffrax_available",
]
