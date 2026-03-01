"""Differential-Algebraic Equation (DAE) support for dynamic modeling.

This module extends the unified dynamic modeling framework to support
systems with algebraic constraints. A DAE system has the form:

    dx/dt = f(t, x, z)    (differential equations)
    0 = g(t, x, z)        (algebraic equations)

where x are differential states and z are algebraic states.

Key Use Cases
-------------
- Flash equilibrium: Phase split computed at each time step
- Pressure controllers: Pressure held at setpoint
- Fast reactions: Assumed at equilibrium
- Mass/energy balance constraints

Key Components
--------------

**Algebraic State Specification**:
- AlgebraicVar: Single algebraic variable specification
- AlgebraicSpec: Collection of algebraic variables
- algebraic_states(): Helper for common patterns

**DAE Unit Protocol**:
- DAEUnit: Extended protocol with algebraic_residual method
- DAEUnitBase: Base class with common functionality

**DAE Integration**:
- integrate_dae(): Main DAE integration interface
- solve_algebraic(): Newton solver for algebraic constraints
- dae_step(): Single DAE integration step

Example Usage
-------------

>>> from difflow.dynamic.dae import DAEUnit, DAEUnitBase, integrate_dae
>>> import jax.numpy as jnp
>>>
>>> # Define a flash drum with VLE equilibrium
>>> class DynamicFlash(DAEUnitBase):
...     def _build_state_spec(self):
...         # Differential: total moles, temperature
...         return reactor_states(["A", "B"], include_T=True)
...
...     def _build_algebraic_spec(self):
...         # Algebraic: vapor fraction, component K-values
...         return AlgebraicSpec([
...             AlgebraicVar("beta", description="Vapor fraction"),
...             AlgebraicVar("K_A", description="K-value for A"),
...             AlgebraicVar("K_B", description="K-value for B"),
...         ])
...
...     def _derivatives(self, t, x, z, inputs):
...         # Material/energy balances
...         return dn_dt, dT_dt
...
...     def _algebraic_residual(self, t, x, z, inputs):
...         # VLE equilibrium: K*x_L - y = 0, sum(x) - 1 = 0, etc.
...         return residuals
"""

from typing import Protocol, Callable, Any, Literal, runtime_checkable
from dataclasses import dataclass, field
from functools import partial
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import Array, lax
import optimistix as optx

from difflow.streams import Stream, get_flows, make_stream
from difflow.dynamic.state import StateSpec, StateVar, StateVector
from difflow.thermo import IdealThermo


# Type aliases
Params = dict[str, Any]


# =============================================================================
# Algebraic State Specification
# =============================================================================

@dataclass(frozen=True)
class AlgebraicVar:
    """Specification for a single algebraic variable.

    Algebraic variables are not integrated but solved at each time step
    to satisfy algebraic constraints (residual = 0).

    Attributes:
        name: Unique identifier (e.g., "beta", "K_A")
        units: Physical units string
        description: Human-readable description
        bounds: Optional (lower, upper) bounds
        scale: Characteristic scale for Newton solver
        initial_guess: Initial guess for Newton iteration
    """
    name: str
    units: str = ""
    description: str = ""
    bounds: tuple[float | None, float | None] = (None, None)
    scale: float = 1.0
    initial_guess: float = 1.0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, AlgebraicVar):
            return self.name == other.name
        return False


@dataclass
class AlgebraicSpec:
    """Specification of algebraic variables for a DAE unit.

    Attributes:
        variables: List of algebraic variable specifications
        n_algebraic: Number of algebraic variables
    """
    variables: list[AlgebraicVar] = field(default_factory=list)

    def __post_init__(self):
        self._index_map: dict[str, int] = {
            var.name: i for i, var in enumerate(self.variables)
        }

    @property
    def n_algebraic(self) -> int:
        """Number of algebraic variables."""
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        """List of algebraic variable names in order."""
        return [var.name for var in self.variables]

    def get_index(self, name: str) -> int:
        """Get index of an algebraic variable by name."""
        return self._index_map[name]

    def get_var(self, name: str) -> AlgebraicVar:
        """Get algebraic variable specification by name."""
        return self.variables[self._index_map[name]]

    def get_scales(self) -> Array:
        """Get array of scales for all variables."""
        return jnp.array([var.scale for var in self.variables])

    def get_initial_guess(self) -> Array:
        """Get initial guess array for Newton solver."""
        return jnp.array([var.initial_guess for var in self.variables])

    def get_bounds(self) -> tuple[Array, Array]:
        """Get lower and upper bound arrays."""
        lower = jnp.array([
            var.bounds[0] if var.bounds[0] is not None else -jnp.inf
            for var in self.variables
        ])
        upper = jnp.array([
            var.bounds[1] if var.bounds[1] is not None else jnp.inf
            for var in self.variables
        ])
        return lower, upper

    def __add__(self, other: "AlgebraicSpec") -> "AlgebraicSpec":
        """Combine two AlgebraicSpecs."""
        return AlgebraicSpec(self.variables + other.variables)


@dataclass
class AlgebraicVector:
    """Runtime algebraic vector with named access.

    Similar to StateVector but for algebraic variables.
    """
    values: Array
    spec: AlgebraicSpec

    def __getitem__(self, name: str) -> Array:
        """Get algebraic value by name."""
        return self.values[self.spec.get_index(name)]

    def to_dict(self) -> dict[str, Array]:
        """Convert to dictionary."""
        return {name: self.values[i] for i, name in enumerate(self.spec.names)}


# =============================================================================
# DAE Unit Protocol
# =============================================================================

@runtime_checkable
class DAEUnit(Protocol):
    """Protocol for dynamic units with algebraic constraints.

    Extends DynamicUnit with algebraic state support. Units implementing
    this protocol can be used with DAE integrators.

    The key methods are:
    - state_spec: Differential state specification
    - algebraic_spec: Algebraic state specification
    - derivatives: dx/dt = f(t, x, z, inputs)
    - algebraic_residual: 0 = g(t, x, z, inputs)
    """

    def state_spec(self) -> StateSpec:
        """Return specification of differential state variables."""
        ...

    def algebraic_spec(self) -> AlgebraicSpec:
        """Return specification of algebraic variables."""
        ...

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial differential state."""
        ...

    def initial_algebraic(
        self,
        x: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial algebraic state (solve g(x,z)=0)."""
        ...

    def derivatives(
        self,
        t: Array,
        x: Array,
        z: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute differential state derivatives: dx/dt = f(t, x, z)."""
        ...

    def algebraic_residual(
        self,
        t: Array,
        x: Array,
        z: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute algebraic residual: g(t, x, z) = 0 at solution."""
        ...

    def outputs(
        self,
        t: Array,
        x: Array,
        z: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet streams from differential and algebraic states."""
        ...


class DAEUnitBase(ABC):
    """Abstract base class for DAE units with common functionality.

    Provides:
    - State/algebraic spec caching
    - Newton solver for algebraic constraints
    - StateVector/AlgebraicVector wrapping
    """

    def __init__(
        self,
        params: Params | None = None,
        name: str | None = None,
    ):
        """Initialize DAE unit.

        Args:
            params: Unit parameters
            name: Optional unit name
        """
        self._params = params or {}
        self.name = name or self.__class__.__name__
        self._state_spec: StateSpec | None = None
        self._algebraic_spec: AlgebraicSpec | None = None

    @property
    def params(self) -> Params:
        """Unit parameters."""
        return self._params

    @abstractmethod
    def _build_state_spec(self) -> StateSpec:
        """Build differential state specification."""
        ...

    @abstractmethod
    def _build_algebraic_spec(self) -> AlgebraicSpec:
        """Build algebraic state specification."""
        ...

    def state_spec(self) -> StateSpec:
        """Return differential state specification (cached)."""
        if self._state_spec is None:
            self._state_spec = self._build_state_spec()
        return self._state_spec

    def algebraic_spec(self) -> AlgebraicSpec:
        """Return algebraic state specification (cached)."""
        if self._algebraic_spec is None:
            self._algebraic_spec = self._build_algebraic_spec()
        return self._algebraic_spec

    @abstractmethod
    def _derivatives(
        self,
        t: Array,
        x: StateVector,
        z: AlgebraicVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute derivatives using wrapped state vectors."""
        ...

    @abstractmethod
    def _algebraic_residual(
        self,
        t: Array,
        x: StateVector,
        z: AlgebraicVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute algebraic residual using wrapped state vectors."""
        ...

    @abstractmethod
    def _outputs(
        self,
        t: Array,
        x: StateVector,
        z: AlgebraicVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outputs using wrapped state vectors."""
        ...

    def derivatives(
        self,
        t: Array,
        x: Array,
        z: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute derivatives (public interface)."""
        if params is not None:
            self._params.update(params)
        x_vec = StateVector(x, self.state_spec())
        z_vec = AlgebraicVector(z, self.algebraic_spec())
        return self._derivatives(t, x_vec, z_vec, inputs)

    def algebraic_residual(
        self,
        t: Array,
        x: Array,
        z: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute algebraic residual (public interface)."""
        if params is not None:
            self._params.update(params)
        x_vec = StateVector(x, self.state_spec())
        z_vec = AlgebraicVector(z, self.algebraic_spec())
        return self._algebraic_residual(t, x_vec, z_vec, inputs)

    def outputs(
        self,
        t: Array,
        x: Array,
        z: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outputs (public interface)."""
        if params is not None:
            self._params.update(params)
        x_vec = StateVector(x, self.state_spec())
        z_vec = AlgebraicVector(z, self.algebraic_spec())
        return self._outputs(t, x_vec, z_vec, inputs)

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Default initial state from spec defaults."""
        return self.state_spec().get_default_initial()

    def initial_algebraic(
        self,
        x: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Solve for initial algebraic state."""
        z0 = self.algebraic_spec().get_initial_guess()

        # Solve g(x, z) = 0 for z
        def residual_fn(z):
            return self.algebraic_residual(jnp.array(0.0), x, z, inputs, params)

        z_solved, _ = newton_solve(residual_fn, z0)
        return z_solved

    def __repr__(self) -> str:
        n_diff = self.state_spec().n_states
        n_alg = self.algebraic_spec().n_algebraic
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"n_diff={n_diff}, n_alg={n_alg})"
        )


# =============================================================================
# Newton Solver for Algebraic Constraints (using optimistix)
# =============================================================================

def newton_solve(
    residual_fn: Callable[[Array], Array],
    z0: Array,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
) -> tuple[Array, dict]:
    """Solve g(z) = 0 using Newton's method via optimistix.

    Uses optimistix.Newton solver for robust root finding with
    automatic differentiation for the Jacobian.

    Args:
        residual_fn: Function z -> g(z) where we want g(z) = 0
        z0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum iterations
        damping: Step size damping factor (0 < damping <= 1)
                 Note: optimistix handles line search internally

    Returns:
        (z_solution, info): Solution and convergence info
    """
    # Wrap residual function for optimistix (expects (y, args) signature)
    def optx_residual(z, args):
        return residual_fn(z)

    # Create Newton solver with specified tolerances
    solver = optx.Newton(rtol=tol, atol=tol)

    # Solve using optimistix root_find
    sol = optx.root_find(
        optx_residual,
        solver,
        z0,
        args=None,
        max_steps=max_iter,
        throw=False,
    )

    # Build info dict compatible with previous interface
    info = {
        "converged": jnp.array(sol.result == optx.RESULTS.successful),
        "residual_history": jnp.array([]),  # optimistix doesn't expose this
        "n_iter": max_iter,  # Could use sol.stats if available
        "result": sol.result,
    }

    return sol.value, info


def solve_algebraic(
    unit: DAEUnit,
    t: Array,
    x: Array,
    z_guess: Array,
    inputs: dict[str, Stream],
    params: Params | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[Array, dict]:
    """Solve algebraic constraints for a DAE unit.

    Finds z such that unit.algebraic_residual(t, x, z, inputs) = 0.

    Args:
        unit: DAE unit
        t: Current time
        x: Differential state
        z_guess: Initial guess for algebraic state
        inputs: Input streams
        params: Optional parameters
        tol: Convergence tolerance
        max_iter: Maximum Newton iterations

    Returns:
        (z_solution, info): Algebraic solution and convergence info
    """
    def residual_fn(z):
        return unit.algebraic_residual(t, x, z, inputs, params)

    return newton_solve(residual_fn, z_guess, tol=tol, max_iter=max_iter)


# =============================================================================
# DAE Integration
# =============================================================================

@dataclass
class DAEResult:
    """Result from DAE integration.

    Attributes:
        x_final: Final differential state
        z_final: Final algebraic state
        t_history: Time points
        x_history: Differential state history
        z_history: Algebraic state history
        info: Integration statistics
    """
    x_final: Array
    z_final: Array
    t_history: Array
    x_history: Array
    z_history: Array
    info: dict


def dae_step_euler(
    unit: DAEUnit,
    t: Array,
    x: Array,
    z: Array,
    dt: Array,
    inputs: dict[str, Stream],
    params: Params | None = None,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 50,
) -> tuple[Array, Array]:
    """Single implicit Euler step for DAE.

    For DAE: x_{n+1} = x_n + dt * f(t_{n+1}, x_{n+1}, z_{n+1})
            0 = g(t_{n+1}, x_{n+1}, z_{n+1})

    We use a semi-explicit approach:
    1. Predict x_{n+1} using explicit Euler
    2. Solve for z_{n+1} using Newton
    3. Correct x_{n+1} using the solved z

    Args:
        unit: DAE unit
        t: Current time
        x: Current differential state
        z: Current algebraic state
        dt: Time step
        inputs: Input streams
        params: Optional parameters
        newton_tol: Tolerance for Newton solver
        newton_max_iter: Max Newton iterations

    Returns:
        (x_new, z_new): States at t + dt
    """
    t_new = t + dt

    # Explicit Euler prediction for x
    dx = unit.derivatives(t, x, z, inputs, params)
    x_pred = x + dt * dx

    # Solve algebraic constraints at new time with predicted x
    z_new, _ = solve_algebraic(
        unit, t_new, x_pred, z, inputs, params,
        tol=newton_tol, max_iter=newton_max_iter
    )

    # Corrector step (optional - could iterate)
    dx_new = unit.derivatives(t_new, x_pred, z_new, inputs, params)
    x_new = x + dt * dx_new

    return x_new, z_new


def dae_step_rk4(
    unit: DAEUnit,
    t: Array,
    x: Array,
    z: Array,
    dt: Array,
    inputs: dict[str, Stream],
    params: Params | None = None,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 50,
) -> tuple[Array, Array]:
    """RK4 step for DAE with algebraic solve at each stage.

    At each RK4 stage, we solve the algebraic constraints to get z,
    then compute the derivatives using that z.

    Args:
        unit: DAE unit
        t: Current time
        x: Current differential state
        z: Current algebraic state
        dt: Time step
        inputs: Input streams
        params: Optional parameters

    Returns:
        (x_new, z_new): States at t + dt
    """
    # Stage 1: k1 at (t, x, z)
    z1 = z  # Use current z
    k1 = unit.derivatives(t, x, z1, inputs, params)

    # Stage 2: k2 at (t + dt/2, x + dt*k1/2, z2)
    x2 = x + 0.5 * dt * k1
    z2, _ = solve_algebraic(
        unit, t + 0.5 * dt, x2, z1, inputs, params,
        tol=newton_tol, max_iter=newton_max_iter
    )
    k2 = unit.derivatives(t + 0.5 * dt, x2, z2, inputs, params)

    # Stage 3: k3 at (t + dt/2, x + dt*k2/2, z3)
    x3 = x + 0.5 * dt * k2
    z3, _ = solve_algebraic(
        unit, t + 0.5 * dt, x3, z2, inputs, params,
        tol=newton_tol, max_iter=newton_max_iter
    )
    k3 = unit.derivatives(t + 0.5 * dt, x3, z3, inputs, params)

    # Stage 4: k4 at (t + dt, x + dt*k3, z4)
    x4 = x + dt * k3
    z4, _ = solve_algebraic(
        unit, t + dt, x4, z3, inputs, params,
        tol=newton_tol, max_iter=newton_max_iter
    )
    k4 = unit.derivatives(t + dt, x4, z4, inputs, params)

    # Combine
    x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Final algebraic solve at new state
    z_new, _ = solve_algebraic(
        unit, t + dt, x_new, z4, inputs, params,
        tol=newton_tol, max_iter=newton_max_iter
    )

    return x_new, z_new


DAEMethod = Literal["Euler", "RK4"]


def integrate_dae(
    unit: DAEUnit,
    inputs: dict[str, Stream],
    t_span: tuple[float, float],
    x0: Array | None = None,
    z0: Array | None = None,
    method: DAEMethod = "RK4",
    n_steps: int = 100,
    params: Params | None = None,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 50,
) -> DAEResult:
    """Integrate a DAE unit over time.

    Integrates the differential states while solving algebraic constraints
    at each time step.

    Args:
        unit: DAE unit to integrate
        inputs: Input streams
        t_span: (t_start, t_end) time interval
        x0: Initial differential state (uses unit.initial_state if None)
        z0: Initial algebraic state (uses unit.initial_algebraic if None)
        method: Integration method ("Euler" or "RK4")
        n_steps: Number of integration steps
        params: Optional parameters
        newton_tol: Tolerance for algebraic solver
        newton_max_iter: Max iterations for algebraic solver

    Returns:
        DAEResult with trajectories and final states
    """
    t0, t_final = t_span
    dt = (t_final - t0) / n_steps

    t0 = jnp.asarray(t0)
    dt = jnp.asarray(dt)

    # Initialize states
    if x0 is None:
        x0 = unit.initial_state(inputs, params)
    if z0 is None:
        z0 = unit.initial_algebraic(x0, inputs, params)

    x0 = jnp.asarray(x0)
    z0 = jnp.asarray(z0)

    # Select step function
    if method == "Euler":
        step_fn = partial(
            dae_step_euler,
            unit=unit,
            inputs=inputs,
            params=params,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
        )
    else:  # RK4
        step_fn = partial(
            dae_step_rk4,
            unit=unit,
            inputs=inputs,
            params=params,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
        )

    def scan_step(carry, _):
        t, x, z = carry
        x_new, z_new = step_fn(t=t, x=x, z=z, dt=dt)
        t_new = t + dt
        return (t_new, x_new, z_new), (t_new, x_new, z_new)

    # Run integration
    (t_end, x_final, z_final), (t_hist, x_hist, z_hist) = lax.scan(
        scan_step, (t0, x0, z0), None, length=n_steps
    )

    # Prepend initial state
    t_history = jnp.concatenate([jnp.array([t0]), t_hist])
    x_history = jnp.vstack([x0, x_hist])
    z_history = jnp.vstack([z0, z_hist])

    # Compute final algebraic residual for validation
    final_residual = unit.algebraic_residual(t_end, x_final, z_final, inputs, params)
    max_residual = jnp.max(jnp.abs(final_residual))

    return DAEResult(
        x_final=x_final,
        z_final=z_final,
        t_history=t_history,
        x_history=x_history,
        z_history=z_history,
        info={
            "method": method,
            "n_steps": n_steps,
            "dt": dt,
            "max_algebraic_residual": max_residual,
            "algebraic_converged": max_residual < newton_tol * 100,
        },
    )


# =============================================================================
# Utility Functions for Common Algebraic Specifications
# =============================================================================

def vapor_fraction_algebraic() -> AlgebraicSpec:
    """Create AlgebraicSpec for vapor fraction in VLE."""
    return AlgebraicSpec([
        AlgebraicVar(
            name="beta",
            units="-",
            description="Vapor fraction",
            bounds=(0.0, 1.0),
            scale=1.0,
            initial_guess=0.5,
        )
    ])


def k_value_algebraic(species: list[str]) -> AlgebraicSpec:
    """Create AlgebraicSpec for K-values in VLE."""
    return AlgebraicSpec([
        AlgebraicVar(
            name=f"K_{s}",
            units="-",
            description=f"K-value for {s}",
            bounds=(1e-6, 1e6),
            scale=1.0,
            initial_guess=1.0,
        )
        for s in species
    ])


def pressure_algebraic(name: str = "P", scale: float = 101325.0) -> AlgebraicSpec:
    """Create AlgebraicSpec for pressure (algebraic constraint)."""
    return AlgebraicSpec([
        AlgebraicVar(
            name=name,
            units="Pa",
            description="Pressure (algebraic)",
            bounds=(0.0, None),
            scale=scale,
            initial_guess=101325.0,
        )
    ])


def equilibrium_extent_algebraic(n_rxns: int = 1) -> AlgebraicSpec:
    """Create AlgebraicSpec for equilibrium reaction extents."""
    return AlgebraicSpec([
        AlgebraicVar(
            name=f"xi_{i}",
            units="mol",
            description=f"Extent of equilibrium reaction {i}",
            bounds=(None, None),
            scale=1.0,
            initial_guess=0.0,
        )
        for i in range(n_rxns)
    ])


# =============================================================================
# Example DAE Units
# =============================================================================

class DynamicFlashDrum(DAEUnitBase):
    """Dynamic flash drum with VLE equilibrium constraint.

    A flash drum where the vapor-liquid equilibrium is assumed to be
    instantaneous (algebraic constraint) while the total holdup
    evolves dynamically.

    Differential states:
    - n_i: Total moles of each species in drum (mol)
    - H: Total enthalpy in drum (J)

    Algebraic states:
    - beta: Vapor fraction
    - K_i: K-values for each species

    The VLE equilibrium is enforced at each time step via:
    - Rachford-Rice equation for beta
    - K-value relations (e.g., Raoult's law)
    """

    def __init__(
        self,
        volume: float,
        species_order: list[str],
        thermo: IdealThermo | None = None,
        P: float = 101325.0,
        K_func: Callable[[Array], Array] | None = None,
        name: str | None = None,
    ):
        """Initialize dynamic flash drum.

        Args:
            volume: Drum volume (m³)
            species_order: List of species names
            thermo: Thermodynamic property calculator.  When provided, the
                energy balance is computed from actual inlet/outlet enthalpy
                flows and K-values are evaluated at the current drum
                temperature.  When None, dH/dt = 0 (isothermal) and K=2.
            P: Operating pressure (Pa).  Used for K-value calculation when
                thermo is provided.
            K_func: Deprecated — use thermo instead.  Function T -> K-values
                array used when thermo is None (default: constant K=2).
            name: Unit name
        """
        params = {
            "V": jnp.asarray(volume),
            "species_order": species_order,
            "K_func": K_func,
            "P": jnp.asarray(P),
        }
        self.thermo = thermo
        super().__init__(params, name)

    def _drum_temperature(
        self,
        H_total: Array,
        x_comp: Array,
        n_total: Array,
        T_guess: Array,
    ) -> Array:
        """Solve for drum temperature from total enthalpy and composition.

        Inverts h_mix(T) = H_total / n_total where h_mix is the mole-fraction-
        weighted liquid-phase enthalpy.  Uses a Newton solve via optimistix.

        Args:
            H_total: Total enthalpy in drum (J)
            x_comp: Mole fractions array (nc,)
            n_total: Total moles in drum (mol)
            T_guess: Initial temperature guess (K)

        Returns:
            Drum temperature (K)
        """
        species = self.params["species_order"]
        h_target = H_total / jnp.maximum(n_total, 1e-10)

        def residual(T, args):
            h_mix = sum(
                x_comp[i] * self.thermo.H_pure(s, T, "liquid")
                for i, s in enumerate(species)
            )
            return h_mix - h_target

        solver = optx.Newton(rtol=1e-6, atol=1e-6)
        sol = optx.root_find(
            residual, solver, T_guess, args=None, max_steps=50, throw=False
        )
        return sol.value

    def _k_values(self, x_comp: Array, T: Array) -> Array:
        """Return K-values at current drum conditions.

        Uses thermo.K_values_array when a thermo object is available,
        otherwise falls back to K_func(T) or constant K=2.

        Args:
            x_comp: Liquid mole fractions (unused, kept for API symmetry)
            T: Drum temperature (K)

        Returns:
            K-values array (nc,)
        """
        p = self.params
        species = p["species_order"]
        if self.thermo is not None:
            return self.thermo.K_values_array(T, p["P"])
        elif p["K_func"] is not None:
            return p["K_func"](T)
        else:
            return jnp.ones(len(species)) * 2.0

    def _build_state_spec(self) -> StateSpec:
        """Differential: moles + enthalpy."""
        from difflow.dynamic.state import molar_states

        species = self.params["species_order"]
        spec = molar_states(species)

        # Add enthalpy state
        spec = spec.add(StateVar(
            name="H",
            category="generic",
            units="J",
            description="Total enthalpy",
            scale=1e6,
            initial_value=0.0,
        ))

        return spec

    def _build_algebraic_spec(self) -> AlgebraicSpec:
        """Algebraic: vapor fraction."""
        # For simplicity, just beta (vapor fraction)
        # Full VLE would include K-values
        return vapor_fraction_algebraic()

    def _derivatives(
        self,
        t: Array,
        x: StateVector,
        z: AlgebraicVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Material and energy balances."""
        p = self.params
        species = p["species_order"]
        n_sp = len(species)

        # Get moles
        n = jnp.array([x[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10

        # Inlet stream
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in = jnp.array([inlet_flows.get(s, 0.0) for s in species])

        # Vapor fraction
        beta = z["beta"]

        # Outlet flows (assume vapor and liquid both exit)
        # Simplified: total outlet = total inlet (steady level)
        F_out_total = jnp.sum(F_in)

        # Composition in drum
        x_comp = n / n_total

        # Outlet composition (simplified: mixed)
        F_out = F_out_total * x_comp

        # Material balance: dn/dt = F_in - F_out
        dn_dt = F_in - F_out

        # Energy balance: dH/dt = H_dot_in - H_dot_out
        # H_dot = sum_i F_i * h_i(T)  (enthalpy flow rate, J/s)
        if self.thermo is not None:
            species = p["species_order"]
            T_in = inlet["T"]

            # Inlet enthalpy flow rate (liquid-phase)
            H_dot_in = sum(
                float_flow * self.thermo.H_pure(s, T_in, "liquid")
                for float_flow, s in zip(F_in, species)
            )

            # Drum temperature: solve h_mix(T) = H_state / n_total
            H_state = x["H"]
            T_drum = self._drum_temperature(H_state, x_comp, n_total, T_in)

            # Outlet enthalpy flow rate at drum conditions (liquid approximation)
            H_dot_out = sum(
                F_out[i] * self.thermo.H_pure(s, T_drum, "liquid")
                for i, s in enumerate(species)
            )

            dH_dt = jnp.array([H_dot_in - H_dot_out])
        else:
            # No thermo available: isothermal assumption (dH/dt = 0)
            dH_dt = jnp.array([0.0])

        return jnp.concatenate([dn_dt, dH_dt])

    def _algebraic_residual(
        self,
        t: Array,
        x: StateVector,
        z: AlgebraicVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """VLE equilibrium constraint (Rachford-Rice)."""
        p = self.params
        species = p["species_order"]

        # Get compositions
        n = jnp.array([x[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10
        z_comp = n / n_total  # Overall composition

        beta = z["beta"]

        # K-values at current drum temperature.
        # Derive drum T from the enthalpy state so K-values respond to
        # the energy balance rather than being stuck at a hardcoded 300 K.
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        T_guess = inlet["T"]
        if self.thermo is not None:
            H_state = x["H"]
            T_drum = self._drum_temperature(H_state, z_comp, n_total, T_guess)
        else:
            T_drum = T_guess
        K = self._k_values(z_comp, T_drum)

        # Rachford-Rice: sum_i[ z_i(K_i - 1) / (1 + beta(K_i - 1)) ] = 0
        num = z_comp * (K - 1.0)
        den = 1.0 + beta * (K - 1.0)
        residual = jnp.sum(num / den)

        return jnp.array([residual])

    def _outputs(
        self,
        t: Array,
        x: StateVector,
        z: AlgebraicVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute vapor and liquid outlet streams."""
        p = self.params
        species = p["species_order"]

        # Get compositions
        n = jnp.array([x[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10
        z_comp = n / n_total

        beta = z["beta"]

        # K-values at current drum temperature
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        T_guess = inlet["T"]
        if self.thermo is not None:
            H_state = x["H"]
            n = jnp.array([x[f"n_{s}"] for s in species])
            n_total = jnp.sum(n) + 1e-10
            z_comp = n / n_total
            T_drum = self._drum_temperature(H_state, z_comp, n_total, T_guess)
        else:
            T_drum = T_guess
        K = self._k_values(z_comp, T_drum)

        # Liquid and vapor compositions
        x_L = z_comp / (1.0 + beta * (K - 1.0))
        y_V = K * x_L

        # Normalize
        x_L = x_L / jnp.sum(x_L)
        y_V = y_V / jnp.sum(y_V)

        # Get inlet for flow rates
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in_total = sum(inlet_flows.values())

        # Split flows
        F_L = F_in_total * (1 - beta)
        F_V = F_in_total * beta

        # Create outlet streams
        liquid_flows = {s: F_L * x_L[i] for i, s in enumerate(species)}
        vapor_flows = {s: F_V * y_V[i] for i, s in enumerate(species)}

        T = inlet["T"]
        P = inlet["P"]

        return {
            "liquid": make_stream(liquid_flows, T, P),
            "vapor": make_stream(vapor_flows, T, P),
        }

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Initialize with inlet composition."""
        p = self.params
        species = p["species_order"]

        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)

        # Initial moles (assume residence time of 60s)
        tau = 60.0
        n0 = jnp.array([inlet_flows.get(s, 0.0) * tau for s in species])

        # Initial enthalpy: compute from inlet conditions if thermo available
        if self.thermo is not None:
            T_in = inlet["T"]
            n_total = jnp.sum(n0) + 1e-10
            x_comp0 = n0 / n_total
            H0_val = n_total * sum(
                x_comp0[i] * self.thermo.H_pure(s, T_in, "liquid")
                for i, s in enumerate(species)
            )
            H0 = jnp.array([H0_val])
        else:
            H0 = jnp.array([0.0])

        return jnp.concatenate([n0, H0])
