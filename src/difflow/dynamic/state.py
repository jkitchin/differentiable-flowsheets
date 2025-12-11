"""State specification and management for dynamic modeling.

This module provides data structures for defining and managing state variables
in dynamic simulations. It enables a unified approach where units declare their
states, and the integrator assembles them into a global state vector.

Key concepts:
- StateVar: A single state variable with metadata
- StateSpec: Collection of state variables for a unit
- StateVector: Runtime container mapping states to array positions

All structures are designed to work seamlessly with JAX pytrees.
"""

from typing import Literal, Callable, Any
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array


# State variable categories
StateCategory = Literal[
    "moles",        # Molar amounts (mol)
    "concentration", # Concentrations (mol/m³)
    "temperature",   # Temperature (K)
    "pressure",      # Pressure (Pa)
    "volume",        # Volume (m³)
    "mass",          # Mass (kg)
    "fraction",      # Mole/mass fractions (dimensionless)
    "extent",        # Reaction extent (mol)
    "generic",       # Other state variables
]


@dataclass(frozen=True)
class StateVar:
    """Specification for a single state variable.

    Attributes:
        name: Unique identifier within the unit (e.g., "n_A", "T", "V")
        category: Type of state variable for unit checking and scaling
        units: Physical units string for documentation
        description: Human-readable description
        bounds: Optional (lower, upper) bounds for the variable
        scale: Characteristic scale for normalization (default 1.0)
        initial_value: Default initial value if not specified
    """
    name: str
    category: StateCategory = "generic"
    units: str = ""
    description: str = ""
    bounds: tuple[float | None, float | None] = (None, None)
    scale: float = 1.0
    initial_value: float | None = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, StateVar):
            return self.name == other.name
        return False


@dataclass
class StateSpec:
    """Specification of all state variables for a unit.

    This defines the state "shape" of a dynamic unit - what variables
    it tracks and their metadata. Used by the integrator to assemble
    the global state vector.

    Attributes:
        variables: List of state variable specifications
        n_states: Total number of state variables

    Example:
        >>> spec = StateSpec([
        ...     StateVar("n_A", "moles", "mol", "Moles of species A"),
        ...     StateVar("n_B", "moles", "mol", "Moles of species B"),
        ...     StateVar("T", "temperature", "K", "Reactor temperature"),
        ... ])
        >>> spec.n_states
        3
        >>> spec.get_index("T")
        2
    """
    variables: list[StateVar] = field(default_factory=list)

    def __post_init__(self):
        # Build name-to-index mapping
        self._index_map: dict[str, int] = {
            var.name: i for i, var in enumerate(self.variables)
        }

    @property
    def n_states(self) -> int:
        """Total number of state variables."""
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        """List of state variable names in order."""
        return [var.name for var in self.variables]

    def get_index(self, name: str) -> int:
        """Get index of a state variable by name."""
        return self._index_map[name]

    def get_var(self, name: str) -> StateVar:
        """Get state variable specification by name."""
        return self.variables[self._index_map[name]]

    def get_indices(self, names: list[str]) -> list[int]:
        """Get indices for multiple state variables."""
        return [self._index_map[name] for name in names]

    def get_scales(self) -> Array:
        """Get array of scales for all variables."""
        return jnp.array([var.scale for var in self.variables])

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

    def get_default_initial(self) -> Array:
        """Get default initial values (0.0 if not specified)."""
        return jnp.array([
            var.initial_value if var.initial_value is not None else 0.0
            for var in self.variables
        ])

    def add(self, var: StateVar) -> "StateSpec":
        """Add a state variable and return new StateSpec."""
        new_vars = self.variables + [var]
        return StateSpec(new_vars)

    def __add__(self, other: "StateSpec") -> "StateSpec":
        """Combine two StateSpecs."""
        return StateSpec(self.variables + other.variables)

    def subset(self, names: list[str]) -> "StateSpec":
        """Create a new StateSpec with only the named variables."""
        return StateSpec([self.get_var(name) for name in names])

    def by_category(self, category: StateCategory) -> list[StateVar]:
        """Get all variables of a given category."""
        return [v for v in self.variables if v.category == category]


@dataclass
class StateVector:
    """Runtime state vector with named access.

    Wraps a JAX array with the ability to access elements by name.
    This is the primary interface for working with states in
    derivative functions.

    Attributes:
        values: The underlying JAX array of state values
        spec: The state specification defining variable names/indices

    Example:
        >>> spec = StateSpec([StateVar("T"), StateVar("P")])
        >>> state = StateVector(jnp.array([300.0, 101325.0]), spec)
        >>> state["T"]
        Array(300., dtype=float32)
        >>> state.get(["T", "P"])
        Array([300., 101325.], dtype=float32)
    """
    values: Array
    spec: StateSpec

    def __getitem__(self, name: str) -> Array:
        """Get state value by name."""
        return self.values[self.spec.get_index(name)]

    def get(self, names: list[str]) -> Array:
        """Get multiple state values as array."""
        indices = self.spec.get_indices(names)
        return self.values[jnp.array(indices)]

    def to_dict(self) -> dict[str, Array]:
        """Convert to dictionary of name -> value."""
        return {name: self.values[i] for i, name in enumerate(self.spec.names)}

    def with_update(self, updates: dict[str, Array]) -> "StateVector":
        """Create new StateVector with updated values."""
        new_values = self.values.at[
            jnp.array([self.spec.get_index(k) for k in updates])
        ].set(jnp.array(list(updates.values())))
        return StateVector(new_values, self.spec)

    @classmethod
    def from_dict(cls, values: dict[str, Array], spec: StateSpec) -> "StateVector":
        """Create StateVector from dictionary."""
        arr = jnp.array([values[name] for name in spec.names])
        return cls(arr, spec)

    @property
    def normalized(self) -> Array:
        """Return state values normalized by their scales."""
        return self.values / self.spec.get_scales()


# =============================================================================
# Utility functions for common state specifications
# =============================================================================

def molar_states(species: list[str], prefix: str = "n") -> StateSpec:
    """Create StateSpec for molar amounts of species.

    Args:
        species: List of species names
        prefix: Prefix for state names (default "n" for moles)

    Returns:
        StateSpec with one state per species

    Example:
        >>> spec = molar_states(["A", "B", "C"])
        >>> spec.names
        ['n_A', 'n_B', 'n_C']
    """
    return StateSpec([
        StateVar(
            name=f"{prefix}_{s}",
            category="moles",
            units="mol",
            description=f"Moles of {s}",
            bounds=(0.0, None),
            scale=1.0,
        )
        for s in species
    ])


def concentration_states(species: list[str], prefix: str = "C") -> StateSpec:
    """Create StateSpec for concentrations of species."""
    return StateSpec([
        StateVar(
            name=f"{prefix}_{s}",
            category="concentration",
            units="mol/m³",
            description=f"Concentration of {s}",
            bounds=(0.0, None),
            scale=1.0,
        )
        for s in species
    ])


def thermal_state(name: str = "T", scale: float = 300.0) -> StateSpec:
    """Create StateSpec for temperature."""
    return StateSpec([
        StateVar(
            name=name,
            category="temperature",
            units="K",
            description="Temperature",
            bounds=(0.0, None),
            scale=scale,
            initial_value=300.0,
        )
    ])


def volume_state(name: str = "V", scale: float = 1.0) -> StateSpec:
    """Create StateSpec for volume."""
    return StateSpec([
        StateVar(
            name=name,
            category="volume",
            units="m³",
            description="Volume",
            bounds=(0.0, None),
            scale=scale,
            initial_value=1.0,
        )
    ])


def pressure_state(name: str = "P", scale: float = 101325.0) -> StateSpec:
    """Create StateSpec for pressure."""
    return StateSpec([
        StateVar(
            name=name,
            category="pressure",
            units="Pa",
            description="Pressure",
            bounds=(0.0, None),
            scale=scale,
            initial_value=101325.0,
        )
    ])


def reactor_states(species: list[str], include_T: bool = True) -> StateSpec:
    """Create common reactor state specification.

    Creates states for molar amounts and optionally temperature.

    Args:
        species: List of species names
        include_T: Whether to include temperature state

    Returns:
        Combined StateSpec
    """
    spec = molar_states(species)
    if include_T:
        spec = spec + thermal_state()
    return spec
