"""Base classes and protocols for dynamic unit operations.

This module defines the interface that all dynamic units must implement
to work with the unified integrator framework. Units can operate in
different modes:

- **Pure ODE**: dx/dt = f(x, u, t) - state evolves continuously
- **Pure algebraic**: 0 = g(x, u) - instantaneous equilibrium (steady-state)
- **DAE (mixed)**: dx/dt = f(x, y, u, t), 0 = g(x, y, u, t)

The DynamicUnit protocol is the core abstraction that allows the
integrator to work with any unit type uniformly.
"""

from typing import Protocol, Callable, Any, Literal, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, make_stream
from difflow.dynamic.state import StateSpec, StateVector, StateVar


# Type aliases
Params = dict[str, Any]
RateFunction = Callable[[dict[str, Array], Array, Params], Array]


@runtime_checkable
class DynamicUnit(Protocol):
    """Protocol defining the interface for dynamic unit operations.

    All dynamic units must implement these methods to work with the
    unified integrator. This enables polymorphic handling of different
    unit types (reactors, separators, tanks, etc.) in a flowsheet.

    The key methods are:
    - state_spec: Declares what state variables the unit tracks
    - derivatives: Computes dx/dt for ODE integration
    - outputs: Maps internal states to outlet stream(s)

    Optional methods for advanced functionality:
    - residual: For steady-state or algebraic constraints
    - events: For discontinuity detection (phase changes, etc.)
    """

    def state_spec(self) -> StateSpec:
        """Return specification of state variables for this unit.

        This defines the "shape" of the unit's dynamic state - what
        variables are tracked, their units, bounds, and scales.

        Returns:
            StateSpec describing all state variables
        """
        ...

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial state from inlet streams and parameters.

        Args:
            inputs: Dictionary of inlet streams by name
            params: Optional parameters to override defaults

        Returns:
            Initial state array matching state_spec order
        """
        ...

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives of state variables.

        This is the core ODE function: dx/dt = f(t, x, inputs, params)

        Args:
            t: Current time
            state: Current state array (matches state_spec order)
            inputs: Dictionary of inlet streams by name
            params: Optional parameter overrides

        Returns:
            Array of derivatives dx/dt (same shape as state)
        """
        ...

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet streams from current state.

        Maps the internal state to outlet stream(s) that can be
        connected to downstream units.

        Args:
            t: Current time
            state: Current state array
            inputs: Dictionary of inlet streams
            params: Optional parameter overrides

        Returns:
            Dictionary of outlet streams by name
        """
        ...


class DynamicUnitBase(ABC):
    """Abstract base class for dynamic units with common functionality.

    Provides default implementations and utilities while requiring
    subclasses to implement the core dynamic methods.

    This base class handles:
    - Parameter management
    - State vector wrapping/unwrapping
    - Common validation
    - Steady-state solving via derivatives=0

    Subclasses must implement:
    - _build_state_spec(): Define state variables
    - _derivatives(): Compute dx/dt
    - _outputs(): Map states to outlet streams
    """

    def __init__(
        self,
        params: Params | None = None,
        name: str | None = None,
    ):
        """Initialize dynamic unit.

        Args:
            params: Unit parameters (kinetics, geometry, etc.)
            name: Optional name for the unit
        """
        self._params = params or {}
        self.name = name or self.__class__.__name__
        self._state_spec: StateSpec | None = None

    @property
    def params(self) -> Params:
        """Unit parameters."""
        return self._params

    @abstractmethod
    def _build_state_spec(self) -> StateSpec:
        """Build and return the state specification.

        Subclasses must implement this to define their state variables.
        """
        ...

    def state_spec(self) -> StateSpec:
        """Return specification of state variables (cached)."""
        if self._state_spec is None:
            self._state_spec = self._build_state_spec()
        return self._state_spec

    @abstractmethod
    def _derivatives(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute derivatives using StateVector interface.

        Subclasses implement this with convenient named access to states.
        """
        ...

    @abstractmethod
    def _outputs(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outputs using StateVector interface."""
        ...

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives (public interface).

        Wraps state array in StateVector for convenient access.
        """
        if params is not None:
            self._params.update(params)
        state_vec = StateVector(state, self.state_spec())
        return self._derivatives(t, state_vec, inputs)

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet streams (public interface)."""
        if params is not None:
            self._params.update(params)
        state_vec = StateVector(state, self.state_spec())
        return self._outputs(t, state_vec, inputs)

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Default initial state from spec defaults.

        Override for unit-specific initialization logic.
        """
        return self.state_spec().get_default_initial()

    def residual(
        self,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Residual for steady-state: r(x) = dx/dt.

        Default implementation uses derivatives at t=0.
        Override for algebraic constraints.
        """
        return self.derivatives(jnp.array(0.0), state, inputs, params)

    def __repr__(self) -> str:
        n_states = self.state_spec().n_states
        return f"{self.__class__.__name__}(name='{self.name}', n_states={n_states})"


# =============================================================================
# Common Dynamic Unit Implementations
# =============================================================================

class DynamicCSTR(DynamicUnitBase):
    """Dynamic CSTR with holdup dynamics.

    State variables:
    - n_i: Moles of each species in reactor (mol)
    - T: Reactor temperature (K) [if non-isothermal]

    Material balance:
        dn_i/dt = F_in * x_in_i - F_out * x_out_i + V * sum_j(nu_ij * r_j)

    Energy balance (non-isothermal):
        d(n_total * Cp * T)/dt = F_in*H_in - F_out*H_out + Q + V*sum_j(r_j*(-dH_j))

    For perfectly mixed: x_out = n / sum(n)
    """

    def __init__(
        self,
        volume: float | Array,
        rate_fn: RateFunction,
        stoich: Array,
        species_order: list[str],
        rate_params: Params | None = None,
        dH_rxn: Array | None = None,
        mode: Literal["isothermal", "adiabatic", "specified_duty"] = "isothermal",
        name: str | None = None,
    ):
        """Initialize dynamic CSTR.

        Args:
            volume: Reactor volume (m³)
            rate_fn: Reaction rate function: rate_fn(C, T, params) -> r
            stoich: Stoichiometry matrix (n_species, n_reactions)
            species_order: List of species names
            rate_params: Parameters for rate function
            dH_rxn: Heats of reaction (J/mol), negative for exothermic
            mode: Energy balance mode
            name: Unit name
        """
        params = {
            "V": jnp.asarray(volume),
            "rate_fn": rate_fn,
            "stoich": jnp.asarray(stoich),
            "species_order": species_order,
            "rate_params": rate_params or {},
            "dH_rxn": jnp.asarray(dH_rxn) if dH_rxn is not None else None,
            "mode": mode,
        }
        super().__init__(params, name)

    def _build_state_spec(self) -> StateSpec:
        """Build state spec for moles and optionally temperature."""
        from difflow.dynamic.state import molar_states, thermal_state

        species = self.params["species_order"]
        spec = molar_states(species)

        if self.params["mode"] != "isothermal":
            spec = spec + thermal_state()

        return spec

    def _derivatives(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute dn/dt and optionally dT/dt."""
        p = self.params
        species = p["species_order"]
        V = p["V"]

        # Get moles from state
        n = jnp.array([state[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10

        # Compute concentrations
        C = {s: n[i] / V for i, s in enumerate(species)}

        # Get inlet stream (assume single inlet named "inlet")
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in = jnp.array([inlet_flows.get(s, 0.0) for s in species])
        F_in_total = jnp.sum(F_in)

        # Outlet flow (assume constant density, F_out = F_in)
        # For more accurate: compute from pressure/level controller
        F_out_total = F_in_total
        x_out = n / n_total
        F_out = F_out_total * x_out

        # Temperature
        if p["mode"] == "isothermal":
            T = inlet["T"]  # Use inlet T or could use T_spec
        else:
            T = state["T"]

        # Reaction rates
        r = p["rate_fn"](C, T, p["rate_params"])

        # Material balance: dn/dt = F_in - F_out + V * stoich @ r
        dn_dt = F_in - F_out + V * (p["stoich"] @ r)

        if p["mode"] == "isothermal":
            return dn_dt

        # Energy balance for non-isothermal
        # Simplified: d(n*Cp*T)/dt = F_in*Cp*(T_in - T) + Q_rxn + Q_ext
        # Assume constant Cp for simplicity
        Cp = 75.0  # J/mol/K (typical liquid)

        # Heat of reaction
        if p["dH_rxn"] is not None:
            Q_rxn = -V * jnp.sum(r * p["dH_rxn"])  # Positive for exothermic
        else:
            Q_rxn = 0.0

        # Heat from flow
        T_in = inlet["T"]
        Q_flow = F_in_total * Cp * (T_in - T)

        # External heat (for adiabatic: Q_ext = 0)
        Q_ext = 0.0 if p["mode"] == "adiabatic" else p.get("Q_spec", 0.0)

        # dT/dt = (Q_flow + Q_rxn + Q_ext) / (n_total * Cp)
        dT_dt = (Q_flow + Q_rxn + Q_ext) / (n_total * Cp + 1e-10)

        return jnp.concatenate([dn_dt, jnp.array([dT_dt])])

    def _outputs(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outlet stream from reactor state."""
        p = self.params
        species = p["species_order"]

        # Get moles and compute outlet composition
        n = jnp.array([state[f"n_{s}"] for s in species])
        n_total = jnp.sum(n) + 1e-10
        x_out = n / n_total

        # Get inlet for flow rate
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in_total = sum(inlet_flows.values())

        # Outlet flows (assume F_out = F_in for constant volume)
        outlet_flows = {s: F_in_total * x_out[i] for i, s in enumerate(species)}

        # Temperature
        if p["mode"] == "isothermal":
            T_out = inlet["T"]
        else:
            T_out = state["T"]

        # Pressure (assume constant)
        P = inlet["P"]

        return {"outlet": make_stream(outlet_flows, T_out, P)}

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Initialize from inlet stream composition."""
        p = self.params
        species = p["species_order"]
        V = p["V"]

        # Get inlet
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)

        # Initial moles = inlet composition * volume * concentration factor
        F_total = sum(inlet_flows.values())
        x_in = {s: inlet_flows.get(s, 0.0) / F_total for s in species}

        # Assume residence time of 1 minute for initial holdup
        tau = 60.0  # seconds
        n0 = jnp.array([F_total * x_in[s] * tau for s in species])

        if p["mode"] != "isothermal":
            T0 = inlet["T"]
            return jnp.concatenate([n0, jnp.array([T0])])

        return n0


class DynamicTank(DynamicUnitBase):
    """Dynamic storage tank with level/volume tracking.

    State variables:
    - V: Liquid volume (m³)
    - n_i: Moles of each species (mol)
    - T: Temperature (K) [if non-isothermal]

    Material balance:
        dV/dt = F_in - F_out  (volumetric)
        dn_i/dt = F_in * C_in_i - F_out * C_out_i

    Assumes perfectly mixed (C_out = n / V).
    """

    def __init__(
        self,
        max_volume: float,
        species_order: list[str],
        outlet_flow_fn: Callable[[Array, Array], Array] | None = None,
        isothermal: bool = True,
        name: str | None = None,
    ):
        """Initialize dynamic tank.

        Args:
            max_volume: Maximum tank volume (m³)
            species_order: List of species names
            outlet_flow_fn: Function(V, t) -> F_out, defaults to constant
            isothermal: Whether to track temperature
            name: Unit name
        """
        params = {
            "V_max": jnp.asarray(max_volume),
            "species_order": species_order,
            "outlet_flow_fn": outlet_flow_fn,
            "isothermal": isothermal,
        }
        super().__init__(params, name)

    def _build_state_spec(self) -> StateSpec:
        """Build state spec for volume, moles, and optionally temperature."""
        from difflow.dynamic.state import (
            volume_state, molar_states, thermal_state
        )

        spec = volume_state()
        spec = spec + molar_states(self.params["species_order"])

        if not self.params["isothermal"]:
            spec = spec + thermal_state()

        return spec

    def _derivatives(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> Array:
        """Compute dV/dt and dn/dt."""
        p = self.params
        species = p["species_order"]

        V = state["V"]
        n = jnp.array([state[f"n_{s}"] for s in species])

        # Concentrations
        C_out = n / jnp.maximum(V, 1e-10)

        # Inlet
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in_mol = jnp.array([inlet_flows.get(s, 0.0) for s in species])

        # Assume molar density for volumetric conversion
        rho_mol = 50.0  # mol/m³ (rough liquid estimate)
        Q_in = jnp.sum(F_in_mol) / rho_mol

        # Outlet flow
        if p["outlet_flow_fn"] is not None:
            Q_out = p["outlet_flow_fn"](V, t)
        else:
            Q_out = Q_in  # Default: maintain level

        F_out_mol = Q_out * C_out

        # Derivatives
        dV_dt = Q_in - Q_out
        dn_dt = F_in_mol - F_out_mol

        derivs = jnp.concatenate([jnp.array([dV_dt]), dn_dt])

        if not p["isothermal"]:
            # Simplified energy balance
            T = state["T"]
            T_in = inlet["T"]
            Cp = 75.0
            n_total = jnp.sum(n) + 1e-10
            dT_dt = jnp.sum(F_in_mol) * Cp * (T_in - T) / (n_total * Cp + 1e-10)
            derivs = jnp.concatenate([derivs, jnp.array([dT_dt])])

        return derivs

    def _outputs(
        self,
        t: Array,
        state: StateVector,
        inputs: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Compute outlet stream."""
        p = self.params
        species = p["species_order"]

        V = state["V"]
        n = jnp.array([state[f"n_{s}"] for s in species])
        C_out = n / jnp.maximum(V, 1e-10)

        # Get outlet flow rate
        if p["outlet_flow_fn"] is not None:
            Q_out = p["outlet_flow_fn"](V, t)
        else:
            inlet = inputs.get("inlet") or list(inputs.values())[0]
            inlet_flows = get_flows(inlet)
            Q_out = sum(inlet_flows.values()) / 50.0  # Assume same as inlet

        F_out = {s: Q_out * C_out[i] for i, s in enumerate(species)}

        # Temperature
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        if p["isothermal"]:
            T = inlet["T"]
        else:
            T = state["T"]

        return {"outlet": make_stream(F_out, T, inlet["P"])}

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Initialize at half capacity with inlet composition."""
        p = self.params
        species = p["species_order"]

        V0 = p["V_max"] / 2.0

        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_total = sum(inlet_flows.values())
        x_in = {s: inlet_flows.get(s, 0.0) / F_total for s in species}

        # Initial moles based on inlet composition
        rho_mol = 50.0
        n0 = jnp.array([V0 * rho_mol * x_in[s] for s in species])

        state0 = jnp.concatenate([jnp.array([V0]), n0])

        if not p["isothermal"]:
            T0 = inlet["T"]
            state0 = jnp.concatenate([state0, jnp.array([T0])])

        return state0
