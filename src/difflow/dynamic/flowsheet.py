"""Dynamic flowsheet solver for connected dynamic units.

This module provides the DynamicFlowsheet class that enables simulation of
interconnected dynamic units over time. It handles:

- Connecting multiple DynamicUnit instances via streams
- Time-varying feed streams
- Combined state vector management
- Integrated system of ODEs

The DynamicFlowsheet is the Phase 3 component of the unified dynamic
modeling framework, building on the DynamicUnit protocol (Phase 1 & 2).

Example Usage
-------------

>>> from difflow.dynamic import DynamicFlowsheet, DynamicCSTR, DynamicTank
>>> from difflow.streams import make_stream
>>> import jax.numpy as jnp
>>>
>>> # Create units
>>> def rate_fn(C, T, params):
...     k = params["k"]
...     return jnp.array([k * C["A"]])
>>>
>>> cstr = DynamicCSTR(
...     volume=1.0,
...     rate_fn=rate_fn,
...     stoich=jnp.array([[-1], [1]]),
...     species_order=["A", "B"],
...     rate_params={"k": 0.1},
...     name="reactor",
... )
>>>
>>> tank = DynamicTank(
...     max_volume=10.0,
...     species_order=["A", "B"],
...     name="storage",
... )
>>>
>>> # Build dynamic flowsheet
>>> fs = DynamicFlowsheet(species_order=["A", "B"])
>>>
>>> # Add feed
>>> feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
>>> fs.add_feed("feed", feed)
>>>
>>> # Add units
>>> fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
>>> fs.add_unit(tank, inlet_names=["reactor_out"], outlet_names=["product"])
>>>
>>> # Connect units (explicit - though inferred from outlet->inlet name match)
>>> fs.connect("reactor_out", "reactor_out")  # Same name = direct connection
>>>
>>> # Simulate
>>> result = fs.simulate(t_span=(0.0, 1000.0), method="RK4", n_steps=500)
>>>
>>> # Access results
>>> print(result.y_final)  # Final combined state
>>> print(result.unit_states("reactor"))  # Reactor states over time
>>> print(result.stream_history("reactor_out"))  # Stream history
"""

from typing import Callable, Any, NamedTuple
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, make_stream
from difflow.dynamic.state import StateSpec, StateVar
from difflow.dynamic.base import DynamicUnit, Params
from difflow.dynamic.integrators import (
    integrate,
    IntegrationResult,
    Trajectory,
    IntegrationInfo,
    Method,
    EventSpec,
    EventResult,
    detect_events,
)


@dataclass
class DynamicUnitEntry:
    """Entry for a unit in the dynamic flowsheet.

    Attributes:
        unit: The DynamicUnit instance
        name: Unique name for this unit
        inlet_names: Names of inlet streams (keys in inputs dict)
        outlet_names: Names of outlet streams (keys returned by outputs())
        state_slice: Slice into combined state vector for this unit's states
    """
    unit: DynamicUnit
    name: str
    inlet_names: list[str]
    outlet_names: list[str]
    state_slice: slice = field(default_factory=lambda: slice(0, 0))

    @property
    def n_states(self) -> int:
        """Number of state variables for this unit."""
        return self.unit.state_spec().n_states


@dataclass
class Connection:
    """Connection between streams in the flowsheet.

    Attributes:
        source: Name of source stream (output of a unit or feed)
        dest: Name of destination stream (input to a unit)
    """
    source: str
    dest: str


class FlowsheetState(NamedTuple):
    """Container for flowsheet state with named unit access.

    Attributes:
        combined: Full combined state array
        unit_states: Dictionary mapping unit names to their state arrays
        t: Current time
    """
    combined: Array
    unit_states: dict[str, Array]
    t: Array


@dataclass
class DynamicFlowsheetResult:
    """Result from dynamic flowsheet simulation.

    Extends IntegrationResult with flowsheet-specific access methods.

    Attributes:
        y_final: Final combined state array
        trajectory: Time and combined state history
        info: Integration statistics
        flowsheet: Reference to the flowsheet for state parsing
    """
    y_final: Array
    trajectory: Trajectory
    info: IntegrationInfo
    flowsheet: "DynamicFlowsheet"
    events: list = field(default_factory=list)

    def unit_trajectory(self, unit_name: str) -> Trajectory:
        """Get trajectory for a specific unit.

        Args:
            unit_name: Name of the unit

        Returns:
            Trajectory with time and unit-specific states
        """
        entry = self.flowsheet._units_by_name[unit_name]
        unit_states = self.trajectory.y[:, entry.state_slice]
        return Trajectory(self.trajectory.t, unit_states)

    def unit_state_at(self, unit_name: str, idx: int = -1) -> Array:
        """Get unit state at a specific trajectory index.

        Args:
            unit_name: Name of the unit
            idx: Index into trajectory (-1 for final)

        Returns:
            State array for the unit
        """
        entry = self.flowsheet._units_by_name[unit_name]
        return self.trajectory.y[idx, entry.state_slice]

    def get_state_dict(self, unit_name: str, idx: int = -1) -> dict[str, Array]:
        """Get unit state as a named dictionary.

        Args:
            unit_name: Name of the unit
            idx: Index into trajectory

        Returns:
            Dictionary mapping state variable names to values
        """
        entry = self.flowsheet._units_by_name[unit_name]
        state = self.trajectory.y[idx, entry.state_slice]
        spec = entry.unit.state_spec()
        return {name: state[i] for i, name in enumerate(spec.names)}


class DynamicFlowsheet:
    """Dynamic flowsheet connecting multiple DynamicUnit instances.

    The DynamicFlowsheet manages:
    - Multiple dynamic units with their interconnections
    - Feed streams (constant or time-varying)
    - Combined state vector for system integration
    - Stream routing at each time step

    Units are connected via named streams. The outlet streams from upstream
    units become the inlet streams for downstream units based on connections.

    Attributes:
        species_order: List of species names for stream arrays
        units: List of DynamicUnitEntry objects
        feeds: Dictionary of feed streams (constant or time-varying)
        connections: List of stream connections
    """

    def __init__(self, species_order: list[str]):
        """Initialize empty dynamic flowsheet.

        Args:
            species_order: List of species names for stream arrays
        """
        self.species_order = species_order
        self._units: list[DynamicUnitEntry] = []
        self._units_by_name: dict[str, DynamicUnitEntry] = {}
        self._feeds: dict[str, Stream | Callable[[Array], Stream]] = {}
        self._connections: list[Connection] = []
        self._state_indices_valid = False
        self._n_total_states = 0

    def add_feed(
        self,
        name: str,
        stream: Stream | Callable[[Array], Stream],
    ) -> None:
        """Add a feed stream to the flowsheet.

        Args:
            name: Name of the feed stream
            stream: Either a constant Stream or a function t -> Stream
                   for time-varying feeds
        """
        self._feeds[name] = stream

    def add_unit(
        self,
        unit: DynamicUnit,
        inlet_names: list[str],
        outlet_names: list[str],
        name: str | None = None,
    ) -> None:
        """Add a dynamic unit to the flowsheet.

        Args:
            unit: DynamicUnit instance
            inlet_names: Names of inlet streams (will be looked up from
                        feeds or other unit outputs)
            outlet_names: Names of outlet streams produced by this unit
            name: Optional name override (uses unit.name if available)
        """
        # Get unit name
        if name is None:
            name = getattr(unit, "name", f"unit_{len(self._units)}")

        if name in self._units_by_name:
            raise ValueError(f"Unit with name '{name}' already exists")

        entry = DynamicUnitEntry(
            unit=unit,
            name=name,
            inlet_names=inlet_names,
            outlet_names=outlet_names,
        )
        self._units.append(entry)
        self._units_by_name[name] = entry
        self._state_indices_valid = False

    def connect(self, source: str, dest: str) -> None:
        """Explicitly connect a source stream to a destination.

        This is optional - by default, streams are matched by name.
        Use connect() when source and destination have different names.

        Args:
            source: Name of source stream (unit outlet or feed)
            dest: Name of destination stream (unit inlet)
        """
        self._connections.append(Connection(source, dest))

    def _update_state_indices(self) -> None:
        """Update state slice indices for all units."""
        if self._state_indices_valid:
            return

        idx = 0
        for entry in self._units:
            n = entry.n_states
            entry.state_slice = slice(idx, idx + n)
            idx += n

        self._n_total_states = idx
        self._state_indices_valid = True

    @property
    def n_states(self) -> int:
        """Total number of state variables in the flowsheet."""
        self._update_state_indices()
        return self._n_total_states

    @property
    def units(self) -> list[DynamicUnitEntry]:
        """List of unit entries."""
        return self._units

    @property
    def unit_names(self) -> list[str]:
        """List of unit names."""
        return [e.name for e in self._units]

    def combined_state_spec(self) -> StateSpec:
        """Get combined state specification for all units.

        Returns:
            StateSpec with all unit state variables, prefixed by unit name
        """
        self._update_state_indices()

        all_vars = []
        for entry in self._units:
            spec = entry.unit.state_spec()
            for var in spec.variables:
                # Prefix variable name with unit name
                new_var = StateVar(
                    name=f"{entry.name}.{var.name}",
                    category=var.category,
                    units=var.units,
                    description=f"[{entry.name}] {var.description}",
                    bounds=var.bounds,
                    scale=var.scale,
                    initial_value=var.initial_value,
                )
                all_vars.append(new_var)

        return StateSpec(all_vars)

    def _get_feed(self, name: str, t: Array) -> Stream:
        """Get feed stream at time t.

        Args:
            name: Feed name
            t: Current time

        Returns:
            Stream at time t
        """
        feed = self._feeds.get(name)
        if feed is None:
            raise ValueError(f"Feed '{name}' not found")

        if callable(feed):
            return feed(t)
        return feed

    def _resolve_streams(
        self,
        t: Array,
        unit_outputs: dict[str, dict[str, Stream]],
    ) -> dict[str, Stream]:
        """Resolve all available streams at current time.

        Combines feed streams and unit output streams, applying
        any explicit connections. Also renames unit outputs according
        to the outlet_names specified when adding the unit.

        Args:
            t: Current time
            unit_outputs: Dict[unit_name, Dict[outlet_name, Stream]]

        Returns:
            Dictionary of all available streams by name
        """
        streams = {}

        # Add feeds
        for name in self._feeds:
            streams[name] = self._get_feed(name, t)

        # Add unit outputs, renaming according to outlet_names
        for entry in self._units:
            if entry.name in unit_outputs:
                unit_out = unit_outputs[entry.name]
                # Map unit's output names to flowsheet's outlet_names
                # e.g., unit returns {"outlet": stream}, but flowsheet
                # expects "reactor_out" based on outlet_names=["reactor_out"]
                out_keys = list(unit_out.keys())
                for i, outlet_name in enumerate(entry.outlet_names):
                    if i < len(out_keys):
                        streams[outlet_name] = unit_out[out_keys[i]]
                    elif len(out_keys) == 1:
                        # Single output, use it for all outlet names
                        streams[outlet_name] = unit_out[out_keys[0]]

        # Apply explicit connections (rename source to dest)
        for conn in self._connections:
            if conn.source in streams and conn.source != conn.dest:
                streams[conn.dest] = streams[conn.source]

        return streams

    def _get_unit_inputs(
        self,
        entry: DynamicUnitEntry,
        available_streams: dict[str, Stream],
    ) -> dict[str, Stream]:
        """Get input streams for a unit.

        Args:
            entry: Unit entry
            available_streams: All available streams

        Returns:
            Dictionary of inlet streams for the unit
        """
        inputs = {}
        for inlet_name in entry.inlet_names:
            if inlet_name in available_streams:
                inputs[inlet_name] = available_streams[inlet_name]
            else:
                raise ValueError(
                    f"Inlet '{inlet_name}' for unit '{entry.name}' not found. "
                    f"Available: {list(available_streams.keys())}"
                )
        return inputs

    def initial_state(self, params: Params | None = None) -> Array:
        """Compute initial state for all units.

        Uses each unit's initial_state method with appropriate feeds.

        Args:
            params: Optional parameters to pass to units

        Returns:
            Combined initial state array
        """
        self._update_state_indices()

        # Get initial streams (feeds at t=0)
        t0 = jnp.array(0.0)

        # Initialize units in order, using outputs from previous units
        states = []
        unit_outputs = {}

        for entry in self._units:
            # Resolve all available streams (feeds + previous outputs with renaming)
            all_streams = self._resolve_streams(t0, unit_outputs)

            inputs = self._get_unit_inputs(entry, all_streams)

            # Get initial state
            y0 = entry.unit.initial_state(inputs, params)
            states.append(y0)

            # Get initial outputs for downstream units
            outputs = entry.unit.outputs(t0, y0, inputs, params)
            unit_outputs[entry.name] = outputs

        return jnp.concatenate(states)

    def derivatives(
        self,
        t: Array,
        state: Array,
        params: Params | None = None,
    ) -> Array:
        """Compute combined derivatives for all units.

        This is the main ODE function for the flowsheet system:
        dy/dt = f(t, y)

        At each evaluation:
        1. Split combined state into per-unit states
        2. Compute outputs (outlet streams) from each unit
        3. Resolve stream routing
        4. Compute derivatives for each unit
        5. Concatenate into combined derivative array

        Args:
            t: Current time
            state: Combined state array
            params: Optional parameters

        Returns:
            Combined derivatives array
        """
        self._update_state_indices()

        # Split state into per-unit states
        unit_states = {}
        for entry in self._units:
            unit_states[entry.name] = state[entry.state_slice]

        # First pass: compute outputs from all units in order
        # Each unit's outputs become available for downstream units
        unit_outputs = {}

        for entry in self._units:
            y_unit = unit_states[entry.name]
            # Get resolved streams (feeds + previous units' outputs with renaming)
            available_streams = self._resolve_streams(t, unit_outputs)
            inputs = self._get_unit_inputs(entry, available_streams)
            outputs = entry.unit.outputs(t, y_unit, inputs, params)
            unit_outputs[entry.name] = outputs

        # Get final stream resolution for derivatives
        all_streams = self._resolve_streams(t, unit_outputs)

        # Second pass: compute derivatives for all units
        derivs = []
        for entry in self._units:
            y_unit = unit_states[entry.name]
            inputs = self._get_unit_inputs(entry, all_streams)
            dy_unit = entry.unit.derivatives(t, y_unit, inputs, params)
            derivs.append(dy_unit)

        return jnp.concatenate(derivs)

    def outputs(
        self,
        t: Array,
        state: Array,
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute all outlet streams at current state.

        Args:
            t: Current time
            state: Combined state array
            params: Optional parameters

        Returns:
            Dictionary of all streams (feeds + unit outputs)
        """
        self._update_state_indices()

        unit_outputs = {}

        for entry in self._units:
            y_unit = state[entry.state_slice]
            # Get resolved streams (feeds + previous units' outputs with renaming)
            available_streams = self._resolve_streams(t, unit_outputs)
            inputs = self._get_unit_inputs(entry, available_streams)
            outputs = entry.unit.outputs(t, y_unit, inputs, params)
            unit_outputs[entry.name] = outputs

        return self._resolve_streams(t, unit_outputs)

    def simulate(
        self,
        t_span: tuple[float, float],
        y0: Array | None = None,
        method: Method = "RK4",
        params: Params | None = None,
        events: list[EventSpec] | None = None,
        **kwargs,
    ) -> DynamicFlowsheetResult:
        """Simulate the flowsheet over time.

        Integrates the combined system of ODEs for all units.

        Args:
            t_span: (t_start, t_end) time interval
            y0: Initial state (uses automatic initialization if None)
            method: Integration method ("RK4", "RK45", "Euler")
            params: Optional parameters to pass to units
            events: Optional list of :class:`EventSpec` describing state
                events to detect (e.g. tank overflow, phase change, a
                threshold crossing). After integration the trajectory is
                scanned for zero crossings of each event's condition and the
                detected crossings are returned on
                ``DynamicFlowsheetResult.events`` (#130).
            **kwargs: Additional arguments for the integrator

        Returns:
            DynamicFlowsheetResult with trajectories and state access
        """
        self._update_state_indices()

        if y0 is None:
            y0 = self.initial_state(params)

        # Create derivatives function with params baked in
        def f(t, y):
            return self.derivatives(t, y, params)

        # Integrate
        result = integrate(f, y0, t_span, method, **kwargs)

        # Post-hoc event detection over the trajectory (#130)
        detected_events = detect_events(result, events) if events else []

        return DynamicFlowsheetResult(
            y_final=result.y_final,
            trajectory=result.trajectory,
            info=result.info,
            flowsheet=self,
            events=detected_events,
        )

    def steady_state(
        self,
        y0: Array | None = None,
        params: Params | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> Array:
        """Find steady-state by integrating until derivatives are small.

        Simple approach: integrate for a long time and check convergence.
        For faster convergence, consider Newton methods on residual.

        Args:
            y0: Initial guess (uses initial_state if None)
            params: Optional parameters
            tol: Tolerance for derivatives norm
            max_iter: Maximum integration steps

        Returns:
            Steady-state state array
        """
        if y0 is None:
            y0 = self.initial_state(params)

        def f(t, y):
            return self.derivatives(t, y, params)

        # Simple approach: integrate and check convergence
        t = 0.0
        y = y0
        dt = 1.0

        for _ in range(max_iter):
            dy = f(jnp.array(t), y)
            norm_dy = jnp.max(jnp.abs(dy))

            if norm_dy < tol:
                return y

            # Take a step
            from difflow.dynamic.integrators import rk4_step
            y = rk4_step(f, jnp.array(t), y, jnp.array(dt))
            t += dt

        # Return current state even if not converged
        return y

    def __repr__(self) -> str:
        self._update_state_indices()
        return (
            f"DynamicFlowsheet("
            f"n_units={len(self._units)}, "
            f"n_states={self._n_total_states}, "
            f"units={self.unit_names})"
        )
