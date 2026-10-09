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


def _takes_args(fn) -> bool:
    """True for a feed written ``(t, args)``: a second positional parameter
    that is required or is named ``args`` (or a ``*args``)."""
    import inspect
    try:
        prms = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = []
    for prm in prms:
        if prm.kind is prm.VAR_POSITIONAL:
            return len(positional) <= 1
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
            positional.append(prm)
    if len(positional) < 2:
        return False
    second = positional[1]
    return second.default is second.empty or second.name == "args"


@dataclass
class DynamicUnitEntry:
    """Entry for a unit in the dynamic flowsheet.

    Attributes:
        unit: The DynamicUnit instance
        name: Unique name for this unit
        inlet_names: Names of the flowsheet streams feeding the unit
        outlet_names: Names of the flowsheet streams the unit produces
        inlet_ports: ``{unit port: flowsheet stream}`` for the inlets. The
            unit sees its inputs under the port names. When the unit was
            added with a plain ``inlet_names`` list each port is the stream's
            own name (the original behaviour).
        outlet_ports: ``{unit port: flowsheet stream}`` for the outlets, the
            port being a key of the dict ``outputs()`` returns; None when the
            unit was added with a plain ``outlet_names`` list and does not
            declare ``output_ports``, in which case the outputs are matched to
            ``outlet_names`` by position.
        state_slice: Slice into combined state vector for this unit's states
    """
    unit: DynamicUnit
    name: str
    inlet_names: list[str]
    outlet_names: list[str]
    inlet_ports: dict[str, str] = field(default_factory=dict)
    outlet_ports: dict[str, str] | None = None
    state_slice: slice = field(default_factory=lambda: slice(0, 0))

    @property
    def n_states(self) -> int:
        """Number of state variables for this unit."""
        return self.unit.state_spec().n_states

    def output_dependencies(self) -> dict[str, set[str]] | None:
        """``{outlet port: inlet ports it reads}``, or None if unknown.

        Read from the unit's ``output_dependencies`` (port names). Only used
        when the outlet ports are known and the unit can evaluate a subset of
        its outputs (``partial_outputs``); otherwise every output is taken to
        depend on every input.
        """
        deps = getattr(self.unit, "output_dependencies", None)
        if (deps is None or self.outlet_ports is None
                or not hasattr(self.unit, "partial_outputs")):
            return None
        return {port: set(deps.get(port, self.inlet_ports))
                for port in self.outlet_ports}


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
        self._feed_takes_args: dict[str, bool] = {}
        self._state_indices_valid = False
        self._n_total_states = 0
        self._plan: list[tuple[DynamicUnitEntry, frozenset[str] | None]] | None = None

    def add_feed(
        self,
        name: str,
        stream: Stream | Callable[..., Stream],
    ) -> None:
        """Add a feed stream to the flowsheet.

        Args:
            name: Name of the feed stream
            stream: A constant Stream, a function ``t -> Stream``, or a
                function ``(t, args) -> Stream`` that reads the integrator
                ``args`` passed to :meth:`simulate`. The last form is how a
                disturbance enters without a new closure, so every scenario
                runs on one compiled right-hand side.
        """
        self._feeds[name] = stream
        self._feed_takes_args[name] = callable(stream) and _takes_args(stream)
        self._plan = None

    def add_unit(
        self,
        unit: DynamicUnit,
        inlet_names: list[str] | dict[str, str] | None = None,
        outlet_names: list[str] | dict[str, str] | None = None,
        name: str | None = None,
        *,
        inlets: dict[str, str] | None = None,
        outlets: dict[str, str] | None = None,
    ) -> None:
        """Add a dynamic unit to the flowsheet.

        Streams are named freely and wired to the unit's fixed port names
        with ``inlets``/``outlets``::

            fs.add_unit(fehe, inlets={"hot": "effluent", "cold": "feed"},
                        outlets={"hot_out": "cooled", "cold_out": "preheated"})

        Args:
            unit: DynamicUnit instance, or a stateless
                :class:`~difflow.dynamic.instantaneous.InstantaneousUnit`.
            inlet_names: Inlet stream names. A list keys each input by its
                stream name (the unit then has to read that name); a dict is
                the same as ``inlets``.
            outlet_names: Outlet stream names. A list is matched to the
                unit's outputs by position; a dict is the same as ``outlets``.
            name: Optional name override (uses unit.name if available)
            inlets: ``{unit port: stream name}`` for the inlets.
            outlets: ``{unit port: stream name}`` for the outlets, the port
                being a key of the dict the unit's ``outputs()`` returns.
        """
        # Get unit name
        if name is None:
            name = getattr(unit, "name", f"unit_{len(self._units)}")

        if name in self._units_by_name:
            raise ValueError(f"Unit with name '{name}' already exists")

        if isinstance(inlet_names, dict):
            inlets, inlet_names = inlet_names, None
        if isinstance(outlet_names, dict):
            outlets, outlet_names = outlet_names, None
        if inlets is not None and inlet_names is not None:
            raise ValueError("pass inlet_names or inlets, not both")
        if outlets is not None and outlet_names is not None:
            raise ValueError("pass outlet_names or outlets, not both")

        if inlets is None:
            inlets = {s: s for s in (inlet_names or [])}
        if outlets is None:
            outlet_names = list(outlet_names or [])
            ports = getattr(unit, "output_ports", None)
            if ports is not None and len(ports) >= len(outlet_names):
                outlets = dict(zip(ports, outlet_names))

        entry = DynamicUnitEntry(
            unit=unit,
            name=name,
            inlet_names=list(inlets.values()),
            outlet_names=list(outlets.values()) if outlets is not None else outlet_names,
            inlet_ports=dict(inlets),
            outlet_ports=dict(outlets) if outlets is not None else None,
        )
        self._units.append(entry)
        self._units_by_name[name] = entry
        self._state_indices_valid = False
        self._plan = None

    def connect(self, source: str, dest: str) -> None:
        """Explicitly connect a source stream to a destination.

        This is optional - by default, streams are matched by name.
        Use connect() when source and destination have different names.

        Args:
            source: Name of source stream (unit outlet or feed)
            dest: Name of destination stream (unit inlet)
        """
        self._connections.append(Connection(source, dest))
        self._plan = None

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

    def _get_feed(self, name: str, t: Array, args: Any = None) -> Stream:
        """Get feed stream at time t.

        Args:
            name: Feed name
            t: Current time
            args: Integrator args, handed to a feed declared ``(t, args)``

        Returns:
            Stream at time t
        """
        feed = self._feeds.get(name)
        if feed is None:
            raise ValueError(f"Feed '{name}' not found")

        if callable(feed):
            if self._feed_takes_args.get(name, False):
                return feed(t, args)
            return feed(t)
        return feed

    # -- stream routing ---------------------------------------------------

    def _source_of(self, stream: str) -> str:
        """Follow explicit connections back to the stream that carries data."""
        aliases = {c.dest: c.source for c in self._connections if c.source != c.dest}
        seen = set()
        while stream in aliases and stream not in seen:
            seen.add(stream)
            stream = aliases[stream]
        return stream

    def _evaluation_plan(self) -> list[tuple[DynamicUnitEntry, frozenset[str] | None]]:
        """Order in which unit outputs can be evaluated within one RHS call.

        Each step is ``(entry, ports)``: evaluate ``entry`` for the outlet
        ``ports`` (None: all of them, from all inputs). The plan is built
        from stream availability, not insertion order, and is static -- it
        depends only on the wiring -- so it is computed once.

        A unit that declares ``output_dependencies`` and ``partial_outputs``
        can be evaluated for some outputs before all its inputs exist. That
        is what makes a loop such as FEHE cold side -> heater -> reactor ->
        FEHE hot side evaluable: the exchanger's cold outlet depends on its
        duty state and the cold inlet alone.

        Raises:
            ValueError: An algebraic loop no declared dependency breaks, or
                an inlet nothing produces.
        """
        if self._plan is not None:
            return self._plan

        known = set(self._feeds)
        pending: dict[str, set[str] | None] = {}
        for e in self._units:
            pending[e.name] = set(e.outlet_ports) if e.outlet_ports is not None else None

        def is_known(stream):
            return self._source_of(stream) in known

        def produce(e, ports):
            if ports is None or e.outlet_ports is None:
                known.update(e.outlet_names)
                pending[e.name] = set()
            else:
                known.update(e.outlet_ports[p] for p in ports)
                pending[e.name] -= set(ports)

        plan = []
        done = lambda e: pending[e.name] is not None and not pending[e.name]
        while not all(done(e) for e in self._units):
            progress = False
            for e in self._units:
                if done(e):
                    continue
                avail = {port for port, st in e.inlet_ports.items() if is_known(st)}
                if len(avail) == len(e.inlet_ports):
                    ports = None if e.outlet_ports is None else frozenset(pending[e.name])
                    plan.append((e, ports))
                    produce(e, ports)
                    progress = True
                    continue
                deps = e.output_dependencies()
                if deps is None:
                    continue
                ready = frozenset(p for p in pending[e.name] if deps[p] <= avail)
                if ready:
                    plan.append((e, ready))
                    produce(e, ready)
                    progress = True
            if not progress:
                stuck = [e for e in self._units if not done(e)]
                missing = sorted({st for e in stuck for st in e.inlet_ports.values()
                                  if not is_known(st)})
                produced = {st for e in self._units for st in e.outlet_names} | set(self._feeds)
                orphans = [st for st in missing if self._source_of(st) not in produced]
                if orphans:
                    raise ValueError(
                        f"Inlet stream(s) {orphans} of unit(s) "
                        f"{[e.name for e in stuck]} not found: they are not a "
                        f"feed and no unit produces them. Available: "
                        f"{sorted(produced)}"
                    )
                raise ValueError(
                    f"Algebraic loop: units {[e.name for e in stuck]} wait on "
                    f"streams {missing} that only they produce. Break it with a "
                    "unit whose output reads a subset of its inputs (declare "
                    "output_dependencies and partial_outputs, as "
                    "DynamicCounterCurrentHX does), or give a unit in the loop a "
                    "state."
                )
        self._plan = plan
        return plan

    def _unit_inputs(
        self,
        entry: DynamicUnitEntry,
        streams: dict[str, Stream],
        ports: Any = None,
    ) -> dict[str, Stream]:
        """Inputs keyed by the unit's port names; ``ports`` limits them."""
        inputs = {}
        for port, stream in entry.inlet_ports.items():
            src = self._source_of(stream)
            if src in streams:
                inputs[port] = streams[src]
            elif ports is None:
                raise ValueError(
                    f"Inlet '{stream}' for unit '{entry.name}' not found. "
                    f"Available: {list(streams.keys())}"
                )
        return inputs

    def _store_outputs(self, entry, outputs, ports, streams) -> None:
        """Name a unit's outputs as flowsheet streams."""
        if entry.outlet_ports is not None:
            for port in (ports if ports is not None else entry.outlet_ports):
                if port not in outputs:
                    raise ValueError(
                        f"Unit '{entry.name}' returned no output '{port}' "
                        f"(it returned {list(outputs)})"
                    )
                streams[entry.outlet_ports[port]] = outputs[port]
            return
        # Positional matching for a unit added with a plain outlet list.
        out_keys = list(outputs.keys())
        for i, outlet_name in enumerate(entry.outlet_names):
            if i < len(out_keys):
                streams[outlet_name] = outputs[out_keys[i]]
            elif len(out_keys) == 1:
                streams[outlet_name] = outputs[out_keys[0]]

    def _evaluate(
        self,
        t: Array,
        state: Array | None,
        params: Params | None,
        args: Any,
        init: bool = False,
    ) -> tuple[dict[str, Stream], dict[str, Array]]:
        """Run the plan: every stream, and (with ``init``) the initial states.

        Returns:
            (streams, unit_states)
        """
        self._update_state_indices()
        plan = self._evaluation_plan()
        streams = {name: self._get_feed(name, t, args) for name in self._feeds}
        unit_states: dict[str, Array] = {}
        if not init:
            for e in self._units:
                unit_states[e.name] = state[e.state_slice]

        for entry, ports in plan:
            inputs = self._unit_inputs(entry, streams, ports)
            if entry.name not in unit_states:
                # init: a unit starts from the inputs it can see when it is
                # first evaluated (a partially-evaluated unit sees a subset).
                unit_states[entry.name] = jnp.atleast_1d(
                    entry.unit.initial_state(inputs, params)
                ) if entry.n_states else jnp.zeros(0)
            y = unit_states[entry.name]
            if ports is None or len(inputs) == len(entry.inlet_ports):
                outputs = entry.unit.outputs(t, y, inputs, params)
            else:
                outputs = entry.unit.partial_outputs(t, y, inputs, params)
            self._store_outputs(entry, outputs, ports, streams)

        for c in self._connections:
            if c.source in streams and c.source != c.dest:
                streams[c.dest] = streams[c.source]
        return streams, unit_states

    def initial_state(self, params: Params | None = None, args: Any = None) -> Array:
        """Compute initial state for all units.

        Uses each unit's initial_state method with the streams available
        when the unit is first evaluated (feeds at t = 0 and upstream
        outputs at their own initial states).

        Args:
            params: Optional parameters to pass to units
            args: Optional integrator args (read by ``(t, args)`` feeds)

        Returns:
            Combined initial state array
        """
        _, unit_states = self._evaluate(jnp.array(0.0), None, params, args, init=True)
        return jnp.concatenate(
            [jnp.zeros(0)] + [unit_states[e.name] for e in self._units]
        )

    def derivatives(
        self,
        t: Array,
        state: Array,
        params: Params | None = None,
        args: Any = None,
    ) -> Array:
        """Compute combined derivatives for all units.

        This is the main ODE function for the flowsheet system:
        dy/dt = f(t, y)

        At each evaluation:
        1. Split combined state into per-unit states
        2. Evaluate every unit's outputs in the order of
           :meth:`_evaluation_plan` (which resolves loops broken by a
           unit's declared output dependencies)
        3. Compute derivatives for each stateful unit from its full inputs
        4. Concatenate into combined derivative array

        Args:
            t: Current time
            state: Combined state array
            params: Optional parameters
            args: Optional integrator args (read by ``(t, args)`` feeds)

        Returns:
            Combined derivatives array
        """
        streams, unit_states = self._evaluate(t, state, params, args)
        derivs = [jnp.zeros(0)]
        for entry in self._units:
            if entry.n_states == 0:
                continue
            inputs = self._unit_inputs(entry, streams)
            derivs.append(jnp.atleast_1d(
                entry.unit.derivatives(t, unit_states[entry.name], inputs, params)
            ))
        return jnp.concatenate(derivs)

    def outputs(
        self,
        t: Array,
        state: Array,
        params: Params | None = None,
        args: Any = None,
    ) -> dict[str, Stream]:
        """Compute all outlet streams at current state.

        Args:
            t: Current time
            state: Combined state array
            params: Optional parameters
            args: Optional integrator args (read by ``(t, args)`` feeds)

        Returns:
            Dictionary of all streams (feeds + unit outputs)
        """
        streams, _ = self._evaluate(t, state, params, args)
        return streams

    def vector_field(self, t: Array, y: Array, args: Any) -> Array:
        """``dy/dt`` in diffrax's ``f(t, y, args)`` form.

        ``args`` is ``(params, user_args)``. This is a method, not a closure
        made per call, so it is the same object every time: a diffrax solve
        built on it compiles once and is reused for every ``params`` and
        ``args`` with the same structure (#390).
        """
        params, user_args = args
        return self.derivatives(t, y, params, user_args)

    def simulate(
        self,
        t_span: tuple[float, float],
        y0: Array | None = None,
        method: Method = "RK4",
        params: Params | None = None,
        events: list[EventSpec] | None = None,
        args: Any = None,
        **kwargs,
    ) -> DynamicFlowsheetResult:
        """Simulate the flowsheet over time.

        Integrates the combined system of ODEs for all units.

        Args:
            t_span: (t_start, t_end) time interval
            y0: Initial state (uses automatic initialization if None)
            method: Integration method ("RK4", "RK45", "Euler", "diffrax:...")
            params: Optional parameters to pass to units
            events: Optional list of :class:`EventSpec` describing state
                events to detect (e.g. tank overflow, phase change, a
                threshold crossing). After integration the trajectory is
                scanned for zero crossings of each event's condition and the
                detected crossings are returned on
                ``DynamicFlowsheetResult.events`` (#130).
            args: Optional pytree handed to every ``(t, args)`` feed. It and
                ``params`` are passed to the integrator as its ``args``
                rather than closed over, so with a diffrax method one compiled
                right-hand side serves every scenario (a new disturbance is a
                new value, not a new function), and ``jax.grad`` with respect
                to a disturbance or a parameter does not retrace. Values meant
                to vary should be JAX arrays: a Python float is static.
            **kwargs: Additional arguments for the integrator

        Returns:
            DynamicFlowsheetResult with trajectories and state access
        """
        self._update_state_indices()

        if y0 is None:
            y0 = self.initial_state(params, args)

        result = integrate(self.vector_field, y0, t_span, method,
                           args=(params, args), **kwargs)

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
        args: Any = None,
    ) -> Array:
        """Find steady-state by integrating until derivatives are small.

        Simple approach: integrate for a long time and check convergence.
        For faster convergence, consider Newton methods on residual.

        Args:
            y0: Initial guess (uses initial_state if None)
            params: Optional parameters
            tol: Tolerance for derivatives norm
            max_iter: Maximum integration steps
            args: Optional integrator args (read by ``(t, args)`` feeds)

        Returns:
            Steady-state state array
        """
        if y0 is None:
            y0 = self.initial_state(params, args)

        def f(t, y):
            return self.derivatives(t, y, params, args)

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
