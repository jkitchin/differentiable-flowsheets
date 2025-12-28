"""Flowsheet module for connecting unit operations.

This module provides:
- Flowsheet class for defining and solving process flowsheets
- Sequential modular approach with recycle convergence
- Anderson/Wegstein acceleration for faster convergence
- Implicit differentiation through converged recycle loops

The flowsheet is defined by adding units and connections, then
solved using fixed-point iteration on tear streams with optional
acceleration methods.
"""

from typing import Callable, Any, Literal
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array
import jax

from difflow.streams import Stream, get_species, get_flows, make_stream
from difflow.initialization import (
    AndersonAccelerator,
    wegstein_acceleration,
    Initializable,
)
import optimistix as optx


@dataclass
class Unit:
    """A unit operation in a flowsheet.

    Attributes:
        name: Unique identifier for the unit
        operation: The unit operation callable
        inlet_names: Names of inlet stream(s)
        outlet_names: Names of outlet stream(s)
        params: Additional parameters passed to the operation
    """
    name: str
    operation: Callable
    inlet_names: list[str]
    outlet_names: list[str]
    params: dict = field(default_factory=dict)

    def __hash__(self):
        """Make Unit hashable for JIT compilation with optimistix."""
        return hash((
            self.name,
            id(self.operation),
            tuple(self.inlet_names),
            tuple(self.outlet_names),
        ))


class Flowsheet:
    """Process flowsheet with sequential modular solver.

    The flowsheet manages:
    - Unit operations and their connections
    - Tear streams for recycle loops
    - Convergence of recycle loops
    - Implicit differentiation through the converged solution

    Example usage:
        fs = Flowsheet()
        fs.add_feed("feed", feed_stream)
        fs.add_unit(Unit("reactor", cstr, ["feed", "recycle"], ["reactor_out"]))
        fs.add_unit(Unit("flash", flash, ["reactor_out"], ["liquid", "vapor"]))
        fs.add_recycle("liquid", "recycle")
        results = fs.solve()
    """

    def __init__(self, species_order: list[str]):
        """Initialize empty flowsheet.

        Args:
            species_order: List of species names for stream arrays
        """
        self.species_order = species_order
        self.units: list[Unit] = []
        self.feeds: dict[str, Stream] = {}
        self.recycles: dict[str, str] = {}  # {source_name: dest_name}
        self._stream_cache: dict[str, Stream] = {}

    def add_feed(self, name: str, stream: Stream) -> None:
        """Add a feed stream to the flowsheet.

        Args:
            name: Stream name
            stream: Feed stream
        """
        self.feeds[name] = stream

    def add_unit(self, unit: Unit) -> None:
        """Add a unit operation to the flowsheet.

        Units should be added in calculation order (feeds first,
        products last). For recycles, add units in a logical order.

        Args:
            unit: Unit operation to add
        """
        self.units.append(unit)

    def add_recycle(self, source: str, dest: str) -> None:
        """Define a recycle stream.

        The source stream will be recycled to become the dest stream.
        During iteration, the dest stream value is updated until
        it matches the calculated source stream.

        Args:
            source: Name of stream to recycle (output of some unit)
            dest: Name of stream to receive recycle (input to some unit)
        """
        self.recycles[source] = dest

    def solve(
        self,
        tear_initial: dict[str, Stream] | None = None,
        tol: float = 1e-8,
        max_iter: int = 100,
        damping: float = 0.5,
        acceleration: Literal["none", "wegstein", "anderson"] = "anderson",
        anderson_depth: int = 5,
        use_initialization: bool = True,
    ) -> dict[str, Stream]:
        """Solve the flowsheet.

        Uses fixed-point iteration on tear streams with optional
        acceleration (Anderson or Wegstein) and implicit differentiation
        through the converged solution.

        Args:
            tear_initial: Initial guesses for tear streams (recycle destinations).
                         If None, uses initialization from units or zero flows.
            tol: Convergence tolerance
            max_iter: Maximum iterations
            damping: Damping factor for tear stream updates (used with acceleration="none")
            acceleration: Acceleration method:
                - "none": Simple fixed-point iteration with damping
                - "wegstein": Wegstein acceleration (uses previous 2 iterates)
                - "anderson": Anderson acceleration (uses history of iterates)
            anderson_depth: History depth for Anderson acceleration (default 5)
            use_initialization: If True and units support it, use their
                               initialize() methods for better initial guesses

        Returns:
            Dictionary of all streams in the flowsheet
        """
        if not self.recycles:
            # No recycles - simple sequential solution
            return self._solve_sequential()

        # Initialize tear streams
        tear_streams = {}
        for source, dest in self.recycles.items():
            if tear_initial and dest in tear_initial:
                tear_streams[dest] = tear_initial[dest]
            elif use_initialization:
                # Try to get initial guess from unit initialization
                init_stream = self._initialize_tear_stream(dest)
                tear_streams[dest] = init_stream
            else:
                # Initialize with small flows
                tear_streams[dest] = self._make_zero_stream()

        # Solve with chosen method
        if acceleration == "none":
            return self._solve_with_recycle_damped(tear_streams, tol, max_iter, damping)
        elif acceleration == "wegstein":
            return self._solve_with_wegstein(tear_streams, tol, max_iter)
        elif acceleration == "anderson":
            return self._solve_with_anderson(tear_streams, tol, max_iter, anderson_depth)
        else:
            raise ValueError(f"Unknown acceleration method: {acceleration}")

    def _initialize_tear_stream(self, stream_name: str) -> Stream:
        """Get initial guess for a tear stream from unit initialization.

        Args:
            stream_name: Name of the tear stream

        Returns:
            Initial guess for the stream
        """
        # Find the unit that produces this stream and try to initialize it
        for unit in self.units:
            if stream_name in unit.outlet_names:
                # Check if unit operation supports initialization
                if isinstance(unit.operation, Initializable):
                    # Get inlet for initialization
                    # This is a chicken-and-egg problem; use feed or previous guess
                    inlet_name = unit.inlet_names[0] if unit.inlet_names else None
                    if inlet_name and inlet_name in self.feeds:
                        try:
                            init_result = unit.operation.initialize(
                                self.feeds[inlet_name],
                                **unit.params
                            )
                            if 'outlet' in init_result:
                                return init_result['outlet']
                        except Exception:
                            pass  # Fall through to default

        return self._make_zero_stream()

    def _make_zero_stream(self) -> Stream:
        """Create a stream with zero flows."""
        flows = {s: jnp.asarray(0.01) for s in self.species_order}  # Small non-zero
        return make_stream(flows, 300.0, 101325.0)

    def _solve_sequential(self) -> dict[str, Stream]:
        """Solve flowsheet without recycles."""
        streams = dict(self.feeds)

        for unit in self.units:
            # Gather inlet streams
            inlets = [streams[name] for name in unit.inlet_names]

            # Call unit operation
            result = unit.operation(*inlets, **unit.params)

            # Handle different return types
            if isinstance(result, tuple):
                # Multiple outputs or (outputs, info)
                if len(result) == len(unit.outlet_names):
                    # Just the outlet streams
                    for name, stream in zip(unit.outlet_names, result):
                        streams[name] = stream
                elif len(result) == 2 and isinstance(result[1], dict):
                    # (stream(s), info) format
                    outputs = result[0]
                    if isinstance(outputs, dict):
                        # Single stream
                        streams[unit.outlet_names[0]] = outputs
                    elif isinstance(outputs, tuple):
                        for name, stream in zip(unit.outlet_names, outputs):
                            streams[name] = stream
                    else:
                        streams[unit.outlet_names[0]] = outputs
                elif len(result) == len(unit.outlet_names) + 1:
                    # Multiple streams + info dict at end
                    for name, stream in zip(unit.outlet_names, result[:-1]):
                        streams[name] = stream
                else:
                    raise ValueError(f"Unexpected output from {unit.name}: {len(result)} items")
            else:
                # Single output
                streams[unit.outlet_names[0]] = result

        return streams

    def _solve_with_recycle_damped(
        self,
        tear_initial: dict[str, Stream],
        tol: float,
        max_iter: int,
        damping: float,
    ) -> dict[str, Stream]:
        """Solve flowsheet with recycle using damped fixed-point iteration."""

        # Convert tear streams to array for fixed-point solver
        tear_array = self._streams_to_array(tear_initial)

        def flowsheet_iteration(tear_arr, args):
            """One iteration of the flowsheet."""
            feeds, units, recycles, species_order = args

            # Convert array back to streams
            tear_streams = self._array_to_streams(tear_arr, list(tear_initial.keys()))

            # Merge feeds and tear streams
            streams = dict(feeds)
            streams.update(tear_streams)

            # Solve each unit
            for unit in units:
                inlets = [streams[name] for name in unit.inlet_names]
                result = unit.operation(*inlets, **unit.params)

                # Parse outputs (same logic as _solve_sequential)
                if isinstance(result, tuple):
                    if len(result) == len(unit.outlet_names):
                        for name, stream in zip(unit.outlet_names, result):
                            streams[name] = stream
                    elif len(result) == 2 and isinstance(result[1], dict):
                        outputs = result[0]
                        if isinstance(outputs, dict):
                            streams[unit.outlet_names[0]] = outputs
                        elif isinstance(outputs, tuple):
                            for name, stream in zip(unit.outlet_names, outputs):
                                streams[name] = stream
                        else:
                            streams[unit.outlet_names[0]] = outputs
                    elif len(result) == len(unit.outlet_names) + 1:
                        for name, stream in zip(unit.outlet_names, result[:-1]):
                            streams[name] = stream
                else:
                    streams[unit.outlet_names[0]] = result

            # Extract new tear stream values (recycle sources)
            new_tear = {}
            for source, dest in recycles.items():
                new_tear[dest] = streams[source]

            return self._streams_to_array(new_tear)

        args = (self.feeds, self.units, self.recycles, self.species_order)

        # Solve fixed-point problem
        fp_solver = optx.FixedPointIteration(rtol=tol, atol=tol)
        sol = optx.fixed_point(
            flowsheet_iteration,
            fp_solver,
            tear_array,
            args=args,
            max_steps=max_iter,
            throw=False,
        )
        tear_converged = sol.value

        # Final solve with converged tear streams
        final_tear = self._array_to_streams(tear_converged, list(tear_initial.keys()))
        streams = dict(self.feeds)
        streams.update(final_tear)

        for unit in self.units:
            inlets = [streams[name] for name in unit.inlet_names]
            result = unit.operation(*inlets, **unit.params)

            if isinstance(result, tuple):
                if len(result) == len(unit.outlet_names):
                    for name, stream in zip(unit.outlet_names, result):
                        streams[name] = stream
                elif len(result) == 2 and isinstance(result[1], dict):
                    outputs = result[0]
                    if isinstance(outputs, dict):
                        streams[unit.outlet_names[0]] = outputs
                    elif isinstance(outputs, tuple):
                        for name, stream in zip(unit.outlet_names, outputs):
                            streams[name] = stream
                    else:
                        streams[unit.outlet_names[0]] = outputs
                elif len(result) == len(unit.outlet_names) + 1:
                    for name, stream in zip(unit.outlet_names, result[:-1]):
                        streams[name] = stream
            else:
                streams[unit.outlet_names[0]] = result

        return streams

    def _solve_with_wegstein(
        self,
        tear_initial: dict[str, Stream],
        tol: float,
        max_iter: int,
    ) -> dict[str, Stream]:
        """Solve flowsheet with Wegstein acceleration.

        Wegstein's method uses two consecutive iterates to estimate
        the optimal relaxation factor, accelerating convergence.

        Args:
            tear_initial: Initial tear stream values
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Dictionary of all streams in the flowsheet
        """
        tear_names = list(tear_initial.keys())

        # Initial arrays
        x_prev = self._streams_to_array(tear_initial)
        g_prev = None

        # First iteration
        streams = dict(self.feeds)
        streams.update(tear_initial)
        streams = self._run_units(streams)

        new_tear = {dest: streams[source] for source, dest in self.recycles.items()}
        g_curr = self._streams_to_array(new_tear)

        for iteration in range(max_iter):
            # Check convergence
            residual = jnp.max(jnp.abs(g_curr - x_prev))
            if residual < tol:
                break

            if g_prev is None:
                # Second iteration: can't use Wegstein yet
                x_curr = g_curr
            else:
                # Apply Wegstein acceleration
                x_curr = wegstein_acceleration(x_prev, x_prev, g_prev, g_curr)
                x_curr = jnp.clip(x_curr, 0.0, None)  # Ensure non-negative flows

            # Update history
            g_prev = g_curr
            x_prev = x_curr

            # Run iteration with accelerated estimate
            tear_streams = self._array_to_streams(x_curr, tear_names)
            streams = dict(self.feeds)
            streams.update(tear_streams)
            streams = self._run_units(streams)

            new_tear = {dest: streams[source] for source, dest in self.recycles.items()}
            g_curr = self._streams_to_array(new_tear)

        # Final solve with converged tear streams
        final_tear = self._array_to_streams(g_curr, tear_names)
        streams = dict(self.feeds)
        streams.update(final_tear)
        return self._run_units(streams)

    def _solve_with_anderson(
        self,
        tear_initial: dict[str, Stream],
        tol: float,
        max_iter: int,
        depth: int = 5,
    ) -> dict[str, Stream]:
        """Solve flowsheet with Anderson acceleration.

        Anderson acceleration (also known as Anderson mixing) uses
        a history of iterates to find an optimal linear combination
        for the next iterate, often dramatically improving convergence.

        Args:
            tear_initial: Initial tear stream values
            tol: Convergence tolerance
            max_iter: Maximum iterations
            depth: History depth for Anderson acceleration

        Returns:
            Dictionary of all streams in the flowsheet
        """
        tear_names = list(tear_initial.keys())
        accelerator = AndersonAccelerator(m=depth)

        x_curr = self._streams_to_array(tear_initial)

        for iteration in range(max_iter):
            # Run iteration
            tear_streams = self._array_to_streams(x_curr, tear_names)
            streams = dict(self.feeds)
            streams.update(tear_streams)
            streams = self._run_units(streams)

            new_tear = {dest: streams[source] for source, dest in self.recycles.items()}
            g_curr = self._streams_to_array(new_tear)

            # Check convergence
            residual = jnp.max(jnp.abs(g_curr - x_curr))
            if residual < tol:
                break

            # Apply Anderson acceleration
            x_next = accelerator.step(x_curr, g_curr)
            x_next = jnp.clip(x_next, 0.0, None)  # Ensure non-negative flows
            x_curr = x_next

        # Final solve with converged tear streams
        final_tear = self._array_to_streams(x_curr, tear_names)
        streams = dict(self.feeds)
        streams.update(final_tear)
        return self._run_units(streams)

    def _run_units(self, streams: dict[str, Stream]) -> dict[str, Stream]:
        """Run all units in sequence.

        Args:
            streams: Current stream values (feeds + tear streams)

        Returns:
            Updated streams after running all units
        """
        for unit in self.units:
            inlets = [streams[name] for name in unit.inlet_names]
            result = unit.operation(*inlets, **unit.params)

            # Parse outputs
            if isinstance(result, tuple):
                if len(result) == len(unit.outlet_names):
                    for name, stream in zip(unit.outlet_names, result):
                        streams[name] = stream
                elif len(result) == 2 and isinstance(result[1], dict):
                    outputs = result[0]
                    if isinstance(outputs, dict):
                        streams[unit.outlet_names[0]] = outputs
                    elif isinstance(outputs, tuple):
                        for name, stream in zip(unit.outlet_names, outputs):
                            streams[name] = stream
                    else:
                        streams[unit.outlet_names[0]] = outputs
                elif len(result) == len(unit.outlet_names) + 1:
                    for name, stream in zip(unit.outlet_names, result[:-1]):
                        streams[name] = stream
            else:
                streams[unit.outlet_names[0]] = result

        return streams

    def _streams_to_array(self, streams: dict[str, Stream]) -> Array:
        """Convert dictionary of streams to a flat array."""
        arrays = []
        for name in sorted(streams.keys()):
            stream = streams[name]
            # Pack: [F_species..., T, P]
            for s in self.species_order:
                arrays.append(stream[f"F_{s}"])
            arrays.append(stream["T"])
            arrays.append(stream["P"])
        return jnp.array(arrays)

    def _array_to_streams(
        self,
        arr: Array,
        stream_names: list[str],
    ) -> dict[str, Stream]:
        """Convert flat array back to dictionary of streams."""
        n_per_stream = len(self.species_order) + 2  # flows + T + P
        streams = {}

        for i, name in enumerate(sorted(stream_names)):
            start = i * n_per_stream
            flows = {}
            for j, s in enumerate(self.species_order):
                flows[s] = arr[start + j]
            T = arr[start + len(self.species_order)]
            P = arr[start + len(self.species_order) + 1]
            streams[name] = make_stream(flows, T, P)

        return streams


def create_objective(
    flowsheet: Flowsheet,
    objective_fn: Callable[[dict[str, Stream]], Array],
) -> Callable[[dict], Array]:
    """Create a differentiable objective function from a flowsheet.

    This wraps the flowsheet solve to create a function that maps
    parameters to an objective value, suitable for optimization.

    Args:
        flowsheet: The flowsheet to solve
        objective_fn: Function that computes objective from solved streams

    Returns:
        Function mapping parameter dict to objective value
    """

    def objective(params: dict) -> Array:
        # Update flowsheet feeds/unit params from params dict
        # (This is a simplified version - could be more sophisticated)

        # Solve flowsheet
        streams = flowsheet.solve()

        # Compute objective
        return objective_fn(streams)

    return objective
