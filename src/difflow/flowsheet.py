"""Flowsheet module for connecting unit operations.

This module provides:
- Flowsheet class for defining and solving process flowsheets
- Sequential modular approach with recycle convergence
- Implicit differentiation through converged recycle loops

The flowsheet is defined by adding units and connections, then
solved using fixed-point iteration on tear streams.
"""

from typing import Callable, Any
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array
import jax

from difflow.streams import Stream, get_species, get_flows, make_stream
from difflow.solvers import fixed_point_solve


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
    ) -> dict[str, Stream]:
        """Solve the flowsheet.

        Uses fixed-point iteration on tear streams with implicit
        differentiation through the converged solution.

        Args:
            tear_initial: Initial guesses for tear streams (recycle destinations).
                         If None, uses zero flows.
            tol: Convergence tolerance
            max_iter: Maximum iterations
            damping: Damping factor for tear stream updates

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
            else:
                # Initialize with small flows
                tear_streams[dest] = self._make_zero_stream()

        # Solve with fixed-point iteration
        return self._solve_with_recycle(tear_streams, tol, max_iter, damping)

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

    def _solve_with_recycle(
        self,
        tear_initial: dict[str, Stream],
        tol: float,
        max_iter: int,
        damping: float,
    ) -> dict[str, Stream]:
        """Solve flowsheet with recycle using fixed-point iteration."""

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
        tear_converged = fixed_point_solve(
            flowsheet_iteration,
            tear_array,
            args,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )

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

    def to_graph(
        self,
        solved_streams: dict[str, Stream] | None = None,
        name: str | None = None,
    ):
        """Convert flowsheet to a visualization graph.

        Creates a FlowsheetGraph from the flowsheet topology,
        optionally populated with solved stream data.

        Args:
            solved_streams: Dictionary of solved streams from fs.solve().
                           If None, graph shows topology only.
            name: Name for the graph (defaults to "Flowsheet")

        Returns:
            FlowsheetGraph with nodes for units and edges for streams

        Example:
            >>> fs = Flowsheet(["A", "B"])
            >>> fs.add_feed("feed", feed_stream)
            >>> fs.add_unit(Unit("reactor", cstr, ["feed"], ["product"]))
            >>> streams = fs.solve()
            >>> graph = fs.to_graph(streams)
            >>> render_flowsheet(graph)
        """
        from difflow.visualization import FlowsheetGraph

        graph = FlowsheetGraph(name=name or "Flowsheet")

        # Track which streams are unit outputs
        stream_sources: dict[str, str] = {}  # stream_name -> unit_name

        # Add feed nodes
        for feed_name in self.feeds:
            feed_node_id = f"_feed_{feed_name}"
            graph.add_node(
                feed_node_id,
                name=feed_name,
                unit_type="feed",
            )
            stream_sources[feed_name] = feed_node_id

        # Add unit nodes
        for unit in self.units:
            unit_type = _get_unit_type(unit.operation)

            # Extract params for tooltip
            params = dict(unit.params) if unit.params else {}

            graph.add_node(
                unit.name,
                name=unit.name,
                unit_type=unit_type,
                params=params,
            )

            # Record outlet streams
            for outlet_name in unit.outlet_names:
                stream_sources[outlet_name] = unit.name

        # Build edges from connections
        for unit in self.units:
            for inlet_name in unit.inlet_names:
                # Find source of this inlet
                if inlet_name in stream_sources:
                    source_unit = stream_sources[inlet_name]

                    # Get stream data if available
                    stream_data = None
                    if solved_streams and inlet_name in solved_streams:
                        stream_data = solved_streams[inlet_name]

                    graph.add_edge(
                        source_unit,
                        unit.name,
                        edge_id=inlet_name,
                        stream_data=stream_data,
                    )

        # Add product nodes for streams that aren't consumed
        consumed_streams = set()
        for unit in self.units:
            consumed_streams.update(unit.inlet_names)

        # Recycle destinations are also "consumed"
        consumed_streams.update(self.recycles.values())

        for unit in self.units:
            for outlet_name in unit.outlet_names:
                if outlet_name not in consumed_streams:
                    # This is a product stream
                    product_node_id = f"_product_{outlet_name}"
                    graph.add_node(
                        product_node_id,
                        name=outlet_name,
                        unit_type="product",
                    )

                    stream_data = None
                    if solved_streams and outlet_name in solved_streams:
                        stream_data = solved_streams[outlet_name]

                    graph.add_edge(
                        unit.name,
                        product_node_id,
                        edge_id=outlet_name,
                        stream_data=stream_data,
                    )

        # Add recycle edges
        for source, dest in self.recycles.items():
            # Find which unit produces the source stream
            if source in stream_sources:
                source_unit = stream_sources[source]

                # Find which unit consumes the dest stream
                for unit in self.units:
                    if dest in unit.inlet_names:
                        stream_data = None
                        if solved_streams and source in solved_streams:
                            stream_data = solved_streams[source]

                        # Check if edge already exists
                        edge_id = f"{source}_recycle"
                        if edge_id not in graph.edges:
                            graph.add_edge(
                                source_unit,
                                unit.name,
                                edge_id=edge_id,
                                stream_data=stream_data,
                                type="recycle",
                            )
                        break

        return graph

    def visualize(
        self,
        solved_streams: dict[str, Stream] | None = None,
        **kwargs,
    ):
        """Render and display the flowsheet interactively.

        Convenience method that creates a graph and renders it.

        Args:
            solved_streams: Dictionary of solved streams from fs.solve()
            **kwargs: Additional arguments passed to render_flowsheet

        Returns:
            Plotly Figure object
        """
        from difflow.visualization import render_flowsheet

        graph = self.to_graph(solved_streams)
        return render_flowsheet(graph, **kwargs)


def _get_unit_type(operation: Callable) -> str:
    """Extract unit type name from an operation callable."""
    # Try to get class name
    if hasattr(operation, '__class__'):
        cls = operation.__class__
        if cls.__name__ != 'function':
            return cls.__name__

    # Try to get function name
    if hasattr(operation, '__name__'):
        return operation.__name__

    return "generic"


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
