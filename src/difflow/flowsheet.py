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
from dataclasses import dataclass, field, replace, is_dataclass
from dataclasses import fields as dc_fields
import copy
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


FEED_PREFIX = "feed:"


def _concrete(value) -> float | None:
    """Return ``float(value)`` when it is concrete, else ``None``.

    The solve loops record a residual for the report layer.  Under
    ``jax.grad``/``jit`` that residual is a tracer and has no numeric value, so
    the diagnostic is simply unavailable -- reporting ``None`` is honest,
    whereas ``float()`` would raise and make the flowsheet undifferentiable.
    """
    try:
        return float(value)
    except Exception:
        return None


def _has_tracer(obj) -> bool:
    """True if any pytree leaf of ``obj`` is a JAX tracer."""
    return any(isinstance(leaf, jax.core.Tracer)
               for leaf in jax.tree_util.tree_leaves(obj))

def _update_feed_stream(stream: Stream, updates: dict[str, Any]) -> Stream:
    """Return a copy of ``stream`` with planning-style updates applied.

    Feed streams are the classic levers of an LP planning model -- feed rate
    and feed composition -- so they need to be addressable the same way unit
    parameters are.  The recognised field names are:

    ``F_<species>``
        Set that species' molar flow directly (mol/s).
    ``x_<species>``
        Set that species' mole fraction, rescaling the *other* species
        proportionally so the total flow is unchanged.
    ``total_flow``
        Scale every species flow by one factor so the total matches, leaving
        the composition unchanged.
    ``T`` / ``P``
        Set temperature (K) or pressure (Pa).

    Updates are applied in that order regardless of dict ordering, so
    ``{"x_A": 0.3, "total_flow": 10.0}`` means "30 mol% A at 10 mol/s".
    All ``x_`` keys are applied together, so two of them do not fight.

    Every operation is a pure ``jnp`` expression, so the result differentiates
    and traces under ``jit``/``vmap``.

    Args:
        stream: The feed stream to update.  Not modified.
        updates: Field name -> value.

    Returns:
        A new stream dict.

    Raises:
        KeyError: If a field name is not one of the forms above, or names a
            species the stream does not carry.
    """
    new = dict(stream)
    flow_keys = [k for k in new if k.startswith("F_")]

    def _known(field: str) -> bool:
        if field in ("T", "P", "total_flow") or field in flow_keys:
            return True
        return field.startswith("x_") and f"F_{field[2:]}" in flow_keys

    unknown = [k for k in updates if not _known(k)]
    if unknown:
        species = [k[2:] for k in flow_keys]
        raise KeyError(
            f"Unknown feed field(s) {unknown!r}. Use 'T', 'P', 'total_flow', "
            f"'F_<species>' or 'x_<species>' with species in {species}"
        )

    # 1. Direct species flows.
    for k in flow_keys:
        if k in updates:
            new[k] = jnp.asarray(updates[k], dtype=jnp.float64)

    # 2. Mole fractions, all at once, at constant total flow.
    x_updates = {k[2:]: v for k, v in updates.items() if k.startswith("x_")}
    if x_updates:
        total = sum(new[k] for k in flow_keys)
        targets = {f"F_{s}": jnp.asarray(v, dtype=jnp.float64) * total
                   for s, v in x_updates.items()}
        held_old = sum(new[k] for k in targets)
        held_new = sum(targets.values())
        rest_old = total - held_old
        # Guard the degenerate case where the untouched species carry no flow:
        # there is nothing to rescale, so leave them at zero.
        safe = jnp.where(rest_old > 0.0, rest_old, 1.0)
        scale = jnp.where(rest_old > 0.0, (total - held_new) / safe, 0.0)
        for k in flow_keys:
            new[k] = targets[k] if k in targets else new[k] * scale

    # 3. Total flow, holding composition.
    if "total_flow" in updates:
        total = sum(new[k] for k in flow_keys)
        scale = jnp.asarray(updates["total_flow"], dtype=jnp.float64) / total
        for k in flow_keys:
            new[k] = new[k] * scale

    # 4. Conditions.
    for k in ("T", "P"):
        if k in updates:
            new[k] = jnp.asarray(updates[k], dtype=jnp.float64)

    return new


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

    def __init__(
        self,
        species_order: list[str],
        default_flow: float = 0.01,
        default_T: float = 300.0,
        default_P: float = 101325.0,
    ):
        """Initialize empty flowsheet.

        Args:
            species_order: List of species names for stream arrays
            default_flow: Default molar flow (mol/s) for tear stream
                initialization when no explicit initial guess is provided.
                Use a smaller value (e.g. 1e-6) for trace species.
            default_T: Default temperature (K) for tear stream initialization
            default_P: Default pressure (Pa) for tear stream initialization
        """
        self.species_order = species_order
        self.default_flow = default_flow
        self.default_T = default_T
        self.default_P = default_P
        self.units: list[Unit] = []
        self.feeds: dict[str, Stream] = {}
        self.recycles: dict[str, str] = {}  # {source_name: dest_name}
        #: Presentation state, round-tripped by :mod:`difflow.serialize` and
        #: read by nothing numeric.  The editor keeps canvas positions
        #: (``view["nodes"]``), its code-context snippet and its planning
        #: selection here, so a flowsheet that has been laid out by hand opens
        #: the way it was left.  Anything may go in; nothing in the solve path
        #: looks.
        self.view: dict[str, Any] = {}
        self._stream_cache: dict[str, Stream] = {}
        #: tear iterations used by the last accelerated solve (Wegstein
        #: or Anderson); equals max_iter if it did not converge
        self.last_solve_iterations: int | None = None
        #: final tear residual (max abs change) of the last recycle solve
        self.last_solve_residual: float | None = None
        #: whether the last recycle solve met its tolerance
        self.last_solve_converged: bool | None = None
        #: acceleration method of the last solve ("anderson", "wegstein",
        #: "damped", or "direct" for a recycle-free sequential solve)
        self.last_solve_method: str | None = None
        #: tolerance requested of the last recycle solve
        self.last_solve_tol: float | None = None
        #: tear-stream (recycle destination) names of the last recycle solve
        self.last_solve_tear_streams: list[str] = []

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
        clip_negative_flows: bool = True,
    ) -> dict[str, Stream]:
        """Solve the flowsheet.

        Uses fixed-point iteration on tear streams with optional
        acceleration (Anderson or Wegstein) and implicit differentiation
        through the converged solution.

        Args:
            tear_initial: Initial guesses for tear streams (recycle destinations).
                If None, uses initialization from units or the default stream
                created by ``_make_zero_stream()`` (controlled by
                ``default_flow``, ``default_T``, ``default_P`` on the
                Flowsheet). Pass a dict mapping destination stream names to
                Stream objects to provide explicit guesses::

                    fs.solve(tear_initial={"recycle": my_guess_stream})

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
            clip_negative_flows: If True (default), the accelerated
                iterations (Wegstein, Anderson) project the tear-stream
                flows onto [0, inf) after each step. This is a safeguard
                for molar flows in chemical flowsheets, but it must be
                disabled for flowsheets whose tear flows are signed
                (e.g. bidirectional flows in gas transmission networks),
                where a negative fixed point is legitimate and clipping
                prevents convergence. Only flow entries are clipped;
                temperature and pressure are never touched.

        Returns:
            Dictionary of all streams in the flowsheet
        """
        if not self.recycles:
            # No recycles - simple sequential solution
            self.last_solve_method = "direct"
            self.last_solve_iterations = 0
            self.last_solve_residual = 0.0
            self.last_solve_converged = True
            self.last_solve_tol = tol
            self.last_solve_tear_streams = []
            return self._solve_sequential()

        self.last_solve_method = acceleration
        self.last_solve_tol = tol
        self.last_solve_tear_streams = [dest for dest in self.recycles.values()]

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

        # The accelerated solvers are Python loops that branch on the residual
        # to stop early, which a tracer cannot answer.  Under jax.grad / jit,
        # fall back to the optimistix fixed point instead: it converges without
        # a Python branch and carries an implicit-differentiation rule, so the
        # gradient comes from the converged solution rather than from an
        # unrolled iteration.  Recorded in last_solve_method so the report
        # layer tells the truth about what ran.
        if acceleration != "none" and self._is_traced():
            acceleration = "none"
            self.last_solve_method = "fixed_point (traced)"

        # Solve with chosen method
        if acceleration == "none":
            return self._solve_with_recycle_damped(tear_streams, tol, max_iter, damping)
        elif acceleration == "wegstein":
            return self._solve_with_wegstein(
                tear_streams, tol, max_iter,
                clip_negative_flows=clip_negative_flows,
            )
        elif acceleration == "anderson":
            return self._solve_with_anderson(
                tear_streams, tol, max_iter, anderson_depth,
                clip_negative_flows=clip_negative_flows,
            )
        else:
            raise ValueError(f"Unknown acceleration method: {acceleration}")

    def _is_traced(self) -> bool:
        """True when a feed or unit parameter currently holds a JAX tracer.

        Used by :meth:`solve` to pick a differentiable solver.  Unit ``Params``
        dataclasses are not registered pytrees, so their fields are collected
        explicitly.
        """
        if _has_tracer(self.feeds):
            return True
        for unit in self.units:
            params = getattr(unit.operation, "params", None)
            if params is None or not is_dataclass(params):
                continue
            values = [getattr(params, f.name) for f in dc_fields(params)]
            if _has_tracer([v for v in values if not callable(v)]):
                return True
        return False

    def _tear_flow_mask(self, n_streams: int) -> Array:
        """Boolean mask marking the flow entries of a packed tear array.

        The packed layout (see ``_streams_to_array``) is, per stream,
        ``[F_species..., T, P]``. Flow clipping must only ever touch the
        flow entries, so the accelerated solvers use this mask.
        """
        per_stream = [True] * len(self.species_order) + [False, False]
        return jnp.array(per_stream * n_streams)

    def _clip_flows(self, x: Array, mask: Array) -> Array:
        """Project the flow entries of a packed tear array onto [0, inf)."""
        return jnp.where(mask, jnp.clip(x, 0.0, None), x)

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
        """Create a stream with small default flows for tear initialization."""
        flows = {s: jnp.asarray(self.default_flow) for s in self.species_order}
        return make_stream(flows, self.default_T, self.default_P)

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

        # Record convergence diagnostics for the report layer.
        final_residual = _concrete(
            jnp.max(jnp.abs(flowsheet_iteration(tear_converged, args) - tear_converged))
        )
        self.last_solve_residual = final_residual
        self.last_solve_converged = (None if final_residual is None
                                     else bool(final_residual < tol))
        try:
            self.last_solve_iterations = int(sol.stats.get("num_steps", max_iter))
        except Exception:
            self.last_solve_iterations = max_iter

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
        clip_negative_flows: bool = True,
    ) -> dict[str, Stream]:
        """Solve flowsheet with Wegstein acceleration.

        Wegstein's method uses two consecutive iterates to estimate
        the optimal relaxation factor, accelerating convergence.

        Args:
            tear_initial: Initial tear stream values
            tol: Convergence tolerance
            max_iter: Maximum iterations
            clip_negative_flows: Project tear flows onto [0, inf) after
                each accelerated step; disable for signed flows.

        Returns:
            Dictionary of all streams in the flowsheet
        """
        tear_names = list(tear_initial.keys())
        flow_mask = self._tear_flow_mask(len(tear_names))

        # Initial arrays
        x_prev = self._streams_to_array(tear_initial)
        self.last_solve_iterations = max_iter
        x_old = None
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
            res = _concrete(residual)
            self.last_solve_residual = res
            if res is not None and res < tol:
                self.last_solve_iterations = iteration
                self.last_solve_converged = True
                break
            self.last_solve_converged = False

            if g_prev is None:
                # Second iteration: can't use Wegstein yet
                x_curr = g_curr
            else:
                # Apply Wegstein acceleration
                x_curr = wegstein_acceleration(x_old, x_prev, g_prev, g_curr)
                if clip_negative_flows:
                    x_curr = self._clip_flows(x_curr, flow_mask)

            # Update history
            g_prev = g_curr
            x_old = x_prev
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
        clip_negative_flows: bool = True,
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
            clip_negative_flows: Project tear flows onto [0, inf) after
                each accelerated step; disable for signed flows.

        Returns:
            Dictionary of all streams in the flowsheet
        """
        tear_names = list(tear_initial.keys())
        flow_mask = self._tear_flow_mask(len(tear_names))
        accelerator = AndersonAccelerator(m=depth)

        x_curr = self._streams_to_array(tear_initial)

        self.last_solve_iterations = max_iter
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
            res = _concrete(residual)
            self.last_solve_residual = res
            if res is not None and res < tol:
                self.last_solve_iterations = iteration
                self.last_solve_converged = True
                break
            self.last_solve_converged = False

            # Apply Anderson acceleration
            x_next = accelerator.step(x_curr, g_curr)
            if clip_negative_flows:
                x_next = self._clip_flows(x_next, flow_mask)
            x_curr = x_next

        # Final solve with converged tear streams
        final_tear = self._array_to_streams(x_curr, tear_names)
        streams = dict(self.feeds)
        streams.update(final_tear)
        return self._run_units(streams)

    def _apply_params(self, params: dict) -> "Flowsheet":
        """Return a copy of this flowsheet with unit params updated from a dict.

        Keys use dot notation to target specific unit parameters:
        ``"<unit_name>.<param_name>"`` updates ``param_name`` on the unit
        operation whose :attr:`Unit.name` matches ``unit_name``.  For example,
        ``{"reactor.V": 2.0}`` will call
        ``unit.operation.params.update(V=2.0)`` on the unit named
        ``"reactor"``.

        Feed streams are addressed with a ``"feed:"`` prefix:
        ``{"feed:F1.total_flow": 12.0, "feed:F1.x_A": 0.3, "feed:F1.T": 340.0}``.
        See :func:`_update_feed_stream` for the recognised field names
        (``T``, ``P``, ``total_flow``, ``F_<species>``, ``x_<species>``).

        Args:
            params: Flat dict of ``"unit.field"`` or ``"feed:stream.field"``
                -> value entries.

        Returns:
            New :class:`Flowsheet` instance with the requested parameters
            applied.  The original flowsheet is not modified.

        Raises:
            KeyError: If the unit or feed name in a dotted key does not exist,
                or a feed field name is unrecognised.
            AttributeError: If the unit's operation has no ``.params``
                attribute or the field name is not present in params.
            ValueError: If a key does not contain exactly one dot.
        """
        # Build a name -> Unit index map
        unit_index = {unit.name: i for i, unit in enumerate(self.units)}

        # Collect per-unit and per-feed updates: {name: {field: value}}
        unit_updates: dict[str, dict[str, Any]] = {}
        feed_updates: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            if key.startswith(FEED_PREFIX):
                feed_parts = key[len(FEED_PREFIX):].split(".", 1)
                if len(feed_parts) != 2:
                    raise ValueError(
                        f"Parameter key {key!r} must use dot notation "
                        f"'feed:<stream_name>.<field>'"
                    )
                feed_name, feed_field = feed_parts
                if feed_name not in self.feeds:
                    raise KeyError(
                        f"No feed stream named {feed_name!r} in flowsheet. "
                        f"Available feeds: {list(self.feeds.keys())}"
                    )
                feed_updates.setdefault(feed_name, {})[feed_field] = value
                continue
            parts = key.split(".", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Parameter key {key!r} must use dot notation "
                    f"'<unit_name>.<param_name>'"
                )
            unit_name, field_name = parts
            if unit_name not in unit_index:
                raise KeyError(
                    f"No unit named {unit_name!r} in flowsheet. "
                    f"Available units: {list(unit_index.keys())}"
                )
            unit_updates.setdefault(unit_name, {})[field_name] = value

        # Clone the units list, applying updates where needed
        new_units = list(self.units)
        for unit_name, updates in unit_updates.items():
            idx = unit_index[unit_name]
            unit = self.units[idx]
            operation = unit.operation
            if not hasattr(operation, "params"):
                raise AttributeError(
                    f"Unit {unit_name!r} operation has no .params attribute; "
                    f"cannot apply parameter updates {list(updates.keys())}"
                )
            new_op_params = operation.params.update(**updates)
            # Shallow-copy the operation object and replace its .params so that
            # all other attributes (e.g. thermo, mode) are preserved without
            # having to know the constructor signature of each unit type.
            new_operation = copy.copy(operation)
            new_operation.params = new_op_params
            new_unit = replace(unit, operation=new_operation)
            new_units[idx] = new_unit

        # Build a shallow copy of the flowsheet with the new units
        new_fs = Flowsheet(
            self.species_order,
            default_flow=self.default_flow,
            default_T=self.default_T,
            default_P=self.default_P,
        )
        new_feeds = dict(self.feeds)
        for feed_name, updates in feed_updates.items():
            new_feeds[feed_name] = _update_feed_stream(
                self.feeds[feed_name], updates
            )
        new_fs.feeds = new_feeds
        new_fs.recycles = dict(self.recycles)
        new_fs.units = new_units
        new_fs.view = copy.deepcopy(self.view)
        return new_fs

    def make_objective_fn(
        self,
        objective_fn: Callable[[dict[str, Stream]], Array],
    ) -> Callable[[dict], Array]:
        """Create a differentiable objective function from this flowsheet.

        The returned function accepts a flat parameter dict whose keys use
        dot notation (``"<unit_name>.<param_name>"``) to identify which unit
        and field to update before solving.  For example::

            obj = fs.make_objective_fn(lambda s: s["product"]["F_B"])
            value = obj({"reactor.V": 2.0})
            grad  = jax.grad(obj)({"reactor.V": 2.0})

        The convention for parameter keys is:
        ``"<unit_name>.<param_name>"`` where ``unit_name`` is the
        :attr:`Unit.name` as registered with :meth:`add_unit`, and
        ``param_name`` is a field on that unit's ``operation.params``
        :class:`~difflow.params_mixin.ParamsMixin` dataclass.  Feed streams
        are addressed as ``"feed:<stream_name>.<field>"`` -- see
        :meth:`_apply_params` -- so feed rate, composition and conditions are
        levers too::

            grad = jax.grad(obj)({"reactor.V": 2.0, "feed:F1.total_flow": 10.0})

        Args:
            objective_fn: Callable that maps the solved stream dict returned
                by :meth:`solve` to a scalar :class:`jax.Array`.

        Returns:
            Callable ``objective(params_dict) -> Array`` that updates unit
            parameters, solves the flowsheet, then evaluates
            ``objective_fn``.
        """

        def objective(params: dict) -> Array:
            # Apply parameter updates to a temporary copy of the flowsheet
            # using dot-notation keys of the form "<unit_name>.<param_name>".
            updated_fs = self._apply_params(params)
            streams = updated_fs.solve()
            return objective_fn(streams)

        return objective

    def report(
        self,
        streams: dict[str, Stream] | None = None,
        include_git: bool = True,
        optimization=None,
        db_access=None,
        notes: list[str] | None = None,
    ):
        """Build a self-documenting :class:`~difflow.report.ir.Report` for this flowsheet.

        Args:
            streams: Optional solved streams from :meth:`solve`. When omitted,
                only the configuration sections (topology, units, species,
                feeds) are populated and the ``results`` /
                ``balance_checks`` fields are ``None``.
            include_git: Whether to capture git commit + dirty flag in the
                provenance section.
            optimization: Optional
                :class:`~difflow.report.ir.OptimizationReport` (build one with
                :func:`difflow.report.build_optimization_report`) to include the
                optimization / sensitivity section.
            notes: Optional free-form notes to attach to the report.

        Returns:
            :class:`difflow.report.Report` with ``to_markdown()``,
            ``to_json()``, ``to_html()``, and ``to_latex()`` methods.
        """
        from difflow.report.builder import build_report

        return build_report(
            self,
            streams=streams,
            include_git=include_git,
            optimization=optimization,
            db_access=db_access,
            notes=notes,
        )

    def solve_eo(
        self,
        initial_guess: dict[str, Stream] | None = None,
        use_sm_init: bool = True,
        tol: float = 1e-8,
        max_steps: int = 100,
    ) -> dict[str, Stream]:
        """Solve the flowsheet using the equation-oriented approach.

        Assembles all unit equations into a single system F(x) = 0
        and solves simultaneously with Newton's method. Provides
        implicit differentiation through the solution.

        Args:
            initial_guess: Initial values for unknown streams.
                          If None, uses SM initialization or feed propagation.
            use_sm_init: If True and no initial_guess, run SM solver first
                        to get a good starting point.
            tol: Convergence tolerance
            max_steps: Maximum Newton iterations

        Returns:
            Dictionary of all streams in the flowsheet
        """
        from difflow.eo_solver import EOSolver

        solver = EOSolver(self)
        return solver.solve_streams(
            initial_guess=initial_guess,
            use_sm_init=use_sm_init,
            tol=tol,
            max_steps=max_steps,
        )

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

    Parameter keys use dot notation ``"<unit_name>.<param_name>"`` to
    identify which unit operation parameter to update before solving.
    See :meth:`Flowsheet.make_objective_fn` for full documentation.

    Args:
        flowsheet: The flowsheet to solve
        objective_fn: Function that computes objective from solved streams

    Returns:
        Function mapping parameter dict to objective value
    """
    return flowsheet.make_objective_fn(objective_fn)
