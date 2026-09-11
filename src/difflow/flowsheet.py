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
import warnings
import jax.numpy as jnp
from jax import Array
import jax

from difflow.streams import Stream, get_species, get_flows, make_stream
from difflow.initialization import (
    AndersonAccelerator,
    wegstein_acceleration,
    Initializable,
    TearInitializationWarning,
)
import optimistix as optx


FEED_PREFIX = "feed:"


class ConvergenceWarning(UserWarning):
    """A recycle solve returned without meeting its tolerance.

    Running out of iterations leaves :meth:`Flowsheet.solve` returning the
    last iterate, which looks like any other result: every stream is there
    and every number is plausible. It is not a solution, though --- the
    material balance around the loop is off by the tear residual, which for
    a loop that is limit-cycling can be large. This warning is what makes
    that visible instead of silent (#249), the way
    :class:`difflow.CSTRDensityWarning` does for an assumed density.

    Silence it with ``solve(on_nonconvergence="ignore")`` or escalate it to
    an exception with ``solve(on_nonconvergence="raise")``.
    """


class ConvergenceError(RuntimeError):
    """A recycle solve did not converge and was asked to raise.

    Only ``solve(on_nonconvergence="raise")`` produces this; the default is
    a :class:`ConvergenceWarning`. It carries the same message.
    """


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


def _parse_outlets(result: Any, unit: "Unit") -> dict[str, Stream]:
    """Name the streams a unit returned.

    A unit operation may hand back a single stream, a tuple of streams, a
    tuple of streams with an info dict on the end, or ``(streams, info)``.
    The mapping onto names is positional in every one of those shapes,
    which is what makes the order of ``outlet_names`` part of a unit's
    contract rather than a detail of how it was written down.
    """
    if not isinstance(result, tuple):
        return {unit.outlet_names[0]: result}

    if len(result) == len(unit.outlet_names):
        return dict(zip(unit.outlet_names, result))

    if len(result) == 2 and isinstance(result[1], dict):
        outputs = result[0]
        if isinstance(outputs, tuple):
            return dict(zip(unit.outlet_names, outputs))
        return {unit.outlet_names[0]: outputs}

    if len(result) == len(unit.outlet_names) + 1:
        return dict(zip(unit.outlet_names, result[:-1]))

    raise ValueError(f"Unexpected output from {unit.name}: {len(result)} items")


def _initialized_outlets(result: Any, unit: "Unit") -> dict[str, Stream]:
    """Name the streams a unit's ``initialize()`` returned.

    Positional, the same way :func:`_parse_outlets` is and for the same
    reason: ``outlets`` is the sequence in ``__call__`` order, and a
    caller holding only ``outlet_names`` has nothing else to match them
    up by. ``outlet`` is the singular spelling and means the same thing
    for a unit with one outlet -- which is why a flash, whose initializer
    used to answer only in ``liquid``/``vapor``, could never be read.
    """
    if not isinstance(result, dict):
        return {}

    outlets = result.get("outlets")
    if outlets is None and len(unit.outlet_names) == 1 and "outlet" in result:
        outlets = (result["outlet"],)
    if outlets is None:
        return {}

    return dict(zip(unit.outlet_names, outlets))


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
        #: "none", "fixed_point (traced)" when a tracer forced the
        #: differentiable path, or "direct" for a recycle-free solve)
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
        damping: float = 1.0,
        acceleration: Literal["none", "wegstein", "anderson"] = "anderson",
        anderson_depth: int = 5,
        use_initialization: bool = True,
        clip_negative_flows: bool = True,
        on_nonconvergence: Literal["warn", "raise", "ignore"] = "warn",
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
            damping: Step fraction for the tear update, used with
                ``acceleration="none"`` (and therefore on the traced path,
                which falls back to it). The iteration becomes
                ``x + damping * (g(x) - x)``, which has the same fixed point
                as ``g`` but contracts where plain substitution overshoots:
                for a tear map with eigenvalues in ``[-m, 0)`` it contracts
                for ``damping < 2/(1 + m)``, optimally near ``2/(2 + m)``.
                ``0.3`` is safe up to a spectral radius of about 5 and is the
                usual first thing to try on a loop whose residual *rises*.

                ``1.0`` (the default) is plain undamped substitution. The
                residual, the convergence verdict and the meaning of ``tol``
                are all in the undamped convention, so this changes how the
                solve gets to the answer and not which answer it is, nor how
                tightly it is converged.

                This argument was previously accepted and silently ignored;
                the default is ``1.0`` so that making it work changes nothing
                for a caller who never set it.
            acceleration: Acceleration method:
                - "none": Fixed-point iteration, damped by ``damping``
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
            on_nonconvergence: What to do when the tear iteration runs out
                of iterations without meeting ``tol``. The returned streams
                are the last iterate either way, and the diagnostics are in
                ``last_solve_*`` either way; this only decides how loudly
                the flowsheet says so.

                - ``"warn"`` (default): emit a :class:`ConvergenceWarning`
                  naming the tear streams, residual, tolerance and
                  iteration count.
                - ``"raise"``: raise :class:`ConvergenceError` instead.
                - ``"ignore"``: return silently.

                A solve under ``jax.grad``/``jit`` has no concrete residual
                to judge (:func:`_concrete` returns ``None``), so it stays
                silent whatever this is set to.

        Returns:
            Dictionary of all streams in the flowsheet

        Raises:
            ConvergenceError: If the solve did not converge and
                ``on_nonconvergence="raise"``.
            ValueError: If ``on_nonconvergence`` is not one of the three
                recognised values, ``acceleration`` is unknown, or
                ``damping`` is not positive.
        """
        if on_nonconvergence not in ("warn", "raise", "ignore"):
            raise ValueError(
                f"Unknown on_nonconvergence: {on_nonconvergence!r}. "
                'Expected "warn", "raise" or "ignore".'
            )
        if not damping > 0:
            # Checked here rather than only where it is used, so a nonsense
            # value is a failure at the call and not a silent no-op on a
            # flowsheet that happens to have no recycles.
            raise ValueError(
                f"damping must be positive, got {damping!r}. Damping scales "
                "the step toward g(x); at zero the iteration is the identity "
                "and every point is a fixed point."
            )

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

        # Initialize tear streams.  The guess comes from the SOURCE end of
        # each recycle: `dest` is an inlet, so nothing in the flowsheet
        # computes it, and asking for it was why this path used to return
        # the default every time (#247).  One propagation pass serves every
        # tear, so it is computed at most once however many recycles there
        # are.  Under tracing it is skipped: the initial guess cannot
        # affect a gradient that comes from the converged solution, and a
        # best-effort pass that catches exceptions has no business running
        # over tracers.
        tear_streams = {}
        propagated = None
        guess_tears = use_initialization and not self._is_traced()
        for source, dest in self.recycles.items():
            if tear_initial and dest in tear_initial:
                tear_streams[dest] = tear_initial[dest]
            elif guess_tears:
                if propagated is None:
                    propagated = self._propagate_from_feeds()
                tear_streams[dest] = self._initialize_tear_stream(source, propagated)
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

        # Solve with chosen method.  Every path records its verdict in
        # last_solve_converged, so the non-convergence check is done once
        # here rather than at each of the three returns.
        if acceleration == "none":
            streams = self._solve_with_recycle_damped(
                tear_streams, tol, max_iter, damping
            )
        elif acceleration == "wegstein":
            streams = self._solve_with_wegstein(
                tear_streams, tol, max_iter,
                clip_negative_flows=clip_negative_flows,
            )
        elif acceleration == "anderson":
            streams = self._solve_with_anderson(
                tear_streams, tol, max_iter, anderson_depth,
                clip_negative_flows=clip_negative_flows,
            )
        else:
            raise ValueError(f"Unknown acceleration method: {acceleration}")

        self._report_nonconvergence(on_nonconvergence, max_iter)
        return streams

    def _report_nonconvergence(self, action: str, max_iter: int) -> None:
        """Warn, raise or stay silent about the solve that just finished.

        ``last_solve_converged`` is tri-state on purpose: ``True`` met the
        tolerance, ``False`` ran out of iterations, and ``None`` means the
        residual was a tracer and there is nothing to judge.  Only ``False``
        is a finding -- guessing under tracing would either raise inside a
        gradient or warn on a solve that was fine.
        """
        if action == "ignore" or self.last_solve_converged is not False:
            return

        residual = self.last_solve_residual
        message = (
            "Recycle solve did not converge: tear stream(s) "
            f"{', '.join(self.last_solve_tear_streams) or '(none)'} reached a "
            f"residual of {'unknown' if residual is None else f'{residual:.3e}'} "
            f"after {self.last_solve_iterations} of {max_iter} iterations, "
            f"against a tolerance of {self.last_solve_tol:.3e} "
            f"(method: {self.last_solve_method}). The returned streams are the "
            "last iterate, not a solution -- material balances around the loop "
            "are off by the tear residual. Try more iterations (max_iter), a "
            "better tear guess (tear_initial) or a different acceleration."
        )
        if action == "raise":
            raise ConvergenceError(message)
        warnings.warn(message, ConvergenceWarning, stacklevel=3)

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

    def _initialize_tear_stream(
        self,
        stream_name: str,
        propagated: dict[str, Stream] | None = None,
    ) -> Stream:
        """Initial guess for a recycled stream, before any iteration.

        Args:
            stream_name: The **source** end of the recycle --- the outlet
                being recycled, not the inlet receiving it. A tear
                destination is an inlet and no unit computes one, so
                looking that end up could only ever find nothing.
            propagated: The result of :meth:`_propagate_from_feeds`, when
                the caller already has it. Computed here otherwise, which
                makes this callable on its own.

        Returns:
            The propagated value of the stream, or the flowsheet default
            when the walk could not produce a usable one.
        """
        if propagated is None:
            propagated = self._propagate_from_feeds()

        guess = propagated.get(stream_name)
        if guess is None or not self._usable_tear(guess):
            return self._make_zero_stream()
        return guess

    def _usable_tear(self, stream: Any) -> bool:
        """Whether a guess can stand in for a tear stream.

        ``_streams_to_array`` indexes every species in ``species_order``
        plus T and P, so a stream missing any of them does not fail as a
        bad guess --- it fails as a ``KeyError`` several frames away.
        """
        try:
            return (
                all(f"F_{s}" in stream for s in self.species_order)
                and "T" in stream
                and "P" in stream
            )
        except TypeError:
            return False

    def _propagate_from_feeds(self) -> dict[str, Stream]:
        """Run the units once from the feeds, with the recycles unknown.

        A tear has to start somewhere, and the cheapest honest answer is
        wherever one pass from the feeds puts it. Streams not available
        yet --- the tear destinations themselves, and the outlets of any
        unit that could not run --- stand in as the flowsheet default, so
        the walk always finishes and always has a value for every inlet
        it needs.

        A unit that raises on this pass is not necessarily broken. The
        pass runs before any recycle is known, so a unit inside a loop
        sees 0.01 mol/s arriving, and a CSTR's root find or a flash's
        Rachford-Rice can genuinely fail to converge on an almost empty
        stream. That is the one place ``initialize()`` earns its keep: an
        analytic estimate that cannot fail, standing in for the call that
        did.
        """
        streams: dict[str, Stream] = dict(self.feeds)
        default = self._make_zero_stream()

        for unit in self.units:
            inlets = [streams.get(name, default) for name in unit.inlet_names]
            try:
                result = unit.operation(*inlets, **unit.params)
            except Exception as failure:
                # Only the call is guarded. A unit that returns a shape
                # `_parse_outlets` cannot read is a bug in the unit, not a
                # stream too empty to run on, and it should say so here the
                # same way it does in a real solve.
                guessed = self._estimate_outlets(unit, inlets, failure)
                for name in unit.outlet_names:
                    streams[name] = guessed.get(name, default)
            else:
                streams.update(_parse_outlets(result, unit))

        return streams

    def _estimate_outlets(
        self,
        unit: Unit,
        inlets: list[Stream],
        failure: Exception,
    ) -> dict[str, Stream]:
        """What a unit's outlets look like when the unit itself would not run.

        Returns an empty dict when there is nothing better than the
        flowsheet default to say, having warned about it --- a tear guess
        that is quietly worse than it looks is the thing worth avoiding.
        """
        operation = unit.operation
        if not inlets or not isinstance(operation, Initializable):
            warnings.warn(
                f"{unit.name} could not run while guessing the recycle "
                f"({failure!r}), and has no initialize() to estimate it; "
                f"its outlets start from the flowsheet default.",
                TearInitializationWarning,
                stacklevel=4,
            )
            return {}

        try:
            estimate = operation.initialize(inlets[0], **unit.params)
        except Exception as also_failed:
            warnings.warn(
                f"{unit.name} could not run while guessing the recycle "
                f"({failure!r}), and neither could its initialize() "
                f"({also_failed!r}); its outlets start from the flowsheet "
                f"default.",
                TearInitializationWarning,
                stacklevel=4,
            )
            return {}

        return _initialized_outlets(estimate, unit)

    def _make_zero_stream(self) -> Stream:
        """Create a stream with small default flows for tear initialization."""
        flows = {s: jnp.asarray(self.default_flow) for s in self.species_order}
        return make_stream(flows, self.default_T, self.default_P)

    def _solve_sequential(self) -> dict[str, Stream]:
        """Solve flowsheet without recycles."""
        streams = dict(self.feeds)

        for unit in self.units:
            inlets = [streams[name] for name in unit.inlet_names]
            result = unit.operation(*inlets, **unit.params)
            streams.update(_parse_outlets(result, unit))

        return streams

    def _solve_with_recycle_damped(
        self,
        tear_initial: dict[str, Stream],
        tol: float,
        max_iter: int,
        damping: float,
    ) -> dict[str, Stream]:
        """Solve flowsheet with recycle using damped fixed-point iteration.

        Iterates ``x + alpha (g(x) - x)``, which has the same fixed point as
        ``g`` but a contractive iteration wherever ``g`` overshoots: for a
        tear map with eigenvalues in ``[-m, 0)`` the damped map contracts for
        ``alpha < 2/(1 + m)``, optimally near ``2/(2 + m)``.  ``alpha = 1``
        is plain substitution and is the default.

        Two things are deliberately kept in the *undamped* convention, so
        that ``damping`` changes how the solve gets there and nothing else:

        * the residual and the convergence verdict are computed from
          ``g(x) - x``, not from the damped step, which is ``alpha`` times
          smaller and would flatter a loose solve by exactly that factor;
        * the tolerance handed to optimistix is scaled by ``alpha``, because
          optimistix stops on the *step* it takes.  Without the scaling a
          damped solve would stop ``1/alpha`` short of the tolerance it was
          asked for, and then the verdict above would -- correctly -- call it
          non-converged.

        ``damping`` is validated in :meth:`solve`.
        """
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

        if damping == 1.0:
            # Not just an optimisation: `x + 1.0 * (g(x) - x)` is g(x) in
            # exact arithmetic and not quite in floating point, and the
            # undamped path is the one every existing result was computed on.
            iteration = flowsheet_iteration
        else:
            def iteration(tear_arr, args):
                """The damped map, which shares g's fixed point."""
                return tear_arr + damping * (
                    flowsheet_iteration(tear_arr, args) - tear_arr
                )

        # Solve fixed-point problem.  See the docstring for why the
        # tolerance carries the damping factor.
        fp_solver = optx.FixedPointIteration(
            rtol=tol * damping, atol=tol * damping
        )
        sol = optx.fixed_point(
            iteration,
            fp_solver,
            tear_array,
            args=args,
            max_steps=max_iter,
            throw=False,
        )
        tear_converged = sol.value

        # Record convergence diagnostics for the report layer.  The verdict
        # is judged against the criterion optimistix ACTUALLY stopped on --
        # elementwise |dx| < atol + rtol |x| on the step it took, with
        # tol * damping serving as both, which is |g(x) - x| < tol (1 + |x|)
        # once the damping cancels -- not against a bare |dx| < tol.  The two
        # differ by the magnitude of
        # the tear: with flows of order 1 and tol=1e-8, a solve optimistix
        # calls successful lands near 2e-8, and calling that non-converged
        # would warn on almost every damped solve.  The residual reported
        # alongside stays the plain max-norm, which is the number to compare
        # against tol by eye.
        delta = flowsheet_iteration(tear_converged, args) - tear_converged
        final_residual = _concrete(jnp.max(jnp.abs(delta)))
        scaled = _concrete(
            jnp.max(jnp.abs(delta) / (tol + tol * jnp.abs(tear_converged)))
        )
        self.last_solve_residual = final_residual
        self.last_solve_converged = None if scaled is None else bool(scaled < 1.0)
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
        # Pessimistic defaults, so that a loop that never reaches its
        # convergence test (max_iter=0) reports "did not converge" rather
        # than whatever the previous solve left behind.
        self.last_solve_iterations = max_iter
        self.last_solve_residual = None
        self.last_solve_converged = False
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
            if res is None:
                # A traced residual cannot be compared, so the loop simply
                # runs to max_iter.  There is no verdict to report either:
                # None says "unavailable", which is what keeps the
                # non-convergence warning quiet under jax.grad / jit.
                self.last_solve_converged = None
            elif res < tol:
                self.last_solve_iterations = iteration
                self.last_solve_converged = True
                break
            else:
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

        # Pessimistic defaults; see _solve_with_wegstein.
        self.last_solve_iterations = max_iter
        self.last_solve_residual = None
        self.last_solve_converged = False
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
            if res is None:
                # See _solve_with_wegstein: no concrete residual, no verdict.
                self.last_solve_converged = None
            elif res < tol:
                self.last_solve_iterations = iteration
                self.last_solve_converged = True
                break
            else:
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
            streams.update(_parse_outlets(result, unit))

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
