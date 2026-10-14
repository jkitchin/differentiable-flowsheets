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
    AcceleratedFixedPoint,
    wegstein_acceleration,
    Initializable,
    TearInitializationWarning,
    FlowsheetGraph,
    TearAnalysis,
    analyze_tears,
    calculation_order,
    calculation_order_problems,
    select_tear_streams,
)
import optimistix as optx


FEED_PREFIX = "feed:"

#: How many times ``solve(tol_basis="error")`` may tighten its step test
#: before giving up and reporting what it reached.  Each tightening is a
#: fresh solve warm-started from the last one, so the cost is bounded; three
#: is enough for the gain estimate to be wrong by a decade twice over.
_MAX_TOLERANCE_TIGHTENINGS = 3

#: How much bigger than the reported step the measured error has to be
#: before :class:`TearToleranceWarning` fires under the default
#: ``tol_basis="step"``.  Ten is a loop gain of 0.9 --- the high-gain
#: regime the warning is for.  See ``Flowsheet._report_tolerance`` for why
#: a bare ``error > tol`` is the wrong test.
_TOLERANCE_WARNING_AMPLIFICATION = 10.0


class _DefaultAcceleration(str):
    """The ``acceleration`` default: ``"anderson"``, but knows nobody asked.

    Eagerly it is ``"anderson"``.  Under tracing a *requested* acceleration
    runs the traced Anderson/Wegstein solver (#440), while the default keeps
    the plain optimistix fixed point that ``solve`` has always used there, so
    a flowsheet that never named an accelerator keeps its traced answers.
    """


_DEFAULT_ACCELERATION = _DefaultAcceleration("anderson")


class _UndetectedTracer(Exception):
    """Raised by an accelerated tear loop whose residual turned out traced."""


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


class TearToleranceWarning(UserWarning):
    """A recycle solve met its step test and is still outside its tolerance.

    ``tol`` tests the max-norm of ``g(x) - x``, the *step* between
    successive tear iterates.  On a loop of gain ``g`` a step of ``d``
    leaves an error of about ``d / (1 - g)``, so at a gain of 0.97 a solve
    that stops inside ``1e-8`` is about ``3e-7`` out --- and reports
    convergence, meaning it (#264).  The solver is right on its own terms
    and wrong on the caller's, and nothing in the result said so: the gain
    was not reported anywhere and ``last_solve_residual`` is the step, not
    the error.

    This warning fires when the error :meth:`Flowsheet.solve` measures
    after converging exceeds the tolerance that was asked for.  The
    measurement is in :attr:`Flowsheet.last_solve_error_estimate` and the
    gain behind it in :attr:`Flowsheet.last_solve_gain`.

    Silence it with ``solve(error_probe=0)``, which also skips the
    measurement, or act on it with ``solve(tol_basis="error")``, which
    keeps tightening the step test until the measured error is inside
    ``tol``.
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


_STREAM_SCALAR_KEYS = frozenset({"T", "P", "phase"})


def _is_stream(x: Any) -> bool:
    """Whether a value a unit returned is a stream rather than an info dict.

    A :data:`~difflow.streams.Stream` is itself a ``dict``, so
    ``isinstance(x, dict)`` cannot tell an outlet from the info payload
    that rides beside it: a unit returning ``(S, S)`` with no info had
    its second stream taken for info whenever the outlet count was off
    (audit of the outlet parsing). A stream is a dict carrying ``T`` and
    ``P`` and nothing else but ``F_*`` flows and an optional ``phase``
    label; an info dict that happens to report a ``T`` is still not one.

    Args:
        x: Any value.

    Returns:
        True when ``x`` is a stream.

    Example:
        >>> _is_stream({"F_A": 1.0, "T": 300.0, "P": 1e5})
        True
        >>> _is_stream({"T": 300.0, "P": 1e5, "duty": 2.0})
        False
    """
    if not isinstance(x, dict) or "T" not in x or "P" not in x:
        return False
    return all(
        isinstance(k, str) and (k in _STREAM_SCALAR_KEYS or k.startswith("F_"))
        for k in x
    )


def _is_info(x: Any) -> bool:
    """A mapping that is not a stream: a unit's info payload."""
    return isinstance(x, dict) and not _is_stream(x)


def _split_result(
    result: Any, outlet_names: list[str], unit_name: str = "unit"
) -> tuple[tuple, dict | None]:
    """Split a unit's return value into its streams and its info dict.

    Every place that reads a unit's result goes through here (the
    sequential solve, the optimistix fixed-point path, the info
    recorder, the equation-oriented solver), so they cannot disagree
    about a shape; there used to be four copies, and they did. The
    shapes are a single stream, ``(S1, ..., Sn)``,
    ``(S1, ..., Sn, info)``, and ``((S1, ..., Sn), info)``.

    The nested shape is recognised FIRST. Checking ``len(result) ==
    len(outlet_names)`` before it handed a two-outlet Ultrafiltration
    its ``(retentate, permeate)`` tuple as the first outlet and its info
    dict as the second, and a one-outlet wiring silently dropped a
    stream.

    Args:
        result: What the unit's ``__call__`` returned.
        outlet_names: The unit's outlet stream names, in order.
        unit_name: Used in the error message.

    Returns:
        ``(streams, info)``: the streams in ``__call__`` order and the
        info dict, or None when the unit returned none.

    Raises:
        ValueError: A non-stream mapping (a results dict) or a non-dict
            sits where a stream belongs, or the unit returned a different
            number of streams than it has outlets (a stream left over
            used to be dropped without a word).

    Example:
        >>> s = {"F_A": 1.0, "T": 300.0, "P": 1e5}
        >>> _split_result(((s, s), {"k": 1}), ["a", "b"])[1]
        {'k': 1}
    """
    if not isinstance(result, tuple):
        streams, info = (result,), None
    elif (len(result) == 2 and isinstance(result[0], tuple)
          and _is_info(result[1])):
        streams, info = result[0], result[1]
    elif len(result) >= 2 and _is_info(result[-1]):
        streams, info = result[:-1], result[-1]
    else:
        streams, info = result, None

    for item in streams:
        if not _is_stream(item):
            if isinstance(item, dict):
                what = f"a mapping with keys {sorted(map(str, item))[:8]}"
                hint = _RESULTS_HINT
            else:
                what = f"a {type(item).__name__}"
                hint = ""
            raise ValueError(
                f"Unit {unit_name!r} returned {what} where an outlet stream "
                "belongs. A unit operation in a flowsheet must return "
                "streams (dicts of T, P and F_* flows), optionally followed "
                "by an info dict." + hint
            )
    # An info outlet: a unit wired with one outlet more than it returns
    # streams, and an info dict to put there, carries its info on that
    # last outlet so the solve hands it back with the streams. The
    # refinery alkylation unit and its columns are wired this way
    # (``["effluent", "reactor_info"]``) and read ``streams["<name>_info"]``
    # after the solve; the old zip-by-length parser allowed it, and the
    # strict count below must not take it away.
    if info is not None and len(streams) + 1 == len(outlet_names):
        return tuple(streams) + (info,), info
    if len(streams) != len(outlet_names):
        raise ValueError(
            f"Unit {unit_name!r} has {len(outlet_names)} outlets "
            f"{list(outlet_names)} but returned {len(streams)} stream(s)."
        )
    return tuple(streams), info


#: the hint added when a unit hands back a results dict instead of streams
_RESULTS_HINT = (
    " A results dict is not a stream: wire a stream-returning wrapper "
    "instead (for the REE circuits, the palette entries ExtractStripUnit, "
    "ExtractScrubStripUnit, SplitShellUnit and SeparationTrainUnit in "
    "difflow_ree, or the modules in difflow_ree.flowsheets.modules)."
)


def _parse_outlets(result: Any, unit: "Unit") -> dict[str, Stream]:
    """Name the streams a unit returned.

    A unit operation may hand back a single stream, a tuple of streams, a
    tuple of streams with an info dict on the end, or ``(streams, info)``
    (see :func:`_split_result`). The mapping onto names is positional in
    every one of those shapes, which is what makes the order of
    ``outlet_names`` part of a unit's contract rather than a detail of
    how it was written down.
    """
    streams, _ = _split_result(result, unit.outlet_names, unit.name)
    return dict(zip(unit.outlet_names, streams))


def _unit_info(result: Any, unit: "Unit") -> dict | None:
    """The info dict in a unit's result, read by the same
    :func:`_split_result` that :func:`_parse_outlets` reads the streams
    with, so the two cannot disagree about which entry is which."""
    return _split_result(result, unit.outlet_names, unit.name)[1]


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


@dataclass(eq=False)
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


@dataclass
class DegreesOfFreedom:
    """Degree-of-freedom count of a flowsheet (:meth:`Flowsheet.degrees_of_freedom`).

    Every stream carries ``n_species + 2`` variables (the species flows, T
    and P). The feeds are the specified streams; every unit contributes the
    equations that fix its outlets from its inlets (its ``eo_residuals`` when
    it has them, else one per outlet variable); a recycle whose source and
    destination streams are named differently adds the ``n_species + 2``
    equations ``dest = source``. Unit *parameters* (a conversion, a split
    fraction, a heat duty, ...) are the design specifications that make a
    unit's equations determinate; they are listed in ``unit_parameters`` and
    are not counted as variables.

    ``degrees_of_freedom = n_variables - n_specified - n_equations`` is 0 for
    a well-posed flowsheet. Positive means variables that nothing fixes (an
    inlet that is neither a feed, a unit outlet nor a recycle destination);
    negative means streams fixed twice (a stream produced by two units, or a
    feed that a unit also produces). ``problems`` names the culprits.

    Attributes:
        n_per_stream: Variables per stream, ``n_species + 2``.
        streams: Every stream name in the flowsheet.
        n_variables: ``len(streams) * n_per_stream``.
        specified: The feed stream names.
        n_specified: ``len(specified) * n_per_stream``.
        n_equations: Unit equations plus recycle connection equations.
        unit_equations: Equations contributed by each unit, by unit name.
        recycle_equations: Connection equations of each recycle, keyed by
            ``"source -> dest"``.
        unit_parameters: The parameter (design-specification) names of each
            unit, by unit name.
        unknowns: The unspecified streams (everything but the feeds); the
            equation-oriented solver's unknowns.
        degrees_of_freedom: ``n_variables - n_specified - n_equations``.
        problems: Human-readable reasons the count is not zero.
    """

    n_per_stream: int
    streams: list[str]
    n_variables: int
    specified: list[str]
    n_specified: int
    n_equations: int
    unit_equations: dict[str, int]
    recycle_equations: dict[str, int]
    unit_parameters: dict[str, list[str]]
    unknowns: list[str]
    degrees_of_freedom: int
    problems: list[str] = field(default_factory=list)

    @property
    def is_well_posed(self) -> bool:
        """True when the count is zero and no problem was found."""
        return self.degrees_of_freedom == 0 and not self.problems

    def __str__(self) -> str:
        lines = [
            f"Degrees of freedom: {self.degrees_of_freedom}",
            f"  variables:  {self.n_variables} "
            f"({len(self.streams)} streams x {self.n_per_stream})",
            f"  specified:  {self.n_specified} (feeds: {', '.join(self.specified) or 'none'})",
            f"  equations:  {self.n_equations}",
        ]
        for name, n in self.unit_equations.items():
            params = self.unit_parameters.get(name) or []
            spec = f"  [specs: {', '.join(params)}]" if params else ""
            lines.append(f"    {name}: {n}{spec}")
        for name, n in self.recycle_equations.items():
            lines.append(f"    recycle {name}: {n}")
        for problem in self.problems:
            lines.append(f"  problem: {problem}")
        return "\n".join(lines)


def _unit_parameter_names(unit: "Unit") -> list[str]:
    """The design-specification names a unit carries: the fields of its
    ``params`` dataclass, if it has one, and its call-time keyword params."""
    names: list[str] = []
    params = getattr(unit.operation, "params", None)
    if is_dataclass(params) and not isinstance(params, type):
        names.extend(f.name for f in dc_fields(params))
    names.extend(k for k in unit.params if k not in names)
    return names


def _unit_equation_count(unit: "Unit", n_per_stream: int, zero: Stream) -> int:
    """Equations a unit contributes: the length of its ``eo_residuals`` when
    it has them (shape-only evaluation, nothing is run), else one per outlet
    variable."""
    fallback = len(unit.outlet_names) * n_per_stream
    op = unit.operation
    if not hasattr(op, "eo_residuals"):
        return fallback
    try:
        shape = jax.eval_shape(
            lambda ins, outs: op.eo_residuals(ins, outs, **unit.params),
            [zero for _ in unit.inlet_names],
            [zero for _ in unit.outlet_names],
        )
        return int(jnp.size(jnp.zeros(shape.shape)))
    except Exception:
        return fallback


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
        default_T: float | None = None,
        default_P: float | None = None,
    ):
        """Initialize empty flowsheet.

        Args:
            species_order: List of species names for stream arrays
            default_flow: Default molar flow (mol/s) for tear stream
                initialization when no explicit initial guess is provided.
                Use a smaller value (e.g. 1e-6) for trace species.
            default_T: Temperature (K) of a tear stream's starting guess.
                ``None`` (the default) *infers* it: a tear starts at the
                flow-weighted temperature of the other streams entering the
                unit it feeds (the stream "feeding the loop"), and at
                300 K only when that unit has no other known inlet. A
                number pins the guess to it.
            default_P: Pressure (Pa) of a tear stream's starting guess.
                ``None`` (the default) infers it as the *highest* pressure
                of the other streams entering the unit the tear feeds,
                falling back to 101325 Pa. A number pins the guess.

                Why inferred, and why the highest: a :class:`Mixer` passes
                on the lowest inlet pressure, so a loop with no pressure
                change inside it has *every* pressure at or below its feed
                as a fixed point, and a 1 atm guess used to be one of them
                --- a 50 bar loop reported 1 bar, and ``jax.grad`` through
                it failed on a singular tear Jacobian (pressure had a loop
                gain of exactly one). Starting at or above the feed lands
                on the physical fixed point, where the pressure is pinned
                to the feed.

        The attributes :attr:`infer_tear_T` / :attr:`infer_tear_P` record
        which of the two is inferred; assigning a number to
        ``default_T``/``default_P`` after construction pins it (and assigning
        ``None`` goes back to inferring).
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
        #: what accelerated the last solve: "anderson", "wegstein" or "none".
        #: Differs from ``last_solve_method`` under tracing, where the method
        #: is "fixed_point (traced)" and this says whether the traced solver
        #: was accelerated or fell back to plain substitution (#440).
        self.last_solve_acceleration: str | None = None
        #: tolerance requested of the last recycle solve
        self.last_solve_tol: float | None = None
        #: tear-stream (recycle destination) names of the last recycle solve
        self.last_solve_tear_streams: list[str] = []
        #: signed dominant contraction ratio of the tear map, measured at
        #: the last solution, for every acceleration method.  Measured from
        #: the probe's residual ratio, or -- when the residual is already at
        #: round-off, as after a typical Anderson solve -- as the dominant
        #: eigenvalue of the tear map's Jacobian (``jax.jacfwd``).  ``None``
        #: when it was not measured: no recycles, a solve that did not
        #: converge, ``error_probe=0``, a traced solve, or a round-off
        #: residual on a loop too large (> 400 tear entries) or not
        #: forward-differentiable.
        self.last_solve_gain: float | None = None
        #: estimated error remaining in the tear at the last solution, in
        #: the same max-norm as ``last_solve_residual``.  This is the
        #: number ``tol`` reads as if it were, and on a high-gain loop the
        #: two differ by ``1 / (1 - gain)`` (#264).
        self.last_solve_error_estimate: float | None = None
        #: iterations of the last solve on which ``clip_negative_flows``
        #: actually moved the proposed iterate.  Zero on the unaccelerated
        #: path, which does not clip at all (#263), and zero on an
        #: accelerated solve whose iterates all stayed non-negative.
        self.last_solve_clip_active: int = 0
        #: the tear residual (max abs change) after each iteration of the
        #: last solve, in order, across any tol_basis="error" retries.
        #: ``None`` when the iteration was optimistix's fixed point
        #: (``acceleration="none"`` or a traced solve), which keeps no
        #: history; empty for a recycle-free solve.
        self.last_solve_history: list[float] | None = None
        #: unit name -> the info dict its operation returned on the last
        #: concrete evaluation of the last solve (the error probe's pass at
        #: the solution, or the final iteration). A unit returning no info
        #: is absent; nothing is recorded while tracing.
        self.last_solve_unit_info: dict[str, dict] = {}

    @property
    def default_T(self) -> float:
        """Fallback tear temperature (K); see ``__init__``."""
        return self._default_T

    @default_T.setter
    def default_T(self, value: float | None) -> None:
        #: True when the tear guess's T is inferred from the unit it feeds
        self.infer_tear_T = value is None
        self._default_T = 300.0 if value is None else value

    @property
    def default_P(self) -> float:
        """Fallback tear pressure (Pa); see ``__init__``."""
        return self._default_P

    @default_P.setter
    def default_P(self, value: float | None) -> None:
        #: True when the tear guess's P is inferred from the unit it feeds
        self.infer_tear_P = value is None
        self._default_P = 101325.0 if value is None else value

    def _default_kwargs(self) -> dict[str, Any]:
        """Constructor keywords that reproduce this flowsheet's tear defaults.

        A copy made with ``default_T=self.default_T`` would *pin* a
        temperature this flowsheet infers; ``None`` is what keeps it
        inferred.
        """
        return {
            "default_flow": self.default_flow,
            "default_T": None if self.infer_tear_T else self.default_T,
            "default_P": None if self.infer_tear_P else self.default_P,
        }

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
        acceleration: Literal["none", "wegstein", "anderson"] = _DEFAULT_ACCELERATION,
        anderson_depth: int = 5,
        use_initialization: bool = True,
        clip_negative_flows: bool = True,
        on_nonconvergence: Literal["warn", "raise", "ignore"] = "warn",
        tears: Literal["declared", "auto", "heuristic", "minimum"] = "declared",
        error_probe: int = 2,
        tol_basis: Literal["step", "error"] = "step",
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

                Under ``jax.jit``/``jax.grad`` a method you NAME runs as a
                traced solver (``AcceleratedFixedPoint`` inside
                ``optimistix.fixed_point``, implicit derivative); the
                unnamed default uses plain substitution there.
                ``last_solve_acceleration`` records what ran.
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

                **It binds on the accelerated paths only** (#263), and
                that is deliberate rather than an oversight. The
                projection is a safeguard on *extrapolation*: Wegstein
                and Anderson each propose an iterate no unit computed,
                and that guess can leave the physical orthant on the way
                to a solution entirely inside it. ``acceleration="none"``
                extrapolates nothing --- its iterate is
                ``x + damping (g(x) - x)``, which for the usual
                ``damping <= 1`` cannot leave a region both ends are in
                --- so there is no guess to safeguard, and clipping there
                would clip ``g`` itself. That is a different act with two
                costs: it invents fixed points the model does not have,
                and it sits inside the map ``optx.fixed_point``
                implicitly differentiates. Wherever a tear flow converges
                to exactly zero --- an ordinary species absent from a
                recycle --- ``jnp.maximum(x, 0)`` contributes a
                derivative of one half at the kink, and the gradient
                comes back wrong while the forward pass stays right.

                Under ``jax.grad`` or ``jit`` the clip applies only when an
                acceleration was named (the traced Anderson/Wegstein
                solver, #440); the unnamed default falls back to ``"none"``,
                which never clips.

                When the projection actually *moves* an iterate it is
                counted in :attr:`last_solve_clip_active`, and a solve
                that then fails to converge says so in its warning ---
                a clipped signed tear is the one failure mode this flag
                causes, and it used to be silent.
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
                when ``solve`` runs (:func:`_concrete` returns ``None``, and
                ``last_solve_converged`` stays ``None``), so the check is
                deferred to run time with ``jax.debug.callback``: an
                unconverged traced solve emits the same
                :class:`ConvergenceWarning` when the computation runs, for
                ``"warn"`` and ``"raise"`` alike (an exception inside a
                gradient would make the flowsheet undifferentiable; escalate
                with ``warnings.simplefilter("error", ConvergenceWarning)``
                if that is wanted), and nothing for ``"ignore"``. This
                matters more there, not less: the gradient comes from
                implicit differentiation at the returned point and is wrong
                if it is not a fixed point.
            tears: Where the tear streams come from.

                - ``"declared"`` (default): the recycles given to
                  :meth:`add_recycle`, and nothing else. A loop closed in
                  the topology but never declared stays untorn, which is
                  what it has always done.
                - ``"auto"``: when --- and only when --- no recycle has
                  been declared, pick the tear streams with
                  :func:`~difflow.select_tear_streams`, sequence the units
                  around them with
                  :func:`~difflow.calculation_order`, and solve on those.
                  The choice is recorded in ``last_solve_tear_streams``
                  and neither it nor the sequence is kept:
                  ``self.recycles`` and ``self.units`` are what was
                  declared, before and after.
                - ``"heuristic"`` / ``"minimum"``: the same, naming the
                  selection strategy explicitly (``"auto"`` is
                  ``"heuristic"``).

                A declared recycle always wins: an automatic choice is
                there to fill a gap, not to overrule one. Use
                :meth:`tear_analysis` to see what the strategies would
                pick without solving anything.
            error_probe: Extra plain-substitution passes to spend, after a
                solve converges, measuring the error that ``tol`` does not
                (#264). ``tol`` tests the max-norm of ``g(x) - x``, a
                *step*; the error is ``(I - M)^-1 (g(x) - x)``, and on a
                loop of gain 0.97 those differ by a factor of 33. Each pass
                measures one more term of that inverse, and the result is
                reported in :attr:`last_solve_gain` and
                :attr:`last_solve_error_estimate`. Two passes --- the
                default --- is enough for a loop with one dominant mode,
                which is the usual case; more refines it. ``0`` skips the
                measurement entirely, along with
                :class:`TearToleranceWarning`.

                The probe never runs on a solve that did not converge (it
                has a warning of its own, and no solution to extrapolate
                from) or under ``jax.grad``/``jit``.
            tol_basis: What ``tol`` tests.

                - ``"step"`` (default): the max-norm of ``g(x) - x``, which
                  is what it has always tested. A high-gain loop is
                  accepted well outside it, and the measurement above is
                  what now says so.
                - ``"error"``: the *measured* error. When the probe says
                  the answer is outside ``tol``, ``solve`` tightens its
                  step test by the factor just measured and solves again,
                  warm-started, up to three times. This is the honest
                  reading of ``tol`` and it costs iterations --- a loop
                  that converges today needs more of them to satisfy it,
                  which is why it is opt-in.

        Returns:
            Dictionary of all streams in the flowsheet

        Raises:
            ConvergenceError: If the solve did not converge and
                ``on_nonconvergence="raise"``.
            ValueError: If ``on_nonconvergence`` is not one of the three
                recognised values, ``acceleration`` or ``tol_basis`` is
                unknown, ``damping`` is not positive, or ``tol_basis`` is
                ``"error"`` with no ``error_probe`` to measure it.
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
        if tears not in ("declared", "auto", "heuristic", "minimum"):
            raise ValueError(
                f"Unknown tears: {tears!r}. "
                'Expected "declared", "auto", "heuristic" or "minimum".'
            )
        if tol_basis not in ("step", "error"):
            raise ValueError(
                f"Unknown tol_basis: {tol_basis!r}. Expected \"step\" (tol "
                'tests |g(x) - x|, the historical meaning) or "error" (tol '
                "tests the measured error, which needs error_probe > 0)."
            )
        if tol_basis == "error" and error_probe < 1:
            raise ValueError(
                'tol_basis="error" needs error_probe > 0: the error is '
                "measured with extra substitution passes, and with none of "
                "them there is nothing to test against."
            )

        self.last_solve_history = []
        self.last_solve_unit_info = {}

        if tears != "declared" and not self.recycles:
            return self._solve_auto_torn(
                "heuristic" if tears == "auto" else tears,
                tear_initial=tear_initial,
                tol=tol,
                max_iter=max_iter,
                damping=damping,
                acceleration=acceleration,
                anderson_depth=anderson_depth,
                use_initialization=use_initialization,
                clip_negative_flows=clip_negative_flows,
                on_nonconvergence=on_nonconvergence,
                error_probe=error_probe,
                tol_basis=tol_basis,
            )

        if not self.recycles:
            # No recycles - simple sequential solution
            self.last_solve_method = "direct"
            self.last_solve_acceleration = "none"
            self.last_solve_iterations = 0
            self.last_solve_residual = 0.0
            self.last_solve_converged = True
            self.last_solve_tol = tol
            self.last_solve_tear_streams = []
            self.last_solve_gain = None
            self.last_solve_error_estimate = 0.0
            self.last_solve_clip_active = 0
            return self._solve_sequential()

        self.last_solve_method = acceleration
        self.last_solve_tol = tol
        self.last_solve_tear_streams = [dest for dest in self.recycles.values()]
        self.last_solve_clip_active = 0

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
        upstream = None
        guess_tears = use_initialization and not self._is_traced()
        for source, dest in self.recycles.items():
            if tear_initial and dest in tear_initial:
                tear_streams[dest] = tear_initial[dest]
            elif guess_tears:
                if propagated is None:
                    propagated = self._propagate_from_feeds()
                tear_streams[dest] = self._initialize_tear_stream(
                    source, propagated, dest=dest
                )
            else:
                # Small flows, at the T and P of the streams the tear
                # joins (when inferred).  Under tracing that needs only
                # the units the loop does not feed back into, which run on
                # the same inputs in every iteration anyway.
                if upstream is None:
                    upstream = self._upstream_of_tears()
                tear_streams[dest] = self._tear_default_stream(dest, upstream)

        if acceleration not in ("none", "wegstein", "anderson"):
            raise ValueError(f"Unknown acceleration method: {acceleration}")

        # The eager accelerated solvers are Python loops that branch on the
        # residual to stop early, which a tracer cannot answer.  Under
        # jax.grad / jit the solve goes through optimistix's fixed point
        # instead: it converges without a Python branch and carries an
        # implicit-differentiation rule, so the gradient comes from the
        # converged solution rather than from an unrolled iteration.
        # ``last_solve_method`` says "fixed_point (traced)" for it, and
        # ``last_solve_acceleration`` says what ran inside it.
        #
        # An acceleration the caller NAMED runs there too, as a traced
        # Anderson/Wegstein solver (#440).  The unnamed default keeps the
        # plain substitution it has always used under tracing: it is the
        # same "anderson" eagerly, but nobody asked for it, and changing
        # what every jitted default solve computes is not this change.
        traced_accel = None
        explicit = type(acceleration) is not _DefaultAcceleration
        self.last_solve_acceleration = str(acceleration)
        if acceleration != "none" and self._is_traced():
            if explicit:
                traced_accel = str(acceleration)
            acceleration = "none"
            self.last_solve_method = "fixed_point (traced)"
            self.last_solve_acceleration = traced_accel or "none"

        tear_names = list(self.recycles.values())
        self.last_solve_gain = None
        self.last_solve_error_estimate = None

        # Solve, measure, and under tol_basis="error" solve again against a
        # tighter step test.  One pass through this loop is the historical
        # behaviour with a measurement bolted on the end; the retries only
        # happen when the caller asked for tol to mean the error and the
        # measurement says it does not yet.  Every path records its verdict
        # in last_solve_converged, so the non-convergence check is done once
        # after the loop rather than at each of the three returns.
        tol_step = tol
        iterations = 0
        requested = acceleration
        for attempt in range(_MAX_TOLERANCE_TIGHTENINGS + 1):
            if acceleration == "none":
                # No clip_negative_flows here, and not by omission (#263).
                # The projection safeguards an *extrapolated* guess; this
                # path extrapolates nothing, so clipping it would clip g
                # itself -- inventing fixed points, and putting a kink
                # inside the map that optx.fixed_point implicitly
                # differentiates.  The argument, and the gradient it
                # breaks, are in solve's docstring.
                self.last_solve_history = None    # optx keeps no history
                streams = self._solve_with_recycle_damped(
                    tear_streams, tol_step, max_iter, damping,
                    accelerate=traced_accel, anderson_depth=anderson_depth,
                    clip_negative_flows=clip_negative_flows,
                    on_nonconvergence=on_nonconvergence,
                )
            else:
                try:
                    if acceleration == "wegstein":
                        streams = self._solve_with_wegstein(
                            tear_streams, tol_step, max_iter,
                            clip_negative_flows=clip_negative_flows,
                        )
                    else:
                        streams = self._solve_with_anderson(
                            tear_streams, tol_step, max_iter, anderson_depth,
                            clip_negative_flows=clip_negative_flows,
                        )
                except _UndetectedTracer:
                    # Same fallback as the _is_traced branch above, for a
                    # tracer that only showed up once the units ran.
                    acceleration = "none"
                    self.last_solve_method = "fixed_point (traced)"
                    self.last_solve_history = None
                    if explicit:
                        traced_accel = str(requested)
                    self.last_solve_acceleration = traced_accel or "none"
                    streams = self._solve_with_recycle_damped(
                        tear_streams, tol_step, max_iter, damping,
                        accelerate=traced_accel, anderson_depth=anderson_depth,
                        clip_negative_flows=clip_negative_flows,
                        on_nonconvergence=on_nonconvergence,
                    )

            # The tolerance reported is the one the caller asked for, not
            # the internal step test a retry is running against.
            self.last_solve_tol = tol
            iterations += self.last_solve_iterations or 0
            self.last_solve_iterations = iterations

            if error_probe < 1 or not self.last_solve_converged:
                # A failed solve has a warning of its own and nothing to
                # extrapolate from; probing it would spend flowsheet passes
                # to describe a point that is not a solution.
                break

            gain, error = self._estimate_tear_error(
                streams, tear_names, error_probe
            )
            self.last_solve_gain = gain
            self.last_solve_error_estimate = error

            if (tol_basis != "error" or error is None or error <= tol
                    or attempt == _MAX_TOLERANCE_TIGHTENINGS):
                break

            # Aim the next step test at the tolerance the caller asked for:
            # the error runs about 1/(1 - gain) times the step, so scaling
            # the step test by the ratio just measured lands on it.  The
            # floor keeps a gain of 0.9999 from asking for a step below the
            # arithmetic, which no iteration could ever meet.
            tol_step = max(tol_step * (tol / error), tol * 1e-8)
            tear_streams = {name: streams[name] for name in tear_names}

        self._report_nonconvergence(on_nonconvergence, max_iter)
        self._report_tolerance(tol, tol_basis)
        return streams

    def _solve_auto_torn(self, method: str, **solve_kwargs) -> dict[str, Stream]:
        """Solve on tear streams this flowsheet never declared.

        The selected streams are installed as recycles of themselves ---
        ``add_recycle(s, s)``, which is how a loop closed by a single
        stream name is torn: the iteration seeds ``s``, the units run, and
        the unit that computes ``s`` overwrites it with the value the next
        iterate is compared against.  They are installed for this solve
        only; ``self.recycles`` is what the user declared, before and
        after, so a later ``solve()`` still does what it always did.
        """
        chosen = select_tear_streams(self, method=method)
        if not chosen:
            # Nothing to tear.  Either the flowsheet is acyclic, and the
            # sequential path is right, or the loop is closed by something
            # the graph cannot see -- and then the declared path saying
            # "no recycles" is still the honest answer.
            return self.solve(tears="declared", **solve_kwargs)

        graph = FlowsheetGraph.from_flowsheet(self)
        missing, _ = calculation_order_problems(graph, chosen, self.feeds)
        if missing:
            raise ValueError(
                "Automatic tear selection cannot solve this flowsheet: no "
                "unit computes and no feed supplies "
                f"{', '.join(missing)}, so neither a tear set nor a "
                "calculation order can fill it in. Add the missing feed with "
                "add_feed(), or check the stream names. "
                "Flowsheet.tear_analysis() reports this alongside the loops "
                "it found."
            )

        order = calculation_order(graph, chosen)
        if order is None:
            # Defensive: every strategy tears at least one edge of every
            # cycle it was given, so what is left is a DAG.  A cycle that
            # survives means the selection missed a loop (a truncated
            # enumeration, say), and running anyway would KeyError deep in
            # the iteration instead of saying so.
            raise ValueError(
                "Automatic tear selection left a recycle loop unbroken, so "
                f"there is no calculation order for the tears it chose "
                f"({', '.join(chosen)}). See Flowsheet.tear_analysis(), and "
                "declare the recycles explicitly with add_recycle()."
            )

        by_name = {unit.name: unit for unit in self.units}
        declared, sequence = self.recycles, self.units
        self.recycles = {name: name for name in chosen}
        self.units = [by_name[name] for name in order]
        try:
            return self.solve(tears="declared", **solve_kwargs)
        finally:
            self.recycles, self.units = declared, sequence

    def degrees_of_freedom(self) -> DegreesOfFreedom:
        """Count the flowsheet's variables, specifications and equations.

        Topology only: nothing is solved. See :class:`DegreesOfFreedom` for
        the accounting. A flowsheet built from difflow units, with every
        unit inlet supplied by a feed, another unit or a recycle, has zero
        degrees of freedom: each unit's parameters are its specifications.

        Returns:
            :class:`~difflow.DegreesOfFreedom`; ``print()`` it for a summary.

        Example:
            For a mixer -> reactor -> splitter loop over three species
            (6 streams, one feed, the recycle torn as ``gas -> recycle``):

            >>> dof = fs.degrees_of_freedom()
            >>> dof.n_variables, dof.n_specified, dof.n_equations
            (30, 5, 25)
            >>> dof.degrees_of_freedom
            0
        """
        nps = len(self.species_order) + 2
        streams: list[str] = list(self.feeds)
        for unit in self.units:
            for name in (*unit.inlet_names, *unit.outlet_names):
                if name not in streams:
                    streams.append(name)
        for source, dest in self.recycles.items():
            for name in (source, dest):
                if name not in streams:
                    streams.append(name)

        zero = self._make_zero_stream()
        unit_equations = {
            u.name: _unit_equation_count(u, nps, zero) for u in self.units
        }
        unit_parameters = {u.name: _unit_parameter_names(u) for u in self.units}
        recycle_equations = {
            f"{source} -> {dest}": nps
            for source, dest in self.recycles.items() if source != dest
        }

        producers: dict[str, list[str]] = {}
        for unit in self.units:
            for name in unit.outlet_names:
                producers.setdefault(name, []).append(unit.name)
        problems = []
        for name in streams:
            made_by = producers.get(name, [])
            if len(made_by) > 1:
                problems.append(
                    f"stream {name!r} is produced by {len(made_by)} units "
                    f"({', '.join(made_by)}): over-specified by {nps * (len(made_by) - 1)}"
                )
            if name in self.feeds and made_by:
                problems.append(
                    f"stream {name!r} is a feed and is also produced by "
                    f"{', '.join(made_by)}: over-specified by {nps}"
                )
            fed_by_recycle = any(
                d == name and src != name for src, d in self.recycles.items()
            )
            if name not in self.feeds and not made_by and not fed_by_recycle:
                problems.append(
                    f"stream {name!r} is not a feed, a unit outlet or a "
                    f"recycle destination: {nps} unspecified variables"
                )
        for source, dest in self.recycles.items():
            if source not in producers:
                problems.append(f"recycle source {source!r} is produced by no unit")

        n_variables = len(streams) * nps
        n_specified = len(self.feeds) * nps
        n_equations = sum(unit_equations.values()) + sum(recycle_equations.values())
        return DegreesOfFreedom(
            n_per_stream=nps,
            streams=streams,
            n_variables=n_variables,
            specified=list(self.feeds),
            n_specified=n_specified,
            n_equations=n_equations,
            unit_equations=unit_equations,
            recycle_equations=recycle_equations,
            unit_parameters=unit_parameters,
            unknowns=[n for n in streams if n not in self.feeds],
            degrees_of_freedom=n_variables - n_specified - n_equations,
            problems=problems,
        )

    def tear_analysis(self, max_cycles: int = 1000) -> TearAnalysis:
        """Report this flowsheet's recycle loops and where they could be torn.

        Diagnosis only: it reads the topology, runs no unit and changes
        nothing.  The result names the loops it found, the tears already
        declared with :meth:`add_recycle`, what
        :func:`~difflow.select_tear_streams` would have chosen instead,
        any loop left untorn, and any inlet the declared unit order cannot
        supply.

        Args:
            max_cycles: Give up enumerating elementary cycles after this
                many, with a
                :class:`~difflow.CycleEnumerationWarning`.

        Returns:
            :class:`~difflow.TearAnalysis`.  ``print(fs.tear_analysis())``
            for the plain-text summary.

        Example:
            >>> print(fs.tear_analysis())
            Tear analysis: 1 recycle loop(s)
              loop: mixer -> reactor -> flash -> mixer
              declared tears:  recycle
              heuristic would: recycle
              minimum would:   recycle
        """
        return analyze_tears(self, max_cycles=max_cycles)

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
        if self.last_solve_clip_active:
            # The one failure this flag causes by itself, and the reason it
            # is worth a sentence of its own: a tear whose answer is
            # genuinely signed cannot be reached through the projection, and
            # nothing else in the message would ever say so (#263).
            message += (
                f" The negative-flow clip moved the proposed iterate on "
                f"{self.last_solve_clip_active} of those iterations: if any "
                "tear flow here is legitimately signed (a gas network, a "
                "power flow, any tear carrying a signed quantity), that "
                "projection puts the fixed point out of reach -- pass "
                "clip_negative_flows=False."
            )
        if action == "raise":
            raise ConvergenceError(message)
        warnings.warn(message, ConvergenceWarning, stacklevel=3)

    def _report_tolerance(self, tol: float, tol_basis: str) -> None:
        """Warn when a converged solve is still outside the tolerance asked for.

        The step test being met is not the same claim as the answer being
        accurate, and on a high-gain loop the two are decades apart (#264).
        Only a *measured* error says so; without ``error_probe`` there is
        nothing here to report.

        The error exceeding ``tol`` is not on its own enough to warn about.
        The unaccelerated path stops on ``|g(x) - x| < tol (1 + |x|)`` --- a
        deliberate choice, since with flows of order one and ``tol = 1e-8``
        optimistix calls a step near ``2e-8`` successful --- so on any loop
        with a gain worth the name the error sits a little above ``tol`` on
        an ordinary, healthy solve. Warning there would put a warning on
        almost every damped solve, and a warning that fires every time is
        read as noise exactly when it is not.

        So the default test is on the *amplification*: the error has to be
        an order of magnitude past the step the solve reported, which is a
        loop gain of 0.9 --- the regime this is about. A factor of two or
        three is the ordinary slack of a step test on a loop with any gain
        at all, and is not news.

        Under ``tol_basis="error"`` the gate comes off. There the caller
        asked for ``tol`` to mean the error and solve spent extra solves
        trying to deliver it, so falling short is a finding at any factor.
        """
        error = self.last_solve_error_estimate
        residual = self.last_solve_residual
        if (error is None or not self.last_solve_converged or error <= tol):
            return
        if (tol_basis != "error" and residual is not None
                and error < _TOLERANCE_WARNING_AMPLIFICATION * residual):
            return

        gain = self.last_solve_gain
        message = (
            "Recycle solve met its step tolerance and is still outside it: "
            f"tear stream(s) {', '.join(self.last_solve_tear_streams) or '(none)'} "
            f"stopped at a step of "
            f"{'unknown' if residual is None else f'{residual:.3e}'} against "
            f"tol={tol:.3e}, but the error measured from there is "
            f"{error:.3e}"
        )
        if gain is not None:
            message += f" -- the tear map has a gain of {gain:+.4f} there"
        message += (
            ". tol tests the STEP between iterates, not the error in the "
            "answer, and the two differ by 1/(1 - gain). "
        )
        if tol_basis == "error":
            message += (
                f"tol_basis=\"error\" tightened the step test "
                f"{_MAX_TOLERANCE_TIGHTENINGS} times without reaching it; "
                "the loop needs more iterations (max_iter) or a method that "
                "converges on it."
            )
        else:
            message += (
                f"Ask for {tol * tol / error:.3e} instead, or pass "
                'tol_basis="error" to have solve do that for you.'
            )
        warnings.warn(message, TearToleranceWarning, stacklevel=3)

    def _is_traced(self) -> bool:
        """True when a feed or unit parameter currently holds a JAX tracer.

        Used by :meth:`solve` to pick a differentiable solver.  Unit ``Params``
        dataclasses are not registered pytrees, so their fields are collected
        explicitly.
        """
        if _has_tracer(self.feeds):
            return True
        for unit in self.units:
            if _has_tracer([v for v in unit.params.values() if not callable(v)]):
                return True
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

    def _tear_substitution(self, x: Array, tear_names: list[str]) -> Array:
        """One plain substitution pass: packed tear in, packed tear out.

        This is ``g``, the map whose fixed point a recycle solve is looking
        for, with no damping and no acceleration --- which is what makes it
        the right instrument for measuring the map itself.
        """
        streams = dict(self.feeds)
        streams.update(self._array_to_streams(x, tear_names))
        streams = self._run_units(streams)
        return self._streams_to_array(
            {dest: streams[source] for source, dest in self.recycles.items()}
        )

    def _estimate_tear_error(
        self, streams: dict[str, Stream], tear_names: list[str], probe: int
    ) -> tuple[float | None, float | None]:
        """Measure the error the step residual hides, and the gain behind it.

        ``tol`` tests ``|g(x) - x|``, a *step*.  The error is
        ``x* - x = (I - M)^-1 (g(x) - x)`` with ``M = dg/dx``, and on a
        high-gain loop those differ by a factor the caller has no way to
        see (#264).  Each extra substitution pass from the converged tear
        yields one more term of that inverse, measured rather than
        assumed::

            r_j = g(x_j) - x_j,   x_{j+1} = g(x_j)
            x* - x_0 = sum_j r_j   ~   sum_{j<k} r_j + r_{k-1} s / (1 - s)

        where ``s = <r_{k-2}, r_{k-1}> / <r_{k-2}, r_{k-2}>`` is the signed
        ratio of the dominant mode.  Signed, not a ratio of norms: an
        oscillating loop has ``s < 0`` and a magnitude-only estimate would
        put its error on the wrong side and inflate it.  The closed form
        ``r / (1 - s)`` is also right for an *expanding* dominant mode,
        where the Neumann sum it continues does not converge --- a map with
        ``s = 3`` has its fixed point half a residual away, not infinitely
        far.

        Entries whose residual is already at round-off are dropped before
        any of this. They carry no information about the error, and a ratio
        taken from floating-point noise lands near one, which is exactly
        where the amplification blows up: that is how an estimator like
        this invents an alarm. A tear carrying a pressure of 1e5 alongside
        flows of order one has such an entry on almost every solve.

        Args:
            streams: The solution, which carries the converged tear.
            tear_names: Packing order for the tear vector.
            probe: Extra substitution passes to spend. Two is enough for a
                loop with one dominant mode, which is the usual case.

        Returns:
            ``(gain, error)``, either of which is ``None`` when there was
            nothing to measure --- a residual at round-off, or a tracer.
        """
        x = self._streams_to_array({name: streams[name] for name in tear_names})
        # Round-off floor, per entry: a tear carries pressures of 1e5 beside
        # flows of order one, and one absolute floor for both would either
        # keep the pressure noise or throw the flows away.
        floor = 1e2 * float(jnp.finfo(x.dtype).eps) * jnp.maximum(jnp.abs(x), 1.0)

        terms = []
        xi = x
        for _ in range(max(probe, 2)):
            gi = self._tear_substitution(xi, tear_names)
            terms.append(jnp.where(jnp.abs(gi - xi) > floor, gi - xi, 0.0))
            xi = gi

        first = _concrete(jnp.max(jnp.abs(terms[0])))
        if first is None:
            return None, None          # traced: no number to report
        if first == 0.0:
            # Every entry at its own round-off. The step test was not
            # merely met, it was met to the limit of the arithmetic, and
            # there is no amplification left to find -- but the gain is
            # still a property of the loop worth reporting, and an
            # accelerated solve (Anderson, Wegstein) lands here routinely.
            return self._tear_jacobian_gain(x, tear_names), 0.0

        denom = _concrete(jnp.vdot(terms[-2], terms[-2]))
        if not denom:
            return self._tear_jacobian_gain(x, tear_names), first
        gain = _concrete(jnp.vdot(terms[-2], terms[-1])) / denom
        if abs(1.0 - gain) < 1e-12:
            # An eigenvalue at one: the loop has no isolated fixed point
            # and the error is unbounded. Saying so beats reporting 1e12.
            return gain, float("inf")

        total = terms[0]
        for term in terms[1:]:
            total = total + term
        total = total + terms[-1] * (gain / (1.0 - gain))
        # Not clamped to the residual from below: an oscillating loop
        # (gain < 0) genuinely sits closer to its answer than its step
        # suggests, and reporting the larger of the two would hide that
        # as surely as the bare step hides the high-gain case.
        return gain, _concrete(jnp.max(jnp.abs(total)))

    #: Largest packed tear for which :meth:`_tear_jacobian_gain` forms the
    #: Jacobian (one forward-mode pass per entry).
    _MAX_JACOBIAN_GAIN_SIZE = 400

    def _tear_jacobian_gain(
        self, x: Array, tear_names: list[str]
    ) -> float | None:
        """The tear map's dominant eigenvalue, from its Jacobian at ``x``.

        The residual-ratio estimate in :meth:`_estimate_tear_error` needs
        residuals above round-off, and an accelerated solve usually leaves
        none: Anderson converges to ~1e-12 where substitution would stop at
        ``tol``. The tear map is a JAX function, so its Jacobian ``M = dg/dx``
        is available exactly; its eigenvalue of largest modulus is the gain.
        Returned signed (its real part) when the dominant eigenvalue is real,
        else its modulus carrying the sign of its real part. Variables the
        loop merely carries through unchanged (a Jacobian row equal to
        ``e_i``, eigenvalue exactly 1) are left out: they are neutral, not
        slow.

        Returns ``None`` when the tear is larger than
        :attr:`_MAX_JACOBIAN_GAIN_SIZE` or a unit in the loop cannot be
        differentiated in forward mode.
        """
        if x.size > self._MAX_JACOBIAN_GAIN_SIZE:
            return None
        try:
            jac = jax.jacfwd(lambda y: self._tear_substitution(y, tear_names))(x)
        except Exception:
            jac = None
        try:
            # The Jacobian ran every unit on forward-mode tracers, and a unit
            # that keeps its last result (an info dict, a cached state) now
            # holds tracers that outlived their trace. One concrete pass at
            # the same point puts concrete values back.
            self._tear_substitution(x, tear_names)
        except Exception:
            pass
        if jac is None:
            return None
        try:
            import numpy as np
            jac = np.asarray(jac, dtype=float)
            if not np.all(np.isfinite(jac)):
                return None
            # A variable the loop only carries (a unit passes the tear's T or
            # P straight through) has a Jacobian row that is exactly e_i: any
            # value of it is a fixed point, the iteration never moves it, and
            # its eigenvalue of exactly 1 says nothing about convergence.
            # Such a row contributes the factor (1 - lambda) alone to the
            # characteristic polynomial, so dropping its row and column
            # leaves the other eigenvalues unchanged.
            off = jac - np.diag(np.diag(jac))
            carried = (np.diag(jac) == 1.0) & np.all(off == 0.0, axis=1)
            keep = np.flatnonzero(~carried)
            if keep.size == 0:
                return None
            eig = np.linalg.eigvals(jac[np.ix_(keep, keep)])
        except Exception:
            return None
        lam = eig[int(np.argmax(np.abs(eig)))]
        if abs(lam.imag) <= 1e-9 * max(abs(lam), 1e-300):
            return float(lam.real)
        return float(np.copysign(abs(lam), lam.real))

    def _clip_and_count(
        self, x: Array, mask: Array, hits: Array
    ) -> tuple[Array, Array]:
        """:meth:`_clip_flows`, counting the steps where it changed something.

        A projection that moves an iterate is the one failure mode
        ``clip_negative_flows=True`` can cause on its own: on a tear whose
        answer is genuinely signed it makes the fixed point unreachable,
        and the solve then fails for a reason nothing in the output named
        (#263).  Counting the steps where it bound is what lets
        :meth:`_report_nonconvergence` name it.

        The count stays on device and is read once, after the loop: the
        accelerated iterations already pay one synchronisation per step
        for the residual, and a diagnostic has no business adding a
        second.
        """
        clipped = self._clip_flows(x, mask)
        return clipped, hits + jnp.any(clipped != x)

    def _record_clip_activity(self, hits: Array) -> None:
        """Read the clip counter off the device, once the loop is done."""
        count = _concrete(hits)
        self.last_solve_clip_active = 0 if count is None else int(count)

    def _initialize_tear_stream(
        self,
        stream_name: str,
        propagated: dict[str, Stream] | None = None,
        dest: str | None = None,
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
            dest: The tear destination, used to infer T and P for the
                fallback (see :meth:`_tear_default_stream`).

        Returns:
            The propagated value of the stream, or the flowsheet default
            when the walk could not produce a usable one.
        """
        if propagated is None:
            propagated = self._propagate_from_feeds()

        guess = propagated.get(stream_name)
        if guess is None or not self._usable_tear(guess):
            if dest is None:
                return self._make_zero_stream()
            return self._tear_default_stream(dest, propagated)
        return guess

    def _tear_default_stream(
        self, dest: str, available: dict[str, Stream]
    ) -> Stream:
        """The starting guess for tear ``dest`` when nothing better is known.

        Flows are ``default_flow``. Temperature and pressure come from the
        *other* streams entering the unit(s) that ``dest`` feeds --- the
        streams the recycle joins --- unless ``default_T`` / ``default_P``
        were given explicitly:

        * ``P`` is the highest of their pressures. A :class:`Mixer` passes on
          the lowest inlet pressure, so in a loop with no pressure change
          every pressure at or below the feed's is a fixed point; only a
          guess at or above it converges to the physical one, where the
          loop's pressure is pinned to the feed (and has a well-defined,
          non-unit loop gain, so the implicit gradient exists).
        * ``T`` is their flow-weighted mean, the adiabatic-mixing estimate.

        Both are wrapped in ``stop_gradient``: the guess is a starting point
        and the gradient comes from the converged fixed point.

        Args:
            dest: The tear destination stream name.
            available: Streams already known (feeds, plus whatever has been
                computed). Streams that are themselves tear destinations
                are ignored.

        Returns:
            A stream with every species at ``default_flow``.
        """
        base = self._make_zero_stream()
        if not (self.infer_tear_T or self.infer_tear_P):
            return base
        tears = set(self.recycles.values())
        siblings = []
        for unit in self.units:
            if dest not in unit.inlet_names:
                continue
            for name in unit.inlet_names:
                if name in tears or name not in available:
                    continue
                stream = available[name]
                if isinstance(stream, dict) and "T" in stream and "P" in stream:
                    siblings.append(stream)
        if not siblings:
            return base

        sg = jax.lax.stop_gradient
        T, P = base["T"], base["P"]
        if self.infer_tear_P:
            P = sg(jnp.max(jnp.stack([jnp.asarray(s["P"], float) for s in siblings])))
        if self.infer_tear_T:
            temps = jnp.stack([jnp.asarray(s["T"], float) for s in siblings])
            flows = jnp.stack([
                jnp.asarray(sum(get_flows(s).values()), float) if get_flows(s)
                else jnp.asarray(0.0) for s in siblings
            ])
            total = jnp.sum(flows)
            weights = jnp.where(
                total > 0, flows / jnp.where(total > 0, total, 1.0),
                1.0 / len(siblings),
            )
            T = sg(jnp.sum(weights * temps))
        flows = {s: jnp.asarray(self.default_flow) for s in self.species_order}
        return make_stream(flows, T, P)

    def _upstream_of_tears(self) -> dict[str, Stream]:
        """Feeds plus every stream computable without any tear.

        Runs, in order, the units whose inlets are all known before the
        loop closes. Those units see exactly these inputs in every iteration
        of the solve, so running them once more is safe under tracing (no
        exception is swallowed, nothing is guessed). It is skipped when every
        tear joins only feeds, which is the common case.
        """
        streams: dict[str, Stream] = dict(self.feeds)
        tears = set(self.recycles.values())
        needed = {
            name
            for unit in self.units
            if tears & set(unit.inlet_names)
            for name in unit.inlet_names
            if name not in tears
        }
        if needed <= set(streams):
            return streams
        for unit in self.units:
            if any(n in tears or n not in streams for n in unit.inlet_names):
                continue
            result = unit.operation(
                *[streams[n] for n in unit.inlet_names], **unit.params
            )
            streams.update(_parse_outlets(result, unit))
        return streams

    def _usable_tear(self, stream: Any) -> bool:
        """Whether a guess can stand in for a tear stream.

        ``_streams_to_array`` reads a species the stream does not carry as
        zero flow, so an outlet that omits some --- an aqueous liquor with
        no extractant in it --- is a usable guess as it stands. T and P it
        cannot invent: a stream missing either does not fail as a bad
        guess, it fails as a ``KeyError`` several frames away.
        """
        try:
            return "T" in stream and "P" in stream
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
        tears = set(self.recycles.values())

        for unit in self.units:
            inlets = [
                streams[name] if name in streams
                else self._tear_default_stream(name, streams) if name in tears
                else default
                for name in unit.inlet_names
            ]
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
            self._record_unit_info(result, unit)

        return streams

    def _solve_with_recycle_damped(
        self,
        tear_initial: dict[str, Stream],
        tol: float,
        max_iter: int,
        damping: float,
        accelerate: str | None = None,
        anderson_depth: int = 5,
        clip_negative_flows: bool = True,
        on_nonconvergence: str = "warn",
    ) -> dict[str, Stream]:
        """Solve flowsheet with recycle using damped fixed-point iteration.

        With ``accelerate="anderson"`` or ``"wegstein"`` the iteration is
        :class:`~difflow.initialization.AcceleratedFixedPoint` instead (the
        traced accelerated solver, #440): the same optimistix fixed point
        and implicit derivative, a much shorter iteration.  ``damping`` is
        not applied there, as it is not on the eager accelerated paths.

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

                streams.update(_parse_outlets(result, unit))

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
        if accelerate is None:
            fp_solver = optx.FixedPointIteration(
                rtol=tol * damping, atol=tol * damping
            )
        else:
            fp_solver = AcceleratedFixedPoint(
                atol=tol,
                method=accelerate,
                depth=anderson_depth,
                flow_mask=self._tear_flow_mask(len(tear_initial)),
                clip=clip_negative_flows,
            )
            iteration = flowsheet_iteration
        sol = optx.fixed_point(
            iteration,
            fp_solver,
            tear_array,
            args=args,
            max_steps=max_iter,
            throw=False,
        )
        tear_converged = sol.value
        if accelerate is not None:
            self._record_clip_activity(sol.state.clip_hits)

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

        if scaled is None and on_nonconvergence != "ignore":
            # Traced (jax.grad / jit): there is no number to judge here,
            # but there will be one when the computation runs.  A gradient
            # taken by implicit differentiation at an unconverged point is
            # silently wrong (the implicit function theorem assumes
            # g(x*) = x*), so the check is deferred to run time.
            jax.debug.callback(
                self._traced_convergence_check(
                    on_nonconvergence, max_iter, tol,
                    list(tear_initial.keys()),
                    method=accelerate or "plain substitution",
                ),
                jnp.max(jnp.abs(delta) / (tol + tol * jnp.abs(tear_converged))),
                jnp.max(jnp.abs(delta)),
                sol.stats.get("num_steps", jnp.asarray(max_iter)),
            )

        # Final solve with converged tear streams
        final_tear = self._array_to_streams(tear_converged, list(tear_initial.keys()))
        streams = dict(self.feeds)
        streams.update(final_tear)

        for unit in self.units:
            inlets = [streams[name] for name in unit.inlet_names]
            result = unit.operation(*inlets, **unit.params)

            streams.update(_parse_outlets(result, unit))

        return streams

    @staticmethod
    def _traced_convergence_check(
        action: str, max_iter: int, tol: float, tear_names: list[str],
        method: str = "plain substitution",
    ) -> Callable:
        """A run-time callback reporting a traced recycle solve that failed.

        Built for ``jax.debug.callback`` in the traced fixed-point path, where
        the residual is a tracer when ``solve`` runs and only has a value when
        the computation does. It reports the way :meth:`_report_nonconvergence`
        does for a concrete solve, but always as a :class:`ConvergenceWarning`:
        a traced solve never raises, ``on_nonconvergence="raise"`` included,
        because an exception inside a gradient makes the flowsheet
        undifferentiable. The caller
        compiles no check at all for ``"ignore"``.
        """
        def check(scaled, residual, steps):
            import numpy as np
            scaled = np.asarray(scaled)
            if not np.all(scaled < 1.0):      # also catches NaN
                message = (
                    "Recycle solve under jax.grad/jit did not converge: tear "
                    f"stream(s) {', '.join(tear_names) or '(none)'} reached a "
                    f"residual of {float(np.max(residual)):.3e} after "
                    f"{int(np.max(steps))} of {max_iter} iterations "
                    f"(tolerance {tol:.3e}, method: fixed_point (traced), "
                    f"{method}). Values AND GRADIENTS from this solve "
                    "are wrong: implicit differentiation assumes the loop is "
                    "converged. Raise max_iter (a loop of gain g needs about "
                    "log(tol)/log(g) iterations of plain substitution), or "
                    "converge the loop eagerly first and pass the result as "
                    "tear_initial."
                )
                warnings.warn(message, ConvergenceWarning, stacklevel=2)
        return check

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
        clip_hits = jnp.zeros((), dtype=jnp.int32)

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
            if res is not None and self.last_solve_history is not None:
                self.last_solve_history.append(res)
            if res is None:
                # A tracer reached the loop that _is_traced did not see (a
                # closure, say).  Unrolling this loop under jax.grad
                # differentiates the acceleration, whose least-squares
                # step is 0/0 once the iterates agree: a silent NaN.
                # solve() catches this and reruns on the fixed-point path.
                raise _UndetectedTracer
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
                    x_curr, clip_hits = self._clip_and_count(
                        x_curr, flow_mask, clip_hits
                    )

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

        self._record_clip_activity(clip_hits)

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
        clip_hits = jnp.zeros((), dtype=jnp.int32)

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
            if res is not None and self.last_solve_history is not None:
                self.last_solve_history.append(res)
            if res is None:
                raise _UndetectedTracer       # see _solve_with_wegstein
            elif res < tol:
                self.last_solve_iterations = iteration
                self.last_solve_converged = True
                break
            else:
                self.last_solve_converged = False

            # Apply Anderson acceleration
            x_next = accelerator.step(x_curr, g_curr)
            if clip_negative_flows:
                x_next, clip_hits = self._clip_and_count(
                    x_next, flow_mask, clip_hits
                )
            x_curr = x_next

        self._record_clip_activity(clip_hits)

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

        A field that is a key of the unit's own :attr:`Unit.params` dict
        (the keyword arguments the flowsheet passes to the operation on every
        call, e.g. ``{"split_frac": 0.1}`` on a :class:`Splitter`) updates
        that dict instead: ``{"purge.split_frac": 0.05}``. Those keywords
        override the operation's own parameters at call time, so they take
        precedence here too.

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
            # Keywords the flowsheet passes to the call (Unit.params) first:
            # they override the operation's own parameters anyway.
            call_updates = {k: v for k, v in updates.items() if k in unit.params}
            op_updates = {k: v for k, v in updates.items() if k not in unit.params}
            operation = unit.operation
            if op_updates:
                if not hasattr(operation, "params"):
                    raise AttributeError(
                        f"Unit {unit_name!r} operation has no .params attribute "
                        f"and {sorted(op_updates)} are not keys of its Unit.params "
                        f"{sorted(unit.params)}; cannot apply these updates"
                    )
                new_op_params = operation.params.update(**op_updates)
                # Shallow-copy the operation object and replace its .params so
                # that all other attributes (e.g. thermo, mode) are preserved
                # without having to know the constructor signature of each
                # unit type.
                operation = copy.copy(operation)
                operation.params = new_op_params
            new_call_params = {**unit.params, **call_updates}
            new_units[idx] = replace(unit, operation=operation, params=new_call_params)

        # Build a shallow copy of the flowsheet with the new units
        new_fs = Flowsheet(self.species_order, **self._default_kwargs())
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
        **solve_kwargs: Any,
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

        A key whose field is in the unit's :attr:`Unit.params` dict updates
        that dict (``"purge.split_frac"`` on a :class:`Splitter`).

        Keyword arguments are passed on to :meth:`solve` on every call. A
        recycle with a loop gain near 1 needs them under ``jax.grad``, where
        the solve falls back to plain substitution and the default
        ``max_iter=100`` can run out::

            obj = fs.make_objective_fn(product, max_iter=2000)

        Args:
            objective_fn: Callable that maps the solved stream dict returned
                by :meth:`solve` to a scalar :class:`jax.Array`.
            **solve_kwargs: Keyword arguments for :meth:`solve` (``max_iter``,
                ``tol``, ``acceleration``, ``damping``, ``tol_basis``, ...).

        Returns:
            Callable ``objective(params_dict) -> Array`` that updates unit
            parameters, solves the flowsheet, then evaluates
            ``objective_fn``.

        Example:
            >>> obj = fs.make_objective_fn(lambda s: s["crude"]["F_MeOH"], max_iter=500)
            >>> dM = jax.grad(obj)({"purge.split_frac": 0.1})
        """

        def objective(params: dict) -> Array:
            # Apply parameter updates to a temporary copy of the flowsheet
            # using dot-notation keys of the form "<unit_name>.<param_name>".
            updated_fs = self._apply_params(params)
            streams = updated_fs.solve(**solve_kwargs)
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
            self._record_unit_info(result, unit)

        return streams

    def _record_unit_info(self, result: Any, unit: "Unit") -> None:
        """Keep the info dict a unit returned, when it holds numbers.

        ``_parse_outlets`` takes the streams and drops the rest, which is
        where a unit says how its own inner solve went. Under tracing the
        dict holds tracers that must not outlive the trace, so nothing is
        kept then.
        """
        info = _unit_info(result, unit)
        if info is not None and not _has_tracer(info):
            self.last_solve_unit_info[unit.name] = info

    def _streams_to_array(self, streams: dict[str, Stream]) -> Array:
        """Convert dictionary of streams to a flat array.

        A species a stream does not carry packs as zero flow. Plugin units
        routinely return only the species their outlet holds --- a solvent
        extraction scrubber's aqueous liquor has no extractant or diluent
        in it --- and tearing such a stream used to fail here as a
        ``KeyError``. Absent means none, which is how the units that sum
        streams (``Flash``, the psychrometric helpers) already read it.
        """
        arrays = []
        for name in sorted(streams.keys()):
            stream = streams[name]
            # Pack: [F_species..., T, P]
            for s in self.species_order:
                arrays.append(stream.get(f"F_{s}", 0.0))
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
    **solve_kwargs: Any,
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
        **solve_kwargs: Passed to :meth:`Flowsheet.solve` on every call
            (``max_iter``, ``tol``, ``acceleration``, ...).

    Returns:
        Function mapping parameter dict to objective value
    """
    return flowsheet.make_objective_fn(objective_fn, **solve_kwargs)
