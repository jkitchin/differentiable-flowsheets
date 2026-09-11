"""Initialization utilities for difflow flowsheets.

This module provides initialization strategies for improving convergence:
1. Unit-level initialization methods
2. Sequential decomposition with automatic tear stream selection
3. Anderson/Wegstein acceleration for recycle convergence
4. Hierarchical initialization for complex units

Based on best practices from IDAES and equation-oriented process simulators.
"""

from typing import Callable, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
import warnings
import heapq
import jax.numpy as jnp
from jax import Array
import jax

from difflow.streams import Stream, get_flows, make_stream
from difflow.numerics import safe_divide


# =============================================================================
# Protocols and Base Classes
# =============================================================================

class TearInitializationWarning(UserWarning):
    """A unit could not be run or estimated while guessing a tear stream.

    :meth:`difflow.Flowsheet.solve` starts a recycle from one pass of the
    units over the feeds. A unit that raises on that pass is not
    necessarily broken --- a CSTR's root find and a flash's Rachford-Rice
    both have trouble near an empty inlet, and the pass runs before any
    recycle is known. When the unit also has no usable ``initialize()`` to
    fall back on, its outlets stand in as the flowsheet's default stream
    and this warning says so, because the alternative is a tear guess that
    is quietly worse than it looks.
    """


@runtime_checkable
class Initializable(Protocol):
    """Protocol for units that support initialization."""

    def initialize(
        self,
        inlet: Stream,
        **kwargs
    ) -> dict[str, Any]:
        """Generate initial guesses for unit outputs and internal states.

        Args:
            inlet: Inlet stream
            **kwargs: Unit-specific parameters

        Returns:
            Dictionary containing:
            - 'outlets': Initial guesses for the outlet streams, as a
              sequence in the order ``__call__`` returns them. This is
              how a caller with only ``Unit.outlet_names`` to go on can
              tell which guess is which, so a multi-outlet unit has to
              provide it to be usable by the flowsheet.
            - 'outlet': The singular spelling, for a unit with one outlet.
              Means the same as a one-element ``outlets``.
            - 'states': Optional dict of internal state guesses
            - 'info': Optional additional information
        """
        ...


@dataclass
class InitializationResult:
    """Result from initialization procedure."""
    success: bool
    outlet: Stream | None = None
    states: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    message: str = ""


# =============================================================================
# Acceleration Methods
# =============================================================================

def wegstein_acceleration(
    x_prev: Array,
    x_curr: Array,
    g_prev: Array,
    g_curr: Array,
    bounds: tuple[float, float] = (-5.0, 0.0),
) -> Array:
    """Wegstein acceleration for fixed-point iteration.

    Wegstein's method accelerates convergence by estimating the optimal
    relaxation factor from the previous two iterations.

    The update is: x_new = q * x_curr + (1 - q) * g_curr
    where q = s / (s - 1) and s = (g_curr - g_prev) / (x_curr - x_prev).

    ``q`` is the weight on the *old* iterate ``x_curr``; ``(1 - q)`` weights
    the direct-substitution value ``g_curr``. For a linear ``g`` this cancels
    the fixed-point map's slope exactly, so the update lands on the solution in
    a single step. (Writing it as ``x_curr + q * (g_curr - x_curr)`` instead
    swaps the two weights and makes the effective slope ``1 + s``, which
    diverges for the common ``0 < s < 1`` contraction.)

    Args:
        x_prev: Previous iterate
        x_curr: Current iterate
        g_prev: g(x_prev) - result of fixed-point function at x_prev
        g_curr: g(x_curr) - result of fixed-point function at x_curr
        bounds: (q_min, q_max) bounds for acceleration parameter

    Returns:
        Accelerated next iterate
    """
    # Compute slope
    dx = x_curr - x_prev
    dg = g_curr - g_prev

    # Avoid division by zero
    dx_safe = jnp.where(jnp.abs(dx) > 1e-10, dx, 1e-10)

    # Wegstein parameter: s = dg/dx
    s = dg / dx_safe

    # Acceleration factor: q = s / (s - 1), the weight on the old iterate.
    # For 0 < s < 1: q < 0 (extrapolation past g_curr)
    # For s < 0: q in (0, 1) (damped toward x_curr)
    q = safe_divide(s, s - 1)

    # Clip to bounds for stability
    q_min, q_max = bounds
    q = jnp.clip(q, q_min, q_max)

    # Accelerated update: weight q on x_curr, (1 - q) on g_curr.
    x_new = q * x_curr + (1 - q) * g_curr

    return x_new


def anderson_acceleration_step(
    x_hist: Array,
    g_hist: Array,
    m: int = 5,
    regularization: float = 1e-10,
) -> Array:
    """One step of Anderson acceleration (AA-I variant).

    Anderson acceleration uses a linear combination of previous iterates
    to find a better next iterate. It's equivalent to GMRES applied to
    the fixed-point iteration.

    Args:
        x_hist: History of iterates, shape (n_hist, n_vars)
        g_hist: History of g(x) values, shape (n_hist, n_vars)
        m: Maximum history depth
        regularization: Tikhonov regularization for stability

    Returns:
        Accelerated next iterate
    """
    n_hist = x_hist.shape[0]
    m_use = min(m, n_hist - 1)

    if m_use < 1:
        # Not enough history, just return direct iteration
        return g_hist[-1]

    # Residuals: f_k = g(x_k) - x_k
    F = g_hist - x_hist  # (n_hist, n_vars)

    # Build the difference matrix
    # dF[:, k] = f_{n-m+k+1} - f_{n-m+k}
    dF = F[1:] - F[:-1]  # (n_hist-1, n_vars)
    dF = dF[-m_use:]  # Use last m differences

    # Current residual
    f_curr = F[-1]  # (n_vars,)

    # Solve the least squares  min_alpha || dF.T @ alpha - f_curr ||^2
    # via an SVD-based solver rather than the normal equations
    # (dF @ dF.T + reg*I) alpha = dF @ f_curr. The normal equations square
    # the condition number, which overflows to NaN for a badly-scaled packed
    # tear vector — e.g. gas transmission networks whose [F, T, P] entries
    # span ~1e1 (kg/s) to ~1e6 (Pa), so dF @ dF.T has a condition number of
    # order (P/F)^4. lstsq works on dF.T directly (condition number, not its
    # square) and drops singular values below rcond, giving a Tikhonov-like
    # regularization that is invariant to the overall problem scale.
    alpha, _, _, _ = jnp.linalg.lstsq(dF.T, f_curr, rcond=regularization)

    # Compute accelerated iterate
    # x_new = g_curr - sum(alpha_k * (g_{k+1} - g_k))
    dG = g_hist[1:] - g_hist[:-1]
    dG = dG[-m_use:]

    x_new = g_hist[-1] - dG.T @ alpha

    # Safety: if the accelerated step is not finite (degenerate history), fall
    # back to plain substitution so the iteration cannot propagate NaN/Inf.
    return jnp.where(jnp.all(jnp.isfinite(x_new)), x_new, g_hist[-1])


class AndersonAccelerator:
    """Anderson acceleration with history management.

    Maintains a rolling history of iterates for acceleration.
    Can be used in fixed-point iteration loops.
    """

    def __init__(self, m: int = 5, regularization: float = 1e-10):
        """Initialize accelerator.

        Args:
            m: Maximum history depth
            regularization: Tikhonov regularization parameter
        """
        self.m = m
        self.regularization = regularization
        self.x_hist = []
        self.g_hist = []

    def reset(self):
        """Clear history."""
        self.x_hist = []
        self.g_hist = []

    def step(self, x: Array, g: Array) -> Array:
        """Perform one acceleration step.

        Args:
            x: Current iterate
            g: g(x) - result of fixed-point function

        Returns:
            Accelerated next iterate
        """
        self.x_hist.append(x)
        self.g_hist.append(g)

        # Limit history size
        if len(self.x_hist) > self.m + 1:
            self.x_hist.pop(0)
            self.g_hist.pop(0)

        if len(self.x_hist) < 2:
            return g  # Not enough history

        x_arr = jnp.stack(self.x_hist)
        g_arr = jnp.stack(self.g_hist)

        return anderson_acceleration_step(
            x_arr, g_arr, self.m, self.regularization
        )


# =============================================================================
# Tear Stream Selection
# =============================================================================

class CycleEnumerationWarning(UserWarning):
    """Cycle enumeration stopped before it ran out of cycles.

    :func:`find_cycles` enumerates *elementary* cycles, of which a densely
    recycled graph can have exponentially many.  ``max_cycles`` bounds the
    work; hitting it means the tear set that comes out covers the cycles
    that were found and says nothing about the rest.
    """


@dataclass(frozen=True)
class StreamEdge:
    """One directed connection between two units, and the name a tear uses.

    Attributes:
        stream: The stream to tear on.  For a declared recycle this is the
            **destination** name, because that is the name
            :meth:`difflow.Flowsheet.solve` seeds, iterates on and reports
            in ``last_solve_tear_streams``.
        source: Unit that computes the stream.
        dest: Unit that consumes it.
        declared: Whether the connection is a recycle the user declared
            with ``add_recycle``.
        feedback: Whether ``dest`` comes at or before ``source`` in the
            flowsheet's unit order --- i.e. whether a sequential pass in
            that order reads the stream before anything writes it.  Every
            cycle contains at least one such edge, and an untorn one is
            what a sequential solve trips over.
    """
    stream: str
    source: str
    dest: str
    declared: bool = False
    feedback: bool = False


@dataclass
class FlowsheetGraph:
    """Graph representation of a flowsheet for analysis.

    Attributes:
        units: Unit names, in the flowsheet's calculation order.
        streams: ``stream -> (source_unit, dest_unit)``.  A declared
            recycle's destination is credited to the unit that computes
            the recycle's source, since that is where its value comes
            from; a stream feeding several units lists the first of them
            (``edges`` has them all).
        adjacency: ``unit -> downstream units``.
        edges: Every connection, one entry per (stream, consumer) pair.
            This is the object the tear selection works on: two units
            joined by two streams are two edges, and tearing one of them
            leaves the other closing the loop.
        inlets: ``unit -> inlet stream names``.
        outlets: ``unit -> outlet stream names``.
    """
    units: list[str]
    streams: dict[str, tuple[str | None, str | None]]  # stream -> (source_unit, dest_unit)
    adjacency: dict[str, list[str]]  # unit -> list of downstream units
    edges: list[StreamEdge] = field(default_factory=list)
    inlets: dict[str, list[str]] = field(default_factory=dict)
    outlets: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_flowsheet(cls, flowsheet) -> "FlowsheetGraph":
        """Build graph from a Flowsheet object.

        The declared recycles are part of the graph.  They are the only
        thing that closes a loop in a flowsheet built the usual way --- a
        recycle destination is an inlet no unit computes, so without the
        ``add_recycle`` map the digraph of stream names is a DAG and
        :func:`find_cycles` would report a flowsheet with a recycle as
        having none.
        """
        units = [u.name for u in flowsheet.units]
        order = {name: i for i, name in enumerate(units)}
        inlets = {u.name: list(u.inlet_names) for u in flowsheet.units}
        outlets = {u.name: list(u.outlet_names) for u in flowsheet.units}

        producer: dict[str, str] = {}
        consumers: dict[str, list[str]] = {}
        for unit in flowsheet.units:
            for name in unit.outlet_names:
                producer.setdefault(name, unit.name)
            for name in unit.inlet_names:
                consumers.setdefault(name, []).append(unit.name)

        recycles = dict(getattr(flowsheet, "recycles", None) or {})

        def _feedback(src: str, dst: str) -> bool:
            # `<=` and not `<`: a unit recycling to itself is its own
            # feedback edge, and it has to be torn like any other.
            return order[dst] <= order[src]

        edges: list[StreamEdge] = []
        for name, src in producer.items():
            for dst in consumers.get(name, []):
                edges.append(StreamEdge(
                    stream=name,
                    source=src,
                    dest=dst,
                    # A recycle written as add_recycle(s, s) tears a
                    # stream that already has both ends; it is this edge,
                    # not a second one.
                    declared=recycles.get(name) == name,
                    feedback=_feedback(src, dst),
                ))

        for source, dest in recycles.items():
            if source == dest:
                continue
            src = producer.get(source)
            if src is None:
                continue  # recycling something no unit computes
            for dst in consumers.get(dest, []):
                edges.append(StreamEdge(
                    stream=dest,
                    source=src,
                    dest=dst,
                    declared=True,
                    feedback=_feedback(src, dst),
                ))

        recycled_by = {dest: source for source, dest in recycles.items()}
        streams: dict[str, tuple[str | None, str | None]] = {}
        for name in list(producer) + [n for n in consumers if n not in producer]:
            src = producer.get(name)
            if src is None and name in recycled_by:
                src = producer.get(recycled_by[name])
            dst = consumers.get(name, [None])[0]
            streams[name] = (src, dst)

        adjacency: dict[str, list[str]] = {u: [] for u in units}
        for edge in edges:
            downstream = adjacency.setdefault(edge.source, [])
            if edge.dest not in downstream:
                downstream.append(edge.dest)

        return cls(units, streams, adjacency, edges, inlets, outlets)


def _edge_cycles(
    graph: FlowsheetGraph,
    max_cycles: int = 1000,
) -> list[list[StreamEdge]]:
    """Every elementary cycle of the graph, as sequences of edges.

    Elementary means no unit repeats, and each cycle is reported once:
    the search from unit ``i`` only visits units at or after ``i``, so a
    cycle is enumerated from its earliest member and nowhere else (the
    standard Johnson restriction).

    Edges, not unit names, because a cycle is only broken by tearing an
    edge on it: two units joined by two parallel streams sit on two
    distinct cycles that happen to name the same units, and tearing one
    stream leaves the other one closing the loop.
    """
    out_edges: dict[str, list[StreamEdge]] = {u: [] for u in graph.units}
    for edge in graph.edges:
        if edge.source in out_edges and edge.dest in out_edges:
            out_edges[edge.source].append(edge)

    index = {u: i for i, u in enumerate(graph.units)}
    cycles: list[list[StreamEdge]] = []
    path: list[StreamEdge] = []

    def extend(start: int, node: str, on_path: set[str]) -> None:
        for edge in out_edges[node]:
            if len(cycles) >= max_cycles:
                return
            position = index[edge.dest]
            if position < start:
                continue
            if position == start:
                cycles.append(path + [edge])
            elif edge.dest not in on_path:
                on_path.add(edge.dest)
                path.append(edge)
                extend(start, edge.dest, on_path)
                path.pop()
                on_path.discard(edge.dest)

    for start, unit in enumerate(graph.units):
        if len(cycles) >= max_cycles:
            break
        extend(start, unit, {unit})

    if len(cycles) >= max_cycles:
        warnings.warn(
            f"Stopped enumerating cycles at {max_cycles}; the tear set covers "
            "the cycles that were found and says nothing about the rest. "
            "Raise max_cycles to look further.",
            CycleEnumerationWarning,
            stacklevel=3,
        )

    return cycles


def find_cycles(
    graph: FlowsheetGraph,
    max_cycles: int = 1000,
) -> list[list[str]]:
    """Find the elementary cycles in the flowsheet graph.

    Args:
        graph: FlowsheetGraph to analyze
        max_cycles: Stop after this many cycles, with a
            :class:`CycleEnumerationWarning`.  Elementary cycles can be
            exponentially many in a densely recycled graph.

    Returns:
        List of cycles, each a list of unit names with the first unit
        repeated at the end (``["mixer", "reactor", "mixer"]``).  Two
        cycles over the same units by different streams appear once.
    """
    return _cycle_names(_edge_cycles(graph, max_cycles))


def _cycle_names(edge_cycles: list[list[StreamEdge]]) -> list[list[str]]:
    """Edge cycles as unit names, with cycles over the same units merged."""
    seen: set[tuple[str, ...]] = set()
    cycles: list[list[str]] = []
    for cycle in edge_cycles:
        names = [edge.source for edge in cycle] + [cycle[0].source]
        key = tuple(names)
        if key in seen:
            continue
        seen.add(key)
        cycles.append(names)
    return cycles


def _tear_score(graph: FlowsheetGraph, edge: StreamEdge, counts: dict) -> float:
    """How good a tear this edge would make; larger is better.

    The ordering is deliberate:

    - A stream the user already declared as a recycle wins outright.  The
      point of an automatic choice is to fill a gap, not to overrule a
      decision someone made.
    - Then the number of loops the edge lies on, which is what makes one
      tear serve several nested recycles instead of one each.
    - Then the classical heuristic: tear downstream of a mixing point,
      where the stream is the sum of everything entering it, so a guess
      that is wrong in composition is still right in order of magnitude.
    - A feedback edge last, as a tie-breaker only.  It is the edge the
      flowsheet's own unit order already treats as the loop closure, so
      preferring it keeps the automatic choice close to what whoever
      ordered the units had in mind.  It is not a requirement:
      :func:`calculation_order` re-sequences the units around whatever is
      torn, which is what lets ``method="minimum"`` tear a forward edge.
    """
    score = 0.0
    if edge.declared:
        score += 100.0
    score += 4.0 * min(counts.get(edge, 0), 5)
    if len(graph.inlets.get(edge.source, ())) > 1:
        score += 10.0
    if "mix" in edge.source.lower():
        score += 10.0
    if edge.feedback:
        score += 1.0
    return score


def _best_tear(graph: FlowsheetGraph, edges, counts: dict) -> StreamEdge:
    """Highest-scoring edge, ties broken by name so the choice is stable."""
    return min(
        edges,
        key=lambda e: (-_tear_score(graph, e, counts), e.stream, e.source, e.dest),
    )


def select_tear_streams(
    flowsheet,
    method: str = "heuristic",
    max_cycles: int = 1000,
) -> list[str]:
    """Select tear streams for recycle convergence.

    Tear streams are where the recycle loop is "torn" to convert
    a cyclic system into an acyclic one for sequential solving.

    Args:
        flowsheet: Flowsheet object, or a :class:`FlowsheetGraph` already
            built from one.
        method: Selection method
            - "heuristic": One tear per loop, scored by :func:`_tear_score`
            - "minimum": Fewest tears that break every loop (greedy)
        max_cycles: Passed to :func:`find_cycles`.

    Returns:
        Stream names to tear, in the order they were chosen.  These are
        the names :meth:`difflow.Flowsheet.solve` seeds and reports: for
        a declared recycle, the destination name.

    Example:
        >>> select_tear_streams(fs)
        ['recycle']
    """
    graph = (
        flowsheet if isinstance(flowsheet, FlowsheetGraph)
        else FlowsheetGraph.from_flowsheet(flowsheet)
    )
    cycles = _edge_cycles(graph, max_cycles)

    if not cycles:
        return []

    if method == "heuristic":
        return _select_tears_heuristic(graph, cycles)
    elif method == "minimum":
        return _select_tears_minimum(graph, cycles)
    else:
        raise ValueError(f"Unknown tear selection method: {method}")


def _select_tears_heuristic(
    graph: FlowsheetGraph,
    cycles: list[list[StreamEdge]],
) -> list[str]:
    """Heuristic tear selection: the best-scoring edge on each open loop.

    Loops are taken in the order they were found, and one already broken
    by an earlier choice is skipped --- so nested recycles sharing a
    stream still cost one tear, without the set-cover search that
    ``method="minimum"`` pays for.
    """
    counts: dict[StreamEdge, int] = {}
    for cycle in cycles:
        for edge in set(cycle):
            counts[edge] = counts.get(edge, 0) + 1

    tears: list[str] = []
    torn: set[str] = set()
    for cycle in cycles:
        if any(edge.stream in torn for edge in cycle):
            continue
        best = _best_tear(graph, cycle, counts)
        tears.append(best.stream)
        torn.add(best.stream)

    return tears


def _select_tears_minimum(
    graph: FlowsheetGraph,
    cycles: list[list[StreamEdge]],
) -> list[str]:
    """Fewest tear streams that break every loop (greedy set cover).

    Exactly minimising the tear set is NP-hard (it is a feedback arc set),
    so this is the standard greedy approximation: take the stream lying on
    the most still-unbroken loops, tie-broken by :func:`_tear_score`.  It
    is optimal on the nested and shared-stream topologies that occur in
    practice, and within a log factor in general.
    """
    counts: dict[StreamEdge, int] = {}
    for cycle in cycles:
        for edge in set(cycle):
            counts[edge] = counts.get(edge, 0) + 1

    by_stream: dict[str, float] = {}
    for edge in graph.edges:
        score = _tear_score(graph, edge, counts)
        by_stream[edge.stream] = max(by_stream.get(edge.stream, score), score)

    tears: list[str] = []
    remaining = list(cycles)
    while remaining:
        covered: dict[str, int] = {}
        for cycle in remaining:
            for name in {edge.stream for edge in cycle}:
                covered[name] = covered.get(name, 0) + 1

        pick = min(
            covered,
            key=lambda n: (-covered[n], -by_stream.get(n, 0.0), n),
        )
        tears.append(pick)
        remaining = [
            cycle for cycle in remaining
            if all(edge.stream != pick for edge in cycle)
        ]

    return tears


def calculation_order(
    graph: FlowsheetGraph,
    tears: list[str] | set[str],
) -> list[str] | None:
    """The order to run the units in, once ``tears`` are seeded.

    A topological sort (Kahn) of the graph with the torn streams removed,
    taking the flowsheet's own unit order whenever more than one unit is
    ready.  A flowsheet already written in a runnable order therefore
    comes back unchanged, and one torn somewhere other than where its
    author happened to start comes back re-sequenced rather than broken:
    tearing the mixer outlet of a ``mixer -> reactor -> splitter`` loop
    means running the reactor first and the mixer last, and the fixed
    point is the same one.

    Returns:
        Unit names in calculation order, or ``None`` when the torn graph
        still has a cycle --- i.e. when ``tears`` does not break every
        loop, and no order exists.
    """
    tears = set(tears)
    index = {unit: i for i, unit in enumerate(graph.units)}
    indegree = {unit: 0 for unit in graph.units}
    downstream: dict[str, list[str]] = {unit: [] for unit in graph.units}

    for edge in graph.edges:
        if edge.stream in tears:
            continue
        if edge.source not in indegree or edge.dest not in indegree:
            continue
        indegree[edge.dest] += 1
        downstream[edge.source].append(edge.dest)

    ready = [index[u] for u in graph.units if indegree[u] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        unit = graph.units[heapq.heappop(ready)]
        order.append(unit)
        for consumer in downstream[unit]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, index[consumer])

    return order if len(order) == len(graph.units) else None


def calculation_order_problems(
    graph: FlowsheetGraph,
    tears: list[str] | set[str],
    feeds: list[str] | set[str] = (),
) -> tuple[list[str], list[str]]:
    """Inlets a sequential pass in unit order could not supply.

    A sequential modular solve walks ``flowsheet.units`` in order and looks
    each inlet up in what it has computed so far, so an inlet that is not a
    feed, not torn, and not already written is a ``KeyError`` several frames
    down rather than a diagnosis.

    Returns:
        ``(missing, out_of_order)``.  ``missing`` names inlets no unit
        computes and nothing supplies; ``out_of_order`` names inlets whose
        producing unit runs later than the unit that reads them.  Either is
        a topology the declared unit order cannot solve as it stands.
    """
    tears = set(tears)
    feeds = set(feeds)
    missing: list[str] = []
    out_of_order: list[str] = []

    for name, (source, _) in graph.streams.items():
        if name in tears or name in feeds:
            continue
        consumed = any(name in ins for ins in graph.inlets.values())
        if not consumed:
            continue
        if source is None:
            missing.append(name)

    for edge in graph.edges:
        if edge.feedback and edge.stream not in tears and edge.stream not in feeds:
            if edge.stream not in out_of_order:
                out_of_order.append(edge.stream)

    return missing, out_of_order


@dataclass
class TearAnalysis:
    """What :func:`analyze_tears` found: the loops, and where to cut them.

    Attributes:
        cycles: Elementary cycles, as unit names (see :func:`find_cycles`).
        declared: Tear streams the flowsheet already has, from
            ``add_recycle`` --- the destination names, which is what
            ``last_solve_tear_streams`` reports.
        heuristic: What ``select_tear_streams(method="heuristic")`` would
            choose from scratch.
        minimum: What ``select_tear_streams(method="minimum")`` would
            choose from scratch.
        uncovered: Cycles no declared tear breaks.  A non-empty list on a
            flowsheet with recycles means a loop is closed in the topology
            with nothing seeding it.
        missing_inputs: Inlets nothing supplies (see
            :func:`calculation_order_problems`).
        out_of_order: Inlets read before they are written, given the
            declared tears and the declared unit order.
    """
    cycles: list[list[str]] = field(default_factory=list)
    declared: list[str] = field(default_factory=list)
    heuristic: list[str] = field(default_factory=list)
    minimum: list[str] = field(default_factory=list)
    uncovered: list[list[str]] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    out_of_order: list[str] = field(default_factory=list)

    @property
    def torn(self) -> bool:
        """Whether the declared tears break every cycle that was found."""
        return not self.uncovered

    def summary(self) -> str:
        """A few lines of plain text, for a notebook or a report."""
        def names(items) -> str:
            return ", ".join(items) if items else "(none)"

        lines = [f"Tear analysis: {len(self.cycles)} recycle loop(s)"]
        for cycle in self.cycles:
            lines.append("  loop: " + " -> ".join(cycle))
        lines.append(f"  declared tears:  {names(self.declared)}")
        lines.append(f"  heuristic would: {names(self.heuristic)}")
        lines.append(f"  minimum would:   {names(self.minimum)}")
        if self.uncovered:
            for cycle in self.uncovered:
                lines.append("  NOT TORN: " + " -> ".join(cycle))
        if self.missing_inputs:
            lines.append(f"  inlets nothing supplies: {names(self.missing_inputs)}")
        if self.out_of_order:
            lines.append(
                f"  inlets read before they are written: {names(self.out_of_order)}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def analyze_tears(flowsheet, max_cycles: int = 1000) -> TearAnalysis:
    """Report the recycle loops of a flowsheet and where they could be torn.

    Pure diagnosis: it reads the flowsheet, runs nothing and changes
    nothing.  See :meth:`difflow.Flowsheet.tear_analysis`.
    """
    graph = FlowsheetGraph.from_flowsheet(flowsheet)
    edge_cycles = _edge_cycles(graph, max_cycles)
    cycles = _cycle_names(edge_cycles)

    recycles = getattr(flowsheet, "recycles", None) or {}
    declared = list(dict.fromkeys(recycles.values()))

    uncovered = _cycle_names([
        cycle for cycle in edge_cycles
        if all(edge.stream not in declared for edge in cycle)
    ])

    missing, out_of_order = calculation_order_problems(
        graph, declared, getattr(flowsheet, "feeds", None) or {}
    )

    return TearAnalysis(
        cycles=cycles,
        declared=declared,
        heuristic=(
            _select_tears_heuristic(graph, edge_cycles) if edge_cycles else []
        ),
        minimum=(
            _select_tears_minimum(graph, edge_cycles) if edge_cycles else []
        ),
        uncovered=uncovered,
        missing_inputs=missing,
        out_of_order=out_of_order,
    )

# =============================================================================
# Initialization Helpers
# =============================================================================

def estimate_outlet_temperature(
    inlet: Stream,
    heat_duty: float = 0.0,
    heat_of_reaction: float = 0.0,
    Cp_avg: float = 75.0,  # J/mol/K
) -> Array:
    """Estimate outlet temperature from energy balance.

    Args:
        inlet: Inlet stream
        heat_duty: External heat duty (W), positive = heat added
        heat_of_reaction: Heat released by reactions (W), positive = exothermic
        Cp_avg: Average heat capacity (J/mol/K)

    Returns:
        Estimated outlet temperature (K)
    """
    flows = get_flows(inlet)
    F_total = sum(flows.values())

    if F_total < 1e-10:
        return inlet["T"]

    # Q = F * Cp * dT
    # dT = (heat_duty + heat_of_reaction) / (F * Cp)
    dT = safe_divide(heat_duty + heat_of_reaction, F_total * Cp_avg)
    T_out = inlet["T"] + dT

    # Clip to reasonable range
    T_out = jnp.clip(T_out, 200.0, 1000.0)

    return T_out


def estimate_cstr_conversion(
    k: float,
    tau: float,
    order: int = 1,
) -> float:
    """Estimate CSTR conversion for nth-order reaction.

    For first-order: X = k*tau / (1 + k*tau)
    For second-order: X = (sqrt(1 + 4*k*C0*tau) - 1) / (2*k*C0*tau)

    Args:
        k: Rate constant
        tau: Residence time (s)
        order: Reaction order (1 or 2)

    Returns:
        Estimated conversion (0 to 1)
    """
    if order == 1:
        X = k * tau / (1 + k * tau)
    elif order == 2:
        # Assuming C0 = 1 for estimation
        X = safe_divide(jnp.sqrt(1 + 4*k*tau) - 1, 2*k*tau)
    else:
        # General approximation
        X = 1 - 1 / (1 + k * tau)

    return jnp.clip(X, 0.0, 0.999)


def estimate_flash_split(
    inlet: Stream,
    T: float | None = None,
    P: float | None = None,
    K_values: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Estimate vapor fraction and composition for flash.

    Uses Rachford-Rice for initial estimate with assumed K-values.

    Args:
        inlet: Inlet stream
        T: Flash temperature (K), defaults to inlet T
        P: Flash pressure (Pa), defaults to inlet P
        K_values: Optional K-values by species

    Returns:
        Tuple of (vapor_fraction, vapor_mole_fractions)
    """
    flows = get_flows(inlet)
    T = T if T is not None else inlet["T"]
    P = P if P is not None else inlet["P"]

    # If no K-values provided, use rough estimates based on T
    if K_values is None:
        # Very rough: K increases with T, decreases with molecular weight
        K_values = {s: jnp.exp(0.01 * (T - 350)) for s in flows.keys()}

    F_total = sum(flows.values())
    if F_total < 1e-10:
        return 0.0, {s: 0.0 for s in flows.keys()}

    z = {s: f / F_total for s, f in flows.items()}

    # Rachford-Rice: sum(z_i * (K_i - 1) / (1 + V*(K_i - 1))) = 0
    # Simple approximation: if avg K > 1, mostly vapor; if < 1, mostly liquid
    K_avg = sum(z[s] * K_values.get(s, 1.0) for s in flows.keys())

    if K_avg < 0.1:
        V = 0.0
    elif K_avg > 10:
        V = 1.0
    else:
        V = (K_avg - 1) / (K_avg + 1)

    V = float(jnp.clip(V, 0.0, 1.0))

    # Vapor composition
    y = {}
    for s in flows.keys():
        K = K_values.get(s, 1.0)
        y[s] = safe_divide(z[s] * K, 1 + V * (K - 1))

    return V, y


def initialize_from_experiment(
    unit,
    inlet: Stream,
    experiment_data: dict[str, float],
) -> InitializationResult:
    """Initialize unit from experimental data.

    Uses experimental measurements as initial guesses.

    Args:
        unit: Unit operation
        inlet: Inlet stream
        experiment_data: Dict with keys like 'T_out', 'conversion', 'vapor_fraction'

    Returns:
        InitializationResult with experimental-based guesses
    """
    flows = get_flows(inlet)

    # Build outlet guess from experimental data
    T_out = experiment_data.get('T_out', inlet["T"])

    if 'conversion' in experiment_data:
        X = experiment_data['conversion']
        # Assume first species is reactant
        species_list = list(flows.keys())
        outlet_flows = dict(flows)
        if species_list:
            key_species = species_list[0]
            outlet_flows[key_species] = flows[key_species] * (1 - X)
    else:
        outlet_flows = flows

    outlet = make_stream(outlet_flows, T_out, inlet["P"])

    return InitializationResult(
        success=True,
        outlet=outlet,
        states=experiment_data,
        info={'source': 'experimental'},
        message="Initialized from experimental data"
    )
