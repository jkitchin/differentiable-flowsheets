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
import jax.numpy as jnp
from jax import Array
import jax

from difflow.streams import Stream, get_flows, make_stream


# =============================================================================
# Protocols and Base Classes
# =============================================================================

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
            - 'outlet': Initial guess for outlet stream
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

    The update is: x_new = x_curr + q * (g_curr - x_curr)
    where q = s / (s - 1) and s = (g_curr - g_prev) / (x_curr - x_prev)

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

    # Acceleration factor: q = s / (s - 1)
    # For s < 1: q < 0 (under-relaxation)
    # For s > 1: q > 1 (over-relaxation, can be unstable)
    q = s / (s - 1 + 1e-10)

    # Clip to bounds for stability
    q_min, q_max = bounds
    q = jnp.clip(q, q_min, q_max)

    # Accelerated update
    x_new = x_curr + q * (g_curr - x_curr)

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

    # Solve least squares: min ||dF @ alpha - f_curr||^2
    # Normal equations: (dF @ dF.T + reg*I) @ alpha = dF @ f_curr
    dF_T = dF.T  # (n_vars, m_use)

    # Gram matrix with regularization
    G = dF @ dF_T + regularization * jnp.eye(m_use)

    # Right-hand side
    rhs = dF @ f_curr

    # Solve for coefficients
    alpha = jnp.linalg.solve(G, rhs)

    # Compute accelerated iterate
    # x_new = g_curr - sum(alpha_k * (g_{k+1} - g_k))
    dG = g_hist[1:] - g_hist[:-1]
    dG = dG[-m_use:]

    x_new = g_hist[-1] - dG.T @ alpha

    return x_new


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

@dataclass
class FlowsheetGraph:
    """Graph representation of a flowsheet for analysis."""
    units: list[str]
    streams: dict[str, tuple[str | None, str | None]]  # stream -> (source_unit, dest_unit)
    adjacency: dict[str, list[str]]  # unit -> list of downstream units

    @classmethod
    def from_flowsheet(cls, flowsheet) -> "FlowsheetGraph":
        """Build graph from a Flowsheet object."""
        units = [u.name for u in flowsheet.units]
        streams = {}
        adjacency = {u: [] for u in units}

        # Map streams to their source and destination units
        for unit in flowsheet.units:
            for outlet in unit.outlet_names:
                streams[outlet] = (unit.name, None)
            for inlet in unit.inlet_names:
                if inlet not in streams:
                    streams[inlet] = (None, unit.name)
                else:
                    src, _ = streams[inlet]
                    streams[inlet] = (src, unit.name)

        # Build adjacency from stream connections
        for stream, (src, dst) in streams.items():
            if src is not None and dst is not None and src in adjacency:
                adjacency[src].append(dst)

        return cls(units, streams, adjacency)


def find_cycles(graph: FlowsheetGraph) -> list[list[str]]:
    """Find all cycles in the flowsheet graph using DFS.

    Args:
        graph: FlowsheetGraph to analyze

    Returns:
        List of cycles, where each cycle is a list of unit names
    """
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.adjacency.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for unit in graph.units:
        if unit not in visited:
            dfs(unit)

    return cycles


def select_tear_streams(
    flowsheet,
    method: str = "heuristic",
) -> list[str]:
    """Select tear streams for recycle convergence.

    Tear streams are where the recycle loop is "torn" to convert
    a cyclic system into an acyclic one for sequential solving.

    Args:
        flowsheet: Flowsheet object
        method: Selection method
            - "heuristic": Select streams after mixers or before splitters
            - "minimum": Find minimum number of tears (more expensive)

    Returns:
        List of stream names to use as tear streams
    """
    graph = FlowsheetGraph.from_flowsheet(flowsheet)
    cycles = find_cycles(graph)

    if not cycles:
        return []

    if method == "heuristic":
        return _select_tears_heuristic(flowsheet, graph, cycles)
    elif method == "minimum":
        return _select_tears_minimum(flowsheet, graph, cycles)
    else:
        raise ValueError(f"Unknown tear selection method: {method}")


def _select_tears_heuristic(flowsheet, graph, cycles) -> list[str]:
    """Heuristic tear selection: prefer streams after mixers."""
    tear_streams = set()

    # For each cycle, find a good tear point
    for cycle in cycles:
        best_tear = None
        best_score = -1

        for i, unit_name in enumerate(cycle[:-1]):
            # Get the stream connecting this unit to next in cycle
            next_unit = cycle[i + 1]
            unit = next(u for u in flowsheet.units if u.name == unit_name)

            for outlet in unit.outlet_names:
                # Check if this stream goes to next_unit
                src, dst = graph.streams.get(outlet, (None, None))
                if dst == next_unit:
                    score = 0

                    # Prefer streams after mixers (composition known)
                    if "mixer" in unit_name.lower() or "mix" in unit_name.lower():
                        score += 10

                    # Prefer streams with fewer components (simpler)
                    # This would need stream info, so skip for now

                    # Prefer streams that are already recycles
                    if outlet in [s for s in flowsheet.recycles.keys()]:
                        score += 5

                    if score > best_score:
                        best_score = score
                        best_tear = outlet

        if best_tear:
            tear_streams.add(best_tear)

    return list(tear_streams)


def _select_tears_minimum(flowsheet, graph, cycles) -> list[str]:
    """Find minimum number of tear streams (greedy approximation)."""
    # Greedy: pick stream that appears in most cycles
    stream_counts = {}

    for cycle in cycles:
        for i, unit_name in enumerate(cycle[:-1]):
            unit = next(u for u in flowsheet.units if u.name == unit_name)
            for outlet in unit.outlet_names:
                stream_counts[outlet] = stream_counts.get(outlet, 0) + 1

    tear_streams = []
    remaining_cycles = list(cycles)

    while remaining_cycles:
        # Find stream in most remaining cycles
        best_stream = max(stream_counts.keys(),
                         key=lambda s: stream_counts.get(s, 0))
        tear_streams.append(best_stream)

        # Remove cycles that are now broken
        remaining_cycles = [
            c for c in remaining_cycles
            if best_stream not in _cycle_streams(flowsheet, graph, c)
        ]

        # Remove this stream from consideration
        stream_counts.pop(best_stream, None)

    return tear_streams


def _cycle_streams(flowsheet, graph, cycle) -> set[str]:
    """Get all streams in a cycle."""
    streams = set()
    for i, unit_name in enumerate(cycle[:-1]):
        unit = next(u for u in flowsheet.units if u.name == unit_name)
        streams.update(unit.outlet_names)
    return streams


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
    dT = (heat_duty + heat_of_reaction) / (F_total * Cp_avg + 1e-10)
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
        X = (jnp.sqrt(1 + 4*k*tau) - 1) / (2*k*tau + 1e-10)
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
        y[s] = z[s] * K / (1 + V * (K - 1) + 1e-10)

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
