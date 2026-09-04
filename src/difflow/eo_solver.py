"""Equation-Oriented (EO) solver for difflow flowsheets.

This module provides an alternative to the sequential modular (SM) solver.
Instead of evaluating units one-by-one and iterating on tear streams,
the EO solver assembles all unit equations and connectivity constraints
into a single nonlinear system F(x) = 0 and solves simultaneously
using Newton's method via optimistix.

Key classes:
- EOStateLayout: Maps between flat state vector and named streams
- EOSolver: Assembles and solves the full system
- EOSolveResult: Solution container with convergence info

Key function:
- solve_residual_system: Section-scope entry point for callers that already
  have a residual r(z; args) = 0 and do not want a Flowsheet built around it
  (#196).

The EO approach is advantageous for:
- Tightly coupled recycle loops (fewer iterations)
- Sensitivity analysis (full Jacobian available)
- Implicit differentiation through the solution (via optimistix)

Usage:
    solver = EOSolver(flowsheet)
    result = solver.solve(tol=1e-8)
    # or via Flowsheet:
    streams = flowsheet.solve_eo()
"""

from typing import Any
from dataclasses import dataclass, field
import time

import jax.numpy as jnp
from jax import Array
import jax

from difflow.streams import Stream, make_stream, get_flows, get_species
from difflow.params_mixin import ParamsMixin
import optimistix as optx


def solve_residual_system(
    residual_fn,
    z0: Array,
    args: Any = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_steps: int = 200,
    feasible_tol: float | None = None,
) -> tuple[Array, Array, Array]:
    """Solve ``r(z; args) = 0`` with soft failure and implicit differentiation.

    This is the section-scope entry point (#196). ``EOSolver`` assembles a
    residual from a :class:`~difflow.flowsheet.Flowsheet`; some models --
    a counter-current equilibrium section, for instance -- already *are* a
    residual and do not need a Flowsheet built around them. Routing them
    through here rather than through a hand-rolled Newton loop is what makes
    the whole section one ``optimistix.root_find``, so:

    - the reverse-mode tape is constant size rather than proportional to
      stages times iterations (optimistix differentiates the converged
      solution implicitly, it does not tape the iteration);
    - the section Jacobian ``dr/dz`` is available as an ordinary
      ``jax.jacobian`` of ``residual_fn``, which is the object the
      linearization, back-off and estimation layers want;
    - recycle tears stop being a separate mechanism -- a tear is just another
      row of ``r``.

    SOFT FAILURE. Nothing is raised. One cannot raise from inside ``vmap`` or
    ``scan``, so failure is reported as a value: the returned residual norm
    and boolean flag are traced arrays that a caller can branch on with
    ``jnp.where``. A non-converged ``z`` is still returned, because under
    ``vmap`` the converged members of the batch have to come back too.

    TOLERANCE. ``rtol``/``atol`` default to 1e-12, far below any outer
    flowsheet tolerance (typically 1e-6 to 1e-8). Keep it that way. A loosely
    converged inner solve gives an implicit-function gradient that is exact
    for the solution manifold but inconsistent with the value the code
    actually returned, and the resulting finite-difference disagreement is
    very hard to diagnose after the fact.

    Args:
        residual_fn: Callable ``(z, args) -> r`` with ``r`` the same shape as
            ``z``. Must be JAX-traceable.
        z0: Initial guess. Scale it well; Newton is not globally convergent.
        args: Any pytree passed through to ``residual_fn``. Differentiable.
        rtol: Relative tolerance for the Newton solver.
        atol: Absolute tolerance for the Newton solver.
        max_steps: Maximum Newton steps before giving up (softly).
        feasible_tol: Residual infinity-norm below which the solution is
            called feasible. None uses ``max(atol, rtol) * 1e4``, i.e. four
            orders of margin above the solver tolerance, so a solve that
            merely stalled just short of ``atol`` is not reported as failed.

    Returns:
        ``(z, residual_norm, feasible)``:

        - ``z``: solution (or the last iterate, on failure);
        - ``residual_norm``: scalar infinity norm of ``residual_fn(z, args)``;
        - ``feasible``: scalar boolean array, True when ``z`` is finite and
          the norm is below ``feasible_tol``.

    Example:
        >>> import jax.numpy as jnp
        >>> f = lambda z, a: z ** 2 - a
        >>> z, norm, ok = solve_residual_system(f, jnp.array([1.0]), 2.0)
        >>> bool(ok), round(float(z[0]), 6)
        (True, 1.414214)
    """
    if feasible_tol is None:
        feasible_tol = max(atol, rtol) * 1e4

    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(
        residual_fn,
        solver,
        z0,
        args=args,
        max_steps=max_steps,
        throw=False,
    )
    z = sol.value
    residual = residual_fn(z, args)
    residual_norm = jnp.max(jnp.abs(residual))
    feasible = jnp.logical_and(
        jnp.all(jnp.isfinite(z)),
        residual_norm <= feasible_tol,
    )
    return z, residual_norm, feasible


@dataclass
class EOStateLayout:
    """Maps between a flat state vector and named streams.

    The state vector contains variables for all non-feed streams in
    the flowsheet. Feed streams are treated as parameters (known values),
    not unknowns.

    State vector layout:
        [stream_1_vars | stream_2_vars | ... ]
        Per stream: [F_species_1, ..., F_species_n, T, P]

    Attributes:
        species_order: List of species names
        stream_names: Sorted list of non-feed stream names (unknowns)
        n_per_stream: Number of variables per stream (n_species + 2)
    """
    species_order: list[str]
    stream_names: list[str]

    @property
    def n_per_stream(self) -> int:
        return len(self.species_order) + 2

    @property
    def total_vars(self) -> int:
        return len(self.stream_names) * self.n_per_stream

    def pack(self, streams: dict[str, Stream]) -> Array:
        """Flatten streams dict to a 1D state vector.

        Args:
            streams: Dictionary mapping stream names to Stream dicts.
                     Only streams in self.stream_names are packed.

        Returns:
            Flat JAX array of length total_vars
        """
        arrays = []
        for name in self.stream_names:
            stream = streams[name]
            for s in self.species_order:
                arrays.append(jnp.asarray(stream[f"F_{s}"]))
            arrays.append(jnp.asarray(stream["T"]))
            arrays.append(jnp.asarray(stream["P"]))
        return jnp.concatenate([jnp.atleast_1d(a) for a in arrays])

    def unpack(self, x: Array) -> dict[str, Stream]:
        """Reconstruct streams dict from flat state vector.

        Args:
            x: Flat array of length total_vars

        Returns:
            Dictionary mapping stream names to Stream dicts
        """
        streams = {}
        for i, name in enumerate(self.stream_names):
            start = i * self.n_per_stream
            flows = {}
            for j, s in enumerate(self.species_order):
                flows[s] = x[start + j]
            T = x[start + len(self.species_order)]
            P = x[start + len(self.species_order) + 1]
            streams[name] = make_stream(flows, T, P)
        return streams

    def stream_slice(self, stream_name: str) -> slice:
        """Get the slice of the state vector for a given stream.

        Args:
            stream_name: Name of the stream

        Returns:
            slice object for indexing into the state vector
        """
        idx = self.stream_names.index(stream_name)
        start = idx * self.n_per_stream
        return slice(start, start + self.n_per_stream)


@dataclass
class EOSolveResult:
    """Result from equation-oriented solve.

    Attributes:
        streams: All streams (feeds + solved unknowns)
        converged: Whether the solver converged
        residual_norm: L-infinity norm of the residual at solution
        n_iterations: Number of Newton iterations
        wall_time: Wall-clock time for the solve (seconds)
    """
    streams: dict[str, Stream]
    converged: bool
    residual_norm: float
    n_iterations: int
    wall_time: float


class EOSolver:
    """Equation-oriented solver for difflow flowsheets.

    Assembles all unit residuals and connectivity constraints into
    a single system F(x) = 0 and solves via Newton's method using
    optimistix.root_find, which provides implicit differentiation
    through the converged solution.
    """

    def __init__(self, flowsheet):
        """Initialize EO solver from a Flowsheet.

        Args:
            flowsheet: A difflow.flowsheet.Flowsheet instance
        """
        self.flowsheet = flowsheet
        self.species_order = flowsheet.species_order

        # Identify all stream names
        all_streams = set()
        for unit in flowsheet.units:
            all_streams.update(unit.inlet_names)
            all_streams.update(unit.outlet_names)

        # Feed streams are known — everything else is an unknown
        feed_names = set(flowsheet.feeds.keys())
        # Recycle destinations are unknowns (they get their values from
        # recycle sources, not from feeds)
        unknown_names = sorted(all_streams - feed_names)

        self.layout = EOStateLayout(
            species_order=self.species_order,
            stream_names=unknown_names,
        )
        self.feed_names = feed_names

    def _build_residual_fn(self):
        """Build the residual function F(x) = 0.

        Returns a function that takes (x, args) where x is the state
        vector and args contains feed streams and unit parameters.
        The function returns a residual vector of the same size as x.
        """
        layout = self.layout
        flowsheet = self.flowsheet
        species_order = self.species_order

        def residual_fn(x, args):
            feeds = args

            # Reconstruct all streams from state vector + feeds
            unknown_streams = layout.unpack(x)
            all_streams = dict(feeds)
            all_streams.update(unknown_streams)

            # Apply recycle connections: source -> dest
            # The recycle dest stream in the state vector should match
            # the recycle source stream computed by units.
            # This is handled implicitly: the dest stream IS in the
            # state vector, and the source stream will be computed by
            # units. The residual will enforce they match.

            # Collect residuals from each unit
            residuals = []

            for unit in flowsheet.units:
                # Gather inlet streams for this unit
                inlets = [all_streams[name] for name in unit.inlet_names]

                # Get outlet stream names and their current values
                outlet_names = unit.outlet_names

                # Check if unit has eo_residuals method
                op = unit.operation
                if hasattr(op, 'eo_residuals'):
                    outlets = [all_streams[name] for name in outlet_names]
                    unit_resid = op.eo_residuals(
                        inlets, outlets, **unit.params
                    )
                    residuals.append(unit_resid)
                else:
                    # Fallback: run unit forward and compute difference
                    # between computed outlets and state vector outlets
                    result = op(*inlets, **unit.params)

                    # Parse the result to get outlet streams
                    computed_outlets = _parse_unit_result(
                        result, outlet_names
                    )

                    # Residual = computed - current state
                    for name in outlet_names:
                        if name not in feed_names:
                            computed = computed_outlets[name]
                            current = all_streams[name]
                            for s in species_order:
                                residuals.append(
                                    jnp.atleast_1d(
                                        computed[f"F_{s}"] - current[f"F_{s}"]
                                    )
                                )
                            residuals.append(
                                jnp.atleast_1d(computed["T"] - current["T"])
                            )
                            residuals.append(
                                jnp.atleast_1d(computed["P"] - current["P"])
                            )

            return jnp.concatenate(residuals)

        return residual_fn

    def _solve_core(
        self,
        x0: Array,
        feeds: dict[str, Stream],
        tol: float = 1e-8,
        max_steps: int = 100,
    ):
        """Core solve routine that is JAX-traceable.

        Args:
            x0: Initial state vector
            feeds: Feed stream dictionary
            tol: Convergence tolerance
            max_steps: Maximum Newton iterations

        Returns:
            optimistix solution object
        """
        residual_fn = self._build_residual_fn()

        solver = optx.Newton(rtol=tol, atol=tol)
        sol = optx.root_find(
            residual_fn,
            solver,
            x0,
            args=feeds,
            max_steps=max_steps,
            throw=False,
        )
        return sol

    def solve_streams(
        self,
        initial_guess: dict[str, Stream] | None = None,
        use_sm_init: bool = True,
        tol: float = 1e-8,
        max_steps: int = 100,
    ) -> dict[str, Stream]:
        """Solve and return streams dict. JAX-traceable for use with jax.grad.

        Args:
            initial_guess: Initial values for unknown streams
            use_sm_init: If True and no initial_guess, run SM solver first
            tol: Convergence tolerance
            max_steps: Maximum Newton iterations

        Returns:
            Dictionary of all streams (feeds + solved)
        """
        # Build initial guess (not traced — happens before root_find)
        if initial_guess is not None:
            x0_streams = initial_guess
        elif use_sm_init:
            x0_streams = self._sm_init()
        else:
            x0_streams = self._feed_propagation_init()

        x0 = self.layout.pack(x0_streams)
        feeds = dict(self.flowsheet.feeds)

        sol = self._solve_core(x0, feeds, tol, max_steps)
        x_sol = sol.value

        # Reconstruct solution streams
        solved_streams = self.layout.unpack(x_sol)
        all_streams = dict(feeds)
        all_streams.update(solved_streams)
        return all_streams

    def solve(
        self,
        initial_guess: dict[str, Stream] | None = None,
        use_sm_init: bool = True,
        tol: float = 1e-8,
        max_steps: int = 100,
    ) -> EOSolveResult:
        """Solve the flowsheet using the EO approach.

        For use inside jax.grad, use solve_streams() instead.

        Args:
            initial_guess: Initial values for unknown streams.
                          If None, uses SM initialization or feed propagation.
            use_sm_init: If True and no initial_guess, run SM solver first
                        to get a good starting point.
            tol: Convergence tolerance for residual norm
            max_steps: Maximum Newton iterations

        Returns:
            EOSolveResult with solved streams and convergence info
        """
        t_start = time.time()

        # Build initial guess
        if initial_guess is not None:
            x0_streams = initial_guess
        elif use_sm_init:
            x0_streams = self._sm_init()
        else:
            x0_streams = self._feed_propagation_init()

        x0 = self.layout.pack(x0_streams)
        feeds = dict(self.flowsheet.feeds)

        sol = self._solve_core(x0, feeds, tol, max_steps)
        x_sol = sol.value

        # Reconstruct solution streams
        solved_streams = self.layout.unpack(x_sol)
        all_streams = dict(feeds)
        all_streams.update(solved_streams)

        # Compute final residual norm (not traced, so float() is fine)
        residual_fn = self._build_residual_fn()
        final_residual = residual_fn(x_sol, feeds)
        residual_norm = float(jnp.max(jnp.abs(final_residual)))

        wall_time = time.time() - t_start

        # Determine convergence
        converged = residual_norm < tol

        # Get iteration count from optimistix result
        n_iterations = int(sol.stats.get("num_steps", 0))

        return EOSolveResult(
            streams=all_streams,
            converged=converged,
            residual_norm=residual_norm,
            n_iterations=n_iterations,
            wall_time=wall_time,
        )

    def _sm_init(self) -> dict[str, Stream]:
        """Get initial guess by running the SM solver.

        Returns:
            Dictionary of stream values from SM solution
        """
        try:
            sm_streams = self.flowsheet.solve(
                tol=1e-6,
                max_iter=50,
                acceleration="anderson",
            )
            # Filter to only unknown streams
            return {
                name: sm_streams[name]
                for name in self.layout.stream_names
                if name in sm_streams
            }
        except Exception:
            return self._feed_propagation_init()

    def _feed_propagation_init(self) -> dict[str, Stream]:
        """Initialize by propagating feed values through units.

        For each unknown stream, tries to run units forward from feeds.
        Falls back to copying feed values for unreachable streams.

        Returns:
            Dictionary of initial stream values
        """
        streams = dict(self.flowsheet.feeds)

        # Try to run units in order, catching failures
        for unit in self.flowsheet.units:
            try:
                inlets = []
                for name in unit.inlet_names:
                    if name in streams:
                        inlets.append(streams[name])
                    else:
                        # Use a default stream
                        inlets.append(self._make_default_stream())

                result = unit.operation(*inlets, **unit.params)
                computed = _parse_unit_result(result, unit.outlet_names)
                streams.update(computed)
            except Exception:
                # If unit fails, fill outlets with defaults
                for name in unit.outlet_names:
                    if name not in streams:
                        streams[name] = self._make_default_stream()

        # Ensure all unknown streams have values
        for name in self.layout.stream_names:
            if name not in streams:
                streams[name] = self._make_default_stream()

        return {
            name: streams[name]
            for name in self.layout.stream_names
        }

    def _make_default_stream(self) -> Stream:
        """Create a default stream with small flows."""
        flows = {s: 0.01 for s in self.species_order}
        return make_stream(flows, 300.0, 101325.0)


def _parse_unit_result(
    result: Any,
    outlet_names: list[str],
) -> dict[str, Stream]:
    """Parse the return value of a unit operation into named streams.

    Handles the various return patterns:
    - Single stream (dict)
    - (stream, info) tuple
    - (stream1, stream2, info) tuple
    - (stream1, stream2) tuple
    - Multiple streams matching outlet_names

    Args:
        result: Return value from unit.__call__
        outlet_names: Expected outlet stream names

    Returns:
        Dictionary mapping outlet names to streams
    """
    if isinstance(result, dict) and "T" in result:
        # Single stream returned directly
        return {outlet_names[0]: result}

    if isinstance(result, tuple):
        if len(result) == len(outlet_names):
            # Exact match: each element is an outlet stream
            return dict(zip(outlet_names, result))
        elif len(result) == 2 and isinstance(result[1], dict) and "T" not in result[1]:
            # (stream, info) pattern - info dict without T key
            stream = result[0]
            if isinstance(stream, dict) and "T" in stream:
                return {outlet_names[0]: stream}
            elif isinstance(stream, tuple):
                return dict(zip(outlet_names, stream))
        elif len(result) == len(outlet_names) + 1:
            # Multiple streams + info dict at end
            if isinstance(result[-1], dict) and "T" not in result[-1]:
                return dict(zip(outlet_names, result[:-1]))

    # Last resort: try treating all non-dict-info items as streams
    streams = {}
    stream_idx = 0
    for item in (result if isinstance(result, tuple) else (result,)):
        if isinstance(item, dict) and "T" in item and stream_idx < len(outlet_names):
            streams[outlet_names[stream_idx]] = item
            stream_idx += 1

    if len(streams) == len(outlet_names):
        return streams

    raise ValueError(
        f"Could not parse unit result with {len(outlet_names)} outlets: "
        f"got {type(result)}"
    )
