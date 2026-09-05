"""The Swaney--Grossmann flexibility index.

.. math:: F = \\max\\ \\{\\delta \\ge 0 : \\psi(d, \\delta) \\le 0\\}

the largest scaling of the uncertainty set over which the design stays
feasible.  ``F >= 1`` means the design covers the envelope it was given;
``F = 0.6`` means it covers only 60% of it and will spend campaigns outside;
``F = 2`` means it is carrying margin that was paid for and may not be needed.

The number on its own is close to useless for redesign, so this module returns
three things with it: **which vertex** binds, **which constraint** binds
there, and **how much scaling each vertex individually tolerates**.  The last
of these is the one that changes decisions -- a single vertex far below the
rest names one direction of feed variability as the problem, and that is a
specification to renegotiate or a control to add, not a column to make taller.

Method.  For each vertex direction the inner value is a monotone function of
``delta`` in the usual case, so a bisection on ``delta`` finds the exact
scaling at which that direction leaves feasibility.  The index is the smallest
such scaling and the binding vertex is its argmin.  Because every evaluation
of ``psi`` already visits every vertex, the per-vertex bisection costs the
same as a single bisection on ``psi`` and yields the whole diagnosis.

Reference:
    Swaney and Grossmann, "An index for operational flexibility in chemical
    process design", AIChE J. 31 (1985) 621 and 631,
    doi:10.1002/aic.690310412, doi:10.1002/aic.690310413.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from difflow.flexibility.feasibility import (
    ModelFn, _n_constraints, _names, inner_value,
)
from difflow.flexibility.inner import DEFAULT_OPTIONS, SolverOptions
from difflow.flexibility.sets import (
    ControlSpec, UncertaintySet, as_control_spec, as_uncertainty_set,
)


@dataclass
class FlexibilityResult:
    """The index, and the diagnosis that makes it actionable.

    Attributes:
        index: ``F``, the largest feasible scaling of the set.
        limited_by_vertex: Index of the vertex that runs out first.
        critical_theta: That vertex, evaluated at ``delta = F``.
        binding_constraint: The constraint active there.
        binding_index: Its row in the constraint vector.
        controls: The re-optimized controls at that point.
        constraint_values: All constraints there.
        vertex_limits: Per-vertex limiting scaling, in ``signs()`` order.
        vertex_directions: The ``signs()`` matrix, so a limit can be read as a
            direction of parameter movement.
        saturated: True if every vertex survived to ``delta_max``, so ``F`` is
            only a lower bound.
        delta_max: The upper bound searched.
        nominal_feasible: Whether the design is feasible at the nominal point
            at all.  When False the index is zero and nothing else is
            meaningful.
        set: The uncertainty set.
        constraint_names: Names of the constraints.
        control_names: Names of the recourse variables.

    Example:
        >>> res.index, res.binding_constraint          # doctest: +SKIP
        (0.62, 'purity')
    """

    index: float
    limited_by_vertex: int
    critical_theta: np.ndarray
    binding_constraint: str
    binding_index: int
    controls: np.ndarray
    constraint_values: np.ndarray
    vertex_limits: np.ndarray
    vertex_directions: np.ndarray
    saturated: bool
    delta_max: float
    nominal_feasible: bool
    set: UncertaintySet
    constraint_names: tuple[str, ...] = ()
    control_names: tuple[str, ...] = ()

    @property
    def covers_envelope(self) -> bool:
        """Whether the design survives the stated envelope, ``F >= 1``."""
        return bool(self.index >= 1.0)

    def critical_point(self) -> dict[str, float]:
        """``{parameter name: value at the limiting realization}``."""
        return {n: float(v) for n, v in
                zip(self.set.names, np.atleast_1d(self.critical_theta))}

    def direction(self) -> dict[str, str]:
        """``{parameter: '+'/'-'/'.'}`` for the limiting vertex direction."""
        s = self.vertex_directions[self.limited_by_vertex]
        return {n: ("+" if v > 0 else "-" if v < 0 else ".")
                for n, v in zip(self.set.names, s)}

    def slack_vertices(self, factor: float = 1.5) -> list[int]:
        """Vertices with at least ``factor`` times the binding vertex's room.

        Args:
            factor: Ratio above the index to count as slack.

        Returns:
            Vertex indices.
        """
        return [i for i, v in enumerate(self.vertex_limits)
                if v >= factor * self.index]

    def summary(self) -> str:
        """The index, the binding direction, and the per-vertex table."""
        if not self.nominal_feasible:
            return ("flexibility index F = 0: the design is infeasible at the "
                    "nominal point, so there is no set to scale.\n"
                    f"  binding constraint: {self.binding_constraint}")
        head = (f"flexibility index F = {self.index:.4g}"
                + ("  (>= delta_max; search saturated)" if self.saturated
                   else ("  -- covers the stated envelope"
                         if self.covers_envelope else
                         f"  -- covers only {100 * self.index:.0f}% of the "
                         "stated envelope")))
        dirs = "".join(self.direction().values())
        lines = [
            head,
            f"  limited by vertex {self.limited_by_vertex} [{dirs}]: "
            f"{self.set.label(self.critical_theta)}",
            f"  binding constraint: {self.binding_constraint}",
        ]
        if len(self.control_names):
            lines.append("  controls there: " + ", ".join(
                f"{n}={v:.6g}" for n, v in
                zip(self.control_names, np.atleast_1d(self.controls))))
        lines.append(f"  {'vertex':<10s}{'direction':<14s}{'limit':>12s}")
        order = np.argsort(self.vertex_limits)
        for i in order:
            s = self.vertex_directions[i]
            dd = "".join("+" if v > 0 else "-" if v < 0 else "." for v in s)
            mark = "  <-- binds" if i == self.limited_by_vertex else ""
            lines.append(f"  {i:<10d}{dd:<14s}{self.vertex_limits[i]:12.4g}"
                         f"{mark}")
        return "\n".join(lines)

    def describe(self) -> str:
        """State the problem that was solved, then the answer."""
        lines = [
            "flexibility index  F = max{delta : psi(d, delta) <= 0}",
            f"  parameters : {self.set.n} ({len(self.vertex_limits)} vertices)",
            f"  recourse   : {len(self.control_names)} controls"
            + (" (none -- no re-optimization allowed)"
               if not self.control_names else ""),
            f"  constraints: {len(self.constraint_names)}",
            f"  searched   : delta in [0, {self.delta_max:g}]",
            "",
            self.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"FlexibilityResult(index={self.index:.5g}, "
                f"vertex={self.limited_by_vertex}, "
                f"binding={self.binding_constraint!r})")


def _bisect_limit(psi_of_delta: Callable[[Array], Array], delta_max: float,
                  steps: int) -> Array:
    """Largest ``delta`` in ``[0, delta_max]`` with ``psi <= 0``.

    Assumes ``psi`` is nondecreasing in ``delta``, which holds whenever
    growing the set cannot help -- true for a nested family of sets and a
    fixed design.  Runs a fixed number of bisection steps so that it is
    ``jit``- and ``vmap``-safe.
    """

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        feasible = psi_of_delta(mid) <= 0.0
        return (jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid)), None

    (lo, _), _ = lax.scan(body, (jnp.asarray(0.0), jnp.asarray(float(delta_max))),
                          None, length=int(steps))
    return lo


def vertex_limits(model_fn: ModelFn, d, uncertainty_set, controls=None, *,
                  delta_max: float = 4.0,
                  options: SolverOptions = DEFAULT_OPTIONS) -> Array:
    """Limiting scaling for each vertex direction, as a traceable array.

    Args:
        model_fn: ``f(d, u, theta) -> array``, feasible where ``<= 0``.
        d: Design vector.
        uncertainty_set: The set.
        controls: The recourse box, or ``None``.
        delta_max: Upper end of the bisection.
        options: Search settings; ``bisection_steps`` sets the resolution.

    Returns:
        A ``(n_vertices,)`` array of limiting scalings.
    """
    T = as_uncertainty_set(uncertainty_set)
    cs = as_control_spec(controls)
    d = jnp.asarray(d, dtype=float)
    signs = jnp.asarray(T.signs())
    dev = jnp.where(signs > 0, T.upper[None, :], -T.lower[None, :])
    step = jnp.asarray(signs != 0) * dev

    def limit_for(direction):
        def psi_of_delta(delta):
            theta = T.nominal + delta * direction
            val, _ = inner_value(model_fn, d, theta, cs, options)
            return val
        return _bisect_limit(psi_of_delta, delta_max, options.bisection_steps)

    return jax.vmap(limit_for)(step)


def flexibility_index(model_fn: ModelFn, d, uncertainty_set, controls=None, *,
                      delta_max: float = 4.0,
                      options: SolverOptions = DEFAULT_OPTIONS,
                      constraint_names: Sequence[str] | None = None,
                      ) -> FlexibilityResult:
    """The largest scaling of the uncertainty set the design survives.

    Args:
        model_fn: ``f(d, u, theta) -> array`` of constraint values, feasible
            where every entry is ``<= 0``.  With no controls, ``u`` arrives as
            a length-zero array.
        d: The design being rated.
        uncertainty_set: An :class:`~difflow.flexibility.sets.UncertaintySet`
            or a ``{name: (nominal, pm)}`` mapping.  ``delta = 1`` is this set.
        controls: The recourse variables --- a
            :class:`~difflow.flexibility.sets.ControlSpec`,
            ``{name: (lo, hi)}``, or ``None`` for no recourse.
        delta_max: Largest scaling searched.  If every vertex survives it, the
            result is reported as saturated and ``index`` is a lower bound.
        options: Search settings.
        constraint_names: Names for the rows of ``f``.

    Returns:
        A :class:`FlexibilityResult` carrying the index, the binding vertex,
        the binding constraint, and every vertex's own limit.

    Note:
        The bisection assumes the inner value is nondecreasing in ``delta``
        along each vertex direction.  That is the standard assumption behind
        the vertex characterization of the index; a constraint that is
        genuinely non-monotone in a parameter can hide a small infeasible
        island inside a feasible outer set, and no bisection will find it.
        :func:`~difflow.flexibility.stochastic.expected_feasibility` will.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.flexibility import flexibility_index
        >>> # u must exceed theta and must not exceed the design d = 2
        >>> f = lambda d, u, th: jnp.array([u[0] - d[0], th[0] - u[0]])
        >>> res = flexibility_index(f, [2.0], {"feed": (1.0, 0.5)},
        ...                         {"u": (-10.0, 10.0)})
        >>> bool(abs(res.index - 2.0) < 1e-3)   # (2 - 1) / 0.5
        True
    """
    T = as_uncertainty_set(uncertainty_set)
    cs = as_control_spec(controls)
    d = jnp.asarray(d, dtype=float)
    n_f = _n_constraints(model_fn, d, cs, T)
    c_names = _names(constraint_names, n_f, "f")

    nominal_val, u_nom = inner_value(model_fn, d, T.nominal, cs, options)
    nominal_feasible = bool(float(nominal_val) <= 0.0)
    if not nominal_feasible:
        f_nom = np.asarray(jnp.atleast_1d(model_fn(d, u_nom, T.nominal)),
                           dtype=float)
        j = int(np.argmax(f_nom))
        signs = T.signs()
        return FlexibilityResult(
            index=0.0, limited_by_vertex=0,
            critical_theta=np.asarray(T.nominal, dtype=float),
            binding_constraint=c_names[j], binding_index=j,
            controls=np.asarray(u_nom, dtype=float), constraint_values=f_nom,
            vertex_limits=np.zeros(signs.shape[0]), vertex_directions=signs,
            saturated=False, delta_max=float(delta_max),
            nominal_feasible=False, set=T, constraint_names=c_names,
            control_names=tuple(cs.names))

    limits = np.asarray(vertex_limits(model_fn, d, T, cs,
                                      delta_max=delta_max, options=options),
                        dtype=float)
    k = int(np.argmin(limits))
    F = float(limits[k])
    saturated = bool(F >= delta_max * (1.0 - 1e-9))

    theta_k = T.vertices(F)[k]
    _, u_k = inner_value(model_fn, d, theta_k, cs, options)
    f_k = np.asarray(jnp.atleast_1d(model_fn(d, u_k, theta_k)), dtype=float)
    j = int(np.argmax(f_k))
    return FlexibilityResult(
        index=F, limited_by_vertex=k,
        critical_theta=np.asarray(theta_k, dtype=float),
        binding_constraint=c_names[j], binding_index=j,
        controls=np.asarray(u_k, dtype=float), constraint_values=f_k,
        vertex_limits=limits, vertex_directions=T.signs(),
        saturated=saturated, delta_max=float(delta_max),
        nominal_feasible=True, set=T, constraint_names=c_names,
        control_names=tuple(cs.names))
