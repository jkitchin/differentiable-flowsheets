"""The Halemane--Grossmann feasibility function.

.. math:: \\psi(d) = \\max_{\\theta \\in T}\\ \\min_u\\ \\max_j\\
          f_j(d, u, \\theta)

A design ``d`` is feasible over the whole set ``T`` exactly when
``psi(d) <= 0``: there exists, for *every* realization the set allows, some
setting of the controls that satisfies every constraint at once.  The value
itself is the worst residual left over after the controls have done all they
can, in the units of the constraint that binds.

Two solution paths are provided.

``method="vertex"`` evaluates the inner problem at every vertex of the box and
takes the largest.  This is the standard first approach and it is *exact*
whenever the critical realization is a vertex, which holds when each
``f_j`` is jointly quasi-convex in ``theta`` -- the usual situation for a
monotone process constraint over a feed envelope.  It parallelizes trivially
under ``vmap`` and its cost is ``2^n`` inner solves.

``method="continuous"`` then runs a projected ascent over ``theta`` inside the
box, seeded at the best vertex.  This is the general fallback: it converges to
a KKT point of the outer maximization and so can find a critical realization
*interior* to the set, which is what vertex enumeration structurally cannot
do.  Because it is seeded at the vertex answer it never reports less than the
vertex answer, and because it is a local ascent it is a lower bound on the
true ``psi`` -- both facts are in the direction of honesty about which way the
error goes, but neither makes it a global guarantee.

Reference:
    Halemane and Grossmann, "Optimal process design under uncertainty",
    AIChE J. 29 (1983) 425, doi:10.1002/aic.690290312.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.flexibility.inner import (
    DEFAULT_OPTIONS, SolverOptions, minimax_value,
)
from difflow.flexibility.sets import (
    ControlSpec, UncertaintySet, as_control_spec, as_uncertainty_set,
)

ModelFn = Callable[[Array, Array, Array], Array]

METHODS = ("vertex", "continuous")


@dataclass
class FeasibilityResult:
    """``psi(d)``, and the realization that produced it.

    The number says whether the design is feasible; the rest of the fields say
    *why*, which is what a redesign needs.

    Attributes:
        psi: The feasibility function value.  Feasible iff ``<= 0``.
        feasible: ``psi <= tol``.
        critical_theta: The worst-case parameter realization.
        critical_vertex: Index of the worst vertex, or ``-1`` if the
            continuous search moved off the vertex.
        binding_constraint: Name of the constraint attaining the maximum.
        binding_index: Its position in the constraint vector.
        controls: The re-optimized controls at the critical realization.
        constraint_values: All ``f_j`` at the critical realization.
        vertex_psi: Inner value at each vertex, in ``signs()`` order.
        vertices: The vertices evaluated.
        scale: The set scaling used.
        method: ``"vertex"`` or ``"continuous"``.
        set: The uncertainty set.
        control_names: Names of the recourse variables.
        constraint_names: Names of the constraints.

    Example:
        >>> res.feasible, res.binding_constraint          # doctest: +SKIP
        (False, 'purity')
        >>> print(res.summary())                          # doctest: +SKIP
    """

    psi: float
    feasible: bool
    critical_theta: np.ndarray
    critical_vertex: int
    binding_constraint: str
    binding_index: int
    controls: np.ndarray
    constraint_values: np.ndarray
    vertex_psi: np.ndarray
    vertices: np.ndarray
    scale: float
    method: str
    set: UncertaintySet
    control_names: tuple[str, ...] = ()
    constraint_names: tuple[str, ...] = ()

    @property
    def margin(self) -> float:
        """How much room is left, ``-psi``.  Negative when infeasible."""
        return -float(self.psi)

    def critical_point(self) -> dict[str, float]:
        """``{parameter name: critical value}``."""
        return {n: float(v) for n, v in
                zip(self.set.names, np.atleast_1d(self.critical_theta))}

    def active_constraints(self, tol: float = 1e-6) -> list[str]:
        """Constraints within ``tol`` of the worst one at the critical point.

        Args:
            tol: Absolute slack below the maximum to still count as active.

        Returns:
            Constraint names, worst first.
        """
        f = np.atleast_1d(self.constraint_values)
        order = np.argsort(-f)
        return [self.constraint_names[i] for i in order
                if f[i] >= f.max() - tol]

    def summary(self) -> str:
        """A verdict, the critical realization, and the constraint table."""
        verdict = "FEASIBLE" if self.feasible else "INFEASIBLE"
        where = ("interior point" if self.critical_vertex < 0
                 else f"vertex {self.critical_vertex}")
        lines = [
            f"psi = {self.psi:.6g}  ->  {verdict} over the set at "
            f"scale {self.scale:g}  [{self.method}]",
            f"  critical realization ({where}): "
            f"{self.set.label(self.critical_theta)}",
        ]
        if len(self.control_names):
            lines.append("  controls re-optimized to: " + ", ".join(
                f"{n}={v:.6g}" for n, v in
                zip(self.control_names, np.atleast_1d(self.controls))))
        else:
            lines.append("  no recourse: controls could not respond")
        lines.append(f"  binding constraint: {self.binding_constraint}")
        lines.append(f"  {'constraint':<24s}{'value':>14s}{'slack':>14s}")
        for i, nm in enumerate(self.constraint_names):
            v = float(self.constraint_values[i])
            mark = "  <-- binding" if i == self.binding_index else ""
            lines.append(f"  {nm:<24s}{v:14.5g}{-v:14.5g}{mark}")
        return "\n".join(lines)

    def describe(self) -> str:
        """State the problem that was solved, then the answer."""
        lines = [
            "feasibility function  psi(d) = max_theta min_u max_j f_j",
            f"  parameters : {self.set.n} "
            f"({self.set.n_vertices} vertices, scale {self.scale:g})",
            f"  recourse   : {len(self.control_names)} controls"
            + (" (none -- no re-optimization allowed)"
               if not self.control_names else ""),
            f"  constraints: {len(self.constraint_names)}",
            f"  method     : {self.method}",
            "",
            self.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"FeasibilityResult(psi={self.psi:.5g}, "
                f"feasible={self.feasible}, "
                f"binding={self.binding_constraint!r}, method={self.method!r})")


def _n_constraints(model_fn: ModelFn, d, controls: ControlSpec,
                   uncertainty_set: UncertaintySet) -> int:
    u0 = controls.starts(1)[0]
    f = jnp.atleast_1d(model_fn(jnp.asarray(d), u0, uncertainty_set.nominal))
    return int(f.shape[0])


def _names(given: Sequence[str] | None, n: int, stem: str) -> tuple[str, ...]:
    if given is None:
        return tuple(f"{stem}{i}" for i in range(n))
    given = tuple(given)
    if len(given) != n:
        raise ValueError(f"expected {n} {stem} names, got {len(given)}")
    return given


def inner_value(model_fn: ModelFn, d, theta, controls: ControlSpec,
                options: SolverOptions = DEFAULT_OPTIONS
                ) -> tuple[Array, Array]:
    """``min_u max_j f_j(d, u, theta)`` at one realization.

    Args:
        model_fn: ``f(d, u, theta) -> array``, feasible where every entry
            is ``<= 0``.
        d: Design vector.
        theta: The realized parameters.
        controls: The recourse box.
        options: Search settings.

    Returns:
        ``(value, u_star)``.
    """
    d = jnp.asarray(d, dtype=float)
    theta = jnp.asarray(theta, dtype=float)
    return minimax_value(lambda u: jnp.atleast_1d(model_fn(d, u, theta)),
                         controls, options)


def vertex_values(model_fn: ModelFn, d, uncertainty_set: UncertaintySet,
                  controls: ControlSpec, scale=1.0,
                  options: SolverOptions = DEFAULT_OPTIONS
                  ) -> tuple[Array, Array, Array]:
    """The inner value at every vertex of the scaled box, in one ``vmap``.

    Args:
        model_fn: ``f(d, u, theta) -> array``.
        d: Design vector.
        uncertainty_set: The set to enumerate.
        controls: The recourse box.
        scale: Set scaling ``delta``; may be traced.
        options: Search settings.

    Returns:
        ``(values, controls_at_each_vertex, vertices)``.
    """
    verts = uncertainty_set.vertices(scale)

    def one(theta):
        return inner_value(model_fn, d, theta, controls, options)

    vals, us = jax.vmap(one)(verts)
    return vals, us, verts


def feasibility_value(model_fn: ModelFn, d, uncertainty_set, controls=None,
                      scale=1.0, options: SolverOptions = DEFAULT_OPTIONS
                      ) -> Array:
    """``psi(d)`` as a bare traceable scalar, by vertex enumeration.

    This is the ``jit``/``grad``/``vmap`` entry point.  Use
    :func:`feasibility_function` when you want the diagnosis with it.

    Args:
        model_fn: ``f(d, u, theta) -> array``, feasible where every entry
            is ``<= 0``.  With no controls, ``u`` is a length-zero array.
        d: Design vector.
        uncertainty_set: An :class:`~difflow.flexibility.sets.UncertaintySet`
            or a ``{name: (nominal, pm)}`` mapping.
        controls: A :class:`~difflow.flexibility.sets.ControlSpec`,
            ``{name: (lo, hi)}``, or ``None`` for no recourse.
        scale: Set scaling ``delta``; may be traced.
        options: Search settings.

    Returns:
        A scalar ``psi``.  Feasible over the set iff ``psi <= 0``.

    Note:
        The derivative with respect to ``d`` is the envelope-theorem
        derivative evaluated at the critical realization: the outer ``max``
        selects ``theta*``, and the inner value contributes the
        multiplier-weighted sum ``sum_j lambda_j df_j/dd`` at ``u*`` (see
        :func:`difflow.flexibility.inner.constraint_multipliers` for why the
        weights, and not the single active row, are what is correct at the
        kink).  That is the derivative of the max-min wherever the critical
        realization is unique, which is almost everywhere, and it is exactly
        the sensitivity a designer wants -- how much the worst case moves per
        unit of design change.

    Example:
        >>> import jax
        >>> g = jax.grad(lambda d: feasibility_value(f, d, T, u))  # doctest: +SKIP
    """
    T = as_uncertainty_set(uncertainty_set)
    cs = as_control_spec(controls)
    vals, _, _ = vertex_values(model_fn, d, T, cs, scale, options)
    return jnp.max(vals)


def _refine_continuous(model_fn: ModelFn, d, T: UncertaintySet,
                       cs: ControlSpec, scale, theta0, u0,
                       options: SolverOptions):
    """Projected ascent on ``theta`` inside the box, seeded at a vertex."""
    from difflow.flexibility.inner import box_adam

    lo, hi = T.bounds(scale)
    warm = cs.update(start=u0) if cs.n else cs
    inner_opts = options.update(n_starts=1)

    def objective(theta, _progress):
        val, _ = inner_value(model_fn, d, theta, warm, inner_opts)
        return -val

    return box_adam(objective, jnp.asarray(theta0, dtype=float), lo, hi,
                    options.outer_steps, options.learning_rate)


def feasibility_function(model_fn: ModelFn, d, uncertainty_set, controls=None,
                         *, scale: float = 1.0, method: str = "vertex",
                         options: SolverOptions = DEFAULT_OPTIONS,
                         constraint_names: Sequence[str] | None = None,
                         tol: float = 0.0) -> FeasibilityResult:
    """Evaluate ``psi(d)`` and report the realization that produced it.

    Args:
        model_fn: ``f(d, u, theta) -> array`` of constraint values, written so
            that ``f_j <= 0`` means constraint ``j`` is satisfied.  With no
            controls, ``u`` arrives as a length-zero array and is ignored.
        d: The design being tested.
        uncertainty_set: An :class:`~difflow.flexibility.sets.UncertaintySet`
            or a ``{name: (nominal, pm)}`` / ``{name: (nominal, minus, plus)}``
            mapping.
        controls: The recourse variables --- a
            :class:`~difflow.flexibility.sets.ControlSpec`,
            ``{name: (lo, hi)}``, or ``None`` for a design that cannot
            respond at all.
        scale: Scaling of the uncertainty set to test at.
        method: ``"vertex"`` (default) or ``"continuous"``.  See the module
            documentation for when the vertex answer is exact.
        options: Search settings.
        constraint_names: Names for the rows of ``f``, used in reports.
        tol: Slack allowed when calling the design feasible.

    Returns:
        A :class:`FeasibilityResult`.

    Raises:
        ValueError: If ``method`` is not one of :data:`METHODS`.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.flexibility import feasibility_function
        >>> # u must exceed theta, and must not exceed the design d
        >>> f = lambda d, u, th: jnp.array([u[0] - d[0], th[0] - u[0]])
        >>> res = feasibility_function(
        ...     f, [2.0], {"feed": (1.0, 0.5)}, {"u": (-10.0, 10.0)})
        >>> res.feasible
        True
        >>> float(res.critical_theta[0])      # the rich end of the envelope
        1.5
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    T = as_uncertainty_set(uncertainty_set)
    cs = as_control_spec(controls)
    d = jnp.asarray(d, dtype=float)
    n_f = _n_constraints(model_fn, d, cs, T)
    c_names = _names(constraint_names, n_f, "f")

    vals, us, verts = vertex_values(model_fn, d, T, cs, scale, options)
    vals = np.asarray(vals, dtype=float)
    k = int(np.argmax(vals))
    theta_star = np.asarray(verts[k], dtype=float)
    u_star = np.asarray(us[k], dtype=float)
    vertex_index = k

    if method == "continuous":
        theta_c = _refine_continuous(model_fn, d, T, cs, scale,
                                     verts[k], us[k], options)
        val_c, u_c = inner_value(model_fn, d, theta_c, cs, options)
        if float(val_c) > vals[k]:
            theta_new = np.asarray(theta_c, dtype=float)
            on_vertex = bool(np.allclose(theta_new, theta_star, rtol=1e-8,
                                         atol=1e-10))
            theta_star, u_star = theta_new, np.asarray(u_c, dtype=float)
            vertex_index = k if on_vertex else -1
            psi = float(val_c)
        else:
            psi = float(vals[k])
    else:
        psi = float(vals[k])

    f_star = np.asarray(
        jnp.atleast_1d(model_fn(d, jnp.asarray(u_star), jnp.asarray(theta_star))),
        dtype=float)
    j = int(np.argmax(f_star))
    return FeasibilityResult(
        psi=psi, feasible=bool(psi <= tol), critical_theta=theta_star,
        critical_vertex=vertex_index, binding_constraint=c_names[j],
        binding_index=j, controls=u_star, constraint_values=f_star,
        vertex_psi=vals, vertices=np.asarray(verts, dtype=float),
        scale=float(scale), method=method, set=T,
        control_names=tuple(cs.names), constraint_names=c_names)
