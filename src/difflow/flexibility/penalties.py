"""Two kinds of uncertainty, two different bills.

This is the module the rest of the package exists to support.

**Feed uncertainty is answerable.**  The composition of an ore-derived liquor
varies, but you find out what today's liquor is before you have to run it, and
the controls can be re-optimized against it.  What that variability costs is
therefore not the full swing of the constraint, it is whatever swing survives
after the controls have done their best --- and *that* is what the flexibility
function measures.  The difference between the two is the recourse credit, and
it is often most of the number.

**Parameter uncertainty is not answerable.**  An equilibrium constant is not
revealed before the campaign; it is simply not known.  No control move can be
scheduled against a constant you do not know, so the entire propagated swing
lands on the constraint and has to be bought as margin up front.  That is what
back-off is for, and it is sized here the same way
:mod:`difflow.planning.backoff` sizes it --- ``kappa * sigma`` from the
gradient of the constraint with respect to the parameters, through
:func:`difflow.uncertainty.propagate_covariance`.

Reporting the two separately is the deliverable.  A single "uncertainty
allowance" hides the one fact a designer can act on: feed penalty is bought
down with *controls and instrumentation*, parameter back-off is bought down
with *experiments*.  They are different purchase orders and they have
different prices.

The model here takes both kinds explicitly::

    model_fn(d, u, theta, phi) -> constraint values, feasible where <= 0

``theta`` is the feed envelope the controls may respond to; ``phi`` are the
parameters they cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.uncertainty import propagate_covariance

from difflow.flexibility.feasibility import (
    _n_constraints, _names, inner_value, vertex_values,
)
from difflow.flexibility.inner import DEFAULT_OPTIONS, SolverOptions
from difflow.flexibility.sets import (
    UncertaintySet, as_control_spec, as_uncertainty_set,
)


@dataclass
class PenaltyReport:
    """The feed penalty and the parameter back-off, side by side.

    All arrays are indexed by constraint, in the order ``model_fn`` returns
    them.

    Attributes:
        nominal: Constraint values at the nominal feed, nominal parameters,
            and the controls optimized there.
        feed_worst: Worst constraint value over the feed set *with* the
            controls re-optimized at each realization.
        feed_penalty: ``feed_worst - nominal``: the margin feed variability
            costs after recourse has done what it can.
        recourse_credit: What re-optimizing the controls saves --- the same
            worst case computed with the controls frozen at their nominal
            setting, minus ``feed_worst``.
        backoff: ``kappa * sigma`` from parameter covariance, at the critical
            feed realization with the controls held fixed.
        sigma: One-sigma constraint uncertainty from the parameters.
        required_margin: ``max(feed_penalty, 0) + backoff``: how far below
            zero the nominal design has to sit for the constraint to hold.
        psi_feed: The feasibility function over the feed set alone.
        critical_theta: The feed realization that produced ``psi_feed``.
        critical_controls: The controls re-optimized there.
        jacobian: ``d(constraint)/d(parameter)`` at that point.
        kappa: Coverage factor used for the back-off.
        parameter_names: Parameter names, in ``jacobian`` column order.
        constraint_names: Constraint names, in row order.
        feed_set: The feed uncertainty set.

    Example:
        >>> print(report.summary())                      # doctest: +SKIP
    """

    nominal: np.ndarray
    feed_worst: np.ndarray
    feed_penalty: np.ndarray
    recourse_credit: np.ndarray
    backoff: np.ndarray
    sigma: np.ndarray
    required_margin: np.ndarray
    psi_feed: float
    critical_theta: np.ndarray
    critical_controls: np.ndarray
    jacobian: np.ndarray
    kappa: float
    parameter_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    feed_set: UncertaintySet

    @property
    def dominant(self) -> dict[str, str]:
        """``{constraint: 'feed' | 'parameter' | 'neither'}``.

        Which of the two bills is the larger one for each constraint --- and
        so which lever, control or experiment, is worth pulling first.
        """
        out = {}
        for i, nm in enumerate(self.constraint_names):
            f = max(float(self.feed_penalty[i]), 0.0)
            p = float(self.backoff[i])
            out[nm] = ("neither" if max(f, p) <= 0.0 else
                       "feed" if f >= p else "parameter")
        return out

    def worst_case(self) -> np.ndarray:
        """``feed_worst + backoff``: the value to compare against zero."""
        return self.feed_worst + self.backoff

    def feasible(self, tol: float = 0.0) -> bool:
        """Whether every constraint clears both penalties at once.

        Args:
            tol: Slack allowed.

        Returns:
            True if ``max(feed_worst + backoff) <= tol``.
        """
        return bool(np.max(self.worst_case()) <= tol)

    def as_dict(self) -> dict[str, dict[str, float]]:
        """``{constraint: {'feed': ..., 'parameter': ..., 'total': ...}}``."""
        return {nm: {"feed": float(max(self.feed_penalty[i], 0.0)),
                     "parameter": float(self.backoff[i]),
                     "total": float(self.required_margin[i])}
                for i, nm in enumerate(self.constraint_names)}

    def summary(self) -> str:
        """The two-column table, plus the totals and the verdict."""
        lines = [
            f"uncertainty penalties (kappa = {self.kappa:g}), "
            f"feed set {self.feed_set.n_vertices} vertices, "
            f"{len(self.parameter_names)} parameters",
            f"  psi over the feed set = {self.psi_feed:.6g} at "
            f"{self.feed_set.label(self.critical_theta)}",
            f"  {'constraint':<22s}{'nominal':>11s}{'feed':>11s}"
            f"{'recourse':>11s}{'param':>11s}{'margin':>11s}  dominant",
        ]
        for i, nm in enumerate(self.constraint_names):
            lines.append(
                f"  {nm:<22s}{self.nominal[i]:11.4g}"
                f"{self.feed_penalty[i]:11.4g}"
                f"{self.recourse_credit[i]:11.4g}"
                f"{self.backoff[i]:11.4g}"
                f"{self.required_margin[i]:11.4g}"
                f"  {self.dominant[nm]}")
        tot_f = float(np.sum(np.clip(self.feed_penalty, 0.0, None)))
        tot_p = float(np.sum(self.backoff))
        lines += [
            f"  feed penalty is bought down with controls and instruments: "
            f"{tot_f:.5g} total",
            f"  parameter back-off is bought down with experiments: "
            f"{tot_p:.5g} total",
            f"  verdict: {'feasible' if self.feasible() else 'INFEASIBLE'} "
            "once both penalties are charged",
        ]
        return "\n".join(lines)

    def describe(self) -> str:
        """State what was computed, then the table."""
        return "\n".join([
            "penalty split: feed uncertainty (recourse) vs parameter "
            "uncertainty (back-off)",
            f"  feed parameters  : {list(self.feed_set.names)}",
            f"  model parameters : {list(self.parameter_names)}",
            f"  constraints      : {list(self.constraint_names)}",
            "",
            self.summary(),
        ])

    def __repr__(self) -> str:
        return (f"PenaltyReport(constraints={len(self.constraint_names)}, "
                f"psi_feed={self.psi_feed:.5g}, kappa={self.kappa:g})")


def uncertainty_penalties(model_fn: Callable[..., Array], d, feed_set,
                          controls=None, *,
                          parameters: Mapping[str, float],
                          covariance,
                          parameter_order: Sequence[str] | None = None,
                          kappa: float = 2.0,
                          options: SolverOptions = DEFAULT_OPTIONS,
                          constraint_names: Sequence[str] | None = None,
                          ) -> PenaltyReport:
    """Charge feed uncertainty and parameter uncertainty separately.

    Args:
        model_fn: ``f(d, u, theta, phi) -> array`` of constraint values,
            feasible where every entry is ``<= 0``.  ``theta`` is the feed
            realization, ``phi`` the model parameters in ``parameter_order``.
        d: The design being charged.
        feed_set: The feed envelope --- an
            :class:`~difflow.flexibility.sets.UncertaintySet` or a
            ``{name: (nominal, pm)}`` mapping.  Controls may respond to this.
        controls: The recourse variables, or ``None``.  With ``None`` the
            recourse credit is identically zero, which is the correct answer
            and usually an expensive one.
        parameters: ``{name: nominal value}`` for the parameters the controls
            cannot respond to.
        covariance: Their covariance matrix, ordered by ``parameter_order``.
            A one-dimensional array is read as a vector of variances.
        parameter_order: Parameter names in covariance order.  Defaults to
            ``list(parameters)``.
        kappa: Coverage factor for the back-off.  ``2.0`` is roughly 95% for
            one Gaussian constraint; use more for a joint statement.
        options: Search settings.
        constraint_names: Names for the rows of ``f``.

    Returns:
        A :class:`PenaltyReport`.

    Raises:
        ValueError: If ``covariance`` does not match ``parameter_order``.

    Note:
        The back-off is evaluated at the *critical feed realization* with the
        controls frozen there, not at the nominal point.  That is the
        conservative and the honest choice: the parameters are unknown at the
        moment the worst feed arrives, and the two do not take turns.

    Example:
        >>> import jax.numpy as jnp
        >>> f = lambda d, u, th, ph: jnp.array([ph[0] * th[0] - u[0],
        ...                                     u[0] - d[0]])
        >>> rep = uncertainty_penalties(
        ...     f, [2.0], {"feed": (1.0, 0.2)}, {"u": (0.0, 5.0)},
        ...     parameters={"K": 1.0}, covariance=[[0.01]])
        >>> rep.dominant                # f0 sees K, f1 does not
        {'f0': 'parameter', 'f1': 'feed'}
        >>> float(rep.backoff[0])       # kappa * |d f0/d K| * sigma_K
        0.24
    """
    T = as_uncertainty_set(feed_set)
    cs = as_control_spec(controls)
    d = jnp.asarray(d, dtype=float)

    order = list(parameter_order) if parameter_order is not None \
        else list(parameters)
    missing = [p for p in order if p not in parameters]
    if missing:
        raise KeyError(f"no nominal value for parameter(s) {missing}")
    phi0 = jnp.asarray([float(parameters[p]) for p in order])

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    cov = np.atleast_2d(cov)
    if cov.shape != (len(order), len(order)):
        raise ValueError(
            f"covariance has shape {cov.shape}, expected "
            f"{(len(order), len(order))} to match parameter_order {order}")

    def g(dd, u, theta):
        return jnp.atleast_1d(model_fn(dd, u, theta, phi0))

    n_f = _n_constraints(g, d, cs, T)
    c_names = _names(constraint_names, n_f, "f")

    # Nominal feed, nominal parameters, controls optimized there.
    _, u_nom = inner_value(g, d, T.nominal, cs, options)
    f_nom = np.asarray(g(d, u_nom, T.nominal), dtype=float)

    # Worst case over the feed set, controls re-optimized at each vertex.
    vals, us, verts = vertex_values(g, d, T, cs, 1.0, options)
    with_recourse = np.asarray(
        jax.vmap(lambda u, th: g(d, u, th))(us, verts), dtype=float)
    frozen = np.asarray(
        jax.vmap(lambda th: g(d, u_nom, th))(verts), dtype=float)

    feed_worst = with_recourse.max(axis=0)
    frozen_worst = frozen.max(axis=0)
    feed_penalty = feed_worst - f_nom
    recourse_credit = frozen_worst - feed_worst

    k = int(np.argmax(np.asarray(vals)))
    theta_star, u_star = verts[k], us[k]

    def parametric(p: dict) -> Array:
        phi = jnp.asarray([p[name] for name in order])
        return jnp.atleast_1d(model_fn(d, u_star, theta_star, phi))

    _, value_cov, jac = propagate_covariance(
        parametric, {p: float(parameters[p]) for p in order},
        jnp.asarray(cov), order)
    sigma = np.sqrt(np.clip(np.diag(np.asarray(value_cov)), 0.0, None))
    backoff = float(kappa) * sigma

    return PenaltyReport(
        nominal=f_nom, feed_worst=feed_worst, feed_penalty=feed_penalty,
        recourse_credit=recourse_credit, backoff=backoff, sigma=sigma,
        required_margin=np.clip(feed_penalty, 0.0, None) + backoff,
        psi_feed=float(np.max(np.asarray(vals))),
        critical_theta=np.asarray(theta_star, dtype=float),
        critical_controls=np.asarray(u_star, dtype=float),
        jacobian=np.atleast_2d(np.asarray(jac)), kappa=float(kappa),
        parameter_names=tuple(order), constraint_names=c_names,
        feed_set=T)
