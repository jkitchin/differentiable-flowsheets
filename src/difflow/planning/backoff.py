"""Coefficient covariance and principled constraint back-off.

The delta vectors are functions of the model parameters.  When those
parameters come from a reconciliation or an estimation they carry a
covariance, and that covariance propagates through the block Jacobian onto the
LP's coefficients — and hence onto the constraint values the plan is built to
respect.

That matters more than it sounds.  Stale or uncertain parameters do not
usually cost *optimality*; they cost *feasibility*.  A plan that predicts it
sits exactly on a specification will, under coefficient uncertainty, spend
roughly half its periods on the wrong side of it.  The remedy is not a tighter
model but a back-off sized by the propagated uncertainty::

    back-off = kappa * sigma,   sigma^2 = g . Sigma_theta . g^T

where ``g`` is the gradient of the constraint value with respect to the
uncertain parameters — again from AD, through the whole flowsheet.

This module is a thin, planning-shaped layer over
:func:`difflow.uncertainty.propagate_covariance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from difflow.planning.lp import Spec
from difflow.uncertainty import propagate_covariance


@dataclass
class BackOffResult:
    """Propagated constraint uncertainty and the back-off it implies.

    Attributes:
        spec_names: Spec names, in row order.
        values: Constraint values at the evaluation point.
        sigma: One-sigma uncertainty of each constraint value.
        backoff: ``kappa * sigma`` per spec.
        kappa: The coverage factor used.
        covariance: Full covariance of the constraint values.
        jacobian: ``d(constraint value)/d(parameter)``.
        param_order: Parameter names, in column order.
    """

    spec_names: list[str]
    values: np.ndarray
    sigma: np.ndarray
    backoff: np.ndarray
    kappa: float
    covariance: np.ndarray
    jacobian: np.ndarray
    param_order: list[str]

    def as_dict(self) -> dict[str, float]:
        """``{spec name: back-off}``."""
        return {n: float(v) for n, v in zip(self.spec_names, self.backoff)}

    def summary(self) -> str:
        lines = [f"Constraint back-off (kappa = {self.kappa:g})",
                 f"  {'spec':<28s}{'value':>12s}{'sigma':>12s}"
                 f"{'back-off':>12s}"]
        for i, n in enumerate(self.spec_names):
            lines.append(f"  {n:<28s}{self.values[i]:12.5g}"
                         f"{self.sigma[i]:12.5g}{self.backoff[i]:12.5g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"BackOffResult(specs={len(self.spec_names)}, "
                f"kappa={self.kappa:g})")


def _flatten_scalar_theta(theta: Mapping[str, Mapping[str, Any]]
                          ) -> tuple[list[str], list[tuple[str, str]]]:
    names, layout = [], []
    for b in sorted(theta):
        for k in sorted(theta[b]):
            v = np.atleast_1d(np.asarray(theta[b][k], dtype=float))
            if v.size == 1:
                names.append(f"{b}.{k}")
                layout.append((b, k))
    return names, layout


def constraint_backoff(planner, decisions, covariance,
                       param_order: Sequence[str] | None = None,
                       kappa: float = 2.0,
                       specs: Sequence[Spec] | None = None) -> BackOffResult:
    """Size a back-off for each spec from parameter covariance.

    Args:
        planner: A :class:`~difflow.planning.planner.DeltaBasePlanner`.  Its
            blocks must declare scalar ``theta`` parameters, and ``planner``
            (or the blocks) must hold their nominal values.
        decisions: The operating point to evaluate at — typically a plan.
        covariance: Parameter covariance matrix, ordered by ``param_order``.
        param_order: Parameter names as ``"<block>.<param>"``.  Defaults to
            every scalar parameter, sorted.
        kappa: Coverage factor.  ``2.0`` is roughly 95% for a scalar Gaussian
            constraint; use a larger value for a joint guarantee.
        specs: Which specs to size.  Defaults to the planner's own.

    Returns:
        A :class:`BackOffResult`.

    Example:
        >>> bo = constraint_backoff(planner, res.decisions, cov,
        ...                         ["ngl.UA", "ngl.eta"], kappa=2.0)
        >>> planner.specs = apply_backoff(planner.specs, bo)
    """
    theta = planner.theta
    if theta is None:
        theta = {b.name: b.theta for b in planner.network.blocks
                 if b.theta is not None}
    if not theta:
        raise ValueError(
            "no block declares scalar `theta` parameters, so there is nothing "
            "to propagate. Give the blocks a `theta` dict and an "
            "`fn(u, theta)` signature.")

    all_names, all_layout = _flatten_scalar_theta(theta)
    if param_order is None:
        param_order = all_names
    param_order = list(param_order)
    unknown = [p for p in param_order if p not in all_names]
    if unknown:
        raise KeyError(f"unknown parameter(s) {unknown}; available: {all_names}")
    layout = {n: l for n, l in zip(all_names, all_layout)}

    cov = np.atleast_2d(np.asarray(covariance, dtype=float))
    if cov.shape != (len(param_order), len(param_order)):
        raise ValueError(
            f"covariance has shape {cov.shape}, expected "
            f"{(len(param_order), len(param_order))} to match param_order")

    spec_list = list(planner.specs if specs is None else specs)
    if not spec_list:
        raise ValueError("there are no specs to size a back-off for")

    net = planner.evaluation_network
    base = {b: dict(d) for b, d in theta.items()}
    nominal = {p: float(np.asarray(base[layout[p][0]][layout[p][1]]))
               for p in param_order}

    def model(params: dict) -> Any:
        th = {b: dict(d) for b, d in base.items()}
        for p in param_order:
            b, k = layout[p]
            th[b][k] = params[p]
        values = net.evaluate(decisions, th).values
        return jnp.stack([
            sum(c * values[v] for v, c in s.coeffs.items()) for s in spec_list])

    values, value_cov, jac = propagate_covariance(
        model, nominal, jnp.asarray(cov), param_order)

    value_cov = np.asarray(value_cov)
    sigma = np.sqrt(np.clip(np.diag(value_cov), 0.0, None))
    return BackOffResult(
        spec_names=[s.name for s in spec_list],
        values=np.asarray(values), sigma=sigma, backoff=float(kappa) * sigma,
        kappa=float(kappa), covariance=value_cov, jacobian=np.asarray(jac),
        param_order=param_order)


def apply_backoff(specs: Sequence[Spec],
                  backoff: BackOffResult | Mapping[str, float]) -> list[Spec]:
    """Return copies of ``specs`` carrying the given back-offs.

    Back-off tightens a spec inside the LP but is *not* part of the promise
    the spec makes: :meth:`difflow.planning.lp.Spec.violation` still measures
    against the stated right-hand side, so eating into the margin is not
    scored as a violation.

    Args:
        specs: The specs to tighten.
        backoff: A :class:`BackOffResult` or ``{spec name: back-off}``.

    Returns:
        New :class:`~difflow.planning.lp.Spec` objects.
    """
    table = (backoff.as_dict() if isinstance(backoff, BackOffResult)
             else {k: float(v) for k, v in backoff.items()})
    out = []
    for s in specs:
        new = Spec(dict(s.coeffs), s.op, s.rhs, elastic=s.elastic,
                   penalty=s.penalty, name=s.name,
                   backoff=float(table.get(s.name, s.backoff)))
        out.append(new)
    return out
