r"""A quadratic subproblem: the curvature the delta vectors leave on the table.

:mod:`difflow.planning.curvature` measures what a second-order model would buy.
This builds one and solves it.

The model stays a **QP**, not a QCQP: the objective picks up curvature, and the
constraint rows stay first order. That is a deliberate line. The objective is
linear in the block outputs, so the priced combination of one block's outputs
has a single Hessian and the whole correction collapses to one term per block:

.. math::

    p^{\\mathsf T} y \;\\approx\;
    p^{\\mathsf T} y_0 + p^{\\mathsf T} J\\,\\delta u
    + \\tfrac{1}{2}\\,\\delta u^{\\mathsf T} H_p\\, \\delta u,
    \\qquad H_p = \\sum_i p_i H_i .

Making the *rows* quadratic instead would turn each trust-region subproblem
into a nonconvex QCQP, and the subproblem being solvable to global optimality
is what every guarantee in this package rests on.

Which leaves the one thing that can still break it: :math:`H_p` need not be
definite. A reduced flowsheet or power-flow model is convex at some operating
points and not at others, so an exact-Hessian QP is nonconvex wherever the
underlying physics is. :func:`convexify` handles that by spectral
modification — the eigenvalues that point the wrong way are clipped, the rest
are kept — which is the standard SQP remedy and is *reported*, never silent.
A convexified model is no longer the true second-order model, and the thing
that keeps it honest is the same thing that keeps the first-order model
honest: the trust region, and an acceptance test against the caller's own
nonlinear blocks.

The QP is warm-started from the LP solution and falls back to it if the
nonlinear solve struggles, so a quadratic subproblem can only match or beat
the linear one. That is what makes ``model_order="quadratic"`` safe to switch
on.

See also:
    :func:`difflow.planning.curvature.check_model_order`, which says whether
    the curvature is worth taking before you take it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from difflow.planning.block import Block
from difflow.planning.curvature import hessian_of
from difflow.planning.lp import LPModel, LPSolution

__all__ = [
    "ConvexificationReport",
    "QPModel",
    "block_objective_hessian",
    "build_qp",
    "convexify",
]

#: Smallest eigenvalue a convexified Hessian is allowed to keep, relative to
#: its largest magnitude.  Zero would leave a singular model with no curvature
#: in the flat directions, which a QP solver handles badly.
MIN_CURVATURE = 1e-8


@dataclass
class ConvexificationReport:
    """What :func:`convexify` had to change, and by how much.

    Attributes:
        modified: Whether any eigenvalue was moved.
        n_flipped: Eigenvalues that pointed the wrong way for the sense.
        worst: The most badly signed eigenvalue in the original Hessian,
            in the convention of the *minimised* objective.  Zero or
            positive means the model was already convex.
        eigenvalues: The original eigenvalues, ascending.
    """

    modified: bool
    n_flipped: int
    worst: float
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> str:
        if not self.modified:
            return "convexification: not needed"
        return (f"convexification: {self.n_flipped} of "
                f"{self.eigenvalues.size} eigenvalues clipped "
                f"(worst {self.worst:.3e})")


def convexify(H: np.ndarray, sense: str = "min",
              min_curvature: float = MIN_CURVATURE,
              ) -> tuple[np.ndarray, ConvexificationReport]:
    """Clip a Hessian's wrong-signed eigenvalues, and say what was clipped.

    For ``sense="min"`` the returned matrix is positive definite; for
    ``"max"`` it is negative definite. Eigenvectors are untouched, so the
    directions the model curves in are preserved and only the *amount* of
    curvature in the offending ones is changed.

    This is not the true second-order model any more. It is the standard SQP
    compromise, and the trust region plus the acceptance test against the
    caller's nonlinear blocks are what stop it from being a licence to be
    wrong.

    Args:
        H: Symmetric Hessian.
        sense: ``"min"`` or ``"max"``.
        min_curvature: Floor on the kept eigenvalues, relative to the largest
            magnitude present.

    Returns:
        ``(H_convex, report)``.

    Raises:
        ValueError: On an unknown ``sense``.

    Example:
        >>> import numpy as np
        >>> H = np.array([[1.0, 0.0], [0.0, -2.0]])     # a saddle
        >>> Hc, rep = convexify(H, "min")
        >>> bool(np.all(np.linalg.eigvalsh(Hc) > 0)), rep.n_flipped
        (True, 1)
    """
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    eig, vec = np.linalg.eigh(H)

    # Work in the convention of the minimised objective, so "wrong-signed"
    # means the same thing either way.
    signed = eig if sense == "min" else -eig
    floor = max(min_curvature * max(float(np.max(np.abs(eig))), 1.0),
                min_curvature)
    bad = signed < floor
    report = ConvexificationReport(
        modified=bool(np.any(bad)), n_flipped=int(np.sum(bad)),
        worst=float(np.min(signed)) if signed.size else 0.0,
        eigenvalues=eig)
    if not report.modified:
        return H, report

    fixed = np.where(bad, floor, signed)
    if sense == "max":
        fixed = -fixed
    return (vec * fixed) @ vec.T, report


def block_objective_hessian(block: Block, prices: Mapping[str, float],
                            u0: np.ndarray | None = None,
                            theta: Mapping[str, Any] | None = None,
                            ) -> np.ndarray | None:
    """Hessian of one block's priced outputs, ``sum_i p_i H_i``.

    Args:
        block: The block.
        prices: Qualified-name prices, as the planner holds them. Prices on
            the block's *inputs* are ignored: a lever enters the objective
            linearly and has no curvature of its own.
        u0: Point. Defaults to ``block.u0``.
        theta: Parameter override.

    Returns:
        The ``(n_u, n_u)`` Hessian, or ``None`` when none of this block's
        outputs is priced — in which case there is no curvature to add and
        the caller should not pay for an AD pass to learn it.
    """
    weights = {}
    for name in block.y_names:
        price = prices.get(f"{block.name}.{name}")
        if price:
            weights[name] = float(price)
    if not weights:
        return None
    return np.asarray(hessian_of(block, weights, u0, theta), dtype=float)


@dataclass
class QPModel:
    """An :class:`~difflow.planning.lp.LPModel` with a quadratic objective.

    The problem solved is::

        minimize    c . x + 0.5 x' Q x
        subject to  A_eq x  = b_eq
                    A_ub x <= b_ub
                    lb <= x <= ub

    ``Q`` is nonzero only on the block-input columns: the outputs are tied to
    the inputs by the model rows, so all the curvature belongs to ``u``.

    Attributes:
        lp: The underlying linear model, whose ``c`` already carries the
            linear part of the expansion about ``u0``.
        Q: Symmetric quadratic term, shape ``(n_cols, n_cols)``.
        convexification: ``{block name: ConvexificationReport}`` for the
            blocks whose curvature had to be modified.
    """

    lp: LPModel
    Q: np.ndarray
    convexification: dict[str, ConvexificationReport] = field(
        default_factory=dict)

    @property
    def convexified(self) -> bool:
        """Did any block's curvature have to be clipped?"""
        return any(r.modified for r in self.convexification.values())

    def minimised(self, x: np.ndarray) -> float:
        """``c . x + 0.5 x' Q x`` — the objective the solver minimises.

        This is the internal convention: lower is better whatever the
        caller's ``sense``. :meth:`objective` is the one to report.
        """
        x = np.asarray(x, dtype=float)
        return float(self.lp.c @ x + 0.5 * x @ (self.Q @ x))

    def objective(self, x: np.ndarray) -> float:
        """The caller's objective at ``x``, sign- and offset-corrected.

        Matches :attr:`~difflow.planning.lp.LPSolution.objective`, so a
        quadratic subproblem's predicted value is directly comparable with a
        linear one's. Keeping the two conventions apart matters: the
        expansion constant is a constant of the *minimised* objective while
        ``objective_offset`` is in the caller's, and conflating them shifts
        the reported value by twice the constant.
        """
        return (self.lp.sense * self.minimised(x)
                + self.lp.objective_offset)

    def solve(self, x0: np.ndarray | None = None,
              lp_solution: LPSolution | None = None) -> LPSolution:
        """Solve the QP, warm-started from the LP.

        The QP is a refinement of the linear subproblem, never a replacement
        for it: the LP solution is the starting point, and it is also the
        answer returned whenever the nonlinear solve fails to improve on it.
        A quadratic subproblem can therefore only match or beat the linear
        one, which is what makes it safe to switch on.

        Args:
            x0: Starting point. Defaults to ``lp_solution.x``, else the LP's
                own solution.
            lp_solution: An already-computed solution of ``self.lp``, to save
                re-solving it.

        Returns:
            An :class:`~difflow.planning.lp.LPSolution` over the same columns.
            Its ``objective`` is the caller's objective under the *quadratic*
            model, which is what the trust-region acceptance test compares
            against.
        """
        from scipy.optimize import (
            Bounds, LinearConstraint, minimize,
        )

        base = lp_solution if lp_solution is not None else self.lp.solve()
        if not base.success or base.x is None:
            return base
        if self.lp.integer_cols:
            # A quadratic objective over integer columns is a MIQP, which the
            # HiGHS path here does not solve. Say so rather than silently
            # dropping the curvature or the integrality.
            raise NotImplementedError(
                "quadratic subproblems are not available for models with "
                "integer columns (piecewise SOS2 blocks); use "
                "model_order='linear' for this network")

        start = np.asarray(base.x if x0 is None else x0, dtype=float)
        lb = np.where(np.isfinite(self.lp.lb), self.lp.lb, -np.inf)
        ub = np.where(np.isfinite(self.lp.ub), self.lp.ub, np.inf)
        start = np.clip(start, lb, ub)

        constraints = []
        if self.lp.A_eq.size:
            constraints.append(
                LinearConstraint(self.lp.A_eq, self.lp.b_eq, self.lp.b_eq))
        if self.lp.A_ub.size:
            constraints.append(
                LinearConstraint(self.lp.A_ub, -np.inf, self.lp.b_ub))

        c, Q = self.lp.c, self.Q

        def fun(x):
            return float(c @ x + 0.5 * x @ (Q @ x))

        def jac(x):
            return c + Q @ x

        res = minimize(fun, start, jac=jac, hess=lambda x: Q,
                       bounds=Bounds(lb, ub), constraints=constraints,
                       method="trust-constr",
                       options={"gtol": 1e-10, "xtol": 1e-12,
                                "maxiter": 400, "verbose": 0})

        x = np.asarray(res.x, dtype=float) if res.x is not None else None
        if x is None:
            return base
        x = np.clip(x, lb, ub)

        # Keep the QP point only when it beats the LP point under the
        # quadratic model *and* satisfies the rows at least as well. The LP
        # point is always available and always row-feasible, so there is no
        # reason to accept anything worse.
        if self._row_violation(x) > self._row_violation(base.x) + 1e-8:
            return base
        if self.minimised(x) > self.minimised(base.x):
            return base

        lp_objective = float(c @ x)
        objective = self.objective(x)
        return LPSolution(
            model=self.lp, x=x, success=True, status=int(res.status),
            message=f"QP ({res.message})", lp_objective=lp_objective,
            objective=objective, duals={})

    def _row_violation(self, x: np.ndarray) -> float:
        """Worst absolute row violation at ``x``, equalities included."""
        worst = 0.0
        if self.lp.A_eq.size:
            worst = max(worst,
                        float(np.max(np.abs(self.lp.A_eq @ x - self.lp.b_eq))))
        if self.lp.A_ub.size:
            worst = max(worst,
                        float(np.max(self.lp.A_ub @ x - self.lp.b_ub)),
                        0.0)
        return worst


def build_qp(lp: LPModel, hessians: Mapping[str, np.ndarray],
             columns: Mapping[str, Sequence[str]],
             centers: Mapping[str, np.ndarray],
             sense: str = "min",
             convex: bool = True,
             min_curvature: float = MIN_CURVATURE) -> QPModel:
    """Attach block curvature to an assembled LP.

    The quadratic model is expanded about each block's linearisation point,
    and that expansion is folded into the standard form rather than carried
    separately::

        0.5 (u - u0)' H (u - u0)
            = 0.5 u' H u  -  u' H u0  +  0.5 u0' H u0

    so ``Q`` takes ``H``, the linear term ``-H u0`` is added to ``c``, and the
    constant goes into ``objective_offset``. Getting that wrong shifts the
    optimum rather than the reported value, which is why it is written out.

    Args:
        lp: The assembled linear subproblem. Not modified.
        hessians: ``{block name: H}`` in the block's own input order, in the
            convention of the *caller's* objective (before any maximisation
            sign fold).
        columns: ``{block name: qualified input names}``, to place ``H``.
        centers: ``{block name: u0}``, the expansion points.
        sense: ``"max"`` or ``"min"``, as passed to the planner.
        convex: Clip wrong-signed eigenvalues via :func:`convexify`. Setting
            this ``False`` keeps the exact Hessian and gives up global
            optimality of the subproblem; it exists so the difference can be
            measured, not because it is a reasonable default.
        min_curvature: Passed to :func:`convexify`.

    Returns:
        A :class:`QPModel`.

    Raises:
        KeyError: If a named block's columns are not in the LP.
    """
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    sign = -1.0 if sense == "max" else 1.0

    n = lp.n_cols
    Q = np.zeros((n, n))
    c = np.array(lp.c, dtype=float)
    offset = float(lp.objective_offset)
    reports: dict[str, ConvexificationReport] = {}

    for name, H in hessians.items():
        if H is None:
            continue
        H = np.asarray(H, dtype=float)
        if not np.all(np.isfinite(H)):
            continue
        if not np.any(H):
            # An exactly linear block. Convexification would floor its zero
            # eigenvalues up to `min_curvature` and hand the solver curvature
            # the model does not have, so skip it: there is nothing to add.
            continue
        idx = np.array([lp.col(q) for q in columns[name]], dtype=int)
        u0 = np.asarray(centers[name], dtype=float)

        # Fold the maximisation sign in before convexifying, so "convex"
        # always means convex for the problem that is actually solved.
        H_signed = sign * H
        if convex:
            H_signed, report = convexify(H_signed, "min", min_curvature)
            reports[name] = report

        Q[np.ix_(idx, idx)] += H_signed
        c[idx] -= H_signed @ u0
        offset += 0.5 * float(u0 @ (H_signed @ u0)) * lp.sense

    return QPModel(lp=replace(lp, c=c, objective_offset=offset), Q=Q,
                   convexification=reports)
