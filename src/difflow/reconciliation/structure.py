"""Observability, redundancy and solvability of a reconciliation problem.

Whether the KKT system can be solved and whether the unmeasured
variables can be observed are *the same question*. For

.. math::

    K = \\begin{bmatrix} W & A^T \\\\ A & 0 \\end{bmatrix},
    \\qquad W \\succeq 0,

:math:`K` is nonsingular if and only if :math:`A` has full row rank and
:math:`Z^T W Z \\succ 0` on a basis :math:`Z` of :math:`\\ker A`. With
:math:`W` diagonal, positive on measured entries and zero elsewhere,
the second condition collapses to

    **the unmeasured columns** :math:`A_U` **must have full column rank**

which is exactly the classical observability condition of data
reconciliation. So this module runs *before* the solve: a problem that
would produce a singular KKT matrix raises
:class:`ReconciliationStructureError` naming the variables responsible,
instead of returning NaN.

Ranks and variable classes are discrete, so they are not differentiable
and do not pretend to be: this module works in NumPy, mirroring the
split in :mod:`difflow.estimation.confidence`, which takes derivatives
with JAX and then does statistics with SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow.reconciliation.core import (
    Scaling,
    jacobian_of,
    measured_mask,
    reconciled_covariance,
)

#: measured, and checked by at least one constraint
MEASURED_REDUNDANT = "measured-redundant"
#: measured, but nothing constrains it -- it passes through unchanged
MEASURED_JUST_DETERMINED = "measured-just-determined"
#: unmeasured, but determined by the constraints
UNMEASURED_OBSERVABLE = "unmeasured-observable"
#: unmeasured and not determined -- the problem is ill-posed
UNMEASURED_UNOBSERVABLE = "unmeasured-unobservable"

#: below this, a measurement is treated as carrying no redundancy
REDUNDANCY_TOL = 1e-8


class ReconciliationStructureError(ValueError):
    """The reconciliation problem is structurally unsolvable.

    Raised when the unmeasured variables cannot be determined from the
    constraints, or when the constraints are linearly dependent. The
    message names the variables involved.
    """


@dataclass
class StructureReport(ParamsMixin):
    """Observability and redundancy structure of a reconciliation problem.

    Attributes:
        classes: variable name -> one of the four class constants.
        degree_of_redundancy: ``m - rank(A_U)``; the degrees of freedom
            of the global chi-squared test.
        redundancy: measured variable -> ``1 - Var(x_hat_i)/sigma_i^2``
            in ``[0, 1]``. Zero means the measurement is unchecked.
        rank_A, rank_A_unmeasured: numerical ranks.
        singular_values_A, singular_values_A_unmeasured: full spectra,
            so a marginal case can be inspected rather than guessed at.
        rank_tol: the threshold used.
        rank_gap: ratio of the smallest retained to the largest
            discarded singular value of ``A_U``; small means the
            classification is tolerance-dependent.
        solvable: whether the KKT system is nonsingular.
        reason: empty when solvable, else a short diagnosis.
        unobservable: names of the unmeasured variables implicated in a
            null-space direction.
    """

    classes: dict[str, str]
    degree_of_redundancy: int
    redundancy: dict[str, float]
    rank_A: int
    rank_A_unmeasured: int
    n_equations: int
    n_measured: int
    n_unmeasured: int
    singular_values_A: np.ndarray
    singular_values_A_unmeasured: np.ndarray
    rank_tol: float
    rank_gap: float
    solvable: bool
    reason: str = ""
    unobservable: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def raise_if_unsolvable(self) -> None:
        """Raise :class:`ReconciliationStructureError` if not solvable."""
        if self.solvable:
            return
        detail = ""
        if self.unobservable:
            detail = f" Unobservable: {', '.join(self.unobservable)}."
        raise ReconciliationStructureError(
            f"reconciliation problem is not solvable: {self.reason}.{detail} "
            f"({self.n_equations} equations, {self.n_measured} measured and "
            f"{self.n_unmeasured} unmeasured variables; "
            f"rank(A)={self.rank_A}, rank(A_unmeasured)="
            f"{self.rank_A_unmeasured}). Measure one of the variables above, "
            "give it a finite sigma as a prior, or remove it from the state."
        )

    def summary(self) -> str:
        """Human-readable table of the classification."""
        lines = [
            f"degrees of redundancy : {self.degree_of_redundancy}",
            f"equations             : {self.n_equations}",
            f"measured / unmeasured : {self.n_measured} / {self.n_unmeasured}",
            f"solvable              : {self.solvable}"
            + (f"  ({self.reason})" if self.reason else ""),
            "",
            f"{'variable':<20} {'class':<28} {'redundancy':>10}",
            "-" * 60,
        ]
        for name in self.names:
            red = self.redundancy.get(name)
            red_s = f"{red:10.3f}" if red is not None else " " * 10
            lines.append(f"{name:<20} {self.classes[name]:<28} {red_s}")
        return "\n".join(lines)


def _rank_and_spectrum(a: np.ndarray, tol: float | None):
    """Numerical rank of ``a`` by SVD, with the spectrum and threshold.

    SVD, never the eigenvalues of ``a.T @ a``: squaring the matrix
    squares its condition number, so a structurally zero singular value
    reappears near ``sqrt(eps) * sigma_max`` and an unobservable system
    is reported as full rank.
    """
    if a.size == 0 or a.shape[1] == 0:
        return 0, np.zeros(0), (tol or 0.0), float("inf")
    s = np.linalg.svd(a, compute_uv=False)
    if tol is None:
        # sqrt(eps), not eps: A is evaluated at an approximate solution,
        # so structural zeros are only zero to about that.
        tol = max(a.shape) * np.sqrt(np.finfo(float).eps) * s[0]
    rank = int((s > tol).sum())
    if 0 < rank < len(s):
        gap = float(s[rank - 1] / s[rank]) if s[rank] > 0 else float("inf")
    else:
        gap = float("inf")
    return rank, s, float(tol), gap


def classify(
    residual_fn: Callable,
    x: Array,
    sigma: Array,
    *,
    scaling: Scaling,
    params: Any = None,
    names: Sequence[str] | None = None,
    rank_tol: float | None = None,
    covariance: Array | None = None,
) -> StructureReport:
    """Classify every variable and decide whether the problem is solvable.

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        x: point to linearize about.
        sigma: standard deviations; ``inf`` = unmeasured.
        scaling: the scaling the problem is solved in; the Jacobian is
            equilibrated with it before any rank test, so the answer
            does not depend on the unit system.
        params: extra argument threaded to ``residual_fn``.
        names: variable names, defaulting to ``x0, x1, ...``.
        rank_tol: override the singular-value threshold.
        covariance: precomputed reconciled covariance, to avoid solving
            the KKT system twice.

    Returns:
        A :class:`StructureReport`. It does *not* raise; call
        :meth:`StructureReport.raise_if_unsolvable` for that.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    names = list(names) if names is not None else [f"x{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"got {len(names)} names for {n} variables")

    mask = np.asarray(measured_mask(sigma))
    a = np.asarray(jacobian_of(residual_fn, x, params), dtype=float)
    # Equilibrate rows and columns before any rank decision.
    a_s = np.asarray(scaling.r)[:, None] * a * np.asarray(scaling.d)[None, :]
    m = a_s.shape[0]

    rank_a, sv_a, tol, _ = _rank_and_spectrum(a_s, rank_tol)
    a_u = a_s[:, ~mask]
    n_u = int((~mask).sum())
    rank_u, sv_u, _, gap = _rank_and_spectrum(a_u, tol)

    unobservable: list[str] = []
    if n_u and rank_u < n_u:
        # Right singular vectors below the threshold span the null
        # space; a direction implicates a *set* of variables, not one.
        _, _, vh = np.linalg.svd(a_u, full_matrices=True)
        u_names = [nm for nm, mk in zip(names, mask) if not mk]
        implicated: set[str] = set()
        for k in range(rank_u, len(u_names)):
            vec = np.abs(vh[k])
            if vec.max() <= 0:
                continue
            for j in np.where(vec > 0.1 * vec.max())[0]:
                implicated.add(u_names[j])
        unobservable = [nm for nm in u_names if nm in implicated]

    reasons = []
    if rank_a < m:
        reasons.append(
            f"the {m} constraints are linearly dependent (rank {rank_a})"
        )
    if n_u and rank_u < n_u:
        reasons.append(
            f"{n_u - rank_u} unmeasured variable(s) cannot be determined "
            "from the constraints"
        )
    solvable = not reasons

    redundancy: dict[str, float] = {}
    classes: dict[str, str] = {}
    if solvable:
        if covariance is None:
            covariance = reconciled_covariance(
                residual_fn, x, sigma, scaling=scaling, params=params
            )
        var = np.diag(np.asarray(covariance, dtype=float))
        sig = np.asarray(sigma, dtype=float)
        for i, nm in enumerate(names):
            if mask[i]:
                # Var(x_hat_i - y_i) = sigma_i^2 - Var(x_hat_i); zero
                # means nothing checks this measurement, which is the
                # same quantity the measurement test divides by, so
                # "non-redundant" and "not testable" cannot disagree.
                red = float(1.0 - var[i] / (sig[i] ** 2))
                redundancy[nm] = min(max(red, 0.0), 1.0)
                classes[nm] = (
                    MEASURED_REDUNDANT
                    if red > REDUNDANCY_TOL
                    else MEASURED_JUST_DETERMINED
                )
            else:
                classes[nm] = UNMEASURED_OBSERVABLE
    else:
        for i, nm in enumerate(names):
            if mask[i]:
                classes[nm] = MEASURED_REDUNDANT
            elif nm in unobservable:
                classes[nm] = UNMEASURED_UNOBSERVABLE
            else:
                classes[nm] = UNMEASURED_OBSERVABLE

    reason = "; ".join(reasons)
    if solvable and gap < 1e3:
        reason = (
            f"no clean rank gap (ratio {gap:.3g}); the classification is "
            "sensitive to rank_tol"
        )

    return StructureReport(
        classes=classes,
        degree_of_redundancy=int(m - rank_u),
        redundancy=redundancy,
        rank_A=rank_a,
        rank_A_unmeasured=rank_u,
        n_equations=m,
        n_measured=int(mask.sum()),
        n_unmeasured=n_u,
        singular_values_A=sv_a,
        singular_values_A_unmeasured=sv_u,
        rank_tol=tol,
        rank_gap=gap,
        solvable=solvable,
        reason=reason,
        unobservable=unobservable,
        names=names,
    )
