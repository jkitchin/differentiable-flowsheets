"""Gross error detection: the global test, the measurement test, elimination.

Reconciliation assumes the measurement errors are zero-mean and normal.
A miscalibrated, drifting or failed sensor breaks that assumption, and
because least squares spreads a single large error across *all* the
adjustments ("smearing"), a gross error left in the data corrupts every
reconciled value, not just its own.

Two classical tests, both read off quantities the reconciliation has
already produced:

* the **global test** asks whether the whole data set is consistent
  with the model. The optimal objective is itself the statistic, and it
  is distributed :math:`\\chi^2` on the degrees of redundancy. (The
  often-quoted form :math:`F(y)^T (A \\Sigma A^T)^{-1} F(y)` is a
  special case that cannot be evaluated at all when some variable is
  unmeasured, and carries the wrong degrees of freedom.)
* the **measurement test** standardizes each adjustment by its own
  standard deviation to point at the sensor responsible.

A measurement nothing checks has zero adjustment variance, so it is not
testable. Because both that fact and the redundancy classification come
from the same :math:`\\Sigma_{\\hat x}`, they cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from scipy import stats

from difflow.params_mixin import ParamsMixin

from difflow.reconciliation.core import measured_mask
from difflow.reconciliation.reconcile import ReconcileResult, reconcile
from difflow.reconciliation.structure import REDUNDANCY_TOL


@dataclass
class GlobalTestResult(ParamsMixin):
    """Whether the data set as a whole is consistent with the model.

    Attributes:
        statistic: the reconciliation objective.
        dof: degrees of redundancy.
        critical: the ``1 - alpha`` quantile of ``chi2(dof)``.
        p_value: probability of a statistic this large under H0.
        detected: ``statistic > critical``.
        alpha: significance level used.
    """

    statistic: float
    dof: int
    critical: float
    p_value: float
    detected: bool
    alpha: float

    def __str__(self) -> str:
        verdict = "GROSS ERROR DETECTED" if self.detected else "no gross error"
        return (
            f"global test: chi2 = {self.statistic:.3f} on {self.dof} dof, "
            f"critical = {self.critical:.3f}, p = {self.p_value:.3g} "
            f"-> {verdict}"
        )


@dataclass
class MeasurementTestResult(ParamsMixin):
    """Per-sensor standardized adjustments.

    Attributes:
        z: variable -> standardized adjustment; ``nan`` when the
            measurement carries no redundancy and so cannot be tested.
        critical: two-sided normal critical value, Bonferroni-corrected
            across the testable measurements by default.
        suspect: the testable variable with the largest ``|z|``, or
            ``None`` if nothing exceeds ``critical``.
        z_max: that largest ``|z|``.
        detected: whether any ``|z|`` exceeds ``critical``.
        testable: variable -> whether it carries enough redundancy.
    """

    z: dict[str, float]
    critical: float
    suspect: str | None
    z_max: float
    detected: bool
    testable: dict[str, bool]
    alpha: float = 0.05

    def ranked(self) -> list[tuple[str, float]]:
        """Testable variables sorted by decreasing ``|z|``."""
        items = [
            (nm, v) for nm, v in self.z.items()
            if self.testable.get(nm, False) and np.isfinite(v)
        ]
        return sorted(items, key=lambda kv: -abs(kv[1]))

    def __str__(self) -> str:
        rows = self.ranked()[:5]
        head = (
            f"measurement test (|z| > {self.critical:.3f}): "
            + (f"suspect {self.suspect}" if self.suspect else "nothing flagged")
        )
        return "\n".join([head] + [f"  {nm:<20} z = {v:+.3f}" for nm, v in rows])


def global_test(result: ReconcileResult, alpha: float = 0.05) -> GlobalTestResult:
    """Chi-squared test on the whole data set.

    Args:
        result: a finished reconciliation.
        alpha: significance level.

    Returns:
        A :class:`GlobalTestResult`.
    """
    dof = int(result.structure.degree_of_redundancy)
    stat = float(result.objective)
    if dof <= 0:
        return GlobalTestResult(
            statistic=stat, dof=0, critical=float("inf"), p_value=1.0,
            detected=False, alpha=alpha,
        )
    critical = float(stats.chi2.ppf(1.0 - alpha, dof))
    p_value = float(stats.chi2.sf(stat, dof))
    return GlobalTestResult(
        statistic=stat, dof=dof, critical=critical, p_value=p_value,
        detected=bool(stat > critical), alpha=alpha,
    )


def measurement_test(
    result: ReconcileResult,
    alpha: float = 0.05,
    *,
    bonferroni: bool = True,
    min_redundancy: float = REDUNDANCY_TOL,
) -> MeasurementTestResult:
    """Standardized adjustments, to identify which sensor is at fault.

    The adjustment covariance is :math:`\\Sigma_{adj} = \\Sigma -
    \\Sigma_{\\hat x}` on the measured block, so
    :math:`z_i = (\\hat x_i - y_i)/\\sqrt{\\Sigma_{adj,ii}}` is standard
    normal under the null hypothesis.

    Args:
        result: a finished reconciliation.
        alpha: significance level.
        bonferroni: correct the critical value for the number of
            simultaneous tests (recommended; without it, one false
            positive per twenty sensors is expected by construction).
        min_redundancy: adjustment variance below this fraction of
            ``sigma^2`` marks a measurement as untestable.

    Returns:
        A :class:`MeasurementTestResult`.
    """
    sigma = np.asarray(result.sigma, dtype=float)
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    var_hat = np.diag(np.asarray(result.covariance, dtype=float))
    mask = np.asarray(measured_mask(result.sigma))

    z: dict[str, float] = {}
    testable: dict[str, bool] = {}
    for i, nm in enumerate(result.names):
        var_adj = sigma[i] ** 2 - var_hat[i] if mask[i] else 0.0
        ok = bool(mask[i] and var_adj > min_redundancy * sigma[i] ** 2)
        testable[nm] = ok
        z[nm] = float((x[i] - y[i]) / np.sqrt(var_adj)) if ok else float("nan")

    n_tests = max(sum(testable.values()), 1)
    level = alpha / n_tests if bonferroni else alpha
    critical = float(stats.norm.ppf(1.0 - level / 2.0))

    candidates = [(nm, abs(v)) for nm, v in z.items() if testable[nm]]
    if candidates:
        suspect, z_max = max(candidates, key=lambda kv: kv[1])
    else:
        suspect, z_max = None, 0.0
    detected = bool(z_max > critical)
    return MeasurementTestResult(
        z=z, critical=critical,
        suspect=suspect if detected else None,
        z_max=float(z_max), detected=detected, testable=testable, alpha=alpha,
    )


@dataclass
class EliminationStep(ParamsMixin):
    """One round of serial elimination.

    Attributes:
        removed: the measurement discarded to reach this round, or
            ``None`` for the initial round.
        statistic, dof, critical, p_value: the global test of this round.
        detected: whether the global test still rejects.
        suspect: the measurement the measurement test now points at.
        z_max: its standardized adjustment.
    """

    removed: str | None
    statistic: float
    dof: int
    critical: float
    p_value: float
    detected: bool
    suspect: str | None
    z_max: float


def serial_elimination(
    residual_fn: Callable,
    y,
    sigma,
    *,
    alpha: float = 0.05,
    max_removed: int = 3,
    names=None,
    **reconcile_kw: Any,
) -> list[EliminationStep]:
    """Repeatedly discard the most suspect measurement until data is clean.

    Each round reconciles, runs the global test, and --- if it still
    rejects --- marks the measurement with the largest standardized
    adjustment as unmeasured (``sigma = inf``) and repeats. Discarding a
    measurement costs one degree of redundancy, so the procedure stops
    once the global test passes, ``max_removed`` is reached, or the
    remaining problem would become unobservable.

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        y: measurements.
        sigma: standard deviations.
        alpha: significance level for both tests.
        max_removed: cap on discarded measurements.
        names: variable names.
        **reconcile_kw: forwarded to :func:`reconcile`.

    Returns:
        One :class:`EliminationStep` per round, oldest first. The last
        entry has ``detected=False`` if the data ended up clean.
    """
    from difflow.reconciliation.structure import ReconciliationStructureError

    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    steps: list[EliminationStep] = []
    removed: str | None = None

    for _ in range(max_removed + 1):
        try:
            res = reconcile(
                residual_fn, y, sigma, names=names, **reconcile_kw
            )
        except ReconciliationStructureError:
            break
        gt = global_test(res, alpha=alpha)
        mt = measurement_test(res, alpha=alpha)
        steps.append(
            EliminationStep(
                removed=removed, statistic=gt.statistic, dof=gt.dof,
                critical=gt.critical, p_value=gt.p_value,
                detected=gt.detected, suspect=mt.suspect, z_max=mt.z_max,
            )
        )
        if not gt.detected or mt.suspect is None:
            break
        idx = res.names.index(mt.suspect)
        sigma = sigma.at[idx].set(jnp.inf)
        removed = mt.suspect

    return steps
