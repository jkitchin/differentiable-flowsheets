"""Monitoring a plant over time, and telling model drift from a bad sensor.

A single reconciliation answers "is today's data consistent with the
model?". Running one every day turns that into an instrument: the
global-test statistic becomes a time series, and its behaviour --- not
any single value --- says what is wrong.

Both failure modes push the statistic above its critical value, so the
statistic alone cannot separate them. The measurement test can, because
of *where* it points:

* a **biased sensor** is one broken equation term, so the same sensor
  is blamed every day. Blame is concentrated.
* **model drift** --- a fouling pipe, a decaying catalyst --- breaks
  the balance the model imposes rather than any one reading, so least
  squares smears the inconsistency over whichever measurements happen
  to be cheapest to move that day. Blame wanders.

:func:`blame_concentration` measures exactly that. But blame needs
redundancy to wander: on a small unit --- an exchanger with one energy
balance and one rate equation --- fouling moves both outlet
temperatures together and the measurement test blames the same one
every day, exactly like a biased thermocouple. So :func:`monitor` also
runs a second, physics-based test whenever the model has one to offer:
it finds the constraints that contain **no model parameter** (the
energy balance, here; rows on which ``dF/dparams`` is identically
zero) and reconciles against those alone. They are true whatever the
parameters are, so they test the instruments and nothing else:

* the parameter-free constraints **pass** while the full model rejects
  --- the meters agree with each other and the model is wrong: drift;
* they **reject** --- a meter disagrees with the balances: a fault.

:meth:`MonitorResult.diagnose` reads that test first and falls back on
the blame pattern only where it cannot (no parameter-free constraint,
or a suspect the parameter-free constraints do not check). The point of separating them is that they call
for opposite responses: recalibrate the instrument, or re-estimate the
parameter. Doing the second when the first is true manufactures a
parameter estimate out of a calibration error --- see
``examples/29_model_updating.ipynb``, which does exactly that on
purpose.

Example:
    >>> mon = monitor(residual_fn, daily_measurements, sigma,
    ...               names=layout.names)          # doctest: +SKIP
    >>> mon.diagnose()                             # doctest: +SKIP
    model drift: 14/15 recent days reject, blame concentration 0.33
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from difflow.params_mixin import ParamsMixin

from difflow.reconciliation.core import Scaling
from difflow.reconciliation.gross_error import global_test, measurement_test
from difflow.reconciliation.reconcile import ReconcileResult, reconcile
from difflow.reconciliation.structure import (
    MEASURED_REDUNDANT,
    ReconciliationStructureError,
)

#: The data are consistent with the model: too few days reject.
MONITOR_CONSISTENT = "consistent"
#: Persistent rejection blaming the same sensor: recalibrate it.
MONITOR_INSTRUMENT_FAULT = "instrument fault"
#: Persistent rejection with a wandering suspect: re-estimate the model.
MONITOR_MODEL_DRIFT = "model drift"
#: Persistent rejection with nothing testable to blame.
MONITOR_UNDIAGNOSED = "undiagnosed"

#: Fraction of the window that must reject before a verdict is drawn.
REJECTION_THRESHOLD = 0.5
#: Blame concentration at or above which a single sensor is blamed.
CONCENTRATION_THRESHOLD = 0.6

#: :attr:`MonitorDiagnosis.basis` when the verdict was drawn from the
#: constraints that contain no model parameter.
BASIS_PARAMETER_FREE = "parameter-free constraints"
#: :attr:`MonitorDiagnosis.basis` when only the blame pattern was
#: available. It cannot tell drift from a fault on a unit with too
#: little redundancy for blame to wander.
BASIS_BLAME = "blame pattern"


@dataclass
class MonitorStep(ParamsMixin):
    """One data set's reconciliation, reduced to its diagnostics.

    Attributes:
        index: position in the sequence handed to :func:`monitor`.
        statistic: the global-test statistic (the objective).
        dof: degrees of redundancy.
        critical: the global test's critical value.
        p_value: probability of a statistic this large under H0.
        detected: whether the global test rejects.
        suspect: the sensor the measurement test blames, or ``None``.
        z_max: the largest standardized adjustment.
        failed: set when the reconciliation could not be posed at all,
            carrying the reason; every other field is then a placeholder.
        result: the full reconciliation, kept only when :func:`monitor`
            is called with ``keep_results=True``.
        balance_statistic: the global-test statistic of the
            reconciliation against the parameter-free constraints
            alone, ``nan`` when there was none to run.
        balance_detected: whether that test rejects; ``None`` when
            it could not be run (no parameter-free constraint, none
            with redundancy, or a model with no parameter to drift).
        balance_checks: the measured variables the parameter-free
            constraints check (those redundant in them).
    """

    index: int
    statistic: float
    dof: int
    critical: float
    p_value: float
    detected: bool
    suspect: str | None
    z_max: float
    failed: str = ""
    result: ReconcileResult | None = None
    balance_statistic: float = float("nan")
    balance_detected: bool | None = None
    balance_checks: tuple[str, ...] = ()


@dataclass
class MonitorResult(ParamsMixin):
    """A campaign of reconciliations against one fixed model.

    Attributes:
        steps: one :class:`MonitorStep` per data set, in order.
        names: the variable names shared by every step.
        alpha: the significance level used throughout.
    """

    steps: list[MonitorStep]
    names: list[str] = field(default_factory=list)
    alpha: float = 0.05

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def statistic(self) -> np.ndarray:
        """Global-test statistic of each step, shape ``(n_steps,)``."""
        return np.array([s.statistic for s in self.steps], dtype=float)

    @property
    def detected(self) -> np.ndarray:
        """Boolean rejection flag of each step."""
        return np.array([s.detected for s in self.steps], dtype=bool)

    @property
    def suspects(self) -> list[str | None]:
        """The sensor blamed on each step, ``None`` where nothing flagged."""
        return [s.suspect for s in self.steps]

    @property
    def critical(self) -> float:
        """The global test's critical value.

        Constant across the campaign whenever the structure is, which
        is the normal case: the model and the sigmas are fixed, only
        the data change. Reported as ``nan`` if the steps disagree.
        """
        crits = {s.critical for s in self.steps if not s.failed}
        if len(crits) == 1:
            return crits.pop()
        return float("nan")

    @property
    def dof(self) -> int:
        """Degrees of redundancy, or ``-1`` if the steps disagree."""
        dofs = {s.dof for s in self.steps if not s.failed}
        return dofs.pop() if len(dofs) == 1 else -1

    def rejection_rate(self, window: int | None = None) -> float:
        """Fraction of the last ``window`` steps whose global test rejects."""
        flags = self.detected[_tail(len(self), window)]
        return float(flags.mean()) if flags.size else 0.0

    def balance_rejection_rate(self, window: int | None = None) -> float | None:
        """Fraction of the last ``window`` steps whose parameter-free test rejects.

        ``None`` when no step in the window could run that test.
        """
        flags = [
            s.balance_detected
            for s in self.steps[_tail(len(self), window)]
            if s.balance_detected is not None
        ]
        return float(np.mean(flags)) if flags else None

    def diagnose(
        self,
        window: int | None = 15,
        *,
        rejection_threshold: float = REJECTION_THRESHOLD,
        concentration_threshold: float = CONCENTRATION_THRESHOLD,
    ) -> "MonitorDiagnosis":
        """Decide whether a sensor or the model is at fault.

        Persistent rejection of the full model is the trigger. What
        decides between the two causes is, where the model offers one,
        the test against its **parameter-free** constraints (see
        :func:`monitor`): they are true whatever the parameters are, so

        * they pass on most of the window --- the meters agree with
          each other, the model does not: :data:`MONITOR_MODEL_DRIFT`;
        * they reject on most of it --- a meter disagrees with the
          balances: :data:`MONITOR_INSTRUMENT_FAULT` on the
          concentrated suspect, :data:`MONITOR_UNDIAGNOSED` if the
          blame wanders.

        Only where that test is unavailable --- no parameter-free
        constraint with redundancy, or a suspect those constraints do
        not check --- is the verdict drawn from the blame pattern:
        concentrated blame is a fault, wandering blame is drift. That
        rule needs redundancy for the blame to wander, and on a small
        unit it reads drift as a fault; :attr:`MonitorDiagnosis.basis`
        says which rule decided.

        Args:
            window: number of most recent steps to judge on; ``None``
                uses the whole campaign. One bad day is noise, so the
                window should span enough days for persistence to mean
                something.
            rejection_threshold: fraction of the window that must
                reject before any fault is declared (applied to both
                tests).
            concentration_threshold: blame concentration at or above
                which the fault is attributed to one sensor.

        Returns:
            A :class:`MonitorDiagnosis`.
        """
        rate = self.rejection_rate(window)
        concentration, culprit = blame_concentration(self.suspects, window)
        balance_rate = self.balance_rejection_rate(window)
        checked = {
            nm
            for s in self.steps[_tail(len(self), window)]
            for nm in s.balance_checks
        }
        basis = BASIS_BLAME

        if rate < rejection_threshold:
            verdict, culprit = MONITOR_CONSISTENT, None
        elif balance_rate is not None and balance_rate >= rejection_threshold:
            # The meters disagree with constraints no parameter can fix.
            basis = BASIS_PARAMETER_FREE
            if culprit is not None and concentration >= concentration_threshold:
                verdict = MONITOR_INSTRUMENT_FAULT
            else:
                verdict = MONITOR_UNDIAGNOSED
        elif balance_rate is not None and (culprit is None or culprit in checked):
            # The meters agree with each other; the model does not.
            basis = BASIS_PARAMETER_FREE
            verdict, culprit = MONITOR_MODEL_DRIFT, None
        elif culprit is None:
            verdict = MONITOR_UNDIAGNOSED
        elif concentration >= concentration_threshold:
            verdict = MONITOR_INSTRUMENT_FAULT
        else:
            verdict, culprit = MONITOR_MODEL_DRIFT, None

        return MonitorDiagnosis(
            verdict=verdict,
            culprit=culprit,
            rejection_rate=rate,
            blame_concentration=concentration,
            window=min(window, len(self)) if window else len(self),
            balance_rejection_rate=balance_rate,
            basis=basis,
        )

    def summary(self) -> str:
        """One line per step, oldest first."""
        lines = [
            f"{'step':>5} {'chi2':>10} {'verdict':>8} {'suspect':>16}",
            "-" * 42,
        ]
        for s in self.steps:
            if s.failed:
                lines.append(f"{s.index:5d} {'-':>10} {'failed':>8} {s.failed:>16}")
                continue
            lines.append(
                f"{s.index:5d} {s.statistic:10.3f} "
                f"{'REJECT' if s.detected else 'accept':>8} "
                f"{str(s.suspect):>16}"
            )
        return "\n".join(lines)


@dataclass
class MonitorDiagnosis(ParamsMixin):
    """Verdict on a monitoring campaign.

    Attributes:
        verdict: one of :data:`MONITOR_CONSISTENT`,
            :data:`MONITOR_INSTRUMENT_FAULT`, :data:`MONITOR_MODEL_DRIFT`
            or :data:`MONITOR_UNDIAGNOSED`.
        culprit: the sensor to recalibrate, set only for an instrument
            fault.
        rejection_rate: fraction of the window that rejected.
        blame_concentration: fraction of the window blaming the single
            most-blamed sensor.
        window: number of steps the verdict was drawn from.
        balance_rejection_rate: fraction of the window whose test
            against the parameter-free constraints rejected, ``None``
            when no step could run it.
        basis: :data:`BASIS_PARAMETER_FREE` or :data:`BASIS_BLAME`,
            the rule that decided between drift and a fault.
    """

    verdict: str
    culprit: str | None
    rejection_rate: float
    blame_concentration: float
    window: int
    balance_rejection_rate: float | None = None
    basis: str = BASIS_BLAME

    @property
    def drifting(self) -> bool:
        """Whether the verdict calls for re-estimating the model."""
        return self.verdict == MONITOR_MODEL_DRIFT

    def __str__(self) -> str:
        head = f"{self.verdict}: {self.rejection_rate:.0%} of the last "
        head += f"{self.window} steps reject, blame concentration "
        head += f"{self.blame_concentration:.0%}"
        if self.balance_rejection_rate is not None:
            head += (
                f", parameter-free constraints reject on "
                f"{self.balance_rejection_rate:.0%}"
            )
        if self.culprit is not None:
            head += f", on {self.culprit}"
        return head


def _tail(n: int, window: int | None) -> slice:
    """Slice of the last ``window`` of ``n`` items."""
    if window is None or window >= n:
        return slice(0, n)
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return slice(n - window, n)


def blame_concentration(
    suspects: Sequence[str | None], window: int | None = 15
) -> tuple[float, str | None]:
    """How concentrated the measurement test's blame is, and on whom.

    The fraction is taken over the whole window, *including* the steps
    that flagged nothing: a campaign that rejects rarely should not
    read as a concentrated fault just because the few rejections it did
    produce agreed with each other.

    Args:
        suspects: the blamed sensor per step, ``None`` where the
            measurement test flagged nothing.
        window: number of most recent steps to count over; ``None``
            counts them all.

    Returns:
        ``(concentration, most_blamed)``, the latter ``None`` when
        nothing was flagged in the window.
    """
    tail = list(suspects)[_tail(len(suspects), window)]
    if not tail:
        return 0.0, None
    counts = Counter(s for s in tail if s is not None)
    if not counts:
        return 0.0, None
    culprit, hits = counts.most_common(1)[0]
    return hits / len(tail), culprit


class _RowSubset:
    """``F(x, params)[rows]``, hashable so the jitted solve is reused.

    :func:`~difflow.reconciliation.solve_reconciliation` takes the
    residual function as a static argument; a fresh closure per call
    would retrace and recompile it every period.
    """

    def __init__(self, residual_fn: Callable, rows: Sequence[int]):
        self.residual_fn = residual_fn
        self.rows = tuple(int(r) for r in rows)

    def __call__(self, x: Any, params: Any) -> Any:
        return self.residual_fn(x, params)[jnp.asarray(self.rows)]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _RowSubset)
            and other.residual_fn == self.residual_fn
            and other.rows == self.rows
        )

    def __hash__(self) -> int:
        return hash((_RowSubset, self.residual_fn, self.rows))


def parameter_free_rows(fn_of_params: Callable, params: Any) -> list[int] | None:
    """Constraints that do not depend on any floating-point parameter.

    Args:
        fn_of_params: ``params -> F`` at a fixed state, shape ``(m,)``.
        params: the parameter pytree; its floating-point leaves are the
            parameters, every other leaf is held fixed.

    Returns:
        The indices of the rows on which ``dF/dparams`` is identically
        zero, or ``None`` when the split is not informative: no
        floating-point parameter, no row free of them, every row free
        of them, or a ``params`` the Jacobian cannot be taken through.
    """
    try:
        leaves, treedef = jax.tree_util.tree_flatten(params)
        which = [
            i for i, leaf in enumerate(leaves)
            if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)
        ]
        if not which:
            return None

        def of(vals):
            full = list(leaves)
            for i, v in zip(which, vals):
                full[i] = v
            return fn_of_params(jax.tree_util.tree_unflatten(treedef, full))

        vals = [jnp.asarray(leaves[i], dtype=jnp.float64) for i in which]
        jac = jax.jacobian(of)(vals)
        m = int(np.asarray(of(vals)).shape[0])
        dep = np.zeros(m, dtype=bool)
        for block in jac:
            dep |= np.any(np.asarray(block).reshape(m, -1) != 0.0, axis=1)
    except Exception:  # noqa: BLE001 - an opaque params just skips the split
        return None
    free = [i for i in range(m) if not dep[i]]
    if not free or len(free) == m:
        return None
    return free


def balance_test(
    residual_fn: Callable,
    y: Any,
    sigma: Any,
    rows: Sequence[int] | None,
    *,
    params: Any = None,
    names: Sequence[str] | None = None,
    alpha: float = 0.05,
    **reconcile_kw: Any,
) -> dict[str, Any]:
    """Global test against the parameter-free constraints alone.

    Those constraints hold whatever the model parameters are, so the
    test checks the instruments and nothing else.

    Returns:
        The :class:`MonitorStep` fields ``balance_statistic``,
        ``balance_detected`` and ``balance_checks``; placeholders when
        ``rows`` is ``None`` or the subset has no redundancy or cannot
        be posed.
    """
    blank = {
        "balance_statistic": float("nan"),
        "balance_detected": None,
        "balance_checks": (),
    }
    if not rows:
        return blank
    given = reconcile_kw.get("scaling")
    if isinstance(given, Scaling):
        # A caller's residual scaling is per constraint; keep its rows.
        reconcile_kw["scaling"] = Scaling(
            d=given.d, r=jnp.asarray(given.r)[jnp.asarray(rows)]
        )
    try:
        res = reconcile(
            _RowSubset(residual_fn, rows), y, sigma, params=params,
            names=names, **reconcile_kw,
        )
    except ReconciliationStructureError:
        return blank
    gt = global_test(res, alpha=alpha)
    if gt.dof <= 0:
        return blank
    return {
        "balance_statistic": gt.statistic,
        "balance_detected": gt.detected,
        "balance_checks": tuple(
            nm for nm, cls in res.structure.classes.items()
            if cls == MEASURED_REDUNDANT
        ),
    }


def monitor(
    residual_fn: Callable,
    measurements: Sequence[Any],
    sigma: Any,
    *,
    params: Any = None,
    names: Sequence[str] | None = None,
    alpha: float = 0.05,
    keep_results: bool = False,
    parameter_free: Sequence[int] | bool | None = None,
    **reconcile_kw: Any,
) -> MonitorResult:
    """Reconcile a sequence of data sets against one fixed model.

    The model and the sigmas stay fixed while the data change, which is
    what makes the resulting statistic series readable: every movement
    in it comes from the data. Estimating a parameter at the same time
    would let the model absorb the very inconsistency the test is
    supposed to expose, so re-estimation is a separate, deliberate step
    --- :func:`difflow.reconciliation.reconcile_multi`.

    A data set whose problem cannot be posed at all does not abort the
    campaign; its step records ``failed`` and the rest continue.

    Each period is also reconciled against the **parameter-free**
    constraints alone --- the rows of ``F`` that do not depend on any
    floating-point leaf of ``params`` (an energy balance, beside a rate
    equation carrying ``UA``). That test is the instrument check
    :meth:`MonitorResult.diagnose` uses to tell drift from a fault. It
    is skipped when ``params`` has no such leaf or every row is free
    of them.

    Args:
        residual_fn: ``F(x, params) -> (m,)``, JAX-traceable.
        measurements: one measurement vector per period, each shape
            ``(n,)``.
        sigma: standard deviations, shape ``(n,)``, shared by every
            period; ``inf`` marks an unmeasured variable.
        params: extra argument threaded to ``residual_fn``, fixed
            across the campaign.
        names: variable names, used for the suspects.
        alpha: significance level for both tests.
        keep_results: retain each full
            :class:`~difflow.reconciliation.ReconcileResult` on its
            step. Off by default --- each carries an ``(n, n)``
            covariance, which a long campaign multiplies.
        parameter_free: the parameter-free constraint rows. ``None``
            (default) finds them from ``dF/dparams``; a sequence of row
            indices names them, for a ``params`` that also holds fixed
            constants; ``False`` turns the second test off.
        **reconcile_kw: forwarded to
            :func:`~difflow.reconciliation.reconcile`.

    Returns:
        A :class:`MonitorResult`.

    Example:
        >>> mon = monitor(F, [y_day1, y_day2], sigma)   # doctest: +SKIP
        >>> mon.statistic                               # doctest: +SKIP
        array([12.3, 48.9])
    """
    names = list(names) if names is not None else None
    steps: list[MonitorStep] = []
    find_rows = parameter_free is None or parameter_free is True
    rows: list[int] | None = (
        None if isinstance(parameter_free, bool) or parameter_free is None
        else [int(r) for r in parameter_free]
    )

    for i, y in enumerate(measurements):
        try:
            res = reconcile(
                residual_fn, y, sigma, params=params, names=names,
                **reconcile_kw,
            )
        except ReconciliationStructureError as err:
            steps.append(
                MonitorStep(
                    index=i, statistic=float("nan"), dof=-1,
                    critical=float("nan"), p_value=float("nan"),
                    detected=False, suspect=None, z_max=float("nan"),
                    failed=str(err).split(":", 1)[0],
                )
            )
            continue

        gt = global_test(res, alpha=alpha)
        mt = measurement_test(res, alpha=alpha)
        if names is None:
            names = list(res.names)
        if find_rows:
            # The dependence is structural, so one look is enough.
            find_rows = False
            x_hat = res.x
            rows = parameter_free_rows(
                lambda ps, x=x_hat: residual_fn(x, ps), params
            )
        steps.append(
            MonitorStep(
                index=i, statistic=gt.statistic, dof=gt.dof,
                critical=gt.critical, p_value=gt.p_value,
                detected=gt.detected, suspect=mt.suspect, z_max=mt.z_max,
                result=res if keep_results else None,
                **balance_test(
                    residual_fn, y, sigma, rows, params=params,
                    names=names, alpha=alpha, **reconcile_kw,
                ),
            )
        )

    return MonitorResult(steps=steps, names=names or [], alpha=alpha)
