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

:func:`blame_concentration` measures exactly that, and
:meth:`MonitorResult.diagnose` reads it together with the rejection
rate to reach a verdict. The point of separating them is that they call
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

import numpy as np

from difflow.params_mixin import ParamsMixin

from difflow.reconciliation.gross_error import global_test, measurement_test
from difflow.reconciliation.reconcile import ReconcileResult, reconcile
from difflow.reconciliation.structure import ReconciliationStructureError

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

    def diagnose(
        self,
        window: int | None = 15,
        *,
        rejection_threshold: float = REJECTION_THRESHOLD,
        concentration_threshold: float = CONCENTRATION_THRESHOLD,
    ) -> "MonitorDiagnosis":
        """Decide whether a sensor or the model is at fault.

        Args:
            window: number of most recent steps to judge on; ``None``
                uses the whole campaign. One bad day is noise, so the
                window should span enough days for persistence to mean
                something.
            rejection_threshold: fraction of the window that must
                reject before any fault is declared.
            concentration_threshold: blame concentration at or above
                which the fault is attributed to one sensor.

        Returns:
            A :class:`MonitorDiagnosis`.
        """
        rate = self.rejection_rate(window)
        concentration, culprit = blame_concentration(self.suspects, window)

        if rate < rejection_threshold:
            verdict, culprit = MONITOR_CONSISTENT, None
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
    """

    verdict: str
    culprit: str | None
    rejection_rate: float
    blame_concentration: float
    window: int

    @property
    def drifting(self) -> bool:
        """Whether the verdict calls for re-estimating the model."""
        return self.verdict == MONITOR_MODEL_DRIFT

    def __str__(self) -> str:
        head = f"{self.verdict}: {self.rejection_rate:.0%} of the last "
        head += f"{self.window} steps reject, blame concentration "
        head += f"{self.blame_concentration:.0%}"
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


def monitor(
    residual_fn: Callable,
    measurements: Sequence[Any],
    sigma: Any,
    *,
    params: Any = None,
    names: Sequence[str] | None = None,
    alpha: float = 0.05,
    keep_results: bool = False,
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
        steps.append(
            MonitorStep(
                index=i, statistic=gt.statistic, dof=gt.dof,
                critical=gt.critical, p_value=gt.p_value,
                detected=gt.detected, suspect=mt.suspect, z_max=mt.z_max,
                result=res if keep_results else None,
            )
        )

    return MonitorResult(steps=steps, names=names or [], alpha=alpha)
