"""Tracking a drifting parameter from a stream of reconciliations.

:func:`~difflow.reconciliation.monitor` runs the *routine* clock: the
model is frozen, so the global test is a genuine instrument-health
monitor. :func:`~difflow.reconciliation.reconcile_multi` runs the
*campaign* clock: parameters go free and a window is pooled. The two
are described in ``examples/29_model_updating.ipynb`` as a discipline a
person applies. This module turns that discipline into an update law,
because in a running twin nobody applies it by hand.

Two things are added, and neither is a new estimator.

**A gate.** :func:`update_gate` reads
:meth:`~difflow.reconciliation.MonitorResult.diagnose` and lets the
parameter move only on :data:`MONITOR_MODEL_DRIFT`. A concentrated
suspect means go calibrate a meter; letting the parameter move then
manufactures a model fault out of a calibration error --- section 6 of
notebook 29 does exactly that on purpose, and the global test *falls*
while it happens, so the twin looks healthier as it gets wronger. No
filter gain prevents this: a slow filter reaches the wrong answer
gracefully. Only a gate prevents it, and the gate has to be structural
rather than advisory.

**A memory.** Pooling a window is a *rectangular* filter --- hard edges,
a ten-day-old period weighted like this morning's, recomputed from
scratch each time. :func:`time_update` and :func:`measurement_update`
replace it with the random walk

.. math::

    \\theta_{k+1} = \\theta_k + w_k, \\qquad
    \\mathrm{cov}(w_k) = Q\\,\\Delta t,

filtered against each period's reconciliation. That is the same model
:func:`difflow.mhe.augment_parameters` puts on a drifting parameter,
and ``drift_std`` here is the same knob as ``process_std`` there,
carrying the same warning: too large and the parameter absorbs sensor
noise, too small and a genuine drift is rejected. It is a required
argument, never a default.

Nothing here re-derives an estimator. A single period's reconciliation
already returns both halves of a Kalman measurement update --- the
estimate and, from :func:`~difflow.reconciliation.reconciled_covariance`,
its covariance --- so the update is a combination of two Gaussians and
the model physics stays inside :func:`~difflow.reconciliation.reconcile`.

Invariants, which are the point of the module:

* The measurement covariance is kept **full**. Correlated parameters
  have a *difference* variance a diagonal covariance gets wrong by a
  factor of a few, so the prior is a matrix and the update is the
  matrix form, in Joseph factorisation so it stays symmetric positive
  semidefinite under a long run.
* The **time update always runs**; only the measurement update is
  gated. A held parameter is not a known parameter: while the twin
  refuses to move it, its uncertainty grows at exactly the rate the
  drift model claims, and the next update that *is* allowed takes a
  correspondingly larger step. ``max_std`` bounds that growth if a
  quiet year would otherwise make the next step a jump.
* A period that cannot identify the parameter is not rejected. Its
  reconciliation returns a large covariance, the gain goes to zero and
  the estimate does not move --- which is what a weakly informative
  period *should* do, and needs no special case.
* The parameter is threaded through ``params``, so the frozen and free
  problems are the same ``residual_fn``. A twin whose two clocks run
  different code drifts apart in a second, less interesting way.

Example:
    >>> from difflow.reconciliation import track_parameters, TrackerState
    >>> state = TrackerState.initial(["eta"], [1.0], std=[0.05])
    >>> run = track_parameters(              # doctest: +SKIP
    ...     F, sixty_days, sigma, state=state,
    ...     drift_std=[2e-3], names=layout.names,
    ... )
    >>> run.summary()                        # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow.reconciliation.core import measured_mask
from difflow.reconciliation.gross_error import global_test, measurement_test
from difflow.reconciliation.monitoring import (
    CONCENTRATION_THRESHOLD,
    MONITOR_CONSISTENT,
    MONITOR_INSTRUMENT_FAULT,
    MONITOR_MODEL_DRIFT,
    MONITOR_UNDIAGNOSED,
    MonitorDiagnosis,
    MonitorResult,
    MonitorStep,
    REJECTION_THRESHOLD,
)
from difflow.reconciliation.reconcile import ReconcileResult, reconcile
from difflow.reconciliation.structure import ReconciliationStructureError

#: Verdicts on which the parameter is allowed to move. Model drift is
#: the only one: a concentrated suspect calls for a calibration, and a
#: consistent campaign gives the model nothing to correct.
UPDATE_WHEN_DRIFTING = frozenset({MONITOR_MODEL_DRIFT})

#: Why an update was withheld, keyed by the verdict that withheld it.
_HOLD_REASON = {
    MONITOR_CONSISTENT:
        "data are consistent with the model; nothing to correct",
    MONITOR_INSTRUMENT_FAULT:
        "blame is concentrated on one sensor; calibrate it",
    MONITOR_UNDIAGNOSED:
        "persistent rejection with nothing testable to blame",
}


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


@dataclass
class GateDecision(ParamsMixin):
    """Whether the monitoring verdict lets the parameter move.

    Attributes:
        allowed: whether a measurement update may run this step.
        verdict: the :class:`~difflow.reconciliation.MonitorDiagnosis`
            verdict it was read from.
        reason: why the update was allowed or withheld, for the log.
        culprit: the sensor to recalibrate, on an instrument fault.
    """

    allowed: bool
    verdict: str
    reason: str
    culprit: str | None = None

    def __str__(self) -> str:
        head = "update" if self.allowed else "hold"
        return f"{head} ({self.verdict}): {self.reason}"


def update_gate(
    diagnosis: MonitorDiagnosis,
    *,
    allow: frozenset[str] | Sequence[str] = UPDATE_WHEN_DRIFTING,
) -> GateDecision:
    """Decide whether a monitoring verdict permits a parameter update.

    The default policy is the one ``examples/29_model_updating.ipynb``
    argues for: move the parameter only when rejection is persistent
    *and* the blame wanders, which is the signature of a model fault.
    Every other verdict holds, and :attr:`GateDecision.reason` says
    which of the three reasons it was.

    Widening ``allow`` is a deliberate act with a known cost. Adding
    :data:`MONITOR_INSTRUMENT_FAULT` lets a biased meter be absorbed
    into the model; adding :data:`MONITOR_CONSISTENT` re-estimates on
    data that gave no reason to, so the parameter tracks whatever the
    period's noise happens to favour.

    Args:
        diagnosis: from
            :meth:`~difflow.reconciliation.MonitorResult.diagnose`.
        allow: verdicts that permit an update.

    Returns:
        A :class:`GateDecision`.

    Example:
        >>> mon = monitor(F, days, sigma)             # doctest: +SKIP
        >>> update_gate(mon.diagnose()).allowed       # doctest: +SKIP
        True
    """
    allow = frozenset(allow)
    verdict = diagnosis.verdict
    if verdict in allow:
        return GateDecision(
            allowed=True,
            verdict=verdict,
            reason=(
                f"{diagnosis.rejection_rate:.0%} of the last "
                f"{diagnosis.window} steps reject, blame concentration "
                f"{diagnosis.blame_concentration:.0%}"
            ),
            culprit=diagnosis.culprit,
        )
    return GateDecision(
        allowed=False,
        verdict=verdict,
        reason=_HOLD_REASON.get(verdict, "verdict does not permit an update"),
        culprit=diagnosis.culprit,
    )


# --------------------------------------------------------------------------
# the filter state
# --------------------------------------------------------------------------


@dataclass
class TrackerState(ParamsMixin):
    """A tracked parameter vector, with its covariance and its clock.

    Attributes:
        names: parameter names, in the order of ``mean``.
        mean: current estimate, shape ``(p,)``.
        covariance: its covariance, shape ``(p, p)``, kept full.
        time: the clock reading this state is current at.
        n_updates: measurement updates applied so far.
        n_held: steps the gate withheld an update on.
    """

    names: list[str]
    mean: Array
    covariance: Array
    time: float = 0.0
    n_updates: int = 0
    n_held: int = 0

    @classmethod
    def initial(
        cls,
        names: Sequence[str],
        mean: Sequence[float] | Array,
        *,
        std: Sequence[float] | Array | float | None = None,
        covariance: Array | None = None,
        time: float = 0.0,
    ) -> "TrackerState":
        """Build a starting state from a mean and a prior spread.

        Give exactly one of ``std`` (independent parameters) or
        ``covariance`` (correlated ones --- prefer it when the prior
        came from :func:`difflow.estimation.predicted_covariance` or
        :func:`~difflow.reconciliation.reconciled_covariance`, which
        return the correlations rather than discarding them).

        ``time`` is the clock this prior is current at, not the clock
        the record starts at. :func:`track_parameters` advances the
        state by ``t - state.time`` each period, so leaving both at
        their defaults means the first period does no drift update ---
        the prior is taken as current, rather than one period stale.
        """
        names = list(names)
        mean = jnp.asarray(mean, dtype=jnp.float64).reshape(-1)
        if mean.shape[0] != len(names):
            raise ValueError(
                f"got {len(names)} names for a mean of {mean.shape[0]}"
            )
        if (std is None) == (covariance is None):
            raise ValueError("give exactly one of std= or covariance=")
        if covariance is None:
            s = jnp.broadcast_to(
                jnp.asarray(std, dtype=jnp.float64), mean.shape
            )
            if not bool(jnp.all(s > 0)):
                raise ValueError("prior std must be strictly positive")
            cov = jnp.diag(s ** 2)
        else:
            cov = jnp.asarray(covariance, dtype=jnp.float64)
            if cov.shape != (mean.shape[0], mean.shape[0]):
                raise ValueError(
                    f"covariance has shape {cov.shape}, expected "
                    f"{(mean.shape[0], mean.shape[0])}"
                )
        return cls(names=names, mean=mean, covariance=_symmetrize(cov),
                   time=float(time))

    @property
    def n(self) -> int:
        """How many parameters are tracked."""
        return len(self.names)

    @property
    def std(self) -> dict[str, float]:
        """Standard deviation of each parameter, as ``{name: value}``."""
        s = np.sqrt(np.clip(np.diag(np.asarray(self.covariance, dtype=float)),
                            0.0, np.inf))
        return {nm: float(s[i]) for i, nm in enumerate(self.names)}

    def as_params(self) -> dict[str, float]:
        """The estimate as ``{name: value}``.

        This is the shape :func:`~difflow.reconciliation.reconcile`
        takes as ``params`` and
        :class:`difflow.planning.Block` takes as ``theta``, so a
        tracked parameter feeds a planning or optimisation layer with
        no adapter --- the same convention
        :class:`difflow.mhe.MHERunResult` reports in.
        """
        m = np.asarray(self.mean, dtype=float)
        return {nm: float(m[i]) for i, nm in enumerate(self.names)}

    def summary(self) -> str:
        """One line per parameter: estimate and standard error."""
        std = self.std
        lines = [
            f"t = {self.time:g}, {self.n_updates} updates, "
            f"{self.n_held} held",
            "",
            f"{'parameter':<24} {'estimate':>12} {'std error':>12}",
            "-" * 50,
        ]
        params = self.as_params()
        for nm in self.names:
            lines.append(f"{nm:<24} {params[nm]:12.6g} {std[nm]:12.6g}")
        return "\n".join(lines)


def drift_std_from_time_constant(
    spread: Sequence[float] | Array | float,
    tau: float,
) -> Array:
    """Convert "wanders by ``spread`` over ``tau``" into a drift std.

    ``drift_std`` is a rate, and a rate is hard to have an opinion
    about. The question an engineer can answer is the other one: *how
    far does this parameter move, and over how long?* A random walk
    accumulates a standard deviation of ``drift_std * sqrt(t)``, so a
    parameter that wanders by ``spread`` over a time ``tau`` has

    .. math:: \\sqrt{Q} = \\mathrm{spread} / \\sqrt{\\tau}.

    Fouling that takes a month to cost five percent of duty is
    ``drift_std_from_time_constant(0.05, 30.0)`` on a daily clock.

    Args:
        spread: how far the parameter moves over ``tau``, scalar or
            per parameter.
        tau: the time it takes, in the clock ``dt`` is measured in.

    Returns:
        The drift standard deviation, same shape as ``spread``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return jnp.asarray(spread, dtype=jnp.float64) / jnp.sqrt(tau)


def time_update(
    state: TrackerState,
    dt: float,
    drift_std: Sequence[float] | Array | float,
    *,
    max_std: Sequence[float] | Array | float | None = None,
) -> TrackerState:
    """Advance the random walk by ``dt``: mean held, covariance grown.

    The parameter has no dynamics of its own beyond :math:`\\theta_{k+1}
    = \\theta_k + w_k`, so the mean does not move and the covariance
    grows by :math:`Q\\,\\Delta t`. Running this even on a step whose
    measurement update is gated off is deliberate: the parameter drifts
    whether or not the data were usable, and a twin that holds an
    estimate without widening its error bar is claiming knowledge it
    stopped collecting.

    Args:
        state: the current state.
        dt: elapsed time, in whatever clock ``drift_std`` is a rate
            in. Zero is legal and is a no-op, which is what the first
            period of a campaign gets when the state's clock and the
            record's first reading agree.
        drift_std: :math:`\\sqrt{\\mathrm{diag}(Q)}`, per unit time,
            scalar or per parameter. See
            :func:`drift_std_from_time_constant`.
        max_std: ceiling on each parameter's standard deviation. The
            covariance is shrunk toward it by a symmetric diagonal
            congruence, so correlations survive and the result stays
            positive semidefinite. ``None`` lets it grow without
            bound, which is right over a campaign and wrong over a
            quiet year.

    Returns:
        The advanced state.
    """
    if dt < 0:
        raise ValueError(f"dt must not be negative, got {dt}")
    q = jnp.broadcast_to(
        jnp.asarray(drift_std, dtype=jnp.float64), (state.n,)
    )
    if not bool(jnp.all(q >= 0)):
        raise ValueError("drift_std must not be negative")
    cov = _symmetrize(state.covariance + jnp.diag(q ** 2) * dt)

    if max_std is not None:
        cap = jnp.broadcast_to(
            jnp.asarray(max_std, dtype=jnp.float64), (state.n,)
        )
        if not bool(jnp.all(cap > 0)):
            raise ValueError("max_std must be strictly positive")
        std = jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0, jnp.inf))
        shrink = jnp.where(std > cap, cap / jnp.where(std > 0, std, 1.0), 1.0)
        cov = _symmetrize(cov * shrink[:, None] * shrink[None, :])

    return replace(state, covariance=cov, time=state.time + float(dt))


@dataclass
class Innovation(ParamsMixin):
    """What one measurement update did, and whether it was in scale.

    Attributes:
        residual: ``estimate - prior mean``, shape ``(p,)``.
        nis: normalised innovation squared, ``v^T S^-1 v``. Under a
            correct drift model and correct sigmas this is
            :math:`\\chi^2` on ``p`` degrees of freedom, so a series of
            them running well above ``p`` says the parameter is moving
            faster than ``drift_std`` admits --- the filter's own
            version of the global test.
        gain: the Kalman gain applied, shape ``(p, p)``. Its diagonal
            is the fraction of the discrepancy each parameter took.
        step: how far the mean actually moved, shape ``(p,)``.
    """

    residual: Array
    nis: float
    gain: Array
    step: Array


def measurement_update(
    state: TrackerState,
    estimate: Sequence[float] | Array,
    covariance: Array,
) -> tuple[TrackerState, Innovation]:
    """Combine the prior with one period's estimate of the parameter.

    The measurement is a reconciliation's own estimate of the same
    quantity, so the observation matrix is the identity and the update
    is the combination of two Gaussians:

    .. math::

        S = P^- + R, \\quad K = P^- S^{-1}, \\quad
        \\theta^+ = \\theta^- + K(\\hat\\theta - \\theta^-),

    with the covariance taken in Joseph form, :math:`P^+ = (I - K) P^-
    (I - K)^T + K R K^T`. The short form :math:`(I-K)P^-` is algebraically
    equal and numerically worse: it loses symmetry over a long run and
    can go indefinite, and a twin is a long run.

    A weakly informative period needs no special handling. Its ``R`` is
    large, ``K`` goes to zero and the estimate does not move.

    Args:
        state: the prior, normally straight out of :func:`time_update`.
        estimate: the period's estimate, shape ``(p,)``.
        covariance: its covariance, shape ``(p, p)``, full.

    Returns:
        ``(posterior, innovation)``.
    """
    p = state.n
    z = jnp.asarray(estimate, dtype=jnp.float64).reshape(-1)
    if z.shape != (p,):
        raise ValueError(f"estimate has shape {z.shape}, expected {(p,)}")
    r = jnp.asarray(covariance, dtype=jnp.float64)
    if r.shape != (p, p):
        raise ValueError(f"covariance has shape {r.shape}, expected {(p, p)}")

    prior = state.covariance
    v = z - state.mean
    s = _symmetrize(prior + r)
    # K = P S^-1, solved rather than inverted; S is symmetric.
    gain = jnp.linalg.solve(s, prior).T
    mean = state.mean + gain @ v
    i_k = jnp.eye(p) - gain
    cov = _symmetrize(i_k @ prior @ i_k.T + gain @ r @ gain.T)
    nis = float(v @ jnp.linalg.solve(s, v))

    posterior = replace(
        state, mean=mean, covariance=cov, n_updates=state.n_updates + 1
    )
    innovation = Innovation(
        residual=v, nis=nis, gain=gain, step=mean - state.mean
    )
    return posterior, innovation


# --------------------------------------------------------------------------
# one period's estimate of the parameter
# --------------------------------------------------------------------------


def _merge_params(base: Any, theta: Mapping[str, Any]) -> Any:
    """Default injection of tracked parameters into ``params``."""
    if base is None:
        return dict(theta)
    if isinstance(base, Mapping):
        merged = dict(base)
        merged.update(theta)
        return merged
    raise TypeError(
        "the default parameter injection handles params=None and a mapping "
        f"params, but got {type(base).__name__}; pass an explicit inject=..."
    )


def parameter_measurement(
    residual_fn: Callable,
    y: Array,
    sigma: Array,
    state: TrackerState,
    *,
    params: Any = None,
    names: Sequence[str] | None = None,
    inject: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    x0: Array | None = None,
    unmeasured_init: Array | float | None = None,
    **reconcile_kw: Any,
) -> tuple[Array, Array, ReconcileResult]:
    """Estimate the tracked parameters from one period, with covariance.

    The tracked parameters are appended to the state vector with
    ``sigma = inf`` --- free, not merely regularised --- and the
    problem is handed to :func:`~difflow.reconciliation.reconcile`
    unchanged. What comes back is the pair a Kalman update wants: the
    period's own estimate, and the block of
    :func:`~difflow.reconciliation.reconciled_covariance` belonging to
    it.

    Leaving the parameters free here rather than passing the prior in
    as a finite ``sigma`` is deliberate on two counts. ``sigma`` is a
    vector, so a prior smuggled through it would be diagonal and would
    throw away exactly the correlations this module keeps. And with the
    prior outside, the reconciliation's objective stays a test of data
    against model, uninflated by how confident the twin already was.

    The cost is that the period must identify the parameters on its
    own. It if cannot,
    :class:`~difflow.reconciliation.ReconciliationStructureError` is
    raised by the structure check rather than a NaN being returned ---
    pool several periods with
    :func:`~difflow.reconciliation.reconcile_multi` and pass its shared
    estimate and standard errors to :func:`measurement_update` instead.

    Args:
        residual_fn: ``F(x, params) -> (m,)``, JAX-traceable.
        y: the period's measurements, shape ``(n,)``.
        sigma: standard deviations, shape ``(n,)``; ``inf`` marks a
            plant variable that is itself unmeasured.
        state: the tracker, for the parameter names and the starting
            point of the solve.
        params: extra argument threaded to ``residual_fn``, into which
            the tracked parameters are injected.
        names: plant variable names; the parameter names are appended.
        inject: ``inject(params, {name: value}) -> params'``. The
            default handles ``params=None`` and a mapping, matching
            :func:`difflow.mhe.augment_parameters`.
        x0: starting point for the plant block, shape ``(n,)``.
        unmeasured_init: starting value for unmeasured plant entries.
        **reconcile_kw: forwarded to
            :func:`~difflow.reconciliation.reconcile`.

    Returns:
        ``(estimate, covariance, result)`` --- shapes ``(p,)``,
        ``(p, p)`` and the full augmented
        :class:`~difflow.reconciliation.ReconcileResult`.
    """
    inject_fn = _merge_params if inject is None else inject
    p = state.n
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(-1)
    n = y.shape[0]
    if sigma.shape != y.shape:
        raise ValueError(
            f"sigma has shape {sigma.shape} but y has shape {y.shape}"
        )
    plant_names = (
        list(names) if names is not None else [f"x{i}" for i in range(n)]
    )
    if len(plant_names) != n:
        raise ValueError(f"got {len(plant_names)} names for {n} variables")
    clash = set(plant_names) & set(state.names)
    if clash:
        raise ValueError(
            f"parameter names {sorted(clash)} already name plant variables; "
            "the augmented problem needs them distinct"
        )

    def augmented(z: Array, ps: Any) -> Array:
        theta = {nm: z[n + i] for i, nm in enumerate(state.names)}
        return residual_fn(z[:n], inject_fn(ps, theta))

    mask = measured_mask(sigma)
    if x0 is None:
        init = 1.0 if unmeasured_init is None else unmeasured_init
        init = jnp.broadcast_to(jnp.asarray(init, dtype=jnp.float64), y.shape)
        plant0 = jnp.where(mask, y, init)
    else:
        plant0 = jnp.asarray(x0, dtype=jnp.float64).reshape(-1)
        if plant0.shape != y.shape:
            raise ValueError(
                f"x0 has shape {plant0.shape}, expected {y.shape}"
            )

    z0 = jnp.concatenate([plant0, state.mean])
    y_aug = jnp.concatenate([y, state.mean])
    sigma_aug = jnp.concatenate([sigma, jnp.full((p,), jnp.inf)])

    # Scale the parameter block by its own prior spread rather than
    # letting auto_scaling floor it at 1.0, which would ruin the
    # conditioning of a parameter whose natural magnitude is small.
    # A caller-supplied unmeasured_scale describes the plant block only
    # -- the layouts that fill it in, dg.reconcile_network among them,
    # know nothing of the tracked parameters -- so it is extended here
    # rather than passed through at the wrong length.
    prior_std = jnp.sqrt(jnp.clip(jnp.diag(state.covariance), 0.0, jnp.inf))
    theta_scale = jnp.maximum(
        jnp.maximum(jnp.abs(state.mean), prior_std), 1e-12
    )
    given = reconcile_kw.get("unmeasured_scale")
    if given is None:
        plant_scale = jnp.maximum(jnp.abs(plant0), 1.0)
    else:
        plant_scale = jnp.broadcast_to(
            jnp.asarray(given, dtype=jnp.float64), y.shape
        )
    reconcile_kw["unmeasured_scale"] = jnp.concatenate(
        [plant_scale, theta_scale]
    )

    result = reconcile(
        augmented, y_aug, sigma_aug, params=params,
        names=plant_names + list(state.names), x0=z0, **reconcile_kw,
    )
    return result.x[n:], result.covariance[n:, n:], result


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


@dataclass
class TrackStep(ParamsMixin):
    """One period of the tracking loop.

    Attributes:
        index: position in the sequence handed to
            :func:`track_parameters`.
        time: the clock reading of this period.
        monitor: the frozen-model reconciliation's diagnostics, the
            same :class:`~difflow.reconciliation.MonitorStep` the
            routine clock would have recorded.
        diagnosis: the verdict drawn from the campaign so far.
        decision: whether the gate let the parameter move.
        state: the tracker after this period.
        estimate: the period's own estimate, ``None`` when no
            measurement update ran.
        innovation: what the update did, ``None`` when none ran.
        objective: the augmented reconciliation's objective, ``None``
            when no measurement update ran.
        failed: set when the period's estimation could not be posed,
            carrying the reason; the update is then held.
    """

    index: int
    time: float
    monitor: MonitorStep
    diagnosis: MonitorDiagnosis
    decision: GateDecision
    state: TrackerState
    estimate: dict[str, float] | None = None
    innovation: Innovation | None = None
    objective: float | None = None
    failed: str = ""

    @property
    def updated(self) -> bool:
        """Whether the parameter actually moved this period."""
        return self.innovation is not None


@dataclass
class TrackResult(ParamsMixin):
    """A tracking campaign: the gate's decisions and the filtered path.

    Attributes:
        steps: one :class:`TrackStep` per period, in order.
        initial: the state the campaign started from.
        names: the tracked parameter names.
        monitor: the routine clock's campaign, in the form
            :func:`~difflow.reconciliation.monitor` returns. It is not
            the same series that function would have produced on this
            data, because each period is reconciled against the
            *current* estimate rather than the original model --- which
            is the point: the statistic coming back down after an
            update is the loop closing, and its failure to come down is
            how structural mismatch announces itself.
    """

    steps: list[TrackStep]
    initial: TrackerState
    names: list[str] = field(default_factory=list)
    monitor: MonitorResult | None = None

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def final(self) -> TrackerState:
        """The state after the last period."""
        return self.steps[-1].state if self.steps else self.initial

    @property
    def n_updates(self) -> int:
        """How many periods moved the parameter."""
        return sum(s.updated for s in self.steps)

    @property
    def trajectory(self) -> np.ndarray:
        """Estimate after each period, shape ``(n_steps, p)``."""
        if not self.steps:
            return np.zeros((0, len(self.names)))
        return np.stack([np.asarray(s.state.mean, dtype=float)
                         for s in self.steps])

    @property
    def std_trajectory(self) -> np.ndarray:
        """Standard error after each period, shape ``(n_steps, p)``."""
        if not self.steps:
            return np.zeros((0, len(self.names)))
        return np.stack([
            np.sqrt(np.clip(np.diag(np.asarray(s.state.covariance,
                                               dtype=float)), 0.0, np.inf))
            for s in self.steps
        ])

    def of(self, name: str) -> np.ndarray:
        """The path of one parameter, shape ``(n_steps,)``."""
        return self.trajectory[:, self.names.index(name)]

    def std_of(self, name: str) -> np.ndarray:
        """The standard-error path of one parameter, shape ``(n_steps,)``."""
        return self.std_trajectory[:, self.names.index(name)]

    def summary(self) -> str:
        """One line per period, oldest first, then the final estimate."""
        lines = [
            f"{'step':>5} {'chi2':>10} {'verdict':>16} {'gate':>7} "
            f"{'NIS':>8}  " + "  ".join(f"{nm:>12}" for nm in self.names),
            "-" * (50 + 14 * len(self.names)),
        ]
        for s in self.steps:
            chi2 = "-" if s.monitor.failed else f"{s.monitor.statistic:10.3f}"
            nis = "-" if s.innovation is None else f"{s.innovation.nis:8.2f}"
            vals = "  ".join(
                f"{float(np.asarray(s.state.mean)[i]):12.6g}"
                for i in range(len(self.names))
            )
            gate = "update" if s.updated else "hold"
            lines.append(
                f"{s.index:5d} {chi2:>10} {s.diagnosis.verdict:>16} "
                f"{gate:>7} {nis:>8}  {vals}"
            )
        lines += ["", self.final.summary()]
        return "\n".join(lines)


def track_parameters(
    residual_fn: Callable,
    measurements: Sequence[Any],
    sigma: Any,
    *,
    state: TrackerState,
    drift_std: Sequence[float] | Array | float,
    times: Sequence[float] | None = None,
    params: Any = None,
    names: Sequence[str] | None = None,
    inject: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    alpha: float = 0.05,
    window: int | None = 15,
    rejection_threshold: float = REJECTION_THRESHOLD,
    concentration_threshold: float = CONCENTRATION_THRESHOLD,
    allow: frozenset[str] | Sequence[str] = UPDATE_WHEN_DRIFTING,
    max_std: Sequence[float] | Array | float | None = None,
    keep_results: bool = False,
    **reconcile_kw: Any,
) -> TrackResult:
    """Run the gated filter over a campaign, one period at a time.

    Each period does four things, in this order:

    1. reconcile with the parameters **frozen** at the current estimate
       and record the global and measurement tests --- the routine
       clock, and the only reason the tests mean anything;
    2. draw a verdict from the campaign so far and put it to
       :func:`update_gate`;
    3. :func:`time_update` the tracker, whatever the verdict said;
    4. only if the gate opened, estimate the parameters from this
       period with :func:`parameter_measurement` and fold the result in
       with :func:`measurement_update`.

    Step 1 is run against the *current* estimate rather than the
    original model, so the statistic series shows the loop closing: a
    drift is detected, the gate opens, the parameter moves and the
    statistic falls back under its critical value. Watching it fail to
    fall is how a structural mismatch announces itself --- no value of
    the parameter fits, and
    :func:`difflow.planning.modifiers.update_modifiers` is the answer
    rather than a faster filter.

    This is the offline driver, and it is also a backtest: run it over
    a recorded campaign to choose ``drift_std`` before trusting the
    loop live. The four pieces are public and stateless, so an online
    loop is the same four calls with the state carried between them.

    Args:
        residual_fn: ``F(x, params) -> (m,)``, JAX-traceable.
        measurements: one measurement vector per period, each ``(n,)``.
        sigma: standard deviations, shape ``(n,)``, shared by every
            period; ``inf`` marks an unmeasured plant variable.
        state: the starting tracker, from
            :meth:`TrackerState.initial`.
        drift_std: :math:`\\sqrt{\\mathrm{diag}(Q)}` per unit time. Required:
            it is the bandwidth of the twin, not a nuisance. See
            :func:`drift_std_from_time_constant`.
        times: the clock reading of each period. Defaults to
            ``state.time, state.time + 1, ...``, which makes
            ``drift_std`` a per-period figure. Note what that means for
            the first period: it carries the state's own clock reading,
            so its ``dt`` is zero and ``n`` periods produce ``n - 1``
            drift increments. That is deliberate --- the prior is
            as-of its own clock, not one period stale --- and if the
            state was last updated some time before the record starts,
            say so by passing ``times`` explicitly rather than by
            back-dating ``state.time``.
        params: extra argument threaded to ``residual_fn``, into which
            the tracked parameters are injected.
        names: plant variable names.
        inject: ``inject(params, {name: value}) -> params'``.
        alpha: significance level for both gross-error tests.
        window: periods the verdict is drawn from; see
            :meth:`~difflow.reconciliation.MonitorResult.diagnose`.
        rejection_threshold: fraction of the window that must reject
            before any fault is declared.
        concentration_threshold: blame concentration at or above which
            the fault is read as one sensor rather than the model.
            Together with ``window`` this is what decides whether a
            real instrument fault is *always* caught: the defaults are
            a rule, not a guarantee, and a campaign where a few days
            slip through as ``model drift`` is telling you to lengthen
            the window or lower this, not to widen ``allow``.
        allow: verdicts the gate updates on.
        max_std: ceiling on each parameter's standard error; see
            :func:`time_update`.
        keep_results: retain each full
            :class:`~difflow.reconciliation.ReconcileResult` on its
            monitoring step.
        **reconcile_kw: forwarded to
            :func:`~difflow.reconciliation.reconcile`.

    Returns:
        A :class:`TrackResult`.

    Raises:
        ValueError: if ``times`` does not match ``measurements``.

    Example:
        >>> run = track_parameters(                    # doctest: +SKIP
        ...     F, sixty_days, sigma,
        ...     state=TrackerState.initial(["eta"], [1.0], std=[0.05]),
        ...     drift_std=drift_std_from_time_constant(0.05, 30.0),
        ...     names=layout.names,
        ... )
        >>> run.final.as_params()                      # doctest: +SKIP
        {'eta': 0.913...}
    """
    inject_fn = _merge_params if inject is None else inject
    measurements = list(measurements)
    if times is None:
        clock = [float(state.time + i) for i in range(len(measurements))]
    else:
        clock = [float(t) for t in times]
        if len(clock) != len(measurements):
            raise ValueError(
                f"got {len(clock)} times for {len(measurements)} periods"
            )

    plant_names = list(names) if names is not None else None
    monitor_steps: list[MonitorStep] = []
    steps: list[TrackStep] = []
    initial = state

    for i, (t, y) in enumerate(zip(clock, measurements)):
        # 1. the routine clock: the model is frozen at the current
        #    estimate, so the tests are about the sensors.
        frozen = inject_fn(params, state.as_params())
        try:
            res = reconcile(
                residual_fn, y, sigma, params=frozen, names=plant_names,
                **reconcile_kw,
            )
        except ReconciliationStructureError as err:
            monitor_steps.append(
                MonitorStep(
                    index=i, statistic=float("nan"), dof=-1,
                    critical=float("nan"), p_value=float("nan"),
                    detected=False, suspect=None, z_max=float("nan"),
                    failed=str(err).split(":", 1)[0],
                )
            )
        else:
            gt = global_test(res, alpha=alpha)
            mt = measurement_test(res, alpha=alpha)
            if plant_names is None:
                plant_names = list(res.names)
            monitor_steps.append(
                MonitorStep(
                    index=i, statistic=gt.statistic, dof=gt.dof,
                    critical=gt.critical, p_value=gt.p_value,
                    detected=gt.detected, suspect=mt.suspect, z_max=mt.z_max,
                    result=res if keep_results else None,
                )
            )

        # 2. the verdict, and the gate.
        campaign = MonitorResult(
            steps=list(monitor_steps), names=plant_names or [], alpha=alpha
        )
        diagnosis = campaign.diagnose(
            window,
            rejection_threshold=rejection_threshold,
            concentration_threshold=concentration_threshold,
        )
        decision = update_gate(diagnosis, allow=allow)

        # 3. the random walk runs whether or not the data were usable.
        state = time_update(state, t - state.time, drift_std, max_std=max_std)

        # 4. the campaign clock, only where the gate opened.
        estimate = innovation = objective = None
        failed = ""
        if decision.allowed:
            try:
                theta_hat, r, aug = parameter_measurement(
                    residual_fn, y, sigma, state, params=params,
                    names=plant_names, inject=inject_fn, **reconcile_kw,
                )
            except ReconciliationStructureError as err:
                failed = str(err).split(":", 1)[0]
            else:
                state, innovation = measurement_update(state, theta_hat, r)
                estimate = {
                    nm: float(np.asarray(theta_hat, dtype=float)[j])
                    for j, nm in enumerate(state.names)
                }
                objective = aug.objective
        if innovation is None:
            state = replace(state, n_held=state.n_held + 1)

        steps.append(
            TrackStep(
                index=i, time=t, monitor=monitor_steps[-1],
                diagnosis=diagnosis, decision=decision, state=state,
                estimate=estimate, innovation=innovation,
                objective=objective, failed=failed,
            )
        )

    return TrackResult(
        steps=steps,
        initial=initial,
        names=list(initial.names),
        monitor=MonitorResult(
            steps=monitor_steps, names=plant_names or [], alpha=alpha
        ),
    )


def _symmetrize(p: Array) -> Array:
    """Average a matrix with its transpose, killing round-off asymmetry."""
    return 0.5 * (p + p.T)
