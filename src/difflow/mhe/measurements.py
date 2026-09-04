"""Timestamped, multi-rate and delayed measurements over an estimation window.

A running plant does not hand an estimator a tidy matrix. Pressures
arrive every second, temperatures every minute, and the assay that says
what the process actually did is drawn on a shift cadence and reported
hours after the sample was taken. Two facts follow, and this module
exists so that neither is bolted on later:

* **A measurement has two times.** ``time`` is when the sample was
  *taken* --- the state it constrains --- and ``reported`` is when the
  value became available. An estimator running at ``now`` may use every
  record with ``reported <= now``, but each one is placed against the
  state at its own ``time``. Placing a six-hour-old assay against the
  current state is not a small error: it asserts the plant is where it
  was six hours ago, and the estimate is wrong by whatever happened in
  between. :func:`build_window` gets this right by construction.
* **A channel that is not sampled is not a missing value.** It is a
  measurement of infinite variance. Writing it that way keeps the array
  shapes fixed under ``jit`` and reuses
  :func:`difflow.reconciliation.measured_mask`, so "unmeasured" means
  exactly the same thing here as it does in steady-state
  reconciliation.

The window is therefore dense: ``y`` and ``sigma`` are both
``(K + 1, n_y)``, with ``sigma = inf`` wherever a channel carries no
information at that time.

Example:
    >>> import numpy as np
    >>> times = np.arange(4.0)
    >>> records = [Measurement(time=0.0, values={"T": 350.0},
    ...                        sigma={"T": 1.0}),
    ...            Measurement(time=1.0, values={"assay": 0.31},
    ...                        sigma={"assay": 0.01}, reported=3.0)]
    >>> win, dropped = build_window(times, records,
    ...                             y_names=["T", "assay"])
    >>> win.n_measurements
    2
    >>> bool(np.isfinite(np.asarray(win.sigma)[1, 1]))   # assay sits at t=1
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.reconciliation import measured_mask


@dataclass
class Measurement:
    """One timestamped reading, of any subset of the channels.

    Attributes:
        time: when the sample was taken. This is the time whose state
            the reading constrains, and it is *not* the time the value
            arrived.
        values: ``{channel: value}``, or a full ``(n_y,)`` array.
        sigma: standard deviation, in the same form as ``values``, or a
            scalar applied to every channel given.
        reported: when the value became available, defaulting to
            ``time`` (an instrument with no delay). A lab assay sets
            this to the time the result came back; an estimator running
            at ``now`` then correctly cannot see it before then.
        label: free-form tag carried through for reporting.
    """

    time: float
    values: Mapping[str, float] | Sequence[float] | Array
    sigma: Mapping[str, float] | Sequence[float] | Array | float
    reported: float | None = None
    label: str = ""

    @property
    def available_at(self) -> float:
        """When the reading can first be used."""
        return float(self.time if self.reported is None else self.reported)

    @property
    def delay(self) -> float:
        """How long after the sample the value arrived."""
        return self.available_at - float(self.time)


@dataclass
class MeasurementWindow(ParamsMixin):
    """Everything an estimator needs about one horizon.

    Attributes:
        times: grid times ``t_0 ... t_K``, shape ``(K + 1,)``.
        y: values, shape ``(K + 1, n_y)``. Entries whose sigma is
            infinite are meaningless and may be anything finite.
        sigma: standard deviations, shape ``(K + 1, n_y)``; ``inf``
            marks a channel not sampled at that time.
        u: inputs over the ``K`` intervals, shape ``(K, n_u)``.
        y_names: channel names.
        records: the :class:`Measurement` records that produced the
            window, kept for reporting; not used in the arithmetic.
    """

    times: Array
    y: Array
    sigma: Array
    u: Array
    y_names: list[str] = field(default_factory=list)
    records: list[Measurement] = field(default_factory=list)

    @property
    def horizon(self) -> int:
        """Number of intervals ``K`` in the window."""
        return int(self.times.shape[0]) - 1

    @property
    def n_y(self) -> int:
        return int(self.y.shape[1])

    @property
    def mask(self) -> Array:
        """Boolean ``(K + 1, n_y)``: which entries carry information."""
        return measured_mask(self.sigma)

    @property
    def n_measurements(self) -> int:
        """How many scalar readings the window actually contains."""
        return int(jnp.sum(self.mask))

    def summary(self) -> str:
        """Per-channel sampling counts and the delays seen."""
        mask = np.asarray(self.mask)
        names = self.y_names or [f"y{j}" for j in range(self.n_y)]
        lines = [
            f"window {float(self.times[0]):g} -> {float(self.times[-1]):g} "
            f"({self.horizon} intervals), "
            f"{self.n_measurements} of {mask.size} entries measured",
            "",
            f"{'channel':<20} {'samples':>8} {'duty':>8}",
            "-" * 40,
        ]
        for j, nm in enumerate(names):
            k = int(mask[:, j].sum())
            lines.append(f"{nm:<20} {k:8d} {k / mask.shape[0]:8.2f}")
        delays = [r.delay for r in self.records if r.delay > 0]
        if delays:
            lines.append("")
            lines.append(
                f"{len(delays)} delayed record(s), "
                f"max delay {max(delays):g}"
            )
        return "\n".join(lines)


def build_window(
    times: Sequence[float] | Array,
    measurements: Iterable[Measurement],
    *,
    y_names: Sequence[str] | None = None,
    n_y: int | None = None,
    u: Array | None = None,
    n_u: int = 0,
    now: float | None = None,
    snap_tol: float | None = None,
) -> tuple[MeasurementWindow, list[Measurement]]:
    """Place timestamped records on an estimation grid.

    Each record is assigned to the grid time nearest its ``time`` --- the
    time the *sample* was taken --- so a delayed reading constrains the
    state it was actually drawn from. Two records landing on the same
    grid time and channel are combined by inverse-variance weighting,
    which is the exact posterior for independent Gaussian readings of
    the same quantity, not an approximation.

    Args:
        times: grid times, shape ``(K + 1,)``, assumed increasing and
            uniformly spaced.
        measurements: the records to place. Order does not matter.
        y_names: channel names; required if any record uses mapping
            ``values``.
        n_y: number of channels; inferred from ``y_names`` if absent.
        u: inputs, shape ``(K, n_u)``; defaults to zeros.
        n_u: input dimension, used when ``u`` is not given.
        now: the estimation time. Records with ``available_at > now``
            have not been reported yet and are excluded, which is how a
            realistic replay of a delayed assay is set up. ``None``
            (default) uses every record.
        snap_tol: how far a record's time may sit from a grid point,
            defaulting to half the grid spacing. A record further away
            than this from every grid point --- including one before
            the window starts, whose information now lives in the
            arrival cost --- is returned rather than silently misplaced.

    Returns:
        ``(window, dropped)``, the assembled
        :class:`MeasurementWindow` and the records that fell outside it.

    Raises:
        ValueError: on inconsistent shapes, or if a record names a
            channel that is not in ``y_names``.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"times must be 1-D with at least 2 points, got {t.shape}")
    n_t = t.size
    k = n_t - 1

    if y_names is not None:
        y_names = list(y_names)
        if n_y is not None and n_y != len(y_names):
            raise ValueError(
                f"n_y={n_y} disagrees with {len(y_names)} y_names"
            )
        n_y = len(y_names)
    elif n_y is None:
        raise ValueError("build_window needs y_names or n_y")
    else:
        y_names = [f"y{j}" for j in range(n_y)]

    if snap_tol is None:
        spacing = float(np.min(np.diff(t))) if n_t > 1 else 1.0
        snap_tol = 0.5 * spacing

    # Accumulate in precision (1/sigma^2) form: adding a second reading
    # of the same quantity then costs one addition and is exact.
    precision = jnp.zeros((n_t, n_y), dtype=jnp.float64)
    weighted = jnp.zeros((n_t, n_y), dtype=jnp.float64)
    kept: list[Measurement] = []
    dropped: list[Measurement] = []

    for record in measurements:
        if now is not None and record.available_at > float(now) + 1e-12:
            dropped.append(record)
            continue
        idx = int(np.argmin(np.abs(t - float(record.time))))
        if abs(t[idx] - float(record.time)) > snap_tol:
            dropped.append(record)
            continue
        cols, vals, sigs = _unpack(record, y_names)
        sigs = np.asarray(sigs, dtype=float)
        good = np.asarray(measured_mask(jnp.asarray(sigs)))
        if not good.any():
            dropped.append(record)
            continue
        cols = np.asarray(cols)[good]
        vals = jnp.asarray(np.asarray(vals)[good], dtype=jnp.float64)
        w = jnp.asarray(1.0 / sigs[good] ** 2, dtype=jnp.float64)
        rows = np.full(cols.shape, idx)
        precision = precision.at[rows, cols].add(w)
        weighted = weighted.at[rows, cols].add(w * vals)
        kept.append(record)

    have = precision > 0
    y = jnp.where(have, weighted / jnp.where(have, precision, 1.0), 0.0)
    sigma = jnp.where(
        have, 1.0 / jnp.sqrt(jnp.where(have, precision, 1.0)), jnp.inf
    )

    if u is None:
        u_arr = jnp.zeros((k, n_u), dtype=jnp.float64)
    else:
        u_arr = jnp.atleast_2d(jnp.asarray(u, dtype=jnp.float64))
        if u_arr.shape[0] != k:
            raise ValueError(
                f"u has {u_arr.shape[0]} rows but the window has {k} intervals"
            )

    return (
        MeasurementWindow(
            times=jnp.asarray(t, dtype=jnp.float64),
            y=y,
            sigma=sigma,
            u=u_arr,
            y_names=list(y_names),
            records=kept,
        ),
        dropped,
    )


def slice_window(window: MeasurementWindow, start: int, horizon: int) -> MeasurementWindow:
    """The sub-window ``[start, start + horizon]`` of a longer record.

    Used by :func:`~difflow.mhe.estimator.run_mhe` to walk a campaign,
    where the whole history is assembled once and the estimator sees a
    fixed-length view of it.

    Args:
        window: the full record.
        start: index of the first grid point of the sub-window.
        horizon: number of intervals in it.

    Returns:
        A :class:`MeasurementWindow` with ``horizon + 1`` grid points.
    """
    stop = start + horizon + 1
    if start < 0 or stop > window.times.shape[0]:
        raise IndexError(
            f"sub-window [{start}, {stop}) does not fit a record of "
            f"{int(window.times.shape[0])} points"
        )
    return MeasurementWindow(
        times=window.times[start:stop],
        y=window.y[start:stop],
        sigma=window.sigma[start:stop],
        u=window.u[start:stop - 1],
        y_names=list(window.y_names),
        records=[
            r for r in window.records
            if float(window.times[start]) - 1e-12
            <= float(r.time)
            <= float(window.times[stop - 1]) + 1e-12
        ],
    )


def _unpack(
    record: Measurement, y_names: Sequence[str]
) -> tuple[Any, Any, Any]:
    """Return ``(column indices, values, sigmas)`` for one record."""
    values, sigma = record.values, record.sigma
    if isinstance(values, Mapping):
        cols, vals = [], []
        for name, v in values.items():
            try:
                cols.append(list(y_names).index(name))
            except ValueError:
                raise ValueError(
                    f"measurement at t={record.time} names channel "
                    f"{name!r}, which is not one of {list(y_names)}"
                ) from None
            vals.append(v)
        if isinstance(sigma, Mapping):
            sigs = [sigma[name] for name in values]
        else:
            sigs = [float(sigma)] * len(vals)
        return cols, vals, sigs

    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != len(y_names):
        raise ValueError(
            f"measurement at t={record.time} has {vals.size} values but "
            f"the model has {len(y_names)} channels; use a mapping to give "
            "a subset"
        )
    if isinstance(sigma, Mapping):
        raise ValueError(
            "a mapping sigma needs a mapping values, so the channels line up"
        )
    sigs = np.broadcast_to(
        np.asarray(sigma, dtype=float), vals.shape
    ).astype(float)
    return list(range(len(y_names))), vals, sigs
