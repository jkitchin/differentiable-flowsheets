"""Sensor placement: what a proposed measurement would actually buy.

Reconciled precision comes from the constraints as much as from the
meters, so the value of a new sensor is not its own accuracy but how
much it shrinks the uncertainty of the quantity you care about --- and
that is a question the covariance answers before anyone buys anything.

Everything here reuses :func:`difflow.reconciliation.reconciled_
covariance`; adding a candidate sensor just means giving its variable a
finite sigma.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.reconciliation.core import (
    Scaling,
    auto_scaling,
    reconciled_covariance,
)


def _std_of(
    residual_fn, x, sigma, target_index, *, params, scaling
) -> float:
    cov = reconciled_covariance(
        residual_fn, x, sigma, scaling=scaling, params=params
    )
    return float(jnp.sqrt(jnp.clip(cov[target_index, target_index], 0.0, jnp.inf)))


def sensor_value(
    residual_fn: Callable,
    x: Array,
    sigma: Array,
    *,
    target: int | str,
    candidate: int | str,
    candidate_sigma: float,
    names: Sequence[str] | None = None,
    params: Any = None,
    scaling: Scaling | None = None,
) -> dict:
    """Effect of adding one sensor on the precision of a target quantity.

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        x: state to linearize about (normally the reconciled state).
        sigma: current standard deviations; ``inf`` = unmeasured.
        target: index or name of the quantity to be estimated well.
        candidate: index or name of the variable a sensor would measure.
        candidate_sigma: accuracy of the proposed sensor.
        names: variable names, needed if ``target``/``candidate`` are
            given by name.
        params: extra argument threaded to ``residual_fn``.
        scaling: scaling to use; derived automatically if omitted.

    Returns:
        ``{'sd_before', 'sd_after', 'variance_reduction', 'sd_reduction'}``,
        where the reductions are fractions in ``[0, 1]``.
    """
    names = list(names) if names is not None else None

    def _idx(v):
        if isinstance(v, str):
            if names is None:
                raise ValueError("names are required to look up a variable")
            return names.index(v)
        return int(v)

    t, c = _idx(target), _idx(candidate)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    sc_before = scaling or auto_scaling(residual_fn, x, sigma, params=params)
    sd_before = _std_of(
        residual_fn, x, sigma, t, params=params, scaling=sc_before
    )

    sigma_after = sigma.at[c].set(candidate_sigma)
    sc_after = auto_scaling(residual_fn, x, sigma_after, params=params)
    sd_after = _std_of(
        residual_fn, x, sigma_after, t, params=params, scaling=sc_after
    )

    var_red = (
        1.0 - (sd_after ** 2) / (sd_before ** 2) if sd_before > 0 else 0.0
    )
    return {
        "target": names[t] if names else t,
        "candidate": names[c] if names else c,
        "sd_before": sd_before,
        "sd_after": sd_after,
        "variance_reduction": float(var_red),
        "sd_reduction": float(1.0 - sd_after / sd_before) if sd_before > 0 else 0.0,
    }


def sensor_ranking(
    residual_fn: Callable,
    x: Array,
    sigma: Array,
    *,
    target: int | str,
    candidates: Sequence[int | str],
    candidate_sigma: float | Sequence[float],
    names: Sequence[str] | None = None,
    params: Any = None,
) -> list[dict]:
    """Rank candidate sensors by how much they sharpen a target estimate.

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        x: state to linearize about.
        sigma: current standard deviations.
        target: the quantity to be estimated well.
        candidates: variables a sensor could be added to.
        candidate_sigma: accuracy of each candidate; a scalar applies
            to all of them.
        names: variable names.
        params: extra argument threaded to ``residual_fn``.

    Returns:
        The :func:`sensor_value` dicts, best first.
    """
    if np.isscalar(candidate_sigma):
        sigmas = [float(candidate_sigma)] * len(candidates)
    else:
        sigmas = [float(s) for s in candidate_sigma]
        if len(sigmas) != len(candidates):
            raise ValueError(
                f"got {len(sigmas)} sigmas for {len(candidates)} candidates"
            )

    out = [
        sensor_value(
            residual_fn, x, sigma, target=target, candidate=c,
            candidate_sigma=s, names=names, params=params,
        )
        for c, s in zip(candidates, sigmas)
    ]
    return sorted(out, key=lambda d: -d["variance_reduction"])
