"""The extended Kalman filter: the cheap baseline, and the arrival cost.

The EKF earns its place here twice over. It is the estimator to beat ---
one Jacobian and one linear solve per sample, against an optimisation
per sample for moving-horizon estimation --- so a comparison against it
is the honest way to say whether the extra cost bought anything. And its
covariance recursion is what summarises the past that falls out of the
back of a moving horizon: the arrival cost of
:mod:`difflow.mhe.arrival` *is* the EKF covariance, propagated along the
trajectory the optimiser found rather than along the filter's own.

Where it fails is exactly where MHE is worth paying for. The EKF
linearises once, at the current mean, and then commits: a bad
linearisation cannot be revisited, and there is no way to say that a
concentration is non-negative --- the update is a linear correction that
will happily produce one. It also has no notion of a measurement drawn
in the past, because its state is a single mean and covariance at the
current time.

Multi-rate and missing data are handled without changing any shape. A
channel whose sigma is infinite has its row of ``H`` zeroed and its
noise variance set to one, which makes the corresponding column of the
gain exactly zero: the reading contributes nothing, at fixed cost and
fixed shape, so the whole filter stays inside one ``lax.scan``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.reconciliation import measured_mask

from difflow.mhe.measurements import MeasurementWindow
from difflow.mhe.model import StateSpaceModel


@dataclass
class EKFState(ParamsMixin):
    """Mean and covariance of the state at one time.

    Attributes:
        x: state estimate, shape ``(n_x,)``.
        P: its covariance, shape ``(n_x, n_x)``.
        time: the time it refers to.
    """

    x: Array
    P: Array
    time: float = 0.0

    @property
    def std(self) -> Array:
        """Marginal standard deviations, shape ``(n_x,)``."""
        return jnp.sqrt(jnp.clip(jnp.diag(self.P), 0.0, jnp.inf))


def ekf_predict(
    model: StateSpaceModel,
    x: Array,
    p: Array,
    u: Array,
    process_std: Array,
    theta: Any = None,
    *,
    x_lin: Array | None = None,
) -> tuple[Array, Array]:
    """Propagate mean and covariance one sampling interval.

    Args:
        model: the plant model.
        x: current mean, shape ``(n_x,)``.
        p: current covariance, shape ``(n_x, n_x)``.
        u: input over the interval, shape ``(n_u,)``.
        process_std: standard deviation of ``w``, shape ``(n_w,)``.
        theta: model parameters.
        x_lin: point to take the Jacobians at, defaulting to ``x``.
            Passing the moving-horizon solution here is what turns this
            recursion into the MHE arrival cost.

    Returns:
        ``(x_next, P_next)``.
    """
    w0 = jnp.zeros((int(model.n_w),), dtype=jnp.float64)
    lin = x if x_lin is None else x_lin
    f_x, f_w = model.jacobians(lin, u, w0, theta)
    x_next = model.step(x, u, w0, theta)
    q = jnp.diag(jnp.asarray(process_std, dtype=jnp.float64) ** 2)
    p_next = f_x @ p @ f_x.T + f_w @ q @ f_w.T
    return x_next, _symmetrize(p_next)


def ekf_update(
    model: StateSpaceModel,
    x: Array,
    p: Array,
    y: Array,
    sigma: Array,
    u: Array,
    theta: Any = None,
    *,
    x_lin: Array | None = None,
    clip: bool = True,
) -> tuple[Array, Array, Array]:
    """Correct mean and covariance with one (possibly partial) reading.

    Channels with infinite sigma are switched off exactly, not
    approximated by a large variance: their rows of ``H`` are zeroed and
    their variance set to one, so the corresponding gain columns vanish
    and the numbers stay well conditioned however sparse the sampling.

    Args:
        model: the plant model.
        x: prior mean, shape ``(n_x,)``.
        p: prior covariance, shape ``(n_x, n_x)``.
        y: reading, shape ``(n_y,)``.
        sigma: its standard deviations, shape ``(n_y,)``; ``inf`` where
            the channel was not sampled.
        u: current input, shape ``(n_u,)``.
        theta: model parameters.
        x_lin: point to take ``H`` at, defaulting to ``x``.
        clip: project the corrected mean onto the model's state bounds.
            This is a projection, not a constrained solve --- the
            covariance does not know about it --- and it is one of the
            reasons to prefer MHE when the bounds are active.

    Returns:
        ``(x_post, P_post, innovation)``, where the innovation is zero
        on unsampled channels.
    """
    mask = measured_mask(sigma)
    lin = x if x_lin is None else x_lin
    h = model.observation_jacobian(lin, u, theta)
    h = jnp.where(mask[:, None], h, 0.0)
    var = jnp.where(mask, jnp.where(mask, sigma, 1.0) ** 2, 1.0)
    r = jnp.diag(var)

    innovation = jnp.where(mask, y - model.observe(x, u, theta), 0.0)
    s = h @ p @ h.T + r
    gain = jnp.linalg.solve(s.T, (p @ h.T).T).T  # P H^T S^{-1}
    x_post = x + gain @ innovation

    # Joseph form: stays symmetric positive semi-definite even when the
    # gain is not the optimal one, which it is not once the model is
    # nonlinear or the mean has been clipped.
    ikh = jnp.eye(p.shape[0]) - gain @ h
    p_post = ikh @ p @ ikh.T + gain @ r @ gain.T

    if clip and model.is_bounded:
        lb, ub = model.bounds
        x_post = jnp.clip(x_post, lb, ub)
    return x_post, _symmetrize(p_post), innovation


@dataclass
class EKFRunResult(ParamsMixin):
    """Filtered trajectory over a window.

    Attributes:
        times: grid times, shape ``(K + 1,)``.
        x: filtered means, shape ``(K + 1, n_x)``; row ``k`` is the
            estimate of ``x_k`` given readings up to and including
            ``k``.
        P: their covariances, shape ``(K + 1, n_x, n_x)``.
        innovations: shape ``(K + 1, n_y)``, zero where unsampled.
        x_names: state names.
    """

    times: Array
    x: Array
    P: Array
    innovations: Array
    x_names: list[str] = field(default_factory=list)

    @property
    def final(self) -> EKFState:
        """The last filtered state."""
        return EKFState(x=self.x[-1], P=self.P[-1],
                        time=float(self.times[-1]))

    @property
    def std(self) -> Array:
        """Marginal standard deviations, shape ``(K + 1, n_x)``."""
        return jnp.sqrt(
            jnp.clip(jnp.diagonal(self.P, axis1=1, axis2=2), 0.0, jnp.inf)
        )

    def named(self, k: int = -1) -> dict[str, float]:
        """One filtered state as ``{name: value}``."""
        return {nm: float(self.x[k, i])
                for i, nm in enumerate(self.x_names)}


def run_ekf(
    model: StateSpaceModel,
    window: MeasurementWindow,
    *,
    x0: Array,
    P0: Array,
    process_std: Array,
    theta: Any = None,
    clip: bool = True,
) -> EKFRunResult:
    """Filter a whole window, update-then-predict at each grid point.

    Args:
        model: the plant model.
        window: the measurements, from
            :func:`~difflow.mhe.build_window`.
        x0: prior mean at ``window.times[0]``, shape ``(n_x,)``.
        P0: prior covariance, shape ``(n_x, n_x)``.
        process_std: standard deviation of ``w``, shape ``(n_w,)``.
        theta: model parameters.
        clip: project each corrected mean onto the state bounds.

    Returns:
        An :class:`EKFRunResult`.

    Example:
        >>> res = run_ekf(model, window, x0=x0, P0=P0,   # doctest: +SKIP
        ...               process_std=jnp.array([0.01]))
        >>> res.final.x                                  # doctest: +SKIP
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    p0 = jnp.asarray(P0, dtype=jnp.float64)
    process_std = jnp.broadcast_to(
        jnp.asarray(process_std, dtype=jnp.float64), (int(model.n_w),)
    )
    # One trailing input row so the scan is rectangular; the prediction
    # it drives is computed and then thrown away with the carry.
    u = jnp.concatenate(
        [window.u,
         jnp.zeros((1, window.u.shape[1]), dtype=jnp.float64)],
        axis=0,
    )

    def body(carry, inputs):
        x, p = carry
        y_k, sigma_k, u_k = inputs
        x_up, p_up, innov = ekf_update(
            model, x, p, y_k, sigma_k, u_k, theta, clip=clip
        )
        x_next, p_next = ekf_predict(
            model, x_up, p_up, u_k, process_std, theta
        )
        return (x_next, p_next), (x_up, p_up, innov)

    _, (xs, ps, innovations) = jax.lax.scan(
        body, (x0, p0), (window.y, window.sigma, u)
    )
    return EKFRunResult(
        times=window.times,
        x=xs,
        P=ps,
        innovations=innovations,
        x_names=list(model.x_names),
    )


def _symmetrize(p: Array) -> Array:
    """Average a covariance with its transpose.

    Rounding makes the recursions drift out of symmetry over a long
    campaign, and an asymmetric covariance eventually produces a
    Cholesky failure in the arrival cost.
    """
    return 0.5 * (p + p.T)
