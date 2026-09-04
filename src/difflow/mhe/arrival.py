"""The arrival cost: what a finite horizon must remember about the past.

Full-information estimation uses every measurement ever taken, and its
cost grows without bound. A moving horizon keeps the last ``K``
intervals and replaces everything before them with a single term on the
first state of the window,

.. math::

    \\Gamma_{k-K}(x) = (x - \\bar x)^T P^{-1} (x - \\bar x),

the *arrival cost*. Choosing it well is the whole difficulty of the
method: it is the only thing standing between a bounded computation and
throwing away the plant's history. Rao, Rawlings and Mayne
(doi:10.1109/TAC.2002.808470) show that a *smoothing* update of this
term keeps the estimator stable, and that the exact arrival cost is
generally unavailable for a nonlinear model --- so what is used in
practice is its Gaussian approximation, whose covariance follows the
extended Kalman filter recursion.

That is what :func:`advance_arrival_cost` does, with one difference
from running an EKF on its own that matters: the Jacobians are taken
along the trajectory the *optimiser* found, not along the filter's own
mean. When the two differ --- which is precisely when constraints bind
or the model is strongly nonlinear, i.e. when MHE is worth its cost ---
the MHE trajectory is the better linearisation point.

The mean is likewise the optimiser's, not the filter's: after solving
over ``[k-K, k]`` the estimate of :math:`x_{k-K+1}` is a *smoothed* one,
informed by every measurement in the window, and it is that value the
next window inherits.

A word of warning that the theory makes explicit. The arrival cost is
a summary, and a summary of a constrained problem by an unconstrained
quadratic can be over-confident: information the constraints supplied
is not represented in ``P``, so the term can become tighter than the
data warrant and the estimator stops responding to new information.
:func:`inflate` is the blunt, standard remedy, and
:attr:`ArrivalCost.condition` is how you notice you need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import solve_triangular

from difflow.params_mixin import ParamsMixin

from difflow.mhe.ekf import ekf_predict, ekf_update
from difflow.mhe.measurements import MeasurementWindow
from difflow.mhe.model import StateSpaceModel

#: Added to the diagonal before a Cholesky, relative to its trace.
CHOLESKY_JITTER = 1e-12


@dataclass
class ArrivalCost(ParamsMixin):
    """A Gaussian summary of everything before the horizon.

    Attributes:
        x_bar: prior mean of the first state in the window, ``(n_x,)``.
        P: its covariance, ``(n_x, n_x)``. A large entry says the past
            has little to say about that state; an infinite one is not
            representable, so use a large finite variance instead.
        time: the grid time it applies at.
    """

    x_bar: Array
    P: Array
    time: float = 0.0

    @classmethod
    def diagonal(
        cls, x_bar: Array, std: Array | float, time: float = 0.0
    ) -> "ArrivalCost":
        """An arrival cost with independent states of given std."""
        x_bar = jnp.asarray(x_bar, dtype=jnp.float64)
        std = jnp.broadcast_to(
            jnp.asarray(std, dtype=jnp.float64), x_bar.shape
        )
        return cls(x_bar=x_bar, P=jnp.diag(std ** 2), time=time)

    @classmethod
    def vague(cls, x_bar: Array, scale: float = 1e6) -> "ArrivalCost":
        """A deliberately uninformative prior.

        Useful for a first window, and for checking that a result is
        driven by the data rather than by the prior: run it twice, once
        vague, and see how much the estimate moves.
        """
        return cls.diagonal(x_bar, scale)

    @property
    def factor(self) -> Array:
        """Lower Cholesky factor ``L`` with ``P = L L^T``."""
        p = jnp.asarray(self.P, dtype=jnp.float64)
        p = 0.5 * (p + p.T)
        jitter = CHOLESKY_JITTER * jnp.trace(p) / p.shape[0]
        return jnp.linalg.cholesky(p + jitter * jnp.eye(p.shape[0]))

    def whiten(self, x: Array) -> Array:
        """``L^{-1} (x - x_bar)``, the arrival term's least-squares residual.

        Working with the residual rather than the quadratic form is what
        lets the whole moving-horizon objective be handed to a
        least-squares solver, which exploits the Gauss-Newton structure
        the problem actually has.
        """
        delta = jnp.asarray(x, dtype=jnp.float64) - self.x_bar
        return solve_triangular(self.factor, delta, lower=True)

    def cost(self, x: Array) -> Array:
        """The scalar arrival cost at ``x``."""
        r = self.whiten(x)
        return jnp.dot(r, r)

    @property
    def std(self) -> Array:
        """Marginal standard deviations, shape ``(n_x,)``."""
        return jnp.sqrt(jnp.clip(jnp.diag(self.P), 0.0, jnp.inf))

    @property
    def condition(self) -> float:
        """Condition number of ``P``.

        A large value means some direction of the state has been
        summarised as almost perfectly known. That is sometimes true and
        sometimes the over-confidence described in the module docstring;
        either way, an estimator that has stopped moving with a
        condition number of 1e12 is explained by this number.
        """
        p = np.asarray(self.P, dtype=float)
        s = np.linalg.svd(p, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")

    def inflate(self, factor: float = 2.0) -> "ArrivalCost":
        """Scale the covariance up, loosening the summary of the past."""
        if factor <= 0:
            raise ValueError("inflation factor must be positive")
        return ArrivalCost(x_bar=self.x_bar, P=factor * self.P,
                           time=self.time)

    def summary(self, names: list[str] | None = None) -> str:
        """Per-state prior mean and standard deviation."""
        x = np.asarray(self.x_bar, dtype=float)
        sd = np.asarray(self.std, dtype=float)
        names = names or [f"x{i}" for i in range(x.size)]
        lines = [
            f"arrival cost at t = {self.time:g}, cond(P) = "
            f"{self.condition:.3g}",
            "",
            f"{'state':<20} {'x_bar':>12} {'std':>12}",
            "-" * 46,
        ]
        for nm, xv, sv in zip(names, x, sd):
            lines.append(f"{nm:<20} {xv:12.5g} {sv:12.5g}")
        return "\n".join(lines)


def advance_arrival_cost(
    model: StateSpaceModel,
    arrival: ArrivalCost,
    window: MeasurementWindow,
    x_hat: Array,
    *,
    process_std: Array,
    theta: Any = None,
) -> ArrivalCost:
    """Roll the arrival cost forward one sampling interval.

    Given a solved window over ``[k-K, k]`` this returns the arrival
    cost for the next window, which starts at ``k-K+1``. The covariance
    follows the EKF recursion --- update with the oldest measurement,
    then predict one step --- linearised at the MHE trajectory, and the
    mean is the MHE's own smoothed estimate of the second state.

    Args:
        model: the plant model.
        arrival: the arrival cost the solved window used.
        window: that window.
        x_hat: the solved trajectory, shape ``(K + 1, n_x)``.
        process_std: standard deviation of ``w``, shape ``(n_w,)``.
        theta: model parameters.

    Returns:
        The :class:`ArrivalCost` for the next window.
    """
    x_hat = jnp.asarray(x_hat, dtype=jnp.float64)
    u0 = window.u[0]
    # Update at the oldest grid point, linearised at the MHE estimate of
    # that state rather than at the prior mean.
    _, p_up, _ = ekf_update(
        model, arrival.x_bar, arrival.P, window.y[0], window.sigma[0], u0,
        theta, x_lin=x_hat[0], clip=False,
    )
    _, p_next = ekf_predict(
        model, x_hat[0], p_up, u0, process_std, theta, x_lin=x_hat[0]
    )
    return ArrivalCost(
        x_bar=x_hat[1], P=p_next, time=float(window.times[1])
    )
