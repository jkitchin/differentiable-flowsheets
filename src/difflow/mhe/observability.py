"""Can this window determine this state?

The question a moving-horizon estimator has to answer before it runs is
the same one :mod:`difflow.reconciliation.structure` asks of a
steady-state problem: is the unknown determined by the data, or is the
solver about to return whatever the prior said and call it an estimate?
The answer here is the rank of the *window observability matrix*

.. math::

    O = \\frac{\\partial}{\\partial x_0}
        \\begin{bmatrix} R^{-1/2}(y_k - h(x_k)) \\end{bmatrix}_{k=0}^{K},

taken along the trajectory, with the process noise held at zero. If
:math:`O` has full column rank the window pins every state --- including
any augmented parameter, which is exactly the question "is this
degradation visible from what I measure?". If it does not, the estimate
of the deficient directions comes entirely from the arrival cost, and
tightening the sensors will not help: those directions need a different
sensor, a longer horizon, or a moving input.

Two conventions are taken from reconciliation rather than reinvented, so
the two modules cannot disagree about what "rank deficient" means. Ranks
come from an SVD of the scaled matrix, never from the eigenvalues of
:math:`O^T O` --- squaring the matrix squares its condition number and
an unobservable system comes back looking full rank --- and the columns
are equilibrated first, through the same
:class:`difflow.reconciliation.Scaling` object the reconciliation solve
carries, so the verdict does not depend on whether pressures are in Pa
or bar.

Like its steady-state counterpart this is discrete arithmetic, so it
works in NumPy and does not pretend to be differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.reconciliation import Scaling, measured_mask
# The same rank convention as steady-state reconciliation, deliberately
# shared rather than reimplemented: one definition of "rank deficient".
from difflow.reconciliation.structure import _rank_and_spectrum

from difflow.mhe.measurements import MeasurementWindow
from difflow.mhe.model import StateSpaceModel


@dataclass
class ObservabilityReport(ParamsMixin):
    """What a window can and cannot determine.

    Attributes:
        rank: numerical rank of the scaled observability matrix.
        n_states: how many states there are, i.e. the rank needed.
        observable: ``rank == n_states``.
        singular_values: the full spectrum, so a marginal case can be
            inspected rather than guessed at.
        rank_tol: the threshold used.
        rank_gap: smallest retained over largest discarded singular
            value; small means the verdict is tolerance-dependent.
        unobservable: names of the states implicated in a null-space
            direction. A direction implicates a *set* of states, not
            one, and they are all listed.
        gramian_std: ``1 / s_i`` for each direction, in scaled units ---
            the standard deviation the measurements alone leave, before
            the arrival cost is applied. Large entries name the
            directions the prior is carrying.
        names: state names.
    """

    rank: int
    n_states: int
    observable: bool
    singular_values: np.ndarray
    rank_tol: float
    rank_gap: float
    unobservable: list[str] = field(default_factory=list)
    gramian_std: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    names: list[str] = field(default_factory=list)

    def raise_if_unobservable(self) -> None:
        """Raise ``ValueError`` naming the states that cannot be seen."""
        if self.observable:
            return
        detail = (f" Implicated: {', '.join(self.unobservable)}."
                  if self.unobservable else "")
        raise ValueError(
            f"the estimation window determines only {self.rank} of "
            f"{self.n_states} states.{detail} The rest are whatever the "
            "arrival cost says. Lengthen the horizon, add a sensor, or "
            "drop the state from the model."
        )

    def summary(self) -> str:
        """Table of directions, worst conditioned first."""
        lines = [
            f"rank {self.rank} of {self.n_states} "
            f"(tol {self.rank_tol:.3g}, gap {self.rank_gap:.3g}), "
            f"observable = {self.observable}",
            "",
            f"{'direction':>10} {'singular value':>16} {'std (scaled)':>14}",
            "-" * 44,
        ]
        for i, s in enumerate(self.singular_values):
            sd = (float(self.gramian_std[i])
                  if i < len(self.gramian_std) else float("inf"))
            lines.append(f"{i:10d} {float(s):16.6g} {sd:14.6g}")
        if self.unobservable:
            lines += ["", "unobservable: " + ", ".join(self.unobservable)]
        return "\n".join(lines)


def check_observability(
    model: StateSpaceModel,
    window: MeasurementWindow,
    x0: Array,
    *,
    theta: Any = None,
    scale: Array | Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    rank_tol: float | None = None,
) -> ObservabilityReport:
    """Rank of the window observability matrix at a trajectory.

    Args:
        model: the plant model.
        window: the measurements. Only ``sigma`` matters --- which
            channels were sampled, and how precisely --- not ``y``.
        x0: the state to linearise about, shape ``(n_x,)``. Usually the
            current best estimate; observability of a nonlinear model is
            a local property and the answer can differ elsewhere.
        theta: fixed model parameters.
        scale: column scales, shape ``(n_x,)``, defaulting to
            ``max(|x0|, 1)``. Pass the prior standard deviations when
            you have them --- the question "can this be resolved?" is
            only meaningful relative to a scale.
        names: state names, defaulting to the model's.
        rank_tol: override the singular-value threshold.

    Returns:
        An :class:`ObservabilityReport`. It does not raise; call
        :meth:`ObservabilityReport.raise_if_unobservable` for that.

    Example:
        >>> rep = check_observability(model, window, x0)  # doctest: +SKIP
        >>> rep.raise_if_unobservable()               # doctest: +SKIP
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    n_x = model.n_x
    names = list(names) if names is not None else list(model.x_names)
    if len(names) != n_x:
        raise ValueError(f"got {len(names)} names for {n_x} states")

    k = window.horizon
    w0 = jnp.zeros((k, int(model.n_w)), dtype=jnp.float64)
    u_obs = jnp.concatenate([window.u, window.u[-1:]], axis=0)
    mask = measured_mask(window.sigma)
    safe_sigma = jnp.where(mask, window.sigma, 1.0)

    def predicted(xx):
        xs = model.rollout(xx, window.u, w0, theta)
        h = jax.vmap(lambda xk, uk: model.observe(xk, uk, theta))(xs, u_obs)
        return jnp.where(mask, h / safe_sigma, 0.0).reshape(-1)

    o = np.asarray(jax.jacobian(predicted)(x0), dtype=float)

    if scale is None:
        d = np.maximum(np.abs(np.asarray(x0, dtype=float)), 1.0)
    else:
        d = np.broadcast_to(np.asarray(scale, dtype=float), (n_x,)).copy()
        d[d <= 0] = 1.0
    # The residual rows are already whitened by 1/sigma above, so only
    # the columns are scaled -- r is the identity. Carried as a Scaling
    # so the object, and not merely the idea, is the one reconciliation
    # uses.
    scaling = Scaling(d=jnp.asarray(d, dtype=jnp.float64),
                      r=jnp.ones(o.shape[0], dtype=jnp.float64))
    o_s = np.asarray(scaling.r, dtype=float)[:, None] * o * d[None, :]

    rank, sv, tol, gap = _rank_and_spectrum(o_s, rank_tol)
    gramian_std = np.where(sv > 0, 1.0 / np.where(sv > 0, sv, 1.0), np.inf)

    unobservable: list[str] = []
    if rank < n_x:
        _, _, vh = np.linalg.svd(o_s, full_matrices=True)
        implicated: set[str] = set()
        for i in range(rank, n_x):
            vec = np.abs(vh[i])
            if vec.max() <= 0:
                continue
            for j in np.where(vec > 0.1 * vec.max())[0]:
                implicated.add(names[j])
        unobservable = [nm for nm in names if nm in implicated]

    return ObservabilityReport(
        rank=int(rank),
        n_states=int(n_x),
        observable=bool(rank >= n_x),
        singular_values=sv,
        rank_tol=float(tol),
        rank_gap=float(gap),
        unobservable=unobservable,
        gramian_std=gramian_std,
        names=names,
    )
