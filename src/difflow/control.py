"""Feedback control: PI(D) in a dynamic flowsheet, and linear MPC.

Two controllers, at the two levels a plant runs them:

* :class:`~difflow.dynamic.controller.PIDController` (re-exported here) is a
  dynamic unit: it sits in a :class:`~difflow.dynamic.DynamicFlowsheet`,
  measures a stream and drives a port, with output limits and
  back-calculation anti-windup. The closed loop is one JAX function of the
  gains, so ``jax.grad`` tunes it.
* :class:`LinearMPC` is a receding-horizon controller on a linear(ized)
  discrete-time model -- a :class:`~difflow.mhe.StateSpaceModel` linearized
  by autodiff, or matrices you give. At every sample it predicts ``horizon``
  steps ahead, chooses the input moves that minimize the predicted tracking
  error plus a move penalty *subject to input bounds*, applies the first,
  and repeats. :func:`run_mpc` runs that loop against any JAX plant.

The MPC problem is a box-constrained quadratic program in the stacked input
deviations ``U`` (``horizon * n_u`` numbers)::

    min_U  sum_i |y_i - r_i|^2_Q + sum_k |u_k - u_{k-1}|^2_R + |u_k - u_op|^2_Ru
    s.t.   u_min <= u_k <= u_max

with ``y_i = y_op + C dx_i + bias`` predicted by ``dx_{k+1} = A dx_k + B du_k
+ c0``. It is solved exactly (to round-off) by a primal-dual active-set
method (Hintermueller, Ito and Kunisch, *SIAM J. Optim.* 13 (2003) 865): a
dense solve per iteration, a handful of iterations, all in JAX. ``bias`` --
the measured output minus the model's -- is the constant output disturbance
that makes the loop offset-free on a plant the linear model does not match.

Limits, stated rather than hidden: input bounds only (no output or
move-rate constraints, which would make the QP general rather than a box);
the model is linear in deviations around one operating point (re-linearize
yourself for gain scheduling); the measurement map's dependence on ``u`` is
ignored; the state of the linear model is propagated open loop from the
applied inputs (no state estimator; feed :func:`difflow.mhe.run_ekf` output
in through ``xm`` of :meth:`LinearMPC.solve` if you have one).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.dynamic.controller import PIDController

__all__ = ["LinearMPC", "MPCResult", "PIDController", "box_qp", "run_mpc"]


def box_qp(H: Array, g: Array, lb: Array, ub: Array, max_iter: int = 50,
           tol: float = 1e-10) -> tuple[Array, Array]:
    """Minimize ``0.5 u'Hu + g'u`` subject to ``lb <= u <= ub``.

    Primal-dual active-set iterations (a free/active split, one dense solve
    each) under ``lax.fori_loop``; for a positive-definite ``H`` they reach
    the exact minimizer once the active set settles, typically in a few
    iterations. If they have not (a cycling active set, which is rare), a
    projected-gradient (FISTA) polish runs on the result. ``lb``/``ub`` may
    hold infinities.

    Args:
        H: Positive-definite Hessian, shape ``(n, n)``.
        g: Linear term, shape ``(n,)``.
        lb, ub: Bounds, shape ``(n,)``.
        max_iter: Active-set iterations.
        tol: Tolerance on the projected-gradient (KKT) residual, relative to
            ``1 + |g|``.

    Returns:
        ``(u, kkt)``: the minimizer and its projected-gradient residual
        ``max|u - clip(u - (Hu + g))|``.

    Example:
        >>> import jax.numpy as jnp
        >>> u, kkt = box_qp(jnp.eye(2), jnp.array([-2.0, 1.0]),
        ...                 jnp.array([-1.0, 0.0]), jnp.array([1.0, 1.0]))
        >>> [round(float(v), 6) for v in u]
        [1.0, 0.0]
    """
    H = jnp.asarray(H, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64)
    lb = jnp.broadcast_to(jnp.asarray(lb, dtype=jnp.float64), g.shape)
    ub = jnp.broadcast_to(jnp.asarray(ub, dtype=jnp.float64), g.shape)
    c = 1.0 / jnp.diag(H)

    def kkt(u):
        return jnp.max(jnp.abs(u - jnp.clip(u - (H @ u + g), lb, ub)), initial=0.0)

    def body(_, carry):
        u, lam = carry
        pred = u - c * lam
        lower = pred < lb
        upper = (pred > ub) & ~lower
        free = ~(lower | upper)
        fixed = jnp.where(lower, lb, jnp.where(upper, ub, 0.0))
        fixed = jnp.where(free, 0.0, fixed)
        M = jnp.where(free[:, None] & free[None, :], H, 0.0) + jnp.diag(jnp.where(free, 0.0, 1.0))
        rhs = jnp.where(free, -g - H @ fixed, fixed)
        u_new = jnp.linalg.solve(M, rhs)
        lam_new = jnp.where(free, 0.0, H @ u_new + g)
        return u_new, lam_new

    u0 = jnp.clip(jnp.linalg.solve(H, -g), lb, ub)
    u, _ = jax.lax.fori_loop(0, max_iter, body, (u0, H @ u0 + g))
    u = jnp.clip(u, lb, ub)

    def polish(u):
        L = jnp.max(jnp.linalg.eigvalsh(H))

        def step(_, carry):
            x, z, t = carry
            x_new = jnp.clip(z - (H @ z + g) / L, lb, ub)
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            return x_new, x_new + (t - 1.0) / t_new * (x_new - x), t_new

        return jax.lax.fori_loop(0, 5000, step, (u, u, jnp.asarray(1.0)))[0]

    ok = kkt(u) <= tol * (1.0 + jnp.max(jnp.abs(g), initial=0.0))
    u = jax.lax.cond(ok, lambda v: v, polish, u)
    return u, kkt(u)


def _vec(v, n: int, fill: float) -> Array:
    if v is None:
        return jnp.full((n,), fill)
    return jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), (n,))


class MPCResult(NamedTuple):
    """A closed-loop receding-horizon run.

    Attributes:
        x: Plant states ``x_0 ... x_N``, shape ``(N + 1, n_x)``.
        y: Measured outputs at those states, shape ``(N + 1, n_y)``.
        u: Inputs applied over each interval, shape ``(N, n_u)``.
        kkt: The QP's KKT residual at each sample, shape ``(N,)``.
    """

    x: Array
    y: Array
    u: Array
    kkt: Array


@dataclass
class LinearMPC:
    """Receding-horizon controller on a linear discrete-time model.

    The model is in deviations from an operating point
    ``(x_op, u_op, y_op)``: ``dx_{k+1} = A dx_k + B du_k + c0`` and
    ``y = y_op + C dx``, where ``c0 = f(x_op, u_op) - x_op`` (zero at a
    steady state). Build it from a :class:`~difflow.mhe.StateSpaceModel`
    with :meth:`from_model`, or give the matrices.

    Attributes:
        A: State matrix, shape ``(n_x, n_x)``.
        B: Input matrix, shape ``(n_x, n_u)``.
        C: Output matrix, shape ``(n_y, n_x)``.
        horizon: Prediction (and control) horizon, in samples.
        q: Output tracking weight, a scalar or one per output.
        r_move: Weight on input moves ``u_k - u_{k-1}``, scalar or per input.
        r_u: Weight on input deviations ``u_k - u_op``, scalar or per input.
        u_min: Lower input bounds (absolute; None for none).
        u_max: Upper input bounds (absolute; None for none).
        x_op: Operating-point state (default zeros).
        u_op: Operating-point input (default zeros).
        y_op: Operating-point output (default zeros).
        c0: Drift of the model at the operating point (default zeros).
        max_iter: Active-set iterations per QP.
    """

    A: Array
    B: Array
    C: Array
    horizon: int
    q: Any = 1.0
    r_move: Any = 0.0
    r_u: Any = 0.0
    u_min: Any = None
    u_max: Any = None
    x_op: Any = None
    u_op: Any = None
    y_op: Any = None
    c0: Any = None
    max_iter: int = 50

    def __post_init__(self) -> None:
        self.A = jnp.atleast_2d(jnp.asarray(self.A, dtype=jnp.float64))
        self.B = jnp.asarray(self.B, dtype=jnp.float64)
        if self.B.ndim == 1:
            self.B = self.B[:, None]
        self.C = jnp.atleast_2d(jnp.asarray(self.C, dtype=jnp.float64))
        n_x, n_u, n_y = self.A.shape[0], self.B.shape[1], self.C.shape[0]
        if self.A.shape != (n_x, n_x) or self.B.shape[0] != n_x or self.C.shape[1] != n_x:
            raise ValueError(f"inconsistent shapes: A {self.A.shape}, B {self.B.shape}, C {self.C.shape}")
        self.horizon = int(self.horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")
        self.x_op = _vec(self.x_op, n_x, 0.0)
        self.u_op = _vec(self.u_op, n_u, 0.0)
        self.y_op = _vec(self.y_op, n_y, 0.0)
        self.c0 = _vec(self.c0, n_x, 0.0)
        self.u_min = _vec(self.u_min, n_u, -jnp.inf)
        self.u_max = _vec(self.u_max, n_u, jnp.inf)
        if np.any(np.asarray(self.u_min) >= np.asarray(self.u_max)):
            raise ValueError("u_min must be below u_max")
        for name, n in (("q", n_y), ("r_move", n_u), ("r_u", n_u)):
            w = _vec(getattr(self, name), n, 0.0)
            if np.any(np.asarray(w) < 0):
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, w)

    @property
    def n_x(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_u(self) -> int:
        return int(self.B.shape[1])

    @property
    def n_y(self) -> int:
        return int(self.C.shape[0])

    @classmethod
    def from_model(cls, model, x_op, u_op, horizon: int, theta: Any = None,
                   **kwargs: Any) -> LinearMPC:
        """Linearize a :class:`~difflow.mhe.StateSpaceModel` at ``(x_op, u_op)``.

        ``A``, ``B`` and ``C`` are Jacobians of the model's step and
        measurement by autodiff (process noise at zero); ``c0`` is the drift
        ``f(x_op, u_op) - x_op``, which is zero at a steady state.

        Args:
            model: The discrete-time model (its ``dt`` is the MPC sample).
            x_op, u_op: Operating point.
            horizon: Prediction horizon, in samples.
            theta: Model parameters.
            **kwargs: Weights and bounds, as for the class.

        Returns:
            A :class:`LinearMPC`.
        """
        x_op = jnp.asarray(x_op, dtype=jnp.float64)
        u_op = jnp.atleast_1d(jnp.asarray(u_op, dtype=jnp.float64))
        w0 = jnp.zeros(model.n_w)
        A = jax.jacfwd(lambda x: model.step(x, u_op, w0, theta))(x_op)
        B = jax.jacfwd(lambda u: model.step(x_op, u, w0, theta))(u_op)
        C = jax.jacfwd(lambda x: model.observe(x, u_op, theta))(x_op)
        c0 = model.step(x_op, u_op, w0, theta) - x_op
        y_op = model.observe(x_op, u_op, theta)
        return cls(A=A, B=B, C=C, horizon=horizon, x_op=x_op, u_op=u_op,
                   y_op=y_op, c0=c0, **kwargs)

    # -- the condensed problem -------------------------------------------

    def prediction_matrices(self) -> tuple[Array, Array, Array]:
        """``(Phi, Gamma, f0)`` with stacked predicted output deviations
        ``Y = Phi dx_0 + Gamma U + f0`` over the horizon (``y_1 ... y_N``)."""
        A, B, C, N = self.A, self.B, self.C, self.horizon
        powers = [jnp.eye(self.n_x)]
        for _ in range(N):
            powers.append(A @ powers[-1])
        Phi = jnp.vstack([C @ powers[i + 1] for i in range(N)])
        zero = jnp.zeros((self.n_y, self.n_u))
        Gam = jnp.block([[C @ powers[i - j] @ B if j <= i else zero for j in range(N)]
                         for i in range(N)])
        drift, f0 = jnp.zeros(self.n_x), []
        for i in range(N):
            drift = A @ drift + self.c0
            f0.append(C @ drift)
        return Phi, Gam, jnp.concatenate(f0)

    def solve(self, xm: Array, u_prev: Array, y_ref: Array, bias: Array = 0.0) -> tuple[Array, Array]:
        """The optimal input sequence from model state ``xm``.

        Args:
            xm: Model state deviation ``dx_0`` (from ``x_op``).
            u_prev: Input applied over the last interval (absolute).
            y_ref: Output reference (absolute), shape ``(n_y,)`` held over
                the horizon or ``(horizon, n_y)``.
            bias: Output disturbance added to every prediction (measured
                output minus the model's).

        Returns:
            ``(u, kkt)``: the inputs over the horizon (absolute), shape
            ``(horizon, n_u)``, and the QP's KKT residual.
        """
        N, n_u, n_y = self.horizon, self.n_u, self.n_y
        Phi, Gam, f0 = self.prediction_matrices()
        y_ref = jnp.broadcast_to(jnp.asarray(y_ref, dtype=jnp.float64), (N, n_y)).reshape(-1)
        bias = jnp.broadcast_to(jnp.asarray(bias, dtype=jnp.float64), (n_y,))
        target = y_ref - jnp.tile(self.y_op + bias, N) - Phi @ jnp.asarray(xm) - f0
        Q = jnp.tile(self.q, N)
        Rm = jnp.tile(self.r_move, N)
        Ru = jnp.tile(self.r_u, N)
        Dm = jnp.eye(N * n_u) - jnp.eye(N * n_u, k=-n_u)
        d0 = jnp.zeros(N * n_u).at[:n_u].set(jnp.atleast_1d(u_prev) - self.u_op)
        H = Gam.T @ (Q[:, None] * Gam) + Dm.T @ (Rm[:, None] * Dm) + jnp.diag(Ru)
        g = -(Gam.T @ (Q * target) + Dm.T @ (Rm * d0))
        lb = jnp.tile(self.u_min - self.u_op, N)
        ub = jnp.tile(self.u_max - self.u_op, N)
        U, res = box_qp(H, g, lb, ub, max_iter=self.max_iter)
        return U.reshape(N, n_u) + self.u_op, res


def run_mpc(
    mpc: LinearMPC,
    plant: Callable[[Array, Array], Array] | Any,
    x0: Array,
    n_steps: int,
    y_ref: Array,
    *,
    measure: Callable[[Array], Array] | None = None,
    u0: Array | None = None,
    theta: Any = None,
) -> MPCResult:
    """Run a closed-loop receding-horizon simulation.

    At each sample: measure the plant, take ``bias`` = measured output minus
    the linear model's, solve the QP, apply the first input over one
    interval of the plant, and advance the linear model with the same input.
    The loop is a ``lax.scan``, so the plant must be JAX-traceable.

    Args:
        mpc: The controller.
        plant: ``plant(x, u) -> x_next`` over one sample, or a
            :class:`~difflow.mhe.StateSpaceModel` (stepped with zero
            process noise).
        x0: Initial plant state.
        n_steps: Number of samples.
        y_ref: Output reference: ``(n_y,)`` (or a scalar) for all samples, or
            ``(n_steps, n_y)``, each sample's value held over its horizon.
        measure: ``measure(x) -> y`` for the plant. Default: the
            ``StateSpaceModel``'s ``observe``, else ``y_op + C (x - x_op)``
            (the plant state is the model's).
        u0: Input before the first sample (default ``u_op``), for the first
            move penalty.
        theta: Parameters for a ``StateSpaceModel`` plant.

    Returns:
        :class:`MPCResult`.

    Example:
        >>> import jax.numpy as jnp
        >>> mpc = LinearMPC(A=[[0.9]], B=[[0.1]], C=[[1.0]], horizon=10,
        ...                 r_move=0.01, u_min=-1.0, u_max=1.0)
        >>> res = run_mpc(mpc, lambda x, u: 0.9 * x + 0.1 * u, jnp.zeros(1), 30, 1.0)
        >>> bool(jnp.all(jnp.abs(res.u) <= 1.0 + 1e-12))
        True
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if hasattr(plant, "step") and hasattr(plant, "observe"):
        model = plant
        w0 = jnp.zeros(model.n_w)
        step = lambda x, u: model.step(x, u, w0, theta)
        if measure is None:
            measure = lambda x: model.observe(x, mpc.u_op, theta)
    elif callable(plant):
        step = plant
    else:
        raise TypeError("plant must be a callable (x, u) -> x_next or a StateSpaceModel")
    if measure is None:
        measure = lambda x: mpc.y_op + mpc.C @ (x - mpc.x_op)
    y_ref = jnp.asarray(y_ref, dtype=jnp.float64)
    if y_ref.ndim < 2:
        y_ref = jnp.broadcast_to(y_ref, (n_steps, mpc.n_y))
    if y_ref.shape != (n_steps, mpc.n_y):
        raise ValueError(f"y_ref must be (n_y,) or (n_steps, n_y) = ({n_steps}, {mpc.n_y}), got {y_ref.shape}")
    u0 = mpc.u_op if u0 is None else jnp.atleast_1d(jnp.asarray(u0, dtype=jnp.float64))
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    def body(carry, r):
        x, xm, u_prev = carry
        bias = jnp.atleast_1d(measure(x)) - (mpc.y_op + mpc.C @ xm)
        U, kkt = mpc.solve(xm, u_prev, r, bias)
        u = U[0]
        x_next = step(x, u)
        xm_next = mpc.A @ xm + mpc.B @ (u - mpc.u_op) + mpc.c0
        return (x_next, xm_next, u), (x_next, u, kkt)

    xm0 = x0[: mpc.n_x] - mpc.x_op if x0.shape == mpc.x_op.shape else jnp.zeros(mpc.n_x)
    _, (xs, us, kkts) = jax.lax.scan(body, (x0, xm0, u0), y_ref)
    xs = jnp.concatenate([x0[None], xs])
    ys = jax.vmap(lambda x: jnp.atleast_1d(measure(x)))(xs)
    return MPCResult(x=xs, y=ys, u=us, kkt=kkts)
