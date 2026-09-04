"""Constrained moving-horizon estimation.

At each sampling time the estimator solves

.. math::

    \\min_{x_0, w} \\; \\|x_0 - \\bar x\\|^2_{P^{-1}}
        + \\sum_{k} \\|w_k\\|^2_{Q^{-1}}
        + \\sum_{k} \\|y_k - h(x_k)\\|^2_{R^{-1}}
    \\quad \\text{s.t.} \\quad x_{k+1} = f(x_k, u_k, w_k),
    \\; x_k \\in [\\ell, u],

over the last ``K`` intervals, with the arrival cost of
:mod:`difflow.mhe.arrival` standing in for everything before them.
Reference: Rao, Rawlings and Mayne, *IEEE Trans. Automat. Contr.* 48
(2003) 246, doi:10.1109/TAC.2002.808470.

Three implementation choices are worth stating, because each is the
reason something later works.

**The dynamics are eliminated, not imposed.** The decision variables are
:math:`x_0` and the noise sequence; the states follow from a
``lax.scan``. That makes the problem an unconstrained (in the equality
sense) *least-squares* problem whose residual vector is the whitened
concatenation of the three terms above, so it goes to
``optimistix.LevenbergMarquardt`` and inherits Gauss-Newton convergence
and implicit differentiation --- ``jax.grad`` through the estimate
works, which is what lets an estimate be a link in a larger
differentiable chain rather than the end of one.

**Bounds on the state are handled two ways, for two reasons.** The
initial state is *reparameterised* through a smooth bijection, so
:math:`x_0` cannot leave its bounds at all, at any iterate. Later states
are the image of the dynamics and cannot be reparameterised, so they
carry a penalty residual; :attr:`MHEResult.max_violation` reports what
is left. An unconstrained fit of a nearly-unobservable concentration
will happily return a negative number, and reporting a negative
concentration is worse than reporting a slightly biased one.

**Every term is whitened before it is summed.** A residual is in units
of its own standard deviation, so the objective is dimensionless, is
comparable across problems, and is a :math:`\\chi^2` statistic ---
:func:`mhe_global_test` reads it as one, returning the same
:class:`~difflow.reconciliation.GlobalTestResult` that steady-state
reconciliation produces, so a dynamic and a steady-state consistency
check can be compared without translation.

Example:
    >>> problem = MHEProblem(model, arrival,          # doctest: +SKIP
    ...                      process_std=jnp.array([0.01]))
    >>> res = solve_mhe(problem, window)          # doctest: +SKIP
    >>> res.x_final, res.parameters               # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array
from scipy import stats

from difflow.params_mixin import ParamsMixin
from difflow.reconciliation import GlobalTestResult, Scaling, measured_mask

from difflow.mhe.arrival import ArrivalCost, advance_arrival_cost
from difflow.mhe.ekf import EKFRunResult, run_ekf
from difflow.mhe.measurements import MeasurementWindow, slice_window
from difflow.mhe.model import StateSpaceModel

#: Default weight on the state-bound penalty, relative to one sigma.
CONSTRAINT_WEIGHT = 1.0e3
#: Floor on a decision-variable scale, so a zero prior std cannot divide.
SCALE_FLOOR = 1.0e-12


@dataclass
class MHEProblem(ParamsMixin):
    """Everything fixed about one moving-horizon estimation.

    Attributes:
        model: the plant model.
        arrival: the arrival cost at the first grid point of the window.
        process_std: standard deviation of ``w``, shape ``(n_w,)``. This
            is the tuning knob: it says how much of a mismatch between
            model and data is charged to the process rather than to the
            sensors, and on an augmented parameter it sets how fast that
            parameter is allowed to drift.
        theta: fixed model parameters, threaded to ``f`` and ``h``.
        constraint_weight: weight on the bound-violation penalty applied
            to states after the first. Raise it if
            :attr:`MHEResult.max_violation` is not small enough; the
            first state is bounded exactly and needs no penalty.
        rtol, atol: Levenberg-Marquardt tolerances.
        max_steps: iteration cap. The solve does not raise on failure;
            :attr:`MHEResult.success` reports it, so a sliding run
            continues rather than stopping on one bad window.
    """

    model: StateSpaceModel
    arrival: ArrivalCost
    process_std: Array
    theta: Any = None
    constraint_weight: float = CONSTRAINT_WEIGHT
    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 256

    def __post_init__(self) -> None:
        self.process_std = jnp.broadcast_to(
            jnp.asarray(self.process_std, dtype=jnp.float64),
            (int(self.model.n_w),),
        )
        if not bool(jnp.all(self.process_std > 0)):
            raise ValueError(
                "process_std must be strictly positive; a state believed "
                "exact still needs a small value, since 1/Q appears in the "
                "objective"
            )

    def scaling(self, horizon: int) -> Scaling:
        """Decision-variable scaling for a window of ``horizon`` intervals.

        Reuses :class:`difflow.reconciliation.Scaling`, but sets only its
        variable scales ``d``: the residual rows are already whitened by
        the covariances, so equilibrating them --- which is right for
        the reconciliation KKT system --- would destroy exactly the
        statistical weighting that makes the objective a
        :math:`\\chi^2`. The scales are the prior standard deviations,
        so a scaled step of 1.0 means "one sigma" in both blocks.
        """
        n_x, n_w = self.model.n_x, int(self.model.n_w)
        d = jnp.concatenate([
            jnp.clip(self.arrival.std, SCALE_FLOOR, jnp.inf),
            jnp.tile(self.process_std, horizon),
        ])
        return Scaling(d=d, r=jnp.ones(n_x + horizon * n_w))


@dataclass
class MHEResult(ParamsMixin):
    """Outcome of one window solve.

    Attributes:
        times: grid times, shape ``(K + 1,)``.
        x: estimated trajectory, shape ``(K + 1, n_x)``.
        w: estimated process noise, shape ``(K, n_w)``.
        x_final: the estimate of the current state, ``x[-1]``.
        objective: the whitened sum of squares; a :math:`\\chi^2`
            statistic on :attr:`n_measurements` degrees of freedom.
        arrival_objective, process_objective, measurement_objective: its
            three parts, which say *where* a large objective came from.
        residual: the whitened residual vector at the solution.
        covariance: covariance of ``x_final``, shape ``(n_x, n_x)``,
            from the Gauss-Newton Hessian of the statistical terms.
        n_measurements: scalar readings in the window.
        max_violation: the largest state-bound violation left by the
            penalty. Not zero by construction; check it.
        success: whether Levenberg-Marquardt converged.
        n_steps: iterations it took.
        x_names, y_names, param_names: names for reporting.
    """

    times: Array
    x: Array
    w: Array
    x_final: Array
    objective: float
    arrival_objective: float
    process_objective: float
    measurement_objective: float
    residual: Array
    covariance: Array
    n_measurements: int
    max_violation: float
    success: bool
    n_steps: int
    x_names: list[str] = field(default_factory=list)
    y_names: list[str] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)

    @property
    def std(self) -> Array:
        """Standard deviations of ``x_final``, shape ``(n_x,)``."""
        return jnp.sqrt(jnp.clip(jnp.diag(self.covariance), 0.0, jnp.inf))

    @property
    def x_named(self) -> dict[str, float]:
        """``x_final`` as ``{name: value}``."""
        return {nm: float(self.x_final[i])
                for i, nm in enumerate(self.x_names)}

    @property
    def parameters(self) -> dict[str, float]:
        """Estimated augmented parameters as ``{name: value}``.

        This is the mapping shape :attr:`difflow.planning.Block.theta`
        takes, and the shape
        :func:`difflow.planning.update_modifiers` accepts as its
        ``theta`` override, so the current estimate goes into a
        real-time optimisation layer without an adapter. That loop is
        the point of estimating parameters at all: an optimiser acting
        on a model whose parameters drifted last week is optimising the
        wrong plant.
        """
        if not self.param_names:
            return {}
        offset = len(self.x_names) - len(self.param_names)
        return {nm: float(self.x_final[offset + i])
                for i, nm in enumerate(self.param_names)}

    @property
    def parameter_std(self) -> dict[str, float]:
        """Standard errors of the estimated parameters."""
        if not self.param_names:
            return {}
        offset = len(self.x_names) - len(self.param_names)
        sd = np.asarray(self.std, dtype=float)
        return {nm: float(sd[offset + i])
                for i, nm in enumerate(self.param_names)}

    def summary(self) -> str:
        """Current state, its standard error, and where the cost sits."""
        x = np.asarray(self.x_final, dtype=float)
        sd = np.asarray(self.std, dtype=float)
        names = self.x_names or [f"x{i}" for i in range(x.size)]
        lines = [
            f"MHE over [{float(self.times[0]):g}, {float(self.times[-1]):g}] "
            f"({int(self.times.shape[0]) - 1} intervals, "
            f"{self.n_measurements} readings)",
            f"objective {self.objective:.4g} = arrival "
            f"{self.arrival_objective:.4g} + process "
            f"{self.process_objective:.4g} + measurement "
            f"{self.measurement_objective:.4g}",
            f"converged = {self.success} in {self.n_steps} steps, "
            f"max bound violation {self.max_violation:.3g}",
            "",
            f"{'state':<20} {'estimate':>14} {'std':>12}",
            "-" * 48,
        ]
        for nm, xv, sv in zip(names, x, sd):
            lines.append(f"{nm:<20} {xv:14.6g} {sv:12.4g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# bound reparameterisation
# ---------------------------------------------------------------------

def _to_state(z: Array, lb: Array, ub: Array) -> Array:
    """Map an unconstrained vector into ``[lb, ub]``, smoothly.

    A sigmoid on a two-sided bound, a softplus on a one-sided one, and
    the identity where there is no bound. Every branch is evaluated on
    *sanitised* limits, so an infinite bound never enters the arithmetic
    of a branch that is not taken --- which would otherwise put a NaN
    into the gradient even though the value is right.
    """
    has_l = jnp.isfinite(lb)
    has_u = jnp.isfinite(ub)
    both = has_l & has_u
    lo_only = has_l & ~has_u
    up_only = has_u & ~has_l

    lb_s = jnp.where(has_l, lb, 0.0)
    ub_s = jnp.where(has_u, ub, 0.0)
    span = jnp.where(both, ub_s - lb_s, 1.0)

    x_both = lb_s + span * jax.nn.sigmoid(z)
    soft = jax.nn.softplus(z)
    x_lo = lb_s + soft
    x_up = ub_s - soft
    return jnp.where(
        both, x_both, jnp.where(lo_only, x_lo, jnp.where(up_only, x_up, z))
    )


def _to_free(x: Array, lb: Array, ub: Array) -> Array:
    """Inverse of :func:`_to_state`, used only for the initial guess."""
    eps = 1e-9
    has_l = jnp.isfinite(lb)
    has_u = jnp.isfinite(ub)
    both = has_l & has_u
    lo_only = has_l & ~has_u
    up_only = has_u & ~has_l

    lb_s = jnp.where(has_l, lb, 0.0)
    ub_s = jnp.where(has_u, ub, 0.0)
    span = jnp.where(both, ub_s - lb_s, 1.0)
    p = jnp.clip((x - lb_s) / span, eps, 1.0 - eps)
    z_both = jnp.log(p / (1.0 - p))

    d_lo = jnp.clip(x - lb_s, eps, jnp.inf)
    d_up = jnp.clip(ub_s - x, eps, jnp.inf)
    inv_soft = lambda d: d + jnp.log(-jnp.expm1(-d))  # noqa: E731
    return jnp.where(
        both, z_both,
        jnp.where(lo_only, inv_soft(d_lo),
                  jnp.where(up_only, inv_soft(d_up), x)),
    )


def _violation(x: Array, lb: Array, ub: Array) -> Array:
    """Non-negative bound violation, safe with infinite limits."""
    lo = jnp.where(jnp.isfinite(lb),
                   jnp.maximum(jnp.where(jnp.isfinite(lb), lb, 0.0) - x, 0.0),
                   0.0)
    hi = jnp.where(jnp.isfinite(ub),
                   jnp.maximum(x - jnp.where(jnp.isfinite(ub), ub, 0.0), 0.0),
                   0.0)
    return lo + hi


# ---------------------------------------------------------------------
# the least-squares residual
# ---------------------------------------------------------------------

def _make_parts(problem: MHEProblem, horizon: int) -> Callable:
    """Build ``parts(z, x_bar, chol, y, sigma, u, d) -> (xs, w, blocks)``.

    Returned as a closure over the *static* parts of the problem (the
    model, the horizon, the flags) so that a sliding run can ``jit`` it
    once and call it with fresh arrays at every sampling time, rather
    than retracing per window.
    """
    model = problem.model
    theta = problem.theta
    n_x, n_w = model.n_x, int(model.n_w)
    lb, ub = model.bounds
    bounded = model.is_bounded
    weight = float(problem.constraint_weight)
    process_std = problem.process_std

    def parts(z, x_bar, chol, y, sigma, u, d):
        v = d * z
        x0 = _to_state(v[:n_x], lb, ub) if bounded else v[:n_x]
        w = v[n_x:].reshape(horizon, n_w)
        xs = model.rollout(x0, u, w, theta)

        r_arrival = jax.scipy.linalg.solve_triangular(
            chol, x0 - x_bar, lower=True
        )
        r_process = (w / process_std[None, :]).reshape(-1)

        u_obs = jnp.concatenate([u, u[-1:]], axis=0)
        h = jax.vmap(lambda xk, uk: model.observe(xk, uk, theta))(xs, u_obs)
        mask = measured_mask(sigma)
        safe_sigma = jnp.where(mask, sigma, 1.0)
        r_meas = jnp.where(mask, (y - h) / safe_sigma, 0.0).reshape(-1)

        if bounded:
            scale_x = jnp.clip(d[:n_x], SCALE_FLOOR, jnp.inf)
            r_pen = (weight * _violation(xs, lb, ub)
                     / scale_x[None, :]).reshape(-1)
        else:
            r_pen = jnp.zeros((0,), dtype=jnp.float64)
        return xs, w, (r_arrival, r_process, r_meas, r_pen)

    return parts


def _make_core(problem: MHEProblem, horizon: int) -> Callable:
    """Build the array-in / array-out window solve.

    Signature ``core(x_bar, chol, y, sigma, u, d, z0) -> (xs, w, blocks,
    covariance, success, n_steps)``. Everything it needs that is not an
    array is captured, which is what makes it safe to ``jit`` --- and a
    sliding run traces it once and calls it at every sampling time,
    because only the arrays change from window to window.
    """
    parts = _make_parts(problem, horizon)
    solver = optx.LevenbergMarquardt(rtol=problem.rtol, atol=problem.atol)
    max_steps = int(problem.max_steps)

    def core(x_bar, chol, y, sigma, u, d, z0):
        def fn(z, args):
            _, _, blocks = parts(z, x_bar, chol, y, sigma, u, d)
            return jnp.concatenate(blocks)

        sol = optx.least_squares(
            fn, solver, z0, max_steps=max_steps, throw=False
        )
        z = sol.value
        xs, w, blocks = parts(z, x_bar, chol, y, sigma, u, d)
        success = sol.result == optx.RESULTS.successful

        # Linearised covariance of the last state: the Gauss-Newton
        # Hessian of the *statistical* residual --- the bound penalty is
        # a numerical device, not information, so it is left out ---
        # inverted and pushed through the map to x_K.
        def stat_residual(zz):
            _, _, bl = parts(zz, x_bar, chol, y, sigma, u, d)
            return jnp.concatenate(bl[:3])

        def last_state(zz):
            xx, _, _ = parts(zz, x_bar, chol, y, sigma, u, d)
            return xx[-1]

        j = jax.jacobian(stat_residual)(z)
        g = jax.jacobian(last_state)(z)
        covariance = g @ jnp.linalg.pinv(j.T @ j) @ g.T
        return xs, w, blocks, covariance, success, sol.stats["num_steps"]

    return core


def _initial_z(
    problem: MHEProblem,
    horizon: int,
    d: Array,
    x_guess: Array | None,
    w_guess: Array | None,
) -> Array:
    """Pack an initial guess into the scaled, unconstrained variables."""
    model = problem.model
    n_w = int(model.n_w)
    lb, ub = model.bounds
    x_g = (problem.arrival.x_bar if x_guess is None
           else jnp.asarray(x_guess, dtype=jnp.float64))
    if model.is_bounded:
        # Start strictly inside, or the inverse map is at an asymptote.
        span = jnp.where(jnp.isfinite(ub) & jnp.isfinite(lb),
                         jnp.where(jnp.isfinite(ub), ub, 0.0)
                         - jnp.where(jnp.isfinite(lb), lb, 0.0),
                         1.0)
        pad = 1e-6 * jnp.maximum(span, 1.0)
        x_g = jnp.clip(x_g, jnp.where(jnp.isfinite(lb), lb + pad, -jnp.inf),
                       jnp.where(jnp.isfinite(ub), ub - pad, jnp.inf))
        v_x = _to_free(x_g, lb, ub)
    else:
        v_x = x_g
    w_g = (jnp.zeros((horizon, n_w), dtype=jnp.float64) if w_guess is None
           else jnp.asarray(w_guess, dtype=jnp.float64).reshape(horizon, n_w))
    return jnp.concatenate([v_x, w_g.reshape(-1)]) / d


# ---------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------

def estimate(
    problem: MHEProblem,
    window: MeasurementWindow,
    *,
    x_guess: Array | None = None,
    w_guess: Array | None = None,
) -> Array:
    """The estimated trajectory, shape ``(K + 1, n_x)``.

    The pure array-valued form of :func:`solve_mhe`: no diagnostics, no
    Python floats, so it composes under ``jit``, ``vmap`` and ``grad``.
    Differentiation goes through Levenberg-Marquardt by implicit
    differentiation of the optimality conditions, not by unrolling, so
    its cost does not grow with the iteration count.

    Args:
        problem: the estimation problem.
        window: the measurements.
        x_guess: initial guess for ``x_0``; defaults to the arrival mean.
        w_guess: initial guess for the noise sequence; defaults to zero.

    Returns:
        The estimated states ``x_0 ... x_K``.

    Example:
        >>> traj = jax.jit(lambda yy: estimate(       # doctest: +SKIP
        ...     problem, replace(window, y=yy)))(window.y)
    """
    k = window.horizon
    sc = problem.scaling(k)
    core = _make_core(problem, k)
    z0 = _initial_z(problem, k, sc.d, x_guess, w_guess)
    xs, _, _, _, _, _ = core(
        problem.arrival.x_bar, problem.arrival.factor, window.y,
        window.sigma, window.u, sc.d, z0,
    )
    return xs


def solve_mhe(
    problem: MHEProblem,
    window: MeasurementWindow,
    *,
    x_guess: Array | None = None,
    w_guess: Array | None = None,
    _core: Callable | None = None,
) -> MHEResult:
    """Solve one moving-horizon estimation problem.

    Args:
        problem: the estimation problem, holding the model, the arrival
            cost and the noise levels.
        window: the measurements over the horizon, from
            :func:`~difflow.mhe.build_window`.
        x_guess: initial guess for ``x_0``; defaults to the arrival mean.
        w_guess: initial guess for the noise sequence; defaults to zero.
        _core: a pre-built solve closure, used by :func:`run_mhe` to
            avoid re-tracing at every sampling time.

    Returns:
        An :class:`MHEResult`.

    Example:
        >>> res = solve_mhe(problem, window)      # doctest: +SKIP
        >>> print(res.summary())              # doctest: +SKIP
    """
    k = window.horizon
    sc = problem.scaling(k)
    core = _make_core(problem, k) if _core is None else _core
    z0 = _initial_z(problem, k, sc.d, x_guess, w_guess)
    xs, w, blocks, covariance, success, n_steps = core(
        problem.arrival.x_bar, problem.arrival.factor, window.y,
        window.sigma, window.u, sc.d, z0,
    )

    r_arrival, r_process, r_meas, _ = blocks
    lb, ub = problem.model.bounds
    residual = jnp.concatenate([r_arrival, r_process, r_meas])

    return MHEResult(
        times=window.times,
        x=xs,
        w=w,
        x_final=xs[-1],
        objective=float(jnp.sum(residual ** 2)),
        arrival_objective=float(jnp.sum(r_arrival ** 2)),
        process_objective=float(jnp.sum(r_process ** 2)),
        measurement_objective=float(jnp.sum(r_meas ** 2)),
        residual=residual,
        covariance=covariance,
        n_measurements=window.n_measurements,
        max_violation=float(jnp.max(_violation(xs, lb, ub)))
        if problem.model.is_bounded else 0.0,
        success=bool(success),
        n_steps=int(n_steps),
        x_names=list(problem.model.x_names),
        y_names=list(problem.model.y_names),
        param_names=list(problem.model.param_names),
    )


@dataclass
class MHERunResult(ParamsMixin):
    """A sliding moving-horizon run over a whole record.

    Attributes:
        times: grid times, shape ``(N + 1,)``.
        x: the estimate of ``x_j`` made at time ``j``, shape
            ``(N + 1, n_x)``. Before the first full window this is the
            extended Kalman filter's estimate; see :attr:`source`.
        source: ``"ekf"`` or ``"mhe"`` per grid point, so a plot never
            silently mixes the two.
        windows: the :class:`MHEResult` of every window solved.
        ekf: the filter run over the same record, kept as the baseline
            the horizon has to beat.
        x_names, param_names: names for reporting.
    """

    times: Array
    x: Array
    source: list[str]
    windows: list[MHEResult]
    ekf: EKFRunResult
    x_names: list[str] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)

    @property
    def x_final(self) -> Array:
        """The last estimate."""
        return self.x[-1]

    @property
    def parameters(self) -> dict[str, Array]:
        """Trajectory of each augmented parameter, ``{name: (N+1,)}``."""
        if not self.param_names:
            return {}
        offset = len(self.x_names) - len(self.param_names)
        return {nm: self.x[:, offset + i]
                for i, nm in enumerate(self.param_names)}

    @property
    def converged(self) -> bool:
        """Whether every window solve converged."""
        return all(w.success for w in self.windows)

    def summary(self) -> str:
        """The run, window by window."""
        lines = [
            f"{len(self.windows)} moving-horizon solves over "
            f"[{float(self.times[0]):g}, {float(self.times[-1]):g}], "
            f"all converged = {self.converged}",
            "",
            f"{'time':>10} {'objective':>12} {'steps':>7} {'viol':>10}",
            "-" * 44,
        ]
        for w in self.windows:
            lines.append(
                f"{float(w.times[-1]):10g} {w.objective:12.4g} "
                f"{w.n_steps:7d} {w.max_violation:10.2g}"
            )
        return "\n".join(lines)


def run_mhe(
    model: StateSpaceModel,
    window: MeasurementWindow,
    *,
    horizon: int,
    process_std: Array,
    x0: Array,
    P0: Array,
    theta: Any = None,
    constraint_weight: float = CONSTRAINT_WEIGHT,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 256,
    warm_start: bool = True,
) -> MHERunResult:
    """Slide a fixed horizon over a whole record.

    The first ``horizon`` grid points have no full window behind them,
    so they are filtered rather than optimised; from then on each
    sampling time solves one window and the arrival cost is rolled
    forward by :func:`~difflow.mhe.advance_arrival_cost`. The filter is
    run over the whole record anyway and returned, because "MHE was
    better" is only a claim if the alternative was computed.

    The window solve is traced once and reused, so the cost of a run is
    ``N`` optimisations and one compilation, not ``N`` of each.

    Args:
        model: the plant model.
        window: the full record, from
            :func:`~difflow.mhe.build_window`.
        horizon: intervals per window.
        process_std: standard deviation of ``w``, shape ``(n_w,)``.
        x0: prior mean at the first grid point.
        P0: prior covariance there.
        theta: fixed model parameters.
        constraint_weight: weight on the state-bound penalty.
        rtol, atol, max_steps: Levenberg-Marquardt settings.
        warm_start: seed each window from the previous solution, shifted
            one step. Off, every window starts from its arrival mean.

    Returns:
        An :class:`MHERunResult`.

    Example:
        >>> run = run_mhe(model, record, horizon=10,   # doctest: +SKIP
        ...               process_std=q, x0=x0, P0=P0)
        >>> run.parameters["efficiency"][-1]           # doctest: +SKIP
    """
    n_points = int(window.times.shape[0])
    if not 1 <= horizon <= n_points - 1:
        raise ValueError(
            f"horizon {horizon} does not fit a record of {n_points} points"
        )
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    p0 = jnp.asarray(P0, dtype=jnp.float64)

    ekf = run_ekf(model, window, x0=x0, P0=p0,
                  process_std=process_std, theta=theta)

    arrival = ArrivalCost(x_bar=x0, P=p0, time=float(window.times[0]))
    problem = MHEProblem(
        model=model, arrival=arrival, process_std=process_std, theta=theta,
        constraint_weight=constraint_weight, rtol=rtol, atol=atol,
        max_steps=max_steps,
    )
    # One trace for the whole run: the scale is an argument, not a
    # constant, so a changing arrival covariance cannot force a retrace.
    core = jax.jit(_make_core(problem, horizon))

    estimates = [np.asarray(ekf.x[j], dtype=float) for j in range(horizon)]
    source = ["ekf"] * horizon
    results: list[MHEResult] = []
    x_guess: Array | None = None
    w_guess: Array | None = None

    for start in range(0, n_points - horizon):
        sub = slice_window(window, start, horizon)
        problem = replace(problem, arrival=arrival)
        res = solve_mhe(problem, sub, x_guess=x_guess, w_guess=w_guess,
                        _core=core)
        results.append(res)
        estimates.append(np.asarray(res.x_final, dtype=float))
        source.append("mhe")
        if start + horizon < n_points - 1:
            arrival = advance_arrival_cost(
                model, arrival, sub, res.x,
                process_std=problem.process_std, theta=theta,
            )
            if warm_start:
                x_guess = res.x[1]
                w_guess = jnp.concatenate(
                    [res.w[1:], jnp.zeros((1, int(model.n_w)))], axis=0
                )

    return MHERunResult(
        times=window.times,
        x=jnp.asarray(np.stack(estimates), dtype=jnp.float64),
        source=source,
        windows=results,
        ekf=ekf,
        x_names=list(model.x_names),
        param_names=list(model.param_names),
    )


def mhe_global_test(
    result: MHEResult, alpha: float = 0.05
) -> GlobalTestResult:
    """Chi-squared consistency test on one window.

    Because every term is whitened, the optimal objective is a sum of
    squared standard normals minus the freedom used to fit them: with
    ``M`` scalar readings and the arrival and process terms exactly
    balancing the decision variables, the degrees of freedom are ``M``.
    A rejection says the window's data and the model disagree by more
    than the stated noise --- a failed sensor, a disturbance the process
    noise does not cover, or a parameter that has drifted out from under
    a fixed model.

    Returns the same
    :class:`~difflow.reconciliation.GlobalTestResult` that steady-state
    reconciliation produces, so the two can be read side by side.

    Args:
        result: a finished window solve.
        alpha: significance level.

    Returns:
        A :class:`~difflow.reconciliation.GlobalTestResult`.
    """
    dof = int(result.n_measurements)
    stat = float(result.objective)
    if dof <= 0:
        return GlobalTestResult(statistic=stat, dof=0, critical=float("inf"),
                                p_value=1.0, detected=False, alpha=alpha)
    critical = float(stats.chi2.ppf(1.0 - alpha, dof))
    return GlobalTestResult(
        statistic=stat, dof=dof, critical=critical,
        p_value=float(stats.chi2.sf(stat, dof)),
        detected=bool(stat > critical), alpha=alpha,
    )
