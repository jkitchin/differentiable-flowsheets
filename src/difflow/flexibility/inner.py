"""The inner problem: ``min over u, max over j, f_j``.

Everything in this package rests on one primitive -- given a design ``d`` and
a revealed parameter ``theta``, how close to feasible can the controls be
driven?  That is a box-constrained minimax, and it is solved here with a
smoothed maximum and projected Adam, entirely in ``jnp`` so that the whole
stack stays ``jit``-able, ``vmap``-able and differentiable.

Two decisions are worth stating because they are what make the derivatives
usable rather than merely defined.

**The maximum is smoothed only for the search, never for the answer.**  The
descent direction comes from a log-sum-exp with a temperature annealed toward
zero; the value that is reported is the exact ``max_j f_j`` at the point the
search lands on.  A smoothed value would be *below* the true maximum, which
would make an infeasible design look feasible.  An imperfectly converged
search reports a value that is too *high*, which is the safe direction.

**The gradient is the multiplier-weighted one, not the active constraint's.**
The minimizer is wrapped in ``stop_gradient`` and the sensitivity taken from
the inner problem's own optimality conditions,

.. math:: \\frac{\\partial}{\\partial p}\\min_u \\max_j f_j(u, p)
          = \\sum_j \\lambda_j \\frac{\\partial f_j}{\\partial p}\\Big|_{u^*},
          \\qquad \\sum_j \\lambda_j = 1,\\quad
          \\sum_j \\lambda_j \\nabla_u f_j = 0 ,

where the second condition holds in the control directions that are not
pinned at a bound.  The weights matter: at the minimum of a maximum the
optimal ``u`` sits on the *kink* where two or more constraints are equal, and
differentiating whichever one ``jnp.max`` happened to select there gives a
number that is not the derivative at all.  In the textbook one-control,
two-constraint case the true sensitivity is the average of the two, and the
naive answer is off by a factor of two or is identically zero.

:func:`constraint_multipliers` recovers ``lambda`` from a small least-squares
problem at ``u*``.  The reported *value* is still the exact ``max_j f_j``; only
its derivative is redefined, so a forward evaluation is unaffected.  This
costs one Jacobian of the model rather than a reverse pass through a few
hundred Adam steps, and it does not accumulate the solver's own error into the
derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.params_mixin import ParamsMixin

from difflow.flexibility.sets import ControlSpec


@dataclass
class SolverOptions(ParamsMixin):
    """Settings for the inner minimax search and the outer searches built on it.

    Attributes:
        steps: Projected-Adam iterations per start.
        learning_rate: Step size as a fraction of the box width, decayed on a
            cosine schedule to zero over ``steps``.
        tau: Initial log-sum-exp temperature, as a fraction of the constraint
            scale at the starting point.
        tau_decay: Factor by which ``tau`` shrinks over the run, so the search
            ends on an essentially exact maximum.
        n_starts: Deterministic starting points for the inner problem.
        active_tol: Gap below the worst constraint, relative to the constraint
            scale, at which a constraint stops counting toward the inner
            multipliers.
        bisection_steps: Bisection iterations used by
            :func:`~difflow.flexibility.index.flexibility_index`.
        outer_steps: Projected-Adam iterations for the continuous (KKT)
            search over ``theta``.

    Example:
        >>> opts = SolverOptions(steps=400, n_starts=5)
        >>> opts["steps"]
        400
    """

    steps: int = 250
    learning_rate: float = 0.08
    tau: float = 0.1
    tau_decay: float = 1e-3
    n_starts: int = 3
    active_tol: float = 1e-3
    bisection_steps: int = 40
    outer_steps: int = 150


DEFAULT_OPTIONS = SolverOptions()


def smooth_max(f: Array, tau: Array | float) -> Array:
    """A log-sum-exp upper bound on ``max(f)`` with temperature ``tau``.

    Args:
        f: Constraint values.
        tau: Temperature.  As ``tau -> 0`` this tends to ``max(f)``; it always
            lies within ``tau * log(len(f))`` above it.

    Returns:
        A scalar.
    """
    t = jnp.maximum(jnp.asarray(tau, dtype=float), 1e-300)
    m = jnp.max(f)
    return m + t * jnp.log(jnp.sum(jnp.exp((f - m) / t)))


def box_adam(objective: Callable[[Array, Array], Array],
             x0: Array, lower: Array, upper: Array,
             steps: int, learning_rate: float) -> Array:
    """Projected Adam on a box, with a cosine-decayed step.

    Args:
        objective: ``f(x, progress) -> scalar``, where ``progress`` runs from
            ``0`` to ``1`` and may be used to anneal a smoothing temperature.
        x0: Starting point.
        lower: Lower bounds.
        upper: Upper bounds.
        steps: Number of iterations.
        learning_rate: Step size as a fraction of the box width at
            ``progress = 0``.

    Returns:
        The final iterate, inside the box.
    """
    span = jnp.where(jnp.isfinite(upper - lower), upper - lower, 1.0)
    span = jnp.where(span > 0, span, 1.0)
    n = int(steps)
    b1, b2 = 0.9, 0.999

    def body(carry, k):
        x, m, v = carry
        progress = k / max(n - 1, 1)
        g = jax.grad(objective)(x, progress)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        t = k + 1.0
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        lr = learning_rate * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        x = jnp.clip(x - lr * span * mhat / (jnp.sqrt(vhat) + 1e-12),
                     lower, upper)
        return (x, m, v), None

    init = (jnp.asarray(x0, dtype=float),
            jnp.zeros_like(x0, dtype=float), jnp.zeros_like(x0, dtype=float))
    if n == 0:
        return init[0]
    (x, _, _), _ = lax.scan(body, init, jnp.arange(n, dtype=float))
    return x


def minimax_controls(constraints: Callable[[Array], Array],
                     controls: ControlSpec,
                     options: SolverOptions = DEFAULT_OPTIONS) -> Array:
    """Controls that minimize the worst constraint, ``argmin_u max_j f_j``.

    Args:
        constraints: ``u -> f`` returning the constraint vector, feasible
            where every entry is ``<= 0``.
        controls: The recourse box.
        options: Search settings.

    Returns:
        The best control vector found, of shape ``(controls.n,)``.

    Note:
        The result carries no gradient: it is wrapped in ``stop_gradient`` by
        :func:`minimax_value`, which is what makes the envelope-theorem
        derivative correct and cheap.
    """
    if controls.n == 0:
        return jnp.zeros(0)
    starts = controls.starts(options.n_starts)

    def one(u0):
        scale = lax.stop_gradient(
            jnp.max(jnp.abs(constraints(u0))) + 1.0)

        def obj(u, progress):
            tau = options.tau * scale * options.tau_decay ** progress
            return smooth_max(constraints(u), tau)

        u = box_adam(obj, u0, controls.lower, controls.upper,
                     options.steps, options.learning_rate)
        return u, jnp.max(constraints(u))

    us, vals = jax.vmap(one)(starts)
    return us[jnp.argmin(vals)]


def constraint_multipliers(constraints: Callable[[Array], Array],
                           u_star: Array, controls: ControlSpec,
                           active_tol: float = 1e-3) -> Array:
    """Inner-problem multipliers at a minimax solution.

    Writing the inner problem as ``min_{u,t} t`` subject to ``f_j(u) <= t``,
    the multipliers ``lambda`` satisfy ``sum_j lambda_j = 1``, ``lambda >= 0``,
    ``lambda_j = 0`` for inactive constraints, and stationarity
    ``sum_j lambda_j d f_j / d u_i = 0`` in every control direction ``i`` not
    pinned at a bound.  They are recovered here from that (small, dense,
    overdetermined) linear system by least squares, with inactive constraints
    driven out by a penalty that grows with their slack rather than by a hard
    threshold, and the result projected back onto the simplex.

    Args:
        constraints: ``u -> f``.
        u_star: The minimax point.
        controls: The recourse box, used to detect pinned coordinates.
        active_tol: Slack below the worst constraint, relative to the
            constraint scale, at which a constraint stops contributing.

    Returns:
        A ``(n_constraints,)`` array on the simplex.

    Note:
        With no free control direction there is no stationarity information,
        and a tie between constraints is resolved as an equal split.  Any
        convex combination is a valid subgradient there; the balanced one is
        the one that agrees with a central difference.
    """
    f = jnp.atleast_1d(constraints(u_star))
    n_f = f.shape[0]
    scale = jnp.max(jnp.abs(f)) + 1.0
    gap = jnp.max(f) - f
    penalty = jnp.minimum(10.0 * gap / (active_tol * scale), 1e6)

    if controls.n:
        G = jnp.atleast_2d(jax.jacobian(constraints)(u_star))
        span = jnp.where(controls.upper > controls.lower,
                         controls.upper - controls.lower, 1.0)
        free = ((u_star > controls.lower + 1e-9 * span)
                & (u_star < controls.upper - 1e-9 * span))
        gscale = jnp.max(jnp.abs(G)) + 1e-30
        top = (G / gscale * free[None, :]).T
    else:
        top = jnp.zeros((0, n_f))

    A = jnp.concatenate([top, jnp.ones((1, n_f)), jnp.diag(penalty)], axis=0)
    b = jnp.concatenate([jnp.zeros(top.shape[0]), jnp.ones(1),
                         jnp.zeros(n_f)])
    lam = jnp.linalg.lstsq(A, b, rcond=None)[0]
    lam = jnp.clip(lam, 0.0, None)
    total = jnp.sum(lam)
    fallback = (f >= jnp.max(f)).astype(float)
    lam = jnp.where(total > 1e-12, lam / jnp.where(total > 1e-12, total, 1.0),
                    fallback / jnp.sum(fallback))
    return lam


def minimax_value(constraints: Callable[[Array], Array],
                  controls: ControlSpec,
                  options: SolverOptions = DEFAULT_OPTIONS
                  ) -> tuple[Array, Array]:
    """The inner value ``min_u max_j f_j`` and the controls achieving it.

    Args:
        constraints: ``u -> f``, feasible where every entry is ``<= 0``.
        controls: The recourse box.  Use
            :data:`~difflow.flexibility.sets.NO_CONTROLS` for a design with no
            recourse, in which case the value is simply ``max_j f_j``.
        options: Search settings.

    Returns:
        ``(value, u_star)``.  ``value`` is the *exact* maximum at ``u_star``;
        its derivative with respect to anything ``constraints`` closes over is
        the multiplier-weighted sensitivity from
        :func:`constraint_multipliers`, which is the derivative of the minimax
        value rather than of whichever constraint happened to be selected.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.flexibility import ControlSpec, minimax_value
        >>> spec = ControlSpec(lower=[-5.0], upper=[5.0])
        >>> v, u = minimax_value(lambda u: jnp.array([u[0] - 1.0,
        ...                                           -u[0] - 3.0]), spec)
        >>> bool(abs(v + 2.0) < 1e-3)   # balanced at u = -1, where both are -2
        True
    """
    u_star = lax.stop_gradient(minimax_controls(constraints, controls, options))
    lam = lax.stop_gradient(
        constraint_multipliers(constraints, u_star, controls,
                               options.active_tol))
    f = jnp.atleast_1d(constraints(u_star))
    weighted = jnp.dot(lam, f)
    # Straight-through: the forward value is the exact maximum, and the whole
    # derivative comes from the multiplier-weighted sum.  The exact maximum
    # has to be cut off with stop_gradient as well -- leaving it attached
    # would add ``d f_{argmax} / d p`` on top of the multiplier sum, which is
    # the naive derivative this construction exists to avoid.  See the module
    # documentation.
    exact = lax.stop_gradient(jnp.max(f))
    return exact + (weighted - lax.stop_gradient(weighted)), u_star
