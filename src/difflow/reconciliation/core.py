"""Scaling and the KKT core of constrained weighted least squares.

Data reconciliation solves

.. math::

    \\min_x (x - y)^T W (x - y) \\quad \\text{s.t.} \\quad F(x, \\theta) = 0

with :math:`W = \\mathrm{diag}(1/\\sigma_i^2)` on measured entries and
:math:`W_{ii} = 0` on unmeasured ones (unknown parameters, unmetered
streams). Everything in this module works on a *scaled* problem; see
:class:`Scaling` for why that is not optional.

The first-order conditions are the KKT system

.. math::

    \\begin{bmatrix} W & A^T \\\\ A & 0 \\end{bmatrix}
    \\begin{bmatrix} \\Delta x \\\\ \\lambda \\end{bmatrix} =
    \\begin{bmatrix} -W(x - y) \\\\ -F(x) \\end{bmatrix},
    \\qquad A = \\partial F / \\partial x,

iterated to convergence. Dropping the :math:`\\sum_k \\lambda_k
\\nabla^2 F_k` term of the exact Newton Jacobian makes this
Gauss-Newton, which is what :func:`solve_reconciliation` does by
default: that term contains the second derivative of the pipe law's
:math:`q|q|`, and the multipliers are small at a consistent solution,
so dropping it costs almost nothing and buys robustness. Exact
gradients are recovered afterwards from one implicit-function-theorem
correction that *does* use the full Jacobian --- see
:func:`implicit_correction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin


def measured_mask(sigma: Array) -> Array:
    """Boolean mask of the entries that carry a measurement.

    An entry is measured when its standard deviation is finite and
    strictly positive. ``inf`` (or ``nan``) marks a variable to be
    estimated rather than reconciled; a *finite* sigma on a variable
    that is not really metered acts as a prior, which is a legitimate
    way to regularize a weakly identified parameter.
    """
    sigma = jnp.asarray(sigma)
    return jnp.isfinite(sigma) & (sigma > 0.0)


@dataclass
class Scaling(ParamsMixin):
    """Diagonal variable and residual scaling of the KKT system.

    The KKT matrix mixes ``W`` (units of 1/variable^2) with ``A``
    (units of residual/variable), so its conditioning depends on the
    unit system: the same gas network posed in Pa and Pa^2 rather than
    bar and bar^2 is many orders worse conditioned, and can exceed what
    float64 resolves. Scaling makes the formulation unit-invariant, so
    it is on by default.

    With ``d_i = sigma_i`` on measured entries the scaled weight matrix
    becomes a 0/1 mask, which has a second benefit: ``1/sigma^2`` is
    never evaluated, so an infinite sigma cannot introduce a NaN.

    Attributes:
        d: variable scales, shape ``(n,)``, strictly positive.
        r: residual row scales, shape ``(m,)``, strictly positive.
    """

    d: Array
    r: Array

    @property
    def n(self) -> int:
        return int(self.d.shape[0])

    @property
    def m(self) -> int:
        return int(self.r.shape[0])


def identity_scaling(n: int, m: int) -> Scaling:
    """Unit scaling, i.e. no scaling at all."""
    return Scaling(d=jnp.ones(n), r=jnp.ones(m))


def auto_scaling(
    residual_fn: Callable,
    x0: Array,
    sigma: Array,
    *,
    params: Any = None,
    unmeasured_scale: Array | float | None = None,
) -> Scaling:
    """Build variable and residual scales from the problem itself.

    Variable scales are the measurement standard deviations, so that a
    scaled adjustment of 1.0 means "one sigma". Unmeasured entries have
    no sigma to borrow, so they take ``unmeasured_scale`` (by default
    the magnitude of their initial value, floored at 1). Residual rows
    are equilibrated to unit 2-norm in the scaled variables, which
    adapts automatically as variables are added or removed.

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        x0: state to linearize about, shape ``(n,)``.
        sigma: measurement standard deviations, shape ``(n,)``.
        params: extra argument threaded to ``residual_fn``.
        unmeasured_scale: scale for unmeasured entries; scalar or
            ``(n,)``.

    Returns:
        A :class:`Scaling`.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    mask = measured_mask(sigma)

    if unmeasured_scale is None:
        fallback = jnp.maximum(jnp.abs(x0), 1.0)
    else:
        fallback = jnp.broadcast_to(
            jnp.asarray(unmeasured_scale, dtype=jnp.float64), x0.shape
        )
    d = jnp.where(mask, jnp.where(mask, sigma, 1.0), fallback)
    d = jnp.where(d > 0, d, 1.0)

    a = jacobian_of(residual_fn, x0, params)
    row_norms = jnp.linalg.norm(a * d[None, :], axis=1)
    r = jnp.where(row_norms > 0, 1.0 / jnp.where(row_norms > 0, row_norms, 1.0), 1.0)
    return Scaling(d=d, r=r)


def jacobian_of(residual_fn: Callable, x: Array, params: Any = None) -> Array:
    """Constraint Jacobian ``A = dF/dx`` at ``x``, shape ``(m, n)``."""
    return jax.jacobian(lambda xx: residual_fn(xx, params))(x)


def kkt_matrix(a: Array, weights: Array) -> Array:
    """Assemble ``[[diag(w), A^T], [A, 0]]``, shape ``(n+m, n+m)``.

    Args:
        a: constraint Jacobian, shape ``(m, n)``.
        weights: diagonal of ``W``, shape ``(n,)``. Zeros are allowed
            and mark unmeasured variables.
    """
    m, n = a.shape
    top = jnp.concatenate([jnp.diag(weights), a.T], axis=1)
    bottom = jnp.concatenate([a, jnp.zeros((m, m), dtype=a.dtype)], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def _scaled_problem(residual_fn, sigma, scaling, params):
    """Return ``(F_tilde, w_tilde, to_u, to_x)`` for the scaled problem."""
    d, r = scaling.d, scaling.r
    mask = measured_mask(sigma)
    w = jnp.where(mask, 1.0, 0.0)

    def f_tilde(u):
        return r * residual_fn(d * u, params)

    return f_tilde, w, (lambda x: x / d), (lambda u: d * u)


def _gauss_newton_step(f_tilde, w, u, v):
    """One KKT step of the scaled problem; returns ``(du, lam)``."""
    a = jax.jacobian(f_tilde)(u)
    f = f_tilde(u)
    n = u.shape[0]
    k = kkt_matrix(a, w)
    rhs = jnp.concatenate([-w * (u - v), -f])
    sol = jnp.linalg.solve(k, rhs)
    return sol[:n], sol[n:]


def stationarity(f_tilde, w, u, lam, v) -> Array:
    """Scaled first-order conditions ``G(u, lambda)``, shape ``(n+m,)``."""
    f, vjp = jax.vjp(f_tilde, u)
    (at_lam,) = vjp(lam)
    return jnp.concatenate([w * (u - v) + at_lam, f])


def implicit_correction(f_tilde, w, u, lam, v) -> tuple[Array, Array]:
    """One exact Newton step on the stationarity system.

    Applied after Gauss-Newton has converged, this leaves the *value*
    essentially unchanged (``G`` is already ~0) while giving reverse-
    and forward-mode differentiation the exact implicit-function
    derivative, taken through the full Jacobian --- including the
    ``sum_k lambda_k grad^2 F_k`` term that Gauss-Newton drops. Using
    the Gauss-Newton Jacobian here instead would make gradients subtly
    wrong.

    Differentiating the unrolled Gauss-Newton loop would also work, but
    costs one reverse pass per iteration and is only correct at full
    convergence.
    """
    n = u.shape[0]
    z = jnp.concatenate([u, lam])
    z_star = jax.lax.stop_gradient(z)

    def g(zz):
        return stationarity(f_tilde, w, zz[:n], zz[n:], v)

    jac = jax.jacobian(g)(z_star)
    z_new = z_star - jnp.linalg.solve(jac, g(z_star))
    return z_new[:n], z_new[n:]


def solve_reconciliation(
    residual_fn: Callable,
    y: Array,
    sigma: Array,
    *,
    x0: Array,
    scaling: Scaling,
    params: Any = None,
    n_steps: int = 12,
    method: str = "gauss_newton",
    correct: bool = True,
) -> tuple[Array, Array]:
    """Solve the constrained WLS problem. Traceable and differentiable.

    Runs a fixed number of steps so the trip count is static under
    ``jit``/``vmap``; convergence is judged by the caller from the
    residual at the returned point (:func:`difflow.reconciliation.
    reconcile` does this).

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        y: measurements, shape ``(n,)``. Entries whose sigma is
            infinite are ignored and may be nan.
        sigma: standard deviations, shape ``(n,)``; ``inf`` = unmeasured.
        x0: initial state, shape ``(n,)``.
        scaling: variable and residual scaling to solve in.
        params: extra argument threaded to ``residual_fn``.
        n_steps: number of iterations.
        method: ``"gauss_newton"`` (default) or ``"newton"``, which
            iterates the full stationarity system instead.
        correct: apply :func:`implicit_correction` at the end, giving
            exact derivatives w.r.t. ``y``, ``sigma`` and ``params``.

    Returns:
        ``(x_hat, multipliers)`` in unscaled units.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    f_tilde, w, to_u, to_x = _scaled_problem(
        residual_fn, sigma, scaling, params
    )
    mask = measured_mask(sigma)
    # y is meaningless where unmeasured and may be nan; zero it so it
    # cannot poison the arithmetic even though its weight is zero.
    y_safe = jnp.where(mask, y, 0.0)
    v = to_u(y_safe)
    u = to_u(jnp.where(mask, y_safe, x0))
    lam = jnp.zeros(scaling.m, dtype=jnp.float64)

    if method == "gauss_newton":
        def body(_, state):
            uu, _lam = state
            du, ll = _gauss_newton_step(f_tilde, w, uu, v)
            return uu + du, ll
    elif method == "newton":
        def body(_, state):
            uu, ll = state
            z = jnp.concatenate([uu, ll])
            n = uu.shape[0]

            def g(zz):
                return stationarity(f_tilde, w, zz[:n], zz[n:], v)

            dz = jnp.linalg.solve(jax.jacobian(g)(z), -g(z))
            z = z + dz
            return z[:n], z[n:]
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'gauss_newton' or 'newton'"
        )

    u, lam = jax.lax.fori_loop(0, n_steps, body, (u, lam))
    if correct:
        u, lam = implicit_correction(f_tilde, w, u, lam, v)
    return to_x(u), scaling.r * lam


def reconciled_covariance(
    residual_fn: Callable,
    x: Array,
    sigma: Array,
    *,
    scaling: Scaling,
    params: Any = None,
) -> Array:
    """Covariance of the reconciled estimates, shape ``(n, n)``.

    Computed as the leading block of the inverse KKT matrix,

    .. math:: \\Sigma_{\\hat x} = [K^{-1}]_{11},

    which for a fully measured problem equals the textbook projection
    :math:`\\Sigma - \\Sigma A^T (A \\Sigma A^T)^{-1} A \\Sigma` but,
    unlike it, stays well defined when some variables are unmeasured
    --- so the standard error of an estimated parameter comes out of
    the same expression as the reconciled variance of a metered flow.

    Args:
        residual_fn: ``F(x, params) -> (m,)``.
        x: point to linearize about, normally the reconciled state.
        sigma: standard deviations, ``inf`` = unmeasured.
        scaling: the scaling the problem was solved in.
        params: extra argument threaded to ``residual_fn``.
    """
    d = scaling.d
    f_tilde, w, _, _ = _scaled_problem(residual_fn, sigma, scaling, params)
    u = x / d
    a = jax.jacobian(f_tilde)(u)
    n = u.shape[0]
    k = kkt_matrix(a, w)
    cov_u = jnp.linalg.solve(k, jnp.eye(k.shape[0]))[:n, :n]
    return cov_u * d[:, None] * d[None, :]


def measurement_sensitivity(
    residual_fn: Callable,
    y: Array,
    sigma: Array,
    *,
    x0: Array,
    scaling: Scaling,
    params: Any = None,
    n_steps: int = 12,
) -> Array:
    """``S = d x_hat / d y``, shape ``(n, n)``, by forward-mode autodiff.

    This is the matrix every reconciliation textbook writes down as a
    projection; here it falls out of differentiating the solver.

    For **linear** constraints it satisfies ``S Sigma S^T ==
    Sigma_xhat`` exactly, since ``S = Sigma_xhat W`` and ``W Sigma W =
    W``. For nonlinear constraints the two differ by the curvature term
    ``sum_k lambda_k grad^2 F_k`` that :func:`reconciled_covariance`
    drops: they agree to machine precision when the data are consistent
    (the multipliers vanish) and diverge in proportion to how
    inconsistent the data are --- around 1% of the largest covariance
    entry at a typical noise level on the gas network of
    ``examples/28_data_reconciliation.ipynb``, 4% at three times that
    noise.

    So the classical covariance formula is itself a linearization, and
    differentiating the solver is the more faithful of the two. The
    discrepancy is a useful diagnostic: a large one means the model is
    strongly nonlinear over the range the adjustments span.
    """
    def solve_for(yy):
        x_hat, _ = solve_reconciliation(
            residual_fn, yy, sigma, x0=x0, scaling=scaling,
            params=params, n_steps=n_steps,
        )
        return x_hat

    return jax.jacfwd(solve_for)(jnp.asarray(y, dtype=jnp.float64))
