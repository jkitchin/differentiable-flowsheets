"""Differentiable scalar root finding: :func:`solve_scalar`.

``solve_scalar(f, x0, args)`` finds ``x*`` with ``f(x*, *args) = 0`` by
Newton's method, optionally safeguarded by a bracket ``[lo, hi]`` over which
``f`` changes sign (a Newton step that leaves the current bracket, or is not
finite, is replaced by a bisection step, so a bracketed solve always
converges).

**Gradients.** The iterations run on stop-gradient values inside a
``lax.while_loop`` (so the solve ``jit``-compiles and is never unrolled);
the root is then attached to its inputs by a ``custom_jvp`` carrying the
implicit-function theorem,

    dx*/dp = -(df/dp) / (df/dx)   at x*,

for every array in ``args`` *and* every traced value ``f`` closes over
(those are hoisted into explicit arguments with :func:`jax.closure_convert`).
The JVP rule is written in JAX operations evaluated at the root, so
``jax.grad``, ``jax.jacobian``, ``jax.hessian`` and ``jax.vmap`` all work.
The derivative is exact at the root; it is as good as the root, so a solve
that did not converge (``return_info=True`` reports it) has a meaningless
gradient.

The ``difflow.staged._solve_monotone`` helper the units use internally is a
fixed-iteration bisection for a monotone ``f`` on a bracket; this is the
public, general-purpose counterpart (Newton, an initial guess, an optional
bracket and a convergence report).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array, lax

__all__ = ["solve_scalar"]


def _concrete(value) -> float | None:
    from difflow.flowsheet import _concrete as concrete

    return concrete(value)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _implicit_root(fn, x, *args):
    """Identity on the converged root ``x``; its JVP is the implicit-function theorem."""
    return x


@_implicit_root.defjvp
def _implicit_root_jvp(fn, primals, tangents):
    x, *args = primals
    _, *dargs = tangents
    # re-enter the custom_jvp so that differentiating this rule again (Hessians) sees dx*/dp too
    x = _implicit_root(fn, x, *args)
    df_dx = jax.grad(lambda xx: fn(xx, *args))(x)
    _, df_dp = jax.jvp(lambda *a: fn(x, *a), tuple(args), tuple(dargs))
    safe = jnp.where(jnp.abs(df_dx) > 0.0, df_dx, 1.0)
    return x, jnp.where(jnp.abs(df_dx) > 0.0, -df_dp / safe, 0.0)


def _newton(fn, x0, lo, hi, bracketed: bool, tol, max_iter: int, args):
    """Safeguarded Newton on stop-gradient inputs; returns ``(x, converged, iterations, residual)``."""
    g = jax.value_and_grad(lambda x: fn(x, *args))
    f_lo = fn(lo, *args)

    def cond(c):
        x, a, b, it, done = c
        return (~done) & (it < max_iter)

    def body(c):
        x, a, b, it, _ = c
        fx, dfx = g(x)
        # keep the sign change inside [a, b]
        same = jnp.sign(fx) == jnp.sign(f_lo)
        a = jnp.where(bracketed & same, x, a)
        b = jnp.where(bracketed & ~same, x, b)
        slope = jnp.where(jnp.abs(dfx) > 0.0, dfx, 1.0)
        x_new = x - fx / slope
        bad = ~jnp.isfinite(x_new) | (jnp.abs(dfx) == 0.0)
        outside = (x_new <= jnp.minimum(a, b)) | (x_new >= jnp.maximum(a, b))
        x_new = jnp.where(bracketed & (bad | outside), 0.5 * (a + b), x_new)
        step = jnp.abs(x_new - x)
        done = (step <= tol * jnp.maximum(1.0, jnp.abs(x_new))) | (fx == 0.0)
        x_new = jnp.where(fx == 0.0, x, x_new)
        return x_new, a, b, it + 1, done

    x, _, _, it, done = lax.while_loop(cond, body, (x0, lo, hi, jnp.asarray(0), jnp.asarray(False)))
    res, d_res = g(x)
    # the next Newton step must be within tolerance too: this rejects a "converged" step that is only
    # small relative to an |x| that ran off to infinity, where df/dx underflows to zero
    next_ok = (jnp.abs(d_res) > 0.0) & (
        jnp.abs(res) <= tol * jnp.maximum(1.0, jnp.abs(x)) * jnp.abs(d_res))
    converged = done & jnp.isfinite(x) & jnp.isfinite(res) & ((res == 0.0) | next_ok)
    if bracketed:
        converged = converged & (jnp.sign(f_lo) != jnp.sign(fn(hi, *args)))
    return x, converged, it, res


def solve_scalar(f: Callable[..., Any], x0: Array | float | None = None, args: tuple = (), *,
                 bracket: tuple | None = None, tol: float = 1e-12, max_iter: int = 100,
                 return_info: bool = False):
    """Root ``x*`` of the scalar equation ``f(x, *args) = 0``, differentiable by the implicit-function theorem.

    Args:
        f: ``f(x, *args) -> scalar``, written in JAX operations and smooth near
            the root (Newton needs ``df/dx``, taken by ``jax.grad``).
        x0: Initial guess. May be ``None`` when ``bracket`` is given (the
            bracket's midpoint is used); a guess outside the bracket is clipped
            into it.
        args: Tuple of extra arguments (arrays or pytrees of arrays) passed to
            ``f``; a single non-tuple value is wrapped. The root is
            differentiable with respect to these and to any traced value ``f``
            closes over.
        bracket: Optional ``(lo, hi)`` with ``lo < hi`` and ``f(lo)``,
            ``f(hi)`` of opposite sign (or one of them zero). Newton steps that
            leave the current bracket, or are not finite, become bisection
            steps, so the solve cannot diverge.
        tol: Step tolerance, relative: stop when
            ``|x_{k+1} - x_k| <= tol max(1, |x_{k+1}|)`` (or ``f(x_k) == 0``).
        max_iter: Iteration cap (Newton or bisection steps).
        return_info: If ``True`` return ``(x, info)``.

    Returns:
        The root ``x*`` (a 0-d array). With ``return_info=True``, also ``info``
        with ``converged`` (bool: step below ``tol`` within ``max_iter``, the
        next Newton step ``|f/f'|`` at the root below it too, the root and
        residual finite and, with a bracket, a sign change over it),
        ``iterations`` and ``residual`` (``f(x*)``). ``info`` carries no
        gradient. Check ``converged`` under ``jit``, where a bad bracket cannot
        raise.

    Raises:
        ValueError: (concrete inputs only) ``bracket`` not a pair with
            ``lo < hi``, ``f`` without a sign change over it, neither ``x0``
            nor ``bracket`` given, ``max_iter < 1`` or ``tol <= 0``.

    Example:
        >>> import jax, jax.numpy as jnp
        >>> from difflow import solve_scalar
        >>> root = lambda a: solve_scalar(lambda x, a: x**2 - a, 1.0, (a,))
        >>> float(root(2.0))                       # sqrt(2)
        1.4142135623730951
        >>> float(jax.grad(root)(2.0))             # 1 / (2 sqrt(2)), by the IFT
        0.35355339059327373
        >>> float(solve_scalar(lambda x: jnp.cos(x) - x, bracket=(0.0, 1.0)))
        0.7390851332151607
    """
    if not isinstance(args, tuple):
        args = (args,)
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")
    if not tol > 0:
        raise ValueError("tol must be positive")
    if x0 is None and bracket is None:
        raise ValueError("solve_scalar needs an initial guess x0, a bracket, or both")

    bracketed = bracket is not None
    if bracketed:
        if len(bracket) != 2:
            raise ValueError("bracket must be a pair (lo, hi)")
        lo, hi = (jnp.asarray(v, dtype=float) for v in bracket)
        lo_c, hi_c = _concrete(lo), _concrete(hi)
        if lo_c is not None and hi_c is not None and not lo_c < hi_c:
            raise ValueError(f"bracket must have lo < hi, got ({lo_c}, {hi_c})")
        if x0 is None:
            x0 = 0.5 * (lo + hi)
        x0 = jnp.clip(jnp.asarray(x0, dtype=float), lo, hi)
    else:
        x0 = jnp.asarray(x0, dtype=float)
        lo = hi = x0

    # hoist the traced values f closes over into explicit arguments, so the IFT covers them too
    fn, consts = jax.closure_convert(lambda x, *a: f(x, *a), x0, *args)
    all_args = (*args, *consts)
    sg_args = lax.stop_gradient(all_args)

    if bracketed:
        f_lo, f_hi = _concrete(fn(lax.stop_gradient(lo), *sg_args)), _concrete(fn(lax.stop_gradient(hi), *sg_args))
        if f_lo is not None and f_hi is not None and f_lo * f_hi > 0:
            raise ValueError(
                f"f has the same sign at both ends of the bracket (f(lo) = {f_lo:.6g}, f(hi) = {f_hi:.6g})")

    x, converged, it, res = _newton(fn, lax.stop_gradient(x0), lax.stop_gradient(lo), lax.stop_gradient(hi),
                                    bracketed, tol, int(max_iter), sg_args)
    root = _implicit_root(fn, x, *all_args)
    if return_info:
        return root, {"converged": converged, "iterations": it, "residual": res}
    return root
