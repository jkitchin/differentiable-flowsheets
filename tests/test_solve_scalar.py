"""Smoke and contract tests for :func:`difflow.solve_scalar`."""

import jax
import jax.numpy as jnp
import pytest

from difflow import solve_scalar


def _sq(x, a):
    return x**2 - a


def test_solve_scalar_smoke():
    x, info = solve_scalar(_sq, 1.0, (2.0,), return_info=True)
    assert jnp.isfinite(x) and bool(info["converged"])
    assert set(info) == {"converged", "iterations", "residual"}
    assert jnp.isfinite(solve_scalar(_sq, None, (2.0,), bracket=(0.0, 5.0)))


def test_solve_scalar_grad_finite():
    def root(a):
        return solve_scalar(_sq, 1.0, (a,))

    def root_bracket_closure(a):
        return solve_scalar(lambda x: x**2 - a, bracket=(0.0, 5.0))

    assert jnp.isfinite(jax.grad(root)(2.0))
    assert jnp.isfinite(jax.grad(root_bracket_closure)(2.0))


def test_solve_scalar_jit():
    f = jax.jit(lambda a: solve_scalar(_sq, 1.0, (a,), bracket=(0.0, 5.0), return_info=True))
    x, info = f(3.0)
    assert jnp.isfinite(x) and bool(info["converged"])
    assert jnp.isfinite(jax.jit(jax.grad(lambda a: solve_scalar(_sq, 1.0, (a,))))(3.0))


def test_solve_scalar_validation():
    with pytest.raises(ValueError):
        solve_scalar(_sq, 1.0, (2.0,), bracket=(3.0, 5.0))          # no sign change
    with pytest.raises(ValueError):
        solve_scalar(_sq, 1.0, (2.0,), bracket=(5.0, 0.0))          # lo > hi
    with pytest.raises(ValueError):
        solve_scalar(_sq, 1.0, (2.0,), bracket=(0.0, 1.0, 2.0))     # not a pair
    with pytest.raises(ValueError):
        solve_scalar(_sq, None, (2.0,))                             # no guess, no bracket
    with pytest.raises(ValueError):
        solve_scalar(_sq, 1.0, (2.0,), max_iter=0)
    with pytest.raises(ValueError):
        solve_scalar(_sq, 1.0, (2.0,), tol=0.0)


def test_solve_scalar_bad_bracket_under_jit_not_converged():
    f = jax.jit(lambda a: solve_scalar(_sq, 1.0, (a,), bracket=(3.0, 5.0), return_info=True)[1]["converged"])
    assert not bool(f(2.0))
