"""Smoke and validation tests for difflow_power.ipm."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow_power.ipm import (
    NLP,
    differentiable_solution,
    kkt_residuals,
    solve_nlp,
)


def _bounded_nlp():
    """min (x-3)^2 + (y-2)^2 s.t. x + y = p, x >= 2."""
    return NLP(
        objective=lambda x, p: (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2,
        n=2,
        m_eq=1,
        m_in=1,
        equalities=lambda x, p: jnp.atleast_1d(x[0] + x[1] - p),
        inequalities=lambda x, p: jnp.atleast_1d(2.0 - x[0]),
    )


def test_solve_nlp_smoke():
    nlp = _bounded_nlp()
    result = solve_nlp(nlp, jnp.zeros(2), 4.0)
    assert result.converged
    assert bool(jnp.all(jnp.isfinite(result.x)))
    residuals = kkt_residuals(result.kkt, nlp, 4.0, result.mu)
    assert bool(jnp.all(jnp.isfinite(residuals)))
    assert result.history and "mu" in result.history[0]


def test_differentiable_solution_jacobian_is_finite():
    nlp = _bounded_nlp()
    result = solve_nlp(nlp, jnp.zeros(2), 4.0)
    jacobian = jax.jacobian(
        lambda p: differentiable_solution(nlp, result.kkt, p, result.mu)[:2]
    )(4.0)
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_infeasible_problem_is_reported_not_raised():
    nlp = NLP(
        objective=lambda x, p: jnp.sum(x ** 2),
        n=1,
        m_in=2,
        inequalities=lambda x, p: jnp.stack([x[0] - 1.0, 2.0 - x[0]]),
    )
    result = solve_nlp(nlp, jnp.asarray([0.0]), max_iterations=40)
    assert not result.converged
    assert "DID NOT CONVERGE" in result.summary()


def test_wrong_sized_start_is_rejected():
    nlp = NLP(objective=lambda x, p: jnp.sum(x ** 2), n=3)
    with pytest.raises(ValueError, match="expected"):
        solve_nlp(nlp, jnp.zeros(2))
