"""Tests for difflow_power.ipm, the JAX interior-point NLP solver.

Every problem here has a solution that can be written down, so the
tests compare against algebra rather than against another solver. The
awkward cases are deliberate: a weakly convex problem (zero Hessian, so
the inertia correction has real work to do), a degenerate active set
(where the barrier smooths a kink), and an infeasible one (which must
be reported, not raised or silently mis-answered).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow_power.ipm import (
    NLP,
    differentiable_solution,
    kkt_residuals,
    solve_nlp,
)


def test_unconstrained_quadratic():
    nlp = NLP(objective=lambda x, p: jnp.sum((x - 3.0) ** 2), n=3)
    result = solve_nlp(nlp, jnp.zeros(3))
    assert result.converged
    np.testing.assert_allclose(result.x, 3.0, atol=1e-8)


def test_equality_constrained_quadratic_matches_the_kkt_algebra():
    """min |x|^2 s.t. sum(x) = 1 has x = 1/n and lambda = -2/n."""
    n = 4
    nlp = NLP(
        objective=lambda x, p: jnp.sum(x ** 2),
        n=n,
        m_eq=1,
        equalities=lambda x, p: jnp.atleast_1d(jnp.sum(x) - 1.0),
    )
    result = solve_nlp(nlp, jnp.zeros(n))
    assert result.converged
    np.testing.assert_allclose(result.x, 1.0 / n, atol=1e-8)
    assert float(result.lam[0]) == pytest.approx(-2.0 / n, abs=1e-8)


def test_active_bound_gives_a_positive_multiplier():
    """min x^2 s.t. x >= 1 puts x on the bound with multiplier 2."""
    nlp = NLP(
        objective=lambda x, p: jnp.sum(x ** 2),
        n=1,
        m_in=1,
        inequalities=lambda x, p: jnp.atleast_1d(1.0 - x[0]),
    )
    result = solve_nlp(nlp, jnp.asarray([5.0]))
    assert result.converged
    assert float(result.x[0]) == pytest.approx(1.0, abs=1e-7)
    assert float(result.z[0]) == pytest.approx(2.0, abs=1e-5)
    assert bool(result.active()[0])


def test_inactive_bound_has_a_zero_multiplier():
    nlp = NLP(
        objective=lambda x, p: jnp.sum(x ** 2),
        n=1,
        m_in=1,
        inequalities=lambda x, p: jnp.atleast_1d(-1.0 - x[0]),
    )
    result = solve_nlp(nlp, jnp.asarray([5.0]))
    assert result.converged
    assert float(result.x[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(result.z[0]) < 1e-6
    assert not bool(result.active()[0])


def test_linear_objective_converges_despite_zero_curvature():
    """A pure LP: the Hessian is zero, so the inertia correction has to
    supply every bit of curvature the Newton system needs.

    min -x0 - 2 x1 s.t. x0 + x1 <= 1, x >= 0  ->  x = (0, 1), f = -2.
    """
    nlp = NLP(
        objective=lambda x, p: -x[0] - 2.0 * x[1],
        n=2,
        m_in=3,
        inequalities=lambda x, p: jnp.stack(
            [x[0] + x[1] - 1.0, -x[0], -x[1]]
        ),
    )
    result = solve_nlp(nlp, jnp.asarray([0.3, 0.3]))
    assert result.converged
    np.testing.assert_allclose(result.x, [0.0, 1.0], atol=1e-6)
    assert result.objective == pytest.approx(-2.0, abs=1e-6)


def test_nonconvex_problem_finds_a_minimum_not_a_saddle():
    """``x^4 - 4 x^2`` has a saddle at 0 and minima at +-sqrt(2).

    Started AT the saddle, an uncorrected Newton step stays there. The
    inertia correction is what moves it off.
    """
    nlp = NLP(objective=lambda x, p: x[0] ** 4 - 4.0 * x[0] ** 2, n=1)
    result = solve_nlp(nlp, jnp.asarray([0.1]))
    assert result.converged
    assert abs(float(result.x[0])) == pytest.approx(np.sqrt(2.0), abs=1e-6)


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


def test_sensitivity_matches_the_analytic_answer_on_both_sides():
    """min (x-3)^2 + (y-2)^2 s.t. x + y = p, x >= 2.

    For p > 3 the bound is slack and the solution slides along the line
    at ``dx/dp = dy/dp = 1/2``. For p < 3 the bound holds x at 2 and all
    the movement goes into y.
    """
    nlp = NLP(
        objective=lambda x, p: (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2,
        n=2,
        m_eq=1,
        m_in=1,
        equalities=lambda x, p: jnp.atleast_1d(x[0] + x[1] - p),
        inequalities=lambda x, p: jnp.atleast_1d(2.0 - x[0]),
    )
    for value, want in ((4.0, [0.5, 0.5]), (2.0, [0.0, 1.0])):
        result = solve_nlp(nlp, jnp.zeros(2), value)
        assert result.converged
        jacobian = jax.jacobian(
            lambda p: differentiable_solution(
                nlp, result.kkt, p, result.mu
            )[:2]
        )(value)
        np.testing.assert_allclose(jacobian, want, atol=1e-6)


def test_kkt_residuals_vanish_at_the_solution():
    nlp = NLP(
        objective=lambda x, p: jnp.sum((x - p) ** 2),
        n=2,
        m_in=1,
        inequalities=lambda x, p: jnp.atleast_1d(x[0] - 0.5),
    )
    params = jnp.asarray([1.0, 1.0])
    result = solve_nlp(nlp, jnp.zeros(2), params)
    assert result.converged
    residuals = kkt_residuals(result.kkt, nlp, params, result.mu)
    assert float(jnp.max(jnp.abs(residuals))) < 1e-7


def test_history_records_the_descent():
    nlp = NLP(objective=lambda x, p: jnp.sum((x - 3.0) ** 2), n=2)
    result = solve_nlp(nlp, jnp.zeros(2))
    assert result.history
    for key in (
        "objective", "feasibility", "stationarity",
        "complementarity", "mu", "step", "regularization",
    ):
        assert key in result.history[0]
    assert result.history[-1]["mu"] <= result.history[0]["mu"]


def test_wrong_sized_start_is_rejected():
    nlp = NLP(objective=lambda x, p: jnp.sum(x ** 2), n=3)
    with pytest.raises(ValueError, match="expected"):
        solve_nlp(nlp, jnp.zeros(2))


def test_verbose_logging_runs(capsys):
    nlp = NLP(objective=lambda x, p: jnp.sum((x - 1.0) ** 2), n=1)
    solve_nlp(nlp, jnp.zeros(1), verbose=True)
    assert "feas" in capsys.readouterr().out
