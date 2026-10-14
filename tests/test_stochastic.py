"""Tests for difflow.stochastic: smoke, validation and one solver regression."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow.stochastic as st
from difflow.stochastic.risk import _softplus

FAST = st.SAAOptions(steps=30, rounds=1, n_starts=1)


def quadratic(x, u, theta):
    return {"cost": (x["x"] - theta["t"]) ** 2 + (u["u"] - theta["t"]) ** 2,
            "level": x["x"] - theta["t"]}


@pytest.fixture(scope="module")
def scen():
    return st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=32, seed=0)


def problem(**kw):
    return st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                              recourse={"u": (0.0, 10.0)}, objective="cost", **kw)


def test_scenario_set_smoke():
    s = st.ScenarioSet.normal({"a": (1.0, 0.1), "b": (2.0, 0.2)}, n=32, seed=0)
    assert (s.n_scenarios, s.n_parameters) == (32, 2)
    assert float(jnp.sum(s.weights)) == pytest.approx(1.0)
    assert s.redraw(7).draws.shape == s.draws.shape


def test_scenario_set_validation():
    with pytest.raises(ValueError, match="must be positive"):
        st.ScenarioSet.lognormal({"D": (0.0, 0.2)}, n=8)
    with pytest.raises(ValueError, match="reversed"):
        st.ScenarioSet.uniform({"x": (3.0, 1.0)}, n=8)
    with pytest.raises(ValueError, match="no distribution to redraw"):
        st.ScenarioSet.from_samples(np.zeros((4, 1)), ["a"]).redraw(1)
    with pytest.raises(ValueError, match="names for"):
        st.ScenarioSet.from_samples(np.zeros((4, 2)), ["a"])


def test_risk_and_constraint_validation():
    with pytest.raises(ValueError, match="strictly in"):
        st.CVaR(1.0)
    with pytest.raises(ValueError, match="operator"):
        st.Expected("g", "<", 1.0)
    with pytest.raises(ValueError, match="probability zero"):
        st.Chance("g", "==", 1.0, alpha=0.9)


def test_cvar_smoothed_positive_part_has_a_live_slope_at_zero():
    """Regression: a zero slope here froze every one-scenario CVaR solve."""
    assert float(jax.grad(lambda z: _softplus(z, 0.1))(0.0)) == pytest.approx(0.5)


def test_problem_validation(scen):
    with pytest.raises(ValueError, match="both the first stage"):
        st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 1.0)},
                           recourse={"x": (0.0, 1.0)}, objective="cost")
    with pytest.raises(ValueError, match="at least one first-stage"):
        st.TwoStageProblem(model=quadratic, first_stage={}, objective="cost")
    p = st.TwoStageProblem(model=lambda x, u, th: x["x"],
                           first_stage={"x": (0.0, 1.0)}, objective="cost")
    with pytest.raises(TypeError, match="mapping of named outputs"):
        p.outputs(jnp.array([1.0]), jnp.zeros((scen.n_scenarios, 0)), scen)
    with pytest.raises(KeyError, match="not declared decisions"):
        st.evaluate(problem(), {"x": 1.0, "y": 2.0}, None, scen)


def test_solve_bounds_and_health_smoke(scen):
    p = problem(risk=("cvar", 0.8), constraints=[("level", "<=", 2.0, 0.9)])
    res = st.solve_saa(p, scen, options=FAST)
    assert np.isfinite(res.first_stage["x"]) and np.isfinite(res.risk_value)
    assert "CVaR_0.8" in res.summary()
    rep = st.bounds(problem(), scen, options=FAST)
    assert np.isfinite(rep.vss) and np.isfinite(rep.evpi)
    assert isinstance(st.check_scenario_health(p, scen, res).warnings, list)
    with pytest.raises(ValueError, match="at least two replications"):
        st.optimality_gap(p, scen, res, n_replications=1)
