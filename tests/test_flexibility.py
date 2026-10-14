"""Smoke and contract tests for :mod:`difflow.flexibility`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.flexibility import (
    MAX_VERTICES,
    NO_CONTROLS,
    ControlSpec,
    FeasibilityResult,
    FlexibilityResult,
    PenaltyReport,
    StochasticFeasibilityResult,
    UncertaintySet,
    as_control_spec,
    as_uncertainty_set,
    draw_flexibility_region,
    expected_feasibility,
    feasibility_function,
    feasibility_value,
    flexibility_index,
    minimax_value,
    sample_set,
    uncertainty_penalties,
)


def corner_model(d, u, theta):
    return jnp.array([theta[0] + theta[1] - d[0], -theta[0] - 2.0])


CORNER_SET = {"a": (0.0, 1.0), "b": (0.0, 1.0)}


def penalty_model(d, u, theta, phi):
    x = u.sum()
    return jnp.array([phi[0] * theta[0] - x - 1.0, x - d[0]])


def test_uncertainty_set_smoke_and_validation():
    T = as_uncertainty_set({"C": (1.0, 0.2), "T": (300.0, 5.0, 10.0)})
    assert T.names == ("C", "T") and T.n_vertices == 4
    assert np.asarray(T.vertices()).shape == (4, 2)
    assert bool(T.contains(jnp.array([1.0, 300.0])))
    assert "nominal" in T

    big = UncertaintySet(nominal=np.zeros(20), lower=np.ones(20), upper=np.ones(20))
    with pytest.raises(ValueError, match="MAX_VERTICES"):
        big.signs()
    assert big.n_vertices > MAX_VERTICES
    with pytest.raises(ValueError, match="names has length"):
        UncertaintySet(nominal=[1.0], lower=[1.0], upper=[1.0], names=("a", "b"))
    with pytest.raises(ValueError, match="nominal"):
        as_uncertainty_set({"a": (1.0,)})


def test_control_spec_validation():
    with pytest.raises(ValueError, match="finite"):
        ControlSpec(lower=[0.0], upper=[np.inf])
    with pytest.raises(ValueError, match="below lower bound"):
        ControlSpec(lower=[1.0], upper=[0.0])
    with pytest.raises(ValueError, match=r"\(lower, upper\)"):
        as_control_spec({"a": (1.0, 2.0, 3.0)})
    assert as_control_spec(None) is NO_CONTROLS
    assert np.asarray(ControlSpec(lower=[0.0], upper=[4.0]).starts(3)).shape == (3, 1)


def test_minimax_value_no_recourse():
    v, u = minimax_value(lambda u: jnp.array([-1.0, 3.0]), NO_CONTROLS)
    assert float(v) == 3.0 and u.shape == (0,)


def test_feasibility_function_smoke():
    res = feasibility_function(corner_model, [1.0], CORNER_SET, None)
    assert isinstance(res, FeasibilityResult)
    assert np.isfinite(res.psi)
    assert res.summary() and res.describe() and "psi=" in repr(res)
    with pytest.raises(ValueError, match="method must be one of"):
        feasibility_function(corner_model, [1.0], CORNER_SET, None, method="kkt")
    with pytest.raises(ValueError, match="expected 2 f names"):
        feasibility_function(corner_model, [1.0], CORNER_SET, None,
                             constraint_names=("only-one",))


def test_feasibility_grad_finite():
    g = jax.grad(lambda d: feasibility_value(corner_model, d, CORNER_SET, None))(
        jnp.array([1.0]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_flexibility_index_smoke():
    res = flexibility_index(corner_model, [1.0], CORNER_SET, None)
    assert isinstance(res, FlexibilityResult)
    assert np.isfinite(res.index)
    assert res.describe() and "index=" in repr(res)


def test_expected_feasibility_smoke():
    model = lambda d, u, th: jnp.array([th[0] - d[0], -5.0])
    res = expected_feasibility(model, [1.0], {"feed": (1.0, 0.5)}, None,
                               n_samples=32, key=0)
    assert isinstance(res, StochasticFeasibilityResult)
    assert 0.0 <= res.probability <= 1.0
    with pytest.raises(ValueError, match="distribution must be one of"):
        sample_set({"a": (1.0, 0.5)}, 4, 0, distribution="beta")


def test_uncertainty_penalties_smoke_and_validation():
    rep = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                {"u": (0.0, 5.0)}, parameters={"K": 1.0},
                                covariance=[[0.01]])
    assert isinstance(rep, PenaltyReport)
    assert np.all(np.isfinite(rep.required_margin))
    assert rep.describe() and rep.as_dict()
    with pytest.raises(ValueError, match="covariance has shape"):
        uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)}, None,
                              parameters={"K": 1.0}, covariance=np.eye(2))
    with pytest.raises(KeyError, match="no nominal value"):
        uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)}, None,
                              parameters={"K": 1.0}, covariance=[[0.01]],
                              parameter_order=["Q"])


def test_draw_flexibility_region():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = flexibility_index(corner_model, [1.0], CORNER_SET, None)
    draw_flexibility_region(res)
    plt.close("all")
