"""Smoke and API tests for difflow.mhe."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.mhe import (  # noqa: E402
    ArrivalCost,
    Measurement,
    MeasurementWindow,
    MHEProblem,
    StateSpaceModel,
    augment_parameters,
    build_window,
    check_observability,
    ekf_update,
    estimate,
    linear_model,
    run_ekf,
    run_mhe,
    slice_window,
)


def window_of(n=6, n_y=1):
    records = [Measurement(time=float(k), values=[1.0 + 0.1 * k] * n_y, sigma=[0.2] * n_y)
               for k in range(n + 1)]
    window, dropped = build_window(np.arange(n + 1.0), records, n_y=n_y)
    assert not dropped
    return window


def model_2d():
    return linear_model(jnp.array([[0.9, 0.1], [0.0, 0.85]]), jnp.array([[1.0, 0.0]]),
                        x_names=["a", "b"], y_names=["y"])


def test_model_construction_and_rollout():
    model = linear_model(jnp.eye(2) * 0.9, jnp.eye(2))
    assert model.rollout(jnp.ones(2), jnp.zeros((3, 0)), jnp.zeros((3, 2))).shape == (4, 2)
    ode = StateSpaceModel.from_ode(lambda t, x, u, th: -0.5 * x, lambda x, u, th: x,
                                   n_x=1, n_y=1, dt=1.0)
    assert np.isfinite(float(ode.step(jnp.array([1.0]), jnp.zeros(0), jnp.zeros(1))[0]))


def test_model_validation():
    with pytest.raises(ValueError, match="lower bound above upper"):
        linear_model(jnp.eye(1), jnp.eye(1), lb=1.0, ub=0.0)
    with pytest.raises(ValueError, match="x_names"):
        linear_model(jnp.eye(2), jnp.eye(2), x_names=["only-one"])


def test_augment_parameters():
    base = StateSpaceModel(f=lambda x, u, w, th: th["k"] * x + w, h=lambda x, u, th: x,
                           n_x=1, n_y=1, x_names=["c"], y_names=["yc"])
    aug = augment_parameters(base, ["k"], lb=0.0, ub=1.0)
    assert aug.n_x == 2 and aug.x_names == ["c", "k"] and aug.param_names == ["k"]
    with pytest.raises(ValueError, match="already carries"):
        augment_parameters(aug, ["j"])
    with pytest.raises(ValueError, match="at least one"):
        augment_parameters(base, [])


def test_build_window_places_delayed_records_and_validates():
    record = Measurement(time=1.0, values={"assay": 0.3}, sigma={"assay": 0.01}, reported=6.0)
    window, _ = build_window(np.arange(7.0), [record], y_names=["assay"])
    mask = np.asarray(window.mask)
    assert mask[1, 0] and mask.sum() == 1
    early, dropped = build_window(np.arange(7.0), [record], y_names=["assay"], now=3.0)
    assert len(dropped) == 1 and early.n_measurements == 0
    assert "assay" in window.summary()
    with pytest.raises(ValueError, match="not one of"):
        build_window(np.arange(4.0), [Measurement(0.0, {"nope": 1.0}, 0.1)], y_names=["yes"])
    with pytest.raises(ValueError, match="but the model has"):
        build_window(np.arange(4.0), [Measurement(0.0, [1.0, 2.0], 0.1)], n_y=1)
    with pytest.raises(ValueError, match="at least 2 points"):
        build_window([0.0], [], n_y=1)


def test_slice_window():
    window = window_of(8)
    assert slice_window(window, 3, 4).horizon == 4
    with pytest.raises(IndexError):
        slice_window(window, 6, 5)


def test_ekf_unsampled_channel_gradient_finite():
    """An inf sigma (unsampled channel) used to give a NaN cotangent."""
    model = linear_model(jnp.array([[0.9]]), jnp.array([[1.0], [1.0]]))

    def posterior(sigma):
        x_up, _, _ = ekf_update(model, jnp.array([1.0]), jnp.array([[0.25]]),
                                jnp.array([1.5, 99.0]), sigma, jnp.zeros(0))
        return jnp.sum(x_up)
    assert np.all(np.isfinite(np.asarray(jax.grad(posterior)(jnp.array([0.3, jnp.inf])))))


def test_run_ekf_smoke():
    run = run_ekf(model_2d(), window_of(), x0=jnp.zeros(2), P0=jnp.eye(2),
                  process_std=jnp.array([0.03, 0.03]))
    assert np.asarray(run.P).shape == (7, 2, 2)
    assert np.all(np.isfinite(np.asarray(run.x)))


def test_run_mhe_smoke():
    model, window = model_2d(), window_of()
    q = jnp.array([0.05, 0.05])
    run = run_mhe(model, window, horizon=3, process_std=q, x0=jnp.zeros(2), P0=jnp.eye(2))
    assert np.asarray(run.x).shape == (7, 2)
    assert "moving-horizon" in run.summary()


def test_estimate_grad_finite():
    model = linear_model(jnp.array([[0.9]]), jnp.array([[1.0]]))
    window = window_of(4)
    problem = MHEProblem(model, ArrivalCost.diagonal(jnp.zeros(1), 1.0),
                         process_std=jnp.array([0.05]))
    g = jax.grad(lambda y: estimate(problem, replace(window, y=y))[-1, 0])(window.y)
    assert np.all(np.isfinite(np.asarray(g)))


def test_check_observability_flags_hidden_state():
    decoupled = linear_model(jnp.array([[0.9, 0.0], [0.0, 0.8]]), jnp.array([[1.0, 0.0]]),
                             x_names=["seen", "hidden"], y_names=["y"])
    report = check_observability(decoupled, window_of(4), jnp.array([1.0, 1.0]))
    assert report.unobservable == ["hidden"]
    with pytest.raises(ValueError, match="hidden"):
        report.raise_if_unobservable()


def test_arrival_cost():
    arrival = ArrivalCost.diagonal(jnp.array([1.0, -2.0]), jnp.array([0.5, 2.0]))
    assert np.isclose(float(arrival.cost(arrival.x_bar)), 0.0)
    assert np.isfinite(float(arrival.inflate(4.0).cost(jnp.zeros(2))))
    with pytest.raises(ValueError, match="positive"):
        arrival.inflate(0.0)


def test_validation():
    model = linear_model(jnp.eye(1), jnp.eye(1))
    with pytest.raises(ValueError, match="strictly positive"):
        MHEProblem(model, ArrivalCost.diagonal(jnp.zeros(1), 1.0), process_std=jnp.array([0.0]))
    with pytest.raises(ValueError, match="does not fit"):
        run_mhe(model, window_of(4), horizon=99, process_std=jnp.array([0.02]),
                x0=jnp.zeros(1), P0=jnp.eye(1))


def test_window_is_params_like():
    window = window_of(3)
    assert isinstance(window, MeasurementWindow)
    assert "y" in window and {"times", "y", "sigma", "u"} <= set(window.keys())


def _counting_model():
    traces = []

    def f(x, u, w, th):
        traces.append(1)              # runs once per trace, not per call
        return 0.9 * x + w

    return StateSpaceModel(f=f, h=lambda x, u, th: x, n_x=1, n_y=1), traces


def test_run_mhe_compiles_once_across_windows_and_runs():
    # Regression (#519): the arrival-cost update ran eagerly and recompiled
    # at every window, and a second identical run compiled everything again.
    kw = {"horizon": 3, "process_std": jnp.array([0.05]), "x0": jnp.zeros(1), "P0": jnp.eye(1)}
    counts = []
    for n in (6, 12):
        model, traces = _counting_model()
        run_mhe(model, window_of(n), **kw)
        counts.append(len(traces))
        run_mhe(model, window_of(n), **kw)
        assert len(traces) == counts[-1]          # the second run traces nothing
    assert counts[0] == counts[1]                 # independent of the number of windows
