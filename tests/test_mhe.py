"""Tests for difflow.mhe.

The load-bearing test is :func:`test_mhe_full_information_matches_kalman`.
For a linear model with Gaussian noise and no active constraints, the
moving-horizon problem over the *whole* record with the true prior as its
arrival cost is the maximum-a-posteriori problem the Kalman filter solves
in closed form, and the smoothed estimate of the *last* state in a window
is the filtered estimate at that time. So the two must agree to solver
tolerance, not to a fudge factor -- and they do, to about 1e-13 in the
mean and 1e-15 in the covariance. Every other numerical choice in the
package (the whitening, the scaling, the elimination of the dynamics, the
Gauss-Newton covariance, the ``inf``-sigma convention for an unsampled
channel) is exercised by that one comparison, because getting any of them
wrong breaks the agreement.

The rest of the suite pins the things the issue asks for that the linear
comparison cannot see: that a delayed sample is placed at the time it was
*taken* (checked against the wrong answer that placing it at the current
time gives), that a bound keeps a concentration non-negative where an
unconstrained fit goes negative, that an augmented parameter tracks a
known drift, and that ``jit`` and ``grad`` go through the estimate.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

jax.config.update("jax_enable_x64", True)

from difflow.mhe import (
    ArrivalCost,
    Measurement,
    MeasurementWindow,
    MHEProblem,
    StateSpaceModel,
    advance_arrival_cost,
    augment_parameters,
    build_window,
    check_observability,
    ekf_predict,
    ekf_update,
    estimate,
    linear_model,
    mhe_global_test,
    run_ekf,
    run_mhe,
    slice_window,
    solve_mhe,
)


# =============================================================================
# Helpers
# =============================================================================


def simulate(a, c, x0, n, q_std, r_std, seed):
    """Simulate ``x+ = A x + w``, ``y = C x + v``.

    Returns ``(xs, ys)`` with ``xs`` shape ``(n + 1, n_x)`` and ``ys``
    shape ``(n + 1, n_y)``; ``ys[k]`` is a reading of ``xs[k]``.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    q_std = np.broadcast_to(np.asarray(q_std, dtype=float), (a.shape[0],))
    r_std = np.broadcast_to(np.asarray(r_std, dtype=float), (c.shape[0],))
    x = np.asarray(x0, dtype=float).copy()
    xs, ys = [], []
    for _ in range(n + 1):
        xs.append(x.copy())
        ys.append(c @ x + r_std * rng.normal(size=c.shape[0]))
        x = a @ x + q_std * rng.normal(size=a.shape[0])
    return np.array(xs), np.array(ys)


def dense_window(ys, r_std, y_names):
    """Every channel sampled at every grid point."""
    n = ys.shape[0] - 1
    r_std = np.broadcast_to(np.asarray(r_std, dtype=float), (ys.shape[1],))
    records = [
        Measurement(time=float(k), values=list(ys[k]), sigma=list(r_std))
        for k in range(n + 1)
    ]
    window, dropped = build_window(
        np.arange(n + 1.0), records, y_names=list(y_names)
    )
    assert not dropped
    return window


# =============================================================================
# StateSpaceModel
# =============================================================================


def test_rollout_matches_manual_iteration():
    a = jnp.array([[0.9, 0.1], [0.0, 0.8]])
    model = linear_model(a, jnp.eye(2))
    x0 = jnp.array([1.0, -2.0])
    w = jnp.array([[0.1, 0.0], [0.0, -0.1], [0.05, 0.05]])

    xs = model.rollout(x0, jnp.zeros((3, 0)), w)

    assert xs.shape == (4, 2)
    expected = np.asarray(x0, dtype=float)
    for k in range(3):
        expected = np.asarray(a) @ expected + np.asarray(w[k])
        assert np.allclose(np.asarray(xs[k + 1]), expected)


def test_jacobians_of_a_linear_model_are_its_matrices():
    a = jnp.array([[0.9, 0.1], [-0.2, 0.8]])
    g = jnp.array([[1.0], [2.0]])
    c = jnp.array([[1.0, 0.0]])
    model = linear_model(a, c, g=g)

    f_x, f_w = model.jacobians(
        jnp.array([1.0, 1.0]), jnp.zeros(0), jnp.zeros(1)
    )
    h = model.observation_jacobian(jnp.array([1.0, 1.0]), jnp.zeros(0))

    assert np.allclose(np.asarray(f_x), np.asarray(a))
    assert np.allclose(np.asarray(f_w), np.asarray(g))
    assert np.allclose(np.asarray(h), np.asarray(c))
    assert model.n_w == 1


def test_from_ode_discretises_with_difflow_dynamic():
    """The flow map over one interval, against the analytic solution."""
    model = StateSpaceModel.from_ode(
        lambda t, x, u, th: -0.5 * x,
        lambda x, u, th: x,
        n_x=1, n_y=1, dt=1.0,
    )
    x1 = model.step(jnp.array([1.0]), jnp.zeros(0), jnp.zeros(1))
    assert np.isclose(float(x1[0]), np.exp(-0.5), atol=1e-6)

    # The same through a diffrax backend, with theta threaded to the rhs.
    stiff = StateSpaceModel.from_ode(
        lambda t, x, u, th: -th["k"] * x,
        lambda x, u, th: x,
        n_x=1, n_y=1, dt=1.0, method="diffrax:tsit5",
    )
    x1b = stiff.step(jnp.array([1.0]), jnp.zeros(0), jnp.zeros(1), {"k": 0.5})
    assert np.isclose(float(x1b[0]), np.exp(-0.5), atol=1e-6)


def test_model_validates_bounds_and_names():
    with pytest.raises(ValueError, match="lower bound above upper"):
        linear_model(jnp.eye(1), jnp.eye(1), lb=1.0, ub=0.0)
    with pytest.raises(ValueError, match="x_names"):
        linear_model(jnp.eye(2), jnp.eye(2), x_names=["only-one"])

    unbounded = linear_model(jnp.eye(1), jnp.eye(1))
    assert not unbounded.is_bounded
    bounded = linear_model(jnp.eye(1), jnp.eye(1), lb=0.0)
    assert bounded.is_bounded
    lb, ub = bounded.bounds
    assert float(lb[0]) == 0.0 and np.isinf(float(ub[0]))


def test_augment_parameters_appends_a_random_walk():
    base = StateSpaceModel(
        f=lambda x, u, w, th: th["k"] * x + w,
        h=lambda x, u, th: x,
        n_x=1, n_y=1, x_names=["c"], y_names=["yc"], lb=0.0,
    )
    aug = augment_parameters(base, ["k"], lb=0.0, ub=1.0)

    assert aug.n_x == 2 and aug.n_w == 2
    assert aug.x_names == ["c", "k"] and aug.param_names == ["k"]
    lb, ub = aug.bounds
    assert np.allclose(np.asarray(lb), [0.0, 0.0])
    assert np.isinf(float(ub[0])) and float(ub[1]) == 1.0

    # The parameter has no dynamics beyond its own noise, and it is the
    # value used by f, not the fixed theta.
    z1 = aug.step(jnp.array([2.0, 0.5]), jnp.zeros(0), jnp.zeros(2),
                  {"k": 99.0})
    assert np.allclose(np.asarray(z1), [1.0, 0.5])
    z2 = aug.step(jnp.array([2.0, 0.5]), jnp.zeros(0), jnp.array([0.0, 0.3]),
                  {"k": 99.0})
    assert np.isclose(float(z2[1]), 0.8)


def test_augment_parameters_rejects_a_second_augmentation():
    base = linear_model(jnp.eye(1), jnp.eye(1))
    aug = augment_parameters(base, ["k"])
    with pytest.raises(ValueError, match="already carries"):
        augment_parameters(aug, ["j"])
    with pytest.raises(ValueError, match="at least one"):
        augment_parameters(base, [])


# =============================================================================
# Timestamped, multi-rate, delayed measurements
# =============================================================================


def test_delayed_record_is_placed_at_its_sample_time():
    """A record is a statement about ``time``, not about ``reported``."""
    record = Measurement(time=1.0, values={"assay": 0.3}, sigma={"assay": 0.01},
                         reported=6.0)
    assert record.delay == 5.0
    assert record.available_at == 6.0

    window, dropped = build_window(np.arange(7.0), [record],
                                   y_names=["assay"])
    assert not dropped
    mask = np.asarray(window.mask)
    assert mask.sum() == 1
    assert mask[1, 0]                       # at the sample time
    assert not mask[6, 0]                   # not at the reporting time


def test_now_excludes_records_not_yet_reported():
    record = Measurement(time=1.0, values=[0.3], sigma=[0.01], reported=6.0)
    early, dropped = build_window(np.arange(7.0), [record], n_y=1, now=3.0)
    assert len(dropped) == 1
    assert early.n_measurements == 0

    late, dropped = build_window(np.arange(7.0), [record], n_y=1, now=6.0)
    assert not dropped
    assert late.n_measurements == 1


def test_repeat_readings_combine_by_inverse_variance():
    records = [
        Measurement(time=0.0, values=[1.0], sigma=[1.0]),
        Measurement(time=0.0, values=[2.0], sigma=[0.5]),
    ]
    window, _ = build_window([0.0, 1.0], records, n_y=1)
    # Exact Gaussian pooling: 1/s^2 = 1 + 4, mean = (1*1 + 4*2)/5.
    assert np.isclose(float(window.sigma[0, 0]), 1.0 / np.sqrt(5.0))
    assert np.isclose(float(window.y[0, 0]), 9.0 / 5.0)
    assert np.isinf(float(window.sigma[1, 0]))


def test_off_grid_and_malformed_records():
    good = Measurement(time=1.05, values=[1.0], sigma=[0.1])
    far = Measurement(time=17.0, values=[1.0], sigma=[0.1])
    unmeasured = Measurement(time=2.0, values=[1.0], sigma=[np.inf])
    window, dropped = build_window(np.arange(4.0), [good, far, unmeasured],
                                   n_y=1)
    assert dropped == [far, unmeasured]
    assert window.n_measurements == 1
    assert bool(np.asarray(window.mask)[1, 0])

    with pytest.raises(ValueError, match="not one of"):
        build_window(np.arange(4.0), [Measurement(0.0, {"nope": 1.0}, 0.1)],
                     y_names=["yes"])
    with pytest.raises(ValueError, match="but the model has"):
        build_window(np.arange(4.0), [Measurement(0.0, [1.0, 2.0], 0.1)],
                     n_y=1)
    with pytest.raises(ValueError, match="at least 2 points"):
        build_window([0.0], [], n_y=1)


def test_slice_window_takes_a_consistent_view():
    _, ys = simulate([[0.9]], [[1.0]], [1.0], 8, 0.02, 0.1, seed=0)
    window = dense_window(ys, 0.1, ["y"])
    sub = slice_window(window, 3, 4)

    assert sub.horizon == 4
    assert sub.times.shape == (5,) and sub.u.shape == (4, 0)
    assert np.allclose(np.asarray(sub.times), np.arange(3.0, 8.0))
    assert np.allclose(np.asarray(sub.y), np.asarray(window.y[3:8]))
    with pytest.raises(IndexError):
        slice_window(window, 6, 5)


def test_window_summary_reports_sampling_and_delay():
    records = [
        Measurement(time=0.0, values={"fast": 1.0}, sigma={"fast": 0.1}),
        Measurement(time=1.0, values={"fast": 1.1}, sigma={"fast": 0.1}),
        Measurement(time=1.0, values={"assay": 0.3}, sigma={"assay": 0.01},
                    reported=3.0),
    ]
    window, _ = build_window(np.arange(4.0), records,
                             y_names=["fast", "assay"])
    text = window.summary()
    assert "fast" in text and "assay" in text
    assert "delayed" in text and "max delay 2" in text
    assert window.n_measurements == 3


# =============================================================================
# Extended Kalman filter
# =============================================================================


def test_ekf_scalar_step_matches_the_closed_form():
    a, q, r = 0.8, 0.04, 0.09
    m0, p0 = 1.0, 0.25
    model = linear_model(jnp.array([[a]]), jnp.array([[1.0]]))
    y = 1.5

    x_up, p_up, innov = ekf_update(
        model, jnp.array([m0]), jnp.array([[p0]]), jnp.array([y]),
        jnp.array([np.sqrt(r)]), jnp.zeros(0),
    )
    gain = p0 / (p0 + r)
    assert np.isclose(float(innov[0]), y - m0)
    assert np.isclose(float(x_up[0]), m0 + gain * (y - m0))
    assert np.isclose(float(p_up[0, 0]), p0 * r / (p0 + r))

    x_pr, p_pr = ekf_predict(
        model, x_up, p_up, jnp.zeros(0), jnp.array([np.sqrt(q)])
    )
    assert np.isclose(float(x_pr[0]), a * float(x_up[0]))
    assert np.isclose(float(p_pr[0, 0]), a ** 2 * float(p_up[0, 0]) + q)


def test_ekf_treats_an_infinite_sigma_as_no_reading():
    model = linear_model(jnp.array([[0.9]]), jnp.array([[1.0]]))
    x = jnp.array([1.0])
    p = jnp.array([[0.25]])
    x_up, p_up, innov = ekf_update(
        model, x, p, jnp.array([99.0]), jnp.array([jnp.inf]), jnp.zeros(0)
    )
    assert np.allclose(np.asarray(x_up), np.asarray(x))
    assert np.allclose(np.asarray(p_up), np.asarray(p))
    assert float(innov[0]) == 0.0


def test_ekf_covariances_stay_symmetric_and_psd():
    xs, ys = simulate([[0.9, 0.1], [0.0, 0.85]], [[1.0, 0.0]], [1.0, -1.0],
                      12, 0.03, 0.2, seed=1)
    window = dense_window(ys, 0.2, ["y"])
    run = run_ekf(model_2d(), window, x0=jnp.zeros(2), P0=jnp.eye(2),
                  process_std=jnp.array([0.03, 0.03]))
    p = np.asarray(run.P)
    assert p.shape == (13, 2, 2)
    assert np.allclose(p, np.transpose(p, (0, 2, 1)), atol=1e-14)
    assert np.all(np.linalg.eigvalsh(p) > -1e-12)
    assert np.all(np.isfinite(np.asarray(run.std)))
    assert set(run.named().keys()) == {"a", "b"}


def model_2d():
    return linear_model(
        jnp.array([[0.9, 0.1], [0.0, 0.85]]), jnp.array([[1.0, 0.0]]),
        x_names=["a", "b"], y_names=["y"],
    )


# =============================================================================
# The load-bearing comparison: MHE == Kalman filter
# =============================================================================


def test_mhe_full_information_matches_kalman():
    """No constraints, linear model, whole record in one window.

    The MHE estimate of the last state is then exactly the Kalman
    filter's estimate at that time -- and so is its covariance. Any
    error in the whitening, the arrival term, the elimination of the
    dynamics or the Gauss-Newton covariance would show up here.
    """
    a = np.array([[0.9, 0.1], [0.0, 0.8]])
    c = np.array([[1.0, 0.0]])
    q_std = jnp.array([0.05, 0.05])
    _, ys = simulate(a, c, [1.0, -0.5], 12, np.asarray(q_std), 0.2, seed=0)
    window = dense_window(ys, 0.2, ["y"])
    model = linear_model(jnp.asarray(a), jnp.asarray(c),
                         x_names=["a", "b"], y_names=["y"])

    x0 = jnp.zeros(2)
    p0 = jnp.eye(2)
    kf = run_ekf(model, window, x0=x0, P0=p0, process_std=q_std)
    mhe = solve_mhe(
        MHEProblem(model, ArrivalCost(x_bar=x0, P=p0), process_std=q_std),
        window,
    )

    assert mhe.success
    assert np.allclose(np.asarray(mhe.x_final), np.asarray(kf.x[-1]),
                       rtol=0.0, atol=1e-9)
    assert np.allclose(np.asarray(mhe.covariance), np.asarray(kf.P[-1]),
                       rtol=0.0, atol=1e-9)
    # And the smoothed trajectory is *not* the filtered one, so the
    # agreement above is a property of the last state, not an accident
    # of the two being the same object.
    assert np.max(np.abs(np.asarray(mhe.x) - np.asarray(kf.x))) > 1e-3


def test_mhe_matches_kalman_with_multirate_gaps():
    """The ``sigma = inf`` convention has to mean the same in both."""
    a = np.array([[0.9, 0.15], [-0.1, 0.85]])
    c = np.eye(2)
    q_std = jnp.array([0.04, 0.03])
    _, ys = simulate(a, c, [0.5, 0.5], 10, np.asarray(q_std), [0.2, 0.5],
                     seed=5)
    records = []
    for k in range(11):
        records.append(Measurement(time=float(k), values={"ya": float(ys[k, 0])},
                                   sigma={"ya": 0.2}))
        if k % 4 == 0:  # a slow channel, on a shift cadence
            records.append(Measurement(time=float(k),
                                       values={"yb": float(ys[k, 1])},
                                       sigma={"yb": 0.5}))
    window, dropped = build_window(np.arange(11.0), records,
                                   y_names=["ya", "yb"])
    assert not dropped and window.n_measurements == 11 + 3

    model = linear_model(jnp.asarray(a), jnp.asarray(c), y_names=["ya", "yb"])
    x0 = jnp.array([0.2, -0.3])
    p0 = jnp.array([[0.5, 0.1], [0.1, 0.4]])
    kf = run_ekf(model, window, x0=x0, P0=p0, process_std=q_std)
    mhe = solve_mhe(
        MHEProblem(model, ArrivalCost(x_bar=x0, P=p0), process_std=q_std),
        window,
    )
    assert np.allclose(np.asarray(mhe.x_final), np.asarray(kf.x[-1]),
                       atol=1e-9)
    assert np.allclose(np.asarray(mhe.covariance), np.asarray(kf.P[-1]),
                       atol=1e-9)


def test_mhe_recovers_states_better_than_the_raw_measurements():
    a = np.array([[0.95, 0.05], [0.0, 0.9]])
    c = np.array([[1.0, 0.0]])
    q_std = np.array([0.02, 0.02])
    xs, ys = simulate(a, c, [2.0, 1.0], 30, q_std, 0.3, seed=11)
    window = dense_window(ys, 0.3, ["ya"])
    model = linear_model(jnp.asarray(a), jnp.asarray(c),
                         x_names=["a", "b"], y_names=["ya"])

    run = run_mhe(model, window, horizon=6, process_std=jnp.asarray(q_std),
                  x0=jnp.array([2.0, 1.0]), P0=jnp.eye(2) * 0.25)

    est = np.asarray(run.x)
    raw_rmse = np.sqrt(np.mean((ys[:, 0] - xs[:, 0]) ** 2))
    mhe_rmse = np.sqrt(np.mean((est[:, 0] - xs[:, 0]) ** 2))
    assert run.converged
    assert mhe_rmse < 0.5 * raw_rmse
    # The unmeasured state is recovered too, which no amount of
    # smoothing of the raw signal could do.
    assert np.sqrt(np.mean((est[:, 1] - xs[:, 1]) ** 2)) < 0.15

    assert run.source[:6] == ["ekf"] * 6
    assert set(run.source[6:]) == {"mhe"}
    assert est.shape == (31, 2)
    assert len(run.windows) == 31 - 6
    assert "moving-horizon solves" in run.summary()


def test_mhe_objective_is_a_chi_squared_statistic():
    """Whitened, so the optimal objective is chi-squared on M readings.

    Checked by sampling: 400 windows drawn from the very model and prior
    the estimator is given, solved through one traced core, must produce
    a mean objective equal to the number of readings and reject at
    roughly the nominal 5%. A wrong weight anywhere -- a missing
    ``1/sigma``, a covariance instead of a standard deviation, the
    arrival term counted twice -- moves the mean off ``dof`` at once.
    """
    from difflow.mhe.estimator import _initial_z, _make_core

    a, q_std, r_std, prior_std, prior_mean = 0.9, 0.05, 0.2, 0.3, 0.3
    k = 8
    dof = k + 1
    model = linear_model(jnp.array([[a]]), jnp.array([[1.0]]))
    problem = MHEProblem(
        model, ArrivalCost.diagonal(jnp.array([prior_mean]), prior_std),
        process_std=jnp.array([q_std]),
    )

    core = jax.jit(_make_core(problem, k))
    d = problem.scaling(k).d
    z0 = _initial_z(problem, k, d, None, None)
    u = jnp.zeros((k, 0))
    sigma = jnp.full((k + 1, 1), r_std)

    def objective(y):
        _, _, blocks, _, _, _ = core(
            problem.arrival.x_bar, problem.arrival.factor, y, sigma, u, d, z0
        )
        return jnp.sum(jnp.concatenate(blocks[:3]) ** 2)

    rng = np.random.default_rng(17)
    n_trials = 400
    draws = []
    for _ in range(n_trials):
        x = prior_mean + prior_std * rng.normal()
        row = []
        for _ in range(k + 1):
            row.append(x + r_std * rng.normal())
            x = a * x + q_std * rng.normal()
        draws.append(row)
    values = np.asarray(
        jax.vmap(objective)(jnp.asarray(np.array(draws)[:, :, None]))
    )

    standard_error = np.sqrt(2.0 * dof / n_trials)
    assert abs(values.mean() - dof) < 3.0 * standard_error
    critical = float(scipy_stats.chi2.ppf(0.95, dof))
    assert 0.0 < (values > critical).mean() < 0.12      # nominal 0.05

    # And the reporting layer reads it as the same statistic, with the
    # parts summing to the whole.
    window = dense_window(np.array(draws[0])[:, None], r_std, ["y"])
    clean = solve_mhe(problem, window)
    assert mhe_global_test(clean).dof == window.n_measurements == dof
    assert np.isclose(
        clean.objective,
        clean.arrival_objective + clean.process_objective
        + clean.measurement_objective,
    )

    spiked = solve_mhe(problem, replace(window, y=window.y.at[4, 0].add(5.0)))
    assert spiked.objective > clean.objective
    assert mhe_global_test(spiked).detected
    assert mhe_global_test(spiked).p_value < 1e-3


# =============================================================================
# Delayed measurements, end to end
# =============================================================================


def test_a_delayed_assay_constrains_the_state_it_was_drawn_from():
    """Placing a late assay at ``now`` instead is measurably wrong.

    The plant halves every interval, so the sample taken at ``t = 1`` and
    reported at ``t = 6`` says nothing directly about the state at
    ``t = 6``: it says the state five intervals ago was 2.0, hence the
    state now is ``2.0 / 2**5``. An implementation that stamped the
    reading with its arrival time would answer ``2.0``, a factor of 32
    out.
    """
    model = linear_model(jnp.array([[0.5]]), jnp.array([[1.0]]),
                         x_names=["c"], y_names=["assay"])
    times = np.arange(7.0)
    problem = MHEProblem(
        model, ArrivalCost.diagonal(jnp.array([3.0]), 5.0),
        process_std=jnp.array([1e-3]),
    )

    late = Measurement(time=1.0, values=[2.0], sigma=[0.01], reported=6.0)
    correct, _ = build_window(times, [late], y_names=["assay"], now=6.0)
    naive, _ = build_window(
        times, [Measurement(time=6.0, values=[2.0], sigma=[0.01])],
        y_names=["assay"],
    )

    right = solve_mhe(problem, correct)
    wrong = solve_mhe(problem, naive)

    assert np.isclose(float(right.x_final[0]), 2.0 * 0.5 ** 5, rtol=1e-3)
    assert np.isclose(float(wrong.x_final[0]), 2.0, rtol=5e-2)
    assert float(wrong.x_final[0]) > 20.0 * float(right.x_final[0])


# =============================================================================
# Constraints
# =============================================================================


def test_a_bound_keeps_a_concentration_physical():
    """The same data: unconstrained goes negative, bounded does not."""
    times = np.arange(5.0)
    values = [-0.15, -0.05, -0.12, -0.20, -0.08]
    records = [
        Measurement(time=float(k), values=[values[k]], sigma=[0.05])
        for k in range(5)
    ]
    window, _ = build_window(times, records, y_names=["yC"])

    free = linear_model(jnp.array([[1.0]]), jnp.array([[1.0]]),
                        x_names=["C"], y_names=["yC"])
    bounded = linear_model(jnp.array([[1.0]]), jnp.array([[1.0]]),
                           x_names=["C"], y_names=["yC"], lb=0.0)
    assert not free.is_bounded and bounded.is_bounded

    unconstrained = solve_mhe(
        MHEProblem(free, ArrivalCost.diagonal(jnp.array([0.0]), 10.0),
                   process_std=jnp.array([1e-3])),
        window,
    )
    constrained = solve_mhe(
        MHEProblem(bounded, ArrivalCost.diagonal(jnp.array([0.05]), 10.0),
                   process_std=jnp.array([1e-3]), constraint_weight=1e4),
        window,
    )

    assert float(unconstrained.x_final[0]) < -0.05
    assert unconstrained.max_violation == 0.0        # no bounds to violate
    assert float(constrained.x_final[0]) > -1e-3
    assert constrained.max_violation < 1e-3
    assert float(jnp.min(constrained.x)) > -1e-3


def test_the_first_state_can_never_leave_its_bounds():
    """x_0 is reparameterised, so no iterate violates its bounds."""
    model = linear_model(jnp.array([[1.0]]), jnp.array([[1.0]]),
                         lb=0.0, ub=1.0)
    records = [Measurement(time=float(k), values=[5.0], sigma=[0.01])
               for k in range(4)]
    window, _ = build_window(np.arange(4.0), records, n_y=1)
    res = solve_mhe(
        MHEProblem(model, ArrivalCost.diagonal(jnp.array([0.5]), 1.0),
                   process_std=jnp.array([1e-4])),
        window,
    )
    assert 0.0 <= float(res.x[0, 0]) <= 1.0


# =============================================================================
# jit / grad
# =============================================================================


def test_estimate_is_jittable_and_differentiable():
    model = linear_model(jnp.array([[0.9]]), jnp.array([[1.0]]))
    records = [Measurement(time=float(k), values=[1.0 + 0.1 * k],
                           sigma=[0.2]) for k in range(5)]
    window, _ = build_window(np.arange(5.0), records, n_y=1)
    problem = MHEProblem(
        model, ArrivalCost.diagonal(jnp.zeros(1), 1.0),
        process_std=jnp.array([0.05]),
    )

    def last_state(y):
        return estimate(problem, replace(window, y=y))[-1, 0]

    plain = float(last_state(window.y))
    jitted = float(jax.jit(last_state)(window.y))
    assert np.isclose(plain, jitted, rtol=0.0, atol=1e-12)

    grad = np.asarray(jax.grad(last_state)(window.y))
    assert grad.shape == (5, 1)
    assert np.all(np.isfinite(grad))
    assert np.all(grad > 0.0)          # more signal, more state

    # Implicit differentiation, checked against a central difference.
    #
    # The step is deliberately large. `last_state` is the output of an
    # iterative solve, so it carries that solve's own convergence noise --
    # here around 7e-11 -- and a central difference divides the noise by
    # 2*eps. At eps = 1e-6 that amplifies it into the fourth decimal place,
    # which is not truncation error and does not shrink with a smaller step:
    # the relative disagreement is non-monotonic in eps (1.6e-4 at 1e-8,
    # 8e-11 at 1e-6, 2.4e-6 at 1e-5 on one machine, 2.3e-4 at 1e-6 on
    # another). This is the interaction #196 warns about, where the
    # implicit-function gradient is exact on the solution manifold while the
    # code's own output is only converged to the inner tolerance.
    #
    # eps = 1e-3 sits where truncation is still negligible and the noise is
    # divided by a thousand times more; across 5e-4 to 5e-3 the worst
    # relative error measured is 2.9e-7, so rtol = 1e-4 keeps ~345x margin
    # and does not depend on the platform's linear algebra.
    eps = 1e-3
    fd = (float(last_state(window.y.at[2, 0].add(eps)))
          - float(last_state(window.y.at[2, 0].add(-eps)))) / (2 * eps)
    assert np.isclose(fd, grad[2, 0], rtol=1e-4)


# =============================================================================
# Joint state and parameter estimation
# =============================================================================


@pytest.fixture(scope="module")
def drift_run():
    """A first-order plant whose gain drifts, estimated jointly."""
    rng = np.random.default_rng(3)
    base = StateSpaceModel(
        f=lambda x, u, w, th: th["a"] * x + 0.1 + w,
        h=lambda x, u, th: x,
        n_x=1, n_y=1, x_names=["c"], y_names=["yc"],
    )
    model = augment_parameters(base, ["a"], lb=0.0, ub=1.0)

    n = 40
    a_true = 0.9 - 0.004 * np.arange(n + 1)
    x = 0.5
    ys = []
    for k in range(n + 1):
        ys.append(x + 0.02 * rng.normal())
        x = a_true[k] * x + 0.1
    records = [Measurement(time=float(k), values=[ys[k]], sigma=[0.02])
               for k in range(n + 1)]
    window, _ = build_window(np.arange(n + 1.0), records, y_names=["yc"])

    run = run_mhe(
        model, window, horizon=8, process_std=jnp.array([1e-4, 3e-3]),
        theta={"a": 0.0}, x0=jnp.array([0.5, 0.95]),
        P0=jnp.diag(jnp.array([0.05, 0.05]) ** 2),
    )
    return model, window, run, a_true


def test_joint_estimation_tracks_a_drifting_parameter(drift_run):
    _, _, run, a_true = drift_run
    est = np.asarray(run.parameters["a"])

    assert run.converged
    assert est.shape == a_true.shape
    # It starts from a prior of 0.95 and has to walk down to 0.74.
    prior_rmse = np.sqrt(np.mean((0.95 - a_true[8:]) ** 2))
    tracked_rmse = np.sqrt(np.mean((est[8:] - a_true[8:]) ** 2))
    assert tracked_rmse < 0.3 * prior_rmse
    assert abs(est[-1] - a_true[-1]) < 0.05
    # And it moves in the right direction, monotonically enough to be a
    # drift rather than a noise fit.
    assert est[-1] < est[8] - 0.1


def test_parameters_come_back_as_a_planning_ready_mapping(drift_run):
    model, _, run, a_true = drift_run
    last = run.windows[-1]

    assert last.param_names == ["a"]
    assert set(last.parameters) == {"a"}
    assert isinstance(last.parameters["a"], float)
    assert abs(last.parameters["a"] - a_true[-1]) < 0.05
    assert last.parameter_std["a"] > 0.0
    assert set(last.x_named) == {"c", "a"}
    assert "a" in last.summary()
    assert model.param_names == ["a"]


def test_an_unobservable_parameter_is_flagged_before_the_solve():
    """A state with no path to any sensor cannot be estimated."""
    decoupled = linear_model(
        jnp.array([[0.9, 0.0], [0.0, 0.8]]), jnp.array([[1.0, 0.0]]),
        x_names=["seen", "hidden"], y_names=["y"],
    )
    records = [Measurement(time=float(k), values=[1.0], sigma=[0.1])
               for k in range(5)]
    window, _ = build_window(np.arange(5.0), records, y_names=["y"])

    report = check_observability(decoupled, window, jnp.array([1.0, 1.0]))
    assert not report.observable
    assert report.rank == 1 and report.n_states == 2
    assert report.unobservable == ["hidden"]
    assert np.isinf(report.gramian_std[-1])
    assert "hidden" in report.summary()
    with pytest.raises(ValueError, match="hidden"):
        report.raise_if_unobservable()

    # Couple the two and the same window resolves both.
    coupled = linear_model(
        jnp.array([[0.9, 0.1], [0.0, 0.8]]), jnp.array([[1.0, 0.0]]),
        x_names=["seen", "hidden"], y_names=["y"],
    )
    ok = check_observability(coupled, window, jnp.array([1.0, 1.0]))
    assert ok.observable and ok.rank == 2
    ok.raise_if_unobservable()

    with pytest.raises(ValueError, match="names for"):
        check_observability(coupled, window, jnp.array([1.0, 1.0]),
                            names=["only-one"])


# =============================================================================
# Arrival cost
# =============================================================================


def test_arrival_cost_whitens_and_reports():
    arrival = ArrivalCost.diagonal(jnp.array([1.0, -2.0]), jnp.array([0.5, 2.0]))
    assert np.isclose(float(arrival.cost(arrival.x_bar)), 0.0)
    # One standard deviation away in each coordinate costs one each.
    assert np.isclose(float(arrival.cost(jnp.array([1.5, -2.0]))), 1.0)
    assert np.isclose(float(arrival.cost(jnp.array([1.0, 0.0]))), 1.0)
    assert np.allclose(np.asarray(arrival.std), [0.5, 2.0])
    assert np.isclose(arrival.condition, 16.0)

    correlated = ArrivalCost(x_bar=jnp.zeros(2),
                             P=jnp.array([[1.0, 0.8], [0.8, 1.0]]))
    r = correlated.whiten(jnp.array([1.0, 1.0]))
    assert np.isclose(float(jnp.dot(r, r)), float(correlated.cost(
        jnp.array([1.0, 1.0]))))
    expected = np.asarray([1.0, 1.0]) @ np.linalg.solve(
        np.asarray(correlated.P), np.asarray([1.0, 1.0])
    )
    assert np.isclose(float(correlated.cost(jnp.array([1.0, 1.0]))), expected)

    loose = arrival.inflate(4.0)
    assert np.allclose(np.asarray(loose.std), 2.0 * np.asarray(arrival.std))
    assert np.isclose(float(loose.cost(jnp.array([1.5, -2.0]))), 0.25)
    with pytest.raises(ValueError, match="positive"):
        arrival.inflate(0.0)

    assert float(ArrivalCost.vague(jnp.zeros(1)).std[0]) == 1e6
    assert "arrival cost" in arrival.summary(["a", "b"])


def test_advancing_the_arrival_cost_reproduces_the_filter():
    """On a linear model the smoothing update *is* the EKF recursion."""
    a = np.array([[0.9, 0.1], [0.0, 0.8]])
    c = np.array([[1.0, 0.0]])
    q_std = jnp.array([0.05, 0.05])
    _, ys = simulate(a, c, [1.0, 0.0], 6, np.asarray(q_std), 0.2, seed=2)
    window = dense_window(ys, 0.2, ["y"])
    model = linear_model(jnp.asarray(a), jnp.asarray(c))

    x0, p0 = jnp.zeros(2), jnp.eye(2)
    arrival = ArrivalCost(x_bar=x0, P=p0, time=0.0)
    problem = MHEProblem(model, arrival, process_std=q_std)
    solved = solve_mhe(problem, window)

    nxt = advance_arrival_cost(model, arrival, window, solved.x,
                               process_std=q_std)
    kf = run_ekf(model, window, x0=x0, P0=p0, process_std=q_std)

    assert nxt.time == 1.0
    # The covariance is the filter's one-step-ahead covariance at t=1,
    # which is the prior for the next window.
    f_x, _ = model.jacobians(solved.x[0], window.u[0], jnp.zeros(2))
    expected = (np.asarray(f_x) @ np.asarray(kf.P[0]) @ np.asarray(f_x).T
                + np.diag(np.asarray(q_std) ** 2))
    assert np.allclose(np.asarray(nxt.P), expected, atol=1e-9)
    assert np.allclose(np.asarray(nxt.x_bar), np.asarray(solved.x[1]))


# =============================================================================
# Validation
# =============================================================================


def test_problem_rejects_a_non_positive_process_std():
    model = linear_model(jnp.eye(1), jnp.eye(1))
    arrival = ArrivalCost.diagonal(jnp.zeros(1), 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        MHEProblem(model, arrival, process_std=jnp.array([0.0]))


def test_run_mhe_rejects_a_horizon_that_does_not_fit():
    _, ys = simulate([[0.9]], [[1.0]], [1.0], 4, 0.02, 0.1, seed=0)
    window = dense_window(ys, 0.1, ["y"])
    model = linear_model(jnp.array([[0.9]]), jnp.array([[1.0]]))
    with pytest.raises(ValueError, match="does not fit"):
        run_mhe(model, window, horizon=99, process_std=jnp.array([0.02]),
                x0=jnp.zeros(1), P0=jnp.eye(1))


def test_window_and_result_are_params_like():
    """Everything user-facing follows the ParamsMixin convention."""
    _, ys = simulate([[0.9]], [[1.0]], [1.0], 3, 0.02, 0.1, seed=0)
    window = dense_window(ys, 0.1, ["y"])
    assert isinstance(window, MeasurementWindow)
    assert "y" in window
    assert np.allclose(np.asarray(window["y"]), np.asarray(window.y))
    assert set(["times", "y", "sigma", "u"]).issubset(set(window.keys()))
