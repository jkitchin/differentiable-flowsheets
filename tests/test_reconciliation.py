"""Tests for difflow.reconciliation: smoke, API contracts and input validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.reconciliation import (
    MONITOR_INSTRUMENT_FAULT,
    MONITOR_MODEL_DRIFT,
    ReconciliationStructureError,
    blame_concentration,
    auto_scaling,
    global_test,
    kkt_matrix,
    measurement_test,
    monitor,
    reconcile,
    reconcile_multi,
    sensor_value,
    serial_elimination,
    solve_reconciliation,
)

LEAK_NAMES = ["feed", "top", "bottom", "leak"]


def one_balance(x, params=None):
    return jnp.array([x[0] + x[1] - 10.0])


def leak_balance(x, params=None):
    return jnp.array([x[0] - x[1] - x[2] - x[3]])


def three_terms(x, params=None):
    return jnp.array([x[0] + x[1] + x[2] - 10.0])


@pytest.fixture(scope="module")
def leak_campaign():
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(3, 4)) * np.array([2.0, 1.0, 1.0, 0.0])
    ys = [jnp.array([100.0, 60.0, 35.0, 5.0]) + jnp.asarray(n) for n in noise]
    return ys, jnp.array([2.0, 1.0, 1.0, jnp.inf])


def test_reconcile_smoke_and_tests():
    res = reconcile(three_terms, jnp.array([6.0, 2.0, 3.0]), jnp.array([1.0, 0.5, 1.0]),
                    names=["a", "b", "c"])
    assert res.converged
    assert np.all(np.isfinite(np.asarray(res.x)))
    assert np.all(np.isfinite(np.diag(np.asarray(res.covariance))))
    assert abs(float(three_terms(res.x)[0])) < 1e-8
    assert np.isfinite(global_test(res).statistic)
    assert set(measurement_test(res).z) == {"a", "b", "c"}


def test_nan_at_unmeasured_entry_does_not_leak():
    res = reconcile(one_balance, jnp.array([6.0, jnp.nan]), jnp.array([1.0, jnp.inf]))
    assert bool(jnp.all(jnp.isfinite(res.x)))


def test_unobservable_raises():
    with pytest.raises(ReconciliationStructureError, match="Unobservable"):
        reconcile(three_terms, jnp.array([6.0, jnp.nan, jnp.nan]),
                  jnp.array([1.0, jnp.inf, jnp.inf]), names=["a", "b", "c"])


def test_structure_and_kkt_smoke():
    res = reconcile(three_terms, jnp.array([6.0, 2.0, jnp.nan]), jnp.array([1.0, 0.5, jnp.inf]),
                    names=["a", "b", "c"])
    assert res.structure.solvable and res.structure.degree_of_redundancy == 0
    assert kkt_matrix(jnp.array([[1.0, 2.0, 3.0]]), jnp.array([1.0, 0.0, 4.0])).shape == (4, 4)


def test_grad_wrt_measurement_finite():
    y0, sigma = jnp.array([6.0, 6.0]), jnp.array([1.0, 2.0])
    sc = auto_scaling(one_balance, y0, sigma)
    g = jax.grad(lambda y: solve_reconciliation(one_balance, y, sigma, x0=y0, scaling=sc,
                                                n_steps=2)[0][0])(y0)
    assert np.all(np.isfinite(np.asarray(g)))


def test_serial_elimination_and_sensor_value_smoke():
    sigma = jnp.array([1.0, 1.0, 1.0])
    steps = serial_elimination(three_terms, jnp.array([6.0, 2.0, 30.0]), sigma)
    assert len(steps) >= 1
    out = sensor_value(three_terms, jnp.array([6.0, 2.0, 3.0]), jnp.array([1.0, 1.0, jnp.inf]),
                       target=2, candidate=2, candidate_sigma=1.0)
    assert np.isfinite(out["sd_after"])


def test_reconcile_multi_smoke_and_validation(leak_campaign):
    ys, sigma = leak_campaign
    res = reconcile_multi(leak_balance, ys, sigma, shared=["leak"], names=LEAK_NAMES)
    assert np.isfinite(res.shared["leak"]) and np.isfinite(res.shared_std["leak"])
    with pytest.raises(ValueError, match="at least one data set"):
        reconcile_multi(leak_balance, [], sigma, shared=["leak"], names=LEAK_NAMES)
    with pytest.raises(ValueError, match="expected"):
        reconcile_multi(leak_balance, [ys[0], ys[1][:3]], sigma, shared=["leak"], names=LEAK_NAMES)
    with pytest.raises(KeyError, match="not a variable"):
        reconcile_multi(leak_balance, ys, sigma, shared=["nope"], names=LEAK_NAMES)
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_multi(leak_balance, ys, sigma, shared=["leak", "leak"], names=LEAK_NAMES)
    with pytest.raises(ValueError, match="different sigmas"):
        reconcile_multi(leak_balance, ys, jnp.stack([sigma] * 2 + [sigma.at[3].set(0.5)]),
                        shared=["leak"], names=LEAK_NAMES)


def test_monitor_smoke(leak_campaign):
    ys, sigma = leak_campaign
    mon = monitor(leak_balance, ys, sigma.at[3].set(1.0), names=LEAK_NAMES)
    assert mon.statistic.shape == (len(ys),)
    assert mon.diagnose(window=None).verdict
    with pytest.raises(ValueError, match="window must be positive"):
        blame_concentration(["a"], window=0)


def test_monitor_records_an_unsolvable_problem(leak_campaign):
    ys, sigma = leak_campaign
    mon = monitor(leak_balance, ys, sigma.at[2].set(jnp.inf), names=LEAK_NAMES)
    assert all(s.failed for s in mon.steps)
    assert "failed" in mon.summary()


def test_wandering_suspect_verdict():
    from difflow.reconciliation.monitoring import MonitorResult, MonitorStep

    steps = [MonitorStep(index=i, statistic=99.0, dof=3, critical=7.8, p_value=0.0,
                         detected=True, suspect=s, z_max=5.0)
             for i, s in enumerate(["a", "b", "c", "a", "d", "b", "c", "d"])]
    assert MonitorResult(steps=steps).diagnose(window=None).verdict == MONITOR_MODEL_DRIFT


def test_reconcile_multi_reuses_the_compiled_solve(leak_campaign):
    # Regression (#516): a fresh stacked closure per call recompiled the solve
    # for every new window of data.
    ys, sigma = leak_campaign
    reconcile_multi(leak_balance, ys, sigma, shared=["leak"], names=LEAK_NAMES)
    size = solve_reconciliation._cache_size()
    shifted = [y + 0.5 for y in ys]
    reconcile_multi(leak_balance, shifted, sigma, shared=["leak"], names=LEAK_NAMES)
    assert solve_reconciliation._cache_size() == size


def split_flow(x, params):
    """A parameter-free balance and a rate equation carrying ``k``."""
    a, b, c = x
    return jnp.array([a - b - c, b - params["k"] * a])


def test_diagnose_uses_the_parameter_free_constraints():
    # Regression (#516): with two equations, drift of k moved the blame onto one
    # meter every day and was read as an instrument fault.
    sigma = jnp.array([1.0, 1.0, 1.0])
    drift = [jnp.array([100.0, 60.0, 40.0])] * 4             # k = 0.6, model says 0.5
    mon = monitor(split_flow, drift, sigma, params={"k": 0.5}, names=["a", "b", "c"])
    assert mon.diagnose(window=None).verdict == MONITOR_MODEL_DRIFT

    biased = [jnp.array([100.0, 50.0, 58.0])] * 4            # k right, c reads 8 high
    mon = monitor(split_flow, biased, sigma, params={"k": 0.5}, names=["a", "b", "c"])
    diag = mon.diagnose(window=None)
    assert diag.verdict == MONITOR_INSTRUMENT_FAULT and diag.culprit == "c"
