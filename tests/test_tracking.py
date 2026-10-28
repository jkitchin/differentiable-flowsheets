"""Tests for difflow.reconciliation.tracking."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.reconciliation import (
    MONITOR_CONSISTENT,
    MONITOR_INSTRUMENT_FAULT,
    MONITOR_MODEL_DRIFT,
    MONITOR_UNDIAGNOSED,
    MonitorDiagnosis,
    ReconciliationStructureError,
    TrackerState,
    drift_std_from_time_constant,
    measurement_update,
    parameter_measurement,
    time_update,
    track_parameters,
    update_gate,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================

K1 = K2 = 1.0
NAMES = ["q0", "q1", "q2", "dp1", "dp2"]
SIGMA = jnp.array([0.5, 0.5, 0.5, 25.0, 25.0])
DRIFT = drift_std_from_time_constant(0.05, 30.0)


def parallel_pipes(x, params):
    """Two pipes in parallel; pipe 1 carries a fouling factor ``eta``.

    Five variables and four equations, so one degree of redundancy with
    ``eta`` frozen --- and ``eta`` is recoverable from ``dp1`` and
    ``q1`` alone, so a single period identifies it.
    """
    eta = params["eta"]
    q0, q1, q2, dp1, dp2 = x
    return jnp.array([
        q0 - q1 - q2,
        dp1 - eta * K1 * q1 ** 2,
        dp2 - K2 * q2 ** 2,
        dp1 - dp2,
    ])


def truth(eta, q0=100.0):
    """The state the plant is actually in at a given fouling factor."""
    q1 = q0 / (1.0 + np.sqrt(eta * K1 / K2))
    q2 = q0 - q1
    dp = eta * K1 * q1 ** 2
    return np.array([q0, q1, q2, dp, dp])


def campaign(etas, seed=0, bias=None):
    """One noisy measurement vector per period."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(SIGMA, dtype=float)
    out = []
    for e in etas:
        y = truth(e) + rng.normal(0.0, sig)
        if bias is not None:
            y = y + np.asarray(bias, dtype=float)
        out.append(jnp.asarray(y))
    return out


def start(std=0.02, mean=1.0):
    return TrackerState.initial(["eta"], [mean], std=[std])


def test_tracker_state_validation():
    with pytest.raises(ValueError, match="exactly one"):
        TrackerState.initial(["a"], [1.0])
    with pytest.raises(ValueError, match="2 names"):
        TrackerState.initial(["a", "b"], [1.0], std=[0.1])
    st = TrackerState.initial(["eta", "k"], [0.9, 2.0], std=0.1)
    assert st.as_params() == pytest.approx({"eta": 0.9, "k": 2.0})
    assert "eta" in st.summary()


def test_time_update_and_validation():
    out = time_update(start(), 1.0, DRIFT)
    assert float(out.covariance[0, 0]) > 0.02 ** 2
    with pytest.raises(ValueError, match="dt"):
        time_update(start(), -1.0, DRIFT)
    with pytest.raises(ValueError, match="tau"):
        drift_std_from_time_constant(0.05, 0.0)


def test_measurement_update_and_validation():
    out, inn = measurement_update(start(), [0.9], jnp.array([[1e-4]]))
    assert np.isfinite(float(out.mean[0])) and inn.nis >= 0.0
    with pytest.raises(ValueError, match="estimate has shape"):
        measurement_update(start(), [1.0, 2.0], jnp.eye(1))


def test_a_parameter_the_period_cannot_see_raises():
    def blind(x, params):
        return jnp.array([x[0] - x[1] - x[2]])

    with pytest.raises(ReconciliationStructureError):
        parameter_measurement(blind, jnp.array([100.0, 60.0, 41.0]),
                              jnp.array([1.0, 1.0, 1.0]), start(), names=["a", "b", "c"])


def test_only_model_drift_opens_the_gate():
    def diag(v):
        return MonitorDiagnosis(verdict=v, culprit=None, rejection_rate=0.8,
                                blame_concentration=0.7, window=10)

    assert update_gate(diag(MONITOR_MODEL_DRIFT)).allowed
    for v in (MONITOR_INSTRUMENT_FAULT, MONITOR_CONSISTENT, MONITOR_UNDIAGNOSED):
        assert not update_gate(diag(v)).allowed


def test_track_parameters_smoke():
    data = campaign(np.ones(4), seed=1)
    run = track_parameters(parallel_pipes, data, SIGMA, state=start(), names=NAMES,
                           drift_std=DRIFT, window=2,
                           allow=[MONITOR_CONSISTENT, MONITOR_MODEL_DRIFT,
                                  MONITOR_INSTRUMENT_FAULT, MONITOR_UNDIAGNOSED])
    assert run.n_updates > 0
    assert len(run) == 4 and run.trajectory.shape == (4, 1)
    assert np.all(np.isfinite(np.asarray(run.trajectory)))
    with pytest.raises(ValueError, match="times for"):
        track_parameters(parallel_pipes, data, SIGMA, state=start(), names=NAMES,
                         drift_std=DRIFT, times=[0.0, 1.0])


def test_a_new_meter_fault_does_not_move_the_parameter():
    # Regression (#516): while a fault was young, blame concentration over the
    # window was low, the verdict read "model drift" and the bias went into eta.
    data = campaign(np.ones(8), seed=2, bias=np.zeros(5))
    data = data[:4] + [y.at[2].add(8.0) for y in data[4:]]   # q2 reads 8 high from period 4
    run = track_parameters(parallel_pipes, data, SIGMA, state=start(), names=NAMES,
                           drift_std=DRIFT, window=4)
    assert run.n_updates == 0
