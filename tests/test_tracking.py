"""Tests for difflow.reconciliation.tracking.

Two things are being asserted, and they pull in opposite directions.

The *filter* has to move: a genuinely drifting parameter must be
tracked, and the load-bearing checks of the update law are the linear
ones, where a Kalman update is the exact combination of two Gaussians
and can be compared with the closed form rather than a tolerance.

The *gate* has to refuse to move: fed the data of a biased meter, the
loop must leave the parameter alone.
``TestGate::test_an_ungated_loop_would_have_invented_a_fouling_factor``
is the regression, and it is the reason the module exists --- section 6
of ``examples/29_model_updating.ipynb`` shows a free parameter
manufacturing a confident fouling estimate out of a calibration error,
with the global test *improving* while it happens.
"""

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
    UPDATE_WHEN_DRIFTING,
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


#: Periods per synthetic campaign. Long enough that a ten-period
#: diagnosis window sees a drift build, short enough to test with.
N_DAYS = 40


@pytest.fixture(scope="module")
def fouling():
    """A campaign over which eta falls from 1.00 to 0.70."""
    etas = 1.0 - 0.30 * np.arange(N_DAYS) / (N_DAYS - 1)
    return etas, campaign(etas, seed=1)


@pytest.fixture(scope="module")
def biased_dp1():
    """A clean pipe, read through a dp1 meter 8% high."""
    return campaign(np.ones(N_DAYS), seed=2,
                    bias=[0.0, 0.0, 0.0, 0.08 * 2500.0, 0.0])


# A campaign is ~40 reconciliations, each re-traced, so the runs the
# assertions share are built once for the module rather than per test.


@pytest.fixture(scope="module")
def tracked_run(fouling):
    """The loop under its default policy."""
    _, data = fouling
    return track_parameters(
        parallel_pipes, data, SIGMA, state=start(), names=NAMES,
        drift_std=DRIFT, window=10,
    )


@pytest.fixture(scope="module")
def frozen_run(fouling):
    """The same campaign with the gate wired shut."""
    _, data = fouling
    return track_parameters(
        parallel_pipes, data, SIGMA, state=start(), names=NAMES,
        drift_std=DRIFT, window=10, allow=[],
    )


@pytest.fixture(scope="module")
def biased_run(biased_dp1):
    """The biased-meter campaign under the default policy."""
    return track_parameters(
        parallel_pipes, biased_dp1, SIGMA, state=start(), names=NAMES,
        drift_std=DRIFT, window=10,
    )


def start(std=0.02, mean=1.0):
    return TrackerState.initial(["eta"], [mean], std=[std])


def diagnosis(verdict, culprit=None):
    return MonitorDiagnosis(verdict=verdict, culprit=culprit,
                            rejection_rate=0.8, blame_concentration=0.7,
                            window=10)


# =============================================================================
# The state
# =============================================================================


class TestTrackerState:

    def test_std_or_covariance_but_not_both(self):
        with pytest.raises(ValueError, match="exactly one"):
            TrackerState.initial(["a"], [1.0], std=[0.1],
                                 covariance=jnp.eye(1))
        with pytest.raises(ValueError, match="exactly one"):
            TrackerState.initial(["a"], [1.0])

    def test_names_must_match_the_mean(self):
        with pytest.raises(ValueError, match="2 names"):
            TrackerState.initial(["a", "b"], [1.0], std=[0.1])

    def test_a_covariance_prior_keeps_its_correlations(self):
        cov = jnp.array([[4.0, 1.8], [1.8, 1.0]])
        st = TrackerState.initial(["a", "b"], [1.0, 2.0], covariance=cov)
        assert np.allclose(np.asarray(st.covariance), np.asarray(cov))

    def test_as_params_is_the_shape_a_model_takes(self):
        st = TrackerState.initial(["eta", "k"], [0.9, 2.0], std=0.1)
        assert st.as_params() == pytest.approx({"eta": 0.9, "k": 2.0})
        assert set(st.std) == {"eta", "k"}

    def test_summary_names_every_parameter(self):
        st = TrackerState.initial(["eta", "k"], [0.9, 2.0], std=0.1)
        assert "eta" in st.summary() and "k" in st.summary()


# =============================================================================
# The time update
# =============================================================================


class TestTimeUpdate:

    def test_the_mean_does_not_move(self):
        st = start()
        assert float(time_update(st, 1.0, DRIFT).mean[0]) == float(st.mean[0])

    def test_variance_grows_by_q_dt(self):
        st = start(std=0.02)
        out = time_update(st, 3.0, [0.01])
        assert float(out.covariance[0, 0]) == pytest.approx(
            0.02 ** 2 + 0.01 ** 2 * 3.0
        )

    def test_growth_is_additive_over_split_steps(self):
        st = start()
        once = time_update(st, 4.0, [0.01])
        twice = time_update(time_update(st, 1.0, [0.01]), 3.0, [0.01])
        assert float(once.covariance[0, 0]) == pytest.approx(
            float(twice.covariance[0, 0])
        )
        assert once.time == pytest.approx(twice.time)

    def test_a_held_parameter_widens_its_error_bar(self, biased_run):
        """Holding is not knowing. The twin must say so."""
        assert biased_run.n_updates == 0
        assert biased_run.std_of("eta")[-1] > biased_run.std_of("eta")[0]

    def test_max_std_caps_the_growth_and_keeps_correlation(self):
        cov = jnp.array([[4.0, 1.8], [1.8, 1.0]])
        st = TrackerState.initial(["a", "b"], [1.0, 2.0], covariance=cov)
        out = time_update(st, 100.0, [1.0, 1.0], max_std=[2.0, 1.0])
        std = np.sqrt(np.diag(np.asarray(out.covariance)))
        assert std == pytest.approx([2.0, 1.0])

        def corr(m):
            m = np.asarray(m)
            d = np.sqrt(np.diag(m))
            return m[0, 1] / (d[0] * d[1])

        # capping is a diagonal congruence, so it rescales but does not
        # decorrelate -- and the result is still a covariance.
        assert corr(out.covariance) == pytest.approx(
            corr(time_update(st, 100.0, [1.0, 1.0]).covariance)
        )
        assert np.all(np.linalg.eigvalsh(np.asarray(out.covariance)) >= -1e-12)

    def test_negative_dt_and_negative_drift_are_refused(self):
        with pytest.raises(ValueError, match="dt"):
            time_update(start(), -1.0, DRIFT)
        with pytest.raises(ValueError, match="drift_std"):
            time_update(start(), 1.0, [-0.1])


class TestDriftFromTimeConstant:

    def test_a_random_walk_reaches_the_stated_spread(self):
        q = drift_std_from_time_constant(0.05, 30.0)
        assert float(q) * np.sqrt(30.0) == pytest.approx(0.05)

    def test_it_is_elementwise(self):
        q = drift_std_from_time_constant([0.05, 0.2], 4.0)
        assert np.asarray(q) == pytest.approx([0.025, 0.1])

    def test_tau_must_be_positive(self):
        with pytest.raises(ValueError, match="tau"):
            drift_std_from_time_constant(0.05, 0.0)


# =============================================================================
# The measurement update
# =============================================================================


class TestMeasurementUpdate:

    def test_scalar_case_is_the_exact_two_gaussian_combination(self):
        st = start(std=0.02, mean=1.0)
        p, r, z = 0.02 ** 2, 0.01 ** 2, 0.90
        out, _ = measurement_update(st, [z], jnp.array([[r]]))
        # posterior mean = (z/r + mu/p) / (1/r + 1/p)
        assert float(out.mean[0]) == pytest.approx(
            (z / r + 1.0 / p) / (1 / r + 1 / p)
        )
        assert float(out.covariance[0, 0]) == pytest.approx(
            1.0 / (1 / r + 1 / p)
        )

    def test_a_weakly_informative_period_does_not_move_the_estimate(self):
        """No special case needed: a large R sends the gain to zero."""
        st = start(std=0.02)
        out, inn = measurement_update(st, [0.5], jnp.array([[1e8]]))
        assert float(out.mean[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(inn.gain[0, 0]) < 1e-8

    def test_a_sharp_measurement_nearly_replaces_the_prior(self):
        st = start(std=1.0)
        out, _ = measurement_update(st, [0.5], jnp.array([[1e-10]]))
        assert float(out.mean[0]) == pytest.approx(0.5, abs=1e-6)

    def test_the_posterior_is_tighter_than_either_input(self):
        st = start(std=0.02)
        out, _ = measurement_update(st, [0.9], jnp.array([[0.01 ** 2]]))
        assert float(out.covariance[0, 0]) < min(0.02 ** 2, 0.01 ** 2)

    def test_covariance_stays_symmetric_and_psd_over_a_long_run(self):
        """What the Joseph form is for."""
        rng = np.random.default_rng(0)
        cov = jnp.array([[4e-4, 1.9e-4], [1.9e-4, 1e-4]])
        st = TrackerState.initial(["a", "b"], [1.0, 2.0], covariance=cov)
        r = jnp.array([[1e-4, 0.9e-4], [0.9e-4, 1e-4]])
        for _ in range(300):
            st = time_update(st, 1.0, [1e-3, 1e-3])
            z = np.asarray(st.mean) + rng.normal(0.0, 1e-2, size=2)
            st, _ = measurement_update(st, z, r)
            m = np.asarray(st.covariance)
            assert np.allclose(m, m.T, atol=1e-18)
            assert np.linalg.eigvalsh(m).min() >= -1e-18

    def test_correlated_parameters_move_together(self):
        """A diagonal prior would leave b alone; a full one must not."""
        cov = jnp.array([[1e-2, 0.9e-2], [0.9e-2, 1e-2]])
        st = TrackerState.initial(["a", "b"], [1.0, 1.0], covariance=cov)
        # a sharp reading on a only
        r = jnp.array([[1e-6, 0.0], [0.0, 1e6]])
        out, _ = measurement_update(st, [1.5, 1.0], r)
        assert float(out.mean[0]) == pytest.approx(1.5, abs=1e-3)
        assert float(out.mean[1]) > 1.3

    def test_nis_is_in_scale_when_the_noise_model_is_right(self):
        rng = np.random.default_rng(3)
        nis = []
        for _ in range(400):
            st = start(std=0.02)
            z = 1.0 + rng.normal(0.0, np.sqrt(0.02 ** 2 + 0.01 ** 2))
            _, inn = measurement_update(st, [z], jnp.array([[0.01 ** 2]]))
            nis.append(inn.nis)
        # chi-squared on one degree of freedom has mean 1
        assert np.mean(nis) == pytest.approx(1.0, abs=0.25)

    def test_shape_mismatches_are_refused(self):
        st = start()
        with pytest.raises(ValueError, match="estimate has shape"):
            measurement_update(st, [1.0, 2.0], jnp.eye(1))
        with pytest.raises(ValueError, match="covariance has shape"):
            measurement_update(st, [1.0], jnp.eye(2))


# =============================================================================
# One period's estimate
# =============================================================================


class TestParameterMeasurement:

    def test_it_recovers_a_known_fouling_factor(self):
        y = jnp.asarray(truth(0.8))
        theta, cov, res = parameter_measurement(
            parallel_pipes, y, SIGMA, start(), names=NAMES,
        )
        assert float(theta[0]) == pytest.approx(0.8, abs=1e-6)
        assert cov.shape == (1, 1)
        assert float(cov[0, 0]) > 0.0
        assert res.converged

    def test_the_prior_does_not_leak_into_the_estimate(self):
        """sigma = inf on the parameter: free, not regularised."""
        y = jnp.asarray(truth(0.8))
        a, _, _ = parameter_measurement(parallel_pipes, y, SIGMA,
                                        start(mean=1.0), names=NAMES)
        b, _, _ = parameter_measurement(parallel_pipes, y, SIGMA,
                                        start(mean=0.5), names=NAMES)
        assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-6)

    def test_a_parameter_the_period_cannot_see_raises(self):
        def blind(x, params):
            del params                      # eta does not enter at all
            return jnp.array([x[0] - x[1] - x[2]])

        with pytest.raises(ReconciliationStructureError):
            parameter_measurement(
                blind, jnp.array([100.0, 60.0, 41.0]),
                jnp.array([1.0, 1.0, 1.0]), start(), names=["a", "b", "c"],
            )

    def test_a_name_clash_with_a_plant_variable_is_refused(self):
        with pytest.raises(ValueError, match="already name plant variables"):
            parameter_measurement(
                parallel_pipes, jnp.asarray(truth(1.0)), SIGMA,
                TrackerState.initial(["q1"], [1.0], std=[0.1]), names=NAMES,
            )

    def test_extra_params_are_carried_alongside(self):
        def with_extra(x, params):
            assert set(params) == {"eta", "scale"}
            return parallel_pipes(x, params)

        theta, _, _ = parameter_measurement(
            with_extra, jnp.asarray(truth(0.8)), SIGMA, start(),
            names=NAMES, params={"scale": 1.0},
        )
        assert float(theta[0]) == pytest.approx(0.8, abs=1e-6)

    def test_a_caller_supplied_scale_describes_the_plant_block_only(self):
        """The layouts that fill ``unmeasured_scale`` in know nothing of
        the tracked parameters, so it arrives one block short."""
        theta, _, _ = parameter_measurement(
            parallel_pipes, jnp.asarray(truth(0.8)), SIGMA, start(),
            names=NAMES, unmeasured_scale=jnp.full(len(NAMES), 50.0),
        )
        assert float(theta[0]) == pytest.approx(0.8, abs=1e-6)

    def test_it_scales_a_small_parameter_properly(self):
        """auto_scaling floors an unmeasured scale at 1.0; a parameter
        of order 1e-6 needs its own prior spread instead."""
        def tiny(x, params):
            q0, q1, q2, dp1, dp2 = x
            return jnp.array([
                q0 - q1 - q2,
                dp1 - params["k"] * 1e6 * K1 * q1 ** 2,
                dp2 - K2 * q2 ** 2,
                dp1 - dp2,
            ])

        y = jnp.asarray(truth(0.8))
        st = TrackerState.initial(["k"], [1e-6], std=[2e-8])
        theta, _, _ = parameter_measurement(tiny, y, SIGMA, st, names=NAMES)
        assert float(theta[0]) == pytest.approx(0.8e-6, rel=1e-5)


# =============================================================================
# The gate
# =============================================================================


class TestGate:

    @pytest.mark.parametrize("verdict, allowed", [
        (MONITOR_MODEL_DRIFT, True),
        (MONITOR_INSTRUMENT_FAULT, False),
        (MONITOR_CONSISTENT, False),
        (MONITOR_UNDIAGNOSED, False),
    ])
    def test_only_model_drift_opens_it(self, verdict, allowed):
        assert update_gate(diagnosis(verdict)).allowed is allowed

    def test_every_decision_says_why(self):
        for v in (MONITOR_MODEL_DRIFT, MONITOR_INSTRUMENT_FAULT,
                  MONITOR_CONSISTENT, MONITOR_UNDIAGNOSED):
            assert update_gate(diagnosis(v)).reason

    def test_widening_allow_is_possible_and_deliberate(self):
        d = diagnosis(MONITOR_CONSISTENT)
        assert not update_gate(d).allowed
        assert update_gate(d, allow=[MONITOR_CONSISTENT]).allowed

    def test_a_biased_meter_never_moves_the_parameter(self, biased_run):
        verdicts = {s.diagnosis.verdict for s in biased_run.steps}
        assert verdicts <= {MONITOR_INSTRUMENT_FAULT, MONITOR_CONSISTENT}
        assert biased_run.n_updates == 0
        assert biased_run.final.as_params()["eta"] == pytest.approx(1.0)

    def test_an_ungated_loop_would_have_invented_a_fouling_factor(
        self, biased_dp1
    ):
        """The regression this module exists for.

        Same data, same filter, gate opened: the loop reports a pipe
        several percent off clean, from a meter fault. Section 6 of
        ``examples/29_model_updating.ipynb``, run automatically.
        """
        ungated = track_parameters(
            parallel_pipes, biased_dp1, SIGMA, state=start(), names=NAMES,
            drift_std=DRIFT, window=10,
            allow=[MONITOR_MODEL_DRIFT, MONITOR_INSTRUMENT_FAULT,
                   MONITOR_CONSISTENT, MONITOR_UNDIAGNOSED],
        )
        assert ungated.n_updates > 0
        assert abs(ungated.final.as_params()["eta"] - 1.0) > 0.02


# =============================================================================
# The loop
# =============================================================================


class TestTrackParameters:

    def test_it_tracks_a_real_drift(self, fouling, tracked_run):
        etas, _ = fouling
        assert tracked_run.final.as_params()["eta"] == pytest.approx(
            etas[-1], abs=0.03
        )
        # a deadband, not a free fit
        assert 0 < tracked_run.n_updates < len(tracked_run)

    def test_a_frozen_model_would_have_stayed_at_the_start(self, frozen_run):
        assert frozen_run.final.as_params()["eta"] == pytest.approx(1.0)
        assert frozen_run.n_updates == 0

    def test_the_loop_closes_on_the_statistic(self, tracked_run, frozen_run):
        """Updating against the current estimate must bring chi2 down."""
        tail = slice(-15, None)
        assert (tracked_run.monitor.statistic[tail].mean()
                < 0.5 * frozen_run.monitor.statistic[tail].mean())

    def test_a_faster_drift_std_tracks_sooner_and_noisier(self, fouling):
        """The bandwidth knob does what it claims in both directions."""
        etas, data = fouling
        n = 25
        slow, fast = [
            track_parameters(parallel_pipes, data[:n], SIGMA, state=start(),
                             names=NAMES, drift_std=[q], window=10)
            for q in (1e-4, 2e-2)
        ]
        assert (np.std(np.diff(fast.of("eta")))
                > np.std(np.diff(slow.of("eta"))))
        assert (abs(fast.of("eta") - etas[:n]).mean()
                < abs(slow.of("eta") - etas[:n]).mean())

    def test_times_set_the_clock_the_drift_rate_is_read_in(self, fouling):
        _, data = fouling
        run = track_parameters(
            parallel_pipes, data[:10], SIGMA, state=start(), names=NAMES,
            drift_std=DRIFT, times=np.arange(10) * 2.0, window=5,
        )
        assert run.final.time == pytest.approx(18.0)

    def test_mismatched_times_are_refused(self, fouling):
        _, data = fouling
        with pytest.raises(ValueError, match="times for"):
            track_parameters(
                parallel_pipes, data[:5], SIGMA, state=start(), names=NAMES,
                drift_std=DRIFT, times=[0.0, 1.0],
            )

    def test_the_result_reports_the_whole_campaign(self, fouling, tracked_run):
        _, data = fouling
        run = tracked_run
        assert len(run) == len(data)
        assert run.trajectory.shape == (len(data), 1)
        assert run.std_trajectory.shape == (len(data), 1)
        assert run.of("eta").shape == (len(data),)
        assert len(run.monitor) == len(data)
        assert "eta" in run.summary()
        assert run.n_updates == sum(s.updated for s in run.steps)
        assert run.initial.as_params()["eta"] == pytest.approx(1.0)

    def test_every_updated_step_records_what_it_did(self, tracked_run):
        moved = [s for s in tracked_run.steps if s.updated]
        assert moved
        for s in moved:
            assert s.estimate is not None and s.objective is not None
            assert s.innovation.nis >= 0.0
            assert s.decision.allowed

    def test_params_carries_fixed_arguments_alongside(self, fouling):
        """Both clocks inject the tracked parameter into the user's
        ``params`` and leave the rest of it alone."""
        _, data = fouling
        seen = []

        def with_extra(x, params):
            seen.append(set(params))
            return parallel_pipes(x, {"eta": params["eta"] * params["scale"]})

        # the gate is opened on every verdict so both clocks run in a
        # campaign short enough to be cheap.
        run = track_parameters(
            with_extra, data[:8], SIGMA, state=start(), names=NAMES,
            drift_std=DRIFT, window=5, params={"scale": 1.0},
            allow=[MONITOR_CONSISTENT, MONITOR_MODEL_DRIFT,
                   MONITOR_INSTRUMENT_FAULT, MONITOR_UNDIAGNOSED],
        )
        assert run.n_updates == 8
        assert seen and all(s == {"eta", "scale"} for s in seen)

    def test_default_allow_is_the_documented_policy(self):
        assert UPDATE_WHEN_DRIFTING == frozenset({MONITOR_MODEL_DRIFT})
