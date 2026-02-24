"""Tests for difflow.estimation module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from difflow.estimation import (
    Experiment,
    Estimator,
    EstimationResult,
    ConfidenceResult,
    DiagnosticsResult,
    BootstrapResult,
    CrossValidationResult,
    sum_squared_errors,
    weighted_sum_squared_errors,
    negative_log_likelihood,
    fisher_confidence_intervals,
    compute_diagnostics,
)


# ---------------------------------------------------------------------------
# Test fixtures: a simple linear model  y = a * x + b
# ---------------------------------------------------------------------------

def linear_model(theta, exp):
    """y = a * x + b"""
    return {'y': theta['a'] * exp.inputs['x'] + theta['b']}


def make_linear_experiments(a_true=2.0, b_true=1.0, n=10, noise_std=0.0, seed=0):
    """Generate experiments from a linear model."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.5, 5.0, n)
    experiments = []
    for x in xs:
        y = a_true * x + b_true + noise_std * rng.standard_normal()
        experiments.append(Experiment(inputs={'x': float(x)}, observed={'y': float(y)}))
    return experiments


# ---------------------------------------------------------------------------
# Experiment tests
# ---------------------------------------------------------------------------

class TestExperiment:
    def test_creation(self):
        exp = Experiment(inputs={'x': 1.0}, observed={'y': 2.0})
        assert exp.inputs == {'x': 1.0}
        assert exp.observed == {'y': 2.0}
        assert exp.output_names == ['y']

    def test_observed_array(self):
        exp = Experiment(inputs={}, observed={'a': 1.0, 'b': 2.0})
        arr = exp.observed_array
        assert arr.shape == (2,)

    def test_weights_no_uncertainty(self):
        exp = Experiment(inputs={}, observed={'y': 1.0})
        w = exp.weights
        np.testing.assert_allclose(w, [1.0])

    def test_weights_with_uncertainty(self):
        exp = Experiment(inputs={}, observed={'y': 1.0}, uncertainties={'y': 0.5})
        w = exp.weights
        np.testing.assert_allclose(w, [4.0])  # 1/0.5^2

    def test_params_mixin(self):
        exp = Experiment(inputs={'x': 1.0}, observed={'y': 2.0}, name='test')
        assert 'inputs' in exp
        assert exp['name'] == 'test'


# ---------------------------------------------------------------------------
# Objective function tests
# ---------------------------------------------------------------------------

class TestObjectives:
    def test_sse_perfect_fit(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=5)
        theta = jnp.array([2.0, 1.0])
        sse = sum_squared_errors(linear_model, theta, exps, ['a', 'b'])
        np.testing.assert_allclose(float(sse), 0.0, atol=1e-10)

    def test_sse_nonzero(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=5)
        theta = jnp.array([1.0, 0.0])  # wrong parameters
        sse = sum_squared_errors(linear_model, theta, exps, ['a', 'b'])
        assert float(sse) > 0

    def test_wsse_with_uniform_weights(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=5)
        theta = jnp.array([1.0, 0.0])
        sse = sum_squared_errors(linear_model, theta, exps, ['a', 'b'])
        wsse = weighted_sum_squared_errors(linear_model, theta, exps, ['a', 'b'])
        # No uncertainties => weights=1 => wsse == sse
        np.testing.assert_allclose(float(wsse), float(sse), rtol=1e-10)

    def test_nll_perfect_fit(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=5)
        theta = jnp.array([2.0, 1.0])
        nll = negative_log_likelihood(linear_model, theta, exps, ['a', 'b'])
        # With sigma=1, NLL at perfect fit = n/2 * log(2*pi)
        expected = 5 * 0.5 * float(jnp.log(2 * jnp.pi))
        np.testing.assert_allclose(float(nll), expected, rtol=1e-10)

    def test_sse_gradient(self):
        """Verify jax.grad works through SSE."""
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=5)
        grad_fn = jax.grad(lambda t: sum_squared_errors(linear_model, t, exps, ['a', 'b']))
        g = grad_fn(jnp.array([2.0, 1.0]))
        # At true params, gradient should be zero
        np.testing.assert_allclose(g, [0.0, 0.0], atol=1e-8)

    def test_nll_gradient(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=5)
        grad_fn = jax.grad(lambda t: negative_log_likelihood(linear_model, t, exps, ['a', 'b']))
        g = grad_fn(jnp.array([2.0, 1.0]))
        np.testing.assert_allclose(g, [0.0, 0.0], atol=1e-8)


# ---------------------------------------------------------------------------
# Estimator.fit tests
# ---------------------------------------------------------------------------

class TestEstimatorFit:
    def test_fit_recovers_params_noiseless(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, theta_init={'a': 1.0, 'b': 0.0})

        assert result.converged
        np.testing.assert_allclose(result.theta_opt['a'], 2.0, atol=1e-6)
        np.testing.assert_allclose(result.theta_opt['b'], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.objective_value, 0.0, atol=1e-10)

    def test_fit_recovers_params_noisy(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=20, noise_std=0.1)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, theta_init={'a': 1.0, 'b': 0.0})

        assert result.converged
        np.testing.assert_allclose(result.theta_opt['a'], 2.0, atol=0.2)
        np.testing.assert_allclose(result.theta_opt['b'], 1.0, atol=0.5)

    def test_fit_with_bounds(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10)
        est = Estimator(linear_model, ['a', 'b'],
                         param_bounds={'a': (0, 10), 'b': (-5, 5)})
        result = est.fit(exps, theta_init={'a': 1.0, 'b': 0.0})
        assert result.converged
        np.testing.assert_allclose(result.theta_opt['a'], 2.0, atol=1e-6)

    def test_fit_wsse_objective(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, theta_init={'a': 1.0, 'b': 0.0}, objective='wsse')
        assert result.converged
        np.testing.assert_allclose(result.theta_opt['a'], 2.0, atol=1e-6)

    def test_fit_nll_objective(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, theta_init={'a': 1.0, 'b': 0.0}, objective='nll')
        assert result.converged
        np.testing.assert_allclose(result.theta_opt['a'], 2.0, atol=1e-6)

    def test_result_is_paramsmixin(self):
        exps = make_linear_experiments(n=5)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        assert 'theta_opt' in result
        assert result['converged'] is True


# ---------------------------------------------------------------------------
# Confidence interval tests
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_intervals(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=20, noise_std=0.1)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        ci = est.confidence_intervals(result, exps)

        assert isinstance(ci, ConfidenceResult)
        # True value should be within CI
        assert ci.ci_lower['a'] < 2.0 < ci.ci_upper['a']
        assert ci.ci_lower['b'] < 1.0 < ci.ci_upper['b']
        # Covariance matrix should be 2x2 positive semi-definite
        assert ci.covariance.shape == (2, 2)
        eigvals = jnp.linalg.eigvalsh(ci.covariance)
        assert jnp.all(eigvals >= -1e-10)

    def test_confidence_noiseless(self):
        """With noiseless data, CIs should be very tight."""
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        ci = est.confidence_intervals(result, exps)

        # Std errors should be extremely small
        assert ci.std_errors['a'] < 1e-6
        assert ci.std_errors['b'] < 1e-6


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_diagnostics_perfect_fit(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        diag = est.diagnostics(result, exps)

        assert isinstance(diag, DiagnosticsResult)
        np.testing.assert_allclose(diag.r_squared, 1.0, atol=1e-10)
        np.testing.assert_allclose(diag.rmse, 0.0, atol=1e-6)
        assert diag.n_obs == 10
        assert diag.n_params == 2

    def test_diagnostics_noisy(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=20, noise_std=0.1)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        diag = est.diagnostics(result, exps)

        assert diag.r_squared > 0.95  # should be close to 1
        assert diag.rmse > 0
        assert diag.rmse < 0.2

    def test_residuals_length(self):
        exps = make_linear_experiments(n=5)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        diag = est.diagnostics(result, exps)
        assert len(diag.residuals) == 5


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_nonparametric_bootstrap(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=15, noise_std=0.1)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        bs = est.bootstrap(result, exps, n_bootstrap=20, method='nonparametric', seed=42)

        assert isinstance(bs, BootstrapResult)
        assert bs.theta_samples.shape == (20, 2)
        assert bs.method == 'nonparametric'
        # Mean should be close to true values
        np.testing.assert_allclose(bs.mean['a'], 2.0, atol=0.5)
        np.testing.assert_allclose(bs.mean['b'], 1.0, atol=1.0)

    def test_parametric_bootstrap(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=15, noise_std=0.1)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        bs = est.bootstrap(result, exps, n_bootstrap=20, method='parametric', seed=42)

        assert isinstance(bs, BootstrapResult)
        assert bs.method == 'parametric'
        np.testing.assert_allclose(bs.mean['a'], 2.0, atol=0.5)

    def test_bootstrap_ci_contains_true(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=20, noise_std=0.05)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        bs = est.bootstrap(result, exps, n_bootstrap=50, alpha=0.05)

        assert bs.ci_lower['a'] < 2.0 < bs.ci_upper['a']


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_leave_one_out(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=8, noise_std=0.05)
        est = Estimator(linear_model, ['a', 'b'])
        cv = est.cross_validate(exps, {'a': 1.0, 'b': 0.0}, n=1)

        assert isinstance(cv, CrossValidationResult)
        assert cv.n_folds == 8  # C(8,1) = 8
        assert cv.n_holdout == 1
        assert len(cv.cv_scores) == 8
        assert cv.mean_score >= 0

    def test_leave_two_out_with_max_folds(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=8, noise_std=0.05)
        est = Estimator(linear_model, ['a', 'b'])
        cv = est.cross_validate(exps, {'a': 1.0, 'b': 0.0}, n=2, max_folds=5)

        assert cv.n_folds == 5
        assert cv.n_holdout == 2


# ---------------------------------------------------------------------------
# Summary test
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_string(self):
        exps = make_linear_experiments(a_true=2.0, b_true=1.0, n=10, noise_std=0.1)
        est = Estimator(linear_model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 0.0})
        s = est.summary(result, exps)

        assert "Parameter Estimation Summary" in s
        assert "R-squared" in s
        assert "a" in s
        assert "b" in s
        assert "Converged: True" in s


# ---------------------------------------------------------------------------
# Multi-output model test
# ---------------------------------------------------------------------------

class TestMultiOutput:
    def test_multi_output_model(self):
        """Test with a model that predicts multiple outputs."""
        def model(theta, exp):
            x = exp.inputs['x']
            return {
                'y1': theta['a'] * x,
                'y2': theta['b'] * x ** 2,
            }

        exps = []
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            exps.append(Experiment(
                inputs={'x': x},
                observed={'y1': 3.0 * x, 'y2': 0.5 * x ** 2},
            ))

        est = Estimator(model, ['a', 'b'])
        result = est.fit(exps, {'a': 1.0, 'b': 1.0})
        assert result.converged
        np.testing.assert_allclose(result.theta_opt['a'], 3.0, atol=1e-5)
        np.testing.assert_allclose(result.theta_opt['b'], 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# Nonlinear model test
# ---------------------------------------------------------------------------

class TestNonlinear:
    def test_exponential_decay(self):
        """Test parameter estimation for y = A * exp(-k * t)."""
        def model(theta, exp):
            t = exp.inputs['t']
            return {'y': theta['A'] * jnp.exp(-theta['k'] * t)}

        A_true, k_true = 5.0, 0.3
        exps = []
        for t in jnp.linspace(0.0, 10.0, 15):
            y = A_true * jnp.exp(-k_true * t)
            exps.append(Experiment(inputs={'t': float(t)}, observed={'y': float(y)}))

        est = Estimator(model, ['A', 'k'], param_bounds={'A': (0.1, 20.0), 'k': (0.01, 5.0)})
        result = est.fit(exps, {'A': 3.0, 'k': 0.1})
        assert result.converged
        np.testing.assert_allclose(result.theta_opt['A'], A_true, atol=1e-4)
        np.testing.assert_allclose(result.theta_opt['k'], k_true, atol=1e-4)

    def test_exponential_gradient_check(self):
        """Verify gradients through a nonlinear model."""
        def model(theta, exp):
            t = exp.inputs['t']
            return {'y': theta['A'] * jnp.exp(-theta['k'] * t)}

        exps = [Experiment(inputs={'t': 1.0}, observed={'y': 3.0})]

        def obj(theta):
            return sum_squared_errors(model, theta, exps, ['A', 'k'])

        # Compare JAX grad to finite differences
        theta = jnp.array([5.0, 0.3])
        jax_grad = jax.grad(obj)(theta)
        eps = 1e-5
        fd_grad = jnp.array([
            (obj(theta.at[0].set(theta[0] + eps)) - obj(theta.at[0].set(theta[0] - eps))) / (2 * eps),
            (obj(theta.at[1].set(theta[1] + eps)) - obj(theta.at[1].set(theta[1] - eps))) / (2 * eps),
        ])
        np.testing.assert_allclose(jax_grad, fd_grad, rtol=1e-4)
