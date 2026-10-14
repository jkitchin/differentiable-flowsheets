"""Tests for protein degradation / stability kinetics.

Covers the nonlinear-in-time degradation models added for issue #131
(stretched-exponential / KWW and Lumry-Eyring), plus the existing
first-order integrated forms they generalize.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    aggregate_fraction,
    stretched_exponential_fraction,
    lumry_eyring_fraction,
)

jax.config.update("jax_enable_x64", True)


class TestStretchedExponential:
    def test_beta_one_recovers_first_order(self):
        """beta = 1 must match the first-order integrated form."""
        t, k = 5.0, 0.1
        f_kww = stretched_exponential_fraction(t, k, beta=1.0)
        f_fo = aggregate_fraction(t, k)
        assert float(f_kww) == pytest.approx(float(f_fo), rel=1e-9)

    def test_bounds_and_monotonic(self):
        ts = jnp.linspace(0.0, 50.0, 20)
        f = stretched_exponential_fraction(ts, 0.05, beta=0.6)
        assert float(f[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(f[-1]) <= 1.0 and float(f[-1]) > float(f[0])
        # non-decreasing
        assert bool(jnp.all(jnp.diff(f) >= -1e-9))

    def test_beta_less_than_one_faster_early(self):
        """Dispersive kinetics (beta<1) degrade faster at short times."""
        t = 1.0
        k = 0.2
        f_disp = stretched_exponential_fraction(t, k, beta=0.5)
        f_fo = stretched_exponential_fraction(t, k, beta=1.0)
        assert float(f_disp) > float(f_fo)

    def test_beta_greater_than_one_has_lag(self):
        """Sigmoidal kinetics (beta>1) degrade slower at short times."""
        t = 0.5
        k = 0.5
        f_sig = stretched_exponential_fraction(t, k, beta=2.0)
        f_fo = stretched_exponential_fraction(t, k, beta=1.0)
        assert float(f_sig) < float(f_fo)

    def test_differentiable(self):
        g = jax.grad(lambda k: stretched_exponential_fraction(5.0, k, beta=0.8))(0.1)
        assert jnp.isfinite(g) and float(g) > 0.0


class TestLumryEyring:
    def test_bounds_and_lag(self):
        """Consecutive model shows a lag: slower than one-step at short t."""
        t = 0.5
        f_le = lumry_eyring_fraction(t, k_unfold=1.0, k_agg=1.0)
        f_fo = aggregate_fraction(t, 1.0)
        assert 0.0 <= float(f_le) <= 1.0
        assert float(f_le) < float(f_fo)  # lag from the unfolding step

    def test_equal_rate_limit_matches_closed_form(self):
        """k_unfold == k_agg uses the degenerate 1-(1+kt)e^{-kt} limit."""
        t, k = 3.0, 0.4
        f = lumry_eyring_fraction(t, k, k)
        expected = 1.0 - (1.0 + k * t) * jnp.exp(-k * t)
        assert float(f) == pytest.approx(float(expected), rel=1e-9)

    def test_continuous_near_equal_rates(self):
        """No blow-up when the two rate constants are nearly equal."""
        t = 2.0
        f_close = lumry_eyring_fraction(t, 0.30000001, 0.3)
        f_equal = lumry_eyring_fraction(t, 0.3, 0.3)
        assert float(f_close) == pytest.approx(float(f_equal), abs=1e-6)

    def test_approaches_one_at_long_time(self):
        f = lumry_eyring_fraction(1000.0, 0.5, 0.2)
        assert float(f) == pytest.approx(1.0, abs=1e-6)

    def test_differentiable(self):
        g = jax.grad(lambda k: lumry_eyring_fraction(5.0, 0.3, k))(0.5)
        assert jnp.isfinite(g)
