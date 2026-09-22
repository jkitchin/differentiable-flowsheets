"""Tests for the safe numerical primitives in difflow.numerics."""

import jax
import jax.numpy as jnp
import pytest

from difflow.constants import EPS_DIVISION
from difflow.numerics import safe_divide, safe_exp, safe_log, safe_sqrt


class TestSafeDivide:
    def test_zero_denominator_is_finite(self):
        assert float(safe_divide(1.0, 0.0)) == pytest.approx(1.0 / EPS_DIVISION)

    def test_small_negative_denominator_keeps_its_sign(self):
        """`sign(d + eps)` is positive for every d in (-eps, 0).

        That flipped the sign of the result exactly inside the window the
        clip is meant to protect, so a quantity heading for -inf came back
        as +1e10.
        """
        for d in (-1e-15, -1e-12, -EPS_DIVISION / 2, -EPS_DIVISION):
            assert float(safe_divide(1.0, d)) < 0.0

    def test_no_jump_at_the_clip_boundary(self):
        """The only discontinuity allowed is the one at d = 0 itself."""
        just_inside = float(safe_divide(1.0, -EPS_DIVISION * 0.999))
        just_outside = float(safe_divide(1.0, -EPS_DIVISION * 1.001))

        assert just_inside == pytest.approx(just_outside, rel=1e-2)

    def test_large_denominators_are_untouched(self):
        assert float(safe_divide(1.0, -4.0)) == pytest.approx(-0.25)
        assert float(safe_divide(1.0, 4.0)) == pytest.approx(0.25)

    def test_is_jittable_and_differentiable(self):
        g = jax.jit(jax.grad(lambda d: safe_divide(1.0, d)))
        assert jnp.isfinite(g(0.0))
        assert float(g(2.0)) == pytest.approx(-0.25)


class TestSafeElementary:
    def test_safe_log_of_zero_and_negative_is_finite(self):
        assert jnp.isfinite(safe_log(0.0))
        assert jnp.isfinite(safe_log(-1.0))

    def test_safe_sqrt_of_negative_is_not_nan(self):
        assert not jnp.isnan(safe_sqrt(-1.0))

    def test_safe_exp_does_not_overflow(self):
        assert jnp.isfinite(safe_exp(1e6))
