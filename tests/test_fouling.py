"""Smoke tests for difflow.fouling (Ebert-Panchal threshold fouling)."""

import math

import jax
import pytest

from difflow import fouling as fo

K = dict(alpha=2.0e-3, gamma=4.0e-13, E=48.0e3)     # illustrative


def test_rate_and_thresholds_finite():
    Re, Pr, T, tau = 3.0e4, 10.0, 600.0, 2.0
    assert math.isfinite(float(fo.ebert_panchal_rate(Re, Pr, T, tau, **K)))
    assert float(fo.ebert_panchal_rate(Re, Pr, T, 1e3, floor_at_zero=True, **K)) == 0.0
    assert math.isfinite(float(fo.threshold_wall_shear(Re, Pr, T, K["alpha"], K["gamma"], K["E"])))
    assert math.isfinite(float(fo.threshold_film_temperature(Re, Pr, 0.02, **K)))
    assert math.isfinite(float(fo.film_temperature(500.0, 560.0)))
    assert math.isfinite(float(fo.wall_shear_stress(0.02, 800.0, 1.5)))
    assert math.isfinite(float(jax.grad(
        lambda t: fo.ebert_panchal_rate(Re, Pr, t, tau, **K))(T)))


def test_fouling_time():
    assert float(fo.fouling_time(3e-4, 1e-11)) == pytest.approx(3e7, rel=1e-14)
    assert math.isinf(float(fo.fouling_time(3e-4, 0.0)))


def test_integrate_fouling():
    out = fo.integrate_fouling(lambda R_: 1e-11 + 0.0 * R_, 3e7, n_steps=10)
    assert out["R_f"].shape == (11,)
    assert math.isfinite(float(out["R_f_final"]))
