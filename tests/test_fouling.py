"""Tests for difflow.fouling (Ebert-Panchal threshold fouling in core).

All constants (alpha, gamma, E) are ILLUSTRATIVE, chosen for the arithmetic; they are not fitted to any
fluid. The cross-check against the refinery plugin's copy is skipped if the plugin is not importable.
"""

import math

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import fluids
from difflow import fouling as fo

K = dict(alpha=2.0e-3, gamma=4.0e-13, E=48.0e3)     # illustrative
R = 8.314462618


def test_rate_matches_hand_formula():
    Re, Pr, T, tau = 3.0e4, 10.0, 600.0, 2.0
    hand = K["alpha"] * Re ** -0.66 * Pr ** -0.33 * math.exp(-K["E"] / (R * T)) - K["gamma"] * tau
    assert float(fo.ebert_panchal_rate(Re, Pr, T, tau, **K)) == pytest.approx(hand, rel=1e-13)


def test_matches_refinery_function_for_same_constants():
    pf = pytest.importorskip("difflow_refinery.preheat.fouling")
    model = pf.EbertPanchal(alpha=K["alpha"], beta=-0.66, E=K["E"], gamma=K["gamma"])
    for Re, Pr, T, tau in [(3e4, 10.0, 600.0, 2.0), (8e3, 25.0, 520.0, 0.5), (5e4, 7.0, 650.0, 8.0)]:
        a = fo.ebert_panchal_rate(Re, Pr, T, tau, **K)
        b = pf.fouling_rate(model, Re, Pr, T, tau)
        assert float(a) == pytest.approx(float(b), rel=1e-13, abs=0.0)
    assert float(fo.film_temperature(500.0, 560.0)) == pytest.approx(float(pf.film_temperature(500.0, 560.0)), rel=1e-14)


def test_floor_and_threshold_shear():
    Re, Pr, T = 3.0e4, 10.0, 600.0
    tau_star = fo.threshold_wall_shear(Re, Pr, T, K["alpha"], K["gamma"], K["E"])
    assert float(fo.ebert_panchal_rate(Re, Pr, T, tau_star, **K)) == pytest.approx(0.0, abs=1e-22)
    assert float(fo.ebert_panchal_rate(Re, Pr, T, 2.0 * tau_star, **K)) < 0.0
    assert float(fo.ebert_panchal_rate(Re, Pr, T, 2.0 * tau_star, floor_at_zero=True, **K)) == 0.0
    assert float(fo.ebert_panchal_rate(Re, Pr, T, 0.5 * tau_star, floor_at_zero=True, **K)) > 0.0


def test_threshold_film_temperature_zero_rate():
    Re, Pr, tau = 3.0e4, 10.0, 0.02
    T_star = fo.threshold_film_temperature(Re, Pr, tau, **{"alpha": K["alpha"], "gamma": K["gamma"], "E": K["E"]})
    assert float(fo.ebert_panchal_rate(Re, Pr, T_star, tau, **K)) == pytest.approx(0.0, abs=1e-22)
    # colder than the threshold: net removal; hotter: fouling
    assert float(fo.ebert_panchal_rate(Re, Pr, T_star - 20.0, tau, **K)) < 0.0
    assert float(fo.ebert_panchal_rate(Re, Pr, T_star + 20.0, tau, **K)) > 0.0
    # no threshold exists if removal beats deposition at any T
    assert math.isnan(float(fo.threshold_film_temperature(Re, Pr, 1e30, **{"alpha": K["alpha"], "gamma": K["gamma"], "E": K["E"]})))


def test_wall_shear_from_friction_and_velocity():
    rho, mu, D, v = 800.0, 5e-3, 0.02, 1.5
    Re = fluids.reynolds_number(rho, v, D, mu)
    f = fluids.friction_factor(Re, 0.0)
    tau = fo.wall_shear_stress(f, rho, v)
    assert float(tau) == pytest.approx(float(f) / 8.0 * rho * v**2, rel=1e-14)
    # the force balance: tau_w * pi D L = dP * pi D^2 / 4  with dP = f L/D rho v^2/2
    L = 1.0
    dP = float(f) * L / D * 0.5 * rho * v**2
    assert float(tau) * math.pi * D * L == pytest.approx(dP * math.pi * D**2 / 4.0, rel=1e-12)


def test_fouling_time():
    assert float(fo.fouling_time(3e-4, 1e-11)) == pytest.approx(3e7, rel=1e-14)
    assert float(fo.fouling_time(3e-4, 1e-11, R_f0=1e-4)) == pytest.approx(2e7, rel=1e-14)
    assert math.isinf(float(fo.fouling_time(3e-4, -1e-12)))
    assert math.isinf(float(fo.fouling_time(3e-4, 0.0)))
    g = jax.grad(lambda r: fo.fouling_time(3e-4, r))(1e-11)
    assert float(g) == pytest.approx(-3e-4 / 1e-22, rel=1e-12)


def test_integrate_constant_and_feedback():
    out = fo.integrate_fouling(lambda R_: 1e-11 + 0.0 * R_, 3e7, n_steps=10)
    assert float(out["R_f_final"]) == pytest.approx(3e-4, rel=1e-12)
    assert out["R_f"].shape == (11,) and float(out["t"][-1]) == pytest.approx(3e7)
    # linear feedback dR/dt = c R + r0 has the closed form (r0/c)(exp(c t) - 1)
    c, r0, t = 1e-8, 1e-11, 2e7
    out = fo.integrate_fouling(lambda R_: c * R_ + r0, t, n_steps=200)
    assert float(out["R_f_final"]) == pytest.approx(r0 / c * (math.exp(c * t) - 1.0), rel=1e-9)
    # a net-removal rate cannot take R_f below zero
    assert float(fo.integrate_fouling(lambda R_: -1e-11 + 0.0 * R_, 1e7, R_f0=1e-5)["R_f_final"]) == 0.0


def test_gradients():
    f = lambda Re, Pr, T, tau: fo.ebert_panchal_rate(Re, Pr, T, tau, **K)
    check_grads(f, (3.0e4, 10.0, 600.0, 2.0), order=1, modes=["rev"], rtol=1e-5, atol=1e-20)
    # scaled variable s = c / 1e-8 and a scaled output so check_grads' step sizes are sensible
    g = lambda s: 1e4 * fo.integrate_fouling(lambda R_: 1e-8 * s * R_ + 1e-11, 2e7, n_steps=50)["R_f_final"]
    check_grads(g, (1.0,), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)
    # the fouling rate is differentiable in the tube velocity through Re and tau_w
    def rate_v(v):
        Re = fluids.reynolds_number(800.0, v, 0.02, 5e-3)
        return fo.ebert_panchal_rate(Re, 10.0, 600.0, fo.wall_shear_stress(fluids.friction_factor(Re, 0.0), 800.0, v), **K)
    assert float(jax.grad(rate_v)(1.5)) < 0.0     # faster flow, more shear, less fouling
