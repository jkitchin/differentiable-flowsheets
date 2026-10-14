"""Tests for the non-Newtonian friction factors in difflow.fluids.

Convention under test: every public function returns the Darcy factor
(``4 x`` the Fanning factor of the source correlations), like
``friction_factor``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

from difflow import fluids
from difflow.fluids import (
    bingham_friction_factor,
    bingham_pressure_drop,
    friction_factor,
    hedstrom_number,
    metzner_reed_reynolds,
    power_law_friction_factor,
    power_law_pressure_drop,
    reynolds_number,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Power law
# ---------------------------------------------------------------------------


def test_metzner_reed_reduces_to_newtonian_reynolds():
    Re_mr = metzner_reed_reynolds(998.0, 1.3, 0.05, 1.0e-3, 1.0)
    np.testing.assert_allclose(float(Re_mr), float(reynolds_number(998.0, 1.3, 0.05, 1.0e-3)), rtol=1e-14)


@pytest.mark.parametrize("n", [0.4, 0.7, 1.0, 1.3])
def test_power_law_laminar_is_64_over_Re(n):
    for Re in (10.0, 500.0, 2000.0):
        assert float(power_law_friction_factor(Re, n)) == pytest.approx(64.0 / Re, rel=1e-14)


@pytest.mark.parametrize("n", [0.3, 0.5, 0.8, 1.0, 1.5])
def test_power_law_laminar_pressure_drop_is_rabinowitsch_mooney(n):
    Q, L, D, rho, K = 2e-4, 12.0, 0.04, 1100.0, 2.0
    v = Q / (math.pi / 4 * D**2)
    assert float(metzner_reed_reynolds(rho, v, D, K, n)) < 2100
    exact = 4 * K * L / D * (8 * v / D) ** n * ((3 * n + 1) / (4 * n)) ** n
    np.testing.assert_allclose(float(power_law_pressure_drop(Q, L, D, rho, K, n)), exact, rtol=1e-12)


def test_power_law_newtonian_limit_equals_hagen_poiseuille():
    Q, L, D, rho, mu = 1e-4, 10.0, 0.05, 900.0, 0.2
    dp = power_law_pressure_drop(Q, L, D, rho, mu, 1.0)
    np.testing.assert_allclose(float(dp), 128 * mu * L * Q / (math.pi * D**4), rtol=1e-12)


@pytest.mark.parametrize("Re", [1e4, 3e4, 1e5, 1e6, 1e7])
def test_dodge_metzner_at_n1_is_close_to_smooth_colebrook(Re):
    f_dm = float(power_law_friction_factor(Re, 1.0))
    f_cb = float(friction_factor(Re, 0.0, "colebrook"))
    assert abs(f_dm / f_cb - 1.0) < 0.01


@pytest.mark.parametrize("n", [0.36, 0.6, 1.0])
@pytest.mark.parametrize("Re", [5e3, 3e4, 1e5])
def test_dodge_metzner_satisfies_its_equation(n, Re):
    fF = float(power_law_friction_factor(Re, n)) / 4.0  # Fanning
    lhs = 1.0 / math.sqrt(fF)
    rhs = 4.0 / n**0.75 * math.log10(Re * fF ** (1.0 - n / 2.0)) - 0.4 / n**1.2
    assert abs(lhs - rhs) < 1e-10


def test_dodge_metzner_shear_thinning_lowers_turbulent_friction():
    f = [float(power_law_friction_factor(5e4, n)) for n in (1.0, 0.8, 0.6, 0.4)]
    assert all(a > b for a, b in zip(f, f[1:]))


def test_power_law_gradients():
    for Re in (800.0, 3000.0, 2e4, 2e5):
        for n in (0.5, 0.9):
            # arguments scaled to O(1) so one finite-difference step suits both
            check_grads(lambda u, m: power_law_friction_factor(Re * u, m) * 100.0, (1.0, n), order=1,
                        modes=["rev", "fwd"], eps=1e-6, rtol=1e-4, atol=1e-6)
    check_grads(lambda D: power_law_pressure_drop(3e-3, 50.0, D, 1050.0, 0.05, 0.7) / 1e3,
                (0.05,), order=1, modes=["rev"], eps=1e-7, rtol=1e-4)


def test_power_law_implicit_gradient_matches_finite_difference():
    f = lambda R: power_law_friction_factor(R, 0.6)
    g = float(jax.grad(f)(3e4))
    fd = (float(f(3e4 + 1.0)) - float(f(3e4 - 1.0))) / 2.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# Bingham plastic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Re", [10.0, 300.0, 1500.0])
def test_bingham_newtonian_laminar_limit(Re):
    assert float(bingham_friction_factor(Re, 0.0, "laminar")) == pytest.approx(64.0 / Re, rel=1e-14)
    # the composite equals the laminar result well below transition
    assert float(bingham_friction_factor(Re, 0.0)) == pytest.approx(64.0 / Re, rel=1e-10)


@pytest.mark.parametrize("Re,He", [(100.0, 1e3), (1000.0, 1e5), (500.0, 1e6), (2000.0, 3e4)])
def test_buckingham_reiner_matches_independent_root(Re, He):
    """Fanning f_L = (16/Re)(1 + He/(6Re) - He^4/(3 f^3 Re^7)), bracketed with brentq."""
    def r(f):
        return f - 16.0 / Re * (1 + He / (6 * Re) - He**4 / (3 * f**3 * Re**7))
    # physical root has xi = 2 He / (f Re^2) < 1
    lo = 2 * He / Re**2 * (1 + 1e-12)
    hi = 16.0 / Re * (1 + He / (6 * Re))
    f_ref = brentq(r, max(lo, 1e-12), hi, xtol=1e-16, rtol=1e-14)
    f = float(bingham_friction_factor(Re, He, "laminar")) / 4.0
    np.testing.assert_allclose(f, f_ref, rtol=1e-10)


def test_bingham_laminar_flow_rate_from_buckingham_equation():
    """Q = pi D^4 dP / (128 mu_p L) [1 - 4/3 xi + xi^4/3], xi = tau_y / tau_w."""
    Q, L, D, rho, tau_y, mu_p = 5e-4, 20.0, 0.05, 1200.0, 8.0, 0.05
    dP = float(bingham_pressure_drop(Q, L, D, rho, tau_y, mu_p, "laminar"))
    tau_w = dP * D / (4 * L)
    xi = tau_y / tau_w
    assert 0.0 < xi < 1.0
    Q_buck = math.pi * D**4 * dP / (128 * mu_p * L) * (1 - 4 / 3 * xi + xi**4 / 3)
    np.testing.assert_allclose(Q_buck, Q, rtol=1e-10)


def test_bingham_turbulent_darby_fit():
    """At high Re the composite tends to f_T = 10^a Re^-0.193 (Fanning)."""
    Re, He = 1e6, 1e5
    a = -1.47 * (1 + 0.146 * math.exp(-2.9e-5 * He))
    fT = 10**a * Re**-0.193
    fL = float(bingham_friction_factor(Re, He, "laminar")) / 4.0
    m = 1.7 + 40000 / Re
    f = float(bingham_friction_factor(Re, He)) / 4.0
    np.testing.assert_allclose(f, (fL**m + fT**m) ** (1 / m), rtol=1e-12)
    assert f >= max(fL, fT)


def test_bingham_composite_finite_for_extreme_m():
    f = bingham_friction_factor(jnp.array([1.0, 10.0, 1e3, 1e5, 1e7]), 1e4)
    assert np.all(np.isfinite(np.asarray(f)))


def test_hedstrom_number():
    np.testing.assert_allclose(float(hedstrom_number(1200.0, 8.0, 0.05, 0.05)), 1200 * 8 * 0.0025 / 0.0025, rtol=1e-14)


def test_bingham_gradients():
    for Re, He in ((300.0, 1e4), (5e3, 1e5), (1e5, 1e6)):
        for method in fluids.BINGHAM_METHODS:
            check_grads(lambda u, w: bingham_friction_factor(Re * u, He * w, method) * 100.0, (1.0, 1.0),
                        order=1, modes=["rev", "fwd"], eps=1e-6, rtol=1e-4, atol=1e-6)
    check_grads(lambda ty: bingham_pressure_drop(5e-4, 20.0, 0.05, 1200.0, ty, 0.05) / 1e3, (8.0,),
                order=1, modes=["rev"], eps=1e-5, rtol=1e-4)
    # Newtonian limit is differentiable too
    g = jax.grad(lambda R: bingham_friction_factor(R, 0.0, "laminar"))(500.0)
    np.testing.assert_allclose(float(g), -64.0 / 500.0**2, rtol=1e-10)


def test_bingham_bad_method():
    with pytest.raises(ValueError):
        bingham_friction_factor(1e3, 1e3, "darby1981")
