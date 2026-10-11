"""Textbook-style checks of difflow.particles, computed independently.

Status of every test here, stated plainly: **none reproduces a printed
answer**. The problems are set up the way McCabe, Smith & Harriott (Ch. 7),
Kunii & Levenspiel (Ch. 3) and Geankoplis (Sec. 3.1) set them up, but the
book's printed numbers could not be recalled with confidence, and a number
recalled without a check is not a reference. So each expected value is
computed **by hand in the test**, from the textbook equation, in the
textbook's own units (cgs for Kunii & Levenspiel, US customary for the
Ergun check), with plain Python arithmetic that shares no code with
``difflow.particles``. That checks the equations and, above all, the unit
conversions; it does not check the correlations against data.

The subject of these tests is the answer, so they are marked ``release``.
"""

import math

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.particles import (
    drag_coefficient,
    ergun_pressure_gradient,
    minimum_fluidization_velocity,
    terminal_velocity,
)

pytestmark = pytest.mark.release

G_CGS = 980.665   # cm/s^2


def test_kunii_levenspiel_umf_cgs():
    """K&L Ch. 3 form of the Ergun u_mf, solved by hand in cgs (not the printed answer).

    Sand-like solid, as in K&L's u_mf examples: d_p = 160 micron, phi = 0.67,
    rho_s = 2.6 g/cm^3, eps_mf = 0.47, ambient air.
    """
    d, phi, rho_s, eps = 160e-4, 0.67, 2.6, 0.47            # cm, -, g/cm^3, -
    rho_g, mu = 1.2e-3, 1.8e-4                               # g/cm^3, g/(cm s)
    Ar = d**3 * rho_g * (rho_s - rho_g) * G_CGS / mu**2
    a = 1.75 / (eps**3 * phi)
    b = 150.0 * (1.0 - eps) / (eps**3 * phi**2)
    Re = (-b + math.sqrt(b * b + 4.0 * a * Ar)) / (2.0 * a)
    u_cms = Re * mu / (rho_g * d)
    got = minimum_fluidization_velocity(d / 100.0, rho_s * 1000.0, rho_g * 1000.0, mu / 10.0, eps, phi)
    assert float(got) * 100.0 == pytest.approx(u_cms, rel=1e-10)
    assert 0.5 < u_cms < 5.0          # a few cm/s, the size K&L find for such sands


def test_wen_yu_cgs():
    """Wen & Yu (1966) in K&L's form, Re_mf = sqrt(33.7^2 + 0.0408 Ar) - 33.7 (hand, cgs)."""
    d, rho_s, rho_g, mu = 450e-4, 2.5, 1.2e-3, 1.8e-4
    Ar = d**3 * rho_g * (rho_s - rho_g) * G_CGS / mu**2
    u_cms = (math.sqrt(33.7**2 + 0.0408 * Ar) - 33.7) * mu / (rho_g * d)
    got = minimum_fluidization_velocity(d / 100.0, rho_s * 1e3, rho_g * 1e3, mu / 10.0, method="wen_yu")
    assert float(got) * 100.0 == pytest.approx(u_cms, rel=1e-10)


def test_mccabe_stokes_settling():
    """Stokes's law, u_t = g d^2 (rho_p - rho)/(18 mu) (MSH Eq. 7.40 form; hand, not printed).

    20 micron glass bead (2500 kg/m^3) in water: Re_t ~ 3e-3, where the drag
    fits are within 1 % of Stokes.
    """
    d, rho_p, rho, mu = 20e-6, 2500.0, 998.0, 1.0e-3
    u_stokes = 9.80665 * d**2 * (rho_p - rho) / (18.0 * mu)
    for method in ("haider_levenspiel", "turton_levenspiel", "schiller_naumann"):
        assert float(terminal_velocity(d, rho_p, rho, mu, method=method)) == pytest.approx(u_stokes, rel=1e-2)


def test_mccabe_newton_settling():
    """Newton's law, u_t = 1.75 sqrt(g d (rho_p - rho)/rho) for C_D = 0.44 (MSH form; hand).

    With ``schiller_naumann`` the drag coefficient is exactly 0.44 above
    Re ~ 1000, so the result is the closed form exactly:
    u_t = sqrt(4 g d (rho_p - rho) / (3 * 0.44 * rho)). 5 mm steel ball in air.
    """
    d, rho_p, rho, mu = 5e-3, 7800.0, 1.2, 1.8e-5
    u_newton = math.sqrt(4.0 * 9.80665 * d * (rho_p - rho) / (3.0 * 0.44 * rho))
    u = float(terminal_velocity(d, rho_p, rho, mu, method="schiller_naumann"))
    assert u == pytest.approx(u_newton, rel=1e-8)
    assert rho * u * d / mu > 1000.0
    assert math.sqrt(4.0 / (3.0 * 0.44)) == pytest.approx(1.74, abs=0.005)   # MSH's 1.74


def test_geankoplis_drag_on_sphere():
    """Drag force on a sphere, F_D = C_D (rho u^2/2) (pi D^2/4), Geankoplis Sec. 3.1 setup (hand).

    Air at 37.8 C (rho = 1.137 kg/m^3, mu = 1.90e-5 Pa s) past a 42 mm sphere
    at 2.3 m/s, Re ~ 5800, on the Newton plateau. The check is the force
    arithmetic around ``drag_coefficient``, and that C_D sits in the
    0.38-0.5 band of the standard curve at this Re (the book reads C_D off
    its Fig. 3.1-2).
    """
    rho, mu, D, u = 1.137, 1.90e-5, 0.042, 2.3
    Re = rho * u * D / mu
    C_D = float(drag_coefficient(Re))
    F = C_D * 0.5 * rho * u**2 * math.pi * D**2 / 4.0
    assert 0.38 < C_D < 0.5
    assert F == pytest.approx(C_D * 0.5 * 1.137 * 2.3**2 * math.pi * 0.042**2 / 4.0, rel=1e-14)
    assert 0.0015 < F < 0.0025   # N


def test_ergun_us_customary():
    """Ergun (1952) in US customary units with g_c, by hand (MSH/Geankoplis form; not printed).

    dP/L [lbf/ft^2 per ft] = 150 mu u (1-eps)^2 / (g_c phi^2 d^2 eps^3)
                            + 1.75 rho u^2 (1-eps) / (g_c phi d eps^3)
    with g_c = 32.174 lb ft/(lbf s^2) (exact value used); converted to Pa/m and compared.
    """
    d_ft, eps, phi = 0.01, 0.38, 0.9                  # ~3 mm particles
    u_fts, rho_lbft3, mu_lbfts = 1.5, 0.075, 1.2e-5   # air-like
    gc = 9.80665 / 0.3048          # 32.174 lb ft/(lbf s^2), exact
    dpdl = (150.0 * mu_lbfts * u_fts * (1 - eps) ** 2 / (gc * phi**2 * d_ft**2 * eps**3)
            + 1.75 * rho_lbft3 * u_fts**2 * (1 - eps) / (gc * phi * d_ft * eps**3))   # lbf/ft^3
    lbf_ft3_to_Pa_m = 4.4482216152605 / 0.3048**3
    got, _ = ergun_pressure_gradient(
        u_fts * 0.3048, d_ft * 0.3048, eps, rho_lbft3 * 16.01846337396, mu_lbfts * 1.4881639435695, phi
    )
    assert float(got) == pytest.approx(dpdl * lbf_ft3_to_Pa_m, rel=1e-9)
