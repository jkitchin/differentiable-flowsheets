"""Tests for difflow.particles (drag, terminal velocity, Ergun, fluidization).

What is and is not verified, stated plainly:

- Stokes limit, the Ergun equation (hand arithmetic), the closed-form
  u_mf quadratic (residual of the Ergun balance), and terminal velocity
  (independent scipy ``brentq`` solve of the force balance) are verified
  against first principles, not against a copied table value.
- The drag correlations are checked against the standard sphere drag curve
  by (a) their own limits, (b) the Newton plateau C_D ~ 0.44-0.5 and (c) mutual
  agreement of the Haider-Levenspiel and Turton-Levenspiel fits, which were
  fitted independently to the same data. No textbook worked-example number
  is quoted from memory.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

jax.config.update("jax_enable_x64", True)

from difflow.particles import (
    G_STD,
    archimedes_number,
    bed_expansion,
    drag_coefficient,
    ergun_alpha,
    ergun_pressure_gradient,
    fluidization_window,
    geldart_group,
    hindered_settling_velocity,
    kozeny_carman,
    minimum_fluidization_velocity,
    richardson_zaki_exponent,
    terminal_velocity,
)

RE_GRID = np.logspace(-1, np.log10(2e5), 40)


class TestDrag:
    @pytest.mark.parametrize("method", ["haider_levenspiel", "turton_levenspiel", "schiller_naumann"])
    def test_stokes_limit(self, method):
        Re = 1e-4
        assert float(drag_coefficient(Re, 1.0, method)) * Re / 24.0 == pytest.approx(1.0, abs=1e-3)

    def test_newton_plateau(self):
        # Standard sphere drag curve: C_D ~ 0.38-0.50 for 1e3 < Re < 2e5 (the fits dip to ~0.39 near Re ~ 3e3).
        for Re in np.logspace(3, np.log10(2e5), 10):
            for m in ("haider_levenspiel", "turton_levenspiel", "schiller_naumann"):
                assert 0.38 <= float(drag_coefficient(Re, 1.0, m)) <= 0.50

    def test_correlations_mutually_consistent(self):
        # Independently fitted to the same data; they agree within 8 %
        # over Re 0.1 .. 2e5 (observed max ~ 8 %, correlations' own scatter).
        hl = np.array([float(drag_coefficient(r, 1.0, "haider_levenspiel")) for r in RE_GRID])
        tl = np.array([float(drag_coefficient(r, 1.0, "turton_levenspiel")) for r in RE_GRID])
        assert np.max(np.abs(hl / tl - 1.0)) < 0.08

    def test_schiller_naumann_agrees_below_800(self):
        for r in np.logspace(-1, np.log10(800), 20):
            sn = float(drag_coefficient(r, 1.0, "schiller_naumann"))
            hl = float(drag_coefficient(r, 1.0, "haider_levenspiel"))
            assert abs(sn / hl - 1.0) < 0.12

    def test_nonspherical_drag_higher(self):
        for Re in (1.0, 100.0, 1e4):
            assert float(drag_coefficient(Re, 0.6)) > float(drag_coefficient(Re, 1.0))

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            drag_coefficient(1.0, 1.0, "nope")

    def test_monotone_decreasing_to_plateau(self):
        cd = np.array([float(drag_coefficient(r)) for r in np.logspace(-1, 3, 30)])
        assert np.all(np.diff(cd) < 0)


def _brent_terminal(d, rp, rf, mu, method, phi=1.0):
    Ar = d**3 * rf * (rp - rf) * G_STD / mu**2

    def f(logRe):
        Re = np.exp(logRe)
        return float(drag_coefficient(Re, phi, method)) * Re**2 - 4.0 / 3.0 * Ar

    Re = np.exp(brentq(f, -20, 20, xtol=1e-14, rtol=1e-14))
    return Re * mu / (rf * d)


class TestTerminalVelocity:
    def test_stokes(self):
        d, rp, rf, mu = 2e-6, 2650.0, 998.0, 1e-3
        v = float(terminal_velocity(d, rp, rf, mu))
        assert v == pytest.approx(G_STD * d**2 * (rp - rf) / (18 * mu), rel=1e-3)

    # Stokes, intermediate and Newton regimes (water and air).
    CASES = [
        (5e-6, 2650.0, 998.0, 1e-3),    # Re ~ 1e-5
        (1e-4, 2650.0, 998.0, 1e-3),    # intermediate
        (1e-3, 2650.0, 998.0, 1e-3),    # Re ~ 150
        (1e-2, 2650.0, 998.0, 1e-3),    # Newton-ish, Re ~ 1e4
        (5e-2, 7800.0, 1.2, 1.8e-5),    # steel ball in air, Re ~ 1e5
    ]

    @pytest.mark.parametrize("method", ["haider_levenspiel", "turton_levenspiel", "schiller_naumann"])
    @pytest.mark.parametrize("case", CASES)
    def test_matches_independent_root_find(self, case, method):
        v = float(terminal_velocity(*case, method=method))
        assert v == pytest.approx(_brent_terminal(*case, method), rel=1e-8)

    def test_force_balance_residual(self):
        d, rp, rf, mu = 3e-3, 2500.0, 1.2, 1.8e-5
        v = float(terminal_velocity(d, rp, rf, mu))
        Re = rf * v * d / mu
        cd = float(drag_coefficient(Re))
        drag = cd * 0.5 * rf * v**2 * np.pi * d**2 / 4
        weight = np.pi / 6 * d**3 * (rp - rf) * G_STD
        assert drag == pytest.approx(weight, rel=1e-9)

    def test_sphericity_slows_particle(self):
        a = float(terminal_velocity(1e-3, 2650.0, 998.0, 1e-3, 1.0))
        b = float(terminal_velocity(1e-3, 2650.0, 998.0, 1e-3, 0.7))
        assert b < a

    def test_jit_and_vmap(self):
        d = jnp.array([1e-5, 1e-4, 1e-3, 1e-2])
        v = jax.jit(jax.vmap(lambda x: terminal_velocity(x, 2650.0, 998.0, 1e-3)))(d)
        assert np.all(np.diff(np.asarray(v)) > 0)

    @pytest.mark.parametrize("d", [5e-6, 1e-4, 1e-3, 1e-2, 5e-2])
    def test_grad_d_p_across_regimes(self, d):
        # scaled variable: check_grads' finite-difference step is absolute
        f = lambda s: terminal_velocity(s * d, 2650.0, 998.0, 1e-3)
        check_grads(f, (1.0,), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)

    def test_grad_other_inputs(self):
        f = lambda a, b, c: terminal_velocity(1e-3, 2650.0 * a, 998.0 * b, 1e-3 * c)
        check_grads(f, (1.0, 1.0, 1.0), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)

    def test_grad_matches_stokes_derivative(self):
        # v ~ d^2 in Stokes: dv/dd = 2 v / d
        d = 1e-6
        v = terminal_velocity(d, 2650.0, 998.0, 1e-3)
        g = jax.grad(lambda x: terminal_velocity(x, 2650.0, 998.0, 1e-3))(d)
        assert float(g) == pytest.approx(2 * float(v) / d, rel=2e-3)


class TestHindered:
    def test_rz_exponent_values(self):
        assert float(richardson_zaki_exponent(0.1)) == pytest.approx(4.65)
        assert float(richardson_zaki_exponent(10.0)) == pytest.approx(4.45 * 10.0**-0.1)
        assert float(richardson_zaki_exponent(1000.0)) == pytest.approx(2.39)

    def test_hindered(self):
        assert float(hindered_settling_velocity(1.0, 0.5, n=4.65)) == pytest.approx(0.5**4.65)
        assert float(hindered_settling_velocity(1.0, 1.0, Re_t=1e-3)) == pytest.approx(1.0)

    def test_requires_n_or_re(self):
        with pytest.raises(ValueError):
            hindered_settling_velocity(1.0, 0.5)

    def test_grad(self):
        check_grads(lambda e: hindered_settling_velocity(0.01, e, n=4.65), (0.6,), order=1, modes=["rev"])


class TestErgun:
    # 3 mm spheres, eps = 0.4, air (rho 1.2, mu 1.8e-5), u = 1 m/s: hand arithmetic.
    #   viscous  = 150*(0.6^2/0.4^3)*mu*u/d^2 = 150*5.625*1.8e-5/9e-6 = 1687.5 Pa/m
    #   inertial = 1.75*(0.6/0.064)*rho*u^2/d = 1.75*9.375*1.2/3e-3 = 6562.5 Pa/m
    def test_hand_calculation(self):
        dp, info = ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5)
        assert float(info["viscous"]) == pytest.approx(1687.5, rel=1e-12)
        assert float(info["inertial"]) == pytest.approx(6562.5, rel=1e-12)
        assert float(dp) == pytest.approx(8250.0, rel=1e-12)

    def test_limits(self):
        # Low velocity: viscous only and equal to Kozeny-Carman with 150/36 = 4.1667.
        dp, info = ergun_pressure_gradient(1e-6, 3e-3, 0.4, 1.2, 1.8e-5)
        assert float(info["inertial"]) / float(dp) < 1e-3
        kc = kozeny_carman(1e-6, 3e-3, 0.4, 1.8e-5, k0=150.0 / 36.0)
        assert float(kc) == pytest.approx(float(info["viscous"]), rel=1e-12)
        # high velocity: inertial dominates and scales as u^2
        d1, i1 = ergun_pressure_gradient(100.0, 3e-3, 0.4, 1.2, 1.8e-5)
        assert float(i1["inertial"]) / float(d1) > 0.99

    def test_kozeny_carman_prefactor(self):
        v = kozeny_carman(1.0, 1e-3, 0.4, 1e-3)
        assert float(v) == pytest.approx(180 * 0.36 / 0.064 * 1e-3 / 1e-6, rel=1e-12)

    def test_sphericity(self):
        a, _ = ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5, 0.5)
        b, _ = ergun_pressure_gradient(1.0, 6e-3, 0.4, 1.2, 1.8e-5, 0.25)
        assert float(a) > float(ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5)[0])
        assert float(a) == pytest.approx(float(ergun_pressure_gradient(1.0, 1.5e-3, 0.4, 1.2, 1.8e-5)[0]))
        assert float(b) == pytest.approx(float(a))  # only phi*d_p enters

    def test_grads(self):
        f = lambda d, e: ergun_pressure_gradient(1.0, d, e, 1.2, 1.8e-5)[0]
        fs = lambda s, e: f(s * 3e-3, e)   # scaled: FD step is absolute
        check_grads(fs, (1.0, 0.4), order=1, modes=["rev"], rtol=1e-6)
        g = jax.grad(f, argnums=(0, 1))(3e-3, 0.4)
        assert float(g[0]) < 0 and float(g[1]) < 0  # larger particles/voids: lower drop

    def test_kc_grads(self):
        check_grads(lambda s, e: kozeny_carman(0.01, s * 1e-3, e, 1e-3), (1.0, 0.4), order=1, modes=["rev"])


class TestFluidization:
    AIR = (300e-6, 2600.0, 1.2, 1.8e-5)

    def test_ergun_balance_at_umf(self):
        # Independent check: the Ergun gradient at u_mf equals the bed weight.
        d, rp, rf, mu = self.AIR
        for eps, phi in [(0.45, 1.0), (0.45, 0.8), (0.40, 0.7)]:
            u = float(minimum_fluidization_velocity(d, rp, rf, mu, eps, phi))
            dp, _ = ergun_pressure_gradient(u, d, eps, rf, mu, phi)
            assert float(dp) == pytest.approx((1 - eps) * (rp - rf) * G_STD, rel=1e-10)

    def test_scipy_root_matches_closed_form(self):
        d, rp, rf, mu = self.AIR
        eps, phi = 0.45, 0.8
        w = (1 - eps) * (rp - rf) * G_STD
        u = brentq(lambda x: float(ergun_pressure_gradient(x, d, eps, rf, mu, phi)[0]) - w, 1e-6, 10.0, xtol=1e-14)
        assert float(minimum_fluidization_velocity(d, rp, rf, mu, eps, phi)) == pytest.approx(u, rel=1e-9)

    def test_wen_yu_close_to_ergun(self):
        # Wen & Yu is a fit to many beds (eps_mf, phi folded in); with phi ~ 0.8
        # and eps_mf ~ 0.45 it is within ~20 % of the Ergun form for this case.
        d, rp, rf, mu = self.AIR
        a = float(minimum_fluidization_velocity(d, rp, rf, mu, 0.45, 0.8, "ergun"))
        b = float(minimum_fluidization_velocity(d, rp, rf, mu, method="wen_yu"))
        assert abs(a / b - 1.0) < 0.20

    def test_wen_yu_formula(self):
        d, rp, rf, mu = self.AIR
        Ar = float(archimedes_number(d, rp, rf, mu))
        re = np.sqrt(33.7**2 + 0.0408 * Ar) - 33.7
        assert float(minimum_fluidization_velocity(d, rp, rf, mu, method="wen_yu")) == pytest.approx(
            re * mu / (rf * d), rel=1e-12
        )

    def test_bad_method(self):
        with pytest.raises(ValueError):
            minimum_fluidization_velocity(*self.AIR, method="x")

    def test_grad_d_p_and_regimes(self):
        # fine (viscous), medium and coarse (inertial) particles
        for d in (30e-6, 300e-6, 3e-3):
            f = lambda s: minimum_fluidization_velocity(s * d, 2600.0, 1.2, 1.8e-5, 0.45, 0.8)
            check_grads(f, (1.0,), order=1, modes=["rev"], rtol=1e-6)
            fw = lambda s: minimum_fluidization_velocity(s * d, 2600.0, 1.2, 1.8e-5, method="wen_yu")
            check_grads(fw, (1.0,), order=1, modes=["rev"], rtol=1e-6)

    def test_window_and_expansion(self):
        u_mf, u_t = fluidization_window(*self.AIR, 0.45, 0.8)
        assert float(u_mf) < float(u_t)
        u_mf, u_t = float(u_mf), float(u_t)
        n = 2.39
        eps, H = bed_expansion(0.5 * u_t, u_t, n=n, voidage_mf=0.45, H_mf=1.0)
        assert float(eps) == pytest.approx(0.5 ** (1 / n))
        assert float(H) == pytest.approx(0.55 / (1 - float(eps)))
        # at u = u_t the voidage is 1 (entrainment)
        assert float(bed_expansion(u_t, u_t, n=n)[0]) == pytest.approx(1.0)

    def test_expansion_grad(self):
        f = lambda u: bed_expansion(u, 1.0, n=2.39)[1]
        check_grads(f, (0.5,), order=1, modes=["rev"], rtol=1e-6)

    def test_geldart(self):
        assert geldart_group(300e-6, 2600.0, 1.2) == "B"
        assert geldart_group(60e-6, 1000.0, 1.2) == "A"
        assert geldart_group(5e-6, 2000.0, 1.0) == "C"
        assert geldart_group(3e-3, 2600.0, 1.2) == "D"


class TestGasPFRPhysicalBed:
    """GasPFR with physical bed parameters reproduces a known alpha."""

    def _params(self, **kw):
        from difflow.units.pfr import GasPFRParams

        return GasPFRParams(
            V=1.0,
            rate_fn=lambda C, T, p: jnp.zeros(1),
            stoich=jnp.array([[-1.0], [1.0]]),
            rate_params={},
            species_order=["A", "B"],
            **kw,
        )

    BED = dict(d_p=3e-3, voidage=0.4, mu=1.8e-5, rho_gas=1.2, u_s0=1.0, bed_area=0.5)

    def test_alpha_known_value(self):
        # Hand-computed Ergun gradient 8250 Pa/m (see TestErgun) over area 0.5 m^2.
        p = self._params(**self.BED)
        assert float(p.effective_alpha) == pytest.approx(8250.0 / 0.5, rel=1e-12)
        assert float(ergun_alpha(1.0, 3e-3, 0.4, 1.2, 1.8e-5, 0.5)) == pytest.approx(16500.0, rel=1e-12)

    def test_backward_compatible_alpha(self):
        assert self._params(alpha=123.0).effective_alpha == 123.0
        assert self._params().effective_alpha == 0.0

    def test_reactor_pressure_drop_matches_explicit_alpha(self):
        from difflow import make_stream
        from difflow.units.pfr import GasPFR

        inlet = make_stream({"A": 1.0, "B": 0.0}, 400.0, 2.0e5)
        _, info_a = GasPFR(self._params(alpha=16500.0))(inlet)
        _, info_b = GasPFR(self._params(**self.BED))(inlet)
        assert float(info_b["pressure_drop"]) == pytest.approx(float(info_a["pressure_drop"]), rel=1e-10)
        assert float(info_b["pressure_drop"]) > 0

    def test_update_physical_parameter(self):
        p = self._params(**self.BED)
        q = p.update(d_p=6e-3)
        assert float(q.effective_alpha) < float(p.effective_alpha)

    def test_validation(self):
        with pytest.raises(ValueError):
            self._params(alpha=1.0, **self.BED)
        with pytest.raises(ValueError):
            self._params(d_p=3e-3)

    def test_alpha_grad_wrt_d_p(self):
        f = lambda s: ergun_alpha(1.0, s * 3e-3, 0.4, 1.2, 1.8e-5, 0.5)
        check_grads(f, (1.0,), order=1, modes=["rev"])
