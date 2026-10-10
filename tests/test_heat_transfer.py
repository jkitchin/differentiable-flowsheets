"""Tests for difflow.heat_transfer: film coefficients, phase change and overall U.

What "validated" means here -- please read before trusting a number.

No Incropera/Bergman or Cengel *worked-example number* is quoted: none could
be reproduced from memory with confidence, and a fabricated textbook value is
worse than none. Instead:

* every correlation is compared with an **independent re-implementation of the
  stated formula** written out in plain NumPy/SciPy in this file (hand
  calculation), to 1e-10;
* where a physical derivation exists, the correlation's *constant* is checked
  against it numerically: Nusselt's 0.943 (film theory integrated with
  ``scipy.integrate.quad``), the cylindrical-shell resistance (``quad`` of
  Fourier's law), the straight-fin efficiency (exact convective-tip fin
  solution), the critical insulation radius (``jax.grad`` of the heat loss);
* sanity bands (Gnielinski vs Dittus-Boelter, Rohsenow vs Mostinski, Zuber
  CHF of water ~1.1 MW/m^2, Nu = 3.66/4.36) are stated with their tolerance;
* overall U is checked against the resistance sum, ``U_o A_o = U_i A_i`` and
  its limits, *not* against a published shell-and-tube example.

Property values for water are rounded textbook-style numbers used as inputs
(they only need to be self-consistent); no test asserts a textbook output.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.integrate import quad
from scipy.optimize import brentq

from difflow import heat_transfer as ht
from difflow.fluids import friction_factor

# water, ~40 C (inputs only)
RHO, MU, CP, K = 992.0, 653e-6, 4179.0, 0.631
PR = CP * MU / K


# -- dimensionless groups ---------------------------------------------------

class TestGroups:
    def test_reynolds_prandtl_nusselt(self):
        assert float(ht.reynolds(RHO, 1.5, 0.025, MU)) == pytest.approx(RHO * 1.5 * 0.025 / MU, rel=1e-14)
        assert float(ht.prandtl(CP, MU, K)) == pytest.approx(PR, rel=1e-14)
        assert float(ht.nusselt_to_h(100.0, K, 0.025)) == pytest.approx(100 * K / 0.025, rel=1e-14)

    def test_grashof_rayleigh(self):
        beta, dT, L, nu, alpha = 3.0e-3, 20.0, 0.5, 1.5e-5, 2.1e-5
        gr = 9.80665 * beta * dT * L**3 / nu**2
        assert float(ht.grashof(beta, dT, L, nu)) == pytest.approx(gr, rel=1e-13)
        assert float(ht.rayleigh(beta, dT, L, nu, alpha)) == pytest.approx(gr * nu / alpha, rel=1e-13)
        # |dT|: sign of the temperature difference does not matter
        assert float(ht.grashof(beta, -dT, L, nu)) == pytest.approx(gr, rel=1e-13)


# -- conduction ---------------------------------------------------------------

class TestConduction:
    def test_slab(self):
        assert float(ht.slab_resistance(0.01, 0.5, 2.0)) == pytest.approx(0.01, rel=1e-14)
        assert float(ht.slab_resistance(0.01, 0.5)) == pytest.approx(0.02, rel=1e-14)

    def test_cylinder_vs_numerical_fourier(self):
        r1, r2, k, L = 0.02, 0.05, 0.04, 3.0
        # R = int dr / (k 2 pi r L)
        R_num, _ = quad(lambda r: 1.0 / (k * 2 * np.pi * r * L), r1, r2)
        assert float(ht.cylinder_resistance(r1, r2, k, L)) == pytest.approx(R_num, rel=1e-10)

    def test_sphere_vs_numerical_fourier(self):
        r1, r2, k = 0.02, 0.05, 0.04
        R_num, _ = quad(lambda r: 1.0 / (k * 4 * np.pi * r**2), r1, r2)
        assert float(ht.sphere_resistance(r1, r2, k)) == pytest.approx(R_num, rel=1e-10)

    def test_composite(self):
        assert float(ht.composite_wall([1.0, 2.0, 3.5])) == pytest.approx(6.5, rel=1e-14)
        assert float(ht.composite_wall(jnp.array([1.0, 2.0]))) == pytest.approx(3.0, rel=1e-14)

    def test_critical_radius_is_the_maximum_of_heat_loss(self):
        k, h, r_i, L, dT = 0.05, 10.0, 0.005, 1.0, 80.0

        def q(r_o):
            R = ht.cylinder_resistance(r_i, r_o, k, L) + 1.0 / (h * 2 * jnp.pi * r_o * L)
            return dT / R

        rc = float(ht.critical_insulation_radius(k, h))
        assert rc == pytest.approx(k / h, rel=1e-14)
        assert float(jax.grad(q)(rc)) == pytest.approx(0.0, abs=1e-9)
        assert float(jax.grad(jax.grad(q))(rc)) < 0.0   # a maximum
        assert float(q(rc)) > float(q(r_i * 1.0001)) and float(q(rc)) > float(q(3 * rc))
        assert float(ht.critical_insulation_radius(k, h, "sphere")) == pytest.approx(2 * k / h, rel=1e-14)
        with pytest.raises(ValueError):
            ht.critical_insulation_radius(k, h, "cube")

    @pytest.mark.parametrize("h,k,t,L", [(50, 200, 0.002, 0.03), (100, 50, 0.003, 0.02), (25, 200, 0.001, 0.05)])
    def test_fin_efficiency_vs_exact_convective_tip(self, h, k, t, L):
        # Exact 1-D fin with a convecting tip, efficiency on the area 2L + t
        # (tip included): corrected-length result must agree to 1e-3.
        m = np.sqrt(2 * h / (k * t))
        Bi = h / (m * k)
        th = np.tanh(m * L)
        q = k * t * m * (th + Bi) / (1 + Bi * th)
        eta_exact = q / (h * (2 * L + t))
        assert float(ht.fin_efficiency_straight(h, k, t, L)) == pytest.approx(eta_exact, rel=1e-3)

    def test_fin_limits(self):
        assert float(ht.fin_efficiency_straight(1e-6, 200.0, 0.002, 0.03)) == pytest.approx(1.0, abs=1e-6)
        assert float(ht.fin_efficiency_straight(1e5, 1.0, 0.002, 1.0)) < 0.05


# -- internal flow ------------------------------------------------------------

def _gnielinski_np(Re, Pr, f):
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


class TestInternal:
    def test_dittus_boelter_hand(self):
        Re, Pr = 5e4, 4.3
        assert float(ht.dittus_boelter(Re, Pr, True)) == pytest.approx(0.023 * Re**0.8 * Pr**0.4, rel=1e-12)
        assert float(ht.dittus_boelter(Re, Pr, False)) == pytest.approx(0.023 * Re**0.8 * Pr**0.3, rel=1e-12)

    def test_sieder_tate_hand(self):
        Re, Pr, mr = 3e4, 80.0, 1.7
        assert float(ht.sieder_tate(Re, Pr, mr)) == pytest.approx(0.027 * Re**0.8 * Pr ** (1 / 3) * mr**0.14, rel=1e-12)
        assert float(ht.sieder_tate(Re, Pr)) == pytest.approx(0.027 * Re**0.8 * Pr ** (1 / 3), rel=1e-12)

    def test_sieder_tate_laminar_hand(self):
        Re, Pr, DL, mr = 800.0, 6.0, 0.02, 0.8
        assert float(ht.sieder_tate_laminar(Re, Pr, DL, mr)) == pytest.approx(
            1.86 * (Re * Pr * DL) ** (1 / 3) * mr**0.14, rel=1e-12)

    def test_gnielinski_hand_and_petukhov(self):
        Re, Pr = 5e4, 4.3
        f = 0.0206
        assert float(ht.gnielinski(Re, Pr, f)) == pytest.approx(_gnielinski_np(Re, Pr, f), rel=1e-12)

    @pytest.mark.parametrize("Re", [1e4, 3e4, 1e5])
    @pytest.mark.parametrize("Pr", [0.7, 5.0, 50.0])
    def test_gnielinski_close_to_dittus_boelter(self, Re, Pr):
        # Sanity band (not a validation): the two classical turbulent
        # correlations agree within 20% over Re 1e4-1e5, Pr 0.7-50
        # (they drift apart above ~1e5, where Dittus-Boelter is outside its usual range).
        Nu_g = float(ht.gnielinski(Re, Pr, friction_factor(Re, 0.0)))
        Nu_d = float(ht.dittus_boelter(Re, Pr, True))
        assert Nu_g == pytest.approx(Nu_d, rel=0.20)

    def test_laminar_fully_developed_values(self):
        for Re in (10.0, 500.0, 2000.0, 2300.0):
            assert float(ht.internal_nusselt(Re, 7.0)) == 3.66
            assert float(ht.internal_nusselt(Re, 7.0, boundary="q")) == 4.36

    def test_laminar_entry_floor_and_growth(self):
        short = float(ht.internal_nusselt(1000.0, 7.0, D_over_L=0.2))
        long_ = float(ht.internal_nusselt(1000.0, 7.0, D_over_L=1e-9))
        assert long_ == pytest.approx(3.66, rel=1e-5)       # fully developed floor
        assert short > long_
        en = 1.86 * (1000.0 * 7.0 * 0.2) ** (1 / 3)
        assert short == pytest.approx((3.66**3 + en**3) ** (1 / 3), rel=1e-12)

    @pytest.mark.parametrize("method,ref", [
        ("gnielinski", lambda Re, Pr: _gnielinski_np(Re, Pr, float(friction_factor(Re, 0.0)))),
        ("dittus_boelter", lambda Re, Pr: 0.023 * Re**0.8 * Pr**0.4),
        ("sieder_tate", lambda Re, Pr: 0.027 * Re**0.8 * Pr ** (1 / 3)),
    ])
    def test_turbulent_limit_is_the_named_correlation(self, method, ref):
        for Re in (4000.0, 2e4, 3e5):
            assert float(ht.internal_nusselt(Re, 5.0, method=method)) == pytest.approx(ref(Re, 5.0), rel=1e-12)

    def test_internal_h_is_nusselt_k_over_D(self):
        Re = float(ht.reynolds(RHO, 1.5, 0.025, MU))
        Nu = float(ht.internal_nusselt(Re, PR, 0.025 / 3.0))
        assert float(ht.internal_h(Re, PR, K, 0.025, L=3.0)) == pytest.approx(Nu * K / 0.025, rel=1e-13)

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            ht.internal_nusselt(1e4, 5.0, method="nope")
        with pytest.raises(ValueError):
            ht.internal_nusselt(1e4, 5.0, boundary="x")

    def test_transition_is_continuous_monotone_and_bounded(self):
        Re = np.linspace(2000.0, 10000.0, 4001)
        for Pr in (0.7, 5.0, 100.0):
            Nu = np.asarray(ht.internal_nusselt(jnp.asarray(Re), Pr))
            assert np.all(np.isfinite(Nu))
            assert np.all(np.diff(Nu) >= -1e-12)           # non-decreasing in Re
            # no jump: the largest step between neighbours is small relative to the range
            assert np.max(np.diff(Nu)) < 0.01 * (Nu[-1] - Nu[0])
            assert Nu[0] == 3.66

    def test_traced_Re_has_no_python_branching(self):
        f = jax.jit(lambda Re: ht.internal_h(Re, PR, K, 0.025, L=5.0))
        for Re in (500.0, 3000.0, 3e4):
            assert np.isfinite(float(f(Re)))
        assert float(f(500.0)) < float(f(3000.0)) < float(f(3e4))

    @pytest.mark.parametrize("method", ["gnielinski", "dittus_boelter", "sieder_tate"])
    def test_check_grads_across_the_transition(self, method):
        # Re 2000-10000 spans laminar, blend and turbulent. check_grads w.r.t. Re and Pr.
        for Re0 in (2000.0, 2150.0, 2500.0, 3000.0, 3300.0, 3700.0, 4300.0, 5500.0, 8000.0, 10000.0):
            for Pr0 in (0.7, 5.0, 40.0):
                # scale factors (O(1) inputs); order 1 because the Colebrook friction factor inside
                # Gnielinski carries exact first (implicit-function) but not exact second derivatives
                fn = lambda a, b: ht.internal_nusselt(Re0 * a, Pr0 * b, method=method)
                check_grads(fn, (jnp.asarray(1.0), jnp.asarray(1.0)), order=1, modes=["fwd", "rev"],
                            atol=1e-6, rtol=1e-4)

    def test_gradient_is_exactly_continuous_at_the_blend_ends(self):
        # At 2300 and 4000 the weight and its first two derivatives vanish; a finite-difference
        # check there only sees the third-derivative kink of the smootherstep, so test the
        # analytic gradient directly: left and right limits agree and the Re-gradient is
        # zero at 2300 (laminar, fully developed) and continuous at 4000.
        g = jax.grad(lambda Re: ht.internal_nusselt(Re, 5.0))
        assert float(g(2300.0)) == 0.0
        assert float(g(2300.0 * (1 + 1e-9))) == pytest.approx(0.0, abs=1e-6)
        assert float(g(4000.0 * (1 - 1e-9))) == pytest.approx(float(g(4000.0 * (1 + 1e-9))), rel=1e-6)
        assert float(g(4000.0 * (1 + 1e-9))) > 0

    def test_check_grads_laminar_entry(self):
        for Re0 in (300.0, 2000.0, 3000.0, 9000.0):
            fn = lambda a, b: ht.internal_nusselt(Re0 * a, 6.0 * b, D_over_L=0.05, mu_ratio=1.1)
            check_grads(fn, (jnp.asarray(1.0), jnp.asarray(1.0)), order=1, modes=["fwd", "rev"],
                        atol=1e-6, rtol=1e-4)

    def test_grad_of_U_wrt_velocity_and_diameter(self):
        def U(v, D):
            Re = ht.reynolds(RHO, v, D, MU)
            h_i = ht.internal_h(Re, PR, K, D, L=4.0)
            return ht.overall_U(h_i, 1500.0, D, D + 0.004, 16.0, R_fi=1e-4)

        for v0 in (0.05, 0.3, 1.5, 3.0):          # laminar, blend and turbulent
            for D0 in (0.016, 0.025):
                gv, gD = jax.grad(U, argnums=(0, 1))(v0, D0)
                ev, eD = 1e-6 * v0, 1e-7
                fd_v = (U(v0 + ev, D0) - U(v0 - ev, D0)) / (2 * ev)
                fd_D = (U(v0, D0 + eD) - U(v0, D0 - eD)) / (2 * eD)
                assert float(gv) == pytest.approx(float(fd_v), rel=1e-5, abs=1e-8)
                assert float(gD) == pytest.approx(float(fd_D), rel=1e-5, abs=1e-8)
        # turbulent: U increases with velocity
        assert float(jax.grad(U)(1.5, 0.025)) > 0.0
        check_grads(U, (jnp.asarray(1.5), jnp.asarray(0.025)), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)


# -- external ----------------------------------------------------------------

class TestExternal:
    def test_churchill_bernstein_hand(self):
        Re, Pr = 2.0e4, 0.71
        ref = 0.3 + (0.62 * Re**0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25) * (
            1 + (Re / 282000) ** (5 / 8)) ** (4 / 5)
        assert float(ht.churchill_bernstein(Re, Pr)) == pytest.approx(ref, rel=1e-12)

    def test_churchill_bernstein_low_Re_limit(self):
        # Nu -> 0.3 + small as Re -> 0 (conduction-like floor of the correlation)
        assert float(ht.churchill_bernstein(1e-6, 0.7)) == pytest.approx(0.3, abs=2e-3)

    def test_flat_plate_hand(self):
        Re, Pr = 2e5, 0.7
        assert float(ht.flat_plate_nusselt(Re, Pr, "laminar")) == pytest.approx(0.664 * Re**0.5 * Pr ** (1 / 3), rel=1e-13)
        assert float(ht.flat_plate_nusselt(Re, Pr, "turbulent")) == pytest.approx(0.037 * Re**0.8 * Pr ** (1 / 3), rel=1e-13)
        # mixed = laminar up to Re_c = 5e5, blended to (0.037 Re^0.8 - 871) Pr^(1/3), which holds from 5.5e5
        assert float(ht.flat_plate_nusselt(Re, Pr)) == pytest.approx(0.664 * Re**0.5 * Pr ** (1 / 3), rel=1e-13)
        assert float(ht.flat_plate_nusselt(5e5, Pr)) == pytest.approx(0.664 * 5e5**0.5 * Pr ** (1 / 3), rel=1e-13)
        assert float(ht.flat_plate_nusselt(5e5, Pr)) == pytest.approx((0.037 * 5e5**0.8 - 871) * Pr ** (1 / 3), rel=1e-2)
        for Re in (5.5e5, 1e6, 5e7):
            assert float(ht.flat_plate_nusselt(Re, Pr)) == pytest.approx((0.037 * Re**0.8 - 871) * Pr ** (1 / 3), rel=1e-12)
        with pytest.raises(ValueError):
            ht.flat_plate_nusselt(1e5, 0.7, "bad")

    def test_flat_plate_mixed_continuous_and_differentiable(self):
        Re = np.linspace(2e5, 1.5e6, 3001)
        Nu = np.asarray(ht.flat_plate_nusselt(jnp.asarray(Re), 0.7))
        assert np.all(np.diff(Nu) > 0)
        assert np.max(np.diff(Nu)) < 0.01 * (Nu[-1] - Nu[0])
        for Re0 in (3e5, 4.5e5, 5e5, 6e5):
            check_grads(lambda r: ht.flat_plate_nusselt(r, 0.7), (jnp.asarray(Re0),), order=2, modes=["fwd", "rev"],
                        atol=1e-6, rtol=1e-6)


# -- natural convection --------------------------------------------------------

class TestNaturalConvection:
    def test_vertical_plate_hand(self):
        Ra, Pr = 1e9, 0.71
        ref = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
        assert float(ht.churchill_chu_vertical_plate(Ra, Pr)) == pytest.approx(ref, rel=1e-12)

    def test_horizontal_cylinder_hand(self):
        Ra, Pr = 1e6, 0.71
        ref = (0.60 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
        assert float(ht.churchill_chu_horizontal_cylinder(Ra, Pr)) == pytest.approx(ref, rel=1e-12)

    def test_grad_wrt_Ra_and_Pr(self):
        for f in (ht.churchill_chu_vertical_plate, ht.churchill_chu_horizontal_cylinder):
            check_grads(f, (jnp.asarray(1e7), jnp.asarray(0.71)), order=2, modes=["fwd", "rev"], atol=1e-5, rtol=1e-5)


# -- phase change -------------------------------------------------------------

# water at ~100 C (inputs only)
WATER_SAT = dict(mu_l=279e-6, h_fg=2257e3, rho_l=957.9, rho_v=0.5955, sigma=58.9e-3, Cp_l=4217.0, Pr_l=1.76)


class TestCondensation:
    P = dict(k_l=0.68, rho_l=957.9, rho_v=0.5955, mu_l=279e-6, h_fg=2257e3, Cp_l=4217.0)

    def test_vertical_hand(self):
        dT, L = 8.0, 0.5
        hp = 2257e3 + 0.68 * 4217.0 * dT
        ref = 0.943 * (9.80665 * 957.9 * (957.9 - 0.5955) * 0.68**3 * hp / (279e-6 * dT * L)) ** 0.25
        assert float(ht.nusselt_film_condensation(dT=dT, L=L, geometry="vertical_plate", **self.P)) == pytest.approx(ref, rel=1e-12)

    def test_horizontal_hand_and_tube_bank(self):
        dT, D = 10.0, 0.0254
        hp = 2257e3 + 0.68 * 4217.0 * dT
        ref = 0.729 * (9.80665 * 957.9 * (957.9 - 0.5955) * 0.68**3 * hp / (279e-6 * dT * D)) ** 0.25
        h1 = float(ht.nusselt_film_condensation(dT=dT, L=D, **self.P))
        assert h1 == pytest.approx(ref, rel=1e-12)
        h10 = float(ht.nusselt_film_condensation(dT=dT, L=D, n_tubes=10, **self.P))
        assert h10 == pytest.approx(h1 * 10 ** -0.25, rel=1e-12)

    def test_vertical_constant_from_film_theory(self):
        # Nusselt film: delta(x) = [4 mu k dT x / (g rho (rho - rho_v) h')]^(1/4), h_x = k / delta.
        # The plate-mean coefficient must equal 0.943 [...]^(1/4) (derivation, not a table value).
        dT, L = 8.0, 0.5
        hp = 2257e3 + 0.68 * 4217.0 * dT
        p = self.P
        c = 4 * p["mu_l"] * p["k_l"] * dT / (9.80665 * p["rho_l"] * (p["rho_l"] - p["rho_v"]) * hp)
        h_mean, _ = quad(lambda x: p["k_l"] / (c * x) ** 0.25, 0.0, L)
        h_mean /= L
        got = float(ht.nusselt_film_condensation(dT=dT, L=L, geometry="vertical_plate", **p))
        assert got == pytest.approx(h_mean, rel=1e-3)   # 0.943 vs the exact 0.9428

    def test_horizontal_to_vertical_ratio(self):
        # Nusselt: 0.729/0.943 for the same length scale
        a = float(ht.nusselt_film_condensation(dT=5.0, L=0.03, geometry="horizontal_tube", **self.P))
        b = float(ht.nusselt_film_condensation(dT=5.0, L=0.03, geometry="vertical_plate", **self.P))
        assert a / b == pytest.approx(0.729 / 0.943, rel=1e-12)

    def test_steam_on_horizontal_tube_magnitude(self):
        # Sanity band only (no textbook value claimed): 1 atm steam on a 25 mm tube
        # with 10 K driving force gives ~1e4 W/m^2/K in every textbook.
        h = float(ht.nusselt_film_condensation(dT=10.0, L=0.025, **self.P))
        assert 8e3 < h < 2e4

    def test_grad_and_errors(self):
        f = lambda a, b: ht.nusselt_film_condensation(dT=10.0 * a, L=0.025 * b, **self.P)
        check_grads(f, (jnp.asarray(1.0), jnp.asarray(1.0)), order=2, modes=["fwd", "rev"], atol=1e-5, rtol=1e-4)
        with pytest.raises(ValueError):
            ht.nusselt_film_condensation(dT=5.0, L=0.03, geometry="sphere", **self.P)


class TestBoiling:
    def test_rohsenow_hand(self):
        dT = 10.0
        p = WATER_SAT
        ref = p["mu_l"] * p["h_fg"] * np.sqrt(9.80665 * (p["rho_l"] - p["rho_v"]) / p["sigma"]) * (
            p["Cp_l"] * dT / (0.013 * p["h_fg"] * p["Pr_l"])) ** 3
        assert float(ht.rohsenow_heat_flux(dT, **p)) == pytest.approx(ref, rel=1e-12)

    def test_superheat_is_inverse_of_flux(self):
        for q in (2e4, 1e5, 6e5):
            dT = float(ht.rohsenow_superheat(q, **WATER_SAT))
            assert float(ht.rohsenow_heat_flux(dT, **WATER_SAT)) == pytest.approx(q, rel=1e-12)
            root = brentq(lambda x: float(ht.rohsenow_heat_flux(x, **WATER_SAT)) - q, 0.1, 100.0, xtol=1e-12)
            assert dT == pytest.approx(root, rel=1e-9)
        # non-water exponent
        dT = float(ht.rohsenow_superheat(1e5, n=1.7, C_sf=0.006, **WATER_SAT))
        assert float(ht.rohsenow_heat_flux(dT, n=1.7, C_sf=0.006, **WATER_SAT)) == pytest.approx(1e5, rel=1e-12)

    def test_water_at_1atm_magnitude(self):
        # Sanity band: nucleate boiling of water at 1 atm, q = 1e5 W/m^2 needs a single-digit
        # to low-teens excess temperature in practice (Nukiyama curve).
        dT = float(ht.rohsenow_superheat(1e5, **WATER_SAT))
        assert 5.0 < dT < 15.0

    def test_mostinski_hand_and_agreement_with_rohsenow(self):
        q, Pc, P = 1e5, 22064.0, 101.325
        pr = P / Pc
        ref = 0.00417 * Pc**0.69 * q**0.7 * (1.8 * pr**0.17 + 4 * pr**1.2 + 10 * pr**10)
        assert float(ht.mostinski(q, Pc, P)) == pytest.approx(ref, rel=1e-12)
        h_r = q / float(ht.rohsenow_superheat(q, **WATER_SAT))
        # the two empirical correlations are good to ~+-30% each; stated band: within 30% of each other
        assert float(ht.mostinski(q, Pc, P)) == pytest.approx(h_r, rel=0.30)

    def test_rohsenow_h_scales_as_q_two_thirds(self):
        h = lambda q: q / float(ht.rohsenow_superheat(q, **WATER_SAT))
        assert h(8e4) / h(1e4) == pytest.approx(8 ** (2 / 3), rel=1e-12)

    def test_critical_heat_flux_hand_and_water(self):
        p = WATER_SAT
        ref = (np.pi / 24) * p["h_fg"] * np.sqrt(p["rho_v"]) * (p["sigma"] * 9.80665 * (p["rho_l"] - p["rho_v"])) ** 0.25
        q = float(ht.critical_heat_flux(p["h_fg"], p["rho_v"], p["rho_l"], p["sigma"]))
        assert q == pytest.approx(ref, rel=1e-12)
        assert q == pytest.approx(1.1e6, rel=0.05)      # water at 1 atm is commonly quoted as ~1.1 MW/m^2
        q149 = float(ht.critical_heat_flux(p["h_fg"], p["rho_v"], p["rho_l"], p["sigma"], C=0.149))
        assert q149 / q == pytest.approx(0.149 / (np.pi / 24), rel=1e-12)

    def test_rohsenow_below_chf_at_operating_point(self):
        dT = float(ht.rohsenow_superheat(1e5, **WATER_SAT))
        assert float(ht.rohsenow_heat_flux(dT, **WATER_SAT)) < float(
            ht.critical_heat_flux(2257e3, 0.5955, 957.9, 58.9e-3))

    def test_grads(self):
        args = tuple(WATER_SAT.values())
        f = lambda q, sigma: ht.rohsenow_superheat(q, 279e-6, 2257e3, 957.9, 0.5955, sigma, 4217.0, 1.76)
        check_grads(f, (jnp.asarray(1e5), jnp.asarray(58.9e-3)), order=2, modes=["fwd", "rev"], atol=1e-4, rtol=1e-6)
        check_grads(lambda q, P: ht.mostinski(q, 22064.0, P), (jnp.asarray(1e5), jnp.asarray(101.325)),
                    order=2, modes=["fwd", "rev"], atol=1e-4, rtol=1e-6)
        assert len(args) == 7


# -- overall U -----------------------------------------------------------------

class TestOverallU:
    D_i, D_o, k_w = 0.0229, 0.0254, 16.0
    h_i, h_o, R_fi, R_fo = 4500.0, 350.0, 1.76e-4, 5.3e-4   # illustrative inputs

    def _hand_outer(self):
        A_o_per_L = np.pi * self.D_o
        A_i_per_L = np.pi * self.D_i
        R_total_per_L = (1 / (self.h_i * A_i_per_L) + self.R_fi / A_i_per_L
                         + np.log(self.D_o / self.D_i) / (2 * np.pi * self.k_w)
                         + self.R_fo / A_o_per_L + 1 / (self.h_o * A_o_per_L))
        return 1.0 / (R_total_per_L * A_o_per_L), 1.0 / (R_total_per_L * A_i_per_L)

    def test_outer_and_inner_vs_resistance_network(self):
        # hand calculation from the series network written in terms of areas (a different route
        # from the closed form in the module)
        Uo, Ui = self._hand_outer()
        args = (self.h_i, self.h_o, self.D_i, self.D_o, self.k_w)
        kw = dict(R_fi=self.R_fi, R_fo=self.R_fo)
        assert float(ht.overall_U(*args, basis="outer", **kw)) == pytest.approx(Uo, rel=1e-12)
        assert float(ht.overall_U(*args, basis="inner", **kw)) == pytest.approx(Ui, rel=1e-12)

    def test_UA_is_basis_independent(self):
        args = (self.h_i, self.h_o, self.D_i, self.D_o, self.k_w)
        Uo = float(ht.overall_U(*args, R_fi=self.R_fi, R_fo=self.R_fo))
        Ui = float(ht.overall_U(*args, R_fi=self.R_fi, R_fo=self.R_fo, basis="inner"))
        assert Uo * self.D_o == pytest.approx(Ui * self.D_i, rel=1e-13)

    def test_limits(self):
        # flat wall, no fouling: 1/U = 1/hi + 1/ho (+ t/k)
        assert float(ht.overall_U(1000.0, 250.0, 0.02, 0.02, 40.0)) == pytest.approx(200.0, rel=1e-13)
        assert float(ht.overall_U(1000.0, 250.0, 0.02, 0.02, jnp.inf)) == pytest.approx(200.0, rel=1e-13)
        # fouling lowers U, adding exactly its resistance on the same area
        U0 = float(ht.overall_U(1000.0, 250.0, 0.02, 0.02, jnp.inf))
        U1 = float(ht.overall_U(1000.0, 250.0, 0.02, 0.02, jnp.inf, R_fi=1e-3, R_fo=2e-3))
        assert 1 / U1 == pytest.approx(1 / U0 + 3e-3, rel=1e-13)
        # a controlling film: U -> h_o
        assert float(ht.overall_U(1e9, 100.0, 0.02, 0.025, 400.0)) == pytest.approx(100.0, rel=2e-3)

    def test_bad_basis(self):
        with pytest.raises(ValueError):
            ht.overall_U(1.0, 1.0, 1.0, 1.0, 1.0, basis="mean")

    def test_grads(self):
        f = lambda a, b, c: ht.overall_U(4000.0 * a, 900.0 * b, 0.02 * c, 0.02 * c + 0.003, 16.0, R_fi=1e-4)
        check_grads(f, (jnp.asarray(1.0),) * 3, order=2, modes=["fwd", "rev"], atol=1e-5, rtol=1e-4)

    def test_tables(self):
        assert all(v > 0 for v in ht.FOULING_RESISTANCES.values())
        for lo, hi in ht.TYPICAL_U_RANGES.values():
            assert 0 < lo < hi
        assert ht.typical_U_range("water_to_oil") == ht.TYPICAL_U_RANGES["water_to_oil"]
        with pytest.raises(KeyError):
            ht.typical_U_range("nope")

    def test_water_to_oil_U_is_in_typical_band(self):
        # Sanity band for a computed U: turbulent water in a tube, viscous oil on the shell.
        Re = float(ht.reynolds(RHO, 1.5, self.D_i, MU))
        h_i = float(ht.internal_h(Re, PR, K, self.D_i, L=4.0))
        U = float(ht.overall_U(h_i, 300.0, self.D_i, self.D_o, self.k_w, R_fi=1e-4, R_fo=2e-4))
        lo, hi = ht.typical_U_range("water_to_oil")
        assert lo <= U <= hi


# -- integration with the exchangers ------------------------------------------

def test_computed_U_feeds_counter_current_hx():
    """Compute U from correlations, hand UA to CounterCurrentHX, check Q = UA * LMTD."""
    from difflow import CounterCurrentHX, HeatExchangerParams, make_stream
    from difflow.units.heat_exchanger import log_mean_temperature_difference

    D_i, D_o, L, n = 0.0229, 0.0254, 4.0, 20
    Re = ht.reynolds(RHO, 1.2, D_i, MU)
    h_i = ht.internal_h(Re, PR, K, D_i, L=L)
    U_o = ht.overall_U(h_i, 800.0, D_i, D_o, 16.0, R_fi=1e-4, basis="outer")
    UA = U_o * (jnp.pi * D_o * L * n)

    hot = make_stream({"water": 30.0}, T=400.0, P=101325.0)
    cold = make_stream({"water": 40.0}, T=300.0, P=101325.0)
    hx = CounterCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.3, Cp_cold=75.3))
    h_out, c_out, info = hx(hot, cold)
    lmtd = log_mean_temperature_difference(400.0 - info["T_cold_out"], info["T_hot_out"] - 300.0)
    assert float(info["Q"]) == pytest.approx(float(UA * lmtd), rel=1e-6)
    # and the whole chain is differentiable in the tube diameter
    def Q_of_D(D):
        Re = ht.reynolds(RHO, 1.2, D, MU)
        U = ht.overall_U(ht.internal_h(Re, PR, K, D, L=L), 800.0, D, D + 0.0025, 16.0)
        return hx(hot, cold, UA=U * jnp.pi * (D + 0.0025) * L * n)[2]["Q"]
    assert np.isfinite(float(jax.grad(Q_of_D)(0.0229)))
