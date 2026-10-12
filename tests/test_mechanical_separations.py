"""Tests for the mechanical-separation units (issue #434): Settler, CakeFilter, Cyclone.

Oracles (none is a published textbook number; see the end of this docstring):

* Ruth: the quadratic root against ``dt/dV`` integrated with ``scipy.integrate.quad``,
  ``filtrate_volume(ruth_constant_pressure(V)) = V``, the B -> 0 and K_p -> 0 limits,
  ``fit_ruth`` recovering K_p and B (and alpha, R_m) from synthetic data and agreeing
  with ``numpy.polyfit`` on noisy data;
* the batch cycle: with R_m = 0 and no wash ``t_f = t_down``; with a wash the closed form
  ``a V^2 = t_d``; ``scipy.optimize.minimize_scalar`` for a nonlinear wash model; a
  gradient-based optimizer (BFGS with ``jax.grad``) finding the same optimum;
* the thickener: Coe-Clevenger against a NumPy grid maximum, the smooth maximum
  converging to it, the particle route against a hand calculation with
  ``hindered_settling_velocity``, Kynch's construction against a batch curve
  generated *from* a known ``v(C)`` by the envelope of its tangent lines, Talmage-Fitch
  against a NumPy tangent scan and the algebraic identity with Coe-Clevenger;
* the cyclone: ``lapple_efficiency(d50, d50) = 0.5``, d50 and the grade efficiency by hand,
  the standard Lapple geometry (N_e = 6, ``8`` velocity heads), monotonicity in flow;
* species balances to 1e-12; ``check_grads`` of the throughput, the thickener area and
  the cyclone efficiency; a ``Flowsheet`` and the Dryer -> Cyclone hand-off (wex21-3 structure).

Published examples (``TestGeankoplisFiltration``, added for unit-ops book Ch. 21): the
Geankoplis 5e Example 14.2-1 CaCO3 data (t, V) are fitted by ``fit_ruth`` and give the
book's alpha = 1.863e11 m/kg and R_m = 10.63e10 1/m within 0.5 % (the book fits the
straight part of the graph, so the first point is dropped); the plate-and-frame scale-up
(Example 14.2-2, 269.7 s) and the wash time at a quarter of the final rate (Example 14.2-3,
194 s) are reproduced. The data table and the answers were typed from memory and should
be checked against the printed book. NOT reproduced: the Geankoplis rotary-drum example,
McCabe Ch. 29-30, Coulson & Richardson and Cooper & Alley (Ch. 4) worked examples. The other
data below are SYNTHETIC (illustrative). The Lapple constants (K = 16, the standard
proportions, the Stairmand set) are from memory and flagged unverified in the module.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy import integrate, optimize

from difflow import (
    CakeFilter,
    CakeFilterParams,
    Cyclone,
    CycloneParams,
    Dryer,
    DryerParams,
    Settler,
    SettlerParams,
)
from difflow.particles import G_STD, hindered_settling_velocity, kozeny_carman, terminal_velocity
from difflow.psychrometrics import MW_WATER, air_stream_from_state, state
from difflow.units.mechanical import (
    CYCLONE_GEOMETRIES,
    compressible_cake,
    cycle_throughput,
    cyclone_geometry,
    cyclone_pressure_drop,
    dust_laden_gas,
    filtrate_rate,
    filtrate_volume,
    fit_ruth,
    kynch_construction,
    lapple_cut_size,
    lapple_efficiency,
    lapple_turns,
    lse_max,
    optimal_cycle,
    overall_efficiency,
    ruth_constant_pressure,
    ruth_constant_rate,
    ruth_constants,
    solids_flux,
    specific_cake_resistance,
    thickener_area_coe_clevenger,
    thickener_area_kynch,
    thickener_area_talmage_fitch,
    time_to_pressure,
    wash_time,
)

K0 = 273.15


# =============================================================================
# Ruth
# =============================================================================


class TestRuth:
    @pytest.mark.parametrize("K_p,B", [(2.0e3, 5.0e2), (1.0e-3, 4.0), (7.0, 0.0)])
    def test_round_trip(self, K_p, B):
        V = jnp.array([0.01, 0.3, 2.0, 40.0])
        t = ruth_constant_pressure(V, K_p, B)
        np.testing.assert_allclose(filtrate_volume(t, K_p, B), V, rtol=1e-12)

    def test_matches_integration_of_the_rate_law(self):
        # dt/dV = K_p V + B, integrated independently
        K_p, B, V = 3.2e3, 1.5e2, 0.7
        t_quad, _ = integrate.quad(lambda v: K_p * v + B, 0.0, V)
        assert float(ruth_constant_pressure(V, K_p, B)) == pytest.approx(t_quad, rel=1e-12)
        assert float(filtrate_volume(t_quad, K_p, B)) == pytest.approx(V, rel=1e-12)

    def test_zero_medium_resistance(self):
        # B -> 0: V = sqrt(2 t / K_p)
        assert float(filtrate_volume(50.0, 4.0, 0.0)) == pytest.approx(math.sqrt(2 * 50.0 / 4.0), rel=1e-14)
        g = jax.grad(filtrate_volume, argnums=(0, 1, 2))(50.0, 4.0, 0.0)
        assert all(np.isfinite(float(x)) for x in g)

    def test_no_cancellation_when_the_cake_is_negligible(self):
        # K_p -> 0: V -> t / B; the textbook (-B + sqrt(B^2 + 2 K t)) / K loses every digit here
        K, B, t = 1e-18, 3.0, 12.0
        naive = (-B + math.sqrt(B * B + 2 * K * t)) / K
        assert abs(naive - t / B) > 1e-3 * (t / B)          # the naive root is destroyed
        assert float(filtrate_volume(t, K, B)) == pytest.approx(t / B, rel=1e-12)

    def test_zero_time(self):
        assert float(filtrate_volume(0.0, 3.0, 0.0)) == 0.0
        assert np.isfinite(float(jax.grad(filtrate_volume)(0.0, 3.0, 1.0)))

    def test_rate(self):
        K_p, B = 40.0, 2.0
        V = 3.0
        t = ruth_constant_pressure(V, K_p, B)
        dVdt = jax.grad(lambda tt: filtrate_volume(tt, K_p, B))(t)
        assert float(dVdt) == pytest.approx(float(filtrate_rate(V, K_p, B)), rel=1e-10)

    def test_constants_definition(self):
        K_p, B = ruth_constants(A=2.0, dP=1e5, mu=1e-3, alpha=1e11, c_s=20.0, R_m=1e10)
        assert float(K_p) == pytest.approx(1e-3 * 1e11 * 20.0 / (4.0 * 1e5), rel=1e-14)
        assert float(B) == pytest.approx(1e-3 * 1e10 / (2.0 * 1e5), rel=1e-14)

    def test_constant_rate_is_the_constant_pressure_law_at_that_rate(self):
        A, mu, alpha, c_s, R_m, q = 3.0, 1e-3, 5e10, 15.0, 2e10, 2e-3
        t = 600.0
        dP = float(ruth_constant_rate(t, q, A, mu, alpha, c_s, R_m))
        # at the volume V = q t the constant-pressure rate at this dP equals q
        K_p, B = ruth_constants(A, dP, mu, alpha, c_s, R_m)
        assert float(filtrate_rate(q * t, K_p, B)) == pytest.approx(q, rel=1e-12)
        # and the time-to-pressure inverse
        assert float(time_to_pressure(dP, q, A, mu, alpha, c_s, R_m)) == pytest.approx(t, rel=1e-12)

    def test_fit_ruth_recovers_the_constants(self):
        # synthetic data, labelled illustrative
        A, dP, mu, c_s, alpha, R_m = 0.0439, 267e3, 1.0e-3, 23.5, 3.0e10, 6.0e9
        K_p, B = ruth_constants(A, dP, mu, alpha, c_s, R_m)
        V = jnp.linspace(0.05, 1.0, 12) * 1e-3
        t = ruth_constant_pressure(V, K_p, B)
        fit = fit_ruth(t, V, A=A, dP=dP, mu=mu, c_s=c_s)
        assert float(fit["K_p"]) == pytest.approx(float(K_p), rel=1e-10)
        assert float(fit["B"]) == pytest.approx(float(B), rel=1e-9)
        assert float(fit["alpha"]) == pytest.approx(alpha, rel=1e-10)
        assert float(fit["R_m"]) == pytest.approx(R_m, rel=1e-9)
        assert float(fit["r2"]) == pytest.approx(1.0, abs=1e-12)

    def test_fit_ruth_matches_polyfit_on_noisy_data(self):
        rng = np.random.default_rng(3)
        V = np.linspace(0.05, 1.0, 15)
        t = (0.5 * 120.0 * V**2 + 8.0 * V) * (1.0 + 0.02 * rng.standard_normal(V.size))
        fit = fit_ruth(t, V)
        slope, icpt = np.polyfit(V, t / V, 1)
        assert float(fit["K_p"]) == pytest.approx(2 * slope, rel=1e-10)
        assert float(fit["B"]) == pytest.approx(icpt, rel=1e-8)
        assert "alpha" not in fit

    def test_fit_ruth_is_differentiable_in_the_data(self):
        V = jnp.linspace(0.1, 1.0, 8)
        t = ruth_constant_pressure(V, 50.0, 3.0)
        g = jax.grad(lambda tt: fit_ruth(tt, V)["K_p"])(t)
        assert np.all(np.isfinite(np.asarray(g)))
        check_grads(lambda tt: fit_ruth(tt, V)["B"], (t,), order=1, modes=["rev"])

    def test_specific_resistance_is_the_kozeny_carman_form(self):
        d, eps, rho, phi, mu, u = 20e-6, 0.45, 2650.0, 0.8, 1e-3, 1e-4
        alpha = specific_cake_resistance(d, eps, rho, phi)
        grad = kozeny_carman(u, d, eps, mu, phi)            # dP / L
        assert float(alpha) == pytest.approx(float(grad / (mu * u * (1 - eps) * rho)), rel=1e-12)

    def test_compressible_cake(self):
        assert float(compressible_cake(2e9, 4e5, 0.0)) == pytest.approx(2e9)
        assert float(compressible_cake(2.0, 1e5, 0.5)) == pytest.approx(2.0 * math.sqrt(1e5))

    def test_wash_time(self):
        assert float(wash_time(2.0, 0.5)) == pytest.approx(4.0)
        assert float(wash_time(2.0, 0.5, 0.25)) == pytest.approx(16.0)


# =============================================================================
# Batch cycle and the optimum
# =============================================================================


class TestGeankoplisFiltration:
    """Geankoplis 5e Examples 14.2-1 to 14.2-3 (CaCO3 slurry, -dP = 338 kPa, 298.2 K).

    Data and answers recalled (check against the printed book): A = 0.0439 m^2,
    c_s = 23.47 kg/m^3, mu = 8.937e-4 Pa s; alpha = 1.863e11 m/kg, R_m = 10.63e10 1/m;
    a 20-frame press of 0.873 m^2 frames collects 3.37 m^3 in 269.7 s; a wash of 10 % of the
    filtrate at a quarter of the final rate takes 194 s.
    """

    t = np.array([4.4, 9.5, 16.3, 24.6, 34.7, 46.1, 59.0, 73.6, 89.4, 107.3])
    V = 1e-3 * np.array([0.498, 1.000, 1.501, 2.000, 2.498, 3.002, 3.506, 4.004, 4.502, 5.009])
    lab = dict(A=0.0439, dP=338e3, mu=8.937e-4, c_s=23.47)

    def test_example_14_2_1_alpha_and_Rm(self):
        f = fit_ruth(self.t[1:], self.V[1:], **self.lab)
        assert float(f["alpha"]) == pytest.approx(1.863e11, rel=5e-3)
        assert float(f["R_m"]) == pytest.approx(10.63e10, rel=5e-3)
        assert float(f["r2"]) > 0.999

    def test_example_14_2_2_press_time(self):
        K_p, B = ruth_constants(20 * 0.873, 338e3, 8.937e-4, 1.863e11, 23.47, 10.63e10)
        assert float(ruth_constant_pressure(3.37, K_p, B)) == pytest.approx(269.7, rel=1e-3)

    def test_example_14_2_3_wash_time(self):
        K_p, B = ruth_constants(20 * 0.873, 338e3, 8.937e-4, 1.863e11, 23.47, 10.63e10)
        t_w = wash_time(0.1 * 3.37, filtrate_rate(3.37, K_p, B), rate_factor=0.25)
        assert float(t_w) == pytest.approx(194.0, rel=2e-3)

    def test_cake_filter_unit_reproduces_the_press(self):
        """The CakeFilter rating at the example's area and time collects 3.37 m^3 per cycle."""
        rho, c_s, X = 997.0, 23.47, 0.0
        Q_f = 3.37 / 1663.7                       # filtrate m^3/s over one cycle (feed rate is immaterial)
        m_s = c_s * Q_f
        slurry = {"F_solid": m_s, "F_H2O": Q_f * rho / 18.015e-3, "T": 298.15, "P": 101325.0}
        _, _, info = CakeFilter(CakeFilterParams(
            delta_P=338e3, area=20 * 0.873, alpha=1.863e11, R_m=10.63e10, viscosity=8.937e-4,
            liquid_density=rho, filtration_time=269.7, wash_ratio=0.1, wash_rate_factor=0.25,
            downtime=1200.0, cake_moisture=X))(slurry)
        assert float(info["c_s"]) == pytest.approx(c_s, rel=1e-12)
        assert float(info["V_cycle"]) == pytest.approx(3.37, rel=1e-3)
        assert float(info["t_wash"]) == pytest.approx(194.0, rel=2e-3)


class TestCycle:
    def test_optimum_is_the_downtime_with_no_medium_resistance_and_no_wash(self):
        K_p, t_d = 12.0, 900.0
        opt = optimal_cycle(K_p, 0.0, t_d)
        assert float(opt["t_f"]) == pytest.approx(t_d, rel=1e-12)
        g = jax.grad(cycle_throughput)(t_d, K_p, 0.0, t_d)
        assert abs(float(g)) < 1e-12 * float(cycle_throughput(t_d, K_p, 0.0, t_d)) / t_d
        # either side is worse
        q = lambda t: float(cycle_throughput(t, K_p, 0.0, t_d))
        assert q(t_d) > q(0.8 * t_d) and q(t_d) > q(1.25 * t_d)

    @pytest.mark.parametrize("B", [0.0, 3.0])
    @pytest.mark.parametrize("r,phi", [(0.0, 1.0), (0.2, 1.0), (0.3, 0.25)])
    def test_closed_form_matches_a_scalar_optimizer(self, B, r, phi):
        K_p, t_d = 15.0, 700.0
        opt = optimal_cycle(K_p, B, t_d, r, phi)
        res = optimize.minimize_scalar(
            lambda lt: -float(cycle_throughput(math.exp(lt), K_p, B, t_d, r, phi)),
            bounds=(math.log(1.0), math.log(1e6)), method="bounded", options={"xatol": 1e-12})
        assert float(opt["t_f"]) == pytest.approx(math.exp(res.x), rel=1e-5)
        assert float(opt["throughput"]) == pytest.approx(-res.fun, rel=1e-10)
        # the analytic result a V^2 = t_d  (a = K_p (1 + 2 r / phi) / 2)
        V_star = math.sqrt(2 * t_d / (K_p * (1 + 2 * r / phi)))
        assert float(opt["V"]) == pytest.approx(V_star, rel=1e-12)

    def test_gradient_of_the_throughput_vanishes_at_the_optimum_with_a_wash(self):
        K_p, B, t_d, r, phi = 15.0, 3.0, 700.0, 0.2, 1.0
        opt = optimal_cycle(K_p, B, t_d, r, phi)
        g = jax.grad(cycle_throughput)(opt["t_f"], K_p, B, t_d, r, phi)
        assert abs(float(g)) < 1e-12 * float(opt["throughput"]) / float(opt["t_f"])

    def test_a_gradient_based_optimizer_finds_it_with_a_wash(self):
        # wex21-2 structure (a filter-press cycle with a wash), on SYNTHETIC constants
        A, dP, mu, alpha, c_s, R_m = 20.0, 4e5, 1e-3, 2e11, 30.0, 5e10
        K_p, B = ruth_constants(A, dP, mu, alpha, c_s, R_m)
        t_d, r = 1500.0, 0.15

        def neg(lt):
            return -cycle_throughput(jnp.exp(lt[0]), K_p, B, t_d, r, 1.0) * 1e3

        grad = jax.jit(jax.grad(neg))
        res = optimize.minimize(lambda x: float(neg(x)), x0=np.array([math.log(500.0)]),
                                jac=lambda x: np.asarray(grad(jnp.asarray(x))), method="BFGS",
                                options={"gtol": 1e-12})
        opt = optimal_cycle(K_p, B, t_d, r, 1.0)
        assert math.exp(res.x[0]) == pytest.approx(float(opt["t_f"]), rel=1e-5)

    def test_numerical_route_agrees_with_the_closed_form(self):
        K_p, B, t_d, r, phi = 15.0, 3.0, 700.0, 0.2, 0.5
        lin = lambda V: r * V * (K_p * V + B) / phi         # the standard model as a user function
        a = optimal_cycle(K_p, B, t_d, r, phi)
        b = optimal_cycle(K_p, B, t_d, t_wash_fn=lin)
        assert float(b["V"]) == pytest.approx(float(a["V"]), rel=1e-10)
        assert float(b["throughput"]) == pytest.approx(float(a["throughput"]), rel=1e-10)

    def test_numerical_route_with_a_nonlinear_wash_against_a_scalar_optimizer(self):
        K_p, B, t_d = 15.0, 3.0, 700.0
        tw = lambda V: 40.0 * V**2 + 25.0 * V
        b = optimal_cycle(K_p, B, t_d, t_wash_fn=tw)
        res = optimize.minimize_scalar(
            lambda lt: -float(cycle_throughput(math.exp(lt), K_p, B, t_d, t_wash_fn=tw)),
            bounds=(0.0, 14.0), method="bounded", options={"xatol": 1e-12})
        assert float(b["t_f"]) == pytest.approx(math.exp(res.x), rel=1e-5)

    def test_optimum_is_differentiable(self):
        f = lambda K, B, td: optimal_cycle(K, B, td, 0.2, 1.0)["throughput"]
        check_grads(f, (15.0, 3.0, 700.0), order=1, modes=["rev"], rtol=1e-5)
        fn = lambda K, B, td: optimal_cycle(K, B, td, t_wash_fn=lambda V: 40.0 * V**2 + 25.0 * V)["throughput"]
        g = jax.grad(fn, argnums=(0, 1, 2))(15.0, 3.0, 700.0)
        h = 1e-6
        fd = (fn(15.0 * (1 + h), 3.0, 700.0) - fn(15.0 * (1 - h), 3.0, 700.0)) / (30.0 * h)
        assert float(g[0]) == pytest.approx(float(fd), rel=1e-5)

    def test_check_grads_of_the_throughput(self):
        A, mu, c_s, R_m = 20.0, 1e-3, 30.0, 5e10

        def thr(t_f, dP, alpha):
            K_p, B = ruth_constants(A, dP, mu, alpha, c_s, R_m)
            return cycle_throughput(t_f, K_p, B, 1500.0, 0.15, 1.0)

        check_grads(thr, (1200.0, 4e5, 2e11), order=1, modes=["rev"], rtol=1e-5)


# =============================================================================
# CakeFilter
# =============================================================================


def slurry(m_s=2.0, m_liq=40.0, extra=0.0):
    s = {"F_solid": m_s, "F_H2O": m_liq * 1000.0 / 18.015, "T": 300.0, "P": 2e5}
    if extra:
        s["F_NaCl"] = extra
    return s


class TestCakeFilter:
    def params(self, **kw):
        base = dict(delta_P=3e5, alpha=1e11, R_m=1e10, cake_moisture=0.4, cycle_time=3600.0, downtime=600.0,
                    wash_ratio=0.1)
        base.update(kw)
        return CakeFilterParams(**base)

    def test_species_balance_closes(self):
        feed = slurry(extra=3.0)
        cake, filt, info = CakeFilter(self.params())(feed)
        for k in ("F_solid", "F_H2O", "F_NaCl"):
            assert float(cake[k] + filt[k]) == pytest.approx(float(feed[k]), rel=1e-12, abs=1e-14)
        assert float(info["balance_error"]) < 1e-12
        assert float(filt["F_solid"]) == pytest.approx(0.0, abs=1e-14)
        # the cake carries the moisture and all the solid
        liq = float(cake["F_H2O"]) * 18.015 / 1000.0
        assert liq == pytest.approx(0.4 * 2.0, rel=1e-12)
        # the dissolved species is split in the same proportion as the liquid
        assert float(cake["F_NaCl"] / feed["F_NaCl"]) == pytest.approx(float(cake["F_H2O"] / feed["F_H2O"]), rel=1e-12)

    def test_design_then_rate_round_trip(self):
        feed = slurry()
        cake, filt, d = CakeFilter(self.params())(feed)
        rated = self.params(area=float(d["area"]), cycle_time=None, filtration_time=float(d["t_filtration"]))
        _, _, r = CakeFilter(rated)(feed)
        assert float(r["t_cycle"]) == pytest.approx(3600.0, rel=1e-10)
        assert float(r["utilization"]) == pytest.approx(1.0, rel=1e-10)
        assert float(r["V_cycle"]) == pytest.approx(float(d["V_cycle"]), rel=1e-10)

    def test_design_matches_the_ruth_law_by_hand(self):
        feed = slurry()
        _, _, d = CakeFilter(self.params(wash_ratio=0.0))(feed)
        # c_s = rho w / (1 - m w) with w the solids mass fraction and m the wet/dry cake ratio
        w = 2.0 / 42.0
        c_s = 998.0 * w / (1.0 - 1.4 * w)
        assert float(d["c_s"]) == pytest.approx(c_s, rel=1e-12)
        V, t_f = float(d["V_cycle"]), float(d["t_filtration"])
        A = float(d["area"])
        # t_f = mu (alpha c_s V^2/(2 A) + R_m V) / (A dP)
        t = 1e-3 * (1e11 * c_s * V * V / (2 * A) + 1e10 * V) / (A * 3e5)
        assert t_f == pytest.approx(t, rel=1e-10)
        assert t_f == pytest.approx(3600.0 - 600.0, rel=1e-10)

    def test_wash_time_follows_the_final_rate(self):
        _, _, i = CakeFilter(self.params(wash_ratio=0.25, wash_rate_factor=0.25))(slurry())
        assert float(i["t_wash"]) == pytest.approx(0.25 * float(i["V_cycle"]) / (0.25 * float(i["final_rate"])), rel=1e-12)
        assert float(i["t_filtration"] + i["t_wash"] + i["t_down"]) == pytest.approx(3600.0, rel=1e-10)

    def test_optimized_cycle_for_a_fixed_area(self):
        p = self.params(area=60.0, cycle_time=None, optimize_cycle=True, downtime=900.0, wash_ratio=0.2)
        _, _, i = CakeFilter(p)(slurry())
        opt = optimal_cycle(i["K_p"], i["B"], 900.0, 0.2, 1.0)
        assert float(i["t_filtration"]) == pytest.approx(float(opt["t_f"]), rel=1e-12)
        assert float(i["throughput"]) == pytest.approx(float(opt["throughput"]), rel=1e-12)

    def test_compressible_and_particle_sources_for_alpha(self):
        f = slurry()
        i1 = CakeFilter(self.params(alpha=None, alpha0=2.0e8, s=0.4))(f)[2]
        assert float(i1["alpha"]) == pytest.approx(2e8 * (3e5) ** 0.4, rel=1e-12)
        i2 = CakeFilter(self.params(alpha=None, d_p=10e-6, voidage=0.45, rho_p=2650.0, sphericity=0.9))(f)[2]
        assert float(i2["alpha"]) == pytest.approx(
            float(specific_cake_resistance(10e-6, 0.45, 2650.0, 0.9)), rel=1e-12)
        assert float(i2["cake_thickness"]) > 0.0

    def test_rotary_mode(self):
        feed = slurry()
        p = CakeFilterParams(delta_P=6e4, alpha=5e10, R_m=2e10, cake_moisture=0.35, mode="rotary",
                             cycle_time=120.0, submergence=0.3)
        _, _, d = CakeFilter(p)(feed)
        # a filter of that area: the filtrate per revolution is the Ruth volume at psi * T_c
        K_p, B = d["K_p"], d["B"]
        assert float(ruth_constant_pressure(d["V_cycle"], K_p, B)) == pytest.approx(0.3 * 120.0, rel=1e-10)
        assert float(d["V_cycle"] / 120.0) == pytest.approx(float(d["filtrate_flow"]), rel=1e-12)
        # rating at that area is the same
        r = CakeFilter(CakeFilterParams(delta_P=6e4, alpha=5e10, R_m=2e10, cake_moisture=0.35, mode="rotary",
                                        cycle_time=120.0, submergence=0.3, area=float(d["area"])))(feed)[2]
        assert float(r["utilization"]) == pytest.approx(1.0, rel=1e-10)
        # a bigger drum has spare capacity
        r2 = CakeFilter(CakeFilterParams(delta_P=6e4, alpha=5e10, R_m=2e10, cake_moisture=0.35, mode="rotary",
                                         cycle_time=120.0, submergence=0.3, area=1.3 * float(d["area"])))(feed)[2]
        assert float(r2["utilization"]) < 1.0
        assert float(d["area_for_cost"]) == float(d["area"])

    @pytest.mark.parametrize("kw", [
        dict(alpha=None), dict(alpha0=1.0), dict(cycle_time=None), dict(area=10.0),
        dict(area=10.0, cycle_time=None), dict(mode="rotary", wash_ratio=0.1), dict(mode="plate"),
        dict(d_p=1e-5, alpha=None),
    ])
    def test_parameter_validation(self, kw):
        with pytest.raises(ValueError):
            self.params(**kw)

    def test_check_grads_in_dP_and_alpha(self):
        feed = slurry()

        def thr(dP, alpha):
            p = CakeFilterParams(delta_P=dP, alpha=alpha, R_m=1e10, cake_moisture=0.4, area=60.0,
                                 filtration_time=1500.0, downtime=600.0, wash_ratio=0.1)
            return CakeFilter(p)(feed)[2]["throughput"]

        check_grads(thr, (3e5, 1e11), order=1, modes=["rev"], rtol=1e-5)

    def test_jit(self):
        feed = slurry()

        @jax.jit
        def f(dP):
            p = self.params(delta_P=dP)
            return CakeFilter(p)(feed)[2]["area"]

        assert float(f(3e5)) == pytest.approx(float(CakeFilter(self.params())(feed)[2]["area"]), rel=1e-12)


# =============================================================================
# Thickener
# =============================================================================

RHO_S, RHO_L, MU = 2650.0, 998.0, 1.0e-3


def v_exp(C, v0=2.0e-4, k=0.012):
    """A synthetic hindered-settling law v = v0 exp(-k C) (illustrative)."""
    return v0 * np.exp(-k * np.asarray(C))


class TestThickenerLibrary:
    def test_solids_flux(self):
        assert float(solids_flux(50.0, 1e-4)) == pytest.approx(5e-3)
        assert float(solids_flux(50.0, 1e-4, 2e-5)) == pytest.approx(50.0 * 1.2e-4)

    def test_smooth_max_bounds_and_limit(self):
        a = jnp.asarray(np.random.default_rng(0).uniform(1.0, 3.0, 50))
        mx = float(jnp.max(a))
        prev = None
        for s in (10.0, 100.0, 1e3, 1e5):
            lse = float(lse_max(a, s))
            scale = float(jnp.mean(jnp.abs(a)))
            assert mx <= lse <= mx + scale / s * math.log(50) + 1e-12
            if prev is not None:
                assert lse <= prev + 1e-15       # monotonically approaches the maximum from above
            prev = lse
        assert prev == pytest.approx(mx, rel=1e-4)
        # a mask excludes entries
        m = jnp.arange(50) < 10
        assert float(lse_max(a, 1e6, m)) == pytest.approx(float(jnp.max(a[:10])), rel=1e-5)

    def test_coe_clevenger_converges_to_the_brute_force_grid_maximum(self):
        Q0, C0, C_u = 0.05, 20.0, 150.0
        C = np.linspace(C0, C_u, 400)
        v = v_exp(C)
        brute = Q0 * C0 * np.max((1 / C - 1 / C_u) / v)
        prev = np.inf
        for s in (30.0, 300.0, 3e3, 3e5):
            A = float(thickener_area_coe_clevenger(Q0, C0, C_u, C, v, s))
            assert A >= brute * (1 - 1e-12)             # never under-predicts the area
            assert A <= prev + 1e-12
            prev = A
        assert prev == pytest.approx(brute, rel=1e-4)

    def test_solids_flux_form_agrees(self):
        # Kynch: A = F_s / min_C [ C v / (1 - C/C_u) ]   (same algebra as Coe-Clevenger)
        Q0, C0, C_u = 0.05, 20.0, 150.0
        C = np.linspace(C0, C_u * (1 - 1e-9), 4000)
        v = v_exp(C)
        G_L = np.min(C * v / (1 - C / C_u))
        A_kynch = Q0 * C0 / G_L
        A_cc = float(thickener_area_coe_clevenger(Q0, C0, C_u, C, v, 1e7))
        assert A_cc == pytest.approx(A_kynch, rel=1e-4)

    def test_details(self):
        C = np.linspace(20.0, 150.0, 200)
        A, d = thickener_area_coe_clevenger(0.05, 20.0, 150.0, C, v_exp(C), 1e4, True)
        assert float(d["limiting_flux"]) == pytest.approx(0.05 * 20.0 / float(A), rel=1e-12)
        assert 20.0 <= float(d["limiting_concentration"]) <= 150.0

    @pytest.mark.slow
    def test_check_grads_in_Cu_Q0_and_curve_parameters(self):
        C0 = 20.0

        def area(Q0, C_u, sv, sk):          # sv, sk scale the curve v = 2e-4 sv exp(-0.012 sk C)
            C = C0 + (C_u - C0) * jnp.linspace(0.0, 1.0, 60)
            v = 2.0e-4 * sv * jnp.exp(-0.012 * sk * C)
            return thickener_area_coe_clevenger(Q0, C0, C_u, C, v, 200.0)

        check_grads(area, (0.05, 150.0, 1.0, 1.0), order=1, modes=["rev"], rtol=1e-5)
        g = jax.grad(area, argnums=(0, 1, 2, 3))(0.05, 150.0, 1.0, 1.0)
        # more feed, a thicker underflow, or slower settling (smaller v0, steeper decay) all need more area
        assert float(g[0]) > 0 and float(g[1]) > 0 and float(g[2]) < 0 and float(g[3]) > 0


def batch_curve(v_of_C, dv_of_C, C0, z0, C_max, n=160):
    """The interface curve z(t) generated from a known v(C): the envelope of the tangent lines
    z = C0 z0 / C - v(C) t  (Kynch), preceded by the constant-rate line of the initial concentration.
    Returns t, z, and the (C, v) at the envelope nodes."""
    C = np.linspace(C0, C_max, n)[1:]
    t_env = (-C0 * z0 / C**2) / dv_of_C(C)
    z_env = C0 * z0 / C - v_of_C(C) * t_env
    t_line = np.linspace(0.0, t_env[0], 8)[:-1]
    z_line = z0 - v_of_C(C0) * t_line
    return np.r_[t_line, t_env], np.r_[z_line, z_env], C


def v_pow(C, v0=2.0e-4, C0=20.0, m=2.0):
    """A synthetic settling law v = v0 (C0/C)^m (illustrative). Its flux C v is decreasing and convex, so
    Kynch's concentration fan rises from the bottom (the regime of Kynch's construction); m = 2 gives
    the analytic Coe-Clevenger area  A = Q0 C_u / (4 v0 C0)  (maximum at C = C_u/2)."""
    return v0 * (C0 / np.asarray(C)) ** m


class TestKynchAndTalmageFitch:
    C0, Z0, V0 = 20.0, 0.5, 2.0e-4

    def curve(self, n=400):
        v = lambda C: v_pow(C)
        dv = lambda C: -2.0 * self.V0 * self.C0**2 / np.asarray(C) ** 3
        return batch_curve(v, dv, self.C0, self.Z0, 160.0, n)

    def test_coe_clevenger_analytic_oracle(self):
        # a(C) = (1/C - 1/C_u) (C/C0)^2 / v0 = (C - C^2/C_u)/(v0 C0^2): maximum at C_u/2
        Q0, C_u = 0.2, 100.0
        C = np.linspace(self.C0, C_u, 2001)
        A = float(thickener_area_coe_clevenger(Q0, self.C0, C_u, C, v_pow(C), 1e7))
        assert A == pytest.approx(Q0 * C_u / (4 * self.V0 * self.C0), rel=1e-6)

    def test_the_generated_curve_is_a_valid_settling_curve(self):
        t, z, _ = self.curve()
        assert np.all(np.diff(t) > 0) and np.all(np.diff(z) < 0)

    def test_kynch_construction_recovers_the_generating_flux_curve(self):
        t, z, C_env = self.curve()
        C, v, _ = kynch_construction(t, z, self.C0)
        C, v = np.asarray(C), np.asarray(v)
        # skip the first/last nodes (one-sided slopes); the interior is a smooth monotone-cubic derivative
        sl = slice(9, -2)
        expected_C = np.r_[np.full(7, self.C0), C_env]       # 7 constant-rate nodes, then the envelope
        np.testing.assert_allclose(C[sl], expected_C[sl], rtol=2e-3)
        np.testing.assert_allclose(v[sl], v_pow(expected_C[sl]), rtol=2e-3)

    def test_kynch_area_matches_coe_clevenger_of_the_generating_law(self):
        t, z, _ = self.curve()
        F_s, C_u = 1.0, 120.0
        A_k = float(thickener_area_kynch(F_s, C_u, t, z, self.C0, sharpness=1e5))
        Cg = np.linspace(self.C0, C_u, 4000)
        A_cc = F_s * np.max((1 / Cg - 1 / C_u) / v_pow(Cg))
        assert A_k == pytest.approx(A_cc, rel=5e-3)
        assert A_k == pytest.approx(F_s * C_u / (4 * self.V0 * self.C0**2), rel=5e-3)   # F_s = Q0 C0

    def test_talmage_fitch_equals_coe_clevenger_on_the_same_nodes(self):
        t, z, _ = self.curve(120)
        Q0, C_u = 0.2, 100.0
        # the algebraic identity A_TF = F_s max (1/C - 1/C_u)/v  with F_s = Q0 C0 on the Kynch nodes. The two
        # smooth maxima use different temperatures (the arrays differ by a factor and a mask), so the
        # identity holds in the sharpness limit; 1e9 makes the smoothing bias ~1e-8 relative.
        A_tf = float(thickener_area_talmage_fitch(Q0, self.C0, self.Z0, C_u, t, z, 1e9))
        A_k = float(thickener_area_kynch(Q0 * self.C0, C_u, t, z, self.C0, 1e9))
        assert A_tf == pytest.approx(A_k, rel=1e-7)

    def test_talmage_fitch_against_a_numpy_tangent_scan(self):
        t, z, _ = self.curve(250)
        Q0, C_u = 0.2, 100.0
        z_u = self.C0 * self.Z0 / C_u
        # independent tangent construction: slopes by np.gradient on the dense data (second order)
        slope = np.gradient(z, t)
        ok = (z > z_u) & (slope < 0)
        t_u = t[ok] + (z[ok] - z_u) / (-slope[ok])
        A_np = Q0 * t_u.max() / self.Z0
        A_tf = float(thickener_area_talmage_fitch(Q0, self.C0, self.Z0, C_u, t, z, 1e5))
        assert A_tf == pytest.approx(A_np, rel=1e-2)
        # and it agrees with the Coe-Clevenger area of the generating law
        Cg = np.linspace(self.C0, C_u, 4000)
        A_cc = Q0 * self.C0 * np.max((1 / Cg - 1 / C_u) / v_pow(Cg))
        assert A_tf == pytest.approx(A_cc, rel=1e-2)

    def test_talmage_fitch_t_u_is_where_the_tangent_meets_z_u(self):
        t, z, _ = self.curve(250)
        A, d = thickener_area_talmage_fitch(0.2, self.C0, self.Z0, 100.0, t, z, 1e5, True)
        assert float(d["z_u"]) == pytest.approx(self.C0 * self.Z0 / 100.0)
        assert float(A) == pytest.approx(0.2 * float(d["t_u"]) / self.Z0, rel=1e-12)

    @pytest.mark.slow
    def test_gradient_in_Cu(self):
        t, z, _ = self.curve(80)
        f = lambda C_u: thickener_area_talmage_fitch(0.2, self.C0, self.Z0, C_u, t, z, 300.0)
        check_grads(f, (100.0,), order=1, modes=["rev"], rtol=1e-5)
        assert float(jax.grad(f)(100.0)) > 0.0           # a thicker underflow needs more area


def thick_feed(m_s=5.0, m_l=50.0):
    return {"F_solid": m_s, "F_H2O": m_l * 1000.0 / 18.015, "T": 300.0, "P": 1e5}


class TestSettlerUnit:
    def test_particle_route_matches_a_hand_calculation(self):
        d_p, C_u, n = 25e-6, 300.0, 4.65
        feed = thick_feed()
        p = SettlerParams(underflow_concentration=C_u, d_p=d_p, rho_p=RHO_S, richardson_zaki_n=n,
                          liquid_density=RHO_L, viscosity=MU, n_grid=200, sharpness=1e6)
        _, _, info = Settler(p)(feed)
        m_s, m_l = 5.0, 50.0
        Q0 = m_l / RHO_L + m_s / RHO_S
        C0 = m_s / Q0
        v_t = float(terminal_velocity(d_p, RHO_S, RHO_L, MU))
        C = np.linspace(C0, C_u, 200)
        v = v_t * (1 - C / RHO_S) ** n                  # Richardson-Zaki
        assert v[0] == pytest.approx(float(hindered_settling_velocity(v_t, 1 - C0 / RHO_S, n=n)), rel=1e-12)
        A_hand = m_s * np.max((1 / C - 1 / C_u) / v)
        assert float(info["area_required"]) == pytest.approx(A_hand, rel=1e-4)
        assert float(info["C_feed"]) == pytest.approx(C0, rel=1e-12)
        # design mode multiplies by the safety factor
        p2 = SettlerParams(underflow_concentration=C_u, d_p=d_p, rho_p=RHO_S, richardson_zaki_n=n,
                           liquid_density=RHO_L, viscosity=MU, n_grid=200, sharpness=1e6, safety_factor=1.5)
        i2 = Settler(p2)(feed)[2]
        assert float(i2["area"]) == pytest.approx(1.5 * float(i2["area_required"]), rel=1e-12)

    def test_stokes_velocity_hand_value(self):
        # for a 10 micron particle Re_t ~ 1e-3: Stokes law v = (rho_p - rho) g d^2 / (18 mu)
        d = 10e-6
        v = float(terminal_velocity(d, RHO_S, RHO_L, MU))
        assert v == pytest.approx((RHO_S - RHO_L) * G_STD * d**2 / (18 * MU), rel=2e-2)

    def test_piecewise_exponent_route_runs(self):
        p = SettlerParams(underflow_concentration=300.0, d_p=25e-6, rho_p=RHO_S)
        _, _, i = Settler(p)(thick_feed())
        assert float(i["area_required"]) > 0.0

    @pytest.mark.slow
    def test_species_balance_all_methods(self):
        t, z, _ = TestKynchAndTalmageFitch().curve(120)
        C_u = 120.0
        cd = np.linspace(10.0, 200.0, 25)
        common = dict(underflow_concentration=C_u, rho_p=RHO_S, liquid_density=RHO_L)
        plist = [
            SettlerParams(method="coe_clevenger", C_data=list(cd), v_data=list(v_exp(cd)), **common),
            SettlerParams(method="coe_clevenger", d_p=20e-6, richardson_zaki_n=4.65, viscosity=MU, **common),
            SettlerParams(method="kynch", t_data=list(t), z_data=list(z), batch_C0=20.0, **common),
            SettlerParams(method="talmage_fitch", t_data=list(t), z_data=list(z), batch_C0=20.0, **common),
            SettlerParams(method="overflow_rate", d_p=20e-6, viscosity=MU, d_bins=[5e-6, 10e-6, 20e-6, 40e-6],
                          mass_fractions=[0.2, 0.3, 0.3, 0.2], **common),
        ]
        feed = {"F_solid": 5.0, "F_H2O": 50000.0 / 18.015, "F_NaCl": 7.0, "T": 300.0, "P": 1e5}
        for p in plist:
            o, u, info = Settler(p)(feed)
            for k in ("F_solid", "F_H2O", "F_NaCl"):
                assert float(o[k] + u[k]) == pytest.approx(feed[k], rel=1e-12, abs=1e-13), (p.method, k)
            assert float(info["balance_error"]) < 1e-12
            # the underflow slurry is at C_u (solids mass over slurry volume)
            vol_u = float(u["F_H2O"]) * 18.015 / 1000.0 / RHO_L + float(u["F_solid"]) / RHO_S
            assert float(u["F_solid"]) / vol_u == pytest.approx(C_u, rel=1e-10)

    def test_data_routes_agree_with_each_other(self):
        t, z, _ = TestKynchAndTalmageFitch().curve(400)
        C_u, m_s = 100.0, 4.0
        feed = {"F_solid": m_s, "F_H2O": 0.0, "T": 300.0, "P": 1e5}
        # set the feed to the batch-test concentration C0 = 20 kg/m^3 exactly (solid volume included)
        Q0 = m_s / 20.0
        feed["F_H2O"] = (Q0 - m_s / RHO_S) * RHO_L * 1000.0 / 18.015
        common = dict(underflow_concentration=C_u, rho_p=RHO_S, liquid_density=RHO_L, sharpness=1e9)
        cd = np.linspace(15.0, 150.0, 60)
        a_cc = Settler(SettlerParams(method="coe_clevenger", C_data=list(cd), v_data=list(v_pow(cd)),
                                     n_grid=400, **common))(feed)[2]["area_required"]
        a_k = Settler(SettlerParams(method="kynch", t_data=list(t), z_data=list(z), batch_C0=20.0, **common))(feed)[2]["area_required"]
        a_tf = Settler(SettlerParams(method="talmage_fitch", t_data=list(t), z_data=list(z), batch_C0=20.0, **common))(feed)[2]["area_required"]
        assert float(a_k) == pytest.approx(float(a_tf), rel=1e-7)     # identical in the sharpness limit
        assert float(a_k) == pytest.approx(float(a_cc), rel=1e-2)

    def test_overflow_rate_mode(self):
        feed = thick_feed(m_s=0.5, m_l=100.0)
        d_bins = [5e-6, 10e-6, 20e-6, 40e-6]
        w = [0.2, 0.3, 0.3, 0.2]
        common = dict(underflow_concentration=200.0, rho_p=RHO_S, liquid_density=RHO_L, viscosity=MU,
                      d_bins=d_bins, mass_fractions=w, method="overflow_rate")
        _, _, i = Settler(SettlerParams(d_p=20e-6, **common))(feed)
        v_t = np.array([float(terminal_velocity(d, RHO_S, RHO_L, MU)) for d in d_bins])
        # design particle is captured (to the smoothing width); smaller ones in proportion v_t / v_o
        eta = np.minimum(1.0, v_t / float(i["overflow_rate"]))
        assert eta[2] == pytest.approx(1.0, rel=1e-9) and eta[3] == 1.0
        np.testing.assert_allclose(np.asarray(i["grade_efficiency"]), eta, atol=6e-5)
        assert float(i["solids_capture"]) == pytest.approx(float(np.dot(w, eta)), abs=6e-5)
        # a safety factor adds area and raises the capture
        _, _, i2 = Settler(SettlerParams(d_p=20e-6, safety_factor=2.0, **common))(feed)
        assert float(i2["solids_capture"]) > float(i["solids_capture"])
        # rating a given area: the overflow rate is Q_o / A
        _, _, i3 = Settler(SettlerParams(area=float(i["area"]), **common))(feed)
        assert float(i3["overflow_rate"]) == pytest.approx(float(i["overflow_rate"]), rel=1e-12)

    @pytest.mark.slow
    def test_check_grads_of_the_area(self):
        def area(Cu, m_s, sd):              # sd scales the particle size, d_p = 25 micron * sd
            p = SettlerParams(underflow_concentration=Cu, d_p=25e-6 * sd, rho_p=RHO_S, richardson_zaki_n=4.65,
                              liquid_density=RHO_L, viscosity=MU, n_grid=60, sharpness=300.0)
            feed = {"F_solid": m_s, "F_H2O": 50000.0 / 18.015, "T": 300.0, "P": 1e5}
            return Settler(p)(feed)[2]["area_required"]

        check_grads(area, (300.0, 5.0, 1.0), order=1, modes=["rev"], rtol=1e-4)
        g = jax.grad(area, argnums=(0, 1, 2))(300.0, 5.0, 1.0)
        # a thicker underflow needs more area; bigger particles settle faster so need less (feed solids at
        # fixed water raises C0 as well as F_s, so the sign of that derivative is not asserted)
        assert float(g[0]) > 0 and float(g[2]) < 0

    @pytest.mark.slow
    def test_check_grads_of_data_route_in_Cu(self):
        cd = np.linspace(10.0, 250.0, 25)
        vd = v_exp(cd)

        def area(Cu, scale):
            p = SettlerParams(underflow_concentration=Cu, C_data=list(cd), v_data=list(vd), rho_p=RHO_S,
                              liquid_density=RHO_L, n_grid=60, sharpness=300.0)
            feed = {"F_solid": 5.0 * scale, "F_H2O": 50000.0 / 18.015, "T": 300.0, "P": 1e5}
            return Settler(p)(feed)[2]["area_required"]

        check_grads(area, (150.0, 1.0), order=1, modes=["rev"], rtol=1e-4)

    @pytest.mark.parametrize("kw", [
        dict(method="foo"), dict(method="coe_clevenger"), dict(method="kynch"),
        dict(method="overflow_rate"), dict(method="coe_clevenger", d_p=1e-5, C_data=[1.0], v_data=[1.0]),
        dict(method="overflow_rate", d_p=1e-5, d_bins=[1e-5], mass_fractions=[0.5, 0.5]),
    ])
    def test_parameter_validation(self, kw):
        with pytest.raises(ValueError):
            SettlerParams(underflow_concentration=100.0, **kw)

    def test_jit(self):
        p = SettlerParams(underflow_concentration=300.0, d_p=25e-6, richardson_zaki_n=4.65, viscosity=MU)
        f = jax.jit(lambda m: Settler(p)({"F_solid": m, "F_H2O": 2000.0, "T": 300.0, "P": 1e5})[2]["area"])
        assert np.isfinite(float(f(3.0)))


# =============================================================================
# Cyclone
# =============================================================================

D_BINS = [2e-6, 5e-6, 10e-6, 20e-6, 50e-6]
W_BINS = [0.1, 0.2, 0.3, 0.3, 0.1]


def dusty_gas(Q=0.85, T=300.0, P=101325.0, m_dust=0.05):
    F = P * Q / (8.314462618 * T)
    return {"F_air": F, "F_solid": m_dust, "T": T, "P": P}


class TestCycloneLibrary:
    def test_efficiency_is_one_half_at_the_cut_size(self):
        for d50 in (1e-6, 7.3e-6, 4e-5):
            assert float(lapple_efficiency(d50, d50)) == pytest.approx(0.5, rel=1e-15)
        assert float(lapple_efficiency(1e-3, 1e-6)) == pytest.approx(1.0, abs=1e-6)

    def test_cut_size_by_hand(self):
        mu, W, N_e, v, rp, rg = 1.8e-5, 0.25, 6.0, 15.0, 2000.0, 1.2
        d50 = math.sqrt(9 * mu * W / (2 * math.pi * N_e * v * (rp - rg)))
        assert float(lapple_cut_size(mu, W, N_e, v, rp, rg)) == pytest.approx(d50, rel=1e-14)

    def test_grade_efficiency_and_overall_by_hand(self):
        d = np.array(D_BINS)
        d50 = 8e-6
        eta = 1.0 / (1.0 + (d50 / d) ** 2)
        np.testing.assert_allclose(lapple_efficiency(d, d50), eta, rtol=1e-14)
        assert float(overall_efficiency(d, W_BINS, d50)) == pytest.approx(float(np.dot(W_BINS, eta)), rel=1e-14)
        # normalisation of the fractions
        assert float(overall_efficiency(d, 3 * np.array(W_BINS), d50)) == pytest.approx(
            float(np.dot(W_BINS, eta)), rel=1e-14)

    def test_standard_lapple_proportions(self):
        g = cyclone_geometry(2.0, "lapple")
        assert float(g["H"]) == pytest.approx(1.0) and float(g["W"]) == pytest.approx(0.5)
        assert float(g["S"]) == pytest.approx(1.25)            # S = 5D/8 (outlet-duct length)
        assert float(lapple_turns(g["H"], g["L_b"], g["L_c"])) == pytest.approx(6.0)
        assert set(CYCLONE_GEOMETRIES) == {"lapple", "stairmand_he"}
        c = cyclone_geometry(1.0, "custom", [0.5, 0.2, 0.5, 0.5, 1.5, 2.5, 0.375])
        assert float(c["J"]) == pytest.approx(0.375)
        with pytest.raises(ValueError):
            cyclone_geometry(1.0, "custom")

    def test_pressure_drop_form(self):
        g = cyclone_geometry(1.0, "lapple")
        dP = cyclone_pressure_drop(1.2, 20.0, g["H"], g["W"], g["D_e"])
        # K H W / D_e^2 = 16 * 0.5 * 0.25 / 0.25 = 8 velocity heads
        assert float(dP) == pytest.approx(8.0 * 0.5 * 1.2 * 400.0, rel=1e-14)
        assert float(cyclone_pressure_drop(1.2, 20.0, 0.5, 0.25, 0.5, velocity_heads=6.0)) == pytest.approx(
            6.0 * 0.5 * 1.2 * 400.0)


class TestCycloneUnit:
    def params(self, **kw):
        base = dict(D=1.0, rho_p=2000.0, d_bins=D_BINS, mass_fractions=W_BINS)
        base.update(kw)
        return CycloneParams(**base)

    def test_matches_the_library_by_hand(self):
        gas = dusty_gas()
        cl, so, info = Cyclone(self.params())(gas)
        Q = 0.85
        v_i = Q / (0.5 * 0.25)
        rho_g = 28.96e-3 * 101325.0 / (8.314462618 * 300.0)
        d50 = math.sqrt(9 * 1.8e-5 * 0.25 / (2 * math.pi * 6.0 * v_i * (2000.0 - rho_g)))
        assert float(info["inlet_velocity"]) == pytest.approx(v_i, rel=1e-12)
        assert float(info["d50"]) == pytest.approx(d50, rel=1e-12)
        eta = 1 / (1 + (d50 / np.array(D_BINS)) ** 2)
        np.testing.assert_allclose(info["grade_efficiency"], eta, rtol=1e-12)
        assert float(info["overall_efficiency"]) == pytest.approx(float(np.dot(W_BINS, eta)), rel=1e-12)
        assert float(info["delta_P"]) == pytest.approx(8.0 * 0.5 * rho_g * v_i**2, rel=1e-12)
        assert float(so["F_solid"]) == pytest.approx(0.05 * float(np.dot(W_BINS, eta)), rel=1e-12)
        assert float(info["dust_loading"]) == pytest.approx(0.05 / Q, rel=1e-12)

    def test_species_balance(self):
        gas = dusty_gas()
        gas["F_H2O"] = 1.5
        cl, so, info = Cyclone(self.params())(gas)
        for k in ("F_air", "F_H2O", "F_solid"):
            assert float(cl[k] + so[k]) == pytest.approx(gas[k], rel=1e-12, abs=1e-14)
        assert float(info["balance_error"]) < 1e-12
        assert float(so["F_air"]) == 0.0

    def test_efficiency_rises_with_gas_flow_at_fixed_D(self):
        eff = [float(Cyclone(self.params())(dusty_gas(Q=q))[2]["overall_efficiency"])
               for q in (0.2, 0.4, 0.8, 1.6, 3.2)]
        assert all(b > a for a, b in zip(eff, eff[1:]))
        dP = [float(Cyclone(self.params())(dusty_gas(Q=q))[2]["delta_P"]) for q in (0.2, 0.8, 3.2)]
        assert dP[1] / dP[0] == pytest.approx(16.0, rel=1e-12)       # dP ~ v_i^2

    def test_smaller_cyclone_collects_more(self):
        e = [float(Cyclone(self.params(D=d))(dusty_gas())[2]["overall_efficiency"]) for d in (1.5, 1.0, 0.5)]
        assert e[0] < e[1] < e[2]

    def test_stairmand_and_custom_geometry_and_turns(self):
        gas = dusty_gas()
        i1 = Cyclone(self.params(geometry="stairmand_he"))(gas)[2]
        assert float(i1["N_e"]) == pytest.approx((1.5 + 1.25) / 0.5)
        i2 = Cyclone(self.params(geometry="custom", ratios=CYCLONE_GEOMETRIES["stairmand_he"]))(gas)[2]
        assert float(i2["d50"]) == pytest.approx(float(i1["d50"]), rel=1e-14)
        i3 = Cyclone(self.params(n_turns=8.0))(gas)[2]
        i4 = Cyclone(self.params())(gas)[2]
        assert float(i3["d50"]) / float(i4["d50"]) == pytest.approx(math.sqrt(6.0 / 8.0), rel=1e-12)
        i5 = Cyclone(self.params(delta_P_velocity_heads=6.0))(gas)[2]
        assert float(i5["delta_P"]) == pytest.approx(
            6.0 * 0.5 * float(i4["gas_density"]) * float(i4["inlet_velocity"]) ** 2, rel=1e-12)

    def test_gas_without_dust(self):
        gas = {"F_air": 30.0, "T": 300.0, "P": 101325.0}
        cl, so, info = Cyclone(self.params())(gas)
        assert float(so["F_solid"]) == 0.0 and float(info["dust_loading"]) == 0.0

    def test_check_grads_of_the_efficiency(self):
        def eff(D, Q, rho_p):
            p = CycloneParams(D=D, rho_p=rho_p, d_bins=D_BINS, mass_fractions=W_BINS)
            gas = {"F_air": 101325.0 * Q / (8.314462618 * 300.0), "F_solid": 0.05, "T": 300.0, "P": 101325.0}
            return Cyclone(p)(gas)[2]["overall_efficiency"]

        check_grads(eff, (1.0, 0.85, 2000.0), order=1, modes=["rev"], rtol=1e-6)
        g = jax.grad(eff, argnums=(0, 1, 2))(1.0, 0.85, 2000.0)
        assert float(g[0]) < 0 and float(g[1]) > 0 and float(g[2]) > 0

    @pytest.mark.parametrize("kw", [
        dict(geometry="foo"), dict(geometry="custom"), dict(mass_fractions=[1.0]),
    ])
    def test_parameter_validation(self, kw):
        with pytest.raises(ValueError):
            self.params(**kw)


# =============================================================================
# Flowsheets and the Dryer exhaust (wex21-3 structure, synthetic numbers)
# =============================================================================


class TestFlowsheets:
    def test_cyclone_in_a_flowsheet(self):
        from difflow.flowsheet import Flowsheet, Unit

        fs = Flowsheet(species_order=["air", "solid"])
        gas = {"F_air": 35.0, "F_solid": 0.1, "T": 300.0, "P": 101325.0}
        fs.add_feed("gas", gas)
        fs.add_unit(Unit("cyc", Cyclone(CycloneParams(D=1.0, rho_p=2000.0, d_bins=D_BINS, mass_fractions=W_BINS)),
                         ["gas"], ["clean", "dust"]))
        out = fs.solve()
        assert float(out["clean"]["F_solid"] + out["dust"]["F_solid"]) == pytest.approx(0.1, rel=1e-12)

    @pytest.mark.slow
    def test_dryer_exhaust_to_cyclone(self):
        """Documented hand-off: Dryer(entrainment=f) puts the fines in info['fines'], NOT in air_out
        (dry_air_flow counts every non-water F_* as dry air); dust_laden_gas() merges them for the cyclone."""
        wet = {"F_solid": 0.5, "F_H2O": 0.5 * 0.30 / MW_WATER, "T": 300.0, "P": 101325.0}
        air = air_stream_from_state(state(T=100.0 + K0, W=0.01), 10.0)
        dp = DryerParams(mode="continuous", X_c=0.15, X_eq=0.02, product_moisture=0.05, h=60.0,
                         solids_per_area=10.0, max_solids_T=340.0, entrainment=0.05)
        dry, air_out, dinfo = Dryer(dp)(wet, air)
        fines = dinfo["fines"]
        assert "F_solid" not in air_out
        assert float(fines["F_solid"]) == pytest.approx(0.05 * 0.5, rel=1e-12)
        inlet = dust_laden_gas(air_out, fines)
        cp = CycloneParams(D=0.6, rho_p=1500.0, d_bins=D_BINS, mass_fractions=W_BINS, gas_MW=28.5)
        clean, dust, info = Cyclone(cp)(inlet)
        # dust loading is the fines over the gas volume flow
        assert float(info["dust_loading"]) == pytest.approx(float(fines["F_solid"]) / float(info["gas_flow"]), rel=1e-12)
        # every species of the exhaust+fines is conserved, and so is the dry solid across the whole train
        for k in inlet:
            if k.startswith("F_"):
                assert float(clean[k] + dust[k]) == pytest.approx(float(inlet[k]), rel=1e-12, abs=1e-14)
        total_solid = float(dry["F_solid"] + clean["F_solid"] + dust["F_solid"])
        assert total_solid == pytest.approx(0.5, rel=1e-12)
        assert 0.0 < float(info["overall_efficiency"]) < 1.0
        assert float(dust["F_solid"]) > 0.0
