"""Tests for difflow.adsorption: isotherms, batch and fixed-bed adsorption,
ion exchange and chromatography (unit-ops book Ch. 23)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import adsorption as ad

jax.config.update("jax_enable_x64", True)

# Geankoplis 5e Ex. 12.1-1 phenol on activated carbon (recalled, unit-ops book verify-against-print E-016)
C_PHENOL = jnp.array([0.322, 0.117, 0.039, 0.0061, 0.0011])     # kg/m^3
Q_PHENOL = jnp.array([0.150, 0.122, 0.094, 0.059, 0.045])       # kg/kg
# Geankoplis 5e Ex. 12.3-1 breakthrough data (recalled, E-019)
T_BT = jnp.array([0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.2, 6.5, 6.8])   # h
R_BT = jnp.array([0.0, 0.0, 0.002, 0.030, 0.155, 0.396, 0.658, 0.903, 0.933, 0.975, 0.993])


class TestIsotherms:
    def test_forms(self):
        assert float(ad.linear_isotherm(2.0, 3.0)) == 6.0
        assert float(ad.langmuir_isotherm(1.0, 2.0, 1.0)) == pytest.approx(1.0)
        # Langmuir tends to q_max and to the Henry slope q_max K
        assert float(ad.langmuir_isotherm(1e9, 2.0, 1.0)) == pytest.approx(2.0)
        assert float(jax.grad(ad.langmuir_isotherm)(0.0, 2.0, 3.0)) == pytest.approx(6.0)
        assert float(ad.freundlich_isotherm(8.0, 0.5, 1.0 / 3.0)) == pytest.approx(1.0)

    def test_freundlich_zero_is_finite(self):
        assert float(ad.freundlich_isotherm(0.0, 0.2, 0.3)) == 0.0
        assert np.isfinite(float(jax.grad(ad.freundlich_isotherm, argnums=1)(0.0, 0.2, 0.3)))

    @pytest.mark.parametrize("model,params", [
        ("langmuir", {"q_max": 3.0, "K": 0.8}),
        ("freundlich", {"K": 0.2, "n": 0.3}),
        ("linear", {"K": 1.7}),
    ])
    def test_fit_recovers_exact_parameters(self, model, params):
        c = jnp.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
        q = ad.ISOTHERMS[model](c, **params)
        fit = ad.fit_isotherm(c, q, model)
        for k, v in params.items():
            assert float(fit["params"][k]) == pytest.approx(v, rel=1e-6)
        assert float(fit["r2"]) == pytest.approx(1.0)

    def test_fit_geankoplis_phenol_freundlich(self):
        """The log-log fit of Ex. 12.1-1 is close to the book's q = 0.199 c^0.229."""
        fit = ad.fit_isotherm(C_PHENOL, Q_PHENOL, "freundlich")
        lin = fit["linearized"]
        assert float(lin["n"]) == pytest.approx(0.229, abs=0.015)
        assert float(lin["K"]) == pytest.approx(0.199, rel=0.05)
        assert float(fit["r2"]) > 0.99
        # Langmuir fits these data much worse (the book's conclusion)
        assert float(ad.fit_isotherm(C_PHENOL, Q_PHENOL, "langmuir")["r2"]) < 0.9


class TestBatch:
    def test_linear_closed_form(self):
        # V(c0 - c) = m K c  ->  c = c0 V/(V + m K)
        r = ad.batch_adsorption(2.0, 3.0, 0.5, "linear", K=4.0)
        assert float(r["c"]) == pytest.approx(2.0 * 3.0 / (3.0 + 2.0), rel=1e-10)

    def test_geankoplis_12_2_1(self):
        """Ex. 12.2-1: 1 m^3 at 0.21 kg/m^3 phenol, 1.40 kg carbon, q = 0.199 c^0.229."""
        r = ad.batch_adsorption(0.21, 1.0, 1.40, "freundlich", K=0.199, n=0.229)
        assert float(r["c"]) == pytest.approx(0.062, abs=0.001)
        assert float(r["q"]) == pytest.approx(0.106, abs=0.001)
        # the balance closes
        assert float(1.0 * (0.21 - r["c"])) == pytest.approx(float(1.40 * r["q"]), rel=1e-10)

    def test_preloaded_adsorbent_desorbs(self):
        r = ad.batch_adsorption(0.0, 1.0, 1.0, "linear", q0=1.0, K=1.0)
        assert float(r["c"]) == pytest.approx(0.5)

    def test_gradients(self):
        def c_out(m, K):
            return ad.batch_adsorption(0.21, 1.0, m, "langmuir", q_max=0.15, K=K)["c"]
        check_grads(c_out, (1.4, 60.0), order=1, modes=["rev"], rtol=1e-4)


class TestLUB:
    def test_ideal_step(self):
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r = ad.lub_analysis(t, jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]), 1.0, c_break=0.5)
        assert float(r["t_stoich"]) == pytest.approx(2.0)

    def test_geankoplis_12_3_1(self):
        """Ex. 12.3-1: L = 14 cm, break at c/c0 = 0.01; the book integrates
        graphically and reports t_t = 5.15 h, t_u = 3.65 h, H_B = 9.92 cm."""
        r = ad.lub_analysis(T_BT, R_BT, 14.0, c_break=0.01)
        assert float(r["t_break"]) == pytest.approx(3.65, abs=0.02)
        assert float(r["t_stoich"]) == pytest.approx(5.15, rel=0.01)
        assert float(r["L_used"]) == pytest.approx(9.92, rel=0.015)
        assert float(r["L_used"] + r["LUB"]) == pytest.approx(14.0)

    def test_break_time_scales(self):
        r = ad.lub_analysis(T_BT, R_BT, 14.0, c_break=0.01)
        # the test bed itself
        tb = ad.lub_break_time(14.0, r["t_stoich"], 14.0, r["LUB"])
        assert float(tb) == pytest.approx(float(r["t_usable"]), rel=1e-12)
        assert float(jax.grad(ad.lub_break_time)(20.0, 5.0, 14.0, 4.0)) == pytest.approx(5.0 / 14.0)


class TestFixedBed:
    T = jnp.linspace(0.0, 6000.0, 241)

    def test_klinkenberg_midpoint_at_stoichiometric_time(self):
        L, v, eps, rho, K, k = 1.0, 0.01, 0.4, 1.0, 16.0, 0.1
        ts = float(ad.stoichiometric_time(L, v, eps, rho, 1.0, K))
        assert float(ad.klinkenberg_breakthrough(ts, L, v, eps, rho, K, k)) == pytest.approx(0.5, abs=0.02)

    def test_matches_klinkenberg_linear_isotherm(self):
        out = ad.fixed_bed_breakthrough(self.T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                        k_ldf=0.02, isotherm="linear", n_cells=200, K=16.0)
        kl = ad.klinkenberg_breakthrough(self.T, 1.0, 0.01, 0.4, 1.0, 16.0, 0.02)
        assert float(jnp.max(jnp.abs(out["c_out"] - kl))) < 0.02

    def test_material_balance_langmuir(self):
        """Area above the curve = stoichiometric time (overall solute balance)."""
        out = ad.fixed_bed_breakthrough(self.T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                        k_ldf=0.05, isotherm="langmuir", q_max=30.0, K=1.0)
        assert float(out["t_stoich"]) == pytest.approx(1540.0)
        assert float(out["c_out"][-1]) == pytest.approx(1.0, abs=1e-4)
        area = float(jnp.trapezoid(1.0 - out["c_out"], self.T))
        assert area == pytest.approx(1540.0, rel=2e-3)
        # and the bed ends saturated
        assert float(jnp.min(out["q_profile"][-1])) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.slow
    def test_favourable_front_is_constant_pattern(self):
        """A Langmuir front keeps its width: LUB is the same for two bed lengths."""
        kw = dict(v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0, k_ldf=0.01, isotherm="langmuir",
                  q_max=30.0, K=1.0)
        t = jnp.linspace(0.0, 9000.0, 1801)
        lub = []
        for L in (1.0, 2.0):   # same cell size in both beds
            out = ad.fixed_bed_breakthrough(t, L=L, n_cells=int(60 * L), **kw)
            lub.append(float(ad.lub_analysis(t, out["c_out"], L, 0.05)["LUB"]))
        assert lub[1] == pytest.approx(lub[0], rel=0.15)

    def test_gradients_small_bed(self):
        # Reverse mode through the adaptive solve; without the stability cap on
        # the step these came out with the wrong sign (see fixed_bed_breakthrough)
        T = jnp.linspace(0.0, 1000.0, 101)

        def area(q_max, k=0.05):
            out = ad.fixed_bed_breakthrough(T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                            k_ldf=k, isotherm="langmuir", n_cells=15, q_max=q_max, K=1.0,
                                            method="mol")
            return jnp.trapezoid(1.0 - out["c_out"], T)
        assert float(jax.grad(area)(5.0)) == pytest.approx(50.0, rel=5e-3)

    def test_gradient_corrected_time(self):
        T = jnp.linspace(0.0, 1000.0, 101)

        def area_ct(q_max):
            out = ad.fixed_bed_breakthrough(T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                            k_ldf=0.05, isotherm="langmuir", n_cells=15, q_max=q_max, K=1.0,
                                            method="corrected_time")
            return jnp.trapezoid(1.0 - out["c_out"], T)
        assert float(jax.grad(area_ct)(5.0)) == pytest.approx(50.0, rel=5e-3)

    def test_methods_agree(self):
        T = jnp.linspace(0.0, 1000.0, 101)
        kw = dict(L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0, k_ldf=0.05, isotherm="langmuir",
                  n_cells=30, q_max=5.0, K=1.0)
        a = ad.fixed_bed_breakthrough(T, method="mol", **kw)["c_out"]
        b = ad.fixed_bed_breakthrough(T, method="corrected_time", **kw)["c_out"]
        np.testing.assert_allclose(a, b, atol=0.02)

    def test_liquid_bed_lasting_months_is_cheap(self):
        # residence time 12 min, bed life months: the real-time method of lines
        # would need ~1e6 steps; the corrected-time form a few hundred
        day = 86400.0
        T = jnp.linspace(0.0, 300.0 * day, 301)
        r = ad.fixed_bed_breakthrough(T, L=2.0, v_s=10 / 3600, epsilon=0.4, rho_b=450.0, c_feed=0.5,
                                      k_ldf=5e-6, isotherm="freundlich", n_cells=30, K=28.0, n=0.62)
        assert float(jnp.trapezoid(1.0 - r["c_out"], T)) == pytest.approx(float(r["t_stoich"]), rel=2e-3)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            ad.fixed_bed_breakthrough(jnp.linspace(0, 10, 3), L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0,
                                      c_feed=1.0, k_ldf=0.05, isotherm="linear", K=1.0, method="euler")

    @pytest.mark.slow
    def test_gradient_of_area_matches_fd(self):
        def area(q_max):
            out = ad.fixed_bed_breakthrough(self.T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                            k_ldf=0.05, isotherm="langmuir", n_cells=40, q_max=q_max, K=1.0,
                                            method="mol")
            return jnp.trapezoid(1.0 - out["c_out"], self.T)
        g = float(jax.grad(area)(30.0))
        # d t_s / d q_max = (L/v) rho K c/(1 + K c)/c = 50 s per unit q_max
        assert g == pytest.approx(50.0, rel=5e-3)

    @pytest.mark.slow
    def test_gradient_wrt_ldf_coefficient_matches_fd(self):
        T = jnp.linspace(0.0, 1000.0, 101)

        def width(k):
            out = ad.fixed_bed_breakthrough(T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                            k_ldf=k, isotherm="langmuir", n_cells=15, q_max=5.0, K=1.0)
            return jnp.sum(out["c_out"] * (1.0 - out["c_out"]))  # spread of the front
        g = float(jax.grad(width)(0.05))
        fd = float((width(0.0505) - width(0.0495)) / 0.001)
        assert g == pytest.approx(fd, rel=1e-2)


class TestIonExchange:
    def test_equal_valence_closed_form(self):
        x = jnp.array([0.1, 0.5, 0.9])
        y = ad.ion_exchange_equilibrium(x, 3.0)
        np.testing.assert_allclose(y, 3.0 * x / (1.0 + 2.0 * x), rtol=1e-10)

    def test_divalent_satisfies_mass_action(self):
        x, K, C, Q = 0.2, 2.5, 0.01, 2.0
        y = float(ad.ion_exchange_equilibrium(x, K, z_A=2, z_B=1, C_total=C, Q=Q))
        assert (y / x) * ((1 - x) / (1 - y)) ** 2 == pytest.approx(K * Q / C, rel=1e-8)
        # dilution favours the divalent ion
        y_conc = float(ad.ion_exchange_equilibrium(x, K, z_A=2, z_B=1, C_total=1.0, Q=Q))
        assert y > y_conc

    def test_ends_and_gradient(self):
        assert float(ad.ion_exchange_equilibrium(0.0, 3.0)) == 0.0
        assert float(ad.ion_exchange_equilibrium(1.0, 3.0)) == 1.0
        f = lambda x, K: ad.ion_exchange_equilibrium(x, K, z_A=2, z_B=1, C_total=0.05, Q=2.0)
        check_grads(f, (0.3, 2.0), order=1, modes=["rev"], rtol=1e-4)


class TestChromatography:
    def test_definitions(self):
        assert float(ad.retention_factor(6.0, 2.0)) == 2.0
        assert float(ad.plate_count(10.0, 1.0)) == 1600.0
        # half-height width of a Gaussian is 2 sqrt(2 ln 2) sigma; base width 4 sigma
        s = 0.25
        assert float(ad.plate_count(10.0, 2 * np.sqrt(2 * np.log(2)) * s, "half")) == pytest.approx(
            float(ad.plate_count(10.0, 4 * s)))
        assert float(ad.resolution(10.0, 12.0, 1.0, 1.0)) == 2.0

    def test_purnell_consistent_with_peaks(self):
        """Two Gaussian peaks of N plates: R_s from widths equals Purnell (equal-N, k1 ~ k2 form)."""
        N, t0, k1, alpha = 2500.0, 1.0, 3.0, 1.2
        k2 = alpha * k1
        t1, t2 = t0 * (1 + k1), t0 * (1 + k2)
        Rs_w = ad.resolution(t1, t2, 4 * t1 / np.sqrt(N), 4 * t2 / np.sqrt(N))
        # Purnell with the width of the later peak: R_s = (sqrt N/4)(t2 - t1)/t2
        assert float(ad.purnell_resolution(N, alpha, k2)) == pytest.approx(
            np.sqrt(N) / 4 * (t2 - t1) / t2, rel=1e-12)
        assert float(Rs_w) == pytest.approx(float(ad.purnell_resolution(N, alpha, k2)), rel=0.08)
        assert float(ad.plates_for_resolution(ad.purnell_resolution(N, alpha, k2), alpha, k2)) == pytest.approx(N)

    def test_van_deemter_optimum_is_minimum(self):
        A, B, C = 0.002, 0.004, 0.01
        u_opt, H_min = ad.van_deemter_optimum(A, B, C)
        assert float(jax.grad(ad.van_deemter)(u_opt, A, B, C)) == pytest.approx(0.0, abs=1e-12)
        assert float(ad.van_deemter(u_opt, A, B, C)) == pytest.approx(float(H_min))

    def test_gaussian_peak_area_and_plates(self):
        t = jnp.linspace(0.0, 20.0, 4001)
        y = ad.gaussian_peak(t, 10.0, 400.0, area=2.0)
        assert float(jnp.trapezoid(y, t)) == pytest.approx(2.0, rel=1e-6)
        mean = jnp.trapezoid(t * y, t) / 2.0
        var = jnp.trapezoid((t - mean) ** 2 * y, t) / 2.0
        assert float(mean ** 2 / var) == pytest.approx(400.0, rel=1e-4)
