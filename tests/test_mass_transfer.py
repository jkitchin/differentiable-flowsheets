"""Tests for difflow.mass_transfer (#427).

What is and is NOT verified here
--------------------------------
Verified against independent oracles: the Fuller formula re-evaluated by hand
(cm^2/s) and against two data points I am reasonably (not perfectly) sure of
(water vapour-air at 25 C, about 2.5e-5 m^2/s; O2-N2 at 273 K, about 0.18
cm^2/s); Wilke-Chang against a hand evaluation and against the experimental
D of ethanol and CO2 in water at 25 C (about 1.2e-9 and 1.9e-9 m^2/s, from
recollection, to the 10-20% accuracy that correlation claims); the stagnant
film against ``ln((1-y2)/(1-y1))``; Colburn's NTU against scipy quadrature
and the Gauss-Legendre integral; Onda against an independent NumPy
transcription of the published equations.

Published worked examples (added for book Ch. 14, ``TestPublishedExamples``):
Poling et al. 5e Example 11-3 (Fuller, allyl chloride-air), Geankoplis Example
6.2-5 (Fuller with the 1966 volumes, n-butanol-air) and Example 6.3-2
(Wilke-Chang with Le Bas, acetone-water). The reference answers are recalled
from the printed books, not copied from a scan.

NOT verified: no published Onda worked example is reproduced. The packing
and Le Bas / Fuller tables are from memory (the entries used by the three
examples above are consistent with them). The Onda regression fixes the order of magnitude of
k_L (1e-4 m/s, a well-known property of packed-column liquid films) and of
k_G; it is not a published-example match.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.integrate import quad

from difflow import heat_transfer
from difflow import mass_transfer as mt

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------
# Diffusivities
# ----------------------------------------------------------------------------


class TestDiffusivities:
    def test_fuller_hand_calculation(self):
        T, P, MA, MB, VA, VB = 350.0, 2.0e5, 44.01, 28.96, 26.9, 19.7
        expected = (
            1e-3 * T**1.75 * math.sqrt(1 / MA + 1 / MB) / ((P / 101325.0) * (VA ** (1 / 3) + VB ** (1 / 3)) ** 2) * 1e-4
        )
        assert float(mt.fuller_diffusivity(T, P, MA, MB, VA, VB)) == pytest.approx(expected, rel=1e-12)

    def test_fuller_water_air_data_point(self):
        # Data (Cussler / Perry, from recollection): ~2.5-2.6e-5 m^2/s at 298 K, 1 atm.
        D = mt.fuller_diffusivity(298.15, 101325.0, 18.015, 28.96, mt.FULLER_VOLUMES["H2O"], mt.FULLER_VOLUMES["air"])
        assert float(D) == pytest.approx(2.55e-5, rel=0.04)

    def test_fuller_o2_n2_data_point(self):
        # ~0.18 cm^2/s at 273 K, 1 atm (recollection, moderate confidence); Fuller claims ~5-10%.
        D = mt.fuller_diffusivity(273.15, 101325.0, 32.0, 28.01, mt.FULLER_VOLUMES["O2"], mt.FULLER_VOLUMES["N2"])
        assert float(D) * 1e4 == pytest.approx(0.18, rel=0.06)

    def test_fuller_scaling(self):
        base = mt.fuller_diffusivity(300.0, 1e5, 18.0, 29.0, 13.1, 19.7)
        assert float(mt.fuller_diffusivity(600.0, 1e5, 18.0, 29.0, 13.1, 19.7) / base) == pytest.approx(2**1.75)
        assert float(mt.fuller_diffusivity(300.0, 2e5, 18.0, 29.0, 13.1, 19.7) / base) == pytest.approx(0.5)

    def test_wilke_chang_hand_calculation(self):
        T, mu, MB, VA, phi = 298.15, 0.89e-3, 18.015, 59.2, 2.6
        expected = 7.4e-8 * math.sqrt(phi * MB) * T / (0.89 * VA**0.6) * 1e-4
        assert float(mt.wilke_chang_diffusivity(T, mu, MB, VA, phi)) == pytest.approx(expected, rel=1e-12)

    def test_wilke_chang_data_points(self):
        # Experimental D at 25 C (recollection): ethanol ~1.2e-9, CO2 ~1.9e-9 m^2/s;
        # Wilke-Chang's stated accuracy ~10-20% (it over-predicts ethanol by ~20%).
        V_eth = mt.molar_volume_nbp({"C": 2, "H": 6, "O": 1})
        assert V_eth == pytest.approx(59.2)
        D_eth = mt.wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, V_eth, mt.ASSOCIATION_FACTORS["water"])
        assert float(D_eth) == pytest.approx(1.24e-9, rel=0.22)
        with pytest.warns(mt.WilkeChangAccuracyWarning, match="small solute"):   # #448
            D_co2 = mt.wilke_chang_diffusivity(
                298.15, 0.89e-3, 18.015, mt.MOLAR_VOLUMES_NBP["CO2"], mt.ASSOCIATION_FACTORS["water"]
            )
        assert float(D_co2) == pytest.approx(1.92e-9, rel=0.15)

    def test_wilke_chang_no_warning_for_organic_solute(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", mt.WilkeChangAccuracyWarning)
            mt.wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, 59.2, 2.6)
            jax.grad(lambda v: mt.wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, v, 2.6))(25.8)

    def test_stokes_einstein(self):
        D = mt.stokes_einstein_diffusivity(298.15, 0.89e-3, 1.0e-9)
        assert float(D) == pytest.approx(1.380649e-23 * 298.15 / (6 * math.pi * 0.89e-3 * 1e-9))

    def test_tables(self):
        assert mt.diffusion_volume("air") == 19.7
        assert mt.diffusion_volume({"C": 1, "O": 2}) == pytest.approx(15.9 + 2 * 6.11)
        assert mt.molar_volume_nbp({"C": 6, "H": 6, "ring6": 1}) == pytest.approx(6 * 14.8 + 6 * 3.7 - 15.0)  # benzene 96.0
        with pytest.raises(KeyError):
            mt.diffusion_volume("unobtainium")

    def test_grads(self):
        f = lambda T, P: mt.fuller_diffusivity(T, P, 18.015, 28.96, 13.1, 19.7)  # noqa: E731
        check_grads(f, (300.0, 1.2e5), order=1, modes=["rev"], rtol=1e-5)
        g = lambda T, mu: mt.wilke_chang_diffusivity(T, mu, 18.015, 59.2, 2.6)  # noqa: E731
        check_grads(g, (300.0, 0.8e-3), order=1, modes=["rev"], rtol=1e-5)


class TestPublishedExamples:
    """Textbook worked examples (book Ch. 14). The reference numbers are recalled
    from the printed books, not copied from a scan; see plan/gaps-ch14.md."""

    def test_poling_example_11_3_allyl_chloride_air(self):
        # Poling, Prausnitz & O'Connell 5e Ex. 11-3: allyl chloride in air, 298 K, 1 bar.
        # Sum v = 3(15.9) + 5(2.31) + 21.0 = 80.25; Fuller gives 0.096 cm^2/s (exp. 0.098).
        V = mt.diffusion_volume({"C": 3, "H": 5, "Cl": 1})
        assert V == pytest.approx(80.25)
        D = mt.fuller_diffusivity(298.0, 1.0e5, 76.53, 28.97, V, mt.FULLER_VOLUMES["air"])
        assert float(D) * 1e4 == pytest.approx(0.096, abs=0.0006)

    def test_geankoplis_example_6_2_5_butanol_air_1966_volumes(self):
        # Geankoplis Ex. 6.2-5: n-butanol in air, 1966 Fuller volumes (91.28, 20.1);
        # about 7.7e-6 m^2/s at 0 C, 1 atm, half that at 2 atm; ~9.0e-6 at 25.9 C.
        V = mt.diffusion_volume({"C": 4, "H": 10, "O": 1}, table="1966")
        assert V == pytest.approx(91.28)
        assert mt.diffusion_volume("air", table="1966") == 20.1
        D0 = mt.fuller_diffusivity(273.0, 101325.0, 74.1, 29.0, V, 20.1)
        assert float(D0) == pytest.approx(7.7e-6, rel=0.01)
        assert float(mt.fuller_diffusivity(273.0, 2 * 101325.0, 74.1, 29.0, V, 20.1)) == pytest.approx(float(D0) / 2)
        assert float(mt.fuller_diffusivity(299.1, 101325.0, 74.1, 29.0, V, 20.1)) == pytest.approx(9.05e-6, rel=0.01)

    def test_geankoplis_example_6_3_2_acetone_water(self):
        # Geankoplis Ex. 6.3-2: acetone in water, Wilke-Chang with Le Bas V_A = 0.0740 m^3/kmol:
        # 1.277e-9 m^2/s at 25 C (exp. 1.28e-9) and 2.251e-9 at 50 C.
        V = mt.molar_volume_nbp({"C": 3, "H": 6, "O": 1})
        assert V == pytest.approx(74.0)
        D25 = mt.wilke_chang_diffusivity(298.15, 0.8937e-3, 18.02, V, 2.6)
        D50 = mt.wilke_chang_diffusivity(323.15, 0.5494e-3, 18.02, V, 2.6)
        assert float(D25) == pytest.approx(1.277e-9, rel=0.002)
        assert float(D50) == pytest.approx(2.251e-9, rel=0.002)

    def test_unknown_table(self):
        with pytest.raises(ValueError):
            mt.diffusion_volume("air", table="1999")

    def test_le_bas_oxygen_environments(self):
        # Le Bas: methyl ester 9.1, methyl ether 9.9, higher ester/ether 11.0, acid 12.0.
        inc = mt.LE_BAS_INCREMENTS
        assert (inc["O_methyl_ester"], inc["O_methyl_ether"], inc["O_ester_ether"], inc["O_acid"]) == (
            9.1, 9.9, 11.0, 12.0)
        assert inc["O_ether"] == inc["O_methyl_ether"]


class TestPenetration:
    def test_penetration_and_renewal(self):
        D, t = 2.0e-9, 0.5
        assert float(mt.penetration_coefficient(D, t)) == pytest.approx(2 * math.sqrt(D / (math.pi * t)))
        # Danckwerts with s = 4/(pi t_c) matches Higbie
        assert float(mt.surface_renewal_coefficient(D, 4 / (math.pi * t))) == pytest.approx(
            float(mt.penetration_coefficient(D, t)))

    def test_falling_film(self):
        G, rho, mu, D, L = 0.02, 998.0, 1.0e-3, 2.0e-9, 1.0
        delta = (3 * mu * G / (rho**2 * mt.GRAVITY)) ** (1 / 3)
        assert float(mt.falling_film_thickness(G, rho, mu)) == pytest.approx(delta)
        vmax = 1.5 * G / (rho * delta)
        k = mt.falling_film_liquid_coefficient(D, G, rho, mu, L)
        assert float(k) == pytest.approx(math.sqrt(4 * D * vmax / (math.pi * L)))
        assert float(k) == pytest.approx(float(mt.penetration_coefficient(D, L / vmax)))
        # k_L ~ Gamma^(1/3): doubling the rate raises k_L by 2^(1/3)
        assert float(mt.falling_film_liquid_coefficient(D, 2 * G, rho, mu, L) / k) == pytest.approx(2 ** (1 / 3))

    def test_grads(self):
        # D scaled to O(1) so the finite-difference step in check_grads is sensible
        f = lambda d, G: 1e5 * mt.falling_film_liquid_coefficient(d * 1e-9, G, 998.0, 1.0e-3, 1.0)  # noqa: E731
        check_grads(f, (2.0, 0.02), order=1, modes=["rev"], rtol=1e-4)


# ----------------------------------------------------------------------------
# Film theory
# ----------------------------------------------------------------------------


class TestFilm:
    def test_log_mean(self):
        assert float(mt.log_mean(3.0, 1.0)) == pytest.approx(2.0 / math.log(3.0))
        assert float(mt.log_mean(1.0, 3.0)) == pytest.approx(2.0 / math.log(3.0))

    @pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-6, 0.0999 * 0.5, 0.1001 * 0.5])
    def test_log_mean_smooth_through_equality(self, eps):
        a = 0.7
        b = a * (1.0 + eps)
        ref = a if eps == 0 else a * eps / math.log1p(eps)  # cancellation-free
        assert float(mt.log_mean(a, b)) == pytest.approx(ref, rel=1e-12)

    def test_log_mean_series_matches_closed_form_at_switch(self):
        # x = (a-b)/(a+b) = 0.05 +- tiny: series and arctanh branches agree to ~1e-13.
        a, b = 1.1052631578947368, 1.0  # x = 0.05 exactly
        below = mt.log_mean(a * (1 - 1e-9), b)
        above = mt.log_mean(a * (1 + 1e-9), b)
        assert float(abs(above - below)) < 1e-8
        assert float(mt.log_mean(a, b)) == pytest.approx((a - b) / math.log(a / b), rel=1e-12)

    def test_log_mean_gradient_at_equality(self):
        da, db = jax.grad(mt.log_mean, argnums=(0, 1))(2.0, 2.0)
        assert float(da) == pytest.approx(0.5, abs=1e-12)
        assert float(db) == pytest.approx(0.5, abs=1e-12)
        check_grads(mt.log_mean, (2.0, 2.5), order=1, modes=["rev"])

    def test_stagnant_film_closed_form(self):
        D, c, d = 2.5e-5, 41.0, 0.1
        y1, y2 = 0.3, 0.05
        expected = D * c / d * math.log((1 - y2) / (1 - y1))
        assert float(mt.stagnant_film_flux(D, c, y1, y2, d)) == pytest.approx(expected, rel=1e-12)

    def test_stagnant_film_reduces_to_equimolar_dilute(self):
        D, c, d = 1e-5, 40.0, 0.01
        for y1 in [1e-3, 1e-5, 1e-8]:
            n_s = float(mt.stagnant_film_flux(D, c, y1, 0.0, d))
            n_e = float(mt.equimolar_flux(D, c, y1, 0.0, d))
            assert n_s / n_e == pytest.approx(1.0, abs=y1)
        assert float(mt.stagnant_film_flux(D, c, 0.2, 0.0, d)) > float(mt.equimolar_flux(D, c, 0.2, 0.0, d))

    def test_stagnant_film_equal_ends_finite_gradient(self):
        g = jax.grad(lambda y: mt.stagnant_film_flux(1e-5, 40.0, y, 0.2, 0.01))(0.2)
        assert np.isfinite(float(g)) and float(g) == pytest.approx(1e-5 * 40.0 / 0.01 / 0.8, rel=1e-9)
        assert float(mt.stagnant_film_flux(1e-5, 40.0, 0.2, 0.2, 0.01)) == 0.0

    def test_water_evaporation_physical_range(self):
        from difflow.units.evaporator import water_saturation_pressure

        P, T = 101325.0, 298.15
        D = mt.fuller_diffusivity(T, P, 18.015, 28.96, 13.1, 19.7)
        y1 = water_saturation_pressure(T) / P
        N = mt.stagnant_film_flux(D, P / (8.314462618 * T), y1, 0.0, 0.01)  # 1 cm film
        # order check only: D c y/d = 2.5e-5 * 40.9 * 0.031 / 0.01 ~ 3.2e-3 mol/m^2/s
        assert 2.5e-3 < float(N) < 4.5e-3

    def test_coefficient_conversions(self):
        T, P = 320.0, 2.0e5
        k_c = 0.02
        k_y = float(mt.mass_transfer_coefficient_conversions(k_c, "c", "y", T, P))
        assert k_y == pytest.approx(k_c * P / (8.314462618 * T))
        k_G = float(mt.mass_transfer_coefficient_conversions(k_c, "c", "G", T, P))
        assert k_G == pytest.approx(k_c / (8.314462618 * T))
        assert float(mt.mass_transfer_coefficient_conversions(k_y, "y", "c", T, P)) == pytest.approx(k_c)
        assert float(mt.mass_transfer_coefficient_conversions(k_G, "G", "y", T, P)) == pytest.approx(k_y)
        assert float(mt.k_L_from_k_x(mt.k_x_from_k_L(1e-4, 55500.0), 55500.0)) == pytest.approx(1e-4)
        with pytest.raises(ValueError):
            mt.mass_transfer_coefficient_conversions(1.0, "c", "z", T, P)


# ----------------------------------------------------------------------------
# Sherwood
# ----------------------------------------------------------------------------


class TestSherwood:
    def test_ranz_marshall_limit(self):
        assert float(mt.ranz_marshall(0.0, 1.0)) == 2.0
        assert float(mt.ranz_marshall(1e-12, 1000.0)) == pytest.approx(2.0, abs=1e-5)
        assert float(mt.frossling(0.0, 1.0)) == 2.0

    def test_ranz_marshall_value(self):
        assert float(mt.ranz_marshall(100.0, 1.0)) == pytest.approx(2.0 + 0.6 * 10.0)
        assert float(mt.frossling(100.0, 8.0)) == pytest.approx(2.0 + 0.552 * 10.0 * 2.0)

    def test_turbulent_pipe(self):
        assert float(mt.linton_sherwood(1e4, 1000.0)) == pytest.approx(0.023 * 1e4**0.83 * 10.0)
        assert float(mt.gilliland_sherwood(1e4, 1.0)) == pytest.approx(0.023 * 1e4**0.83)

    def test_pipe_blend(self):
        assert float(mt.pipe_sherwood(100.0, 1.0)) == pytest.approx(3.66)
        assert float(mt.pipe_sherwood(1e5, 5.0)) == pytest.approx(float(mt.linton_sherwood(1e5, 5.0)))
        vals = [float(mt.pipe_sherwood(r, 5.0)) for r in np.geomspace(2000, 5000, 40)]
        assert all(np.isfinite(vals))
        assert np.all(np.diff(vals) > -1e-9)  # no dip across the blend
        with pytest.raises(ValueError):
            mt.pipe_sherwood(1e4, 1.0, method="nope")

    def test_flat_plate_is_heat_analogue(self):
        for reg in ("laminar", "turbulent", "mixed"):
            assert float(mt.flat_plate_sherwood(1e6, 2.0, reg)) == pytest.approx(
                float(heat_transfer.flat_plate_nusselt(1e6, 2.0, reg))
            )
        assert float(mt.flat_plate_sherwood(1e4, 1.0, "laminar")) == pytest.approx(0.664 * 100.0)

    def test_chilton_colburn(self):
        Nu, Re, Pr, Sc = 150.0, 1e4, 0.7, 0.6
        j = float(mt.chilton_colburn_jD(Nu, Re, Pr))
        assert j == pytest.approx(Nu / (Re * 0.7 ** (1 / 3)))
        assert float(mt.sherwood_from_heat_transfer(Nu, Re, Pr, Sc)) == pytest.approx(Nu * (Sc / Pr) ** (1 / 3))

    def test_naphthalene_sphere_physical(self):
        # Sublimation of a 1 cm sphere in air at 1 m/s: Sh of order 20-60, k_c of order 0.01-0.1 m/s.
        Re = 1.18 * 1.0 * 0.01 / 1.85e-5
        Sc = 1.85e-5 / (1.18 * 0.6e-5)
        Sh = mt.ranz_marshall(Re, Sc)
        k = mt.film_coefficient(Sh, 0.6e-5, 0.01)
        assert 10 < float(Sh) < 60 and 0.003 < float(k) < 0.05
        assert float(mt.sherwood(k, 0.01, 0.6e-5)) == pytest.approx(float(Sh))
        assert float(mt.schmidt(1.85e-5, 1.18, 0.6e-5)) == pytest.approx(Sc)


# ----------------------------------------------------------------------------
# Two-resistance
# ----------------------------------------------------------------------------


class TestTwoResistance:
    def test_overall_coefficients(self):
        k_y, k_x, m = 0.5, 2.0, 3.0
        assert float(mt.overall_gas_coefficient(k_y, k_x, m)) == pytest.approx(1 / (1 / 0.5 + 3 / 2.0))
        # K_y = m K_x ... 1/K_x = 1/(m k_y) + 1/k_x  =>  K_x = m K_y
        assert float(mt.overall_liquid_coefficient(k_y, k_x, m)) == pytest.approx(
            m * float(mt.overall_gas_coefficient(k_y, k_x, m))
        )
        # enhancement shrinks the liquid resistance
        assert float(mt.overall_gas_coefficient(k_y, k_x, m, E=10.0)) > float(mt.overall_gas_coefficient(k_y, k_x, m))

    def test_overall_K_G(self):
        k_G, k_L, H = 1e-6, 1e-4, 3000.0  # mol/m2/s/Pa, m/s, Pa m3/mol
        assert float(mt.overall_K_G(k_G, k_L, H)) == pytest.approx(1 / (1 / k_G + H / k_L))

    def test_resistance_fractions(self):
        g, l = mt.resistance_fractions(0.5, 2.0, 3.0)
        assert float(g + l) == pytest.approx(1.0)
        # very soluble gas (small m): gas-film control; sparingly soluble (large m): liquid control
        g_nh3, _ = mt.resistance_fractions(0.05, 5.0, 0.01)
        g_co2, _ = mt.resistance_fractions(0.05, 5.0, 1.0e5)
        assert float(g_nh3) > 0.95 and float(g_co2) < 0.05


# ----------------------------------------------------------------------------
# Onda
# ----------------------------------------------------------------------------


def _onda_reference(u_G, rho_G, mu_G, D_G, u_L, rho_L, mu_L, D_L, sigma, sigma_c, a_p, d_p):
    """Independent NumPy transcription of Onda et al. (1968), mass-flux form."""
    g = 9.80665
    L, G = rho_L * u_L, rho_G * u_G
    a_w = a_p * (
        1
        - math.exp(
            -1.45
            * (sigma_c / sigma) ** 0.75
            * (L / (a_p * mu_L)) ** 0.1
            * (L**2 * a_p / (rho_L**2 * g)) ** -0.05
            * (L**2 / (rho_L * sigma * a_p)) ** 0.2
        )
    )
    C = 5.23 if d_p >= 0.015 else 2.00
    k_c = C * a_p * D_G * (G / (a_p * mu_G)) ** 0.7 * (mu_G / (rho_G * D_G)) ** (1 / 3) * (a_p * d_p) ** -2
    kL_lhs = 0.0051 * (L / (a_w * mu_L)) ** (2 / 3) * (mu_L / (rho_L * D_L)) ** -0.5 * (a_p * d_p) ** 0.4
    k_L = kL_lhs / (rho_L / (mu_L * g)) ** (1 / 3)
    return a_w, k_c, k_L


class TestOnda:
    kw = dict(u_G=1.0, rho_G=1.2, mu_G=1.8e-5, D_G=1.6e-5, u_L=0.01, rho_L=1000.0, mu_L=1e-3, D_L=1.5e-9, sigma=0.072)

    @pytest.mark.parametrize("d_p,a_p,sigma_c", [(0.025, 250.0, 0.075), (0.05, 100.0, 0.061), (0.010, 400.0, 0.033)])
    def test_matches_independent_transcription(self, d_p, a_p, sigma_c):
        k = self.kw
        a_w, k_c, k_L = _onda_reference(
            k["u_G"], k["rho_G"], k["mu_G"], k["D_G"], k["u_L"], k["rho_L"], k["mu_L"], k["D_L"], k["sigma"], sigma_c, a_p, d_p
        )
        pk = mt.PackingData("t", d_p, a_p, 0.9, sigma_c)
        assert float(mt.onda_wetted_area(k["u_L"], k["rho_L"], k["mu_L"], k["sigma"], pk)) == pytest.approx(a_w, rel=2e-5)
        # (g differs by 9.81 vs 9.80665 in the old code only; ours matches the reference exactly)
        assert float(mt.onda_gas_coefficient(k["u_G"], k["rho_G"], k["mu_G"], k["D_G"], pk, smooth=False)) == pytest.approx(
            k_c, rel=1e-12
        )
        assert float(
            mt.onda_liquid_coefficient(k["u_L"], k["rho_L"], k["mu_L"], k["D_L"], k["sigma"], pk)
        ) == pytest.approx(k_L, rel=1e-9)

    def test_k_L_order_of_magnitude_regression(self):
        """Guards the old bug: (rho g/mu)^(1/3) instead of (mu g/rho)^(1/3) gave 0.49 m/s."""
        k_L = float(
            mt.onda_liquid_coefficient(
                0.01, 1010.0, 2e-3, 1.5e-9, 0.072, a_p=250.0, d_p=0.05, sigma_c=0.075
            )
        )
        assert 5e-5 < k_L < 5e-4  # packed-column liquid films: ~1e-4 m/s
        # with a_p in place of a_w (the old Re basis) and the correct factor: issue's 7.7e-5
        a_w = float(mt.onda_wetted_area(0.01, 1010.0, 2e-3, 0.072, a_p=250.0, sigma_c=0.075))
        assert a_w < 250.0
        Re_p = 1010.0 * 0.01 / (250.0 * 2e-3)
        k_old_basis = 0.0051 * Re_p ** (2 / 3) * (2e-3 / (1010.0 * 1.5e-9)) ** -0.5 * (250 * 0.05) ** 0.4 * (2e-3 * 9.80665 / 1010.0) ** (1 / 3)
        assert k_old_basis == pytest.approx(7.7e-5, rel=0.02)
        assert k_L > k_old_basis  # smaller a_w -> larger Re -> larger k_L

    def test_k_G_order_of_magnitude(self):
        k_c = float(mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, a_p=250.0, d_p=0.025))
        assert 5e-3 < k_c < 0.2  # k_c of order 1e-2 m/s for gas films in packing
        # old value was larger by a_p d_p = 6.25
        old = 5.23 * (1.6e-5 / 0.025) * (1.2 / (250 * 1.8e-5)) ** 0.7 * (1.8e-5 / (1.2 * 1.6e-5)) ** (1 / 3)
        assert old / k_c == pytest.approx(250 * 0.025, rel=1e-9)

    def test_small_packing_constant_and_smooth_blend(self):
        a = dict(a_p=300.0)
        r = lambda d, **kw: float(  # noqa: E731
            mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, d_p=d, **a, **kw)
        )
        # C = 2.00 below 15 mm, 5.23 above (published step), after removing the d_p^-2 factor
        assert r(0.010, smooth=False) * 0.010**2 * 1.0 / (r(0.020, smooth=False) * 0.020**2) == pytest.approx(2.00 / 5.23)
        assert r(0.010) == pytest.approx(r(0.010, smooth=False), rel=1e-12)
        assert r(0.020) == pytest.approx(r(0.020, smooth=False), rel=1e-12)
        ds = np.linspace(0.0121, 0.0179, 50)
        v = np.array([r(d) * d**2 for d in ds])
        assert np.all(np.diff(v) > 0)  # monotone, continuous through 15 mm

    def test_wetted_area_bounds_and_trend(self):
        lo = float(mt.onda_wetted_area(0.005, 1000.0, 1e-3, 0.072, a_p=250.0, sigma_c=0.075))
        hi = float(mt.onda_wetted_area(0.02, 1000.0, 1e-3, 0.072, a_p=250.0, sigma_c=0.075))
        assert 0 < lo < hi < 250.0

    def test_packing_record(self):
        pk = mt.PACKINGS["pall_metal_50"]
        a1 = mt.onda_wetted_area(0.01, 1000.0, 1e-3, 0.072, pk)
        a2 = mt.onda_wetted_area(0.01, 1000.0, 1e-3, 0.072, a_p=pk.a_p, sigma_c=pk.sigma_c)
        assert float(a1) == float(a2)
        with pytest.raises(ValueError):
            mt.onda_wetted_area(0.01, 1000.0, 1e-3, 0.072)
        with pytest.raises(ValueError):
            mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, a_p=100.0)

    def test_grads(self):
        pk = mt.PACKINGS["pall_metal_25"]
        # d_p in mm so the finite-difference step of check_grads (1e-4) is small relative to it
        kG = lambda uG, d: mt.onda_gas_coefficient(uG, 1.2, 1.8e-5, 1.6e-5, a_p=pk.a_p, d_p=d * 1e-3)  # noqa: E731
        check_grads(kG, (1.0, 25.0), order=1, modes=["rev"], rtol=1e-5)
        kL = lambda uL, d: mt.onda_liquid_coefficient(  # noqa: E731
            uL * 1e-2, 1000.0, 1e-3, 1.5e-9, 0.072, a_p=pk.a_p, d_p=d * 1e-3, sigma_c=pk.sigma_c
        )
        check_grads(kL, (1.0, 25.0), order=1, modes=["rev"], rtol=1e-5)
        aw = lambda uL: mt.onda_wetted_area(uL * 1e-2, 1000.0, 1e-3, 0.072, a_p=pk.a_p, sigma_c=pk.sigma_c)  # noqa: E731
        check_grads(aw, (1.0,), order=1, modes=["rev"], rtol=1e-5)
        g = jax.grad(kG, argnums=1)(1.0, 0.0149)
        assert np.isfinite(float(g))


# ----------------------------------------------------------------------------
# HTU / NTU / HETP
# ----------------------------------------------------------------------------


class TestHtuNtu:
    def test_htu(self):
        assert float(mt.htu_og(2.0, 0.5, 100.0)) == pytest.approx(0.04)
        assert float(mt.htu_ol(10.0, 0.5, 100.0)) == pytest.approx(0.2)
        assert float(mt.htu_og_from_films(0.3, 0.2, 0.5)) == pytest.approx(0.4)
        # H_OG = H_G + lam H_L agrees with 1/(K_y a) route
        G, L, k_y, k_x, m, a = 2.0, 8.0, 1.5, 6.0, 0.8, 120.0
        lam = m * G / L
        via_K = mt.htu_og(G, mt.overall_gas_coefficient(k_y, k_x, m), a)
        via_films = mt.htu_og_from_films(mt.htu_og(G, k_y, a), mt.htu_ol(L, k_x, a), lam)
        assert float(via_K) == pytest.approx(float(via_films), rel=1e-12)

    # (lam, y_out): lam > 1 needs a feasible (non-pinched) spec, so a modest recovery there
    @pytest.mark.parametrize(
        "lam,y_out", [(0.0, 0.002), (0.3, 0.002), (0.9, 0.002), (1.0, 0.002), (1.0 + 1e-7, 0.002), (1.5, 0.04), (3.0, 0.049)]
    )
    def test_colburn_equals_quadrature(self, lam, y_out):
        y_in, x_in, m = 0.05, 1e-4, 1.2
        closed = float(mt.ntu_og_colburn(y_in, y_out, x_in, m, lam))
        # independent: scipy quad of dy/(y - y*) with the straight operating line
        ystar = lambda y: m * x_in + lam * (y - y_out)  # noqa: E731
        ref, _ = quad(lambda y: 1.0 / (y - ystar(y)), y_out, y_in, epsabs=1e-13, epsrel=1e-13)
        assert closed == pytest.approx(ref, rel=1e-9)
        n64 = float(mt.ntu_og_integral(y_in, y_out, lambda y: m * x_in + lam * (y - y_out), n=64))
        assert n64 == pytest.approx(closed, abs=1e-6)

    def test_colburn_textbook_formula(self):
        y_in, y_out, x_in, m, lam = 0.05, 0.002, 0.0, 1.0, 0.7
        direct = math.log((1 - lam) * (y_in - m * x_in) / (y_out - m * x_in) + lam) / (1 - lam)
        assert float(mt.ntu_og_colburn(y_in, y_out, x_in, m, lam)) == pytest.approx(direct, rel=1e-12)
        # lam -> 1 limit
        assert float(mt.ntu_og_colburn(y_in, y_out, x_in, m, 1.0)) == pytest.approx((y_in - y_out) / y_out)

    def test_curved_equilibrium_and_concentrated(self):
        ystar = lambda y: 0.5 * y + 2.0 * y**2  # noqa: E731  (any smooth curve below the diagonal)
        n = float(mt.ntu_og_integral(0.2, 0.01, ystar, n=64))
        ref, _ = quad(lambda y: 1.0 / (y - ystar(y)), 0.01, 0.2, epsabs=1e-13, epsrel=1e-13)
        assert n == pytest.approx(ref, rel=1e-9)
        nc = float(mt.ntu_og_integral(0.2, 0.01, ystar, n=64, concentrated=True))
        lm = lambda a, b: (a - b) / math.log(a / b)  # noqa: E731
        refc, _ = quad(
            lambda y: lm(1 - ystar(y), 1 - y) / ((1 - y) * (y - ystar(y))), 0.01, 0.2, epsabs=1e-13, epsrel=1e-13
        )
        assert nc == pytest.approx(refc, rel=1e-9)
        # a dilute limit
        n_d = float(mt.ntu_og_integral(1e-4, 1e-5, lambda y: 0.5 * y, concentrated=True))
        assert n_d == pytest.approx(float(mt.ntu_og_integral(1e-4, 1e-5, lambda y: 0.5 * y)), rel=1e-3)

    def test_hetp_continuous_at_lambda_one(self):
        H = 0.6
        assert float(mt.hetp_from_htu(H, 1.0)) == H
        for lam in [1 - 1e-2, 1 - 1e-4, 1 - 1e-6, 1 + 1e-6, 1 + 1e-4, 1 + 1e-2]:
            ref = H * math.log(lam) / (lam - 1)
            assert float(mt.hetp_from_htu(H, lam)) == pytest.approx(ref, rel=1e-9)
        assert float(mt.hetp_from_htu(H, 1.0 + 5e-4)) == pytest.approx(H * (1 - 2.5e-4), rel=1e-6)
        g = jax.grad(mt.hetp_from_htu, argnums=1)(H, 1.0)
        assert float(g) == pytest.approx(-H / 2, rel=1e-9)

    def test_grad_height_wrt_L_over_G(self):
        """d(H_OG * N_OG)/d(L/G): column height from the Onda film coefficients."""
        pk = mt.PACKINGS["pall_metal_50"]
        G_m = 40.0  # mol/m^2/s
        y_in, y_out, m = 0.02, 0.002, 0.8

        def height(LG):
            a_w = mt.onda_wetted_area(0.01, 1000.0, 1e-3, 0.072, pk)
            k_c = mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, pk)
            k_y = k_c * 101325.0 / (8.314462618 * 300.0)
            k_x = mt.k_x_from_k_L(mt.onda_liquid_coefficient(0.01, 1000.0, 1e-3, 1.5e-9, 0.072, pk), 55500.0)
            lam = m / LG
            Hog = mt.htu_og_from_films(mt.htu_og(G_m, k_y, a_w), mt.htu_ol(G_m * LG, k_x, a_w), lam)
            return Hog * mt.ntu_og_colburn(y_in, y_out, 0.0, m, lam)

        check_grads(height, (1.5,), order=1, modes=["rev"], rtol=1e-5)
        assert float(jax.grad(height)(1.5)) < 0  # more solvent, shorter column

    def test_colburn_grads(self):
        f = lambda lam, y: mt.ntu_og_colburn(0.05, y * 1e-3, 0.0, 1.0, lam)  # y_out in 1e-3 units
        check_grads(f, (0.7, 2.0), order=1, modes=["rev"], rtol=1e-5)
        for lam in (0.0, 1.0):
            assert np.isfinite(float(jax.grad(f)(lam, 2.0)))

    def test_ntu_integral_grad(self):
        f = lambda m: mt.ntu_og_integral(0.05, 0.002, lambda y: m * y, n=32)  # noqa: E731
        check_grads(f, (0.8,), order=1, modes=["rev"], rtol=1e-5)


# ----------------------------------------------------------------------------
# cc wrappers
# ----------------------------------------------------------------------------


class TestCCWrappers:
    def test_wrappers_import_warn_and_delegate(self):
        from difflow_cc import kinetics as K

        with pytest.warns(DeprecationWarning):
            k_G = K.gas_film_coefficient(1.0, 0.025, 1.8e-5, 1.2, 1.6e-5, 250.0)
        assert float(k_G) == pytest.approx(
            float(mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, a_p=250.0, d_p=0.025))
        )
        with pytest.warns(DeprecationWarning):
            k_L = K.liquid_film_coefficient(u_L=0.01, d_p=0.05, mu_L=2e-3, rho_L=1010.0, D_L=1.5e-9, a_p=250.0)
        assert 5e-5 < float(k_L) < 5e-4  # was 0.487 m/s before #427
        with pytest.warns(DeprecationWarning):
            a_w = K.interfacial_area(0.01, 1.0, 1000.0, 1e-3, 0.072, 250.0)
        assert 0 < float(a_w) < 250.0

    def test_ntu_htu_wrappers(self):
        from difflow_cc.kinetics.mass_transfer import height_of_transfer_unit, number_of_transfer_units

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert float(number_of_transfer_units(0.1, 0.01, 0.0)) == pytest.approx(math.log(10.0))
            assert float(number_of_transfer_units(0.1, 0.01, 0.005)) == pytest.approx(math.log(0.095 / 0.005))
            assert float(height_of_transfer_unit(1.0, 2.0, 3.0)) == pytest.approx(1.0 / 6.0)
