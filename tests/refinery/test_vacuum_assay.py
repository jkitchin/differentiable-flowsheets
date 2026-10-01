"""Tests for difflow_refinery: correlations, thermo and characterization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.vacuum import correlations as corr
from difflow_refinery.vacuum.assay import (
    atmospheric_residue,
    characterize,
    heavy_crude,
    light_crude,
    product_properties,
    tbp_fraction,
    tbp_temperature,
)
from difflow_refinery.vacuum.thermo import ColumnThermo, steam_enthalpy

C_TO_K = 273.15


# -- correlations ---------------------------------------------------------


class TestTwu:
    def test_reproduces_n_hexadecane(self):
        # Tb 560 K, SG 0.777: Tc 723 K, Pc 14.0 bar, MW 226.4 (Poling et al.)
        t = corr.twu_critical_properties(560.0, 0.777)
        assert float(t["Tc"]) == pytest.approx(723.0, rel=0.01)
        assert float(t["Pc"]) == pytest.approx(14.0e5, rel=0.03)
        assert float(t["MW"]) == pytest.approx(226.4, rel=0.01)

    def test_acentric_factor_of_n_hexadecane(self):
        t = corr.twu_critical_properties(560.0, 0.777)
        om = corr.lee_kesler_acentric(560.0, t["Tc"], t["Pc"],
                                      corr.watson_k(560.0, 0.777))
        assert float(om) == pytest.approx(0.742, abs=0.02)

    def test_sane_through_the_heavy_end(self):
        """Up to 800 C TBP: Tc above Tb, Pc falling, MW and omega rising."""
        Tb = jnp.linspace(300.0, 800.0, 21) + C_TO_K
        SG = corr.sg_from_watson_k(Tb, 11.8)
        t = corr.twu_critical_properties(Tb, SG)
        om = corr.lee_kesler_acentric(Tb, t["Tc"], t["Pc"], 11.8)
        for v in (t["Tc"], t["Pc"], t["MW"], om):
            assert jnp.all(jnp.isfinite(v))
        assert jnp.all(t["Tc"] > Tb)
        assert jnp.all(jnp.diff(t["Pc"]) < 0)
        assert jnp.all(jnp.diff(t["MW"]) > 0)
        assert jnp.all(jnp.diff(om) > 0)

    def test_breaks_past_the_alkane_reference(self):
        """Why the residue lump's properties are set directly."""
        Tb = 900.0 + C_TO_K
        t = corr.twu_critical_properties(Tb, corr.sg_from_watson_k(Tb, 11.8))
        assert not np.isfinite(complex(t["MW"]).real) or not np.isreal(t["Pc"])

    def test_riazi_daubert_agrees_where_both_are_fitted(self):
        Tb = jnp.asarray([350.0, 400.0, 450.0]) + C_TO_K
        SG = corr.sg_from_watson_k(Tb, 11.8)
        twu = corr.twu_critical_properties(Tb, SG)["MW"]
        rd = corr.riazi_daubert_mw(Tb, SG)
        assert jnp.all(jnp.abs(rd / twu - 1.0) < 0.15)


class TestMaxwellBonnell:
    def test_normal_boiling_point_is_one_atmosphere(self):
        for TbC in (350.0, 500.0, 700.0):
            Tb = TbC + C_TO_K
            p = corr.maxwell_bonnell_psat(Tb, Tb, 11.8)
            assert float(p) == pytest.approx(101325.0, rel=3e-3)

    def test_n_hexadecane_against_its_antoine_equation(self):
        # NIST: log10(P/bar) = 4.17312 - 1845.672 / (T - 117.054), 463-560 K
        Tb, SG = 560.0, 0.777
        for T in (480.0, 520.0):
            ref = 10 ** (4.17312 - 1845.672 / (T - 117.054)) * 1e5
            p = corr.maxwell_bonnell_psat(T, Tb, corr.watson_k(Tb, SG))
            assert float(p) == pytest.approx(ref, rel=0.05)

    def test_d1160_conversion(self):
        """A 500 C AEBP cut boils near 330 C at 10 mmHg (D1160 tables)."""
        T = corr.maxwell_bonnell_boiling_point(10 * corr.MMHG, 500.0 + C_TO_K)
        assert float(T) - C_TO_K == pytest.approx(330.0, abs=8.0)

    def test_boiling_point_inverts_psat(self):
        Tb = 550.0 + C_TO_K
        T = corr.maxwell_bonnell_boiling_point(1000.0, Tb, 11.5)
        assert float(corr.maxwell_bonnell_psat(T, Tb, 11.5)) == pytest.approx(1000.0, rel=1e-9)

    def test_smooth_and_monotone_down_to_a_fraction_of_a_kpa(self):
        """The VDU range: ln psat is smooth across the formula joins."""
        Tb = 600.0 + C_TO_K
        T = jnp.linspace(450.0, 750.0, 600)
        lnp = jnp.log(corr.maxwell_bonnell_psat(T, Tb, 11.8))
        assert float(jnp.exp(lnp[0])) < 100.0          # well below 1 kPa
        d = jnp.diff(lnp) / jnp.diff(T)
        assert jnp.all(d > 0)
        # no kink: slope changes by less than 1% between neighbours
        assert float(jnp.max(jnp.abs(jnp.diff(d)) / d[1:])) < 0.01

    def test_lee_kesler_agrees_near_the_boiling_point(self):
        Tb = 450.0 + C_TO_K
        SG = corr.sg_from_watson_k(Tb, 11.8)
        t = corr.twu_critical_properties(Tb, SG)
        om = corr.lee_kesler_acentric(Tb, t["Tc"], t["Pc"], 11.8)
        T = Tb - 60.0
        mb = corr.maxwell_bonnell_psat(T, Tb, 11.8)
        lk = corr.lee_kesler_psat(T, t["Tc"], t["Pc"], om)
        assert float(mb / lk) == pytest.approx(1.0, abs=0.15)

    def test_antoine_fit_over_a_vacuum_range(self):
        (A, B, C), err = corr.fit_antoine(500.0 + C_TO_K, 11.8, T_range=(450.0, 650.0))
        assert float(err) < 0.05
        assert jnp.isfinite(A) and B > 0


@pytest.fixture(scope="module")
def thermo():
    return ColumnThermo(characterize(heavy_crude()).components)


class TestThermo:
    def test_heat_of_vaporization_is_plausible(self, thermo):
        # J/kg at 350 C: a few hundred kJ/kg for gas oils
        dh = thermo.dh_vap(623.15) / thermo.c.MW * 1000.0
        assert jnp.all((dh[:10] > 1.5e5) & (dh[:10] < 4.0e5))

    def test_liquid_cp_is_plausible(self, thermo):
        cp = jax.jacfwd(lambda T: thermo.h_liquid(T))(623.15) / thermo.c.MW * 1000.0
        assert jnp.all((cp > 2.0e3) & (cp < 3.6e3))

    def test_enthalpies_consistent_with_k_values(self, thermo):
        """dHvap is the Clausius-Clapeyron slope of the same psat."""
        T = 600.0
        h = 1e-3
        slope = (thermo.log_psat(T + h) - thermo.log_psat(T - h)) / (2 * h)
        np.testing.assert_allclose(thermo.dh_vap(T), 8.314462618 * T ** 2 * slope,
                                   rtol=1e-6)

    def test_steam_enthalpy(self):
        # ideal-gas water, 25 -> 300 C: about 9.6 kJ/mol
        assert float(steam_enthalpy(573.15)) == pytest.approx(9.57e3, rel=0.03)


# -- the TBP curve and the characterization -------------------------------


class TestTBP:
    @pytest.mark.parametrize("assay", [light_crude(), heavy_crude()])
    def test_passes_through_the_data(self, assay):
        x = tbp_fraction(assay, jnp.asarray(assay.tbp_C, dtype=float))
        np.testing.assert_allclose(x, np.asarray(assay.tbp_wt) / 100.0, rtol=1e-12)

    @pytest.mark.parametrize("assay", [light_crude(), heavy_crude()])
    def test_monotone_and_extended_past_565(self, assay):
        x = tbp_fraction(assay, jnp.linspace(-50.0, 1000.0, 2000))
        assert jnp.all(jnp.diff(x) > 0)
        assert 0.0 < float(x[0]) and float(x[-1]) < 1.0

    def test_inverse(self):
        a = heavy_crude()
        for f in (0.05, 0.3, 0.7, 0.9):
            assert float(tbp_fraction(a, tbp_temperature(a, f))) == pytest.approx(f, rel=1e-10)

    def test_derivative_is_single_valued_at_a_data_point(self):
        """C1 at the knots, so a cut boundary on a TBP point has one slope."""
        a = heavy_crude()

        def x_at_500(t):
            return tbp_fraction(a.with_tbp_point(9, t), 500.0)

        g = jax.grad(x_at_500)(500.0)
        h = 1e-4
        fd_up = (x_at_500(500.0 + h) - x_at_500(500.0)) / h
        fd_dn = (x_at_500(500.0) - x_at_500(500.0 - h)) / h
        assert float(fd_up) == pytest.approx(float(g), rel=1e-3)
        assert float(fd_dn) == pytest.approx(float(g), rel=1e-3)


@pytest.fixture(scope="module", params=["light", "heavy"])
def char(request):
    return characterize(light_crude() if request.param == "light" else heavy_crude())


class TestCharacterize:
    def test_mass_closes(self, char):
        assert float(jnp.sum(char.yields) + char.light_ends) == pytest.approx(1.0, abs=1e-12)

    def test_boiling_points_inside_their_cuts(self, char):
        c = char.components
        assert jnp.all((c.Tb[:-1] > c.T_lo[:-1]) & (c.Tb[:-1] < c.T_hi[:-1]))
        assert jnp.all(jnp.diff(c.Tb) > 0)

    def test_heavier_cuts_are_denser_and_dirtier(self, char):
        c = char.components
        for v in (c.SG, c.sulfur, c.nitrogen, c.ccr, c.nickel_vanadium):
            assert jnp.all(jnp.diff(v) >= 0)
        assert jnp.all(jnp.diff(c.MW[:-1]) > 0)

    def test_residue_lump_properties_are_set_directly(self):
        char = characterize(heavy_crude(), residue_mw=1800.0, residue_sg=1.08)
        assert float(char.components.MW[-1]) == 1800.0
        assert float(char.components.SG[-1]) == pytest.approx(1.08)

    def test_watson_k_reproduces_bulk_gravity(self):
        """Constant Kw, fitted so the whole crude has its bulk SG."""
        a = heavy_crude()
        fine = np.arange(0.0, 800.1, 10.0)
        ch = characterize(a, cuts_C=fine)
        c = ch.components
        Tb0 = tbp_temperature(a, 0.5 * ch.light_ends) + C_TO_K
        sg0 = corr.sg_from_watson_k(Tb0, ch.Kw)
        inv = ch.light_ends / sg0 + jnp.sum(ch.yields / c.SG)
        assert float(1.0 / inv) == pytest.approx(float(a.bulk_sg()), rel=1e-10)

    def test_contaminants_reproduce_bulk_values(self):
        a = heavy_crude()
        ch = characterize(a, cuts_C=np.arange(0.0, 800.1, 10.0))
        s = float(jnp.sum(ch.yields * ch.components.sulfur))
        # the part below 0 C carries next to no sulfur
        assert s == pytest.approx(a.sulfur_wt / 100.0, rel=1e-3)

    def test_atmospheric_residue_is_the_heavy_end(self, char):
        feed = atmospheric_residue(char, 100.0, cut_point_C=370.0)
        c = char.components
        m = jnp.stack([feed[f"F_{n}"] for n in c.names]) * c.MW / 1000.0
        props = product_properties(c, m)
        assert 350.0 < float(props["T05"]) - C_TO_K < 420.0
        assert float(props["rate"]) == pytest.approx(
            100.0 * (1.0 - float(tbp_fraction(
                light_crude() if float(c.SG[0]) < 0.86 else heavy_crude(), 370.0))),
            rel=0.05)

    def test_characterization_is_differentiable_in_a_tbp_point(self):
        a = heavy_crude()

        def mw_bulk(t):
            c = characterize(a.with_tbp_point(10, t))
            return jnp.sum(c.yields * c.components.MW)

        g = jax.grad(mw_bulk)(550.0)
        h = 0.01
        fd = (mw_bulk(550.0 + h) - mw_bulk(550.0 - h)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-5)
