"""Kent-Eisenberg H2S-CO2-amine equilibrium (``difflow_cc.equilibrium.acid_gas``, #471).

The fixed constants are checked against independent literature values, the
speciation against its own balances, and the fitted model against the data
it was fitted to AND against data held out of the fit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow_cc.equilibrium import acid_gas as ag
from tests.cc import acid_gas_data as D
from tests.reference.amine_ke_fit import FIT_SETS, aapd, log_error

jax.config.update("jax_enable_x64", True)

T25 = 298.15
LOG10 = np.log(10.0)


def pK(c, T=T25):
    return -float(ag._lnk(c, T)) / LOG10


class TestFixedConstants:
    """The Edwards (1978) / Posey (1996) constants against independent values."""

    def test_water(self):
        # IAPWS (Bandura & Lvov 2006, J. Phys. Chem. Ref. Data 35:15):
        # pKw = 13.995 at 25 C and saturation pressure
        assert abs(pK(ag.LNK_WATER) - 13.995) < 0.02

    def test_carbonic_acid(self):
        # Harned & Davis (1943) JACS 65:2030, doi:10.1021/ja01250a059,
        # pK1 = 6.352; Harned & Scholes (1941) JACS 63:1706,
        # doi:10.1021/ja01851a058, pK2 = 10.329 (25 C; the values usually
        # quoted for these papers, the tables not re-read)
        assert abs(pK(ag.LNK_CO2) - 6.352) < 0.02
        assert abs(pK(ag.LNK_HCO3) - 10.329) < 0.02

    def test_hydrogen_sulfide(self):
        # Hershey, Plese & Millero (1988) Geochim. Cosmochim. Acta 52:2047:
        # pK1 = 6.98 at 25 C
        assert abs(pK(ag.LNK_H2S) - 6.98) < 0.04

    def test_amine_pka(self):
        # Bower, Robinson & Bates (1962) J. Res. NBS 66A:71, DEA pKa = 8.883
        # at 25 C (read in the paper)
        assert abs(pK(ag.KE_AMINES["DEA"].protonation) - 8.883) < 0.02

    @pytest.mark.parametrize("T", [278.15, 298.15, 323.15, 353.15])
    def test_mdea_pka_against_perez_salado_kamps(self, T):
        # Perez-Salado Kamps & Maurer (1996) J. Chem. Eng. Data 41:1505,
        # doi:10.1021/je960141+, molality scale, 278-368 K:
        # ln K = -819.7/T - 79.474 + 10.9756 ln T  (8.551 at 25 C; the
        # correlation as quoted by OSTI 2418163, Eq. 26). Posey's constant
        # is an independent fit to other data and agrees to 0.02
        lnK = -819.7 / T - 79.474 + 10.9756 * np.log(T)
        assert abs(pK(ag.KE_AMINES["MDEA"].protonation, T) + lnK / LOG10) < 0.025

    def test_henry_constants(self):
        # Sander (2015) Atmos. Chem. Phys. 15:4399, doi:10.5194/acp-15-4399-2015,
        # Hcp at 25 C: CO2 3.4e-4 (top-ranked entries; 3.3e-4 also listed),
        # H2S 1.0e-3 mol/(m3 Pa); dilute water, mol/L ~ mol/kg
        assert abs(float(ag.henry_molal(T25, "CO2")) * 3.4e-7 - 1.0) < 0.03
        assert abs(float(ag.henry_molal(T25, "H2S")) * 1.0e-6 - 1.0) < 0.03

    def test_solubility_falls_with_temperature(self):
        for gas in ("CO2", "H2S"):
            H = [float(ag.henry_molal(T, gas)) for T in (300.0, 340.0, 380.0)]
            assert H[0] < H[1] < H[2]


@pytest.mark.parametrize("amine,aC,aS", [
    ("MDEA", 0.1, 0.05), ("MDEA", 0.6, 0.0), ("MDEA", 0.0, 0.4),
    ("DEA", 0.3, 0.1), ("DEA", 0.05, 0.0), ("DEA", 0.0, 0.8),
])
def test_speciation_closes_its_balances(amine, aC, aS):
    m = ag.amine_molality(0.4, amine)
    out = ag.ke_speciate(amine, 330.0, m, aC, aS)
    assert bool(out["converged"])
    sp = out["species"]
    a, c, s, pos, neg = (float(v) for v in ag._sums(sp))
    m = float(m)
    np.testing.assert_allclose(a, m, rtol=1e-10)
    np.testing.assert_allclose(c, max(aC, ag.TRACE) * m, rtol=1e-10)
    np.testing.assert_allclose(s, max(aS, ag.TRACE) * m, rtol=1e-10)
    np.testing.assert_allclose(pos, neg, rtol=1e-10)
    # every reaction is at its equilibrium
    lnK = ag._constants(ag.ke_amine(amine), 330.0, max(aC, ag.TRACE) + max(aS, ag.TRACE),
                        np.log(m))
    np.testing.assert_allclose(np.log(sp["H+"] * sp["OH-"]), lnK["w"], atol=1e-10)
    np.testing.assert_allclose(np.log(sp["H+"] * sp["HS-"] / sp["H2S"]), lnK["d"], atol=1e-10)
    np.testing.assert_allclose(np.log(sp["H+"] * sp["Am"] / sp["AmH+"]), lnK["f"], atol=1e-10)


@pytest.mark.parametrize("amine", ["MDEA", "DEA"])
def test_trace_loading_is_henrys_law(amine):
    """At infinite dilution of one acid gas (the other sets the pH) its
    pressure is linear in its loading, down to loadings a stripped bottoms
    reaches (~1e-17, #475); TRACE only guards the log, far below that."""
    m = ag.amine_molality(0.45, amine)
    k = [float(ag.ke_speciate(amine, 395.0, m, a, 0.01)["p_CO2"]) / a for a in (1e-12, 1e-17, 1e-24)]
    np.testing.assert_allclose(k[1:], k[0], rtol=1e-8)
    assert ag.TRACE < 1e-24


@pytest.mark.parametrize("amine", ["MDEA", "DEA"])
def test_unloaded_amine_is_the_weak_base(amine):
    # closed form: [OH-] = sqrt(Kb m) with Kb = Kw / K_f, the thermodynamic
    # K_f (Posey 1996). The apparent correction at zero loading is
    # d0 + d1 ln(alpha0); it moves pH by half of it over ln 10, so it must
    # stay bounded (a bare ln(alpha) moved it by 3 units)
    am = ag.KE_AMINES[amine]
    m = float(ag.amine_molality(0.4, amine))
    out = ag.ke_speciate(amine, T25, m, 0.0, 0.0)
    pKw = -float(ag._lnk(ag.LNK_WATER, T25)) / LOG10
    pKf = -float(ag._lnk(am.protonation, T25)) / LOG10
    pH_thermo = pKw - 0.5 * (pKw - pKf - np.log10(m))
    shift = -0.5 * (am.apparent[0] + am.apparent[1] * np.log(ag.ALPHA0)) / LOG10
    np.testing.assert_allclose(float(out["pH"]), pH_thermo + shift, atol=0.01)
    assert abs(shift) < 1.0


class TestAgainstData:
    """Fit quality (AAPD in p) on the fitting data, and on data held out."""

    @pytest.mark.parametrize("amine,i,bound", [("MDEA", 0, 20.0), ("MDEA", 1, 10.0),
                                               ("DEA", 0, 15.0), ("DEA", 1, 10.0)])
    def test_fitting_data(self, amine, i, bound):
        rows, gas = FIT_SETS[amine][i]
        assert aapd(log_error(ag.KE_AMINES[amine], rows, gas)) < bound

    def test_mdea_co2_held_out(self):
        # Huttenhuis et al. 2007 Table 1, 10 and 25 C: not in the fit
        # (which has CO2 at 40 C only); 11.5 %
        assert aapd(log_error(ag.KE_AMINES["MDEA"], D.HUTTENHUIS_CO2_MDEA, "CO2")) < 15.0

    def test_dea_held_out(self):
        # Lee et al. 1972 and Lawson & Garst 1976, as tabulated by Barreau.
        # The model is 18.0 % (CO2) and 12.8 % (H2S) off them; the two data
        # sources are 14.2 % and 5.2 % off EACH OTHER at the same points.
        am = ag.KE_AMINES["DEA"]
        assert aapd(log_error(am, D.LITERATURE_CO2_DEA, "CO2")) < 20.0
        assert aapd(log_error(am, D.LITERATURE_H2S_DEA, "H2S")) < 15.0

    def test_no_point_off_by_a_factor_of_two(self):
        for amine, sets in FIT_SETS.items():
            for rows, gas in sets:
                assert np.max(np.abs(np.asarray(log_error(ag.KE_AMINES[amine], rows, gas)))) \
                    < np.log(2.0)


class TestChemistry:
    def test_dea_holds_co2_tighter_than_mdea_at_low_loading(self):
        # the carbamate: a secondary amine binds CO2 1:2, so at low loading
        # the CO2 back-pressure over DEA is far below that over MDEA
        pd, _ = ag.ke_partial_pressures("DEA", 313.15, 0.3, 0.1, 0.0)
        pm, _ = ag.ke_partial_pressures("MDEA", 313.15, 0.3, 0.1, 0.0)
        assert float(pd) < 0.2 * float(pm)

    def test_co2_raises_the_h2s_back_pressure(self):
        # both acids compete for the same amine
        _, p0 = ag.ke_partial_pressures("MDEA", 313.15, 0.45, 0.0, 0.1)
        _, p1 = ag.ke_partial_pressures("MDEA", 313.15, 0.45, 0.2, 0.1)
        assert float(p1) > 1.5 * float(p0)

    def test_stripping_temperature_raises_back_pressure(self):
        for amine in ("MDEA", "DEA"):
            p = [float(ag.ke_partial_pressures(amine, T, 0.4, 0.2, 0.05)[1])
                 for T in (313.15, 353.15, 393.15)]
            assert p[0] < p[1] < p[2]

    @pytest.mark.parametrize("amine", ["MDEA", "DEA"])
    def test_loading_inverts_the_pressures(self, amine):
        pc, ps = ag.ke_partial_pressures(amine, 323.15, 0.4, 0.25, 0.12)
        out = ag.ke_loading(amine, 323.15, 0.4, pc, ps)
        assert bool(out["converged"])
        np.testing.assert_allclose(float(out["alpha_CO2"]), 0.25, rtol=1e-8)
        np.testing.assert_allclose(float(out["alpha_H2S"]), 0.12, rtol=1e-8)

    def test_loading_with_one_gas_absent(self):
        out = ag.ke_loading("MDEA", 313.15, 0.5, 0.0, 1e3)
        assert float(out["alpha_CO2"]) < 1e-10
        assert 0.0 < float(out["alpha_H2S"]) < 1.0

    def test_unknown_amine(self):
        with pytest.raises(ValueError, match="no Kent-Eisenberg"):
            ag.ke_amine("MEA")


class TestGradients:
    def test_pressures(self):
        def f(x):
            pc, ps = ag.ke_partial_pressures("DEA", x[0], 0.35, x[1], x[2])
            return jnp.log(pc) + 2.0 * jnp.log(ps)
        check_grads(f, (jnp.array([340.0, 0.3, 0.1]),), order=1, modes=["fwd", "rev"],
                    atol=1e-5, rtol=1e-5)

    def test_loading(self):
        def f(x):
            out = ag.ke_loading("MDEA", x[0], 0.45, x[1], x[2])
            return out["alpha_CO2"] + 3.0 * out["alpha_H2S"]
        check_grads(f, (jnp.array([320.0, 5e3, 2e3]),), order=1, modes=["fwd", "rev"],
                    atol=1e-5, rtol=1e-5)

    def test_jit(self):
        f = jax.jit(lambda T: ag.ke_partial_pressures("MDEA", T, 0.5, 0.2, 0.1)[1])
        np.testing.assert_allclose(
            float(f(330.0)), float(ag.ke_partial_pressures("MDEA", 330.0, 0.5, 0.2, 0.1)[1]),
            rtol=1e-12)


@pytest.mark.parametrize("amine", ["MDEA", "DEA"])
def test_constants_are_the_fit(amine):
    """Refit: the constants in KE_AMINES are still the least-squares optimum."""
    from tests.reference.amine_ke_fit import fit
    x = fit(amine)
    am = ag.KE_AMINES[amine]
    have = list(am.apparent[:2]) + (list(am.carbamate) if am.carbamate else [])
    np.testing.assert_allclose(have, x, rtol=1e-5, atol=1e-6)
