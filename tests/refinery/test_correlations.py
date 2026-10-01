"""Petroleum characterisation correlations against pure-compound data.

The reference set is thirteen hydrocarbons spanning the paraffins, naphthenes
and aromatics a crude cut is made of, C5 to C20. Critical constants and
acentric factors: Poling, Prausnitz & O'Connell, *The Properties of Gases and
Liquids*, 5th ed., Appendix A; SG at 60 F: API Technical Data Book.

The tolerances are the correlations' published accuracy, not this
implementation's: a correlation that missed them would be wrongly coded, and
one that met them much better than published would be suspicious too.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import difflow  # noqa: F401  (enables x64)
from difflow_refinery import correlations as corr

# name, Tb (K), SG, MW, Tc (K), Pc (bar), omega
REF = [
    ("n-pentane", 309.22, 0.6311, 72.15, 469.7, 33.70, 0.252),
    ("n-hexane", 341.88, 0.6640, 86.18, 507.6, 30.25, 0.300),
    ("n-heptane", 371.58, 0.6882, 100.20, 540.2, 27.40, 0.350),
    ("n-octane", 398.83, 0.7070, 114.23, 568.7, 24.90, 0.399),
    ("n-decane", 447.30, 0.7342, 142.28, 617.7, 21.10, 0.490),
    ("n-dodecane", 489.47, 0.7536, 170.34, 658.0, 18.20, 0.576),
    ("n-hexadecane", 559.98, 0.7773, 226.45, 723.0, 14.00, 0.718),
    ("n-eicosane", 616.93, 0.7928, 282.55, 768.0, 11.60, 0.907),
    ("cyclohexane", 353.87, 0.7834, 84.16, 553.5, 40.73, 0.211),
    ("methylcyclohexane", 374.08, 0.7740, 98.19, 572.2, 34.71, 0.236),
    ("benzene", 353.24, 0.8845, 78.11, 562.0, 48.95, 0.210),
    ("toluene", 383.78, 0.8719, 92.14, 591.8, 41.08, 0.264),
    ("ethylbenzene", 409.35, 0.8717, 106.17, 617.2, 36.09, 0.304),
]
NAMES, TB, SG, MW, TC, PC, OMEGA = (np.array(c) if i else list(c) for i, c in enumerate(zip(*REF)))
PC = PC * 1e5
MW_REF = MW

# Average absolute deviation (%) allowed for (MW, Tc, Pc), per method: the
# published accuracy for this kind of compound, with some room.
AAD_LIMITS = {
    "twu": (1.5, 1.0, 3.0),
    "twu_1984": (1.5, 1.0, 3.0),
    "twu_legacy": (4.0, 1.0, 3.0),
    "riazi_daubert_1987": (4.0, 1.0, 4.5),
    "riazi_daubert_1980": (6.0, 1.5, 5.0),
    "lee_kesler": (7.0, 1.5, 5.5),
}


def _aad(estimate, reference):
    return 100.0 * float(np.mean(np.abs(np.asarray(estimate) / reference - 1.0)))


@pytest.mark.parametrize("method", corr.CRITICAL_METHODS)
def test_critical_properties_meet_published_accuracy(method):
    MW, Tc, Pc = corr.critical_properties(jnp.asarray(TB), jnp.asarray(SG), method)
    limits = AAD_LIMITS[method]
    for est, ref, limit, label in zip((MW, Tc, Pc), (MW_REF, TC, PC), limits, ("MW", "Tc", "Pc")):
        assert _aad(est, ref) < limit, f"{method} {label} AAD {_aad(est, ref):.2f}%"



def test_twu_reproduces_its_own_reference_series():
    """Twu's correlation is built on the n-alkanes: it should all but return them."""
    alkanes = slice(0, 7)  # to n-hexadecane; n-eicosane's Pc is the weak point
    MW, Tc, Pc = corr.critical_properties(jnp.asarray(TB[alkanes]), jnp.asarray(SG[alkanes]), "twu")
    np.testing.assert_allclose(MW, MW_REF[alkanes], rtol=0.01)
    np.testing.assert_allclose(Tc, TC[alkanes], rtol=0.01)
    np.testing.assert_allclose(Pc, PC[alkanes], rtol=0.02)


# Molecular weights, Poling-Prausnitz-O'Connell 5th ed. Appendix A; SG at
# 60 F, API Technical Data Book. Polynuclear aromatics are where Twu's SG
# perturbation does the most work, so they are what tells the two codings
# of it apart.
AROMATICS = [  # name, Tb (K), SG, MW
    ("naphthalene", 491.14, 1.0253, 128.17),
    ("phenanthrene", 613.0, 1.1800, 178.23),
    ("tetralin", 480.77, 0.9752, 132.20),
]


def test_twu_is_the_published_molecular_weight():
    """``twu`` carries Twu's Rankine constants; ``twu_legacy`` the Kelvin-form
    ones against sqrt(Tb in R), which under-corrects aromatics' MW.

    The published-constant coding is checked against independent codings of
    Twu (1984) Eqs. 21-22 (pychemqt ``lib/petro.py`` and sim21
    ``data/twu.py`` both have 0.328086 and 0.193168 with Tb in Rankine).
    """
    _, tb, sg, mw = (np.array(c) if i else c for i, c in enumerate(zip(*AROMATICS)))
    good = corr.critical_properties(jnp.asarray(tb), jnp.asarray(sg), "twu")[0]
    old = corr.critical_properties(jnp.asarray(tb), jnp.asarray(sg), "twu_legacy")[0]
    assert _aad(good, mw) < 3.0
    assert _aad(old, mw) > 3.0 * _aad(good, mw)
    # Tc and Pc do not involve the MW constants: the two codings agree.
    for a, b in zip(corr.critical_properties(TB, SG, "twu_legacy")[1:],
                    corr.critical_properties(TB, SG, "twu")[1:]):
        np.testing.assert_array_equal(a, b)
    # "twu_1984" is the same function under its paper's name.
    for a, b in zip(corr.critical_properties(tb, sg, "twu"),
                    corr.critical_properties(tb, sg, "twu_1984")):
        np.testing.assert_array_equal(a, b)
    # The dict form is the same function.
    d = corr.twu_critical_properties(tb, sg)
    np.testing.assert_allclose(d["MW"], good, rtol=1e-14)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="method"):
        corr.critical_properties(400.0, 0.75, "made_up")


def test_acentric_factor():
    omega = corr.acentric_factor(jnp.asarray(TB), jnp.asarray(TC), jnp.asarray(PC), jnp.asarray(SG))
    np.testing.assert_allclose(omega, OMEGA, atol=0.03)


def test_vapor_pressure_is_one_atmosphere_at_tb_with_lee_kesler_omega():
    """Lee-Kesler's omega is defined by inverting its own vapour pressure at Tb."""
    Tbr = TB / TC
    omega = (-np.log(PC / corr.P_ATM) - corr._lk_f0(Tbr)) / corr._lk_f1(Tbr)
    np.testing.assert_allclose(corr.vapor_pressure(jnp.asarray(TB), TC, PC, omega), corr.P_ATM, rtol=1e-10)


# n-hexane, n-heptane, n-decane, n-hexadecane, toluene, cyclohexane at 298 K:
# ideal-gas Cp, liquid Cp (J/mol/K) and Hvap at Tb (kJ/mol). Poling et al. and
# NIST Webbook.
THERMAL = {
    "n-hexane": (143.1, 195.6, 28.85),
    "n-heptane": (165.98, 224.7, 31.77),
    "n-decane": (233.1, 314.4, 38.75),
    "n-hexadecane": (370.2, 499.7, 51.8),
    "toluene": (103.7, 157.1, 33.18),
    "cyclohexane": (106.3, 156.0, 29.97),
}


def _cubic(coeffs, T):
    return sum(c * T**k for k, c in enumerate(np.asarray(coeffs)))


@pytest.mark.parametrize("name", list(THERMAL))
def test_heat_capacities_and_hvap(name):
    i = NAMES.index(name)
    cp_ig, cp_l, hvap = THERMAL[name]
    est_ig = _cubic(corr.cp_ideal_gas_coeffs(TB[i], SG[i], MW_REF[i]), 298.15)
    est_l = _cubic(corr.cp_liquid_coeffs(TB[i], SG[i], MW_REF[i]), 298.15)
    est_h = float(corr.hvap_at_tb(TB[i], TC[i], PC[i])) / 1e3
    # Watson-Nelson ideal gas and Kesler-Lee liquid are ~5% methods for
    # petroleum fractions; benzene-ring compounds are their worst case.
    assert abs(est_ig / cp_ig - 1) < 0.10, est_ig
    assert abs(est_l / cp_l - 1) < 0.10, est_l
    assert abs(est_h / hvap - 1) < 0.05, est_h


def test_ideal_gas_cp_over_temperature():
    """Watson-Nelson for n-heptane, 300-800 K (TRC tables)."""
    i = NAMES.index("n-heptane")
    coeffs = corr.cp_ideal_gas_coeffs(TB[i], SG[i], MW_REF[i])
    for T, ref in [(300, 166.5), (400, 210.9), (500, 252.6), (600, 288.0), (800, 342.3)]:
        assert abs(_cubic(coeffs, T) / ref - 1) < 0.05


def test_watson_k_and_api():
    # n-alkanes sit near 12.7, aromatics near 10
    assert 12.3 < float(corr.watson_k(TB[4], SG[4])) < 13.0
    assert 9.6 < float(corr.watson_k(TB[10], SG[10])) < 10.2
    assert float(corr.api_from_sg(1.0)) == pytest.approx(10.0)
    assert float(corr.sg_from_api(corr.api_from_sg(0.85))) == pytest.approx(0.85)


@pytest.mark.parametrize("method", corr.CRITICAL_METHODS)
def test_gradients_match_finite_differences(method):
    from jax.test_util import check_grads

    def f(Tb, SG):
        MW, Tc, Pc = corr.critical_properties(Tb, SG, method)
        return MW / 100 + Tc / 500 + Pc / 1e6 + corr.acentric_factor(Tb, Tc, Pc, SG)

    for Tb, sg in [(400.0, 0.75), (650.0, 0.90), (850.0, 0.98)]:
        check_grads(f, (Tb, sg), order=1, modes=["fwd", "rev"], rtol=1e-4)
