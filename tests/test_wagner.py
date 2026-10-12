"""Wagner vapor pressures for the light gases (issue #442, Ch. 2).

References are NOT the coefficients' sources: saturation pressures and
normal boiling points from the reference equations of state that the NIST
WebBook fluid tables use (Setzmann & Wagner for methane, Span et al. for
N2, ...), evaluated with CoolProp 8.0. Methane's normal boiling point,
111.667 K at 101 325 Pa, is the issue's example.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import IdealThermo
from difflow.database import WAGNER_COEFFS, WAGNER_SOURCES, get_species_data
from difflow.thermo import AntoineRangeWarning, psat_wagner

jax.config.update("jax_enable_x64", True)

#: Reference-EOS normal boiling points (K); CO2 has none (it sublimes).
TB_REF = {
    "methane": 111.6672, "ethane": 184.5686, "propane": 231.0362,
    "n_butane": 272.6599, "isobutane": 261.401, "nitrogen": 77.355,
    "oxygen": 90.1878, "ammonia": 239.8343, "ethylene": 169.3786,
    "propylene": 225.5308, "carbon_monoxide": 81.6382,
}

#: (species, T K, reference-EOS Psat Pa) at 25/50/90% of the Antoine range.
REF_PSAT = [
    ("methane", 115.6, 138464), ("methane", 140.3, 651189),
    ("methane", 179.7, 3.25283e6),
    ("ethane", 164.3, 29176.5), ("ethane", 192.9, 155500),
    ("ethane", 238.6, 925613),
    ("propane", 195.9, 15614.4), ("propane", 243.8, 172167),
    ("propane", 320.6, 1.62021e6),
    ("n_butane", 233.2, 16837.3), ("n_butane", 265.9, 77792.1),
    ("n_butane", 318.2, 434746),
    ("isobutane", 208.9, 7055.31), ("isobutane", 254.7, 77332.1),
    ("isobutane", 327.9, 768398),
    ("nitrogen", 71.3, 46456.6), ("nitrogen", 79.4, 128088),
    ("nitrogen", 92.5, 443864),
    ("oxygen", 73.5, 11447.6), ("oxygen", 92.7, 130762),
    ("oxygen", 123.3, 1.23173e6),
    ("carbon_dioxide", 234.3, 1.0477e6), ("carbon_dioxide", 252.7, 1.94244e6),
    ("carbon_dioxide", 282.0, 4.37518e6),
    ("ammonia", 280.1, 552656), ("ammonia", 320.7, 1.90672e6),
    ("ammonia", 385.6, 7.93218e6),
    ("ethylene", 150.2, 27803.4), ("ethylene", 173.4, 127624),
    ("ethylene", 210.6, 687354),
    ("propylene", 189.1, 13345.6), ("propylene", 231.9, 134162),
    ("propylene", 300.5, 1.22352e6),
    ("carbon_monoxide", 74.3, 40105.9), ("carbon_monoxide", 78.9, 73316.4),
    ("carbon_monoxide", 86.2, 165281),
]


def _th(species, vp="wagner"):
    return IdealThermo({species: get_species_data(species, vapor_pressure=vp)})


def test_every_set_is_tested_and_sourced():
    assert set(TB_REF) | {"carbon_dioxide"} == set(WAGNER_COEFFS)
    assert {s for s, *_ in REF_PSAT} == set(WAGNER_COEFFS)
    for s in WAGNER_COEFFS:
        assert ("Poling" in WAGNER_SOURCES[s]) or ("McGarry" in WAGNER_SOURCES[s])


@pytest.mark.parametrize("species", sorted(TB_REF))
def test_normal_boiling_point(species):
    """Psat(Tb_ref) = 1 atm to 0.5%, and Tb itself to 0.1 K."""
    th = _th(species)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntoineRangeWarning)
        P = float(th.Psat(species, TB_REF[species]))
        Tb = float(th.saturation_T(species, 101325.0))
    assert P / 101325.0 == pytest.approx(1.0, abs=5e-3)
    assert Tb == pytest.approx(TB_REF[species], abs=0.1)


def test_methane_issue_example():
    th = _th("methane")
    assert float(th.Psat("methane", 111.67)) == pytest.approx(101325.0, rel=2e-3)
    assert float(th.saturation_T("methane", 101325.0)) == pytest.approx(
        111.667, abs=0.02)
    form, co, _ = WAGNER_COEFFS["methane"]
    # At its own Tc the Wagner equation returns its own Pc exactly.
    assert float(psat_wagner(co[4], co, form)) == pytest.approx(co[5], rel=1e-12)


@pytest.mark.parametrize("species,T,P_ref", REF_PSAT)
def test_against_reference_eos(species, T, P_ref):
    P = float(_th(species).Psat(species, T))
    assert P == pytest.approx(P_ref, rel=0.01)


def test_methane_error_decreases_vs_prydz_goodwin():
    """The point of the issue: 2.4% (Antoine) -> 0.15% (Wagner) mid-range."""
    pts = [(T, P) for s, T, P in REF_PSAT if s == "methane"] + [(150.0, 1.03996e6)]
    ant, wag = _th("methane", "antoine"), _th("methane")
    e_ant = max(abs(float(ant.Psat("methane", T)) / P - 1) for T, P in pts)
    e_wag = max(abs(float(wag.Psat("methane", T)) / P - 1) for T, P in pts)
    assert e_ant > 0.02
    assert e_wag < 0.0015
    assert e_wag < e_ant / 10


def test_wagner_within_one_percent_to_near_tc():
    """Every set within 1% of the reference EOS at its test points.

    (Oxygen's NIST Antoine set is tighter mid-range, 0.3% vs 0.7%, but stops
    at 131 K; the McGarry set covers the liquid to Tc.)
    """
    for s in WAGNER_COEFFS:
        pts = [(T, P) for sp, T, P in REF_PSAT if sp == s]
        e_w = max(abs(float(_th(s).Psat(s, T)) / P - 1) for T, P in pts)
        assert e_w < 0.01, s


def test_default_is_still_antoine():
    """Existing results do not move: the default record has no Wagner set."""
    d = get_species_data("methane")
    assert d.wagner_coeffs is None
    A, B, C = d.antoine_coeffs
    th = IdealThermo({"methane": d})
    assert float(th.Psat("methane", 150.0)) == pytest.approx(10 ** (A - B / (150.0 + C)))


def test_lookup_modes():
    assert get_species_data("water", vapor_pressure="best").wagner_coeffs is None
    assert get_species_data("co2", vapor_pressure="best").wagner_form == "3-6"
    with pytest.raises(KeyError, match="Wagner"):
        get_species_data("water", vapor_pressure="wagner")
    with pytest.raises(ValueError):
        get_species_data("methane", vapor_pressure="raoult")


def test_supercritical_continuation_is_smooth():
    """Above Tc the leading term continues ln P with matching value and slope."""
    form, co, _ = WAGNER_COEFFS["methane"]
    Tc = co[4]
    f = lambda T: jnp.log(psat_wagner(T, co, form))
    lo, hi = f(Tc - 1e-6), f(Tc + 1e-6)
    assert float(hi - lo) == pytest.approx(0.0, abs=1e-6)
    dlo, dhi = jax.grad(f)(Tc - 1e-4), jax.grad(f)(Tc + 1e-4)
    assert float(dhi) == pytest.approx(float(dlo), rel=1e-3)
    assert jnp.isfinite(jax.grad(f)(300.0)) and float(jax.grad(f)(300.0)) > 0
    # analytic supercritical form
    T = 300.0
    assert float(f(T)) == pytest.approx(math.log(co[5]) + co[0] * (Tc / T - 1.0))


def test_range_warning_uses_wagner_range():
    th = _th("methane")
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntoineRangeWarning)
        th.Psat("methane", 185.0)   # inside Wagner range (to Tc 190.55)
    with pytest.warns(AntoineRangeWarning, match="Wagner"):
        th.Psat("methane", 250.0)


@pytest.mark.parametrize("species", ["methane", "propane", "carbon_dioxide"])
def test_gradients(species):
    th = _th(species)
    T = {"methane": 150.0, "propane": 280.0, "carbon_dioxide": 260.0}[species]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_grads(lambda t: jnp.log(th.Psat(species, t)), (T,), order=2,
                    modes=["rev", "fwd"])
    # Clausius-Clapeyron: d ln P / d(1/T) = -Hvap/R < 0
    g = jax.grad(lambda t: jnp.log(th.Psat(species, t)))(T)
    assert float(g) > 0.0


def test_gradient_wrt_coefficients():
    form, co, _ = WAGNER_COEFFS["nitrogen"]

    def f(c):
        return jnp.log(psat_wagner(90.0, tuple(c), form))

    check_grads(f, (jnp.array(co),), order=1, modes=["rev"])
