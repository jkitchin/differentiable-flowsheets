"""Tests for the product blending pool (difflow_refinery.blending)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow.streams import make_stream
from difflow_refinery import (
    PSI,
    RHO_WATER_15C,
    BlendComponent,
    BlendPool,
    BlendSpec,
    BlendCharacterization,
    EthylRT70,
    cetane_index_d4737,
    cetane_index_d976,
    ethyl_rt70,
    flash_point_blend,
    raoult_rvp,
    refutas_blend,
    refutas_vbn,
    refutas_viscosity,
    rvp_index_blend,
    smooth_violation,
    tbp_evaporated,
    tbp_temperature,
    tbp_to_d86,
    temperature_index_blend,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid():
    """n-butane plus a naphtha-to-diesel pseudocomponent grid."""
    names = ["nC4"] + [f"pc{i}" for i in range(14)]
    Tb = [272.66] + list(np.linspace(320.0, 640.0, 14))
    SG = [0.584] + list(np.linspace(0.66, 0.88, 14))
    nan = float("nan")
    return BlendCharacterization(
        names=names, Tb=Tb, SG=SG,
        MW=[58.12] + [nan] * 14, Tc=[425.12] + [nan] * 14,
        Pc=[37.96e5] + [nan] * 14, omega=[0.200] + [nan] * 14,
        qualities={
            "S_ppm": [0.0] + list(np.linspace(5.0, 4000.0, 14)),
            "aromatics_vol": [0.0] + list(np.linspace(5.0, 30.0, 14)),
            "olefins_vol": [0.0] * 15,
        })


@pytest.fixture(scope="module")
def char():
    return _grid()


def _stream(char, weights, scale=10.0):
    return make_stream({n: scale * w for n, w in zip(char.names, weights)},
                       T=310.0, P=2e5)


@pytest.fixture(scope="module")
def stream_components(char):
    n = char.n
    light = np.zeros(n); light[1:6] = [3, 4, 4, 3, 2]
    heavy = np.zeros(n); heavy[4:10] = [1, 2, 3, 3, 2, 1]
    butane = np.zeros(n); butane[0] = 1.0
    return [
        BlendComponent.from_stream("lsr", _stream(char, light), char,
                                   RON=70.0, MON=68.0),
        BlendComponent.from_stream("reformate", _stream(char, heavy), char,
                                   RON=98.0, MON=88.0, aromatics_vol=65.0),
        BlendComponent.from_stream("butane", _stream(char, butane, 2.0),
                                   char, RON=93.0, MON=92.0),
    ]


def _gasoline_components(**kw):
    base = dict(
        reformate=dict(SG=0.80, RON=98.0, MON=88.0, RVP_psi=3.5, S_ppm=1.0,
                       olefins_vol=1.0, aromatics_vol=65.0),
        fcc=dict(SG=0.74, RON=92.0, MON=80.0, RVP_psi=7.0, S_ppm=20.0,
                 olefins_vol=30.0, aromatics_vol=25.0),
        alkylate=dict(SG=0.70, RON=96.0, MON=93.5, RVP_psi=4.5, S_ppm=5.0,
                      olefins_vol=0.5, aromatics_vol=0.5),
        butane=dict(SG=0.584, RON=93.0, MON=92.0, RVP_psi=51.6, S_ppm=1.0,
                    olefins_vol=0.0, aromatics_vol=0.0),
    )
    for name, over in kw.items():
        base[name].update(over)
    return [BlendComponent.from_properties(n, **p) for n, p in base.items()]


RECIPE = jnp.array([0.35, 0.35, 0.25, 0.05])


# ---------------------------------------------------------------------------
# Rules, against hand calculation and their defining identities
# ---------------------------------------------------------------------------

class TestRefutas:
    def test_round_trip(self):
        for nu in [1.5, 10.0, 180.0, 380.0, 700.0]:
            assert float(refutas_viscosity(refutas_vbn(nu))) == \
                pytest.approx(nu, rel=1e-12)

    def test_hand_calculation(self):
        # 30 wt% of a 2.5 cSt cutter in 70 wt% of a 700 cSt residue, worked
        # through the Refutas equations by hand (here: with math, not jax).
        def vbn(nu):
            return 14.534 * math.log(math.log(nu + 0.8)) + 10.975
        blend_vbn = 0.3 * vbn(2.5) + 0.7 * vbn(700.0)
        expected = math.exp(math.exp((blend_vbn - 10.975) / 14.534)) - 0.8
        # equal SG, so volume fractions are the mass fractions
        got = refutas_blend(jnp.array([0.3, 0.7]), jnp.array([0.95, 0.95]),
                            jnp.array([2.5, 700.0]))
        assert float(got) == pytest.approx(expected, rel=1e-12)
        # a little cutter goes a long way: far below the 490 cSt linear mix
        assert 40.0 < float(got) < 200.0

    def test_published_worked_example(self):
        # K. Johnsen, "Blending by Index", Haverly Systems blog no. 60
        # (2 Oct 2019), https://www.haverly.com/kathy-blog/716-blog-60-blend-index:
        # a 30 cSt @ 100 C fuel oil from a 34 cSt base and a 1.5 cSt
        # diluent.  The post tabulates VBI 29.389, 8.318 and 28.880 and
        # finds 0.0242 weight fraction diluent (12.3 % if blended linearly).
        # Its formula is printed with 14.543, a typo for 14.534: 14.534
        # reproduces the diluent's 8.318 and both constants land within
        # 0.01 of the other two, inside the post's rounding.
        base, diluent, target = (float(refutas_vbn(nu)) for nu in (34.0, 1.5, 30.0))
        assert base == pytest.approx(29.389, abs=0.003)
        assert diluent == pytest.approx(8.318, abs=0.001)
        assert target == pytest.approx(28.880, abs=0.003)
        x = (base - target) / (base - diluent)
        assert x == pytest.approx(0.0242, abs=1e-4)
        # and blending that fraction back by mass lands on the spec
        nu = float(refutas_viscosity((1 - x) * base + x * diluent))
        assert nu == pytest.approx(30.0, rel=1e-12)
        assert (34.0 - 30.0) / (34.0 - 1.5) == pytest.approx(0.123, abs=5e-4)

    def test_mass_basis(self):
        # the heavier component carries more weight than its volume says
        light = refutas_blend(jnp.array([0.5, 0.5]), jnp.array([0.8, 0.8]),
                              jnp.array([2.0, 400.0]))
        heavy = refutas_blend(jnp.array([0.5, 0.5]), jnp.array([0.8, 1.0]),
                              jnp.array([2.0, 400.0]))
        assert float(heavy) > float(light)


class TestEthylRT70:
    def test_hand_calculation(self):
        v = [0.6, 0.4]
        r, m, o, a = [98.0, 92.0], [88.0, 80.0], [1.0, 30.0], [65.0, 25.0]
        s = [r[i] - m[i] for i in range(2)]

        def avg(x):
            return sum(v[i] * x[i] for i in range(2))

        rs = avg([r[i] * s[i] for i in range(2)]) - avg(r) * avg(s)
        ms = avg([m[i] * s[i] for i in range(2)]) - avg(m) * avg(s)
        o2 = avg([x * x for x in o]) - avg(o) ** 2
        a2 = avg([x * x for x in a]) - avg(a) ** 2
        ron = avg(r) + 0.03224 * rs + 0.00101 * o2
        mon = avg(m) + 0.04450 * ms + 0.00081 * o2 - 0.00645 * (a2 / 100) ** 2
        got = ethyl_rt70(jnp.array(v), jnp.array(r), jnp.array(m),
                         jnp.array(o), jnp.array(a))
        assert float(got[0]) == pytest.approx(ron, rel=1e-13)
        assert float(got[1]) == pytest.approx(mon, rel=1e-13)

    def test_reduces_to_linear_without_spread(self):
        # same sensitivity, olefins and aromatics -> every correction is zero
        v = jnp.array([0.2, 0.5, 0.3])
        ron = jnp.array([90.0, 95.0, 99.0])
        mon = ron - 10.0
        o = jnp.full(3, 12.0)
        a = jnp.full(3, 30.0)
        got = ethyl_rt70(v, ron, mon, o, a)
        assert float(got[0]) == pytest.approx(float(jnp.sum(v * ron)))
        assert float(got[1]) == pytest.approx(float(jnp.sum(v * mon)))

    def test_coefficients_are_the_published_fit(self):
        # Maples, Petroleum Refinery Process Economics, 2nd ed. (PennWell,
        # 2000), 75-blend fit, as tabulated in J. Jechura, CBEN 409 "Product
        # Blending & Optimization Considerations", Colorado School of Mines
        # (2019), slide 6.  b3 was -0.0645 before #301 -- ten times too big.
        c = EthylRT70()
        assert (c.a1, c.a2, c.a3) == (0.03224, 0.00101, 0.0)
        assert (c.b1, c.b2, c.b3) == (0.04450, 0.00081, -0.00645)

    def test_aromatic_term_scaling_matches_the_written_out_form(self):
        # Jechura slide 19 writes the 135-blend MON aromatic term out as
        # -6.32e-7 (A^2 - A A)^2 against b3 = -0.00632 in the table: the
        # spread is divided by 100 BEFORE squaring.  Pin that scaling.
        v = jnp.array([0.5, 0.5])
        r = jnp.array([95.0, 95.0])
        m = jnp.array([85.0, 85.0])
        o = jnp.zeros(2)
        a = jnp.array([10.0, 60.0])
        c135 = EthylRT70(a1=0.03324, a2=0.00085, b1=0.04285, b2=0.00066,
                         b3=-0.00632)
        _, mon = ethyl_rt70(v, r, m, o, a, c135)
        spread = 0.5 * (10.0 ** 2 + 60.0 ** 2) - 35.0 ** 2
        assert float(mon) == pytest.approx(85.0 - 6.32e-7 * spread ** 2, rel=1e-13)

    def test_mon_interaction_is_on_sensitivity_not_olefins(self):
        # Same olefins everywhere, different sensitivity: the b1 term must
        # still act (Maples 2000; Singh et al., J. Process Control 10 (2000)
        # 43-58, both write it as M*J with J = RON - MON).
        v = jnp.array([0.5, 0.5])
        r = jnp.array([100.0, 90.0])
        m = jnp.array([88.0, 88.0])
        o = jnp.full(2, 5.0)
        a = jnp.full(2, 20.0)
        _, mon = ethyl_rt70(v, r, m, o, a)
        assert float(mon) == pytest.approx(88.0, abs=1e-12)  # M constant: zero covariance
        m2 = jnp.array([90.0, 84.0])
        _, mon2 = ethyl_rt70(v, r, m2, o, a)
        s = r - m2
        cov = float(jnp.mean(m2 * s) - jnp.mean(m2) * jnp.mean(s))
        assert float(mon2) == pytest.approx(87.0 + 0.04450 * cov, rel=1e-13)
        assert cov != 0.0


class TestRVP:
    def test_index_hand_calculation(self):
        v, rvp = [0.95, 0.05], [7.0, 51.6]
        expected = (sum(v[i] * rvp[i] ** 1.25 for i in range(2))) ** 0.8
        got = rvp_index_blend(jnp.array(v), jnp.array(rvp))
        assert float(got) == pytest.approx(expected, rel=1e-13)

    def test_published_worked_example(self):
        # J. Jechura, CBEN 409 "Product Blending & Optimization
        # Considerations", Colorado School of Mines (2019), slide 22,
        # "Gasoline Blending Example - All Into Regular": 30,000 gal
        # butane (54 psi), 35,000 straight-run naphtha (11.2), 60,000 high
        # octane reformate (3.2), 70,000 FCC naphtha (1.4) and 40,000
        # alkylate (4.6) give RVP^1.25 = 24.43 and RVP = 12.9 psi.
        v = jnp.array([30_000.0, 35_000.0, 60_000.0, 70_000.0, 40_000.0])
        rvp = jnp.array([54.0, 11.2, 3.2, 1.4, 4.6])
        got = float(rvp_index_blend(v, rvp))
        assert got ** 1.25 == pytest.approx(24.43, abs=0.005)
        assert got == pytest.approx(12.9, abs=0.05)
        # the slide's per-component index column
        np.testing.assert_allclose(np.asarray(rvp) ** 1.25,
                                   [146.4, 20.5, 4.3, 1.5, 6.7], atol=0.05)

    def test_index_is_linear_at_exponent_one(self):
        v, rvp = jnp.array([0.3, 0.7]), jnp.array([4.0, 10.0])
        assert float(rvp_index_blend(v, rvp, exponent=1.0)) == \
            pytest.approx(8.2)

    def test_raoult_pure_component_is_its_vapor_pressure(self):
        got = raoult_rvp(jnp.array([3.0]), jnp.array([2.5e5]),
                         jnp.array([1e-4]))
        assert float(got) == pytest.approx(2.5e5, rel=1e-12)

    def test_raoult_air_space_strips_light_ends(self):
        z = jnp.array([0.1, 0.9])
        psat = jnp.array([3.5e5, 2e4])
        vm = jnp.array([1.0e-4, 1.3e-4])
        bomb = raoult_rvp(z, psat, vm)
        no_air = raoult_rvp(z, psat, vm, vapor_liquid_ratio=0.0)
        assert float(no_air) == pytest.approx(float(jnp.sum(z * psat)))
        assert float(bomb) < float(no_air)

    def test_butane_vapor_pressure(self, char):
        # Lee-Kesler with n-butane's own constants: 51.6 psi at 100 degF
        rvp = char.psat(310.928)[0] / PSI
        assert float(rvp) == pytest.approx(51.6, rel=0.02)


class TestFlashPoint:
    def test_identical_components(self):
        assert float(flash_point_blend(jnp.array([0.3, 0.7]),
                                       jnp.array([330.0, 330.0]))) == \
            pytest.approx(330.0)

    def test_matches_wickey_chittenden_in_fahrenheit(self):
        v, T = [0.4, 0.6], [311.0, 345.0]
        F = [(t - 273.15) * 1.8 + 32.0 for t in T]
        bi = sum(v[i] * 10 ** (-6.1188 + 4345.2 / (F[i] + 383.0))
                 for i in range(2))
        F_blend = 4345.2 / (math.log10(bi) + 6.1188) - 383.0
        got = flash_point_blend(jnp.array(v), jnp.array(T))
        assert (float(got) - 273.15) * 1.8 + 32.0 == \
            pytest.approx(F_blend, abs=0.05)

    def test_low_flash_component_dominates(self):
        got = flash_point_blend(jnp.array([0.5, 0.5]),
                                jnp.array([311.0, 345.0]))
        assert float(got) < 328.0  # below the arithmetic mean


class TestTemperatureIndex:
    def test_hand_calculation(self):
        v, T, n = [0.25, 0.75], [253.15, 268.15], 20.0
        expected = (v[0] * T[0] ** n + v[1] * T[1] ** n) ** (1 / n)
        got = temperature_index_blend(jnp.array(v), jnp.array(T), n)
        assert float(got) == pytest.approx(expected, rel=1e-12)

    def test_unit_scale_cancels(self):
        v, T = jnp.array([0.4, 0.6]), jnp.array([250.0, 270.0])
        k = temperature_index_blend(v, T, 12.5)
        rankine = temperature_index_blend(v, 1.8 * T, 12.5)
        assert float(rankine) == pytest.approx(1.8 * float(k), rel=1e-12)

    def test_warm_component_dominates(self):
        got = temperature_index_blend(jnp.array([0.5, 0.5]),
                                      jnp.array([240.0, 270.0]), 20.0)
        assert 255.0 < float(got) < 270.0


class TestCetane:
    def test_d4737_reference_fuel(self):
        assert float(cetane_index_d4737(0.85, 215.0, 260.0, 310.0)) == \
            pytest.approx(45.2, abs=1e-12)

    def test_d976_hand_calculation(self):
        D, B = 0.84, 270.0
        expected = (454.74 - 1641.416 * D + 774.74 * D ** 2 - 0.554 * B
                    + 97.803 * math.log10(B) ** 2)
        assert float(cetane_index_d976(D, B)) == \
            pytest.approx(expected, rel=1e-13)

    def test_lighter_fuel_higher_index(self):
        assert float(cetane_index_d4737(0.83, 215.0, 260.0, 310.0)) > \
            float(cetane_index_d4737(0.87, 215.0, 260.0, 310.0))


class TestDistillation:
    def test_d86_round_trip(self):
        from difflow_refinery import TBP_D86
        for pct, (a, b) in TBP_D86.items():
            d86 = 500.0
            tbp = a * d86 ** b
            assert float(tbp_to_d86(tbp, pct)) == pytest.approx(d86)

    def test_d86_curve_flatter_than_tbp(self):
        # D86 is a poorer separation: higher at 10%, lower at 90%
        assert float(tbp_to_d86(400.0, 10)) > 400.0
        assert float(tbp_to_d86(600.0, 90)) < 600.0

    def test_inversion(self):
        Tb = jnp.array([330.0, 360.0, 400.0, 450.0])
        phi = jnp.array([0.1, 0.3, 0.4, 0.2])
        for f in [0.1, 0.5, 0.9]:
            T = tbp_temperature(f, Tb, phi, 5.0)
            assert float(tbp_evaporated(T, Tb, phi, 5.0)) == \
                pytest.approx(f, abs=1e-10)

    @pytest.mark.release
    def test_implicit_gradient_matches_fd(self):
        Tb = jnp.array([330.0, 360.0, 400.0, 450.0])
        phi = jnp.array([0.1, 0.3, 0.4, 0.2])
        check_grads(lambda tb, p: tbp_temperature(0.9, tb, p, 5.0),
                    (Tb, phi), order=1, modes=["rev", "fwd"],
                    atol=1e-5, rtol=1e-5)


class TestCharacterization:
    def test_heptane_like_cut(self):
        c = BlendCharacterization(["h"], [371.6], [0.688])
        assert float(c.tc[0]) == pytest.approx(540.2, rel=0.01)
        assert float(c.pc[0]) == pytest.approx(27.4e5, rel=0.05)
        assert float(c.mw[0]) == pytest.approx(100.2, rel=0.08)
        # Lee-Kesler with Edmister's omega gives ~1 atm at the boiling point
        assert float(c.psat(371.6)[0]) == pytest.approx(101325.0, rel=0.02)

    def test_overrides_only_finite_entries(self, char):
        assert float(char.mw[0]) == 58.12
        assert float(char.mw[1]) != 58.12

    def test_rejects_foreign_species(self, char):
        with pytest.raises(ValueError, match="outside the characterization"):
            char.flows(make_stream({"benzene": 1.0}, T=300.0, P=1e5))


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

class TestPoolPropertyMode:
    def test_properties_and_margins(self):
        pool = BlendPool("gasoline")
        res = pool(_gasoline_components(), RECIPE)
        assert res.stream is None
        assert set(res.margins) == {"RON >= 91", "MON >= 82", "RVP_psi <= 9",
                                    "S_ppm <= 10"}
        # margins are the signed distance to each limit
        assert float(res.margins["RON >= 91"]) == \
            pytest.approx(float(res.properties["RON"]) - 91.0)
        assert float(res.margins["RVP_psi <= 9"]) == \
            pytest.approx(9.0 - float(res.properties["RVP_psi"]))
        assert float(res.properties["AKI"]) == pytest.approx(
            0.5 * float(res.properties["RON"] + res.properties["MON"]))

    def test_sulfur_by_mass(self):
        comps = _gasoline_components()
        res = BlendPool("gasoline")(comps, RECIPE)
        sg = np.array([0.80, 0.74, 0.70, 0.584])
        s = np.array([1.0, 20.0, 5.0, 1.0])
        w = np.asarray(RECIPE) * sg
        assert float(res.properties["S_ppm"]) == \
            pytest.approx(float(np.sum(w * s) / np.sum(w)))

    def test_recipe_bases_agree(self):
        comps = _gasoline_components()
        pool = BlendPool("gasoline")
        a = pool(comps, RECIPE)
        b = pool(comps, 7.0 * RECIPE, basis="volume_flow")
        for k in a.properties:
            assert float(a.properties[k]) == \
                pytest.approx(float(b.properties[k]), rel=1e-12)
        assert float(b.volume) == pytest.approx(7.0)
        assert float(b.mass) == pytest.approx(
            7.0 * float(a.properties["SG"]) * RHO_WATER_15C)

    def test_linear_rule_option(self):
        comps = _gasoline_components()
        res = BlendPool("gasoline", rules={"octane": "volume",
                                           "RVP_psi": "volume"})(comps,
                                                                 RECIPE)
        ron = np.array([98.0, 92.0, 96.0, 93.0])
        assert float(res.properties["RON"]) == \
            pytest.approx(float(np.sum(np.asarray(RECIPE) * ron)))

    def test_rules_are_per_property(self):
        comps = _gasoline_components()
        ethyl = BlendPool("gasoline")(comps, RECIPE).properties
        mixed = BlendPool("gasoline", rules={"RON": "volume"})(
            comps, RECIPE).properties
        ron = np.array([98.0, 92.0, 96.0, 93.0])
        assert float(mixed["RON"]) == \
            pytest.approx(float(np.sum(np.asarray(RECIPE) * ron)))
        assert float(mixed["MON"]) == pytest.approx(float(ethyl["MON"]))

    def test_linear_blend_error(self):
        comps = _gasoline_components()
        pool = BlendPool("gasoline")
        err = pool.linear_blend_error(comps, RECIPE)
        # linear-by-volume properties carry no error ...
        assert float(err["SG"]) == pytest.approx(0.0, abs=1e-14)
        assert float(err["aromatics_vol"]) == pytest.approx(0.0, abs=1e-12)
        # ... the nonlinear rules do, with the signs the rules imply
        assert float(err["RVP_psi"]) > 0.0      # power mean >= mean
        assert float(err["MON"]) < 0.0          # olefin x MON interaction
        lin = pool.linear_properties(comps, RECIPE)
        nl = pool(comps, RECIPE).properties
        assert float(err["RON"]) == \
            pytest.approx(float(nl["RON"] - lin["RON"]))

    def test_exact_properties_carry_no_error(self):
        comps = _gasoline_components()
        pool = BlendPool("gasoline")
        err = pool.linear_blend_error(comps, RECIPE,
                                      exact=("RVP_psi", "S_ppm"))
        assert float(err["RVP_psi"]) == pytest.approx(0.0, abs=1e-12)
        assert float(err["S_ppm"]) == pytest.approx(0.0, abs=1e-12)
        assert abs(float(err["MON"])) > 1.0
        with pytest.raises(ValueError, match="cannot hold"):
            pool.linear_properties(comps, RECIPE, exact=("RON",))

    def test_backoff(self):
        comps = _gasoline_components()
        pool = BlendPool("gasoline")
        b = pool.backoff(comps, RECIPE, exact=("RVP_psi", "S_ppm"))
        err = pool.linear_blend_error(comps, RECIPE,
                                      exact=("RVP_psi", "S_ppm"))
        # MON blends below its linear value: the LP overstates its margin
        assert float(b["MON >= 82"]) == pytest.approx(-float(err["MON"]))
        # RON blends above it: nothing to protect against
        assert float(err["RON"]) > 0 and float(b["RON >= 91"]) == 0.0
        assert float(b["RVP_psi <= 9"]) == 0.0
        # a back-off of exactly that size makes the linear margin the true one
        lin = pool.linear_properties(comps, RECIPE)["MON"]
        nl = pool(comps, RECIPE).properties["MON"]
        assert float(lin - 82.0 - b["MON >= 82"]) == \
            pytest.approx(float(nl - 82.0))

    def test_weighted_margins(self):
        comps = _gasoline_components()
        pool = BlendPool("gasoline")
        V = 3.0 * RECIPE
        m = pool.spec_margins(comps, V, basis="volume_flow")
        w = pool.spec_margins(comps, V, basis="volume_flow", weighted=True)
        np.testing.assert_allclose(np.asarray(w), 3.0 * np.asarray(m))
        # and an empty pool is finite rather than 0/0
        z = pool.spec_margins(comps, jnp.zeros(4), basis="volume_flow",
                              weighted=True)
        assert np.all(np.asarray(z) == 0.0)

    def test_single_component_has_no_blending_error(self):
        comps = _gasoline_components()[:1]
        err = BlendPool("gasoline").linear_blend_error(comps, jnp.ones(1))
        for k, v in err.items():
            assert abs(float(v)) < 1e-10, k

    def test_other_products(self):
        jet = [BlendComponent.from_properties(
                   "kero", SG=0.80, S_ppm=1500.0, freeze_C=-47.0,
                   smoke_mm=22.0, flash_C=45.0),
               BlendComponent.from_properties(
                   "hck", SG=0.81, S_ppm=10.0, freeze_C=-38.0,
                   smoke_mm=26.0, flash_C=40.0)]
        res = BlendPool("jet")(jet, [0.6, 0.4])
        assert -47.0 < float(res.properties["freeze_C"]) < -38.0
        assert 40.0 < float(res.properties["flash_C"]) < 45.0
        fo = [BlendComponent.from_properties(
                  "resid", SG=1.00, S_ppm=9000.0, viscosity_cSt=3000.0,
                  CCR_wt=20.0),
              BlendComponent.from_properties(
                  "cutter", SG=0.86, S_ppm=100.0, viscosity_cSt=3.0,
                  CCR_wt=0.1)]
        res = BlendPool("fuel_oil")(fo, [0.6, 0.4])
        assert set(res.margins) == {"S_ppm <= 5000", "viscosity_cSt <= 380",
                                    "SG <= 0.991", "CCR_wt <= 18"}

    def test_errors(self):
        comps = _gasoline_components()
        with pytest.raises(ValueError, match="no default specs"):
            BlendPool("asphalt")
        with pytest.raises(ValueError, match="not available"):
            BlendPool(rules={"RON": "refutas"})
        with pytest.raises(ValueError, match="unknown properties"):
            BlendComponent.from_properties("x", SG=0.7, octane=90)
        with pytest.raises(ValueError, match="needs an SG"):
            BlendComponent.from_properties("x", RON=90.0)
        no_olefins = [BlendComponent.from_properties(
            "x", SG=0.7, RON=90.0, MON=80.0, RVP_psi=7.0, S_ppm=5.0)] * 2
        with pytest.raises(KeyError, match="Ethyl RT-70"):
            BlendPool("gasoline")(no_olefins, [0.5, 0.5])
        with pytest.raises(KeyError, match="has no 'cetane_index'"):
            BlendPool("ulsd")(comps, RECIPE)
        with pytest.raises(ValueError, match="recipe needs 4"):
            BlendPool("gasoline")(comps, [0.5, 0.5])


class TestPoolStreamMode:
    def test_mass_and_volume_balance(self, char, stream_components):
        pool = BlendPool("gasoline")
        V = jnp.array([0.6, 0.3, 0.05]) * 1e-3
        res = pool(stream_components, V, basis="volume_flow")
        F = char.flows(res.stream)
        mass = float(jnp.sum(F * char.mw)) * 1e-3
        vol = float(jnp.sum(F * char.molar_volume))
        assert mass == pytest.approx(float(res.mass), rel=1e-12)
        assert vol == pytest.approx(float(res.volume), rel=1e-12)
        assert float(res.volume) == pytest.approx(float(jnp.sum(V)),
                                                  rel=1e-14)
        # product SG from the composition = volume average of component SG
        sg = float(jnp.sum(char.volume_fractions(F) * char.SG))
        assert sg == pytest.approx(float(res.properties["SG"]), rel=1e-12)

    def test_split_basis_closes_per_pseudocomponent(self, char,
                                                    stream_components):
        n = char.n
        light = np.zeros(n); light[1:6] = [3, 4, 4, 3, 2]
        heavy = np.zeros(n); heavy[4:10] = [1, 2, 3, 3, 2, 1]
        butane = np.zeros(n); butane[0] = 1.0
        inlets = [10.0 * light, 10.0 * heavy, 2.0 * butane]
        split = jnp.array([0.5, 1.0, 0.25])
        res = BlendPool("gasoline")(stream_components, split, basis="split")
        expected = sum(s * f for s, f in zip(np.asarray(split), inlets))
        np.testing.assert_allclose(np.asarray(char.flows(res.stream)),
                                   expected, rtol=1e-12, atol=1e-14)

    def test_derived_properties(self, stream_components):
        res = BlendPool("gasoline")(stream_components, [0.6, 0.3, 0.1])
        p = res.properties
        assert 0.0 < float(p["E70_tbp"]) < float(p["E100_tbp"]) < 100.0
        assert float(p["T10_d86_C"]) < float(p["T50_d86_C"]) < \
            float(p["T90_d86_C"])
        # the RVP index (on the components' own Raoult RVPs) and Raoult on
        # the blend are different rules and should land near each other
        assert float(p["RVP_psi"]) == \
            pytest.approx(float(p["RVP_raoult_psi"]), rel=0.3)

    def test_raoult_rule(self, stream_components):
        pool = BlendPool("gasoline", rules={"RVP_psi": "raoult"})
        res = pool(stream_components, [0.6, 0.3, 0.1])
        assert float(res.properties["RVP_psi"]) == \
            float(res.properties["RVP_raoult_psi"])

    def test_ulsd_cetane_index(self, char):
        n = char.n
        diesel = np.zeros(n); diesel[8:14] = [1, 2, 3, 3, 2, 1]
        kero = np.zeros(n); kero[5:10] = [1, 2, 3, 2, 1]
        comps = [BlendComponent.from_stream("diesel", _stream(char, diesel),
                                            char, flash_C=70.0),
                 BlendComponent.from_stream("kero", _stream(char, kero),
                                            char, flash_C=45.0)]
        pool = BlendPool("ulsd", specs=[("cetane_index", ">=", 40.0),
                                        ("T90_d86_C", "<=", 360.0)])
        res = pool(comps, [0.8, 0.2])
        assert 20.0 < float(res.properties["cetane_index"]) < 80.0
        d976 = BlendPool("ulsd", specs=[], rules={"cetane": "d976"})
        assert float(d976(comps, [0.8, 0.2]).properties["cetane_index"]) \
            != float(res.properties["cetane_index"])

    def test_mixed_modes_rejected(self, stream_components):
        prop = BlendComponent.from_properties(
            "x", SG=0.7, RON=90.0, MON=80.0, olefins_vol=0.0,
            aromatics_vol=0.0, RVP_psi=5.0)
        with pytest.raises(ValueError, match="mix of stream-mode"):
            BlendPool("gasoline", specs=[])(
                list(stream_components) + [prop], [0.25] * 4)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def _fd_check(f, x, rel=1e-5, h=1e-6):
    """Central finite differences against jax.jacrev, column by column."""
    x = jnp.asarray(x, dtype=jnp.float64)
    f = jax.jit(f)
    J = np.atleast_2d(np.asarray(jax.jacrev(f)(x)).reshape(-1, x.size))
    for i in range(x.size):
        e = jnp.zeros_like(x).at[i].set(h * max(1.0, abs(float(x[i]))))
        fd = (np.asarray(f(x + e)) - np.asarray(f(x - e))).reshape(-1) \
            / (2 * float(e[i]))
        np.testing.assert_allclose(J[:, i], fd, rtol=rel,
                                   atol=rel * max(1.0, np.abs(fd).max()))


@pytest.mark.release
class TestGradients:
    """Every property's derivative against central finite differences."""

    PROPS = ["SG", "S_ppm", "aromatics_vol", "olefins_vol", "RON", "MON",
             "RVP_psi"]

    def test_property_mode_wrt_recipe(self):
        comps = _gasoline_components()
        pool = BlendPool("gasoline")

        def f(r):
            p = pool(comps, r).properties
            return jnp.stack([p[k] for k in self.PROPS])

        _fd_check(f, RECIPE)

    def test_property_mode_wrt_component_properties(self):
        pool = BlendPool("gasoline")
        names = ["SG", "RON", "MON", "RVP_psi", "S_ppm", "olefins_vol",
                 "aromatics_vol"]
        base = {n: jnp.stack([c.properties[n]
                              for c in _gasoline_components()])
                for n in names}
        sizes = [4] * len(names)

        def f(flat):
            cols = jnp.split(flat, np.cumsum(sizes)[:-1])
            comps = [BlendComponent.from_properties(
                f"c{i}", **{n: cols[k][i] for k, n in enumerate(names)})
                for i in range(4)]
            p = pool(comps, RECIPE).properties
            return jnp.stack([p[k] for k in self.PROPS])

        _fd_check(f, jnp.concatenate([base[n] for n in names]))

    def test_index_rules_wrt_recipe_and_properties(self):
        def jet(x):
            r, freeze, flash = x[:2], x[2:4], x[4:6]
            comps = [BlendComponent.from_properties(
                f"c{i}", SG=0.8, S_ppm=100.0, smoke_mm=20.0,
                freeze_C=freeze[i], flash_C=flash[i]) for i in range(2)]
            p = BlendPool("jet")(comps, r).properties
            return jnp.stack([p["freeze_C"], p["flash_C"]])

        _fd_check(jet, jnp.array([0.6, 0.4, -47.0, -38.0, 45.0, 40.0]))

        def fo(x):
            r, nu, sg = x[:2], x[2:4], x[4:6]
            comps = [BlendComponent.from_properties(
                f"c{i}", SG=sg[i], S_ppm=1000.0, CCR_wt=5.0,
                viscosity_cSt=nu[i]) for i in range(2)]
            return BlendPool("fuel_oil")(comps, r).properties["viscosity_cSt"]

        _fd_check(fo, jnp.array([0.6, 0.4, 3000.0, 3.0, 1.0, 0.86]))

    def test_stream_mode_wrt_recipe_and_boiling_points(self, char,
                                                       stream_components):
        pool = BlendPool("gasoline")
        keys = ["E70_tbp", "T50_d86_C", "T90_d86_C", "RVP_raoult_psi",
                "RVP_psi", "SG"]

        def f(r):
            p = pool(stream_components, r).properties
            return jnp.stack([p[k] for k in keys])

        _fd_check(f, jnp.array([0.6, 0.3, 0.1]))

        def g(tb):
            c = BlendCharacterization(names=char.names, Tb=tb, SG=char.SG,
                                 MW=char.MW, Tc=char.Tc, Pc=char.Pc,
                                 omega=char.omega, qualities=char.qualities)
            comps = [BlendComponent.from_stream(
                s.name, {f"F_{n}": s.moles_per_volume[j]
                         for j, n in enumerate(char.names)}, c,
                RON=s.properties["RON"], MON=s.properties["MON"])
                for s in stream_components]
            p = pool(comps, [0.6, 0.3, 0.1]).properties
            return jnp.stack([p["T90_d86_C"], p["RVP_raoult_psi"]])

        _fd_check(g, char.Tb, rel=1e-4)


class TestSmoothMargins:
    def test_smooth_violation_at_the_boundary(self):
        d = jax.grad(lambda m: smooth_violation(m, 0.1))(0.0)
        assert float(d) == pytest.approx(-0.5)
        assert float(smooth_violation(-1.0)) == 1.0
        assert float(smooth_violation(1.0)) == 0.0
        assert float(smooth_violation(5.0, 0.1)) < 1e-15

    def test_finite_gradient_at_an_active_spec(self):
        comps = _gasoline_components()
        ron = float(BlendPool("gasoline")(comps, RECIPE).properties["RON"])
        pool = BlendPool("gasoline", specs=[BlendSpec("RON", ">=", ron)])
        m = pool.spec_margins(comps, RECIPE)
        assert abs(float(m[0])) < 1e-12   # the spec is exactly active
        g = jax.grad(lambda r: jnp.sum(pool.spec_violations(
            comps, r, temperature=0.05)))(RECIPE)
        assert np.all(np.isfinite(np.asarray(g)))
        assert float(jnp.linalg.norm(g)) > 0.0
        # and it is half the margin's own gradient, with the sign flipped
        gm = jax.grad(lambda r: pool.spec_margins(comps, r)[0])(RECIPE)
        np.testing.assert_allclose(np.asarray(g), -0.5 * np.asarray(gm),
                                   rtol=1e-10, atol=1e-14)


class TestPlanningBlock:
    def test_as_block_delta_vectors(self, stream_components):
        pool = BlendPool("gasoline")
        blk = pool.as_block(stream_components)
        assert blk.u_names == ["lsr_V", "reformate_V", "butane_V"]
        assert blk.y_names[0] == "volume"
        assert "margin:RON >= 91" in blk.y_names
        u0 = 0.5 * jnp.asarray(blk.ub)
        y = blk.fn(u0)
        assert y.shape == (len(blk.y_names),)
        # d(volume)/dV_i = 1: the volume balance, as a delta vector
        J = jax.jacobian(blk.fn)(u0)
        np.testing.assert_allclose(np.asarray(J[0]), 1.0, rtol=1e-12)
        _fd_check(blk.fn, u0 * 1e3 / 1e3, rel=1e-5, h=1e-7)
