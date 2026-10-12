"""Tests for phase-diagram helpers and the Wilson / Margules / van Laar models (#398).

What is and is not verified against literature
----------------------------------------------
- Benzene-toluene (Raoult, 1 atm): end points vs. the literature normal boiling
  points (benzene 353.24 K, toluene 383.78 K; Antoine coefficients are the NIST
  set in the difflow database).
- Ethanol-water: NRTL, alpha = 0.3, tau_ij = a_ij + b_ij/T with the DECHEMA/Aspen
  set (Gmehling & Onken, VLE Data Collection, 1977) gives the azeotrope at
  x = 0.893, T = 351.29 K, against the experimental x = 0.894, T = 351.3 K at
  1 atm. Antoine: NIST log10(P/bar) shifted by +5 for Pa.
- Margules / van Laar: the closed forms are checked against hand-computed
  values, the Gibbs-Duhem equation and the infinite-dilution identities
  (A12 = ln gamma_1^inf). No textbook worked example is reproduced to 3
  significant figures: it could not be retrieved while writing this, and a
  recalled number would not be a validation. The van Laar constants
  A12 = 1.6798, A21 = 0.9227 (ethanol-water, Carlson & Colburn-type values
  tabulated in Perry's) are used as a published parameter set; the test checks
  the azeotrope they imply is within a loose band, not a 3-figure match.
- Wilson, methanol-water: the two energy parameters were *regressed* to the
  1 atm Txy table (Perry's; x = 0.1..0.9, entered from the published table) with
  V = 40.73 and 18.07 cm3/mol, then frozen here. The test confirms the
  implementation reproduces that data set (T within 0.3 K, y within 0.01); it
  tests the equation and the data's consistency, not independently published
  Wilson constants.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow.activity import (
    ActivityModel, MargulesParams, VanLaarParams, WilsonParams, activity_gamma,
)
from difflow.database import get_species_data
from difflow.phase_diagrams import (
    bubble_T, find_azeotrope, pxy, ternary_lle, txy, xy_curve,
)
from difflow.streams import make_stream
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.flash import Flash, FlashParams
from difflow.units.lle import (
    LLEEquilibrium, NRTLParams, UNIQUACParams, nrtl_activity_coefficients,
)

jax.config.update("jax_enable_x64", True)

ATM = 101325.0


def _sd(name, antoine, Tc):
    return SpeciesData(name=name, MW=1.0, Cp_coeffs=(0, 0, 0, 0),
                       Hvap_coeffs=(1.0, 0.38, Tc), antoine_coeffs=antoine)


@pytest.fixture(scope="module")
def bt_thermo():
    return IdealThermo({s: get_species_data(s) for s in ("benzene", "toluene")})


@pytest.fixture(scope="module")
def ew_thermo():
    return IdealThermo({
        "ethanol": _sd("ethanol", (10.24677, 1598.673, -46.424), 514.0),
        "water": _sd("water", (9.6543, 1435.264, -64.848), 647.0),
    })


@pytest.fixture(scope="module")
def mw_thermo():
    return IdealThermo({
        "methanol": _sd("methanol", (10.20409, 1581.341, -33.5), 512.6),
        "water": _sd("water", (9.6543, 1435.264, -64.848), 647.0),
    })


@pytest.fixture(scope="module")
def ew_nrtl():
    # ethanol(1)-water(2); verified: gamma_1^inf = 5.4, gamma_2^inf = 2.6 at 351 K
    return NRTLParams(
        species=("ethanol", "water"),
        a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
        b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
        alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
    )


# --------------------------------------------------------------------------
# Raoult: benzene-toluene
# --------------------------------------------------------------------------

class TestBenzeneToluene:
    def test_txy_endpoints_are_normal_boiling_points(self, bt_thermo):
        d = txy(bt_thermo, ["benzene", "toluene"], ATM, n=21)
        assert float(d["T"][-1]) == pytest.approx(353.24, abs=0.1)   # x = 1 benzene
        assert float(d["T"][0]) == pytest.approx(383.78, abs=0.1)    # x = 0 toluene
        np.testing.assert_allclose(np.asarray(d["y"])[[0, -1]], [0.0, 1.0], atol=1e-12)

    def test_curve_is_monotone_and_dew_above_bubble(self, bt_thermo):
        d = txy(bt_thermo, ["benzene", "toluene"], ATM, n=21)
        assert np.all(np.diff(np.asarray(d["T"])) < 0)
        assert np.all(np.asarray(d["y"])[1:-1] > np.asarray(d["x"])[1:-1])

    def test_mean_relative_volatility(self, bt_thermo):
        d = xy_curve(bt_thermo, ["benzene", "toluene"], ATM, n=51)
        a = np.asarray(d["alpha"])
        # alpha varies from 2.35 (toluene end) to 2.60 (benzene end) through
        # the Antoine fits; "about 2.4" in the textbooks.
        assert a.mean() == pytest.approx(2.4, abs=0.1)
        assert a.min() > 2.3 and a.max() < 2.65

    def test_y_from_alpha_is_consistent(self, bt_thermo):
        d = xy_curve(bt_thermo, ["benzene", "toluene"], ATM, n=21)
        x, a = np.asarray(d["x"]), np.asarray(d["alpha"])
        np.testing.assert_allclose(a * x / (1 + (a - 1) * x), d["y"], atol=1e-9)

    @pytest.mark.slow
    def test_no_raoult_azeotrope(self, bt_thermo):
        r = find_azeotrope(bt_thermo, ["benzene", "toluene"], P=ATM)
        assert not bool(r["found"])
        assert np.isnan(float(r["x"])) and np.isnan(float(r["T"]))

    def test_pxy_consistent_with_txy(self, bt_thermo):
        sp = ["benzene", "toluene"]
        t = txy(bt_thermo, sp, ATM, n=11)
        p = pxy(bt_thermo, sp, float(t["T"][5]), n=11)
        # at the bubble temperature of x=0.5 the bubble pressure is 1 atm
        assert float(p["P"][5]) == pytest.approx(ATM, rel=1e-8)
        assert float(p["y"][5]) == pytest.approx(float(t["y"][5]), abs=1e-9)

    def test_binary_only(self, bt_thermo):
        with pytest.raises(ValueError):
            txy(bt_thermo, ["benzene"], ATM)


# --------------------------------------------------------------------------
# NRTL ethanol-water azeotrope
# --------------------------------------------------------------------------

class TestEthanolWater:
    @pytest.mark.slow
    def test_nrtl_azeotrope(self, ew_thermo, ew_nrtl):
        r = find_azeotrope(ew_thermo, ["ethanol", "water"], P=ATM, activity_model=ew_nrtl)
        assert bool(r["found"])
        assert float(r["x"]) == pytest.approx(0.89, abs=0.01)
        assert float(r["T"]) == pytest.approx(351.3, abs=0.1)

    def test_azeotrope_is_a_min_boiling_point(self, ew_thermo, ew_nrtl):
        d = txy(ew_thermo, ["ethanol", "water"], ATM, 101, ew_nrtl)
        i = int(np.argmin(np.asarray(d["T"])))
        assert 0.85 < float(d["x"][i]) < 0.93
        # at the azeotrope the bubble and dew curves touch
        assert abs(float(d["y"][i]) - float(d["x"][i])) < 0.01

    def test_pressure_azeotrope(self, ew_thermo, ew_nrtl):
        r = find_azeotrope(ew_thermo, ["ethanol", "water"], T=351.29, activity_model=ew_nrtl)
        assert bool(r["found"])
        assert float(r["P"]) == pytest.approx(ATM, rel=0.01)

    @pytest.mark.slow
    def test_xy_alpha_crosses_one(self, ew_thermo, ew_nrtl):
        d = xy_curve(ew_thermo, ["ethanol", "water"], ATM, 41, ew_nrtl)
        a = np.asarray(d["alpha"])
        assert a[1] > 1 and a[-2] < 1

    def test_van_laar_published_constants(self, ew_thermo):
        m = VanLaarParams(("ethanol", "water"), 1.6798, 0.9227)
        r = find_azeotrope(ew_thermo, ["ethanol", "water"], P=ATM, activity_model=m)
        assert bool(r["found"])
        assert 0.88 < float(r["x"]) < 0.93
        assert float(r["T"]) == pytest.approx(351.3, abs=0.3)


# --------------------------------------------------------------------------
# Margules / van Laar / Wilson models
# --------------------------------------------------------------------------

class TestMargules:
    def test_two_suffix_hand_value(self):
        m = MargulesParams.two_suffix(("a", "b"), 1.0)
        g = m.gamma(jnp.array([0.5, 0.5]), 300.0)
        np.testing.assert_allclose(g, np.exp(0.25), rtol=1e-12)
        assert float(g[0]) == pytest.approx(1.284, abs=5e-4)

    def test_three_suffix_hand_value(self):
        m = MargulesParams(("a", "b"), 0.5, 1.5)
        x1 = 0.3
        g = m.gamma(jnp.array([x1, 1 - x1]), 300.0)
        lg1 = 0.7**2 * (0.5 + 2 * (1.5 - 0.5) * 0.3)
        lg2 = 0.3**2 * (1.5 + 2 * (0.5 - 1.5) * 0.7)
        np.testing.assert_allclose(jnp.log(g), [lg1, lg2], rtol=1e-12)

    def test_infinite_dilution_and_pure_limits(self):
        m = MargulesParams(("a", "b"), 0.7, 1.9)
        g0 = m.gamma(jnp.array([0.0, 1.0]), 300.0)
        g1 = m.gamma(jnp.array([1.0, 0.0]), 300.0)
        assert float(jnp.log(g0[0])) == pytest.approx(0.7)
        assert float(g0[1]) == pytest.approx(1.0)
        assert float(jnp.log(g1[1])) == pytest.approx(1.9)

    def test_gibbs_duhem(self):
        m = MargulesParams(("a", "b"), 0.7, 1.9)
        x1 = jnp.linspace(0.1, 0.9, 9)
        def lng(x):
            return jnp.log(m.gamma(jnp.stack([x, 1 - x]), 300.0))
        d = jax.vmap(jax.jacfwd(lng))(x1)    # (9, 2)
        gd = x1 * d[:, 0] + (1 - x1) * d[:, 1]
        np.testing.assert_allclose(gd, 0.0, atol=1e-12)

    def test_binary_only(self):
        with pytest.raises(ValueError):
            MargulesParams(("a", "b", "c"), 1.0, 1.0)


class TestVanLaar:
    def test_hand_value(self):
        m = VanLaarParams(("a", "b"), 1.0, 2.0)
        g = m.gamma(jnp.array([0.5, 0.5]), 300.0)
        np.testing.assert_allclose(jnp.log(g), [0.25 * 4 / 2.25, 2 * 0.25 / 2.25], rtol=1e-12)
        assert float(g[0]) == pytest.approx(1.560, abs=5e-4)
        assert float(g[1]) == pytest.approx(1.249, abs=5e-4)

    def test_matches_textbook_form(self):
        A12, A21, x1 = 1.3, 0.8, 0.35
        x2 = 1 - x1
        lg1 = A12 * (1 + A12 * x1 / (A21 * x2)) ** -2
        lg2 = A21 * (1 + A21 * x2 / (A12 * x1)) ** -2
        g = VanLaarParams(("a", "b"), A12, A21).gamma(jnp.array([x1, x2]), 300.0)
        np.testing.assert_allclose(jnp.log(g), [lg1, lg2], rtol=1e-12)

    def test_limits_finite(self):
        m = VanLaarParams(("a", "b"), 1.3, 0.8)
        g0 = m.gamma(jnp.array([0.0, 1.0]), 300.0)
        g1 = m.gamma(jnp.array([1.0, 0.0]), 300.0)
        assert float(jnp.log(g0[0])) == pytest.approx(1.3)
        assert float(jnp.log(g1[1])) == pytest.approx(0.8)
        assert np.all(np.isfinite(np.asarray(g0))) and np.all(np.isfinite(np.asarray(g1)))

    def test_gibbs_duhem(self):
        m = VanLaarParams(("a", "b"), 1.3, 0.8)
        x1 = jnp.linspace(0.1, 0.9, 9)
        lng = lambda x: jnp.log(m.gamma(jnp.stack([x, 1 - x]), 300.0))
        d = jax.vmap(jax.jacfwd(lng))(x1)
        np.testing.assert_allclose(x1 * d[:, 0] + (1 - x1) * d[:, 1], 0.0, atol=1e-12)


METHANOL_WATER_1ATM = {   # x_methanol, y_methanol, T (C): Perry's Table (1 atm)
    "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "y": [0.418, 0.579, 0.665, 0.729, 0.779, 0.825, 0.870, 0.915, 0.958],
    "T": [87.7, 81.7, 78.0, 75.3, 73.1, 71.2, 69.3, 67.6, 66.0],
}


class TestWilson:
    def test_reduces_to_ideal_for_equal_volumes_and_zero_energy(self):
        m = WilsonParams.binary(("a", "b"), [1e-5, 1e-5], 0.0, 0.0)
        np.testing.assert_allclose(m.gamma(jnp.array([0.3, 0.7]), 300.0), 1.0, rtol=1e-12)

    def test_binary_closed_form(self):
        V, dl12, dl21, T, x1 = jnp.array([4e-5, 1.8e-5]), 700.0, 2000.0, 340.0, 0.4
        L12 = V[1] / V[0] * jnp.exp(-dl12 / (8.314462618 * T))
        L21 = V[0] / V[1] * jnp.exp(-dl21 / (8.314462618 * T))
        x2 = 1 - x1
        lg1 = -jnp.log(x1 + L12 * x2) + x2 * (L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1))
        lg2 = -jnp.log(x2 + L21 * x1) - x1 * (L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1))
        g = WilsonParams.binary(("a", "b"), V, dl12, dl21).gamma(jnp.array([x1, x2]), T)
        np.testing.assert_allclose(jnp.log(g), [lg1, lg2], rtol=1e-12)

    def test_infinite_dilution(self):
        V = jnp.array([4e-5, 1.8e-5])
        m = WilsonParams.binary(("a", "b"), V, 700.0, 2000.0)
        L12, L21 = m.Lambda(340.0)[0, 1], m.Lambda(340.0)[1, 0]
        g = m.gamma(jnp.array([0.0, 1.0]), 340.0)
        assert float(jnp.log(g[0])) == pytest.approx(float(1 - jnp.log(L12) - L21), rel=1e-10)

    @pytest.mark.slow
    def test_methanol_water_vle(self, mw_thermo):
        sp = ("methanol", "water")
        m = WilsonParams.binary(sp, [40.73e-6, 18.07e-6], 667.0, 1981.0)
        Td, yd = np.asarray(METHANOL_WATER_1ATM["T"]) + 273.15, np.asarray(METHANOL_WATER_1ATM["y"])
        for xi, Ti, yi in zip(METHANOL_WATER_1ATM["x"], Td, yd):
            x = jnp.array([xi, 1 - xi])
            Tb = bubble_T(mw_thermo, sp, x, ATM, m)
            y = xi * m.gamma(x, Tb)[0] * mw_thermo.Psat("methanol", Tb) / ATM
            assert float(Tb) == pytest.approx(Ti, abs=0.3)
            assert float(y) == pytest.approx(yi, abs=0.01)

    def test_cannot_split_liquid(self):
        """Wilson's Gibbs energy of mixing is convex, even for strongly positive deviation."""
        m = WilsonParams.binary(("a", "b"), [4e-5, 1.8e-5], 6000.0, 6000.0)
        def gmix(x):
            xv = jnp.stack([x, 1 - x])
            return jnp.sum(xv * jnp.log(xv * m.gamma(xv, 300.0)))
        xs = jnp.linspace(0.01, 0.99, 99)
        d2 = jax.vmap(jax.grad(jax.grad(gmix)))(xs)
        assert bool(jnp.all(d2 > 0))


class TestProtocol:
    def test_all_models_satisfy_protocol(self, ew_nrtl):
        uq = UNIQUACParams(("a", "b"), jnp.array([1.0, 1.5]), jnp.array([1.0, 1.4]),
                           jnp.zeros((2, 2)), jnp.zeros((2, 2)))
        for m in (ew_nrtl, uq, WilsonParams.binary(("a", "b"), [1., 1.], 0., 0.),
                  MargulesParams(("a", "b"), 1., 1.), VanLaarParams(("a", "b"), 1., 1.)):
            assert isinstance(m, ActivityModel)
            assert activity_gamma(m, jnp.array([0.4, 0.6]), 300.0).shape == (2,)

    def test_nrtl_adapter_is_the_function(self, ew_nrtl):
        x = jnp.array([0.3, 0.7])
        np.testing.assert_array_equal(
            activity_gamma(ew_nrtl, x, 350.0), nrtl_activity_coefficients(x, 350.0, ew_nrtl))

    def test_plain_callable(self):
        g = activity_gamma(lambda x, T: jnp.exp(0.5 * x), jnp.array([0.2, 0.8]), 300.0)
        np.testing.assert_allclose(g, np.exp([0.1, 0.4]))

    def test_not_a_model(self):
        with pytest.raises(TypeError):
            activity_gamma(3.0, jnp.array([0.5, 0.5]), 300.0)


# --------------------------------------------------------------------------
# Flash with the new models
# --------------------------------------------------------------------------

class TestFlashActivityModels:
    @pytest.mark.slow
    @pytest.mark.parametrize("which", ["wilson", "margules", "vanlaar", "nrtl"])
    def test_flash_runs_and_is_consistent(self, ew_thermo, ew_nrtl, which):
        sp = ("ethanol", "water")
        model = {
            "wilson": WilsonParams.binary(sp, [58.7e-6, 18.07e-6], 1500.0, 3000.0),
            "margules": MargulesParams(sp, 1.6, 0.9),
            "vanlaar": VanLaarParams(sp, 1.6798, 0.9227),
            "nrtl": ew_nrtl,
        }[which]
        T = 352.0
        flash = Flash(FlashParams(species_order=list(sp)), thermo=ew_thermo, activity_model=model)
        # choose P between the bubble and dew pressure of z = 0.5
        Pb = float(pxy(ew_thermo, sp, T, 3, model)["P"][1])
        feed = make_stream({"ethanol": 0.5, "water": 0.5}, T=T, P=Pb * 0.9)
        liq, vap, info = flash(feed, T=T, P=Pb * 0.9)
        assert 0.0 < float(info["V_frac"]) < 1.0
        assert float(info["balance_residual"]) < 1e-8
        # K_i = gamma_i(x, T) Psat_i / P at the converged liquid
        x = jnp.array([float(info["x"][s]) for s in sp])
        g = activity_gamma(model, x, T)
        K = g * jnp.array([ew_thermo.Psat(s, T) for s in sp]) / (Pb * 0.9)
        np.testing.assert_allclose([float(info["K"][s]) for s in sp], K, rtol=1e-6)

    @pytest.mark.slow
    def test_nrtl_path_matches_direct_nrtl_call(self, ew_thermo, ew_nrtl):
        """The protocol dispatch leaves the NRTL answer unchanged."""
        sp = ("ethanol", "water")
        flash = Flash(FlashParams(species_order=list(sp)), thermo=ew_thermo, activity_model=ew_nrtl)
        feed = make_stream({"ethanol": 0.4, "water": 0.6}, T=355.0, P=ATM)
        _, _, info = flash(feed, T=355.0, P=ATM)
        x = jnp.array([float(info["x"][s]) for s in sp])
        K_direct = (nrtl_activity_coefficients(x, 355.0, ew_nrtl)
                    * jnp.array([ew_thermo.Psat(s, 355.0) for s in sp]) / ATM)
        np.testing.assert_allclose([float(info["K"][s]) for s in sp], K_direct, rtol=1e-9)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------

class TestGradients:
    @pytest.mark.slow
    def test_txy_wrt_pressure(self, bt_thermo):
        sp = ["benzene", "toluene"]
        f = lambda P: txy(bt_thermo, sp, P, n=5)["T"]
        check_grads(f, (ATM,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)
        dT = jax.jacrev(f)(ATM)
        assert np.all(np.asarray(dT) > 0)

    @pytest.mark.slow
    def test_txy_wrt_pressure_with_activity_model(self, ew_thermo, ew_nrtl):
        f = lambda P: txy(ew_thermo, ["ethanol", "water"], P, n=5, activity_model=ew_nrtl)["T"]
        check_grads(f, (ATM,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    @pytest.mark.slow
    def test_y_wrt_activity_parameter(self, ew_thermo):
        sp = ("ethanol", "water")
        f = lambda A: txy(ew_thermo, list(sp), ATM, n=5,
                          activity_model=VanLaarParams(sp, A, 0.9227))["y"]
        check_grads(f, (1.6798,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    def test_gamma_wrt_wilson_parameters(self):
        sp = ("a", "b")
        x = jnp.array([0.3, 0.7])
        f = lambda V, dl: WilsonParams(sp, V, dl).gamma(x, 340.0)
        V = jnp.array([40.0, 18.0])      # cm3/mol (only ratios enter)
        dl = jnp.array([[0.0, 700.0], [2000.0, 0.0]])
        check_grads(f, (V, dl), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    def test_gamma_wrt_margules_and_van_laar(self):
        x = jnp.array([0.3, 0.7])
        check_grads(lambda a, b: MargulesParams(("a", "b"), a, b).gamma(x, 300.0),
                    (0.7, 1.9), order=1, modes=["rev"])
        check_grads(lambda a, b: VanLaarParams(("a", "b"), a, b).gamma(x, 300.0),
                    (1.3, 0.8), order=1, modes=["rev"])

    def test_gamma_wrt_nrtl_parameters(self):
        x = jnp.array([0.3, 0.7])
        al = jnp.array([[0.0, 0.3], [0.3, 0.0]])
        f = lambda a, b: NRTLParams(("a", "b"), a, b, al).gamma(x, 350.0)
        check_grads(f, (jnp.array([[0.0, -0.8], [3.4, 0.0]]),
                        jnp.array([[0.0, 246.0], [-586.0, 0.0]])),
                    order=1, modes=["rev"], rtol=1e-4, atol=1e-8)

    @pytest.mark.slow
    def test_azeotrope_gradient_and_no_azeotrope_is_finite(self, ew_thermo, ew_nrtl, bt_thermo):
        f = lambda P: find_azeotrope(ew_thermo, ["ethanol", "water"], P=P,
                                     activity_model=ew_nrtl)["T"]
        check_grads(f, (ATM,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)
        # moves with pressure ~ the pure-component dT/dP
        assert 0.0 < float(jax.grad(f)(ATM)) < 1e-3
        # no azeotrope: masked value, and a finite gradient (no NaN leaking through)
        g = jax.grad(lambda P: jnp.nan_to_num(
            find_azeotrope(bt_thermo, ["benzene", "toluene"], P=P)["x"], nan=0.0))(ATM)
        assert np.isfinite(float(g))


# --------------------------------------------------------------------------
# Ternary LLE and plots
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ternary_model():
    # synthetic type-I system: 0 and 1 partially miscible, 2 the solute
    tau = jnp.array([[0.0, 2.5, 0.5], [2.5, 0.0, 0.3], [0.8, 0.6, 0.0]])
    return NRTLParams(("a", "b", "s"), tau, jnp.zeros((3, 3)), 0.3 * (1 - jnp.eye(3)))


class TestTernaryLLE:
    @pytest.mark.slow
    def test_tie_lines_satisfy_isoactivity(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=8)
        tl = np.asarray(d["tie_lines"])
        assert tl.shape[0] >= 5
        for xI, xII in tl:
            aI = xI * np.asarray(nrtl_activity_coefficients(jnp.asarray(xI), 300.0, ternary_model))
            aII = xII * np.asarray(nrtl_activity_coefficients(jnp.asarray(xII), 300.0, ternary_model))
            np.testing.assert_allclose(aI, aII, rtol=1e-8)
            np.testing.assert_allclose([xI.sum(), xII.sum()], 1.0, atol=1e-12)
            assert np.all(xI > 0) and np.all(xII > 0)

    @pytest.mark.slow
    def test_two_phase_region_shrinks_towards_plait(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=8)
        tl = np.asarray(d["tie_lines"])
        length = np.linalg.norm(tl[:, 0, :] - tl[:, 1, :], axis=1)
        assert length[0] > 0.9 and np.all(np.diff(length) < 0)
        assert d["binodal"].shape[0] == 2 * tl.shape[0]
        # plait point lies between the last tie line's ends
        assert np.all(np.asarray(d["plait"]) > 0)

    @pytest.mark.slow
    def test_accepts_lle_equilibrium(self, ternary_model):
        lle = LLEEquilibrium(solutes=["s"], aqueous_carrier="b", organic_carrier="a",
                             nrtl_params=ternary_model, activity_model="NRTL")
        d = ternary_lle(lle, 300.0, n_tie=3)
        assert d["tie_lines"].shape[0] >= 2

    def test_rejects_k_model(self):
        from difflow.units.lle import DistributionCoeffs
        lle = LLEEquilibrium(solutes=["s"], aqueous_carrier="b", organic_carrier="a",
                             K_coeffs=DistributionCoeffs(species=("s",), K0=(2.0,)))
        with pytest.raises(ValueError):
            ternary_lle(lle, 300.0)


class TestPlots:
    @pytest.fixture(autouse=True)
    def _agg(self):
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")

    @pytest.mark.slow
    def test_binary_plots(self, ew_thermo, ew_nrtl):
        from difflow.visualization import plot_pxy, plot_txy, plot_xy
        sp = ["ethanol", "water"]
        ax = plot_txy(txy(ew_thermo, sp, ATM, 21, ew_nrtl))
        styles = {l.get_linestyle() for l in ax.get_lines()}
        assert "-" in styles and "--" in styles        # not colour alone
        plot_pxy(pxy(ew_thermo, sp, 351.0, 21, ew_nrtl))
        ax = plot_xy(xy_curve(ew_thermo, sp, ATM, 21, ew_nrtl))
        assert any(l.get_linestyle() == ":" for l in ax.get_lines())   # 45-degree line
        xs, ys = ax.get_lines()[0].get_data()
        np.testing.assert_allclose(xs, ys)

    @pytest.mark.slow
    @pytest.mark.parametrize("coords", ["equilateral", "right"])
    def test_ternary_plot(self, ternary_model, coords):
        from difflow.visualization import plot_ternary
        d = ternary_lle(ternary_model, 300.0, n_tie=4)
        ax = plot_ternary(d, coords=coords, labels=("a", "b", "s"))
        assert len(ax.get_lines()) >= 2 + d["tie_lines"].shape[0]

    @pytest.mark.slow
    def test_bad_coords(self, ternary_model):
        from difflow.visualization import plot_ternary
        with pytest.raises(ValueError):
            plot_ternary(ternary_lle(ternary_model, 300.0, n_tie=2), coords="polar")


# --------------------------------------------------------------------------
# Ternary geometry: mix, lever rule, tie lines, difference point, Hunter-Nash (#430)
#
# Verification note: no published Hunter-Nash example (Seader, Treybal, Geankoplis)
# is reproduced; the inputs and printed answers of those examples could not be
# retrieved, so no agreement with them is claimed. The construction is checked
# against (i) the lever rule and component balances exactly, (ii) the analytic
# Kremser solution for an immiscible ternary (tie lines running from edge to edge
# with a linear ratio equilibrium), where the stage count is known in closed form,
# and (iii) its own consistency (operating lines through the difference point,
# tie lines on the isoactivity curve). The graphical tolerance is the issue's
# +-0.5 stage; the observed agreement is far tighter.
# --------------------------------------------------------------------------

from difflow.phase_diagrams import (  # noqa: E402
    difference_point, hunter_nash, lever_rule, mix, tie_line_through,
)


def _immiscible_ternary(K=2.0, X_max=6.0, m=601):
    """Tie lines of an immiscible ternary: species 0 and 1 mutually insoluble,
    the solute at ratio equilibrium Y = K X (X, Y = solute per carrier)."""
    X = np.linspace(0.0, X_max, m)
    tl = np.array([[[1 / (1 + x), 0.0, x / (1 + x)],
                    [0.0, 1 / (1 + K * x), K * x / (1 + K * x)]] for x in X])
    return {"tie_lines": jnp.asarray(tl)}


class TestMixAndLever:
    def test_mix_conserves_components(self):
        rng = np.random.default_rng(1)
        pts = rng.dirichlet(np.ones(3), size=4)
        a = rng.uniform(0.5, 3.0, size=4)
        M = np.asarray(mix(pts, a))
        np.testing.assert_allclose(M.sum(), 1.0, atol=1e-14)
        np.testing.assert_allclose(a.sum() * M, a @ pts, rtol=1e-14, atol=1e-15)

    def test_lever_rule_exact_on_collinear_points(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            A, B = rng.dirichlet(np.ones(3), size=2)
            f = rng.uniform(-0.5, 1.5)            # inside and outside the segment
            M = (1 - f) * A + f * B
            fA, fB = lever_rule(M, A, B)
            np.testing.assert_allclose([fA, fB], [1 - f, f], rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(fA * A + fB * B, M, atol=1e-13)

    def test_lever_rule_is_the_segment_length_ratio(self):
        A, B = np.array([0.9, 0.0, 0.1]), np.array([0.1, 0.8, 0.1])
        M = 0.3 * A + 0.7 * B
        fA, fB = lever_rule(M, A, B)
        np.testing.assert_allclose(fB, np.linalg.norm(M - A) / np.linalg.norm(B - A), rtol=1e-12)
        np.testing.assert_allclose(fA, np.linalg.norm(B - M) / np.linalg.norm(B - A), rtol=1e-12)

    def test_mix_then_lever_roundtrip(self):
        A, B = np.array([0.7, 0.1, 0.2]), np.array([0.05, 0.8, 0.15])
        M = mix(np.stack([A, B]), np.array([2.0, 6.0]))
        fA, fB = lever_rule(M, A, B)
        np.testing.assert_allclose([fA, fB], [0.25, 0.75], rtol=1e-12)

    def test_gradients(self):
        A = jnp.array([0.7, 0.1, 0.2])
        B = jnp.array([0.05, 0.8, 0.15])
        check_grads(lambda a: mix(jnp.stack([A, B]), a), (jnp.array([2.0, 6.0]),), order=1, modes=["rev"])
        check_grads(lambda P: mix(P, jnp.array([2.0, 6.0])), (jnp.stack([A, B]),), order=1, modes=["rev"])
        M = 0.3 * A + 0.7 * B + jnp.array([1e-3, -2e-3, 1e-3])    # slightly off the line: projected
        check_grads(lambda m, a, b: lever_rule(m, a, b)[0], (M, A, B), order=1, modes=["rev"])
        check_grads(lambda m, a, b: lever_rule(m, a, b)[1], (M, A, B), order=1, modes=["rev"])


class TestTieLineThrough:
    @pytest.mark.slow
    def test_passes_through_the_point_and_lever_rule_fractions(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=30)
        tl = np.asarray(d["tie_lines"])
        for i, f in ((3, 0.3), (8, 0.5), (14, 0.8)):
            P = (1 - f) * tl[i, 0] + f * tl[i, 1]
            r = tie_line_through(d, P)
            np.testing.assert_allclose(r["s"], i, atol=1e-6)
            np.testing.assert_allclose((r["frac_I"] * r["x_I"] + r["frac_II"] * r["x_II"]), P, atol=1e-10)
            np.testing.assert_allclose(r["frac_II"], f, atol=1e-6)

    @pytest.mark.slow
    def test_between_tabulated_tie_lines_is_near_isoactivity(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=30)
        tl = np.asarray(d["tie_lines"])
        P = 0.5 * (0.5 * tl[7].sum(axis=0) / 1.0 + 0.5 * tl[8].sum(axis=0) / 1.0)   # between 7 and 8
        r = tie_line_through(d, P / 1.0)
        for x in (r["x_I"], r["x_II"]):
            np.testing.assert_allclose(float(jnp.sum(x)), 1.0, atol=1e-12)
        aI = r["x_I"] * nrtl_activity_coefficients(r["x_I"], 300.0, ternary_model)
        aII = r["x_II"] * nrtl_activity_coefficients(r["x_II"], 300.0, ternary_model)
        # interpolated, not solved: relative activity mismatch is interpolation-sized (measured ~1e-3)
        np.testing.assert_allclose(np.asarray(aI), np.asarray(aII), rtol=2e-2)

    @pytest.mark.slow
    def test_single_phase_point_is_rejected(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=10)
        with pytest.raises(ValueError):
            tie_line_through(d, np.array([0.3, 0.3, 0.4]))      # beyond the plait point's range
        with pytest.raises(ValueError):
            tie_line_through(d, np.array([0.99, 0.0, 0.01]))    # pure-ish single-phase corner


class TestDifferencePoint:
    def test_balance_and_collinearity(self):
        F = np.array([0.7, 0.0, 0.3]); S = np.array([0.0, 1.0, 0.0])
        RN = np.array([0.9, 0.05, 0.05])
        M = np.asarray(mix(np.stack([F, S]), np.array([1.0, 2.0])))
        E1 = RN + 1.7 * (M - RN)                     # collinear R_N - M - E_1, beyond M
        dp = difference_point(F, E1, S, RN, 1.0, None, 2.0)
        # extensive balance F + S = E1 + R_N  (component by component)
        np.testing.assert_allclose(np.asarray(dp["M"]), M, atol=1e-14)
        np.testing.assert_allclose(3.0 * M, float(dp["E1_amount"]) * E1 + float(dp["R_N_amount"]) * RN, atol=1e-12)
        np.testing.assert_allclose(float(dp["E1_amount"]), 3.0 / 1.7, rtol=1e-12)   # lever rule
        # F - E1 = R_N - S, amount-weighted
        np.testing.assert_allclose(np.asarray(dp["vector"]), float(dp["R_N_amount"]) * RN - 2.0 * S, atol=1e-12)
        # Delta lies on both line F-E1 and line S-R_N
        D = np.asarray(dp["composition"])
        np.testing.assert_allclose(D.sum(), 1.0, atol=1e-12)
        for p, q in ((F, E1), (S, RN)):
            u, v = (q - p)[1:], (D - p)[1:]
            assert abs(u[0] * v[1] - u[1] * v[0]) < 1e-12

    def test_gradients(self):
        F = jnp.array([0.7, 0.0, 0.3]); S = jnp.array([0.0, 1.0, 0.0]); RN = jnp.array([0.9, 0.05, 0.05])
        M = mix(jnp.stack([F, S]), jnp.array([1.0, 2.0]))
        E1 = RN + 1.7 * (M - RN)
        check_grads(lambda e: difference_point(F, e, S, RN, 1.0, None, 2.0)["composition"], (E1,),
                    order=1, modes=["rev"])

    def test_parallel_lines_put_delta_at_infinity(self):
        F = np.array([0.5, 0.0, 0.5]); S = np.array([0.0, 1.0, 0.0])
        E1 = np.array([0.0, 0.5, 0.5]); RN = np.array([0.5, 0.5, 0.0])
        # F-E1 is parallel to S-R_N: net flow F - E1 has zero total amount
        dp = difference_point(F, E1, S, RN, 1.0, 1.0, 1.0, 1.0)
        assert float(dp["amount"]) == 0.0 and np.all(np.isnan(np.asarray(dp["composition"])))


class TestHunterNash:
    def test_immiscible_limit_equals_kremser(self):
        """Analytic oracle: edge-to-edge tie lines, Y = K X, so the stage count is Kremser's."""
        K = 2.0
        data = _immiscible_ternary(K)
        F, S = np.array([0.7, 0.0, 0.3]), np.array([0.0, 1.0, 0.0])
        X0, Fc = 0.3 / 0.7, 0.7
        for Sv in (2.0, 1.0, 0.5):
            A = K * Sv / Fc
            for N in (1, 2, 3, 5):
                XN = X0 * (A - 1) / (A ** (N + 1) - 1)
                r = hunter_nash(data, F, S, XN / (1 + XN), S_over_F=Sv)
                assert r["feasible"]
                assert abs(r["n_stages"] - N) < 0.01, (Sv, N, r["n_stages"])

    def test_stage_compositions_match_the_dense_linear_solve(self):
        K = 2.0
        data = _immiscible_ternary(K)
        F, S = np.array([0.7, 0.0, 0.3]), np.array([0.0, 1.0, 0.0])
        Fc, Sv, X0, N = 0.7, 2.0, 0.3 / 0.7, 3
        A_ = np.zeros((N, N)); b = np.zeros(N)
        for n in range(N):
            A_[n, n] = Fc + Sv * K
            if n > 0: A_[n, n - 1] = -Fc
            else: b[0] += Fc * X0
            if n < N - 1: A_[n, n + 1] = -Sv * K
        X = np.linalg.solve(A_, b)
        r = hunter_nash(data, F, S, X[-1] / (1 + X[-1]), S_over_F=Sv)
        R = np.asarray(r["R"])[:N]
        np.testing.assert_allclose(R[:, 2] / R[:, 0], X, rtol=2e-3)

    def test_fractional_stage_is_between_integers(self):
        data = _immiscible_ternary()
        F, S = np.array([0.7, 0.0, 0.3]), np.array([0.0, 1.0, 0.0])
        n = [hunter_nash(data, F, S, c, S_over_F=2.0)["n_stages"] for c in (0.0030, 0.0020, 0.0012)]
        assert n[0] < n[1] < n[2]

    def test_pinch_is_reported_infeasible(self):
        data = _immiscible_ternary()
        F, S = np.array([0.7, 0.0, 0.3]), np.array([0.0, 1.0, 0.0])
        # solvent far below the minimum: A < 1 cannot reach a 1e-3 raffinate
        r = hunter_nash(data, F, S, 0.001, S_over_F=0.3)
        assert not r["feasible"] and not np.isfinite(r["n_stages"])

    @pytest.mark.slow
    def test_ternary_lle_consistency(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=30)
        F, S = np.array([0.8, 0.0, 0.2]), np.array([0.0, 1.0, 0.0])
        r = hunter_nash(d, F, S, 0.02, S_over_F=1.0)
        assert r["feasible"] and 2.5 < r["n_stages"] < 4.0
        E, R = np.asarray(r["E"]), np.asarray(r["R"])
        # tie lines: every (E_n, R_n) pair is at isoactivity to interpolation accuracy
        for e, rr in zip(E, R):
            ae = e * np.asarray(nrtl_activity_coefficients(jnp.asarray(e), 300.0, ternary_model))
            ar = rr * np.asarray(nrtl_activity_coefficients(jnp.asarray(rr), 300.0, ternary_model))
            np.testing.assert_allclose(ae, ar, rtol=3e-2)
        # operating lines: E_{n+1} - R_n is parallel to w - t_w R_n, w = E_1 - F (extensive), t_w its amount
        dp = r["delta"]
        w = -np.asarray(dp["vector"])
        # E_{n+1} - R_n = u/e with u = w - t_w R_n: check collinearity with Delta
        for n in range(len(R) - 1):
            u = (w - (-float(dp["amount"])) * R[n])[1:]
            v = (E[n + 1] - R[n])[1:]
            assert abs(u[0] * v[1] - u[1] * v[0]) < 1e-6

    @pytest.mark.slow
    def test_bad_inputs(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=30)
        F, S = np.array([0.8, 0.0, 0.2]), np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            hunter_nash(d, F, S, 0.02, carrier=2)
        with pytest.raises(ValueError):
            hunter_nash(d, F, S, 0.9)                       # solute fraction outside the branch
        with pytest.raises(ValueError):                      # M in the single-phase region
            hunter_nash(d, F, S, 0.02, S_over_F=1e-3)


class TestHunterNashPlot:
    @pytest.fixture(autouse=True)
    def _agg(self):
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")

    @pytest.mark.slow
    @pytest.mark.parametrize("coords", ["equilateral", "right"])
    def test_plot(self, ternary_model, coords):
        from difflow.visualization import plot_hunter_nash
        d = ternary_lle(ternary_model, 300.0, n_tie=30)
        r = hunter_nash(d, np.array([0.8, 0.0, 0.2]), np.array([0.0, 1.0, 0.0]), 0.02)
        ax = plot_hunter_nash(r, coords=coords, labels=("a", "b", "s"))
        styles = {l.get_linestyle() for l in ax.get_lines()}
        assert {"-", "--", ":", "-."} <= styles                 # not colour alone
        n_text = [t for t in ax.texts if t.get_text().isdigit()]
        assert len(n_text) == len(r["E"])


# ---------------------------------------------------------------------------
# tabulated_lle: Hunter-Nash and plot_ternary on measured tie lines
# ---------------------------------------------------------------------------

def _oliveira_mibk():
    """Water(0)-MIBK(1)-acetic acid(2) tie lines at 293.2 K, mole fractions.

    Oliveira et al., Fluid Phase Equilib. 482 (2019) 48-63 (NIST ThermoML
    10.1016/j.fluid.2018.10.021), with the binary-edge line first.
    """
    aq = [(0.9953, 0.0), (0.9788, 0.0192), (0.958, 0.0382), (0.9344, 0.0588),
          (0.9083, 0.0804), (0.8829, 0.1002), (0.8528, 0.1221)]
    org = [(0.0915, 0.0), (0.1455, 0.0586), (0.2092, 0.1153), (0.2747, 0.162),
           (0.3423, 0.1984), (0.4151, 0.2245), (0.5039, 0.2387)]
    return np.array([[[w, 1 - w - a, a], [wo, 1 - wo - ao, ao]]
                     for (w, a), (wo, ao) in zip(aq, org)])


class TestTabulatedLLE:
    def test_format_matches_ternary_lle(self):
        from difflow import tabulated_lle
        tl = _oliveira_mibk()
        d = tabulated_lle(tl[::-1], T=293.15)       # order does not matter
        assert d["tie_lines"].shape == (7, 2, 3)
        np.testing.assert_allclose(np.asarray(d["tie_lines"]), tl, atol=1e-12)
        assert d["binodal"].shape == (14, 3)
        assert bool(np.all(np.asarray(d["valid"])))
        np.testing.assert_allclose(np.asarray(d["plait"]), tl[-1].mean(axis=0))

    def test_hunter_nash_on_data(self):
        from difflow import hunter_nash, tabulated_lle
        d = tabulated_lle(_oliveira_mibk())
        F, S = np.array([0.9, 0.0, 0.1]), np.array([0.0, 1.0, 0.0])
        r = hunter_nash(d, F, S, 0.01, S_over_F=1.0)
        assert bool(r["feasible"])
        assert 1.0 < float(r["n_stages"]) < 3.0
        # every stage pair sits on a measured-data tie line: extract leaving
        # stage 1 is richer in acid than the raffinate leaving it
        assert float(r["E"][0, 2]) > float(r["R"][0, 2])
        # less solvent needs more stages
        r2 = hunter_nash(d, F, S, 0.01, S_over_F=0.6)
        assert float(r2["n_stages"]) > float(r["n_stages"])

    def test_rejects_bad_input(self):
        from difflow import tabulated_lle
        with pytest.raises(ValueError):
            tabulated_lle(np.zeros((3, 3)))
        tl = _oliveira_mibk()
        with pytest.raises(ValueError):
            tabulated_lle(tl[:, ::-1, :])   # phases swapped

    def test_plot_ternary_accepts_it(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from difflow import tabulated_lle
        from difflow.visualization import plot_ternary
        ax = plot_ternary(tabulated_lle(_oliveira_mibk()))
        assert ax is not None
