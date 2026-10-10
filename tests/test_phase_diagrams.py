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
    def test_txy_wrt_pressure(self, bt_thermo):
        sp = ["benzene", "toluene"]
        f = lambda P: txy(bt_thermo, sp, P, n=5)["T"]
        check_grads(f, (ATM,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)
        dT = jax.jacrev(f)(ATM)
        assert np.all(np.asarray(dT) > 0)

    def test_txy_wrt_pressure_with_activity_model(self, ew_thermo, ew_nrtl):
        f = lambda P: txy(ew_thermo, ["ethanol", "water"], P, n=5, activity_model=ew_nrtl)["T"]
        check_grads(f, (ATM,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

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

    def test_two_phase_region_shrinks_towards_plait(self, ternary_model):
        d = ternary_lle(ternary_model, 300.0, n_tie=8)
        tl = np.asarray(d["tie_lines"])
        length = np.linalg.norm(tl[:, 0, :] - tl[:, 1, :], axis=1)
        assert length[0] > 0.9 and np.all(np.diff(length) < 0)
        assert d["binodal"].shape[0] == 2 * tl.shape[0]
        # plait point lies between the last tie line's ends
        assert np.all(np.asarray(d["plait"]) > 0)

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

    @pytest.mark.parametrize("coords", ["equilateral", "right"])
    def test_ternary_plot(self, ternary_model, coords):
        from difflow.visualization import plot_ternary
        d = ternary_lle(ternary_model, 300.0, n_tie=4)
        ax = plot_ternary(d, coords=coords, labels=("a", "b", "s"))
        assert len(ax.get_lines()) >= 2 + d["tie_lines"].shape[0]

    def test_bad_coords(self, ternary_model):
        from difflow.visualization import plot_ternary
        with pytest.raises(ValueError):
            plot_ternary(ternary_lle(ternary_model, 300.0, n_tie=2), coords="polar")
