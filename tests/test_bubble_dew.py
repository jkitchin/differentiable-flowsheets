"""Bubble/dew points: EOS functions, modified Raoult, Flash helpers, warnings.

Covers the Ch. 2 gaps: the Flash bubble/dew helpers ignored ``activity_model``
and ``eos`` and clipped silently to [150, 700] K; there were no EOS bubble/dew
functions; and Antoine extrapolation was silent.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import (
    IdealThermo, PengRobinson, SRK, SpeciesData, make_stream,
    AntoineRangeWarning, BubbleDewConvergenceWarning,
    eos_bubble_P, eos_bubble_T, eos_dew_P, eos_dew_T,
    bubble_T, bubble_P, dew_T, dew_P, txy,
)
from difflow.database import get_critical_props, get_species_data
from difflow.eos import flash_TP_eos, _phase_stability
from difflow.units.flash import Flash, FlashParams
from difflow.units.lle import NRTLParams, nrtl_activity_coefficients

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _ethanol_water():
    # Same NRTL/Antoine set as tests/test_flash_azeotrope.py (default ranges).
    th = IdealThermo({
        "ethanol": SpeciesData(name="ethanol", MW=46.07, Cp_coeffs=(0, 0, 0, 0),
                               Hvap_coeffs=(1.0, 0.38, 514.0),
                               antoine_coeffs=(10.24677, 1598.673, -46.424)),
        "water": SpeciesData(name="water", MW=18.02, Cp_coeffs=(0, 0, 0, 0),
                             Hvap_coeffs=(1.0, 0.38, 647.0),
                             antoine_coeffs=(9.6543, 1435.264, -64.848)),
    })
    nrtl = NRTLParams(
        species=("ethanol", "water"),
        a=jnp.array([[0.0, 3.4578], [-0.8009, 0.0]]),
        b=jnp.array([[0.0, -586.0809], [246.18, 0.0]]),
        alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
    )
    return th, nrtl


SP_EW = ("ethanol", "water")


def _c1c3():
    return PengRobinson({s: get_critical_props(s) for s in ("methane", "propane")})


def _ss_bubble_P(eos, x, T, P0, n=3000):
    """Independent bubble P: plain successive substitution in NumPy."""
    x = np.asarray(x, float)
    P = P0
    K = np.asarray(eos.K_values_wilson(T, P))
    for _ in range(n):
        y = K * x / np.sum(K * x)
        phiL = np.asarray(eos.fugacity_coefficient(T, P, jnp.asarray(x), "liquid"))
        phiV = np.asarray(eos.fugacity_coefficient(T, P, jnp.asarray(y), "vapor"))
        K = phiL / phiV
        S = np.sum(K * x)
        P = P * S
        if abs(S - 1) < 1e-12:
            break
    return P, y


# ---------------------------------------------------------------------------
# EOS bubble/dew
# ---------------------------------------------------------------------------

class TestEOSBubbleDew:
    def test_bubble_P_matches_independent_successive_substitution(self):
        eos = _c1c3()
        x = jnp.array([0.2, 0.8])
        r = eos_bubble_P(eos, x, 277.6)
        P_ref, y_ref = _ss_bubble_P(eos, x, 277.6, 30e5)
        assert bool(r["converged"]) and not bool(r["trivial"])
        assert float(r["P"]) == pytest.approx(P_ref, rel=1e-8)
        np.testing.assert_allclose(np.asarray(r["y"]), y_ref, rtol=1e-6)
        assert float(r["P"]) / 1e5 == pytest.approx(33.41, abs=0.05)

    def test_bubble_P_brackets_the_tp_flash(self):
        eos = _c1c3()
        z = jnp.array([0.2, 0.8])
        Pb = float(eos_bubble_P(eos, z, 277.6)["P"])
        V_below, _, _ = flash_TP_eos(eos, z, 277.6, 0.97 * Pb)
        assert 0.0 < float(V_below) < 0.1
        # Above it the feed is one (liquid) phase.
        two_phase, is_vapor = _phase_stability(eos, z, 277.6, 1.03 * Pb)
        assert not bool(two_phase) and not bool(is_vapor)
        V_above, x, y = flash_TP_eos(eos, z, 277.6, 1.03 * Pb)
        assert float(V_above) == 0.0

    @pytest.mark.parametrize("P_bar", [34.0, 36.0, 50.0, 100.0])
    def test_single_root_compressed_liquid_is_liquid(self, P_bar):
        # Above the bubble point the cubic has one root; it used to be labelled
        # vapour because "largest root" and "smallest root" coincide.
        eos = _c1c3()
        z = jnp.array([0.2, 0.8])
        Zv = eos.solve_Z(277.6, P_bar * 1e5, z, "vapor")
        Zl = eos.solve_Z(277.6, P_bar * 1e5, z, "liquid")
        assert float(Zv) == pytest.approx(float(Zl))  # one root
        V, _, _ = flash_TP_eos(eos, z, 277.6, P_bar * 1e5)
        assert float(V) == 0.0

    def test_single_root_gas_is_vapour(self):
        # Methane well above its critical temperature: one root, a gas.
        eos = _c1c3()
        z = jnp.array([0.999, 0.001])
        for P in (1e5, 50e5, 100e5):
            V, _, _ = flash_TP_eos(eos, z, 300.0, P)
            assert float(V) == 1.0

    def test_phase_identification_parameter(self):
        from difflow import phase_identification_parameter, eos_pressure_TV
        eos = _c1c3()
        # A subcooled liquid is PIP > 1, a dilute gas PIP < 1.
        assert float(phase_identification_parameter(
            eos, 250.0, 20e5, jnp.array([0.0, 1.0]))) > 1.0
        assert float(phase_identification_parameter(
            eos, 300.0, 1e5, jnp.array([1.0, 0.0]))) < 1.0
        # Two roots: the liquid root is liquid-like, the vapour root vapour-like.
        z = jnp.array([0.2, 0.8])
        assert float(phase_identification_parameter(eos, 277.6, 20e5, z, phase="liquid")) > 1
        assert float(phase_identification_parameter(eos, 277.6, 20e5, z, phase="vapor")) < 1
        # eos_pressure_TV inverts solve_Z.
        Z = eos.solve_Z(277.6, 20e5, z, "liquid")
        v = Z * 8.314462618 * 277.6 / 20e5
        assert float(eos_pressure_TV(eos, 277.6, v, z)) == pytest.approx(20e5, rel=1e-6)

    def test_dew_P_brackets_the_tp_flash(self):
        eos = SRK({s: get_critical_props(s) for s in ("propane", "n_butane", "n_pentane")})
        z = jnp.array([0.3, 0.4, 0.3])
        r = eos_dew_P(eos, z, 300.0)
        Pd = float(r["P"])
        assert bool(r["converged"]) and not bool(r["trivial"])
        assert float(jnp.sum(r["x"])) == pytest.approx(1.0)
        V_above, _, _ = flash_TP_eos(eos, z, 300.0, 1.03 * Pd)
        V_below, _, _ = flash_TP_eos(eos, z, 300.0, 0.97 * Pd)
        assert 0.9 < float(V_above) < 1.0
        assert float(V_below) == 1.0

    @pytest.mark.parametrize("kind", ["bubble", "dew"])
    def test_T_and_P_forms_are_inverse(self, kind):
        eos = SRK({s: get_critical_props(s) for s in ("propane", "n_butane", "n_pentane")})
        z = jnp.array([0.3, 0.4, 0.3])
        fP, fT = (eos_bubble_P, eos_bubble_T) if kind == "bubble" else (eos_dew_P, eos_dew_T)
        P = fP(eos, z, 290.0)["P"]
        r = fT(eos, z, P)
        assert bool(r["converged"])
        assert float(r["T"]) == pytest.approx(290.0, abs=1e-6)

    def test_gradients_bubble_P(self):
        eos = _c1c3()
        x = jnp.array([0.2, 0.8])
        check_grads(lambda T: eos_bubble_P(eos, x, T)["P"] / 1e5, (277.6,),
                    order=1, modes=["rev"], atol=1e-4, rtol=1e-4)
        check_grads(lambda x1: eos_bubble_P(eos, jnp.stack([x1, 1 - x1]), 277.6)["P"] / 1e5,
                    (0.2,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)

    def test_gradients_dew_T(self):
        eos = SRK({s: get_critical_props(s) for s in ("propane", "n_butane", "n_pentane")})
        z = jnp.array([0.3, 0.4, 0.3])
        check_grads(lambda lp: eos_dew_T(eos, z, jnp.exp(lp))["T"], (jnp.log(101325.0),),
                    order=1, modes=["rev"], atol=1e-4, rtol=1e-4)

    def test_clausius_clapeyron_sign(self):
        # dP_bubble/dT > 0
        eos = _c1c3()
        g = jax.grad(lambda T: eos_bubble_P(eos, jnp.array([0.2, 0.8]), T)["P"])(277.6)
        assert float(g) > 0

    def test_near_critical_bubble_point_avoids_the_trivial_root(self):
        # x = 0.3 at 300 K: two-phase up to ~57 bar (stability test). Plain
        # substitution from the Wilson pressure (107 bar) slides onto K = 1 at
        # 63.4 bar; the safeguarded start finds the real point.
        eos = _c1c3()
        r = eos_bubble_P(eos, jnp.array([0.3, 0.7]), 300.0)
        assert bool(r["converged"]) and not bool(r["trivial"])
        assert float(r["K"][0]) > 2.0
        two_lo, _ = _phase_stability(eos, jnp.array([0.3, 0.7]), 300.0, 0.99 * r["P"])
        two_hi, _ = _phase_stability(eos, jnp.array([0.3, 0.7]), 300.0, 1.01 * r["P"])
        assert bool(two_lo) and not bool(two_hi)

    @pytest.mark.parametrize("x1, T, P_ref", [
        (0.90, 205.0, 55.58e5), (0.95, 200.0, 52.54e5)])
    def test_near_cricondenbar_bubble_P_by_continuation(self, x1, T, P_ref):
        # Methane-rich liquids near the cricondenbar: every Wilson start
        # failed (unconverged, ~15 bar). The continuation in T from 0.75 T
        # finds the point; P_ref is that root, and the stability test
        # brackets it.
        eos = _c1c3()
        x = jnp.array([x1, 1 - x1])
        r = eos_bubble_P(eos, x, T)
        assert bool(r["converged"]) and not bool(r["trivial"])
        assert float(r["P"]) == pytest.approx(P_ref, rel=1e-3)
        two_lo, _ = _phase_stability(eos, x, T, 0.995 * r["P"])
        two_hi, _ = _phase_stability(eos, x, T, 1.005 * r["P"])
        assert bool(two_lo) and not bool(two_hi)
        # The root carries implicit derivatives: dP/dT > 0.
        g = jax.grad(lambda T_: eos_bubble_P(eos, x, T_)["P"])(T)
        assert np.isfinite(float(g)) and float(g) > 0

    def test_near_critical_bubble_T_heavy_liquid(self):
        # A 25/75 heptane/octane liquid at 20 bar bubbles at ~543 K, 25 K
        # from octane's Tc. The 0.95-of-Wilson start alone fails here
        # (unconverged at 515 K); the multi-start finds the root, which the
        # stability test brackets.
        from difflow.database import get_critical_props
        eos = PengRobinson({s: get_critical_props(s) for s in ("n_heptane", "n_octane")})
        x = jnp.array([0.25, 0.75])
        r = eos_bubble_T(eos, x, 20e5)
        assert bool(r["converged"]) and not bool(r["trivial"])
        assert 535.0 < float(r["T"]) < 550.0
        two_lo, _ = _phase_stability(eos, x, 0.998 * r["T"], 20e5)
        two_hi, _ = _phase_stability(eos, x, 1.002 * r["T"], 20e5)
        assert not bool(two_lo) and bool(two_hi)

    def test_upper_saturation_point_of_equimolar_mixture(self):
        # x = 0.5 at 300 K is on the liquid side of the CH4/C3H8 critical
        # composition (~0.65 at 300 K): it saturates at ~86.6 bar against a
        # methane-richer phase (y_CH4 ~ 0.70). Both phases have a single
        # cubic root there, so the root labels do not matter, and the
        # stability test brackets the point.
        eos = _c1c3()
        x = jnp.array([0.5, 0.5])
        r = eos_bubble_P(eos, x, 300.0)
        assert bool(r["converged"]) and not bool(r["trivial"])
        assert float(r["y"][0]) > 0.6
        two_lo, _ = _phase_stability(eos, x, 300.0, 0.99 * r["P"])
        two_hi, _ = _phase_stability(eos, x, 300.0, 1.01 * r["P"])
        assert bool(two_lo) and not bool(two_hi)

    def test_trivial_solution_is_flagged_and_warned(self):
        # Started from 150 bar, above the saturation point, the single
        # (guessed) attempt slides onto K = 1; the flag and warning say so.
        eos = _c1c3()
        with pytest.warns(BubbleDewConvergenceWarning, match="trivial"):
            r = eos_bubble_P(eos, jnp.array([0.5, 0.5]), 300.0, P_guess=150e5)
        assert bool(r["trivial"])

    def test_no_saturation_point_is_flagged(self):
        # x = 0.9 at 300 K never splits (stability test over 5-150 bar): there
        # is no bubble point, and the solve says so instead of returning one.
        eos = _c1c3()
        x = jnp.array([0.9, 0.1])
        assert not any(bool(_phase_stability(eos, x, 300.0, P)[0])
                       for P in np.linspace(5e5, 150e5, 30))
        with pytest.warns(BubbleDewConvergenceWarning):
            r = eos_bubble_P(eos, x, 300.0)
        assert not (bool(r["converged"]) and not bool(r["trivial"]))

    def test_no_warning_under_jit(self):
        eos = _c1c3()
        f = jax.jit(lambda T: eos_bubble_P(eos, jnp.array([0.5, 0.5]), T,
                                           P_guess=150e5)["trivial"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert bool(f(300.0))


# ---------------------------------------------------------------------------
# modified Raoult (phase_diagrams)
# ---------------------------------------------------------------------------

def _gamma(nrtl, x, T):
    return np.asarray(nrtl_activity_coefficients(jnp.asarray(x), T, nrtl))


def _psat(th, T):
    return np.array([float(th.Psat(s, T)) for s in SP_EW])


def _dew_T_reference(th, nrtl, y, P):
    """Independent dew T: bisection on T with an inner fixed point for x."""
    y = np.asarray(y, float)

    def g(T):
        x = y.copy()
        for _ in range(500):
            xn = y * P / (_gamma(nrtl, x, T) * _psat(th, T))
            s = xn.sum()
            xn = xn / s
            if np.max(np.abs(xn - x)) < 1e-14:
                break
            x = xn
        return s - 1.0

    lo, hi = 330.0, 380.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if g(lo) * g(m) <= 0:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


class TestModifiedRaoult:
    def test_bubble_P_is_explicit(self):
        th, nrtl = _ethanol_water()
        x, T = np.array([0.3, 0.7]), 350.0
        ref = np.sum(x * _gamma(nrtl, x, T) * _psat(th, T))
        assert float(bubble_P(th, SP_EW, x, T, nrtl)) == pytest.approx(ref, rel=1e-12)

    def test_dew_T_matches_independent_solve(self):
        th, nrtl = _ethanol_water()
        y = [0.4, 0.6]
        T = float(dew_T(th, SP_EW, y, 101325.0, nrtl))
        assert T == pytest.approx(_dew_T_reference(th, nrtl, y, 101325.0), abs=1e-6)

    def test_dew_P_and_dew_T_are_inverse(self):
        th, nrtl = _ethanol_water()
        y = jnp.array([0.4, 0.6])
        P = dew_P(th, SP_EW, y, 360.0, nrtl)
        assert float(dew_T(th, SP_EW, y, P, nrtl)) == pytest.approx(360.0, abs=1e-6)

    def test_raoult_dew_P_closed_form(self):
        th, _ = _ethanol_water()
        y = np.array([0.4, 0.6])
        ref = 1.0 / np.sum(y / _psat(th, 360.0))
        assert float(dew_P(th, SP_EW, y, 360.0)) == pytest.approx(ref, rel=1e-12)

    def test_dew_T_gradient(self):
        th, nrtl = _ethanol_water()
        check_grads(lambda P: dew_T(th, SP_EW, jnp.array([0.4, 0.6]), P * 1e5, nrtl),
                    (1.01325,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Flash helpers
# ---------------------------------------------------------------------------

class TestFlashHelpers:
    def _flash(self, model=True):
        th, nrtl = _ethanol_water()
        return Flash(FlashParams(species_order=list(SP_EW)), th,
                     activity_model=nrtl if model else None), th, nrtl

    def test_bubble_point_pressure_uses_activity_model(self):
        fl, th, nrtl = self._flash()
        fl0, _, _ = self._flash(model=False)
        feed = make_stream({"ethanol": 30.0, "water": 70.0}, T=350.0, P=101325.0)
        x = np.array([0.3, 0.7])
        ref = np.sum(x * _gamma(nrtl, x, 350.0) * _psat(th, 350.0))
        assert float(fl.bubble_point_pressure(feed)) == pytest.approx(ref, rel=1e-10)
        # positive deviations: gamma > 1 raises the bubble pressure over Raoult
        assert float(fl.bubble_point_pressure(feed)) > 1.2 * float(fl0.bubble_point_pressure(feed))

    def test_bubble_point_temperature_uses_activity_model(self):
        fl, th, nrtl = self._flash()
        feed = make_stream({"ethanol": 30.0, "water": 70.0}, T=350.0, P=101325.0)
        T = float(fl.bubble_point_temperature(feed))
        x = np.array([0.3, 0.7])
        assert np.sum(x * _gamma(nrtl, x, T) * _psat(th, T)) == pytest.approx(101325.0, rel=1e-8)

    def test_dew_point_temperature_uses_activity_model(self):
        fl, th, nrtl = self._flash()
        feed = make_stream({"ethanol": 40.0, "water": 60.0}, T=350.0, P=101325.0)
        T = float(fl.dew_point_temperature(feed))
        assert T == pytest.approx(_dew_T_reference(th, nrtl, [0.4, 0.6], 101325.0), abs=1e-6)

    def test_eos_flash_helpers_delegate(self):
        eos = _c1c3()
        th = IdealThermo({s: get_species_data(s) for s in ("methane", "propane")})
        fl = Flash(FlashParams(species_order=["methane", "propane"]), th, eos=eos)
        feed = make_stream({"methane": 20.0, "propane": 80.0}, T=277.6, P=30e5)
        z = jnp.array([0.2, 0.8])
        assert float(fl.bubble_point_pressure(feed)) == pytest.approx(
            float(eos_bubble_P(eos, z, 277.6)["P"]), rel=1e-12)
        assert float(fl.dew_point_pressure(feed)) == pytest.approx(
            float(eos_dew_P(eos, z, 277.6)["P"]), rel=1e-12)
        Pb = fl.bubble_point_pressure(feed)
        assert float(fl.bubble_point_temperature(feed, P=Pb)) == pytest.approx(277.6, abs=1e-6)

    def test_no_silent_clip_above_700K(self):
        # Antoine Tb = 3000/(9 - log10 101325) = 751 K; the old helper clipped to 700.
        heavy = SpeciesData(name="H", MW=300.0, Cp_coeffs=(0, 0, 0, 0),
                            Hvap_coeffs=(1.0, 0.38, 900.0), antoine_coeffs=(9.0, 3000.0, 0.0))
        fl = Flash(FlashParams(species_order=["H"]), IdealThermo({"H": heavy}))
        feed = make_stream({"H": 1.0}, T=300.0, P=101325.0)
        T = float(fl.bubble_point_temperature(feed))
        assert T == pytest.approx(3000.0 / (9.0 - np.log10(101325.0)), rel=1e-8)
        assert T > 700.0

    def test_unconverged_solve_warns(self):
        fl, _, _ = self._flash(model=False)
        feed = make_stream({"ethanol": 30.0, "water": 70.0}, T=350.0, P=101325.0)
        with pytest.warns(BubbleDewConvergenceWarning):
            T = fl.bubble_point_temperature(feed, T_guess=float("nan"))
        assert not np.isfinite(float(T))  # returned as is, not clipped


# ---------------------------------------------------------------------------
# Antoine extrapolation warnings
# ---------------------------------------------------------------------------

class TestAntoineRangeWarning:
    def _thermo(self):
        return IdealThermo({s: get_species_data(s) for s in ("benzene", "toluene")})

    def test_psat_in_range_is_silent(self):
        th = self._thermo()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            th.Psat("benzene", 350.0)

    def test_psat_out_of_range_warns(self):
        th = self._thermo()
        with pytest.warns(AntoineRangeWarning, match="'benzene' extrapolated below"):
            th.Psat("benzene", 250.0)
        with pytest.warns(AntoineRangeWarning, match="above"):
            th.Psat("benzene", 500.0)

    def test_no_warning_under_jit_or_grad(self):
        th = self._thermo()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jax.jit(lambda T: th.Psat("benzene", T))(250.0)
            jax.grad(lambda T: th.Psat("benzene", T))(250.0)

    def test_txy_warns_once_per_species(self):
        th = self._thermo()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            txy(th, ("benzene", "toluene"), 20 * 101325.0)
        msgs = [str(x.message) for x in w if issubclass(x.category, AntoineRangeWarning)]
        assert len(msgs) == 2
        assert any("'benzene'" in m for m in msgs) and any("'toluene'" in m for m in msgs)

    def test_txy_in_range_is_silent(self):
        th = self._thermo()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            txy(th, ("benzene", "toluene"), 101325.0, n=11)

    def test_bubble_T_warns_on_concrete_result(self):
        th = self._thermo()
        with pytest.warns(AntoineRangeWarning):
            bubble_T(th, ("benzene", "toluene"), [0.5, 0.5], 30 * 101325.0)

    def test_psat_with_info_works_under_jit(self):
        th = self._thermo()
        ok = jax.jit(lambda T: th.Psat_with_info("benzene", T)[1]["antoine_in_range"])
        assert bool(ok(350.0)) and not bool(ok(250.0))
        assert th.Psat_with_info("benzene", 350.0)[1]["antoine_in_range"] is True


def test_pure_component_is_not_flagged_trivial():
    """K = 1 is the answer for a pure species; no trivial flag, no warning."""
    import warnings as _w

    from difflow import PengRobinson, get_critical_props
    from difflow.eos import eos_bubble_P, eos_bubble_T, eos_dew_P

    pr = PengRobinson({"propane": get_critical_props("propane")})
    with _w.catch_warnings():
        _w.simplefilter("error")
        r = eos_bubble_P(pr, jnp.array([1.0]), 300.0)
        d = eos_dew_P(pr, jnp.array([1.0]), 300.0)
        t = eos_bubble_T(pr, jnp.array([1.0]), float(r["P"]))
    assert bool(r["converged"]) and not bool(r["trivial"])
    assert float(r["P"]) == pytest.approx(float(d["P"]), rel=1e-8)
    assert float(t["T"]) == pytest.approx(300.0, abs=1e-6)
    assert 9.5e5 < float(r["P"]) < 1.05e6   # NIST propane Psat(300 K) = 9.97 bar
