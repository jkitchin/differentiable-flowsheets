"""Tests for the drying library and the Dryer unit (issue #432).

Oracles (nothing here is a published textbook number; see the end of the module
docstring):

* the closed-form constant + linear-falling-rate time, against an independent
  ``scipy.integrate.quad`` of ``dX/R`` over the same piecewise rate curve;
* the smooth switch converging to the exact piecewise answer as ``sharpness`` grows;
* GAB / BET against hand-written NumPy formulas, and their inverses;
* the continuous dryer's water and enthalpy balances, recomputed from the
  *streams* with plain ASHRAE formulas (not the library's functions);
* the wet bulb solved independently by ``scipy.optimize.brentq``;
* the gradient-based design example against an independent root find.

Not reproduced: the Geankoplis Ch. 9 and McCabe Ch. 24 examples, GAB parameters
for a real material and the h = 0.0204 G^0.8 tray correlation. The issue calls
all of those numbers "unverified, from recollection" and none of their inputs
or printed answers could be reproduced with confidence here, so no test claims
agreement with a published example; the correlation is tested against its own
formula only.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy import integrate, optimize

from difflow import (
    Dryer,
    DryerParams,
    constant_drying_rate,
    dry_basis,
    drying_periods,
    drying_rate_curve,
    drying_time,
    equilibrium_moisture,
    free_moisture,
    heat_transfer_coefficient_tray,
    solids_moisture,
    water_activity,
    wet_basis,
)
from difflow.psychrometrics import (
    MW_DRY_AIR,
    MW_WATER,
    air_stream_from_state,
    saturation_pressure,
    state,
    wet_bulb_temperature,
)

K0 = 273.15

# a representative (invented, not published) drying problem
X1, X2, XC, XE = 0.35, 0.10, 0.20, 0.05
RC = 2.78e-4          # kg/m^2/s
LSA = 25.0            # kg dry solid / m^2


def closed_form(X1, X2, Xc, Xe, Rc, LsA):
    """Piecewise closed form, written with explicit Python branches (the thing the smooth form replaces)."""
    t = 0.0
    xa = min(X1, Xc)
    if X1 > Xc:
        t += LsA * (X1 - max(X2, Xc)) / Rc
    if X2 < xa:
        t += LsA * (Xc - Xe) / Rc * math.log((xa - Xe) / (X2 - Xe))
    return t


# =============================================================================
# Moisture bookkeeping and isotherms
# =============================================================================


class TestMoisture:
    def test_basis_conversions(self):
        assert float(dry_basis(0.2)) == pytest.approx(0.25)
        assert float(wet_basis(0.25)) == pytest.approx(0.2)
        w = jnp.linspace(0.01, 0.9, 7)
        np.testing.assert_allclose(wet_basis(dry_basis(w)), w, rtol=1e-14)
        assert float(free_moisture(0.3, 0.05)) == pytest.approx(0.25)

    def test_solids_moisture(self):
        s = {"F_solid": 2.0, "F_H2O": 0.5 / MW_WATER}
        assert float(solids_moisture(s)) == pytest.approx(0.25, rel=1e-12)
        s = {"F_pellet": 2.0, "F_H2O": 0.5 / MW_WATER}
        assert float(solids_moisture(s, "pellet", 0.5)) == pytest.approx(0.5, rel=1e-12)

    def test_gab_against_numpy(self):
        a = np.array([0.0, 0.1, 0.3, 0.5, 0.8, 0.9])
        Xm, C, K = 0.07, 12.0, 0.85
        ref = Xm * C * K * a / ((1 - K * a) * (1 - K * a + C * K * a))
        got = equilibrium_moisture(a, 300.0, "gab", {"X_m": Xm, "C": C, "K": K})
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)

    def test_bet_is_gab_with_unit_K_and_matches_numpy(self):
        a = np.array([0.05, 0.2, 0.35])
        Xm, C = 0.06, 9.0
        ref = Xm * C * a / ((1 - a) * (1 + (C - 1) * a))
        got = equilibrium_moisture(a, 300.0, "bet", {"X_m": Xm, "C": C})
        np.testing.assert_allclose(got, ref, rtol=1e-12)
        gab = equilibrium_moisture(a, 300.0, "gab", {"X_m": Xm, "C": C, "K": 1.0})
        np.testing.assert_allclose(got, gab, rtol=1e-14)

    def test_monolayer_and_monotone(self):
        # BET at a_w -> small: X ~ X_m C a (Henry limit)
        Xm, C = 0.06, 9.0
        X = float(equilibrium_moisture(1e-6, 300.0, "bet", {"X_m": Xm, "C": C}))
        assert X == pytest.approx(Xm * C * 1e-6, rel=1e-5)
        a = jnp.linspace(0.0, 0.95, 40)
        X = equilibrium_moisture(a, 300.0, "gab", {"X_m": Xm, "C": C, "K": 0.9})
        assert bool(jnp.all(jnp.diff(X) > 0.0))

    @pytest.mark.parametrize("model,params", [
        ("gab", {"X_m": 0.07, "C": 10.0, "K": 0.9}),
        ("gab", {"X_m": 0.05, "C": 0.8, "K": 0.95}),     # C < 1
        ("gab", {"X_m": 0.05, "C": 1.0, "K": 0.9}),      # C = 1 (the quadratic degenerates)
        ("bet", {"X_m": 0.06, "C": 9.0}),
        ("table", {"a_w": [0.0, 0.2, 0.5, 0.8, 0.95], "X": [0.0, 0.04, 0.09, 0.2, 0.5]}),
    ])
    def test_water_activity_inverts(self, model, params):
        a = jnp.array([0.05, 0.2, 0.45, 0.7])
        X = equilibrium_moisture(a, 310.0, model, params)
        back = water_activity(X, 310.0, model, params)
        np.testing.assert_allclose(back, a, rtol=1e-8, atol=1e-10)

    def test_temperature_dependence(self):
        p0 = {"X_m": 0.07, "C": 10.0, "K": 0.9}
        p1 = {**p0, "Hc": 5000.0, "Hk": -1500.0}
        # without Hc / Hk the isotherm ignores T
        assert float(equilibrium_moisture(0.5, 280.0, "gab", p0)) == float(equilibrium_moisture(0.5, 350.0, "gab", p0))
        a = 0.5
        for T in (280.0, 340.0):
            C = 10.0 * math.exp(5000.0 / (8.314462618 * T))
            K = 0.9 * math.exp(-1500.0 / (8.314462618 * T))
            ref = 0.07 * C * K * a / ((1 - K * a) * (1 - K * a + C * K * a))
            assert float(equilibrium_moisture(a, T, "gab", p1)) == pytest.approx(ref, rel=1e-12)
        # the inverse honours the temperature too
        X = equilibrium_moisture(0.4, 330.0, "gab", p1)
        assert float(water_activity(X, 330.0, "gab", p1)) == pytest.approx(0.4, rel=1e-9)

    def test_table_and_callable(self):
        tab = {"a_w": [0.0, 0.5, 1.0], "X": [0.0, 0.1, 0.4]}
        assert float(equilibrium_moisture(0.5, 300.0, "table", tab)) == pytest.approx(0.1)
        assert float(equilibrium_moisture(0.0, 300.0, "table", tab)) == pytest.approx(0.0)
        f = lambda a, T: 0.2 * a  # noqa: E731
        assert float(equilibrium_moisture(0.5, 300.0, f)) == pytest.approx(0.1)
        assert float(water_activity(0.1, 300.0, f)) == pytest.approx(0.5, rel=1e-8)
        with pytest.raises(ValueError):
            equilibrium_moisture(0.5, 300.0, "gab")

    def test_isotherm_gradient(self):
        p = {"X_m": 0.07, "C": 10.0, "K": 0.9}
        check_grads(lambda a, Xm: equilibrium_moisture(a, 300.0, "gab", {**p, "X_m": Xm}), (0.4, 0.07),
                    order=1, modes=["rev"], rtol=1e-6, atol=1e-9)
        check_grads(lambda X: water_activity(X, 300.0, "gab", p), (0.1,), order=1, modes=["rev"],
                    rtol=1e-6, atol=1e-9)
        tab = {"a_w": [0.0, 0.2, 0.5, 0.8, 0.95], "X": [0.0, 0.04, 0.09, 0.2, 0.5]}
        check_grads(lambda X: water_activity(X, 300.0, "table", tab), (0.07,), order=1, modes=["rev"],
                    rtol=1e-5, atol=1e-8)


# =============================================================================
# Constant rate
# =============================================================================


def _wet_bulb_independent(T, W, P=101325.0):
    """Wet bulb by brentq on the Lewis-relation balance, ASHRAE constants (psychrometrics' saturation_pressure)."""
    def res(Ts):
        t = Ts - K0
        ps = float(saturation_pressure(Ts))
        Ws = 0.621945 * ps / (P - ps)
        lam = 2501e3 + (1860.0 - 4186.0) * t
        return (T - Ts) * (1006.0 + W * 1860.0) - (Ws - W) * lam
    return optimize.brentq(res, 250.0, min(T, 372.0) + 1e-9, xtol=1e-12)


class TestConstantRate:
    def test_heat_transfer_form(self):
        T, W, h = 65.0 + K0, 0.01, 30.0
        Twb = _wet_bulb_independent(T, W)
        lam = 2501e3 + (1860.0 - 4186.0) * (Twb - K0)
        ref = h * (T - Twb) / lam
        assert float(constant_drying_rate(T, W, h)) == pytest.approx(ref, rel=1e-9)

    def test_hot_air_above_boiling_point(self):
        # a dryer's 100-200 C air: the wet-bulb Newton must not start from W_s(T) (negative above 100 C)
        for T in (373.15, 423.15, 473.15):
            Twb = float(wet_bulb_temperature(T, 0.01))
            assert Twb == pytest.approx(_wet_bulb_independent(T, 0.01), abs=1e-8)

    def test_mass_transfer_form(self):
        T, W = 60.0 + K0, 0.02
        Twb = _wet_bulb_independent(T, W)
        ps = float(saturation_pressure(Twb))
        Ws = 0.621945 * ps / (101325.0 - ps)
        assert float(constant_drying_rate(T, W, None, k_W=0.05)) == pytest.approx(0.05 * (Ws - W), rel=1e-9)

    def test_tray_correlation_is_its_formula(self):
        # the correlation itself is UNVERIFIED (typed from memory); only the unit handling is tested
        G = 1.5  # kg/m^2/s = 5400 kg/m^2/h
        assert float(heat_transfer_coefficient_tray(G)) == pytest.approx(0.0204 * 5400.0 ** 0.8, rel=1e-12)
        assert float(heat_transfer_coefficient_tray(G, "perpendicular")) == pytest.approx(1.17 * 5400.0 ** 0.37, rel=1e-12)
        with pytest.raises(ValueError):
            heat_transfer_coefficient_tray(G, "diagonal")

    def test_rate_gradients(self):
        check_grads(lambda T, W, h: constant_drying_rate(T, W, h), (330.0, 0.015, 30.0), order=1,
                    modes=["rev"], rtol=1e-5, atol=1e-10)


# =============================================================================
# Drying time
# =============================================================================


class TestDryingTime:
    @pytest.mark.parametrize("x1,x2", [
        (0.35, 0.10),     # X1 > Xc > X2: both periods
        (0.18, 0.08),     # X1 < Xc: falling only
        (0.35, 0.25),     # X2 > Xc: constant only
    ])
    def test_closed_forms(self, x1, x2):
        ref = closed_form(x1, x2, XC, XE, RC, LSA)
        got = float(drying_time(x1, x2, XC, XE, RC, LSA))
        assert got == pytest.approx(ref, rel=1e-7)

    def test_periods_split(self):
        t_c, t_f = drying_periods(X1, X2, XC, XE, RC, LSA)
        assert float(t_c) == pytest.approx(LSA * (X1 - XC) / RC, rel=1e-9)
        assert float(t_f) == pytest.approx(LSA * (XC - XE) / RC * math.log((XC - XE) / (X2 - XE)), rel=1e-9)
        # falling only: no constant period; constant only: no falling period
        t_c, t_f = drying_periods(0.18, 0.08, XC, XE, RC, LSA)
        assert float(t_c) == pytest.approx(0.0, abs=1e-6)
        t_c, t_f = drying_periods(0.35, 0.25, XC, XE, RC, LSA)
        assert float(t_f) == pytest.approx(0.0, abs=1e-6)

    def test_equals_numerical_integral_of_dX_over_R(self):
        # the oracle of the issue: closed form == numerical int dX/R of the SAME piecewise rate curve
        def R(x):
            return RC if x >= XC else RC * (x - XE) / (XC - XE)

        for x1, x2 in ((0.35, 0.10), (0.18, 0.08), (0.35, 0.25), (0.30, 0.06)):
            pts = [XC] if x2 < XC < x1 else None
            num, _ = integrate.quad(lambda x: 1.0 / R(x), x2, x1, points=pts, epsabs=0, epsrel=1e-12)
            assert float(drying_time(x1, x2, XC, XE, RC, LSA)) == pytest.approx(LSA * num, rel=1e-7)

    def test_rate_curve_integral_round_trip(self):
        # t = Ls int dX / R(X) with R from drying_rate_curve (the smooth curve, integrated numerically)
        xs = np.linspace(X2, X1, 20001)
        R = np.asarray(drying_rate_curve(xs, XC, XE, RC))
        num = LSA * np.trapezoid(1.0 / R, xs)
        assert num == pytest.approx(float(drying_time(X1, X2, XC, XE, RC, LSA)), rel=2e-3)

    def test_tabulated_curve_matches_linear(self):
        # the falling="integral" path with a table sampled from the linear curve (kink at X_c in the table)
        Xt = np.array([XE, 0.06, 0.08, 0.10, 0.14, 0.17, XC, 0.30, 0.40])
        Rt = np.where(Xt >= XC, RC, RC * (Xt - XE) / (XC - XE))
        for x1, x2 in ((X1, X2), (0.18, 0.08), (0.35, 0.25)):
            lin = float(drying_time(x1, x2, XC, XE, RC, LSA))
            tab = float(drying_time(x1, x2, XC, XE, RC, LSA, falling="integral", rate_curve=(Xt, Rt)))
            assert tab == pytest.approx(lin, rel=2e-4)

    def test_tabulated_nonlinear_curve(self):
        # a falling-rate curve that is concave: compare with scipy quad of the interpolated table
        Xt = np.array([0.06, 0.09, 0.12, 0.16, 0.20, 0.40])
        Rt = np.array([0.4, 0.7, 0.85, 0.95, 1.0, 1.0]) * RC
        got = float(drying_time(0.30, 0.08, 0.20, 0.05, RC, LSA, falling="integral", rate_curve=(Xt, Rt)))
        num, _ = integrate.quad(lambda x: 1.0 / np.interp(x, Xt, Rt), 0.08, 0.20, epsabs=0, epsrel=1e-12)
        ref = LSA * ((0.30 - 0.20) / RC + num)
        assert got == pytest.approx(ref, rel=1e-6)

    def test_smooth_switch_converges_as_sharpness_grows(self):
        # worst case: X1 exactly at X_c (the corner of the switch); error shrinks like 1/sharpness
        ref = closed_form(XC, X2, XC, XE, RC, LSA)
        errs = [abs(float(drying_time(XC, X2, XC, XE, RC, LSA, sharpness=s)) - ref) / ref
                for s in (20.0, 100.0, 500.0, 2500.0, 12500.0)]
        assert all(b < a for a, b in zip(errs, errs[1:]))
        assert errs[-1] < 5e-5
        # and the other corner: X2 exactly at X_c
        ref = closed_form(X1, XC, XC, XE, RC, LSA)
        errs = [abs(float(drying_time(X1, XC, XC, XE, RC, LSA, sharpness=s)) - ref) / ref
                for s in (20.0, 100.0, 500.0, 2500.0, 12500.0)]
        assert all(b < a for a, b in zip(errs, errs[1:]))
        assert errs[-1] < 5e-4

    def test_default_sharpness_error(self):
        # default sharpness: well under 0.5 % on ordinary cases, and under 0.5 % even at the corner here
        cases = [(0.35, 0.10), (0.30, 0.06), (0.18, 0.08), (0.35, 0.25), (0.45, 0.07)]
        for x1, x2 in cases:
            ref = closed_form(x1, x2, XC, XE, RC, LSA)
            assert abs(float(drying_time(x1, x2, XC, XE, RC, LSA)) - ref) / ref < 5e-5
        for x1, x2 in ((XC, X2), (X1, XC), (XC, 0.12)):
            ref = closed_form(x1, x2, XC, XE, RC, LSA)
            assert abs(float(drying_time(x1, x2, XC, XE, RC, LSA)) - ref) / ref < 5e-3

    def test_below_equilibrium_is_finite(self):
        t = float(drying_time(0.35, 0.04, XC, XE, RC, LSA))
        assert np.isfinite(t) and t > 1e5

    def test_input_checks(self):
        with pytest.raises(ValueError):
            drying_time(X1, X2, XC, XE, RC, LSA, falling="diffusion")
        with pytest.raises(ValueError):
            drying_time(X1, X2, XC, XE, RC, LSA, falling="integral")

    def test_gradients_of_drying_time(self):
        # R_c is scaled to O(1) so the finite-difference step is meaningful for every argument
        def f(x1, x2, xc, xe, rc4, lsa):
            return drying_time(x1, x2, xc, xe, rc4 * 1e-4, lsa) / 1e4

        rc4 = RC / 1e-4
        for x1, x2 in ((X1, X2), (0.18, 0.08), (0.35, 0.25)):   # both periods, falling only, constant only
            check_grads(f, (x1, x2, XC, XE, rc4, LSA), order=1, modes=["rev"], eps=1e-6, rtol=1e-5, atol=1e-7)

    def test_gradient_values_against_closed_form(self):
        g = jax.grad(drying_time, argnums=(0, 1, 2, 3))(X1, X2, XC, XE, RC, LSA)
        # analytic derivatives of t = LsA/Rc [ (X1 - Xc) + (Xc - Xe) ln((Xc - Xe)/(X2 - Xe)) ]
        ln = math.log((XC - XE) / (X2 - XE))
        ref = (LSA / RC,
               -LSA / RC * (XC - XE) / (X2 - XE),
               LSA / RC * ln,
               LSA / RC * (-ln - 1.0 + (XC - XE) / (X2 - XE)))
        for a, b in zip(g, ref):
            assert float(a) == pytest.approx(b, rel=1e-6)

    def test_gradient_wrt_Xc_exists_across_the_corner(self):
        # X_c scanned through X1 and X2: finite, no jump larger than the smoothing allows
        gs = [float(jax.grad(drying_time, argnums=2)(X1, X2, xc, XE, RC, LSA)) for xc in np.linspace(0.08, 0.40, 41)]
        assert np.all(np.isfinite(gs))

    def test_rate_curve_shape(self):
        x = jnp.array([0.4, 0.25, XC, 0.12, XE])
        R = np.asarray(drying_rate_curve(x, XC, XE, RC))
        assert R[0] == pytest.approx(RC, rel=1e-9) and R[1] == pytest.approx(RC, rel=1e-9)
        assert R[3] == pytest.approx(RC * (0.12 - XE) / (XC - XE), rel=1e-9)
        assert R[4] < 0.01 * RC
        assert np.all(np.diff(R) <= 1e-18)


# =============================================================================
# The Dryer unit
# =============================================================================


def make_air(T_C=100.0, W=0.01, G=10.0):
    return air_stream_from_state(state(T=T_C + K0, W=W), G)


def make_wet(m_s=0.5, X=0.30, T=300.0):
    return {"F_solid": m_s, "F_H2O": m_s * X / MW_WATER, "T": T, "P": 101325.0}


def stream_balances(wet, dry, fines, air_in, air_out, cps=1500.0, cpw=4186.0, loss=0.0):
    """Water (kg/s) and enthalpy (W) imbalance from the streams, with plain ASHRAE / constant-cp formulas."""
    def air(s):
        G = (s["F_air"]) * MW_DRY_AIR
        W = s["F_H2O"] * MW_WATER / G
        t = s["T"] - K0
        return G, W, G * (1006.0 * t + W * (2501e3 + 1860.0 * t))

    def sol(s):
        ms = s["F_solid"] * 1.0
        mw = s["F_H2O"] * MW_WATER
        return mw, (ms * cps + mw * cpw) * (s["T"] - K0)

    G_i, W_i, H_i = air(air_in)
    G_o, W_o, H_o = air(air_out)
    outs = [dry] + ([fines] if fines is not None else [])
    w_in = G_i * W_i + sol(wet)[0]
    w_out = G_o * W_o + sum(sol(s)[0] for s in outs)
    e_in = H_i + sol(wet)[1]
    e_out = H_o + sum(sol(s)[1] for s in outs) + loss
    return w_out - w_in, e_out - e_in, e_in


class TestContinuousDryer:
    def params(self, **kw):
        base = dict(mode="continuous", X_c=0.15, X_eq=0.02, product_moisture=0.05, h=60.0,
                    solids_per_area=10.0, max_solids_T=340.0)
        base.update(kw)
        return DryerParams(**base)

    @pytest.mark.parametrize("flow", ["countercurrent", "cocurrent"])
    def test_balances_close(self, flow):
        wet, air = make_wet(), make_air()
        dry, air_out, info = Dryer(self.params(flow=flow))(wet, air)
        dw, de, scale = stream_balances(wet, dry, None, air, air_out)
        assert abs(dw) < 1e-10 * float(wet["F_H2O"] * MW_WATER)
        assert abs(de) < 1e-10 * scale
        assert abs(float(info["water_balance_residual"])) < 1e-12
        assert abs(float(info["energy_balance_residual"])) < 1e-10 * scale
        # the evaporation is the moisture drop of the solids
        assert float(info["evaporation"]) == pytest.approx(0.5 * (info["X_in"] - 0.05), rel=1e-12)

    @pytest.mark.parametrize("flow", ["countercurrent", "cocurrent"])
    def test_balances_with_loss_and_entrainment(self, flow):
        wet, air = make_wet(), make_air()
        dry, air_out, info = Dryer(self.params(flow=flow, adiabatic=False, heat_loss=20e3, entrainment=0.05))(wet, air)
        dw, de, scale = stream_balances(wet, dry, info["fines"], air, air_out, loss=20e3)
        assert abs(dw) < 1e-10 * float(wet["F_H2O"] * MW_WATER)
        assert abs(de) < 1e-10 * scale
        # fines are 5 % of the dry solids, at the product moisture and temperature
        assert float(info["fines"]["F_solid"]) == pytest.approx(0.05 * 0.5, rel=1e-12)
        assert float(dry["F_solid"]) == pytest.approx(0.95 * 0.5, rel=1e-12)
        assert float(info["fines"]["T"]) == float(dry["T"])
        # the loss cools the exhaust relative to the adiabatic dryer
        _, ao_ad, _ = Dryer(self.params(flow=flow))(wet, air)
        assert float(air_out["T"]) < float(ao_ad["T"])
        # an air-side heat loss left in `adiabatic=True` is ignored
        _, ao_ign, _ = Dryer(self.params(flow=flow, heat_loss=20e3))(wet, air)
        assert float(ao_ign["T"]) == float(ao_ad["T"])

    def test_independent_countercurrent_hand_calculation(self):
        # solids leave at the hot end: T_s = T_a - (T_a - T_wb) (X_out - X*)/(X_c - X*); then the energy balance
        wet, air = make_wet(), make_air()
        _, air_out, info = Dryer(self.params())(wet, air)
        T_a, W_a, G = 100.0 + K0, 0.01, 10.0
        Twb = _wet_bulb_independent(T_a, W_a)
        f2 = (0.05 - 0.02) / (0.15 - 0.02)
        Ts_out = T_a - (T_a - Twb) * f2
        # soft-clip offset of the rate fraction is O(1/sharpness): allow it
        assert float(info["T_solids_out"]) == pytest.approx(Ts_out, abs=0.1)
        Xin = 0.5 * 0.3 / 0.5 / 1.0 * 0 + float(info["X_in"])
        W_out = W_a + 0.5 * (Xin - 0.05) / G
        assert float(info["W_out"]) == pytest.approx(W_out, rel=1e-12)
        hs_in = (1500.0 + Xin * 4186.0) * (300.0 - K0)
        hs_out = (1500.0 + 0.05 * 4186.0) * (float(info["T_solids_out"]) - K0)
        h_in = 1006.0 * (T_a - K0) + W_a * (2501e3 + 1860.0 * (T_a - K0))
        h_out = h_in + 0.5 * (hs_in - hs_out) / G
        t_out = (h_out - W_out * 2501e3) / (1006.0 + 1860.0 * W_out)
        assert float(info["T_air_out"]) == pytest.approx(t_out + K0, abs=1e-8)

    def test_cocurrent_vs_countercurrent(self):
        wet, air = make_wet(), make_air()
        _, ao_cc, i_cc = Dryer(self.params(flow="countercurrent"))(wet, air)
        _, ao_co, i_co = Dryer(self.params(flow="cocurrent"))(wet, air)
        # the countercurrent solids meet the hot air: hotter product, a bigger quality problem
        assert float(i_cc["T_solids_out"]) > float(i_co["T_solids_out"])
        assert float(i_cc["solids_T_margin"]) < float(i_co["solids_T_margin"])
        # ... but the wet solids meet the cold, humid air: a slower constant rate, a longer residence time
        assert float(i_cc["R_c"]) < float(i_co["R_c"])
        assert float(i_cc["residence_time_required"]) > float(i_co["residence_time_required"])
        # cocurrent product cannot be hotter than the exhaust air
        assert float(i_co["T_solids_out"]) <= float(i_co["T_air_out"]) + 1e-9

    def test_cocurrent_root_residual(self):
        # the bisection result satisfies the energy balance to rounding (checked from the streams above) and
        # the exhaust temperature is the T that the closed forms in the info reproduce
        wet, air = make_wet(), make_air()
        _, ao, info = Dryer(self.params(flow="cocurrent"))(wet, air)
        assert float(ao["T"]) == float(info["T_air_out"])
        assert float(info["dew_point_margin"]) > 0.0

    def test_not_supersaturated_and_flag(self):
        wet = make_wet()
        _, _, ok = Dryer(self.params())(wet, make_air(G=10.0))
        assert not bool(ok["supersaturated"])
        assert float(ok["dew_point_margin"]) > 0.0
        # starve the dryer of air: the exhaust needs to hold more water than it can
        _, ao, bad = Dryer(self.params())(wet, make_air(T_C=60.0, G=5.0))
        assert bool(bad["supersaturated"])
        assert float(bad["dew_point_margin"]) < 0.0

    def test_margins_and_energy(self):
        wet, air = make_wet(), make_air()
        _, _, info = Dryer(self.params())(wet, air)
        assert float(info["solids_T_margin"]) == pytest.approx(340.0 - float(info["T_solids_out"]), rel=1e-12)
        Q = 10.0 * ((1006.0 + 0.01 * 1860.0) * (100.0 + K0 - 293.15))
        assert float(info["Q_air"]) == pytest.approx(Q, rel=1e-12)
        assert float(info["energy_per_kg"]) == pytest.approx(Q / float(info["evaporation"]), rel=1e-12)
        assert float(info["thermal_efficiency"]) == pytest.approx(
            float(info["evaporation"]) * 2.4537e6 / Q, rel=2e-3)   # lambda(20 C) = 2453.7 kJ/kg (ideal-gas form)
        _, _, nolimit = Dryer(self.params(max_solids_T=None))(wet, air)
        assert np.isinf(float(nolimit["solids_T_margin"]))

    def test_isotherm_sets_equilibrium_from_air_rh(self):
        wet, air = make_wet(), make_air(T_C=80.0, W=0.02)
        iso = [0.07, 10.0, 0.9]
        d = Dryer(self.params(X_eq=None, isotherm_model="gab", isotherm_params=iso))
        _, _, info = d(wet, air)
        RH = float(state(T=80.0 + K0, W=0.02)["RH"])
        Xm, C, K = iso
        ref = Xm * C * K * RH / ((1 - K * RH) * (1 - K * RH + C * K * RH))
        assert float(info["X_eq"]) == pytest.approx(ref, rel=1e-12)

    def test_residence_time_mode_round_trip(self):
        wet, air = make_wet(), make_air()
        for flow in ("countercurrent", "cocurrent"):
            _, _, spec = Dryer(self.params(flow=flow))(wet, air)
            t = float(spec["residence_time_required"])
            _, _, back = Dryer(self.params(flow=flow, product_moisture=None, residence_time=t))(wet, air)
            assert float(back["X_out"]) == pytest.approx(0.05, abs=1e-8)
            assert float(back["T_air_out"]) == pytest.approx(float(spec["T_air_out"]), abs=1e-5)

    def test_residence_time_matches_drying_time(self):
        wet, air = make_wet(), make_air()
        _, _, info = Dryer(self.params(flow="cocurrent"))(wet, air)
        # cocurrent: the constant rate is evaluated in the inlet air
        Twb = _wet_bulb_independent(100.0 + K0, 0.01)
        Rc = 60.0 * (100.0 + K0 - Twb) / (2501e3 + (1860.0 - 4186.0) * (Twb - K0))
        t = closed_form(float(info["X_in"]), 0.05, 0.15, 0.02, Rc, 10.0)
        assert float(info["residence_time_required"]) == pytest.approx(t, rel=1e-6)

    def test_extra_species_pass_through(self):
        air = make_air()
        air["F_CO2"] = 0.0   # counted as dry air by dry_air_flow, but must survive
        wet = make_wet()
        _, ao, _ = Dryer(self.params())(wet, air)
        assert "F_CO2" in ao

    def test_missing_spec_raises(self):
        with pytest.raises(ValueError):
            Dryer(self.params(product_moisture=None))(make_wet(), make_air())
        # a residence time needs the loading; the time needs a constant rate
        with pytest.raises(ValueError):
            Dryer(DryerParams(mode="continuous", X_c=0.15, product_moisture=None, residence_time=100.0))(
                make_wet(), make_air())
        with pytest.raises(ValueError):
            Dryer(DryerParams(mode="continuous", X_c=0.15, product_moisture=0.05, solids_per_area=10.0))(
                make_wet(), make_air())
        # without solids_per_area no time is asked for, so no rate is needed
        _, _, info = Dryer(DryerParams(mode="continuous", X_c=0.15, product_moisture=0.05))(make_wet(), make_air())
        assert "residence_time_required" not in info

    def test_gradient_of_outlet_T_wrt_air_flow(self):
        wet = make_wet()

        base = make_air(G=1.0)

        def f(G, flow):
            air = {"F_H2O": base["F_H2O"] * G, "F_air": base["F_air"] * G, "T": base["T"], "P": base["P"]}
            _, ao, _ = Dryer(self.params(flow=flow))(wet, air)
            return ao["T"]

        for flow in ("countercurrent", "cocurrent"):
            check_grads(lambda G: f(G, flow), (10.0,), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)
            # a wetter, hotter exhaust with more air: dT_out/dG > 0 (less cooling per kg of air)
            assert float(jax.grad(lambda G: f(G, flow))(10.0)) > 0.0

    def test_gradients_wrt_inlet_state_and_X_c(self):
        wet = make_wet()

        def f(T, W, xc, flow):
            air = air_stream_from_state(state(T=T, W=W), 10.0)
            _, ao, info = Dryer(self.params(flow=flow, X_c=xc))(wet, air)
            return jnp.stack([ao["T"], info["T_solids_out"], info["residence_time_required"]])

        for flow in ("countercurrent", "cocurrent"):
            check_grads(lambda T, W, xc: f(T, W, xc, flow), (100.0 + K0, 0.01, 0.15), order=1,
                        modes=["rev"], rtol=1e-5, atol=1e-6)

    def test_gradient_example_optimal_inlet_T(self):
        """Design with gradients: the inlet-air T that minimises energy per kg of water.

        A countercurrent dryer of fixed size (residence time 4000 s) must dry the solids to the spec. For each
        inlet T the air flow that just does it is solved for (implicit gradient); hotter air needs less of it, so
        the energy per kg of water falls with T until the product reaches ``max_solids_T``, which enters the
        objective as a soft constraint. The independent answer is the root of T_solids_out(T) = max_solids_T.
        """
        from difflow.staged import _solve_monotone

        wet = make_wet()
        T_max, t_avail = 350.0, 4000.0
        unit = Dryer(self.params(flow="countercurrent", max_solids_T=T_max))

        def run(T, G):
            return unit(wet, air_stream_from_state(state(T=T, W=0.01), G))[2]

        def air_flow(T):
            return _solve_monotone(lambda G: run(T, G)["residence_time_required"] - t_avail, 4.0, 60.0)

        def objective(T):
            info = run(T, air_flow(T))
            return info["energy_per_kg"] / 1e6 + 5.0 * jax.nn.softplus((info["T_solids_out"] - T_max) / 0.1)

        ev = jax.jit(lambda T: (lambda i: (i["energy_per_kg"], i["T_solids_out"], i["residence_time_required"]))(
            run(T, air_flow(T))))
        # unconstrained, the energy per kg falls with the inlet temperature
        Ts = [350.0, 370.0, 390.0, 410.0]
        out = [ev(T) for T in Ts]
        E = np.array([float(o[0]) for o in out])
        assert np.all(np.diff(E) < 0.0)
        assert all(float(o[2]) == pytest.approx(t_avail, rel=1e-8) for o in out)   # the size constraint is met
        # so the optimum sits on the temperature limit: independent root of T_s(T) = T_max
        root = optimize.brentq(lambda T: float(ev(T)[1]) - T_max, 350.0, 410.0, xtol=1e-10)
        val_grad = jax.jit(jax.value_and_grad(objective))

        def fun(x):
            v, g = val_grad(x[0])
            return float(v), np.array([float(g)])

        res = optimize.minimize(fun, x0=[405.0], jac=True, method="L-BFGS-B", bounds=[(350.0, 420.0)])
        T_opt = float(res.x[0])
        assert abs(T_opt - root) < 1.0
        assert float(ev(T_opt)[1]) < T_max + 1.0
        assert float(ev(T_opt)[0]) < float(ev(T_opt - 15.0)[0])          # beats a cooler inlet on energy
        assert float(ev(T_opt + 15.0)[1]) > T_max + 5.0                  # a hotter inlet breaks the limit


class TestBatchDryer:
    def params(self, **kw):
        base = dict(mode="batch", X_c=0.20, X_eq=0.05, product_moisture=0.10, air_mass_velocity=1.0,
                    load=12.5, area=0.5, max_solids_T=330.0)
        base.update(kw)
        return DryerParams(**base)

    def test_time_matches_closed_form(self):
        T, W = 70.0 + K0, 0.01
        air = make_air(70.0, W, G=0.5)
        wet = make_wet(0.01, 0.35, 295.0)
        _, _, info = Dryer(self.params())(wet, air)
        Twb = _wet_bulb_independent(T, W)
        h = 0.0204 * (3600.0 * 1.0) ** 0.8
        Rc = h * (T - Twb) / (2501e3 + (1860.0 - 4186.0) * (Twb - K0))
        LsA = 12.5 / 0.5
        Xin = float(info["X_in"])
        ref = closed_form(Xin, 0.10, 0.20, 0.05, Rc, LsA)
        assert float(info["drying_time"]) == pytest.approx(ref, rel=1e-6)
        assert float(info["R_c"]) == pytest.approx(Rc, rel=1e-9)
        assert float(info["t_constant"]) == pytest.approx(LsA * (Xin - 0.20) / Rc, rel=1e-6)

    def test_R_c_and_h_take_precedence(self):
        air, wet = make_air(70.0, 0.01, 0.5), make_wet(0.01, 0.35, 295.0)
        _, _, a = Dryer(self.params(R_c=3e-4))(wet, air)
        assert float(a["R_c"]) == 3e-4
        _, _, b = Dryer(self.params(h=40.0))(wet, air)
        Twb = _wet_bulb_independent(70.0 + K0, 0.01)
        assert float(b["R_c"]) == pytest.approx(40.0 * (70.0 + K0 - Twb) / (2501e3 + (1860.0 - 4186.0) * (Twb - K0)),
                                                rel=1e-9)

    def test_air_balance_and_bookkeeping(self):
        air, wet = make_air(70.0, 0.01, 0.5), make_wet(0.01, 0.35, 295.0)
        dry, ao, info = Dryer(self.params())(wet, air)
        # over the drying step: solids processed at load / t_dry
        t = float(info["drying_time"])
        rate = 12.5 / t
        Xin = float(info["X_in"])
        assert float(info["evaporation"]) == pytest.approx(rate * (Xin - 0.10), rel=1e-12)
        assert float(info["W_out"]) == pytest.approx(0.01 + rate * (Xin - 0.10) / 0.5, rel=1e-12)
        assert abs(float(info["water_balance_residual"])) < 1e-12
        assert abs(float(info["energy_balance_residual"])) < 1e-9
        # the solids stream carries the throughput through; product moisture and area
        assert float(dry["F_solid"]) == pytest.approx(0.01)
        assert float(solids_moisture(dry)) == pytest.approx(0.10, rel=1e-12)
        assert float(info["area_required"]) == pytest.approx(0.01 * t / 25.0, rel=1e-12)
        assert float(info["cycle_time"]) == pytest.approx(t)
        _, _, t2 = Dryer(self.params(turnaround_time=1800.0))(wet, air)
        assert float(t2["cycle_time"]) == pytest.approx(t + 1800.0)
        # product temperature: between the wet bulb and the air; under the limit here
        Twb = _wet_bulb_independent(70.0 + K0, 0.01)
        assert Twb < float(info["T_solids_out"]) < 70.0 + K0
        assert float(info["solids_T_margin"]) == pytest.approx(330.0 - float(info["T_solids_out"]))
        assert float(info["dew_point_margin"]) > 0.0 and not bool(info["supersaturated"])

    def test_residence_time_mode_round_trip(self):
        air, wet = make_air(70.0, 0.01, 0.5), make_wet(0.01, 0.35, 295.0)
        _, _, spec = Dryer(self.params())(wet, air)
        t = float(spec["drying_time"])
        dry, _, back = Dryer(self.params(product_moisture=None, residence_time=t))(wet, air)
        assert float(back["X_out"]) == pytest.approx(0.10, abs=1e-9)
        # half the time: wetter product, but never wetter than the feed
        _, _, half = Dryer(self.params(product_moisture=None, residence_time=0.5 * t))(wet, air)
        assert 0.10 < float(half["X_out"]) < float(spec["X_in"])

    def test_isotherm_and_tabulated_falling_rate(self):
        air, wet = make_air(70.0, 0.01, 0.5), make_wet(0.01, 0.35, 295.0)
        Xt, Rt = [0.06, 0.09, 0.12, 0.16, 0.20, 0.40], [0.4, 0.7, 0.85, 0.95, 1.0, 1.0]
        d = Dryer(self.params(falling="integral", rate_curve=[Xt, Rt]))
        _, _, tab = d(wet, air)
        _, _, lin = Dryer(self.params())(wet, air)
        Rc = float(tab["R_c"])
        num, _ = integrate.quad(lambda x: 1.0 / (Rc * np.interp(x, Xt, Rt)), 0.10, 0.20, epsabs=0, epsrel=1e-12)
        ref = 25.0 * ((float(tab["X_in"]) - 0.20) / Rc + num)
        assert float(tab["drying_time"]) == pytest.approx(ref, rel=1e-6)
        assert float(tab["drying_time"]) != pytest.approx(float(lin["drying_time"]), rel=1e-2)

    def test_needs_inputs(self):
        air, wet = make_air(70.0, 0.01, 0.5), make_wet(0.01, 0.35, 295.0)
        with pytest.raises(ValueError):
            Dryer(self.params(load=None))(wet, air)
        with pytest.raises(ValueError):
            Dryer(self.params(air_mass_velocity=None))(wet, air)
        with pytest.raises(ValueError):
            Dryer(self.params(product_moisture=None))(wet, air)

    def test_gradients_of_time_and_energy(self):
        wet = make_wet(0.01, 0.35, 295.0)

        def f(T, W, G_mass, xc, load):
            air = air_stream_from_state(state(T=T, W=W), 0.5)
            _, _, info = Dryer(self.params(X_c=xc, air_mass_velocity=G_mass, load=load))(wet, air)
            return jnp.stack([info["drying_time"], info["energy_per_kg"], info["T_air_out"],
                              info["T_solids_out"]])

        x0 = (70.0 + K0, 0.01, 1.0, 0.20, 12.5)
        check_grads(f, x0, order=1, modes=["rev"], rtol=1e-5, atol=1e-6)
        # signs: hotter or faster air dries faster; a bigger load takes longer
        J = jax.jacobian(f, argnums=(0, 2, 4))(*x0)
        assert float(J[0][0]) < 0.0 and float(J[1][0]) < 0.0 and float(J[2][0]) > 0.0


class TestRegistration:
    def test_catalog_and_symbols(self):
        from difflow.catalog import describe_operation

        spec = describe_operation("Dryer")
        assert spec.category == "drying"
        assert spec.ports.n_outlets == 2
        assert spec.ports.outlet_roles == ["dry_solids", "air_out"]
        assert spec.is_declarative

    def test_params_validation(self):
        with pytest.raises(ValueError):
            DryerParams(mode="spray")
        with pytest.raises(ValueError):
            DryerParams(flow="crossflow")
        with pytest.raises(ValueError):
            DryerParams(falling="integral")
