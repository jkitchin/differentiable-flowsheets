"""Tests for the CoolingTower unit and the Merkel library functions.

Oracles:

* wex18-1 (10 MW, 35 -> 27 C at a 22 C wet bulb): the issue's own illustrative
  Merkel numbers (about 1.24, 1.39, 1.59, 1.89 at L/G = 0.8, 1.0, 1.2, 1.4),
  computed there with ideal-gas humid air, IAPWS-IF97 saturation pressure, inlet
  air saturated at 22 C and the trapezoid rule. They are NOT published values.
  Here they are reproduced to 1 % and, separately, against an independent NumPy
  trapezoid integration of this module's own h_s on a fine grid (1e-4).
* The CTI four-point method against fine Gauss-Legendre quadrature.
* Evaporation against the ~0.00085 x 1.8 x range rule of thumb (unverified, from
  recollection): only a factor-of-1.3 sanity band, not a calibration.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import (
    CentrifugalPump,
    CentrifugalPumpParams,
    CoolingTower,
    CoolingTowerParams,
    CounterCurrentHX,
    Flowsheet,
    HeatExchangerParams,
    Unit,
    make_stream,
)
from difflow.psychrometrics import (
    MW_WATER,
    air_stream_from_state,
    humid_enthalpy,
    saturation_enthalpy,
    saturation_humidity_ratio,
    state,
    state_from_air_stream,
)
from difflow.units.cooling_tower import (
    maximum_L_over_G,
    merkel_min_driving_force,
    merkel_number,
    saturation_temperature_from_enthalpy,
)

K = 273.15
T_IN, T_OUT, T_WB = 35.0 + K, 27.0 + K, 22.0 + K
P = 101325.0
L = 10e6 / (4186.0 * 8.0)            # 298.5 kg/s (wex18-1)
H_AIN = float(saturation_enthalpy(T_WB))

ISSUE_NTU = {0.8: 1.24, 1.0: 1.39, 1.2: 1.59, 1.4: 1.89}


def water(T=T_IN, flow_kg=L, P_=3e5):
    return make_stream({"H2O": flow_kg / MW_WATER, "air": 0.0}, T, P_)


def tower(**kw):
    base = dict(mode="design", L_over_G=1.2, T_out=T_OUT)
    base.update(kw)
    return CoolingTower(CoolingTowerParams(**base))


def numpy_trapezoid_merkel(lg, n=4001, T_in=T_IN, T_out=T_OUT, h_in=H_AIN):
    x = np.linspace(0.0, 1.0, n)
    T = T_out + x * (T_in - T_out)
    hs = np.asarray(saturation_enthalpy(jnp.asarray(T)))
    ha = h_in + lg * 4186.0 * (T_in - T_out) * x
    return float(np.trapezoid(4186.0 * (T_in - T_out) / (hs - ha), x))


class TestMerkelNumber:
    @pytest.mark.parametrize("lg", sorted(ISSUE_NTU))
    def test_wex18_1_matches_the_issue_illustration(self, lg):
        got = float(merkel_number(T_IN, T_OUT, H_AIN, lg, quadrature="gauss", n_quad=16))
        assert got == pytest.approx(ISSUE_NTU[lg], rel=0.01)

    @pytest.mark.parametrize("lg", sorted(ISSUE_NTU))
    def test_against_an_independent_numpy_trapezoid(self, lg):
        got = float(merkel_number(T_IN, T_OUT, H_AIN, lg, quadrature="gauss", n_quad=16))
        assert got == pytest.approx(numpy_trapezoid_merkel(lg), rel=1e-4)

    @pytest.mark.parametrize("lg", sorted(ISSUE_NTU))
    def test_cti_four_point_agrees_with_fine_quadrature(self, lg):
        cti = float(merkel_number(T_IN, T_OUT, H_AIN, lg, quadrature="cti"))
        fine = float(merkel_number(T_IN, T_OUT, H_AIN, lg, quadrature="gauss", n_quad=24))
        assert cti == pytest.approx(fine, rel=0.01)
        assert cti == pytest.approx(ISSUE_NTU[lg], rel=0.01)

    def test_cti_is_exactly_the_four_point_formula(self):
        lg = 1.0
        x = np.array([0.1, 0.4, 0.6, 0.9])
        T = T_OUT + x * 8.0
        hs = np.asarray(saturation_enthalpy(jnp.asarray(T)))
        ha = H_AIN + lg * 4186.0 * 8.0 * x
        by_hand = 4186.0 * 8.0 / 4.0 * np.sum(1.0 / (hs - ha))
        assert float(merkel_number(T_IN, T_OUT, H_AIN, lg)) == pytest.approx(by_hand, rel=1e-12)

    def test_gauss_converges_with_points(self):
        ref = float(merkel_number(T_IN, T_OUT, H_AIN, 1.4, quadrature="gauss", n_quad=40))
        errs = [abs(float(merkel_number(T_IN, T_OUT, H_AIN, 1.4, quadrature="gauss", n_quad=n)) - ref)
                for n in (2, 4, 8)]
        assert errs[0] > errs[1] > errs[2]

    def test_zero_range_and_inverted_range_are_masked(self):
        assert float(merkel_number(T_IN, T_IN, H_AIN, 1.0)) == np.inf
        assert float(merkel_number(T_OUT, T_IN, H_AIN, 1.0)) == np.inf

    def test_more_cooling_needs_more_fill(self):
        n = [float(merkel_number(T_IN, T_IN - r, H_AIN, 1.0)) for r in (4.0, 6.0, 8.0, 10.0)]
        assert all(a < b for a, b in zip(n, n[1:]))

    def test_unknown_quadrature(self):
        with pytest.raises(ValueError):
            merkel_number(T_IN, T_OUT, H_AIN, 1.0, quadrature="simpson")


class TestPinch:
    def test_pinch_limit_is_where_the_driving_force_vanishes(self):
        lg_max = float(maximum_L_over_G(T_IN, T_OUT, H_AIN))
        assert 1.4 < lg_max < 3.0
        assert float(merkel_min_driving_force(T_IN, T_OUT, H_AIN, lg_max)) == pytest.approx(0.0, abs=1e-3)
        assert float(merkel_min_driving_force(T_IN, T_OUT, H_AIN, 0.99 * lg_max)) > 0
        assert np.isfinite(float(merkel_number(T_IN, T_OUT, H_AIN, 0.99 * lg_max)))
        assert float(merkel_number(T_IN, T_OUT, H_AIN, 1.01 * lg_max)) == np.inf

    def test_merkel_number_diverges_towards_the_pinch(self):
        lg_max = float(maximum_L_over_G(T_IN, T_OUT, H_AIN))
        n = [float(merkel_number(T_IN, T_OUT, H_AIN, f * lg_max, quadrature="gauss", n_quad=16))
             for f in (0.5, 0.8, 0.95, 0.99)]
        assert all(a < b for a, b in zip(n, n[1:])) and n[-1] > 3 * n[0]

    def test_unit_above_the_pinch_is_infeasible_with_no_nans(self):
        lg_max = float(maximum_L_over_G(T_IN, T_OUT, H_AIN))
        wo, ao, info = tower(L_over_G=1.3 * lg_max)(water(), T_wb=T_WB)
        assert not bool(info["feasible"])
        assert float(info["KaV_L"]) == np.inf
        for name, v in info.items():
            if name in ("KaV_L", "air_out"):
                continue
            assert bool(jnp.all(jnp.isfinite(v))), name
        assert float(wo["T"]) == pytest.approx(T_OUT)
        for v in jax.tree_util.tree_leaves((wo, ao, info["air_out"])):
            assert bool(jnp.all(jnp.isfinite(v)))

    def test_gradients_stay_finite_when_infeasible(self):
        lg_max = float(maximum_L_over_G(T_IN, T_OUT, H_AIN))
        f = lambda lg: tower(L_over_G=lg)(water(), T_wb=T_WB)[2]["min_driving_force"]
        g = float(jax.grad(f)(1.3 * lg_max))
        assert np.isfinite(g) and g < 0
        h = lambda lg: jnp.where(jnp.isfinite(merkel_number(T_IN, T_OUT, H_AIN, lg)),
                                 merkel_number(T_IN, T_OUT, H_AIN, lg), 0.0)
        assert np.isfinite(float(jax.grad(h)(1.3 * lg_max)))

    def test_wet_bulb_above_the_requested_outlet_is_infeasible(self):
        _, _, info = tower(T_out=T_WB - 1.0)(water(), T_wb=T_WB)
        assert not bool(info["feasible"])


class TestDesignMode:
    def test_wex18_1_design(self):
        wo, ao, info = tower(L_over_G=1.2)(water(), T_wb=T_WB)
        assert bool(info["feasible"])
        assert float(info["KaV_L"]) == pytest.approx(1.59, rel=0.01)
        assert float(info["range"]) == pytest.approx(8.0)
        assert float(info["approach"]) == pytest.approx(5.0)
        assert float(info["effectiveness"]) == pytest.approx(8.0 / 13.0)
        assert float(info["Q"]) == pytest.approx(10e6, rel=2e-3)
        assert float(info["L"]) == pytest.approx(L)
        assert float(info["G"]) == pytest.approx(L / 1.2)
        assert float(info["L_over_G_max"]) > 1.2

    def test_approach_replaces_T_out(self):
        a = tower(T_out=None, approach=5.0)(water(), T_wb=T_WB)[2]
        b = tower()(water(), T_wb=T_WB)[2]
        assert float(a["KaV_L"]) == pytest.approx(float(b["KaV_L"]), rel=1e-12)

    def test_heat_balance_across_the_tower(self):
        wo, ao, info = tower()(water(), T_wb=T_WB)
        # water side: L cp (T_in - T_out); air side: G (h_out - h_in); evaporated
        # water is neglected in the Merkel air balance, so they agree to ~1 %
        s_in = state(T=T_WB, RH=1.0)
        s_out = state_from_air_stream(ao)
        dh_air = float(info["G"]) * (float(s_out["h"]) - float(s_in["h"]))
        assert dh_air == pytest.approx(float(info["Q"]), rel=2e-2)
        assert float(s_out["RH"]) == pytest.approx(1.0, rel=1e-8)          # Merkel: leaves saturated
        assert float(ao["T"]) < T_IN                                       # and cooler than the hot water

    def test_air_state_is_read_from_a_stream(self):
        s = state(T=T_WB + 4.0, T_wb=T_WB)
        air = air_stream_from_state(s, L / 1.2)
        _, _, a = tower(L_over_G=None)(water(), air)
        _, _, b = tower()(water(), T_wb=T_WB)
        assert float(a["T_wb"]) == pytest.approx(T_WB, abs=1e-9)
        assert float(a["L_over_G"]) == pytest.approx(1.2, rel=1e-9)
        # the real inlet enthalpy is a little BELOW saturation at the wet bulb
        # (h = h_s(T*) - (W_s* - W) h_liq): more driving force, a smaller tower
        assert float(a["KaV_L"]) == pytest.approx(float(b["KaV_L"]), rel=0.02)
        assert float(a["KaV_L"]) < float(b["KaV_L"])

    def test_needs_air_or_wet_bulb(self):
        with pytest.raises(ValueError, match="air_in or T_wb"):
            tower()(water())
        with pytest.raises(ValueError, match="L_over_G"):
            tower(L_over_G=None)(water(), T_wb=T_WB)
        with pytest.raises(ValueError, match="T_out or approach"):
            tower(T_out=None)(water(), T_wb=T_WB)

    def test_sizing_with_ka_gives_the_fill_volume(self):
        _, _, info = tower(Ka=2.0)(water(), T_wb=T_WB)
        assert float(info["V"]) == pytest.approx(float(info["KaV_L"]) * L / 2.0, rel=1e-12)


class TestLosses:
    def test_evaporation_matches_the_water_balance_of_the_air(self):
        wo, ao, info = tower()(water(), T_wb=T_WB)
        gain = (float(ao["F_H2O"]) * MW_WATER) - float(saturation_humidity_ratio(T_WB)) * float(info["G"])
        assert float(info["evaporation"]) == pytest.approx(gain, rel=1e-10)

    def test_evaporation_is_about_a_percent_of_the_flow(self):
        info = tower()(water(), T_wb=T_WB)[2]
        rule = 0.00085 * 1.8 * 8.0          # rule of thumb (unverified, recalled): fraction of L
        assert float(info["evaporation_fraction"]) == pytest.approx(rule, rel=0.3)
        assert 0.005 < float(info["evaporation_fraction"]) < 0.02

    def test_makeup_is_evaporation_plus_drift_plus_blowdown(self):
        info = tower(drift_fraction=0.0005, cycles_of_concentration=5.0)(water(), T_wb=T_WB)[2]
        E, D, B, M = (float(info[k]) for k in ("evaporation", "drift", "blowdown", "makeup"))
        assert D == pytest.approx(0.0005 * L)
        assert M == pytest.approx(E + D + B, rel=1e-12)
        # cycles of concentration: solids in = solids out  =>  M = C (D + B)
        assert M == pytest.approx(5.0 * (D + B), rel=1e-12)

    def test_outlet_flow_bookkeeping(self):
        with_makeup = tower()(water(), T_wb=T_WB)[0]
        short = tower(makeup_in_outlet=False)(water(), T_wb=T_WB)
        info = short[2]
        assert float(with_makeup["F_H2O"]) == pytest.approx(L / MW_WATER, rel=1e-13)
        assert float(short[0]["F_H2O"]) * MW_WATER == pytest.approx(
            L - float(info["evaporation"]) - float(info["drift"]), rel=1e-12)
        assert float(with_makeup["P"]) == P            # basin at tower pressure

    def test_total_water_is_conserved_with_losses_and_makeup(self):
        wo, ao, info = tower(makeup_in_outlet=False)(water(), T_wb=T_WB)
        water_in = L + float(info["makeup"])
        water_out = (float(wo["F_H2O"]) * MW_WATER + float(info["makeup"]) - float(info["blowdown"])
                     + float(info["evaporation"]) + float(info["drift"]) - float(info["evaporation"]) - float(info["drift"]))
        # basin: cold water in + makeup - blowdown = circulating water
        assert water_out == pytest.approx(L, rel=1e-12)
        assert water_in > L


class TestRatingMode:
    def test_design_rating_round_trip(self):
        for lg in (0.8, 1.2, 1.6):
            d = tower(L_over_G=lg)(water(), T_wb=T_WB)[2]
            r = tower(mode="rating", L_over_G=lg, T_out=None, KaV_L=float(d["KaV_L"]))(water(), T_wb=T_WB)[2]
            assert float(r["T_out"]) == pytest.approx(T_OUT, abs=1e-8)
            assert bool(r["converged"]) and bool(r["feasible"])

    def test_round_trip_with_the_gauss_quadrature(self):
        d = tower(quadrature="gauss", n_quad=12)(water(), T_wb=T_WB)[2]
        r = tower(mode="rating", quadrature="gauss", n_quad=12, KaV_L=float(d["KaV_L"]))(water(), T_wb=T_WB)[2]
        assert float(r["T_out"]) == pytest.approx(T_OUT, abs=1e-8)

    def test_bigger_fill_cools_more(self):
        outs = [float(tower(mode="rating", KaV_L=k)(water(), T_wb=T_WB)[2]["T_out"]) for k in (0.5, 1.0, 2.0, 4.0)]
        assert all(a > b for a, b in zip(outs, outs[1:]))
        assert all(o > T_WB for o in outs)               # never below the wet bulb

    def test_huge_fill_approaches_the_wet_bulb(self):
        # the Merkel number diverges only logarithmically at the pinch: KaV/L = 8
        # still leaves an approach of about 2 K
        a8 = float(tower(mode="rating", KaV_L=8.0)(water(), T_wb=T_WB)[2]["approach"])
        a16 = float(tower(mode="rating", KaV_L=16.0)(water(), T_wb=T_WB)[2]["approach"])
        assert 0.0 < a16 < a8 < 3.0

    def test_ka_and_volume_give_the_merkel_number(self):
        a = tower(mode="rating", Ka=1.5, V=200.0)(water(), T_wb=T_WB)[2]
        b = tower(mode="rating", KaV_L=1.5 * 200.0 / L)(water(), T_wb=T_WB)[2]
        assert float(a["T_out"]) == pytest.approx(float(b["T_out"]), abs=1e-10)

    def test_fill_characteristic(self):
        C, n = 1.6, 0.6
        a = tower(mode="rating", L_over_G=1.0, fill_characteristic=[C, n])(water(), T_wb=T_WB)[2]
        assert float(a["KaV_L"]) == pytest.approx(C, rel=1e-12)
        b = tower(mode="rating", L_over_G=1.4, fill_characteristic=[C, n])(water(), T_wb=T_WB)[2]
        assert float(b["KaV_L"]) == pytest.approx(C * 1.4 ** -n, rel=1e-12)
        # a lower KaV/L at higher L/G, but the tower still has a fixed size: less cooling
        assert float(b["T_out"]) > float(a["T_out"])

    def test_rating_needs_a_size(self):
        with pytest.raises(ValueError, match="KaV_L"):
            tower(mode="rating")(water(), T_wb=T_WB)

    def test_unreachable_kav_l_is_flagged_not_nan(self):
        info = tower(mode="rating", KaV_L=1e6)(water(), T_wb=T_WB)[2]
        assert not bool(info["converged"])
        assert bool(jnp.isfinite(info["T_out"]))

    def test_params_validation(self):
        with pytest.raises(ValueError):
            CoolingTowerParams(mode="sizing")
        with pytest.raises(ValueError):
            CoolingTowerParams(quadrature="simpson")


class TestGradients:
    def test_merkel_number_wrt_LG_Tin_Tout_Twb(self):
        def f(lg, t_in, t_out, t_wb):
            return merkel_number(t_in, t_out, saturation_enthalpy(t_wb), lg)

        check_grads(f, (1.2, T_IN, T_OUT, T_WB), order=1, modes=["rev"], rtol=1e-5, atol=1e-7)

    def test_merkel_number_through_the_unit(self):
        def f(lg, t_in, t_out, t_wb):
            t = CoolingTower(CoolingTowerParams(mode="design", L_over_G=lg, T_out=t_out))
            return t(make_stream({"H2O": 1.0e4, "air": 0.0}, t_in, 3e5), T_wb=t_wb)[2]["KaV_L"]

        check_grads(f, (1.2, T_IN, T_OUT, T_WB), order=1, modes=["rev"], rtol=1e-5, atol=1e-7)

    def test_gradient_signs(self):
        f = lambda lg, t_out, t_wb: tower(L_over_G=lg, T_out=t_out)(water(), T_wb=t_wb)[2]["KaV_L"]
        g = jax.grad(f, argnums=(0, 1, 2))(1.2, T_OUT, T_WB)
        assert g[0] > 0          # more water per air: bigger tower
        assert g[1] < 0          # a warmer outlet is cheaper
        assert g[2] > 0          # a wetter day shrinks the driving force at a fixed outlet

    def test_rating_T_out_wrt_KaV_L(self):
        def f(k):
            return tower(mode="rating", KaV_L=k)(water(), T_wb=T_WB)[2]["T_out"]

        check_grads(f, (1.4,), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)
        assert float(jax.grad(f)(1.4)) < 0

    def test_rating_T_out_wrt_LG_and_wet_bulb(self):
        def f(lg, t_wb, t_in):
            return CoolingTower(CoolingTowerParams(mode="rating", L_over_G=lg, KaV_L=1.5))(
                make_stream({"H2O": 1.0e4, "air": 0.0}, t_in, 3e5), T_wb=t_wb)[2]["T_out"]

        check_grads(f, (1.2, T_WB, T_IN), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)
        g = jax.grad(f, argnums=(0, 1, 2))(1.2, T_WB, T_IN)
        assert g[0] > 0 and g[1] > 0 and g[2] > 0

    def test_design_gradient_of_approach_vs_size_trade_off(self):
        # the textbook design question: dV/d(approach), fill volume with a fixed Ka
        def volume(approach):
            return tower(T_out=None, approach=approach, Ka=1.0)(water(), T_wb=T_WB)[2]["V"]

        g = float(jax.grad(volume)(5.0))
        fd = (float(volume(5.0 + 1e-4)) - float(volume(5.0 - 1e-4))) / 2e-4
        assert g < 0 and g == pytest.approx(fd, rel=1e-6)

    def test_pinch_limit_gradient(self):
        f = lambda t_wb: maximum_L_over_G(T_IN, T_OUT, saturation_enthalpy(t_wb))
        g = float(jax.grad(f)(T_WB))
        fd = (float(f(T_WB + 1e-4)) - float(f(T_WB - 1e-4))) / 2e-4
        assert g < 0 and g == pytest.approx(fd, rel=1e-5)

    def test_jit(self):
        f = jax.jit(lambda lg: tower(L_over_G=lg)(water(), T_wb=T_WB)[2]["KaV_L"])
        assert float(f(1.2)) == pytest.approx(1.59, rel=0.01)


class TestHelpers:
    def test_saturation_temperature_from_enthalpy_inverts_hs(self):
        for t in (5.0, 22.0, 35.0, 55.0):
            h = saturation_enthalpy(t + K)
            assert float(saturation_temperature_from_enthalpy(h)) == pytest.approx(t + K, abs=1e-8)


# -- the cooling-water loop for the exchangers --------------------------------------


class BasinMakeup:
    """Level control: tops the basin up to the circulation rate (mol/s of water)."""

    symbol = "Basin makeup"

    def __init__(self, flow):
        self.flow = flow

    def __call__(self, inlet):
        out = dict(inlet)
        out["F_H2O"] = self.flow
        return out, {"makeup_mol": self.flow - inlet["F_H2O"]}


def _loop(T_process=340.0, kav_l=1.4, lg=1.2, UA=2.0e6, T_wb=T_WB):
    fs = Flowsheet(species_order=["H2O"])
    fs.add_feed("process", make_stream({"H2O": 2000.0}, T_process, 3e5))
    hx = CounterCurrentHX(HeatExchangerParams(UA=UA, Cp_hot=75.3, Cp_cold=75.3))
    ct = CoolingTower(CoolingTowerParams(mode="rating", L_over_G=lg, KaV_L=kav_l, makeup_in_outlet=False))
    basin = BasinMakeup(6000.0)
    pump = CentrifugalPump(CentrifugalPumpParams(head_coeffs=[40.0, 0.0, -2.0e3], rho=995.0,
                                                 eta_max=0.7, Q_bep=0.5, MW=18.015))
    fs.add_unit(Unit("hx", hx, ["process", "cw_cold"], ["process_out", "cw_hot"]))
    fs.add_unit(Unit("tower", ct, ["cw_hot"], ["cw_fill", "air_out"], params={"T_wb": T_wb}))
    fs.add_unit(Unit("basin", basin, ["cw_fill"], ["cw_basin"]))
    fs.add_unit(Unit("pump", pump, ["cw_basin"], ["cw_pumped"]))
    fs.add_recycle("cw_pumped", "cw_cold")
    guess = {"cw_cold": make_stream({"H2O": 6000.0}, 300.0, 3e5)}
    return fs, guess


class TestCoolingWaterLoop:
    def test_loop_converges_and_closes_the_energy_balance(self):
        fs, guess = _loop()
        with warnings.catch_warnings():
            warnings.simplefilter("error")        # no convergence or tolerance warnings
            s = fs.solve(tear_initial=guess, tol=1e-10, max_iter=100)
        # the recycle is closed
        assert float(s["cw_cold"]["T"]) == pytest.approx(float(s["cw_pumped"]["T"]), abs=1e-8)
        assert float(s["cw_cold"]["F_H2O"]) == pytest.approx(6000.0, rel=1e-12)
        # the basin's level control replaces exactly the water the tower lost (evaporation + drift)
        _, _, info = CoolingTower(CoolingTowerParams(mode="rating", L_over_G=1.2, KaV_L=1.4))(
            s["cw_hot"], T_wb=T_WB)
        lost = 6000.0 - float(s["cw_fill"]["F_H2O"])
        assert lost * MW_WATER == pytest.approx(float(info["evaporation"] + info["drift"]), rel=1e-9)
        # heat the exchanger gives the loop = the tower's heat rejection (Cp = 75.3 J/mol/K)
        Q_hx = 2000.0 * 75.3 * (340.0 - float(s["process_out"]["T"]))
        Q_loop = 6000.0 * 75.3 * (float(s["cw_hot"]["T"]) - float(s["cw_cold"]["T"]))
        assert Q_hx == pytest.approx(Q_loop, rel=1e-8)
        # the tower cools to above the wet bulb, and the process stream ends near the cold-water temperature
        assert T_WB < float(s["cw_basin"]["T"]) < float(s["cw_hot"]["T"])
        assert float(s["process_out"]["T"]) > float(s["cw_cold"]["T"])
        # the basin water comes back at the tower pressure and the pump lifts it again
        assert float(s["cw_basin"]["P"]) == P
        assert float(s["cw_pumped"]["P"]) > P

    def test_loop_state_is_the_tower_rating_at_the_loop_inlet(self):
        fs, guess = _loop()
        s = fs.solve(tear_initial=guess, tol=1e-12, max_iter=100)
        ct = CoolingTower(CoolingTowerParams(mode="rating", L_over_G=1.2, KaV_L=1.4))
        _, _, info = ct(s["cw_hot"], T_wb=T_WB)
        assert float(info["T_out"]) == pytest.approx(float(s["cw_basin"]["T"]), abs=1e-8)

    def test_a_wetter_day_warms_the_loop(self):
        def T_cold(t_wb):
            fs, guess = _loop(T_wb=t_wb)
            return float(fs.solve(tear_initial=guess, tol=1e-10, max_iter=100)["cw_cold"]["T"])

        assert 1.5 < T_cold(T_WB + 5.0) - T_cold(T_WB) < 5.0     # warms, by less than the wet bulb did

    def test_gradient_of_the_loop_wrt_fill_size(self):
        def T_cold(kav_l):
            fs, guess = _loop(kav_l=kav_l)
            return fs.solve(tear_initial=guess, tol=1e-12, max_iter=200)["cw_cold"]["T"]

        g = float(jax.grad(T_cold)(1.4))
        h = 1e-4
        fd = (float(T_cold(1.4 + h)) - float(T_cold(1.4 - h))) / (2 * h)
        assert g < 0
        assert g == pytest.approx(fd, rel=1e-4)
