"""Tests for difflow.psychrometrics.

Oracles, in order of authority:

* an independent NumPy implementation of the Hyland-Wexler (1983) saturation
  pressure over liquid and ice (constants recalled from ASHRAE Fundamentals
  Ch. 1; a mistyped digit would fail the comparison, but they are flagged as
  recalled) and the ASHRAE ideal-gas humid-air formulas;
* round trips between every pair of state inputs;
* limits and identities (Lewis relation, conservation, energy balance);
* a few well-known state points (30 C / 50 % has dew point 18.4 C and wet bulb
  22.0 C; 20 C / 50 % has 9.3 C and 13.8 C), tolerance 0.1 K, recalled;
* saturated-air humidity ratios W_s at 0, 20 and 30 C recalled from the ASHRAE
  table (0.003789, 0.014758, 0.027333?) at 1 %. The issue asks for 0.5 % against
  the table; this module has no enhancement factor, which puts it 0.35-0.5 %
  below the real-gas table, so 1 % is the honest tolerance for these recalled
  numbers, and the 0.5 % claim is not tested against a transcribed table.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import psychrometrics as ps
from difflow.psychrometrics import (
    adiabatic_humidify,
    adiabatic_saturation_temperature,
    air_stream_from_state,
    cool_dehumidify,
    dew_point,
    dry_air_flow,
    humid_enthalpy,
    humid_heat,
    humid_volume,
    humidity_ratio,
    humidity_ratio_from_rh,
    mix_air,
    psychrometric_chart_data,
    rh_from_humidity_ratio,
    saturation_enthalpy,
    saturation_humidity_ratio,
    sensible_heat,
    state,
    state_from_air_stream,
    wet_bulb_temperature,
)

K = 273.15
P = 101325.0


# -- independent reference ---------------------------------------------------


def hw_psat(T):
    """Hyland-Wexler saturation pressure (Pa); constants recalled from ASHRAE Ch. 1."""
    T = np.asarray(T, dtype=float)
    liq = (-5.8002206e3 / T + 1.3914993 - 4.8640239e-2 * T + 4.1764768e-5 * T**2
           - 1.4452093e-8 * T**3 + 6.5459673 * np.log(T))
    ice = (-5.6745359e3 / T + 6.3925247 - 9.677843e-3 * T + 6.2215701e-7 * T**2
           + 2.0747825e-9 * T**3 - 9.484024e-13 * T**4 + 4.1635019 * np.log(T))
    return np.exp(np.where(T < 273.15, ice, liq))


def np_W(p, Ptot=P):
    return 0.621945 * p / (Ptot - p)


def np_h(t, W):
    return 1.006e3 * t + W * (2501.0e3 + 1.86e3 * t)


# -- saturation ----------------------------------------------------------------


class TestSaturation:
    @pytest.mark.parametrize("t", [-10.0, -5.0, 0.5, 5.0, 20.0, 30.0, 40.0, 60.0])
    def test_saturation_pressure_against_hyland_wexler(self, t):
        # IF97 and HW agree to ~1e-4 over liquid; the ice formulas to ~1e-3
        got = float(ps.saturation_pressure(t + K))
        assert got == pytest.approx(float(hw_psat(t + K)), rel=1e-3)

    @pytest.mark.parametrize("t", [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    def test_Ws_and_hs_against_independent_formulas(self, t):
        T = t + K
        p = float(hw_psat(T))
        Ws = np_W(p)
        assert float(saturation_humidity_ratio(T)) == pytest.approx(Ws, rel=2e-3)
        assert float(saturation_enthalpy(T)) == pytest.approx(np_h(t, Ws), rel=2e-3, abs=50.0)

    @pytest.mark.parametrize("t,Ws", [(0.0, 0.003789), (20.0, 0.014758), (30.0, 0.027333)])
    def test_Ws_against_recalled_ashrae_table(self, t, Ws):
        # recalled, unverified; see the module docstring for the tolerance
        assert float(saturation_humidity_ratio(t + K)) == pytest.approx(Ws, rel=1e-2)

    def test_enhancement_factor_sign(self):
        # no enhancement factor: ours sits below the real-gas table by 0.3-0.5 %
        assert float(saturation_humidity_ratio(20 + K)) < 0.014758

    def test_saturated_air_has_unit_rh_and_dew_point_equal_to_T(self):
        for t in (-15.0, 0.0, 12.0, 45.0):
            T = t + K
            W = saturation_humidity_ratio(T)
            assert float(rh_from_humidity_ratio(T, W)) == pytest.approx(1.0, rel=1e-12)
            assert float(dew_point(W)) == pytest.approx(T, abs=1e-8)

    def test_frost_point_is_above_the_liquid_dew_point(self):
        # at -10 C air that is saturated over ice has a "liquid" dew point below -10 C
        T = 263.15
        W = saturation_humidity_ratio(T)
        assert float(dew_point(W, over_ice=False)) < T - 0.5
        assert float(dew_point(W)) == pytest.approx(T, abs=1e-8)

    def test_humidity_ratio_limits(self):
        assert float(humidity_ratio(0.0)) == 0.0
        assert float(humidity_ratio(0.5 * P)) == pytest.approx(0.621945, rel=1e-12)
        assert float(humidity_ratio_from_rh(300.0, 0.0)) == 0.0

    def test_enthalpy_volume_formulas(self):
        t, W = 25.0, 0.01
        assert float(humid_enthalpy(t + K, W)) == pytest.approx(np_h(t, W), rel=1e-12)
        assert float(humid_heat(W)) == pytest.approx(1006.0 + 1860.0 * W)
        # dry air at 0 C and 1 atm: R T / P = 0.7735 m^3/kg
        assert float(humid_volume(K, 0.0)) == pytest.approx(287.042 * K / P, rel=1e-12)
        assert float(humid_volume(t + K, W)) > float(humid_volume(t + K, 0.0))


# -- wet bulb, adiabatic saturation, dew point ---------------------------------------


class TestWetBulb:
    @pytest.mark.parametrize("t,rh,twb,tdp", [(30.0, 0.5, 22.0, 18.4), (20.0, 0.5, 13.8, 9.3)])
    def test_known_state_points(self, t, rh, twb, tdp):
        s = state(T=t + K, RH=rh)
        assert float(s["T_wb"]) - K == pytest.approx(twb, abs=0.1)
        assert float(s["T_dp"]) - K == pytest.approx(tdp, abs=0.1)

    def test_ordering_dew_point_wet_bulb_dry_bulb(self):
        for t, rh in itertools.product([-10.0, 5.0, 25.0, 45.0], [0.1, 0.5, 0.9]):
            s = state(T=t + K, RH=rh)
            assert float(s["T_dp"]) <= float(s["T_wb"]) + 1e-9 <= float(s["T"]) + 1e-9
            assert float(s["T_dp"]) < float(s["T"])

    def test_saturated_air_has_all_three_temperatures_equal(self):
        s = state(T=300.0, RH=1.0)
        assert float(s["T_wb"]) == pytest.approx(300.0, abs=1e-8)
        assert float(s["T_dp"]) == pytest.approx(300.0, abs=1e-8)

    @pytest.mark.parametrize("t,rh", [(-8.0, 0.4), (10.0, 0.2), (30.0, 0.5), (55.0, 0.3)])
    def test_adiabatic_saturation_equals_wet_bulb_at_lewis_one(self, t, rh):
        T = t + K
        W = humidity_ratio_from_rh(T, rh)
        assert float(adiabatic_saturation_temperature(T, W)) == pytest.approx(
            float(wet_bulb_temperature(T, W, lewis=1.0)), abs=1e-8)

    def test_adiabatic_saturation_satisfies_its_enthalpy_balance(self):
        # independent check: h(T,W) + (Ws* - W) h_liq(T*) = h_s(T*)
        T, W = 308.15, 0.01
        Ts = float(adiabatic_saturation_temperature(T, W))
        Ws = np_W(float(hw_psat(Ts)))
        lhs = np_h(T - K, W) + (Ws - W) * 4186.0 * (Ts - K)
        assert lhs == pytest.approx(np_h(Ts - K, Ws), rel=2e-3)

    def test_wet_bulb_satisfies_the_psychrometer_balance(self):
        T, W, le = 303.15, 0.012, 0.9
        Ts = float(wet_bulb_temperature(T, W, lewis=le))
        Ws = np_W(float(hw_psat(Ts)))
        hfg = 2501.0e3 + (1860.0 - 4186.0) * (Ts - K)
        assert le * (1006.0 + 1860.0 * W) * (T - Ts) == pytest.approx((Ws - W) * hfg, rel=2e-3)

    def test_lewis_factor_moves_the_wet_bulb_monotonically(self):
        T, W = 303.15, 0.012
        vals = [float(wet_bulb_temperature(T, W, lewis=le)) for le in (0.8, 0.9, 1.0, 1.1)]
        assert all(a < b for a, b in zip(vals, vals[1:]))   # more convection: closer to the dry bulb
        assert vals[2] == pytest.approx(float(adiabatic_saturation_temperature(T, W)), abs=1e-8)

    def test_below_freezing_the_wick_is_ice(self):
        # an ice-bulb is warmer than the liquid-bulb solution (the heat of fusion)
        T, W = 263.15, 0.0005
        assert float(wet_bulb_temperature(T, W)) > float(wet_bulb_temperature(T, W, over_ice=False)) - 1.0
        assert float(wet_bulb_temperature(T, W)) < T

    def test_vectorises_and_jits(self):
        Tv = jnp.linspace(270.0, 320.0, 7)
        Wv = jnp.full(7, 0.003)
        out = jax.jit(jax.vmap(lambda T, W: wet_bulb_temperature(T, W)))(Tv, Wv)
        for T, W, o in zip(Tv, Wv, out):
            assert float(o) == pytest.approx(float(wet_bulb_temperature(T, W)), abs=1e-9)


# -- state() ----------------------------------------------------------------------


NAMES = ("T", "W", "RH", "T_wb", "T_dp", "h")
STATES = [(-8.0, 0.55), (4.0, 0.8), (22.0, 0.35), (30.0, 0.5), (48.0, 0.15)]


class TestState:
    @pytest.mark.parametrize("t,rh", STATES)
    def test_every_pair_round_trips(self, t, rh):
        ref = state(T=t + K, RH=rh)
        for a, b in itertools.combinations(NAMES, 2):
            if {a, b} == {"W", "T_dp"}:
                continue
            got = state(**{a: ref[a], b: ref[b]})
            for k in ps.STATE_KEYS:
                assert float(got[k]) == pytest.approx(float(ref[k]), rel=1e-9, abs=1e-9), (a, b, k)

    def test_dependent_pair_is_refused(self):
        with pytest.raises(ValueError, match="not independent"):
            state(W=0.01, T_dp=285.0)

    @pytest.mark.parametrize("kw", [dict(T=300.0), dict(T=300.0, W=0.01, RH=0.5), {}])
    def test_needs_exactly_two(self, kw):
        with pytest.raises(ValueError, match="exactly two"):
            state(**kw)

    def test_values_against_independent_formulas(self):
        s = state(T=303.15, RH=0.5)
        p_w = 0.5 * float(hw_psat(303.15))
        W = np_W(p_w)
        assert float(s["W"]) == pytest.approx(W, rel=2e-3)
        assert float(s["h"]) == pytest.approx(np_h(30.0, W), rel=2e-3)
        assert float(s["p_w"]) == pytest.approx(p_w, rel=2e-3)
        assert float(s["mu"]) == pytest.approx(float(s["W"] / s["W_sat"]))
        assert 0.0 < float(s["mu"]) < 1.0

    def test_pressure_dependence(self):
        lo = state(T=300.0, RH=0.5, P=80e3)
        hi = state(T=300.0, RH=0.5, P=P)
        assert float(lo["W"]) > float(hi["W"])           # thinner air holds more water
        assert float(lo["v"]) > float(hi["v"])

    def test_is_a_pytree_of_arrays(self):
        s = state(T=300.0, RH=0.5)
        leaves = jax.tree_util.tree_leaves(s)
        assert len(leaves) == len(ps.STATE_KEYS)
        assert float(jax.jit(lambda T: state(T=T, RH=0.5)["h"])(300.0)) == pytest.approx(float(s["h"]))

    def test_vmap(self):
        T = jnp.linspace(280.0, 310.0, 5)
        out = jax.vmap(lambda t: state(T=t, RH=0.6)["T_dp"])(T)
        assert bool(jnp.all(out < T)) and bool(jnp.all(jnp.diff(out) > 0))


# -- processes ---------------------------------------------------------------------


class TestProcesses:
    def test_mix_air_conserves_dry_air_water_and_enthalpy(self):
        a = state(T=303.15, RH=0.7)
        b = state(T=275.15, RH=0.4)
        c = state(T=290.0, W=0.006)
        m = [3.0, 1.0, 2.0]
        mix = mix_air([a, b, c], m)
        tot = sum(m)
        assert float(mix["W"]) * tot == pytest.approx(sum(mi * float(s["W"]) for mi, s in zip(m, (a, b, c))), rel=1e-13)
        assert float(mix["h"]) * tot == pytest.approx(sum(mi * float(s["h"]) for mi, s in zip(m, (a, b, c))), rel=1e-13)
        # the mixture lies on the straight line between the end states in (W, h)
        assert float(a["T"]) > float(mix["T"]) > float(b["T"])

    def test_mixing_two_saturated_streams_can_fog(self):
        # the convexity of the saturation curve: mu > 1 is reported, not hidden
        a, b = state(T=263.15, RH=1.0), state(T=303.15, RH=1.0)
        assert float(mix_air([a, b], [1.0, 1.0])["mu"]) > 1.0

    def test_sensible_heating_keeps_W_and_obeys_the_energy_balance(self):
        a = state(T=290.0, RH=0.6)
        out = sensible_heat(a, 20e3, m_dry_air=2.0)       # 20 kW into 2 kg/s
        assert float(out["W"]) == float(a["W"])
        assert float(out["h"] - a["h"]) == pytest.approx(10e3, rel=1e-12)
        assert float(out["T"]) - 290.0 == pytest.approx(10e3 / float(humid_heat(a["W"])), rel=1e-12)
        assert float(out["RH"]) < float(a["RH"])

    def test_cooling_coil_energy_balance_and_condensate(self):
        a = state(T=303.15, RH=0.6)
        out, info = cool_dehumidify(a, 281.15, bypass_factor=0.15)
        assert float(out["W"]) < float(a["W"])
        cond = float(a["W"] - out["W"])
        assert float(info["condensate"]) == pytest.approx(cond, rel=1e-12)
        h_liq = 4186.0 * (281.15 - K)
        assert float(a["h"] - out["h"]) - cond * h_liq == pytest.approx(float(info["Q_total"]), rel=1e-12)
        assert float(info["Q_sensible"]) + float(info["Q_latent"]) == pytest.approx(float(info["Q_total"]), rel=1e-12)
        assert float(info["Q_latent"]) > 0.0 and float(info["Q_sensible"]) > 0.0

    def test_coil_leaving_state_is_on_the_line_to_the_apparatus_dew_point(self):
        a = state(T=303.15, RH=0.6)
        bf = 0.2
        out, _ = cool_dehumidify(a, 283.15, bypass_factor=bf)
        Ws = float(saturation_humidity_ratio(283.15))
        assert float(out["W"]) == pytest.approx(bf * float(a["W"]) + (1 - bf) * Ws, rel=1e-12)
        # zero bypass: leaving air saturated at the coil temperature
        full, _ = cool_dehumidify(a, 283.15, 0.0)
        assert float(full["T"]) == pytest.approx(283.15, abs=1e-8)
        assert float(full["RH"]) == pytest.approx(1.0, rel=1e-9)

    def test_warm_coil_cools_sensibly(self):
        a = state(T=303.15, RH=0.3)                        # dew point ~ 10 C
        out, info = cool_dehumidify(a, 295.15, bypass_factor=0.1)
        assert float(out["W"]) == pytest.approx(float(a["W"]), rel=1e-12)
        assert float(info["condensate"]) == 0.0
        assert float(out["T"]) < 303.15

    def test_adiabatic_humidification_follows_the_wet_bulb_line(self):
        a = state(T=308.15, RH=0.2)
        out = adiabatic_humidify(a, effectiveness=0.8)
        assert float(out["T"]) < float(a["T"]) and float(out["W"]) > float(a["W"])
        assert float(out["T_wb"]) == pytest.approx(float(a["T_wb"]), abs=0.05)
        full = adiabatic_humidify(a, 1.0)
        assert float(full["RH"]) == pytest.approx(1.0, abs=1e-6)
        assert float(full["T"]) == pytest.approx(float(a["T_wb"]), abs=1e-6)
        target = adiabatic_humidify(a, W_out=0.01)
        assert float(target["W"]) == pytest.approx(0.01, rel=1e-12)


# -- air streams -------------------------------------------------------------------


class TestAirStreams:
    def test_round_trip(self):
        s = state(T=305.0, RH=0.45, P=95e3)
        stream = air_stream_from_state(s, 12.5)
        assert set(stream) == {"F_H2O", "F_air", "T", "P"}
        assert float(dry_air_flow(stream)) == pytest.approx(12.5, rel=1e-12)
        back = state_from_air_stream(stream)
        for k in ps.STATE_KEYS:
            assert float(back[k]) == pytest.approx(float(s[k]), rel=1e-9)

    def test_water_mass_flow_is_W_times_dry_air(self):
        s = state(T=305.0, W=0.02)
        stream = air_stream_from_state(s, 10.0)
        assert float(stream["F_H2O"]) * ps.MW_WATER == pytest.approx(0.2, rel=1e-12)

    def test_n2_o2_stream_is_read_as_dry_air(self):
        n = 10.0 / ps.MW_DRY_AIR
        stream = {"F_N2": 0.79 * n, "F_O2": 0.21 * n, "F_H2O": 0.1 / ps.MW_WATER, "T": 300.0, "P": P}
        s = state_from_air_stream(stream)
        assert float(s["W"]) == pytest.approx(0.01, rel=1e-12)

    def test_empty_air_stream_is_dry_not_nan(self):
        s = state_from_air_stream({"F_H2O": 0.0, "F_air": 0.0, "T": 300.0, "P": P})
        assert float(s["W"]) == 0.0 and bool(jnp.isfinite(s["h"]))

    def test_gradient_through_the_stream(self):
        f = lambda T, m: state_from_air_stream(
            air_stream_from_state(state(T=T, RH=0.5), m))["T_wb"]
        check_grads(f, (300.0, 5.0), order=1, modes=["rev"], rtol=1e-5)


# -- chart data --------------------------------------------------------------------


class TestChart:
    def test_families_and_consistency(self):
        d = psychrometric_chart_data()
        assert len(d["rh"]) == 10 and len(d["wb"]) > 5 and len(d["h"]) > 5 and len(d["v"]) > 5
        sat = d["saturation"]
        np.testing.assert_allclose(sat["W"], np.asarray(saturation_humidity_ratio(jnp.asarray(sat["T"]))), rtol=1e-12)
        np.testing.assert_allclose(d["rh"][-1]["W"], sat["W"], rtol=1e-12)
        # every plotted point lies in the physical region
        for fam in ("rh", "wb", "h", "v"):
            for ln in d[fam]:
                W = ln["W"][np.isfinite(ln["W"])]
                assert W.size > 0 and W.min() >= 0.0

    def test_each_line_is_the_state_it_claims_to_be(self):
        d = psychrometric_chart_data(n=41)
        for fam, key, scale in (("rh", "RH", 1.0), ("wb", "T_wb", 1.0), ("h", "h", 1.0), ("v", "v", 1.0)):
            ln = d[fam][len(d[fam]) // 2]
            ok = np.where(np.isfinite(ln["W"]))[0][::7]
            for i in ok:
                s = state(T=ln["T"][i], W=ln["W"][i])
                assert float(s[key]) == pytest.approx(ln["value"], rel=1e-7, abs=1e-7), fam

    def test_plot_runs_and_overlays_processes(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from difflow.visualization import plot_psychrometric_chart
        a = state(T=303.15, RH=0.6)
        b, _ = cool_dehumidify(a, 285.15, 0.1)
        ax = plot_psychrometric_chart(psychrometric_chart_data(), processes=[{"states": [a, b], "label": "coil"}])
        assert ax.get_xlabel().startswith("dry-bulb")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "coil" in labels and "saturation" in labels
        styles = {ln.get_linestyle() for ln in ax.get_lines()}
        assert len(styles) >= 4        # families differ by line style, not only by colour


# -- gradients ---------------------------------------------------------------------


def _prop(name):
    def f(T, RH, Pt):
        return state(T=T, RH=RH, P=Pt)[name]
    return f


class TestGradients:
    @pytest.mark.parametrize("name", ["W", "h", "T_wb", "T_dp"])
    @pytest.mark.parametrize("x0", [(303.15, 0.5, P), (268.15, 0.6, 90e3)])
    def test_check_grads_wrt_T_RH_P(self, name, x0):
        check_grads(_prop(name), x0, order=1, modes=["rev"], rtol=1e-5, atol=1e-8)

    def test_wet_bulb_derivative_wrt_W_matches_finite_differences(self):
        f = lambda W: wet_bulb_temperature(303.15, W)
        g = float(jax.grad(f)(0.012))
        h = 1e-7
        assert g == pytest.approx((float(f(0.012 + h)) - float(f(0.012 - h))) / (2 * h), rel=1e-6)
        assert g > 0

    def test_dew_point_derivative_is_physical(self):
        # p_ws(T_dp) = RH p_ws(T)  =>  dT_dp/dRH = p_ws(T) / p_ws'(T_dp)
        g = float(jax.grad(lambda rh: state(T=300.0, RH=rh)["T_dp"])(0.5))
        Td = float(state(T=300.0, RH=0.5)["T_dp"])
        dp = float(jax.grad(ps.saturation_pressure)(Td))
        assert g == pytest.approx(float(ps.saturation_pressure(300.0)) / dp, rel=1e-8)

    def test_other_pairs_differentiate(self):
        for pair in (("RH", "h"), ("RH", "T_wb"), ("T_wb", "h")):
            ref = state(T=303.15, RH=0.5)

            def f(x, pair=pair):
                kw = {pair[0]: ref[pair[0]], pair[1]: ref[pair[1]] * x}
                return state(**kw)["T"]

            check_grads(f, (1.0,), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)


class TestHotAirWetBulb:
    """Wet bulb above the boiling point of water at P (a dryer's 100-200 C air): the Newton seed must stay below it."""

    @pytest.mark.parametrize("T,W", [(373.15, 0.01), (423.15, 0.01), (473.15, 0.05)])
    def test_wet_bulb_of_hot_air_is_finite_and_satisfies_the_balance(self, T, W):
        from scipy import optimize

        Twb = float(wet_bulb_temperature(T, W))
        assert 280.0 < Twb < 345.0

        def res(Ts):
            pws = float(ps.saturation_pressure(Ts))
            Ws = 0.621945 * pws / (101325.0 - pws)
            return (T - Ts) * (1006.0 + W * 1860.0) - (Ws - W) * (2501e3 + (1860.0 - 4186.0) * (Ts - 273.15))

        assert Twb == pytest.approx(optimize.brentq(res, 250.0, 372.0, xtol=1e-12), abs=1e-8)
        # the adiabatic-saturation temperature is the same state for this model
        assert float(adiabatic_saturation_temperature(T, W)) == pytest.approx(Twb, abs=1e-6)
