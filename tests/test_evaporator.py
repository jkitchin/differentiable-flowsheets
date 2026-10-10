"""Tests for the evaporators (issue #404).

What is and is not verified here, plainly:

* VERIFIED: mass and energy closure; agreement with an independent NumPy /
  SciPy coding of the same equations (own steam properties, ``fsolve``);
  single effect == one-effect multi effect; design/rating round trips;
  steam economy rising with N; gradients (``check_grads``, order 1, reverse).
* Steam properties: Psat/Tsat are IAPWS-IF97 (checked by round trip and the
  normal boiling point); h_f and h_fg are fits to steam-table points recalled
  from a standard table, checked here against the same recalled points.
* NOT VERIFIED: any textbook worked example to 2-3 %. The Geankoplis Ex.
  8.4-1 inputs are run as a *regression* pin (values this code produced), NOT
  as agreement with the book, whose printed answers were not at hand. The
  NaOH / NaCl / sucrose BPR data are unverified recollections.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import fsolve

from difflow import (
    Evaporator,
    EvaporatorParams,
    MechanicalVaporRecompression,
    MultiEffectEvaporator,
    MultiEffectEvaporatorParams,
    MVRParams,
    UnverifiedDataWarning,
    boiling_point_rise,
    make_stream,
)
from difflow.units.evaporator import (
    water_latent_heat,
    water_liquid_enthalpy,
    water_saturation_pressure,
    water_saturation_temperature,
    water_vapor_enthalpy,
)

jax.config.update("jax_enable_x64", True)

MW_NAOH = 40.0
MW_SUC = 342.3


def naoh_feed(F=10.0, xF=0.10, T=330.0):
    """F kg/s of an xF solute-mass-fraction NaOH solution."""
    return make_stream(
        {"water": F * (1 - xF) * 1000 / 18.015, "NaOH": F * xF * 1000 / MW_NAOH},
        T=T, P=2e5,
    )


def sugar_feed(xF, F=22680 / 3600, T=299.85):
    return make_stream(
        {"water": F * (1 - xF) * 1000 / 18.015, "sucrose": F * xF * 1000 / MW_SUC},
        T=T, P=1e5,
    )


NAOH = dict(solute_species=["NaOH"], solute_MW=[MW_NAOH], bpr_model="ideal",
            vant_hoff_i=2.0)
#: Geankoplis Ex. 8.4-1 inputs as recalled (sugar, BPR = 1.78 x + 6.22 x^2,
#: cp = 4.19 - 2.35 x kJ/kg/K): a regression pin only, see the module docstring.
GEANKOPLIS = dict(solute_species=["sucrose"], solute_MW=[MW_SUC],
                  bpr_model="polynomial", bpr_coeffs=[0.0, 1.78, 6.22],
                  cp_coeffs=[4190.0, -2350.0])


# =============================================================================
# Steam properties
# =============================================================================


class TestSteamProperties:
    def test_normal_boiling_point(self):
        assert float(water_saturation_pressure(373.15)) == pytest.approx(101417.98, rel=1e-5)
        assert float(water_saturation_temperature(101325.0)) == pytest.approx(373.124, abs=0.01)

    def test_psat_tsat_round_trip(self):
        T = jnp.linspace(280.0, 480.0, 9)
        np.testing.assert_allclose(
            water_saturation_temperature(water_saturation_pressure(T)), T, atol=1e-6)

    # (T in C, h_f, h_fg) in kJ/kg, recalled from a standard steam table; the
    # correlations are fits to exactly these points, so this checks the fit
    # and the units, not the table.
    @pytest.mark.parametrize("t,hf,hfg", [
        (50.0, 209.34, 2382.0), (100.0, 419.17, 2256.4),
        (150.0, 632.18, 2113.8), (200.0, 852.26, 1939.8),
    ])
    def test_enthalpies_against_recalled_table(self, t, hf, hfg):
        T = t + 273.15
        assert float(water_liquid_enthalpy(T)) / 1e3 == pytest.approx(hf, rel=2e-3)
        assert float(water_latent_heat(T)) / 1e3 == pytest.approx(hfg, rel=2e-3)

    def test_latent_heat_obeys_clapeyron_independently(self):
        """h_fg = T (dP/dT) (v_g - v_f), with ideal-gas v_g and a Z of ~0.97.

        An independent check of the latent-heat fit that uses only the
        (IAPWS) saturation-pressure equation, good to a few percent up to
        150 C; a constant Z is too crude above that, so it is not tested there.
        """
        for t in (60.0, 100.0, 150.0):
            T = t + 273.15
            dPdT = float(jax.grad(water_saturation_pressure)(T))
            P = float(water_saturation_pressure(T))
            vg = 0.97 * 461.5 * T / P
            h = T * dPdT * (vg - 0.00105)
            assert float(water_latent_heat(T)) == pytest.approx(h, rel=0.04)


# =============================================================================
# Boiling-point rise
# =============================================================================


class TestBoilingPointRise:
    def test_zero_without_solute(self):
        for m in ("ideal", "colligative"):
            assert float(boiling_point_rise(0.0, 101325.0, m, solute_MW=40.0)) == pytest.approx(0.0, abs=1e-9)

    def test_ideal_is_raoults_law(self):
        x, P, MW, i = 0.2, 50e3, 40.0, 2.0
        bpr = float(boiling_point_rise(x, P, "ideal", solute_MW=MW, vant_hoff_i=i))
        T = float(water_saturation_temperature(P)) + bpr
        n_w, n_s = (1 - x) / 18.015, i * x / MW
        assert n_w / (n_w + n_s) * float(water_saturation_pressure(T)) == pytest.approx(P, rel=1e-9)

    def test_colligative_is_the_dilute_limit_of_ideal(self):
        kw = dict(solute_MW=40.0, vant_hoff_i=2.0)
        a = float(boiling_point_rise(0.01, 101325.0, "ideal", **kw))
        b = float(boiling_point_rise(0.01, 101325.0, "colligative", **kw))
        assert a == pytest.approx(b, rel=0.02)
        # ebullioscopic constant of water, 0.51 K kg/mol, as a sanity anchor
        m = 0.01 / 40.0 / 0.99 * 1000.0 * 2.0
        assert b == pytest.approx(0.512 * m, rel=0.02)

    def test_rises_with_concentration_and_falls_with_pressure(self):
        lo = boiling_point_rise(0.1, 101325.0, "ideal", solute_MW=40.0, vant_hoff_i=2.0)
        hi = boiling_point_rise(0.3, 101325.0, "ideal", solute_MW=40.0, vant_hoff_i=2.0)
        low_p = boiling_point_rise(0.3, 20e3, "ideal", solute_MW=40.0, vant_hoff_i=2.0)
        assert float(hi) > float(lo) > 0.0
        assert float(low_p) < float(hi)

    def test_polynomial_and_sucrose(self):
        got = float(boiling_point_rise(0.5, 1e5, "polynomial", coeffs=[0.0, 1.78, 6.22]))
        assert got == pytest.approx(1.78 * 0.5 + 6.22 * 0.25)
        with pytest.warns(UnverifiedDataWarning):
            pass_ = Evaporator(EvaporatorParams(
                solute_species=["s"], solute_MW=[342.3], x_product=0.3,
                bpr_model="sucrose", steam_P=2e5))
        assert pass_ is not None

    def test_duhring_lines_user_table(self):
        # T_soln = a + b T_water; at T_water = 373.15, BPR = a + (b - 1) T_w
        x, a, b = [0.0, 0.5], [0.0, -20.0], [1.0, 1.1]
        got = float(boiling_point_rise(0.5, 101325.0, "duhring", duhring=(x, a, b)))
        Tw = float(water_saturation_temperature(101325.0))
        assert got == pytest.approx(-20.0 + 0.1 * Tw)

    def test_builtin_tables_are_flagged_unverified(self):
        for m in ("naoh", "nacl"):
            with pytest.warns(UnverifiedDataWarning):
                Evaporator(EvaporatorParams(
                    solute_species=["s"], solute_MW=[40.0], x_product=0.3,
                    bpr_model=m, steam_P=2e5))
        # the table is the pinned approximation, NOT a claim about the chart
        v = float(boiling_point_rise(0.5, 101325.0, "naoh"))
        assert v == pytest.approx(43.0, abs=1.0)

    def test_callable_and_errors(self):
        f = lambda x, Tw, P: 10.0 * x
        assert float(boiling_point_rise(0.3, 1e5, "callable", bpr_fn=f)) == pytest.approx(3.0)
        with pytest.raises(ValueError):
            boiling_point_rise(0.3, 1e5, "no_such_model")
        with pytest.raises(ValueError):
            boiling_point_rise(0.3, 1e5, "ideal")  # needs solute_MW

    def test_gradient(self):
        f = lambda x, P: boiling_point_rise(x, P, "ideal", solute_MW=40.0, vant_hoff_i=2.0)
        check_grads(f, (0.2, 60e3), order=1, modes=["rev"], rtol=1e-4)


# =============================================================================
# Single effect
# =============================================================================


def single(**kw):
    base = dict(NAOH, U=1500.0, P_vapor_space=30e3, steam_P=300e3)
    base.update(kw)
    return Evaporator(EvaporatorParams(**base))


class TestSingleEffect:
    def test_design_balances_close(self):
        conc, vap, info = single(x_product=0.3)(naoh_feed())
        assert float(info["mass_balance_error"]) == pytest.approx(0.0, abs=1e-12)
        assert float(info["energy_balance_error"]) / float(info["Q"]) == pytest.approx(0.0, abs=1e-12)
        # solute does not leave with the vapor; the vapor is all solvent
        assert float(vap["F_NaOH"]) == 0.0
        assert float(conc["F_NaOH"]) == pytest.approx(float(naoh_feed()["F_NaOH"]))
        # product is 30 wt % by mass, computed back from the stream
        m_s = float(conc["F_NaOH"]) * MW_NAOH / 1e3
        m_w = float(conc["F_water"]) * 18.015 / 1e3
        assert m_s / (m_s + m_w) == pytest.approx(0.3)
        # vapor flow in mol/s matches V in kg/s
        assert float(vap["F_water"]) * 18.015 / 1e3 == pytest.approx(float(info["V"]))

    def test_vapor_leaves_superheated_by_the_bpr(self):
        _, vap, info = single(x_product=0.3)(naoh_feed())
        assert float(vap["T"]) - float(info["T_sat_vapor"]) == pytest.approx(float(info["BPR"]))
        assert float(info["BPR"]) > 5.0

    def test_matches_independent_numpy_implementation(self):
        """Same equations, separately coded with NumPy and plain floats."""
        P_v, P_s, U, xP, xF, F, TF = 30e3, 300e3, 1500.0, 0.30, 0.10, 10.0, 330.0
        n = (0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
             0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
             -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
             0.65017534844798e3)

        def Ps(T):
            th = T + n[8] / (T - n[9])
            A = th**2 + n[0] * th + n[1]
            B = n[2] * th**2 + n[3] * th + n[4]
            C = n[5] * th**2 + n[6] * th + n[7]
            return (2 * C / (-B + np.sqrt(B * B - 4 * A * C))) ** 4 * 1e6

        def Ts(P):
            b = (P / 1e6) ** 0.25
            E = b * b + n[2] * b + n[5]
            Fq = n[0] * b * b + n[3] * b + n[6]
            G = n[1] * b * b + n[4] * b + n[7]
            D = 2 * G / (-Fq - np.sqrt(Fq * Fq - 4 * E * G))
            return 0.5 * (n[9] + D - np.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D)))

        Tc = 647.096
        hf = lambda T: ((4.53143312e-03 * (T - 273.15) - 6.69677099e-01) * (T - 273.15)
                        + 4.21481377e03) * (T - 273.15) - 2.36660783e02
        hfg = lambda T: 2.92757368e06 * (1 - T / Tc) ** (0.264717681 + 0.0657306406 * T / Tc)
        Tw, Tsteam = Ts(P_v), Ts(P_s)
        xw = ((1 - xP) / 18.015) / ((1 - xP) / 18.015 + 2 * xP / 40.0)
        T = Ts(P_v / xw)
        L, V = F * xF / xP, F - F * xF / xP
        h = lambda T, x: (1 - x) * hf(T) + x * 1500.0 * (T - 273.15)
        HV = hf(Tw) + hfg(Tw) + 1880.0 * (T - Tw)
        Q = L * h(T, xP) + V * HV - F * h(TF, xF)
        S, A = Q / hfg(Tsteam), Q / (1500.0 * (Tsteam - T))

        _, _, info = single(x_product=xP)(naoh_feed(F, xF, TF))
        assert float(info["steam"]) == pytest.approx(S, rel=1e-9)
        assert float(info["A"]) == pytest.approx(A, rel=1e-9)
        assert float(info["T_soln"]) == pytest.approx(T, rel=1e-12)

    def test_economy_is_below_one_for_cold_feed(self):
        _, _, info = single(x_product=0.3)(naoh_feed(T=300.0))
        assert 0.6 < float(info["steam_economy"]) < 1.0

    def test_economy_is_the_latent_heat_ratio_without_bpr_and_a_boiling_feed(self):
        """Water-like solution, feed at the boiling point: S lambda_s = V lambda_w.

        So the economy is lambda_s / lambda_w, a little under 1 because the
        vapor space is colder than the steam (latent heat rises as T falls).
        """
        Tw = float(water_saturation_temperature(30e3))
        Ts = float(water_saturation_temperature(300e3))
        _, _, info = single(x_product=0.2, bpr_model="none")(naoh_feed(T=Tw))
        ratio = float(water_latent_heat(Ts) / water_latent_heat(Tw))
        assert float(info["steam_economy"]) == pytest.approx(ratio, rel=0.01)

    def test_bpr_costs_steam_and_area(self):
        """With the BPR the driving force shrinks, so the area grows."""
        base = single(x_product=0.3, bpr_model="none")(naoh_feed())[2]
        with_bpr = single(x_product=0.3)(naoh_feed())[2]
        assert float(with_bpr["A"]) > float(base["A"])
        assert float(with_bpr["driving_force"]) < float(base["driving_force"])

    def test_rating_inverts_design(self):
        _, _, d = single(x_product=0.3)(naoh_feed())
        _, _, r = single(A=float(d["A"]))(naoh_feed())
        assert bool(r["converged"])
        assert float(r["x_product"]) == pytest.approx(0.3, rel=1e-6)
        assert float(r["steam"]) == pytest.approx(float(d["steam"]), rel=1e-5)

    def test_more_area_gives_more_concentration(self):
        a = single(A=60.0)(naoh_feed())[2]["x_product"]
        b = single(A=120.0)(naoh_feed())[2]["x_product"]
        assert float(b) > float(a)

    def test_steam_stream_and_steam_T_agree_with_steam_P(self):
        feed = naoh_feed()
        _, _, a = single(x_product=0.3)(feed)
        _, _, b = single(x_product=0.3, steam_P=None,
                         steam_T=float(water_saturation_temperature(300e3)))(feed)
        steam = make_stream({"water": 1.0}, T=400.0, P=300e3)
        _, _, c = single(x_product=0.3, steam_P=None)(feed, steam)
        assert float(b["steam"]) == pytest.approx(float(a["steam"]))
        assert float(c["steam"]) == pytest.approx(float(a["steam"]))

    def test_enthalpy_fn_heat_of_dilution(self):
        """Exothermic dilution means concentrating costs extra heat."""
        base = single(x_product=0.3)(naoh_feed())[2]["Q"]

        def h(T, x):
            hw = water_liquid_enthalpy(T)
            return (1 - x) * hw + x * 1500.0 * (T - 273.15) + - 2.0e5 * x * (1 - x)

        p = EvaporatorParams(**dict(NAOH, U=1500.0, P_vapor_space=30e3, steam_P=300e3,
                                    x_product=0.3))
        q = Evaporator(p, enthalpy_fn=h)(naoh_feed())[2]["Q"]
        assert float(q) > float(base)

    def test_bpr_fn(self):
        p = EvaporatorParams(solute_species=["NaOH"], solute_MW=[40.0], U=1500.0,
                             x_product=0.3, P_vapor_space=30e3, steam_P=300e3)
        _, _, info = Evaporator(p, bpr_fn=lambda x, Tw, P: 20.0 * x)(naoh_feed())
        assert float(info["BPR"]) == pytest.approx(6.0)

    def test_input_errors(self):
        with pytest.raises(ValueError, match="exactly one"):
            single()
        with pytest.raises(ValueError, match="exactly one"):
            single(x_product=0.3, A=50.0)
        with pytest.raises(ValueError, match="steam"):
            single(x_product=0.3, steam_P=None)(naoh_feed())
        stray = dict(naoh_feed(), F_other=jnp.asarray(1.0))
        with pytest.raises(ValueError, match="neither"):
            single(x_product=0.3)(stray)

    def test_jit(self):
        ev = single(x_product=0.3)
        f = jax.jit(lambda feed: ev(feed)[2]["steam"])
        assert float(f(naoh_feed())) == pytest.approx(float(ev(naoh_feed())[2]["steam"]))

    def test_in_a_flowsheet(self):
        from difflow.flowsheet import Flowsheet, Unit

        fs = Flowsheet(species_order=["water", "NaOH"])
        fs.add_feed("feed", naoh_feed())
        fs.add_unit(Unit("evap", single(x_product=0.3), ["feed"], ["conc", "vap"]))
        out = fs.solve()
        assert float(out["vap"]["F_NaOH"]) == 0.0
        assert float(out["conc"]["T"]) > float(out["vap"]["T"]) - 1e-9


# =============================================================================
# Multiple effect
# =============================================================================


def multi(N=3, feed="forward", **kw):
    base = dict(NAOH, n_effects=N, feed=feed, U=[2500.0, 2000.0, 1500.0, 1200.0, 1000.0, 900.0][:N],
                P_vapor_space=30e3, steam_P=300e3)
    base.update(kw)
    return MultiEffectEvaporator(MultiEffectEvaporatorParams(**base))


class TestMultiEffect:
    @pytest.mark.parametrize("arrangement", ["forward", "backward", "parallel"])
    def test_design_closes_and_has_equal_areas(self, arrangement):
        conc, vap, info = multi(3, arrangement, x_product=0.3)(naoh_feed())
        assert bool(info["converged"])
        A = np.asarray(info["A"])
        np.testing.assert_allclose(A, A[0], rtol=1e-12)
        assert float(info["mass_balance_error"]) == pytest.approx(0.0, abs=1e-9)
        assert abs(float(info["energy_balance_error"])) < 1e-6 * float(info["steam"]) * 2e6
        assert float(info["x_product"]) == pytest.approx(0.3, rel=1e-9)
        # every effect: Q = U A dT and the temperatures fall down the train
        T = np.asarray(info["T"])
        Tw = np.asarray(info["T_water"])
        Th = np.concatenate([[float(info["T_steam"])], Tw[:-1]])
        U = np.array([2500.0, 2000.0, 1500.0])
        np.testing.assert_allclose(np.asarray(info["Q"]), U * A * (Th - T), rtol=1e-12)
        assert np.all(np.diff(Tw) < 0) and np.all(T < Th)
        assert float(vap["F_NaOH"]) == 0.0

    def test_forward_feed_concentration_profile(self):
        info = multi(3, "forward", x_product=0.3)(naoh_feed())[2]
        x = np.asarray(info["x"])
        assert np.all(np.diff(x) > 0) and x[-1] == pytest.approx(0.3)
        back = multi(3, "backward", x_product=0.3)(naoh_feed())[2]
        xb = np.asarray(back["x"])
        assert np.all(np.diff(xb) < 0) and xb[0] == pytest.approx(0.3)
        par = multi(3, "parallel", x_product=0.3)(naoh_feed())[2]
        np.testing.assert_allclose(np.asarray(par["x"]), 0.3)

    def test_one_effect_equals_the_single_effect_unit(self):
        feed = naoh_feed()
        _, _, s = single(x_product=0.3)(feed)
        for arrangement in ("forward", "backward", "parallel"):
            _, _, m = multi(1, arrangement, U=[1500.0], x_product=0.3)(feed)
            assert float(m["steam"]) == pytest.approx(float(s["steam"]), rel=1e-8)
            assert float(m["A"][0]) == pytest.approx(float(s["A"]), rel=1e-8)
            assert float(m["steam_economy"]) == pytest.approx(float(s["steam_economy"]), rel=1e-8)
        # and in rating mode
        _, _, r = multi(1, "forward", U=[1500.0], A=[float(s["A"])])(feed)
        assert float(r["x_product"]) == pytest.approx(0.3, rel=1e-6)

    def test_matches_independent_scipy_implementation(self):
        """Forward-feed triple effect: the same equations, coded separately.

        Own steam properties and balances in plain floats, solved by
        ``scipy.optimize.fsolve`` from a perturbed guess; the unknowns are
        [V1, V2, V3, S, Tw1, Tw2, A].
        """
        F, xF, TF, xP = 6.3, 0.10, 299.85, 0.5
        Ps_v, Ps_s = 13.4e3, 205.5e3
        U = np.array([3123.0, 1987.0, 1136.0])
        n = (0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
             0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
             -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
             0.65017534844798e3)

        def Tsat(P):
            b = (P / 1e6) ** 0.25
            E = b * b + n[2] * b + n[5]
            Fq = n[0] * b * b + n[3] * b + n[6]
            G = n[1] * b * b + n[4] * b + n[7]
            D = 2 * G / (-Fq - np.sqrt(Fq * Fq - 4 * E * G))
            return 0.5 * (n[9] + D - np.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D)))

        Tc = 647.096
        hf = lambda T: ((4.53143312e-03 * (T - 273.15) - 6.69677099e-01) * (T - 273.15)
                        + 4.21481377e03) * (T - 273.15) - 2.36660783e02
        hfg = lambda T: 2.92757368e06 * (1 - T / Tc) ** (0.264717681 + 0.0657306406 * T / Tc)
        hL = lambda T, x: (4190.0 - 2350.0 * x) * (T - 273.15)
        bpr = lambda x: 1.78 * x + 6.22 * x * x
        Ts, Tw3 = Tsat(Ps_s), Tsat(Ps_v)
        ms = F * xF

        def res(u):
            V1, V2, V3, S, Tw1, Tw2, A = u
            Tw = [Tw1, Tw2, Tw3]
            V = [V1, V2, V3]
            Lin, Tin, xin = F, TF, xF
            out = []
            Th, Qh_prev = Ts, None
            for i in range(3):
                Lo = Lin - V[i]
                x = ms / Lo
                T = Tw[i] + bpr(x)
                HV = hf(Tw[i]) + hfg(Tw[i]) + 1880.0 * (T - Tw[i])
                Qb = Lo * hL(T, x) + V[i] * HV - Lin * hL(Tin, xin)
                Qh = S * hfg(Ts) if i == 0 else V[i - 1] * (hfg(Th) + 1880.0 * bpr_prev)
                out += [(Qh - Qb) / 1e7, (Qh - U[i] * A * (Th - T)) / 1e7]
                Lin, Tin, xin, Th, bpr_prev = Lo, T, x, Tw[i], bpr(x)
            out.append(sum(V) - (F - ms / xP))
            return out

        m = multi(3, "forward", U=list(U), P_vapor_space=Ps_v, steam_P=Ps_s,
                  x_product=xP, solute_species=["sucrose"], solute_MW=[MW_SUC],
                  bpr_model="polynomial", bpr_coeffs=[0.0, 1.78, 6.22],
                  cp_coeffs=[4190.0, -2350.0], vant_hoff_i=1.0)
        _, _, info = m(sugar_feed(xF))
        u_mod = np.array([*np.asarray(info["V"]), float(info["steam"]),
                          *np.asarray(info["T_water"])[:2], float(info["A"][0])])
        sol, _, ier, msg = fsolve(res, u_mod * np.array([1.05, 0.95, 1.05, 1.05, 1.001,
                                                          0.999, 0.95]),
                                  full_output=True, xtol=1e-13)
        assert ier == 1, msg
        np.testing.assert_allclose(sol, u_mod, rtol=1e-6)

    def test_geankoplis_8_4_1_inputs_regression_pin(self):
        """NOT a textbook-agreement claim: the values this code returned for the
        recalled Geankoplis Ex. 8.4-1 inputs, pinned against silent change.
        """
        m = multi(3, "forward", U=[3123.0, 1987.0, 1136.0], x_product=0.5,
                  P_vapor_space=13.4e3, steam_P=205.5e3, solute_species=["sucrose"],
                  solute_MW=[MW_SUC], bpr_model="polynomial",
                  bpr_coeffs=[0.0, 1.78, 6.22], cp_coeffs=[4190.0, -2350.0],
                  vant_hoff_i=1.0)
        _, _, info = m(sugar_feed(0.10))
        assert float(info["steam"]) == pytest.approx(2.4886358721, rel=1e-6)
        assert float(info["A"][0]) == pytest.approx(105.0124441616, rel=1e-6)
        assert float(info["steam_economy"]) == pytest.approx(2.0252058794, rel=1e-6)
        assert float(info["steam_economy"]) < 3.0

    def test_steam_economy_rises_with_n(self):
        """About N x 0.8-1.0 for a water-like solution with a preheated feed.

        No BPR, feed at 370 K: the economy per effect starts near 1 and sags
        as N grows (more sensible heat is lost and the temperature span is
        shared), but stays well above 0.7 N up to six effects.
        """
        feed = naoh_feed(T=370.0)
        eco = []
        for N in range(1, 7):
            _, _, info = multi(N, "forward", U=[2500.0], P_vapor_space=20e3, steam_P=400e3,
                               x_product=0.2, bpr_model="none")(feed)
            assert bool(info["converged"])
            eco.append(float(info["steam_economy"]))
        assert all(b > a for a, b in zip(eco, eco[1:]))
        for N, e in enumerate(eco, start=1):
            assert 0.78 * N <= e <= 1.06 * N, (N, e)

    def test_backward_feed_beats_forward_for_a_cold_feed(self):
        feed = naoh_feed(T=300.0)
        f = multi(3, "forward", x_product=0.3)(feed)[2]
        b = multi(3, "backward", x_product=0.3)(feed)[2]
        assert float(b["steam_economy"]) > float(f["steam_economy"])

    def test_total_area_grows_with_n(self):
        feed = naoh_feed()
        areas = [float(multi(N, "forward", U=[2500.0], x_product=0.3)(feed)[2]["total_area"])
                 for N in (1, 2, 3, 4)]
        assert all(b > a for a, b in zip(areas, areas[1:]))

    def test_parallel_feed_split_is_solved(self):
        conc, _, info = multi(3, "parallel", x_product=0.3)(naoh_feed())
        feed = naoh_feed()
        m_w_in = float(feed["F_water"]) * 18.015 / 1e3
        m_w_out = float(conc["F_water"]) * 18.015 / 1e3
        assert m_w_in - m_w_out == pytest.approx(float(info["V_total"]))
        assert float(info["x_product"]) == pytest.approx(0.3)

    @pytest.mark.parametrize("arrangement", ["forward", "backward", "parallel"])
    def test_rating_inverts_design(self, arrangement):
        feed = naoh_feed()
        _, _, d = multi(3, arrangement, x_product=0.3)(feed)
        extra = dict(feed_split=list(np.asarray(d["feed_split"]))) if arrangement == "parallel" else {}
        _, _, r = multi(3, arrangement, A=list(np.asarray(d["A"])), **extra)(feed)
        assert bool(r["converged"])
        assert float(r["x_product"]) == pytest.approx(0.3, rel=1e-6)
        assert float(r["steam"]) == pytest.approx(float(d["steam"]), rel=1e-6)

    def test_rating_with_unequal_areas_and_parallel_split(self):
        feed = naoh_feed()
        _, _, info = multi(3, "forward", A=[80.0, 100.0, 120.0])(feed)
        assert bool(info["converged"])
        np.testing.assert_allclose(np.asarray(info["A"]), [80.0, 100.0, 120.0])
        _, _, p = multi(3, "parallel", A=[100.0], feed_split=[0.5, 0.3, 0.2])(feed)
        assert bool(p["converged"])
        assert float(p["x_product"]) > 0.1

    def test_scalar_U_and_A_broadcast(self):
        a = multi(3, "forward", U=[2000.0], x_product=0.3)(naoh_feed())[2]
        b = multi(3, "forward", U=[2000.0] * 3, x_product=0.3)(naoh_feed())[2]
        assert float(a["steam"]) == pytest.approx(float(b["steam"]))

    def test_input_errors(self):
        with pytest.raises(ValueError, match="feed must be"):
            multi(3, "sideways", x_product=0.3)
        with pytest.raises(ValueError, match="exactly one"):
            multi(3, "forward")
        with pytest.raises(ValueError, match="n_effects"):
            multi(0, "forward", x_product=0.3)
        with pytest.raises(ValueError, match="entries"):
            multi(3, "forward", U=[1.0, 2.0], x_product=0.3)(naoh_feed())

    def test_in_a_flowsheet(self):
        from difflow.flowsheet import Flowsheet, Unit

        fs = Flowsheet(species_order=["water", "NaOH"])
        fs.add_feed("feed", naoh_feed())
        fs.add_unit(Unit("mee", multi(3, "forward", x_product=0.3), ["feed"], ["conc", "vap"]))
        out = fs.solve()
        assert float(out["vap"]["F_NaOH"]) == 0.0
        assert float(out["conc"]["P"]) == pytest.approx(30e3)


# =============================================================================
# Mechanical vapor recompression
# =============================================================================


class TestMVR:
    def make(self, **kw):
        base = dict(NAOH, U=1500.0, x_product=0.3, P_vapor_space=101325.0,
                    dT_drive=10.0, eta=0.75, steam_P=300e3)
        base.update(kw)
        return MechanicalVaporRecompression(MVRParams(**base))

    def test_work_is_positive_and_saves_steam(self):
        _, cond, info = self.make()(naoh_feed())
        assert float(info["W_compressor"]) > 0.0
        assert float(info["compression_ratio"]) > 1.0
        assert float(info["steam_saved"]) > 0.0
        assert float(info["steam_makeup"]) < float(info["steam_without_mvr"])
        # electricity out is far smaller than the heat it replaces
        assert float(info["W_compressor"]) < 0.5 * float(info["Q"])
        assert float(cond["F_NaOH"]) == 0.0

    def test_energy_balance(self):
        _, _, info = self.make()(naoh_feed())
        np.testing.assert_allclose(
            float(info["Q_compressor_heat"]) + float(info["Q_makeup"]), float(info["Q"]),
            rtol=1e-12)

    def test_isentropic_work_matches_ideal_gas_formula(self):
        _, _, info = self.make()(naoh_feed())
        T1, r = float(info["T_soln"]), float(info["compression_ratio"])
        w_is = 1880.0 * T1 * (r ** (0.3 / 1.3) - 1.0)
        assert float(info["specific_work"]) == pytest.approx(w_is / 0.75)

    def test_work_grows_with_drive_and_falls_with_efficiency(self):
        feed = naoh_feed()
        w = lambda **kw: float(self.make(**kw)(feed)[2]["W_compressor"])
        assert w(dT_drive=15.0) > w(dT_drive=8.0)
        assert w(eta=0.6) > w(eta=0.85)

    def test_gradient_of_work(self):
        def f(x, dT):
            p = MVRParams(**dict(NAOH, U=1500.0, x_product=x, dT_drive=dT, steam_P=300e3))
            return MechanicalVaporRecompression(p)(naoh_feed())[2]["W_compressor"]

        check_grads(f, (0.3, 10.0), order=1, modes=["rev"], rtol=1e-4)


# =============================================================================
# Gradients (check_grads, order 1, reverse)
# =============================================================================


def _steam_and_area(arrangement, N, U1, xF, Ps, design=True):
    U = [U1, 0.8 * U1, 0.6 * U1][:N]
    kw = dict(x_product=0.3) if design else dict(A=[100.0])
    m = MultiEffectEvaporator(MultiEffectEvaporatorParams(
        **NAOH, n_effects=N, feed=arrangement, U=U, P_vapor_space=30e3, steam_P=Ps, **kw))
    info = m(naoh_feed(xF=xF))[2]
    return jnp.stack([info["steam"], info["A"][0], info["x_product"]])


class TestGradients:
    X0 = (2500.0, 0.10, 300e3)

    @pytest.mark.parametrize("design", [True, False])
    def test_single_effect(self, design):
        def f(U, xF, Ps):
            kw = dict(x_product=0.3) if design else dict(A=80.0)
            p = EvaporatorParams(**NAOH, U=U, P_vapor_space=30e3, steam_P=Ps, **kw)
            info = Evaporator(p)(naoh_feed(xF=xF))[2]
            return jnp.stack([info["steam"], info["A"]])

        check_grads(f, self.X0, order=1, modes=["rev"], rtol=1e-4, atol=1e-4)

    def test_multi_effect_forward_design(self):
        f = lambda U, xF, Ps: _steam_and_area("forward", 3, U, xF, Ps)[:2]
        check_grads(f, self.X0, order=1, modes=["rev"], rtol=1e-4, atol=1e-4)

    def test_multi_effect_forward_rating(self):
        f = lambda U, xF, Ps: _steam_and_area("forward", 2, U, xF, Ps, design=False)[jnp.array([0, 2])]
        check_grads(f, self.X0, order=1, modes=["rev"], rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize("arrangement", ["backward", "parallel"])
    def test_multi_effect_other_arrangements(self, arrangement):
        f = lambda U, xF, Ps: _steam_and_area(arrangement, 3, U, xF, Ps)[:2]
        check_grads(f, self.X0, order=1, modes=["rev"], rtol=1e-4, atol=1e-4)

    def test_gradient_signs(self):
        """Steam falls with U? No: steam is set by the duty; AREA falls with U."""
        g = jax.jacrev(lambda U: _steam_and_area("forward", 2, U, 0.10, 300e3)[1])(2500.0)
        assert float(g) < 0.0
        # more steam pressure, bigger driving force, less area
        gp = jax.jacrev(lambda Ps: _steam_and_area("forward", 2, 2500.0, 0.10, Ps)[1])(300e3)
        assert float(gp) < 0.0


# =============================================================================
# Catalog / palette registration
# =============================================================================


@pytest.fixture(scope="module")
def cat():
    from difflow.catalog import catalog

    return catalog()


class TestCatalog:
    @pytest.mark.parametrize("name", ["Evaporator", "MultiEffectEvaporator",
                                      "MechanicalVaporRecompression"])
    def test_registered_declarative_and_documented(self, cat, name):
        spec = cat[name]
        assert spec.category == "evaporation" and spec.plugin == "core"
        assert spec.is_declarative and spec.is_buildable
        assert spec.ports.inlets == ["feed"]
        assert spec.ports.n_outlets == 2
        assert spec.equations and spec.assumptions and spec.references
        assert spec.numerical_method
        units = {p.name: p.units for p in spec.parameters}
        assert units["solute_MW"] == "g/mol" and units["P_vapor_space"] == "Pa"
        assert all(p.description for p in spec.parameters), [
            p.name for p in spec.parameters if not p.description]

    def test_outlet_roles(self, cat):
        assert cat["Evaporator"].ports.outlet_roles == ["concentrate", "vapor"]
        assert cat["MultiEffectEvaporator"].ports.outlet_roles == ["concentrate", "vapor"]
        assert cat["MechanicalVaporRecompression"].ports.outlet_roles == ["concentrate", "condensate"]

    def test_required_parameters(self, cat):
        assert cat["Evaporator"].required_parameters() == ["solute_species", "solute_MW"]


# =============================================================================
# Fast sanity on warnings
# =============================================================================


def test_no_unverified_warning_for_ideal_model():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnverifiedDataWarning)
        single(x_product=0.3)(naoh_feed())
