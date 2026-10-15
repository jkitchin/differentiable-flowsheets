"""Claus SRU and SCOT tail-gas treating (#472).

Literature anchors (read; see ``src/difflow_refinery/sulfur/CLAUDE.md``):

* [FL11] Ferreira & Lobo, J. Chem. Thermodyn. 43 (2011) 95,
  doi:10.1016/j.jct.2010.07.007: liquid-sulfur vapour pressure, eq 4;
  their datum 28.4 Pa at 424.79 K and their Tb 717.652 K.
* [KN97] Kohl & Nielsen, Gas Purification 5th ed. (1997) ch. 8,
  doi:10.1016/B978-088415220-0/50008-2 (excerpts): the Gamson-Elkins curve
  turns at ~1000 F (~811 K) and approaches 100% near 250 F; furnace
  1800-2500 F; thermal conversion typically 60-70%; a 4th bed adds < 1%;
  SCOT treated gas 10-400 ppmv; bed inlet temperatures.
* [AP42] U.S. EPA AP-42 section 8.13 (4/2015, and the 1995 text):
  recoveries 2 beds 92-95 (94-96), 3 beds 95-96 (96-97.5), 4 beds 96-97
  (97-98.5) %; SCOT ~99.9%.
* [EB04] El-Bishtawi & Haimour, Fuel Process. Technol. 86 (2004) 245,
  doi:10.1016/j.fuproc.2004.04.001: furnace equilibrium conversion ~75% max.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import GibbsReactor, GibbsReactorParams, make_stream
from difflow_refinery import thermochemistry as tc
from difflow_refinery.sulfur import claus as cl
from difflow_refinery.sulfur import tgtu as tg
from difflow_refinery.sulfur import vapor as v
from difflow_refinery.treating import TrayParams

jax.config.update("jax_enable_x64", True)

ELEMENTS = ("S", "H", "O", "C", "N")
# bed inlets inside [KN97] (bed 1 505-555 K, bed 2 472-494, bed 3 461-483);
# condensers inside its 140-170 C liquid-sulfur range, the last at 140 C
BEDS = (cl.ConverterParams(523.15, hydrolysis=(0.9, 0.7)), cl.ConverterParams(483.15),
        cl.ConverterParams(473.15), cl.ConverterParams(468.15))
T_COND = (443.15, 438.15, 433.15, 423.15)
TRAY = TrayParams(area=200.0, k_G=1e-5, k_L=1e-4)   # ILLUSTRATIVE tray (as test_treating)


def acid_gas(h2s=90.0):
    return make_stream({"hydrogen_sulfide": h2s, "carbon_dioxide": 5.0, "water": 4.0,
                        "methane": 1.0}, T=323.15, P=1.6e5)


def params(n_beds):
    return cl.ClausParams(converters=BEDS[:n_beds], T_condensers=T_COND[:n_beds] + (413.15,),
                          T_air=473.15)


def atoms(stream_or_vec):
    n = stream_or_vec if isinstance(stream_or_vec, jnp.ndarray) else cl.to_vec(stream_or_vec)
    return np.array([sum(float(n[i]) * tc.elements(s).get(e, 0) for i, s in enumerate(cl.SPECIES))
                     for e in ELEMENTS])


def air_atoms(air):
    return np.array([0.0, 0.0, 2 * cl.AIR_O2 * air, 0.0, 2 * (1 - cl.AIR_O2) * air])


# ---------------------------------------------------------------- vapour
class TestVapour:
    def test_ferreira_lobo(self):
        # the correlation reproduces the paper's own datum and normal boiling point
        assert float(v.total_vapor_pressure(424.79)) == pytest.approx(28.4, rel=0.02)
        assert float(v.total_vapor_pressure(717.652)) == pytest.approx(101325.0, rel=0.002)
        # and the ITS-90 sulfur point (717.764 K) to 0.2%
        assert float(v.total_vapor_pressure(717.764)) == pytest.approx(101325.0, rel=0.002)

    def test_janaf_low_and_corrected(self):
        # the raw JANAF total is 24% high at 400 K and 30% low at 700 K
        # against [FL11] (its 1 atm point is ~745 K); the corrected
        # saturation sums to the measured total
        ratio = {T: float(jnp.sum(v.janaf_saturation(T)) / v.total_vapor_pressure(T))
                 for T in (400.0, 700.0)}
        assert ratio[400.0] == pytest.approx(1.24, abs=0.02)
        assert ratio[700.0] == pytest.approx(0.70, abs=0.02)
        for T in (420.0, 500.0, 600.0, 700.0):
            assert float(jnp.sum(v.saturation_pressures(T))) == pytest.approx(
                float(v.total_vapor_pressure(T)), rel=1e-10)

    def test_s8_dominates_saturated_vapour(self):
        # S8 then S6 at condenser temperatures; S2 grows with T
        f = v.saturation_pressures(450.0) / jnp.sum(v.saturation_pressures(450.0))
        assert int(jnp.argmax(f)) == 6 and float(f[6]) > 0.6
        f7 = v.saturation_pressures(700.0) / jnp.sum(v.saturation_pressures(700.0))
        assert float(f7[0]) > float(f[0])

    def test_saturation_lies_on_gas_curve(self):
        # the liquid correction keeps saturated and gas-phase speciation one model
        G, P, T = 100.0, 1.4e5, 450.0
        sat = v.saturated(G, T, P)
        s = float(jnp.sum(v.SIZES * sat))
        np.testing.assert_allclose(v.speciate(s, G, T, P), sat, rtol=1e-9)

    def test_condense_continuous_at_dew_point(self):
        G, P, s = 100.0, 1.4e5, 1.0
        Td = float(v.dew_point(s, G, P))
        for T in (Td - 1e-6, Td + 1e-6):
            vap, liq = v.condense(s, G, T, P)
            assert float(jnp.sum(v.SIZES * vap) + liq) == pytest.approx(s, rel=1e-12)
            assert float(liq) < 1e-6
        vap, liq = v.condense(s, G, Td - 10.0, P)
        assert 0.0 < float(liq) < s

    def test_dew_point_rises_with_sulfur(self):
        Ts = [float(v.dew_point(s, 100.0, 1.4e5)) for s in (0.1, 1.0, 10.0)]
        assert Ts[0] < Ts[1] < Ts[2]

    def test_grads(self):
        check_grads(lambda T: v.condense(5.0, 100.0, T, 1.4e5)[1], (430.0,), order=1,
                    modes=["fwd", "rev"])
        check_grads(lambda s: v.condense(s, 100.0, 700.0, 1.4e5)[0], (0.5,), order=1,
                    modes=["fwd", "rev"])
        check_grads(lambda s: v.dew_point(s, 100.0, 1.4e5), (1.0,), order=1, modes=["fwd", "rev"])


# ---------------------------------------------------------------- furnace
class TestFurnace:
    def gamson_elkins(self, sulfur):
        """Equilibrium conversion of stoichiometric H2S/SO2 in air, 1 atm, all vapour."""
        names = ["hydrogen_sulfide", "sulfur_dioxide", *sulfur, "water", "nitrogen"]
        th = tc.IdealGasSet(names, cp_basis="wide")
        n2 = 0.5 * (1 - cl.AIR_O2) / cl.AIR_O2
        feed = make_stream({"hydrogen_sulfide": 2 / 3, "sulfur_dioxide": 1 / 3, "water": 1 / 3,
                            "nitrogen": n2}, T=500.0, P=101325.0)
        Ts = np.arange(400.0, 1601.0, 25.0)
        X = []
        for T in Ts:
            o, i = GibbsReactor(GibbsReactorParams(thermo=th, T_out=T))(feed)
            assert bool(i["converged"])
            X.append(1.0 - float(o["F_hydrogen_sulfide"] + o["F_sulfur_dioxide"]))
        return Ts, np.array(X)

    def test_gamson_elkins_curve(self):
        # [KN97]: minimum near 811 K, ~100% at 400 K; thermal region 60-75% [EB04]
        for S in (["sulfur_s2", "sulfur_s6", "sulfur_s8"], list(v.SULFUR_SPECIES)):
            Ts, X = self.gamson_elkins(S)
            assert 760.0 <= Ts[np.argmin(X)] <= 880.0
            assert X[0] > 0.99
            hot = X[(Ts >= 1300) & (Ts <= 1600)]
            assert np.all((hot > 0.60) & (hot < 0.78))

    def test_adiabatic_flame(self):
        # pure H2S, a third burnt, cold air: inside the 1800-2500 F furnace range [KN97]
        acid = make_stream({"hydrogen_sulfide": 100.0}, T=298.15, P=101325.0)
        air = cl.oxygen_demand(acid) / cl.AIR_O2
        out, info = cl.furnace(acid, air)
        assert bool(info["converged"])
        assert 1255.0 < float(info["T"]) < 1644.0
        np.testing.assert_allclose(atoms(out), atoms(acid) + air_atoms(air), atol=1e-9)
        assert abs(float(info["heat_duty"])) < 1e-3


# ---------------------------------------------------------------- train pieces
class TestTrain:
    def test_cool_balances(self):
        out, _ = cl.furnace(acid_gas(), 190.0, T_air=473.15)
        g, liq, q = cl.cool(out, 443.15)
        n0, n1 = cl.to_vec(out), cl.to_vec(g)
        assert float(cl.sulfur_in(n0)) == pytest.approx(float(cl.sulfur_in(n1) + liq), rel=1e-12)
        h_in = float(cl.enthalpy(n0, out["T"]))
        h_out = float(cl.enthalpy(n1, 443.15) + liq * tc.liquid_sulfur_enthalpy(443.15))
        assert h_in - h_out == pytest.approx(float(q), rel=1e-12)
        assert float(liq) > 0.5 * float(cl.sulfur_in(n0))     # thermal stage 60-70% [KN97]

    def test_train_balances(self):
        acid, air = acid_gas(), 190.0
        r = cl.claus_train(acid, air, params(3))
        assert bool(r["converged"])
        S_liq = float(r["sulfur"])
        np.testing.assert_allclose(atoms(r["tail_gas"]) + np.array([S_liq, 0, 0, 0, 0]),
                                   atoms(acid) + air_atoms(air), rtol=0, atol=1e-8)
        # every bed above its sulfur dew point; furnace inside [KN97]
        assert all(float(b["dew_margin"]) > 0 for b in r["bed"])
        assert 1255.0 < float(r["T_furnace"]) < 1644.0
        assert 0.60 < float(r["sulfur_by_condenser"][1] / r["S_in"]) < 0.78

    def test_converter_hydrolysis_and_heat(self):
        # bed 1 hydrolyses COS/CS2 as specified, then Claus runs adiabatically
        g = make_stream({"hydrogen_sulfide": 2.0, "sulfur_dioxide": 1.0, "carbonyl_sulfide": 0.1,
                         "carbon_disulfide": 0.05, "water": 30.0, "nitrogen": 60.0},
                        T=443.15, P=1.4e5)
        out, info = cl.converter(g, cl.ConverterParams(523.15, hydrolysis=(0.9, 0.7)))
        assert float(out["F_carbonyl_sulfide"]) == pytest.approx(0.01, rel=1e-10)
        assert float(out["F_carbon_disulfide"]) == pytest.approx(0.015, rel=1e-10)
        assert float(info["T_out"]) > 523.15                  # exothermic
        np.testing.assert_allclose(atoms(out), atoms(g), atol=1e-10)
        h_in = float(cl.enthalpy(cl.to_vec(g), 523.15))
        assert float(cl.enthalpy(cl.to_vec(out), info["T_out"])) == pytest.approx(h_in, abs=1e-3)

    def test_quench_saturates(self):
        from difflow.water import water_saturation_pressure
        g = make_stream({"hydrogen_sulfide": 1.0, "water": 80.0, "nitrogen": 150.0},
                        T=573.15, P=1.2e5)
        q, cond, duty = tg.quench(g, 313.15)
        y = float(q["F_water"]) / (1.0 + float(q["F_water"]) + 150.0)
        assert y == pytest.approx(float(water_saturation_pressure(313.15)) / 1.2e5, rel=1e-12)
        assert float(cond + q["F_water"]) == pytest.approx(80.0, rel=1e-12)
        assert float(duty) > 0.0


# ---------------------------------------------------------------- solved train
class TestAirControl:
    @pytest.fixture(scope="class")
    def solved(self):
        return {n: jax.jit(lambda g, n=n: cl.solve_air(g, params(n)))(acid_gas()) for n in (2, 3, 4)}

    def test_ratio_two_to_one(self, solved):
        for n, (air, r) in solved.items():
            assert bool(r["air_converged"]) and bool(r["converged"])
            t = r["tail_gas"]
            assert float(t["F_hydrogen_sulfide"]) == pytest.approx(2 * float(t["F_sulfur_dioxide"]),
                                                                   rel=1e-9)

    def test_recovery_vs_beds(self, solved):
        rec = {n: float(r["recovery"]) for n, (_, r) in solved.items()}
        # equilibrium is an upper bound: at or above the AP-42 practical
        # ranges (2015 and 1995 tops: 95/96, 96/97.5, 97/98.5 %), and within ~3%
        for n, top in ((2, 0.96), (3, 0.975), (4, 0.985)):
            assert top - 0.005 < rec[n] < top + 0.03
        assert rec[2] < rec[3] < rec[4]
        assert rec[4] - rec[3] < 0.01                         # [KN97]: 4th bed < 1%

    def test_sulfur_balance(self, solved):
        for air, r in solved.values():
            S_tail = float(cl.sulfur_in(cl.to_vec(r["tail_gas"])))
            assert float(r["sulfur"]) + S_tail == pytest.approx(90.0, rel=1e-11)

    def test_steam(self, solved):
        _, r = solved[3]
        assert float(r["steam_whb"]) > 0 and float(r["steam_condensers"]) > 0
        assert float(r["whb_duty"]) > float(jnp.sum(r["condenser_duty"]))

    def test_air_gradient_matches_fd(self):
        p = params(2)
        f = jax.jit(lambda h: cl.solve_air(dict(acid_gas(), F_hydrogen_sulfide=h), p)[0])
        g = float(jax.jit(jax.grad(lambda h: cl.solve_air(
            dict(acid_gas(), F_hydrogen_sulfide=h), p)[0]))(90.0))
        fd = (float(f(90.01)) - float(f(89.99))) / 0.02
        assert g == pytest.approx(fd, rel=1e-6)
        # below 0.5 O2 per extra H2S: the furnace splits some H2S to H2 + S,
        # which takes no O2 (~0.13 H2 per H2S here, ~0.065 O2 saved)
        assert 0.40 < g * cl.AIR_O2 < 0.5

    def test_train_gradients(self):
        # recovery vs air ratio and bed inlet temperatures (fixed air)
        def rec(x):
            beds = (cl.ConverterParams(x[1], hydrolysis=(0.9, 0.7)), cl.ConverterParams(x[2]))
            p = cl.ClausParams(converters=beds, T_condensers=(443.15, 438.15, 413.15),
                               T_air=473.15)
            return cl.claus_train(acid_gas(), x[0], p)["recovery"]

        x = jnp.array([188.0, 523.15, 483.15])
        g = jax.jit(jax.grad(rec))(x)
        f = jax.jit(rec)
        for i, h in enumerate((0.05, 0.05, 0.05)):
            e = jnp.zeros(3).at[i].set(h)
            fd = (float(f(x + e)) - float(f(x - e))) / (2 * h)
            assert float(g[i]) == pytest.approx(fd, rel=1e-4, abs=1e-9)
        # hotter beds convert less (exothermic equilibrium)
        assert float(g[2]) < 0.0


class TestSRU:
    @pytest.fixture(scope="class")
    def sru(self):
        tp = tg.TGTUParams(tray=TRAY, L=30.0, n_stages=12, P_absorber=1.2e5)
        return tp, jax.jit(lambda g: tg.solve_sru(g, params(2), tp))(acid_gas())

    def test_converged_and_recovery(self, sru):
        _, r = sru
        assert bool(r["converged"])
        assert float(r["tear_residual"]) < 1e-10
        assert float(r["recovery"]) > 0.995                   # #472 acceptance; AP-42 ~99.9%
        assert float(r["recovery"]) > float(r["claus"]["recovery"])

    def test_treated_gas_ppmv(self, sru):
        off = sru[1]["tgtu"]["off_gas"]
        tot = sum(float(x) for k, x in off.items() if k.startswith("F_"))
        ppm = 1e6 * float(cl.sulfur_in(cl.to_vec(off))) / tot
        assert 10.0 < ppm < 400.0                             # SCOT 10-400 ppmv [KN97]

    def test_hydrogenation_complete(self, sru):
        h = sru[1]["tgtu"]["hydrogenated"]
        assert float(h["F_sulfur_dioxide"]) < 1e-9
        assert float(sum(h[f"F_{s}"] for s in v.SULFUR_SPECIES)) < 1e-9
        assert float(h["F_hydrogen"]) > 0.0

    def test_sulfur_balance(self, sru):
        r = sru[1]
        # S in = liquid sulfur + S burnt in the incinerator (the recycle is internal)
        assert float(r["claus"]["sulfur"] + r["SO2_emitted"]) == pytest.approx(90.0, rel=1e-9)

    def test_recycle_small(self, sru):
        r = sru[1]
        assert 0.0 < float(r["recycle"][0]) < 0.08 * 90.0       # [KN97]: 7-8% of the load at most
        assert float(r["gain"]) < 0.5

    def test_gradient_matches_fd(self, sru):
        tp = sru[0]

        def emitted(h):
            return tg.solve_sru(dict(acid_gas(), F_hydrogen_sulfide=h), params(2), tp)["SO2_emitted"]

        vg = jax.jit(jax.value_and_grad(emitted))     # one compile serves the FD points too
        g = float(vg(90.0)[1])
        fd = (float(vg(90.001)[0]) - float(vg(89.999)[0])) / 0.002
        assert g == pytest.approx(fd, rel=1e-5)
