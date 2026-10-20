"""Claus SRU and SCOT tail-gas treating (#472).

Smoke, gradient-finite and regression tests only (the test policy in
CLAUDE.md): the vapour model, furnace, train pieces, air-demand control and
the closed SRU converge, close their element and energy balances, and have
finite derivatives. The literature comparisons (Ferreira & Lobo, Kohl &
Nielsen, AP-42) are in ``src/difflow_refinery/sulfur/CLAUDE.md`` and
``tests/reference/refinery_checks.py``, not per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import make_stream
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
    def test_corrected_saturation_sums_to_the_total(self):
        for T in (420.0, 500.0, 600.0, 700.0):
            assert float(jnp.sum(v.saturation_pressures(T))) == pytest.approx(
                float(v.total_vapor_pressure(T)), rel=1e-10)

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

    def test_grads(self):
        for g in (jax.grad(lambda T: v.condense(5.0, 100.0, T, 1.4e5)[1])(430.0),
                  jax.jacfwd(lambda s: v.condense(s, 100.0, 700.0, 1.4e5)[0])(0.5),
                  jax.grad(lambda s: v.dew_point(s, 100.0, 1.4e5))(1.0)):
            assert np.all(np.isfinite(np.asarray(g)))


# ---------------------------------------------------------------- furnace
class TestFurnace:
    def test_adiabatic_flame(self):
        # pure H2S, a third burnt, cold air: converges, closes elements and energy
        acid = make_stream({"hydrogen_sulfide": 100.0}, T=298.15, P=101325.0)
        air = cl.oxygen_demand(acid) / cl.AIR_O2
        out, info = cl.furnace(acid, air)
        assert bool(info["converged"])
        np.testing.assert_allclose(atoms(out), atoms(acid) + air_atoms(air), atol=1e-9)
        assert abs(float(info["heat_duty"])) < 1e-3

    @pytest.mark.parametrize("co2", [1e-27, 1e-9])
    def test_carbon_trace(self, co2):
        # an amine regenerator passes a 1e-27 CO2 trace; its own balance row
        # used to throw the element-potential Newton off (furnace at 411 K)
        acid = make_stream({"hydrogen_sulfide": 50.0, "water": 5.0, "carbon_dioxide": co2},
                           T=330.0, P=1.6e5)
        air = cl.oxygen_demand(acid) / cl.AIR_O2
        out, info = cl.furnace(acid, air, T_air=473.15)
        ref, _ = cl.furnace(dict(acid, F_carbon_dioxide=0.0), air, T_air=473.15)
        assert bool(info["converged"])
        assert float(info["T"]) == pytest.approx(float(ref["T"]), rel=1e-9)


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

    def test_train_balances(self):
        acid, air = acid_gas(), 190.0
        r = cl.claus_train(acid, air, params(3))
        assert bool(r["converged"])
        S_liq = float(r["sulfur"])
        np.testing.assert_allclose(atoms(r["tail_gas"]) + np.array([S_liq, 0, 0, 0, 0]),
                                   atoms(acid) + air_atoms(air), rtol=0, atol=1e-8)

    def test_converter_hydrolysis_and_heat(self):
        # bed 1 hydrolyses COS/CS2 as specified, then Claus runs adiabatically
        g = make_stream({"hydrogen_sulfide": 2.0, "sulfur_dioxide": 1.0, "carbonyl_sulfide": 0.1,
                         "carbon_disulfide": 0.05, "water": 30.0, "nitrogen": 60.0},
                        T=443.15, P=1.4e5)
        out, info = cl.converter(g, cl.ConverterParams(523.15, hydrolysis=(0.9, 0.7)))
        assert float(out["F_carbonyl_sulfide"]) == pytest.approx(0.01, rel=1e-10)
        assert float(out["F_carbon_disulfide"]) == pytest.approx(0.015, rel=1e-10)
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
        """Two beds; the air's derivative in H2S comes from the same compile.
        tests/reference/refinery_checks.py runs 2-4 beds against AP-42."""
        def air(h):
            a, r = cl.solve_air(dict(acid_gas(), F_hydrogen_sulfide=h), params(2))
            return a, (a, r)

        (_, (a, r)), g = jax.jit(jax.value_and_grad(air, has_aux=True))(90.0)
        return a, r, g

    def test_ratio_two_to_one(self, solved):
        _, r, _ = solved
        assert bool(r["air_converged"]) and bool(r["converged"])
        t = r["tail_gas"]
        assert float(t["F_hydrogen_sulfide"]) == pytest.approx(2 * float(t["F_sulfur_dioxide"]),
                                                               rel=1e-9)

    def test_air_gradient(self, solved):
        """tests/reference/gradient_checks.py compares it with FD."""
        assert np.isfinite(float(solved[2]))

    def test_train_gradients(self):
        # recovery vs air ratio and bed inlet temperatures (fixed air)
        def rec(x):
            beds = (cl.ConverterParams(x[1], hydrolysis=(0.9, 0.7)), cl.ConverterParams(x[2]))
            p = cl.ClausParams(converters=beds, T_condensers=(443.15, 438.15, 413.15),
                               T_air=473.15)
            return cl.claus_train(acid_gas(), x[0], p)["recovery"]

        g = jax.jit(jax.grad(rec))(jnp.array([188.0, 523.15, 483.15]))
        assert np.all(np.isfinite(np.asarray(g)))


class TestSRU:
    @pytest.fixture(scope="class")
    def sru(self):
        """The smallest closed SRU (one bed, three absorber stages) and
        d(SO2 emitted)/d(H2S), from one compile."""
        tp = tg.TGTUParams(tray=TRAY, L=30.0, n_stages=3, P_absorber=1.2e5)

        def emitted(h):
            r = tg.solve_sru(dict(acid_gas(), F_hydrogen_sulfide=h), params(1), tp)
            return r["SO2_emitted"], r

        (_, r), g = jax.jit(jax.value_and_grad(emitted, has_aux=True))(90.0)
        return r, g

    def test_smoke(self, sru):
        r, _ = sru
        assert bool(r["converged"])
        for k in ("recovery", "SO2_emitted", "air"):
            assert np.isfinite(float(r[k])), k

    def test_gradient_is_finite(self, sru):
        """tests/reference/gradient_checks.py compares it with FD."""
        assert np.isfinite(float(sru[1]))

    def test_regression(self, sru):
        """Values recorded before solve_sru / solve_air were rewritten to trace
        the train once (difflow_refinery.implicit); unchanged to round-off."""
        r, g = sru
        assert float(r["SO2_emitted"]) == pytest.approx(2.474353852243781, rel=1e-9)
        assert float(r["air"]) == pytest.approx(203.5544845268881, rel=1e-9)
        assert float(g) == pytest.approx(0.06491886281106696, rel=1e-7)
