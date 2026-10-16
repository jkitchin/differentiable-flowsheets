"""Amine treating unit (#471): absorber, regenerator, rich flash, the closed loop.

What is validated, and against what:

* properties: the rate constants, diffusivity, solubility and viscosities
  reproduce the tables of their sources (``difflow_refinery.treating.
  properties`` docstring); the ILLUSTRATIVE heats of absorption lie
  inside the Kent-Eisenberg model's own van 't Hoff heat;
* the regenerator against the J.M. Campbell MDEA case (Moshfeghian, Tip of
  the Month, July 2018: 50 wt% MDEA, 214 kPa, 10 theoretical stages, rich
  feed 98.9 C, condenser 48.9 C, 1.1 lb steam/gal -> reboiler ~126 C,
  lean loading 0.002-0.004); its mass and energy balances close;
* the absorber: balances; limits (no gas, very many transfer units ->
  equilibrium at the top); MDEA slips CO2 and DEA does not (the reason
  MDEA is the selective solvent, Kohl & Nielsen 1997 ch. 2);
* the flash and the closed loop: every element and the water balance; the
  derivatives (``check_grads``/finite differences).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow_cc.equilibrium import acid_gas as ag  # noqa: E402
from difflow_refinery.treating import properties as pr  # noqa: E402
from difflow_refinery.treating.absorber import TrayParams, absorb  # noqa: E402
from difflow_refinery.treating.regenerator import (M_W, liquid_enthalpy,  # noqa: E402
                                                   regenerate, vapor_enthalpy)
from difflow_refinery.treating.unit import (AmineUnitParams, Contactor,  # noqa: E402
                                            lean_water, rich_flash, solve_amine_unit)

H2S, CO2, WATER = "F_hydrogen_sulfide", "F_carbon_dioxide", "F_water"
TRAY = TrayParams(area=200.0, k_G=1e-5, k_L=1e-4)   # ILLUSTRATIVE tray


def fuel_gas(h2s=20.0, co2=20.0):
    return {"F_methane": 700.0, "F_ethane": 150.0, "F_hydrogen": 110.0, H2S: h2s, CO2: co2,
            "T": 313.15, "P": 7e5}


def lean(amine, w, L=60.0, aC=1e-3, aS=5e-3, T=313.15):
    return dict(L=L, W=lean_water(amine, L, w), CO2=L * aC, H2S=L * aS, T=T)


# ---------------------------------------------------------------- properties
class TestProperties:
    def test_mdea_k2_versteeg_table3(self):
        # Versteeg & van Swaaij 1988 CES Table 3 (m3/kmol/s); eq fit within 7%
        k2 = pr.amine_properties("MDEA").k2
        for T, k in ((298.0, 4.3), (308.0, 7.7), (318.0, 12.4)):
            assert float(k2(T)) * 1e3 == pytest.approx(k, rel=0.07)
        # Blauwhoff et al. 1983: 4.8 L/mol/s at 25 C
        assert float(k2(298.15)) * 1e3 == pytest.approx(4.8, rel=0.10)

    def test_dea_k2_blanc_demarais(self):
        # log10 k2 = 10.4493 - 2274.5/T (L/mol/s): 662 at 25 C
        k2 = pr.amine_properties("DEA").k2
        assert float(k2(298.15)) * 1e3 == pytest.approx(10 ** (10.4493 - 2274.5 / 298.15), rel=1e-12)
        assert float(k2(298.15)) * 1e3 == pytest.approx(662.0, rel=0.01)
        # two orders of magnitude above MDEA: the root of MDEA's selectivity
        assert float(k2(313.15) / pr.amine_properties("MDEA").k2(313.15)) > 100.0

    def test_water_diffusivity_and_solubility(self):
        # Versteeg & van Swaaij 1988 JCED eqs 4, 8: 1.93e-9 m2/s; He 2976 Pa m3/mol at 25 C
        assert float(pr.co2_diffusivity("MDEA", 0.0, 298.15)) == pytest.approx(1.93e-9, rel=0.01)
        assert float(pr.co2_henry_solution("MDEA", 0.0, 298.15)) == pytest.approx(2976.0, rel=0.01)
        assert float(pr.co2_henry_solution("MDEA", 0.0, 313.15)) == pytest.approx(4133.0, rel=0.01)

    @pytest.mark.parametrize("T,eta", [(293.15, 1.0016), (298.15, 0.89002), (313.15, 0.65273),
                                       (333.15, 0.46604), (353.15, 0.35405), (373.15, 0.28158),
                                       (393.15, 0.23203)])
    def test_water_viscosity_iapws2008(self, T, eta):
        # Huber et al. 2009 (IAPWS 2008), evaluated with CoolProp
        assert float(pr.water_viscosity(T)) == pytest.approx(eta, rel=0.002)

    @pytest.mark.parametrize("w,T,eta", [
        (0.3, 298.15, 3.136), (0.3, 313.15, 1.995), (0.3, 323.15, 1.544),
        (0.4, 298.15, 5.290), (0.4, 313.15, 3.170), (0.4, 323.15, 2.376),
        (0.5, 298.15, 9.323), (0.5, 313.15, 5.152), (0.5, 323.15, 3.683)])
    def test_mdea_viscosity_karunarathne(self, w, T, eta):
        # Karunarathne, Eimer & Oi 2020, Table A5
        assert float(pr.solution_viscosity("MDEA", w, T)) == pytest.approx(eta, rel=0.015)

    @pytest.mark.parametrize("w,eta", [(0.20, 1.86), (0.25, 2.33), (0.30, 2.92)])
    def test_dea_viscosity_versteeg(self, w, eta):
        assert float(pr.solution_viscosity("DEA", w, 298.15)) == pytest.approx(eta, rel=0.01)

    @pytest.mark.parametrize("name,H0,B", [("methane", 1.4e-5, 1600.0), ("ethane", 1.9e-5, 2400.0),
                                           ("propane", 1.5e-5, 2700.0), ("hydrogen", 7.8e-6, 530.0)])
    def test_henry_sander(self, name, H0, B):
        assert float(pr.dissolved_gas(name, 298.15)) == pytest.approx(H0, rel=1e-12)
        assert float(pr.dissolved_gas(name, 323.15)) == pytest.approx(
            H0 * math.exp(B * (1 / 323.15 - 1 / 298.15)), rel=1e-12)
        assert float(pr.dissolved_gas("argon_unknown", 300.0)) == 0.0

    @pytest.mark.parametrize("amine,w", [("MDEA", 0.45), ("DEA", 0.30)])
    def test_heats_inside_kent_eisenberg_van_t_hoff(self, amine, w):
        """The constant heats of absorption (ILLUSTRATIVE) must agree with
        the equilibrium model's own -R dln p/d(1/T) over 313-383 K."""
        am = ag.ke_amine(amine)
        m = w / (1 - w) / (am.MW * 1e-3)
        props = pr.amine_properties(amine)
        R = 8.314462618
        for gas, dH in (("p_CO2", props.dH_CO2), ("p_H2S", props.dH_H2S)):
            vals = []
            for a in (0.05, 0.2, 0.4):
                for T in (313.15, 353.15, 383.15):
                    aC, aS = (a, 1e-6) if gas == "p_CO2" else (1e-6, a)
                    f = lambda iT: jnp.log(ag.ke_speciate(am, 1 / iT, m, aC, aS)[gas])  # noqa: E731,B023
                    vals.append(-R * float(jax.grad(f)(1 / T)))
            assert 0.75 * min(vals) <= dH <= 1.1 * max(vals), (gas, dH, min(vals), max(vals))

    def test_unknown_amine(self):
        with pytest.raises(ValueError, match="no treating properties"):
            pr.amine_properties("MEA")


# ---------------------------------------------------------------- regenerator
CAMPBELL = dict(P=2.14e5, n_stages=10, T_condenser=48.9 + 273.15)


def campbell_rich(L=100.0):
    # 50 wt% MDEA, rich loading 0.4 (inside 0.28-0.54), feed 98.9 C
    return dict(L=L, W=lean_water("MDEA", L, 0.5), CO2=0.2 * L, H2S=0.2 * L, T=98.9 + 273.15)


def steam_ratio_duty(L, lb_per_gal):
    """Reboiler duty (W) of ``lb_per_gal`` of 50 psig steam per US gallon of solution."""
    from difflow.water import water_latent_heat

    mass = L * ag.ke_amine("MDEA").MW * 1e-3 * 2.0          # kg/s at 50 wt%
    gal = mass / pr.solution_density("MDEA", 0.5) / 3.785411784e-3
    return lb_per_gal * gal * 0.45359237 * float(water_latent_heat(147.7 + 273.15))


@pytest.fixture(scope="module")
def campbell():
    rich = campbell_rich()
    return rich, regenerate("MDEA", rich, Q=steam_ratio_duty(100.0, 1.1), **CAMPBELL)


class TestRegenerator:
    def test_campbell_mdea_case(self, campbell):
        _, r = campbell
        assert bool(r.converged)
        # reboiler ~126 C at 214 kPa; lean loading 0.002-0.004 at 1.1 lb/gal
        assert float(r.T[-1]) - 273.15 == pytest.approx(126.0, abs=1.5)
        lean = float(r.alpha_CO2[-1] + r.alpha_H2S[-1])
        assert 0.0015 < lean < 0.005

    def test_balances(self, campbell):
        rich, r = campbell
        a = r.acid_gas
        assert float(a["CO2"] + r.lean["CO2"]) == pytest.approx(rich["CO2"], rel=1e-10)
        assert float(a["H2S"] + r.lean["H2S"]) == pytest.approx(rich["H2S"], rel=1e-10)
        assert float(a["water"] + r.lean["W"]) == pytest.approx(rich["W"], rel=1e-10)

    def test_energy_balance(self, campbell):
        rich, r = campbell
        am = ag.ke_amine("MDEA")
        Q = steam_ratio_duty(100.0, 1.1)
        H_in = liquid_enthalpy(am, rich["L"], rich["W"], rich["CO2"], rich["H2S"], rich["T"]) + Q
        ln = r.lean
        a = r.acid_gas
        H_out = (liquid_enthalpy(am, ln["L"], ln["W"], ln["CO2"], ln["H2S"], ln["T"])
                 + vapor_enthalpy(a["water"], a["CO2"], a["H2S"], a["T"]) + r.condenser_duty)
        assert float(H_out) == pytest.approx(float(H_in), rel=1e-8)

    def test_trace_co2_is_resolved(self):
        """A trace of CO2 (an absent one held at a floor) is stripped like a
        real one: each acid gas balances on its own feed, not the total, so
        the trace is resolved and the bottoms loading stays positive (#475)."""
        rich = dict(L=250.0, W=lean_water("MDEA", 250.0, 0.45), CO2=2.5e-10, H2S=31.2, T=380.0)
        r = regenerate("MDEA", rich, Q=1.2e7, P=1.9e5, n_stages=8, T_condenser=318.15)
        assert bool(r.converged)
        assert 0 < float(r.lean["CO2"]) < 1e-3 * rich["CO2"]
        assert float(r.lean["CO2"] + r.acid_gas["CO2"]) == pytest.approx(rich["CO2"], rel=1e-9)
        assert float(r.lean["H2S"] + r.acid_gas["H2S"]) == pytest.approx(rich["H2S"], rel=1e-9)

    def test_more_steam_strips_more(self):
        rich = campbell_rich()
        loads = []
        for s in (0.8, 1.1, 1.5):
            r = regenerate("MDEA", rich, Q=steam_ratio_duty(100.0, s), **CAMPBELL)
            assert bool(r.converged)
            loads.append(float(r.alpha_H2S[-1]))
        assert loads[0] > loads[1] > loads[2]

    def test_gradient_wrt_duty(self):
        rich = campbell_rich()
        Q0 = steam_ratio_duty(100.0, 1.1)

        def f(q):
            return jnp.log(regenerate("MDEA", rich, Q=q * 1e6, **CAMPBELL).lean["H2S"])

        g = float(jax.grad(f)(Q0 / 1e6))
        h = 0.02
        fd = (float(f(Q0 / 1e6 + h)) - float(f(Q0 / 1e6 - h))) / (2 * h)
        assert g < 0
        assert g == pytest.approx(fd, rel=2e-3)


# ---------------------------------------------------------------- absorber
@pytest.fixture(scope="module")
def absorbers():
    out = {}
    for amine, w in (("MDEA", 0.45), ("DEA", 0.25)):
        out[amine] = absorb(amine, fuel_gas(), lean(amine, w), 12, TRAY)
    return out


class TestAbsorber:
    def test_converged(self, absorbers):
        for r in absorbers.values():
            assert bool(r.converged)

    def test_acid_gas_balances(self, absorbers):
        g = fuel_gas()
        for amine, w in (("MDEA", 0.45), ("DEA", 0.25)):
            r, ln = absorbers[amine], lean(amine, w)
            for k, s in ((H2S, "H2S"), (CO2, "CO2")):
                assert float(r.sweet[k] + r.rich[s]) == pytest.approx(g[k] + ln[s], rel=1e-10)
            for n in ("methane", "ethane", "hydrogen"):
                assert float(r.sweet[f"F_{n}"] + r.dissolved[n]) == pytest.approx(g[f"F_{n}"], rel=1e-12)
            assert float(r.sweet[WATER] + r.rich["W"]) == pytest.approx(ln["W"], rel=1e-12)

    def test_mdea_selective(self, absorbers):
        """MDEA slips most of the CO2 (slow, base-catalysed reaction); DEA
        (carbamate, k2 ~ 100x) does not; both hold H2S."""
        slip = {a: float(r.sweet[CO2]) / 20.0 for a, r in absorbers.items()}
        assert slip["MDEA"] > 0.8
        assert slip["DEA"] < 0.6
        Ha = {a: float(jnp.mean(r.hatta)) for a, r in absorbers.items()}
        assert Ha["DEA"] > 5 * Ha["MDEA"]
        # H2S is gas-film controlled: the same efficiency in both
        np.testing.assert_allclose(np.asarray(absorbers["MDEA"].E_H2S),
                                   np.asarray(absorbers["DEA"].E_H2S), rtol=0.02)

    def test_efficiency_closed_form(self, absorbers):
        # H2S: E = 1 - exp(-k_G a P / G) with G the gas leaving the stage
        # leaving the stage (960 mol/s of non-acid carrier through the column)
        r = absorbers["DEA"]
        Gs = 960.0 / (1.0 - r.y_CO2 - r.y_H2S)
        E = 1.0 - jnp.exp(-TRAY.k_G * TRAY.area * 7e5 / Gs)
        np.testing.assert_allclose(np.asarray(r.E_H2S), np.asarray(E), rtol=1e-10)

    def test_equilibrium_limit(self):
        """With an enormous transfer area every stage is an equilibrium
        stage: the gas leaving the top is in equilibrium with the lean."""
        am = ag.ke_amine("DEA")
        big = TrayParams(area=1e6, k_G=1e-3, k_L=1e-2)
        ln = lean("DEA", 0.25)
        r = absorb("DEA", fuel_gas(), ln, 6, big)
        assert bool(r.converged)
        assert float(jnp.min(r.E_H2S)) > 1 - 1e-12
        m = ln["L"] / (ln["W"] * M_W)
        sp = ag.ke_speciate(am, 313.15, m, float(r.alpha_CO2[0]), float(r.alpha_H2S[0]))
        assert float(r.y_H2S[0]) == pytest.approx(float(sp["p_H2S"]) / 7e5, rel=1e-6)

    def test_no_acid_gas(self):
        """A gas with no acid gas STRIPS the lean solvent's (it is the lean
        back-pressure that sets a treated gas's floor); balances still close."""
        ln = lean("MDEA", 0.45, aC=1e-4, aS=1e-4)
        r = absorb("MDEA", fuel_gas(h2s=0.0, co2=0.0), ln, 4, TRAY)
        assert bool(r.converged)
        assert float(r.rich["H2S"]) < ln["H2S"]
        assert float(r.sweet[H2S] + r.rich["H2S"]) == pytest.approx(ln["H2S"], rel=1e-8)

    def test_gradient(self):
        ln = lean("MDEA", 0.45)

        def f(x):
            g = fuel_gas(h2s=x[0], co2=x[1])
            return jnp.log(absorb("MDEA", g, dict(ln, L=x[2]), 6, TRAY).sweet[H2S])

        check_grads(f, (jnp.array([20.0, 20.0, 60.0]),), order=1, modes=["fwd"], atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------- flash
class TestRichFlash:
    def test_balances_and_equilibrium(self):
        am = ag.ke_amine("MDEA")
        rich = dict(L=60.0, W=lean_water("MDEA", 60.0, 0.45), CO2=2.0, H2S=18.0, T=318.15)
        dis = {"methane": jnp.asarray(0.1), "hydrogen": jnp.asarray(0.01)}
        liq, vap, info = rich_flash(am, rich, dis, 5e5)
        assert bool(info["converged"])
        assert float(liq["CO2"] + vap[CO2]) == pytest.approx(2.0, rel=1e-10)
        assert float(liq["H2S"] + vap[H2S]) == pytest.approx(18.0, rel=1e-10)
        assert float(liq["W"] + vap[WATER]) == pytest.approx(rich["W"], rel=1e-12)
        V = sum(float(vap[k]) for k in vap if k.startswith("F_"))
        sp = ag.ke_speciate(am, 318.15, 60.0 / (float(liq["W"]) * M_W),
                            float(liq["CO2"]) / 60.0, float(liq["H2S"]) / 60.0)
        assert float(vap[H2S]) / V * 5e5 == pytest.approx(float(sp["p_H2S"]), rel=1e-4)


    def test_absent_acid_gas_is_exact(self):
        """The flash holds an absent acid gas at a trace (it solves in log
        loadings) and takes the trace back out: none is made (#475)."""
        am = ag.ke_amine("MDEA")
        rich = dict(L=60.0, W=lean_water("MDEA", 60.0, 0.45), CO2=0.0, H2S=18.0, T=318.15)
        liq, vap, info = rich_flash(am, rich, {"methane": jnp.asarray(0.1)}, 5e5)
        assert bool(info["converged"])
        assert abs(float(liq["CO2"] + vap[CO2])) < 1e-20
        assert float(liq["H2S"] + vap[H2S]) == pytest.approx(18.0, rel=1e-10)


# ---------------------------------------------------------------- closed loop
LOOP = AmineUnitParams("MDEA", 0.45, (Contactor("fuel", 120.0, 8, TRAY),), Q_reboiler=5e6,
                       n_regenerator=6)


def loop_gas():
    g = fuel_gas()
    g[WATER] = 1.0
    return {"fuel": g}


def _sweet_h2s(x, gas):
    r = solve_amine_unit(LOOP, gas, circulation=(x[0],), Q_reboiler=x[1] * 1e6)
    return jnp.log(r.sweet["fuel"][H2S]), r


#: The loop at circulation x[0] mol/s and duty x[1] MW with d ln(sweet H2S)/dx,
#: from one forward-mode program: one compile serves the fixture, the gradient
#: test and the CO2-free gas (the gas is traced, so only its keys shape it).
_loop_and_grad_jit = jax.jit(jax.jacfwd(_sweet_h2s, has_aux=True))


def _loop_and_grad(gas):
    J, r = _loop_and_grad_jit(jnp.array([120.0, 5.0]), gas)
    return r, J


@pytest.fixture(scope="module")
def loop_and_grad():
    return _loop_and_grad(loop_gas())


@pytest.fixture(scope="module")
def loop(loop_and_grad):
    return loop_and_grad[0]


class TestAmineUnit:
    def test_converged(self, loop):
        assert bool(loop.converged)

    def test_sulfur_and_carbon_balances(self, loop):
        g = loop_gas()["fuel"]
        out_S = loop.sweet["fuel"][H2S] + loop.acid_gas[H2S] + loop.flash_gas[H2S]
        out_C = loop.sweet["fuel"][CO2] + loop.acid_gas[CO2] + loop.flash_gas[CO2]
        assert float(out_S) == pytest.approx(g[H2S], rel=1e-9)
        assert float(out_C) == pytest.approx(g[CO2], rel=1e-9)

    def test_water_balance(self, loop):
        g = loop_gas()["fuel"]
        out = loop.sweet["fuel"][WATER] + loop.acid_gas[WATER] + loop.flash_gas[WATER]
        assert float(out) == pytest.approx(g[WATER] + float(loop.makeup_water), rel=1e-9)

    def test_lean_is_the_regenerated_solvent(self, loop):
        assert float(loop.lean["H2S"]) == pytest.approx(float(loop.regenerator.lean["H2S"]), rel=1e-12)
        assert float(loop.lean["W"]) == pytest.approx(lean_water("MDEA", 120.0, 0.45), rel=1e-12)

    def test_gradients(self, loop_and_grad):
        """tests/reference/gradient_checks.py compares it with FD."""
        assert np.all(np.isfinite(np.asarray(loop_and_grad[1])))

    def test_gas_without_co2(self):
        """A refinery fuel gas carries H2S and no CO2 (hydrotreater off-gas):
        the loop's CO2 is then a pure decay with no source, the case that
        failed to converge before #475. Balances still close."""
        g = dict(loop_gas()["fuel"], **{CO2: 0.0})   # same keys: reuses the compiled loop
        r, _ = _loop_and_grad({"fuel": g})
        assert bool(r.converged)
        out_S = r.sweet["fuel"][H2S] + r.acid_gas[H2S] + r.flash_gas[H2S]
        assert float(out_S) == pytest.approx(g[H2S], rel=1e-9)
        out_C = r.sweet["fuel"][CO2] + r.acid_gas[CO2] + r.flash_gas[CO2]
        assert abs(float(out_C)) < 1e-12
        assert float(r.sweet["fuel"][H2S]) < 1e-2 * g[H2S]
