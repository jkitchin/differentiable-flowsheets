"""Amine treating unit (#471): absorber, regenerator, rich flash, the closed loop.

Smoke, gradient-finite and regression tests only (the test policy in
CLAUDE.md): each piece converges, closes its balances, and has a finite
derivative. The comparisons with published data (the J.M. Campbell MDEA
regenerator case, the property tables) are not per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow_cc.equilibrium import acid_gas as ag  # noqa: E402
from difflow_refinery.treating import properties as pr  # noqa: E402
from difflow_refinery.treating.absorber import TrayParams, absorb  # noqa: E402
from difflow_refinery.treating.regenerator import (liquid_enthalpy,  # noqa: E402
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
    @pytest.mark.parametrize("amine,w", [("MDEA", 0.45), ("DEA", 0.30)])
    def test_properties_smoke(self, amine, w):
        p = pr.amine_properties(amine)
        for v in (p.k2(313.15), pr.co2_diffusivity(amine, w, 313.15), pr.co2_henry_solution(amine, w, 313.15),
                  pr.solution_viscosity(amine, w, 313.15), pr.water_viscosity(313.15), p.dH_CO2, p.dH_H2S):
            assert np.isfinite(float(v)) and float(v) != 0.0
        for name in ("methane", "ethane", "propane", "hydrogen"):
            assert float(pr.dissolved_gas(name, 313.15)) > 0.0
        assert float(pr.dissolved_gas("argon_unknown", 300.0)) == 0.0

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


#: The Campbell column with the feed and duty traced: one compile serves the
#: case, the steam sweep and the finite differences of the gradient.
_campbell_regen = jax.jit(lambda rich, Q: regenerate("MDEA", rich, Q=Q, **CAMPBELL))


@pytest.fixture(scope="module")
def campbell():
    rich = campbell_rich()
    return rich, _campbell_regen(rich, steam_ratio_duty(100.0, 1.1))


class TestRegenerator:
    def test_smoke(self, campbell):
        _, r = campbell
        assert bool(r.converged)
        assert np.all(np.isfinite(np.asarray(r.T))) and np.all(np.isfinite(np.asarray(r.alpha_H2S)))

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
        r = _campbell_regen(rich, 1.2e7)              # the Campbell column: no new compile
        assert bool(r.converged)
        assert 0 < float(r.lean["CO2"]) < 1e-3 * rich["CO2"]
        assert float(r.lean["CO2"] + r.acid_gas["CO2"]) == pytest.approx(rich["CO2"], rel=1e-9)
        assert float(r.lean["H2S"] + r.acid_gas["H2S"]) == pytest.approx(rich["H2S"], rel=1e-9)

    def test_gradient_wrt_duty(self):
        rich = campbell_rich()
        g = jax.grad(lambda q: jnp.log(_campbell_regen(rich, q * 1e6).lean["H2S"]))(
            steam_ratio_duty(100.0, 1.1) / 1e6)
        assert np.isfinite(float(g))


# ---------------------------------------------------------------- absorber
#: The absorber with the gas and lean traced (amine, stages and tray static):
#: the 12-stage MDEA column compiles once for the fixture, the no-acid-gas
#: case and the gradient.
_absorb = jax.jit(absorb, static_argnums=(0, 3, 4))


@pytest.fixture(scope="module")
def absorbers():
    out = {}
    for amine, w in (("MDEA", 0.45), ("DEA", 0.25)):
        out[amine] = _absorb(amine, fuel_gas(), lean(amine, w), 12, TRAY)   # both kinetics paths
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

    def test_no_acid_gas(self):
        """A gas with no acid gas STRIPS the lean solvent's (it is the lean
        back-pressure that sets a treated gas's floor); balances still close."""
        ln = lean("MDEA", 0.45, aC=1e-4, aS=1e-4)
        r = _absorb("MDEA", fuel_gas(h2s=0.0, co2=0.0), ln, 12, TRAY)
        assert bool(r.converged)
        assert float(r.sweet[H2S] + r.rich["H2S"]) == pytest.approx(ln["H2S"], rel=1e-8)

    def test_gradient(self):
        ln = lean("MDEA", 0.45)

        def f(x):
            g = fuel_gas(h2s=x[0], co2=x[1])
            return jnp.log(_absorb("MDEA", g, dict(ln, L=x[2]), 12, TRAY).sweet[H2S])

        assert np.all(np.isfinite(np.asarray(jax.jacfwd(f)(jnp.array([20.0, 20.0, 60.0])))))


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
#: the smallest loop that exercises it: one three-stage contactor, a
#: three-stage regenerator
LOOP = AmineUnitParams("MDEA", 0.45, (Contactor("fuel", 120.0, 3, TRAY),), Q_reboiler=5e6,
                       n_regenerator=3)


@pytest.fixture(scope="module")
def loop_and_grad():
    """The loop and d ln(sweet H2S)/d circulation from one forward-mode compile."""
    g = fuel_gas()
    g[WATER] = 1.0

    def f(c):
        r = solve_amine_unit(LOOP, {"fuel": g}, circulation=(c,))
        return jnp.log(r.sweet["fuel"][H2S]), r

    (_, r), (dlog, _) = jax.jit(lambda c: jax.jvp(f, (c,), (1.0,)))(120.0)
    return r, dlog


class TestAmineUnit:
    def test_smoke(self, loop_and_grad):
        r, _ = loop_and_grad
        assert bool(r.converged)
        for v in (r.sweet["fuel"][H2S], r.acid_gas[H2S], r.makeup_water, r.lean["H2S"]):
            assert np.isfinite(float(v))

    def test_gradient_is_finite(self, loop_and_grad):
        """tests/reference/gradient_checks.py compares it with FD."""
        assert np.isfinite(float(loop_and_grad[1]))
