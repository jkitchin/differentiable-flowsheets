"""Smoke, gradient and validation tests for the Crystallizer and its library functions.

The solubility curve and kinetic constants are synthetic.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import (
    Crystallizer,
    CrystallizerParams,
    Solubility,
    crystal_balance,
    crystal_yield,
    fit_msmpr,
    growth_rate,
    make_stream,
    msmpr_distribution,
    msmpr_dominant_size,
    msmpr_from_magma,
    msmpr_mass_cv,
    msmpr_mass_median_size,
    msmpr_moments,
    msmpr_steady_state,
    nucleation_rate,
    sieve_population_density,
    supersaturation,
)

K0 = 273.15
M_S, M_W = 142.0, 18.015
SOL0 = Solubility.polynomial([0.35, 0.0075], hydrate_waters=10.0, solute_MW=M_S, solvent_MW=M_W)
KIN = (2.0e-6, 1.5, 1.0e9, 2.0, 0.6)     # synthetic k_g, g, k_b, b, j
RHO_C, KV, RHO_L = 1460.0, math.pi / 6, 1250.0
N0, G, TAU = 3.0e9, 2.0e-8, 1800.0


def feed_stream(S0=0.60, W0=1.0, T=K0 + 60.0):
    return make_stream({"Na2SO4": S0 * 1000.0 / M_S, "H2O": W0 * 1000.0 / M_W}, T=T, P=1.0e5)


def params(**kw):
    base = dict(solute="Na2SO4", solute_MW=M_S, mode="cooling", T=K0 + 5.0,
                solubility_coeffs=[0.35, 0.0075], hydrate_waters=10.0)
    base.update(kw)
    return CrystallizerParams(**base)


def test_solubility_models_smoke():
    T = K0 + 20.0
    for s in (Solubility.polynomial([10.0, 0.5], per_100g=True),
              Solubility.vant_hoff(2.0, -1000.0),
              Solubility.table([280.0, 300.0, 320.0], [0.1, 0.25, 0.6]), SOL0):
        assert np.isfinite(float(s.c_sat(T)))
    fs, fw = SOL0.crystal_fractions(300.0)
    assert float(fs + fw) == pytest.approx(1.0)


def test_solubility_validation():
    with pytest.raises(ValueError):
        Solubility("bogus", coeffs=[1.0])
    with pytest.raises(ValueError):
        Solubility("polynomial")
    with pytest.raises(ValueError):
        Solubility.polynomial([0.1], upper=Solubility.polynomial([0.2]))
    with pytest.raises(ValueError):
        supersaturation(0.3, 300.0, Solubility.polynomial([0.2]), "bogus")


def test_solubility_is_a_pytree_and_jits():
    leaves, tree = jax.tree_util.tree_flatten(SOL0)
    s2 = jax.tree_util.tree_unflatten(tree, leaves)
    assert float(jax.jit(lambda sol, T: sol.c_sat(T))(s2, 300.0)) == pytest.approx(
        float(SOL0.c_sat(300.0)))


def test_transition_gradient_is_finite_through_the_switch():
    upper = Solubility.polynomial([0.45, 0.002], hydrate_waters=0.0, solute_MW=M_S, solvent_MW=M_W)
    s = Solubility.polynomial([0.05, 0.015], hydrate_waters=10.0, solute_MW=M_S, solvent_MW=M_W,
                              upper=upper, T_transition=K0 + 32.0, width=0.5)
    g = jax.grad(lambda T: crystal_yield(0.3, 1.0, T, s))(K0 + 32.0)
    assert np.isfinite(float(g))


def test_crystal_balance_smoke_and_grad():
    b = crystal_balance(0.5, 1.0, K0 + 5.0, SOL0, 0.1)
    assert float(b["crystal_solute"] + b["liquor_solute"]) == pytest.approx(0.5, abs=1e-12)
    g = jax.grad(crystal_yield, argnums=(0, 1, 2, 4))(0.5, 1.0, K0 + 5.0, SOL0, 0.1)
    assert all(np.isfinite(float(x)) for x in g)


def test_undersaturated_yield_is_zero_with_finite_gradient():
    assert float(crystal_yield(0.1, 1.0, K0 + 25.0, SOL0)) == pytest.approx(0.0, abs=1e-12)
    g = jax.grad(crystal_yield, argnums=2)(0.1, 1.0, K0 + 25.0, SOL0)
    assert np.isfinite(float(g))


def test_msmpr_functions_smoke():
    vals = [msmpr_moments(N0, G, TAU, (0, 1, 2, 3)), msmpr_dominant_size(G, TAU),
            msmpr_mass_median_size(G, TAU), msmpr_mass_cv(),
            msmpr_from_magma(10.0, RHO_C, KV, G, TAU),
            growth_rate(0.02, 1e-4, 1.5), nucleation_rate(0.02, 10.0, 1e9, 2.0, 0.6)]
    assert all(np.all(np.isfinite(np.asarray(v))) for v in vals)
    L = np.linspace(1e-5, 3e-4, 20)
    Gt, n0 = fit_msmpr(L, np.asarray(msmpr_distribution(L, N0, G, TAU)))
    assert np.isfinite(float(Gt)) and np.isfinite(float(n0))


def test_sieve_population_density_smoke():
    L, n = sieve_population_density([0.2], [1.0e-3], [0.5e-3], 120.0, 1600.0, 1.0)
    assert np.isfinite(float(L[0])) and np.isfinite(float(n[0]))


def test_msmpr_steady_state_smoke_and_grad():
    def run(tau):
        return msmpr_steady_state(0.60, 1.0, K0 + 5.0, SOL0, KIN, tau, crystal_density=RHO_C,
                                  shape_factor=KV, liquor_density=RHO_L)

    r = run(900.0)
    for k in ("delta_C", "crystals", "L_D", "L_50", "M_T"):
        assert np.isfinite(float(r[k]))
    assert np.isfinite(float(jax.grad(lambda t: run(t)["L_50"])(900.0)))


def test_msmpr_j_above_one_is_rejected():
    with pytest.raises(ValueError):
        msmpr_steady_state(0.60, 1.0, K0 + 5.0, SOL0, KIN[:4] + (1.5,), 900.0,
                           crystal_density=RHO_C, shape_factor=KV, liquor_density=RHO_L)


def test_unit_equilibrium_mass_balance():
    feed = feed_stream()
    cry, liq, info = Crystallizer(params())(feed)
    for k in ("F_Na2SO4", "F_H2O"):
        assert float(cry[k] + liq[k]) == pytest.approx(float(feed[k]), rel=1e-12)
    assert np.isfinite(float(info["yield_kg_s"]))


@pytest.mark.parametrize("kw", [
    dict(mode="evaporative", T=None, water_evaporated=0.3),
    dict(mode="adiabatic", T=K0 + 30.0, cp_solution=3200.0),
])
def test_unit_other_modes_smoke(kw):
    cry, liq, info = Crystallizer(params(**kw))(feed_stream(T=K0 + 60.0))
    assert np.isfinite(float(info["crystal_mass"]))
    assert np.isfinite(float(info["water_evaporated"]))


def test_unit_kinetic_mode_jit_and_traced_kinetics():
    """Regression: float() on kinetics raised under jit and for traced kinetics."""
    def lnL50(logk):
        p = params(kinetics=jnp.append(jnp.exp(logk), 1.0), residence_time=900.0,
                   crystal_density=RHO_C, liquor_density=RHO_L)
        return jnp.log(Crystallizer(p)(feed_stream())[2]["L_50"])

    logk = jnp.log(jnp.array(KIN[:4]))
    assert np.isfinite(float(jax.jit(lnL50)(logk)))
    assert np.all(np.isfinite(np.asarray(jax.jit(jax.grad(lnL50))(logk))))


def test_unit_grad_in_T():
    g = jax.grad(lambda T: Crystallizer(params(T=T))(feed_stream())[2]["yield_kg_s"])(K0 + 5.0)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("kw", [
    dict(mode="bogus"),
    dict(mode="cooling", water_evaporated=0.1),
    dict(mode="cooling", T=None),
    dict(water_evaporated=0.1, evaporation_fraction=0.1, mode="evaporative"),
    dict(kinetics=[1, 1, 1, 1, 1]),
    dict(mode="adiabatic", water_evaporated=0.1),
    dict(kinetics=[1, 1, 1, 1, 2.0], residence_time=1.0),
    dict(transition_T=None, upper_solubility_coeffs=[0.4]),
])
def test_parameter_validation(kw):
    with pytest.raises(ValueError):
        params(**kw)


def test_evaporator_to_crystallizer_flowsheet():
    from difflow import Evaporator, EvaporatorParams
    from difflow.flowsheet import Flowsheet, Unit

    evap = Evaporator(EvaporatorParams(
        solute_species=["Na2SO4"], solute_MW=[M_S], solvent="water", x_product=0.30,
        bpr_model="ideal", vant_hoff_i=3.0, P_vapor_space=60e3, steam_P=300e3, U=1500.0))
    feed = make_stream({"Na2SO4": 0.30 * 1000 / M_S, "water": 1.7 * 1000 / M_W}, T=K0 + 40.0, P=1e5)
    fs = Flowsheet(species_order=["water", "Na2SO4"])
    fs.add_feed("feed", feed)
    fs.add_unit(Unit("evap", evap, ["feed"], ["conc", "vap"]))
    fs.add_unit(Unit("xtal", Crystallizer(params(solvent="water")), ["conc"], ["crystals", "liquor"]))
    out = fs.solve()
    assert float(out["crystals"]["F_Na2SO4"] + out["liquor"]["F_Na2SO4"]) == pytest.approx(
        float(feed["F_Na2SO4"]), rel=1e-12)
