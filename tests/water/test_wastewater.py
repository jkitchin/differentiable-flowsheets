"""Wastewater (ASM1 reactor, Takacs settler, BSM1 plant): code checks."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw
from difflow_water.wastewater import (
    STATES,
    conversion_rates,
    settling_velocity,
    state_to_stream,
    stream_to_state,
)

jax.config.update("jax_enable_x64", True)

Z_MID = jnp.array([30.0, 2.0, 1149.0, 80.0, 2550.0, 148.0, 449.0, 1.0, 5.0, 8.0, 1.0, 5.0, 5.0])


def test_asm1_rates_finite_and_differentiable():
    r = conversion_rates(Z_MID, dw.ASM1Params())
    assert r.shape == (13,) and jnp.isfinite(r).all()
    g = jax.jacfwd(lambda z: conversion_rates(z, dw.ASM1Params()))(Z_MID)
    assert jnp.isfinite(g).all()


def test_stream_state_round_trip():
    s = state_to_stream(18446.0, Z_MID)
    Q, Z = stream_to_state(s)
    assert jnp.allclose(Q, 18446.0) and jnp.allclose(Z, Z_MID)
    assert {f"F_{k}" for k in STATES} <= set(s)


def test_settling_velocity_bounds():
    t = dw.wastewater.TakacsParams()
    v = settling_velocity(jnp.array([0.0, 500.0, 3000.0, 1e4]), 3000.0, t)
    assert jnp.all(v >= 0) and jnp.all(v <= t.v0_max)


def test_asm1_reactor_runs_and_differentiates():
    out, info = dw.ASM1Reactor(dw.ASM1ReactorParams(volume=1000.0, KLa=0.0))(dw.bsm1_influent())
    assert bool(info["converged"]) and jnp.isfinite(out["F_S_NH"])
    g = jax.grad(lambda k: dw.ASM1Reactor(dw.ASM1ReactorParams(KLa=k))(dw.bsm1_influent())[1]
                 ["Z"][STATES.index("S_O")])(240.0)
    assert jnp.isfinite(g)


def test_asm1_reactor_do_setpoint():
    _, info = dw.ASM1Reactor(dw.ASM1ReactorParams(DO_setpoint=2.0))(dw.bsm1_influent())
    assert jnp.allclose(info["Z"][STATES.index("S_O")], 2.0) and jnp.isfinite(info["aeration"])


def test_takacs_settler_runs():
    feed = state_to_stream(36892.0, Z_MID)
    eff, und, info = dw.TakacsSettler(dw.TakacsSettlerParams())(feed)
    assert info["solids"].shape == (10,) and bool(info["converged"])
    assert jnp.isfinite(eff["F_X_BH"]) and jnp.isfinite(und["F_X_BH"])


def test_takacs_settler_rejects_bad_feed_layer():
    with pytest.raises(ValueError, match="feed_layer"):
        dw.TakacsSettler(dw.TakacsSettlerParams(feed_layer=10))


def test_plant_runs_and_differentiates():
    r = dw.activated_sludge_plant(dw.bsm1_influent())
    assert bool(r["converged"]) and r["reactors"].shape == (5, 13)
    assert jnp.isfinite(r["sludge_age"]) and r["settler_solids"].shape == (10,)

    def nh4(kla5):
        p = dw.ActivatedSludgeParams(KLa=(0.0, 0.0, 240.0, 240.0, kla5))
        return dw.activated_sludge_plant(dw.bsm1_influent(), p)["Z_effluent"][STATES.index("S_NH")]

    assert jnp.isfinite(jax.grad(nh4)(84.0))


def test_plant_validation():
    with pytest.raises(ValueError, match="settler"):
        dw.activated_sludge_plant(dw.bsm1_influent(), dw.ActivatedSludgeParams(settler="bogus"))
    with pytest.raises(ValueError, match="one entry per reactor"):
        dw.activated_sludge_plant(dw.bsm1_influent(), dw.ActivatedSludgeParams(KLa=(0.0,)))


def test_plant_split_settler_runs():
    p = dw.ActivatedSludgeParams(settler="split", DO_setpoint=(None, None, None, None, 0.449))
    r = dw.activated_sludge_plant(dw.bsm1_influent(), p)
    assert jnp.isfinite(r["Z_effluent"]).all() and jnp.isfinite(r["aeration"][-1])


def test_bsm1_indices_keys_and_finite():
    from difflow_water.wastewater import BSM1_LIMITS, bsm1_indices
    r = dw.activated_sludge_plant(dw.bsm1_influent())
    i = bsm1_indices(r)
    for k in ("COD", "BOD5", "TSS", "N_tot", "EQ", "AE", "PE"):
        assert jnp.isfinite(i[k]), k
    assert set(i["violations"]) == set(BSM1_LIMITS)


def test_bsm1_flowsheet_wiring():
    from difflow_water.wastewater.units import build_bsm1_flowsheet
    fs = build_bsm1_flowsheet()
    assert set(fs.recycles.values()) == {"int_rec", "ext_rec"}
    assert "influent" in fs.feeds
