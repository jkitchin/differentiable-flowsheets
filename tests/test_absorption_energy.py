"""Smoke tests for the Absorber / Stripper energy balance and live steam."""

import jax
import numpy as np
import pytest

from difflow import make_stream
from difflow.units.absorption import (
    Absorber,
    AbsorptionParams,
    Stripper,
    henry_params,
)

jax.config.update("jax_enable_x64", True)

T0, P0 = 298.15, 101325.0


def _nh3(L=2.0):
    gas = make_stream({"air": 0.95, "NH3": 0.05}, T0, P0)
    liq = make_stream({"H2O": L}, T0, P0)
    return gas, liq


def _absorber(**kw):
    henry, henry_C = henry_params(["NH3"])
    kw.setdefault("model", "stages")
    kw.setdefault("n_stages", 5)
    return Absorber(AbsorptionParams(henry=henry, henry_C=henry_C, **kw))


def _steam_stripper(**kw):
    henry, henry_C = henry_params(["NH3"])
    kw.setdefault("n_stages", 5.0)
    kw.setdefault("cp_gas", 34.0)
    return Stripper(AbsorptionParams(henry=henry, henry_C=henry_C, **kw))


def _sour_water(S=15.0):
    steam = make_stream({"steam": S}, 373.15, P0)
    water = make_stream({"H2O": 100.0, "NH3": 1.0}, 323.15, P0)
    return steam, water


def test_absorber_energy_balance_smoke_and_grad():
    gas, _ = _nh3()

    def f(L):
        g, _, i = _absorber(energy_balance=True, intercooler_stages=(3,))(
            gas, make_stream({"H2O": L}, T0, P0))
        return g["F_NH3"], i["stage_T"]

    (val, stage_T), grad = jax.value_and_grad(f, has_aux=True)(2.0)
    assert np.all(np.isfinite(np.asarray(stage_T)))
    assert np.isfinite(float(val)) and np.isfinite(float(grad))


def test_absorber_energy_validation():
    with pytest.raises(ValueError, match="model='stages'"):
        _absorber(model="kremser", n_stages=5.0, energy_balance=True)
    with pytest.raises(ValueError, match="T=None"):
        _absorber(energy_balance=True, T=300.0)
    with pytest.raises(ValueError, match="energy_balance=True"):
        _absorber(intercooler_stages=(2,))
    with pytest.raises(ValueError, match="outside"):
        _absorber(energy_balance=True, intercooler_stages=(10,))(*_nh3())


def test_live_steam_stripper_smoke():
    g, l, i = _steam_stripper(steam_species="steam", T=373.15)(*_sour_water())
    for s, tot in (("NH3", 1.0), ("steam", 15.0)):
        assert float(g.get(f"F_{s}", 0.0) + l.get(f"F_{s}", 0.0)) == pytest.approx(tot, rel=1e-8)


def test_live_steam_stripper_jit_grad():
    # #496: the steam balance is an implicit fixed point, so jit(grad) traces one column.
    st = _steam_stripper(steam_species="steam", T=373.15)
    steam, _ = _sour_water()

    def f(L):
        return st(steam, make_stream({"H2O": L, "NH3": 1.0}, 323.15, P0))[2]["steam_condensed"]

    assert np.isfinite(float(jax.jit(jax.grad(f))(100.0)))


def test_live_steam_validation():
    steam, water = _sour_water()
    with pytest.raises(ValueError, match="not in gas_in"):
        _steam_stripper(steam_species="H2O")(steam, water)
    with pytest.raises(ValueError, match="Stripper"):
        henry, henry_C = henry_params(["NH3"])
        Absorber(AbsorptionParams(henry=henry, henry_C=henry_C, steam_species="steam"))(steam, water)
    with pytest.raises(ValueError, match="solute"):
        _steam_stripper(steam_species="NH3")
