"""Energy balances of the generic Absorber / Stripper (#448).

* ``energy_balance=True`` (stages model): the liquid of an NH3-water scrubber
  heats up stage by stage, the K values follow the stage temperature, and the
  heat released closes against the sensible heat gained by both phases plus
  the intercooler duty; the default (isothermal) result is unchanged; the
  gradient matches finite differences.
* live steam (``steam_species``): the condensed steam closes the overall energy
  balance and joins the liquid; the per-species mass balance holds.
* the CH4 entry of ``HENRY_WATER`` (Sander 2015) in the file's convention.

The heat-of-absorption numbers come from the (from-memory) van 't Hoff
coefficients of ``HENRY_WATER``, so these are consistency checks, not a
validation of a published column.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import make_stream
from difflow.units.absorption import (
    HENRY_WATER,
    R_GAS,
    Absorber,
    AbsorptionParams,
    Stripper,
    henry_constant,
    henry_params,
)

jax.config.update("jax_enable_x64", True)

T0, P0 = 298.15, 101325.0


def _nh3(L=2.0, y=0.05, T_gas=T0, T_liq=T0):
    gas = make_stream({"air": 1.0 - y, "NH3": y}, T_gas, P0)
    liq = make_stream({"H2O": L}, T_liq, P0)
    return gas, liq


def _absorber(**kw):
    henry, henry_C = henry_params(["NH3"])
    kw.setdefault("model", "stages")
    kw.setdefault("n_stages", 9)
    return Absorber(AbsorptionParams(henry=henry, henry_C=henry_C, **kw))


class TestAbsorberEnergyBalance:
    def test_default_is_unchanged_and_isothermal(self):
        gas, liq = _nh3()
        g0, l0, i0 = _absorber()(gas, liq)
        g1, l1, i1 = _absorber(energy_balance=False)(gas, liq)
        assert float(g0["F_NH3"]) == float(g1["F_NH3"])
        assert float(l0["T"]) == T0 and "stage_T" not in i0

    def test_liquid_heats_down_the_column_and_absorbs_less(self):
        gas, liq = _nh3()
        _, _, iso = _absorber()(gas, liq)
        _, lo, i = _absorber(energy_balance=True)(gas, liq)
        assert bool(i["converged"])
        T = np.asarray(i["stage_T"])
        assert T[-1] > T0 + 5.0                       # about 11 K at L/G = 2
        assert np.all(np.diff(T) > 0)                  # warms stage by stage, top to bottom
        assert float(lo["T"]) == pytest.approx(T[-1])
        assert float(i["recovery"]["NH3"]) < float(iso["recovery"]["NH3"])
        # K follows the stage temperature through the van 't Hoff form
        K = np.asarray(i["stage_K"]["NH3"])
        H = HENRY_WATER["NH3"]
        np.testing.assert_allclose(K, np.asarray(henry_constant(H.H298, T, H.C)) / P0, rtol=1e-12)

    def test_large_heat_effect_converges_and_cuts_recovery(self):
        """At L/G = 1 the temperature pinches the column; more stages hardly help."""
        gas, liq = _nh3(L=1.0)
        _, _, iso = _absorber(n_stages=30)(gas, liq)
        out = {}
        for N in (9, 30):
            _, _, i = _absorber(energy_balance=True, n_stages=N)(gas, liq)
            assert bool(i["converged"])
            out[N] = float(i["recovery"]["NH3"])
        assert float(iso["recovery"]["NH3"]) > 0.99
        assert out[30] < 0.7 and abs(out[30] - out[9]) < 0.01

    @pytest.mark.parametrize("ics", [(), (3, 6), (9,)])
    def test_energy_balance_closes(self, ics):
        gas, liq = _nh3(T_gas=310.0, T_liq=290.0)
        g, l, i = _absorber(energy_balance=True, intercooler_stages=ics, intercooler_T=295.0)(gas, liq)
        # heat released by the solute that moved into the liquid
        Q_rel = R_GAS * HENRY_WATER["NH3"].C * (float(gas["F_NH3"]) - float(g["F_NH3"]))
        assert float(jnp.sum(i["stage_Q_absorption"])) == pytest.approx(Q_rel, rel=1e-10)
        # sensible heat gained by the outlets (independently, from the streams)
        cpL, cpG = 75.3, 29.1
        L_out = float(l["F_H2O"] + l["F_NH3"])
        V_out = float(g["F_air"] + g["F_NH3"])
        sens = (cpL * (L_out * (float(l["T"]) - T0) - 2.0 * (290.0 - T0))
                + cpG * (V_out * (float(g["T"]) - T0) - 1.0 * (310.0 - T0)))
        assert Q_rel == pytest.approx(sens + float(i["Q_intercooler"]), rel=1e-8)
        assert abs(float(i["energy_balance_error"])) < 1e-6 * Q_rel
        if ics:  # (a cooler above 295 K would heat the cold top liquid: duty < 0 there)
            duty = np.asarray(i["intercooler_duty"])
            assert np.count_nonzero(duty) == len(ics)
        if 9 in ics:
            assert float(l["T"]) == pytest.approx(295.0)

    def test_intercoolers_restore_recovery(self):
        gas, liq = _nh3(L=1.3)
        _, _, hot = _absorber(energy_balance=True, n_stages=12)(gas, liq)
        _, _, cool = _absorber(energy_balance=True, n_stages=12, intercooler_stages=(3, 6, 9))(gas, liq)
        assert float(cool["recovery"]["NH3"]) > float(hot["recovery"]["NH3"]) + 0.05
        assert float(jnp.max(cool["stage_T"])) < float(jnp.max(hot["stage_T"]))
        assert np.all(np.asarray(cool["intercooler_duty"]) >= 0.0) and float(cool["Q_intercooler"]) > 0.0

    def test_no_heat_effect_matches_isothermal(self):
        """With C = 0 and equal inlet temperatures the energy balance changes nothing."""
        henry = {"NH3": HENRY_WATER["NH3"].H298}
        gas, liq = _nh3()
        p = dict(henry=henry, model="stages", n_stages=6)
        g0, _, _ = Absorber(AbsorptionParams(**p))(gas, liq)
        g1, _, i1 = Absorber(AbsorptionParams(energy_balance=True, **p))(gas, liq)
        assert float(g1["F_NH3"]) == pytest.approx(float(g0["F_NH3"]), rel=1e-9)
        np.testing.assert_allclose(np.asarray(i1["stage_T"]), T0, atol=1e-9)

    def test_gradient_matches_finite_difference(self):
        gas, _ = _nh3()

        def f(L):
            liq = make_stream({"H2O": L}, T0, P0)
            g, _, _ = _absorber(energy_balance=True, intercooler_stages=(5,))(gas, liq)
            return g["F_NH3"]

        g = float(jax.grad(f)(1.5))
        fd = (float(f(1.5 + 1e-5)) - float(f(1.5 - 1e-5))) / 2e-5
        assert np.isfinite(g) and g < 0.0
        assert g == pytest.approx(fd, rel=1e-5)

    def test_validation(self):
        with pytest.raises(ValueError, match="model='stages'"):
            _absorber(model="kremser", n_stages=5.0, energy_balance=True)
        with pytest.raises(ValueError, match="T=None"):
            _absorber(energy_balance=True, T=300.0)
        with pytest.raises(ValueError, match="energy_balance=True"):
            _absorber(intercooler_stages=(2,))
        with pytest.raises(ValueError, match="outside"):
            _absorber(energy_balance=True, intercooler_stages=(10,))(*_nh3())


def _steam_stripper(**kw):
    henry, henry_C = henry_params(["NH3"])
    kw.setdefault("n_stages", 10.0)
    kw.setdefault("cp_gas", 34.0)
    return Stripper(AbsorptionParams(henry=henry, henry_C=henry_C, **kw))


def _sour_water(S=15.0, T_steam=373.15, T_feed=323.15):
    steam = make_stream({"steam": S}, T_steam, P0)
    water = make_stream({"H2O": 100.0, "NH3": 1.0}, T_feed, P0)
    return steam, water


class TestLiveSteam:
    def test_default_steam_is_inert(self):
        steam, water = _sour_water()
        g, l, i = _steam_stripper()(steam, water)
        assert float(g["F_steam"]) == 15.0 and "F_steam" not in l and "steam_condensed" not in i

    @pytest.mark.parametrize("T_steam", [373.15, 423.15])
    def test_condensed_steam_closes_the_energy_balance(self, T_steam):
        steam, water = _sour_water(T_steam=T_steam)
        g, l, i = _steam_stripper(steam_species="steam", T=373.15)(steam, water)
        assert bool(i["converged"]) and bool(i["feasible"])
        n = float(i["steam_condensed"])
        assert float(l["F_steam"]) == pytest.approx(n) and float(g["F_steam"]) == pytest.approx(15.0 - n)
        Q_pre = 75.3 * 101.0 * (373.15 - 323.15)
        Q_gas = 34.0 * 15.0 * (T_steam - 373.15)
        Q_des = -R_GAS * HENRY_WATER["NH3"].C * (float(water["F_NH3"]) - float(l["F_NH3"]))
        assert float(i["Q_desorption"]) == pytest.approx(-Q_des, rel=1e-8)
        assert n * 40660.0 == pytest.approx(Q_pre - Q_gas - Q_des, rel=1e-8)
        assert float(i["steam_duty"]) == pytest.approx(
            float(i["Q_preheat"] + i["Q_desorption"] - i["Q_gas_sensible"]), rel=1e-8)
        assert float(l["T"]) == pytest.approx(373.15)
        # mass balance per species
        for s, tot in (("NH3", 1.0), ("steam", 15.0)):
            assert float(g.get(f"F_{s}", 0.0) + l.get(f"F_{s}", 0.0)) == pytest.approx(tot, rel=1e-12)
        if T_steam > 373.15:  # superheat condenses less
            _, _, i0 = _steam_stripper(steam_species="steam", T=373.15)(*_sour_water())
            assert n < float(i0["steam_condensed"])

    def test_condensation_costs_stripping(self):
        steam, water = _sour_water()
        _, _, inert = _steam_stripper()(steam, water)
        _, _, live = _steam_stripper(steam_species="steam")(steam, water)
        assert float(live["recovery"]["NH3"]) < float(inert["recovery"]["NH3"])

    def test_too_little_steam_is_infeasible(self):
        steam, water = _sour_water(S=5.0)
        _, _, i = _steam_stripper(steam_species="steam")(steam, water)
        assert not bool(i["feasible"])

    def test_gradient_wrt_steam_rate(self):
        def f(S):
            steam, water = _sour_water(S=S)
            return _steam_stripper(steam_species="steam")(steam, water)[1]["F_NH3"]

        g = float(jax.grad(f)(15.0))
        fd = (float(f(15.0 + 1e-5)) - float(f(15.0 - 1e-5))) / 2e-5
        assert g < 0.0 and g == pytest.approx(fd, rel=1e-6)

    def test_validation(self):
        steam, water = _sour_water()
        with pytest.raises(ValueError, match="not in gas_in"):
            _steam_stripper(steam_species="H2O")(steam, water)
        with pytest.raises(ValueError, match="Stripper"):
            henry, henry_C = henry_params(["NH3"])
            Absorber(AbsorptionParams(henry=henry, henry_C=henry_C, steam_species="steam"))(steam, water)
        with pytest.raises(ValueError, match="solute"):
            _steam_stripper(steam_species="NH3")


def test_methane_henry_constant_follows_sander_2015():
    d = HENRY_WATER["CH4"]
    Hcp = 1.4e-5                      # mol/(m3 Pa), Sander (2015)
    assert d.H298 == pytest.approx(55343.0 / Hcp, rel=2e-3)
    assert d.C == 1600.0
    assert "Sander" in d.source
    # warmer water dissolves less methane
    assert float(henry_constant(d.H298, 318.15, d.C)) > d.H298
