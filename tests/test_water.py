"""Tests for difflow.water: the shared water/steam correlations.

Oracles: the original evaporator correlations (unchanged at 20-200 C), the
IAPWS triple point, Clausius-Clapeyron against the heat of sublimation, and
values the author is confident of (flagged below). No external steam library
is installed, so there is no independent table to compare against.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import difflow.water as water
from difflow.units import evaporator
from difflow.water import (
    P_TRIPLE,
    T_TRIPLE,
    water_latent_heat,
    water_liquid_enthalpy,
    water_saturation_pressure,
    water_sublimation_pressure,
    water_vapor_enthalpy,
    water_vapor_pressure,
)

K = 273.15


def _raw_latent(T):
    a, b, c = water._HFG_COEFFS
    Tr = T / water.T_CRIT_WATER
    return a * (1.0 - Tr) ** (b + c * Tr)


def _raw_liquid(T):
    c3, c2, c1, c0 = water._HF_COEFFS
    t = T - K
    return ((c3 * t + c2) * t + c1) * t + c0


class TestReexport:
    def test_evaporator_names_are_the_water_functions(self):
        for name in ("water_saturation_pressure", "water_saturation_temperature",
                     "water_latent_heat", "water_liquid_enthalpy", "water_vapor_enthalpy",
                     "T_REF", "T_CRIT_WATER", "R_STEAM"):
            assert getattr(evaporator, name) is getattr(water, name)


class TestUnchangedAbove20C:
    @pytest.mark.parametrize("t", [20.0, 25.0, 50.0, 100.0, 150.0, 200.0])
    def test_enthalpies_equal_the_original_fit(self, t):
        T = t + K
        assert float(water_latent_heat(T)) == pytest.approx(float(_raw_latent(T)), rel=1e-14)
        assert float(water_liquid_enthalpy(T)) == pytest.approx(float(_raw_liquid(T)), rel=1e-14)


class TestBelow20C:
    def test_triple_point_values_are_restored(self):
        # 2500.9 kJ/kg and h_f = 0 at the triple point are recalled values (flagged)
        assert float(water_latent_heat(T_TRIPLE)) == pytest.approx(2500.9e3, rel=1e-12)
        assert abs(float(water_liquid_enthalpy(T_TRIPLE))) < 1e-6

    @pytest.mark.parametrize("t,hf,hfg", [(5.0, 20.98, 2489.1), (10.0, 41.99, 2477.2)])
    def test_against_recalled_steam_table(self, t, hf, hfg):
        # recalled from a steam table (Cengel-type): about 0.1 % is the claim
        T = t + K
        assert float(water_liquid_enthalpy(T)) / 1e3 == pytest.approx(hf, rel=2e-3)
        assert float(water_latent_heat(T)) / 1e3 == pytest.approx(hfg, rel=1e-3)

    def test_smooth_through_20C(self):
        f = water_latent_heat
        g = jax.grad(f)
        eps = 1e-6
        jump = abs(float(f(293.15 + eps)) - float(f(293.15 - eps)))
        assert jump < 1.01 * 2 * eps * abs(float(g(293.15)))   # no jump beyond the slope
        assert float(g(293.15 + eps)) == pytest.approx(float(g(293.15 - eps)), rel=1e-4)
        h = jax.grad(water_liquid_enthalpy)
        assert float(h(293.15 + eps)) == pytest.approx(float(h(293.15 - eps)), rel=1e-4)

    def test_monotone_between_minus_20_and_200C(self):
        T = jnp.linspace(253.15, 473.15, 400)
        assert bool(jnp.all(jnp.diff(water_latent_heat(T)) < 0))
        assert bool(jnp.all(jnp.diff(water_liquid_enthalpy(T)) > 0))
        assert bool(jnp.all(water_vapor_enthalpy(T) > water_liquid_enthalpy(T)))


class TestIce:
    def test_triple_point(self):
        assert float(water_sublimation_pressure(T_TRIPLE)) == pytest.approx(P_TRIPLE, rel=1e-12)
        assert float(water_saturation_pressure(T_TRIPLE)) == pytest.approx(P_TRIPLE, rel=2e-6)

    def test_coefficients_are_self_consistent(self):
        # p(T_t) = p_t only if a1 + a2 + a3 = 0
        assert sum(water._ICE_A) == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize("t,p", [(-10.0, 259.9), (-20.0, 103.2)])
    def test_recalled_ice_pressures(self, t, p):
        # values the author is confident of (flagged as recalled): 259.9 Pa
        # at -10 C and 103.2 Pa at -20 C over ice
        assert float(water_sublimation_pressure(t + K)) == pytest.approx(p, rel=2e-3)

    def test_clausius_clapeyron_against_the_heat_of_sublimation(self):
        # d ln p/d ln T = L_sub / (R_v T) with L_sub = 2834.7 kJ/kg at 0 C (recalled)
        T = 273.16
        slope = float(jax.grad(lambda x: jnp.log(water_sublimation_pressure(x)))(T)) * T
        assert slope * 461.5 * T == pytest.approx(2834.7e3, rel=0.01)

    def test_ice_is_below_supercooled_liquid(self):
        T = jnp.linspace(233.15, 273.0, 40)
        assert bool(jnp.all(water_sublimation_pressure(T) < water_saturation_pressure(T)))


class TestVaporPressure:
    def test_follows_each_branch_away_from_the_triple_point(self):
        for t in (-20.0, -5.0):
            T = t + K
            assert float(water_vapor_pressure(T)) == pytest.approx(float(water_sublimation_pressure(T)), rel=1e-12)
        for t in (1.0, 20.0, 80.0):
            T = t + K
            assert float(water_vapor_pressure(T)) == pytest.approx(float(water_saturation_pressure(T)), rel=1e-12)

    def test_flags_force_a_branch(self):
        T = 263.15
        assert float(water_vapor_pressure(T, over_ice=False)) == float(water_saturation_pressure(T))
        assert float(water_vapor_pressure(T + 20.0, over_ice=True)) == float(water_sublimation_pressure(T + 20.0))

    @pytest.mark.slow
    def test_blend_is_continuous_and_has_continuous_slope(self):
        f = lambda T: jnp.log(water_vapor_pressure(T))
        T = jnp.linspace(272.5, 274.0, 601)
        v = np.asarray(jax.vmap(f)(T))
        d = np.asarray(jax.vmap(jax.grad(f))(T))
        assert np.max(np.abs(np.diff(v))) < 2e-3       # no jump
        assert np.max(np.abs(np.diff(d))) < 5e-3       # no kink (slope ~0.1 /K)

    def test_gradient(self):
        check_grads(water_vapor_pressure, (260.0,), order=1, modes=["rev"], rtol=1e-5)
        check_grads(water_vapor_pressure, (273.16,), order=1, modes=["rev"], rtol=1e-5)
        check_grads(water_vapor_pressure, (300.0,), order=1, modes=["rev"], rtol=1e-5)
