"""Tests for combustion / gas-turbine units (issue #173).

Combustor, GasCompressor, GasTurbine on an ideal-gas working fluid, plus the
brayton_cycle assembly validated against F-class machine performance.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow.combustion import (
    IdealGasThermo,
    CYCLE_SPECIES,
    AIR_COMPOSITION,
    lhv_molar,
    o2_demand,
    co2_per_mol,
    h2o_per_mol,
)
from difflow.units.gas_turbine import (
    Combustor,
    CombustorParams,
    GasCompressor,
    GasCompressorParams,
    GasTurbine,
    GasTurbineParams,
    BraytonCycleParams,
    brayton_cycle,
    make_cycle_thermo,
)
from difflow.streams import make_stream, get_flows

jax.config.update("jax_enable_x64", True)

P_ATM = 101325.0


@pytest.fixture
def thermo():
    return make_cycle_thermo()


# ---------------------------------------------------------------------------
# Fuel / combustion data
# ---------------------------------------------------------------------------
class TestCombustionData:
    def test_methane_properties(self):
        assert float(lhv_molar({"methane": 1.0})) == pytest.approx(802.6e3)
        assert float(o2_demand({"methane": 1.0})) == pytest.approx(2.0)
        assert float(co2_per_mol({"methane": 1.0})) == pytest.approx(1.0)
        assert float(h2o_per_mol({"methane": 1.0})) == pytest.approx(2.0)

    def test_diluent_co2_demands_no_oxygen(self):
        """Diluent CO2 passes through: 1 CO2 out, 0 O2 demanded (not combustion)."""
        assert float(o2_demand({"carbon_dioxide": 1.0})) == pytest.approx(0.0)
        assert float(co2_per_mol({"carbon_dioxide": 1.0})) == pytest.approx(1.0)

    def test_air_heat_capacity(self):
        """Ideal-gas air Cp ~ 29.1 J/mol/K near ambient."""
        th = IdealGasThermo(CYCLE_SPECIES)
        n = th.flow_vector(AIR_COMPOSITION)
        cp_molar = float(th.mixture_cp(n, 288.15) / jnp.sum(n))
        assert cp_molar == pytest.approx(29.1, abs=0.3)

    def test_entropy_increases_with_T_decreases_with_P(self):
        th = IdealGasThermo(CYCLE_SPECIES)
        n = th.flow_vector(AIR_COMPOSITION)
        assert float(th.mixture_entropy(n, 400.0, P_ATM)) > float(
            th.mixture_entropy(n, 300.0, P_ATM))
        assert float(th.mixture_entropy(n, 300.0, 5 * P_ATM)) < float(
            th.mixture_entropy(n, 300.0, P_ATM))


# ---------------------------------------------------------------------------
# GasCompressor
# ---------------------------------------------------------------------------
class TestGasCompressor:
    def test_compression_heats_and_consumes_work(self, thermo):
        air = make_stream(dict(AIR_COMPOSITION), 288.15, P_ATM)
        comp = GasCompressor(GasCompressorParams(pressure_ratio=18.0, eta_isentropic=0.89), thermo)
        out, info = comp(air)
        assert float(out["T"]) > 288.15
        assert float(out["P"]) == pytest.approx(18.0 * P_ATM)
        assert float(info["W"]) > 0.0

    def test_ideal_efficiency_reaches_isentropic_T(self, thermo):
        air = make_stream(dict(AIR_COMPOSITION), 288.15, P_ATM)
        comp = GasCompressor(GasCompressorParams(pressure_ratio=18.0, eta_isentropic=1.0), thermo)
        out, info = comp(air)
        assert float(out["T"]) == pytest.approx(float(info["T_isen"]), abs=0.05)

    def test_lower_efficiency_needs_more_work(self, thermo):
        air = make_stream(dict(AIR_COMPOSITION), 288.15, P_ATM)
        hi = GasCompressor(GasCompressorParams(18.0, 0.90), thermo)(air)[1]["W"]
        lo = GasCompressor(GasCompressorParams(18.0, 0.70), thermo)(air)[1]["W"]
        assert float(lo) > float(hi)

    def test_work_differentiable(self, thermo):
        air = make_stream(dict(AIR_COMPOSITION), 288.15, P_ATM)
        def W(rp):
            return GasCompressor(GasCompressorParams(pressure_ratio=rp), thermo)(air)[1]["W"]
        g = jax.grad(W)(jnp.array(18.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0  # higher ratio -> more work


# ---------------------------------------------------------------------------
# GasTurbine
# ---------------------------------------------------------------------------
class TestGasTurbine:
    def _hot_products(self, thermo):
        # Burn 1 mol methane to a firing temperature, get a hot product stream.
        fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
        air = make_stream(dict(AIR_COMPOSITION), 700.0, 18 * P_ATM)
        comb = Combustor(CombustorParams(mode="fixed_T", T_out=1673.15), thermo)
        return comb(fuel, air)[0]

    def test_expansion_cools_and_extracts_work(self, thermo):
        hot = self._hot_products(thermo)
        turb = GasTurbine(GasTurbineParams(P_out=P_ATM, eta_isentropic=0.90), thermo)
        out, info = turb(hot)
        assert float(out["T"]) < float(hot["T"])
        assert float(out["P"]) == pytest.approx(P_ATM)
        assert float(info["W"]) > 0.0

    def test_lower_efficiency_extracts_less_work_and_runs_hotter(self, thermo):
        hot = self._hot_products(thermo)
        hi = GasTurbine(GasTurbineParams(P_ATM, 0.92), thermo)(hot)
        lo = GasTurbine(GasTurbineParams(P_ATM, 0.75), thermo)(hot)
        assert float(lo[1]["W"]) < float(hi[1]["W"])
        assert float(lo[0]["T"]) > float(hi[0]["T"])

    def test_work_differentiable(self, thermo):
        hot = self._hot_products(thermo)
        def W(Pout):
            return GasTurbine(GasTurbineParams(P_out=Pout), thermo)(hot)[1]["W"]
        g = jax.grad(W)(jnp.array(P_ATM))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Combustor
# ---------------------------------------------------------------------------
class TestCombustor:
    def test_adiabatic_flame_and_stoichiometry(self, thermo):
        fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
        air = make_stream({k: 30 * v for k, v in AIR_COMPOSITION.items()}, 700.0, 18 * P_ATM)
        comb = Combustor(CombustorParams(mode="adiabatic"), thermo)
        prod, info = comb(fuel, air)
        fp = get_flows(prod)
        # Complete combustion of 1 mol CH4: +1 CO2, +2 H2O, -2 O2.
        assert float(info["o2_demand"]) == pytest.approx(2.0)
        assert float(fp["oxygen"]) == pytest.approx(30 * AIR_COMPOSITION["oxygen"] - 2.0)
        assert float(fp["carbon_dioxide"]) == pytest.approx(
            1.0 + 30 * AIR_COMPOSITION["carbon_dioxide"])
        assert float(fp["water"]) == pytest.approx(2.0)
        # Lean, very hot but sub-stoichiometric-flame; sane range.
        assert 1000.0 < float(prod["T"]) < 2000.0

    def test_more_air_lowers_flame_temperature(self, thermo):
        fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
        comb = Combustor(CombustorParams(mode="adiabatic"), thermo)
        T_20 = comb(fuel, make_stream({k: 20 * v for k, v in AIR_COMPOSITION.items()}, 700.0, 18 * P_ATM))[0]["T"]
        T_40 = comb(fuel, make_stream({k: 40 * v for k, v in AIR_COMPOSITION.items()}, 700.0, 18 * P_ATM))[0]["T"]
        assert float(T_40) < float(T_20)

    def test_fixed_T_hits_target_and_closes_energy_balance(self, thermo):
        fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
        air = make_stream(dict(AIR_COMPOSITION), 700.0, 18 * P_ATM)
        comb = Combustor(CombustorParams(mode="fixed_T", T_out=1673.15, dp_frac=0.04), thermo)
        prod, info = comb(fuel, air)
        assert float(prod["T"]) == pytest.approx(1673.15)
        assert float(info["air_scale"]) > 1.0  # needs many moles of air per mol fuel
        # Verify the solved air scale actually closes the adiabatic balance at T_out.
        scaled_air = make_stream({k: float(info["air_scale"]) * v for k, v in AIR_COMPOSITION.items()},
                                 700.0, 18 * P_ATM)
        adiab = Combustor(CombustorParams(mode="adiabatic"), thermo)
        assert float(adiab(fuel, scaled_air)[0]["T"]) == pytest.approx(1673.15, abs=0.5)

    def test_fixed_T_pressure_drop(self, thermo):
        fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
        air = make_stream(dict(AIR_COMPOSITION), 700.0, 18 * P_ATM)
        comb = Combustor(CombustorParams(mode="fixed_T", T_out=1673.15, dp_frac=0.04), thermo)
        prod, _ = comb(fuel, air)
        assert float(prod["P"]) == pytest.approx(0.96 * 18 * P_ATM)

    def test_air_scale_differentiable_wrt_TIT(self, thermo):
        fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
        air = make_stream(dict(AIR_COMPOSITION), 700.0, 18 * P_ATM)
        def afr(TIT):
            comb = Combustor(CombustorParams(mode="fixed_T", T_out=TIT), thermo)
            return comb(fuel, air)[1]["air_scale"]
        g = jax.grad(afr)(jnp.array(1673.15))
        assert jnp.isfinite(g)
        assert float(g) < 0.0  # hotter firing -> less dilution air


# ---------------------------------------------------------------------------
# Full Brayton / combined cycle vs. F-class performance
# ---------------------------------------------------------------------------
class TestBraytonCycle:
    FUEL = {"methane": 0.95, "ethane": 0.03, "propane": 0.01,
            "nitrogen": 0.005, "carbon_dioxide": 0.005}

    def test_simple_cycle_matches_f_class(self):
        r = brayton_cycle(self.FUEL, BraytonCycleParams(combined_cycle=False))
        # F-class simple cycle: ~40% LHV efficiency.
        assert float(r["eta_thermal"]) == pytest.approx(0.40, abs=0.02)
        # Reasonable stage temperatures and air/fuel ratio.
        assert 650.0 < float(r["T_compressor_out"]) < 730.0
        assert 850.0 < float(r["T_turbine_out"]) < 1000.0
        assert 18.0 < float(r["air_fuel_molar"]) < 28.0

    def test_combined_cycle_matches_f_class(self):
        r = brayton_cycle(self.FUEL, BraytonCycleParams(combined_cycle=True))
        # F-class CCGT: ~56.8% LHV efficiency.
        assert float(r["eta_thermal"]) == pytest.approx(0.568, abs=0.02)
        assert float(r["eta_thermal"]) > float(r["eta_gt_only"])

    def test_higher_TIT_raises_efficiency(self):
        cool = brayton_cycle(self.FUEL, BraytonCycleParams(TIT_K=1500.0))
        hot = brayton_cycle(self.FUEL, BraytonCycleParams(TIT_K=1700.0))
        assert float(hot["eta_thermal"]) > float(cool["eta_thermal"])

    def test_efficiency_differentiable_wrt_pressure_ratio(self):
        def eta(rp):
            return brayton_cycle(self.FUEL, BraytonCycleParams(pressure_ratio=rp))["eta_thermal"]
        g = jax.grad(eta)(jnp.array(18.0))
        assert jnp.isfinite(g)
