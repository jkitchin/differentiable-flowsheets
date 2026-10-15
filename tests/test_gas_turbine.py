"""Smoke tests for combustion / gas-turbine units (issue #173)."""

import jax
import jax.numpy as jnp
import pytest

from difflow.combustion import (
    AIR_COMPOSITION,
    lhv_mass,
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
from difflow.streams import make_stream, total_flow

jax.config.update("jax_enable_x64", True)

P_ATM = 101325.0


@pytest.fixture
def thermo():
    return make_cycle_thermo()


def test_combustion_data_reads_a_stream():
    """Regression: a Stream's F_ keys used to read as zero (lhv_mass NaN)."""
    stream = make_stream({"methane": 0.9, "nitrogen": 0.1}, 300.0, P_ATM)
    for fn in (lhv_molar, lhv_mass, o2_demand, co2_per_mol, h2o_per_mol):
        assert float(fn(stream)) > 0.0, fn.__name__


def test_compressor_combustor_turbine(thermo):
    air = make_stream(dict(AIR_COMPOSITION), 288.15, P_ATM)
    out, info = GasCompressor(GasCompressorParams(pressure_ratio=18.0), thermo)(air)
    assert float(out["P"]) == pytest.approx(18.0 * P_ATM) and jnp.isfinite(info["W"])

    fuel = make_stream({"methane": 1.0}, 298.15, 18 * P_ATM)
    air = make_stream({k: 30 * v for k, v in AIR_COMPOSITION.items()}, 700.0, 18 * P_ATM)
    prod, info = Combustor(CombustorParams(mode="adiabatic"), thermo)(fuel, air)
    assert jnp.isfinite(prod["T"]) and "o2_demand" in info
    prod, info = Combustor(CombustorParams(mode="fixed_T", T_out=1673.15), thermo)(fuel, air)
    assert jnp.isfinite(info["air_scale"])

    out, info = GasTurbine(GasTurbineParams(P_out=P_ATM), thermo)(prod)
    assert float(out["P"]) == pytest.approx(P_ATM) and jnp.isfinite(info["W"])
    assert float(total_flow(out)) == pytest.approx(float(total_flow(prod)), rel=1e-8)


def test_compressor_grad_finite(thermo):
    air = make_stream(dict(AIR_COMPOSITION), 288.15, P_ATM)
    W = lambda rp: GasCompressor(GasCompressorParams(pressure_ratio=rp), thermo)(air)[1]["W"]
    assert jnp.isfinite(jax.grad(W)(jnp.array(18.0)))


def test_brayton_cycle_smoke():
    r = brayton_cycle({"methane": 1.0}, BraytonCycleParams(combined_cycle=True))
    for k in ("eta_thermal", "eta_gt_only", "T_compressor_out", "T_turbine_out",
              "air_fuel_molar"):
        assert jnp.isfinite(r[k])
