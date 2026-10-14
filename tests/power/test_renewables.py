"""Smoke and validation tests for difflow_power.renewables."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.opf import acopf_structure
from difflow_power.residuals import power_state_layout


def test_resource_models_smoke():
    assert np.isfinite(float(dp.wind_turbine_power(8.0, 3.0)))
    assert np.isfinite(float(dp.pv_power(800.0, 20.0, capacity_mw_dc=100.0)))
    assert np.isfinite(float(dp.hydro_power(100.0, 50.0)))
    g = jax.grad(lambda v: dp.wind_turbine_power(v, 3.0))(8.0)
    assert np.isfinite(float(g))


def test_wind_power_curve_rejects_disordered_speeds():
    with pytest.raises(ValueError, match="cut_in"):
        dp.wind_turbine_power(10.0, 3.0, rated_speed=30.0)


def test_factories_build_generators():
    for gen in (
        dp.renewable_generator("5", 40.0, 50.0, name="farm"),
        dp.wind_generator("5", 12.0, n_turbines=10, turbine_mw=3.0),
        dp.solar_generator("5", 1200.0, 10.0, capacity_mw_dc=120.0),
        dp.hydro_generator("5", 100.0, 50.0, machine_mva=60.0),
    ):
        assert gen.p_min_mw == 0.0
        assert gen.s_max_mva > 0.0


def test_factory_validation():
    with pytest.raises(ValueError, match="non-negative"):
        dp.renewable_generator("5", -1.0, 50.0)
    with pytest.raises(ValueError, match="s_max_mva"):
        dp.Generator("5", s_max_mva=0.0)


def test_structure_carries_a_capability_row():
    net = dp.cases.case9()
    net.generators["w"] = dp.renewable_generator("5", 40.0, 50.0)
    layout = power_state_layout(net)
    structure = acopf_structure(net, layout)
    assert structure.n_capability == 1
