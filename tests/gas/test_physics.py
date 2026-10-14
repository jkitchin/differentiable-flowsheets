"""Smoke tests for difflow_gas.physics."""

import jax
import jax.numpy as jnp
import pytest

from difflow_gas.physics import (
    aga_friction,
    compressor_power,
    isothermal_pipe_beta,
    kg_s_to_knm3h,
    knm3h_to_kg_s,
    nikuradse_friction,
    panhandle_a_friction,
    panhandle_b_friction,
    resistor_xi,
    smoothed_power_w,
    specific_gas_constant,
    standard_flow_scfd,
    weymouth_friction,
)


def test_physics_functions_run_and_are_finite():
    vals = [
        specific_gas_constant(16.0),
        kg_s_to_knm3h(knm3h_to_kg_s(300.0, 0.785), 0.785),
        nikuradse_friction(0.5, 1e-4),
        weymouth_friction(0.6),
        resistor_xi(5.0, 0.5, 283.15, 18.0),
        compressor_power(100.0, 1.2),
        smoothed_power_w(50.0, 1.3),
        standard_flow_scfd(121.54, 16.04),
        panhandle_a_friction(0.6, 121.54, 16.04),
        panhandle_b_friction(0.6, 121.54, 16.04),
        aga_friction(0.6, 45e-6, 121.54, 1.2e-5),
    ]
    assert all(jnp.isfinite(v) for v in vals)


@pytest.mark.parametrize("law", ["nikuradse", "colebrook", "weymouth",
                                 "panhandle_a", "panhandle_b", "aga"])
def test_pipe_beta_friction_laws_run(law):
    beta = isothermal_pipe_beta(50e3, 0.6, 45e-6, 288.15, 16.04, 0.851,
                                friction=law, mass_flow_kg_s=121.54,
                                viscosity_pa_s=1.2e-5)
    assert jnp.isfinite(beta) and float(beta) > 0


def test_pipe_beta_gradient_is_finite():
    g = jax.grad(lambda D: isothermal_pipe_beta(50e3, D, 45e-6, 288.15, 16.04,
                                                0.851, friction="aga",
                                                mass_flow_kg_s=121.54,
                                                viscosity_pa_s=1.2e-5))(0.6)
    assert jnp.isfinite(g)


def test_pipe_beta_rejects_bad_friction_inputs():
    with pytest.raises(ValueError):
        isothermal_pipe_beta(1e3, 0.5, 1e-4, friction="panhandle")
    with pytest.raises(ValueError):
        isothermal_pipe_beta(1e3, 0.5, 1e-4, friction="colebrook")
