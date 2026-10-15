"""Smoke tests for difflow_power.sensitivity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.residuals import power_state_layout


def test_sensitivities_smoke():
    net = dp.cases.case3()
    layout = power_state_layout(net)
    assert dp.loss_sensitivity(net).shape == (net.n_bus,)
    assert dp.demand_sensitivity(net).shape == (layout.size, net.n_bus)
    assert dp.branch_flow_sensitivity(net).shape == (net.n_branch, net.n_bus)
    jac = dp.parameter_sensitivity(net, "x")
    assert jac.shape == (layout.size, net.n_branch)
    assert bool(jnp.all(jnp.isfinite(jac)))
    result = dp.solve_power_flow(net)
    assert np.isfinite(float(dp.voltage_stability_margin(result.x, net)))


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError, match="unknown branch parameter"):
        dp.parameter_sensitivity(dp.cases.case3(), "resistance")
