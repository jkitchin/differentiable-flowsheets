"""Smoke and validation tests for difflow_power.powerflow."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.powerflow import solve_state, specification_from_network
from difflow_power.residuals import power_state_layout


def test_power_flow_smoke():
    net = dp.cases.case9()
    result = dp.solve_power_flow(net)
    assert result.converged
    assert set(result.vm) == set(net.bus_ids)
    assert set(result.branch_loading) == set(net.branch_ids)
    assert "converged" in result.summary()


def test_non_convergence_is_reported_not_raised():
    net = dp.cases.case9().scaled_load(3.0)
    result = dp.solve_power_flow(net, max_steps=15)
    assert not result.converged
    assert "DID NOT CONVERGE" in result.summary()


def test_disagreeing_voltage_setpoints_on_one_bus_are_rejected():
    net = dp.cases.case5()
    net.generators["g2"].vm_setpoint = 1.03
    with pytest.raises(ValueError, match="different voltages"):
        specification_from_network(net)


def test_solve_state_grad_wrt_demand_is_finite():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    spec = specification_from_network(net)
    x0 = dp.flat_start(net, layout, spec)
    pd, qd = net.load_arrays_pu()

    def slack_mw(p):
        x = solve_state(net, layout, spec, x0, demand=(p, qd))[0]
        return x[layout.slice_pg][0]

    assert bool(jnp.all(jnp.isfinite(jax.grad(slack_mw)(pd))))
