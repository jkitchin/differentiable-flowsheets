"""Smoke tests for difflow_power.dc: DC power flow, DC-OPF, PTDF, LODF."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.dc import (
    contingency_flows,
    dc_matrices,
    lodf,
    ptdf,
    solve_dc_power_flow,
    solve_dcopf,
)


def test_dc_power_flow_runs_and_balances():
    net = dp.cases.case9()
    result = solve_dc_power_flow(net)
    m = dc_matrices(net)
    injections = m.b_bus @ result.va + m.p_bus_shift
    assert float(jnp.sum(injections)) == pytest.approx(0.0, abs=1e-9)
    assert bool(jnp.all(jnp.isfinite(result.p_from)))


def test_dcopf_smoke():
    net = dp.cases.case5()
    result = solve_dcopf(net)
    assert result.converged
    assert set(result.pg_mw) == set(net.generators)
    assert set(result.lmp_mw) == set(net.bus_ids)


def test_ptdf_and_contingency_flows_shapes():
    net = dp.cases.case9()
    assert ptdf(net).shape == (net.n_branch, net.n_bus)
    base = solve_dc_power_flow(net)
    flows = contingency_flows(net, base.p_from)
    assert flows.shape == (net.n_branch, net.n_branch)


def test_lodf_marks_an_islanding_outage_as_undefined():
    factors = lodf(dp.cases.radial_feeder())
    assert bool(jnp.all(jnp.isnan(factors)))


def test_dc_flow_grad_wrt_reactance_is_finite():
    net = dp.cases.case9()
    reactances = net.branch_param_arrays()["x"]
    grad = jax.grad(
        lambda xs: solve_dc_power_flow(net, branch_params={"x": xs}).p_from[2]
    )(reactances)
    assert bool(jnp.all(jnp.isfinite(grad)))
