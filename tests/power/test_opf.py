"""Smoke tests for difflow_power.opf, the AC optimal power flow."""

import jax

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.opf import acopf_structure
from difflow_power.residuals import power_state_layout


def test_acopf_smoke():
    net = dp.cases.case3()
    result = dp.solve_acopf(net)
    assert result.converged
    assert set(result.pg_mw) == set(net.generators)
    assert set(result.lmp_mw) == set(net.bus_ids)
    assert "ACOPFResult" in repr(result)


def test_structure_drops_constraints_that_cannot_bind():
    net = dp.cases.case14()          # every branch unrated
    structure = acopf_structure(net, power_state_layout(net))
    assert structure.rated_branches == ()
    assert structure.n_thermal == 0


def test_fixed_bounds_become_equalities():
    net = dp.cases.case9()
    net.generators["g3"].p_min_mw = net.generators["g3"].p_max_mw
    layout = power_state_layout(net)
    structure = acopf_structure(net, layout)
    assert layout.index("pg_g3") in structure.fixed_indices
    assert "pg_g3_min" not in structure.bound_names
