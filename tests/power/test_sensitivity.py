"""Tests for difflow_power.sensitivity.

Every sensitivity here is a derivative of a solved state, so the only
honest oracle is a finite difference of that same solved state. These
tests take one and compare. Where a factor has a classical counterpart
--- the DC PTDF for the AC injection shift factors --- the two are
compared as well, which says how good the linearisation is rather than
whether the code runs.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.residuals import power_state_layout


def test_loss_factors_match_finite_differences():
    net = dp.cases.case9()
    factors = dp.loss_sensitivity(net)
    pd, qd = net.load_arrays_pu()
    step = 1e-5
    for bus in ("2", "5", "7", "9"):
        i = net.bus_index[bus]
        up = dp.solve_power_flow(net, demand=(pd.at[i].add(step), qd))
        down = dp.solve_power_flow(net, demand=(pd.at[i].add(-step), qd))
        finite = (up.losses_mw - down.losses_mw) / (2 * step)
        assert float(factors[i]) * net.base_mva == pytest.approx(
            finite, rel=1e-5
        )


def test_the_slack_buss_own_loss_factor_is_zero():
    """Load added at the slack is served at the slack: nothing moves."""
    net = dp.cases.case9()
    factors = dp.loss_sensitivity(net)
    assert float(factors[net.bus_index[net.slack_bus]]) == pytest.approx(
        0.0, abs=1e-12
    )


def test_loss_factors_are_negative_beside_a_scheduled_generator():
    """A real effect, not a sign error.

    In ``case9`` the units at buses 2 and 3 hold a fixed schedule and
    export it across the network. Load added next to one of them is
    served locally instead of being wheeled, so the total transport ---
    and the loss --- falls.
    """
    net = dp.cases.case9()
    factors = dp.loss_sensitivity(net)
    assert float(factors[net.bus_index["8"]]) < 0.0     # beside gen 2
    assert float(factors[net.bus_index["5"]]) > 0.0     # a load pocket


def test_demand_sensitivity_matches_finite_differences():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    jacobian = dp.demand_sensitivity(net)
    assert jacobian.shape == (layout.size, net.n_bus)

    pd, qd = net.load_arrays_pu()
    step = 1e-5
    i = net.bus_index["9"]
    row = layout.index("vm_9")
    up = dp.solve_power_flow(net, demand=(pd.at[i].add(step), qd))
    down = dp.solve_power_flow(net, demand=(pd.at[i].add(-step), qd))
    finite = (up.vm["9"] - down.vm["9"]) / (2 * step)
    assert float(jacobian[row, i]) == pytest.approx(finite, rel=1e-5)
    assert float(jacobian[row, i]) < 0.0      # load sags the voltage


def test_reactive_demand_moves_voltages_more_than_real_does():
    """Transmission is reactance-dominated, so vars drive magnitude."""
    net = dp.cases.case9()
    layout = power_state_layout(net)
    real = dp.demand_sensitivity(net)
    reactive = dp.demand_sensitivity(net, reactive=True)
    row, column = layout.index("vm_9"), net.bus_index["9"]
    assert abs(float(reactive[row, column])) > abs(
        float(real[row, column])
    )


def test_ac_shift_factors_are_close_to_the_dc_ptdf():
    """Same quantity, one exact and one linearised.

    The sign flips because a shift factor is per unit INJECTED and this
    is per unit of DEMAND.
    """
    net = dp.cases.case9()
    ac = dp.branch_flow_sensitivity(net)
    dc = dp.ptdf(net)
    assert ac.shape == dc.shape
    assert float(jnp.max(jnp.abs(-ac - dc))) < 0.1


def test_branch_flow_sensitivity_matches_finite_differences():
    net = dp.cases.case9()
    factors = dp.branch_flow_sensitivity(net)
    pd, qd = net.load_arrays_pu()
    step = 1e-5
    i = net.bus_index["5"]
    k = net.branch_ids.index("br3")
    up = dp.solve_power_flow(net, demand=(pd.at[i].add(step), qd))
    down = dp.solve_power_flow(net, demand=(pd.at[i].add(-step), qd))
    finite = (
        up.branch_mw["br3"][0] - down.branch_mw["br3"][0]
    ) / (2 * step) / net.base_mva
    assert float(factors[k, i]) == pytest.approx(finite, rel=1e-5)


def test_parameter_sensitivity_matches_finite_differences():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    jacobian = dp.parameter_sensitivity(net, "x")
    reactances = net.branch_param_arrays()["x"]

    row = layout.index("vm_9")
    j = net.branch_ids.index("br8")
    step = 1e-7
    up = dp.solve_power_flow(
        net, branch_params={"x": reactances.at[j].add(step)}
    )
    down = dp.solve_power_flow(
        net, branch_params={"x": reactances.at[j].add(-step)}
    )
    finite = (up.vm["9"] - down.vm["9"]) / (2 * step)
    assert float(jacobian[row, j]) == pytest.approx(finite, rel=1e-4)


def test_tap_sensitivity_is_available_on_a_case_with_transformers():
    net = dp.cases.case14()
    jacobian = dp.parameter_sensitivity(net, "tap")
    layout = power_state_layout(net)
    assert jacobian.shape == (layout.size, net.n_branch)
    # Only the three real transformers can do anything with a tap...
    # every branch has one, but raising a line's "tap" also moves things,
    # so just check the transformer's column is not dead.
    j = net.branch_ids.index("br10")            # the 0.932 tap, 5-6
    assert float(jnp.max(jnp.abs(jacobian[:, j]))) > 1e-3


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError, match="unknown branch parameter"):
        dp.parameter_sensitivity(dp.cases.case3(), "resistance")


def test_voltage_stability_margin_falls_as_the_network_is_loaded():
    """It goes to zero at the nose of the P-V curve."""
    base = dp.cases.case9()
    margins = []
    for factor in (1.0, 1.5, 2.0, 2.3):
        net = base.scaled_load(factor)
        result = dp.solve_power_flow(net)
        margins.append(dp.voltage_stability_margin(result.x, net))
    assert margins == sorted(margins, reverse=True)
    assert margins[-1] < 0.5 * margins[0]
