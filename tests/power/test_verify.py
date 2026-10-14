"""Smoke tests for difflow_power.verify."""

import jax

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.residuals import power_state_layout


def test_operating_report_smoke():
    net = dp.cases.case9()
    x = dp.solve_power_flow(net).x
    report = dp.operating_report(x, net)
    assert report.solved
    assert "mismatch" in report.summary()
    assert set(dp.branch_loss_report(x, net)) == set(net.branch_ids)


def test_an_unsolved_state_is_reported_as_unsolved():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    report = dp.operating_report(dp.flat_start(net, layout), net)
    assert not report.solved
