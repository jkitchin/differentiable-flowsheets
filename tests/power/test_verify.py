"""Tests for difflow_power.verify.

The distinction the report exists to draw is between a state that
SOLVES the equations and one that is also FEASIBLE. A power flow always
produces the first and routinely fails the second, which is the whole
reason an OPF exists.
"""

import jax

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.residuals import power_state_layout


def test_a_solved_power_flow_solves_and_is_feasible():
    net = dp.cases.case9()
    report = dp.operating_report(dp.solve_power_flow(net).x, net)
    assert report.solved
    assert report.feasible
    assert report.max_p_mismatch_mw < 1e-9
    assert abs(report.angle_reference_error_rad) < 1e-12
    assert not report.voltage_violations


def test_an_unsolved_state_is_reported_as_unsolved():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    report = dp.operating_report(dp.flat_start(net, layout), net)
    assert not report.solved
    assert not report.feasible
    assert report.max_p_mismatch_mw > 1.0


def test_an_overloaded_case_solves_but_is_not_feasible():
    """Exactly the case an OPF is for."""
    net = dp.cases.case9().scaled_load(1.8)
    report = dp.operating_report(dp.solve_power_flow(net).x, net)
    assert report.solved
    assert not report.feasible
    assert report.voltage_violations or report.thermal_violations
    assert report.worst_loading > 0.0


def test_violations_carry_their_magnitude_and_sign():
    net = dp.cases.case9()
    net.buses["9"].vm_min = 1.0          # bus 9 solves near 0.958
    report = dp.operating_report(dp.solve_power_flow(net).x, net)
    assert "9" in report.voltage_violations
    assert report.voltage_violations["9"] < 0.0     # under, not over


def test_branch_losses_are_non_negative_on_every_case():
    """The cheapest useful check on a new case file."""
    for case in dp.cases.CASES:
        net = dp.cases.load_case(case)
        losses = dp.branch_loss_report(dp.solve_power_flow(net).x, net)
        assert set(losses) == set(net.branch_ids)
        assert min(losses.values()) > -1e-12
        assert sum(losses.values()) > 0.0


def test_report_summary_reads():
    net = dp.cases.case9().scaled_load(1.8)
    report = dp.operating_report(dp.solve_power_flow(net).x, net)
    text = report.summary()
    assert "mismatch" in text
    assert "OperatingReport" in repr(report)
