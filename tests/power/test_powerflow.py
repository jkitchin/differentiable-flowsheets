"""Tests for difflow_power.powerflow.

The load-bearing tests assert MATPOWER's published answers for the
standard cases, digit for digit. That is the only check that catches
the mistakes a power-flow implementation actually makes: a self-
consistent tool with the phase-shift sign backwards, or the charging
susceptance halved twice, converges beautifully to the wrong numbers.

MATPOWER reference values (``runpf`` on the same case files reproduced
in :mod:`difflow_power.cases`):

case9   Pg = (71.955, 163, 85) MW, Qg = (24.07, 14.46, -3.65) MVAr,
        Va2 = 9.6687 deg, Va3 = 4.7711 deg, losses 4.9547 MW
case14  Pg1 = 232.39 MW, Vm8 = 1.09 pu, losses 13.393 MW
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.powerflow import (
    solve_state,
    specification_from_network,
    specification_names,
)
from difflow_power.residuals import power_state_layout


def test_case9_matches_matpower():
    result = dp.solve_power_flow(dp.cases.case9())
    assert result.converged
    assert result.pg_mw["g1"] == pytest.approx(71.955, abs=1e-3)
    assert result.pg_mw["g2"] == pytest.approx(163.0, abs=1e-6)
    assert result.pg_mw["g3"] == pytest.approx(85.0, abs=1e-6)
    assert result.qg_mvar["g1"] == pytest.approx(24.069, abs=1e-3)
    assert result.qg_mvar["g2"] == pytest.approx(14.460, abs=1e-3)
    assert result.qg_mvar["g3"] == pytest.approx(-3.649, abs=1e-3)
    assert result.va_degrees["2"] == pytest.approx(9.6687, abs=1e-4)
    assert result.va_degrees["3"] == pytest.approx(4.7711, abs=1e-4)
    assert result.vm["9"] == pytest.approx(0.9576, abs=1e-4)
    assert result.losses_mw == pytest.approx(4.9547, abs=1e-3)
    assert result.max_mismatch_mw < 1e-9


def test_case14_matches_matpower():
    result = dp.solve_power_flow(dp.cases.case14())
    assert result.converged
    assert result.pg_mw["g1"] == pytest.approx(232.393, abs=1e-2)
    assert result.losses_mw == pytest.approx(13.393, abs=1e-2)
    # Every voltage-controlled bus sits exactly on its setpoint.
    for gid, gen in dp.cases.case14().generators.items():
        assert result.vm[gen.bus] == pytest.approx(gen.vm_setpoint, abs=1e-9)


def test_case14_reproduces_the_case_files_own_solution():
    """The IEEE case file ships with its solved state in the bus rows.

    The tolerance is 2e-3, not solver precision: those values were
    converted from the 1962 IEEE Common Data Format file, rounded to
    three decimals, and reflect a solution computed from very slightly
    different data. Agreeing with them to a millivolt per unit on every
    bus is the strongest claim the data supports; agreeing to solver
    precision would mean the reference had been fitted to, not checked
    against.
    """
    net = dp.cases.case14()
    result = dp.solve_power_flow(net)
    published_vm = {
        "1": 1.06, "2": 1.045, "3": 1.01, "4": 1.019, "5": 1.02,
        "6": 1.07, "7": 1.062, "8": 1.09, "9": 1.056, "10": 1.051,
        "11": 1.057, "12": 1.055, "13": 1.05, "14": 1.036,
    }
    for bus, value in published_vm.items():
        assert result.vm[bus] == pytest.approx(value, abs=2e-3)


@pytest.mark.parametrize("case", sorted(dp.cases.CASES))
def test_every_case_converges_and_balances(case):
    net = dp.cases.load_case(case)
    result = dp.solve_power_flow(net)
    assert result.converged
    assert result.max_mismatch_mw < 1e-8
    # Generation equals load plus losses, to solver precision.
    assert result.total_generation_mw == pytest.approx(
        net.total_load_mw + result.losses_mw, abs=1e-6
    )
    assert result.losses_mw > 0.0


def test_specification_is_square():
    """``2 n_gen - 1`` setpoint rows, for any arrangement of units."""
    for case in dp.cases.CASES:
        net = dp.cases.load_case(case)
        layout = power_state_layout(net)
        spec = specification_from_network(net)
        names = specification_names(net, spec)
        assert len(names) == 2 * net.n_gen - 1
        assert layout.n_residual + len(names) == layout.size


def test_shared_bus_units_split_vars_by_capability():
    """case5 has two units on bus 1 with a 1:4.25 var range ratio."""
    net = dp.cases.case5()
    result = dp.solve_power_flow(net)
    q1, q2 = result.qg_mvar["g1"], result.qg_mvar["g2"]
    ranges = [
        net.generators[g].q_max_mvar - net.generators[g].q_min_mvar
        for g in ("g1", "g2")
    ]
    assert q1 / (q1 + q2) == pytest.approx(ranges[0] / sum(ranges), rel=1e-9)


def test_disagreeing_voltage_setpoints_on_one_bus_are_rejected():
    net = dp.cases.case5()
    net.generators["g2"].vm_setpoint = 1.03
    with pytest.raises(ValueError, match="different voltages"):
        specification_from_network(net)


def test_violations_are_reported_not_enforced():
    """A power flow is not an OPF: it solves the equations and no more."""
    net = dp.cases.case9().scaled_load(1.8)
    result = dp.solve_power_flow(net)
    assert result.converged
    violations = result.violations()
    assert violations, "a heavily loaded case9 must break something"
    assert any(k.startswith(("vm_", "rate_", "pg_")) for k in violations)


def test_non_convergence_is_reported_not_raised():
    """Past the loadability limit the solution ceases to exist."""
    net = dp.cases.case9().scaled_load(3.0)
    result = dp.solve_power_flow(net, max_steps=15)
    assert not result.converged
    assert "DID NOT CONVERGE" in result.summary()


def test_demand_override_beats_the_networks_own_loads():
    net = dp.cases.case9()
    pd, qd = net.load_arrays_pu()
    result = dp.solve_power_flow(net, demand=(pd * 1.1, qd * 1.1))
    scaled = dp.solve_power_flow(net.scaled_load(1.1))
    assert result.total_generation_mw == pytest.approx(
        scaled.total_generation_mw, abs=1e-8
    )


def test_solution_differentiates_with_respect_to_demand():
    """Implicit function theorem, checked against finite differences."""
    net = dp.cases.case9()
    layout = power_state_layout(net)
    spec = specification_from_network(net)
    x0 = dp.flat_start(net, layout, spec)
    pd, qd = net.load_arrays_pu()

    def slack_mw(demand_block):
        x = solve_state(net, layout, spec, x0, demand=(demand_block, qd))[0]
        return x[layout.slice_pg][0] * net.base_mva

    grad = jax.grad(slack_mw)(pd)
    step = 1e-6
    for i in (4, 6, 8):
        up = slack_mw(pd.at[i].add(step))
        down = slack_mw(pd.at[i].add(-step))
        assert float(grad[i]) == pytest.approx(
            float((up - down) / (2 * step)), rel=1e-6
        )
    # A MW of load costs the slack a MW plus the marginal loss.
    assert float(grad[4]) == pytest.approx(100.0, abs=5.0)


def test_solution_differentiates_with_respect_to_a_setpoint():
    """The AVR knob an operator actually turns."""
    net = dp.cases.case9()
    layout = power_state_layout(net)
    x0 = dp.flat_start(net, layout)

    def losses(vm_set):
        spec = specification_from_network(net)
        spec.vm_setpoint["2"] = vm_set
        x = solve_state(net, layout, spec, x0)[0]
        return dp.total_losses(x, net, layout)

    grad = jax.grad(losses)(1.0)
    step = 1e-6
    want = (losses(1.0 + step) - losses(1.0 - step)) / (2 * step)
    assert float(grad) == pytest.approx(float(want), rel=1e-5)
    # Raising a generator's voltage reduces losses here.
    assert float(grad) < 0.0


def test_solve_state_is_jittable():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    spec = specification_from_network(net)
    x0 = dp.flat_start(net, layout, spec)
    pd, qd = net.load_arrays_pu()

    jitted = jax.jit(
        lambda demand: solve_state(net, layout, spec, x0, demand=demand)[0]
    )
    np.testing.assert_allclose(
        jitted((pd, qd)),
        solve_state(net, layout, spec, x0, demand=(pd, qd))[0],
        atol=1e-10,
    )


def test_result_reports_read_in_engineering_units():
    net = dp.cases.case9()
    result = dp.solve_power_flow(net)
    assert result.vm_kv["1"] == pytest.approx(345.0, abs=1e-6)
    assert set(result.branch_loading) == set(net.branch_ids)
    assert max(result.branch_loading.values()) < 1.0
    assert "converged" in result.summary()
    from_mw, to_mw = result.branch_mw["br1"]
    assert from_mw > 0 and to_mw < 0        # power leaves bus 1 for bus 4
