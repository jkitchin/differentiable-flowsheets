"""Tests for difflow_power.network and difflow_power.cases: build and validation."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.network import Branch, Bus, Generator, Load, PowerNetwork


def minimal(**overrides):
    kwargs = dict(
        buses={"a": Bus(kind="slack"), "b": Bus()},
        branches={"l": Branch("a", "b", 0.01, 0.1)},
        generators={"g": Generator("a", 0.0, 100.0, -50.0, 50.0)},
        loads={"d": Load("b", 50.0, 10.0)},
    )
    kwargs.update(overrides)
    return PowerNetwork(**kwargs)


def test_minimal_network_builds():
    net = minimal()
    assert net.n_bus == 2 and net.n_branch == 1 and net.n_gen == 1
    assert net.slack_bus == "a"


def test_every_case_builds_a_ybus():
    for case in sorted(dp.cases.CASES):
        net = dp.cases.load_case(case)
        assert net.ybus().shape == (net.n_bus, net.n_bus)


def test_exactly_one_slack_is_required():
    with pytest.raises(ValueError, match="exactly one slack"):
        minimal(buses={"a": Bus(), "b": Bus()})


def test_voltage_controlled_bus_needs_a_generator():
    with pytest.raises(ValueError, match="hosts no generator"):
        minimal(buses={"a": Bus(kind="slack"), "b": Bus(kind="pv")})


def test_islands_are_rejected():
    with pytest.raises(ValueError, match="islands"):
        PowerNetwork(
            buses={"a": Bus(kind="slack"), "b": Bus(), "c": Bus(), "d": Bus()},
            branches={
                "l1": Branch("a", "b", 0.01, 0.1),
                "l2": Branch("c", "d", 0.01, 0.1),
            },
            generators={"g": Generator("a", 0.0, 100.0, -50.0, 50.0)},
        )


def test_component_on_an_unknown_bus_is_rejected():
    with pytest.raises(ValueError, match="unknown bus"):
        minimal(loads={"d": Load("nowhere", 10.0, 1.0)})


def test_bad_branch_is_rejected():
    with pytest.raises(ValueError, match="zero impedance"):
        Branch("a", "b", 0.0, 0.0)
    with pytest.raises(ValueError, match="self-loop"):
        Branch("a", "a", 0.01, 0.1)


def test_crossed_limits_are_rejected():
    with pytest.raises(ValueError, match="p_min_mw"):
        Generator("a", 100.0, 10.0)
    with pytest.raises(ValueError, match="limits cross"):
        Bus(vm_min=1.1, vm_max=0.9)


def test_unknown_names_are_rejected():
    net = dp.cases.case9()
    with pytest.raises(KeyError):
        net.with_kinds({"99": "pq"})
    with pytest.raises(KeyError, match="unknown branch parameters"):
        net.ybus({"nonsense": None})
    with pytest.raises(KeyError, match="available"):
        dp.cases.load_case("case1000")


def test_from_matpower_smoke():
    mpc = {
        "baseMVA": 100.0,
        "bus": [[1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
                [2, 1, 10, 1, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]],
        "gen": [[1, 0, 0, 50, -50, 1, 100, 1, 100, 0]],
        "branch": [[1, 2, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 1, -360, 360]],
        "gencost": [[2, 0, 0, 3, 0.1, 20, 5]],
    }
    net = dp.cases.from_matpower(mpc, name="two bus")
    assert net.n_bus == 2 and net.n_gen == 1 and net.n_branch == 1


def test_piecewise_linear_cost_is_refused():
    mpc = {
        "baseMVA": 100.0,
        "bus": [[1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
                [2, 1, 10, 1, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]],
        "gen": [[1, 0, 0, 50, -50, 1, 100, 1, 100, 0]],
        "branch": [[1, 2, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 1, -360, 360]],
        "gencost": [[1, 0, 0, 2, 0, 0, 100, 2000]],
    }
    with pytest.raises(ValueError, match="piecewise-linear"):
        dp.cases.from_matpower(mpc)
