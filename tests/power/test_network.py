"""Tests for difflow_power.network and difflow_power.cases.

The validation is the interesting part. A network object that accepts a
structurally impossible case --- two slack buses, an islanded network,
a PV bus with nothing to hold its voltage --- pushes the failure into
a linear solver, where it surfaces as a singular matrix with no
indication of what is actually wrong.
"""

import math

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
    assert net.is_radial and net.cycle_rank == 0


def test_exactly_one_slack_is_required():
    with pytest.raises(ValueError, match="exactly one slack"):
        minimal(buses={"a": Bus(), "b": Bus()})
    with pytest.raises(ValueError, match="exactly one slack"):
        minimal(
            buses={"a": Bus(kind="slack"), "b": Bus(kind="slack")},
            generators={
                "g": Generator("a", 0.0, 100.0, -50.0, 50.0),
                "h": Generator("b", 0.0, 100.0, -50.0, 50.0),
            },
        )


def test_voltage_controlled_bus_needs_a_generator():
    with pytest.raises(ValueError, match="hosts no generator"):
        minimal(buses={"a": Bus(kind="slack"), "b": Bus(kind="pv")})


def test_islands_are_rejected():
    with pytest.raises(ValueError, match="islands"):
        PowerNetwork(
            buses={
                "a": Bus(kind="slack"), "b": Bus(),
                "c": Bus(), "d": Bus(),
            },
            branches={
                "l1": Branch("a", "b", 0.01, 0.1),
                "l2": Branch("c", "d", 0.01, 0.1),
            },
            generators={"g": Generator("a", 0.0, 100.0, -50.0, 50.0)},
        )


def test_component_on_an_unknown_bus_is_rejected():
    with pytest.raises(ValueError, match="unknown bus"):
        minimal(loads={"d": Load("nowhere", 10.0, 1.0)})


def test_zero_impedance_branch_is_rejected():
    with pytest.raises(ValueError, match="zero impedance"):
        Branch("a", "b", 0.0, 0.0)


def test_self_loop_is_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        Branch("a", "a", 0.01, 0.1)


def test_case_file_sentinels_are_normalised():
    """``tap = 0`` means 1, ``rate = 0`` means unlimited."""
    br = Branch("a", "b", 0.01, 0.1, tap=0.0, rate_mva=0.0)
    assert br.tap == 1.0
    assert br.rate_mva is None
    assert not br.is_transformer


def test_crossed_limits_are_rejected():
    with pytest.raises(ValueError, match="p_min_mw"):
        Generator("a", 100.0, 10.0)
    with pytest.raises(ValueError, match="q_min_mvar"):
        Generator("a", 0.0, 10.0, 50.0, -50.0)
    with pytest.raises(ValueError, match="limits cross"):
        Bus(vm_min=1.1, vm_max=0.9)


def test_loads_on_one_bus_are_summed():
    net = minimal(
        loads={
            "d1": Load("b", 30.0, 5.0),
            "d2": Load("b", 20.0, 5.0),
        }
    )
    pd, qd = net.load_arrays_pu()
    i = net.bus_index["b"]
    assert float(pd[i]) == pytest.approx(0.5)
    assert float(qd[i]) == pytest.approx(0.1)


def test_cycle_rank_counts_loops():
    assert dp.cases.case3().cycle_rank == 1
    assert dp.cases.case9().cycle_rank == 1
    assert dp.cases.case14().cycle_rank == 7
    assert dp.cases.radial_feeder().cycle_rank == 0
    assert dp.cases.radial_feeder().is_radial


def test_scaled_load_and_with_kinds_do_not_mutate_the_original():
    net = dp.cases.case9()
    heavy = net.scaled_load(1.5)
    assert heavy.total_load_mw == pytest.approx(1.5 * net.total_load_mw)
    assert net.total_load_mw == pytest.approx(315.0)

    swapped = net.with_kinds({"2": "slack", "1": "pv"})
    assert swapped.slack_bus == "2"
    assert net.slack_bus == "1"


def test_with_kinds_rejects_unknown_buses():
    with pytest.raises(KeyError):
        dp.cases.case9().with_kinds({"99": "pq"})


def test_bus_order_is_insertion_order_not_sorted():
    """Sorting bus labels as strings would put "10" between "1" and "2"."""
    net = dp.cases.case14()
    assert net.bus_ids[:3] == ["1", "2", "3"]
    assert net.bus_ids[9] == "10"
    assert net.bus_ids != sorted(net.bus_ids)


def test_unknown_branch_parameter_is_rejected():
    with pytest.raises(KeyError, match="unknown branch parameters"):
        dp.cases.case9().ybus({"nonsense": None})


# --- case files -------------------------------------------------------


def test_case9_matches_the_matpower_case_file():
    net = dp.cases.case9()
    assert net.base_mva == 100.0
    assert net.n_bus == 9 and net.n_branch == 9 and net.n_gen == 3
    assert net.total_load_mw == pytest.approx(315.0)
    assert net.total_load_mvar == pytest.approx(115.0)
    assert net.slack_bus == "1"
    assert net.buses_of_kind("pv") == ["2", "3"]
    assert net.generators["g2"].p_max_mw == 300.0
    assert net.generators["g1"].cost == (0.11, 5.0, 150.0)


def test_case14_carries_transformers_and_a_shunt():
    net = dp.cases.case14()
    transformers = [
        aid for aid, br in net.branches.items() if br.is_transformer
    ]
    assert len(transformers) == 3
    taps = sorted(net.branches[a].tap for a in transformers)
    assert taps == pytest.approx([0.932, 0.969, 0.978])
    assert net.buses["9"].b_shunt_mvar == pytest.approx(19.0)
    # Every branch is unrated in this case file (rateA = 0).
    assert all(br.rate_mva is None for br in net.branches.values())


def test_case5_has_two_units_on_one_bus_and_two_rated_branches():
    net = dp.cases.case5()
    assert net.generators_at("1") == ["g1", "g2"]
    rated = [a for a, br in net.branches.items() if br.rate_mva is not None]
    assert len(rated) == 2


def test_matpower_import_converts_degrees_and_drops_out_of_service():
    mpc = {
        "baseMVA": 100.0,
        "bus": [
            [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
            [2, 1, 50, 10, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
        ],
        "gen": [
            [1, 0, 0, 50, -50, 1.0, 100, 1, 100, 0],
            [2, 0, 0, 50, -50, 1.0, 100, 0, 100, 0],     # out of service
        ],
        "branch": [
            [1, 2, 0.01, 0.1, 0.02, 100, 0, 0, 0.98, 30, 1, -30, 30],
            [1, 2, 0.02, 0.2, 0.0, 0, 0, 0, 0, 0, 0, -360, 360],  # off
        ],
        "gencost": [[2, 0, 0, 3, 0.1, 20, 5], [2, 0, 0, 3, 0.1, 20, 5]],
    }
    net = dp.cases.from_matpower(mpc, name="two bus")
    assert net.n_gen == 1 and net.n_branch == 1
    br = net.branches["br1"]
    assert br.shift == pytest.approx(math.radians(30.0))
    assert br.angle_max == pytest.approx(math.radians(30.0))
    assert br.rate_mva == 100.0
    assert net.generators["g1"].cost == (0.1, 20.0, 5.0)


def test_piecewise_linear_cost_is_refused_with_a_reason():
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


def test_load_case_lists_what_it_has():
    with pytest.raises(KeyError, match="available"):
        dp.cases.load_case("case1000")
    assert set(dp.cases.CASES) == {
        "case3", "case5", "case9", "case14", "radial_feeder"
    }


@pytest.mark.parametrize("case", sorted(dp.cases.CASES))
def test_every_case_summarises_and_builds_a_ybus(case):
    net = dp.cases.load_case(case)
    assert net.name in repr(net)
    assert net.ybus().shape == (net.n_bus, net.n_bus)
    assert len(net.islands()) == 1
