"""Tests for difflow_power.flowsheet: the backward/forward sweep.

The sweep and Newton's method are completely different algorithms
solving the same equations, so agreeing to solver precision is a real
check on both. They share nothing but the network data: the sweep goes
through the unit-level admittance blocks in tree order, Newton through
the packed residual vector and a Jacobian factorisation.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.flowsheet import (
    RadialFeederFlowsheet,
    build_ladder_flowsheet,
    feeder_tree,
)
from difflow_power.network import PowerNetwork
from difflow_power.streams import power_stream


def ladder_network():
    """``radial_feeder`` with its two laterals removed: a pure chain."""
    net = dp.cases.radial_feeder()
    return PowerNetwork(
        name="ladder",
        base_mva=net.base_mva,
        buses={k: v for k, v in net.buses.items() if k not in ("l1", "l2")},
        branches={
            k: v for k, v in net.branches.items() if k not in ("b1", "b2")
        },
        generators=net.generators,
        loads={
            k: v for k, v in net.loads.items() if k not in ("dl1", "dl2")
        },
    )


def test_feeder_tree_roots_at_the_slack():
    net = dp.cases.radial_feeder()
    tree = feeder_tree(net)
    assert tree.root == "s"
    assert tree.order[0] == "s"
    assert set(tree.order) == set(net.bus_ids)
    assert tree.parent["n1"][0] == "s"
    assert set(tree.children["n2"]) == {"n3", "l1"}
    # Every non-root bus has exactly one parent.
    assert len(tree.parent) == net.n_bus - 1


def test_a_meshed_network_is_refused():
    with pytest.raises(ValueError, match="radial network"):
        RadialFeederFlowsheet(dp.cases.case9())


def test_sweep_matches_newton_on_every_bus():
    net = dp.cases.radial_feeder()
    streams = RadialFeederFlowsheet(net).solve()
    reference = dp.solve_power_flow(net)
    for bus in net.bus_ids:
        assert float(streams[f"bus_{bus}"]["P"]) == pytest.approx(
            reference.vm[bus], abs=1e-9
        )
        assert float(
            jnp.degrees(streams[f"bus_{bus}"]["T"])
        ) == pytest.approx(reference.va_degrees[bus], abs=1e-7)


def test_sweep_matches_newton_on_the_substation_infeed():
    net = dp.cases.radial_feeder()
    flowsheet = RadialFeederFlowsheet(net)
    streams = flowsheet.solve()
    reference = dp.solve_power_flow(net)
    assert float(streams["bus_s"]["F_P"]) * net.base_mva == pytest.approx(
        reference.pg_mw["sub"], abs=1e-7
    )
    assert flowsheet.last_solve_stats["converged"]


def test_sweep_branch_flows_match_the_equation_oriented_ones():
    net = dp.cases.radial_feeder()
    streams = RadialFeederFlowsheet(net).solve()
    reference = dp.solve_power_flow(net)
    s_from, _ = reference.flows()
    for i, aid in enumerate(net.branch_ids):
        assert float(streams[f"branch_{aid}"]["F_P"]) == pytest.approx(
            float(jnp.real(s_from[i])), abs=1e-9
        )


def test_the_two_ends_of_a_branch_carry_different_currents():
    """What the backward pass has to get right.

    Reusing a child's own accumulated current at the parent's end is
    the textbook shortcut, and it is wrong by the charging current. On a
    feeder with line charging that is a real error, so this asserts the
    two are genuinely different rather than trusting they are close.
    """
    net = dp.cases.radial_feeder()
    charged = replace(
        net,
        branches={
            aid: replace(br, b=0.05) for aid, br in net.branches.items()
        },
    )
    flowsheet = RadialFeederFlowsheet(charged)
    voltages, _ = flowsheet.solve_voltages()
    tree = flowsheet.tree
    near = flowsheet._current_into("t1", True, voltages["n1"], voltages["n2"])
    far = flowsheet._current_into("t1", False, voltages["n2"], voltages["n1"])
    assert abs(complex(near) + complex(far)) > 1e-4

    # And the sweep still lands on Newton's answer with charging present.
    streams = flowsheet.solve()
    reference = dp.solve_power_flow(charged)
    for bus in charged.bus_ids:
        assert float(streams[f"bus_{bus}"]["P"]) == pytest.approx(
            reference.vm[bus], abs=1e-9
        )


def test_sweep_handles_a_transformer_in_the_feeder():
    net = dp.cases.radial_feeder()
    tapped = replace(
        net,
        branches={
            **net.branches,
            "t1": replace(net.branches["t1"], tap=0.975),
        },
    )
    streams = RadialFeederFlowsheet(tapped).solve()
    reference = dp.solve_power_flow(tapped)
    for bus in tapped.bus_ids:
        assert float(streams[f"bus_{bus}"]["P"]) == pytest.approx(
            reference.vm[bus], abs=1e-9
        )


def test_voltage_sags_away_from_the_substation():
    """The defining behaviour of a load feeder."""
    net = dp.cases.radial_feeder()
    streams = RadialFeederFlowsheet(net).solve()
    trunk = ["s", "n1", "n2", "n3", "n4"]
    magnitudes = [float(streams[f"bus_{b}"]["P"]) for b in trunk]
    assert magnitudes == sorted(magnitudes, reverse=True)


@pytest.mark.release
def test_sweep_objective_is_differentiable_in_demand():
    net = dp.cases.radial_feeder()
    flowsheet = RadialFeederFlowsheet(net)
    pd, qd = net.load_arrays_pu()

    objective = flowsheet.make_objective_fn(
        lambda streams: jnp.asarray(streams["bus_s"]["F_P"])
    )
    grad = jax.grad(lambda p: objective((p, qd)))(pd)
    step = 1e-7
    i = net.bus_index["n4"]
    want = (
        objective((pd.at[i].add(step), qd))
        - objective((pd.at[i].add(-step), qd))
    ) / (2 * step)
    assert float(grad[i]) == pytest.approx(float(want), rel=1e-5)
    # A pu of load at the far end costs the substation more than a pu.
    assert float(grad[i]) > 1.0


def test_ladder_flowsheet_matches_newton():
    net = ladder_network()
    flowsheet, order = build_ladder_flowsheet(net)
    streams = flowsheet.solve(
        tear_initial={"infeed": power_stream(0.35, 0.14, 1.02, 0.0)},
        tol=1e-13, max_iter=100, clip_negative_flows=False,
    )
    assert flowsheet.last_solve_converged
    reference = dp.solve_power_flow(net)
    for bus in order:
        assert float(streams[f"bus_{bus}"]["P"]) == pytest.approx(
            reference.vm[bus], abs=1e-9
        )
    assert float(streams["infeed"]["F_P"]) * net.base_mva == pytest.approx(
        reference.pg_mw["sub"], abs=1e-7
    )


def test_ladder_flowsheet_converges_in_a_handful_of_tears():
    """The infeed correction is nearly an exact Newton step."""
    flowsheet, _ = build_ladder_flowsheet(ladder_network())
    flowsheet.solve(
        tear_initial={"infeed": power_stream(0.0, 0.0, 1.02, 0.0)},
        tol=1e-12, max_iter=100, clip_negative_flows=False,
    )
    assert flowsheet.last_solve_converged
    assert flowsheet.last_solve_iterations <= 10


def test_a_branching_feeder_is_not_a_ladder():
    with pytest.raises(ValueError, match="not a ladder"):
        build_ladder_flowsheet(dp.cases.radial_feeder())


def test_stream_names_are_stable():
    assert dp.bus_stream_name("n1") == "bus_n1"
    assert dp.branch_stream_name("t0") == "branch_t0"


# -- every injection is in the ladder (audit, 2026-10) ----------------------
#
# The builder used to place a LoadDraw per non-root bus and nothing else:
# a root-bus load, every shunt and every non-slack generator were dropped,
# and a tapped branch stored against the chain kept its impedance
# unreferred, all while the solve reported convergence. Each variant below
# failed by the amount in its comment; RadialFeederFlowsheet and Newton
# agree on all of them to 1e-12.


def _ladder_variant(variant):
    from difflow_power.network import Branch, Generator, Load

    net = ladder_network()
    buses, branches = dict(net.buses), dict(net.branches)
    gens, loads = dict(net.generators), dict(net.loads)
    if variant == "root_load":          # infeed was 0.800 MW low
        loads["d0"] = Load("s", 0.80, 0.30)
    elif variant == "shunt":            # 0.195 MW, 1.6e-3 pu off
        buses["n3"] = replace(buses["n3"], g_shunt_mw=0.2, b_shunt_mvar=0.3)
    elif variant == "root_shunt":
        buses["s"] = replace(buses["s"], g_shunt_mw=0.1, b_shunt_mvar=-0.2)
    elif variant == "dg":               # 0.533 MW, 7.4e-3 pu off
        gens["pv"] = Generator("n3", 0.0, 1.0, 0.0, 0.0, cost=(0.0,),
                               p_mw=0.5, q_mvar=0.1)
    elif variant == "tap_reversed":     # 0.0043 MW, 1.9e-3 pu off
        branches["t1"] = Branch("n2", "n1", 0.045, 0.050, 0.02, tap=1.05,
                                shift=0.03)
    return PowerNetwork(name="ladder-" + variant, base_mva=net.base_mva,
                        buses=buses, branches=branches, generators=gens,
                        loads=loads)


@pytest.mark.parametrize(
    "variant", ["root_load", "shunt", "root_shunt", "dg", "tap_reversed"]
)
def test_ladder_flowsheet_carries_every_injection(variant):
    net = _ladder_variant(variant)
    flowsheet, order = build_ladder_flowsheet(net)
    streams = flowsheet.solve(
        tear_initial={"infeed": power_stream(0.35, 0.14, 1.02, 0.0)},
        tol=1e-13, max_iter=100, clip_negative_flows=False,
    )
    assert flowsheet.last_solve_converged
    sweep = RadialFeederFlowsheet(net).solve()
    reference = dp.solve_power_flow(net)
    for bus in order:
        for slot in ("P", "T"):         # voltage magnitude and angle
            assert float(streams[f"bus_{bus}"][slot]) == pytest.approx(
                float(sweep[f"bus_{bus}"][slot]), abs=1e-9
            )
    # The infeed is the slack's whole output, root load and shunt included.
    assert float(streams["infeed"]["F_P"]) * net.base_mva == pytest.approx(
        reference.pg_mw["sub"], abs=1e-7
    )
    assert float(streams["infeed"]["F_Q"]) * net.base_mva == pytest.approx(
        reference.qg_mvar["sub"], abs=1e-7
    )


def test_ladder_flowsheet_refuses_a_downstream_pv_bus():
    from difflow_power.network import Generator

    net = ladder_network()
    gens = dict(net.generators)
    gens["pv"] = Generator("n3", 0.0, 1.0, -1.0, 1.0, p_mw=0.5)
    pv_net = PowerNetwork(
        name="ladder-pv", base_mva=net.base_mva,
        buses={**net.buses, "n3": replace(net.buses["n3"], kind="pv")},
        branches=net.branches, generators=gens, loads=net.loads,
    )
    with pytest.raises(ValueError, match="PV buses"):
        build_ladder_flowsheet(pv_net)
