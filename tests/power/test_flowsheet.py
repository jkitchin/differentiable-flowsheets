"""Smoke and validation tests for difflow_power.flowsheet."""

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
    assert set(tree.order) == set(net.bus_ids)
    assert len(tree.parent) == net.n_bus - 1


def test_radial_sweep_smoke():
    net = dp.cases.radial_feeder()
    flowsheet = RadialFeederFlowsheet(net)
    streams = flowsheet.solve()
    assert flowsheet.last_solve_stats["converged"]
    for bus in net.bus_ids:
        assert bool(jnp.isfinite(streams[f"bus_{bus}"]["P"]))


def test_sweep_objective_grad_is_finite():
    net = dp.cases.radial_feeder()
    flowsheet = RadialFeederFlowsheet(net)
    pd, qd = net.load_arrays_pu()
    objective = flowsheet.make_objective_fn(
        lambda streams: jnp.asarray(streams["bus_s"]["F_P"])
    )
    grad = jax.grad(lambda p: objective((p, qd)))(pd)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_ladder_flowsheet_smoke():
    flowsheet, order = build_ladder_flowsheet(ladder_network())
    streams = flowsheet.solve(
        tear_initial={"infeed": power_stream(0.35, 0.14, 1.02, 0.0)},
        tol=1e-12, max_iter=100, clip_negative_flows=False,
    )
    assert flowsheet.last_solve_converged
    assert bool(jnp.isfinite(streams["infeed"]["F_P"]))


def test_a_meshed_network_is_refused():
    with pytest.raises(ValueError, match="radial network"):
        RadialFeederFlowsheet(dp.cases.case9())


def test_a_branching_feeder_is_not_a_ladder():
    with pytest.raises(ValueError, match="not a ladder"):
        build_ladder_flowsheet(dp.cases.radial_feeder())


def test_downstream_pv_bus_is_refused():
    from dataclasses import replace

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
    with pytest.raises(ValueError, match="PV buses"):
        RadialFeederFlowsheet(pv_net)


def test_stream_names_are_stable():
    assert dp.bus_stream_name("n1") == "bus_n1"
    assert dp.branch_stream_name("t0") == "branch_t0"
