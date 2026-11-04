"""Water distribution networks: they solve, are finite, differentiate; inputs validated."""

import jax
import jax.numpy as jnp
import pytest

from difflow_water.network import (
    NetworkConvergenceWarning,
    NetworkModel,
    WaterNetwork,
    load_example,
)


def small(prv=True, pump=True):
    net = WaterNetwork()
    net.add_reservoir("R", head=50.0)
    net.add_junction("A", elevation=10.0)
    net.add_junction("B", elevation=5.0, demand=0.01, emitter=1e-3)
    net.add_junction("C", elevation=0.0, demand=0.005)
    if pump:
        net.add_pump("PU", "R", "A", curve=[(0.02, 40.0)])
    else:
        net.add_pipe("P0", "R", "A", 100.0, 0.2, 120.0)
    net.add_pipe("P1", "A", "B", 500.0, 0.15, 110.0, minor_loss=2.0)
    if prv:
        net.add_valve("V", "B", "C", "PRV", diameter=0.1, setting=30.0)
    else:
        net.add_pipe("P2", "B", "C", 300.0, 0.1, 100.0, check_valve=True)
    return net


@pytest.mark.parametrize("formulation", ["complementarity", "smooth"])
@pytest.mark.parametrize("prv", [True, False], ids=["prv", "check_valve"])
def test_network_solves(formulation, prv):
    res = NetworkModel(small(prv=prv), valve_formulation=formulation).solve()
    for k in ("head", "pressure", "flow", "demand", "emitter_flow", "headloss", "head_gain"):
        assert jnp.all(jnp.isfinite(res[k])), k
    assert res["converged"]


def test_pressure_driven_demand_and_darcy_weisbach_run():
    net = small(pump=False)
    net.headloss = "D-W"
    for p in net.pipes.values():
        p.roughness = 2.6e-4
    res = NetworkModel(net, demand_model="PDD", pdd=(0.0, 20.0, 0.5)).solve()
    assert res["converged"] and jnp.all(jnp.isfinite(res["demand"]))


def test_network_gradient_is_finite():
    model = NetworkModel(small())
    p0 = model.parameters()

    def f(demand, rough):
        return jnp.sum(model.solve({"demand": demand, "roughness": rough})["pressure"])
    g = jax.grad(f, argnums=(0, 1))(p0["demand"], p0["roughness"])
    assert all(jnp.all(jnp.isfinite(x)) for x in g)


def test_example_networks_load_and_solve():
    net, epanet = load_example("Net1")
    assert set(epanet) == {"head", "pressure", "demand", "flow"}
    assert NetworkModel(net).solve()["converged"]


def test_a_pump_with_no_flow_path_is_closed():
    """API contract: a pump feeding only a closed valve is compiled as closed."""
    net = small(prv=False)
    net.pipes["P2"].status = "closed"
    net.junctions["B"].demand = 0.0
    net.junctions["B"].emitter = 0.0
    net.junctions["A"].demand = 0.0
    assert net.compile().auto_closed == ["PU"]


def test_non_convergence_warns():
    net, _ = load_example("Net3")
    with pytest.warns(NetworkConvergenceWarning):
        NetworkModel(net, valve_formulation="smooth", smoothing=0.1, n_iter=1).solve()


@pytest.mark.parametrize("bad", ["kind", "formulation", "demand_model", "no_source",
                                 "endpoint", "param", "shape", "pump", "headloss"])
def test_input_validation(bad):
    net = small()
    with pytest.raises(ValueError):
        if bad == "kind":
            net.add_valve("X", "A", "B", "GPV", 0.1, 1.0)
        elif bad == "formulation":
            NetworkModel(net, valve_formulation="exact")
        elif bad == "demand_model":
            NetworkModel(net, demand_model="CDA")
        elif bad == "no_source":
            del net.reservoirs["R"]
            net.compile()
        elif bad == "endpoint":
            net.add_pipe("X", "A", "nowhere", 10.0, 0.1, 100.0)
            net.compile()
        elif bad == "param":
            NetworkModel(net).solve({"viscosity": 1.0})
        elif bad == "shape":
            NetworkModel(net).solve({"setting": [30.0]})
        elif bad == "pump":
            net.add_pump("X", "R", "A", curve=[(0.01, 10.0)], power=1e3)
        else:
            net.headloss = "C-M"
            net.compile()
