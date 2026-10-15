"""Tests for difflow.pipe_network (looped/branched liquid pipe networks)."""

import jax
import numpy as np
import pytest

from difflow import Pipe, PipeNetwork, PipeParams, ResistanceBranch

jax.config.update("jax_enable_x64", True)

WATER = dict(rho=998.2, mu=1.002e-3)
ENDS = {"AB": ("A", "B"), "AD": ("A", "D"), "DB": ("D", "B"), "BC": ("B", "C"),
        "CE": ("C", "E"), "DE": ("D", "E")}
DEMAND = {"B": 0.02, "C": 0.025, "D": 0.03, "E": 0.025}


def _two_loop(demand=DEMAND):
    net = PipeNetwork()
    net.add_node("A", pressure=5e5)
    for node, d in demand.items():
        net.add_node(node, demand=d)
    for name, (a, b) in ENDS.items():
        net.add_branch(name, a, b, ResistanceBranch(r=1e8, n=2.0))
    return net


@pytest.mark.parametrize("method", ["newton", "hardy_cross"])
def test_two_loop_network_smoke(method):
    net = _two_loop()
    res = net.solve(method=method, iterations=300)
    assert bool(res.converged)
    assert len(net.loops()) == 2
    assert np.all(np.isfinite(np.asarray(res.Q)))
    np.testing.assert_allclose(float(res.supply["A"]), sum(DEMAND.values()), rtol=1e-10)


def test_parallel_pipes_smoke():
    net = PipeNetwork()
    net.add_node("in", pressure=3e5)
    net.add_node("out", demand=6e-3)
    net.add_branch("a", "in", "out", Pipe(PipeParams(L=80.0, D=0.0525, **WATER)))
    net.add_branch("b", "in", "out", Pipe(PipeParams(L=150.0, D=0.0409, **WATER)))
    res = net.solve()
    np.testing.assert_allclose(float(res.Q.sum()), 6e-3, rtol=1e-10)


def test_grad_finite():
    g = jax.grad(lambda d: _two_loop(dict(DEMAND, C=d)).solve().branch_flows["BC"])(0.025)
    assert np.isfinite(float(g))


def test_resistance_branch_gradient_through_zero_flow():
    assert np.isfinite(float(jax.grad(ResistanceBranch(r=1e8, n=2.0).signed_dp)(0.0)))


def test_errors():
    net = PipeNetwork()
    net.add_node("a", demand=0.1)
    net.add_node("b", demand=-0.1)
    net.add_branch("ab", "a", "b", ResistanceBranch(1e8))
    with pytest.raises(ValueError, match="fixed-pressure"):
        net.solve()
    with pytest.raises(ValueError, match="demand"):
        net.add_node("c", pressure=1e5, demand=0.1)
    with pytest.raises(ValueError, match="unknown node"):
        net.add_branch("ax", "a", "x", ResistanceBranch(1e8))
    with pytest.raises(ValueError, match="already exists"):
        net.add_node("a")
    net.add_node("r", pressure=1e5)
    net.add_branch("ra", "r", "a", ResistanceBranch(1e8))
    with pytest.raises(ValueError, match="method"):
        net.solve(method="gauss")
    net.add_node("island", demand=0.0)
    with pytest.raises(ValueError, match="not connected"):
        net.solve()
