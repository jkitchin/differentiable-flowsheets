"""Tests for difflow.pipe_network (looped/branched liquid pipe networks)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import Pipe, PipeNetwork, PipeParams, ResistanceBranch

jax.config.update("jax_enable_x64", True)

METHODS = ("newton", "hardy_cross")
WATER = dict(rho=998.2, mu=1.002e-3)


# ---------------------------------------------------------------------------
# Two-branch parallel header
# ---------------------------------------------------------------------------


def _laminar_header(Q_total=1e-3):
    """Oil header feeding one user through two parallel laminar pipes."""
    oil = dict(rho=900.0, mu=0.5)
    net = PipeNetwork()
    net.add_node("header", pressure=2e5)
    net.add_node("user", demand=Q_total)
    net.add_branch("a", "header", "user", Pipe(PipeParams(L=100.0, D=0.05, **oil)))
    net.add_branch("b", "header", "user", Pipe(PipeParams(L=60.0, D=0.04, **oil)))
    return net, oil


@pytest.mark.parametrize("method", METHODS)
def test_parallel_laminar_header_matches_hagen_poiseuille(method):
    """Laminar branches are linear: Q_k = G_k dP, G_k = pi D^4 / (128 mu L)."""
    Q_total = 1e-3
    net, oil = _laminar_header(Q_total)
    res = net.solve(method=method)
    G = np.array([math.pi * 0.05**4 / (128 * 0.5 * 100.0), math.pi * 0.04**4 / (128 * 0.5 * 60.0)])
    dP = Q_total / G.sum()
    np.testing.assert_allclose(np.asarray(res.Q), G * dP, rtol=1e-9)
    np.testing.assert_allclose(float(res.node_pressures["user"]), 2e5 - dP, rtol=1e-12)
    np.testing.assert_allclose(float(res.supply["header"]), Q_total, rtol=1e-12)
    assert bool(res.converged)
    # really laminar
    for k, D in (("a", 0.05), ("b", 0.04)):
        Re = 4 * oil["rho"] * float(res.branch_flows[k]) / (math.pi * D * oil["mu"])
        assert Re < 2100


@pytest.mark.parametrize("method", METHODS)
def test_parallel_turbulent_header_matches_flow_at_dp(method):
    """Two fixed-pressure headers joined by two pipes: each flow is Pipe.flow_at_dp."""
    pa = Pipe(PipeParams(L=80.0, D=0.0525, **WATER))
    pb = Pipe(PipeParams(L=150.0, D=0.0409, fittings={"globe_valve": 1}, **WATER))
    net = PipeNetwork()
    net.add_node("hi", pressure=3.5e5)
    net.add_node("lo", pressure=2.0e5)
    net.add_branch("a", "hi", "lo", pa)
    net.add_branch("b", "hi", "lo", pb)
    res = net.solve(method=method)
    np.testing.assert_allclose(float(res.branch_flows["a"]), float(pa.flow_at_dp(1.5e5)), rtol=1e-8)
    np.testing.assert_allclose(float(res.branch_flows["b"]), float(pb.flow_at_dp(1.5e5)), rtol=1e-8)
    np.testing.assert_allclose(float(res.supply["hi"]), -float(res.supply["lo"]), rtol=1e-12)


def test_parallel_split_with_demand_has_equal_dp():
    pa = Pipe(PipeParams(L=80.0, D=0.0525, **WATER))
    pb = Pipe(PipeParams(L=150.0, D=0.0409, fittings={"globe_valve": 1}, **WATER))
    net = PipeNetwork()
    net.add_node("in", pressure=3e5)
    net.add_node("out", demand=6e-3)
    net.add_branch("a", "in", "out", pa)
    net.add_branch("b", "in", "out", pb)
    res = net.solve()
    np.testing.assert_allclose(float(res.branch_dP["a"]), float(res.branch_dP["b"]), rtol=1e-9)
    np.testing.assert_allclose(float(res.Q.sum()), 6e-3, rtol=1e-12)


# ---------------------------------------------------------------------------
# Two-loop network with a known answer
# ---------------------------------------------------------------------------

# Manufactured solution: node pressures and branch flows are chosen first
# (continuity holds), and each branch resistance is set to r = dP/(Q|Q|^(n-1)).
# The solver starts from the linearized network, far from this answer, and
# must recover it exactly.  Branch "DB" runs against its flow (Q < 0).
P_TRUE = {"A": 5.0e5, "B": 4.0e5, "C": 3.2e5, "D": 3.5e5, "E": 3.0e5}
Q_TRUE = {"AB": 0.06, "AD": 0.04, "DB": -0.01, "BC": 0.03, "CE": 0.005, "DE": 0.02}
ENDS = {"AB": ("A", "B"), "AD": ("A", "D"), "DB": ("D", "B"), "BC": ("B", "C"),
        "CE": ("C", "E"), "DE": ("D", "E")}
DEMAND = {"B": 0.02, "C": 0.025, "D": 0.03, "E": 0.025}


def _two_loop(n=2.0, demand=None):
    demand = DEMAND if demand is None else demand
    net = PipeNetwork()
    net.add_node("A", pressure=P_TRUE["A"])
    for node, d in demand.items():
        net.add_node(node, demand=d)
    for name, (a, b) in ENDS.items():
        dP = P_TRUE[a] - P_TRUE[b]
        Q = Q_TRUE[name]
        r = dP / (Q * abs(Q) ** (n - 1.0))
        net.add_branch(name, a, b, ResistanceBranch(r=r, n=n))
    return net


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [2.0, 1.852])
def test_two_loop_network_recovers_known_answer(method, n):
    net = _two_loop(n)
    res = net.solve(method=method, iterations=300)
    assert bool(res.converged)
    for k, q in Q_TRUE.items():
        np.testing.assert_allclose(float(res.branch_flows[k]), q, rtol=1e-8, atol=1e-12)
    for node, p in P_TRUE.items():
        np.testing.assert_allclose(float(res.node_pressures[node]), p, rtol=1e-9)
    assert float(res.branch_flows["DB"]) < 0.0  # flow reversal against the branch orientation


@pytest.mark.parametrize("method", METHODS)
def test_two_loop_mass_balance_and_zero_loop_head_loss(method):
    net = _two_loop()
    res = net.solve(method=method, iterations=300)
    # continuity at every free node
    for node, d in DEMAND.items():
        inflow = sum(float(res.branch_flows[k]) for k, (a, b) in ENDS.items() if b == node)
        outflow = sum(float(res.branch_flows[k]) for k, (a, b) in ENDS.items() if a == node)
        assert abs(inflow - outflow - d) < 1e-12
    np.testing.assert_allclose(float(res.supply["A"]), sum(DEMAND.values()), rtol=1e-12)
    # zero pressure (head) loss around every independent loop
    loops = net.loops()
    assert len(loops) == 2  # 6 branches, 5 nodes: 6 - 5 + 1
    for loop in loops:
        total = sum(s * float(res.branch_dP[k]) for k, s in loop.items())
        scale = sum(abs(float(res.branch_dP[k])) for k in loop)
        assert abs(total) < 1e-9 * scale


def test_three_reservoir_pseudo_loop():
    """Classic three-reservoir junction: two fixed-pressure loops through a datum."""
    P = {"R1": 5e5, "R2": 2.5e5, "R3": 1.5e5, "J": 3e5}
    Q = {"1J": 0.05, "J2": 0.03, "J3": 0.02}
    ends = {"1J": ("R1", "J"), "J2": ("J", "R2"), "J3": ("J", "R3")}
    for method in METHODS:
        net = PipeNetwork()
        for r in ("R1", "R2", "R3"):
            net.add_node(r, pressure=P[r])
        net.add_node("J")
        for k, (a, b) in ends.items():
            net.add_branch(k, a, b, ResistanceBranch(r=(P[a] - P[b]) / Q[k] ** 2))
        res = net.solve(method=method, iterations=300)
        for k, q in Q.items():
            np.testing.assert_allclose(float(res.branch_flows[k]), q, rtol=1e-8)
        np.testing.assert_allclose(float(res.node_pressures["J"]), 3e5, rtol=1e-10)
        np.testing.assert_allclose(float(res.supply["R1"]), 0.05, rtol=1e-8)
        np.testing.assert_allclose(float(res.supply["R2"]), -0.03, rtol=1e-8)


def _pipe_loop_net(D_ab=0.1023, demand_c=0.012):
    """A two-loop Darcy pipe network with an uphill leg (pipes from difflow.fluids)."""
    net = PipeNetwork()
    net.add_node("S", pressure=4e5)
    net.add_node("A", demand=0.004)
    net.add_node("B", demand=0.006)
    net.add_node("C", demand=demand_c)
    net.add_node("D", demand=0.003)
    pipe = lambda L, D, **kw: Pipe(PipeParams(L=L, D=D, **WATER, **kw))
    net.add_branch("SA", "S", "A", pipe(300.0, 0.1541))
    net.add_branch("AB", "A", "B", pipe(400.0, D_ab))
    net.add_branch("AC", "A", "C", pipe(500.0, 0.0779, fittings={"elbow_90_standard": 4}))
    net.add_branch("BC", "B", "C", pipe(250.0, 0.0779, dz=5.0))
    net.add_branch("BD", "B", "D", pipe(350.0, 0.0525))
    net.add_branch("CD", "C", "D", pipe(300.0, 0.0525, dz=-5.0))
    return net


def test_pipe_network_methods_agree_and_balance():
    net = _pipe_loop_net()
    r_n = net.solve(method="newton")
    r_h = net.solve(method="hardy_cross", iterations=400)
    assert bool(r_n.converged) and bool(r_h.converged)
    np.testing.assert_allclose(np.asarray(r_n.Q), np.asarray(r_h.Q), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(np.asarray(r_n.P), np.asarray(r_h.P), rtol=1e-9)
    assert float(r_n.mass_imbalance) < 1e-14
    for loop in net.loops():
        total = sum(s * float(r_n.branch_dP[k]) for k, s in loop.items())
        assert abs(total) < 1e-6  # Pa
    # the static heads around the BC/CD/BD loop cancel (+5 m then -5 m)
    # and are part of branch_dP
    rho_g = WATER["rho"] * 9.80665
    h = net.branches[3][3].hydraulics_at_flow(abs(float(r_n.branch_flows["BC"])))
    np.testing.assert_allclose(float(h["dP_static"]), 5.0 * rho_g, rtol=1e-12)


def test_network_jit():
    def q_ab(d):
        return _pipe_loop_net(demand_c=d).solve().branch_flows["AB"]
    np.testing.assert_allclose(float(jax.jit(q_ab)(0.012)), float(q_ab(0.012)), rtol=1e-10)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradient_wrt_demand_and_diameter_newton():
    @jax.jit
    def q_ab_demand(d):
        return _pipe_loop_net(demand_c=d).solve().branch_flows["AB"]

    @jax.jit
    def p_c_diam(D):
        return _pipe_loop_net(D_ab=D).solve().node_pressures["C"] / 1e5

    check_grads(q_ab_demand, (0.012,), order=1, modes=["rev"], eps=1e-6, atol=1e-6, rtol=1e-4)
    check_grads(p_c_diam, (0.1023,), order=1, modes=["rev"], eps=1e-6, atol=1e-6, rtol=1e-4)
    p_c_diam = jax.jit(jax.value_and_grad(lambda D: _pipe_loop_net(D_ab=D).solve().node_pressures["C"] / 1e5))
    g = float(p_c_diam(0.1023)[1])
    h = 1e-6
    fd = (float(p_c_diam(0.1023 + h)[0]) - float(p_c_diam(0.1023 - h)[0])) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_gradient_hardy_cross_matches_newton():
    def total_supply(d, method):
        demand = dict(DEMAND, C=d)
        return _two_loop(demand=demand).solve(method=method, iterations=300).branch_flows["BC"]

    g_n = jax.grad(lambda d: total_supply(d, "newton"))(0.025)
    g_h = jax.grad(lambda d: total_supply(d, "hardy_cross"))(0.025)
    np.testing.assert_allclose(float(g_n), float(g_h), rtol=1e-6)
    check_grads(lambda d: total_supply(d, "hardy_cross"), (0.025,), order=1, modes=["rev"],
                eps=1e-6, atol=1e-6, rtol=1e-4)


def test_resistance_branch_gradient_through_zero_flow():
    b = ResistanceBranch(r=1e8, n=2.0)
    check_grads(lambda q: b.signed_dp(q) / 1e4, (0.003,), order=1, modes=["rev", "fwd"])
    assert np.isfinite(float(jax.grad(b.signed_dp)(0.0)))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


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
