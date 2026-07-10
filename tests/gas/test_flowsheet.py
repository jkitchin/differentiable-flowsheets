"""End-to-end tests: build, solve, verify and differentiate flowsheets."""

import math

import jax
import jax.numpy as jnp
import pytest

import difflow_gas as dg
from tests.gas.test_network import mixed_network, triangle

P_SLACK = 50.0e5


def triangle_reversed() -> dg.GasNetwork:
    """Triangle with the chord arc reversed: its flow is negative."""
    return dg.GasNetwork(
        arcs={
            "p01": ("n0", "n1", "pipe"),
            "p12": ("n1", "n2", "pipe"),
            "p02": ("n2", "n0", "pipe"),  # reversed reference direction
        },
        beta={"p01": 1e8, "p12": 2e8, "p02": 4e8},
        supply_kg_s={"n0": 30.0, "n1": -10.0, "n2": -20.0},
    )


def triangle_closed_form():
    """Exact solution of the triangle loop split.

    With x = q(p01): x^2 + 2 (x - 10)^2 = 4 (30 - x)^2, i.e.
    x^2 - 200 x + 3400 = 0, taking the root with all flows forward.
    """
    x = 100.0 - math.sqrt(100.0**2 - 3400.0)
    return {"p01": x, "p12": x - 10.0, "p02": 30.0 - x}


# ---------------------------------------------------------------------------
# triangle: solves match the closed form
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def triangle_solved():
    net = triangle()
    fs, dec = dg.build_network_flowsheet(net, root="n0", p_slack_pa=P_SLACK)
    streams = fs.solve(tol=1e-8, max_iter=200)
    return net, fs, dec, streams


def test_anderson_solve_matches_closed_form(triangle_solved):
    net, fs, dec, streams = triangle_solved
    q = dg.verify.arc_flows_kg_s(streams, dec)
    for aid, ref in triangle_closed_form().items():
        assert q[aid] == pytest.approx(ref, abs=1e-7)


def test_residual_report_ok(triangle_solved):
    net, fs, dec, streams = triangle_solved
    rep = dg.residual_report(streams, net, dec)
    assert rep.ok
    assert rep.max_node_imbalance_kg_s < 1e-9
    assert rep.max_resistance_residual_bar2 < 1e-9


def test_damped_solve_matches_anderson(triangle_solved):
    net, fs, dec, streams = triangle_solved
    s2, stats = fs.solve_differentiable(return_stats=True)
    assert int(stats["num_steps"]) < 500
    p_a = dg.verify.node_pressures_bar(streams, dec)
    p_d = dg.verify.node_pressures_bar(s2, dec)
    for node in p_a:
        assert p_d[node] == pytest.approx(p_a[node], abs=1e-6)


def test_negative_chord_flow_converges_with_default_solve():
    """The reversed-arc triangle has a negative tear fixed point; the
    GasNetworkFlowsheet solve() default (clip_negative_flows=False)
    must reach it (difflow issue #164)."""
    net = triangle_reversed()
    fs, dec = dg.build_network_flowsheet(net, root="n0", p_slack_pa=P_SLACK)
    streams = fs.solve(tol=1e-8, max_iter=200)
    q = dg.verify.arc_flows_kg_s(streams, dec)
    ref = triangle_closed_form()
    assert q["p02"] == pytest.approx(-ref["p02"], abs=1e-7)  # signed
    assert dg.residual_report(streams, net, dec).ok


def test_solution_independent_of_root():
    net = triangle()
    fs0, dec0 = dg.build_network_flowsheet(net, "n0", P_SLACK)
    s0 = fs0.solve(tol=1e-9, max_iter=300)
    p0 = dg.verify.node_pressures_bar(s0, dec0)

    # root the tree at n2 instead, pinning n2 at the pressure the
    # n0-rooted solve computed there; the states must coincide
    fs2, dec2 = dg.build_network_flowsheet(
        net, "n2", p_slack_pa=p0["n2"] * 1e5
    )
    s2 = fs2.solve(tol=1e-9, max_iter=300)
    p2 = dg.verify.node_pressures_bar(s2, dec2)
    for node in p0:
        assert p2[node] == pytest.approx(p0[node], abs=1e-6)


# ---------------------------------------------------------------------------
# mixed network: every arc kind at once
# ---------------------------------------------------------------------------


RATIOS = {"cs_1": 1.2}
CV_DROPS_PA = {"cv_1": 2.0e5}


@pytest.fixture(scope="module")
def mixed_solved():
    net = mixed_network()
    fs, dec = dg.build_network_flowsheet(
        net, root="s1", p_slack_pa=P_SLACK,
        ratios=RATIOS, cv_drops_pa=CV_DROPS_PA,
    )
    streams = fs.solve(tol=1e-8, max_iter=300)
    return net, fs, dec, streams


def test_mixed_network_residuals(mixed_solved):
    net, fs, dec, streams = mixed_solved
    rep = dg.residual_report(
        streams, net, dec, cv_drops_bar={"cv_1": 2.0}
    )
    assert rep.ok, rep
    # forgetting the control-valve drops must show up as a residual
    rep_wrong = dg.residual_report(streams, net, dec)
    assert rep_wrong.max_control_valve_residual_bar == pytest.approx(2.0)


def test_mixed_network_element_relations(mixed_solved):
    net, fs, dec, streams = mixed_solved
    p = dg.verify.node_pressures_bar(streams, dec)
    q = dg.verify.arc_flows_kg_s(streams, dec)
    # valve and short pipe: pressure equality
    assert p["b"] == pytest.approx(p["a"], abs=1e-9)
    assert p["e"] == pytest.approx(p["d"], abs=1e-9)
    # control valve: parametric drop (2 bar)
    assert p["d"] == pytest.approx(p["c"] - 2.0, abs=1e-9)
    # compressor: p_a = ratio * p_s2 (station feeds the network)
    assert p["a"] == pytest.approx(1.2 * p["s2"], rel=1e-9)
    # boundary flows delivered exactly
    assert q["cs_1"] == pytest.approx(10.0, abs=1e-9)
    assert q["pipe_1"] == pytest.approx(20.0, abs=1e-9)
    assert q["short_1"] == pytest.approx(15.0, abs=1e-9)


def test_total_compressor_power(mixed_solved):
    net, fs, dec, streams = mixed_solved
    W = float(dg.total_compressor_power_w(streams, dec, net.gas_temp_k))
    ref = dg.compressor_power(10.0, 1.2, t_in_k=net.gas_temp_k)
    assert W == pytest.approx(ref, rel=1e-6)


def test_compressor_report(mixed_solved):
    net, fs, dec, streams = mixed_solved
    net.compressor_limits = {
        "cs_1": dg.CompressorLimits(pressure_in_min_bar=31.0,
                                    pressure_out_max_bar=71.0)
    }
    p = dg.verify.node_pressures_bar(streams, dec)
    q = dg.verify.arc_flows_kg_s(streams, dec)
    rep = dg.verify.compressor_report(p, q, RATIOS, net)
    assert rep["cs_1"]["q_kg_s"] == pytest.approx(10.0, abs=1e-6)
    assert rep["cs_1"]["inlet_margin_bar"] == pytest.approx(
        p["s2"] - 31.0
    )


# ---------------------------------------------------------------------------
# differentiability and jit
# ---------------------------------------------------------------------------


def test_power_gradient_matches_finite_differences(mixed_solved):
    net, fs, dec, streams = mixed_solved

    obj = fs.make_objective_fn(
        lambda s: dg.total_compressor_power_w(s, dec, net.gas_temp_k)
    )
    key = f"{dg.cs_unit_name('cs_1')}.ratio"
    params = {key: 1.2}
    g = jax.grad(obj)(params)

    h = 1e-6
    up, dn = {key: 1.2 + h}, {key: 1.2 - h}
    fd = (obj(up) - obj(dn)) / (2 * h)
    assert float(g[key]) == pytest.approx(float(fd), rel=1e-5)
    assert float(g[key]) > 0


def test_slack_pressure_gradient(triangle_solved):
    net, fs, dec, streams = triangle_solved
    obj = fs.make_objective_fn(lambda s: s["node_n2"]["P"])
    key = f"{dg.src_unit_name('n0')}.P_set"
    g = jax.grad(obj)({key: P_SLACK})
    # pressures shift essentially one-to-one with the slack head
    assert float(g[key]) == pytest.approx(1.0, abs=0.05)


def test_jit_compiled_solve(triangle_solved):
    net, fs, dec, streams = triangle_solved

    @jax.jit
    def solve_packed(p_slack):
        prm = {f"{dg.src_unit_name('n0')}.P_set": p_slack}
        s = fs._apply_params(prm).solve_differentiable()
        return s["node_n2"]["P"]

    p_eager = float(streams["node_n2"]["P"])
    p_jit = float(solve_packed(jnp.asarray(P_SLACK)))
    assert p_jit == pytest.approx(p_eager, rel=1e-9)


# ---------------------------------------------------------------------------
# builder validation and plugin registration
# ---------------------------------------------------------------------------


def test_builder_rejects_mismatched_decomposition():
    net = triangle()
    dec = dg.decompose(net, root="n1")
    with pytest.raises(ValueError, match="rooted at"):
        dg.build_network_flowsheet(net, root="n0", p_slack_pa=P_SLACK,
                                   dec=dec)


def test_plugin_registration():
    from difflow.plugins import OperationRegistry

    import difflow_gas

    reg = OperationRegistry()
    difflow_gas.register(reg)
    for name in ("GasPipe", "PressureDrivenPipe", "CompressorBoost",
                 "ControlValveDrop", "AffineFlow", "SourceHead"):
        assert reg.get(name) is not None
