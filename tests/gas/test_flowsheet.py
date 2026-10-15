"""Build, solve and differentiate gas network flowsheets."""

import jax
import jax.numpy as jnp
import pytest

import difflow_gas as dg
from tests.gas.test_network import mixed_network, triangle

P_SLACK = 50.0e5
RATIOS = {"cs_1": 1.2}


@pytest.fixture(scope="module")
def mixed_solved():
    net = mixed_network()
    fs, dec = dg.build_network_flowsheet(
        net, root="s1", p_slack_pa=P_SLACK,
        ratios=RATIOS, cv_drops_pa={"cv_1": 2.0e5},
    )
    streams = fs.solve(tol=1e-8, max_iter=300)
    return net, fs, dec, streams


def test_mixed_network_solves(mixed_solved):
    net, fs, dec, streams = mixed_solved
    assert dg.residual_report(streams, net, dec, cv_drops_bar={"cv_1": 2.0}).ok
    assert jnp.isfinite(dg.total_compressor_power_w(streams, dec, net.gas_temp_k))
    p = dg.verify.node_pressures_bar(streams, dec)
    q = dg.verify.arc_flows_kg_s(streams, dec)
    net.compressor_limits = {"cs_1": dg.CompressorLimits(pressure_in_min_bar=31.0)}
    rep = dg.verify.compressor_report(p, q, RATIOS, net)
    assert "inlet_margin_bar" in rep["cs_1"]


def test_negative_chord_flow_converges_with_default_solve():
    """Regression (difflow issue #164): a negative tear fixed point."""
    net = dg.GasNetwork(
        arcs={"p01": ("n0", "n1", "pipe"), "p12": ("n1", "n2", "pipe"),
              "p02": ("n2", "n0", "pipe")},
        beta={"p01": 1e8, "p12": 2e8, "p02": 4e8},
        supply_kg_s={"n0": 30.0, "n1": -10.0, "n2": -20.0},
    )
    fs, dec = dg.build_network_flowsheet(net, root="n0", p_slack_pa=P_SLACK)
    streams = fs.solve(tol=1e-8, max_iter=200)
    assert dg.residual_report(streams, net, dec).ok


def test_slack_pressure_gradient_is_finite():
    net = triangle()
    fs, dec = dg.build_network_flowsheet(net, root="n0", p_slack_pa=P_SLACK)
    obj = fs.make_objective_fn(lambda s: s["node_n2"]["P"])
    key = f"{dg.src_unit_name('n0')}.P_set"
    g = jax.grad(obj)({key: P_SLACK})
    assert jnp.isfinite(g[key])


def test_builder_rejects_mismatched_decomposition():
    net = triangle()
    dec = dg.decompose(net, root="n1")
    with pytest.raises(ValueError, match="rooted at"):
        dg.build_network_flowsheet(net, root="n0", p_slack_pa=P_SLACK, dec=dec)


def test_plugin_registration():
    from difflow.plugins import OperationRegistry

    reg = OperationRegistry()
    dg.register(reg)
    for name in ("GasPipe", "PressureDrivenPipe", "CompressorBoost",
                 "ControlValveDrop", "AffineFlow", "SourceHead"):
        assert reg.get(name) is not None
