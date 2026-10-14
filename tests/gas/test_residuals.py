"""Smoke and API tests for difflow_gas.residuals."""

import jax
import jax.numpy as jnp
import pytest

from difflow_gas import verify
from difflow_gas.residuals import (
    gas_state_layout,
    network_residuals,
    residual_names,
)
from tests.gas.test_network import mixed_network

RATIOS = {"cs_1": 1.2}
CV_BAR = {"cv_1": 2.0}


@pytest.fixture(scope="module")
def state():
    net = mixed_network()
    p_bar = {n: 50.0 - i for i, n in enumerate(net.nodes)}
    q = {a: 5.0 for a in net.arcs}
    return net, p_bar, q


def test_residuals_and_jacobian_are_finite(state):
    net, p_bar, q = state
    layout = gas_state_layout(net)
    x = layout.pack(p_bar, q, net.supply_kg_s)
    r = network_residuals(x, net, layout, ratios=RATIOS, cv_drops_bar=CV_BAR)
    assert r.shape == (len(residual_names(net, layout)),)
    assert bool(jnp.all(jnp.isfinite(r)))
    a = jax.jacobian(network_residuals)(x, net, layout)
    assert a.shape == (r.shape[0], layout.size)
    assert bool(jnp.all(jnp.isfinite(a)))


def test_verify_report_keys_match_the_network(state):
    net, p_bar, q = state
    report = verify.residuals_from_values(p_bar, q, net, cv_drops_bar=CV_BAR)
    assert set(report.node_imbalance) == set(net.nodes)


def test_pack_unpack_round_trip(state):
    net, p_bar, q = state
    layout = gas_state_layout(net, efficiency_arcs=["pipe_1"])
    x = layout.pack(p_bar, q, net.supply_kg_s, {"eta_pipe_1": 1.5})
    assert len(layout.names) == layout.size
    p2, q2, _, extra = layout.unpack(x)
    assert float(p2["a"]) == pytest.approx(p_bar["a"])
    assert float(q2["pipe_1"]) == pytest.approx(5.0)
    assert float(extra["eta_pipe_1"]) == pytest.approx(1.5)


def test_layout_validation(state):
    net, _, _ = state
    with pytest.raises(KeyError, match="not a state variable"):
        gas_state_layout(net).index("q_nope")
    with pytest.raises(ValueError, match="only pipes and resistors"):
        gas_state_layout(net, efficiency_arcs=["cs_1"])
    with pytest.raises(ValueError, match="not a compressor"):
        gas_state_layout(net, ratio_arcs=["pipe_1"])
