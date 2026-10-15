"""Smoke and API tests for gas network data reconciliation."""

import jax
import numpy as np
import pytest

import difflow_gas as dg
from difflow_gas import verify
from difflow_gas.reconcile import (
    measurement_sigma,
    monitor_network,
    network_residual_fn,
    perturb,
    reconcile_network,
    reconcile_network_multi,
    reconciled_values,
)
from difflow_gas.residuals import gas_state_layout
from difflow.reconciliation import ReconciliationStructureError
from tests.gas.test_network import triangle


@pytest.fixture(scope="module")
def tri():
    net = triangle()
    fs, dec = dg.build_network_flowsheet(net, root="n0", p_slack_pa=50.0e5)
    streams = fs.solve(tol=1e-10, max_iter=300)
    p_bar = verify.node_pressures_bar(streams, dec)
    q = verify.arc_flows_kg_s(streams, dec)
    return net, p_bar, q


def test_reconcile_network_smoke(tri):
    net, p_bar, q = tri
    layout = gas_state_layout(net)
    sigma = measurement_sigma(layout)
    y = perturb(layout.pack(p_bar, q, net.supply_kg_s), sigma,
                jax.random.PRNGKey(0))
    res = reconcile_network(net, y, sigma, layout)
    assert res.converged
    assert np.all(np.isfinite(np.asarray(res.x)))
    p2, q2, _ = reconciled_values(res, layout)
    assert set(p2) == set(net.nodes) and set(q2) == set(net.arcs)


def test_monitor_and_multi_period_smoke(tri):
    net, p_bar, q = tri
    layout = gas_state_layout(net, efficiency_arcs=["p01"])
    sigma = measurement_sigma(layout)
    x = layout.pack(p_bar, q, net.supply_kg_s, {"eta_p01": 1.0})
    days = [perturb(x, sigma, jax.random.PRNGKey(k)) for k in range(1)]
    mon = monitor_network(net, days, sigma, layout)
    assert len(mon) == 1
    pooled = reconcile_network_multi(net, days, sigma, layout, shared=["eta_p01"])
    assert np.isfinite(pooled.shared["eta_p01"])


def test_unobservable_problem_raises(tri):
    net, p_bar, q = tri
    layout = gas_state_layout(net)
    sigma = measurement_sigma(layout, unmeasured=layout.names)
    with pytest.raises(ReconciliationStructureError):
        reconcile_network(net, layout.pack(p_bar, q, net.supply_kg_s),
                          sigma, layout)


def test_residual_fn_efficiencies_survive_repeated_calls(tri):
    """Regression: the closure must not consume its own argument."""
    net, p_bar, q = tri
    layout = gas_state_layout(net)
    x = layout.pack(p_bar, q, net.supply_kg_s)
    fn = network_residual_fn(net, layout, efficiencies={"p01": 1.3})
    assert np.allclose(fn(x), fn(x))
    assert not np.allclose(fn(x), network_residual_fn(net, layout)(x))
