"""Smoke and mechanics tests for difflow_power.residuals."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.residuals import (
    power_flow_residuals,
    power_state_layout,
    residual_names,
)


def _state(network):
    rng = np.random.default_rng(0)
    return (
        0.95 + 0.1 * rng.random(network.n_bus),
        0.3 * (rng.random(network.n_bus) - 0.5),
        rng.random(network.n_gen),
        0.4 * (rng.random(network.n_gen) - 0.5),
    )


def test_residuals_smoke():
    network = dp.cases.case9()
    layout = power_state_layout(network)
    r = power_flow_residuals(layout.pack(*_state(network)), network, layout)
    assert r.shape == (layout.n_residual,)
    assert len(residual_names(network, layout)) == layout.n_residual
    assert bool(jnp.all(jnp.isfinite(r)))


def test_residual_grad_wrt_line_parameter_is_finite():
    network = dp.cases.case9()
    layout = power_state_layout(network)
    x = layout.pack(*_state(network))
    reactances = network.branch_param_arrays()["x"]
    grad = jax.grad(
        lambda xs: jnp.sum(
            power_flow_residuals(x, network, layout, branch_params={"x": xs}) ** 2
        )
    )(reactances)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_layout_pack_unpack_round_trip():
    network = dp.cases.case14()
    layout = power_state_layout(
        network, demand_buses=["2", "3"], tap_branches=["br8"]
    )
    x = layout.pack(
        *_state(network), pd=[0.1, 0.2], qd=[0.03, 0.04],
        extra={"tap_br8": 0.97},
    )
    values = layout.unpack(x)
    assert values["tap_br8"] == pytest.approx(0.97)
    assert values["pd_3"] == pytest.approx(0.2)


def test_layout_embed_maps_by_name():
    network = dp.cases.case9()
    plain = power_state_layout(network)
    extended = power_state_layout(network, demand_buses=["5"])
    x = plain.pack(*_state(network))
    y = extended.embed(x, plain)
    for name in plain.names:
        assert float(y[extended.index(name)]) == float(x[plain.index(name)])
    assert math.isnan(float(y[extended.index("pd_5")]))


def test_layout_validation():
    layout = power_state_layout(dp.cases.case3())
    with pytest.raises(KeyError, match="not a state variable"):
        layout.index("vm_nowhere")
    with pytest.raises(ValueError, match="line, not a transformer"):
        power_state_layout(dp.cases.case9(), tap_branches=["br2"])


def test_branch_flows_smoke():
    network = dp.cases.case9()
    layout = power_state_layout(network)
    s_from, s_to = dp.branch_flows(
        layout.pack(*_state(network)), network, layout
    )
    assert s_from.shape == (network.n_branch,)
    assert bool(jnp.all(jnp.isfinite(s_to)))
