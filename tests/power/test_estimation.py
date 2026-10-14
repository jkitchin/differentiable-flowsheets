"""Smoke and validation tests for difflow_power.estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.estimation import (
    estimate_state,
    estimated_values,
    measurement_sigma,
    perturb,
)
from difflow_power.residuals import power_state_layout


@pytest.fixture(scope="module")
def setup():
    """A true state of case9 packed into a demand-carrying layout."""
    net = dp.cases.case9()
    reference = dp.solve_power_flow(net)
    layout = power_state_layout(net, demand_buses=net.bus_ids)
    x_true = layout.embed(reference.x, power_state_layout(net), fill=0.0)
    pd, qd = net.load_arrays_pu()
    x_true = x_true.at[layout.slice_pd].set(pd).at[layout.slice_qd].set(qd)
    return net, layout, x_true


def test_estimate_state_smoke(setup):
    net, layout, x_true = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    y = perturb(x_true, sigma, jax.random.PRNGKey(0))
    result = estimate_state(net, y, sigma, layout)
    assert result.converged
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(result.x))))
    assert list(result.names) == layout.names
    values = estimated_values(result, layout, net)
    assert set(values) == {"vm", "va_degrees", "pg_mw", "qg_mvar"}


def test_unknown_variable_names_are_rejected(setup):
    _, layout, _ = setup
    with pytest.raises(KeyError):
        measurement_sigma(layout, unmeasured=["va_nowhere"])
    with pytest.raises(KeyError):
        measurement_sigma(layout, overrides={"nonsense": 1.0})


def test_perturb_leaves_unmeasured_entries_alone(setup):
    _, layout, x_true = setup
    sigma = measurement_sigma(layout)          # angles are inf
    y = perturb(x_true, sigma, jax.random.PRNGKey(3))
    np.testing.assert_allclose(
        y[layout.slice_va], x_true[layout.slice_va], atol=0.0
    )


def test_bad_data_needs_a_layout(setup):
    _, layout, x_true = setup
    sigma = measurement_sigma(layout)
    with pytest.raises(ValueError, match="layout is required"):
        perturb(x_true, sigma, jax.random.PRNGKey(0), bad_data={"vm_5": 3.0})
