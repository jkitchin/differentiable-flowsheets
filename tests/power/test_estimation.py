"""Tests for difflow_power.estimation.

State estimation and data reconciliation are the same computation, so
these tests are about the wiring: that the residual closure, the
default meter accuracies and the layout's names all line up with what
:mod:`difflow.reconciliation` expects, and that the estimate lands on
a state that actually satisfies the network equations --- which is the
one property a raw measurement vector does not have.
"""

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
    monitor_network,
    perturb,
)
from difflow_power.residuals import power_flow_residuals, power_state_layout


@pytest.fixture(scope="module")
def setup():
    """A true state of case9 packed into a demand-carrying layout."""
    net = dp.cases.case9()
    reference = dp.solve_power_flow(net)
    layout = power_state_layout(net, demand_buses=net.bus_ids)
    x_true = layout.embed(reference.x, power_state_layout(net), fill=0.0)
    pd, qd = net.load_arrays_pu()
    x_true = x_true.at[layout.slice_pd].set(pd).at[layout.slice_qd].set(qd)
    return net, layout, x_true, reference


def test_the_true_state_satisfies_the_equations(setup):
    net, layout, x_true, _ = setup
    residuals = power_flow_residuals(x_true, net, layout)
    assert float(jnp.max(jnp.abs(residuals))) < 1e-10


def test_estimate_lands_on_a_state_that_satisfies_the_equations(setup):
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    y = perturb(x_true, sigma, jax.random.PRNGKey(0))

    # The raw measurements do NOT satisfy the equations...
    assert float(
        jnp.max(jnp.abs(power_flow_residuals(y, net, layout)))
    ) > 1e-3
    result = estimate_state(net, y, sigma, layout)
    assert result.converged
    # ...but the estimate does. That is what reconciliation buys.
    assert float(
        jnp.max(jnp.abs(power_flow_residuals(jnp.asarray(result.x), net, layout)))
    ) < 1e-8


def test_estimate_is_closer_to_the_truth_than_the_raw_readings(setup):
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    total_raw = total_estimated = 0.0
    for seed in range(5):
        y = perturb(x_true, sigma, jax.random.PRNGKey(seed))
        result = estimate_state(net, y, sigma, layout)
        assert result.converged
        measured = jnp.isfinite(sigma)
        total_raw += float(
            jnp.sum(jnp.where(measured, (y - x_true) ** 2, 0.0))
        )
        total_estimated += float(
            jnp.sum(
                jnp.where(measured, (jnp.asarray(result.x) - x_true) ** 2, 0.0)
            )
        )
    assert total_estimated < total_raw


def test_names_come_from_the_layout(setup):
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    y = perturb(x_true, sigma, jax.random.PRNGKey(1))
    result = estimate_state(net, y, sigma, layout)
    assert list(result.names) == layout.names


def test_default_sigma_leaves_angles_unmeasured(setup):
    net, layout, _, _ = setup
    sigma = measurement_sigma(layout)
    for i, name in enumerate(layout.names):
        if name.startswith("va_"):
            assert not np.isfinite(float(sigma[i]))
        elif name.startswith("vm_"):
            assert float(sigma[i]) == pytest.approx(0.004)


def test_a_pmu_is_an_override(setup):
    net, layout, _, _ = setup
    sigma = measurement_sigma(layout, overrides={"va_7": 0.001})
    assert float(sigma[layout.index("va_7")]) == pytest.approx(0.001)
    assert not np.isfinite(float(sigma[layout.index("va_5")]))


def test_unknown_variable_names_are_rejected(setup):
    net, layout, _, _ = setup
    with pytest.raises(KeyError):
        measurement_sigma(layout, unmeasured=["va_nowhere"])
    with pytest.raises(KeyError):
        measurement_sigma(layout, overrides={"nonsense": 1.0})


def test_perturb_leaves_unmeasured_entries_alone(setup):
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout)          # angles are inf
    y = perturb(x_true, sigma, jax.random.PRNGKey(3))
    np.testing.assert_allclose(
        y[layout.slice_va], x_true[layout.slice_va], atol=0.0
    )
    assert not np.allclose(y[layout.slice_vm], x_true[layout.slice_vm])


def test_bad_data_is_placed_in_sigma_multiples(setup):
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    clean = perturb(x_true, sigma, jax.random.PRNGKey(4))
    dirty = perturb(
        x_true, sigma, jax.random.PRNGKey(4), layout=layout,
        bad_data={"vm_5": 6.0},
    )
    i = layout.index("vm_5")
    assert float(dirty[i] - clean[i]) == pytest.approx(6.0 * 0.004)


def test_bad_data_needs_a_layout(setup):
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout)
    with pytest.raises(ValueError, match="layout is required"):
        perturb(x_true, sigma, jax.random.PRNGKey(0), bad_data={"vm_5": 3.0})


def test_estimated_values_reads_in_engineering_units(setup):
    net, layout, x_true, reference = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    y = perturb(x_true, sigma, jax.random.PRNGKey(5))
    result = estimate_state(net, y, sigma, layout)
    values = estimated_values(result, layout, net)
    assert set(values) == {"vm", "va_degrees", "pg_mw", "qg_mvar"}
    assert values["vm"]["1"] == pytest.approx(reference.vm["1"], abs=0.02)
    assert values["pg_mw"]["g2"] == pytest.approx(163.0, abs=10.0)


@pytest.mark.slow
def test_monitoring_campaign_flags_the_scan_with_bad_data(setup):
    """What a control centre runs: an estimate per scan against one model.

    Four clean scans and one with a meter twelve sigma out. The clean
    ones must pass the global test and the dirty one must not --- a
    monitor that flags everything, or nothing, is no use either way.
    """
    net, layout, x_true, _ = setup
    sigma = measurement_sigma(layout, sigma_va=0.02)
    scans = [
        perturb(x_true, sigma, jax.random.PRNGKey(seed))
        for seed in range(4)
    ]
    scans.append(
        perturb(
            x_true, sigma, jax.random.PRNGKey(11),
            layout=layout, bad_data={"vm_5": 12.0},
        )
    )
    result = monitor_network(net, scans, sigma, layout)

    assert len(result.steps) == 5
    assert all(not step.failed for step in result.steps)
    detected = list(result.detected)
    assert not any(detected[:4]), "clean scans must not be flagged"
    assert detected[4], "a twelve-sigma error must be flagged"
    assert result.steps[4].statistic > max(
        step.statistic for step in result.steps[:4]
    )
