"""Tests for difflow_power.residuals.

This module holds the single definition of a network's equation set;
:mod:`difflow_power.verify` and everything else are consumers of it.
That means ``verify`` cannot serve as an oracle for these tests --- it
would be checking the code against itself --- so
:func:`reference_residuals` below restates the equations independently
in POLAR form, straight from the textbook statement of the power flow
problem, while the implementation works in complex rectangular form.
Two different algebraic routes to the same numbers is what makes the
comparison worth anything.

Keeping the oracle in the test file rather than in production code is
deliberate: a second implementation is worth having precisely as a
check, and worth nothing as an import.
"""

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


def reference_ybus(network):
    """Bus admittance matrix, restated in plain Python complex arithmetic.

    Written from the standard branch model: an ideal transformer of
    complex ratio ``t = tau exp(j theta)`` at the from end, followed by
    a pi section carrying the series admittance and half the charging
    at each end.
    """
    n = network.n_bus
    index = network.bus_index
    y = [[0j] * n for _ in range(n)]
    for br in network.branches.values():
        f, t = index[br.from_bus], index[br.to_bus]
        ys = 1.0 / complex(br.r, br.x)
        yc = complex(br.g, br.b) / 2.0
        ratio = br.tap * complex(math.cos(br.shift), math.sin(br.shift))
        y[f][f] += (ys + yc) / (br.tap * br.tap)
        y[f][t] += -ys / ratio.conjugate()
        y[t][f] += -ys / ratio
        y[t][t] += ys + yc
    for bid, bus in network.buses.items():
        i = index[bid]
        y[i][i] += complex(bus.g_shunt_mw, bus.b_shunt_mvar) / network.base_mva
    return y


def reference_residuals(vm, va, pg, qg, network):
    """Power balance in POLAR form, restated from the textbook.

        P_i = V_i sum_k V_k (G_ik cos t_ik + B_ik sin t_ik)
        Q_i = V_i sum_k V_k (G_ik sin t_ik - B_ik cos t_ik)

    with ``t_ik = theta_i - theta_k``. Returns ``{name: value}`` using
    the same names as :func:`residual_names`.
    """
    y = reference_ybus(network)
    index = network.bus_index
    base = network.base_mva
    n = network.n_bus

    scheduled_p = [0.0] * n
    scheduled_q = [0.0] * n
    for k, gid in enumerate(network.generator_ids):
        i = index[network.generators[gid].bus]
        scheduled_p[i] += pg[k]
        scheduled_q[i] += qg[k]
    for load in network.loads.values():
        i = index[load.bus]
        scheduled_p[i] -= load.p_mw / base
        scheduled_q[i] -= load.q_mvar / base

    out = {}
    for bid in network.bus_ids:
        i = index[bid]
        p_net = q_net = 0.0
        for kid in network.bus_ids:
            k = index[kid]
            g, b = y[i][k].real, y[i][k].imag
            delta = va[i] - va[k]
            p_net += vm[k] * (g * math.cos(delta) + b * math.sin(delta))
            q_net += vm[k] * (g * math.sin(delta) - b * math.cos(delta))
        out[f"p_balance_{bid}"] = scheduled_p[i] - vm[i] * p_net
        out[f"q_balance_{bid}"] = scheduled_q[i] - vm[i] * q_net
    slack = network.slack_bus
    out[f"va_ref_{slack}"] = (
        va[index[slack]] - network.buses[slack].va_reference
    )
    return out


ALL_CASES = ["case3", "case5", "case9", "case14", "radial_feeder"]


def arbitrary_state(network, seed=0):
    """A state that is NOT a solution, so the residuals are nonzero.

    Testing residuals at a solved point would only show that both
    implementations return zero, which any two wrong implementations
    also manage.
    """
    rng = np.random.default_rng(seed)
    vm = 0.95 + 0.1 * rng.random(network.n_bus)
    va = 0.3 * (rng.random(network.n_bus) - 0.5)
    pg = rng.random(network.n_gen)
    qg = 0.4 * (rng.random(network.n_gen) - 0.5)
    return vm, va, pg, qg


@pytest.mark.parametrize("case", ALL_CASES)
def test_residuals_match_independent_polar_restatement(case):
    """The complex-rectangular implementation equals the polar form."""
    network = dp.cases.load_case(case)
    layout = power_state_layout(network)
    vm, va, pg, qg = arbitrary_state(network)

    got = power_flow_residuals(layout.pack(vm, va, pg, qg), network, layout)
    want = reference_residuals(vm, va, pg, qg, network)
    names = residual_names(network, layout)

    assert len(names) == len(got) == layout.n_residual
    for name, value in zip(names, got):
        assert float(value) == pytest.approx(want[name], abs=1e-12)


@pytest.mark.parametrize("case", ALL_CASES)
def test_residuals_are_not_trivially_zero(case):
    """Guard the test above: an arbitrary state must break the equations."""
    network = dp.cases.load_case(case)
    layout = power_state_layout(network)
    r = power_flow_residuals(
        layout.pack(*arbitrary_state(network)), network, layout
    )
    assert float(jnp.max(jnp.abs(r))) > 1e-3


@pytest.mark.parametrize("case", ALL_CASES)
def test_jacobian_has_full_row_rank(case):
    """The reference row is what makes the Jacobian full rank.

    Without it the equations are invariant to a global angle shift and
    the rank is one short, which every downstream method that inverts
    the Jacobian would hit.
    """
    network = dp.cases.load_case(case)
    layout = power_state_layout(network)
    x = layout.pack(*arbitrary_state(network))
    jacobian = jax.jacobian(power_flow_residuals)(x, network, layout)
    assert jacobian.shape == (layout.n_residual, layout.size)
    assert int(jnp.linalg.matrix_rank(jacobian)) == layout.n_residual

    # The balance rows alone are blind to a global angle shift: the
    # all-ones angle direction sits in their null space. That is the
    # structural deficiency the reference row exists to remove, and it
    # is what would make a square power-flow system singular.
    shift = jnp.zeros(layout.size).at[layout.slice_va].set(1.0)
    assert float(jnp.max(jnp.abs(jacobian[:-1] @ shift))) < 1e-9
    assert float(jnp.abs(jacobian[-1] @ shift)) == pytest.approx(1.0)


def test_global_angle_shift_leaves_balances_unchanged():
    """The invariance the reference row exists to break."""
    network = dp.cases.case9()
    layout = power_state_layout(network)
    vm, va, pg, qg = arbitrary_state(network)
    a = power_flow_residuals(layout.pack(vm, va, pg, qg), network, layout)
    b = power_flow_residuals(
        layout.pack(vm, va + 0.17, pg, qg), network, layout
    )
    np.testing.assert_allclose(a[:-1], b[:-1], atol=1e-12)
    assert float(b[-1] - a[-1]) == pytest.approx(0.17)


def test_layout_pack_unpack_round_trip():
    network = dp.cases.case14()
    layout = power_state_layout(
        network, demand_buses=["2", "3"], tap_branches=["br8"]
    )
    vm, va, pg, qg = arbitrary_state(network)
    x = layout.pack(
        vm, va, pg, qg, pd=[0.1, 0.2], qd=[0.03, 0.04],
        extra={"tap_br8": 0.97},
    )
    assert x.shape == (layout.size,)
    values = layout.unpack(x)
    assert values["tap_br8"] == pytest.approx(0.97)
    assert values["pd_3"] == pytest.approx(0.2)

    state = layout.unpack_arrays(x, network)
    assert float(state.tap[network.branch_ids.index("br8")]) == pytest.approx(0.97)
    # A branch not in the layout keeps the network's own tap.
    j = network.branch_ids.index("br9")
    assert float(state.tap[j]) == pytest.approx(network.branches["br9"].tap)


def test_layout_embed_maps_by_name():
    """Extending the state changes the pack order; embed must follow names."""
    network = dp.cases.case9()
    plain = power_state_layout(network)
    extended = power_state_layout(network, demand_buses=["5"])
    x = plain.pack(*arbitrary_state(network))
    y = extended.embed(x, plain)

    for name in plain.names:
        assert float(y[extended.index(name)]) == float(x[plain.index(name)])
    assert math.isnan(float(y[extended.index("pd_5")]))


def test_layout_index_error_names_the_layout():
    layout = power_state_layout(dp.cases.case3())
    with pytest.raises(KeyError, match="not a state variable"):
        layout.index("vm_nowhere")


def test_tap_branch_must_be_a_transformer():
    network = dp.cases.case9()
    with pytest.raises(ValueError, match="line, not a transformer"):
        power_state_layout(network, tap_branches=["br2"])


def test_branch_flows_sum_to_a_positive_loss():
    """A passive branch cannot generate real power."""
    network = dp.cases.case9()
    layout = power_state_layout(network)
    result = dp.solve_power_flow(network)
    s_from, s_to = dp.branch_flows(result.x, network, layout)
    losses = jnp.real(s_from + s_to)
    assert float(jnp.min(losses)) > -1e-12
    assert float(jnp.sum(losses)) * network.base_mva == pytest.approx(
        4.9547, abs=1e-3
    )


def test_residuals_differentiate_with_respect_to_a_line_parameter():
    """branch_params is the traced handle on the model's coefficients."""
    network = dp.cases.case9()
    layout = power_state_layout(network)
    x = layout.pack(*arbitrary_state(network))
    reactances = network.branch_param_arrays()["x"]

    def total(xs):
        return jnp.sum(
            power_flow_residuals(x, network, layout, branch_params={"x": xs})
            ** 2
        )

    grad = jax.grad(total)(reactances)
    step = 1e-6
    for j in (0, 3, 7):
        up = total(reactances.at[j].add(step))
        down = total(reactances.at[j].add(-step))
        assert float(grad[j]) == pytest.approx(
            float((up - down) / (2 * step)), rel=1e-5
        )
