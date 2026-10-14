"""Tests for difflow_power.dc: the linearised model, DC-OPF, PTDF, LODF.

MATPOWER reference: ``rundcopf case5`` gives $17479.90/h with
``Pg = (40, 170, 323.49, 0, 466.51)`` MW and LMPs
``(16.98, 26.38, 30.00, 39.94, 10.00)`` $/MWh.

The PTDF and LODF tests check STRUCTURAL properties --- the reference
column is zero, rows sum correctly, an islanding outage is undefined
--- and then check the factors against what a re-solve actually does.
A distribution factor that satisfies its identities but predicts the
wrong flow is no use.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.dc import (
    contingency_flows,
    dc_matrices,
    lodf,
    ptdf,
    solve_dc_power_flow,
    solve_dcopf,
)


def test_case5_dcopf_matches_matpower():
    result = solve_dcopf(dp.cases.case5())
    assert result.converged
    assert result.cost == pytest.approx(17479.90, abs=0.01)
    want = {"g1": 40.0, "g2": 170.0, "g3": 323.49, "g4": 0.0, "g5": 466.51}
    for gid, value in want.items():
        assert result.pg_mw[gid] == pytest.approx(value, abs=0.02)
    want_lmp = {
        "1": 16.98, "2": 26.38, "3": 30.00, "4": 39.94, "5": 10.00,
    }
    for bus, value in want_lmp.items():
        assert result.lmp_mw[bus] == pytest.approx(value, abs=0.02)


def test_dcopf_is_lossless_so_generation_equals_load():
    for case in ("case5", "case9"):
        net = dp.cases.load_case(case)
        result = solve_dcopf(net)
        assert result.converged
        assert sum(result.pg_mw.values()) == pytest.approx(
            net.total_load_mw, abs=1e-6
        )


def test_dcopf_is_optimistic_about_cost():
    """No losses means nobody generates them, so DC always under-prices."""
    net = dp.cases.case9()
    assert solve_dcopf(net).cost < dp.solve_acopf(net).cost


def test_uncongested_dc_prices_are_identical():
    """With no losses and no binding constraint there is nothing to
    separate one bus's price from another's."""
    result = solve_dcopf(dp.cases.case9(), enforce_ratings=False)
    assert result.converged
    prices = list(result.lmp_mw.values())
    assert max(prices) - min(prices) < 1e-6


def test_dc_power_flow_balances_and_is_lossless():
    net = dp.cases.case9()
    result = solve_dc_power_flow(net)
    m = dc_matrices(net)
    injections = m.b_bus @ result.va + m.p_bus_shift
    # Total injection is zero: the DC model conserves real power exactly.
    assert float(jnp.sum(injections)) == pytest.approx(0.0, abs=1e-9)
    assert result.va_degrees[net.slack_bus] == pytest.approx(0.0)


def test_dc_flows_are_within_a_few_percent_of_ac():
    """The claim that justifies using the DC model at all.

    Accuracy is measured against the network's LARGEST flow, which is
    how screening tolerances are quoted: a 3 MW error on a 250 MW
    corridor is irrelevant, and the same error on a 5 MW tie is not
    news either. It cannot be measured per branch: the DC model is
    lossless, so it hands the slack bus an injection short by exactly
    the total losses, and the branches out of the slack carry that whole
    discrepancy however small their own flow is.
    """
    net = dp.cases.case9()
    ac = dp.solve_power_flow(net)
    injections = jnp.asarray(
        [
            sum(ac.pg_mw[g] for g in net.generators_at(b)) / net.base_mva
            for b in net.bus_ids
        ]
    ) - net.load_arrays_pu()[0]
    dc = solve_dc_power_flow(net, injections=injections)

    largest = max(abs(ac.branch_mw[a][0]) for a in net.branch_ids)
    errors = {
        a: abs(ac.branch_mw[a][0] - dc.branch_mw[a])
        for a in net.branch_ids
    }
    assert max(errors.values()) < 0.05 * largest
    # And the slack's own branch is off by close to the total losses.
    assert errors["br1"] == pytest.approx(ac.losses_mw, rel=0.05)


def test_ptdf_reference_column_is_zero_and_shape_is_right():
    net = dp.cases.case9()
    factors = ptdf(net)
    assert factors.shape == (net.n_branch, net.n_bus)
    np.testing.assert_allclose(
        factors[:, net.bus_index[net.slack_bus]], 0.0, atol=1e-12
    )


def test_ptdf_predicts_a_dc_transfer_exactly():
    """One MW from bus 5 to the slack, compared against a re-solve."""
    net = dp.cases.case9()
    factors = ptdf(net)
    base = solve_dc_power_flow(net)
    pd, _ = net.load_arrays_pu()
    gen_idx = net.generator_bus_indices()
    pg = jnp.asarray(
        [g.p_mw / net.base_mva for g in net.generators.values()]
    )
    injections = jnp.zeros(net.n_bus).at[gen_idx].add(pg) - pd

    i = net.bus_index["5"]
    step = 0.1
    moved = solve_dc_power_flow(net, injections=injections.at[i].add(step))
    np.testing.assert_allclose(
        moved.p_from - base.p_from, step * factors[:, i], atol=1e-10
    )


def test_ptdf_is_slack_independent_for_a_real_transfer():
    """Injecting at 5 and withdrawing at 9 is a physical transfer, so it
    cannot depend on which bus the columns are referenced to."""
    net = dp.cases.case9()
    a = ptdf(net, slack="1")
    b = ptdf(net, slack="7")
    i, j = net.bus_index["5"], net.bus_index["9"]
    np.testing.assert_allclose(
        a[:, i] - a[:, j], b[:, i] - b[:, j], atol=1e-10
    )


def test_lodf_diagonal_is_minus_one_where_the_outage_is_survivable():
    """An outaged branch loses all of its own flow.

    ``case5`` has no bridges --- every branch lies on a cycle --- so
    every diagonal entry is defined. ``case9`` has three (its step-up
    transformers), whose outage islands a machine, and those columns are
    ``nan`` instead.
    """
    np.testing.assert_allclose(
        jnp.diag(lodf(dp.cases.case5())), -1.0, atol=1e-12
    )

    net = dp.cases.case9()
    diagonal = jnp.diag(lodf(net))
    islanding = jnp.isnan(diagonal)
    assert int(jnp.sum(islanding)) == 3
    np.testing.assert_allclose(diagonal[~islanding], -1.0, atol=1e-12)
    # The three are exactly the transformers connecting the machines.
    bridges = {
        net.branch_ids[i] for i in range(net.n_branch) if bool(islanding[i])
    }
    assert bridges == {"br1", "br4", "br7"}


def test_lodf_predicts_an_outage_exactly():
    """Rebuild the network without a branch and compare the DC flows."""
    from dataclasses import replace

    net = dp.cases.case9()
    base = solve_dc_power_flow(net)
    predicted = contingency_flows(net, base.p_from)

    outaged = "br5"
    k = net.branch_ids.index(outaged)
    reduced = replace(
        net, branches={a: b for a, b in net.branches.items() if a != outaged}
    )
    after = solve_dc_power_flow(reduced)
    for i, aid in enumerate(net.branch_ids):
        if aid == outaged:
            continue
        j = reduced.branch_ids.index(aid)
        assert float(predicted[i, k]) == pytest.approx(
            float(after.p_from[j]), abs=1e-9
        )


def test_lodf_marks_an_islanding_outage_as_undefined():
    """A radial branch's outage disconnects the network; there is no
    post-outage flow to redistribute, and nan says so."""
    factors = lodf(dp.cases.radial_feeder())
    assert bool(jnp.all(jnp.isnan(factors)))


def test_dc_matrices_differentiate_with_respect_to_reactance():
    net = dp.cases.case9()
    reactances = net.branch_param_arrays()["x"]

    def flow_on_branch_zero(xs):
        return solve_dc_power_flow(
            net, branch_params={"x": xs}
        ).p_from[2]

    grad = jax.grad(flow_on_branch_zero)(reactances)
    step = 1e-8
    j = 4
    want = (
        flow_on_branch_zero(reactances.at[j].add(step))
        - flow_on_branch_zero(reactances.at[j].add(-step))
    ) / (2 * step)
    assert float(grad[j]) == pytest.approx(float(want), rel=1e-4)


def test_dcopf_binding_constraints_are_named():
    result = solve_dcopf(dp.cases.case5())
    binding = result.binding()
    assert binding
    assert all(
        name.startswith(("pg_", "rate_")) for name in binding
    )
    assert "DCOPFResult" in repr(result)
