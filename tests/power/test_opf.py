"""Tests for difflow_power.opf, the AC optimal power flow.

The load-bearing assertions are MATPOWER's published optima:

===============  =============  ====================================
case             optimum $/h    notes
===============  =============  ====================================
``case5``        17551.89       linear costs, two ratings bind
``case9``        5296.69        the standard AC-OPF benchmark
``case14``       8081.53        transformers and a shunt
===============  =============  ====================================

Matching a cost to the cent is a strong test: the optimum is where the
gradient of a nonconvex objective vanishes against a particular active
set, so an error anywhere --- the branch model, the barrier, the
inertia correction, the cost conversion from per unit to MW --- moves
it.

The price tests are stronger still, because they check the SAME number
two independent ways: read off the equality multipliers, and taken as
``jax.grad`` of the optimal cost with respect to demand through the
KKT system. Agreement to solver precision means the multipliers really
are prices and the implicit differentiation really is exact.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.opf import acopf_problem, acopf_structure, generation_cost
from difflow_power.residuals import power_state_layout


@pytest.fixture(scope="module")
def opf9():
    return dp.solve_acopf(dp.cases.case9())


@pytest.fixture(scope="module")
def opf5():
    return dp.solve_acopf(dp.cases.case5())


def test_case9_matches_the_matpower_optimum(opf9):
    assert opf9.converged
    assert opf9.cost == pytest.approx(5296.69, abs=0.01)
    assert opf9.pg_mw["g1"] == pytest.approx(89.80, abs=0.02)
    assert opf9.pg_mw["g2"] == pytest.approx(134.32, abs=0.02)
    assert opf9.pg_mw["g3"] == pytest.approx(94.19, abs=0.02)


def test_case5_matches_the_matpower_optimum(opf5):
    assert opf5.converged
    assert opf5.cost == pytest.approx(17551.89, abs=0.01)


@pytest.mark.slow
def test_case14_matches_the_matpower_optimum():
    result = dp.solve_acopf(dp.cases.case14())
    assert result.converged
    assert result.cost == pytest.approx(8081.53, abs=0.01)


def test_the_opf_beats_the_power_flow_dispatch(opf9):
    """Optimising has to cost less than the scheduled operating point."""
    net = dp.cases.case9()
    pf = dp.solve_power_flow(net)
    layout = power_state_layout(net)
    assert opf9.cost < float(generation_cost(pf.x, net, layout))


def test_the_solution_is_feasible(opf9):
    """Every limit the OPF carried must hold at its answer."""
    report = dp.operating_report(opf9.x, opf9.network, opf9.layout)
    assert report.solved
    assert report.feasible


def test_case5_congestion_separates_the_prices(opf5):
    """The 4-5 rating binds; without it every LMP would be the cheapest
    unit's marginal cost."""
    binding = opf5.binding()
    assert any(k.startswith("rate_") for k in binding)
    lmp = opf5.lmp_mw
    assert max(lmp.values()) - min(lmp.values()) > 10.0
    # Bus 5 hosts the $10/MWh unit and is priced at its offer.
    assert lmp["5"] == pytest.approx(10.0, abs=0.05)


def test_removing_the_ratings_collapses_the_price_spread():
    """The complement of the test above: no congestion, no spread."""
    result = dp.solve_acopf(dp.cases.case5(), enforce_ratings=False)
    assert result.converged
    lmp = result.lmp_mw
    assert max(lmp.values()) - min(lmp.values()) < 1.0
    assert result.cost < 17551.89


def test_prices_from_multipliers_equal_prices_from_autodiff(opf9):
    """Two independent routes to the LMP must agree."""
    errors = opf9.check_prices()
    assert max(errors.values()) < 1e-6


@pytest.mark.slow
def test_prices_from_multipliers_equal_prices_from_autodiff_congested(opf5):
    """The same check where constraints bind, which is the hard case."""
    errors = opf5.check_prices()
    assert max(errors.values()) < 1e-4


def test_solution_sensitivity_matches_finite_differences():
    """d(dispatch)/d(load), by implicit differentiation of the KKT system."""
    net = dp.cases.case9()
    result = dp.solve_acopf(net)
    jacobian = result.solution_sensitivity()
    layout = result.layout
    row = layout.index("pg_g2")
    column = net.bus_index["5"]

    pd, qd = net.load_arrays_pu()
    step = 1e-4
    up = dp.solve_acopf(net, demand=(pd.at[column].add(step), qd))
    down = dp.solve_acopf(net, demand=(pd.at[column].add(-step), qd))
    finite = (
        up.x[layout.slice_pg][1] - down.x[layout.slice_pg][1]
    ) / (2 * step)
    assert float(jacobian[row, column]) == pytest.approx(
        float(finite), rel=1e-3
    )


def test_lmp_predicts_the_cost_of_another_megawatt(opf9):
    """The economic meaning of the price, checked by re-solving.

    A CENTRAL difference over +-1 MW, not a forward one. The cost curves
    are quadratic, so a forward difference over a whole megawatt carries
    a visible second-order term (about 0.15% here) and would be testing
    the curvature rather than the price.
    """
    net = dp.cases.case9()
    pd, qd = net.load_arrays_pu()
    column = net.bus_index["7"]
    step = 1.0 / net.base_mva          # one MW
    up = dp.solve_acopf(net, demand=(pd.at[column].add(step), qd))
    down = dp.solve_acopf(net, demand=(pd.at[column].add(-step), qd))
    assert (up.cost - down.cost) / 2.0 == pytest.approx(
        opf9.lmp_mw["7"], rel=1e-4
    )


def test_structure_drops_constraints_that_cannot_bind():
    """No rating, no row; a +-360 degree angle limit is not a limit."""
    net = dp.cases.case14()          # every branch unrated
    structure = acopf_structure(net, power_state_layout(net))
    assert structure.rated_branches == ()
    assert structure.angle_branches == ()
    assert structure.n_thermal == 0

    rated = dp.cases.case9()         # every branch rated
    s9 = acopf_structure(rated, power_state_layout(rated))
    assert len(s9.rated_branches) == rated.n_branch
    assert s9.n_thermal == 2 * rated.n_branch


def test_constraint_names_line_up_with_the_rows():
    net = dp.cases.case9()
    layout = power_state_layout(net)
    structure = acopf_structure(net, layout)
    nlp = acopf_problem(net, layout, structure)
    x = dp.flat_start(net, layout)
    h = nlp.h(x, net.load_arrays_pu())
    assert h.shape == (structure.n_inequality,)
    assert len(structure.inequality_names) == structure.n_inequality

    # A bound row must respond to its own variable and no other.
    i = structure.inequality_names.index("vm_5_max")
    j = layout.index("vm_5")
    row = jax.jacobian(nlp.h)(x, net.load_arrays_pu())[i]
    assert float(row[j]) == pytest.approx(1.0)
    assert float(jnp.sum(jnp.abs(row))) == pytest.approx(1.0)


def test_fixed_bounds_become_equalities_not_opposing_inequalities():
    """A unit with no range is pinned, not squeezed from both sides."""
    net = dp.cases.case9()
    net.generators["g3"].p_min_mw = net.generators["g3"].p_max_mw
    layout = power_state_layout(net)
    structure = acopf_structure(net, layout)
    assert layout.index("pg_g3") in structure.fixed_indices
    assert "pg_g3_min" not in structure.bound_names
    assert "pg_g3_max" not in structure.bound_names


def test_a_pinned_unit_is_dispatched_at_its_pin():
    net = dp.cases.case9()
    net.generators["g3"].p_min_mw = 120.0
    net.generators["g3"].p_max_mw = 120.0
    result = dp.solve_acopf(net)
    assert result.converged
    assert result.pg_mw["g3"] == pytest.approx(120.0, abs=1e-6)


@pytest.mark.slow
def test_a_controllable_tap_is_used_and_stays_in_range():
    net = dp.cases.case14()
    result = dp.solve_acopf(net, tap_branches=["br8"], max_iterations=300)
    assert result.converged
    tap = float(result.x[result.layout.index("tap_br8")])
    assert 0.9 - 1e-6 <= tap <= 1.1 + 1e-6
    # Free to move, the OPF does not leave it where the case file had it.
    assert abs(tap - net.branches["br8"].tap) > 1e-4
    assert result.cost <= dp.solve_acopf(net).cost + 1e-6


def test_flat_start_reaches_the_same_optimum_as_the_warm_start(opf9):
    cold = dp.solve_acopf(dp.cases.case9(), warm_start=False)
    assert cold.converged
    assert cold.cost == pytest.approx(opf9.cost, abs=1e-4)


def test_summary_names_the_binding_constraints(opf5):
    text = opf5.summary()
    assert "binding constraints" in text
    assert "$/MWh" in text
    assert "ACOPFResult" in repr(opf5)
