"""Delta-base SLP against AC-OPF and DC-OPF, all scored in the full AC model.

`difflow.planning` claims an AD delta vector is a *better* linearisation than
the fixed one a planning system normally uses. On a power network that claim
is directly testable, because the industry's fixed linearisation --- DC-OPF,
which assumes flat voltage, small angles and no losses --- is right there in
`difflow_power` to compare against, and the exact nonlinear answer is there
too.

The comparison only means anything if all three dispatches are scored the same
way, so each one is fed back into the FULL AC power flow and its true cost and
true violations are read off. A DC-OPF reports a cost below the AC optimum
because it does not pay for losses; that number is not comparable to anything
and is never used here.

MATPOWER reference: ``runopf case9`` gives $5296.69/h.

What this pins:

1. ``test_slp_reaches_the_ac_optimum`` --- the SLP over AD delta vectors lands
   on the AC-OPF cost, AC-feasible. This is the module's accuracy claim, and
   `difflow.planning.benchmark` otherwise only argues about cost.
2. ``test_slp_beats_the_fixed_linearisation`` --- and beats DC-OPF, which is
   the linearisation it is actually competing with.
3. ``test_linear_slp_terminates_on_iteration_budget`` and
   ``test_quadratic_subproblem_terminates`` --- termination, which was the
   weak point. A first-order model of this objective FINDS the optimum
   without CERTIFYING it and runs out its iteration budget. Adding the
   curvature (`difflow.planning.quadratic`) cuts the run from 40 iterations
   that never terminate to about a dozen that do, at the same cost and the
   same feasibility. Both are pinned, because the gap between them is the
   claim.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
import difflow_power.powerflow as pf
from difflow.planning import Block, DeltaBasePlanner, Network
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions

# Every test here pays for a one-off JAX trace/compile of the Newton power
# flow nested inside a planning block, plus its implicit-diff backward pass --
# the repo's `slow` category, where the cost is the computation rather than the
# assertion. CI runs the slow job concurrently, so coverage is unchanged.
pytestmark = pytest.mark.slow

#: MATPOWER ``runopf case9``.
AC_OPF_COST = 5296.6862
VLO, VHI = 0.9, 1.1


@pytest.fixture(scope="module")
def grid():
    """case9 reduced to u -> y: generator schedule in, cost and limits out."""
    net = dp.case9()
    base = net.base_mva
    layout = pf.power_state_layout(net)
    spec0 = pf.specification_from_network(net)
    x0 = pf.flat_start(net, layout, spec0)
    rate = jnp.asarray(net.branch_rate_array_pu())
    gens = layout.generators
    costs = {g: net.generators[g].cost for g in gens}
    pq = [i for i, b in enumerate(net.buses.values()) if b.kind == "pq"]

    def outputs(u):
        """u = [pg2 MW, pg3 MW, vm1, vm2, vm3]."""
        spec = replace(
            spec0,
            pg_setpoint={"g2": u[0] / base, "g3": u[1] / base},
            vm_setpoint={"1": u[2], "2": u[3], "3": u[4]})
        # solve_state, not solve_power_flow: the reporting wrapper takes a
        # float() of its mismatch diagnostic and is deliberately not traceable.
        x, _ = pf.solve_state(net, layout, spec, x0)
        pg = x[layout.slice_pg] * base
        cost = 0.0
        for i, g in enumerate(gens):
            c2, c1, c0 = costs[g]
            cost = cost + c2 * pg[i] ** 2 + c1 * pg[i] + c0
        s_f, s_t = pf.branch_flows(x, net, layout)
        loading = jnp.concatenate([jnp.abs(s_f) ** 2 / rate ** 2,
                                   jnp.abs(s_t) ** 2 / rate ** 2])
        vm = x[layout.slice_vm][jnp.array(pq)]
        return jnp.concatenate([jnp.array([cost]), vm, loading])

    n_branch = len(rate)
    y_names = (["cost"] + [f"vpq{i}" for i in pq]
               + [f"s2_{k}" for k in range(2 * n_branch)])
    incumbent = [net.generators["g2"].p_mw, net.generators["g3"].p_mw,
                 1.0, 1.0, 1.0]

    def score(u):
        """True AC cost and true worst violation of a dispatch."""
        y = np.asarray(outputs(jnp.asarray(u, dtype=float)))
        vm = y[1:1 + len(pq)]
        s2 = y[1 + len(pq):]
        violation = max(0.0,
                        float(np.max(VLO - vm)), float(np.max(vm - VHI)),
                        float(np.max(s2) - 1.0))
        return float(y[0]), violation

    return {
        "net": net, "outputs": outputs, "y_names": y_names, "pq": pq,
        "n_branch": n_branch, "incumbent": incumbent, "score": score,
    }


def _planner(grid, radius, max_iter=40, model_order="linear"):
    block = Block(name="grid", fn=grid["outputs"],
                  u_names=["pg2", "pg3", "vm1", "vm2", "vm3"],
                  y_names=grid["y_names"],
                  lb=[10.0, 10.0, VLO, VLO, VLO],
                  ub=[300.0, 270.0, VHI, VHI, VHI],
                  u0=grid["incumbent"], jit=True)
    specs = (
        [Spec(f"grid.vpq{i}", ">=", VLO, elastic=False) for i in grid["pq"]]
        + [Spec(f"grid.vpq{i}", "<=", VHI, elastic=False) for i in grid["pq"]]
        + [Spec(f"grid.s2_{k}", "<=", 1.0, elastic=False)
           for k in range(2 * grid["n_branch"])])
    return DeltaBasePlanner(
        Network([block]), prices={"grid.cost": 1.0}, specs=specs,
        radius=radius, sense="min", model_order=model_order,
        options=TrustRegionOptions(max_iter=max_iter))


@pytest.fixture(scope="module")
def slp_plan(grid):
    res = _planner(grid, radius=0.2).solve()
    return np.asarray(res.decisions, dtype=float), res


def test_reference_ac_opf_matches_matpower(grid):
    result = dp.solve_acopf(grid["net"])
    assert result.converged
    assert result.cost == pytest.approx(AC_OPF_COST, rel=1e-5)


def test_incumbent_schedule_is_feasible_but_dear(grid):
    """The point the planner starts from: legal, and 2.7% off optimal."""
    cost, violation = grid["score"](grid["incumbent"])
    assert violation < 1e-8
    assert cost > AC_OPF_COST * 1.02


def test_slp_reaches_the_ac_optimum(grid, slp_plan):
    """The accuracy claim: a first-order model, re-taken, finds the NLP answer."""
    plan, _ = slp_plan
    cost, violation = grid["score"](plan)
    assert violation < 1e-6, f"SLP plan violates AC limits by {violation:.2e}"
    assert cost == pytest.approx(AC_OPF_COST, rel=1e-4)


def test_slp_beats_the_fixed_linearisation(grid, slp_plan):
    """DC-OPF is the linearisation the industry actually clears on.

    Its own reported cost is below the AC optimum because it does not pay for
    losses, so what is compared is its *dispatch*, scored in AC --- which is
    what the plant would really cost if you ran it.
    """
    dc = dp.solve_dcopf(grid["net"])
    dc_cost, dc_violation = grid["score"](
        [dc.pg_mw["g2"], dc.pg_mw["g3"], 1.0, 1.0, 1.0])
    slp_cost, _ = grid["score"](slp_plan[0])

    assert dc_violation < 1e-8, "DC dispatch should be AC-feasible on case9"
    assert dc_cost > AC_OPF_COST
    assert slp_cost < dc_cost, (
        f"SLP {slp_cost:.2f} did not beat DC-OPF {dc_cost:.2f}")


def test_linear_slp_terminates_on_iteration_budget(grid, slp_plan):
    """A first-order model finds the optimum without certifying it.

    The delta vectors predict a gain the AC model does not deliver, the
    radius ratchets down, and the loop exhausts its budget sitting on the
    right answer. Pinned so the improvement below has something to be
    measured against.
    """
    _, res = slp_plan
    assert res.reason == "max_iter"


def test_quadratic_subproblem_terminates(grid):
    """Curvature is what lets the loop stop.

    Same optimum, same AC feasibility, a third of the iterations -- and it
    ends on its own rather than on the iteration cap. Iterations are the
    metric that matters here rather than wall time: each one is an
    evaluation of the caller's blocks, which on a real flowsheet is the
    entire cost.
    """
    linear = _planner(grid, radius=0.2, model_order="linear").solve()
    quad = _planner(grid, radius=0.2, model_order="quadratic").solve()

    cost, violation = grid["score"](np.asarray(quad.decisions, dtype=float))
    assert violation < 1e-6
    assert cost == pytest.approx(AC_OPF_COST, rel=1e-4)

    assert quad.reason != "max_iter", "the quadratic loop still did not stop"
    assert quad.n_iterations < linear.n_iterations


def test_auto_model_order_also_terminates(grid):
    """`auto` takes the exact curvature where it is definite, and stops too.

    At the incumbent schedule the reduced AC cost Hessian is positive
    definite, so `auto` gets a true second-order model with nothing clipped.
    """
    res = _planner(grid, radius=0.2, model_order="auto").solve()
    cost, violation = grid["score"](np.asarray(res.decisions, dtype=float))
    assert violation < 1e-6
    assert cost == pytest.approx(AC_OPF_COST, rel=1e-4)
    assert res.reason != "max_iter"


@pytest.mark.parametrize("radius", [0.05, 0.2])
def test_optimum_found_from_either_trust_radius(grid, radius):
    res = _planner(grid, radius=radius).solve()
    cost, violation = grid["score"](np.asarray(res.decisions, dtype=float))
    assert violation < 1e-6
    assert cost == pytest.approx(AC_OPF_COST, rel=1e-4)
