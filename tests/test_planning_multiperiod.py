"""Multi-period planning with inventory carried between periods.

`difflow.planning.chain.two_plant_chain(horizon=n)` replicates blocks per
period and couples them only through a shared cap, which is a horizon built to
make the AD scaling argument measurable. A *planning* model couples periods
through inventory: what is not sold this period is still there next period.

These tests record that this needs no new machinery. A `Link` is
output-to-input and `Network._topological_order` rejects only cycles, so a
forward link from one period's tank level to the next is an ordinary DAG edge.
They also pin the two ways of getting it wrong, both of which were found by
building the model rather than by reading the code:

* ``test_elastic_specs_would_sell_from_an_empty_tank`` — the `Spec` default of
  ``elastic=True`` is right for a commercial specification and wrong for a
  mass balance, and produces a *better-looking* objective by breaking physics.
* ``test_inelastic_spec_violated_at_the_start_is_a_dead_end`` — the planner's
  response to an infeasible LP is to shrink the radius, which cannot restore
  feasibility. There is no restoration phase; the failure is reported, not
  silent, but the model has to start feasible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, DeltaBasePlanner, Network
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions

HORIZON = 4
PRICES = [10.0, 10.0, 40.0, 10.0]      # a spike at t2
CAPACITY = 5.0
MAKE_COST = 6.0


def _gen_fn(u):
    """Curved production, so the SLP has something to linearise."""
    return jnp.array([u[0] - 0.05 * u[0] ** 2])


def _tank_fn(u):
    level_in, inflow, release = u
    return jnp.array([level_in + inflow - release, release])


def _tank0_fn(u):
    """The first period starts empty -- structurally, not by a constraint."""
    inflow, release = u
    return jnp.array([inflow - release, release])


def storage_network(elastic: bool = False, start_as_decision: bool = False):
    """Make cheaply every period, hold in a tank, sell into the price spike."""
    blocks, links, prices, specs = [], [], {}, []
    for t in range(HORIZON):
        blocks.append(Block(name=f"gen@t{t}", fn=_gen_fn, u_names=["rate"],
                            y_names=["make"], lb=[0.0], ub=[4.0]))
        first = t == 0 and not start_as_decision
        if first:
            blocks.append(Block(
                name="tank@t0", fn=_tank0_fn, u_names=["inflow", "release"],
                y_names=["level_out", "sold"], lb=[0.0, 0.0], ub=[4.0, 4.0]))
        else:
            blocks.append(Block(
                name=f"tank@t{t}", fn=_tank_fn,
                u_names=["level_in", "inflow", "release"],
                y_names=["level_out", "sold"],
                lb=[0.0, 0.0, 0.0], ub=[CAPACITY, 4.0, 4.0]))
            if t:
                links.append((f"tank@t{t-1}.level_out", f"tank@t{t}.level_in"))
        links.append((f"gen@t{t}.make", f"tank@t{t}.inflow"))
        prices[f"tank@t{t}.sold"] = PRICES[t]
        prices[f"gen@t{t}.rate"] = -MAKE_COST

        # You cannot release what you do not have, and the tank has a lid.
        draw = {f"tank@t{t}.release": 1.0, f"tank@t{t}.inflow": -1.0}
        if not first:
            draw[f"tank@t{t}.level_in"] = -1.0
        specs += [
            Spec(draw, "<=", 0.0, elastic=elastic, name=f"draw@t{t}"),
            Spec(f"tank@t{t}.level_out", "<=", CAPACITY, elastic=elastic,
                 name=f"cap@t{t}"),
            Spec(f"tank@t{t}.level_out", ">=", 0.0, elastic=elastic,
                 name=f"floor@t{t}"),
        ]
    if start_as_decision:
        specs.append(Spec("tank@t0.level_in", "==", 0.0, elastic=elastic,
                          name="start"))
    return Network(blocks, links=links), prices, specs


def _solve(network, prices, specs, penalty=None):
    """Run the planner.

    Vertex seeding is off and ``radius_min`` is loose: none of these plans is
    bang-bang, so the nine extra starts and the last decade of radius change
    the answer not at all and cost a factor of eighteen in runtime. Checked,
    not assumed -- the same plan comes back either way.
    """
    kwargs = {} if penalty is None else {"penalty": penalty}
    return DeltaBasePlanner(
        network, prices=prices, specs=specs, radius=0.4,
        vertex_seeding=False,
        options=TrustRegionOptions(radius=0.4, radius_min=1e-4),
        **kwargs).solve()


def _levels(network, decisions):
    state = network.evaluate(decisions, None)
    return np.array([float(state[f"tank@t{t}.level_out"])
                     for t in range(HORIZON)])


# -- the structure already works ------------------------------------------

def test_cross_period_link_is_not_a_recycle():
    """A forward link between periods is a DAG edge, not a loop."""
    net, _, _ = storage_network()
    order = net.order
    for t in range(1, HORIZON):
        assert order.index(f"tank@t{t-1}") < order.index(f"tank@t{t}")


def test_linked_inventory_is_not_a_free_decision():
    """Next period's opening level is determined, not chosen."""
    net, _, _ = storage_network()
    free = set(net.decision_names)
    for t in range(1, HORIZON):
        assert f"tank@t{t}.level_in" not in free
        assert f"tank@t{t}.release" in free


def test_storage_arbitrage_beats_selling_everything_immediately():
    """The plan must hold inventory back for the price spike."""
    net, prices, specs = storage_network()
    res = _solve(net, prices, specs)
    levels = _levels(net, res.decisions)

    assert levels[1] > 0.5, "nothing was held back for the spike"
    assert levels[2] < levels[1], "the tank was not drawn down into the spike"
    assert float(res.objective) > 140.0

    state = net.evaluate(res.decisions, None)
    spike = float(state["tank@t2.sold"])
    for t in (0, 1, 3):
        assert spike >= float(state[f"tank@t{t}.sold"])


def test_physics_holds_in_the_accepted_plan():
    """Inventory stays between zero and the lid, to the solver's tolerance.

    The slack here is set by ``radius_min``: the loop stops moving once the
    trust region is smaller than that, so the last decade of an active
    constraint is left on the table. At 1e-4 of a five-unit tank that is a
    residual around 1e-7, which is a convergence tolerance rather than a
    physics violation -- an inelastic spec cannot be traded away.
    """
    net, prices, specs = storage_network()
    res = _solve(net, prices, specs)
    levels = _levels(net, res.decisions)
    tol = 1e-6 * CAPACITY
    assert np.all(levels >= -tol), f"negative inventory: {levels}"
    assert np.all(levels <= CAPACITY + tol), f"over capacity: {levels}"


# -- the two ways to get it wrong -----------------------------------------

def test_elastic_specs_would_sell_from_an_empty_tank():
    """Elastic slack on a mass balance buys a better objective with fiction.

    This is the regression for the `Spec` default. An elastic physical
    constraint lets the planner report a *higher* objective than the feasible
    plan earns, by running the tank negative -- and the run converges and
    reports the number without complaint.
    """
    net, prices, specs = storage_network(elastic=True)
    res = _solve(net, prices, specs, penalty=1.0)
    levels = _levels(net, res.decisions)
    assert np.min(levels) < -1e-6, (
        "expected the elastic formulation to break the inventory balance; "
        f"levels were {levels}")


def test_inelastic_spec_violated_at_the_start_is_a_dead_end():
    """No feasibility restoration: an infeasible start cannot be recovered.

    `tank@t0.level_in` defaults to the midpoint of its bounds, the trust
    region clips around that midpoint, and the `== 0` spec sits outside the
    clip. The LP is infeasible from the first cycle, and shrinking the radius
    only tightens it. The planner reports this rather than hiding it, which
    is what this test pins -- together with the fact that a restoration phase
    is what would fix it.
    """
    net, prices, specs = storage_network(start_as_decision=True)
    res = _solve(net, prices, specs)
    assert not res.converged
    assert res.reason == "lp_infeasible"
    assert max(res.violations.values(), default=0.0) > 1e-6


def test_a_feasible_start_reaches_the_same_plan_with_the_level_as_a_decision():
    """The dead end is the start point, not the formulation."""
    net, prices, specs = storage_network(start_as_decision=True)
    blocks = [b if b.name != "tank@t0"
              else type(b)(**{**{k: getattr(b, k) for k in
                                 ("name", "fn", "u_names", "y_names", "lb",
                                  "ub")},
                              "u0": [0.0, 2.0, 2.0]})
              for b in net.blocks]
    res = _solve(Network(blocks, links=net.links), prices, specs)
    assert max(res.violations.values(), default=0.0) < 1e-6
    levels = _levels(Network(blocks, links=net.links), res.decisions)
    assert np.all(levels >= -1e-6)
