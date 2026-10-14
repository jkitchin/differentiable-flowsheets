"""Multi-period planning: inventory carried between periods by a forward Link.

A `Link` is output-to-input and the network rejects only cycles, so a forward
link from one period's tank level to the next is an ordinary DAG edge.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, Network
from difflow.planning.lp import Spec

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
