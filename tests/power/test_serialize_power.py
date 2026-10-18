"""Power networks and radial feeders through difflow.serialize."""

import jax

jax.config.update("jax_enable_x64", True)

from difflow import serialize
from difflow_power import RadialFeederFlowsheet, case9, radial_feeder


def test_a_network_round_trips_in_insertion_order():
    """Bus order is insertion order; a sorted reload would renumber it."""
    net = case9()
    back = serialize.from_json(serialize.to_json(net))
    assert back == net and list(back.buses) == list(net.buses)


def test_a_radial_feeder_rebuilds_from_its_constructor_call():
    """It keeps the tree derived from `root`, not `root` itself."""
    feeder = RadialFeederFlowsheet(radial_feeder())
    text = serialize.to_json(feeder)
    back = serialize.from_json(text)
    assert type(back) is RadialFeederFlowsheet
    assert serialize.to_json(back) == text
