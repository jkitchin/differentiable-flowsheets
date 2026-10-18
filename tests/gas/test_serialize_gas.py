"""Gas-network flowsheets through difflow.serialize."""

import jax

jax.config.update("jax_enable_x64", True)

from difflow import Unit, serialize
from difflow_gas import GasNetworkFlowsheet, GasPipe


def test_a_gas_network_flowsheet_keeps_its_class():
    """Its own solve was lost when it reloaded as a plain Flowsheet."""
    fs = GasNetworkFlowsheet(species_order=["CH4"])
    fs.add_unit(Unit("p", GasPipe(beta=2.0), ["a"], ["b"]))
    assert type(serialize.from_json(serialize.to_json(fs))) is GasNetworkFlowsheet


def test_a_dataclass_named_like_another_is_written_by_path():
    """A bare name found core PipeParams for the gas plugin's."""
    from difflow_gas.units.pipes import PipeParams

    data = serialize.to_value(PipeParams(beta=3.0))
    assert ":" in data["$dataclass"]
    assert type(serialize.from_value(data)) is PipeParams
