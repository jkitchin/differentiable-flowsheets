"""Refinery thermochemistry objects through serialize and codegen."""

import jax

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, Unit, codegen, serialize
from difflow.catalog import _params_class
from difflow.units.gibbs import GibbsReactor
from difflow_refinery.thermochemistry import IdealGasSet


def test_a_gibbs_reactor_on_an_ideal_gas_set_round_trips():
    """IdealGasSet keeps its constructor arguments by name, so it is
    rebuilt from them: as JSON and as source."""
    gas = IdealGasSet(("hydrogen", "benzene", "cyclohexane"))
    fs = Flowsheet(species_order=list(gas.names))
    fs.add_unit(Unit("rx", GibbsReactor(_params_class(GibbsReactor)(thermo=gas)),
                     ["a"], ["b"]))
    text = serialize.to_json(fs)
    assert serialize.from_json(text).units[0].operation.params.thermo.names == gas.names
    namespace = {}
    exec(codegen.to_python(fs), namespace)
    assert serialize.to_json(namespace["fs"]) == text
