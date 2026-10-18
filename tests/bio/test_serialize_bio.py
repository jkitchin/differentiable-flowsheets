"""Bioreactor flowsheets through difflow.serialize."""

import jax

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, Unit, serialize
from difflow_bio import ContinuousBioreactor, monod_kinetics
from difflow.catalog import _params_class


def test_a_library_growth_law_is_written_by_name():
    """kinetic_fn=monod_kinetics made every bioreactor flowsheet unsaveable."""
    params = _params_class(ContinuousBioreactor)(
        V=10.0, Y_xs=0.5, kinetic_fn=monod_kinetics,
        kinetic_params={"mu_max": 0.5, "K_s": 0.1})
    fs = Flowsheet(species_order=["S", "X"])
    fs.add_unit(Unit("bio", ContinuousBioreactor(params), ["a"], ["b"]))
    back = serialize.from_json(serialize.to_json(fs))
    assert back.units[0].operation.params.kinetic_fn is monod_kinetics
