"""VacuumColumn: smoke, gradient, validation and serialization."""

import jax
import numpy as np
import pytest

from difflow_refinery.vacuum import (
    StageSpec,
    VacuumColumn,
    VacuumColumnParams,
    atmospheric_residue,
    characterize,
    heavy_crude,
)

jax.config.update("jax_enable_x64", True)

PRODUCTS = ("overhead", "lvgo", "hvgo", "slop", "residue")


@pytest.fixture(scope="module")
def heavy():
    char = characterize(heavy_crude())
    feed = atmospheric_residue(char, crude_rate_kg_s=100.0)
    return char, feed, VacuumColumnParams(components=char.components)


def test_solve_smoke_and_grad(heavy):
    """One compile: the column converges, outputs are finite, the gradient is finite."""
    char, feed, params = heavy

    def hvgo_rate(T):
        info = VacuumColumn(params.update(furnace_T=T))(feed)[-1]
        return info["outputs"]["hvgo.rate"], info

    (y, info), g = jax.value_and_grad(hvgo_rate, has_aux=True)(673.15)
    assert bool(info["converged"])
    assert np.isfinite(float(y)) and np.isfinite(float(g))


def test_bad_specs_are_rejected(heavy):
    char, feed, params = heavy
    with pytest.raises(ValueError, match="neither a draw rate"):
        VacuumColumn(params.update(specs=(StageSpec("top.T", 340.0, replaces="nope"),)))
    with pytest.raises(ValueError, match="no value and no spec"):
        VacuumColumn(params.update(specs=()))


def test_json_round_trip(heavy):
    import difflow_refinery as dr
    from difflow import serialize
    from difflow.flowsheet import Flowsheet, Unit
    from difflow.plugins import OperationRegistry

    char, feed, params = heavy
    op = VacuumColumn(params)
    reg = OperationRegistry()
    dr.register(reg)
    fs = Flowsheet([k[2:] for k in feed if k.startswith("F_")])
    fs.add_feed("resid", feed)
    fs.add_unit(Unit("vdu", op, ["resid"], [*PRODUCTS, "info"]))
    text = serialize.to_json(fs, registry=reg)
    back = serialize.from_json(text, registry=reg).units[0].operation
    assert isinstance(back, VacuumColumn) and back.params.specs == op.params.specs
