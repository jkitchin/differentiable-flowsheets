"""Smoke tests for the TEMA design fouling resistances (difflow.fouling)."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.fouling import (
    tema_crude_oil_fouling,
    tema_fouling_resistance,
    tema_water_fouling,
)


def test_lookups_smoke():
    assert float(tema_fouling_resistance("kerosene")) > 0.0
    assert float(tema_crude_oil_fouling(400.0, 1.0)) > 0.0
    assert float(tema_water_fouling("sea_water", 300.0, 0.5)) > 0.0


def test_bad_inputs():
    with pytest.raises(KeyError, match="Unknown service"):
        tema_fouling_resistance("crude")
    with pytest.raises(ValueError):
        tema_fouling_resistance("kerosene", "max")
    with pytest.raises(KeyError, match="Unknown water"):
        tema_water_fouling("lake", 300.0, 1.0)
