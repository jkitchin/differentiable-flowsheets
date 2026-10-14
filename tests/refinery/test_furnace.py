"""Configuration checks for the furnace solved with the crude column.

The furnace solve itself is exercised by ``test_unit.py`` (``CrudeUnit`` adds one).
"""

import pytest

from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo


@pytest.fixture(scope="module")
def thermo():
    crude = characterize(Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85))
    return ColumnThermo.from_characterization(crude)


def _small(thermo, specs, **over):
    kw = dict(n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
              P_condenser=1.2e5, specs=specs, furnace=cc.Furnace())
    kw.update(over)
    return cc.CrudeColumn(cc.CrudeColumnParams(**kw), thermo)


def test_furnace_adds_a_freedom(thermo):
    with pytest.raises(ValueError, match="1 furnace"):
        _small(thermo, (cc.product_rate("naphtha", 0.25, basis="mole"),))


def test_furnace_spec_needs_a_furnace(thermo):
    with pytest.raises(ValueError, match="needs a Furnace"):
        _small(thermo, (cc.coil_outlet_temperature(620.0),), furnace=None)
