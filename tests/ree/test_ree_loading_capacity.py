"""Phase-flow and capacity input validation for the REE extractor (#192, #193)."""

import pytest

from difflow.streams import make_stream
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams


def _run(feed_flows, solvent_flows, **kw):
    params = REEExtractorParams(n_stages=3, extractant="D2EHPA",
                                elements=("Nd",), pH=1.0, **kw)
    feed = make_stream(feed_flows, T=298.15, P=101325.0)
    solvent = make_stream(solvent_flows, T=298.15, P=101325.0)
    return REEExtractor(params)(feed, solvent)


def test_solvent_without_an_organic_carrier_raises():
    with pytest.raises(ValueError, match="no organic phase"):
        _run({"H2O": 10.0, "Nd": 0.1}, {"Organic": 10.0, "Nd": 0.0})


def test_an_empty_aqueous_feed_raises():
    with pytest.raises(ValueError, match="total flow is"):
        _run({"H2O": 0.0, "Nd": 0.0}, {"D2EHPA": 1.0, "kerosene": 5.0})


def test_capacity_needs_the_extractant_flow():
    with pytest.raises(ValueError, match="does not declare a flow"):
        _run({"H2O": 10.0, "Nd": 0.1}, {"kerosene": 5.0}, include_loading=True)
