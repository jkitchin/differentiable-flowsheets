"""The strip acid sets the strip pH (2026 operating-point audit, R9).

``StripperParams.acid_conc`` defaulted to 4 M and was reported in ``info``
but never read: the strip pH was the record's window bottom whatever acid was
named, so 0.01 M, 4 M and 8 M acid stripped identically.
"""

import math
import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.units.stripping import REEStripper, StripperParams

ELS = ("Nd", "Dy", "Y")


def _strip(**kw):
    org = make_stream({"kerosene": 10.0, "D2EHPA": 5.0, "Nd": 0.05, "Dy": 0.05,
                       "Y": 0.05}, 298.15, 101325.0)
    acid = make_stream({"H2O": 5.0}, 298.15, 101325.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # extrapolation report
        product, _, info = REEStripper(StripperParams(
            n_stages=5, extractant="D2EHPA", elements=ELS, **kw))(org, acid)
    return get_flows(product), info


def test_acid_conc_sets_the_pH():
    assert StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                          acid_conc=2.0).pH == pytest.approx(-math.log10(2.0))
    assert StripperParams(n_stages=5, extractant="D2EHPA",
                          elements=ELS).pH == pytest.approx(-math.log10(4.0))


def test_stronger_acid_strips_more():
    weak, _ = _strip(acid_conc=0.01)
    strong, info = _strip(acid_conc=8.0)
    assert float(info["acid_conc"]) == pytest.approx(8.0)
    assert float(strong["Y"]) > float(weak["Y"])
    assert float(strong["Dy"]) > 10 * float(weak["Dy"])


def test_pH_alone_reports_the_implied_acid():
    p = StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS, pH=0.5)
    assert p.acid_conc == pytest.approx(10 ** -0.5)


def test_contradictory_pH_and_acid_raise():
    with pytest.raises(ValueError, match="Give one of them"):
        StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                       pH=0.5, acid_conc=4.0)
