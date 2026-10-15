"""ScrubberParams validation: target_elements labels, scrub_type deprecation (#288)."""

import pytest

from difflow.streams import make_stream
from difflow_ree import REEScrubber, ScrubberParams, ScrubTypeDeprecationWarning

ELEMENTS = ("La", "Nd")


def test_target_elements_are_validated():
    with pytest.raises(ValueError, match="not in elements"):
        ScrubberParams(n_stages=5, extractant="D2EHPA", elements=ELEMENTS,
                       target_elements=("Y",))
    with pytest.raises(TypeError, match="tuple of element names"):
        ScrubberParams(n_stages=5, extractant="D2EHPA", elements=ELEMENTS,
                       target_elements="Nd")


def test_scrub_type_is_deprecated():
    with pytest.warns(ScrubTypeDeprecationWarning, match="deprecated"):
        ScrubberParams(n_stages=5, extractant="D2EHPA", elements=ELEMENTS,
                       scrub_type="ree")
    with pytest.raises(ValueError, match="not one of"):
        ScrubberParams(n_stages=5, extractant="D2EHPA", elements=ELEMENTS,
                       scrub_type="brine")


def test_an_untracked_ree_is_reported_as_dropped():
    org = make_stream({"D2EHPA": 0.5, "kerosene": 1.0, "Nd": 0.10, "Dy": 0.10},
                      298.15, 101325.0)
    scrub = make_stream({"H2O": 1.0}, 298.15, 101325.0)
    _, _, info = REEScrubber(ScrubberParams(
        n_stages=5, extractant="D2EHPA", elements=("Nd",), pH=1.0))(org, scrub)
    assert info["dropped_species"] == ("Dy",)
