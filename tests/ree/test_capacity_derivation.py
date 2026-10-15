"""typical_K_L derives from the correlation's declared reference (#268)."""

import dataclasses
import math

import pytest

from difflow_ree.database import get_extractant, get_extractant_database
from difflow_ree.equilibrium.loading import typical_K_L


def test_typical_K_L_smoke():
    K = typical_K_L("D2EHPA")
    assert K and all(math.isfinite(float(v)) for v in K.values())
    with pytest.raises(KeyError):
        typical_K_L("NoSuchExtractant")


def test_a_reference_outside_the_validity_range_is_refused():
    with pytest.raises(ValueError, match="valid_ph_range"):
        dataclasses.replace(get_extractant("D2EHPA"), reference_pH=9.0)


def test_a_record_with_no_reference_says_so():
    stripped = dataclasses.replace(get_extractant("D2EHPA"), name="NoRef",
                                   reference_pH=None)
    db = get_extractant_database()
    db.add_extractant("NoRef", stripped)
    try:
        with pytest.raises(ValueError, match="reference_pH"):
            typical_K_L("NoRef")
    finally:
        db.remove_extractant("NoRef")
