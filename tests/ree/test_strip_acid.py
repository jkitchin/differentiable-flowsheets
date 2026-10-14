"""StripperParams: acid_conc and pH, and the strip-acid limit (#384)."""

import pytest

from difflow_ree.units.stripping import StripperParams

ELS = ("Nd", "Dy", "Y")


def test_contradictory_pH_and_acid_raise():
    with pytest.raises(ValueError, match="Give one of them"):
        StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                       pH=0.5, acid_conc=4.0)


def test_stripper_refuses_an_acid_beyond_the_limit():
    with pytest.raises(ValueError, match="max_strip_acid=6 M"):
        StripperParams(n_stages=5, extractant="D2EHPA", elements=ELS,
                       acid_conc=8.0)
