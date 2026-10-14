"""pH outside a record's ``valid_ph_range`` is reported, not clamped (#262)."""

import pytest

from difflow_ree.equilibrium.distribution import REEDistribution


def test_outside_the_window_warns_or_raises():
    dist = REEDistribution(extractant="PC88A", elements=("Nd",))
    with pytest.warns(UserWarning, match="validity range"):
        dist.get_D("Nd", pH=6.0)
    strict = REEDistribution(extractant="PC88A", elements=("Nd",),
                             on_out_of_range="raise")
    with pytest.raises(ValueError, match="validity range"):
        strict.get_D("Nd", pH=0.05)
