"""Validation of ``coefficient_overrides`` and the eager ``n_stages`` guard."""

import pytest

from difflow_ree import REEExtractorParams
from difflow_ree.equilibrium.distribution import REEDistribution


def test_an_override_changes_D():
    base = REEDistribution(extractant="D2EHPA", elements=("Nd",))
    d = REEDistribution(extractant="D2EHPA", elements=("Nd",),
                        coefficient_overrides={"Nd": {"a": -5.0}})
    assert d._coefficients("Nd").a == -5.0
    assert float(d.get_D("Nd", pH=1.5)) != float(base.get_D("Nd", pH=1.5))


def test_an_unknown_element_is_named_in_the_error():
    with pytest.raises(ValueError, match="names element 'La'"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",),
                        coefficient_overrides={"La": {"a": 1.0}})


def test_an_unknown_coefficient_is_named_in_the_error():
    with pytest.raises(ValueError, match=r"carries \['z'\]"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",),
                        coefficient_overrides={"Nd": {"z": 1.0}})


def test_the_eager_stage_guard_fires():
    with pytest.raises(ValueError, match="n_stages must be >= 1"):
        REEExtractorParams(n_stages=0.5, extractant="D2EHPA", elements=("Nd",))
