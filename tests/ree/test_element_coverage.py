"""Coefficient coverage is reported before a run (#269)."""

import pytest

from difflow_ree.database import get_extractant_database
from difflow_ree.equilibrium.distribution import REEDistribution


def test_coverage_reports_every_extractant():
    cov = get_extractant_database().coverage()
    assert set(cov.covered) == set(get_extractant_database().list_extractants())
    assert "missing" in cov.as_text()


def test_construction_refuses_an_uncovered_element():
    with pytest.raises(ValueError, match="Ho"):
        REEDistribution(extractant="D2EHPA", elements=("Y", "Ho"))
