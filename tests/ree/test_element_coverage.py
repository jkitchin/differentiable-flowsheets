"""Coverage gaps are answerable before the run, not during it (issue #269).

Coefficient coverage is uneven across the extractant database, and the gap
matters most for exactly the separation people reach for these extractants to
do. Yttrium purification is Y against Ho, Er, Tm, Yb and Lu: Y(III)'s 90.0 pm
ionic radius sits between Ho's 90.1 and Er's 89.0, which is why the heavies are
what it has to be told apart from. A record whose coefficients stop at Dy plus Y
cannot do that job however many elements it lists.

Before this, a flowsheet naming an uncovered element got a `KeyError` from
mid-solve -- raised while iterating stages, naming one element at a time, with
the flowsheet already half-built.
"""

import pytest

from difflow_ree import Coverage
from difflow_ree.database import get_extractant, get_extractant_database
from difflow_ree.equilibrium.distribution import REEDistribution

#: The five the issue names: what Y has to be separated from.
Y_PURIFICATION_NEIGHBOURS = ("Ho", "Er", "Tm", "Yb", "Lu")


class TestTheGapIsVisibleBeforeARun:
    def test_coverage_reports_every_extractant(self):
        cov = get_extractant_database().coverage()
        assert set(cov.covered) == set(
            get_extractant_database().list_extractants())

    def test_coverage_names_what_is_missing(self):
        """Asked about the heavies, it says which records cannot do them."""
        wanted = ("Dy", "Y") + Y_PURIFICATION_NEIGHBOURS
        cov = get_extractant_database().coverage(wanted)
        for name in ("D2EHPA", "PC88A", "Cyanex272", "TBP"):
            assert cov.missing(name) == Y_PURIFICATION_NEIGHBOURS, name
        assert cov.complete() == ("naphthenic_acid",)

    def test_the_shipped_gap_is_the_one_the_issue_describes(self):
        """Four of five extractants cover 10 of 15, and it is the same five.

        This is the table in #269, asserted rather than described. The five
        missing are not an arbitrary tail: they are what yttrium
        purification runs against, so the records that lack them cannot do
        the separation people reach for them to do.
        """
        cov = get_extractant_database().coverage()
        assert len(cov.elements) == 15
        assert cov.complete() == ("naphthenic_acid",)
        for name in ("D2EHPA", "PC88A", "Cyanex272", "TBP"):
            assert len(cov.covered[name]) == 10, name
            assert cov.missing(name) == Y_PURIFICATION_NEIGHBOURS, name

    def test_the_report_shows_the_gap_at_a_glance(self):
        """What a user runs before a run, rather than finding out during one."""
        text = get_extractant_database().coverage().as_text()
        assert "10/15" in text
        assert "15/15" in text
        assert "Ho, Er, Tm, Yb, Lu" in text

    def test_the_report_reads(self):
        text = get_extractant_database().coverage(
            ("Dy", "Ho")).as_text()
        assert "extractant" in text and "missing" in text
        assert "Ho" in text

    def test_the_record_knows_its_own_coverage(self):
        for name in get_extractant_database().list_extractants():
            record = get_extractant(name)
            assert set(record.covered_elements) == set(
                record.driving_coefficients or {})

    def test_driving_coefficients_follow_the_mechanism(self):
        """TBP is solvating, so its coverage is the nitrate block."""
        tbp = get_extractant("TBP")
        assert tbp.driving_coefficients is tbp.nitrate_coefficients
        assert tbp.ph_coefficients is None
        d2 = get_extractant("D2EHPA")
        assert d2.driving_coefficients is d2.ph_coefficients


class TestTheErrorArrivesEarly:
    def test_construction_refuses_an_uncovered_element(self):
        with pytest.raises(ValueError) as exc:
            REEDistribution(extractant="D2EHPA",
                            elements=("Y",) + Y_PURIFICATION_NEIGHBOURS)
        message = str(exc.value)
        # every missing element at once, not one per solve
        for element in Y_PURIFICATION_NEIGHBOURS:
            assert element in message
        # and the coverage that does exist, beside it
        assert "It covers: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y" in message
        assert "coverage()" in message

    def test_it_says_extending_is_a_refit(self):
        """Not an interpolation: these correlations have no source to extend."""
        with pytest.raises(ValueError, match="refit"):
            REEDistribution(extractant="PC88A", elements=("Ho",))

    def test_a_solvating_record_is_checked_against_its_own_block(self):
        with pytest.raises(ValueError, match="Lu"):
            REEDistribution(extractant="TBP", elements=("Nd", "Lu"),
                            nitrate_conc=3.0)

    def test_a_covered_list_is_untouched(self):
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd", "Pr"))
        assert float(dist.get_D("Nd", pH=3.0)) > 0

    def test_an_added_element_becomes_usable(self):
        """The escape the message points at actually works."""
        db = get_extractant_database()
        db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
            temperature_coefficient=-2350.0,
        )
        try:
            assert "Ho" in get_extractant("PC88A").covered_elements
            dist = REEDistribution(extractant="PC88A", elements=("Ho", "Dy"))
            assert float(dist.get_D("Ho", pH=3.5)) > 0
        finally:
            db.remove_element_from_extractant("PC88A", "Ho")


def test_coverage_is_a_plain_report():
    cov = Coverage(elements=("A", "B"), covered={"X": ("A",)})
    assert cov.missing("X") == ("B",)
    assert cov.complete() == ()
