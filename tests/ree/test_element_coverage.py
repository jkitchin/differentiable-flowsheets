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

import dataclasses

import pytest

from difflow_ree import Coverage
from difflow_ree.database import (
    EXTRACTION_MECHANISMS,
    MECHANISM_COEFFICIENT_BLOCKS,
    get_extractant,
    get_extractant_database,
)
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


class TestTheDispatchIsSharedWithTheSolve:
    """Coverage and the solve must read the *same* block (#269).

    The point of a coverage check at construction is that it predicts what
    the solve will do. It only does that if both resolve mechanism -> block
    the same way, which is why the dispatch is one table rather than an
    `if` in each place. Two ways to get it wrong, both regressions here:
    reading the record's mechanism when the caller overrode it, and
    forgetting that `counter_ion_exchange` (#266) has a block of its own.
    """

    @staticmethod
    def _two_block_record():
        """A record carrying both a pH and a counter-ion block.

        Nothing shipped carries both, so an override cannot change the
        answer for any record in the database today. It can for a custom
        one, which is the whole point of `add_extractant`.
        """
        base = get_extractant("D2EHPA")
        return dataclasses.replace(
            base,
            name="two_block",
            counter_ion="NH4",
            counter_ion_coefficients={"Nd": base.ph_coefficients["Nd"]},
            counter_ion_exponent=3.0,
        )

    def test_the_table_is_total_over_the_mechanisms(self):
        """A mechanism left out would fall through to ph_coefficients."""
        assert set(MECHANISM_COEFFICIENT_BLOCKS) == set(EXTRACTION_MECHANISMS)
        for mechanism, block in MECHANISM_COEFFICIENT_BLOCKS.items():
            assert hasattr(get_extractant("D2EHPA"), block), mechanism

    def test_a_counter_ion_record_covers_its_counter_ion_block(self):
        """Not its pH block, which it does not read."""
        record = self._two_block_record()
        assert record.mechanism == "cation_exchange"
        assert record.covered_elements == tuple(record.ph_coefficients)

        exchanging = dataclasses.replace(
            record, mechanism="counter_ion_exchange")
        assert exchanging.covered_elements == ("Nd",)
        assert (exchanging.driving_coefficients
                is exchanging.counter_ion_coefficients)

    def test_an_override_is_checked_against_the_block_it_selects(self):
        """The active mechanism decides coverage, not the record's."""
        db = get_extractant_database()
        db.add_extractant("two_block", self._two_block_record())
        try:
            # On the record's own mechanism La is covered and runs.
            assert float(REEDistribution(extractant="two_block",
                                         elements=("La",)).get_D("La", pH=3.0)) > 0
            # Overridden onto counter-ion exchange it is not, and saying so
            # at construction is the whole point -- before the fix this
            # passed the check and then raised a KeyError mid-solve.
            with pytest.raises(ValueError, match="La"):
                REEDistribution(extractant="two_block", elements=("La", "Nd"),
                                mechanism="counter_ion_exchange")
        finally:
            db.remove_extractant("two_block")

    @pytest.mark.parametrize("mechanism", EXTRACTION_MECHANISMS)
    def test_what_the_check_allows_is_what_the_solve_can_read(self, mechanism):
        """For every mechanism: coverage passes iff `_coefficients` works."""
        db = get_extractant_database()
        tbp = get_extractant("TBP")
        db.add_extractant("two_block", dataclasses.replace(
            self._two_block_record(),
            nitrate_coefficients={"Ce": tbp.nitrate_coefficients["Ce"]},
            reference_nitrate=tbp.reference_nitrate,   # #195 wants the basis
        ))
        try:
            for element in ("La", "Nd", "Ce"):
                try:
                    dist = REEDistribution(extractant="two_block",
                                           elements=(element,),
                                           mechanism=mechanism,
                                           nitrate_conc=3.0)
                except ValueError as refusal:
                    assert "has no coefficients for" in str(refusal), refusal
                    covered = False
                else:
                    covered = True
                    # Allowed through, so the block really does carry it.
                    assert dist._coefficients(element) is not None
                block = getattr(get_extractant("two_block"),
                                MECHANISM_COEFFICIENT_BLOCKS[mechanism]) or {}
                assert covered == (element in block), (mechanism, element)
        finally:
            db.remove_extractant("two_block")
