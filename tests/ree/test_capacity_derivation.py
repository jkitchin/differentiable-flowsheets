"""`typical_K_L` is derived from the correlation, not hand-synced (issue #268).

`EXTRACTANT_CAPACITIES` used to be a second extractant table: per-extractant,
per-element Langmuir constants derived once from `data/extractants.yaml` and
then stored as literals. In the trace limit `q = q_max K_L c` and `q = D c`,
so `K_L = D / q_max` -- a quantity the YAML already determines.

It had drifted. Asked what single pH would reconcile each stored table with the
coefficients it came from, D2EHPA, PC88A and Cyanex272 came back with rms
log10 residuals of 0.62, 0.85 and 0.92 at their best-fitting pH: off by
factors of 4-8 at *any* pH, so not merely evaluated at a different condition.
They described an extractant the database no longer contained.

Nothing failed when that happened. `test_isotherm_exponent_is_monomers_per_ree`
iterates `list_extractants()`, so a *missing* extractant was caught; a *stale*
one was not. This module is the test that would have caught it.
"""

import pytest

from difflow_ree.database import get_extractant, get_extractant_database
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.loading import (
    EXTRACTANT_CAPACITIES,
    get_competitive_K_L,
    get_loading_isotherm,
    typical_K_L,
)


def _reference_driving(record):
    """The condition the record declares its correlation is anchored at."""
    if record.mechanism == "solvating":
        return {"pH": None, "nitrate_conc": record.reference_nitrate}
    return {"pH": record.reference_pH}


class TestTheTableIsDerived:
    """The assertion the issue asks for, which would fail today for 3 of 4."""

    @pytest.mark.parametrize(
        "extractant", ["D2EHPA", "PC88A", "Cyanex272", "TBP"])
    def test_k_l_is_d_at_reference_over_q_max(self, extractant):
        record = get_extractant(extractant)
        stored = EXTRACTANT_CAPACITIES[extractant]["typical_K_L"]
        driving = _reference_driving(record)
        conc = record.reference_concentration
        dist = REEDistribution(
            extractant=extractant, elements=tuple(stored),
            concentration=conc,
            **{k: v for k, v in driving.items() if k != "pH"},
        )
        q_max = conc / record.monomers_per_ree
        for element, K_L in stored.items():
            expected = float(dist.get_D(element, **driving)) / q_max
            assert K_L == pytest.approx(expected, rel=1e-9), element

    def test_every_extractant_is_covered(self):
        """A missing one was already caught; a stale one now cannot exist."""
        names = set(get_extractant_database().list_extractants())
        assert set(EXTRACTANT_CAPACITIES) == names

    def test_every_element_of_the_record_is_covered(self):
        for name in EXTRACTANT_CAPACITIES:
            record = get_extractant(name)
            block = (record.nitrate_coefficients
                     if record.mechanism == "solvating"
                     else record.ph_coefficients)
            assert set(EXTRACTANT_CAPACITIES[name]["typical_K_L"]) == set(block)


class TestItCannotDriftAgain:
    def test_a_new_element_gets_a_constant_with_no_second_edit(self):
        """The drift was a hand-sync step that could be skipped. Now there is none."""
        db = get_extractant_database()
        db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
            temperature_coefficient=-2350.0,
        )
        try:
            fresh = typical_K_L("PC88A")
            assert "Ho" in fresh
            record = get_extractant("PC88A")
            dist = REEDistribution(extractant="PC88A", elements=("Ho",),
                                   concentration=record.reference_concentration)
            q_max = record.reference_concentration / record.monomers_per_ree
            assert fresh["Ho"] == pytest.approx(
                float(dist.get_D("Ho", pH=record.reference_pH)) / q_max,
                rel=1e-9)
        finally:
            db.remove_element_from_extractant("PC88A", "Ho")

    def test_changing_the_reference_moves_the_constants(self):
        """The declared basis is load-bearing, which is why it is declared."""
        record = get_extractant("D2EHPA")
        at_reference = typical_K_L("D2EHPA")["Nd"]
        elsewhere = typical_K_L("D2EHPA", concentration=1.0)["Nd"]
        assert record.reference_concentration != 1.0
        assert at_reference != pytest.approx(elsewhere, rel=1e-6)

    def test_the_literals_are_gone(self):
        """Not a dict of numbers any more."""
        assert not isinstance(EXTRACTANT_CAPACITIES, dict)
        # and it still reads like one, so no call site had to change
        assert set(EXTRACTANT_CAPACITIES["D2EHPA"]) == {"typical_K_L"}
        assert len(EXTRACTANT_CAPACITIES) >= 4


class TestTheReferenceIsDeclared:
    def test_every_cation_exchanger_declares_a_reference_ph(self):
        for name in ("D2EHPA", "PC88A", "Cyanex272"):
            record = get_extractant(name)
            assert record.reference_pH is not None, name
            low, high = record.valid_ph_range
            assert low <= record.reference_pH <= high

    def test_a_solvating_record_declares_a_nitrate_reference_instead(self):
        record = get_extractant("TBP")
        assert record.reference_pH is None
        assert record.reference_nitrate is not None

    def test_a_reference_outside_the_validity_range_is_refused(self):
        import dataclasses
        record = get_extractant("D2EHPA")
        with pytest.raises(ValueError, match="valid_ph_range"):
            dataclasses.replace(record, reference_pH=9.0)

    def test_a_record_with_no_reference_says_so_rather_than_guessing(self):
        import dataclasses
        from difflow_ree.database import Extractant
        record = get_extractant("D2EHPA")
        stripped = dataclasses.replace(record, name="NoRef", reference_pH=None)
        db = get_extractant_database()
        db.add_extractant("NoRef", stripped)
        try:
            with pytest.raises(ValueError, match="reference_pH"):
                typical_K_L("NoRef")
        finally:
            db.remove_extractant("NoRef")


def test_the_isotherm_and_the_competitive_constants_agree():
    isotherm = get_loading_isotherm("D2EHPA", 0.5)
    constants = get_competitive_K_L("D2EHPA")
    assert isotherm.K_L == pytest.approx(
        sum(constants.values()) / len(constants), rel=1e-9)


class TestTheMemoCannotGoStale:
    """The memo must not become the new place the numbers drift.

    `TestItCannotDriftAgain` calls `typical_K_L` directly, which is the one
    path that has no memo in front of it -- so it would pass with a cache
    that never invalidates. `EXTRACTANT_CAPACITIES` is the public name and
    the one every call site uses, so these go through it.

    What makes this a real hazard rather than a theoretical one is that
    `ExtractantDatabase` mutates records IN PLACE:
    `add_element_to_extractant` writes into the record's own
    `ph_coefficients` dict. The record object is the same object afterwards
    and compares equal to itself, so identity and equality both say
    "unchanged" while the basis of every derived constant has moved.
    """

    def test_a_new_element_appears_after_the_table_was_already_read(self):
        """Read first, edit second -- the order a name-keyed memo gets wrong."""
        db = get_extractant_database()
        before = dict(EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"])
        assert "Ho" not in before
        db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
            temperature_coefficient=-2350.0,
        )
        try:
            assert "Ho" in EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]
        finally:
            db.remove_element_from_extractant("PC88A", "Ho")

    def test_a_removed_element_stops_appearing(self):
        db = get_extractant_database()
        db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
            temperature_coefficient=-2350.0,
        )
        assert "Ho" in EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]
        db.remove_element_from_extractant("PC88A", "Ho")
        assert "Ho" not in EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]

    def test_a_changed_coefficient_moves_the_constant(self):
        """`a` is log10(D), so +1 must multiply K_L by exactly ten."""
        import dataclasses

        record = get_extractant("PC88A")
        original = record.ph_coefficients["Nd"]
        before = EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]["Nd"]
        record.ph_coefficients["Nd"] = dataclasses.replace(
            original, a=original.a + 1.0)
        try:
            after = EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]["Nd"]
            assert after / before == pytest.approx(10.0, rel=1e-9)
        finally:
            record.ph_coefficients["Nd"] = original
        assert EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]["Nd"] == (
            pytest.approx(before, rel=1e-12))

    def test_a_changed_reference_moves_every_constant(self):
        record = get_extractant("PC88A")
        original = record.reference_pH
        before = dict(EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"])
        record.reference_pH = original + 0.5
        try:
            after = EXTRACTANT_CAPACITIES["PC88A"]["typical_K_L"]
            assert all(after[el] != pytest.approx(before[el], rel=1e-6)
                       for el in before)
        finally:
            record.reference_pH = original

    def test_it_is_still_a_memo(self):
        """Invalidation must not turn into recomputing on every read.

        `get_loading_isotherm` reads this per stage construction, and
        deriving runs the correlation once per element.
        """
        import difflow_ree.equilibrium.loading as loading

        calls = []
        real = loading.typical_K_L

        def counted(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        EXTRACTANT_CAPACITIES["D2EHPA"]  # prime it
        loading.typical_K_L = counted
        try:
            for _ in range(25):
                EXTRACTANT_CAPACITIES["D2EHPA"]["typical_K_L"]
        finally:
            loading.typical_K_L = real
        assert calls == []

    def test_an_unknown_extractant_still_raises_key_error(self):
        with pytest.raises(KeyError):
            EXTRACTANT_CAPACITIES["definitely_not_an_extractant"]


class TestNaphthenicAcidReproducesTheDeletedLiterals:
    """The derivation is checkable against the table it replaces.

    `naphthenic_acid` landed on main while #268 was open, and it declared
    no `reference_pH` -- so deriving its constants refused outright, which
    is the correct behaviour and a broken merge. Its deleted `typical_K_L`
    literals carried a note saying they had been back-computed as
    `K_L = D(pH 4.5, 0.5 M, 298 K) / (0.5 * 1/3) = 6 D`, which is this
    derivation exactly. Declaring `reference_pH: 4.5` reproduces all
    fifteen of them, which is the strongest evidence available that the
    derived table and the hand table describe the same extractant -- the
    thing that was NOT true of D2EHPA, PC88A or Cyanex272.
    """

    DELETED_LITERALS = {
        "La": 10.02, "Ce": 17.23, "Pr": 23.82, "Nd": 29.56, "Sm": 43.37,
        "Eu": 37.06, "Gd": 29.00, "Tb": 28.93, "Dy": 25.92, "Ho": 21.23,
        "Er": 18.28, "Tm": 17.58, "Yb": 17.72, "Lu": 16.25, "Y": 8.20,
    }

    def test_the_reference_is_declared_and_in_range(self):
        record = get_extractant("naphthenic_acid")
        assert record.reference_pH == 4.5
        low, high = record.valid_ph_range
        assert low <= record.reference_pH <= high

    def test_every_deleted_literal_is_reproduced(self):
        derived = EXTRACTANT_CAPACITIES["naphthenic_acid"]["typical_K_L"]
        assert set(derived) == set(self.DELETED_LITERALS)
        for element, literal in self.DELETED_LITERALS.items():
            assert derived[element] == pytest.approx(literal, rel=2e-3), element

    def test_the_y_selectivity_survives(self):
        """Y lowest is the entire point of this extractant."""
        derived = EXTRACTANT_CAPACITIES["naphthenic_acid"]["typical_K_L"]
        assert derived["Y"] == min(derived.values())
        assert derived["Sm"] == max(derived.values())

