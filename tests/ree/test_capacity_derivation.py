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
