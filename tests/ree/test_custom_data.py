"""Tests for custom element, extractant coefficient, and separation factor APIs.

These tests verify that users can add their own literature data for elements
not in the built-in database (e.g., Ho) and use them in simulations.
"""

import pytest

from difflow_ree.database import (
    REEDatabase,
    REEElement,
    ExtractantDatabase,
    SeparationFactorDatabase,
    PHCoefficients,
    create_custom_element,
    create_custom_extractant,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ree_db():
    """Fresh REE database for each test."""
    return REEDatabase()


@pytest.fixture
def ext_db():
    """Fresh extractant database for each test."""
    return ExtractantDatabase()


@pytest.fixture
def sf_db():
    """Fresh separation factor database for each test."""
    return SeparationFactorDatabase()


@pytest.fixture
def ho_element():
    """Holmium element for testing (real physical properties)."""
    return create_custom_element(
        symbol="Ho",
        name="Holmium",
        atomic_number=67,
        atomic_weight=164.930,
        ionic_radius_pm=90.1,
        density=8.795,
        melting_point=1734,
        group="heavy",
        oxide_formula="Ho2O3",
        oxide_mw=377.86,
        price_usd_kg=60.0,
    )


# =============================================================================
# Tests: create_custom_element
# =============================================================================


class TestCreateCustomElement:

    def test_creates_valid_element(self, ho_element):
        assert ho_element.symbol == "Ho"
        assert ho_element.name == "Holmium"
        assert ho_element.atomic_number == 67
        assert ho_element.atomic_weight == 164.930
        assert ho_element.ionic_radius_pm == 90.1
        assert ho_element.group == "heavy"
        assert ho_element.oxidation_states == (3,)

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            create_custom_element(
                symbol="", name="X", atomic_number=1, atomic_weight=1.0,
                ionic_radius_pm=50.0, density=1.0, melting_point=300,
                group="light", oxide_formula="X2O3", oxide_mw=100.0,
            )

    def test_invalid_group_raises(self):
        with pytest.raises(ValueError, match="group must be"):
            create_custom_element(
                symbol="X", name="X", atomic_number=1, atomic_weight=1.0,
                ionic_radius_pm=50.0, density=1.0, melting_point=300,
                group="superheavy", oxide_formula="X2O3", oxide_mw=100.0,
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="atomic_weight must be positive"):
            create_custom_element(
                symbol="X", name="X", atomic_number=1, atomic_weight=-1.0,
                ionic_radius_pm=50.0, density=1.0, melting_point=300,
                group="light", oxide_formula="X2O3", oxide_mw=100.0,
            )

    def test_custom_oxidation_states(self):
        elem = create_custom_element(
            symbol="X", name="X", atomic_number=1, atomic_weight=1.0,
            ionic_radius_pm=50.0, density=1.0, melting_point=300,
            group="light", oxide_formula="X2O3", oxide_mw=100.0,
            oxidation_states=(2, 3, 4),
        )
        assert elem.oxidation_states == (2, 3, 4)


# =============================================================================
# Tests: REEDatabase.add_element / remove_element / update_element
# =============================================================================


class TestREEDatabaseAddElement:

    def test_add_and_retrieve(self, ree_db, ho_element):
        ree_db.add_element("Ho", ho_element)
        retrieved = ree_db.get("Ho")
        assert retrieved.name == "Holmium"
        assert retrieved.atomic_weight == 164.930

    def test_add_appears_in_list(self, ree_db, ho_element):
        assert "Ho" not in ree_db.list_elements()
        ree_db.add_element("Ho", ho_element)
        assert "Ho" in ree_db.list_elements()

    def test_add_updates_group(self, ree_db, ho_element):
        ree_db.add_element("Ho", ho_element)
        heavy = ree_db.list_by_group("heavy")
        assert "Ho" in heavy

    def test_add_duplicate_raises(self, ree_db, ho_element):
        ree_db.add_element("Ho", ho_element)
        with pytest.raises(ValueError, match="already exists"):
            ree_db.add_element("Ho", ho_element)

    def test_add_wrong_type_raises(self, ree_db):
        with pytest.raises(TypeError, match="REEElement instance"):
            ree_db.add_element("Ho", {"name": "Holmium"})

    def test_remove_element(self, ree_db, ho_element):
        ree_db.add_element("Ho", ho_element)
        ree_db.remove_element("Ho")
        assert "Ho" not in ree_db.list_elements()
        assert "Ho" not in ree_db.list_by_group("heavy")

    def test_remove_nonexistent_raises(self, ree_db):
        with pytest.raises(KeyError, match="not found"):
            ree_db.remove_element("Ho")

    def test_update_element(self, ree_db, ho_element):
        ree_db.add_element("Ho", ho_element)
        updated = create_custom_element(
            symbol="Ho", name="Holmium", atomic_number=67,
            atomic_weight=164.930, ionic_radius_pm=90.1, density=8.795,
            melting_point=1734, group="heavy", oxide_formula="Ho2O3",
            oxide_mw=377.86, price_usd_kg=75.0,  # Updated price
        )
        ree_db.update_element("Ho", updated)
        assert ree_db.get("Ho").price_usd_kg == 75.0

    def test_update_nonexistent_raises(self, ree_db, ho_element):
        with pytest.raises(KeyError, match="not found"):
            ree_db.update_element("Ho", ho_element)

    def test_update_changes_group(self, ree_db, ho_element):
        ree_db.add_element("Ho", ho_element)
        assert "Ho" in ree_db.list_by_group("heavy")

        reclassified = create_custom_element(
            symbol="Ho", name="Holmium", atomic_number=67,
            atomic_weight=164.930, ionic_radius_pm=90.1, density=8.795,
            melting_point=1734, group="middle", oxide_formula="Ho2O3",
            oxide_mw=377.86,
        )
        ree_db.update_element("Ho", reclassified)
        assert "Ho" not in ree_db.list_by_group("heavy")
        assert "Ho" in ree_db.list_by_group("middle")


# =============================================================================
# Tests: ExtractantDatabase.add_element_to_extractant
# =============================================================================


class TestAddElementToExtractant:

    def test_add_element_coefficients(self, ext_db):
        ext_db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.42, "b": 2.86, "c": 0.010},
            temperature_coefficient=-2250,
        )
        extractant = ext_db.get("PC88A")
        assert "Ho" in extractant.ph_coefficients
        assert extractant.ph_coefficients["Ho"].a == -6.42
        assert extractant.ph_coefficients["Ho"].b == 2.86
        assert extractant.temperature_coefficients["Ho"] == -2250

    def test_optional_d_coefficient(self, ext_db):
        ext_db.add_element_to_extractant(
            "D2EHPA", "Ho",
            ph_coefficients={"a": -6.7, "b": 2.76, "c": 0.01, "d": -0.5},
            temperature_coefficient=-2350,
        )
        assert ext_db.get("D2EHPA").ph_coefficients["Ho"].d == -0.5

    def test_missing_extractant_raises(self, ext_db):
        with pytest.raises(KeyError, match="not found"):
            ext_db.add_element_to_extractant(
                "NoSuchExtractant", "Ho",
                ph_coefficients={"a": 0, "b": 0, "c": 0},
                temperature_coefficient=0,
            )

    def test_duplicate_element_raises(self, ext_db):
        """La already exists in all extractants from YAML."""
        with pytest.raises(ValueError, match="already has coefficients"):
            ext_db.add_element_to_extractant(
                "D2EHPA", "La",
                ph_coefficients={"a": 0, "b": 0, "c": 0},
                temperature_coefficient=0,
            )

    def test_missing_coefficient_keys_raises(self, ext_db):
        with pytest.raises(ValueError, match="missing required keys"):
            ext_db.add_element_to_extractant(
                "D2EHPA", "Ho",
                ph_coefficients={"a": 0, "b": 0},  # missing "c"
                temperature_coefficient=0,
            )

    def test_remove_element_from_extractant(self, ext_db):
        ext_db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.42, "b": 2.86, "c": 0.01},
            temperature_coefficient=-2250,
        )
        ext_db.remove_element_from_extractant("PC88A", "Ho")
        assert "Ho" not in ext_db.get("PC88A").ph_coefficients

    def test_remove_nonexistent_element_raises(self, ext_db):
        with pytest.raises(KeyError, match="not found"):
            ext_db.remove_element_from_extractant("PC88A", "Ho")

    def test_adding_a_ph_element_to_tbp_raises(self, ext_db):
        """TBP has NO ph_coefficients block -- it was deleted as
        mechanistically unsupported (neutral extractant, pKa None, zero protons
        released). Adding a pH coefficient here would recreate exactly that
        model, so it must raise rather than lazily create the block."""
        assert ext_db.get("TBP").ph_coefficients is None
        with pytest.raises(ValueError, match="no 'ph_coefficients' block"):
            ext_db.add_element_to_extractant(
                "TBP", "Ho",
                ph_coefficients={"a": -6.42, "b": 2.86, "c": 0.01},
                temperature_coefficient=2300,
            )

    def test_removing_a_ph_element_from_tbp_raises_keyerror(self, ext_db):
        """The None block must not leak a TypeError out of `in`."""
        with pytest.raises(KeyError, match="not found"):
            ext_db.remove_element_from_extractant("TBP", "Nd")


# =============================================================================
# Tests: SeparationFactorDatabase.add_pair / add_separation_factors
# =============================================================================


class TestSeparationFactorDatabase:

    def test_add_pair_adjacent(self, sf_db):
        sf_db.add_pair("D2EHPA", "Ho_Dy", 1.4)
        assert sf_db.get_sf("D2EHPA", "Ho_Dy") == 1.4

    def test_add_pair_group(self, sf_db):
        sf_db.add_pair("D2EHPA", "Ho_Nd", 10.0, adjacent=False)
        assert sf_db.get_sf("D2EHPA", "Ho_Nd") == 10.0

    def test_add_pair_with_stages(self, sf_db):
        sf_db.add_pair("D2EHPA", "Ho_Dy", 1.4, stages_99=22)
        assert sf_db.get_stages_needed("D2EHPA", "Ho_Dy") == 22

    def test_add_pair_missing_extractant_raises(self, sf_db):
        with pytest.raises(KeyError, match="No SF data"):
            sf_db.add_pair("NoSuch", "Ho_Dy", 1.4)

    def test_add_duplicate_pair_raises(self, sf_db):
        with pytest.raises(ValueError, match="already exists"):
            sf_db.add_pair("D2EHPA", "Ce_La", 999.0)

    def test_remove_pair(self, sf_db):
        sf_db.add_pair("D2EHPA", "Ho_Dy", 1.4)
        sf_db.remove_pair("D2EHPA", "Ho_Dy")
        with pytest.raises(KeyError):
            sf_db.get_sf("D2EHPA", "Ho_Dy")

    def test_add_separation_factors_new_extractant(self, sf_db):
        sf_db.add_separation_factors(
            extractant="MyExtractant",
            conditions={"pH": 3.0, "temperature_K": 298},
            adjacent_pairs={"Ho_Dy": 1.4, "Y_Ho": 0.9},
            group_pairs={"Ho_La": 50.0},
        )
        assert sf_db.get_sf("MyExtractant", "Ho_Dy") == 1.4
        assert sf_db.get_sf("MyExtractant", "Ho_La") == 50.0
        assert "MyExtractant" in sf_db.list_extractants()

    def test_add_separation_factors_duplicate_raises(self, sf_db):
        with pytest.raises(ValueError, match="already exists"):
            sf_db.add_separation_factors(
                extractant="D2EHPA",
                conditions={"pH": 3.0},
            )

    def test_remove_separation_factors(self, sf_db):
        sf_db.add_separation_factors(
            extractant="Temp",
            conditions={"pH": 3.0},
            adjacent_pairs={"A_B": 1.5},
        )
        sf_db.remove_separation_factors("Temp")
        assert "Temp" not in sf_db.list_extractants()

    def test_remove_nonexistent_raises(self, sf_db):
        with pytest.raises(KeyError):
            sf_db.remove_separation_factors("NoSuch")


# =============================================================================
# Integration test: full workflow for adding Ho with PC88A
# =============================================================================


class TestIntegrationAddHolmium:
    """Demonstrate the intended workflow from issue #160."""

    def test_add_ho_to_pc88a_workflow(self, ree_db, ext_db, sf_db):
        # Step 1: Add Ho to element database
        ho = create_custom_element(
            symbol="Ho",
            name="Holmium",
            atomic_number=67,
            atomic_weight=164.930,
            ionic_radius_pm=90.1,
            density=8.795,
            melting_point=1734,
            group="heavy",
            oxide_formula="Ho2O3",
            oxide_mw=377.86,
            price_usd_kg=60.0,
        )
        ree_db.add_element("Ho", ho)
        assert ree_db.get("Ho").name == "Holmium"

        # Step 2: Add Ho coefficients to PC88A only
        # (These would come from the user's literature data)
        ext_db.add_element_to_extractant(
            "PC88A", "Ho",
            ph_coefficients={"a": -6.42, "b": 2.86, "c": 0.010},
            temperature_coefficient=-2250,
        )
        assert "Ho" in ext_db.get("PC88A").ph_coefficients
        # Other extractants should NOT have Ho
        assert "Ho" not in ext_db.get("D2EHPA").ph_coefficients

        # Step 3: Add separation factors for Ho with Gd and Y only
        sf_db.add_pair("PC88A", "Ho_Dy", 1.4, stages_99=20)
        sf_db.add_pair("PC88A", "Y_Ho", 0.9)
        assert sf_db.get_sf("PC88A", "Ho_Dy") == 1.4
        assert sf_db.get_sf("PC88A", "Y_Ho") == 0.9
