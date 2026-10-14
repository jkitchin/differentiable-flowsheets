"""Custom element, extractant-coefficient and separation-factor APIs."""

import pytest

from difflow_ree.database import (
    ExtractantDatabase,
    REEDatabase,
    SeparationFactorDatabase,
    create_custom_element,
)

PM = dict(symbol="Pm", name="Promethium", atomic_number=61, atomic_weight=145.0,
          ionic_radius_pm=97.0, density=7.26, melting_point=1042, group="light",
          oxide_formula="Pm2O3", oxide_mw=338.00)


def test_create_custom_element_validates():
    with pytest.raises(ValueError, match="group must be"):
        create_custom_element(**{**PM, "group": "superheavy"})


def test_add_update_remove_element():
    db = REEDatabase()
    db.add_element("Pm", create_custom_element(**PM))
    assert "Pm" in db.list_by_group("light")
    with pytest.raises(ValueError, match="already exists"):
        db.add_element("Pm", create_custom_element(**PM))
    db.update_element("Pm", create_custom_element(**PM, price_usd_kg=75.0))
    assert db.get("Pm").price_usd_kg == 75.0
    db.remove_element("Pm")
    assert "Pm" not in db.list_elements()
    with pytest.raises(KeyError, match="not found"):
        db.remove_element("Pm")


def test_add_and_remove_extractant_coefficients():
    db = ExtractantDatabase()
    db.add_element_to_extractant(
        "PC88A", "Ho", ph_coefficients={"a": -6.42, "b": 2.86, "c": 0.01},
        temperature_coefficient=-2250)
    assert db.get("PC88A").ph_coefficients["Ho"].a == -6.42
    db.remove_element_from_extractant("PC88A", "Ho")
    assert "Ho" not in db.get("PC88A").ph_coefficients
    with pytest.raises(ValueError, match="already has coefficients"):
        db.add_element_to_extractant(
            "D2EHPA", "La", ph_coefficients={"a": 0, "b": 0, "c": 0},
            temperature_coefficient=0)
    with pytest.raises(ValueError, match="missing required keys"):
        db.add_element_to_extractant(
            "D2EHPA", "Ho", ph_coefficients={"a": 0, "b": 0},
            temperature_coefficient=0)
    with pytest.raises(ValueError, match="no 'ph_coefficients' block"):
        db.add_element_to_extractant(
            "TBP", "Ho", ph_coefficients={"a": 0, "b": 0, "c": 0},
            temperature_coefficient=0)


def test_add_and_remove_separation_factors():
    db = SeparationFactorDatabase()
    db.add_pair("D2EHPA", "Ho_Dy", 1.4)
    assert db.get_sf("D2EHPA", "Ho_Dy") == 1.4
    db.remove_pair("D2EHPA", "Ho_Dy")
    with pytest.raises(KeyError):
        db.get_sf("D2EHPA", "Ho_Dy")
    with pytest.raises(ValueError, match="already exists"):
        db.add_pair("D2EHPA", "Ce_La", 999.0)
    db.add_separation_factors(extractant="Mine", conditions={"pH": 3.0},
                              adjacent_pairs={"A_B": 1.5})
    assert db.get_sf("Mine", "A_B") == 1.5
    db.remove_separation_factors("Mine")
    assert "Mine" not in db.list_extractants()
