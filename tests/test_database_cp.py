"""Core-database heat capacities: liquid Cp and ideal-gas Cp are different fields (#393)."""

import pytest

from difflow import IdealThermo, PengRobinson
from difflow.database import get_critical_props, get_species_data
from difflow.thermo import CubicThermo, SpeciesData


def test_cubic_thermo_does_not_use_the_liquid_cp():
    """Regression: CubicThermo used water's liquid 75.3 as its ideal-gas Cp."""
    ideal = IdealThermo({"water": get_species_data("water")})
    thermo = CubicThermo(ideal, PengRobinson({"water": get_critical_props("water")}))
    assert float(thermo.Cp_mix({"water": 1.0}, 300.0)) < 50.0


def test_falls_back_to_cp_coeffs_when_no_ig_field():
    """User/importer data that stores the ideal-gas Cp in Cp_coeffs keeps working."""
    sd = SpeciesData("X", 10.0, (30.0, 0.01, 0.0, 0.0), (1e4, 0.38, 500.0), (9.0, 1000.0, -50.0))
    assert float(IdealThermo({"X": sd}).Cp_ig("X", 300.0)) == pytest.approx(33.0)
