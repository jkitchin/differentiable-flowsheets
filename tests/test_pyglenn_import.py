"""Tests for the NASA Glenn (pyglenn) thermo-data import adapter.

pyglenn is an optional dependency and is not installed in CI, so these tests
drive the adapter with a FakeGlennCalculator that implements the documented
pyglenn API (get_available_species / calculate_properties / connect / close /
context manager). Each fake species has an exact cubic Cp(T) so the cubic fit
is exact and can be checked end-to-end through IdealThermo.
"""

import warnings

import numpy as np
import pytest

from difflow.thermo import IdealThermo
from difflow.pyglenn_import import (
    import_species_data,
    list_available_species,
    fit_cp_coeffs,
)


# Exact cubic Cp coefficients (a, b, c, d) per fake species id: the fit must
# recover the Cp curve these produce.
_CP = {
    1: (25.0, 1.3e-2, -3.0e-6, 1.0e-9),   # O2-like
    2: (22.0, 6.0e-2, -4.0e-5, 1.0e-8),   # CO2-like
    4: (33.0, 8.0e-3, -1.0e-6, 3.0e-10),  # H2O(gas)-like (id 4)
}


class FakeGlennCalculator:
    """Minimal stand-in for pyglenn.ThermochemicalCalculator."""

    _DB = {
        "O2": {
            "id": 1, "name": "O2", "formula": "O2", "phase": "G",
            "molecular_weight": 31.998, "heat_of_formation_298K": 0.0,
        },
        "CO2": {
            "id": 2, "name": "CO2", "formula": "CO2", "phase": "G",
            "molecular_weight": 44.009, "heat_of_formation_298K": -393510.0,
        },
        # A condensed-phase record to exercise phase preference.
        "H2O(cr)": {
            "id": 3, "name": "H2O", "formula": "H2O", "phase": "CR",
            "molecular_weight": 18.015, "heat_of_formation_298K": -285830.0,
        },
        "H2O": {
            "id": 4, "name": "H2O", "formula": "H2O", "phase": "G",
            "molecular_weight": 18.015, "heat_of_formation_298K": -241826.0,
        },
    }

    def __init__(self):
        self._connected = False

    def connect(self):
        self._connected = True
        return True

    def close(self):
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def get_available_species(self, query):
        q = query.lower()
        recs = [
            r for r in self._DB.values()
            if q in r["name"].lower() or q in r["formula"].lower()
        ]
        return recs

    def calculate_properties(self, species_id, temperature):
        if species_id not in _CP:
            # Emulate pyglenn raising for an id with no coefficients.
            raise KeyError(species_id)
        a, b, c, d = _CP[species_id]
        T = float(temperature)
        cp = a + b * T + c * T**2 + d * T**3
        return {
            "temperature": T,
            "cp": cp,
            "h_relative": 0.0,
            "s": 0.0,
            "temp_interval": [200.0, 6000.0],
            "species_name": "fake",
            "phase": "G",
        }


class TestFitCpCoeffs:
    def test_recovers_exact_cubic(self):
        a, b, c, d = 25.0, 1.3e-2, -3.0e-6, 1.0e-9
        coeffs = fit_cp_coeffs(lambda T: a + b * T + c * T**2 + d * T**3, 300, 1000, 8)
        # Compare reconstructed Cp across the window (robust to fit conditioning).
        for T in (300.0, 550.0, 800.0, 1000.0):
            got = coeffs[0] + coeffs[1] * T + coeffs[2] * T**2 + coeffs[3] * T**3
            exp = a + b * T + c * T**2 + d * T**3
            assert got == pytest.approx(exp, rel=1e-6)

    def test_skips_non_finite_and_raising_points(self):
        def cp(T):
            if T > 700.0:
                raise ValueError("out of range")
            return 30.0 + 0.01 * T

        coeffs = fit_cp_coeffs(cp, 300, 1000, 12)  # only T<=700 contribute
        for T in (300.0, 500.0, 700.0):
            got = coeffs[0] + coeffs[1] * T + coeffs[2] * T**2 + coeffs[3] * T**3
            assert got == pytest.approx(30.0 + 0.01 * T, rel=1e-6)

    def test_too_few_points_raises(self):
        # Only one temperature yields a value -> cannot fit a cubic.
        def cp(T):
            if T < 310.0:
                return 29.0
            raise ValueError

        with pytest.raises(ValueError, match="at least 4"):
            fit_cp_coeffs(cp, 300, 1000, 8)


class TestImportSpeciesData:
    def test_builds_speciesdata_and_reproduces_cp(self):
        data = import_species_data(["O2", "CO2"], calc=FakeGlennCalculator())
        assert set(data) == {"O2", "CO2"}

        # MW and Hf carried through from the record.
        assert data["O2"].MW == pytest.approx(31.998)
        assert data["CO2"].MW == pytest.approx(44.009)
        assert data["CO2"].Hf == pytest.approx(-393510.0)

        # End-to-end: IdealThermo Cp reproduces the fake's cubic Cp.
        thermo = IdealThermo(data)
        for name, sid in (("O2", 1), ("CO2", 2)):
            a, b, c, d = _CP[sid]
            for T in (300.0, 600.0, 1000.0):
                exp = a + b * T + c * T**2 + d * T**3
                assert float(thermo.Cp(name, T)) == pytest.approx(exp, rel=1e-5)

    def test_accepts_single_string(self):
        data = import_species_data("O2", calc=FakeGlennCalculator())
        assert list(data) == ["O2"]

    def test_phase_preference_selects_gas(self):
        # Query "H2O" matches both a condensed (CR) and a gas (G) record;
        # phase="G" (default) must pick the gas one (id 4, Hf=-241826).
        data = import_species_data(["H2O"], calc=FakeGlennCalculator())
        assert data["H2O"].Hf == pytest.approx(-241826.0)

    def test_missing_species_warns_and_omits(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = import_species_data(["O2", "Nonexistium"], calc=FakeGlennCalculator())
        assert "O2" in data and "Nonexistium" not in data
        assert any("Nonexistium" in str(rec.message) for rec in w)

    def test_optional_boiling_point_sets_hvap_antoine(self):
        # Supplying Tb makes the Hvap/Antoine estimators produce non-default values.
        default = import_species_data(["O2"], calc=FakeGlennCalculator())["O2"]
        withTb = import_species_data(
            ["O2"], calc=FakeGlennCalculator(), boiling_points={"O2": 90.2}
        )["O2"]
        assert withTb.Hvap_coeffs != default.Hvap_coeffs


class TestListAvailableSpecies:
    def test_returns_records(self):
        recs = list_available_species("O2", calc=FakeGlennCalculator())
        assert recs and recs[0]["name"] == "O2"


def test_missing_pyglenn_raises_importerror():
    """With no calc and pyglenn absent, the adapter raises a clear ImportError."""
    try:
        import pyglenn  # noqa: F401
        pytest.skip("pyglenn is installed; ImportError path not exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pyglenn"):
        import_species_data(["O2"])
    with pytest.raises(ImportError, match="pyglenn"):
        list_available_species("O2")
