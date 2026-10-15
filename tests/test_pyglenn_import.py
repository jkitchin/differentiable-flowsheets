"""Tests for the NASA Glenn (pyglenn) thermo-data import adapter.

pyglenn is an optional dependency and is not installed in CI, so these tests
drive the adapter with a FakeGlennCalculator that implements the documented
pyglenn API.
"""

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


def test_fit_too_few_points_raises():
    def cp(T):
        if T < 310.0:
            return 29.0
        raise ValueError

    with pytest.raises(ValueError, match="at least 4"):
        fit_cp_coeffs(cp, 300, 1000, 8)


def test_import_species_data_smoke():
    data = import_species_data(["O2", "CO2"], calc=FakeGlennCalculator())
    assert set(data) == {"O2", "CO2"}
    assert data["CO2"].MW == pytest.approx(44.009)
    assert np.isfinite(float(IdealThermo(data).Cp("O2", 500.0)))
    assert list_available_species("O2", calc=FakeGlennCalculator())[0]["name"] == "O2"


def test_missing_species_warns_and_omits():
    with pytest.warns(UserWarning, match="Nonexistium"):
        data = import_species_data(["O2", "Nonexistium"], calc=FakeGlennCalculator())
    assert list(data) == ["O2"]


def test_missing_pyglenn_raises_importerror():
    try:
        import pyglenn  # noqa: F401
        pytest.skip("pyglenn is installed; ImportError path not exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pyglenn"):
        import_species_data(["O2"])
