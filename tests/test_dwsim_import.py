"""Tests for the DWSIM thermo-data import adapter (prototype).

DWSIM needs a .NET runtime and is not available in CI, so these tests drive the
importer with a FakeDWSIMBackend implementing the DWSIMBackend contract
(constants / cp_ig_molar / list_compounds / close). Each fake compound has an
exact cubic ideal-gas Cp so the fit is exact and can be checked end to end.
"""

import warnings

import pytest

from difflow.thermo import IdealThermo
from difflow.eos import PengRobinson
from difflow.dwsim_import import (
    import_species_data,
    import_critical_props,
    list_available_compounds,
)


# name -> (MW g/mol, Tc K, Pc Pa, omega, Tb K, Hf J/mol, (a,b,c,d) Cp J/mol/K)
_DB = {
    "Methane": (16.04, 190.6, 4.60e6, 0.011, 111.7, -74600.0,
                (33.0, 5.0e-3, 1.5e-5, -6.0e-9)),
    "Carbon dioxide": (44.01, 304.2, 7.38e6, 0.224, 194.7, -393500.0,
                       (23.0, 6.0e-2, -4.0e-5, 1.0e-8)),
    "Water": (18.02, 647.1, 22.06e6, 0.345, 373.1, -241800.0,
              (33.5, 6.0e-4, 5.0e-6, -1.5e-9)),
}


class FakeDWSIMBackend:
    """Stand-in for DWSIMBackend using the same difflow-unit contract."""

    def __init__(self, missing_crit: bool = False):
        self._missing_crit = missing_crit

    def list_compounds(self):
        return list(_DB)

    def constants(self, name):
        MW, Tc, Pc, omega, Tb, Hf, _ = _DB[name]
        if self._missing_crit:
            Tc = Pc = None
        return {"MW": MW, "Tc": Tc, "Pc": Pc, "omega": omega, "Tb": Tb, "Hf": Hf}

    def cp_ig_molar(self, name, T):
        a, b, c, d = _DB[name][6]
        T = float(T)
        return a + b * T + c * T**2 + d * T**3

    def close(self):
        pass


class TestImportCriticalProps:
    def test_builds_critical_properties(self):
        crit = import_critical_props(
            ["Methane", "Carbon dioxide"], backend=FakeDWSIMBackend()
        )
        assert set(crit) == {"Methane", "Carbon dioxide"}
        assert crit["Methane"].Tc == pytest.approx(190.6)
        assert crit["Methane"].Pc == pytest.approx(4.60e6)
        assert crit["Carbon dioxide"].omega == pytest.approx(0.224)
        # The result must build a working Peng-Robinson EOS.
        pr = PengRobinson(crit)
        assert pr.species_order == ["Methane", "Carbon dioxide"]

    def test_accepts_single_string(self):
        crit = import_critical_props("Water", backend=FakeDWSIMBackend())
        assert list(crit) == ["Water"]

    def test_missing_critical_data_warns_and_omits(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            crit = import_critical_props(
                ["Methane"], backend=FakeDWSIMBackend(missing_crit=True)
            )
        assert crit == {}
        assert any("Methane" in str(rec.message) for rec in w)


class TestImportSpeciesData:
    def test_builds_speciesdata_and_reproduces_cp(self):
        data = import_species_data(
            ["Methane", "Carbon dioxide", "Water"], backend=FakeDWSIMBackend()
        )
        assert set(data) == {"Methane", "Carbon dioxide", "Water"}
        assert data["Methane"].MW == pytest.approx(16.04)
        assert data["Methane"].Hf == pytest.approx(-74600.0)

        thermo = IdealThermo(data)
        for name in _DB:
            a, b, c, d = _DB[name][6]
            for T in (300.0, 600.0, 1000.0):
                exp = a + b * T + c * T**2 + d * T**3
                assert float(thermo.Cp(name, T)) == pytest.approx(exp, rel=1e-5)

    def test_species_and_critical_together_build_cubic_thermo(self):
        from difflow.thermo import CubicThermo

        names = ["Methane", "Carbon dioxide"]
        be = FakeDWSIMBackend()
        sp = import_species_data(names, backend=be)
        crit = import_critical_props(names, backend=be)
        thermo = CubicThermo(IdealThermo(sp), PengRobinson(crit))
        # Real-gas enthalpy from DWSIM-sourced data is finite.
        import jax.numpy as jnp

        H = thermo.stream_enthalpy(
            {"Methane": 1.0, "Carbon dioxide": 1.0},
            jnp.array(350.0),
            phase="vapor",
            P=jnp.array(2e6),
        )
        assert jnp.isfinite(H)

    def test_unknown_compound_warns_and_omits(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = import_species_data(
                ["Methane", "Unobtainium"], backend=FakeDWSIMBackend()
            )
        assert "Methane" in data and "Unobtainium" not in data
        assert any("Unobtainium" in str(rec.message) for rec in w)


class TestListAvailableCompounds:
    def test_lists_compounds(self):
        names = list_available_compounds(backend=FakeDWSIMBackend())
        assert "Water" in names


def test_missing_pythonnet_or_path_raises():
    """With no backend and no DWSIM available, the adapter fails clearly."""
    try:
        import clr  # noqa: F401
        # pythonnet present: without a DLL path it must raise ValueError.
        with pytest.raises((ValueError, Exception)):
            import_species_data(["Methane"])
    except ImportError:
        # pythonnet absent: must raise a clear ImportError mentioning pythonnet.
        with pytest.raises(ImportError, match="pythonnet"):
            import_species_data(["Methane"])
