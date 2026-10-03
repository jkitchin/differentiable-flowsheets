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


def test_missing_path_raises_without_starting_a_runtime(monkeypatch):
    """With no backend and no DWSIM path, the adapter fails clearly -- before
    pythonnet starts any .NET runtime (one CLR per process; starting the
    wrong one here would break every later DWSIM use in the process)."""
    import sys

    monkeypatch.delenv("DWSIM_PATH", raising=False)
    monkeypatch.delenv("DWSIM_DTL_PATH", raising=False)
    had_clr = "clr" in sys.modules
    with pytest.raises(ValueError, match="DWSIM_PATH"):
        import_species_data(["Methane"])
    assert ("clr" in sys.modules) == had_clr


def test_missing_pythonnet_raises_import_error(monkeypatch, tmp_path):
    """With a path but no pythonnet, the error names pythonnet."""
    import sys

    if "clr" in sys.modules:
        pytest.skip("a CLR is already loaded in this process")
    monkeypatch.setitem(sys.modules, "pythonnet", None)
    with pytest.raises(ImportError, match="pythonnet"):
        import_species_data(["Methane"], dwsim_path=str(tmp_path))


def test_dwsim_names_resolve_aliases_and_database_names():
    from difflow.database import list_species
    from difflow.dwsim_import import DWSIM_NAMES, dwsim_name

    assert dwsim_name("n_butane") == "N-butane"
    assert dwsim_name("h2s") == "Hydrogen sulfide"
    assert dwsim_name("isooctane") == "2,2,4-trimethylpentane"
    assert dwsim_name("MCP") == "Methylcyclopentane"
    known = set(list_species())
    assert set(DWSIM_NAMES) <= known, sorted(set(DWSIM_NAMES) - known)
    with pytest.raises(KeyError, match="DWSIM_NAMES"):
        dwsim_name("unobtainium")


# =============================================================================
# Release: the real DWSIM (9.0.5), in a subprocess
# =============================================================================

_REAL_SCRIPT = r"""
import json, sys
from difflow.dwsim_import import (DWSIMBackend, DWSIM_NAMES, import_critical_props,
                                  import_species_data)
be = DWSIMBackend(dwsim_path=sys.argv[1])
have = set(be.list_compounds())
names = sorted(set(DWSIM_NAMES.values()))
crit = import_critical_props(names, backend=be)
sp = import_species_data(["Methane", "Propane", "Benzene"], backend=be)
out = {"missing": sorted(set(names) - have),
       "crit": {n: dict(Tc=float(c.Tc), Pc=float(c.Pc), omega=float(c.omega), MW=float(c.MW))
                for n, c in crit.items()},
       "db": {n: be.constants(n)["db"] for n in names if n in have},
       "cp": {n: [float(v) for v in s.Cp_coeffs] for n, s in sp.items()},
       "cp_raw": {n: [be.cp_ig_molar(n, T) for T in (300.0, 500.0, 800.0)]
                  for n in ("Methane", "Propane", "Benzene")}}
print("JSON" + json.dumps(out))
"""


def _real_dwsim_path():
    import importlib.util
    import os
    import shutil

    from .refinery.reference.dwsim_session import DEFAULT_DWSIM_PATH

    path = os.environ.get("DWSIM_PATH") or DEFAULT_DWSIM_PATH
    if not os.path.exists(os.path.join(path, "DWSIM.Thermodynamics.dll")):
        pytest.skip(f"no DWSIM install at {path} (set DWSIM_PATH; scripts/install_dwsim.sh)")
    if importlib.util.find_spec("pythonnet") is None:
        pytest.skip("pythonnet is not installed")
    if shutil.which("dotnet") is None and not os.path.isdir("/usr/lib/dotnet"):
        pytest.skip("no .NET runtime")
    return path


@pytest.fixture(scope="module")
def real_import():
    """The importer run against the real DWSIM, in its own process (one CLR
    per process: never in pytest's)."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    path = _real_dwsim_path()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-c", _REAL_SCRIPT, path], capture_output=True,
                       text=True, timeout=600, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON")][-1]
    return json.loads(line[4:])


#: What "agrees" means for a database constant: DWSIM's ChemSep tables and
#: difflow's (NIST/Perry/DIPPR, Lemmon for the butenes, PSRK for the C6s) are
#: different compilations, so this is a check that the importer reads the right
#: field in the right unit, with room for honest table differences --
#: Tc 0.6 %, Pc 2.5 %, omega 0.015, MW 0.1 %. Outside that, listed below.
_TOL = dict(Tc=6e-3, Pc=2.5e-2, omega=1.5e-2, MW=1e-3)

#: Known table differences beyond _TOL (DWSIM ChemSep vs difflow.database),
#: measured with DWSIM 9.0.5: kept visible rather than tolerated.
_KNOWN_DIFFERENCES = {
    ("2_methyl_2_butene", "Pc"),      # DWSIM 3.86 MPa, difflow 3.42 (+12.9 %)
    ("2_methyl_2_butene", "omega"),   # 0.339 vs 0.285
    ("carbon_monoxide", "omega"),     # 0.045 vs 0.066
}


@pytest.mark.release
class TestAgainstRealDWSIM:
    def test_every_mapped_name_exists_in_dwsim(self, real_import):
        assert real_import["missing"] == []
        assert set(real_import["db"].values()) == {"ChemSep"}

    def test_critical_constants_match_difflows_database(self, real_import):
        from difflow.database import get_critical_props
        from difflow.dwsim_import import DWSIM_NAMES

        off = []
        for key, dw in DWSIM_NAMES.items():
            ref = get_critical_props(key)
            got = real_import["crit"][dw]
            for k, tol in _TOL.items():
                r = getattr(ref, k)
                bad = (abs(got[k] - r) > tol) if k == "omega" else (abs(got[k] / r - 1) > tol)
                if bad and (key, k) not in _KNOWN_DIFFERENCES:
                    off.append((key, k, got[k], r))
        assert off == []

    def test_known_differences_are_still_differences(self, real_import):
        """If DWSIM or difflow fixes a table, take it off the list."""
        from difflow.database import get_critical_props
        from difflow.dwsim_import import DWSIM_NAMES

        for key, k in _KNOWN_DIFFERENCES:
            r = getattr(get_critical_props(key), k)
            got = real_import["crit"][DWSIM_NAMES[key]][k]
            assert abs(got - r) > _TOL[k] * (1 if k == "omega" else abs(r)), (key, k)

    def test_ideal_gas_cp_fit_reproduces_dwsim(self, real_import):
        """The cubic fit through DWSIM's own Cp (ChemSep equation 16 for
        these) is within 0.5 % at 300-800 K."""
        for n, (c300, c500, c800) in real_import["cp_raw"].items():
            a, b, c, d = real_import["cp"][n]
            for T, ref in ((300.0, c300), (500.0, c500), (800.0, c800)):
                assert a + b * T + c * T * T + d * T ** 3 == pytest.approx(ref, rel=5e-3), (n, T)

    def test_dwsim_cp_is_in_j_per_mol_k(self, real_import):
        """Methane's ideal-gas Cp at 300 K is 35.7 J/mol/K (NIST)."""
        assert real_import["cp_raw"]["Methane"][0] == pytest.approx(35.7, rel=0.01)
