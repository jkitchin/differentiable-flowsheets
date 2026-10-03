"""DWSIM thermodynamic-data import utilities for difflow.

Import compound constants and ideal-gas heat capacities from DWSIM's
thermodynamics library (https://dwsim.org) into difflow's differentiable
:class:`SpeciesData` and :class:`CriticalProperties`.

Why an importer (and not a live backend)
----------------------------------------
DWSIM is a .NET application; its thermodynamics are reached from Python through
`pythonnet` (``import clr``). Calls into DWSIM return concrete numbers through
the CLR and are **not differentiable** -- JAX cannot trace through them. So,
exactly like :mod:`difflow.cantera_import` and :mod:`difflow.pyglenn_import`,
this module uses DWSIM only as a **one-time data source**: it pulls each
compound's constants (Tc, Pc, omega, MW, Tb, Hf) and samples its ideal-gas
Cp(T), then builds difflow's own JAX-native structures. difflow stays
differentiable end to end.

Requirements, verified against DWSIM 9.0.5 on Linux
---------------------------------------------------
* A DWSIM install: the directory holding ``DWSIM.Thermodynamics.dll``
  (DWSIM 6 and later carry the "DTL" calculator,
  ``DWSIM.Thermodynamics.CalculatorInterface.Calculator``, in that assembly;
  an older standalone ``DWSIM.Thermodynamics.StandaloneLibrary.dll`` is used
  if that is what the directory has). On Linux the SourceForge ``.deb``
  extracted with ``dpkg-deb -x`` puts it in ``usr/local/lib/dwsim``; see
  ``scripts/install_dwsim.sh``.
* The .NET 8 runtime (``apt-get install dotnet-runtime-8.0``): DWSIM 9 is
  a .NET 8 build, and pythonnet must load **CoreCLR** for it, not Mono (the
  Linux default). :class:`DWSIMBackend` does ``pythonnet.load("coreclr")``
  itself -- which only works if nothing in the process has imported ``clr``
  yet; one CLR per process.
* ``pip install pythonnet`` (or ``pip install "difflow[dwsim]"``).

Point to the install with ``dwsim_path=...`` or the ``DWSIM_PATH``
environment variable (``DWSIM_DTL_PATH`` and ``dtl_path=`` are accepted for
compatibility with the earlier prototype).

What is imported
----------------
* ``import_critical_props`` -> ``{name: CriticalProperties(Tc, Pc, omega, MW)}``
* ``import_species_data``   -> ``{name: SpeciesData}`` with:
    - ``Cp_coeffs``  : cubic fit of DWSIM's ideal-gas Cp(T) over ``T_fit_range``
    - ``MW``, ``Hf`` : from the compound constants
    - ``Hvap_coeffs``, ``antoine_coeffs`` : estimated from Tb/Tc (DWSIM's
      pressure-dependent vapor-pressure/Hvap models are not transcribed here)

Names are DWSIM's (``"Methane"``, ``"N-butane"``, ``"Hydrogen sulfide"``);
:data:`DWSIM_NAMES` maps difflow's database names to them and
:func:`dwsim_name` resolves one, aliases included.

Usage::

    from difflow.dwsim_import import import_species_data, import_critical_props, dwsim_name
    from difflow.thermo import IdealThermo, CubicThermo
    from difflow.eos import PengRobinson

    names = [dwsim_name(n) for n in ("methane", "co2", "water")]
    sp   = import_species_data(names)      # DWSIM_PATH set
    crit = import_critical_props(names)
    thermo = CubicThermo(IdealThermo(sp), PengRobinson(crit))

Every call without ``backend=`` starts and closes its own DWSIM calculator
(~2 s); pass one :class:`DWSIMBackend` to several calls to pay that once.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from typing import Any

from difflow.thermo import SpeciesData
from difflow.eos import CriticalProperties
# Reuse the cubic Cp fit and the ideal-gas-friendly Hvap/Antoine estimators from
# the sibling importers so all three fill difflow's data the same way.
from difflow.pyglenn_import import fit_cp_coeffs
from difflow.cantera_import import _estimate_antoine_coeffs, _estimate_hvap_coeffs


__all__ = [
    "DWSIMBackend",
    "DWSIM_NAMES",
    "dwsim_name",
    "import_species_data",
    "import_critical_props",
    "list_available_compounds",
]


#: difflow database name -> DWSIM 9.0.5 compound name, for the refinery's
#: species. Every entry was checked to exist in DWSIM 9.0.5 (all in its ChemSep
#: database) by CAS number. Where DWSIM's ChemSep constants differ noticeably
#: from difflow's the comment says so; otherwise Tc agrees to 0.31 %, Pc to
#: 2.0 %, omega to 0.012 and MW to 0.02 % (measured, worst entries commented; see
#: docs/unit-operations-refinery.md, "Validation against DWSIM: setup").
DWSIM_NAMES: dict[str, str] = {
    # light gases
    "hydrogen": "Hydrogen",
    "nitrogen": "Nitrogen",
    "oxygen": "Oxygen",
    "carbon_monoxide": "Carbon monoxide",    # DWSIM omega 0.045 vs difflow 0.066
    "carbon_dioxide": "Carbon dioxide",
    "hydrogen_sulfide": "Hydrogen sulfide",
    "ammonia": "Ammonia",
    "sulfur_dioxide": "Sulfur dioxide",
    "water": "Water",
    # paraffins
    "methane": "Methane",
    "ethane": "Ethane",
    "propane": "Propane",
    "n_butane": "N-butane",
    "isobutane": "Isobutane",
    "n_pentane": "N-pentane",
    "isopentane": "Isopentane",
    "neopentane": "Neopentane",
    "n_hexane": "N-hexane",
    "n_heptane": "N-heptane",
    "n_octane": "N-octane",
    "n_nonane": "N-nonane",
    "n_decane": "N-decane",
    "n_dodecane": "N-dodecane",
    # olefins
    "ethylene": "Ethylene",
    "propylene": "Propylene",
    "1_butene": "1-butene",
    "cis_2_butene": "Cis-2-butene",
    "trans_2_butene": "Trans-2-butene",      # DWSIM Pc 4.10 MPa vs 4.019 (+2 %)
    "isobutylene": "Isobutene",
    "1_pentene": "1-pentene",
    "2_methyl_2_butene": "2-methyl-2-butene",  # DWSIM Pc 3.86 MPa vs 3.42; omega 0.339 vs 0.285
    # C6 isomers and naphthenes
    "2_methylpentane": "2-methylpentane",
    "3_methylpentane": "3-methylpentane",
    "2_2_dimethylbutane": "2,2-dimethylbutane",
    "2_3_dimethylbutane": "2,3-dimethylbutane",
    "methylcyclopentane": "Methylcyclopentane",  # DWSIM omega 0.227 vs difflow 0.239
    "cyclohexane": "Cyclohexane",
    # C7-C9 branched paraffins (alkylate)
    "2_3_dimethylpentane": "2,3-dimethylpentane",
    "2_4_dimethylpentane": "2,4-dimethylpentane",
    "2_2_4_trimethylpentane": "2,2,4-trimethylpentane",
    "2_3_4_trimethylpentane": "2,3,4-trimethylpentane",
    "2_5_dimethylhexane": "2,5-dimethylhexane",
    "2_2_5_trimethylhexane": "2,2,5-trimethylhexane",
    # aromatics
    "benzene": "Benzene",
    "toluene": "Toluene",
    "ethylbenzene": "Ethylbenzene",
    "o_xylene": "O-xylene",
    "m_xylene": "M-xylene",
    "p_xylene": "P-xylene",
}


def dwsim_name(name: str) -> str:
    """DWSIM's name for a difflow species (database name or alias)."""
    from difflow.database import resolve_alias

    key = resolve_alias(name.lower().replace("-", "_").replace(" ", "_"))
    try:
        return DWSIM_NAMES[key]
    except KeyError:
        raise KeyError(f"no DWSIM name recorded for {name!r} (difflow key {key!r}); "
                       "add it to difflow.dwsim_import.DWSIM_NAMES") from None


# =============================================================================
# DWSIM backend (all pythonnet / .NET contact lives here)
# =============================================================================

# Exact ICompoundConstantProperties field names and their DWSIM units:
#   Molar_Weight                 kg/kmol  (== g/mol numerically)
#   Critical_Temperature         K
#   Critical_Pressure            Pa
#   Acentric_Factor              -
#   Normal_Boiling_Point         K
#   IG_Enthalpy_of_Formation_25C kJ/kg    -> J/mol via  * Molar_Weight
_CONST_FIELDS = {
    "MW": ("Molar_Weight", "MolarWeight", "MM"),
    "Tc": ("Critical_Temperature",),
    "Pc": ("Critical_Pressure",),
    "omega": ("Acentric_Factor",),
    "Tb": ("Normal_Boiling_Point", "Normal_Boiling_Temperature"),
    "Hf_mass": ("IG_Enthalpy_of_Formation_25C", "Enthalpy_of_Formation_25C"),
}


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute (or dict key) from ``names``."""
    for n in names:
        if isinstance(obj, dict):
            if obj.get(n) is not None:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return default


def _resolve_path(dwsim_path: str | None) -> str:
    path = (dwsim_path or os.environ.get("DWSIM_PATH")
            or os.environ.get("DWSIM_DTL_PATH"))
    if not path:
        raise ValueError(
            "No DWSIM path. Pass dwsim_path=... pointing at the folder containing "
            "DWSIM.Thermodynamics.dll (DWSIM 9: <install>/usr/local/lib/dwsim on Linux), "
            "or set the DWSIM_PATH environment variable.")
    return path


def _clr():
    """``clr`` with CoreCLR loaded (DWSIM 9 is a .NET 8 build)."""
    if "clr" not in sys.modules:
        try:
            from pythonnet import load
        except ImportError as e:  # pragma: no cover - only without pythonnet
            raise ImportError(
                "pythonnet is required to talk to DWSIM. Install it with "
                "`pip install pythonnet` (or `pip install \"difflow[dwsim]\"`), "
                "plus the .NET 8 runtime (apt-get install dotnet-runtime-8.0), and "
                "point dwsim_path / DWSIM_PATH at a DWSIM install."
            ) from e
        if sys.platform != "win32":
            load("coreclr")
    import clr

    return clr


class DWSIMBackend:
    """Thin wrapper over DWSIM's thermodynamics calculator ("DTL").

    The only place that touches DWSIM/pythonnet. It exposes a small,
    difflow-flavored contract that the importer consumes:

    * ``list_compounds() -> list[str]``
    * ``constants(name) -> dict`` with difflow-unit keys
      ``MW`` (g/mol), ``Tc`` (K), ``Pc`` (Pa), ``omega`` (-), ``Tb`` (K),
      ``Hf`` (J/mol), plus ``db`` (the DWSIM database it came from).
    * ``cp_ig_molar(name, T) -> float`` ideal-gas Cp in J/mol/K.
    * ``close()``

    Args:
        dwsim_path: Folder with ``DWSIM.Thermodynamics.dll`` (or the older
            ``DWSIM.Thermodynamics.StandaloneLibrary.dll``); default
            ``$DWSIM_PATH``, then ``$DWSIM_DTL_PATH``.
        model: Accepted for compatibility; the constants and ideal-gas Cp do
            not depend on a property package.
        dtl_path: Old name of ``dwsim_path``.
    """

    def __init__(self, dwsim_path: str | None = None, model: str = "PengRobinson",
                 dtl_path: str | None = None):
        self._calc = None
        self._const_cache: dict[str, dict] = {}
        self.model = model
        self.path = None
        self._connect(dwsim_path or dtl_path)

    def _connect(self, dwsim_path: str | None) -> None:
        # the path first: a missing install must not start a .NET runtime
        path = _resolve_path(dwsim_path)
        self.path = path
        clr = _clr()
        for dll in ("DWSIM.Thermodynamics.dll", "DWSIM.Thermodynamics.StandaloneLibrary.dll"):
            full = os.path.join(path, dll)
            if os.path.exists(full):
                break
        else:
            raise FileNotFoundError(f"neither DWSIM.Thermodynamics.dll nor the standalone "
                                    f"library is in {path}")
        if path not in sys.path:
            sys.path.append(path)
        clr.AddReference(full)
        from DWSIM.Thermodynamics import CalculatorInterface

        calc = CalculatorInterface.Calculator()
        calc.Initialize()
        self._calc = calc

    # -- DWSIM object access -------------------------------------------------

    def _const_obj(self, name: str) -> Any:
        """The compound's ``ConstantProperties`` object."""
        comps = self._calc.AvailableCompounds
        if not comps.ContainsKey(name):
            raise KeyError(f"DWSIM: compound not found: '{name}'")
        return comps[name]

    def list_compounds(self) -> list[str]:
        return [str(x) for x in self._calc.GetCompoundList()]

    def constants(self, name: str) -> dict:
        if name in self._const_cache:
            return self._const_cache[name]
        obj = self._const_obj(name)
        raw = {k: _attr(obj, *aliases) for k, aliases in _CONST_FIELDS.items()}
        MW = float(raw["MW"])  # kg/kmol == g/mol
        Hf_mass = raw["Hf_mass"]
        rec = {
            "MW": MW,
            "Tc": float(raw["Tc"]) if raw["Tc"] is not None else None,
            "Pc": float(raw["Pc"]) if raw["Pc"] is not None else None,
            "omega": float(raw["omega"]) if raw["omega"] is not None else None,
            "Tb": float(raw["Tb"]) if raw["Tb"] is not None else None,
            # kJ/kg * (kg/kmol) = kJ/kmol = J/mol
            "Hf": float(Hf_mass) * MW if Hf_mass is not None else 0.0,
            "db": str(getattr(obj, "OriginalDB", "")),
        }
        self._const_cache[name] = rec
        return rec

    def cp_ig_molar(self, name: str, T: float) -> float:
        """Ideal-gas Cp (J/mol/K): the calculator's ``idealGasHeatCapacity``,
        DWSIM's own correlation for the compound's database (kJ/kmol/K)."""
        self._const_obj(name)  # a clear KeyError for an unknown name
        return float(str(self._calc.GetCompoundTDepProp(name, "idealGasHeatCapacity",
                                                        float(T))))

    def close(self) -> None:
        self._calc = None


@contextlib.contextmanager
def _backend(backend: Any | None, dwsim_path: str | None, model: str):
    """Yield a backend; construct/close a DWSIMBackend if none was provided."""
    if backend is not None:
        yield backend
        return
    be = DWSIMBackend(dwsim_path=dwsim_path, model=model)
    try:
        yield be
    finally:
        be.close()


# =============================================================================
# Public import API
# =============================================================================


def list_available_compounds(
    *,
    backend: Any | None = None,
    dwsim_path: str | None = None,
    model: str = "PengRobinson",
    dtl_path: str | None = None,
) -> list[str]:
    """List compound names available in the DWSIM database."""
    with _backend(backend, dwsim_path or dtl_path, model) as be:
        return list(be.list_compounds())


def import_critical_props(
    compound_names: str | list[str],
    *,
    backend: Any | None = None,
    dwsim_path: str | None = None,
    model: str = "PengRobinson",
    dtl_path: str | None = None,
) -> dict[str, CriticalProperties]:
    """Import critical properties from DWSIM as CriticalProperties.

    Args:
        compound_names: A DWSIM compound name or list of names.
        backend: A backend implementing the DWSIMBackend contract (for testing
            or a custom DWSIM version). If None, a DWSIMBackend is opened.
        dwsim_path: Folder containing ``DWSIM.Thermodynamics.dll`` (or set the
            ``DWSIM_PATH`` environment variable).
        model: Kept for compatibility (constants do not depend on it).
        dtl_path: Old name of ``dwsim_path``.

    Returns:
        ``{name: CriticalProperties}``. Compounds that cannot be read emit a
        warning and are omitted.
    """
    if isinstance(compound_names, str):
        compound_names = [compound_names]

    result: dict[str, CriticalProperties] = {}
    with _backend(backend, dwsim_path or dtl_path, model) as be:
        for name in compound_names:
            try:
                c = be.constants(name)
                if c.get("Tc") is None or c.get("Pc") is None:
                    warnings.warn(f"DWSIM: missing critical data for '{name}'; skipping")
                    continue
                result[name] = CriticalProperties(
                    name=name,
                    Tc=c["Tc"],
                    Pc=c["Pc"],
                    omega=c.get("omega") or 0.0,
                    MW=c["MW"],
                )
            except Exception as e:  # noqa: BLE001 - report and continue per compound
                warnings.warn(f"DWSIM: could not import critical props for '{name}': {e}")
    return result


def import_species_data(
    compound_names: str | list[str],
    *,
    backend: Any | None = None,
    dwsim_path: str | None = None,
    model: str = "PengRobinson",
    T_fit_range: tuple[float, float] = (300.0, 1000.0),
    n_fit_points: int = 8,
    dtl_path: str | None = None,
) -> dict[str, SpeciesData]:
    """Import ideal-gas species data from DWSIM as SpeciesData.

    Args:
        compound_names: A DWSIM compound name or list of names.
        backend: A backend implementing the DWSIMBackend contract. If None, a
            DWSIMBackend is opened.
        dwsim_path: Folder with ``DWSIM.Thermodynamics.dll`` (or ``DWSIM_PATH``).
        model: Kept for compatibility (Cp does not depend on it).
        T_fit_range: ``(T_lo, T_hi)`` window (K) for the ideal-gas Cp cubic fit.
        n_fit_points: Number of Cp samples across the window.
        dtl_path: Old name of ``dwsim_path``.

    Returns:
        ``{name: SpeciesData}``. Unlike pyglenn, DWSIM also carries critical
        properties, so pair this with :func:`import_critical_props` for a full
        EOS/CubicThermo. Hvap/Antoine are estimated from Tb/Tc (see module doc).
    """
    if isinstance(compound_names, str):
        compound_names = [compound_names]
    T_lo, T_hi = T_fit_range

    result: dict[str, SpeciesData] = {}
    with _backend(backend, dwsim_path or dtl_path, model) as be:
        for name in compound_names:
            try:
                c = be.constants(name)
                Cp_coeffs = fit_cp_coeffs(
                    lambda T, _n=name: be.cp_ig_molar(_n, T), T_lo, T_hi, n_fit_points
                )
                antoine = _estimate_antoine_coeffs(name, c.get("Tb"), c.get("Tc"))
                hvap = _estimate_hvap_coeffs(c.get("Tb"), c.get("Tc"))
                result[name] = SpeciesData(
                    name=name,
                    MW=float(c["MW"]),
                    Cp_coeffs=Cp_coeffs,
                    Hvap_coeffs=hvap,
                    antoine_coeffs=antoine,
                    Hf=float(c.get("Hf", 0.0)),
                )
            except Exception as e:  # noqa: BLE001 - report and continue per compound
                warnings.warn(f"DWSIM: could not import '{name}': {e}")
    return result
