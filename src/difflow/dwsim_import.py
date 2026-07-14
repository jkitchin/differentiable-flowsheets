"""DWSIM thermodynamic-data import utilities for difflow (PROTOTYPE).

Import compound constants and ideal-gas heat capacities from DWSIM's
thermodynamics library (https://dwsim.org) into difflow's differentiable
:class:`SpeciesData` and :class:`CriticalProperties`.

Why an importer (and not a live backend)
----------------------------------------
DWSIM is a .NET application; its thermodynamics are reached from Python through
`pythonnet` (``import clr``) against ``DWSIM.Thermodynamics.StandaloneLibrary.dll``.
Calls into DWSIM return concrete numbers through the CLR and are **not
differentiable** -- JAX cannot trace through them. So, exactly like
:mod:`difflow.cantera_import` and :mod:`difflow.pyglenn_import`, this module uses
DWSIM only as a **one-time data source**: it pulls each compound's constants
(Tc, Pc, omega, MW, Tb, Hf) and samples its ideal-gas Cp(T), then builds
difflow's own JAX-native structures. difflow stays differentiable end to end.

DWSIM is an **optional** dependency and is reached via pythonnet + a local DWSIM
(or DTL) install::

    pip install pythonnet            # or:  pip install "difflow[dwsim]"
    # plus a DWSIM / DTL installation providing the standalone thermo DLL

PROTOTYPE status
----------------
This adapter is built to DWSIM's documented API, but DWSIM cannot be exercised
in difflow's own CI (no .NET runtime), so the exact object graph used to read a
compound's ``ConstantProperties`` and its ideal-gas Cp may need adjusting for a
given DWSIM version. **All** DWSIM contact is isolated in :class:`DWSIMBackend`;
the import logic (Cp fit, SpeciesData/CriticalProperties construction) is
backend-agnostic and unit-tested against a fake backend. To adapt to your DWSIM
build, subclass/replace :class:`DWSIMBackend` and pass it via ``backend=``.

What is imported
----------------
* ``import_critical_props`` -> ``{name: CriticalProperties(Tc, Pc, omega, MW)}``
* ``import_species_data``   -> ``{name: SpeciesData}`` with:
    - ``Cp_coeffs``  : cubic fit of DWSIM's ideal-gas Cp(T) over ``T_fit_range``
    - ``MW``, ``Hf`` : from the compound constants
    - ``Hvap_coeffs``, ``antoine_coeffs`` : estimated from Tb/Tc (DWSIM's
      pressure-dependent vapor-pressure/Hvap models are not transcribed here)

Usage::

    from difflow.dwsim_import import import_species_data, import_critical_props
    from difflow.thermo import IdealThermo, CubicThermo
    from difflow.eos import PengRobinson

    names = ["Methane", "Carbon dioxide", "Water"]
    sp   = import_species_data(names)
    crit = import_critical_props(names)
    thermo = CubicThermo(IdealThermo(sp), PengRobinson(crit))
"""

from __future__ import annotations

import contextlib
import os
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
    "import_species_data",
    "import_critical_props",
    "list_available_compounds",
]


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


class DWSIMBackend:
    """Thin, best-effort wrapper over DWSIM's standalone thermodynamics library.

    PROTOTYPE: this is the only place that touches DWSIM/pythonnet. It exposes a
    small, difflow-flavored contract that the importer consumes:

    * ``list_compounds() -> list[str]``
    * ``constants(name) -> dict`` with difflow-unit keys
      ``MW`` (g/mol), ``Tc`` (K), ``Pc`` (Pa), ``omega`` (-), ``Tb`` (K),
      ``Hf`` (J/mol).
    * ``cp_ig_molar(name, T) -> float`` ideal-gas Cp in J/mol/K.
    * ``close()``

    The unit conversions from DWSIM's mass/kmol basis to difflow's molar basis
    happen here, so the importer is unit-agnostic.
    """

    def __init__(self, dtl_path: str | None = None, model: str = "PengRobinson"):
        self._pp = None
        self._dtlc = None
        self._const_cache: dict[str, dict] = {}
        dtl_path = dtl_path or os.environ.get("DWSIM_DTL_PATH")
        self._connect(dtl_path, model)

    def _connect(self, dtl_path: str | None, model: str) -> None:
        try:
            import clr  # noqa: F401  (pythonnet)
        except ImportError as e:  # pragma: no cover - only without pythonnet
            raise ImportError(
                "pythonnet is required to talk to DWSIM. Install it with "
                "`pip install pythonnet` (or `pip install \"difflow[dwsim]\"`) "
                "and provide a DWSIM/DTL install path via dtl_path or the "
                "DWSIM_DTL_PATH environment variable."
            ) from e
        import clr

        if not dtl_path:
            raise ValueError(
                "No DWSIM library path. Pass dtl_path=... pointing at the folder "
                "containing DWSIM.Thermodynamics.StandaloneLibrary.dll, or set "
                "the DWSIM_DTL_PATH environment variable."
            )

        dll = os.path.join(dtl_path, "DWSIM.Thermodynamics.StandaloneLibrary.dll")
        clr.AddReference(dll)
        from DWSIM.Thermodynamics import CalculatorInterface, PropertyPackages

        self._dtlc = CalculatorInterface.Calculator()
        self._dtlc.Initialize()
        pp_cls = getattr(PropertyPackages, f"{model}PropertyPackage")
        self._pp = pp_cls(True)
        self._dtlc.TransferCompounds(self._pp)

    # -- DWSIM object access (best-effort; adjust here for your DWSIM version) --

    def _const_obj(self, name: str) -> Any:
        """Return the ConstantProperties object for a compound."""
        pp = self._pp
        # Common access paths across DWSIM versions; first that works wins.
        for getter in (
            lambda: pp.CompoundConstants[name],
            lambda: pp.get_compound(name),
            lambda: pp.CurrentMaterialStream.Phases[0].Compounds[name].ConstantProperties,
        ):
            try:
                obj = getter()
                if obj is not None:
                    return obj
            except Exception:
                continue
        raise KeyError(f"DWSIM: compound not found or inaccessible: '{name}'")

    def list_compounds(self) -> list[str]:
        try:
            return [str(x) for x in self._pp.CompoundConstants.Keys]
        except Exception:
            return [str(x) for x in self._dtlc.GetCompoundList()]

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
        }
        self._const_cache[name] = rec
        return rec

    def cp_ig_molar(self, name: str, T: float) -> float:
        # AUX_CPi returns the ideal-gas Cp in kJ/kg/K; * MW(kg/kmol) = J/mol/K.
        cp_mass = float(self._pp.AUX_CPi(name, float(T)))
        return cp_mass * self.constants(name)["MW"]

    def close(self) -> None:
        try:
            if self._dtlc is not None:
                self._dtlc.Dispose()
        except Exception:
            pass


@contextlib.contextmanager
def _backend(backend: Any | None, dtl_path: str | None, model: str):
    """Yield a backend; construct/close a DWSIMBackend if none was provided."""
    if backend is not None:
        yield backend
        return
    be = DWSIMBackend(dtl_path=dtl_path, model=model)
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
    dtl_path: str | None = None,
    model: str = "PengRobinson",
) -> list[str]:
    """List compound names available in the DWSIM database."""
    with _backend(backend, dtl_path, model) as be:
        return list(be.list_compounds())


def import_critical_props(
    compound_names: str | list[str],
    *,
    backend: Any | None = None,
    dtl_path: str | None = None,
    model: str = "PengRobinson",
) -> dict[str, CriticalProperties]:
    """Import critical properties from DWSIM as CriticalProperties.

    Args:
        compound_names: A DWSIM compound name or list of names.
        backend: A backend implementing the DWSIMBackend contract (for testing
            or a custom DWSIM version). If None, a DWSIMBackend is opened.
        dtl_path: Folder containing the DWSIM standalone thermo DLL (or set the
            DWSIM_DTL_PATH environment variable).
        model: DWSIM property-package model name used to source the data.

    Returns:
        ``{name: CriticalProperties}``. Compounds that cannot be read emit a
        warning and are omitted.
    """
    if isinstance(compound_names, str):
        compound_names = [compound_names]

    result: dict[str, CriticalProperties] = {}
    with _backend(backend, dtl_path, model) as be:
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
    dtl_path: str | None = None,
    model: str = "PengRobinson",
    T_fit_range: tuple[float, float] = (300.0, 1000.0),
    n_fit_points: int = 8,
) -> dict[str, SpeciesData]:
    """Import ideal-gas species data from DWSIM as SpeciesData.

    Args:
        compound_names: A DWSIM compound name or list of names.
        backend: A backend implementing the DWSIMBackend contract. If None, a
            DWSIMBackend is opened.
        dtl_path: Folder with the DWSIM standalone thermo DLL (or DWSIM_DTL_PATH).
        model: DWSIM property-package model name used to source the data.
        T_fit_range: ``(T_lo, T_hi)`` window (K) for the ideal-gas Cp cubic fit.
        n_fit_points: Number of Cp samples across the window.

    Returns:
        ``{name: SpeciesData}``. Unlike pyglenn, DWSIM also carries critical
        properties, so pair this with :func:`import_critical_props` for a full
        EOS/CubicThermo. Hvap/Antoine are estimated from Tb/Tc (see module doc).
    """
    if isinstance(compound_names, str):
        compound_names = [compound_names]
    T_lo, T_hi = T_fit_range

    result: dict[str, SpeciesData] = {}
    with _backend(backend, dtl_path, model) as be:
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
