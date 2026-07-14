"""NASA Glenn (pyglenn) thermodynamic-data import utilities for difflow.

This module imports ideal-gas thermodynamic data from the NASA Glenn / CEA
thermodynamic database, as exposed by the ``pyglenn`` package
(https://github.com/ProfLeao/pyglenn), into difflow's :class:`SpeciesData`.

pyglenn is an **optional** dependency::

    pip install pyglenn            # or:  pip install "difflow[pyglenn]"

pyglenn stores NASA-9 coefficient polynomials (a1..a7, b1, b2) for ~2030
species and exposes a small API::

    from pyglenn import ThermochemicalCalculator
    with ThermochemicalCalculator() as calc:
        rec = calc.get_available_species("O2")[0]      # dict: id, name, phase, molecular_weight, ...
        p = calc.calculate_properties(rec["id"], 1000.0)  # dict: cp, h_relative, s, ...

What this adapter maps
----------------------
* Ideal-gas ``Cp(T)`` (from the NASA-9 polynomial, evaluated by pyglenn) ->
  ``SpeciesData.Cp_coeffs`` by a least-squares cubic fit (see below).
* ``molecular_weight``                 -> ``SpeciesData.MW``  (g/mol).
* ``heat_of_formation_298K``           -> ``SpeciesData.Hf``  (J/mol at 298.15 K).

What pyglenn does NOT provide (so this adapter cannot either)
------------------------------------------------------------
* **Critical properties** (Tc, Pc, omega). There is deliberately no
  ``import_critical_props`` here: the NASA Glenn data is ideal-gas only. To
  build a :class:`~difflow.eos.PengRobinson`/:class:`~difflow.eos.SRK` or a
  :class:`~difflow.thermo.CubicThermo`, pair the ``SpeciesData`` returned here
  with :class:`~difflow.eos.CriticalProperties` from ``difflow.database`` or
  ``difflow.cantera_import``.
* **Liquid-phase data** (heat of vaporization, Antoine vapor pressure). Those
  ``SpeciesData`` fields are filled with neutral placeholders, or estimated
  from an optional boiling point / critical temperature you supply via
  ``boiling_points`` / ``critical_temps``. The imported data is therefore only
  meaningful for gas-phase / ideal-gas enthalpy unless you add VLE data.

The Cp cubic fit
----------------
difflow's ideal-gas Cp is a cubic ``Cp = a + b*T + c*T**2 + d*T**3`` while the
NASA-9 form additionally carries ``1/T**2`` and ``1/T`` terms. ``Cp_coeffs`` are
therefore obtained by sampling pyglenn's computed ``Cp(T)`` over a temperature
window (default 300-1000 K) and least-squares fitting the cubic. Accuracy is
best inside that window; set ``T_fit_range`` to the range you actually operate
over. Samples that fall outside a species' valid interval (pyglenn raises) are
dropped, so a species with a narrower range is fit only where it is defined.

Usage::

    from difflow.pyglenn_import import import_species_data, list_available_species
    from difflow.thermo import IdealThermo

    data = import_species_data(["O2", "CO2", "H2O"])
    thermo = IdealThermo(data)
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Callable

import numpy as np

from difflow.thermo import SpeciesData
# Reuse the ideal-gas-friendly placeholder estimators from the Cantera adapter
# so both importers fill Hvap/Antoine the same way.
from difflow.cantera_import import _estimate_antoine_coeffs, _estimate_hvap_coeffs


__all__ = [
    "import_species_data",
    "list_available_species",
    "fit_cp_coeffs",
]


# =============================================================================
# pyglenn connection handling (lazy: importing this module never needs pyglenn)
# =============================================================================


def _lazy_calculator_cls():
    """Import pyglenn.ThermochemicalCalculator, with a clear error if absent."""
    try:
        from pyglenn import ThermochemicalCalculator
    except ImportError as e:  # pragma: no cover - exercised only without pyglenn
        raise ImportError(
            "pyglenn is required for this function. Install it with "
            "`pip install pyglenn` (or `pip install \"difflow[pyglenn]\"`); "
            "see https://github.com/ProfLeao/pyglenn."
        ) from e
    return ThermochemicalCalculator


@contextlib.contextmanager
def _calculator(calc: Any | None = None):
    """Yield a connected calculator.

    If ``calc`` is provided it is used as-is and left open (the caller owns it).
    Otherwise a ``ThermochemicalCalculator`` is created, connected, and closed
    on exit.
    """
    if calc is not None:
        yield calc
        return
    cls = _lazy_calculator_cls()
    owned = cls()
    owned.connect()
    try:
        yield owned
    finally:
        owned.close()


def _field(record: Any, *keys: str, default: Any = None) -> Any:
    """Read a field from a pyglenn record (dict or object), trying key aliases."""
    for k in keys:
        if isinstance(record, dict):
            if record.get(k) is not None:
                return record[k]
        else:
            v = getattr(record, k, None)
            if v is not None:
                return v
    return default


def _resolve_record(calc: Any, query: str, phase: str | None) -> Any | None:
    """Pick a single species record for ``query`` from get_available_species."""
    records = calc.get_available_species(query)
    if not records:
        return None
    if phase is not None:
        matches = [
            r for r in records
            if phase.lower() in str(_field(r, "phase", default="")).lower()
        ]
        if matches:
            records = matches
    return records[0]


# =============================================================================
# Cp cubic fit
# =============================================================================


def fit_cp_coeffs(
    cp_of_T: Callable[[float], float],
    T_lo: float = 300.0,
    T_hi: float = 1000.0,
    n_points: int = 8,
) -> tuple[float, float, float, float]:
    """Least-squares fit ``Cp = a + b*T + c*T**2 + d*T**3`` to sampled Cp(T).

    Args:
        cp_of_T: Callable returning Cp (J/mol/K) at a temperature (K). Points
            where it raises or returns a non-finite value are skipped, so it is
            safe to sample outside a species' valid range.
        T_lo: Low end of the fit window (K).
        T_hi: High end of the fit window (K).
        n_points: Number of evenly spaced sample temperatures.

    Returns:
        ``(a, b, c, d)`` for ``Cp = a + b*T + c*T**2 + d*T**3``.

    Raises:
        ValueError: If fewer than four samples succeed (a cubic needs >= 4).
    """
    Ts, cps = [], []
    for T in np.linspace(T_lo, T_hi, n_points):
        try:
            cp = float(cp_of_T(float(T)))
        except Exception:
            continue
        if np.isfinite(cp):
            Ts.append(float(T))
            cps.append(cp)

    if len(Ts) < 4:
        raise ValueError(
            f"Only {len(Ts)} valid Cp sample(s) in [{T_lo}, {T_hi}] K; "
            "need at least 4 to fit a cubic. Adjust T_fit_range/n_fit_points."
        )

    T = np.asarray(Ts)
    A = np.vstack([np.ones_like(T), T, T**2, T**3]).T
    coeffs, *_ = np.linalg.lstsq(A, np.asarray(cps), rcond=None)
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3]))


# =============================================================================
# Public import API
# =============================================================================


def list_available_species(
    query: str = "",
    *,
    calc: Any | None = None,
) -> list[dict[str, Any]]:
    """List NASA Glenn species records matching ``query``.

    Args:
        query: Name or formula fragment to search (pyglenn's
            ``get_available_species``). Empty string returns whatever pyglenn
            returns for an empty query.
        calc: An already-connected ``ThermochemicalCalculator`` to reuse. If
            None, a temporary one is opened and closed.

    Returns:
        The list of record dicts pyglenn returns (keys include ``id``, ``name``,
        ``phase``, ``molecular_weight``).
    """
    with _calculator(calc) as c:
        return list(c.get_available_species(query))


def import_species_data(
    species_names: str | list[str],
    *,
    calc: Any | None = None,
    phase: str | None = "G",
    T_fit_range: tuple[float, float] = (300.0, 1000.0),
    n_fit_points: int = 8,
    boiling_points: dict[str, float] | None = None,
    critical_temps: dict[str, float] | None = None,
) -> dict[str, SpeciesData]:
    """Import ideal-gas species data from NASA Glenn (pyglenn) as SpeciesData.

    Args:
        species_names: A species name/formula or list thereof (as understood by
            pyglenn's ``get_available_species``).
        calc: An already-connected ``ThermochemicalCalculator`` to reuse. If
            None, a temporary one is opened and closed.
        phase: Preferred phase; the first record whose ``phase`` contains this
            (case-insensitive) is used. Defaults to ``"G"`` (gas). Pass None to
            take pyglenn's first match regardless of phase.
        T_fit_range: ``(T_lo, T_hi)`` window (K) for the Cp cubic fit.
        n_fit_points: Number of Cp samples across the window.
        boiling_points: Optional ``{name: Tb}`` (K) to estimate Hvap/Antoine.
        critical_temps: Optional ``{name: Tc}`` (K) to estimate Hvap/Antoine.

    Returns:
        ``{name: SpeciesData}`` keyed by the requested name. Species that cannot
        be found or fit emit a warning and are omitted. The returned data is
        ideal-gas only (see the module docstring); pair with
        ``CriticalProperties`` for an EOS/CubicThermo.

    Example:
        >>> data = import_species_data(["O2", "CO2", "H2O"])
        >>> thermo = IdealThermo(data)
    """
    if isinstance(species_names, str):
        species_names = [species_names]
    boiling_points = boiling_points or {}
    critical_temps = critical_temps or {}
    T_lo, T_hi = T_fit_range

    result: dict[str, SpeciesData] = {}

    with _calculator(calc) as c:
        for name in species_names:
            try:
                record = _resolve_record(c, name, phase)
                if record is None:
                    warnings.warn(f"pyglenn: species not found: '{name}'")
                    continue

                species_id = _field(record, "id")
                MW = _field(record, "molecular_weight", "mw", "MW", "molar_mass")
                Hf = _field(
                    record,
                    "heat_of_formation_298K",
                    "heat_of_formation",
                    "Hf",
                    default=0.0,
                )
                if MW is None:
                    warnings.warn(f"pyglenn: no molecular weight for '{name}'; skipping")
                    continue

                Cp_coeffs = fit_cp_coeffs(
                    lambda T, _id=species_id: c.calculate_properties(_id, T)["cp"],
                    T_lo,
                    T_hi,
                    n_fit_points,
                )

                Tb = boiling_points.get(name)
                Tc = critical_temps.get(name)
                antoine = _estimate_antoine_coeffs(name, Tb, Tc)
                hvap = _estimate_hvap_coeffs(Tb, Tc)

                result[name] = SpeciesData(
                    name=name,
                    MW=float(MW),
                    Cp_coeffs=Cp_coeffs,
                    Hvap_coeffs=hvap,
                    antoine_coeffs=antoine,
                    Hf=float(Hf),
                )
            except Exception as e:  # noqa: BLE001 - report and continue per species
                warnings.warn(f"pyglenn: could not import '{name}': {e}")

    return result
