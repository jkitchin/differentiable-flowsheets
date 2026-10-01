"""The real species of an alkylation unit and the properties it needs of them.

Every species here is a real molecule in :mod:`difflow.database` (critical
constants and ideal-gas data). This module adds what the alkylation model
needs on top -- the standard-volume basis that the Sauer-Colville-Burwick /
Bracken-McCormick correlations are written in, and the liquid-phase heats of
formation behind the heat of alkylation:

* **Formula** (``n_C``, ``n_H``): the atom balances are checked against it.
* **Liquid molar volume at 60 F** by COSTALD -- Hankinson & Thomson (1979),
  AIChE J. 25(4), 653-663, Eqs. 14-17 (``V_s/V* = V0 (1 - omega_SRK
  Vdelta)``; equation numbers unverified, coefficients checked against the
  implementation in the ``chemicals`` package, which cites the same paper).
  The characteristic volume ``V*`` and ``omega_SRK`` are the fitted COSTALD
  parameters as tabulated in the ``chemicals`` 1.5.2 data file
  "COSTALD Parameters.tsv" (from Hankinson & Thomson 1979; table number
  unverified). For the three species that file does not carry
  (2,3,4-trimethylpentane, 2,5-dimethylhexane, 2,2,5-trimethylhexane) ``V*``
  is fitted here so that COSTALD reproduces the CRC Handbook liquid density
  at 20 C, with ``omega_SRK`` set to the database acentric factor. The
  method is checked against GPA 2145 (propane 0.50736, isobutane 0.56293,
  n-butane 0.58407: COSTALD gives 0.5073, 0.5625, 0.5844).
* **Normal boiling point** (``Tb``) and **enthalpy of vaporisation at
  298.15 K** (``Hvap298``): CRC Handbook of Chemistry and Physics,
  "Enthalpy of Vaporization" table, as transcribed in the ``chemicals``
  1.5.2 file "CRC Handbook Heat of Vaporization.tsv" (edition unverified).
  Isobutylene has no CRC entry: Perry's Chemical Engineers' Handbook
  Table 2-150 (C1 = 32614 J/mol, C2 = 0.38073, Tc = 417.9 K) evaluated at
  298.15 K, the value the database's own note uses (edition unverified).
* **Ideal-gas heat of formation** at 298.15 K: :mod:`difflow.database`
  (``SpeciesData.Hf``), so there is one copy of it.

The liquid heat of formation is ``Hf(l) = Hf(g) - Hvap(298.15 K)`` (Hess's
law across the vaporisation at 298.15 K), which is what the heat of an
alkylation run in the liquid phase is made of.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from difflow.database import get_critical_props, get_species_data

# One standard barrel (m^3) and the density of water at 60 F (kg/m^3): the
# crude unit's own constants, so a barrel here is a barrel there.
from difflow_refinery.column import BARREL
from difflow_refinery.thermo import RHO_WATER_60F

__all__ = ["ALKYLATE", "ALKYLATION_SPECIES", "AlkySpecies", "ATOMIC_WEIGHT", "BARREL",
           "HEAVY_END", "INERTS", "OLEFINS", "RHO_WATER_60F", "T_60F",
           "costald_volume", "property_array", "species"]

#: Standard temperature for petroleum volumes: 60 F (K).
T_60F = 288.70555555555555

#: Atomic weights (IUPAC 2013 conventional values), for the atom balances.
ATOMIC_WEIGHT = {"C": 12.011, "H": 1.008}

# name: (n_C, n_H, Tb [K], Hvap(298.15 K) [J/mol], (V* [m^3/mol], omega_SRK) or
#        None, rho(20 C) [kg/m^3] or None)
#
# Tb, Hvap298: CRC (see module docstring); isobutylene Hvap298 from Perry's
# Table 2-150. V*, omega_SRK: Hankinson & Thomson (1979) COSTALD parameters
# (as in chemicals' "COSTALD Parameters.tsv"). rho(20 C): CRC Handbook
# "Physical Constants of Organic Compounds" (as in chemicals' file of that
# name); used only where COSTALD parameters are missing, and as a check.
_TABLE: dict[str, tuple] = {
    # inert paraffins
    "propane": (3, 8, 231.05, 14790.0, (2.001e-4, 0.1532), None),
    "isobutane": (4, 10, 261.42, 19230.0, (2.568e-4, 0.1825), None),
    "n_butane": (4, 10, 272.65, 21020.0, (2.544e-4, 0.2008), None),
    "isopentane": (5, 12, 301.03, 24850.0, (3.096e-4, 0.2400), 620.12),
    "n_pentane": (5, 12, 309.21, 26430.0, (3.113e-4, 0.2522), None),
    # olefins
    "propylene": (3, 6, 225.46, 14240.0, (1.829e-4, 0.1455), None),
    "1_butene": (4, 8, 266.89, 20220.0, (2.377e-4, 0.1921), None),
    "cis_2_butene": (4, 8, 276.86, 22160.0, (2.311e-4, 0.2039), None),
    "trans_2_butene": (4, 8, 274.03, 21400.0, (2.367e-4, 0.2153), None),
    "isobutylene": (4, 8, 266.25, 20265.0, (2.369e-4, 0.1959), None),
    "1_pentene": (5, 10, 303.11, 25470.0, (2.951e-4, 0.2824), 640.52),
    "2_methyl_2_butene": (5, 10, 311.71, 27060.0, (2.883e-4, 0.2852), 662.32),
    # alkylate
    "2_3_dimethylpentane": (7, 16, 362.93, 34260.0, (4.127e-4, 0.2973), 690.825),
    "2_4_dimethylpentane": (7, 16, 353.64, 32880.0, (4.251e-4, 0.3040), 672.72),
    "2_2_4_trimethylpentane": (8, 18, 372.37, 35140.0, (4.790e-4, 0.3045), 687.825),
    "2_3_4_trimethylpentane": (8, 18, 386.65, 37750.0, None, 719.12),
    "2_5_dimethylhexane": (8, 18, 382.27, 37850.0, None, 690.125),
    "2_2_5_trimethylhexane": (9, 20, 397.24, 40160.0, None, 707.22),
    "n_dodecane": (12, 26, 489.47, 61520.0, (7.558e-4, 0.5807), 749.52),
}

#: Every species the alkylation unit knows, in the order its streams use.
ALKYLATION_SPECIES: tuple[str, ...] = tuple(_TABLE)

#: The olefins and their carbon number. Alkylation of a C_n olefin with
#: isobutane gives a C_(n+4) paraffin.
OLEFINS: dict[str, int] = {
    "propylene": 3, "1_butene": 4, "cis_2_butene": 4, "trans_2_butene": 4,
    "isobutylene": 4, "1_pentene": 5, "2_methyl_2_butene": 5,
}

#: Species that pass through the reactor unchanged.
INERTS: tuple[str, ...] = ("propane", "n_butane", "isopentane", "n_pentane")

#: The alkylate species (everything the reactor makes).
ALKYLATE: tuple[str, ...] = (
    "2_3_dimethylpentane", "2_4_dimethylpentane", "2_2_4_trimethylpentane",
    "2_3_4_trimethylpentane", "2_5_dimethylhexane", "2_2_5_trimethylhexane",
    "n_dodecane",
)

#: The heavy-end lump. It is carried with n-dodecane's properties: the
#: dimer-alkylate heavy ends are C12 (2 C4= + iC4 -> C12H26), and n-dodecane
#: is the C12 paraffin every property table carries. It is a property
#: SURROGATE, not a claim that heavy ends are a normal paraffin; its octane
#: never enters the model (the alkylate octane is the correlation's).
HEAVY_END = "n_dodecane"


# COSTALD (Hankinson & Thomson 1979): V_s = V* V0 (1 - omega_SRK V_delta).
_COSTALD_A = (-1.52816, 1.43907, -0.81446, 0.190454)
_COSTALD_E = (-0.296123, 0.386914, -0.0427258, -0.0480645)


def costald_volume(T: float, Tc: float, v_star: float, omega_srk: float) -> float:
    """Saturated liquid molar volume (m^3/mol) by COSTALD.

    Hankinson & Thomson (1979), AIChE J. 25(4), 653-663:
    ``V0 = 1 + a (1-Tr)^(1/3) + b (1-Tr)^(2/3) + c (1-Tr) + d (1-Tr)^(4/3)``
    (``0.25 < Tr < 0.95``) and ``Vdelta = (e + f Tr + g Tr^2 + h Tr^3) /
    (Tr - 1.00001)``, with the coefficients in ``_COSTALD_A``/``_COSTALD_E``.

    Args:
        T: Temperature (K).
        Tc: Critical temperature (K).
        v_star: COSTALD characteristic volume (m^3/mol).
        omega_srk: COSTALD (SRK-optimised) acentric factor.
    """
    Tr = T / Tc
    tau = (1.0 - Tr) ** (1.0 / 3.0)
    a, b, c, d = _COSTALD_A
    e, f, g, h = _COSTALD_E
    v0 = 1.0 + a * tau + b * tau**2 + c * tau**3 + d * tau**4
    vd = (e + f * Tr + g * Tr**2 + h * Tr**3) / (Tr - 1.00001)
    return v_star * v0 * (1.0 - omega_srk * vd)


@dataclass(frozen=True)
class AlkySpecies:
    """What the alkylation model needs to know of one species.

    Attributes:
        name: Database name.
        n_C: Carbon atoms.
        n_H: Hydrogen atoms.
        MW: Molar mass (g/mol) from the formula and the IUPAC atomic
            weights (the database's value rounded to 0.01 g/mol agrees).
        Tc: Critical temperature (K), from :mod:`difflow.database`.
        omega: Acentric factor, from :mod:`difflow.database`.
        Pc: Critical pressure (Pa), from :mod:`difflow.database`.
        Tb: Normal boiling point (K), CRC.
        Hf_gas: Ideal-gas heat of formation at 298.15 K (J/mol), database.
        Hvap298: Enthalpy of vaporisation at 298.15 K (J/mol), CRC.
        v_star: COSTALD characteristic volume (m^3/mol).
        omega_srk: COSTALD acentric factor.
        v60: Liquid molar volume at 60 F (m^3/mol), COSTALD.
        sg60: Specific gravity 60/60 F.
        v_star_fitted: True where ``v_star`` was fitted to the CRC density.
    """

    name: str
    n_C: int
    n_H: int
    MW: float
    Tc: float
    Pc: float
    omega: float
    Tb: float
    Hf_gas: float
    Hvap298: float
    v_star: float
    omega_srk: float
    v60: float
    sg60: float
    v_star_fitted: bool

    @property
    def Hf_liquid(self) -> float:
        """Liquid heat of formation at 298.15 K (J/mol): ``Hf(g) - Hvap``."""
        return self.Hf_gas - self.Hvap298


@lru_cache(maxsize=None)
def species(name: str) -> AlkySpecies:
    """The :class:`AlkySpecies` record of ``name``."""
    if name not in _TABLE:
        raise KeyError(f"{name!r} is not an alkylation species; known: "
                       f"{', '.join(ALKYLATION_SPECIES)}")
    n_C, n_H, Tb, hvap, costald, rho20 = _TABLE[name]
    crit = get_critical_props(name)
    data = get_species_data(name)
    # The molar mass from the formula, not the database's two-decimal value:
    # the reactor conserves atoms exactly, and with rounded molar masses a
    # mass balance would close only to ~1e-5 (C5= + iC4 -> C9 is
    # 70.13 + 58.12 = 128.25 against 128.26 in the database).
    mw = n_C * ATOMIC_WEIGHT["C"] + n_H * ATOMIC_WEIGHT["H"]
    if costald is None:
        # One-point COSTALD: V* such that V_s(20 C) = MW / rho(20 C).
        omega_srk = crit.omega
        unit = costald_volume(293.15, crit.Tc, 1.0, omega_srk)
        v_star = mw / 1000.0 / rho20 / unit
        fitted = True
    else:
        v_star, omega_srk = costald
        fitted = False
    v60 = costald_volume(T_60F, crit.Tc, v_star, omega_srk)
    sg60 = mw / 1000.0 / v60 / RHO_WATER_60F
    return AlkySpecies(name=name, n_C=n_C, n_H=n_H, MW=mw, Tc=crit.Tc,
                       Pc=crit.Pc, omega=crit.omega, Tb=Tb, Hf_gas=data.Hf,
                       Hvap298=hvap, v_star=v_star, omega_srk=omega_srk,
                       v60=v60, sg60=sg60, v_star_fitted=fitted)


def property_array(names, attribute: str) -> np.ndarray:
    """``[species(n).<attribute> for n in names]`` as a float array."""
    return np.array([float(getattr(species(n), attribute)) for n in names])
