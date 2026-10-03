"""One table of ideal-gas formation data for every refinery model compound (#339).

Every difflow_refinery module that needs a heat of formation, an absolute
entropy or an ideal-gas heat capacity for reaction thermochemistry reads it
here -- the reformer, isomerization, alkylation, the FCC regenerator, the
gas plant's heating values and (since #338) the hydrotreater, hydrocracker
and residue desulfurizer. Before #339 each kept its own copy, entered from a
different secondary source, and the copies disagreed (benzene 82.88-83.18,
cyclohexane -122.08 to -123.13, isopentane -153.7 to -154.5 kJ/mol), so the
same reaction had different heats and equilibria in different units.
``tests/refinery/test_thermochemistry.py`` keeps it that way: no module may
carry its own copy of a species that is in this table.

What is in a row (:class:`FormationData`), per species:

* ``Hf``: ideal-gas standard enthalpy of formation at 298.15 K (J/mol);
* ``S0``: ideal-gas absolute (third-law) entropy at 298.15 K and 1 bar
  (J/mol/K), or ``None`` where no defensible value was found;
* ``cp``: an ideal-gas heat-capacity cubic ``a + bT + cT^2 + dT^3`` (J/mol/K,
  T in K) and the range ``cp_range`` it was fitted over (298.15-1000 K for
  the organics, which covers a reformer at ~800 K and a hydrotreater; 298.15-
  1500 K for the regenerator gases N2, O2, H2O, CO, CO2, SO2);
* where each came from (``Hf_source``, ``S0_source``, ``cp_source``, keys of
  :data:`SOURCES`), a ``status`` of the formation enthalpy, and a ``note``
  wherever a choice was made between sources that disagree.

How the values were chosen
--------------------------

No primary source (the papers, the NIST WebBook, the ATcT or JANAF web
pages) could be opened from the environment this was written in. Values were
read from the data tables of the ``chemicals`` Python package (C. Bell et
al., version 1.5.2), which transcribe the compilations named below, and
compared across them; every value carries its compilation, and
:attr:`FormationData.status` says how far it was checked.

``Hf`` -- one rule, so that every reaction is computed from ONE consistent
evaluation wherever possible (isomer and ring/aromatic differences matter
more than absolute values):

1. Elements in their reference state (H2, N2, O2): zero.
2. Small inorganic molecules (H2O, CO, CO2, SO2, H2S, NH3): the CODATA Key
   Values for Thermodynamics (Cox, Wagman & Medvedev, Hemisphere, 1989). They
   agree with NIST-JANAF and ATcT 1.112 within 0.1 kJ/mol except NH3 (ATcT
   -45.56 against CODATA -45.94; CODATA kept, as JANAF and CRC also give
   -45.9).
3. Hydrocarbons and heteroatom organics: the API Technical Data Book ideal-gas
   values ("API TDB Albahri Hf (g)" in ``chemicals``; the Albahri compilation
   of the API TDB values -- edition unverified). It is the only compilation
   here that covers every hydrocarbon the units carry, so isomer
   differences are internally consistent; it is what the reformer has used
   since #309, and ChemSep (DWSIM's database) uses the same values for the
   C5/C6 paraffins. Cross-checked against CRC and ATcT 1.112 (``chemicals``).
   ATcT is the more accurate network where it exists, but covers only about
   a third of the species; mixing it in would mix evaluations within one
   reaction (benzene from ATcT with cyclohexane from API TDB gives a
   hydrogenation heat of -206.31 kJ/mol, equal to neither consistent set:
   API TDB -206.06, ATcT -205.26).
4. API TDB is overridden where it is absent, or where it disagrees with CRC
   by more than 2 kJ/mol and a third source sides with CRC (an outlier or an
   estimate): 2,3-dimethylpentane, cyclohexylbenzene, quinoline, carbazole.
   Where a third source sides with API TDB it is kept (2-methylnonane: the
   API TDB value is on the homologous series, CRC is 3.7 kJ/mol off it).
   Disagreements of 1-2 kJ/mol are recorded in :data:`HF_CROSSCHECK` and
   left at API TDB (2,2- and 2,3-dimethylbutane: CRC is 1.2 and 1.3 kJ/mol
   lower; the C6 isomerization equilibrium moves with them, see the docs).
5. No tabulated value: an estimate by this project, by group increments taken
   from the same table, marked ``"estimate"``.

Benzothiophene (issue #339 item 3): 166.3 kJ/mol. Sabbah (1979) gives
166.28 +- 0.48 and Good (1972) 166.6 kJ/mol from combustion calorimetry and
sublimation enthalpy, as the NIST WebBook lists them (read through a search
engine's summary of the WebBook page; the page itself was blocked -- so
unverified to the letter); CRC and Yaws give 166.3. ChemSep's 137.0 kJ/mol
(what DWSIM uses) matches no measurement found and is not used.

``S0`` -- Yaws, *Thermophysical Properties of Chemicals and Hydrocarbons*
(2nd ed., 2014; "Yaws Hf S0 (g)" in ``chemicals``) for every organic, so
the set is internally consistent, cross-checked against the CRC / NIST
WebBook values ``chemicals`` carries (:data:`S0_CROSSCHECK`; within 2.5
J/mol/K wherever both exist); CODATA for the inorganics. Yaws's
benzothiophene (212.76) and carbazole (244.95) are 100-150 J/mol/K below
their bicyclic/tricyclic analogues, a condensed-phase magnitude, and are not
used: those species have no ``S0`` (nothing needs one -- HDS and HDN are
irreversible). 1,2,3,4-tetrahydrophenanthrene's ``S0`` is an estimate
(phenanthrene + tetralin - naphthalene), which makes the poly -> di
aromatic-saturation entropy equal to the di -> mono one, as the hydrotreater
assumed before.

``cp`` -- cubics fitted by this project:

* ``"TRC"``: least squares on a 1 K grid over 298.15-1000 K to the TRC
  ideal-gas heat-capacity correlation (Frenkel et al., *Thermodynamics of
  Organic Compounds in the Gas State*, TRC, 1994, as tabulated in
  ``chemicals``);
* ``"TRC_REFORMER"``: the reformer's fits (#309) to the same correlation over
  the same range, kept as they were (they agree with a 1 K-grid refit to
  0.02 %);
* ``"JANAF"``: least squares to the NIST-JANAF tables (Chase, 4th ed., 1998)
  at their tabulated temperatures; ``"SHOMATE"``: to the NIST WebBook's
  Shomate equations, which are fits to the same JANAF tables (N2, O2);
* ``"JOBACK"``: the Joback (1984; Joback & Reid 1987) group-contribution
  cubic, for species no correlation covers (benzothiophene,
  dibenzothiophene, 4,6-DMDBT, cyclohexylbenzene, quinoline, indole,
  carbazole and an estimated product); on naphthalene, biphenyl and tetralin
  it is within 1.1, 2.3 and 2.9 % of TRC over 298-1000 K.

``cp_max_dev`` is the largest relative deviation of the cubic from what it
was fitted to, over ``cp_range``. Outside ``cp_range`` the cubic is an
extrapolation; nothing here clips or warns (that would not trace).

How to use it
-------------

::

    from difflow_refinery import thermochemistry as tc

    tc.Hf("benzene"), tc.S0("benzene")      # (82930.0, 269.18): J/mol, J/mol/K
    tc.species("benzene").cp                # Cp cubic (a, b, c, d), J/mol/K, T in K
    tc.enthalpy("benzene", 600.0)           # Hf + int Cp dT (J/mol)
    tc.entropy("benzene", 600.0, P=1e5)     # S0 + int Cp/T dT - R ln(P/P_STD) (J/mol/K)
    tc.gibbs("benzene", 600.0)              # H - T S at 1 bar (J/mol)
    nu = {"benzene": -1, "hydrogen": -3, "cyclohexane": 1}
    tc.reaction_enthalpy(nu)                # -206060.0 J per mol of reaction, at 298.15 K
    tc.ln_K(nu, 623.15)                     # 1 bar standard state, Cp-integrated

and, vectorised over a unit's species list (what a reactor wants)::

    gas = tc.IdealGasSet(("hydrogen", "benzene", "cyclohexane"))
    gas.enthalpy(T)       # (n,) J/mol      gas.entropy(T)  # (n,) J/mol/K, 1 bar
    gas.gibbs(T)          # (n,) J/mol      gas.ln_K(nu, T) # nu (n,) or (m, n)

Everything is a pure function of ``T`` in ``jax.numpy`` (differentiable, and
traceable under ``jit``); the table itself is concrete Python data, so it
imports without JAX tracing and without importing any unit module.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

__all__ = [
    "R", "T_REF", "P_STD", "SOURCES", "FormationData", "TABLE", "KEYS", "ALIASES",
    "HF_CROSSCHECK", "S0_CROSSCHECK", "species", "resolve", "Hf", "S0", "cp_coefficients", "cp",
    "sensible_enthalpy", "sensible_entropy", "enthalpy", "entropy", "gibbs", "elements",
    "check_balance", "reaction_enthalpy", "reaction_entropy", "reaction_gibbs", "ln_K",
    "IdealGasSet",
]

#: Gas constant (J/mol/K), CODATA 2018 exact.
R = 8.314462618
#: Reference temperature of ``Hf`` and ``S0`` (K).
T_REF = 298.15
#: Standard-state pressure of ``S0``, ``gibbs`` and every ``ln_K`` (Pa): 1 bar.
P_STD = 1.0e5

#: IUPAC conventional atomic weights (g/mol), CIAAW 2021.
ATOMIC_WEIGHT = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}

#: What each source key means. "(unverified)" in a row's note marks a value
#: that could not be traced to a primary source here.
SOURCES: Mapping[str, str] = MappingProxyType({
    "ELEMENT": "Element in its standard reference state: zero by definition.",
    "CODATA": ("Cox, J.D., Wagman, D.D. and Medvedev, V.A., CODATA Key Values for "
               "Thermodynamics, Hemisphere, New York (1989)."),
    "API_TDB": ("API Technical Data Book - Petroleum Refining, ideal-gas enthalpy of "
                "formation, as compiled by Albahri and tabulated in the chemicals package "
                "(Bell et al., v1.5.2, 'API TDB Albahri Hf (g).tsv'); edition unverified."),
    "CRC": ("CRC Handbook of Chemistry and Physics (Haynes, Bruno & Lide, 95th ed., 2014), "
            "'Standard thermodynamic properties of chemical substances', as tabulated in the "
            "chemicals package."),
    "NIST": ("NIST/TRC measurement as listed in the NIST Chemistry WebBook (Linstrom & "
             "Mallard, eds.), read through a search-engine summary: unverified."),
    "YAWS": ("Yaws, C.L., Thermophysical Properties of Chemicals and Hydrocarbons, 2nd ed., "
             "Gulf Professional Publishing (2014), as tabulated in the chemicals package "
             "('Yaws Hf S0 (g).tsv')."),
    "estimate": "Estimate by this project; the row's note gives the construction.",
    "none": "No defensible value: the quantity is absent (None).",
    "TRC": ("Cubic fitted by this project (least squares, 1 K grid, cp_range) to the TRC "
            "ideal-gas Cp correlation: Frenkel, M. et al., Thermodynamics of Organic Compounds "
            "in the Gas State, Thermodynamics Research Center, College Station (1994), as "
            "tabulated in the chemicals package."),
    "TRC_REFORMER": ("The reformer's (#309) cubic fit to the same TRC correlation over "
                     "298-1000 K, kept unchanged."),
    "JANAF": ("Cubic fitted by this project to the NIST-JANAF Thermochemical Tables, 4th ed. "
              "(Chase, M.W., J. Phys. Chem. Ref. Data Monograph 9, 1998), tabulated points "
              "in cp_range."),
    "SHOMATE": ("Cubic fitted by this project to the NIST Chemistry WebBook Shomate equations "
                "(fits to Chase 1998), 1 K grid over cp_range."),
    "JOBACK": ("Joback group-contribution ideal-gas Cp: Joback, K.G. and Reid, R.C., Chem. "
               "Eng. Commun. 57, 233-243 (1987), group values as in the thermo package; an "
               "estimate (1-3 % on tested polycyclic aromatics)."),
})

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


@dataclass(frozen=True)
class FormationData:
    """Ideal-gas formation data of one model compound.

    Attributes:
        key: Table key (the :mod:`difflow.database`-style name, ``"n_hexane"``).
        name: Chemical name.
        cas: CAS registry number ("" for an estimated, unregistered surrogate).
        formula: Hill formula (``"C6H6"``).
        Hf: Ideal-gas enthalpy of formation at 298.15 K (J/mol).
        Hf_source: Key of :data:`SOURCES`.
        S0: Ideal-gas absolute entropy at 298.15 K and 1 bar (J/mol/K), or None.
        S0_source: Key of :data:`SOURCES`.
        cp: Ideal-gas Cp cubic ``(a, b, c, d)``, J/mol/K with T in K.
        cp_range: ``(T_min, T_max)`` the cubic was fitted over (K).
        cp_source: Key of :data:`SOURCES`.
        cp_max_dev: Largest relative deviation of the cubic from its source
            over ``cp_range`` (None for a group-contribution cubic, which is
            its own source).
        note: Why a value was chosen where sources disagree.
    """

    key: str
    name: str
    cas: str
    formula: str
    Hf: float
    Hf_source: str
    S0: Optional[float]
    S0_source: str
    cp: tuple
    cp_range: tuple
    cp_source: str
    cp_max_dev: Optional[float]
    note: str = ""

    @property
    def elements(self) -> dict:
        """``{element: atoms}`` from the formula."""
        out: dict = {}
        for e, n in _FORMULA_RE.findall(self.formula):
            out[e] = out.get(e, 0) + int(n or 1)
        return out

    @property
    def MW(self) -> float:
        """Molar mass (g/mol) from the formula and IUPAC atomic weights."""
        return sum(ATOMIC_WEIGHT[e] * n for e, n in self.elements.items())

    @property
    def status(self) -> str:
        """How far ``Hf`` was checked: ``"definition"`` (an element),
        ``"key value"`` (CODATA), ``"cross-checked"`` (another independent
        compilation -- ATcT, CRC or API TDB, not the chosen one -- within
        1.5 kJ/mol), ``"estimate"``, or ``"unverified"``."""
        if self.Hf_source == "ELEMENT":
            return "definition"
        if self.Hf_source == "CODATA":
            return "key value"
        if self.Hf_source == "estimate":
            return "estimate"
        alts = HF_CROSSCHECK.get(self.key, {})
        indep = [v for k, v in alts.items()
                 if k in ("ATcT", "CRC", "API_TDB") and k != self.Hf_source]
        if any(abs(v * 1e3 - self.Hf) <= 1.5e3 for v in indep) and "(unverified)" not in self.note:
            return "cross-checked"
        return "unverified"


def _r(key, name, cas, formula, Hf_kJ, Hf_src, S0, S0_src, cp, cp_range, cp_src, cp_dev,
       note=""):
    return FormationData(key=key, name=name, cas=cas, formula=formula, Hf=round(Hf_kJ * 1e3, 6),
                         Hf_source=Hf_src, S0=S0, S0_source=S0_src, cp=tuple(cp),
                         cp_range=tuple(cp_range), cp_source=cp_src, cp_max_dev=cp_dev, note=note)


# -----------------------------------------------------------------------------
# The table. Columns: key, name, CAS, formula; Hf (kJ/mol) and its source;
# S0 (J/mol/K) and its source; Cp cubic, its range (K), its source, its
# largest relative deviation from that source; the note on any choice.
# Generated by this project from the chemicals 1.5.2 data files by the
# rules of the module docstring; edit a value only with its source.
# -----------------------------------------------------------------------------
_ROWS = (
    _r("hydrogen", "hydrogen", "1333-74-0", "H2",
       0.0, "ELEMENT", 130.68, "CODATA",
       (27.44, 0.00855236, -1.38696e-05, 8.18211e-09), (298.15, 1000.0), "TRC_REFORMER", 0.0049),
    _r("nitrogen", "nitrogen", "7727-37-9", "N2",
       0.0, "ELEMENT", 191.609, "CODATA",
       (29.21658, -0.003970918, 1.204881e-05, -4.643206e-09), (298.15, 1500.0), "SHOMATE", 0.0049),
    _r("oxygen", "oxygen", "7782-44-7", "O2",
       0.0, "ELEMENT", 205.152, "CODATA",
       (25.05484, 0.01510069, -5.90325e-06, 6.020373e-10), (298.15, 1500.0), "SHOMATE", 0.0114),
    _r("water", "water", "7732-18-5", "H2O",
       -241.826, "CODATA", 188.835, "CODATA",
       (32.43278, 0.0002808831, 1.300131e-05, -4.459023e-09), (298.15, 1500.0), "JANAF", 0.0023),
    _r("carbon_monoxide", "carbon monoxide", "630-08-0", "CO",
       -110.53, "CODATA", 197.66, "CODATA",
       (28.91528, -0.002388137, 1.108704e-05, -4.493443e-09), (298.15, 1500.0), "JANAF", 0.0046),
    _r("carbon_dioxide", "carbon dioxide", "124-38-9", "CO2",
       -393.51, "CODATA", 213.785, "CODATA",
       (21.47427, 0.06418226, -4.120753e-05, 9.909129e-09), (298.15, 1500.0), "JANAF", 0.0033),
    _r("sulfur_dioxide", "sulfur dioxide", "7446-09-5", "O2S",
       -296.81, "CODATA", 248.223, "CODATA",
       (24.78975, 0.06275112, -4.40398e-05, 1.104592e-08), (298.15, 1500.0), "JANAF", 0.0016),
    _r("hydrogen_sulfide", "hydrogen sulfide", "7783-06-4", "H2S",
       -20.6, "CODATA", 205.81, "CODATA",
       (32.04997, 0.0007138444, 2.527715e-05, -1.226284e-08), (298.15, 1000.0), "JANAF", 0.0004),
    _r("ammonia", "ammonia", "7664-41-7", "H3N",
       -45.94, "CODATA", 192.77, "CODATA",
       (27.15735, 0.02472246, 1.556473e-05, -1.098948e-08), (298.15, 1000.0), "JANAF", 0.0030),
    _r("methane", "methane", "74-82-8", "CH4",
       -74.52, "API_TDB", 186.6, "YAWS",
       (24.2384, 0.0207783, 6.75556e-05, -3.99814e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0079),
    _r("ethane", "ethane", "74-84-0", "C2H6",
       -83.85, "API_TDB", 229.45, "YAWS",
       (6.45789, 0.167124, -4.55258e-05, -5.67066e-09), (298.15, 1000.0), "TRC_REFORMER", 0.0079),
    _r("propane", "propane", "74-98-6", "C3H8",
       -104.69, "API_TDB", 270.28, "YAWS",
       (-4.32175, 0.301515, -0.000148719, 2.58421e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0079),
    _r("isobutane", "isobutane", "75-28-5", "C4H10",
       -134.99, "API_TDB", 295.34, "YAWS",
       (-13.2547, 0.435778, -0.000249866, 5.54883e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0079),
    _r("n_butane", "n-butane", "106-97-8", "C4H10",
       -125.65, "API_TDB", 304.4, "YAWS",
       (-1.92085, 0.38816, -0.000190209, 3.08733e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0086),
    _r("isopentane", "isopentane", "78-78-4", "C5H12",
       -153.7, "API_TDB", 343.89, "YAWS",
       (-9.47212, 0.499944, -0.000253443, 4.88841e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0046),
    _r("n_pentane", "n-pentane", "109-66-0", "C5H12",
       -146.71, "API_TDB", 349.25, "YAWS",
       (-4.11219, 0.475647, -0.000218834, 2.81896e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0092),
    _r("neopentane", "neopentane", "463-82-1", "C5H12",
       -168.07, "API_TDB", 305.99, "YAWS",
       (-4.000417, 0.4818834, -0.0002205837, 4.052638e-08), (298.15, 1000.0), "TRC", 0.0017),
    _r("n_hexane", "n-hexane", "110-54-3", "C6H14",
       -166.95, "API_TDB", 388.74, "YAWS",
       (-10.8684, 0.597481, -0.000308925, 5.27867e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0097),
    _r("2_methylpentane", "2-methylpentane", "107-83-5", "C6H14",
       -174.69, "API_TDB", 380.69, "YAWS",
       (-19.4918, 0.644242, -0.000375015, 8.7014e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0045),
    _r("3_methylpentane", "3-methylpentane", "96-14-0", "C6H14",
       -172.06, "API_TDB", 383.04, "YAWS",
       (-18.46845, 0.6260095, -0.0003447855, 7.284083e-08), (298.15, 1000.0), "TRC", 0.0046),
    _r("2_2_dimethylbutane", "2,2-dimethylbutane", "75-83-2", "C6H14",
       -184.68, "API_TDB", 358.22, "YAWS",
       (-15.3392, 0.6105846, -0.0003094691, 6.185449e-08), (298.15, 1000.0), "TRC", 0.0045),
    _r("2_3_dimethylbutane", "2,3-dimethylbutane", "79-29-8", "C6H14",
       -176.8, "API_TDB", 365.94, "YAWS",
       (-21.5962, 0.6412816, -0.0003670808, 8.811045e-08), (298.15, 1000.0), "TRC", 0.0006),
    _r("n_heptane", "n-heptane", "142-82-5", "C7H16",
       -187.65, "API_TDB", 428.23, "YAWS",
       (-17.2465, 0.718155, -0.000398277, 7.81453e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0100),
    _r("2_methylhexane", "2-methylhexane", "591-76-4", "C7H16",
       -194.5, "CRC", 420.52, "YAWS",
       (-23.1965, 0.749886, -0.00044417, 1.03967e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0051,
       'Hf: no API TDB entry; Yaws -194.6'),
    _r("2_3_dimethylpentane", "2,3-dimethylpentane", "565-59-3", "C7H16",
       -198.7, "CRC", 415.15, "YAWS",
       (-35.3966, 0.7961199, -0.0005030922, 1.33604e-07), (298.15, 1000.0), "TRC", 0.0002,
       'Hf: API TDB/Yaws -194.1 is 4.6 kJ/mol above CRC (-198.7) and the WebBook value the chemicals package carries (-199.0); taken as an API TDB outlier'),
    _r("2_4_dimethylpentane", "2,4-dimethylpentane", "108-08-7", "C7H16",
       -201.67, "API_TDB", 397.38, "YAWS",
       (-23.09101, 0.7930496, -0.0005215026, 1.455995e-07), (298.15, 1000.0), "TRC", 0.0006),
    _r("n_octane", "n-octane", "111-65-9", "C8H18",
       -208.82, "API_TDB", 467.05, "YAWS",
       (-21.9528, 0.829387, -0.000471684, 9.5135e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0097),
    _r("2_methylheptane", "2-methylheptane", "592-27-8", "C8H18",
       -215.35, "API_TDB", 459.34, "YAWS",
       (-29.8136, 0.869951, -0.000527505, 1.23802e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0067),
    _r("2_2_4_trimethylpentane", "2,2,4-trimethylpentane", "540-84-1", "C8H18",
       -224.01, "API_TDB", 423.11, "YAWS",
       (-27.88157, 0.8620196, -0.0004983181, 1.184593e-07), (298.15, 1000.0), "TRC", 0.0023),
    _r("2_3_4_trimethylpentane", "2,3,4-trimethylpentane", "565-75-3", "C8H18",
       -217.32, "API_TDB", 428.48, "YAWS",
       (-24.38501, 0.8806812, -0.0005642392, 1.555672e-07), (298.15, 1000.0), "TRC", 0.0030),
    _r("2_5_dimethylhexane", "2,5-dimethylhexane", "592-13-2", "C8H18",
       -222.51, "API_TDB", 442.57, "YAWS",
       (-35.60048, 0.8938265, -0.000560112, 1.432136e-07), (298.15, 1000.0), "TRC", 0.0030),
    _r("n_nonane", "n-nonane", "111-84-2", "C9H20",
       -228.86, "API_TDB", 507.08, "YAWS",
       (-26.8245, 0.941338, -0.00054556, 1.11725e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0100),
    _r("2_methyloctane", "2-methyloctane", "3221-61-2", "C9H20",
       -235.85, "API_TDB", 499.16, "YAWS",
       (-35.2104, 0.984259, -0.00060728, 1.44772e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0067,
       'Hf: API TDB only (Yaws, which follows it, -235.9); on the 2-methylalkane series, -20.5 kJ/mol per CH2 from 2-methylheptane (unverified)'),
    _r("2_2_5_trimethylhexane", "2,2,5-trimethylhexane", "3522-94-9", "C9H20",
       -253.3, "API_TDB", 461.93, "YAWS",
       (-41.63028, 1.005219, -0.0006186863, 1.583727e-07), (298.15, 1000.0), "TRC", 0.0026,
       'Hf: API TDB and Yaws only; the WebBook value chemicals carries is -253.1 (unverified)'),
    _r("n_decane", "n-decane", "124-18-5", "C10H22",
       -249.53, "API_TDB", 546.36, "YAWS",
       (-34.0257, 1.06631, -0.00064092, 1.39661e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0103),
    _r("2_methylnonane", "2-methylnonane", "871-83-0", "C10H22",
       -256.52, "API_TDB", 539.32, "YAWS",
       (-37.3414, 1.08403, -0.000661239, 1.51019e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0060,
       'Hf: CRC -260.2 is 3.7 kJ/mol off the 2-methylalkane series (2-methyloctane - 20.6 per CH2 gives -256.5); API TDB kept (unverified)'),
    _r("n_dodecane", "n-dodecane", "112-40-3", "C12H26",
       -290.79, "API_TDB", 625.21, "YAWS",
       (-44.72979, 1.29526, -0.0007952779, 1.761652e-07), (298.15, 1000.0), "TRC", 0.0103),
    _r("ethylene", "ethylene", "74-85-1", "C2H4",
       52.28, "API_TDB", 219.18, "YAWS",
       (0.831622, 0.1656872, -9.672814e-05, 2.405523e-08), (298.15, 1000.0), "TRC", 0.0143),
    _r("propylene", "propylene", "115-07-1", "C3H6",
       19.71, "API_TDB", 266.71, "YAWS",
       (3.536555, 0.2344359, -0.0001117289, 1.806397e-08), (298.15, 1000.0), "TRC", 0.0059),
    _r("1_butene", "1-butene", "106-98-9", "C4H8",
       0.1, "CRC", 307.88, "YAWS",
       (-2.180126, 0.3441387, -0.000182105, 3.613238e-08), (298.15, 1000.0), "TRC", 0.0053,
       'Hf: no API TDB entry; ATcT -0.03, Yaws -0.5, Prosen-Maron-Rossini 1951 -0.63 (as formerly in the gas plant); CRC chosen, ATcT supports it'),
    _r("cis_2_butene", "cis-2-butene", "590-18-1", "C4H8",
       -6.99, "API_TDB", 301.17, "YAWS",
       (-4.870494, 0.3248636, -0.0001454216, 1.853342e-08), (298.15, 1000.0), "TRC", 0.0082),
    _r("trans_2_butene", "trans-2-butene", "624-64-6", "C4H8",
       -11.17, "API_TDB", 296.48, "YAWS",
       (12.16769, 0.2818207, -0.0001019701, 3.021777e-09), (298.15, 1000.0), "TRC", 0.0061),
    _r("isobutylene", "isobutylene", "115-11-7", "C4H8",
       -16.9, "API_TDB", 293.12, "YAWS",
       (7.239664, 0.312252, -0.0001486434, 2.446902e-08), (298.15, 1000.0), "TRC", 0.0041),
    _r("1_pentene", "1-pentene", "109-67-1", "C5H10",
       -20.92, "API_TDB", 347.03, "YAWS",
       (-10.43572, 0.4730927, -0.0002812622, 6.47033e-08), (298.15, 1000.0), "TRC", 0.0077),
    _r("2_methyl_2_butene", "2-methyl-2-butene", "513-35-9", "C5H10",
       -42.55, "API_TDB", 338.65, "YAWS",
       (-5.280709, 0.4313747, -0.0002278808, 4.663109e-08), (298.15, 1000.0), "TRC", 0.0065),
    _r("1_hexene", "1-hexene", "592-41-6", "C6H12",
       -41.67, "API_TDB", 383.84, "YAWS",
       (-15.77158, 0.5878713, -0.0003595768, 8.373911e-08), (298.15, 1000.0), "TRC", 0.0081,
       'Hf: CRC -43.5 against API TDB -41.67; API TDB is on the 1-alkene series (1-pentene - 20.6 = -41.5) and close to Yaws and the WebBook value in chemicals (-42.0); kept (unverified)'),
    _r("methylcyclopentane", "methylcyclopentane", "96-37-7", "C6H12",
       -106.69, "API_TDB", 339.9, "YAWS",
       (-49.1407, 0.614133, -0.000301819, 3.991e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0121),
    _r("cyclohexane", "cyclohexane", "110-82-7", "C6H12",
       -123.13, "API_TDB", 297.31, "YAWS",
       (-59.2417, 0.630003, -0.000271783, 1.56173e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0140),
    _r("methylcyclohexane", "methylcyclohexane", "108-87-2", "C7H14",
       -154.77, "API_TDB", 343.5, "YAWS",
       (-59.4426, 0.779575, -0.000453102, 9.73564e-08), (298.15, 1000.0), "TRC_REFORMER", 0.0061),
    _r("ethylcyclohexane", "ethylcyclohexane", "1678-91-7", "C8H16",
       -171.75, "API_TDB", 382.99, "YAWS",
       (-51.4099, 0.857249, -0.000490642, 1.02072e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0059),
    _r("n_propylcyclohexane", "n-propylcyclohexane", "1678-92-8", "C9H18",
       -193.3, "API_TDB", 419.97, "YAWS",
       (-54.2228, 0.959797, -0.000550495, 1.12486e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0063),
    _r("n_butylcyclohexane", "n-butylcyclohexane", "1678-93-9", "C10H20",
       -213.17, "API_TDB", 459.59, "YAWS",
       (-57.1935, 1.06263, -0.000611202, 1.23725e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0065),
    _r("trans_decalin", "trans-decahydronaphthalene", "493-02-7", "C10H18",
       -182.1, "CRC", 373.89, "YAWS",
       (-125.9629, 1.204355, -0.0008358164, 2.433981e-07), (298.15, 1000.0), "TRC", 0.0196,
       'Hf: no API TDB entry; Yaws -182.1'),
    _r("benzene", "benzene", "71-43-2", "C6H6",
       82.93, "API_TDB", 269.18, "YAWS",
       (-48.1443, 0.547397, -0.000403617, 1.15717e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0035),
    _r("toluene", "toluene", "108-88-3", "C7H8",
       50.17, "API_TDB", 321.08, "YAWS",
       (-45.7735, 0.612717, -0.000413813, 1.07684e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0079),
    _r("ethylbenzene", "ethylbenzene", "100-41-4", "C8H10",
       29.79, "API_TDB", 361.24, "YAWS",
       (-47.2994, 0.712493, -0.000469658, 1.18049e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0081),
    _r("n_propylbenzene", "n-propylbenzene", "103-65-1", "C9H12",
       7.9, "API_TDB", 399.08, "YAWS",
       (-47.249, 0.814124, -0.000538192, 1.34796e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0084),
    _r("n_butylbenzene", "n-butylbenzene", "104-51-8", "C10H14",
       -13.14, "API_TDB", 440.28, "YAWS",
       (-49.5473, 0.90106, -0.000565273, 1.34063e-07), (298.15, 1000.0), "TRC_REFORMER", 0.0087),
    _r("cyclohexylbenzene", "cyclohexylbenzene", "827-52-1", "C12H16",
       -16.7, "CRC", 429.36, "YAWS",
       (-107.53, 1.187, -0.0007412, 1.667e-07), (298.15, 1000.0), "JOBACK", None,
       "Hf: API TDB/Yaws -20.92 (= -5.00 kcal/mol, an estimate's round number) against CRC -16.7; CRC chosen (unverified)"),
    _r("tetralin", "1,2,3,4-tetrahydronaphthalene", "119-64-2", "C10H12",
       26.61, "API_TDB", 366.22, "YAWS",
       (-83.09248, 0.9777167, -0.000696777, 1.963984e-07), (298.15, 1000.0), "TRC", 0.0052),
    _r("naphthalene", "naphthalene", "91-20-3", "C10H8",
       150.58, "API_TDB", 333.6, "YAWS",
       (-75.44665, 0.873975, -0.0006660575, 1.963721e-07), (298.15, 1000.0), "TRC", 0.0062),
    _r("biphenyl", "biphenyl", "92-52-4", "C12H10",
       182.09, "API_TDB", 391.24, "YAWS",
       (-80.50497, 1.037109, -0.0007909717, 2.361639e-07), (298.15, 1000.0), "TRC", 0.0036),
    _r("3_3_dimethylbiphenyl", "3,3'-dimethylbiphenyl", "612-75-9", "C14H14",
       116.57, "estimate", None, "none",
       (-68.18283, 1.142488, -0.000784635, 2.115048e-07), (298.15, 1000.0), "TRC", 0.0045,
       'Hf: no tabulated value: biphenyl + 2 x (toluene - benzene) (estimate by this project); S0: no tabulated ideal-gas value'),
    _r("methylcyclohexyltoluene", "3-(3-methylcyclohexyl)toluene", "", "C14H20",
       -81.1, "estimate", None, "none",
       (-89.11, 1.29104, -0.00072756, 1.374e-07), (298.15, 1000.0), "JOBACK", None,
       'Hf: no tabulated value: cyclohexylbenzene + (toluene - benzene) + (methylcyclohexane - cyclohexane) (estimate by this project); S0: no tabulated ideal-gas value'),
    _r("phenanthrene", "phenanthrene", "85-01-8", "C14H10",
       207.1, "API_TDB", 396.01, "YAWS",
       (-92.69448, 1.177497, -0.0009067871, 2.703063e-07), (298.15, 1000.0), "TRC", 0.0041),
    _r("tetrahydrophenanthrene", "1,2,3,4-tetrahydrophenanthrene", "1013-08-7", "C14H14",
       92.3, "NIST", 428.63, "estimate",
       (-99.56502, 1.237008, -0.000853078, 2.265193e-07), (298.15, 1000.0), "TRC", 0.0066,
       "Hf: 92.30 +- 1.30 kJ/mol, NIST TRC (Chirico, Steele et al., NIPER measurements; '1,2,3,4-Tetrahydrophenanthrene: Experimental and Derived Thermodynamic Properties', NIST data set) as reported by a search-engine summary of the NIST WebBook (unverified); S0: phenanthrene + (tetralin - naphthalene), Yaws values (estimate by this project): the di-ring step's dS"),
    _r("diethyl_sulfide", "diethyl sulfide", "352-93-2", "C4H10S",
       -83.47, "API_TDB", 368.32, "YAWS",
       (0.5921921, 0.4636281, -0.000281591, 6.808094e-08), (298.15, 1000.0), "TRC", 0.0084),
    _r("thiophene", "thiophene", "110-02-1", "C4H4S",
       114.9, "CRC", 278.81, "YAWS",
       (-31.77668, 0.4575949, -0.0003987942, 1.347712e-07), (298.15, 1000.0), "TRC", 0.0017,
       'Hf: no API TDB entry; Yaws 114.9 (ChemSep, in DWSIM, 115.44)'),
    _r("benzothiophene", "benzo[b]thiophene", "95-15-8", "C8H6S",
       166.3, "CRC", None, "none",
       (-50.57, 0.76121, -0.00065714, 2.251e-07), (298.15, 1000.0), "JOBACK", None,
       "Hf: Sabbah (1979) 166.28 +- 0.48 and Good (1972) 166.6 kJ/mol (combustion calorimetry + sublimation), as the NIST WebBook lists them (seen through a search-engine summary; the WebBook itself could not be opened: unverified); CRC and Yaws 166.3. ChemSep's 137.0 (in DWSIM) has no experimental support found and is rejected; S0: Yaws's 212.76 J/mol/K is 115 below indole (328.4) and naphthalene (333.1), a condensed-phase magnitude; rejected, no ideal-gas value"),
    _r("dibenzothiophene", "dibenzothiophene", "132-65-0", "C12H8S",
       205.1, "CRC", None, "none",
       (-71.35, 1.07801, -0.00094442, 3.289e-07), (298.15, 1000.0), "JOBACK", None,
       'Hf: no API TDB entry; calorimetric (unverified); S0: no tabulated ideal-gas value'),
    _r("4_6_dimethyldibenzothiophene", "4,6-dimethyldibenzothiophene", "1207-12-1", "C14H12S",
       139.58, "estimate", None, "none",
       (-44.57, 1.14905, -0.00091914, 3.029e-07), (298.15, 1000.0), "JOBACK", None,
       'Hf: no tabulated value: dibenzothiophene + 2 x (toluene - benzene), the aromatic C-methyl increment of the same API TDB table (estimate by this project); S0: no tabulated ideal-gas value'),
    _r("pyridine", "pyridine", "110-86-1", "C5H5N",
       140.16, "API_TDB", 282.5, "YAWS",
       (-43.71338, 0.5094745, -0.0003845633, 1.137059e-07), (298.15, 1000.0), "TRC", 0.0076),
    _r("quinoline", "quinoline", "91-22-5", "C9H7N",
       200.5, "CRC", 366.04, "YAWS",
       (-60.58, 0.80996, -0.00064298, 2.043e-07), (298.15, 1000.0), "JOBACK", None,
       'Hf: API TDB/Yaws 222.3 is an estimate; CRC 200.5 is the calorimetric value (Steele et al., NIPER, as in CRC; unverified); S0: (unverified): ~20 J/mol/K above naphthalene + R ln 4 (symmetry)'),
    _r("indole", "indole", "120-72-9", "C8H7N",
       156.6, "API_TDB", 328.44, "YAWS",
       (-55.47, 0.7334, -0.00057784, 1.834e-07), (298.15, 1000.0), "JOBACK", None),
    _r("carbazole", "carbazole", "86-74-8", "C12H9N",
       200.7, "CRC", None, "none",
       (-76.25, 1.0502, -0.00086512, 2.872e-07), (298.15, 1000.0), "JOBACK", None,
       "Hf: API TDB/Yaws 209.6 against CRC 200.7 (calorimetric); CRC chosen (unverified); S0: Yaws's 244.95 J/mol/K is ~150 below biphenyl (391.2); rejected, no ideal-gas value"),
    _r("aniline", "aniline", "62-53-3", "C6H7N",
       86.86, "API_TDB", 319.87, "YAWS",
       (-35.78656, 0.6058877, -0.0004551822, 1.331602e-07), (298.15, 1000.0), "TRC", 0.0020),
)

#: Ideal-gas Hf (kJ/mol) of each species in every compilation the chemicals
#: 1.5.2 tables carry (ATcT 1.112, API TDB, CRC, Yaws): the cross-check
#: behind each choice (see the module docstring).
HF_CROSSCHECK: dict[str, dict[str, float]] = {
    "hydrogen": {"ATcT": 0.0, "CRC": 0.0, "YAWS": 0.0},
    "nitrogen": {"ATcT": 0.0, "CRC": 0.0, "YAWS": 0.0},
    "oxygen": {"ATcT": 0.0, "CRC": 0.0, "YAWS": 0.0},
    "water": {"ATcT": -241.822, "API_TDB": -241.82, "CRC": -241.8, "YAWS": -241.83},
    "carbon_monoxide": {"ATcT": -110.525, "CRC": -110.5, "YAWS": -110.53},
    "carbon_dioxide": {"ATcT": -393.474, "API_TDB": -393.53, "CRC": -393.5, "YAWS": -393.52},
    "sulfur_dioxide": {"API_TDB": -296.85, "CRC": -296.8, "YAWS": -296.84},
    "hydrogen_sulfide": {"CRC": -20.6, "YAWS": -20.5},
    "ammonia": {"ATcT": -45.558, "CRC": -45.9, "YAWS": -45.9},
    "methane": {"ATcT": -74.534, "API_TDB": -74.52, "CRC": -74.6, "YAWS": -74.5},
    "ethane": {"ATcT": -83.78, "API_TDB": -83.85, "CRC": -84.0, "YAWS": -83.8},
    "propane": {"ATcT": -104.39, "API_TDB": -104.69, "CRC": -103.8, "YAWS": -104.7},
    "isobutane": {"ATcT": -135.36, "API_TDB": -134.99, "CRC": -134.2, "YAWS": -135.0},
    "n_butane": {"ATcT": -125.85, "API_TDB": -125.65, "CRC": -125.7, "YAWS": -126.8},
    "isopentane": {"API_TDB": -153.7, "CRC": -153.6, "YAWS": -153.7},
    "n_pentane": {"API_TDB": -146.71, "CRC": -146.9, "YAWS": -146.8},
    "neopentane": {"API_TDB": -168.07, "CRC": -168.0, "YAWS": -167.9},
    "n_hexane": {"ATcT": -166.94, "API_TDB": -166.95, "CRC": -166.9, "YAWS": -166.9},
    "2_methylpentane": {"API_TDB": -174.69, "CRC": -174.6, "YAWS": -174.6},
    "3_methylpentane": {"API_TDB": -172.06, "CRC": -171.9, "YAWS": -172.0},
    "2_2_dimethylbutane": {"API_TDB": -184.68, "CRC": -185.9, "YAWS": -184.0},
    "2_3_dimethylbutane": {"API_TDB": -176.8, "CRC": -178.1, "YAWS": -175.9},
    "n_heptane": {"ATcT": -187.34, "API_TDB": -187.65, "CRC": -187.6, "YAWS": -187.8},
    "2_methylhexane": {"CRC": -194.5, "YAWS": -194.6},
    "2_3_dimethylpentane": {"API_TDB": -194.1, "CRC": -198.7, "YAWS": -194.1},
    "2_4_dimethylpentane": {"API_TDB": -201.67, "CRC": -201.6, "YAWS": -201.7},
    "n_octane": {"ATcT": -208.22, "API_TDB": -208.82, "CRC": -208.5, "YAWS": -208.8},
    "2_methylheptane": {"API_TDB": -215.35, "CRC": -215.3, "YAWS": -215.4},
    "2_2_4_trimethylpentane": {"ATcT": -223.7, "API_TDB": -224.01, "CRC": -224.0, "YAWS": -224.0},
    "2_3_4_trimethylpentane": {"API_TDB": -217.32, "CRC": -217.3, "YAWS": -217.3},
    "2_5_dimethylhexane": {"API_TDB": -222.51, "CRC": -222.5, "YAWS": -222.5},
    "n_nonane": {"API_TDB": -228.86, "CRC": -228.2, "YAWS": -229.03},
    "2_methyloctane": {"API_TDB": -235.85, "YAWS": -235.9},
    "2_2_5_trimethylhexane": {"API_TDB": -253.3, "YAWS": -253.3},
    "n_decane": {"API_TDB": -249.53, "CRC": -249.5, "YAWS": -249.5},
    "2_methylnonane": {"API_TDB": -256.52, "CRC": -260.2, "YAWS": -256.5},
    "n_dodecane": {"API_TDB": -290.79, "CRC": -289.4, "YAWS": -290.83},
    "ethylene": {"ATcT": 52.56, "API_TDB": 52.28, "CRC": 52.4, "YAWS": 52.5},
    "propylene": {"ATcT": 20.37, "API_TDB": 19.71, "CRC": 20.0, "YAWS": 19.7},
    "1_butene": {"ATcT": -0.03, "CRC": 0.1, "YAWS": -0.5},
    "cis_2_butene": {"ATcT": -7.33, "API_TDB": -6.99, "CRC": -7.1, "YAWS": -7.4},
    "trans_2_butene": {"ATcT": -11.18, "API_TDB": -11.17, "CRC": -11.4, "YAWS": -11.0},
    "isobutylene": {"ATcT": -17.6, "API_TDB": -16.9, "CRC": -16.9, "YAWS": -17.1},
    "1_pentene": {"API_TDB": -20.92, "CRC": -21.1, "YAWS": -21.3},
    "2_methyl_2_butene": {"API_TDB": -42.55, "CRC": -41.7, "YAWS": -40.8},
    "1_hexene": {"API_TDB": -41.67, "CRC": -43.5, "YAWS": -42.0},
    "methylcyclopentane": {"API_TDB": -106.69, "CRC": -106.2, "YAWS": -106.0},
    "cyclohexane": {"ATcT": -122.08, "API_TDB": -123.13, "CRC": -123.4, "YAWS": -123.4},
    "methylcyclohexane": {"API_TDB": -154.77, "CRC": -154.7, "YAWS": -154.7},
    "ethylcyclohexane": {"API_TDB": -171.75, "CRC": -171.5, "YAWS": -171.6},
    "n_propylcyclohexane": {"API_TDB": -193.3, "CRC": -192.3, "YAWS": -193.2},
    "n_butylcyclohexane": {"API_TDB": -213.17, "CRC": -213.7, "YAWS": -213.17},
    "trans_decalin": {"CRC": -182.1, "YAWS": -182.1},
    "benzene": {"ATcT": 83.18, "API_TDB": 82.93, "CRC": 82.9, "YAWS": 82.9},
    "toluene": {"ATcT": 50.41, "API_TDB": 50.17, "CRC": 50.5, "YAWS": 50.2},
    "ethylbenzene": {"API_TDB": 29.79, "CRC": 29.9, "YAWS": 29.9},
    "n_propylbenzene": {"API_TDB": 7.9, "CRC": 7.9, "YAWS": 7.91},
    "n_butylbenzene": {"API_TDB": -13.14, "CRC": -11.8, "YAWS": -13.14},
    "cyclohexylbenzene": {"API_TDB": -20.92, "CRC": -16.7, "YAWS": -20.92},
    "tetralin": {"API_TDB": 26.61, "CRC": 26.0, "YAWS": 26.0},
    "naphthalene": {"API_TDB": 150.58, "CRC": 150.6, "YAWS": 150.6},
    "biphenyl": {"API_TDB": 182.09, "CRC": 181.4, "YAWS": 182.4},
    "phenanthrene": {"API_TDB": 207.1, "CRC": 207.5, "YAWS": 207.5},
    "diethyl_sulfide": {"API_TDB": -83.47, "CRC": -83.5, "YAWS": -83.2},
    "thiophene": {"CRC": 114.9, "YAWS": 114.9},
    "benzothiophene": {"CRC": 166.3, "YAWS": 166.3},
    "dibenzothiophene": {"CRC": 205.1},
    "pyridine": {"API_TDB": 140.16, "CRC": 140.4, "YAWS": 140.4},
    "quinoline": {"API_TDB": 222.3, "CRC": 200.5, "YAWS": 222.3},
    "indole": {"API_TDB": 156.6, "CRC": 156.5, "YAWS": 156.6},
    "carbazole": {"API_TDB": 209.6, "CRC": 200.7, "YAWS": 209.6},
    "aniline": {"API_TDB": 86.86, "CRC": 87.5, "YAWS": 86.86},
}

#: Ideal-gas S0 (J/mol/K) in every compilation the chemicals 1.5.2 tables
#: carry (CRC, NIST WebBook, JANAF, Yaws).
S0_CROSSCHECK: dict[str, dict[str, float]] = {
    "hydrogen": {"CRC": 130.7, "WEBBOOK": 130.68, "YAWS": 130.68},
    "nitrogen": {"CRC": 191.6, "WEBBOOK": 191.61, "YAWS": 191.61},
    "oxygen": {"CRC": 205.2, "WEBBOOK": 205.151, "YAWS": 205.15},
    "water": {"CRC": 188.8, "WEBBOOK": 188.838, "JANAF": 188.834, "YAWS": 188.84},
    "carbon_monoxide": {"CRC": 197.7, "WEBBOOK": 197.66, "JANAF": 197.653, "YAWS": 197.87},
    "carbon_dioxide": {"CRC": 213.8, "WEBBOOK": 213.788, "JANAF": 213.795, "YAWS": 213.91},
    "sulfur_dioxide": {"CRC": 248.2, "WEBBOOK": 248.214, "JANAF": 248.212, "YAWS": 248.22},
    "hydrogen_sulfide": {"CRC": 205.8, "WEBBOOK": 205.783, "JANAF": 205.757, "YAWS": 205.76},
    "ammonia": {"CRC": 192.8, "WEBBOOK": 192.77, "JANAF": 192.774, "YAWS": 192.78},
    "methane": {"CRC": 186.3, "WEBBOOK": 186.437, "JANAF": 186.251, "YAWS": 186.6},
    "ethane": {"CRC": 229.2, "YAWS": 229.45},
    "propane": {"CRC": 270.3, "YAWS": 270.28},
    "isobutane": {"YAWS": 295.34},
    "n_butane": {"YAWS": 304.4},
    "isopentane": {"YAWS": 343.89},
    "n_pentane": {"WEBBOOK": 347.82, "YAWS": 349.25},
    "neopentane": {"YAWS": 305.99},
    "n_hexane": {"WEBBOOK": 388.82, "YAWS": 388.74},
    "2_methylpentane": {"YAWS": 380.69},
    "3_methylpentane": {"WEBBOOK": 382.88, "YAWS": 383.04},
    "2_2_dimethylbutane": {"WEBBOOK": 358.65, "YAWS": 358.22},
    "2_3_dimethylbutane": {"YAWS": 365.94},
    "n_heptane": {"YAWS": 428.23},
    "2_methylhexane": {"WEBBOOK": 419.99, "YAWS": 420.52},
    "2_3_dimethylpentane": {"YAWS": 415.15},
    "2_4_dimethylpentane": {"WEBBOOK": 396.73, "YAWS": 397.38},
    "n_octane": {"WEBBOOK": 467.06, "YAWS": 467.05},
    "2_methylheptane": {"WEBBOOK": 459.49, "YAWS": 459.34},
    "2_2_4_trimethylpentane": {"YAWS": 423.11},
    "2_3_4_trimethylpentane": {"WEBBOOK": 427.2, "YAWS": 428.48},
    "2_5_dimethylhexane": {"YAWS": 442.57},
    "n_nonane": {"WEBBOOK": 506.5, "YAWS": 507.08},
    "2_methyloctane": {"YAWS": 499.16},
    "2_2_5_trimethylhexane": {"YAWS": 461.93},
    "n_decane": {"WEBBOOK": 545.8, "YAWS": 546.36},
    "2_methylnonane": {"YAWS": 539.32},
    "n_dodecane": {"WEBBOOK": 622.5, "YAWS": 625.21},
    "ethylene": {"CRC": 219.3, "WEBBOOK": 219.32, "JANAF": 219.33, "YAWS": 219.18},
    "propylene": {"YAWS": 266.71},
    "1_butene": {"YAWS": 307.88},
    "cis_2_butene": {"YAWS": 301.17},
    "trans_2_butene": {"YAWS": 296.48},
    "isobutylene": {"WEBBOOK": 293.59, "YAWS": 293.12},
    "1_pentene": {"YAWS": 347.03},
    "2_methyl_2_butene": {"YAWS": 338.65},
    "1_hexene": {"YAWS": 383.84},
    "methylcyclopentane": {"YAWS": 339.9},
    "cyclohexane": {"WEBBOOK": 298.19, "YAWS": 297.31},
    "methylcyclohexane": {"WEBBOOK": 343.3, "YAWS": 343.5},
    "ethylcyclohexane": {"WEBBOOK": 382.67, "YAWS": 382.99},
    "n_propylcyclohexane": {"WEBBOOK": 419.86, "YAWS": 419.97},
    "n_butylcyclohexane": {"WEBBOOK": 459.78, "YAWS": 459.59},
    "trans_decalin": {"YAWS": 373.89},
    "benzene": {"CRC": 269.2, "YAWS": 269.18},
    "toluene": {"YAWS": 321.08},
    "ethylbenzene": {"WEBBOOK": 360.6, "YAWS": 361.24},
    "n_propylbenzene": {"WEBBOOK": 397.86, "YAWS": 399.08},
    "n_butylbenzene": {"WEBBOOK": 437.86, "YAWS": 440.28},
    "cyclohexylbenzene": {"YAWS": 429.36},
    "tetralin": {"YAWS": 366.22},
    "naphthalene": {"CRC": 333.1, "YAWS": 333.6},
    "biphenyl": {"YAWS": 391.24},
    "phenanthrene": {"YAWS": 396.01},
    "diethyl_sulfide": {"CRC": 368.1, "YAWS": 368.32},
    "thiophene": {"CRC": 278.8, "YAWS": 278.81},
    "benzothiophene": {"YAWS": 212.76},
    "pyridine": {"YAWS": 282.5},
    "quinoline": {"YAWS": 366.04},
    "indole": {"YAWS": 328.44},
    "carbazole": {"YAWS": 244.95},
    "aniline": {"CRC": 317.9, "YAWS": 319.87},
}

#: Every species, by key (read-only).
TABLE: Mapping[str, FormationData] = MappingProxyType({row.key: row for row in _ROWS})
#: Table keys, in table order.
KEYS: tuple = tuple(TABLE)

#: Other names modules use for the same compound.
ALIASES: Mapping[str, str] = MappingProxyType({
    "H2": "hydrogen", "N2": "nitrogen", "O2": "oxygen", "H2O": "water", "steam": "water",
    "CO": "carbon_monoxide", "CO2": "carbon_dioxide", "SO2": "sulfur_dioxide",
    "H2S": "hydrogen_sulfide", "NH3": "ammonia",
    "propylbenzene": "n_propylbenzene", "butylbenzene": "n_butylbenzene",
    "isobutene": "isobutylene", "2_methylpropene": "isobutylene", "isooctane": "2_2_4_trimethylpentane",
    "1_2_3_4_tetrahydronaphthalene": "tetralin", "decalin": "trans_decalin",
    "1_2_3_4_tetrahydrophenanthrene": "tetrahydrophenanthrene",
    "dmdbt": "4_6_dimethyldibenzothiophene", "4_6_dmdbt": "4_6_dimethyldibenzothiophene",
    "dbt": "dibenzothiophene",
})


def resolve(name: str) -> str:
    """The table key of ``name`` (a key, an alias, or a name with dashes,
    commas or spaces for underscores)."""
    if name in TABLE:
        return name
    if name in ALIASES:
        return ALIASES[name]
    k = name.strip().lower().replace("-", "_").replace(",", "_").replace(" ", "_")
    k = ALIASES.get(k, k)
    if k in TABLE:
        return k
    raise KeyError(f"{name!r} is not in the refinery thermochemistry table; known: "
                   f"{', '.join(KEYS)}")


def species(name: str) -> FormationData:
    """The :class:`FormationData` row of ``name``."""
    return TABLE[resolve(name)]


def Hf(name: str) -> float:
    """Ideal-gas enthalpy of formation at 298.15 K (J/mol)."""
    return species(name).Hf


def S0(name: str) -> float:
    """Ideal-gas absolute entropy at 298.15 K, 1 bar (J/mol/K).

    Raises:
        ValueError: The table holds no defensible value for this species.
    """
    s = species(name)
    if s.S0 is None:
        raise ValueError(f"{s.key} has no ideal-gas S0 in the table ({s.note or s.S0_source})")
    return s.S0


def cp_coefficients(name: str) -> tuple:
    """Cp cubic ``(a, b, c, d)`` (J/mol/K, T in K)."""
    return species(name).cp


def elements(name: str) -> dict:
    """``{element: atoms}`` of one species."""
    return species(name).elements


# --- closed-form integrals of the cubic (vectorised over the coefficient rows)

def _cp(C, T):
    a, b, c, d = C[..., 0], C[..., 1], C[..., 2], C[..., 3]
    return a + b * T + c * T**2 + d * T**3


def _int_cp(C, T):
    a, b, c, d = C[..., 0], C[..., 1], C[..., 2], C[..., 3]

    def prim(t):
        return a * t + b * t**2 / 2 + c * t**3 / 3 + d * t**4 / 4

    return prim(T) - prim(T_REF)


def _int_cp_over_T(C, T):
    a, b, c, d = C[..., 0], C[..., 1], C[..., 2], C[..., 3]

    def prim(t):
        return a * jnp.log(t) + b * t + c * t**2 / 2 + d * t**3 / 3

    return prim(T) - prim(T_REF)


def cp(name: str, T):
    """Ideal-gas Cp (J/mol/K) of one species at ``T`` (K)."""
    return _cp(jnp.asarray(species(name).cp), jnp.asarray(T, dtype=float))


def sensible_enthalpy(name: str, T):
    """``int_298.15^T Cp dT`` (J/mol)."""
    return _int_cp(jnp.asarray(species(name).cp), jnp.asarray(T, dtype=float))


def sensible_entropy(name: str, T):
    """``int_298.15^T Cp/T dT`` (J/mol/K)."""
    return _int_cp_over_T(jnp.asarray(species(name).cp), jnp.asarray(T, dtype=float))


def enthalpy(name: str, T):
    """Ideal-gas molar enthalpy ``Hf + int Cp dT`` (J/mol; elements zero at 298.15 K)."""
    return species(name).Hf + sensible_enthalpy(name, T)


def entropy(name: str, T, P=P_STD):
    """Ideal-gas molar entropy ``S0 + int Cp/T dT - R ln(P/P_STD)`` (J/mol/K)."""
    T = jnp.asarray(T, dtype=float)
    return S0(name) + sensible_entropy(name, T) - R * jnp.log(jnp.asarray(P, dtype=float) / P_STD)


def gibbs(name: str, T):
    """Standard (1 bar) ideal-gas molar Gibbs energy ``H - T S`` (J/mol), on the
    basis "elements' enthalpy zero at 298.15 K, third-law entropy": element
    terms cancel in a balanced reaction, so differences are standard Gibbs
    energies of reaction."""
    T = jnp.asarray(T, dtype=float)
    return enthalpy(name, T) - T * (S0(name) + sensible_entropy(name, T))


# --- reactions given as {species: stoichiometric coefficient}

def check_balance(nu: Mapping[str, float], tol: float = 1e-9) -> None:
    """Raise ``ValueError`` unless the reaction ``nu`` conserves every element."""
    tot: dict = {}
    for k, n in nu.items():
        for e, a in elements(k).items():
            tot[e] = tot.get(e, 0.0) + float(n) * a
    bad = {e: v for e, v in tot.items() if abs(v) > tol}
    if bad:
        raise ValueError(f"reaction {dict(nu)} does not balance: {bad}")


def reaction_enthalpy(nu: Mapping[str, float], T=T_REF):
    """Standard enthalpy of reaction (J per mol of reaction as written) at ``T``."""
    check_balance(nu)
    return sum(float(n) * enthalpy(k, T) for k, n in nu.items())


def reaction_entropy(nu: Mapping[str, float], T=T_REF):
    """Standard (1 bar) entropy of reaction (J/K per mol of reaction) at ``T``."""
    check_balance(nu)
    T = jnp.asarray(T, dtype=float)
    return sum(float(n) * (S0(k) + sensible_entropy(k, T)) for k, n in nu.items())


def reaction_gibbs(nu: Mapping[str, float], T=T_REF):
    """Standard (1 bar) Gibbs energy of reaction (J per mol of reaction) at ``T``."""
    check_balance(nu)
    return sum(float(n) * gibbs(k, T) for k, n in nu.items())


def ln_K(nu: Mapping[str, float], T):
    """``ln K`` of the reaction at ``T``, ideal-gas standard state 1 bar
    (partial pressures in bar), Cp-integrated (not a constant dH, dS)."""
    T = jnp.asarray(T, dtype=float)
    return -reaction_gibbs(nu, T) / (R * T)


class IdealGasSet:
    """The table's data over a unit's species list, as arrays: what a reactor
    integrates. Concrete numpy arrays ``HF`` (J/mol), ``S0`` (J/mol/K, NaN
    where absent) and ``CP`` (``(n, 4)``); methods are pure in ``T`` and
    vectorise over the species (and over ``T``'s shape, leading).

    Args:
        names: Table keys or aliases, in the unit's array order.
    """

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        self.keys = tuple(resolve(n) for n in self.names)
        rows = [TABLE[k] for k in self.keys]
        self.HF = np.array([r.Hf for r in rows], dtype=float)
        self.S0 = np.array([np.nan if r.S0 is None else r.S0 for r in rows], dtype=float)
        self.CP = np.array([r.cp for r in rows], dtype=float)
        self.ELEMENTS = {e: np.array([r.elements.get(e, 0) for r in rows], dtype=float)
                         for e in ("C", "H", "N", "O", "S")}
        self.MW = np.array([r.MW for r in rows], dtype=float)

    def __len__(self) -> int:
        return len(self.keys)

    def _need_S0(self):
        missing = [k for k, s in zip(self.keys, self.S0) if math.isnan(s)]
        if missing:
            raise ValueError(f"no ideal-gas S0 in the table for {missing}: entropy, Gibbs "
                             "energy and equilibrium constants are undefined for them")

    def _T(self, T):
        return jnp.asarray(T, dtype=float)[..., None]

    def cp(self, T):
        """Cp of every species at ``T`` (J/mol/K), ``(..., n)``."""
        return _cp(jnp.asarray(self.CP), self._T(T))

    def sensible_enthalpy(self, T):
        """``int_298.15^T Cp dT`` of every species (J/mol), ``(..., n)``."""
        return _int_cp(jnp.asarray(self.CP), self._T(T))

    def sensible_entropy(self, T):
        """``int_298.15^T Cp/T dT`` of every species (J/mol/K), ``(..., n)``."""
        return _int_cp_over_T(jnp.asarray(self.CP), self._T(T))

    def enthalpy(self, T):
        """``Hf + int Cp dT`` of every species (J/mol), ``(..., n)``."""
        return jnp.asarray(self.HF) + self.sensible_enthalpy(T)

    def entropy(self, T):
        """Absolute entropy at 1 bar of every species (J/mol/K), ``(..., n)``."""
        self._need_S0()
        return jnp.asarray(self.S0) + self.sensible_entropy(T)

    def gibbs(self, T):
        """Standard (1 bar) ``H - T S`` of every species (J/mol), ``(..., n)``."""
        return self.enthalpy(T) - self._T(T) * self.entropy(T)

    def reaction_enthalpy(self, nu, T=T_REF):
        """``nu @ H(T)`` for stoichiometry rows ``nu`` (``(n,)`` or ``(m, n)``) at scalar ``T``."""
        return jnp.asarray(nu, dtype=float) @ self.enthalpy(jnp.asarray(T, dtype=float))

    def ln_K(self, nu, T):
        """``ln K`` (1 bar standard state) of stoichiometry rows ``nu`` at scalar ``T``."""
        T = jnp.asarray(T, dtype=float)
        return -(jnp.asarray(nu, dtype=float) @ self.gibbs(T)) / (R * T)
