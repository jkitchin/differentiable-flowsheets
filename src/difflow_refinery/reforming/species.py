"""The reformer's species: lumps by carbon number, each a model compound.

The reaction network (:mod:`.kinetics`) runs on lumps by carbon number,
C6-C10, of four kinds -- normal paraffins ``nP<n>``, iso-paraffins ``iP<n>``,
naphthenes ``N<n>`` and aromatics ``A<n>`` -- plus methylcyclopentane
``N5_6`` (the C6 five-ring naphthene, which must isomerize to cyclohexane
before it can dehydrogenate to benzene), hydrogen and the C1-C5 paraffins the
cracking reactions make. That is a carbon-number (Krane-type) lumping with the
paraffins split by branching, because the octane of a paraffin lump depends
on it more than on anything else.

Every lump is represented by ONE real compound -- its *model compound* -- and
takes that compound's formula, thermochemistry, critical constants, density
and octane numbers (:data:`SPECIES`). The choices:

==========  =================  =====================  ===================
carbon no.  n-paraffin         iso-paraffin           naphthene / aromatic
==========  =================  =====================  ===================
C6          n-hexane           2-methylpentane        cyclohexane (+ MCP) / benzene
C7          n-heptane          2-methylhexane         methylcyclohexane / toluene
C8          n-octane           2-methylheptane        ethylcyclohexane / ethylbenzene
C9          n-nonane           2-methyloctane         n-propylcyclohexane / n-propylbenzene
C10         n-decane           2-methylnonane         n-butylcyclohexane / n-butylbenzene
==========  =================  =====================  ===================

Each naphthene/aromatic pair shares its side chain, so a dehydrogenation
equilibrium is the equilibrium of a real reaction (ethylcyclohexane =
ethylbenzene + 3 H2) rather than of an arbitrary pairing. The cost is that a
lump's free energy is one isomer's, not that of the equilibrium isomer
distribution the catalyst actually holds (a C8 aromatic product is mostly
xylenes, not ethylbenzene). That is the main simplification of the
thermodynamic layer, and it is stated, not hidden: a lump at internal isomer
equilibrium would have ``G_lump = -RT ln sum_i exp(-G_i/RT)`` and a lower
free energy than any one member.

Where the numbers come from (checked as described; "(unverified)" marks
what was not checked against a primary source):

* ``Hf``, ``S0`` and the ideal-gas Cp cubic are NOT kept here: they are the
  model compound's row of :mod:`difflow_refinery.thermochemistry`, the one
  table every refinery unit reads (#339), named per species by
  :attr:`ModelCompound.table_key`. For these species that table holds what
  this module held before #339 -- API Technical Data Book ``Hf`` (CRC for
  2-methylhexane), Yaws ``S0``, cubic fits over 298-1000 K to the TRC
  ideal-gas Cp correlation -- so the reformer's numbers did not move; the
  sources, the cross-checks and the choices where sources disagree are
  recorded there.
* Formula, molar mass, ``Tc``, ``Pc``, ``omega`` and ``Tb`` were read from
  the data tables shipped with the ``chemicals`` Python package (C. Bell et
  al., version 1.5.2), which transcribe the sources named per column below.
  No primary source could be opened from the environment this was written
  in (the NIST WebBook and publishers were unreachable).

  - ``Tc``, ``Pc``, ``omega``, ``Tb``: the first-ranked source in
    ``chemicals`` (CoolProp's reference EOS for the common species; the IUPAC
    critical-property review, CRC and PSRK tables for the rest).
  - ``rho60``: saturated liquid density at 60 F (288.71 K), kg/m^3: the
    DIPPR-105 equation of Perry's Chemical Engineers' Handbook, 8th ed.,
    Table 2-32 (as tabulated in ``chemicals``) where it has the species;
    VDI Heat Atlas PPDS equation for n-propyl- and n-butylcyclohexane;
    Hankinson-Thomson COSTALD with its fitted characteristic volume for
    2-methylhexane. For 2-methylheptane, 2-methyloctane and 2-methylnonane
    no tabulated equation was available and the corresponding-states
    estimates disagree by 5 %: the values are handbook 20 C densities
    recalled by this project, raised by 4 kg/m^3 to 60 F (unverified).
    Hydrogen and methane are supercritical at 60 F and have no liquid
    density; they never enter a liquid-volume yield.

* ``RON``/``MON``: research and motor octane numbers of the pure compound
  (ASTM D2699 / D2700). n-Heptane (0) and 2,2,4-trimethylpentane (100) are
  the primary reference fuels and exact by definition; isooctane is not a
  lump and appears only as that anchor. Every other value is a pure-compound
  octane number as compiled by API Research Project 45 (ASTM STP 225,
  *Knocking Characteristics of Pure Hydrocarbons*, 1958) and widely
  reproduced, RECALLED by this project and NOT checked against STP 225:
  all are (unverified). Where no value was recalled -- the C9 and C10
  paraffins and n-butylcyclohexane -- the number is an EXTRAPOLATION along
  the homologous series by this project (``octane_source="estimate"``),
  marked so per species. Pure-compound numbers are not blending numbers; the
  reformate's octane blends them with the Ethyl RT-70 rule
  (:func:`difflow_refinery.blending.ethyl_rt70`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from difflow_refinery import thermochemistry

#: Carbon numbers the lumps cover.
CARBON_NUMBERS: tuple[int, ...] = (6, 7, 8, 9, 10)

#: Light paraffins the cracking reactions make (and that ride in the recycle gas).
LIGHT_SPECIES: tuple[str, ...] = ("H2", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5")


@dataclass(frozen=True)
class ModelCompound:
    """One reformer species and the real compound that represents it.

    Attributes:
        key: Species name in the reformer (``"nP7"``, ``"A8"``, ``"H2"``...).
        compound: Name of the model compound.
        cas: Its CAS registry number.
        kind: ``"H2"``, ``"light"``, ``"nP"``, ``"iP"``, ``"N"`` or ``"A"``.
        carbon, hydrogen: Atoms per molecule.
        MW: Molar mass (g/mol).
        Hf: Ideal-gas standard enthalpy of formation at 298.15 K (J/mol).
        S0: Ideal-gas standard entropy at 298.15 K and 1 bar (J/mol/K).
        cp: Ideal-gas Cp cubic ``(a, b, c, d)``, J/mol/K, T in K, 298-1000 K.
        Tc, Pc, omega: Critical temperature (K), pressure (Pa), acentric factor.
        Tb: Normal boiling point (K).
        rho60: Liquid density at 60 F (kg/m^3), or ``nan`` when supercritical.
        RON, MON: Pure-compound research / motor octane numbers.
        octane_source: ``"reference"`` (a primary reference fuel, exact),
            ``"recalled"`` (API RP45 value as recalled, unverified) or
            ``"estimate"`` (extrapolated by this project).
        table_key: The model compound's key in
            :mod:`difflow_refinery.thermochemistry`, where ``Hf``, ``S0`` and
            ``cp`` come from.
    """

    key: str
    compound: str
    cas: str
    kind: str
    carbon: int
    hydrogen: int
    MW: float
    Hf: float
    S0: float
    cp: tuple[float, float, float, float]
    Tc: float
    Pc: float
    omega: float
    Tb: float
    rho60: float
    RON: float
    MON: float
    octane_source: Literal["reference", "recalled", "estimate"] = "recalled"
    table_key: str = ""


def _S(key, table_key, compound, cas, kind, C, H, MW, Tc, Pc, omega, Tb, rho60, RON, MON,
       src="recalled"):
    # Hf, S0 and the Cp cubic are the shared table's (#339), never a copy.
    f = thermochemistry.species(table_key)
    return ModelCompound(key, compound, cas, kind, C, H, MW, f.Hf, f.S0, tuple(f.cp), Tc, Pc,
                         omega, Tb, rho60, RON, MON, src, table_key)


_NAN = float("nan")

#: Every reformer species, in the order of every flow array.
#: See the module docstring for where each column comes from.
SPECIES: dict[str, ModelCompound] = {s.key: s for s in [
    # key, table key, compound, CAS, kind, C, H, MW; Tc, Pc, omega, Tb, rho60, RON, MON
    _S("H2", "hydrogen", "hydrogen", "1333-74-0", "H2", 0, 2, 2.01588,
       33.145, 1296400.0, -0.2190, 20.37, _NAN, _NAN, _NAN),
    _S("C1", "methane", "methane", "74-82-8", "light", 1, 4, 16.04246,
       190.564, 4599200.0, 0.0114, 111.67, _NAN, _NAN, _NAN),
    _S("C2", "ethane", "ethane", "74-84-0", "light", 2, 6, 30.06904,
       305.322, 4872200.0, 0.0995, 184.57, 355.0, _NAN, _NAN),
    _S("C3", "propane", "propane", "74-98-6", "light", 3, 8, 44.09562,
       369.890, 4251200.0, 0.1521, 231.04, 505.8, _NAN, _NAN),
    _S("iC4", "isobutane", "isobutane", "75-28-5", "light", 4, 10, 58.12220,
       407.810, 3629000.0, 0.1840, 261.40, 563.2, 102.1, 97.6),
    _S("nC4", "n_butane", "n-butane", "106-97-8", "light", 4, 10, 58.12220,
       425.125, 3796000.0, 0.2010, 272.66, 584.3, 93.8, 89.6),
    _S("iC5", "isopentane", "isopentane", "78-78-4", "light", 5, 12, 72.14878,
       460.350, 3378000.0, 0.2274, 300.98, 626.0, 92.3, 90.3),
    _S("nC5", "n_pentane", "n-pentane", "109-66-0", "light", 5, 12, 72.14878,
       469.700, 3367500.0, 0.2510, 309.21, 631.1, 61.7, 62.6),
    # C6
    _S("nP6", "n_hexane", "n-hexane", "110-54-3", "nP", 6, 14, 86.17536,
       507.820, 3044100.0, 0.3000, 341.87, 664.4, 24.8, 26.0),
    _S("iP6", "2_methylpentane", "2-methylpentane", "107-83-5", "iP", 6, 14, 86.17536,
       497.700, 3040000.0, 0.2797, 333.36, 657.0, 73.4, 73.5),
    _S("N5_6", "methylcyclopentane", "methylcyclopentane", "96-37-7", "N", 6, 12, 84.15948,
       553.800, 4080000.0, 0.2390, 344.95, 753.2, 91.3, 80.0),
    _S("N6", "cyclohexane", "cyclohexane", "110-82-7", "N", 6, 12, 84.15948,
       553.600, 4080500.0, 0.2096, 353.86, 781.5, 83.0, 77.2),
    _S("A6", "benzene", "benzene", "71-43-2", "A", 6, 6, 78.11184,
       562.020, 4907277.0, 0.2110, 353.22, 882.4, 102.7, 105.0),
    # C7
    _S("nP7", "n_heptane", "n-heptane", "142-82-5", "nP", 7, 16, 100.20194,
       540.200, 2735730.0, 0.3490, 371.55, 689.5, 0.0, 0.0, "reference"),
    _S("iP7", "2_methylhexane", "2-methylhexane", "591-76-4", "iP", 7, 16, 100.20194,
       530.400, 2740000.0, 0.3300, 363.15, 683.0, 42.4, 46.4),
    _S("N7", "methylcyclohexane", "methylcyclohexane", "108-87-2", "N", 7, 14, 98.18606,
       572.200, 3470000.0, 0.2340, 374.01, 774.0, 74.8, 71.1),
    _S("A7", "toluene", "toluene", "108-88-3", "A", 7, 8, 92.13842,
       591.750, 4126300.0, 0.2657, 383.75, 872.5, 120.0, 103.5),
    # C8
    _S("nP8", "n_octane", "n-octane", "111-65-9", "nP", 8, 18, 114.22852,
       568.740, 2483590.0, 0.3980, 398.79, 710.3, -19.0, -15.0),
    _S("iP8", "2_methylheptane", "2-methylheptane", "592-27-8", "iP", 8, 18, 114.22852,
       559.700, 2500000.0, 0.3780, 390.75, 702.0, 21.7, 23.8),
    _S("N8", "ethylcyclohexane", "ethylcyclohexane", "1678-91-7", "N", 8, 16, 112.21264,
       606.900, 3270000.0, 0.2444, 404.95, 791.8, 45.6, 40.8),
    _S("A8", "ethylbenzene", "ethylbenzene", "100-41-4", "A", 8, 10, 106.16500,
       617.120, 3622400.0, 0.3050, 409.31, 871.6, 107.4, 97.9),
    # C9
    _S("nP9", "n_nonane", "n-nonane", "111-84-2", "nP", 9, 20, 128.25510,
       594.550, 2281000.0, 0.4433, 423.91, 724.2, -40.0, -35.0, "estimate"),
    _S("iP9", "2_methyloctane", "2-methyloctane", "3221-61-2", "iP", 9, 20, 128.25510,
       582.800, 2310000.0, 0.4499, 416.15, 717.0, 0.0, 5.0, "estimate"),
    _S("N9", "n_propylcyclohexane", "n-propylcyclohexane", "1678-92-8", "N", 9, 18, 126.23922,
       630.800, 2860000.0, 0.3260, 429.86, 797.7, 17.8, 14.0),
    _S("A9", "n_propylbenzene", "n-propylbenzene", "103-65-1", "A", 9, 12, 120.19158,
       638.350, 3200000.0, 0.3440, 432.35, 866.7, 111.0, 98.7),
    # C10
    _S("nP10", "n_decane", "n-decane", "124-18-5", "nP", 10, 22, 142.28168,
       617.700, 2103000.0, 0.4884, 447.27, 734.9, -50.0, -45.0, "estimate"),
    _S("iP10", "2_methylnonane", "2-methylnonane", "871-83-0", "iP", 10, 22, 142.28168,
       610.700, 2120000.0, 0.4720, 440.15, 730.0, -10.0, -5.0, "estimate"),
    _S("N10", "n_butylcyclohexane", "n-butylcyclohexane", "1678-93-9", "N", 10, 20, 140.26580,
       653.100, 2570000.0, 0.3524, 454.05, 802.9, 0.0, 0.0, "estimate"),
    _S("A10", "n_butylbenzene", "n-butylbenzene", "104-51-8", "A", 10, 14, 134.21816,
       660.500, 2890000.0, 0.3920, 456.45, 864.6, 104.4, 94.5),
]}

#: Species names, in flow-array order.
NAMES: tuple[str, ...] = tuple(SPECIES)
N_SPECIES = len(NAMES)
INDEX: dict[str, int] = {k: i for i, k in enumerate(NAMES)}


def kind_of(key: str) -> str:
    return SPECIES[key].kind


def lump(kind: str, n: int) -> str:
    """Name of the ``kind`` lump with ``n`` carbons (``lump("A", 7) == "A7"``)."""
    if kind not in ("nP", "iP", "N", "A"):
        raise ValueError(f"kind must be nP, iP, N or A, not {kind!r}")
    if n not in CARBON_NUMBERS:
        raise ValueError(f"carbon number must be in {CARBON_NUMBERS}, not {n}")
    return f"{kind}{n}"


def paraffin(n: int, iso: bool) -> str:
    """The paraffin species with ``n`` carbons (n <= 3 has no isomer)."""
    if n == 1:
        return "C1"
    if n == 2:
        return "C2"
    if n == 3:
        return "C3"
    if n in (4, 5):
        return f"{'i' if iso else 'n'}C{n}"
    return f"{'iP' if iso else 'nP'}{n}"


def array(attr: str) -> np.ndarray:
    """``(N_SPECIES,)`` array of one attribute of every species."""
    return np.array([getattr(SPECIES[k], attr) for k in NAMES], dtype=float)


#: Element matrix ``(2, N_SPECIES)``: rows carbon, hydrogen atoms.
ELEMENTS = np.stack([array("carbon"), array("hydrogen")])
MW = array("MW")
HF = array("Hf")
S0 = array("S0")
CP = np.array([SPECIES[k].cp for k in NAMES], dtype=float)
TC, PC, OMEGA, TB = array("Tc"), array("Pc"), array("omega"), array("Tb")
RHO60 = array("rho60")
RON, MON = array("RON"), array("MON")

#: Species that end up in the C5+ reformate (everything with five carbons or more).
C5_PLUS = tuple(k for k in NAMES if SPECIES[k].carbon >= 5)
#: Aromatic species.
AROMATICS = tuple(k for k in NAMES if SPECIES[k].kind == "A")
