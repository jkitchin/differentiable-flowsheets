"""Ideal-gas thermochemistry and pure-component octane data for C5/C6
isomerization.

Every species the reactor carries is in :data:`SPECIES`, with its
heat of formation and absolute entropy (ideal gas, 298.15 K, 1 bar), an
ideal-gas Cp cubic, its formula, and its blending properties. The
equilibrium constants of every reaction follow from these alone::

    G_i(T) = dHf_i + int_298^T Cp_i dT - T (S_i + int_298^T Cp_i / T dT)
    ln K_j(T) = -sum_i nu_ij G_i(T) / (R T)

The elements' enthalpies are zero at 298.15 K and their entropies are in
``S_i`` (absolute entropies), so the element terms cancel in any balanced
reaction and ``G_i`` needs no element correction.

Sources
    dHf (gas, 298.15 K), as the NIST WebBook gives them: the paraffins from
    Prosen & Rossini, *J. Res. NBS* 34, 263 (1945) -- the API Project 44
    values; methylcyclopentane and cyclohexane from Prosen, Johnson &
    Rossini, *J. Res. NBS* 37, 51 (1946); benzene from Prosen, Gilmont &
    Rossini, *J. Res. NBS* 34, 65 (1945); ethane, propane and the C4 and
    C7 paraffins (which only carry heat here) from Yaws (1999) or
    :mod:`difflow.database`. S (gas, 298.15 K, 1 bar): Yaws, *Chemical
    Properties Handbook* (1999), as the ``chemicals`` package (Bell et al.)
    transcribes it; the NIST WebBook's tabulated entropies (Scott 1974 for
    the paraffins, Dorofeeva 1986 for cyclohexane) agree to 1.5 J/mol/K or
    better wherever both exist. Ideal-gas Cp: cubics fitted here to the
    NIST WebBook gas tables at 273.15-800 K (Scott 1974 for the C5-C7
    paraffins; TRC 1997 for methylcyclopentane and benzene; Dorofeeva et
    al. 1986 for cyclohexane), worst point 0.5%; hydrogen and the C2-C4
    paraffins from :mod:`difflow.database` / the gas plant's table.

    The equilibrium is sensitive to these numbers: 0.5 kJ/mol in one
    isomer's free energy moves its equilibrium share by about 15% at
    420 K. No test compares the free energies derived here with a
    tabulated set (API Project 44's, say); that check is still to do.

Octane numbers
    Research and motor octane numbers of the pure hydrocarbons, from API
    Research Project 45 as compiled in ASTM STP 225, *Knocking
    Characteristics of Pure Hydrocarbons* (1958). **They were recalled, not
    checked against the printed tables, and are marked verify.** Benzene's
    MON and the C7+ lump's octanes are assumptions (marked as such): the
    lump stands for the methylhexanes, dimethylpentanes and C7 naphthenes
    that boil with a light naphtha's tail, which is not one molecule.
    Specific gravities (60/60 F) are the GPA 2145 values where
    :data:`difflow_refinery.assay.LIGHT_END_SG` holds them, and recalled
    (verify) otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

#: Gas constant (J/mol/K).
R = 8.314462618
#: Reference temperature (K).
T0 = 298.15
#: Standard pressure of the entropies and equilibrium constants (Pa).
P_STD = 1.0e5

_V = "verify"


@dataclass(frozen=True)
class IsomSpecies:
    """One species of the isomerization reactor.

    Attributes:
        name: :mod:`difflow.database` name.
        label: Short label (``"22DMB"``).
        C: Carbon atoms.
        H: Hydrogen atoms.
        Hf: Heat of formation, ideal gas, 298.15 K (J/mol).
        S: Absolute entropy, ideal gas, 298.15 K, 1 bar (J/mol/K).
        cp: Ideal-gas Cp cubic ``(a, b, c, d)``, J/mol/K with T in K.
        RON: Research octane number (pure).
        MON: Motor octane number (pure).
        SG: Liquid specific gravity, 60/60 F.
        aromatic: 1.0 for an aromatic.
        note: Provenance flag for the octane data (``"verify"`` or an
            explanation of an assumption).
    """

    name: str
    label: str
    C: int
    H: int
    Hf: float
    S: float
    cp: tuple
    RON: float
    MON: float
    SG: float
    aromatic: float = 0.0
    note: str = _V

    @property
    def MW(self) -> float:
        return 12.011 * self.C + 1.008 * self.H


# Cp cubics. NIST-table fits (this module's docstring) for the reacting
# species; hydrogen as gasplant.CP_IG; C2-C4 as difflow.database.
_CP = {
    "hydrogen": (27.14, 9.274e-3, -1.381e-5, 7.645e-9),
    "ethane": (5.409, 1.781e-1, -6.938e-5, 8.713e-9),
    "propane": (-4.224, 3.063e-1, -1.586e-4, 3.215e-8),
    "isobutane": (-1.39, 3.847e-1, -1.846e-4, 2.895e-8),
    "n_butane": (9.487, 3.313e-1, -1.108e-4, -2.822e-9),
    "isopentane": (-1.1224, 0.4517, -1.6418e-4, -4.0841e-9),
    "n_pentane": (10.67, 0.39135, -6.5389e-5, -6.1227e-8),
    "2_2_dimethylbutane": (-3.1491, 0.54281, -1.8902e-4, -6.8825e-9),
    "2_3_dimethylbutane": (-20.012, 0.6334, -3.5449e-4, 8.1562e-8),
    "2_methylpentane": (-6.882, 0.57438, -2.5124e-4, 1.6512e-8),
    "3_methylpentane": (-4.5919, 0.54702, -2.0104e-4, -1.1154e-8),
    "n_hexane": (8.3803, 0.48881, -1.1298e-4, -6.0572e-8),
    "methylcyclopentane": (-35.499, 0.53993, -1.5819e-4, -5.2279e-8),
    "cyclohexane": (-30.506, 0.46222, 3.0128e-5, -1.5954e-7),
    "benzene": (-41.768, 0.51126, -3.3693e-4, 7.5704e-8),
    "n_heptane": (16.089, 0.53113, -6.3869e-5, -1.1342e-7),
}

#: Every species, in reactor order (also the column component order).
SPECIES: tuple[IsomSpecies, ...] = (
    IsomSpecies("hydrogen", "H2", 0, 2, 0.0, 130.68, _CP["hydrogen"], 0.0, 0.0, 1.0,
                note="gas; SG is a placeholder, never used"),
    IsomSpecies("ethane", "C2", 2, 6, -83.8e3, 229.45, _CP["ethane"], 0.0, 0.0, 0.35584,
                note="gas; never in the isomerate, so no octanes"),
    IsomSpecies("propane", "C3", 3, 8, -104.7e3, 270.28, _CP["propane"], 0.0, 0.0, 0.50736,
                note="gas; never in the isomerate, so no octanes"),
    IsomSpecies("isobutane", "iC4", 4, 10, -134.2e3, 295.34, _CP["isobutane"], 102.1, 97.6,
                0.56293),
    IsomSpecies("n_butane", "nC4", 4, 10, -125.6e3, 304.4, _CP["n_butane"], 93.8, 89.6, 0.58407),
    IsomSpecies("isopentane", "iC5", 5, 12, -154.5e3, 343.89, _CP["isopentane"], 92.3, 90.3,
                0.62470),
    IsomSpecies("n_pentane", "nC5", 5, 12, -146.4e3, 349.25, _CP["n_pentane"], 61.7, 62.6, 0.63086),
    IsomSpecies("2_2_dimethylbutane", "22DMB", 6, 14, -185.6e3, 358.22,
                _CP["2_2_dimethylbutane"], 91.8, 93.4, 0.6540),
    IsomSpecies("2_3_dimethylbutane", "23DMB", 6, 14, -177.8e3, 365.94,
                _CP["2_3_dimethylbutane"], 103.5, 94.3, 0.6664),
    IsomSpecies("2_methylpentane", "2MP", 6, 14, -174.3e3, 380.69, _CP["2_methylpentane"],
                73.4, 73.5, 0.6579),
    IsomSpecies("3_methylpentane", "3MP", 6, 14, -171.6e3, 383.04, _CP["3_methylpentane"],
                74.5, 74.3, 0.6690),
    IsomSpecies("n_hexane", "nC6", 6, 14, -167.2e3, 388.74, _CP["n_hexane"], 24.8, 26.0, 0.66404),
    IsomSpecies("methylcyclopentane", "MCP", 6, 12, -106.7e3, 339.9, _CP["methylcyclopentane"],
                91.3, 80.0, 0.7535),
    IsomSpecies("cyclohexane", "CH", 6, 12, -123.1e3, 297.31, _CP["cyclohexane"], 83.0, 77.2,
                0.7834),
    IsomSpecies("benzene", "Bz", 6, 6, 82.93e3, 269.18, _CP["benzene"], 101.0, 91.0, 0.8845,
                aromatic=1.0, note="verify; MON is an assumption"),
    IsomSpecies("n_heptane", "C7+", 7, 16, -187.8e3, 428.23, _CP["n_heptane"], 60.0, 58.0, 0.700,
                note="assumption: a C7+ lump of a light-naphtha tail, carried as n-heptane "
                     "for its thermodynamics, with lump octanes and SG"),
)

#: Species names in reactor order.
NAMES: tuple[str, ...] = tuple(s.name for s in SPECIES)
#: Short labels in reactor order.
LABELS: tuple[str, ...] = tuple(s.label for s in SPECIES)
#: Index of each species by name or label.
INDEX: dict[str, int] = {**{s.name: i for i, s in enumerate(SPECIES)},
                         **{s.label: i for i, s in enumerate(SPECIES)}}


def idx(name: str) -> int:
    """Reactor index of a species, by database name or short label."""
    try:
        return INDEX[name]
    except KeyError:
        raise KeyError(f"no isomerization species {name!r}; have {LABELS}") from None


def _arr(attr):
    return jnp.asarray([getattr(s, attr) for s in SPECIES], dtype=float)


CP = jnp.asarray([s.cp for s in SPECIES], dtype=float)
HF = _arr("Hf")
S0 = _arr("S")
MW = jnp.asarray([s.MW for s in SPECIES], dtype=float)
C_ATOMS = _arr("C")
H_ATOMS = _arr("H")
RON = _arr("RON")
MON = _arr("MON")
SG = _arr("SG")
AROMATIC = _arr("aromatic")

#: The isomer families whose equilibrium is computed in closed form.
FAMILIES: dict[str, tuple[str, ...]] = {
    "C5": ("n_pentane", "isopentane"),
    "C6P": ("n_hexane", "2_methylpentane", "3_methylpentane",
            "2_3_dimethylbutane", "2_2_dimethylbutane"),
    "C6N": ("methylcyclopentane", "cyclohexane"),
}


def cp(T):
    """Ideal-gas Cp of every species (J/mol/K), shape ``(n,)``."""
    T = jnp.asarray(T, dtype=float)
    return CP @ jnp.stack([jnp.ones_like(T), T, T**2, T**3])


def _int_cp(T):
    a, b, c, d = CP.T
    return a * (T - T0) + b / 2 * (T**2 - T0**2) + c / 3 * (T**3 - T0**3) + d / 4 * (T**4 - T0**4)


def _int_cp_over_T(T):
    a, b, c, d = CP.T
    return a * jnp.log(T / T0) + b * (T - T0) + c / 2 * (T**2 - T0**2) + d / 3 * (T**3 - T0**3)


def enthalpy(T):
    """Ideal-gas molar enthalpy, elements at 298.15 K as zero (J/mol)."""
    T = jnp.asarray(T, dtype=float)
    return HF + _int_cp(T)


def entropy(T):
    """Absolute ideal-gas molar entropy at 1 bar (J/mol/K)."""
    T = jnp.asarray(T, dtype=float)
    return S0 + _int_cp_over_T(T)


def gibbs(T):
    """``H - T S`` of every species at 1 bar (J/mol); differences are
    standard reaction free energies."""
    T = jnp.asarray(T, dtype=float)
    return enthalpy(T) - T * entropy(T)


def ln_K(nu, T):
    """``ln K`` of a reaction with stoichiometry ``nu`` (shape ``(n,)``),
    pressures in bar."""
    return -jnp.dot(jnp.asarray(nu, dtype=float), gibbs(T)) / (R * T)


def family_equilibrium(family: str, T) -> dict:
    """Equilibrium mole fractions within an isomer family at ``T`` (K).

    Isomers interconvert without changing moles, so the equilibrium is
    pressure-independent and closed-form: ``x_i`` proportional to
    ``exp(-G_i / R T)``.

    Returns:
        ``{species name: mole fraction}``.
    """
    names = FAMILIES[family]
    g = gibbs(T)[jnp.asarray([idx(n) for n in names])]
    w = jax.nn.softmax(-g / (R * jnp.asarray(T, dtype=float)))
    return dict(zip(names, w))


def equilibrium_table(T) -> dict:
    """Every family's equilibrium distribution at ``T``, as
    ``{family: {name: x}}``."""
    return {f: family_equilibrium(f, T) for f in FAMILIES}


def check_formulae() -> None:
    """Raise if a species' molar mass disagrees with the database's."""
    from difflow.database import get_critical_props

    for s in SPECIES:
        mw = get_critical_props(s.name).MW
        if abs(mw - s.MW) > 0.05:
            raise ValueError(f"{s.name}: formula MW {s.MW:.3f} vs database {mw:.3f}")


def liquid_volume(flows) -> jax.Array:
    """Liquid volume (m^3/s at 60 F) of molar flows (mol/s), ideal mixing."""
    return jnp.asarray(flows) * MW / 1000.0 / (SG * 999.0)


__all__ = [
    "R", "T0", "P_STD", "IsomSpecies", "SPECIES", "NAMES", "LABELS", "INDEX", "idx",
    "CP", "HF", "S0", "MW", "C_ATOMS", "H_ATOMS", "RON", "MON", "SG", "AROMATIC",
    "FAMILIES", "cp", "enthalpy", "entropy", "gibbs", "ln_K", "family_equilibrium",
    "equilibrium_table", "check_formulae", "liquid_volume",
]
