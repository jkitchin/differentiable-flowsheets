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
    dHf, S and the Cp cubic of every species are its row of
    :mod:`difflow_refinery.thermochemistry`, the one table every refinery
    unit reads (#339): API Technical Data Book dHf, Yaws S, and cubics
    fitted over 298-1000 K to the TRC ideal-gas Cp correlation, with the
    cross-checks and per-species choices recorded there. Before #339 this
    module kept its own copy -- Prosen & Rossini (1945) dHf as the NIST
    WebBook gives them, and cubics fitted to the WebBook tables over
    273-800 K -- and the C5 and C6 isomer differences have moved with the
    change: nC5 -> iC5 is -6.99 kJ/mol now against -8.10 before (Good
    (1970)'s re-measurement of the pentanes, which the API TDB follows,
    against Prosen & Rossini); relative to n-hexane, 2,2- and
    2,3-dimethylbutane sit 0.67 and 0.75 kJ/mol higher and 2- and
    3-methylpentane 0.64 and 0.71 kJ/mol lower (API TDB against Prosen &
    Rossini; CRC puts the dimethylbutanes 1.2-1.3 kJ/mol lower than API
    TDB). See the docs' "Thermochemical data" section for what that does to
    the equilibria.

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

from difflow_refinery import thermochemistry

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


def _I(name, label, C, H, RON, MON, SG, aromatic=0.0, note=_V):
    """One species: Hf, S and the Cp cubic from the shared table (#339)."""
    f = thermochemistry.species(name)
    if f.elements != ({"C": C, "H": H} if C else {"H": H}):
        raise ValueError(f"{name}: formula C{C}H{H} is not the table's {f.formula}")
    return IsomSpecies(name, label, C, H, f.Hf, f.S0, tuple(f.cp), RON, MON, SG, aromatic, note)


#: Every species, in reactor order (also the column component order).
SPECIES: tuple[IsomSpecies, ...] = (
    _I("hydrogen", "H2", 0, 2, 0.0, 0.0, 1.0, note="gas; SG is a placeholder, never used"),
    _I("ethane", "C2", 2, 6, 0.0, 0.0, 0.35584, note="gas; never in the isomerate, so no octanes"),
    _I("propane", "C3", 3, 8, 0.0, 0.0, 0.50736, note="gas; never in the isomerate, so no octanes"),
    _I("isobutane", "iC4", 4, 10, 102.1, 97.6, 0.56293),
    _I("n_butane", "nC4", 4, 10, 93.8, 89.6, 0.58407),
    _I("isopentane", "iC5", 5, 12, 92.3, 90.3, 0.62470),
    _I("n_pentane", "nC5", 5, 12, 61.7, 62.6, 0.63086),
    _I("2_2_dimethylbutane", "22DMB", 6, 14, 91.8, 93.4, 0.6540),
    _I("2_3_dimethylbutane", "23DMB", 6, 14, 103.5, 94.3, 0.6664),
    _I("2_methylpentane", "2MP", 6, 14, 73.4, 73.5, 0.6579),
    _I("3_methylpentane", "3MP", 6, 14, 74.5, 74.3, 0.6690),
    _I("n_hexane", "nC6", 6, 14, 24.8, 26.0, 0.66404),
    _I("methylcyclopentane", "MCP", 6, 12, 91.3, 80.0, 0.7535),
    _I("cyclohexane", "CH", 6, 12, 83.0, 77.2, 0.7834),
    _I("benzene", "Bz", 6, 6, 101.0, 91.0, 0.8845, aromatic=1.0,
       note="verify; MON is an assumption"),
    _I("n_heptane", "C7+", 7, 16, 60.0, 58.0, 0.700,
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
