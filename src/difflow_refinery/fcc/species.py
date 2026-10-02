"""Real species of the FCC: light products, air and flue gas.

Molar masses are computed from the molecular FORMULA and the IUPAC
conventional atomic weights, so the elemental (C, H, S, N, O) balances of
the unit close to round-off by construction rather than to the third figure
of a tabulated molar mass. They therefore differ in the second decimal from
some ``difflow.database`` entries (methane 16.043 here, 16.04 there).

Species names are ``difflow.database`` names (``tests/refinery/test_fcc.py``
checks that each resolves there), including the butenes and isobutylene
added for the refinery plugins.

Ideal-gas heat capacities are used only by the regenerator (air in, flue gas
out); the riser works on lumped constant heat capacities.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

#: IUPAC conventional atomic weights (g/mol): IUPAC CIAAW, "Standard atomic
#: weights of the elements 2021", Pure Appl. Chem. 2022 (abridged /
#: conventional values: H 1.008, C 12.011, N 14.007, O 15.999, S 32.06).
ATOMIC_WEIGHT = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}

#: Formula of each real species, ``{element: atoms}``.
FORMULA: dict[str, dict[str, int]] = {
    "hydrogen": {"H": 2},
    "methane": {"C": 1, "H": 4},
    "ethane": {"C": 2, "H": 6},
    "ethylene": {"C": 2, "H": 4},
    "hydrogen_sulfide": {"H": 2, "S": 1},
    "propane": {"C": 3, "H": 8},
    "propylene": {"C": 3, "H": 6},
    "isobutane": {"C": 4, "H": 10},
    "n_butane": {"C": 4, "H": 10},
    "1_butene": {"C": 4, "H": 8},
    "isobutylene": {"C": 4, "H": 8},
    "cis_2_butene": {"C": 4, "H": 8},
    "trans_2_butene": {"C": 4, "H": 8},
    "water": {"H": 2, "O": 1},
    "nitrogen": {"N": 2},
    "oxygen": {"O": 2},
    "carbon_dioxide": {"C": 1, "O": 2},
    "carbon_monoxide": {"C": 1, "O": 1},
    "sulfur_dioxide": {"S": 1, "O": 2},
}

#: Dry-gas hydrocarbons (with ``hydrogen_sulfide`` added from the sulfur
#: balance), the C3 cut and the C4 cut, in stream order.
DRY_GAS_SPECIES = ("hydrogen", "methane", "ethane", "ethylene")
C3_SPECIES = ("propane", "propylene")
C4_SPECIES = ("isobutane", "n_butane", "1_butene", "isobutylene",
              "cis_2_butene", "trans_2_butene")
C3_OLEFINS = ("propylene",)
C4_OLEFINS = ("1_butene", "isobutylene", "cis_2_butene", "trans_2_butene")
FLUE_SPECIES = ("nitrogen", "oxygen", "carbon_dioxide", "carbon_monoxide",
                "water", "sulfur_dioxide")


def molar_mass(name: str) -> float:
    """Molar mass (g/mol) of a real species from its formula."""
    return float(sum(ATOMIC_WEIGHT[e] * n for e, n in FORMULA[name].items()))


def element_mass_fraction(name: str, element: str) -> float:
    """Mass fraction of ``element`` in species ``name``."""
    return ATOMIC_WEIGHT[element] * FORMULA[name].get(element, 0) / molar_mass(name)


MW = {name: molar_mass(name) for name in FORMULA}

#: Ideal-gas heat capacity ``Cp = a + b T + c T^2 + d T^3`` (J/mol/K, T in
#: K) of the regenerator gases, Reid, Prausnitz & Poling, *The Properties of
#: Gases and Liquids*, 4th ed. (McGraw-Hill, 1987), Appendix A (stated range
#: roughly 273-1500 K). N2 and CO2 are the coefficients ``difflow.database``
#: already carries (same source); O2, CO, H2O and SO2 were transcribed for
#: this module and are marked (unverified transcription) -- see
#: ``docs/unit-operations-refinery.md``.
CP_IG: dict[str, tuple[float, float, float, float]] = {
    "nitrogen": (31.15, -1.357e-2, 2.680e-5, -1.168e-8),
    "carbon_dioxide": (19.80, 7.344e-2, -5.602e-5, 1.715e-8),
    "oxygen": (28.11, -3.680e-6, 1.746e-5, -1.065e-8),         # (unverified transcription)
    "carbon_monoxide": (30.87, -1.285e-2, 2.789e-5, -1.272e-8),  # (unverified transcription)
    "water": (32.24, 1.924e-3, 1.055e-5, -3.596e-9),            # (unverified transcription)
    "sulfur_dioxide": (23.85, 6.699e-2, -4.961e-5, 1.328e-8),   # (unverified transcription)
}

#: Standard enthalpy of formation at 298.15 K, ideal gas (J/mol): CODATA
#: Key Values for Thermodynamics (Cox, Wagman & Medvedev, Hemisphere, 1989).
#: CO is the NIST-JANAF value (Chase 1998), not a CODATA key value.
HF_298: dict[str, float] = {
    "nitrogen": 0.0,
    "oxygen": 0.0,
    "carbon_dioxide": -393.51e3,
    "carbon_monoxide": -110.53e3,
    "water": -241.826e3,
    "sulfur_dioxide": -296.81e3,
}

T_REF = 298.15

#: Dry air (mole fractions), argon lumped into nitrogen.
AIR_O2 = 0.2095
AIR_N2 = 1.0 - AIR_O2


def h_ideal_gas(name: str, T: Array) -> Array:
    """Molar enthalpy (J/mol) of an ideal gas: ``Hf(298.15) + int Cp dT``."""
    a, b, c, d = CP_IG[name]

    def F(t):
        return a * t + b * t ** 2 / 2 + c * t ** 3 / 3 + d * t ** 4 / 4

    return HF_298[name] + F(T) - F(T_REF)


def flue_enthalpy(moles: dict[str, Array], T: Array) -> Array:
    """Enthalpy flow (W) of a gas mixture given as ``{species: mol/s}``."""
    return sum(moles[n] * h_ideal_gas(n, T) for n in moles)


def mass_of(moles: dict[str, Array]) -> Array:
    """Mass flow (kg/s) of ``{species: mol/s}``."""
    return sum(moles[n] * MW[n] for n in moles) / 1000.0


def element_flow(moles: dict[str, Array], element: str) -> Array:
    """Mass flow (kg/s) of ``element`` in ``{species: mol/s}``."""
    return sum(moles[n] * ATOMIC_WEIGHT[element] * FORMULA[n].get(element, 0)
               for n in moles) / 1000.0


def zeros_like_moles(names, like=0.0) -> dict[str, Array]:
    return {n: jnp.zeros_like(jnp.asarray(like, dtype=float)) for n in names}
