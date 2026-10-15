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

from difflow_refinery import combustion as _cb

#: IUPAC conventional atomic weights (g/mol): IUPAC CIAAW, "Standard atomic
#: weights of the elements 2021", Pure Appl. Chem. 2022 (abridged /
#: conventional values: H 1.008, C 12.011, N 14.007, O 15.999, S 32.06).
ATOMIC_WEIGHT = _cb.ATOMIC_WEIGHT

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

#: The regenerator gases (air in, flue gas out).
REGENERATOR_GASES = ("nitrogen", "oxygen", "carbon_dioxide", "carbon_monoxide", "water",
                     "sulfur_dioxide")

#: Ideal-gas heat capacity ``Cp = a + b T + c T^2 + d T^3`` (J/mol/K, T in K)
#: of the regenerator gases (shared with :mod:`difflow_refinery.combustion`, #470), and their ideal-gas enthalpy of formation at
#: 298.15 K (J/mol): both from the refinery's one thermochemistry table
#: (:mod:`difflow_refinery.thermochemistry`, #339) -- CODATA key values for
#: Hf, and cubics fitted over 298.15-1500 K to the NIST-JANAF tables (Chase
#: 1998), which cover a regenerator at 950-1000 K with room to spare. Before
#: #339 the Cp were Reid, Prausnitz & Poling (4th ed.) cubics, four of them
#: unverified transcriptions; they agree with the JANAF fits to 1.1 % (O2;
#: the rest 0.7 %) up to 1050 K, and fall away from JANAF above 1000 K.
CP_IG = _cb.CP_IG
HF_298 = _cb.HF_298
T_REF = _cb.T_REF

#: Dry air (mole fractions), argon lumped into nitrogen.
AIR_O2 = _cb.AIR_O2
AIR_N2 = _cb.AIR_N2

h_ideal_gas = _cb.h_ideal_gas
flue_enthalpy = _cb.gas_enthalpy


def mass_of(moles: dict[str, Array]) -> Array:
    """Mass flow (kg/s) of ``{species: mol/s}``."""
    return sum(moles[n] * MW[n] for n in moles) / 1000.0


def element_flow(moles: dict[str, Array], element: str) -> Array:
    """Mass flow (kg/s) of ``element`` in ``{species: mol/s}``."""
    return sum(moles[n] * ATOMIC_WEIGHT[element] * FORMULA[n].get(element, 0)
               for n in moles) / 1000.0


def zeros_like_moles(names, like=0.0) -> dict[str, Array]:
    return {n: jnp.zeros_like(jnp.asarray(like, dtype=float)) for n in names}
