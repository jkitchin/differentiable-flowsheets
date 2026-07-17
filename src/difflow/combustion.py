"""Combustion and gas-turbine thermochemistry (ideal-gas working fluid).

A gas-turbine working fluid is air and combustion gas at high temperature and
moderate pressure, where the real-gas (Peng-Robinson) departure is negligible,
so the standard approach for gas-turbine cycle analysis -- and what this module
provides -- is **ideal-gas** properties with temperature-dependent Cp, not the
cubic-EOS flash path used by :mod:`difflow.units.eos_units`.

The pieces here support the combustion / gas-turbine units in
:mod:`difflow.units.gas_turbine` (issue #173):

- :class:`IdealGasThermo` -- temperature-dependent ideal-gas Cp, sensible
  enthalpy (integral of Cp from 298.15 K) and entropy (integral of Cp/T plus the
  -R ln(P/Pref) pressure and -R sum y ln y mixing terms). The entropy structure
  mirrors :meth:`difflow.eos.PengRobinson.entropy_departure` with the departure
  set to zero.
- Combustion data -- lower heating values and complete-combustion stoichiometry
  ``CxHy + (x + y/4) O2 -> x CO2 + (y/2) H2O`` per fuel species.
- Fuel property functions of composition (LHV, O2 demand, products), all
  smooth so a plant's power/emissions can be differentiated with respect to the
  delivered gas quality.

The fuel-hydrocarbon (and N2, CO2) ideal-gas Cp cubics are reused from
:mod:`difflow.database` (added in issue #172); only the air/product species
(O2, H2O vapor, Ar) that the NGL work did not need are defined locally here.
Defining H2O's *vapor* Cp locally also avoids the database's ``water`` record,
whose ``Cp_coeffs`` is the *liquid* heat capacity used for aqueous/distillation
models.

Units: Cp in J/mol/K (``Cp = a + b*T + c*T**2 + d*T**3``); LHV in J/mol; molar
mass in kg/mol. Reference state 298.15 K, 101325 Pa.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.database import get_species_data
from difflow.numerics import safe_log

R_GAS = 8.314462618  # J/mol/K
T_REF = 298.15       # K
P_REF = 101325.0     # Pa


# ---------------------------------------------------------------------------
# Ideal-gas Cp cubics for the air / combustion-product species not covered by
# the NGL fuel table. Reid, Prausnitz & Poling 4th ed., Appendix A.
# ---------------------------------------------------------------------------
_CYCLE_CP: dict[str, tuple[float, float, float, float]] = {
    "oxygen": (28.11, -3.680e-6, 1.746e-5, -1.065e-8),
    "water": (32.24, 1.924e-3, 1.055e-5, -3.596e-9),   # H2O *vapor*
    "argon": (20.786, 0.0, 0.0, 0.0),                  # monatomic
    # nitrogen and carbon_dioxide come from difflow.database (ideal-gas cubics).
}


def cp_coeffs(species: str) -> tuple[float, float, float, float]:
    """Ideal-gas Cp cubic (a, b, c, d) for a cycle species.

    Air/product species (O2, H2O vapor, Ar) come from the local table; all other
    species (fuel hydrocarbons, N2, CO2) come from difflow's database, where they
    are ideal-gas cubics (issue #172).
    """
    if species in _CYCLE_CP:
        return _CYCLE_CP[species]
    return get_species_data(species).Cp_coeffs


# Molar masses (kg/mol) for the cycle and fuel species.
MW: dict[str, float] = {
    "nitrogen": 0.0280134, "carbon_dioxide": 0.0440095, "methane": 0.0160428,
    "ethane": 0.0300690, "propane": 0.0440956, "isobutane": 0.0581222,
    "n_butane": 0.0581222, "isopentane": 0.0721488, "n_pentane": 0.0721488,
    "n_hexane": 0.0861754, "oxygen": 0.0319988, "water": 0.0180153,
    "argon": 0.0399480,
}

# Dry-air composition (mole fractions).
AIR_COMPOSITION: dict[str, float] = {
    "nitrogen": 0.78084,
    "oxygen": 0.20946,
    "argon": 0.00934,
    "carbon_dioxide": 0.00036,
}

# All species that can appear in the cycle working fluid (air + products).
CYCLE_SPECIES: tuple[str, ...] = (
    "nitrogen", "oxygen", "argon", "carbon_dioxide", "water",
)

# ---------------------------------------------------------------------------
# Complete-combustion data per fuel species:
#   (carbon count x, hydrogen count y, LHV [J/mol], combustible)
# Reaction: CxHy + (x + y/4) O2 -> x CO2 + (y/2) H2O.
# Diluents (N2, CO2) are not combustible: they consume no O2 and pass through
# (CO2's carbon still routes to a product CO2). LHV (gaseous H2O) from standard
# tables (kJ/mol -> J/mol).
# ---------------------------------------------------------------------------
FUEL_COMBUSTION: dict[str, tuple[int, int, float, bool]] = {
    "methane":    (1, 4, 802.6e3, True),
    "ethane":     (2, 6, 1428.6e3, True),
    "propane":    (3, 8, 2043.1e3, True),
    "isobutane":  (4, 10, 2649.0e3, True),
    "n_butane":   (4, 10, 2657.3e3, True),
    "isopentane": (5, 12, 3264.0e3, True),
    "n_pentane":  (5, 12, 3272.1e3, True),
    "n_hexane":   (6, 14, 3886.6e3, True),
    "nitrogen":       (0, 0, 0.0, False),
    "carbon_dioxide": (1, 0, 0.0, False),
}

# Fuel species order (matches the NGL component order).
FUEL_COMPONENTS: tuple[str, ...] = (
    "nitrogen", "carbon_dioxide", "methane", "ethane", "propane",
    "isobutane", "n_butane", "isopentane", "n_pentane", "n_hexane",
)


# ---------------------------------------------------------------------------
# Ideal-gas thermo for a fixed species set
# ---------------------------------------------------------------------------
@dataclass
class IdealGasThermo:
    """Ideal-gas properties for a fixed, ordered set of species.

    Temperature-dependent Cp (cubic), sensible enthalpy (integral of Cp from
    ``T_REF``) and entropy (integral of Cp/T, plus the -R ln(P/Pref) pressure and
    -R sum y ln y mixing terms). Per-species reference entropies are omitted;
    they cancel in the constant-composition compressions and expansions the
    gas-turbine units use them for. All methods are JAX-differentiable.

    Attributes:
        species: ordered species names (each resolvable by :func:`cp_coeffs`).
    """

    species: tuple[str, ...]

    def __post_init__(self):
        coeffs = jnp.asarray([cp_coeffs(s) for s in self.species])
        self._a, self._b, self._c, self._d = (
            coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3])
        self._index = {s: i for i, s in enumerate(self.species)}

    @property
    def species_order(self) -> tuple[str, ...]:
        return self.species

    def index(self, species: str) -> int:
        """Position of ``species`` in the ordered set."""
        return self._index[species]

    # ---- per-species integrals (arrays over species) -------------------
    def cp(self, T: Array) -> Array:
        """Ideal-gas Cp per species [J/mol/K]."""
        T = jnp.asarray(T)
        return self._a + self._b * T + self._c * T**2 + self._d * T**3

    def h_sensible(self, T: Array) -> Array:
        """Per-species sensible enthalpy integral(Cp dT) from T_REF [J/mol]."""
        T = jnp.asarray(T)
        return (self._a * (T - T_REF)
                + self._b / 2 * (T**2 - T_REF**2)
                + self._c / 3 * (T**3 - T_REF**3)
                + self._d / 4 * (T**4 - T_REF**4))

    def s_T(self, T: Array) -> Array:
        """Per-species integral(Cp/T dT) from T_REF [J/mol/K]."""
        T = jnp.asarray(T)
        return (self._a * safe_log(T / T_REF)
                + self._b * (T - T_REF)
                + self._c / 2 * (T**2 - T_REF**2)
                + self._d / 3 * (T**3 - T_REF**3))

    # ---- mixture molar properties (n = per-species mole amounts) --------
    def mixture_enthalpy(self, n: Array, T: Array) -> Array:
        """Total sensible enthalpy of a mixture [J or W] (n in mol or mol/s)."""
        return jnp.sum(n * self.h_sensible(T))

    def mixture_entropy(self, n: Array, T: Array, P: Array) -> Array:
        """Total ideal-gas entropy of a mixture [J/K or W/K] vs. the reference.

        Includes the Cp/T integral, the -R ln(P/Pref) pressure term, and the
        -R sum n_i ln(y_i) mixing term.
        """
        n_total = jnp.sum(n)
        y = n / jnp.maximum(n_total, 1e-30)
        s_species = (self.s_T(T)
                     - R_GAS * safe_log(jnp.asarray(P) / P_REF)
                     - R_GAS * safe_log(jnp.maximum(y, 1e-30)))
        return jnp.sum(n * s_species)

    def mixture_cp(self, n: Array, T: Array) -> Array:
        """Total heat capacity of a mixture [J/K or W/K]."""
        return jnp.sum(n * self.cp(T))

    # ---- stream helpers (difflow {species: flow} dicts) ----------------
    def flow_vector(self, flows: dict) -> Array:
        """Per-species mole (flow) vector in ``species`` order from a flow dict.

        Species absent from ``flows`` contribute zero.
        """
        return jnp.asarray([flows.get(s, 0.0) for s in self.species])

    def stream_enthalpy(self, flows: dict, T: Array) -> Array:
        """Total sensible enthalpy of a stream [W] (flows in mol/s)."""
        return self.mixture_enthalpy(self.flow_vector(flows), T)

    def stream_entropy(self, flows: dict, T: Array, P: Array) -> Array:
        """Total ideal-gas entropy of a stream [W/K] (flows in mol/s)."""
        return self.mixture_entropy(self.flow_vector(flows), T, P)


# ---------------------------------------------------------------------------
# Fuel properties as differentiable functions of composition
# ---------------------------------------------------------------------------
def _fuel_arrays(components: tuple[str, ...]):
    """(C, H, LHV, combustible) arrays for a fuel component order."""
    C = jnp.asarray([FUEL_COMBUSTION[c][0] for c in components], dtype=float)
    H = jnp.asarray([FUEL_COMBUSTION[c][1] for c in components], dtype=float)
    LHV = jnp.asarray([FUEL_COMBUSTION[c][2] for c in components], dtype=float)
    comb = jnp.asarray([1.0 if FUEL_COMBUSTION[c][3] else 0.0 for c in components])
    return C, H, LHV, comb


def _fuel_x(fuel, components: tuple[str, ...]) -> Array:
    """Normalized fuel composition array in ``components`` order."""
    if isinstance(fuel, dict):
        x = jnp.asarray([fuel.get(c, 0.0) for c in components])
    else:
        x = jnp.asarray(fuel)
    return x / jnp.maximum(jnp.sum(x), 1e-30)


def lhv_molar(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """Lower heating value per mole of fuel mixture [J/mol]."""
    _, _, LHV, _ = _fuel_arrays(components)
    return jnp.sum(_fuel_x(fuel, components) * LHV)


def molar_mass(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """Fuel mixture molar mass [kg/mol]."""
    mw = jnp.asarray([MW[c] for c in components])
    return jnp.sum(_fuel_x(fuel, components) * mw)


def lhv_mass(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """Lower heating value per kg of fuel [J/kg]."""
    return lhv_molar(fuel, components) / molar_mass(fuel, components)


def o2_demand(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """Stoichiometric O2 per mole of fuel [mol O2 / mol fuel].

    Only combustible species demand O2; diluent CO2/N2 do not.
    """
    C, H, _, comb = _fuel_arrays(components)
    return jnp.sum(_fuel_x(fuel, components) * comb * (C + H / 4.0))


def co2_per_mol(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """CO2 produced per mole of fuel [mol/mol] (fuel carbon, incl. diluent CO2)."""
    C, _, _, _ = _fuel_arrays(components)
    return jnp.sum(_fuel_x(fuel, components) * C)


def h2o_per_mol(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """H2O produced per mole of fuel [mol/mol]."""
    _, H, _, _ = _fuel_arrays(components)
    return jnp.sum(_fuel_x(fuel, components) * H / 2.0)


def n2_per_mol(fuel, components: tuple[str, ...] = FUEL_COMPONENTS) -> Array:
    """Inert N2 carried from the fuel per mole of fuel [mol/mol]."""
    is_n2 = jnp.asarray([1.0 if c == "nitrogen" else 0.0 for c in components])
    return jnp.sum(_fuel_x(fuel, components) * is_n2)
