"""Coke combustion: stoichiometry, air, flue gas and enthalpy (#470).

Shared by the regenerators that burn coke off a catalyst -- the FCC
(:mod:`difflow_refinery.fcc.regenerator`) and the CCR reformer's continuous
regenerator (#474). Coke is given as element mass flows (kg/s) of C, H, S
and N and burns to::

    C + O2 -> CO2         C + 1/2 O2 -> CO
    H + 1/4 O2 -> 1/2 H2O S + O2 -> SO2      N -> 1/2 N2

The CO/CO2 molar ratio is SPECIFIED (0 is full burn) or taken from the
Arthur (1951) correlation for the primary products of carbon combustion,
``CO/CO2 = 10^3.4 exp(-12400 / (R T))`` (R in cal/mol/K). Air is dry,
20.95 mol% O2 with argon lumped into N2; either the flue-gas O2 mole fraction
(wet basis) is specified and the air follows, or the air rate is given and
the excess O2 follows.

Enthalpies are ideal-gas, ``Hf(298.15 K) + int Cp dT``, from the refinery's
one thermochemistry table (:mod:`difflow_refinery.thermochemistry`: CODATA
Hf, Cp cubics fitted to NIST-JANAF over 298.15-1500 K). Coke's enthalpy of
formation is taken as zero (graphite, H2, S, N2 elements), so its heat of
combustion follows from its composition: carbon to CO2 releases the CODATA
-393.51 kJ/mol, hydrogen to water vapour -241.826 kJ/mol H2O.

Sources:

* Atomic weights: Prohaska et al. (IUPAC CIAAW), "Standard atomic weights
  of the elements 2021", Pure Appl. Chem. 94, 573-600 (2022),
  doi:10.1515/pac-2019-0603; abridged/conventional values.
* Dry air O2 = 20.95 mol%: U.S. Standard Atmosphere 1976 (NOAA/NASA/USAF;
  NASA-TM-X-74335, DTIC ADA035728), sea-level dry air O2 20.9476 % by
  volume, rounded (value confirmed through a secondary tabulation; the table
  number in the report was not checked).
* Arthur, J. R., "Reactions between carbon and oxygen", Trans. Faraday Soc.
  47, 164-178 (1951), doi:10.1039/TF9514700164. The constants 10^3.4 and
  12400 cal/mol (51.88 kJ/mol) are the values as commonly quoted in the
  FCC/char-combustion literature (e.g. the char-combustion reviews quoting
  ``A = 2512, E = 51.8 kJ/mol, 730-1170 K``); they were NOT checked against
  the paper itself (its full text was not accessible), so they stay marked
  unverified. Arthur's ratio describes the products at the carbon surface;
  afterburning of CO is not modelled, so ``"arthur"`` is a partial-burn
  model.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from difflow_refinery import thermochemistry

#: IUPAC conventional atomic weights (g/mol), CIAAW 2021 (abridged).
ATOMIC_WEIGHT = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}

#: Arthur (1951) CO/CO2 constants as quoted (unverified against the paper):
#: pre-exponential 10^3.4 and activation 12400 cal/mol.
ARTHUR_A = 10.0 ** 3.4
ARTHUR_E_CAL = 12400.0
#: Gas constant in cal/mol/K (8.314462618 / 4.184, thermochemical calorie).
R_CAL = 1.987204
#: Temperature range (K) the quoted Arthur constants are stated for.
ARTHUR_RANGE = (730.0, 1170.0)

#: Dry air (mole fractions), argon lumped into nitrogen.
AIR_O2 = 0.2095
AIR_N2 = 1.0 - AIR_O2

#: Gases of air and flue gas, and their thermochemistry (shared table).
GASES = ("nitrogen", "oxygen", "carbon_dioxide", "carbon_monoxide", "water",
         "sulfur_dioxide")
CP_IG: dict[str, tuple[float, float, float, float]] = {
    n: thermochemistry.species(n).cp for n in GASES}
HF_298: dict[str, float] = {n: thermochemistry.Hf(n) for n in GASES}
T_REF = 298.15

_MW_O2 = 2 * ATOMIC_WEIGHT["O"]
_MW_N2 = 2 * ATOMIC_WEIGHT["N"]
#: Molar mass of the dry air above (g/mol).
MW_AIR = AIR_O2 * _MW_O2 + AIR_N2 * _MW_N2


def arthur_co_co2(T: Array) -> Array:
    """CO/CO2 molar ratio of carbon combustion, Arthur (1951) form (unverified constants)."""
    return ARTHUR_A * jnp.exp(-ARTHUR_E_CAL / (R_CAL * T))


def h_ideal_gas(name: str, T: Array) -> Array:
    """Molar enthalpy (J/mol) of an air/flue gas: ``Hf(298.15) + int Cp dT``."""
    a, b, c, d = CP_IG[name]

    def F(t):
        return a * t + b * t ** 2 / 2 + c * t ** 3 / 3 + d * t ** 4 / 4

    return HF_298[name] + F(T) - F(T_REF)


def gas_enthalpy(moles: dict[str, Array], T: Array) -> Array:
    """Enthalpy flow (W) of a gas mixture given as ``{species: mol/s}``."""
    return sum(moles[n] * h_ideal_gas(n, T) for n in moles)


def burn(el: dict[str, Array], co_co2: Array, *, flue_o2: Array | None = None,
         air_rate: Array | None = None) -> dict:
    """Air and flue gas (mol/s) for burning coke of elements ``el`` (kg/s).

    Args:
        el: Element mass flows ``{"C", "H", "S", "N"}`` (kg/s).
        co_co2: CO/CO2 molar ratio of the burnt carbon (0 is full burn); e.g.
            :func:`arthur_co_co2` at the bed temperature.
        flue_o2: Flue-gas O2 mole fraction (wet); the air follows. Give this
            or ``air_rate``.
        air_rate: Dry air rate (kg/s); the excess O2 follows (negative means
            too little air for the requested burn -- not clipped).

    Returns:
        ``{"air": {oxygen, nitrogen}, "flue": {nitrogen, oxygen,
        carbon_dioxide, carbon_monoxide, water, sulfur_dioxide},
        "co_co2", "o2_required"}``, flows in mol/s.

    Example:
        >>> out = burn({"C": 1.0, "H": 0.0, "S": 0.0, "N": 0.0}, 0.0, flue_o2=0.02)
        >>> round(float(out["flue"]["carbon_dioxide"]), 3)   # 1 kg/s / 12.011 g/mol
        83.257
    """
    if (flue_o2 is None) == (air_rate is None):
        raise ValueError("give exactly one of flue_o2 and air_rate")
    nC = el["C"] * 1000.0 / ATOMIC_WEIGHT["C"]
    nH = el["H"] * 1000.0 / ATOMIC_WEIGHT["H"]
    nS = el["S"] * 1000.0 / ATOMIC_WEIGHT["S"]
    nN = el["N"] * 1000.0 / ATOMIC_WEIGHT["N"]
    r = co_co2
    n_co2 = nC / (1.0 + r)
    n_co = nC - n_co2
    n_h2o = nH / 2.0
    n_so2 = nS
    n_n2_coke = nN / 2.0
    o2_req = n_co2 + 0.5 * n_co + 0.5 * n_h2o + n_so2
    k = AIR_N2 / AIR_O2
    if flue_o2 is not None:
        y = flue_o2
        other = n_co2 + n_co + n_h2o + n_so2 + n_n2_coke
        o2_ex = y * (other + k * o2_req) / (1.0 - y * (1.0 + k))
        o2_air = o2_req + o2_ex
    else:
        o2_air = air_rate * 1000.0 / MW_AIR * AIR_O2
        o2_ex = o2_air - o2_req
    n2_air = k * o2_air
    air = {"oxygen": o2_air, "nitrogen": n2_air}
    flue = {"nitrogen": n2_air + n_n2_coke, "oxygen": o2_ex, "carbon_dioxide": n_co2,
            "carbon_monoxide": n_co, "water": n_h2o, "sulfur_dioxide": n_so2}
    return {"air": air, "flue": flue, "co_co2": r, "o2_required": o2_req}


def heat_of_combustion(el: dict[str, Array], co_co2: Array, T: Array = T_REF) -> Array:
    """Heat released (W, positive) burning coke ``el`` (kg/s) with exactly the
    stoichiometric O2, reactants and products at ``T`` (water as vapour).

    At 298.15 K and full burn this is the lower heating value of the coke on
    the zero-formation-enthalpy convention: 393.51 kJ/mol C (CODATA CO2) plus
    120.9 kJ/mol H (half the CODATA 241.826 of H2O(g)), 296.81 kJ/mol S.
    """
    out = burn(el, co_co2, air_rate=0.0)
    products = {k: v for k, v in out["flue"].items() if k != "oxygen"}
    return out["o2_required"] * h_ideal_gas("oxygen", T) - gas_enthalpy(products, T)


__all__ = [
    "ATOMIC_WEIGHT", "ARTHUR_A", "ARTHUR_E_CAL", "R_CAL", "ARTHUR_RANGE", "AIR_O2",
    "AIR_N2", "MW_AIR", "GASES", "CP_IG", "HF_298", "T_REF", "arthur_co_co2",
    "h_ideal_gas", "gas_enthalpy", "burn", "heat_of_combustion",
]
