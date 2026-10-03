"""The FCC regenerator: well-mixed coke combustion and its heat balance.

Coke (C, H, S, N by mass) burns with air in one well-mixed bed at the
regenerator temperature ``T_rg``::

    C + O2 -> CO2         C + 1/2 O2 -> CO
    H + 1/4 O2 -> 1/2 H2O S + O2 -> SO2      N -> 1/2 N2

The CO/CO2 molar ratio is either SPECIFIED (``co_co2``; 0 is full burn) or
from the Arthur (1951) correlation for the primary products of carbon
combustion, ``CO/CO2 = 10^3.4 exp(-12400 / (R T))`` (R in cal/mol/K), times
a multiplier ``arthur_factor`` (a fit knob, e.g. for CO promoter). The
constants 10^3.4 and 12400 cal/mol are the values as commonly quoted from
Arthur (1951) in the FCC-modelling literature; they were NOT checked against
the paper for this implementation (unverified). Arthur's ratio describes the
products at the carbon surface; afterburning of CO in the bed or dilute
phase is not modelled, so ``"arthur"`` is a partial-burn model.

Air: the flue-gas O2 mole fraction (wet basis) is specified (``flue_o2``)
and the air rate follows, or the air rate is given and the excess O2 is an
output. Air is dry, 20.95 mol% O2, the rest lumped as N2.

Energy: enthalpies are ideal-gas, ``Hf(298.15 K) + int Cp dT`` with Cp
from :data:`~difflow_refinery.fcc.species.CP_IG`. Coke's enthalpy of
formation is taken as zero (graphite, H2, S, N2 elements), so its heat of
combustion follows from its H content -- the convention the issue asks for.
The regenerated catalyst leaves clean (no carbon on regenerated catalyst).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from difflow_refinery.fcc import species as sp

#: Arthur (1951) CO/CO2 constants as quoted (unverified against the paper):
#: pre-exponential 10^3.4 and activation 12400 cal/mol.
ARTHUR_A = 10.0 ** 3.4
ARTHUR_E_CAL = 12400.0
R_CAL = 1.987204  # cal/mol/K


def arthur_co_co2(T: Array) -> Array:
    """CO/CO2 molar ratio of carbon combustion, Arthur (1951) form (unverified constants)."""
    return ARTHUR_A * jnp.exp(-ARTHUR_E_CAL / (R_CAL * T))


def coke_elements(p: dict, feed, coke_mass: Array) -> dict[str, Array]:
    """Element mass flows (kg/s) in the coke: H and S by the coke's contents,
    N as ``n_to_coke`` of the feed nitrogen, carbon the rest."""
    H = p["coke_hydrogen"] * coke_mass
    S = p["coke_sulfur_factor"] * feed.sulfur * coke_mass
    N = p["n_to_coke"] * feed.nitrogen * feed.mass
    return {"C": coke_mass - H - S - N, "H": H, "S": S, "N": N}


def combustion(p: dict, el: dict, T_rg: Array, co_mode: str, air_mode: str) -> dict:
    """Flue gas and air (mol/s) for burning coke of elements ``el`` (kg/s)."""
    nC = el["C"] * 1000.0 / sp.ATOMIC_WEIGHT["C"]
    nH = el["H"] * 1000.0 / sp.ATOMIC_WEIGHT["H"]
    nS = el["S"] * 1000.0 / sp.ATOMIC_WEIGHT["S"]
    nN = el["N"] * 1000.0 / sp.ATOMIC_WEIGHT["N"]
    if co_mode == "arthur":
        r = p["arthur_factor"] * arthur_co_co2(T_rg)
    else:
        r = p["co_co2"]
    n_co2 = nC / (1.0 + r)
    n_co = nC - n_co2
    n_h2o = nH / 2.0
    n_so2 = nS
    n_n2_coke = nN / 2.0
    o2_req = n_co2 + 0.5 * n_co + 0.5 * n_h2o + n_so2
    k = sp.AIR_N2 / sp.AIR_O2
    if air_mode == "flue_o2":
        y = p["flue_o2"]
        other = n_co2 + n_co + n_h2o + n_so2 + n_n2_coke
        o2_ex = y * (other + k * o2_req) / (1.0 - y * (1.0 + k))
        o2_air = o2_req + o2_ex
    else:
        mw_air = sp.AIR_O2 * sp.MW["oxygen"] + sp.AIR_N2 * sp.MW["nitrogen"]
        o2_air = p["air_rate"] * 1000.0 / mw_air * sp.AIR_O2
        o2_ex = o2_air - o2_req
    n2_air = k * o2_air
    air = {"oxygen": o2_air, "nitrogen": n2_air}
    flue = {"nitrogen": n2_air + n_n2_coke, "oxygen": o2_ex, "carbon_dioxide": n_co2,
            "carbon_monoxide": n_co, "water": n_h2o, "sulfur_dioxide": n_so2}
    return {"air": air, "flue": flue, "co_co2": r, "o2_required": o2_req}


def heat_residual(p: dict, feed, CTO: Array, T_rg: Array, ROT: Array, coke_mass: Array,
                  comb: dict) -> Array:
    """Regenerator energy balance, in minus out (W).

    In: spent catalyst and its coke at the riser outlet temperature (coke
    sensible heat at ``cp_vapor``, formation enthalpy zero), air at
    ``air_T``. Out: regenerated catalyst and flue gas at ``T_rg``, and the
    heat loss ``regen_heat_loss``.
    """
    Fc = CTO * feed.mass
    T0 = sp.T_REF
    h_in = (Fc * p["cp_catalyst"] * (ROT - T0) + coke_mass * p["cp_vapor"] * (ROT - T0)
            + sp.flue_enthalpy(comb["air"], p["air_T"]))
    h_out = Fc * p["cp_catalyst"] * (T_rg - T0) + sp.flue_enthalpy(comb["flue"], T_rg)
    return h_in - h_out - p["regen_heat_loss"]
