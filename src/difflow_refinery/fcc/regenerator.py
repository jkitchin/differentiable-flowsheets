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
from :data:`~difflow_refinery.fcc.species.CP_IG` (the refinery's shared
thermochemistry table, :mod:`difflow_refinery.thermochemistry`). Coke's enthalpy of
formation is taken as zero (graphite, H2, S, N2 elements), so its heat of
combustion follows from its H content -- the convention the issue asks for.
The regenerated catalyst leaves clean (no carbon on regenerated catalyst).

The stoichiometry, air and flue-gas enthalpy live in the shared
:mod:`difflow_refinery.combustion` (#470), which the CCR regenerator reuses;
this module keeps the FCC's coke split and heat balance.
"""

from __future__ import annotations

from jax import Array

from difflow_refinery import combustion as cb
from difflow_refinery.combustion import ARTHUR_A, ARTHUR_E_CAL, R_CAL, arthur_co_co2  # noqa: F401
from difflow_refinery.fcc import species as sp


def coke_elements(p: dict, feed, coke_mass: Array) -> dict[str, Array]:
    """Element mass flows (kg/s) in the coke: H and S by the coke's contents,
    N as ``n_to_coke`` of the feed nitrogen, carbon the rest."""
    H = p["coke_hydrogen"] * coke_mass
    S = p["coke_sulfur_factor"] * feed.sulfur * coke_mass
    N = p["n_to_coke"] * feed.nitrogen * feed.mass
    return {"C": coke_mass - H - S - N, "H": H, "S": S, "N": N}


def combustion(p: dict, el: dict, T_rg: Array, co_mode: str, air_mode: str) -> dict:
    """Flue gas and air (mol/s) for burning coke of elements ``el`` (kg/s)."""
    if co_mode == "arthur":
        r = p["arthur_factor"] * arthur_co_co2(T_rg)
    else:
        r = p["co_co2"]
    if air_mode == "flue_o2":
        return cb.burn(el, r, flue_o2=p["flue_o2"])
    return cb.burn(el, r, air_rate=p["air_rate"])


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
