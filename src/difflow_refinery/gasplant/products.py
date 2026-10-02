"""Gas plant product qualities: fuel gas, LPG and stabilized naphtha.

Each function takes a product's component flows (mol/s, in the
:class:`~difflow_refinery.gasplant.components.GasComponents` order) and
returns JAX scalars, so a quality can be a column spec target's neighbour,
a planning output or a constraint, and is differentiable through the
column that made the flows.

LPG specifications
    :data:`GPA_2140` holds the limits of GPA Standard 2140 (*Liquefied
    Petroleum Gas Specifications and Test Methods*; ASTM D1835 carries the
    same grades) for the properties a gas plant's columns set: vapor
    pressure at 100 F and the composition limits. They are given here as
    recalled and are marked **verify**: check them against the current
    edition before a product is certified on them. Two simplifications,
    both stated in what :func:`lpg_quality` returns:

    * the standard writes composition limits as *liquid volume* percent;
      they are compared here as *mole* fractions. For C3/C4 mixtures the
      two differ by a few percent of the value (a butane molecule has
      about 1.2 times a propane molecule's liquid volume), so a margin
      smaller than that is not a margin;
    * the 95% evaporated temperature (volatile residue), residue on
      evaporation, copper strip, sulfur and moisture tests are not
      computed. H2S is reported as a mole ppm, without a limit: the
      standard's sulfur tests are what decide it, and amine treating
      (:class:`~difflow_refinery.gasplant.units.AmineTreater`) is where
      the plant meets them.

Fuel gas
    The heating value is the molar lower heating value of each component
    (:attr:`GasComponents.lhv`, from heats of formation; see
    :mod:`~difflow_refinery.gasplant.components`), flow-weighted.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from difflow_refinery.gasplant.column import RVP_VL_RATIO, T_100F
from difflow_refinery.gasplant.components import GasComponents
from difflow_refinery.gasplant.thermo import PR, Cubic, CubicThermo, vapor_pressure_vl

#: One standard atmosphere (Pa): gauge = absolute - this.
P_ATM = 101325.0
_PSI = 6894.757

_VERIFY = "verify: GPA 2140 / ASTM D1835 limit as recalled, not checked against the edition"

#: GPA 2140 limits by grade: ``{grade: {quality: (kind, limit, note)}}``.
#: ``kind`` is ``"max"`` or ``"min"``. Vapor pressure is gauge, at 100 F
#: (Pa); composition limits are fractions (liquid volume in the standard,
#: compared as mole fractions here).
GPA_2140: dict[str, dict[str, tuple[str, float, str]]] = {
    "HD-5": {
        "vapor_pressure": ("max", 208.0 * _PSI, _VERIFY),
        "propane": ("min", 0.90, _VERIFY),
        "propylene": ("max", 0.05, _VERIFY),
        "butanes_plus": ("max", 0.025, _VERIFY),
    },
    "commercial_propane": {
        "vapor_pressure": ("max", 208.0 * _PSI, _VERIFY),
    },
    "commercial_butane": {
        "vapor_pressure": ("max", 70.0 * _PSI, _VERIFY),
        "pentanes_plus": ("max", 0.02, _VERIFY),
    },
}

_C3 = ("propane", "propylene")
_C4 = ("isobutane", "n_butane", "1_butene", "isobutylene", "cis_2_butene", "trans_2_butene")
_LIGHT = ("hydrogen", "nitrogen", "carbon_dioxide", "methane", "ethane", "ethylene",
          "hydrogen_sulfide", "water") + _C3 + _C4


def _frac(x, comps, names):
    idx = [comps.names.index(n) for n in names if n in comps.names]
    return jnp.sum(x[jnp.asarray(idx)]) if idx else jnp.zeros(())


def fuel_gas(flows, components: GasComponents) -> dict:
    """Fuel gas rate and heating value.

    Args:
        flows: Component flows (mol/s).
        components: The component set.

    Returns:
        ``rate`` (mol/s), ``mass`` (kg/s), ``mw`` (g/mol), ``lhv`` (J/mol),
        ``lhv_mass`` (J/kg), ``heat`` (W, the LHV times the rate) and
        ``h2s_ppm`` (mole ppm).
    """
    f = jnp.asarray(flows, dtype=float)
    F = jnp.sum(f)
    mass = jnp.sum(f * components.MW) / 1000.0
    heat = jnp.sum(f * components.lhv)
    x = f / F
    return {
        "rate": F, "mass": mass, "mw": 1000.0 * mass / F,
        "lhv": heat / F, "lhv_mass": heat / mass, "heat": heat,
        "h2s_ppm": 1e6 * _frac(x, components, ("hydrogen_sulfide",)),
    }


def reid_vapor_pressure(flows, components: GasComponents, eos: Cubic = PR):
    """Reid vapor pressure (Pa, absolute) of a liquid product.

    The ASTM D323 construction on the EOS -- vapor space four times the
    liquid volume, 100 F -- by
    :func:`~difflow_refinery.gasplant.thermo.vapor_pressure_vl`; the same
    number a column reports as ``<product>.rvp``.
    """
    return vapor_pressure_vl(CubicThermo(components, eos), jnp.asarray(flows, dtype=float),
                             T_100F, RVP_VL_RATIO)


def true_vapor_pressure(flows, components: GasComponents, T=T_100F, eos: Cubic = PR):
    """Bubble-point pressure (Pa, absolute) of a liquid product at ``T``."""
    return vapor_pressure_vl(CubicThermo(components, eos), jnp.asarray(flows, dtype=float),
                             T, 0.0)


def lpg_quality(flows, components: GasComponents, grade: str = "HD-5",
                eos: Cubic = PR, limits: Optional[dict] = None) -> dict:
    """LPG qualities against a GPA 2140 grade.

    Args:
        flows: Component flows of the LPG (mol/s).
        components: The component set.
        grade: A key of :data:`GPA_2140` (``"HD-5"``,
            ``"commercial_propane"``, ``"commercial_butane"``).
        eos: Cubic for the vapor pressure.
        limits: Replaces the grade's table, same shape as a
            :data:`GPA_2140` entry.

    Returns:
        ``values`` (``vapor_pressure`` gauge Pa at 100 F, and mole
        fractions ``propane``, ``propylene``, ``C3``, ``butanes_plus``,
        ``pentanes_plus``, ``ethane_minus``; ``h2s_ppm``),
        ``margins`` (per limit, positive when on spec, in the quality's own
        units), ``on_spec`` (all margins non-negative), ``limits`` and a
        ``basis`` note.
    """
    if limits is None:
        if grade not in GPA_2140:
            raise ValueError(f"grade must be one of {sorted(GPA_2140)}, not {grade!r}")
        limits = GPA_2140[grade]
    f = jnp.asarray(flows, dtype=float)
    x = f / jnp.sum(f)
    heavy = tuple(n for n in components.names if n not in _LIGHT)
    vals = {
        "vapor_pressure": true_vapor_pressure(f, components, T_100F, eos) - P_ATM,
        "propane": _frac(x, components, ("propane",)),
        "propylene": _frac(x, components, ("propylene",)),
        "C3": _frac(x, components, _C3),
        "butanes_plus": _frac(x, components, _C4 + heavy),
        "pentanes_plus": _frac(x, components, heavy),
        "ethane_minus": _frac(x, components, ("hydrogen", "nitrogen", "carbon_dioxide",
                                              "methane", "ethane", "ethylene")),
        "h2s_ppm": 1e6 * _frac(x, components, ("hydrogen_sulfide",)),
    }
    margins = {}
    for q, (kind, lim, _note) in limits.items():
        margins[q] = (lim - vals[q]) if kind == "max" else (vals[q] - lim)
    on_spec = jnp.all(jnp.stack([m >= 0 for m in margins.values()])) if margins else True
    return {"values": vals, "margins": margins, "on_spec": on_spec, "limits": limits,
            "basis": "composition limits compared as mole fractions (the standard: "
                     "liquid volume); vapor pressure gauge at 100 F on the EOS"}
