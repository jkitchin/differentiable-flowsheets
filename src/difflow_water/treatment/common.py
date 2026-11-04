"""Stream conventions shared by the conventional-treatment units.

Treatment streams are plain difflow streams: ``F_H2O`` and one ``F_<name>``
per solute or particle class (mol/s), ``T``, ``P``. Concentrations need
molar masses: each unit takes the ones it uses (``mw`` arguments, kg/mol;
water is :data:`~difflow_water.chemistry.seawater.MW_WATER`). Volumetric
flow is mass flow over a constant density ``rho`` (default 1000 kg/m^3,
WaterTAP's MCAS property package default), so a dilute stream's
concentrations match WaterTAP's.

Two pseudo-components with a stated convention:

- particles (``F_TSS`` and size classes): molar mass 1e-3 kg/mol, i.e.
  ``1 mol = 1 g`` of suspended solids;
- organisms (e.g. ``F_Giardia``): ``1 mol = 1 organism`` (count/s), only
  ever used as a ratio (log inactivation), never with a molar mass.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream
from difflow_water.chemistry.seawater import MW_WATER

RHO_DEFAULT = 1000.0   # kg/m^3, WaterTAP MCAS default (convention)
MW_PARTICLE = 1e-3     # kg/mol: 1 mol of a particle class = 1 g


def volumetric_flow(stream: Stream, mw: dict[str, float] | None = None,
                    rho: float | Array = RHO_DEFAULT) -> Array:
    """Volumetric flow (m^3/s): water plus the solutes in ``mw`` over ``rho``."""
    mass = stream["F_H2O"] * MW_WATER
    for k, m in (mw or {}).items():
        mass = mass + stream.get(f"F_{k}", 0.0) * m
    return mass / rho


def mass_concentration(stream: Stream, name: str, mw: float, V) -> Array:
    """Mass concentration of ``name`` (kg/m^3) in a stream of volume flow V."""
    return stream[f"F_{name}"] * mw / V


def scale_component(stream: Stream, name: str, factor) -> Stream:
    """A copy of ``stream`` with ``F_<name>`` multiplied by ``factor``."""
    out = dict(stream)
    out[f"F_{name}"] = stream[f"F_{name}"] * factor
    return out


def only(stream: Stream, names) -> Stream:
    """A stream with only the components ``names`` (others zero), same T, P."""
    out = {k: (v if (not k.startswith("F_") or k[2:] in names) else jnp.zeros_like(v))
           for k, v in stream.items()}
    return out
