"""Water streams: plain difflow streams plus conversions.

A difflow_water stream is an ordinary difflow stream (``make_stream``):
molar flows in mol/s under ``F_<species>`` plus ``T`` (K) and ``P`` (Pa).
Seawater carries two species:

- ``H2O``: water, molar mass :data:`MW_WATER`;
- ``TDS``: sea salt, counted in moles of the Reference-Composition mean
  molar mass ``M_S = 31.4038218 g/mol`` (TEOS-10 [S4], from Millero et al.
  2008 [S3]), so one "mole" of TDS is ``M_S`` grams of sea salt.

The helpers here convert to the quantities water engineers use (salinity,
mg/L, volumetric flow) and to WaterTAP's state variables (mass flow per
component), so a comparison case can be fed identical streams.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, combine_streams, make_stream
from difflow_water.chemistry.seawater import MW_SEA_SALT, MW_WATER, density

WATER = "H2O"
SALT = "TDS"

#: kg/mol for each species a seawater stream carries
MOLAR_MASS = {WATER: MW_WATER, SALT: MW_SEA_SALT}


def make_seawater_stream(flow_mass=None, salinity=0.035, T=298.15, P=101325.0,
                         flow_vol=None) -> Stream:
    """A seawater stream from a total flow and a salinity.

    Give exactly one of ``flow_mass`` (kg/s of seawater) or ``flow_vol``
    (m^3/s of seawater at ``T`` and ``salinity``).

    Args:
        flow_mass: Total mass flow (kg/s).
        salinity: Mass fraction of sea salt (kg/kg).
        T: Temperature (K).
        P: Pressure (Pa).
        flow_vol: Total volumetric flow (m^3/s).

    Returns:
        Stream with ``F_H2O`` and ``F_TDS`` (mol/s), ``T`` and ``P``.

    Example:
        >>> s = make_seawater_stream(flow_vol=1e-3, salinity=0.035)
        >>> round(float(mass_flow(s)), 4)
        1.0236
    """
    if (flow_mass is None) == (flow_vol is None):
        raise ValueError("give exactly one of flow_mass or flow_vol")
    S = jnp.asarray(salinity, dtype=jnp.float64)
    if flow_mass is None:
        flow_mass = jnp.asarray(flow_vol, dtype=jnp.float64) * density(T, S)
    m = jnp.asarray(flow_mass, dtype=jnp.float64)
    return make_stream({WATER: m * (1.0 - S) / MW_WATER, SALT: m * S / MW_SEA_SALT}, T, P)


def mass_flows(stream: Stream) -> tuple[Array, Array]:
    """``(water, salt)`` mass flows (kg/s)."""
    return stream[f"F_{WATER}"] * MW_WATER, stream[f"F_{SALT}"] * MW_SEA_SALT


def mass_flow(stream: Stream) -> Array:
    """Total mass flow (kg/s)."""
    w, s = mass_flows(stream)
    return w + s


def salinity(stream: Stream) -> Array:
    """Mass fraction of sea salt (kg/kg); 0 for an empty stream."""
    w, s = mass_flows(stream)
    tot = w + s
    return s / jnp.where(tot > 0, tot, 1.0)


def stream_density(stream: Stream) -> Array:
    """Seawater density at the stream's T and salinity (kg/m^3)."""
    return density(stream["T"], salinity(stream))


def volumetric_flow(stream: Stream) -> Array:
    """Volumetric flow (m^3/s) at the stream's T and salinity."""
    return mass_flow(stream) / stream_density(stream)


def concentration_mg_per_L(stream: Stream) -> Array:
    """Total dissolved solids as mg/L (= g/m^3) of solution."""
    return salinity(stream) * stream_density(stream) * 1e3


def to_watertap_state(stream: Stream) -> dict[str, float]:
    """WaterTAP seawater state variables for a stream.

    Returns:
        ``{"flow_mass_phase_comp": {("Liq", "H2O"): kg/s, ("Liq", "TDS"):
        kg/s}, "temperature": K, "pressure": Pa}`` as plain floats.
    """
    w, s = mass_flows(stream)
    return {
        "flow_mass_phase_comp": {("Liq", "H2O"): float(w), ("Liq", "TDS"): float(s)},
        "temperature": float(stream["T"]),
        "pressure": float(stream["P"]),
    }


def from_watertap_state(flow_mass_water, flow_mass_tds, T, P) -> Stream:
    """A stream from WaterTAP seawater state variables (kg/s, K, Pa)."""
    return make_stream({WATER: jnp.asarray(flow_mass_water) / MW_WATER,
                        SALT: jnp.asarray(flow_mass_tds) / MW_SEA_SALT}, T, P)


def combine_seawater(*streams: Stream) -> Stream:
    """Mix seawater streams: summed flows, flow-weighted T, lowest P.

    This is :func:`difflow.streams.combine_streams` (WaterTAP's mixer with
    ``momentum_mixing_type=equality`` instead requires equal pressures).
    """
    return combine_streams(*streams)
