"""Stream representation for difflow.

Streams are represented as dictionaries with:
- Species molar flows (mol/s) with keys like 'F_A', 'F_B', etc.
- Temperature 'T' (K)
- Pressure 'P' (Pa)

This simple representation is a JAX pytree by default, enabling
automatic differentiation through stream operations.
"""

from typing import TypeAlias
import jax.numpy as jnp
from jax import Array

# Type alias for streams
Stream: TypeAlias = dict[str, Array | float]


def make_stream(
    flows: dict[str, float | Array],
    T: float | Array,
    P: float | Array,
) -> Stream:
    """Create a stream dictionary.

    Args:
        flows: Dictionary of species molar flows {species_name: flow_rate}.
               Flow rates in mol/s.
        T: Temperature in K
        P: Pressure in Pa

    Returns:
        Stream dictionary with 'F_<species>', 'T', and 'P' keys.
    """
    stream = {}
    for species, flow in flows.items():
        key = f"F_{species}" if not species.startswith("F_") else species
        stream[key] = jnp.asarray(flow, dtype=jnp.float64)
    stream["T"] = jnp.asarray(T, dtype=jnp.float64)
    stream["P"] = jnp.asarray(P, dtype=jnp.float64)
    return stream


def get_flows(stream: Stream) -> dict[str, Array]:
    """Extract species flows from a stream.

    Returns:
        Dictionary of {species_name: flow_rate} without the 'F_' prefix.
    """
    return {
        k[2:]: v for k, v in stream.items()
        if k.startswith("F_")
    }


def get_species(stream: Stream) -> list[str]:
    """Get list of species names in the stream (without 'F_' prefix)."""
    return [k[2:] for k in stream.keys() if k.startswith("F_")]


def get_flow_array(stream: Stream, species_order: list[str]) -> Array:
    """Get flows as an array in specified species order.

    Args:
        stream: Stream dictionary
        species_order: List of species names (without 'F_' prefix)

    Returns:
        JAX array of flows in the specified order
    """
    return jnp.array([stream[f"F_{s}"] for s in species_order])


def total_flow(stream: Stream) -> Array:
    """Calculate total molar flow rate of a stream."""
    flows = get_flows(stream)
    return sum(flows.values())


def mole_fractions(stream: Stream) -> dict[str, Array]:
    """Calculate mole fractions of each species."""
    flows = get_flows(stream)
    total = total_flow(stream)
    return {species: flow / total for species, flow in flows.items()}


def combine_streams(*streams: Stream) -> Stream:
    """Combine multiple streams by adding flows.

    All streams must have the same species. Temperature is computed as a
    flow-weighted average (equivalent to assuming equal Cp for all species,
    which is a standard first approximation for adiabatic mixing when no
    thermodynamic model is available). Pressure is taken as the minimum of
    the inlet pressures.

    Args:
        *streams: Variable number of streams to combine

    Returns:
        Combined stream with summed flows, flow-weighted average T, and
        minimum P.
    """
    if not streams:
        raise ValueError("At least one stream required")

    result = {}
    species = get_species(streams[0])

    # Sum flows for each species
    for s in species:
        key = f"F_{s}"
        result[key] = sum(stream[key] for stream in streams)

    # Adiabatic mixing: approximate by flow-weighted average T
    # (assumes equal Cp; for accurate results use IdealThermo)
    F_total = sum(v for k, v in result.items() if k.startswith("F_"))
    T_mix = sum(
        streams[i]["T"] * sum(v for k, v in streams[i].items() if k.startswith("F_")) / F_total
        for i in range(len(streams))
    )
    result["T"] = T_mix

    # Use minimum pressure (most conservative for downstream units).
    # Use jnp.minimum reduce to stay JAX-compatible inside jit/vmap.
    P = streams[0]["P"]
    for s in streams[1:]:
        P = jnp.minimum(P, s["P"])
    result["P"] = P

    return result


def scale_stream(stream: Stream, factor: float | Array) -> Stream:
    """Scale all flows in a stream by a factor.

    Args:
        stream: Input stream
        factor: Scaling factor for flows

    Returns:
        New stream with scaled flows, same T and P
    """
    result = {}
    for key, value in stream.items():
        if key.startswith("F_"):
            result[key] = value * factor
        else:
            result[key] = value
    return result
