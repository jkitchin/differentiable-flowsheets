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
    phase: str | None = None,
) -> Stream:
    """Create a stream dictionary.

    Args:
        flows: Dictionary of species molar flows {species_name: flow_rate}.
               Flow rates in mol/s.
        T: Temperature in K
        P: Pressure in Pa
        phase: Optional phase label: 'liquid', 'vapor', 'two_phase', or None

    Returns:
        Stream dictionary with 'F_<species>', 'T', and 'P' keys.
        If phase is provided, includes 'phase' key.
    """
    stream = {}
    for species, flow in flows.items():
        key = f"F_{species}" if not species.startswith("F_") else species
        stream[key] = jnp.asarray(flow, dtype=jnp.float64)
    stream["T"] = jnp.asarray(T, dtype=jnp.float64)
    stream["P"] = jnp.asarray(P, dtype=jnp.float64)
    if phase is not None:
        stream["phase"] = phase
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

    The outlet carries the UNION of the inlet species: a species missing from
    one inlet is taken as zero flow there, so mixing a pure-A feed with a
    pure-B feed gives both. Temperature is computed as a flow-weighted
    average (equivalent to assuming equal Cp for all species, which is a
    standard first approximation for adiabatic mixing when no thermodynamic
    model is available). Pressure is taken as the minimum of the inlet
    pressures.

    Args:
        *streams: Variable number of streams to combine

    Returns:
        Combined stream with summed flows, flow-weighted average T, and
        minimum P.
    """
    if not streams:
        raise ValueError("At least one stream required")

    result = {}
    # Union of species, first-seen order. Taking only the first stream's
    # species drops the others' silently: the flows vanish from the mole
    # balance AND the T weights stop summing to one, which puts the mixed
    # temperature outside the range of the inlet temperatures.
    species: list[str] = []
    for stream in streams:
        for s in get_species(stream):
            if s not in species:
                species.append(s)

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    for s in species:
        key = f"F_{s}"
        result[key] = sum(stream.get(key, zero) for stream in streams)

    # Adiabatic mixing: approximate by flow-weighted average T
    # (assumes equal Cp; for accurate results use IdealThermo)
    totals = [
        sum(
            (v for k, v in stream.items() if k.startswith("F_")),
            start=zero,
        )
        for stream in streams
    ]
    F_total = sum(totals, start=zero)
    # With no flow at all the weights are 0/0. Fall back to an equal-weight
    # average rather than emitting a NaN temperature (and a NaN gradient)
    # into the rest of the flowsheet; a zero stream is an ordinary outcome of
    # a split or a purge.
    any_flow = F_total > 0.0
    denom = jnp.where(any_flow, F_total, 1.0)
    n = len(streams)
    result["T"] = sum(
        streams[i]["T"] * jnp.where(any_flow, totals[i] / denom, 1.0 / n)
        for i in range(n)
    )

    # Use minimum pressure (most conservative for downstream units).
    # Use jnp.minimum reduce to stay JAX-compatible inside jit/vmap.
    P = streams[0]["P"]
    for s in streams[1:]:
        P = jnp.minimum(P, s["P"])
    result["P"] = P

    # Phase compatibility check
    phases = [s.get("phase") for s in streams]
    labeled_phases = [p for p in phases if p is not None]
    if labeled_phases:
        unique = set(labeled_phases)
        if len(unique) == 1:
            result["phase"] = labeled_phases[0]
        else:
            result["phase"] = "two_phase"
            result["phase_mismatch"] = True

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
