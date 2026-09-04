"""Stream conventions for electrical networks.

A difflow stream is ``{"F_<species>": Array, "T": Array, "P": Array}``.
Electrical networks map onto that as follows, and the mapping is worth
stating plainly because two of the three slots are being used for
something other than their names:

=================  ==============================================
stream key         electrical quantity
=================  ==============================================
``F_P``            real power flow (pu on ``base_mva``), signed
``F_Q``            reactive power flow (pu), signed
``P``              VOLTAGE MAGNITUDE (pu)
``T``              VOLTAGE ANGLE (radians)
=================  ==============================================

The ``P`` slot carrying voltage is not a pun. In a flowsheet the
pressure slot is the *potential that drives flow through a resistance*,
and voltage is exactly that; the gas plugin puts pressure there for the
same reason. The angle has no fluid analogue at all --- it is the second
coordinate of the complex potential, and AC networks need both --- so it
takes the remaining slot. Nothing downstream interprets it as a
temperature.

Signs
-----

Power flows are SIGNED and measured along a branch's reference (from ->
to) direction, so a negative ``F_P`` means real power flowing against
the arrow. That is routine in a meshed network and happens on radial
feeders too as soon as there is distributed generation. Flowsheets built
from these streams must therefore be solved with
``clip_negative_flows=False``.

Two pseudo-species rather than one complex flow: difflow's stream
species are real arrays, and splitting the complex power into ``P`` and
``Q`` keeps every difflow utility --- mixing, splitting, mass balance
reporting --- working unchanged. :func:`complex_power` and
:func:`from_complex` convert at the boundary of anything that wants
phasor arithmetic.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream

#: pseudo-species for real power
REAL = "P"

#: pseudo-species for reactive power
REACTIVE = "Q"

#: stream key of the real power flow
P_KEY = f"F_{REAL}"

#: stream key of the reactive power flow
Q_KEY = f"F_{REACTIVE}"

#: the species order every power flowsheet uses
SPECIES = (REAL, REACTIVE)


def power_stream(p_pu, q_pu, vm_pu, va_rad) -> Stream:
    """Make an electrical stream.

    Args:
        p_pu: signed real power flow (pu).
        q_pu: signed reactive power flow (pu).
        vm_pu: voltage magnitude (pu).
        va_rad: voltage angle (radians).

    Returns:
        A difflow Stream ``{"F_P", "F_Q", "T", "P"}``.

    Example:
        >>> s = power_stream(1.0, 0.3, 1.02, 0.0)
        >>> float(s["P"])
        1.02
    """
    return make_stream({REAL: p_pu, REACTIVE: q_pu}, va_rad, vm_pu)


def complex_power(stream: Stream) -> Array:
    """The stream's complex power ``P + jQ`` (pu)."""
    return jnp.asarray(stream[P_KEY]) + 1j * jnp.asarray(stream[Q_KEY])


def complex_voltage(stream: Stream) -> Array:
    """The stream's complex voltage ``|V| exp(j theta)`` (pu)."""
    return jnp.asarray(stream["P"]) * jnp.exp(1j * jnp.asarray(stream["T"]))


def from_complex(s_complex, v_complex) -> Stream:
    """Make a stream from a complex power and a complex voltage.

    The inverse of :func:`complex_power` / :func:`complex_voltage`, and
    the form every unit operation in :mod:`difflow_power.units` returns
    its result through.
    """
    return power_stream(
        jnp.real(s_complex),
        jnp.imag(s_complex),
        jnp.abs(v_complex),
        jnp.angle(v_complex),
    )


def current(stream: Stream) -> Array:
    """The complex current the stream carries (pu), ``conj(S / V)``.

    Undefined at zero voltage, which no converged state has; a stream
    built as a solver's initial guess might, so guard the guess rather
    than this.
    """
    return jnp.conj(complex_power(stream) / complex_voltage(stream))


def apparent_power(stream: Stream) -> Array:
    """``|S|`` of the stream (pu) --- what a thermal rating limits."""
    return jnp.abs(complex_power(stream))


def power_factor(stream: Stream) -> Array:
    """``P / |S|``, signed by the direction of real power.

    1.0 is a purely real flow; a low value means the branch is carrying
    vars it gets no revenue for and losses it does.
    """
    s = complex_power(stream)
    return jnp.real(s) / jnp.maximum(jnp.abs(s), 1e-30)
