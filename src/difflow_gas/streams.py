"""Stream conventions for gas transmission networks.

difflow streams are dicts ``{"F_<species>": Array, "T": Array,
"P": Array}``. Gas networks use ONE pseudo-species, ``"gas"``, whose
"flow" is MASS flow in kg/s (difflow does not care about the unit,
only that it is an array). Flows are SIGNED: a negative flow means
flow against the arc's reference (from -> to) direction, which is
routine in meshed transmission networks. Because of that, flowsheets
built from these streams must be solved with
``clip_negative_flows=False`` (``GasNetworkFlowsheet`` does this by
default).

Units used throughout the plugin:

===========  =======================================
quantity     unit
===========  =======================================
mass flow    kg/s (signed)
pressure     Pa internally; bar in reporting helpers
temperature  K
Weymouth     Pa^2 / (kg/s)^2
===========  =======================================
"""

from __future__ import annotations

from difflow.streams import Stream, make_stream

#: the single pseudo-species used for gas network streams
GAS = "gas"

#: stream key of the gas mass flow
FLOW_KEY = f"F_{GAS}"


class NotAGasStream(KeyError):
    """A gas unit was handed a stream that carries something else.

    A `KeyError` still, so code that already catches one around a solve
    keeps working, but carrying a message that says what went wrong.
    """

    def __str__(self) -> str:
        # `KeyError.__str__` is `repr(args[0])`, which is right for a
        # missing key and wrong for a sentence: it would show the whole
        # explanation wrapped in quotes with the newlines escaped.
        return str(self.args[0]) if self.args else ""


def gas_flow(stream: Stream, what: str = "stream"):
    """The signed mass flow off a gas stream, or say why there is none.

    Every unit in this plugin reads ``F_gas``, because the whole plugin
    is built on the one pseudo-species. Read bare, a flowsheet whose
    species are anything else answers with ``KeyError: 'F_gas'`` --- a
    stream key the user never typed, from somewhere deep in a solve,
    with nothing to say that the species order is the thing to change.

    Args:
        stream: the stream to read.
        what: what this stream is to the caller, for the message.

    Returns:
        The signed mass flow (kg/s).

    Raises:
        NotAGasStream: if the stream has no ``F_gas``.
    """
    try:
        return stream[FLOW_KEY]
    except KeyError:
        pass
    carried = sorted(
        key[2:] for key in stream if key.startswith("F_") and len(key) > 2
    )
    carries = ", ".join(carried) if carried else "no species at all"
    raise NotAGasStream(
        f"this {what} carries {carries}, not {GAS!r}, so there is no "
        f"{FLOW_KEY!r} to read. Every unit in difflow_gas models one "
        f"pseudo-species -- a signed mass flow of {GAS!r} -- so a gas "
        f"network has to be built on a flowsheet whose species_order is "
        f"[{GAS!r}]. Gas units and multi-species units cannot share a "
        f"flowsheet; build the network separately."
    ) from None


def gas_stream(mass_flow_kg_s, T_k, P_pa) -> Stream:
    """Make a single-species gas stream.

    Args:
        mass_flow_kg_s: signed mass flow (kg/s)
        T_k: temperature (K)
        P_pa: pressure (Pa)

    Returns:
        A difflow Stream ``{"F_gas": ..., "T": ..., "P": ...}``.
    """
    return make_stream({GAS: mass_flow_kg_s}, T_k, P_pa)
