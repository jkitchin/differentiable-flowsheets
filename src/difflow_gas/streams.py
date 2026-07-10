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
