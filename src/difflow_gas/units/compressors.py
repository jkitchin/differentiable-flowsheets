"""Compressor station unit operations.

The benchmark compressor model is a fixed pressure ratio with an
isothermal (aftercooled) outlet:

    p_out = ratio * p_in,      T_out = T_in

The ratio is the differentiable decision parameter; shaft power is
evaluated separately from the solved streams
(:func:`difflow_gas.physics.smoothed_power_w`,
:func:`adiabatic_power_w`).
"""

from __future__ import annotations

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_gas.physics import (
    DEFAULT_CP,
    DEFAULT_ETA_AD,
    DEFAULT_KAPPA,
)
from difflow_gas.streams import FLOW_KEY, gas_stream


@dataclass
class CompressorParams(ParamsMixin):
    """Parameters of a compressor station.

    Attributes:
        ratio: outlet/inlet pressure ratio, the decision variable.
        eta_ad: adiabatic efficiency (used by power helpers).
        kappa: isentropic exponent.
        cp: specific heat capacity, J/(kg K).
    """

    ratio: float
    eta_ad: float = DEFAULT_ETA_AD
    kappa: float = DEFAULT_KAPPA
    cp: float = DEFAULT_CP


class Compressor:
    """Fixed-ratio compressor in forward mode (hand-built flowsheets).

    Single inlet; the (signed) flow is carried through and the pressure
    is boosted by the ratio.
    """

    def __init__(self, ratio: float, **kwargs):
        self.params = CompressorParams(ratio=ratio, **kwargs)

    def __call__(self, inlet: Stream) -> Stream:
        return gas_stream(
            inlet[FLOW_KEY], inlet["T"], inlet["P"] * self.params.ratio
        )


class CompressorBoost:
    """Tree-propagation compressor: child node pressure from parent.

    ``direction=+1``: the child node is the station outlet,
    ``P_child = ratio * P_parent``. ``direction=-1``: the child is the
    station inlet, ``P_child = P_parent / ratio``. The output stream
    carries the arc flow at the child node pressure.
    """

    def __init__(self, ratio: float, direction: int, **kwargs):
        if direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")
        self.params = CompressorParams(ratio=ratio, **kwargs)
        self.direction = direction

    def __call__(self, parent: Stream, flow: Stream) -> Stream:
        r = self.params.ratio
        P = parent["P"] * r if self.direction > 0 else parent["P"] / r
        return gas_stream(flow[FLOW_KEY], parent["T"], P)


def adiabatic_power_w(
    inlet: Stream,
    outlet: Stream,
    eta_ad: float = DEFAULT_ETA_AD,
    kappa: float = DEFAULT_KAPPA,
    cp: float = DEFAULT_CP,
):
    """Shaft power (W) of a compressor from its solved in/out streams.

    The ratio is recovered from the stream pressures, so this works
    inside stream-only objective functions:

        P = q c_p T_in ((p_out/p_in)^((k-1)/k) - 1) / eta_ad

    Uses the plain (unsmoothed) flow; see
    :func:`difflow_gas.physics.smoothed_power_w` for the C^1 variant
    matching equation-oriented NLP objectives.
    """
    ratio = outlet["P"] / inlet["P"]
    exponent = (kappa - 1.0) / kappa
    q = inlet[FLOW_KEY]
    return q * cp * inlet["T"] * (ratio ** exponent - 1.0) / eta_ad
