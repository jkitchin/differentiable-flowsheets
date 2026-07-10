"""Valve, control valve and short-pipe unit operations.

GasLib semantics (steady state, discrete states resolved to the
continuous benchmark relaxation):

* **open valve** and **short pipe**: pressure equality across the arc,
  flow free (determined by the rest of the network). In a sequential
  decomposition these must therefore be tree arcs; a closed valve is a
  topology change (remove the arc), not a unit mode.
* **control valve**: reduces pressure by a controllable amount. In the
  sequential decomposition the drop is a differentiable unit parameter
  (default 0), mirroring how a compressor's ratio is a parameter; the
  drop bound |dp| <= dp_max from the data belongs to the optimization
  layer, not the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_gas.streams import FLOW_KEY, gas_stream

#: pressure floor (Pa) after a control-valve drop, matching the pipe
#: units' MIN_P_SQUARED floor of (0.5 bar)^2
MIN_P = 0.5e5


class OpenValve:
    """Open valve in forward mode: no pressure loss (p_out = p_in)."""

    def __call__(self, inlet: Stream) -> Stream:
        return gas_stream(inlet[FLOW_KEY], inlet["T"], inlet["P"])


class PressureEqual:
    """Tree-propagation unit for open valves and short pipes.

    The child node sees the parent pressure unchanged (in either
    traversal direction); the output stream carries the arc flow.
    """

    def __call__(self, parent: Stream, flow: Stream) -> Stream:
        return gas_stream(flow[FLOW_KEY], parent["T"], parent["P"])


@dataclass
class ControlValveParams(ParamsMixin):
    """Parameters of a control valve.

    Attributes:
        dp_pa: controlled pressure reduction across the arc (Pa),
            nonnegative, in the arc's from->to direction. The decision
            variable of the station.
    """

    dp_pa: float = 0.0


class ControlValveDrop:
    """Tree-propagation control valve: parametric linear pressure drop.

    ``direction=+1``: the child node is the arc's ``to`` end,
    ``P_child = P_parent - dp``. ``direction=-1``: the child is the
    ``from`` end, ``P_child = P_parent + dp``. The result is floored
    at :data:`MIN_P` so unphysical parameter/state combinations during
    iteration cannot produce nonpositive pressures.
    """

    def __init__(self, dp_pa: float = 0.0, direction: int = +1):
        if direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")
        self.params = ControlValveParams(dp_pa=dp_pa)
        self.direction = direction

    def __call__(self, parent: Stream, flow: Stream) -> Stream:
        P = parent["P"] - self.direction * self.params.dp_pa
        return gas_stream(
            flow[FLOW_KEY], parent["T"], jnp.maximum(P, MIN_P)
        )
