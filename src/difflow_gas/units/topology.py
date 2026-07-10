"""Topology units: sources, junctions, splits and balance bookkeeping.

These units carry no pipe physics; they implement the mass-balance and
pressure-reference bookkeeping a sequential gas network schedule
needs. :class:`AffineFlow` is the workhorse of the computed
decomposition (one per tree node); the split/junction units serve
hand-built decompositions in the GasLib-11 style.
"""

from __future__ import annotations

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_gas.streams import FLOW_KEY, gas_stream

#: shared literature reference for the topology / balance units
_TOPOLOGY_REFS = [
    "Koch, T. et al., Evaluating Gas Network Capacities, MOS-SIAM Series on Optimization (2015).",
]


@dataclass
class SourceHeadParams(ParamsMixin):
    """Parameters of the slack-pressure source.

    Attributes:
        P_set: prescribed (slack) pressure, Pa.
    """

    P_set: float


class SourceHead:
    """Pin a feed stream to a specified (slack) source pressure.

    Gas nomination scenarios specify boundary *flows*; one node must
    provide the pressure level. Making it a unit parameter (rather
    than baking it into the feed) keeps it differentiable through
    ``Flowsheet._apply_params``.
    """

    symbol = "Source"
    equations = [r"p = p_\mathrm{set}"]
    references = _TOPOLOGY_REFS

    def __init__(self, P_set: float):
        self.params = SourceHeadParams(P_set=P_set)

    def __call__(self, inlet: Stream) -> Stream:
        return gas_stream(inlet[FLOW_KEY], inlet["T"], self.params.P_set)


@dataclass
class AffineFlowParams(ParamsMixin):
    """Parameters of an affine flow balance.

    Attributes:
        const: fixed term (kg/s), typically the node's nomination.
    """

    const: float


class AffineFlow:
    """Parent-arc flow of a tree node from its local mass balance.

    ``flow = const + sum(sign_i * inlet_i.F)`` where the inlets are the
    node's child tree-arc flows and incident chord tear streams (see
    :class:`difflow_gas.network.BalanceSpec`). T and P on the output
    are placeholders: flow streams only carry flow.
    """

    symbol = "Flow balance"
    equations = [r"q = c + \textstyle\sum_i s_i\,q_i"]
    references = _TOPOLOGY_REFS

    def __init__(self, const: float, signs: tuple[float, ...],
                 T_k: float, P_pa: float):
        self.params = AffineFlowParams(const=const)
        self.signs = tuple(signs)
        self.T_k = T_k
        self.P_pa = P_pa

    def __call__(self, *inlets: Stream) -> Stream:
        if len(inlets) != len(self.signs):
            raise ValueError(
                f"AffineFlow expected {len(self.signs)} inlets, "
                f"got {len(inlets)}"
            )
        q = self.params.const
        for s, inlet in zip(self.signs, inlets):
            q = q + s * inlet[FLOW_KEY]
        return gas_stream(q, self.T_k, self.P_pa)


@dataclass
class FlowSplitParams(ParamsMixin):
    """Parameters of a fixed-draw flow split.

    Attributes:
        w: flow (kg/s) drawn into the first outlet.
    """

    w: float


class FlowSplit:
    """Split a node stream: fixed flow ``w`` to outlet 1, rest to outlet 2.

    Used where a demand branch leaves a node in a hand-built
    decomposition: the branch flow is fixed by the nomination, so the
    split is explicit.
    """

    symbol = "Split (fixed)"
    equations = [r"q_1 = w,\quad q_2 = q_\mathrm{in} - w"]
    references = _TOPOLOGY_REFS

    def __init__(self, w: float):
        self.params = FlowSplitParams(w=w)

    def __call__(self, inlet: Stream) -> tuple[Stream, Stream]:
        w = self.params.w
        out1 = gas_stream(w, inlet["T"], inlet["P"])
        out2 = gas_stream(inlet[FLOW_KEY] - w, inlet["T"], inlet["P"])
        return out1, out2


class TearSplit:
    """Split a node stream using a tear stream as the flow specification.

    Outlet 1 takes the tear stream's flow, outlet 2 the remainder;
    both leave at the node's T and P. This is the entry point of a
    hand-built recycle that resolves a network loop.
    """

    symbol = "Split (tear)"
    equations = [r"q_1 = q_\mathrm{tear},\quad q_2 = q_\mathrm{in} - q_\mathrm{tear}"]
    references = _TOPOLOGY_REFS

    def __call__(self, inlet: Stream, spec: Stream) -> tuple[Stream, Stream]:
        w = spec[FLOW_KEY]
        out1 = gas_stream(w, inlet["T"], inlet["P"])
        out2 = gas_stream(inlet[FLOW_KEY] - w, inlet["T"], inlet["P"])
        return out1, out2


class Junction:
    """Network node: sum flows; node pressure comes from the first inlet.

    In a gas network every arc at a node sees the same nodal pressure,
    so unlike difflow's ``combine_streams`` (min P) the junction takes
    its pressure from the designated pressure-defining branch. At a
    converged solution all inlet pressures agree; the difference is a
    convergence residual, not a modeling choice. The outlet temperature
    is the flow-weighted mean.
    """

    symbol = "Junction"
    equations = [r"q = \textstyle\sum_i q_i,\quad p = p_\mathrm{ref}"]
    references = _TOPOLOGY_REFS

    def __call__(self, *inlets: Stream) -> Stream:
        q_tot = sum(s[FLOW_KEY] for s in inlets)
        T = sum(s[FLOW_KEY] * s["T"] for s in inlets) / (q_tot + 1e-30)
        return gas_stream(q_tot, T, inlets[0]["P"])


class FlowMinus:
    """Stream a minus stream b (flows); T and P from stream a.

    Tear-update bookkeeping for hand-built decompositions.
    """

    symbol = "Flow −"
    equations = [r"q = q_a - q_b"]
    references = _TOPOLOGY_REFS

    def __call__(self, a: Stream, b: Stream) -> Stream:
        return gas_stream(a[FLOW_KEY] - b[FLOW_KEY], a["T"], a["P"])
