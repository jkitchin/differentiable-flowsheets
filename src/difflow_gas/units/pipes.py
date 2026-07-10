"""Pipe (and resistor) unit operations.

All pipes obey the squared-pressure Weymouth law

    p_from^2 - p_to^2 = beta q |q|      (Pa^2, q in kg/s, signed)

and come in the four modes a sequential decomposition needs:

* :class:`GasPipe`          - forward: outlet pressure from inlet
                              stream (hand-built flowsheets),
* :class:`BackPipe`         - backward: required source pressure of a
                              flow-specified boundary node,
* :class:`PipePressure`     - tree propagation: child node pressure
                              from the parent node and the arc flow,
                              in either traversal direction,
* :class:`PressureDrivenPipe` - chord: flow from the two end
                              pressures; its output closes a tear.

Resistors use the identical law with beta replaced by xi
(:func:`difflow_gas.physics.resistor_xi`), so they reuse these classes.

While a tear iterates through unphysical intermediate states, a
squared pressure can go negative; the units floor it at
:data:`MIN_P_SQUARED` ((0.5 bar)^2) before the square root. The floor
is inactive at any physically meaningful converged solution. Note the
floor zeroes gradients wherever it is active: optimization-facing code
should pose pressure constraints in squared pressure (where the
network response is nearly linear) so that optimizers do not step into
the floored region in the first place; see the package README.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_gas.streams import FLOW_KEY, gas_stream

#: floor for squared pressure during tear iteration, (0.5 bar)^2 in Pa^2
MIN_P_SQUARED = (0.5e5) ** 2

#: shared literature references for the Weymouth pipe units
_PIPE_REFS = [
    "Koch, T. et al., Evaluating Gas Network Capacities, MOS-SIAM Series on Optimization (2015).",
    "Weymouth, T.R., Trans. ASME 34, 185 (1912).",
]


@dataclass
class PipeParams(ParamsMixin):
    """Parameters of every pipe/resistor unit.

    Attributes:
        beta: squared-pressure resistance coefficient, Pa^2/(kg/s)^2
            (:func:`difflow_gas.physics.weymouth_beta` or
            :func:`difflow_gas.physics.resistor_xi`).
    """

    beta: float


class GasPipe:
    """Weymouth pipe in forward mode: propagate pressure downstream.

    ``p_out = sqrt(p_in^2 - beta q |q|)``, isothermal. The inlet
    stream's (signed) flow is carried through unchanged.
    """

    symbol = "Pipe →"
    equations = [r"p_\mathrm{out} = \sqrt{p_\mathrm{in}^2 - \beta\,q\,|q|}"]
    references = _PIPE_REFS

    def __init__(self, beta: float, min_p_squared: float = MIN_P_SQUARED):
        self.params = PipeParams(beta=beta)
        self.min_p_squared = min_p_squared

    def __call__(self, inlet: Stream) -> Stream:
        q = inlet[FLOW_KEY]
        p2 = inlet["P"] ** 2 - self.params.beta * q * jnp.abs(q)
        P_out = jnp.sqrt(jnp.maximum(p2, self.min_p_squared))
        return gas_stream(q, inlet["T"], P_out)


class BackPipe:
    """Weymouth pipe in backward mode: required upstream (source) pressure.

    Used for flow-specified entries whose pressure is an *output* of
    the load flow: given the node stream at the pipe outlet and the
    flow specification (the feed), compute
    ``p_src = sqrt(p_node^2 + beta q |q|)``.
    """

    symbol = "Pipe ←"
    equations = [r"p_\mathrm{src} = \sqrt{p_\mathrm{node}^2 + \beta\,q\,|q|}"]
    references = _PIPE_REFS

    def __init__(self, beta: float):
        self.params = PipeParams(beta=beta)

    def __call__(self, node: Stream, feed: Stream) -> Stream:
        q = feed[FLOW_KEY]
        p2 = node["P"] ** 2 + self.params.beta * q * jnp.abs(q)
        return gas_stream(q, node["T"], jnp.sqrt(p2))


class PipePressure:
    """Tree-propagation pipe: child node pressure from parent and flow.

    ``direction=+1`` when the tree is traversed with the arc (the
    parent is the arc's ``from`` node):
    ``p_child^2 = p_parent^2 - beta q |q|``. ``direction=-1`` against
    the arc: ``p_child^2 = p_parent^2 + beta q |q|``. The output
    stream carries the arc flow at the child node pressure.
    """

    symbol = "Pipe (tree)"
    equations = [r"p_\mathrm{child}^2 = p_\mathrm{parent}^2 \mp \beta\,q\,|q|"]
    references = _PIPE_REFS

    def __init__(self, beta: float, direction: int,
                 min_p_squared: float = MIN_P_SQUARED):
        if direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")
        self.params = PipeParams(beta=beta)
        self.direction = direction
        self.min_p_squared = min_p_squared

    def __call__(self, parent: Stream, flow: Stream) -> Stream:
        q = flow[FLOW_KEY]
        drop = self.params.beta * q * jnp.abs(q)
        p2 = parent["P"] ** 2 - self.direction * drop
        P = jnp.sqrt(jnp.maximum(p2, self.min_p_squared))
        return gas_stream(q, parent["T"], P)


class PressureDrivenPipe:
    """Weymouth pipe in pressure-driven mode: flow from end pressures.

    ``q = sign(dp2) sqrt(|dp2| / beta)`` with
    ``dp2 = p_from^2 - p_to^2``. These are the chord (loop-closing)
    elements of a decomposition: their computed flows feed the tear
    updates. The outlet stream carries the computed signed flow at the
    downstream node pressure.
    """

    symbol = "Pipe (chord)"
    equations = [
        r"q = \operatorname{sign}(\Delta p^2)\,\sqrt{|\Delta p^2|/\beta},"
        r"\quad \Delta p^2 = p_\mathrm{from}^2 - p_\mathrm{to}^2"
    ]
    references = _PIPE_REFS

    def __init__(self, beta: float):
        self.params = PipeParams(beta=beta)

    def __call__(self, upstream: Stream, downstream: Stream) -> Stream:
        dp2 = upstream["P"] ** 2 - downstream["P"] ** 2
        q = jnp.sign(dp2) * jnp.sqrt(jnp.abs(dp2) / self.params.beta)
        return gas_stream(q, upstream["T"], downstream["P"])
