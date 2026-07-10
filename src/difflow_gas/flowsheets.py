"""Build a sequential difflow flowsheet from a network decomposition.

:func:`build_network_flowsheet` mechanically translates a
:class:`difflow_gas.network.Decomposition` into a
:class:`difflow_gas.flowsheet.GasNetworkFlowsheet`:

* one :class:`~difflow_gas.units.AffineFlow` unit per non-root tree
  node (leaf-to-root mass balances producing every tree-arc flow),
* one :class:`~difflow_gas.units.SourceHead` at the slack node,
* one pressure-propagation unit per tree arc, root-to-leaf
  (:class:`~difflow_gas.units.PipePressure` for pipes and resistors,
  :class:`~difflow_gas.units.CompressorBoost` for compressor stations,
  :class:`~difflow_gas.units.PressureEqual` for valves and short
  pipes, :class:`~difflow_gas.units.ControlValveDrop` for control
  valves),
* one :class:`~difflow_gas.units.PressureDrivenPipe` per chord, whose
  output recycles into the corresponding tear stream.

Stream naming (see also :func:`node_pressure_stream` and
:func:`arc_flow_stream`):

=====================  ===========================================
stream                 meaning
=====================  ===========================================
``node_<node_id>``     node state: pressure of the node, flow of
                       its parent tree arc
``q_<arc_id>``         tree-arc flow (signed, arc direction)
``q_<arc_id>_calc``    chord flow computed by the chord unit
``tear_<arc_id>``      chord tear stream (recycle destination)
``root_feed``          feed carrying the root nomination
=====================  ===========================================

Unit naming: ``bal_<node>``, ``src_<root>``, ``press_<node>`` for
non-parametric tree arcs, ``cs_<arc_id>`` for compressor stations and
``cv_<arc_id>`` for control valves (so their decision parameters are
addressable as ``"cs_<id>.ratio"`` / ``"cv_<id>.dp_pa"`` in
``_apply_params`` and ``make_objective_fn``), and ``chord_<arc_id>``.
"""

from __future__ import annotations

from difflow.flowsheet import Unit

from difflow_gas.flowsheet import GasNetworkFlowsheet
from difflow_gas.network import Decomposition, GasNetwork, decompose
from difflow_gas.physics import smoothed_power_w
from difflow_gas.streams import FLOW_KEY, GAS, gas_stream
from difflow_gas.units import (
    AffineFlow,
    CompressorBoost,
    ControlValveDrop,
    PipePressure,
    PressureDrivenPipe,
    PressureEqual,
    SourceHead,
)


def cs_unit_name(arc_id: str) -> str:
    """Flowsheet unit (and param prefix) of a compressor station."""
    return f"cs_{arc_id}"


def cv_unit_name(arc_id: str) -> str:
    """Flowsheet unit (and param prefix) of a control valve."""
    return f"cv_{arc_id}"


def src_unit_name(root: str) -> str:
    """Flowsheet unit (and param prefix) of the slack-pressure source."""
    return f"src_{root}"


def node_pressure_stream(node_id: str) -> str:
    """Stream whose pressure is the given node's pressure."""
    return f"node_{node_id}"


def arc_flow_stream(dec: Decomposition, arc_id: str) -> str:
    """Stream carrying an arc's signed flow (kg/s, from->to positive)."""
    if arc_id in dec.chord_ids:
        return f"q_{arc_id}_calc"
    return f"q_{arc_id}"


def build_network_flowsheet(
    network: GasNetwork,
    root: str,
    p_slack_pa: float,
    ratios: dict[str, float] | None = None,
    cv_drops_pa: dict[str, float] | None = None,
    dec: Decomposition | None = None,
) -> tuple[GasNetworkFlowsheet, Decomposition]:
    """Assemble the sequential flowsheet for a gas network.

    Args:
        network: the network to solve.
        root: slack node id (its pressure is prescribed).
        p_slack_pa: slack pressure (Pa); parameter of unit
            ``src_<root>`` (key ``"src_<root>.P_set"``).
        ratios: compressor pressure ratios by station arc id
            (default 1.0 everywhere); parameters of units
            ``cs_<id>`` (keys ``"cs_<id>.ratio"``).
        cv_drops_pa: control valve pressure drops (Pa) by arc id
            (default 0.0); parameters of units ``cv_<id>``
            (keys ``"cv_<id>.dp_pa"``).
        dec: reuse an existing decomposition (must match the network
            and root); computed when omitted.

    Returns:
        ``(flowsheet, decomposition)``. The flowsheet has neutral
        zero-flow tear guesses attached as ``flowsheet.tear_guess``.
    """
    if dec is None:
        dec = decompose(network, root)
    elif dec.root != root:
        raise ValueError(
            f"decomposition rooted at {dec.root!r}, expected {root!r}"
        )

    T = network.gas_temp_k
    ratios = {
        cs: (ratios or {}).get(cs, 1.0) for cs in network.compressor_ids()
    }
    cv_drops_pa = {
        cv: (cv_drops_pa or {}).get(cv, 0.0)
        for cv in network.control_valve_ids()
    }

    fs = GasNetworkFlowsheet(
        species_order=[GAS], default_flow=1.0, default_T=T,
        default_P=p_slack_pa,
    )

    fs.add_feed(
        "root_feed",
        gas_stream(network.supply_kg_s.get(root, 0.0), T, p_slack_pa),
    )

    def U(name, op, inlets, outlets):
        fs.add_unit(Unit(name, op, inlets, outlets))

    # --- phase 1: leaf-to-root mass balances (tree-arc flows) ----------
    for bal in dec.balances:
        inlet_names, signs = [], []
        for kind, aid, sign in bal.inlets:
            inlet_names.append(f"q_{aid}" if kind == "tree" else f"tear_{aid}")
            signs.append(sign)
        U(f"bal_{bal.node}",
          AffineFlow(bal.const, tuple(signs), T, p_slack_pa),
          inlet_names, [f"q_{bal.parent_arc}"])

    # --- phase 2: root-to-leaf pressure propagation ---------------------
    U(src_unit_name(root), SourceHead(p_slack_pa),
      ["root_feed"], [node_pressure_stream(root)])
    for v in dec.order[1:]:
        aid = dec.parent_arc[v]
        kind = dec.arcs[aid].kind
        d = dec.traversal_dir[v]
        if kind in ("pipe", "resistor"):
            name, op = f"press_{v}", PipePressure(network.beta[aid], d)
        elif kind == "compressor":
            name, op = cs_unit_name(aid), CompressorBoost(ratios[aid], d)
        elif kind == "control_valve":
            name, op = cv_unit_name(aid), ControlValveDrop(cv_drops_pa[aid], d)
        else:  # valve, short_pipe
            name, op = f"press_{v}", PressureEqual()
        U(name, op,
          [node_pressure_stream(dec.parent[v]), f"q_{aid}"],
          [node_pressure_stream(v)])

    # --- phase 3: chords recompute their flows (tear updates) ----------
    for cid in dec.chord_ids:
        a = dec.arcs[cid]
        U(f"chord_{cid}", PressureDrivenPipe(network.beta[cid]),
          [node_pressure_stream(a.from_node), node_pressure_stream(a.to_node)],
          [f"q_{cid}_calc"])
        fs.add_recycle(f"q_{cid}_calc", f"tear_{cid}")

    # A neutral tear guess: zero flow through every chord at slack head.
    fs.tear_guess = {
        f"tear_{cid}": gas_stream(0.0, T, p_slack_pa)
        for cid in dec.chord_ids
    }

    return fs, dec


def total_compressor_power_w(
    streams, dec: Decomposition, t_in_k: float, **power_kwargs
):
    """Total compressor shaft power (W) from solved streams.

    Uses :func:`difflow_gas.physics.smoothed_power_w` (smoothed |q|)
    with each station's ratio recovered from its end-node pressures and
    its flow from the tree-arc flow stream, so the value matches the
    objective of an equation-oriented power-minimization NLP of the
    same network. Differentiable; usable inside
    :meth:`GasNetworkFlowsheet.make_objective_fn` objectives.
    """
    W = 0.0
    for aid, a in dec.arcs.items():
        if a.kind != "compressor":
            continue
        q = streams[arc_flow_stream(dec, aid)][FLOW_KEY]
        ratio = (
            streams[node_pressure_stream(a.to_node)]["P"]
            / streams[node_pressure_stream(a.from_node)]["P"]
        )
        W = W + smoothed_power_w(q, ratio, t_in_k, **power_kwargs)
    return W
