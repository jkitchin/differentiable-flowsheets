"""Gas network data model and topology-driven sequential decomposition.

The central objects are :class:`GasNetwork` (a plain, parser-agnostic
description of a network: arcs, resistance coefficients, fixed
boundary flows) and :func:`decompose`, which turns a network into a
:class:`Decomposition`, the complete schedule for a sequential-modular
solve:

1. **Spanning tree.** Arcs whose pressure relation is not an
   invertible flow law (compressor stations, valves, control valves,
   short pipes) are forced into the tree; among the resistance arcs
   (pipes, resistors), a minimum spanning tree on the resistance
   coefficient keeps the LEAST resistive arcs, pushing the most
   resistive arc of each independent loop out as the chord. The chord
   choice matters for convergence: the tear-map slope of a chord is
   roughly ``-sum(beta_e |q_e|, loop tree arcs) / (beta_c |q_c|)``, so
   making the chord the most resistive arc of its loop keeps the
   spectral radius small.

2. **Tears.** With every boundary flow fixed, the chord flows are the
   only flow unknowns; there are exactly ``cycle rank = arcs - nodes
   + 1`` of them.

3. **Flows.** Given the tears, every tree-arc flow follows from
   leaf-to-root mass balances (affine in the tears, signs +-1); the
   :class:`BalanceSpec` list records that schedule.

4. **Pressures.** Node pressures propagate root-to-leaf along the
   tree from the slack node: squared-pressure drop across pipes and
   resistors, ratio across compressors, equality across valves and
   short pipes, parametric drop across control valves.

5. **Tear update.** Each chord recomputes its flow from its end
   squared pressures (pressure-driven Weymouth); iterating 3-5 is a
   fixed-point map on the tears.

The fixed-point map typically has real negative eigenvalues (loop
flows overshoot and oscillate), so it needs damping or acceleration;
see :meth:`difflow_gas.flowsheet.GasNetworkFlowsheet.solve_differentiable`
for guidance.

The module is deliberately independent of any file format: build a
:class:`GasNetwork` from GasLib XML, GeoJSON, a database, or by hand,
and everything downstream is identical.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

#: arc kinds whose flow follows invertibly from the end pressures;
#: these are the only arcs allowed to close loops (become chords)
RESISTANCE_KINDS = frozenset({"pipe", "resistor"})

#: arc kinds whose pressure relation is a parameter or an equality;
#: these are forced into the spanning tree
FORCED_TREE_KINDS = frozenset(
    {"compressor", "valve", "control_valve", "short_pipe"}
)

ARC_KINDS = RESISTANCE_KINDS | FORCED_TREE_KINDS


class Arc(NamedTuple):
    """A directed network arc (the direction fixes the flow sign)."""

    from_node: str
    to_node: str
    kind: str  # one of ARC_KINDS


@dataclass
class CompressorLimits:
    """Operating limits of a compressor station (all optional)."""

    pressure_in_min_bar: float = 0.0
    pressure_out_max_bar: float = float("inf")
    ratio_max: float = 2.0


@dataclass
class GasNetwork:
    """A steady-state gas network with fixed nominations.

    Attributes:
        arcs: arc id -> :class:`Arc`. Every arc kind in
            :data:`ARC_KINDS` is supported; parallel arcs between the
            same node pair are not (yet).
        beta: arc id -> squared-pressure resistance coefficient in
            Pa^2/(kg/s)^2, required for every pipe and resistor
            (:func:`difflow_gas.physics.weymouth_beta` /
            :func:`difflow_gas.physics.resistor_xi` compute them from
            geometry). Ignored for other kinds.
        supply_kg_s: node id -> fixed net boundary mass flow, positive
            into the network (entries), negative out (exits). Nodes
            without an entry are transit nodes. The supplies must
            balance to zero for a steady state to exist.
        gas_temp_k: isothermal gas temperature (K), used for stream
            temperatures and compressor power.
        pressure_bounds_bar: optional node id -> (pmin, pmax) in bar,
            used by the verification helpers.
        compressor_limits: optional compressor arc id ->
            :class:`CompressorLimits`.
    """

    arcs: dict[str, Arc]
    beta: dict[str, float]
    supply_kg_s: dict[str, float]
    gas_temp_k: float = 283.15
    pressure_bounds_bar: dict[str, tuple[float, float]] | None = None
    compressor_limits: dict[str, CompressorLimits] | None = None
    #: tolerance (kg/s) on the total supply balance check
    balance_tol_kg_s: float = 1e-6

    def __post_init__(self):
        self.arcs = {
            aid: (a if isinstance(a, Arc) else Arc(*a))
            for aid, a in self.arcs.items()
        }
        seen_pairs = set()
        for aid, a in self.arcs.items():
            if a.kind not in ARC_KINDS:
                raise ValueError(
                    f"arc {aid!r}: unknown kind {a.kind!r} "
                    f"(expected one of {sorted(ARC_KINDS)})"
                )
            if a.from_node == a.to_node:
                raise ValueError(f"arc {aid!r} is a self-loop")
            pair = frozenset((a.from_node, a.to_node))
            if pair in seen_pairs:
                raise NotImplementedError(
                    f"parallel arcs between {sorted(pair)} are not "
                    "supported yet (arc {aid!r})"
                )
            seen_pairs.add(pair)
            if a.kind in RESISTANCE_KINDS:
                if aid not in self.beta:
                    raise ValueError(
                        f"{a.kind} {aid!r} has no resistance coefficient "
                        "in network.beta"
                    )
                if self.beta[aid] <= 0.0:
                    raise ValueError(
                        f"{a.kind} {aid!r}: beta must be positive, got "
                        f"{self.beta[aid]}"
                    )
        unknown = set(self.supply_kg_s) - set(self.nodes)
        if unknown:
            raise ValueError(
                f"supply specified for nodes not in any arc: {sorted(unknown)}"
            )
        imbalance = sum(self.supply_kg_s.values())
        if abs(imbalance) > self.balance_tol_kg_s:
            raise ValueError(
                "boundary flows do not balance: total supply - demand = "
                f"{imbalance:.6g} kg/s (tolerance {self.balance_tol_kg_s})"
            )

    @property
    def nodes(self) -> list[str]:
        """Sorted list of node ids appearing in the arcs."""
        ns = set()
        for a in self.arcs.values():
            ns.add(a.from_node)
            ns.add(a.to_node)
        return sorted(ns)

    @property
    def cycle_rank(self) -> int:
        """Number of independent loops (= number of tears needed)."""
        return len(self.arcs) - len(self.nodes) + 1

    def compressor_ids(self) -> list[str]:
        return sorted(
            aid for aid, a in self.arcs.items() if a.kind == "compressor"
        )

    def control_valve_ids(self) -> list[str]:
        return sorted(
            aid for aid, a in self.arcs.items() if a.kind == "control_valve"
        )


@dataclass
class BalanceSpec:
    """Parent-arc flow of one tree node from its local mass balance.

    ``q[parent_arc]`` (in the arc's own from->to direction) equals
    ``const + sum(sign_i * q_i)`` over the inlet terms, where each
    inlet is ``("tree", arc_id, sign)`` for a child tree arc or
    ``("chord", arc_id, sign)`` for an incident chord tear.
    """

    node: str
    parent_arc: str
    const: float
    inlets: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass
class Decomposition:
    """Spanning-tree schedule for a sequential gas network solve."""

    root: str
    arcs: dict[str, Arc]                   # id -> Arc (from, to, kind)
    order: list[str]                       # BFS node order, root first
    parent: dict[str, str | None]
    parent_arc: dict[str, str]             # node -> tree arc to its parent
    tree_arc_ids: list[str]
    chord_ids: list[str]                   # tear arcs; always resistance arcs
    balances: list[BalanceSpec]            # in leaf-to-root solve order
    #: +1 if the node is its parent arc's "to" end (tree traversed with
    #: the arc), -1 if traversed against it
    traversal_dir: dict[str, int] = field(default_factory=dict)

    def arc_child(self, arc_id: str) -> str:
        """The BFS-child endpoint of a tree arc (e.g. a compressor)."""
        for v, a in self.parent_arc.items():
            if a == arc_id:
                return v
        raise KeyError(arc_id)


class _UnionFind:
    """Minimal union-find for Kruskal's algorithm."""

    def __init__(self, items):
        self._parent = {i: i for i in items}

    def find(self, i):
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:  # path compression
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, i, j) -> bool:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return False
        self._parent[rj] = ri
        return True


def spanning_tree(network: GasNetwork) -> tuple[list[str], list[str]]:
    """Split the arcs into (tree arc ids, chord ids).

    Kruskal's minimum spanning tree with forced-tree arc kinds given
    weight -1 (always preferred) and resistance arcs weighted by their
    coefficient, so the most resistive arc of each loop becomes the
    chord. Ties break on the arc id, which makes the result
    deterministic.

    Raises:
        ValueError: if the network is disconnected, or if some
            independent loop contains no resistance arc (such a loop
            cannot be torn by a pressure-driven flow update).
    """
    edges = []
    for aid, a in sorted(network.arcs.items()):
        w = -1.0 if a.kind in FORCED_TREE_KINDS else network.beta[aid]
        edges.append((w, aid, a.from_node, a.to_node))
    edges.sort(key=lambda e: (e[0], e[1]))

    uf = _UnionFind(network.nodes)
    tree: list[str] = []
    for w, aid, u, v in edges:
        if uf.union(u, v):
            tree.append(aid)

    if len(tree) != len(network.nodes) - 1:
        raise ValueError("network graph is not connected")

    chords = sorted(set(network.arcs) - set(tree))
    bad = [c for c in chords if network.arcs[c].kind not in RESISTANCE_KINDS]
    if bad:
        raise ValueError(
            "every independent loop must contain at least one pipe or "
            f"resistor to serve as its tear; loop-closing arcs {bad} are "
            "not resistance arcs"
        )
    return sorted(tree), chords


def decompose(network: GasNetwork, root: str) -> Decomposition:
    """Compute the sequential-modular schedule for a network.

    Args:
        network: the network to decompose.
        root: slack node id, the pressure reference of the solve. Pick
            a node whose pressure you want to prescribe; a source not
            hidden behind a compressor is the natural choice.

    Returns:
        The :class:`Decomposition` consumed by
        :func:`difflow_gas.flowsheets.build_network_flowsheet` (and by
        any other schedule executor, e.g. a Pyomo
        SequentialDecomposition builder).
    """
    if root not in set(network.nodes):
        raise ValueError(f"root {root!r} is not a node of the network")

    tree_arc_ids, chord_ids = spanning_tree(network)

    # --- root the tree (deterministic BFS, sorted neighbors) -----------
    adj: dict[str, list[tuple[str, str]]] = {n: [] for n in network.nodes}
    for aid in tree_arc_ids:
        a = network.arcs[aid]
        adj[a.from_node].append((a.to_node, aid))
        adj[a.to_node].append((a.from_node, aid))
    for n in adj:
        adj[n].sort()

    parent: dict[str, str | None] = {root: None}
    parent_arc: dict[str, str] = {}
    order = [root]
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v, aid in adj[u]:
            if v not in parent:
                parent[v] = u
                parent_arc[v] = aid
                order.append(v)
                queue.append(v)

    traversal_dir = {
        v: (+1 if network.arcs[parent_arc[v]].to_node == v else -1)
        for v in order[1:]
    }

    # --- children and incident chords, for the balance specs -----------
    children: dict[str, list[str]] = {n: [] for n in order}
    for v in order[1:]:
        children[parent[v]].append(v)

    chords_at: dict[str, list[str]] = {n: [] for n in order}
    for cid in chord_ids:
        a = network.arcs[cid]
        chords_at[a.from_node].append(cid)
        chords_at[a.to_node].append(cid)

    balances = []
    for v in reversed(order[1:]):  # leaf-to-root
        d = traversal_dir[v]
        # flow arriving at v through its parent arc (traversal sense) =
        #   -supply(v) + flows toward children + net chord outflow
        inlets: list[tuple[str, str, float]] = []
        for c in children[v]:
            a = parent_arc[c]
            inlets.append(("tree", a, float(d * traversal_dir[c])))
        for cid in chords_at[v]:
            a = network.arcs[cid]
            sign = +1.0 if a.from_node == v else -1.0  # flow leaving v
            inlets.append(("chord", cid, float(d * sign)))
        const = d * (-network.supply_kg_s.get(v, 0.0))
        balances.append(
            BalanceSpec(node=v, parent_arc=parent_arc[v],
                        const=float(const), inlets=inlets)
        )

    return Decomposition(
        root=root,
        arcs=dict(network.arcs),
        order=order,
        parent=parent,
        parent_arc=parent_arc,
        tree_arc_ids=tree_arc_ids,
        chord_ids=chord_ids,
        balances=balances,
        traversal_dir=traversal_dir,
    )
