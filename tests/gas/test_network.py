"""Tests for the GasNetwork model and the sequential decomposition."""

import pytest

from difflow_gas.network import (
    Arc,
    GasNetwork,
    decompose,
    spanning_tree,
)


def triangle() -> GasNetwork:
    """One loop, three pipes; p02 is the most resistive."""
    return GasNetwork(
        arcs={
            "p01": ("n0", "n1", "pipe"),
            "p12": ("n1", "n2", "pipe"),
            "p02": ("n0", "n2", "pipe"),
        },
        beta={"p01": 1e8, "p12": 2e8, "p02": 4e8},
        supply_kg_s={"n0": 30.0, "n1": -10.0, "n2": -20.0},
    )


def mixed_network() -> GasNetwork:
    """Every supported arc kind; one loop (a-b-c-a) with a resistor.

    s1 --pipe_1--> a --valve_1--> b --pipe_2--> c --cv_1--> d --short_1--> e
    s2 --cs_1----> a --res_1--------------------^
    """
    return GasNetwork(
        arcs={
            "pipe_1": ("s1", "a", "pipe"),
            "cs_1": ("s2", "a", "compressor"),
            "valve_1": ("a", "b", "valve"),
            "pipe_2": ("b", "c", "pipe"),
            "res_1": ("a", "c", "resistor"),
            "cv_1": ("c", "d", "control_valve"),
            "short_1": ("d", "e", "short_pipe"),
        },
        beta={"pipe_1": 1e8, "pipe_2": 2e8, "res_1": 5e8},
        supply_kg_s={"s1": 20.0, "s2": 10.0, "b": -5.0, "c": -10.0,
                     "e": -15.0},
    )


# ---------------------------------------------------------------------------
# GasNetwork validation
# ---------------------------------------------------------------------------


def test_network_basic_properties():
    net = triangle()
    assert net.nodes == ["n0", "n1", "n2"]
    assert net.cycle_rank == 1
    assert isinstance(net.arcs["p01"], Arc)  # tuples normalized to Arc


def test_mixed_network_helpers():
    net = mixed_network()
    assert net.cycle_rank == 1
    assert net.compressor_ids() == ["cs_1"]
    assert net.control_valve_ids() == ["cv_1"]


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        GasNetwork(arcs={"x": ("a", "b", "widget")}, beta={},
                   supply_kg_s={})


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        GasNetwork(arcs={"x": ("a", "a", "pipe")}, beta={"x": 1e8},
                   supply_kg_s={})


def test_parallel_arcs_rejected():
    with pytest.raises(NotImplementedError, match="parallel"):
        GasNetwork(
            arcs={"x": ("a", "b", "pipe"), "y": ("b", "a", "pipe")},
            beta={"x": 1e8, "y": 1e8},
            supply_kg_s={"a": 0.0, "b": 0.0},
        )


def test_missing_beta_rejected():
    with pytest.raises(ValueError, match="no resistance coefficient"):
        GasNetwork(arcs={"x": ("a", "b", "pipe")}, beta={},
                   supply_kg_s={"a": 1.0, "b": -1.0})


def test_nonpositive_beta_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        GasNetwork(arcs={"x": ("a", "b", "pipe")}, beta={"x": 0.0},
                   supply_kg_s={"a": 1.0, "b": -1.0})


def test_unbalanced_supply_rejected():
    with pytest.raises(ValueError, match="do not balance"):
        GasNetwork(arcs={"x": ("a", "b", "pipe")}, beta={"x": 1e8},
                   supply_kg_s={"a": 2.0, "b": -1.0})


def test_supply_on_unknown_node_rejected():
    with pytest.raises(ValueError, match="not in any arc"):
        GasNetwork(arcs={"x": ("a", "b", "pipe")}, beta={"x": 1e8},
                   supply_kg_s={"z": 0.0})


# ---------------------------------------------------------------------------
# spanning tree / chord selection
# ---------------------------------------------------------------------------


def test_chord_is_most_resistive_pipe_of_the_loop():
    tree, chords = spanning_tree(triangle())
    assert chords == ["p02"]
    assert tree == ["p01", "p12"]


def test_forced_tree_kinds_never_become_chords():
    tree, chords = spanning_tree(mixed_network())
    # loop a-valve-b-pipe_2-c-res_1-a: the valve must stay in-tree, so
    # the chord is the most resistive resistance arc of the loop
    assert chords == ["res_1"]
    for aid in ("cs_1", "valve_1", "cv_1", "short_1"):
        assert aid in tree


def test_disconnected_network_rejected():
    net = GasNetwork(
        arcs={"x": ("a", "b", "pipe"), "y": ("c", "d", "pipe")},
        beta={"x": 1e8, "y": 1e8},
        supply_kg_s={"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0},
    )
    with pytest.raises(ValueError, match="not connected"):
        spanning_tree(net)


def test_loop_without_resistance_arc_rejected():
    net = GasNetwork(
        arcs={
            "c1": ("a", "b", "compressor"),
            "c2": ("b", "c", "compressor"),
            "c3": ("c", "a", "compressor"),
        },
        beta={},
        supply_kg_s={},
    )
    with pytest.raises(ValueError, match="pipe or resistor"):
        spanning_tree(net)


# ---------------------------------------------------------------------------
# decomposition schedule
# ---------------------------------------------------------------------------


def test_decompose_structure():
    net = triangle()
    dec = decompose(net, root="n0")
    assert dec.root == "n0"
    assert dec.order[0] == "n0"
    assert sorted(dec.order) == net.nodes
    assert len(dec.balances) == len(net.nodes) - 1
    assert dec.chord_ids == ["p02"]
    # traversal directions match arc orientations
    for v in dec.order[1:]:
        a = dec.arcs[dec.parent_arc[v]]
        assert dec.traversal_dir[v] == (+1 if a.to_node == v else -1)


def test_decompose_rejects_unknown_root():
    with pytest.raises(ValueError, match="not a node"):
        decompose(triangle(), root="nope")


def test_decompose_is_deterministic():
    a = decompose(triangle(), root="n0")
    b = decompose(triangle(), root="n0")
    assert a.order == b.order
    assert a.chord_ids == b.chord_ids
    assert [s.node for s in a.balances] == [s.node for s in b.balances]


def _flows_from_balances(dec, chord_flows):
    """Execute the balance schedule for given chord (tear) flows."""
    q = dict(chord_flows)
    for bal in dec.balances:  # leaf-to-root order
        val = bal.const
        for kind, aid, sign in bal.inlets:
            val += sign * q[aid]
        q[bal.parent_arc] = val
    return q


@pytest.mark.parametrize("net_fn,root", [(triangle, "n0"),
                                         (mixed_network, "s1")])
@pytest.mark.parametrize("tear", [-7.3, 0.0, 4.2])
def test_balance_schedule_closes_mass_balance_for_any_tears(
    net_fn, root, tear
):
    """The affine flow schedule satisfies EVERY node balance for ANY
    tear values; the tears only redistribute flow around loops."""
    net = net_fn()
    dec = decompose(net, root=root)
    q = _flows_from_balances(dec, {cid: tear for cid in dec.chord_ids})
    for node in net.nodes:
        acc = net.supply_kg_s.get(node, 0.0)
        for aid, a in net.arcs.items():
            if a.from_node == node:
                acc -= q[aid]
            if a.to_node == node:
                acc += q[aid]
        assert acc == pytest.approx(0.0, abs=1e-9), f"node {node}"


def test_arc_child_lookup():
    dec = decompose(mixed_network(), root="s1")
    child = dec.arc_child("cs_1")
    # cs_1 connects s2 -> a; whichever end is deeper in the BFS tree
    assert child in ("s2", "a")
    assert dec.parent_arc[child] == "cs_1"
    with pytest.raises(KeyError):
        dec.arc_child("res_1")  # chords have no tree child
