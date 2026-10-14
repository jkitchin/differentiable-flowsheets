"""Tests for automatic tear-stream selection (#248)."""

import pytest
import jax

from difflow import (
    Flowsheet,
    Unit,
    FlowsheetGraph,
    CycleEnumerationWarning,
    calculation_order,
    find_cycles,
    select_tear_streams,
)
from difflow.streams import make_stream, get_flows


def passthrough(*inlets, **kwargs):
    """A unit that is never meant to run: the graph is the point."""
    raise AssertionError("tear analysis must not run the units")


def graph_of(*units, recycles=None, feeds=()):
    """Build a Flowsheet with the given topology and nothing numeric."""
    fs = Flowsheet(["A"])
    for name, inlets, outlets in units:
        fs.add_unit(Unit(name, passthrough, list(inlets), list(outlets)))
    for source, dest in (recycles or {}).items():
        fs.add_recycle(source, dest)
    for feed in feeds:
        fs.add_feed(feed, make_stream({"A": 1.0}, 300.0, 101325.0))
    return fs


LOOP = (
    ("mixer", ["feed", "recycle"], ["mixed"]),
    ("reactor", ["mixed"], ["rx_out"]),
    ("splitter", ["rx_out"], ["product", "rec_src"]),
)


def test_a_declared_recycle_is_an_edge():
    fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})
    graph = FlowsheetGraph.from_flowsheet(fs)
    assert find_cycles(graph) == [["mixer", "reactor", "splitter", "mixer"]]
    assert {e.stream for e in graph.edges if e.feedback} == {"recycle"}
    assert select_tear_streams(fs) == ["recycle"]


def test_a_cycle_reached_by_a_forward_edge_is_still_found():
    """Regression: a DFS reporting only back edges missed ``a -> c -> a``."""
    fs = graph_of(
        ("a", ["feed", "ca"], ["ab", "ac"]),
        ("b", ["ab"], ["bc"]),
        ("c", ["bc", "ac"], ["ca"]),
    )
    cycles = find_cycles(FlowsheetGraph.from_flowsheet(fs))
    assert sorted(cycles) == [["a", "b", "c", "a"], ["a", "c", "a"]]


def test_enumeration_says_so_when_it_gives_up():
    fs = graph_of(
        ("m1", ["feed", "r1"], ["mid"]),
        ("s1", ["mid"], ["on", "r1"]),
        ("m2", ["on", "r2"], ["mid2"]),
        ("s2", ["mid2"], ["product", "r2"]),
    )
    with pytest.warns(CycleEnumerationWarning):
        assert len(find_cycles(FlowsheetGraph.from_flowsheet(fs), max_cycles=1)) == 1


def test_selected_tears_sequence_the_units():
    fs = graph_of(
        ("mixer", ["feed", "short", "long"], ["mixed"]),
        ("reactor", ["mixed"], ["rx_out"]),
        ("splitter", ["rx_out"], ["to_flash", "short"]),
        ("flash", ["to_flash"], ["to_col", "long"]),
        ("column", ["to_col", "reflux"], ["product", "reflux"]),
    )
    graph = FlowsheetGraph.from_flowsheet(fs)
    for method in ("heuristic", "minimum"):
        assert calculation_order(graph, select_tear_streams(fs, method=method)) is not None
    assert calculation_order(graph, []) is None
    with pytest.raises(ValueError, match="Unknown tear selection method"):
        select_tear_streams(fs, method="barkley-motard")


def test_tear_analysis():
    fs = graph_of(*LOOP[:2], ("splitter", ["rx_out"], ["product", "recycle"]),
                  feeds=["feed"])
    analysis = fs.tear_analysis()
    assert not analysis.torn
    assert analysis.uncovered == [["mixer", "reactor", "splitter", "mixer"]]
    assert analysis.heuristic == ["mixed"]
    assert "mixer -> reactor -> splitter -> mixer" in str(analysis)
    fs = graph_of(("reactor", ["feed", "steam"], ["rx_out"]), feeds=["feed"])
    assert fs.tear_analysis().missing_inputs == ["steam"]


def _loop_flowsheet(explicit: bool) -> Flowsheet:
    """mixer -> reactor -> splitter -> mixer, torn by hand or not at all."""

    def mixer(s1, s2):
        f1, f2 = get_flows(s1), get_flows(s2)
        return make_stream({k: f1[k] + f2[k] for k in f1}, s1["T"], s1["P"])

    def reactor(s):
        f = get_flows(s)
        return make_stream(
            {"A": f["A"] * 0.6, "B": f["B"] + 0.4 * f["A"]}, s["T"], s["P"]
        )

    def splitter(s):
        f = get_flows(s)
        return (
            make_stream({k: v * 0.7 for k, v in f.items()}, s["T"], s["P"]),
            make_stream({k: v * 0.3 for k, v in f.items()}, s["T"], s["P"]),
        )

    fs = Flowsheet(["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0))
    fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("reactor", reactor, ["mixed"], ["rx_out"]))
    if explicit:
        fs.add_unit(Unit("splitter", splitter, ["rx_out"], ["product", "rec_src"]))
        fs.add_recycle("rec_src", "recycle")
    else:
        fs.add_unit(Unit("splitter", splitter, ["rx_out"], ["product", "recycle"]))
    return fs


def test_solve_with_automatic_tears():
    fs = _loop_flowsheet(explicit=False)
    with pytest.raises(KeyError):
        fs.solve()
    fs.solve(tears="auto", tol=1e-10)
    assert fs.last_solve_tear_streams == ["mixed"]
    assert fs.last_solve_converged is True
    assert fs.recycles == {}


def test_automatic_tears_bad_inputs():
    with pytest.raises(ValueError, match="Unknown tears"):
        _loop_flowsheet(explicit=True).solve(tears="sometimes")
    fs = _loop_flowsheet(explicit=False)
    fs.units[1].inlet_names.append("steam")
    with pytest.raises(ValueError, match="steam"):
        fs.solve(tears="auto")


def test_grad_through_automatic_tears_is_finite():
    def product_B(feed_A):
        fs = _loop_flowsheet(explicit=False)
        fs.add_feed("feed", make_stream({"A": feed_A, "B": 0.0}, 350.0, 101325.0))
        return get_flows(fs.solve(tears="auto", tol=1e-12, max_iter=200)["product"])["B"]

    assert jax.numpy.isfinite(jax.grad(product_B)(1.0))
