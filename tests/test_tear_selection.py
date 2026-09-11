"""Tests for automatic tear-stream selection (#248).

The selection code existed and was exported long before anything called
it, so these tests pin it down twice over: known-answer graphs for the
topology (cycles, scores, calculation order), and a solve that has to
land on the same fixed point as the same flowsheet torn by hand.
"""

import pytest
import jax.numpy as jnp

from difflow import (
    Flowsheet,
    Unit,
    FlowsheetGraph,
    CycleEnumerationWarning,
    analyze_tears,
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


class TestFlowsheetGraph:
    """The digraph a tear analysis reasons over."""

    def test_a_declared_recycle_is_an_edge(self):
        """Without the recycle map the graph of a recycle loop is a DAG.

        A recycle destination is an inlet no unit computes, so the stream
        names alone never close the loop.  This is what made
        ``find_cycles`` report "no cycles" for every flowsheet that had
        one.
        """
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})
        graph = FlowsheetGraph.from_flowsheet(fs)

        assert graph.adjacency["splitter"] == ["mixer"]
        assert find_cycles(graph) == [["mixer", "reactor", "splitter", "mixer"]]

    def test_the_loop_is_found_whatever_order_the_units_were_added(self):
        """The graph is a property of the topology, not of insertion order."""
        forward = FlowsheetGraph.from_flowsheet(
            graph_of(*LOOP, recycles={"rec_src": "recycle"})
        )
        backward = FlowsheetGraph.from_flowsheet(
            graph_of(*reversed(LOOP), recycles={"rec_src": "recycle"})
        )

        assert set(forward.adjacency["mixer"]) == set(backward.adjacency["mixer"])
        assert {(e.stream, e.source, e.dest) for e in forward.edges} == {
            (e.stream, e.source, e.dest) for e in backward.edges
        }

    def test_a_feedback_edge_is_one_the_declared_order_reads_too_early(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})
        graph = FlowsheetGraph.from_flowsheet(fs)

        feedback = {e.stream for e in graph.edges if e.feedback}
        assert feedback == {"recycle"}

    def test_two_streams_between_the_same_units_are_two_edges(self):
        """Tearing one of a parallel pair leaves the other closing the loop."""
        fs = graph_of(
            ("a", ["feed", "back1", "back2"], ["fwd"]),
            ("b", ["fwd"], ["back1", "back2"]),
        )
        graph = FlowsheetGraph.from_flowsheet(fs)

        assert len([e for e in graph.edges if e.source == "b"]) == 2
        # Two units, two loops: a cycle is a sequence of edges, not of
        # unit names.
        assert len(find_cycles(graph)) == 1
        assert calculation_order(graph, ["back1"]) is None
        assert calculation_order(graph, select_tear_streams(graph)) is not None


class TestFindCycles:
    """Known-answer graphs."""

    def test_no_recycle_no_cycles(self):
        fs = graph_of(
            ("reactor", ["feed"], ["rx_out"]),
            ("flash", ["rx_out"], ["vapor", "liquid"]),
        )
        assert find_cycles(FlowsheetGraph.from_flowsheet(fs)) == []
        assert select_tear_streams(fs) == []

    def test_a_unit_recycling_to_itself(self):
        fs = graph_of(("reactor", ["feed", "back"], ["out", "back"]))
        assert find_cycles(FlowsheetGraph.from_flowsheet(fs)) == [
            ["reactor", "reactor"]
        ]

    def test_two_independent_loops(self):
        fs = graph_of(
            ("m1", ["feed", "r1"], ["mid"]),
            ("s1", ["mid"], ["on", "r1"]),
            ("m2", ["on", "r2"], ["mid2"]),
            ("s2", ["mid2"], ["product", "r2"]),
        )
        cycles = find_cycles(FlowsheetGraph.from_flowsheet(fs))

        assert cycles == [["m1", "s1", "m1"], ["m2", "s2", "m2"]]
        assert len(select_tear_streams(fs)) == 2
        assert len(select_tear_streams(fs, method="minimum")) == 2

    def test_nested_loops_over_a_shared_stream(self):
        """Two loops, one stream on both: one tear breaks them both."""
        fs = graph_of(
            ("mixer", ["feed", "short", "long"], ["mixed"]),
            ("reactor", ["mixed"], ["rx_out"]),
            ("splitter", ["rx_out"], ["to_flash", "short"]),
            ("flash", ["to_flash"], ["product", "long"]),
        )
        cycles = find_cycles(FlowsheetGraph.from_flowsheet(fs))

        assert len(cycles) == 2
        assert select_tear_streams(fs) == ["mixed"]
        assert select_tear_streams(fs, method="minimum") == ["mixed"]

    def test_a_cycle_reached_by_a_forward_edge_is_still_found(self):
        """The DFS-with-a-visited-set version missed this one.

        ``a -> c -> a`` is not a back edge of the tree the walk builds
        from ``a -> b -> c -> a``, and a search that only reports back
        edges never sees it.  Both loops are elementary and both need a
        tear.
        """
        fs = graph_of(
            ("a", ["feed", "ca"], ["ab", "ac"]),
            ("b", ["ab"], ["bc"]),
            ("c", ["bc", "ac"], ["ca"]),
        )
        cycles = find_cycles(FlowsheetGraph.from_flowsheet(fs))

        assert sorted(cycles) == [["a", "b", "c", "a"], ["a", "c", "a"]]

    def test_each_cycle_is_reported_once(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})
        cycles = find_cycles(FlowsheetGraph.from_flowsheet(fs))

        assert len(cycles) == len({tuple(c) for c in cycles}) == 1

    def test_enumeration_says_so_when_it_gives_up(self):
        fs = graph_of(
            ("m1", ["feed", "r1"], ["mid"]),
            ("s1", ["mid"], ["on", "r1"]),
            ("m2", ["on", "r2"], ["mid2"]),
            ("s2", ["mid2"], ["product", "r2"]),
        )
        graph = FlowsheetGraph.from_flowsheet(fs)

        with pytest.warns(CycleEnumerationWarning):
            cycles = find_cycles(graph, max_cycles=1)

        assert len(cycles) == 1


class TestSelectTearStreams:
    """What the two strategies choose, and what they must never do."""

    def test_a_declared_recycle_is_never_overruled(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})

        assert select_tear_streams(fs) == ["recycle"]
        assert select_tear_streams(fs, method="minimum") == ["recycle"]

    def test_the_heuristic_tears_after_the_mixing_point(self):
        """The classical choice: downstream of the mixer, where a wrong
        guess is at least right in order of magnitude."""
        fs = graph_of(*LOOP[:2], ("splitter", ["rx_out"], ["product", "recycle"]))

        assert select_tear_streams(fs) == ["mixed"]

    def test_every_loop_ends_up_torn(self):
        fs = graph_of(
            ("mixer", ["feed", "short", "long"], ["mixed"]),
            ("reactor", ["mixed"], ["rx_out"]),
            ("splitter", ["rx_out"], ["to_flash", "short"]),
            ("flash", ["to_flash"], ["to_col", "long"]),
            ("column", ["to_col", "reflux"], ["product", "reflux"]),
        )
        graph = FlowsheetGraph.from_flowsheet(fs)
        cycles = find_cycles(graph)
        assert len(cycles) == 3

        for method in ("heuristic", "minimum"):
            tears = set(select_tear_streams(fs, method=method))
            # A cycle is broken only by tearing a stream on it.
            for cycle in cycles:
                steps = set(zip(cycle, cycle[1:]))
                on_cycle = {
                    e.stream for e in graph.edges if (e.source, e.dest) in steps
                }
                assert on_cycle & tears, (method, cycle)
            # ... and what is torn is enough to sequence the units.
            assert calculation_order(graph, tears) is not None

    def test_a_graph_can_be_passed_instead_of_a_flowsheet(self):
        graph = FlowsheetGraph.from_flowsheet(
            graph_of(*LOOP, recycles={"rec_src": "recycle"})
        )
        assert select_tear_streams(graph) == ["recycle"]

    def test_an_unknown_method_says_so(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})
        with pytest.raises(ValueError, match="Unknown tear selection method"):
            select_tear_streams(fs, method="barkley-motard")


class TestCalculationOrder:
    """Tearing chooses where the loop opens; the order follows from it."""

    def test_an_order_that_already_works_is_kept(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"})
        graph = FlowsheetGraph.from_flowsheet(fs)

        assert calculation_order(graph, ["recycle"]) == [
            "mixer", "reactor", "splitter",
        ]

    def test_tearing_elsewhere_re_sequences_the_units(self):
        fs = graph_of(*LOOP[:2], ("splitter", ["rx_out"], ["product", "recycle"]))
        graph = FlowsheetGraph.from_flowsheet(fs)

        assert calculation_order(graph, ["mixed"]) == [
            "reactor", "splitter", "mixer",
        ]

    def test_an_unbroken_loop_has_no_order(self):
        fs = graph_of(*LOOP[:2], ("splitter", ["rx_out"], ["product", "recycle"]))
        graph = FlowsheetGraph.from_flowsheet(fs)

        assert calculation_order(graph, []) is None


class TestTearAnalysis:
    """The diagnostic: it reads the flowsheet and runs nothing."""

    def test_a_declared_loop_reports_as_torn(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"}, feeds=["feed"])
        analysis = fs.tear_analysis()

        assert analysis.torn
        assert analysis.declared == ["recycle"]
        assert analysis.uncovered == []
        assert analysis.missing_inputs == []
        assert analysis.out_of_order == []
        assert len(analysis.cycles) == 1

    def test_it_names_the_loop_nobody_declared(self):
        fs = graph_of(
            *LOOP[:2],
            ("splitter", ["rx_out"], ["product", "recycle"]),
            feeds=["feed"],
        )
        analysis = fs.tear_analysis()

        assert not analysis.torn
        assert analysis.declared == []
        assert analysis.uncovered == [["mixer", "reactor", "splitter", "mixer"]]
        # ... and what a solve would do about it
        assert analysis.heuristic == ["mixed"]
        assert analysis.out_of_order == ["recycle"]

    def test_it_names_an_inlet_nothing_supplies(self):
        fs = graph_of(
            ("reactor", ["feed", "steam"], ["rx_out"]),
            feeds=["feed"],
        )
        assert fs.tear_analysis().missing_inputs == ["steam"]

    def test_the_summary_reads(self):
        fs = graph_of(*LOOP, recycles={"rec_src": "recycle"}, feeds=["feed"])
        text = str(fs.tear_analysis())

        assert "mixer -> reactor -> splitter -> mixer" in text
        assert "declared tears:  recycle" in text


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


class TestSolveWithAutomaticTears:
    """``solve(tears="auto")`` on a loop nobody declared."""

    @pytest.mark.parametrize("method", ["auto", "heuristic", "minimum"])
    def test_it_finds_the_same_fixed_point_as_tearing_by_hand(self, method):
        auto = _loop_flowsheet(explicit=False).solve(tears=method, tol=1e-10)
        declared = _loop_flowsheet(explicit=True).solve(tol=1e-10)

        for species in ("A", "B"):
            assert get_flows(auto["product"])[species] == pytest.approx(
                float(get_flows(declared["product"])[species]), rel=1e-6
            )

    def test_the_loop_it_tore_is_reported(self):
        fs = _loop_flowsheet(explicit=False)
        fs.solve(tears="auto", tol=1e-10)

        assert fs.last_solve_tear_streams == ["mixed"]
        assert fs.last_solve_converged is True

    def test_the_flowsheet_is_left_as_it_was(self):
        """An automatic choice is for this solve, not for the flowsheet."""
        fs = _loop_flowsheet(explicit=False)
        fs.solve(tears="auto", tol=1e-10)

        assert fs.recycles == {}
        assert [u.name for u in fs.units] == ["mixer", "reactor", "splitter"]

    def test_a_declared_recycle_still_wins(self):
        fs = _loop_flowsheet(explicit=True)
        fs.solve(tears="auto", tol=1e-10)

        assert fs.last_solve_tear_streams == ["recycle"]
        assert fs.recycles == {"rec_src": "recycle"}

    def test_the_default_is_unchanged(self):
        """Without tears="auto" an undeclared loop is still a KeyError.

        Inferring the tear silently would change the answer for every
        flowsheet that already runs.
        """
        fs = _loop_flowsheet(explicit=False)
        with pytest.raises(KeyError):
            fs.solve()

    def test_an_acyclic_flowsheet_still_takes_the_sequential_path(self):
        fs = Flowsheet(["A"])
        fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
        fs.add_unit(Unit("heater", lambda s: s, ["feed"], ["out"]))

        streams = fs.solve(tears="auto")

        assert fs.last_solve_method == "direct"
        assert fs.last_solve_tear_streams == []
        assert "out" in streams

    def test_an_inlet_nothing_supplies_is_a_sentence_not_a_keyerror(self):
        fs = _loop_flowsheet(explicit=False)
        fs.units[1].inlet_names.append("steam")

        with pytest.raises(ValueError, match="steam"):
            fs.solve(tears="auto")

    def test_an_unknown_setting_says_so(self):
        fs = _loop_flowsheet(explicit=True)
        with pytest.raises(ValueError, match="Unknown tears"):
            fs.solve(tears="sometimes")

    def test_gradients_still_come_from_the_converged_solution(self):
        """The auto path is the ordinary solve with different bookkeeping."""
        import jax

        def conversion(feed_A):
            fs = _loop_flowsheet(explicit=False)
            fs.add_feed("feed", make_stream({"A": feed_A, "B": 0.0}, 350.0, 101325.0))
            streams = fs.solve(tears="auto", tol=1e-12, max_iter=200)
            return get_flows(streams["product"])["B"]

        # B out = 0.7 * B produced, linear in the feed, so the gradient is
        # the ratio itself.
        value = conversion(1.0)
        assert jax.grad(conversion)(1.0) == pytest.approx(float(value), rel=1e-4)
