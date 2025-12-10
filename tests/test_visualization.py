"""Tests for flowsheet visualization module."""

import pytest
import jax.numpy as jnp

from difflow.visualization.graph import FlowsheetGraph, Node, Edge
from difflow.visualization.styles import get_unit_style, UNIT_STYLES, get_stream_color
from difflow import Flowsheet, Unit, make_stream


class TestNode:
    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(id="reactor", name="CSTR-1", unit_type="CSTR")
        assert node.id == "reactor"
        assert node.name == "CSTR-1"
        assert node.unit_type == "CSTR"

    def test_node_with_params(self):
        """Test node with parameters."""
        node = Node(
            id="reactor",
            name="CSTR-1",
            unit_type="CSTR",
            params={"V": 10.0, "T": 350.0},
        )
        assert node.params["V"] == 10.0
        assert node.params["T"] == 350.0

    def test_node_tooltip_html(self):
        """Test tooltip generation."""
        node = Node(
            id="reactor",
            name="CSTR-1",
            unit_type="CSTR",
            params={"V": 10.0},
        )
        html = node.tooltip_html()
        assert "CSTR-1" in html
        assert "CSTR" in html
        assert "V:" in html


class TestEdge:
    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge(id="stream1", source="reactor", target="flash")
        assert edge.id == "stream1"
        assert edge.source == "reactor"
        assert edge.target == "flash"

    def test_edge_with_stream_data(self):
        """Test edge with stream data."""
        stream = {"F_A": 10.0, "F_B": 5.0, "T": 300.0, "P": 101325.0}
        edge = Edge(
            id="stream1",
            source="reactor",
            target="flash",
            stream_data=stream,
        )
        assert edge.stream_data["F_A"] == 10.0
        assert edge.stream_data["T"] == 300.0

    def test_edge_tooltip_html(self):
        """Test edge tooltip generation."""
        stream = {"F_A": 10.0, "F_B": 5.0, "T": 350.0, "P": 101325.0}
        edge = Edge(id="stream1", source="reactor", target="flash", stream_data=stream)
        html = edge.tooltip_html()
        assert "stream1" in html
        assert "reactor" in html
        assert "flash" in html

    def test_edge_total_flow(self):
        """Test total flow calculation."""
        stream = {"F_A": 10.0, "F_B": 5.0, "T": 300.0}
        edge = Edge(id="stream1", source="r", target="f", stream_data=stream)
        assert edge.total_flow() == 15.0

    def test_edge_total_flow_empty(self):
        """Test total flow with no stream data."""
        edge = Edge(id="stream1", source="r", target="f")
        assert edge.total_flow() == 1.0  # Default minimum


class TestFlowsheetGraph:
    def test_graph_creation(self):
        """Test empty graph creation."""
        graph = FlowsheetGraph(name="Test Process")
        assert graph.name == "Test Process"
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes."""
        graph = FlowsheetGraph()
        node = graph.add_node("reactor", "CSTR-1", unit_type="CSTR")
        assert "reactor" in graph.nodes
        assert graph.nodes["reactor"].name == "CSTR-1"

    def test_add_node_default_name(self):
        """Test node with default name."""
        graph = FlowsheetGraph()
        graph.add_node("reactor")
        assert graph.nodes["reactor"].name == "reactor"

    def test_add_edge(self):
        """Test adding edges."""
        graph = FlowsheetGraph()
        graph.add_node("reactor")
        graph.add_node("flash")
        edge = graph.add_edge("reactor", "flash", edge_id="stream1")
        assert "stream1" in graph.edges
        assert edge.source == "reactor"
        assert edge.target == "flash"

    def test_add_edge_auto_id(self):
        """Test auto-generated edge ID."""
        graph = FlowsheetGraph()
        graph.add_node("reactor")
        graph.add_node("flash")
        edge = graph.add_edge("reactor", "flash")
        assert edge.id.startswith("stream_")

    def test_add_edge_invalid_source(self):
        """Test error on invalid source node."""
        graph = FlowsheetGraph()
        graph.add_node("flash")
        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge("reactor", "flash")

    def test_add_edge_invalid_target(self):
        """Test error on invalid target node."""
        graph = FlowsheetGraph()
        graph.add_node("reactor")
        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge("reactor", "flash")

    def test_add_feed(self):
        """Test adding feed stream."""
        graph = FlowsheetGraph()
        graph.add_node("reactor")
        feed_node, feed_edge = graph.add_feed("reactor", "Fresh Feed")
        assert feed_node.unit_type == "feed"
        assert feed_edge.target == "reactor"

    def test_add_product(self):
        """Test adding product stream."""
        graph = FlowsheetGraph()
        graph.add_node("flash")
        prod_node, prod_edge = graph.add_product("flash", "Product")
        assert prod_node.unit_type == "product"
        assert prod_edge.source == "flash"

    def test_get_incoming_edges(self):
        """Test getting incoming edges."""
        graph = FlowsheetGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")

        incoming = graph.get_incoming_edges("c")
        assert len(incoming) == 2

    def test_get_outgoing_edges(self):
        """Test getting outgoing edges."""
        graph = FlowsheetGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        outgoing = graph.get_outgoing_edges("a")
        assert len(outgoing) == 2

    def test_get_predecessors(self):
        """Test getting predecessor nodes."""
        graph = FlowsheetGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")

        preds = graph.get_predecessors("c")
        assert set(preds) == {"a", "b"}

    def test_get_successors(self):
        """Test getting successor nodes."""
        graph = FlowsheetGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        succs = graph.get_successors("a")
        assert set(succs) == {"b", "c"}

    def test_to_networkx(self):
        """Test conversion to NetworkX graph."""
        pytest.importorskip("networkx")

        graph = FlowsheetGraph()
        graph.add_node("a", "Node A", unit_type="CSTR")
        graph.add_node("b", "Node B", unit_type="Flash")
        graph.add_edge("a", "b")

        G = graph.to_networkx()
        assert len(G.nodes()) == 2
        assert len(G.edges()) == 1
        assert G.nodes["a"]["unit_type"] == "CSTR"

    def test_compute_layout_spring(self):
        """Test spring layout computation."""
        pytest.importorskip("networkx")

        graph = FlowsheetGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        positions = graph.compute_layout(algorithm="spring")
        assert len(positions) == 3
        assert all(len(pos) == 2 for pos in positions.values())


class TestStyles:
    def test_get_unit_style_known(self):
        """Test getting style for known unit type."""
        style = get_unit_style("CSTR")
        assert style.color is not None
        assert style.shape is not None

    def test_get_unit_style_unknown(self):
        """Test getting style for unknown unit type."""
        style = get_unit_style("UnknownUnit")
        assert style == UNIT_STYLES["generic"]

    def test_all_unit_types_have_styles(self):
        """Test that common unit types have defined styles."""
        unit_types = [
            "CSTR", "Flash", "Centrifuge", "Ultrafiltration",
            "ProteinAChromatography", "feed", "product",
        ]
        for unit_type in unit_types:
            style = get_unit_style(unit_type)
            assert style is not None

    def test_get_stream_color(self):
        """Test getting stream colors."""
        assert get_stream_color("feed") is not None
        assert get_stream_color("product") is not None
        assert get_stream_color("unknown") == get_stream_color("default")


class TestRender:
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        graph = FlowsheetGraph(name="Test Process")
        graph.add_node("reactor", "CSTR-1", unit_type="CSTR", params={"V": 10.0})
        graph.add_node("flash", "Flash-1", unit_type="Flash")
        graph.add_feed("reactor", "Feed")
        graph.add_edge("reactor", "flash", stream_data={"F_A": 10.0, "T": 350.0})
        graph.add_product("flash", "Product")
        return graph

    def test_render_flowsheet(self, simple_graph):
        """Test rendering a flowsheet."""
        pytest.importorskip("plotly")
        pytest.importorskip("networkx")

        from difflow.visualization.render import render_flowsheet

        fig = render_flowsheet(simple_graph, layout="spring")
        assert fig is not None
        # Check that figure has data
        assert len(fig.data) > 0

    def test_to_html(self, simple_graph):
        """Test exporting to HTML."""
        pytest.importorskip("plotly")
        pytest.importorskip("networkx")

        from difflow.visualization.render import to_html

        html = to_html(simple_graph, layout="spring")
        assert "<html>" in html.lower()
        assert "plotly" in html.lower()

    def test_render_with_title(self, simple_graph):
        """Test rendering with custom title."""
        pytest.importorskip("plotly")
        pytest.importorskip("networkx")

        from difflow.visualization.render import render_flowsheet

        fig = render_flowsheet(simple_graph, title="My Process", layout="spring")
        assert fig.layout.title.text == "My Process"


class TestFlowsheetIntegration:
    """Tests for Flowsheet.to_graph() integration."""

    def test_simple_flowsheet_to_graph(self):
        """Test converting a simple flowsheet to graph."""
        # Create a simple flowsheet
        fs = Flowsheet(species_order=["A", "B"])

        feed = make_stream({"A": 100.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        # Simple pass-through operation for testing
        def passthrough(inlet):
            return inlet

        fs.add_unit(Unit("reactor", passthrough, ["feed"], ["product"]))

        # Convert to graph
        graph = fs.to_graph()

        # Check structure
        assert len(graph.nodes) == 3  # feed node, reactor, product node
        assert len(graph.edges) == 2  # feed->reactor, reactor->product

        # Check unit node exists
        assert "reactor" in graph.nodes

    def test_flowsheet_to_graph_with_streams(self):
        """Test graph with solved stream data."""
        fs = Flowsheet(species_order=["A", "B"])

        feed = make_stream({"A": 100.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        def double_flow(inlet):
            return {k: v * 2 if k.startswith("F_") else v for k, v in inlet.items()}

        fs.add_unit(Unit("reactor", double_flow, ["feed"], ["product"]))

        # Solve and convert to graph
        streams = fs.solve()
        graph = fs.to_graph(solved_streams=streams)

        # Check that stream data is attached to edges
        feed_edge = None
        for edge in graph.edges.values():
            if edge.id == "feed":
                feed_edge = edge
                break

        assert feed_edge is not None
        assert feed_edge.stream_data is not None

    def test_flowsheet_to_graph_multi_unit(self):
        """Test graph with multiple units."""
        fs = Flowsheet(species_order=["A", "B"])

        feed = make_stream({"A": 100.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        def passthrough(inlet):
            return inlet

        fs.add_unit(Unit("reactor", passthrough, ["feed"], ["reactor_out"]))
        fs.add_unit(Unit("separator", passthrough, ["reactor_out"], ["product"]))

        graph = fs.to_graph()

        # Check nodes
        assert "reactor" in graph.nodes
        assert "separator" in graph.nodes

        # Check edges
        edge_ids = {e.id for e in graph.edges.values()}
        assert "feed" in edge_ids
        assert "reactor_out" in edge_ids

    def test_flowsheet_visualize_method(self):
        """Test the visualize() convenience method."""
        pytest.importorskip("plotly")
        pytest.importorskip("networkx")

        fs = Flowsheet(species_order=["A", "B"])

        feed = make_stream({"A": 100.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        def passthrough(inlet):
            return inlet

        fs.add_unit(Unit("reactor", passthrough, ["feed"], ["product"]))

        streams = fs.solve()
        fig = fs.visualize(streams, layout="spring")

        assert fig is not None
        assert len(fig.data) > 0

    def test_flowsheet_to_graph_detects_unit_type(self):
        """Test that unit type is correctly detected from operation class."""
        from difflow import CSTR, CSTRParams

        fs = Flowsheet(species_order=["A", "B"])

        feed = make_stream({"A": 100.0, "B": 0.0}, T=300.0, P=101325.0)
        fs.add_feed("feed", feed)

        # Simple first-order rate law: r = k * C_A
        def rate_fn(C, T, params):
            return params["k"] * C["A"]

        params = CSTRParams(
            V=jnp.array(10.0),
            rate_fn=rate_fn,
            stoich={"A": jnp.array([-1.0]), "B": jnp.array([1.0])},
            rate_params={"k": jnp.array(0.1)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params)
        fs.add_unit(Unit("reactor", cstr, ["feed"], ["product"]))

        graph = fs.to_graph()

        # Check that unit type was detected
        reactor_node = graph.nodes["reactor"]
        assert reactor_node.unit_type == "CSTR"
