"""Graph data structures for flowsheet visualization."""

from dataclasses import dataclass, field
from typing import Any, Optional
import jax.numpy as jnp


@dataclass
class Node:
    """A node representing a unit operation in the flowsheet.

    Attributes:
        id: Unique identifier for the node
        name: Display name for the node
        unit_type: Type of unit operation (e.g., 'CSTR', 'Flash', 'Centrifuge')
        params: Unit operation parameters for tooltip display
        position: Optional (x, y) position for layout
        metadata: Additional metadata for visualization
    """
    id: str
    name: str
    unit_type: str = "generic"
    params: dict = field(default_factory=dict)
    position: Optional[tuple[float, float]] = None
    metadata: dict = field(default_factory=dict)

    def tooltip_html(self) -> str:
        """Generate HTML tooltip content for this node."""
        lines = [f"<b>{self.name}</b>", f"Type: {self.unit_type}"]

        if self.params:
            lines.append("<br><b>Parameters:</b>")
            for key, value in self.params.items():
                if hasattr(value, '__float__'):
                    lines.append(f"  {key}: {float(value):.4g}")
                else:
                    lines.append(f"  {key}: {value}")

        return "<br>".join(lines)


@dataclass
class Edge:
    """A directed edge representing a stream between units.

    Attributes:
        id: Unique identifier for the edge
        source: Source node ID
        target: Target node ID
        source_port: Output port name on source node
        target_port: Input port name on target node
        stream_data: Stream data (flows, T, P, species)
        metadata: Additional metadata for visualization
    """
    id: str
    source: str
    target: str
    source_port: str = "out"
    target_port: str = "in"
    stream_data: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def tooltip_html(self) -> str:
        """Generate HTML tooltip content for this edge."""
        lines = [f"<b>Stream: {self.id}</b>"]
        lines.append(f"{self.source}:{self.source_port} → {self.target}:{self.target_port}")

        if self.stream_data:
            # Temperature and pressure
            if "T" in self.stream_data:
                T = self.stream_data["T"]
                if hasattr(T, '__float__'):
                    lines.append(f"T: {float(T):.1f} K")
            if "P" in self.stream_data:
                P = self.stream_data["P"]
                if hasattr(P, '__float__'):
                    lines.append(f"P: {float(P)/1000:.1f} kPa")

            # Flow rates
            lines.append("<br><b>Flows:</b>")
            for key, value in self.stream_data.items():
                if key.startswith("F_"):
                    species = key[2:]
                    if hasattr(value, '__float__'):
                        lines.append(f"  {species}: {float(value):.4g}")

        return "<br>".join(lines)

    def total_flow(self) -> float:
        """Calculate total mass flow for edge width scaling."""
        if not self.stream_data:
            return 1.0

        total = 0.0
        for key, value in self.stream_data.items():
            if key.startswith("F_"):
                if hasattr(value, '__float__'):
                    total += float(value)

        return max(total, 0.1)  # Minimum for visibility


class FlowsheetGraph:
    """Graph representation of a process flowsheet.

    Provides methods for building, modifying, and querying the flowsheet
    topology for visualization purposes.

    Example:
        >>> graph = FlowsheetGraph()
        >>> graph.add_node("reactor", "CSTR-1", unit_type="CSTR", params={"V": 10.0})
        >>> graph.add_node("flash", "Flash-1", unit_type="Flash")
        >>> graph.add_edge("feed", "reactor", "flash", stream_data=outlet_stream)
        >>> render_flowsheet(graph)
    """

    def __init__(self, name: str = "Flowsheet"):
        """Initialize an empty flowsheet graph.

        Args:
            name: Name of the flowsheet for display
        """
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._edge_counter = 0

    def add_node(
        self,
        node_id: str,
        name: Optional[str] = None,
        unit_type: str = "generic",
        params: Optional[dict] = None,
        position: Optional[tuple[float, float]] = None,
        **metadata
    ) -> Node:
        """Add a unit operation node to the graph.

        Args:
            node_id: Unique identifier for the node
            name: Display name (defaults to node_id)
            unit_type: Type of unit operation
            params: Unit parameters for tooltip
            position: Optional fixed position (x, y)
            **metadata: Additional metadata

        Returns:
            The created Node object
        """
        if name is None:
            name = node_id

        node = Node(
            id=node_id,
            name=name,
            unit_type=unit_type,
            params=params or {},
            position=position,
            metadata=metadata,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_id: Optional[str] = None,
        source_port: str = "out",
        target_port: str = "in",
        stream_data: Optional[dict] = None,
        **metadata
    ) -> Edge:
        """Add a stream edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            edge_id: Unique edge ID (auto-generated if None)
            source_port: Output port name on source
            target_port: Input port name on target
            stream_data: Stream dictionary with flows, T, P
            **metadata: Additional metadata

        Returns:
            The created Edge object

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found in graph")

        if edge_id is None:
            edge_id = f"stream_{self._edge_counter}"
            self._edge_counter += 1

        edge = Edge(
            id=edge_id,
            source=source,
            target=target,
            source_port=source_port,
            target_port=target_port,
            stream_data=stream_data,
            metadata=metadata,
        )
        self.edges[edge_id] = edge
        return edge

    def add_feed(
        self,
        target: str,
        feed_name: str = "Feed",
        stream_data: Optional[dict] = None,
        target_port: str = "in",
    ) -> tuple[Node, Edge]:
        """Add a feed stream to the flowsheet.

        Creates a virtual feed node and connects it to the target.

        Args:
            target: Target node ID
            feed_name: Display name for the feed
            stream_data: Feed stream data
            target_port: Input port on target

        Returns:
            Tuple of (feed_node, feed_edge)
        """
        feed_id = f"feed_{target}"
        node = self.add_node(feed_id, feed_name, unit_type="feed")
        edge = self.add_edge(
            feed_id, target,
            source_port="out",
            target_port=target_port,
            stream_data=stream_data,
        )
        return node, edge

    def add_product(
        self,
        source: str,
        product_name: str = "Product",
        stream_data: Optional[dict] = None,
        source_port: str = "out",
    ) -> tuple[Node, Edge]:
        """Add a product stream from the flowsheet.

        Creates a virtual product node and connects it from the source.

        Args:
            source: Source node ID
            product_name: Display name for the product
            stream_data: Product stream data
            source_port: Output port on source

        Returns:
            Tuple of (product_node, product_edge)
        """
        product_id = f"product_{source}_{source_port}"
        node = self.add_node(product_id, product_name, unit_type="product")
        edge = self.add_edge(
            source, product_id,
            source_port=source_port,
            target_port="in",
            stream_data=stream_data,
        )
        return node, edge

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        """Get all edges incoming to a node."""
        return [e for e in self.edges.values() if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        """Get all edges outgoing from a node."""
        return [e for e in self.edges.values() if e.source == node_id]

    def get_predecessors(self, node_id: str) -> list[str]:
        """Get IDs of all nodes that feed into this node."""
        return [e.source for e in self.get_incoming_edges(node_id)]

    def get_successors(self, node_id: str) -> list[str]:
        """Get IDs of all nodes that this node feeds into."""
        return [e.target for e in self.get_outgoing_edges(node_id)]

    def to_networkx(self):
        """Convert to NetworkX DiGraph for layout algorithms.

        Returns:
            networkx.DiGraph with node and edge attributes
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for graph operations. "
                "Install with: pip install networkx"
            )

        G = nx.DiGraph()

        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                name=node.name,
                unit_type=node.unit_type,
                params=node.params,
                position=node.position,
            )

        for edge_id, edge in self.edges.items():
            G.add_edge(
                edge.source,
                edge.target,
                id=edge_id,
                source_port=edge.source_port,
                target_port=edge.target_port,
                stream_data=edge.stream_data,
            )

        return G

    def compute_layout(
        self,
        algorithm: str = "dot",
        **kwargs
    ) -> dict[str, tuple[float, float]]:
        """Compute node positions using a layout algorithm.

        Args:
            algorithm: Layout algorithm to use:
                - 'dot': Hierarchical left-to-right (best for flowsheets)
                - 'neato': Spring model
                - 'fdp': Force-directed
                - 'sfdp': Scalable force-directed
                - 'circo': Circular
                - 'twopi': Radial
                - 'spring': NetworkX spring layout (fallback)
            **kwargs: Additional arguments for the layout algorithm

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        import networkx as nx

        G = self.to_networkx()

        # Try graphviz layouts first (best quality)
        graphviz_layouts = {'dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi'}

        if algorithm in graphviz_layouts:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                positions = graphviz_layout(G, prog=algorithm, **kwargs)
            except ImportError:
                try:
                    from networkx.drawing.nx_pydot import graphviz_layout
                    positions = graphviz_layout(G, prog=algorithm, **kwargs)
                except ImportError:
                    # Fall back to spring layout
                    print(f"Warning: graphviz not available, using spring layout")
                    positions = nx.spring_layout(G, **kwargs)
        elif algorithm == 'spring':
            positions = nx.spring_layout(G, **kwargs)
        elif algorithm == 'kamada_kawai':
            positions = nx.kamada_kawai_layout(G, **kwargs)
        elif algorithm == 'spectral':
            positions = nx.spectral_layout(G, **kwargs)
        else:
            raise ValueError(f"Unknown layout algorithm: {algorithm}")

        # Update node positions
        for node_id, pos in positions.items():
            if node_id in self.nodes:
                self.nodes[node_id].position = (float(pos[0]), float(pos[1]))

        return positions

    def __repr__(self) -> str:
        return f"FlowsheetGraph(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"
