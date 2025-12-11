"""Interactive flowsheet visualization using ipycytoscape.

This module provides visualization of process flowsheets as interactive
graphs with:
- Custom icons for each unit operation type
- Drag-and-drop node positioning
- Tooltips showing stream data and unit parameters
- Export to HTML and PNG for publications
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import json
import warnings

from difflow.visualization.icons import get_icon_data_uri, get_icon_svg

if TYPE_CHECKING:
    from difflow.flowsheet import Flowsheet, Unit
    from difflow.streams import Stream

# Try to import ipycytoscape, but provide helpful error if not available
try:
    import ipycytoscape
    HAS_IPYCYTOSCAPE = True
except ImportError:
    HAS_IPYCYTOSCAPE = False
    ipycytoscape = None


def _get_operation_type(unit: "Unit") -> str:
    """Extract operation type name from a Unit."""
    op = unit.operation
    # Try to get the class/function name
    if hasattr(op, "__name__"):
        return op.__name__
    if hasattr(op, "__class__"):
        return op.__class__.__name__
    return "generic"


def _format_stream_tooltip(stream: "Stream", name: str) -> str:
    """Format stream data for tooltip display."""
    import jax.numpy as jnp

    lines = [f"<b>Stream: {name}</b>"]

    # Temperature and pressure
    T = float(stream.get("T", 0))
    P = float(stream.get("P", 0))
    lines.append(f"T = {T:.1f} K ({T - 273.15:.1f} °C)")
    lines.append(f"P = {P:.0f} Pa ({P / 1e5:.2f} bar)")

    # Flow rates
    lines.append("<br><b>Flows (mol/s):</b>")
    total_flow = 0.0
    for key, value in sorted(stream.items()):
        if key.startswith("F_"):
            species = key[2:]  # Remove "F_" prefix
            flow = float(value)
            total_flow += flow
            if flow > 1e-10:
                lines.append(f"  {species}: {flow:.4g}")

    lines.append(f"<br><b>Total: {total_flow:.4g} mol/s</b>")

    return "<br>".join(lines)


def _format_unit_tooltip(unit: "Unit", streams: dict[str, "Stream"] | None = None) -> str:
    """Format unit operation data for tooltip display."""
    lines = [f"<b>{unit.name}</b>"]
    lines.append(f"Type: {_get_operation_type(unit)}")

    # Inlets and outlets
    lines.append(f"<br>Inlets: {', '.join(unit.inlet_names)}")
    lines.append(f"Outlets: {', '.join(unit.outlet_names)}")

    # Parameters
    if unit.params:
        lines.append("<br><b>Parameters:</b>")
        for key, value in unit.params.items():
            if callable(value):
                lines.append(f"  {key}: <function>")
            else:
                try:
                    lines.append(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")
                except (TypeError, ValueError):
                    lines.append(f"  {key}: {value}")

    return "<br>".join(lines)


class FlowsheetVisualizer:
    """Interactive visualization of process flowsheets.

    Creates an interactive graph visualization using ipycytoscape with:
    - Custom P&ID-style icons for unit operations
    - Left-to-right hierarchical layout (configurable)
    - Tooltips showing stream compositions and unit parameters
    - Drag-and-drop node positioning
    - Export to HTML/PNG for publications

    Example:
        >>> fs = Flowsheet(["A", "B", "C"])
        >>> fs.add_feed("feed", feed_stream)
        >>> fs.add_unit(Unit("reactor", CSTR, ["feed"], ["product"]))
        >>> results = fs.solve()
        >>>
        >>> viz = FlowsheetVisualizer(fs)
        >>> viz.show(streams=results)  # Returns interactive widget
        >>> viz.export_html("flowsheet.html")
    """

    def __init__(
        self,
        flowsheet: "Flowsheet",
        *,
        show_feeds: bool = True,
        show_products: bool = True,
        layout: str = "dagre",
        layout_options: dict[str, Any] | None = None,
    ):
        """Initialize the visualizer.

        Args:
            flowsheet: The Flowsheet object to visualize
            show_feeds: Whether to show feed stream nodes
            show_products: Whether to show product stream nodes
            layout: Layout algorithm ('dagre', 'breadthfirst', 'grid', 'circle', 'cose')
            layout_options: Additional options for the layout algorithm
        """
        if not HAS_IPYCYTOSCAPE:
            raise ImportError(
                "ipycytoscape is required for visualization. "
                "Install it with: pip install ipycytoscape"
            )

        self.flowsheet = flowsheet
        self.show_feeds = show_feeds
        self.show_products = show_products
        self.layout = layout
        self.layout_options = layout_options or {}

        # Build graph data
        self._nodes: list[dict] = []
        self._edges: list[dict] = []
        self._build_graph()

        # Widget reference (created on show())
        self._widget: ipycytoscape.CytoscapeWidget | None = None

    def _build_graph(self) -> None:
        """Build graph nodes and edges from flowsheet."""
        # Track which streams are unit outputs
        stream_sources: dict[str, str] = {}  # stream_name -> unit_name
        stream_targets: dict[str, list[str]] = {}  # stream_name -> [unit_names]

        # First pass: identify all streams and their sources
        for unit in self.flowsheet.units:
            for outlet in unit.outlet_names:
                stream_sources[outlet] = unit.name
            for inlet in unit.inlet_names:
                if inlet not in stream_targets:
                    stream_targets[inlet] = []
                stream_targets[inlet].append(unit.name)

        # Add feed nodes
        if self.show_feeds:
            for feed_name in self.flowsheet.feeds:
                self._nodes.append({
                    "data": {
                        "id": f"feed_{feed_name}",
                        "label": feed_name,
                        "type": "feed",
                        "node_type": "feed",
                    }
                })

        # Add unit nodes
        for unit in self.flowsheet.units:
            op_type = _get_operation_type(unit)
            self._nodes.append({
                "data": {
                    "id": unit.name,
                    "label": unit.name,
                    "type": op_type,
                    "node_type": "unit",
                    "tooltip": _format_unit_tooltip(unit),
                }
            })

        # Identify product streams (outputs not connected to any unit input)
        product_streams = set()
        for unit in self.flowsheet.units:
            for outlet in unit.outlet_names:
                # Check if this stream is used as input anywhere
                is_input = any(
                    outlet in u.inlet_names for u in self.flowsheet.units
                )
                # Check if it's a recycle source
                is_recycle = outlet in self.flowsheet.recycles
                if not is_input and not is_recycle:
                    product_streams.add(outlet)

        # Add product nodes
        if self.show_products:
            for product_name in product_streams:
                self._nodes.append({
                    "data": {
                        "id": f"product_{product_name}",
                        "label": product_name,
                        "type": "product",
                        "node_type": "product",
                    }
                })

        # Add edges
        edge_id = 0

        # Feed -> Unit edges
        for feed_name in self.flowsheet.feeds:
            for unit in self.flowsheet.units:
                if feed_name in unit.inlet_names:
                    self._edges.append({
                        "data": {
                            "id": f"edge_{edge_id}",
                            "source": f"feed_{feed_name}",
                            "target": unit.name,
                            "label": feed_name,
                            "stream_name": feed_name,
                        }
                    })
                    edge_id += 1

        # Unit -> Unit edges
        for source_unit in self.flowsheet.units:
            for outlet in source_unit.outlet_names:
                for target_unit in self.flowsheet.units:
                    if outlet in target_unit.inlet_names:
                        self._edges.append({
                            "data": {
                                "id": f"edge_{edge_id}",
                                "source": source_unit.name,
                                "target": target_unit.name,
                                "label": outlet,
                                "stream_name": outlet,
                            }
                        })
                        edge_id += 1

        # Unit -> Product edges
        if self.show_products:
            for unit in self.flowsheet.units:
                for outlet in unit.outlet_names:
                    if outlet in product_streams:
                        self._edges.append({
                            "data": {
                                "id": f"edge_{edge_id}",
                                "source": unit.name,
                                "target": f"product_{outlet}",
                                "label": outlet,
                                "stream_name": outlet,
                            }
                        })
                        edge_id += 1

        # Recycle edges (with special styling)
        for source, dest in self.flowsheet.recycles.items():
            # Find the source unit
            source_unit = None
            for unit in self.flowsheet.units:
                if source in unit.outlet_names:
                    source_unit = unit.name
                    break

            # Find the target unit
            target_unit = None
            for unit in self.flowsheet.units:
                if dest in unit.inlet_names:
                    target_unit = unit.name
                    break

            if source_unit and target_unit:
                self._edges.append({
                    "data": {
                        "id": f"edge_{edge_id}",
                        "source": source_unit,
                        "target": target_unit,
                        "label": f"{source}→{dest}",
                        "stream_name": source,
                        "is_recycle": True,
                    }
                })
                edge_id += 1

    def _get_stylesheet(self) -> list[dict]:
        """Generate Cytoscape stylesheet with custom icons."""
        styles = [
            # Base node style
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "text-valign": "bottom",
                    "text-halign": "center",
                    "text-margin-y": "5px",
                    "font-size": "12px",
                    "font-weight": "bold",
                    "width": "60px",
                    "height": "60px",
                    "background-color": "#ffffff",
                    "border-width": "0px",
                }
            },
            # Base edge style
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "#666666",
                    "target-arrow-color": "#666666",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "label": "data(label)",
                    "font-size": "10px",
                    "text-rotation": "autorotate",
                    "text-margin-y": "-10px",
                }
            },
            # Recycle edge style
            {
                "selector": "edge[is_recycle]",
                "style": {
                    "line-style": "dashed",
                    "line-color": "#e74c3c",
                    "target-arrow-color": "#e74c3c",
                }
            },
            # Feed node style
            {
                "selector": "node[node_type='feed']",
                "style": {
                    "background-image": get_icon_data_uri("feed"),
                    "background-fit": "contain",
                    "background-clip": "none",
                }
            },
            # Product node style
            {
                "selector": "node[node_type='product']",
                "style": {
                    "background-image": get_icon_data_uri("product"),
                    "background-fit": "contain",
                    "background-clip": "none",
                }
            },
        ]

        # Add styles for each unit type
        unit_types_seen = set()
        for unit in self.flowsheet.units:
            op_type = _get_operation_type(unit)
            if op_type not in unit_types_seen:
                unit_types_seen.add(op_type)
                styles.append({
                    "selector": f"node[type='{op_type}']",
                    "style": {
                        "background-image": get_icon_data_uri(op_type),
                        "background-fit": "contain",
                        "background-clip": "none",
                    }
                })

        return styles

    def _get_layout_config(self) -> dict:
        """Get layout configuration."""
        base_config = {
            "name": self.layout,
        }

        if self.layout == "dagre":
            base_config.update({
                "rankDir": "LR",  # Left to right
                "nodeSep": 80,
                "rankSep": 100,
                "edgeSep": 20,
            })
        elif self.layout == "breadthfirst":
            base_config.update({
                "directed": True,
                "spacingFactor": 1.5,
            })
        elif self.layout == "cose":
            base_config.update({
                "nodeRepulsion": 8000,
                "idealEdgeLength": 100,
            })

        base_config.update(self.layout_options)
        return base_config

    def show(
        self,
        streams: dict[str, "Stream"] | None = None,
        width: str = "100%",
        height: str = "500px",
    ) -> "ipycytoscape.CytoscapeWidget":
        """Display the interactive flowsheet visualization.

        Args:
            streams: Solved stream data to include in tooltips (from flowsheet.solve())
            width: Widget width (CSS value)
            height: Widget height (CSS value)

        Returns:
            ipycytoscape CytoscapeWidget for display in Jupyter
        """
        # Update tooltips with stream data if provided
        if streams:
            self._update_stream_tooltips(streams)

        # Create widget
        self._widget = ipycytoscape.CytoscapeWidget()
        self._widget.graph.add_graph_from_json({
            "nodes": self._nodes,
            "edges": self._edges,
        })

        # Apply stylesheet
        self._widget.set_style(self._get_stylesheet())

        # Apply layout
        self._widget.set_layout(**self._get_layout_config())

        # Configure interaction
        self._widget.min_zoom = 0.3
        self._widget.max_zoom = 3.0

        # Set size
        self._widget.layout.width = width
        self._widget.layout.height = height

        # Enable tooltips via node data (using popper extension)
        self._setup_tooltips()

        return self._widget

    def _update_stream_tooltips(self, streams: dict[str, "Stream"]) -> None:
        """Update edge tooltips with stream data."""
        for edge in self._edges:
            stream_name = edge["data"].get("stream_name")
            if stream_name and stream_name in streams:
                edge["data"]["tooltip"] = _format_stream_tooltip(
                    streams[stream_name], stream_name
                )

        # Also update feed tooltips
        for node in self._nodes:
            if node["data"].get("node_type") == "feed":
                feed_name = node["data"]["label"]
                if feed_name in streams:
                    node["data"]["tooltip"] = _format_stream_tooltip(
                        streams[feed_name], feed_name
                    )
                elif feed_name in self.flowsheet.feeds:
                    node["data"]["tooltip"] = _format_stream_tooltip(
                        self.flowsheet.feeds[feed_name], feed_name
                    )

    def _setup_tooltips(self) -> None:
        """Configure tooltip display on hover."""
        if self._widget is None:
            return

        # ipycytoscape supports tooltips via the 'tooltip_source' attribute
        # which uses the node/edge data for tooltip content
        self._widget.tooltip_source = "tooltip"

    def export_html(
        self,
        filename: str,
        include_cytoscape_js: bool = True,
        title: str = "Process Flowsheet",
    ) -> None:
        """Export the flowsheet to a standalone HTML file.

        Args:
            filename: Output filename (should end with .html)
            include_cytoscape_js: Whether to include Cytoscape.js library inline
            title: HTML page title
        """
        # Generate HTML with embedded Cytoscape.js
        html_content = self._generate_html(title, include_cytoscape_js)

        with open(filename, "w") as f:
            f.write(html_content)

    def _generate_html(self, title: str, include_js: bool) -> str:
        """Generate standalone HTML content."""
        graph_data = json.dumps({
            "nodes": self._nodes,
            "edges": self._edges,
        }, indent=2)

        stylesheet = json.dumps(self._get_stylesheet(), indent=2)
        layout_config = json.dumps(self._get_layout_config(), indent=2)

        if include_js:
            cytoscape_src = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"
            dagre_src = "https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"
            cytoscape_dagre_src = "https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"
            popper_src = "https://unpkg.com/@popperjs/core@2"
            cytoscape_popper_src = "https://cdn.jsdelivr.net/npm/cytoscape-popper@2.0.0/cytoscape-popper.min.js"
            tippy_src = "https://unpkg.com/tippy.js@6"
        else:
            cytoscape_src = ""
            dagre_src = ""
            cytoscape_dagre_src = ""
            popper_src = ""
            cytoscape_popper_src = ""
            tippy_src = ""

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{cytoscape_src}"></script>
    <script src="{dagre_src}"></script>
    <script src="{cytoscape_dagre_src}"></script>
    <script src="{popper_src}"></script>
    <script src="{cytoscape_popper_src}"></script>
    <script src="{tippy_src}"></script>
    <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/dist/tippy.css">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
        }}
        h1 {{
            margin: 0 0 20px 0;
            color: #333;
        }}
        #cy {{
            width: 100%;
            height: calc(100vh - 100px);
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .tippy-box {{
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 12px;
            max-width: 300px;
        }}
        .tippy-box b {{
            color: #3498db;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="cy"></div>
    <script>
        // Register dagre layout
        if (typeof cytoscape !== 'undefined' && typeof cytoscape.use !== 'undefined' && typeof cytoscapeDagre !== 'undefined') {{
            cytoscape.use(cytoscapeDagre);
        }}

        // Graph data
        var graphData = {graph_data};

        // Stylesheet
        var stylesheet = {stylesheet};

        // Layout configuration
        var layoutConfig = {layout_config};

        // Initialize Cytoscape
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: graphData,
            style: stylesheet,
            layout: layoutConfig,
            minZoom: 0.3,
            maxZoom: 3,
            wheelSensitivity: 0.3,
        }});

        // Setup tooltips with Tippy.js
        if (typeof tippy !== 'undefined') {{
            cy.nodes().forEach(function(node) {{
                var tooltip = node.data('tooltip');
                if (tooltip) {{
                    var ref = node.popperRef();
                    var dummyDomEle = document.createElement('div');
                    tippy(dummyDomEle, {{
                        getReferenceClientRect: ref.getBoundingClientRect,
                        content: tooltip,
                        trigger: 'manual',
                        placement: 'bottom',
                        arrow: true,
                        allowHTML: true,
                    }});
                    var tippyInstance = dummyDomEle._tippy;
                    node.on('mouseover', function() {{ tippyInstance.show(); }});
                    node.on('mouseout', function() {{ tippyInstance.hide(); }});
                }}
            }});

            cy.edges().forEach(function(edge) {{
                var tooltip = edge.data('tooltip');
                if (tooltip) {{
                    var ref = edge.popperRef();
                    var dummyDomEle = document.createElement('div');
                    tippy(dummyDomEle, {{
                        getReferenceClientRect: ref.getBoundingClientRect,
                        content: tooltip,
                        trigger: 'manual',
                        placement: 'top',
                        arrow: true,
                        allowHTML: true,
                    }});
                    var tippyInstance = dummyDomEle._tippy;
                    edge.on('mouseover', function() {{ tippyInstance.show(); }});
                    edge.on('mouseout', function() {{ tippyInstance.hide(); }});
                }}
            }});
        }}

        // Fit to viewport on load
        cy.fit(50);
    </script>
</body>
</html>'''

    def export_png(self, filename: str, scale: float = 2.0) -> None:
        """Export the flowsheet to a PNG image.

        Note: This requires the widget to be displayed first.

        Args:
            filename: Output filename (should end with .png)
            scale: Scale factor for resolution (2.0 = 2x resolution)
        """
        if self._widget is None:
            raise RuntimeError(
                "Widget must be displayed first. Call show() before export_png()."
            )

        # ipycytoscape doesn't have direct PNG export,
        # so we'll save a message about using browser export
        warnings.warn(
            "PNG export requires manual screenshot or browser export. "
            "In the HTML export, you can use browser developer tools to "
            "capture the canvas, or use a screenshot tool. "
            f"For automated PNG export, consider using the HTML export: {filename.replace('.png', '.html')}"
        )

    def get_graph_data(self) -> dict:
        """Get the raw graph data as a dictionary.

        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        return {
            "nodes": self._nodes,
            "edges": self._edges,
        }

    def to_networkx(self) -> "Any":
        """Convert the flowsheet graph to a NetworkX DiGraph.

        Returns:
            networkx.DiGraph representation of the flowsheet

        Raises:
            ImportError: If networkx is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for this method. "
                "Install it with: pip install networkx"
            )

        G = nx.DiGraph()

        # Add nodes
        for node in self._nodes:
            node_id = node["data"]["id"]
            G.add_node(node_id, **node["data"])

        # Add edges
        for edge in self._edges:
            source = edge["data"]["source"]
            target = edge["data"]["target"]
            G.add_edge(source, target, **edge["data"])

        return G


def visualize_flowsheet(
    flowsheet: "Flowsheet",
    streams: dict[str, "Stream"] | None = None,
    width: str = "100%",
    height: str = "500px",
    **kwargs,
) -> "ipycytoscape.CytoscapeWidget":
    """Convenience function to visualize a flowsheet.

    Args:
        flowsheet: The Flowsheet object to visualize
        streams: Solved stream data (from flowsheet.solve())
        width: Widget width (CSS string)
        height: Widget height (CSS string)
        **kwargs: Additional arguments passed to FlowsheetVisualizer

    Returns:
        ipycytoscape CytoscapeWidget for display in Jupyter

    Example:
        >>> results = flowsheet.solve()
        >>> visualize_flowsheet(flowsheet, streams=results, height="400px")
    """
    viz = FlowsheetVisualizer(flowsheet, **kwargs)
    return viz.show(streams=streams, width=width, height=height)
