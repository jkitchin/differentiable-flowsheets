"""Interactive flowsheet rendering using Plotly."""

from typing import Optional, Any
import math

from .graph import FlowsheetGraph, Node, Edge
from .styles import get_unit_style, get_stream_color, UnitStyle


def render_flowsheet(
    graph: FlowsheetGraph,
    layout: str = "dot",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 600,
    show_stream_labels: bool = True,
    edge_width_scale: float = 2.0,
    **layout_kwargs
) -> Any:
    """Render an interactive flowsheet visualization using Plotly.

    Args:
        graph: FlowsheetGraph to visualize
        layout: Layout algorithm ('dot', 'neato', 'spring', etc.)
        title: Optional title for the figure
        width: Figure width in pixels
        height: Figure height in pixels
        show_stream_labels: Whether to show stream ID labels on edges
        edge_width_scale: Scale factor for edge widths based on flow
        **layout_kwargs: Additional arguments for layout algorithm

    Returns:
        Plotly Figure object

    Example:
        >>> graph = FlowsheetGraph()
        >>> graph.add_node("reactor", "CSTR-1", unit_type="CSTR")
        >>> graph.add_node("flash", "Flash-1", unit_type="Flash")
        >>> graph.add_edge("reactor", "flash")
        >>> fig = render_flowsheet(graph)
        >>> fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: pip install plotly"
        )

    # Compute layout if positions not set
    positions = graph.compute_layout(algorithm=layout, **layout_kwargs)

    # Create figure
    fig = go.Figure()

    # Add edges first (so they're behind nodes)
    _add_edges(fig, graph, positions, show_stream_labels, edge_width_scale)

    # Add nodes
    _add_nodes(fig, graph, positions)

    # Configure layout
    fig.update_layout(
        title=title or graph.name,
        showlegend=False,
        hovermode='closest',
        width=width,
        height=height,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Add zoom and pan controls
    fig.update_layout(
        dragmode='pan',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.8)',
        ),
    )

    return fig


def _add_nodes(
    fig,
    graph: FlowsheetGraph,
    positions: dict[str, tuple[float, float]]
) -> None:
    """Add node traces to the figure."""
    import plotly.graph_objects as go

    # Group nodes by type for consistent styling
    node_groups: dict[str, list[Node]] = {}
    for node in graph.nodes.values():
        unit_type = node.unit_type
        if unit_type not in node_groups:
            node_groups[unit_type] = []
        node_groups[unit_type].append(node)

    for unit_type, nodes in node_groups.items():
        style = get_unit_style(unit_type)

        x_coords = []
        y_coords = []
        texts = []
        hovertexts = []

        for node in nodes:
            pos = positions.get(node.id, (0, 0))
            x_coords.append(pos[0])
            y_coords.append(pos[1])

            # Node label (with optional icon)
            if style.icon:
                texts.append(f"{style.icon}<br>{node.name}")
            else:
                texts.append(node.name)

            hovertexts.append(node.tooltip_html())

        # Create node trace
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=style.size,
                color=style.color,
                line=dict(color=style.border_color, width=2),
                symbol=_shape_to_plotly_symbol(style.shape),
            ),
            text=texts,
            textposition='bottom center' if style.label_position == 'bottom' else 'top center',
            textfont=dict(size=10),
            hovertext=hovertexts,
            hoverinfo='text',
            name=unit_type,
        ))


def _add_edges(
    fig,
    graph: FlowsheetGraph,
    positions: dict[str, tuple[float, float]],
    show_labels: bool,
    width_scale: float
) -> None:
    """Add edge traces to the figure."""
    import plotly.graph_objects as go

    for edge in graph.edges.values():
        src_pos = positions.get(edge.source, (0, 0))
        tgt_pos = positions.get(edge.target, (0, 0))

        # Calculate edge width based on flow
        base_width = 1.5
        if edge.stream_data:
            total_flow = edge.total_flow()
            # Log scale for width
            width = base_width + width_scale * math.log1p(total_flow) / 3
        else:
            width = base_width

        # Get color based on metadata
        stream_type = edge.metadata.get('type', 'default')
        color = get_stream_color(stream_type)

        # Draw edge line
        fig.add_trace(go.Scatter(
            x=[src_pos[0], tgt_pos[0]],
            y=[src_pos[1], tgt_pos[1]],
            mode='lines',
            line=dict(color=color, width=width),
            hovertext=edge.tooltip_html(),
            hoverinfo='text',
            showlegend=False,
        ))

        # Add arrow head
        _add_arrow(fig, src_pos, tgt_pos, color, width)

        # Add stream label at midpoint
        if show_labels and edge.stream_data:
            mid_x = (src_pos[0] + tgt_pos[0]) / 2
            mid_y = (src_pos[1] + tgt_pos[1]) / 2

            # Offset label slightly
            dx = tgt_pos[0] - src_pos[0]
            dy = tgt_pos[1] - src_pos[1]
            length = math.sqrt(dx*dx + dy*dy) + 1e-6
            offset = 10 / length

            fig.add_annotation(
                x=mid_x - dy * offset,
                y=mid_y + dx * offset,
                text=edge.id,
                showarrow=False,
                font=dict(size=8, color='#666'),
            )


def _add_arrow(fig, src: tuple, tgt: tuple, color: str, width: float) -> None:
    """Add an arrowhead at the target end of an edge."""
    import plotly.graph_objects as go

    # Calculate arrow position (slightly before target)
    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    length = math.sqrt(dx*dx + dy*dy)

    if length < 1e-6:
        return

    # Normalize
    dx /= length
    dy /= length

    # Position arrow slightly before target
    arrow_back = 25  # pixels
    arrow_x = tgt[0] - dx * arrow_back
    arrow_y = tgt[1] - dy * arrow_back

    # Arrow size
    arrow_size = 8 + width

    fig.add_annotation(
        x=tgt[0] - dx * 20,
        y=tgt[1] - dy * 20,
        ax=arrow_x,
        ay=arrow_y,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=width,
        arrowcolor=color,
    )


def _shape_to_plotly_symbol(shape: str) -> str:
    """Convert shape name to Plotly marker symbol."""
    mapping = {
        'circle': 'circle',
        'rect': 'square',
        'square': 'square',
        'diamond': 'diamond',
        'hexagon': 'hexagon',
        'triangle': 'triangle-up',
    }
    return mapping.get(shape, 'circle')


def show_flowsheet(
    graph: FlowsheetGraph,
    **kwargs
) -> None:
    """Render and display a flowsheet interactively.

    Convenience function that calls render_flowsheet and shows the result.

    Args:
        graph: FlowsheetGraph to visualize
        **kwargs: Arguments passed to render_flowsheet
    """
    fig = render_flowsheet(graph, **kwargs)
    fig.show()


def to_html(
    graph: FlowsheetGraph,
    filename: Optional[str] = None,
    **kwargs
) -> str:
    """Export flowsheet visualization to standalone HTML.

    Args:
        graph: FlowsheetGraph to visualize
        filename: Optional filename to save HTML to
        **kwargs: Arguments passed to render_flowsheet

    Returns:
        HTML string
    """
    fig = render_flowsheet(graph, **kwargs)
    html = fig.to_html(include_plotlyjs=True, full_html=True)

    if filename:
        with open(filename, 'w') as f:
            f.write(html)

    return html


def to_image(
    graph: FlowsheetGraph,
    filename: str,
    format: str = "png",
    scale: float = 2.0,
    **kwargs
) -> None:
    """Export flowsheet visualization to image file.

    Args:
        graph: FlowsheetGraph to visualize
        filename: Output filename
        format: Image format ('png', 'svg', 'pdf', 'jpeg')
        scale: Resolution scale factor
        **kwargs: Arguments passed to render_flowsheet
    """
    fig = render_flowsheet(graph, **kwargs)
    fig.write_image(filename, format=format, scale=scale)
