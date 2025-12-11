"""Visualization module for difflow.

This module provides interactive visualization of process flowsheets
using ipycytoscape with:
- Custom P&ID-style icons for unit operations
- Left-to-right hierarchical layout
- Tooltips showing stream compositions and unit parameters
- Drag-and-drop node positioning
- Export to HTML and PNG for publications

Requirements:
    pip install ipycytoscape

Example usage:
    >>> from difflow import Flowsheet, Unit, CSTR, Flash
    >>> from difflow.visualization import visualize_flowsheet, FlowsheetVisualizer
    >>>
    >>> # Create and solve flowsheet
    >>> fs = Flowsheet(["A", "B", "C"])
    >>> # ... add feeds, units, etc.
    >>> results = fs.solve()
    >>>
    >>> # Quick visualization
    >>> visualize_flowsheet(fs, streams=results)
    >>>
    >>> # Or with more control
    >>> viz = FlowsheetVisualizer(fs)
    >>> widget = viz.show(streams=results)
    >>> viz.export_html("flowsheet.html")
"""

from difflow.visualization.icons import (
    get_icon_svg,
    get_icon_data_uri,
    list_available_icons,
    register_icon,
    ICON_REGISTRY,
)

from difflow.visualization.flowsheet_viz import (
    FlowsheetVisualizer,
    visualize_flowsheet,
)

__all__ = [
    # Main visualizer
    "FlowsheetVisualizer",
    "visualize_flowsheet",
    # Icon utilities
    "get_icon_svg",
    "get_icon_data_uri",
    "list_available_icons",
    "register_icon",
    "ICON_REGISTRY",
]
