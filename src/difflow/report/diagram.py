"""Self-contained inline-SVG flowsheet diagram for HTML reports.

The diagram is generated purely from the :class:`~difflow.report.ir.Report`
intermediate representation (unit inlet/outlet names + topology), so it needs
no ``ipycytoscape``, no external JavaScript, and no network access — it embeds
directly in the standalone HTML report.  The layout is a simple left-to-right
layered (Sugiyama-style) placement: feeds on the left, products on the right,
units in longest-path columns between them; recycle arcs are drawn dashed.
"""

from __future__ import annotations

from html import escape

from difflow.report.ir import Report

# Geometry (px).
_COL_W = 190
_ROW_H = 78
_NODE_W = 130
_NODE_H = 46
_MARGIN = 24


def _unit_columns(unit_names, up_edges) -> dict[str, int]:
    """Longest-path column index per unit, ignoring back (recycle) edges.

    ``up_edges`` maps a unit to the list of upstream units feeding it.  A
    Bellman-Ford-style relaxation capped at ``len(units)`` passes assigns each
    unit one past its deepest predecessor; the cap makes it safe on cyclic
    graphs (recycle edges simply stop lengthening the path).
    """
    col = {u: 0 for u in unit_names}
    for _ in range(len(unit_names)):
        changed = False
        for u in unit_names:
            for p in up_edges.get(u, ()):  # predecessors
                if col[p] + 1 > col[u]:
                    col[u] = col[p] + 1
                    changed = True
        if not changed:
            break
    return col


def _node_svg(x: int, y: int, label: str, sub: str, kind: str) -> str:
    """One rounded-rectangle node with a title line and a subtitle."""
    fill = {"feed": "#eef6ff", "product": "#f0fff0", "unit": "#ffffff"}.get(kind, "#fff")
    stroke = {"feed": "#5b8def", "product": "#3caa5a", "unit": "#888"}.get(kind, "#888")
    parts = [
        f'<rect x="{x}" y="{y}" width="{_NODE_W}" height="{_NODE_H}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + _NODE_W // 2}" y="{y + 19}" text-anchor="middle" '
        f'font-size="13" font-weight="bold">{escape(label)}</text>',
    ]
    if sub:
        parts.append(
            f'<text x="{x + _NODE_W // 2}" y="{y + 35}" text-anchor="middle" '
            f'font-size="10" fill="#555">{escape(sub)}</text>'
        )
    return "".join(parts)


def _edge_svg(x1: int, y1: int, x2: int, y2: int, label: str, recycle: bool) -> str:
    """A straight arrow between two node anchor points with a small label."""
    color = "#e74c3c" if recycle else "#666"
    dash = ' stroke-dasharray="6 4"' if recycle else ""
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2 - 4
    line = (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="1.5"{dash} marker-end="url(#arrow)"/>'
    )
    text = ""
    if label:
        text = (
            f'<text x="{mid_x}" y="{mid_y}" text-anchor="middle" font-size="9" '
            f'fill="{color}">{escape(label)}</text>'
        )
    return line + text


def flowsheet_svg(report: Report) -> str:
    """Render an inline ``<svg>`` flowsheet diagram from a :class:`Report`.

    Returns an empty string when the flowsheet has no units.
    """
    units = report.units
    if not units:
        return ""

    unit_names = [u.name for u in units]
    producer: dict[str, str] = {}
    for u in units:
        for out in u.outlet_names:
            producer[out] = u.name
    all_inlets: set[str] = set()
    for u in units:
        all_inlets.update(u.inlet_names)

    # Unit -> upstream units (only where an inlet is produced by another unit).
    up_edges: dict[str, list[str]] = {u.name: [] for u in units}
    unit_arcs: list[tuple[str, str, str]] = []  # (src_unit, dst_unit, stream)
    recycle_sources = {r.source_stream for r in report.topology.recycles}
    for u in units:
        for inlet in u.inlet_names:
            src = producer.get(inlet)
            if src is not None and inlet not in recycle_sources:
                up_edges[u.name].append(src)
                unit_arcs.append((src, u.name, inlet))

    col = _unit_columns(unit_names, up_edges)
    max_unit_col = max(col.values()) if col else 0

    # Feed streams: inlets not produced by any unit.  Product streams: outlets
    # not consumed anywhere and not a recycle source.
    feed_streams: list[str] = []
    seen_f: set[str] = set()
    for u in units:
        for inlet in u.inlet_names:
            if producer.get(inlet) is None and inlet not in seen_f:
                seen_f.add(inlet)
                feed_streams.append(inlet)
    product_streams: list[str] = []
    for u in units:
        for out in u.outlet_names:
            if out not in all_inlets and out not in recycle_sources:
                product_streams.append(out)

    feed_col = 0
    unit_col_offset = 1 if feed_streams else 0
    product_col = max_unit_col + unit_col_offset + (1 if product_streams else 0)

    # Assign a (col, row) grid slot to every node.
    def _place(items):
        return {name: row for row, name in enumerate(items)}

    node_pos: dict[str, tuple[int, int]] = {}
    feed_rows = _place(feed_streams)
    for name, row in feed_rows.items():
        node_pos[f"feed:{name}"] = (feed_col, row)
    # Units grouped by column.
    by_col: dict[int, list[str]] = {}
    for name in unit_names:
        by_col.setdefault(col[name] + unit_col_offset, []).append(name)
    for c, names in by_col.items():
        for row, name in enumerate(names):
            node_pos[f"unit:{name}"] = (c, row)
    prod_rows = _place(product_streams)
    for name, row in prod_rows.items():
        node_pos[f"product:{name}"] = (product_col, row)

    max_col = max((c for c, _ in node_pos.values()), default=0)
    max_row = max((r for _, r in node_pos.values()), default=0)
    width = 2 * _MARGIN + max_col * _COL_W + _NODE_W
    height = 2 * _MARGIN + max_row * _ROW_H + _NODE_H

    def _xy(key):
        c, r = node_pos[key]
        return _MARGIN + c * _COL_W, _MARGIN + r * _ROW_H

    def _right(key):
        x, y = _xy(key)
        return x + _NODE_W, y + _NODE_H // 2

    def _left(key):
        x, y = _xy(key)
        return x, y + _NODE_H // 2

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px" xmlns="http://www.w3.org/2000/svg" '
        'font-family="system-ui, sans-serif">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" '
        'refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#666"/></marker></defs>',
    ]

    # Edges first, so nodes render on top.
    for src, dst, stream in unit_arcs:
        x1, y1 = _right(f"unit:{src}")
        x2, y2 = _left(f"unit:{dst}")
        out.append(_edge_svg(x1, y1, x2, y2, stream, recycle=False))
    for name in feed_streams:
        # feed -> first consuming unit
        target = next((u.name for u in units if name in u.inlet_names), None)
        if target is None:
            continue
        x1, y1 = _right(f"feed:{name}")
        x2, y2 = _left(f"unit:{target}")
        out.append(_edge_svg(x1, y1, x2, y2, "", recycle=False))
    for name in product_streams:
        source = producer.get(name)
        if source is None:
            continue
        x1, y1 = _right(f"unit:{source}")
        x2, y2 = _left(f"product:{name}")
        out.append(_edge_svg(x1, y1, x2, y2, "", recycle=False))
    for rec in report.topology.recycles:
        src = producer.get(rec.source_stream)
        dst = next(
            (u.name for u in units if rec.dest_stream in u.inlet_names), None
        )
        if src is None or dst is None:
            continue
        x1, y1 = _right(f"unit:{src}")
        x2, y2 = _left(f"unit:{dst}")
        out.append(
            _edge_svg(x1, y1, x2, y2, f"{rec.source_stream}→{rec.dest_stream}", True)
        )

    # Nodes.
    for name in feed_streams:
        x, y = _xy(f"feed:{name}")
        out.append(_node_svg(x, y, name, "feed", "feed"))
    for u in units:
        x, y = _xy(f"unit:{u.name}")
        out.append(_node_svg(x, y, u.name, u.type, "unit"))
    for name in product_streams:
        x, y = _xy(f"product:{name}")
        out.append(_node_svg(x, y, name, "product", "product"))

    out.append("</svg>")
    return "\n".join(out)
