"""Where to put the nodes when nobody has said.

A canvas needs coordinates, and a flowsheet does not carry any: it is a
graph, and the file only became able to remember positions in format
version 2. So the first time a flowsheet is opened --- and whenever the
"Re-layout" button is pressed --- the positions come from here.

The algorithm is the one :mod:`difflow.report.diagram` already used for
its SVG, promoted rather than copied: a longest-path column assignment
per unit, feeds banked on the left and dangling products on the right.
It reads left to right, which is how a process flow diagram reads, and
it is stable --- the same flowsheet lays out the same way every time,
so reopening a file does not shuffle it.

Recycle edges are excluded from the path length. They are exactly the
arcs that make the graph cyclic, and including them would either not
terminate or push a unit arbitrarily far right; leaving them out is what
makes a recycle draw as an arrow going back, which is what it is.
"""

from __future__ import annotations

from typing import Iterable, Mapping

#: Canvas pixels. Roughly the node box plus a gap, matching the spacing
#: :mod:`difflow.report.diagram` uses for the same graph.
COL_W = 220
ROW_H = 110
MARGIN = 40

#: Node key prefixes. Units are keyed by their bare name --- the same
#: vocabulary :meth:`difflow.flowsheet.Flowsheet._apply_params` uses, where
#: a feed is ``"feed:<stream>"`` and a unit is just its name.
FEED_PREFIX = "feed:"
PRODUCT_PREFIX = "product:"


def unit_columns(unit_names: Iterable[str],
                 up_edges: Mapping[str, Iterable[str]]) -> dict[str, int]:
    """Longest-path column index per unit, ignoring back (recycle) edges.

    ``up_edges`` maps a unit to the units feeding it. A Bellman-Ford-style
    relaxation capped at ``len(units)`` passes assigns each unit one past
    its deepest predecessor; the cap is what makes it safe on a cyclic
    graph, where a recycle edge simply stops lengthening the path.

    Args:
        unit_names: every unit, in the order they should break ties.
        up_edges: unit -> its upstream units.

    Returns:
        ``{unit_name: column index}``, starting at 0.
    """
    names = list(unit_names)
    col = {u: 0 for u in names}
    for _ in range(len(names)):
        changed = False
        for u in names:
            for p in up_edges.get(u, ()):  # predecessors
                if col[p] + 1 > col[u]:
                    col[u] = col[p] + 1
                    changed = True
        if not changed:
            break
    return col


def _graph(flowsheet):
    """The four things a layout needs: units, feeds, products, up-edges."""
    units = list(flowsheet.units)
    producer: dict[str, str] = {}
    for u in units:
        for out in u.outlet_names:
            producer[out] = u.name
    consumed: set[str] = set()
    for u in units:
        consumed.update(u.inlet_names)

    recycle_sources = set(getattr(flowsheet, "recycles", {}))
    up_edges: dict[str, list[str]] = {u.name: [] for u in units}
    for u in units:
        for inlet in u.inlet_names:
            src = producer.get(inlet)
            if src is not None and inlet not in recycle_sources:
                up_edges[u.name].append(src)

    # A feed is an inlet nothing produces; a product is an outlet nothing
    # consumes and no recycle carries back.  Declared feeds come first and in
    # their declared order, so the left-hand bank does not reshuffle when a
    # unit is added.  A recycle destination is not a feed: nothing produces it
    # either, but it is fed by the recycle arc, and banking it on the left
    # would claim the flowsheet has an inlet it does not have.
    fed_by_recycle = set(getattr(flowsheet, "recycles", {}).values())
    feeds = [n for n in getattr(flowsheet, "feeds", {})]
    seen = set(feeds) | fed_by_recycle
    for u in units:
        for inlet in u.inlet_names:
            if producer.get(inlet) is None and inlet not in seen:
                seen.add(inlet)
                feeds.append(inlet)
    products = [out for u in units for out in u.outlet_names
                if out not in consumed and out not in recycle_sources]
    return units, feeds, products, up_edges


def auto_layout(flowsheet, *, col_w: int = COL_W, row_h: int = ROW_H,
                margin: int = MARGIN) -> dict[str, tuple[float, float]]:
    """Canvas positions for every node in ``flowsheet``.

    Args:
        flowsheet: the :class:`~difflow.flowsheet.Flowsheet` to lay out.
        col_w: horizontal spacing between columns, in canvas pixels.
        row_h: vertical spacing between rows.
        margin: offset of the first column and row from the origin.

    Returns:
        ``{key: (x, y)}`` where a unit's key is its name, a feed's is
        ``"feed:<stream>"`` and a dangling product's is
        ``"product:<stream>"``. Empty for a flowsheet with no units.

    Example:
        >>> positions = auto_layout(fs)                  # doctest: +SKIP
        >>> fs.view["nodes"] = {k: {"x": x, "y": y}
        ...                     for k, (x, y) in positions.items()}
    """
    units, feeds, products, up_edges = _graph(flowsheet)
    if not units:
        return {}

    col = unit_columns([u.name for u in units], up_edges)
    offset = 1 if feeds else 0
    product_col = (max(col.values()) if col else 0) + offset + 1

    slots: dict[str, tuple[int, int]] = {}
    for row, name in enumerate(feeds):
        slots[FEED_PREFIX + name] = (0, row)
    by_col: dict[int, list[str]] = {}
    for u in units:
        by_col.setdefault(col[u.name] + offset, []).append(u.name)
    for c, names in by_col.items():
        for row, name in enumerate(names):
            slots[name] = (c, row)
    for row, name in enumerate(products):
        slots[PRODUCT_PREFIX + name] = (product_col, row)

    return {key: (float(margin + c * col_w), float(margin + r * row_h))
            for key, (c, r) in slots.items()}
