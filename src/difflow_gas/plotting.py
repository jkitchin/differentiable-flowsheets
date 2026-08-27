"""Schematic drawing of a gas network.

One entry point, :func:`draw_network`, renders a network's topology onto
a matplotlib axes: nodes shaped by their boundary role (entry, exit,
transit) and arcs styled by kind, with optional pressure and flow
annotations from a solved or reconciled state.

This is a *schematic*, not a chart --- it carries topology and
identity, not magnitude. Flow and pressure values are printed as text
rather than encoded in line width or colour, so that a reader takes the
numbers from the numbers.

Arcs are drawn along their reference direction, which is the direction
a positive flow runs; a negative annotation therefore means gas moving
against the arrow, which is routine in a meshed network.
"""

from __future__ import annotations

import math
from typing import Sequence

from difflow_gas.network import GasNetwork

# Light-surface slots of the difflow documentation palette. Node roles
# are an identity encoding, so the hues are assigned in fixed order and
# never cycled; the flagged colour is the reserved "critical" status
# step, which is why it never doubles as a role.
_ENTRY = "#2a78d6"      # categorical slot 1, blue
_EXIT = "#eb6834"       # categorical slot 2, orange
_TRANSIT = "#fcfcfb"    # surface: an open circle
_FLAG = "#d03b3b"       # status: critical
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_LINE = "#8a8985"

#: line style per arc kind
_ARC_STYLE = {
    "pipe": dict(linestyle="-", linewidth=1.8),
    "resistor": dict(linestyle=(0, (6, 2)), linewidth=1.8),
    "valve": dict(linestyle="-", linewidth=1.2),
    "short_pipe": dict(linestyle="-", linewidth=1.0),
    "control_valve": dict(linestyle=(0, (3, 2)), linewidth=1.4),
    "compressor": dict(linestyle="-", linewidth=1.8),
}

#: midpoint marker per arc kind (None = plain line)
_ARC_MARKER = {
    "compressor": ("^", 11.0),
    "valve": ("s", 7.0),
    "control_valve": ("s", 7.0),
}


def circular_positions(nodes: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Evenly spaced positions on a unit circle, in the given order.

    A readable fallback when no layout is supplied. For anything beyond
    a handful of nodes, pass explicit positions --- a schematic that
    mirrors the physical layout is worth far more than an automatic one.
    """
    n = max(len(nodes), 1)
    return {
        node: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }


def _node_role(network: GasNetwork, node: str) -> str:
    supply = network.supply_kg_s.get(node, 0.0)
    if supply > 0:
        return "entry"
    if supply < 0:
        return "exit"
    return "transit"


def draw_network(
    network: GasNetwork,
    *,
    ax=None,
    pos: dict[str, tuple[float, float]] | None = None,
    pressures: dict[str, float] | None = None,
    flows: dict[str, float] | None = None,
    supplies: dict[str, float] | None = None,
    highlight: Sequence[str] = (),
    title: str | None = None,
    legend: bool = True,
    node_size: float = 900.0,
):
    """Draw a gas network schematic.

    Args:
        network: the network to draw.
        ax: matplotlib axes to draw on; a new figure is made if omitted.
        pos: node id -> (x, y). Defaults to a circle, which is only
            readable for very small networks --- pass a layout that
            matches the physical arrangement.
        pressures: node id -> bar, printed under each node.
        flows: arc id -> kg/s, printed beside each arc. Negative means
            flow against the drawn arrow.
        supplies: node id -> kg/s, printed beside entry and exit nodes;
            defaults to ``network.supply_kg_s``. Pass measured or
            reconciled nominations to show those instead.
        highlight: arc or node ids to mark in the critical colour, for
            pointing at a suspect meter or a fouled pipe. Highlighted
            items are also labelled, so colour is never the only cue.
        title: axes title.
        legend: draw the node-role and arc-kind legend.
        node_size: marker area for the node circles.

    Returns:
        The axes drawn on.

    Example:
        >>> ax = draw_network(net, pos={"src": (0, 0), "a": (1, 0)},
        ...                   flows=q_kg_s, highlight=["q_p3"])
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))
    pos = pos or circular_positions(network.nodes)
    missing = set(network.nodes) - set(pos)
    if missing:
        raise ValueError(f"no position given for nodes {sorted(missing)}")
    supplies = network.supply_kg_s if supplies is None else supplies
    highlight = set(highlight)

    kinds_seen = set()
    for aid, arc in network.arcs.items():
        x0, y0 = pos[arc.from_node]
        x1, y1 = pos[arc.to_node]
        flagged = aid in highlight
        colour = _FLAG if flagged else _LINE
        style = dict(_ARC_STYLE.get(arc.kind, _ARC_STYLE["pipe"]))
        kinds_seen.add(arc.kind)

        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=colour,
                shrinkA=16, shrinkB=16,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"] + (0.8 if flagged else 0.0),
            ),
            zorder=1,
        )

        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        marker = _ARC_MARKER.get(arc.kind)
        if marker is not None:
            shape, size = marker
            ax.plot(
                [mx], [my], marker=shape, markersize=size,
                markerfacecolor="#ffffff" if arc.kind == "valve" else colour,
                markeredgecolor=colour, markeredgewidth=1.4,
                linestyle="none", zorder=2,
            )

        # offset the label perpendicular to the arc so it clears the line
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / norm * 0.19, dx / norm * 0.19
        label = aid
        if flows is not None and aid in flows:
            label = f"{aid}\n{flows[aid]:.1f} kg/s"
        ax.text(
            mx + ox, my + oy, label,
            ha="center", va="center", fontsize=8,
            color=_FLAG if flagged else _INK_SOFT,
            fontweight="bold" if flagged else "normal", zorder=3,
        )

    roles_seen = set()
    for node in network.nodes:
        x, y = pos[node]
        role = _node_role(network, node)
        roles_seen.add(role)
        flagged = node in highlight
        ax.scatter(
            [x], [y], s=node_size,
            facecolor={"entry": _ENTRY, "exit": _EXIT, "transit": _TRANSIT}[role],
            edgecolor=_FLAG if flagged else _INK_SOFT,
            linewidth=2.2 if flagged else 1.2, zorder=4,
        )
        ax.text(
            x, y, node, ha="center", va="center", fontsize=9,
            color="#ffffff" if role != "transit" else _INK,
            fontweight="bold", zorder=5,
        )

        caption = []
        if pressures is not None and node in pressures:
            caption.append(f"{pressures[node]:.1f} bar")
        if supplies and node in supplies and supplies[node] != 0.0:
            caption.append(f"{supplies[node]:+.0f} kg/s")
        if caption:
            ax.text(
                x, y - 0.27, "\n".join(caption), ha="center", va="top",
                fontsize=8, color=_INK_SOFT, zorder=5,
            )

    if legend:
        handles = []
        for role, colour, name in [
            ("entry", _ENTRY, "entry (supply)"),
            ("exit", _EXIT, "exit (demand)"),
            ("transit", _TRANSIT, "transit"),
        ]:
            if role in roles_seen:
                handles.append(Line2D(
                    [], [], marker="o", linestyle="none", markersize=9,
                    markerfacecolor=colour, markeredgecolor=_INK_SOFT,
                    label=name,
                ))
        for kind in sorted(kinds_seen):
            style = _ARC_STYLE.get(kind, _ARC_STYLE["pipe"])
            marker = _ARC_MARKER.get(kind)
            handles.append(Line2D(
                [], [], color=_LINE,
                linestyle=style["linestyle"], linewidth=style["linewidth"],
                marker=marker[0] if marker else None,
                markersize=7 if marker else 0,
                markerfacecolor=_LINE, markeredgecolor=_LINE,
                label=kind.replace("_", " "),
            ))
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
            frameon=False, fontsize=8, labelcolor=_INK_SOFT,
        )

    if title:
        ax.set_title(title, fontsize=11, color=_INK, loc="left")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.45
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
