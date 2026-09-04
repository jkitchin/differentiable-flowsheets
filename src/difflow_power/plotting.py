"""Schematic drawing of an electrical network.

One entry point, :func:`draw_network`, renders a network's topology
onto a matplotlib axes: buses shaped by their role (slack, generator,
load, transit) and branches styled by kind (line, transformer, phase
shifter), with optional voltage, flow and loading annotations from a
solved state.

This is a *schematic*, not a chart --- it carries topology and
identity, not magnitude. Voltages and flows are printed as text rather
than encoded in line width or colour, so that a reader takes the
numbers from the numbers. The one exception is
:func:`draw_network`'s ``loading`` argument, where an overload is
flagged in the reserved critical colour, because "which line is over
its rating" is the question a schematic is looked at to answer.

Branches are drawn along their reference direction, which is the
direction a positive flow runs; a negative annotation therefore means
power moving against the arrow, which is routine in a meshed network
and happens on feeders with distributed generation.

matplotlib is imported inside the functions, so importing
:mod:`difflow_power` does not require it.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from difflow_power.network import PowerNetwork

# Light-surface slots of the difflow documentation palette. Bus roles
# are an identity encoding, so the hues are assigned in fixed order and
# never cycled; the flagged colour is the reserved "critical" status
# step, which is why it never doubles as a role.
_SLACK = "#2a78d6"      # categorical slot 1, blue
_GENERATOR = "#eb6834"  # categorical slot 2, orange
_LOAD = "#f2c14e"       # categorical slot 3, amber
_TRANSIT = "#fcfcfb"    # surface: an open circle
_FLAG = "#d03b3b"       # status: critical
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_LINE = "#8a8985"


def circular_positions(
    buses: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """Evenly spaced positions on a unit circle, in the given order.

    A readable fallback when no layout is supplied. For anything beyond
    a handful of buses, pass explicit positions --- a schematic that
    mirrors the geographic or one-line layout is worth far more than an
    automatic one, and no automatic layout recovers the structure an
    engineer already knows.
    """
    n = max(len(buses), 1)
    return {
        b: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, b in enumerate(buses)
    }


def tree_positions(
    network: PowerNetwork, root: str | None = None
) -> dict[str, tuple[float, float]]:
    """Layered positions for a radial network, root at the left.

    Depth from the root sets the horizontal position and siblings are
    stacked vertically, which is how a feeder one-line is always drawn.
    Falls back to :func:`circular_positions` on a meshed network, where
    there is no tree to lay out.
    """
    if not network.is_radial:
        return circular_positions(network.bus_ids)
    from difflow_power.flowsheet import feeder_tree

    tree = feeder_tree(network, root)
    depth = {tree.root: 0}
    for bus in tree.order[1:]:
        depth[bus] = depth[tree.parent[bus][0]] + 1
    by_depth: dict[int, list[str]] = {}
    for bus in tree.order:
        by_depth.setdefault(depth[bus], []).append(bus)
    return {
        bus: (float(d), float(i) - 0.5 * (len(row) - 1))
        for d, row in by_depth.items()
        for i, bus in enumerate(row)
    }


def _bus_role(network: PowerNetwork, bus_id: str) -> str:
    if network.buses[bus_id].kind == "slack":
        return "slack"
    if network.generators_at(bus_id):
        return "generator"
    if any(load.bus == bus_id for load in network.loads.values()):
        return "load"
    return "transit"


_ROLE_COLOR = {
    "slack": _SLACK,
    "generator": _GENERATOR,
    "load": _LOAD,
    "transit": _TRANSIT,
}


def draw_network(
    network: PowerNetwork,
    *,
    pos: Mapping[str, tuple[float, float]] | None = None,
    voltages: Mapping[str, float] | None = None,
    flows: Mapping[str, float] | None = None,
    loading: Mapping[str, float] | None = None,
    prices: Mapping[str, float] | None = None,
    highlight: Sequence[str] = (),
    ax=None,
    title: str | None = None,
):
    """Draw a network schematic.

    Args:
        network: the network to draw.
        pos: bus id -> ``(x, y)``. Defaults to a tree layout for a
            radial network and a circle otherwise; pass a real layout
            for anything of size.
        voltages: bus id -> magnitude (pu), annotated at the bus.
        flows: branch id -> real power (MW), annotated at the midpoint.
        loading: branch id -> fraction of rating. Branches above 1.0 are
            drawn in the critical colour.
        prices: bus id -> LMP ($/MWh), annotated under the bus.
        highlight: bus or branch ids to ring in the critical colour ---
            for marking a contingency, a violated limit, or whatever the
            figure is being made to show.
        ax: existing matplotlib axes; one is created if omitted.
        title: axes title; defaults to the network's name.

    Returns:
        The matplotlib axes.

    Example:
        >>> res = solve_power_flow(net)                   # doctest: +SKIP
        >>> draw_network(net, voltages=res.vm,
        ...              loading=res.branch_loading)      # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))
    if pos is None:
        pos = tree_positions(network)
    missing = set(network.bus_ids) - set(pos)
    if missing:
        raise KeyError(f"no position given for buses {sorted(missing)}")
    highlight = set(highlight)

    for aid, br in network.branches.items():
        x0, y0 = pos[br.from_bus]
        x1, y1 = pos[br.to_bus]
        over = loading is not None and loading.get(aid, 0.0) > 1.0
        color = _FLAG if (over or aid in highlight) else _LINE
        width = 2.4 if (over or aid in highlight) else 1.6
        style = (0, (5, 2)) if br.is_transformer else "-"
        ax.plot(
            [x0, x1], [y0, y1], linestyle=style, linewidth=width,
            color=color, zorder=1, solid_capstyle="round",
        )
        if br.is_transformer:
            ax.plot(
                [(x0 + x1) / 2], [(y0 + y1) / 2], marker="o",
                markersize=8, markerfacecolor="white",
                markeredgecolor=color, zorder=2,
            )
        labels = []
        if flows is not None and aid in flows:
            labels.append(f"{flows[aid]:.1f} MW")
        if loading is not None and aid in loading:
            labels.append(f"{100 * loading[aid]:.0f}%")
        if labels:
            ax.annotate(
                "\n".join(labels),
                ((x0 + x1) / 2, (y0 + y1) / 2),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7,
                color=_FLAG if over else _INK_SOFT, zorder=4,
            )

    for bid in network.bus_ids:
        x, y = pos[bid]
        role = _bus_role(network, bid)
        ax.plot(
            [x], [y], marker="o", markersize=13,
            markerfacecolor=_ROLE_COLOR[role],
            markeredgecolor=_FLAG if bid in highlight else _INK,
            markeredgewidth=2.0 if bid in highlight else 0.9, zorder=3,
        )
        ax.annotate(
            bid, (x, y), textcoords="offset points", xytext=(0, 11),
            ha="center", fontsize=8, color=_INK, zorder=4,
        )
        below = []
        if voltages is not None and bid in voltages:
            below.append(f"{voltages[bid]:.3f} pu")
        if prices is not None and bid in prices:
            below.append(f"${prices[bid]:.1f}")
        if below:
            ax.annotate(
                "  ".join(below), (x, y),
                textcoords="offset points", xytext=(0, -17),
                ha="center", fontsize=7, color=_INK_SOFT, zorder=4,
            )

    ax.set_title(title or (network.name or "network"), fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.18)
    return ax


def draw_legend(ax):
    """Add a bus-role legend to an axes drawn by :func:`draw_network`."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=9,
            markerfacecolor=_ROLE_COLOR[role], markeredgecolor=_INK,
            label=role,
        )
        for role in ("slack", "generator", "load", "transit")
    ] + [
        Line2D([], [], color=_LINE, linewidth=1.6, label="line"),
        Line2D(
            [], [], color=_LINE, linewidth=1.6, linestyle=(0, (5, 2)),
            label="transformer",
        ),
        Line2D([], [], color=_FLAG, linewidth=2.4, label="over rating"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False)
    return ax
