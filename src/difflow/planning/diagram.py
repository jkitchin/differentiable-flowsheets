"""Drawings for a delta-base plan: the flowsheet, the model, the region.

Five entry points, each a schematic rather than a chart --- they carry
identity and structure, and print magnitudes as text so that a reader
takes the numbers from the numbers:

* :func:`draw_chain` --- the reference two-plant chain as a process
  flow diagram: units, streams, the levers the plan may move, the
  priced streams and the constrained one.
* :func:`draw_planning_network` --- the same chain as the *planner*
  sees it: blocks, free decisions, links, priced outputs.  Works for
  any :class:`~difflow.planning.network.Network`.
* :func:`draw_delta_vectors` --- a delta vector as the matrix it is.
* :func:`draw_taylor_model` --- what a delta vector predicts along one
  decision, against what the nonlinear block actually does, inside and
  outside the trust region.
* :func:`draw_trust_region` --- the accept/reject/shrink loop, as boxes
  walking across the merit surface.

matplotlib is imported inside the functions, so importing this module
costs nothing in a headless run.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Light-surface slots of the difflow documentation palette.  Roles are
# an identity encoding: assigned in fixed order, never cycled.
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_LINE = "#8a8985"
_UNIT = "#eef2f7"        # unit-operation fill
_BLOCK = "#2a78d6"       # planning-block outline
_LEVER = "#eb6834"       # a decision the plan may move
_REVENUE = "#1f7a4d"     # a priced output worth money
_COST = "#b0651f"        # a priced output that costs money
_FLAG = "#d03b3b"        # a constrained variable, or a rejected step


# ----------------------------------------------------------------------
# small drawing primitives
# ----------------------------------------------------------------------

def _box(ax, xy, w, h, label, *, fc=_UNIT, ec=_INK_SOFT, fontsize=8.5,
         lw=1.2, zorder=2):
    """A rounded unit box with a centred label."""
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=zorder))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=_INK, zorder=zorder + 1)
    return (x + w / 2, y + h / 2)


def _stream(ax, points, *, color=_LINE, lw=1.4, arrow=True, ls="-",
            zorder=1):
    """A poly-line stream, arrowhead on the last segment."""
    pts = np.asarray(points, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, linestyle=ls,
            solid_capstyle="round", zorder=zorder)
    if arrow:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=zorder,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    shrinkA=0, shrinkB=0))


def _label(ax, xy, text, *, color=_INK_SOFT, fontsize=7.5, ha="center",
           va="center", box=False, weight="normal"):
    """A stream or annotation label, optionally on an opaque patch."""
    kw = dict(ha=ha, va=va, fontsize=fontsize, color=color, zorder=5,
              fontweight=weight)
    if box:
        kw["bbox"] = dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85)
    ax.text(xy[0], xy[1], text, **kw)


def _lever(ax, xy, text, *, dy=0.62, fontsize=7.5, va="bottom", ha=None,
           dx=0.0):
    """A decision marker: the plan is allowed to move this.

    ``ha="left"`` or ``"right"`` puts the label beside the lever head
    instead of above it, which is how crowded corners stay legible.
    """
    x, y = xy
    y_tip = y + (dy if va == "bottom" else -dy)
    ax.annotate("", xy=(x, y), xytext=(x, y_tip), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=_LEVER, lw=1.3))
    ax.plot([x], [y_tip], marker="D", ms=5.0, color=_LEVER, zorder=5)
    if ha in ("left", "right"):
        off = 0.16 if ha == "left" else -0.16
        _label(ax, (x + off + dx, y_tip), text, color=_LEVER,
               fontsize=fontsize, va="center", ha=ha, box=True)
    else:
        _label(ax, (x + dx, y_tip + (0.16 if va == "bottom" else -0.16)),
               text, color=_LEVER, fontsize=fontsize, va=va, box=True)


def _num(values, key, fmt="{:.4g}"):
    """Format a value from a state dict, or nothing when absent."""
    if not values or key not in values:
        return ""
    return fmt.format(float(values[key]))


def _price_tag(price):
    """A priced-output tag and the colour that goes with its sign."""
    if price is None:
        return "", _INK_SOFT
    return (f"${price:g}" if price >= 0 else f"-${abs(price):g}",
            _REVENUE if price >= 0 else _COST)


# ----------------------------------------------------------------------
# the reference chain, as a process flow diagram
# ----------------------------------------------------------------------

def draw_chain(values: Mapping[str, float] | None = None, *,
               prices: Mapping[str, float] | None = None,
               specs: Sequence[Any] = (),
               ax=None, title: str | None = None,
               show_blocks: bool = True, legend: bool = True):
    """Draw the two-plant reference chain as a process flow diagram.

    The picture carries the four things a reader needs before any
    numbers make sense: which units there are, which streams connect
    them, which quantities the plan is allowed to *move* (drawn as
    orange levers), and which are priced or constrained.

    Args:
        values: Optional ``{qualified name: value}`` --- typically
            ``net.evaluate(u).as_dict()`` or ``result.values`` --- whose
            entries are printed on the streams they belong to.  Without
            it the diagram is drawn unannotated.
        prices: ``{qualified name: price}``, printed as a tag on the
            priced stream.  Defaults to
            :data:`difflow.planning.chain.PRICES` in qualified form.
        specs: Specs to flag on the diagram; a single-variable
            ``<=``/``>=`` spec is drawn against its stream.
        ax: Axes to draw on; a new figure is made if omitted.
        title: Axes title.
        show_blocks: Outline the two planning blocks over the units, so
            the flowsheet and the LP's submodels line up.
        legend: Draw the marker legend.

    Returns:
        The axes drawn on.

    Example:
        >>> from difflow.planning import chain, draw_chain
        >>> problem = chain.two_plant_chain()
        >>> state = problem.network.evaluate(problem.network.decision_start())
        >>> ax = draw_chain(state.as_dict())        # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from difflow.planning import chain as chain_mod

    if ax is None:
        _, ax = plt.subplots(figsize=(13.5, 6.0))
    if prices is None:
        prices = {f"ngl.{k}": v for k, v in chain_mod.PRICES.items()
                  if k in ("NGL_C2", "NGL_C3plus", "E_refrig")}
        prices.update({f"power.{k}": chain_mod.PRICES[k]
                       for k in ("Power", "gas_sold", "CO2")})
    v = dict(values or {})

    spec_text: dict[str, str] = {}
    for s in specs:
        coeffs = getattr(s, "coeffs", None)
        if coeffs and len(coeffs) == 1:
            name = next(iter(coeffs))
            spec_text[name] = f"{s.op} {s.rhs:g}"

    def val(key, fmt="{:.4g}", unit=""):
        text = _num(v, key, fmt)
        return f" = {text}{unit}" if text else ""

    # -- feed and expander --------------------------------------------
    feed_total = sum(chain_mod.FEED.values())
    _stream(ax, [(-1.0, 6.0), (1.95, 6.0)])
    _label(ax, (-1.0, 6.3), f"feed gas  {feed_total:g} mol/s", weight="bold",
           ha="left")
    _label(ax, (-1.0, 6.05), "C1 75 / C2 12 / C3 8 / nC4 5", fontsize=6.6,
           ha="left")

    _box(ax, (1.95, 5.55), 0.85, 0.9, "turbo-\nexpander", fontsize=6.6)
    _lever(ax, (2.375, 6.45),
           f"P_expander{val('ngl.P_expander', '{:.3g}', ' Pa')}"
           "\n[2.5, 4.0] MPa", dy=0.75, dx=-0.35)
    _stream(ax, [(2.8, 6.0), (3.05, 6.0)])

    # -- cold box ------------------------------------------------------
    _box(ax, (3.05, 5.1), 1.75, 1.8, "cold box\nflash")
    _lever(ax, (3.9, 6.9), f"T_coldbox{val('ngl.T_coldbox', '{:.1f}', ' K')}"
           "\n[218, 244] K", dy=1.45, dx=0.15)

    # refrigeration duty leaves as a cost
    _stream(ax, [(3.9, 5.1), (3.9, 4.2)], color=_COST)
    tag, colour = _price_tag(prices.get("ngl.E_refrig"))
    _label(ax, (3.9, 3.95), f"E_refrig{val('ngl.E_refrig', '{:.2f}', ' MW')}",
           color=colour)
    _label(ax, (3.9, 3.68), f"{tag}/MW  refrigeration", color=colour,
           fontsize=6.8)

    # -- vapour up, liquid across --------------------------------------
    _stream(ax, [(4.8, 6.5), (5.05, 6.5), (5.05, 8.0), (5.75, 8.0)])
    _label(ax, (5.15, 7.35), "vapour", fontsize=7.0, ha="left")
    _stream(ax, [(4.8, 5.5), (5.75, 5.5)])
    _label(ax, (5.28, 5.25), "liquid", fontsize=7.0)

    # reflux: a split of the vapour is contacted back over the liquid
    ax.plot([5.05], [8.0], marker="o", ms=5.5, color=_INK_SOFT, zorder=4)
    _stream(ax, [(5.05, 8.0), (5.4, 8.0), (5.4, 6.4), (5.75, 6.4)],
            ls=(0, (5, 2)))
    _label(ax, (5.5, 6.7), "reflux", fontsize=6.8, ha="left")
    _lever(ax, (5.05, 8.05), f"split{val('ngl.split', '{:.2f}')}   [0, 1]",
           dy=0.6, ha="left")

    _box(ax, (5.75, 4.9), 1.7, 2.0, "reflux\ncontactor")
    _label(ax, (6.5, 4.62), "heavies leave the residue gas", fontsize=6.6)

    # -- residue gas ---------------------------------------------------
    _stream(ax, [(5.75, 8.0), (12.0, 8.0)])
    _label(ax, (6.35, 7.73), "residue gas", fontsize=7.0)

    # -- deethanizer ---------------------------------------------------
    _stream(ax, [(7.45, 5.9), (8.0, 5.9)])
    flag = spec_text.get("ngl.T_colfeed")
    _label(ax, (7.7, 6.15), "NGL liquid", fontsize=7.0)
    tcol = _num(v, "ngl.T_colfeed", "{:.1f}")
    if flag or tcol:
        text = "T_colfeed" + (f" = {tcol} K" if tcol else "")
        colour = _FLAG if flag else _INK_SOFT
        ax.plot([7.72, 7.72], [4.3, 5.85], color=colour, lw=0.8,
                linestyle=(0, (2, 2)), zorder=3)
        _label(ax, (7.72, 4.05), text + (f"\n{flag} K  (spec)" if flag else ""),
               color=colour, fontsize=7.0, box=True)

    _box(ax, (8.0, 4.7), 2.0, 2.1, "deethanizer\n(C2 recovery\nvs rejection)")
    _lever(ax, (8.55, 6.8),
           f"ethane_recovery{val('ngl.ethane_recovery', '{:.2f}')}"
           "\n[0.30, 0.98]", dy=0.45)

    # rejected ethane joins the residue gas
    _stream(ax, [(9.55, 6.8), (9.55, 8.0)])
    _label(ax, (9.68, 7.3), "rejected C2", fontsize=6.8, ha="left")

    # products leave downwards, clear of the power plant
    for y, key in ((3.55, "ngl.NGL_C2"), (2.85, "ngl.NGL_C3plus")):
        unit = chain_mod.UNITS[key.split(".", 1)[1]]
        x_drop = 8.45 if key.endswith("C2") else 9.05
        _stream(ax, [(x_drop, 4.7), (x_drop, y), (11.75, y)],
                color=_REVENUE)
        tag, colour = _price_tag(prices.get(key))
        short = key.split(".", 1)[1]
        _label(ax, (11.95, y), f"{short}{val(key, '{:.3g}', ' ' + unit)}"
               f"    {tag}", color=colour, ha="left", fontsize=7.4)

    # -- residue header and the allocation lever -----------------------
    ax.plot([12.0], [8.0], marker="o", ms=5.5, color=_INK_SOFT, zorder=4)
    _label(ax, (11.85, 8.62),
           f"residue_F{val('ngl.residue_F', '{:.3g}', ' mol/s')}",
           fontsize=7.6, weight="bold", color=_BLOCK, ha="right")
    _label(ax, (11.85, 8.35), "the link:  ngl.residue_F -> power.fuel_F",
           fontsize=6.6, color=_BLOCK, ha="right")
    _lever(ax, (12.0, 8.0), f"alloc{val('power.alloc', '{:.2f}')}   [0, 1]"
           "\nburn or sell", dy=0.62, ha="left")

    _stream(ax, [(12.0, 8.0), (14.7, 8.0)], color=_REVENUE)
    tag, colour = _price_tag(prices.get("power.gas_sold"))
    sold = val("power.gas_sold", "{:.3g}", " " + chain_mod.UNITS["gas_sold"])
    _label(ax, (14.8, 8.0), f"gas_sold{sold}    {tag}",
           color=colour, ha="left", fontsize=7.4)

    _stream(ax, [(12.0, 8.0), (12.0, 6.3), (12.75, 6.3)])
    _label(ax, (12.25, 7.2), "burned", fontsize=6.8, ha="left")

    # -- gas turbine ---------------------------------------------------
    _box(ax, (12.75, 5.3), 1.9, 1.9, "gas turbine\n(part-load\nefficiency)")
    for y, key in ((6.6, "power.Power"), (5.7, "power.CO2")):
        unit = chain_mod.UNITS[key.split(".", 1)[1]]
        colour = _REVENUE if prices.get(key, 0.0) >= 0 else _COST
        _stream(ax, [(14.65, y), (15.25, y)], color=colour)
        tag, colour = _price_tag(prices.get(key))
        short = key.split(".", 1)[1]
        _label(ax, (15.35, y), f"{short}{val(key, '{:.3g}', ' ' + unit)}"
               f"    {tag}", color=colour, ha="left", fontsize=7.4)

    # -- the planning blocks over the units ----------------------------
    if show_blocks:
        for (x, y, w, h, label) in (
                (2.2, 2.45, 9.6, 5.7, "block  ngl   —   4 decisions, "
                 "5 outputs,   J is 5x4"),
                (11.9, 4.95, 3.3, 3.2, "block  power   —   2 inputs "
                 "(one of them the link),   3 outputs,   J is 3x2")):
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor=_BLOCK, linewidth=1.0,
                                   linestyle=(0, (6, 3)), zorder=0))
            _label(ax, (x + 0.12, y + 0.18), label, color=_BLOCK, ha="left",
                   fontsize=7.2, weight="bold")

    if legend:
        handles = [
            Line2D([], [], marker="D", color=_LEVER, ls="none", ms=6,
                   label="decision the plan may move"),
            Line2D([], [], color=_REVENUE, lw=2, label="priced (revenue)"),
            Line2D([], [], color=_COST, lw=2, label="priced (cost)"),
            Line2D([], [], color=_BLOCK, lw=1.2, ls=(0, (6, 3)),
                   label="planning block (one delta vector)"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=7.2,
                  frameon=False, ncol=4, bbox_to_anchor=(0.0, -0.03))

    ax.set_xlim(-1.15, 18.3)
    ax.set_ylim(2.0, 9.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, color=_INK, loc="left")
    return ax


# ----------------------------------------------------------------------
# the network as the planner sees it
# ----------------------------------------------------------------------

def draw_planning_network(network, *, prices: Mapping[str, float] | None = None,
                          specs: Sequence[Any] = (),
                          values: Mapping[str, float] | None = None,
                          ax=None, title: str | None = None):
    """Draw a planning network: blocks, decisions, links, priced outputs.

    This is the flowsheet stripped to what the LP actually contains ---
    one box per block, the free decisions entering on the left, the
    outputs leaving on the right, and the links as arrows between
    boxes.  Works for any :class:`~difflow.planning.network.Network`,
    so it is worth drawing whenever a network is bigger than the two
    blocks a reader can hold in their head.

    Args:
        network: The network to draw.
        prices: ``{qualified name: price}``; priced outputs are tagged
            and coloured by sign.
        specs: Specs; a single-variable spec is flagged on its output.
        values: Optional ``{qualified name: value}`` to print.
        ax: Axes to draw on; a new figure is made if omitted.
        title: Axes title.

    Returns:
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    prices = dict(prices or {})
    v = dict(values or {})
    spec_of: dict[str, str] = {}
    for s in specs:
        coeffs = getattr(s, "coeffs", None)
        if coeffs and len(coeffs) == 1:
            spec_of[next(iter(coeffs))] = f"{s.op} {s.rhs:g}"

    order = network.order
    depth: dict[str, int] = {name: 0 for name in order}
    for link in network.links:
        depth[link.target_block] = max(depth[link.target_block],
                                       depth[link.source_block] + 1)

    rows = max([3] + [max(len(b.u_names), len(b.y_names))
                      for b in network.blocks])
    dy, x_step = 0.62, 10.0
    height = rows * dy + 0.9
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6 + x_step * (max(depth.values()) + 1)
                                      / 2.6, 1.4 + height / 1.5))

    centres: dict[str, tuple[float, float]] = {}
    port_y: dict[str, float] = {}
    for name in order:
        b = network.block(name)
        x = 1.9 + x_step * depth[name]
        y = 0.0
        _box(ax, (x, y), 2.3, height,
             f"{name}\n\n{len(b.u_names)} in, {len(b.y_names)} out\n"
             f"J is {len(b.y_names)}x{len(b.u_names)}", fontsize=8.0)
        centres[name] = (x, y)
        free = [u for u in b.u_names if not network.is_linked(f"{name}.{u}")]
        for j, u in enumerate(b.u_names):
            qname = f"{name}.{u}"
            yy = y + height - 0.55 - j * dy
            port_y[qname] = yy
            if network.is_linked(qname):
                continue
            if depth[name] == 0:
                # room to the left: levers enter on the inlet face
                _stream(ax, [(x - 1.55, yy), (x, yy)], color=_LEVER, lw=1.2)
                ax.plot([x - 1.62], [yy], marker="D", ms=4.5, color=_LEVER)
                _label(ax, (x - 1.75, yy), u + _num_suffix(v, qname),
                       color=_LEVER, ha="right", fontsize=7.2)
            else:
                # the inlet face carries a link, so levers come in on top
                k = free.index(u)
                xx = x + 2.3 * (k + 1) / (len(free) + 1)
                _lever(ax, (xx, y + height), u + _num_suffix(v, qname),
                       dy=0.45 + 0.35 * k, fontsize=7.2)
        for i, yv in enumerate(b.y_names):
            qname = f"{name}.{yv}"
            yy = y + height - 0.55 - i * dy
            port_y[qname] = yy
            if any(l.source == qname for l in network.links):
                continue
            price = prices.get(qname)
            colour = _INK_SOFT if price is None else (
                _REVENUE if price >= 0 else _COST)
            _stream(ax, [(x + 2.3, yy), (x + 2.9, yy)], color=colour, lw=1.2)
            tag, _ = _price_tag(price)
            flag = spec_of.get(qname)
            text = yv + _num_suffix(v, qname) + (f"   {tag}" if tag else "")
            _label(ax, (x + 3.0, yy), text, color=colour, ha="left",
                   fontsize=7.2)
            if flag:
                _label(ax, (x + 3.0, yy - 0.26), f"spec {flag}", color=_FLAG,
                       ha="left", fontsize=6.8)

    for link in network.links:
        xs = centres[link.source_block][0] + 2.3
        xt = centres[link.target_block][0]
        ys, yt = port_y[link.source], port_y[link.target]
        lane = -0.75
        _stream(ax, [(xs, ys), (xs + 0.35, ys), (xs + 0.35, lane),
                     (xt - 0.35, lane), (xt - 0.35, yt), (xt, yt)],
                color=_BLOCK, lw=1.6)
        _label(ax, ((xs + xt) / 2, lane + 0.22),
               f"link   {link.source} -> {link.target}"
               + _num_suffix(v, link.source),
               color=_BLOCK, fontsize=7.2)

    ax.set_xlim(-1.4, 1.9 + x_step * (max(depth.values()) + 1) + 1.6)
    ax.set_ylim(-1.4, height + 0.9)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, color=_INK, loc="left")
    return ax


def _num_suffix(values, key, fmt="{:.4g}"):
    text = _num(values, key, fmt)
    return f" = {text}" if text else ""


# ----------------------------------------------------------------------
# the delta vector as a model
# ----------------------------------------------------------------------

def draw_delta_vectors(lin, u_names: Sequence[str] | None = None,
                       y_names: Sequence[str] | None = None, *,
                       block=None, ax=None, title: str | None = None):
    """Draw one block's delta vectors as the matrix they are.

    Rows are outputs, columns are decisions, and cell ``(i, j)`` is
    ``dy_i/du_j`` at the linearisation point.  Units differ wildly down
    the rows --- mol/s against K against MW --- so a shared colour scale
    would say nothing.  Each row is therefore coloured by its own
    largest entry, which answers the question a planner actually asks
    ("which lever moves this output?"), and the printed number carries
    the magnitude.

    Args:
        lin: A :class:`~difflow.planning.linearize.Linearization`.
        u_names: Column names; taken from ``block`` when omitted.
        y_names: Row names; taken from ``block`` when omitted.
        block: Optional :class:`~difflow.planning.block.Block` to take
            names from.
        ax: Axes to draw on; a new figure is made if omitted.
        title: Axes title.  Defaults to the block name and AD mode.

    Returns:
        The axes drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if block is not None:
        u_names = u_names or block.u_names
        y_names = y_names or block.y_names
    J = np.asarray(lin.J, dtype=float)
    n_y, n_u = J.shape
    u_names = list(u_names or [f"u{j}" for j in range(n_u)])
    y_names = list(y_names or [f"y{i}" for i in range(n_y)])

    if ax is None:
        _, ax = plt.subplots(figsize=(1.55 * n_u + 2.4, 0.62 * n_y + 1.9))

    scale = np.max(np.abs(J), axis=1, keepdims=True)
    scale[scale == 0.0] = 1.0
    rel = J / scale
    cmap = LinearSegmentedColormap.from_list(
        "delta", [_COST, "#f7f2ec", "#ffffff", "#eaf1f8", _BLOCK])
    ax.imshow(rel, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    for i in range(n_y):
        for j in range(n_u):
            ax.text(j, i, f"{J[i, j]:.3g}", ha="center", va="center",
                    fontsize=8.0,
                    color=_INK if abs(rel[i, j]) < 0.75 else "white")
    ax.set_xticks(range(n_u))
    ax.set_xticklabels(u_names, fontsize=8, rotation=18, ha="right")
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(y_names, fontsize=8)
    ax.set_xlabel("decision  u", fontsize=8.5, color=_INK_SOFT)
    ax.set_ylabel("output  y", fontsize=8.5, color=_INK_SOFT)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title if title is not None else
                 f"delta vectors  J = dy/du  for block '{lin.block}'   "
                 f"(one {lin.mode}-mode AD pass; shaded within each row)",
                 fontsize=9.5, color=_INK, loc="left")
    return ax


def draw_taylor_model(block, variable: str, output: str, *,
                      u0=None, theta: Mapping[str, Any] | None = None,
                      radius: float = 0.25, span: float = 1.0,
                      n: int = 61, ax=None, title: str | None = None):
    """Draw what a delta vector predicts against what the block does.

    One decision is swept while the others are held at the
    linearisation point.  The straight line is the delta-vector model
    the LP is given, ``y ~= y0 + J (u - u0)``; the curve is the block.
    They agree to first order at ``u0`` and part company away from it,
    which is the entire reason a trust region exists: the shaded band
    is how far the LP is allowed to move this decision in one cycle.

    Args:
        block: The :class:`~difflow.planning.block.Block` to sweep.
        variable: Name of the decision to sweep, from ``block.u_names``.
        output: Name of the output to plot, from ``block.y_names``.
        u0: Linearisation point; ``block.u0`` when omitted.
        theta: Parameter override passed to the block.
        radius: Trust-region radius as a fraction of the bound range.
        span: Fraction of the bound range to sweep, 1.0 for all of it.
        n: Number of sweep points.
        ax: Axes to draw on; a new figure is made if omitted.
        title: Axes title.

    Returns:
        The axes drawn on.
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from difflow.planning.linearize import linearize_block

    j = block.u_index(variable)
    i = block.y_index(output)
    u0 = jnp.asarray(block.u0 if u0 is None else u0, dtype=float)
    lb, ub = float(np.asarray(block.lb)[j]), float(np.asarray(block.ub)[j])
    lin = linearize_block(block, u0, theta)

    half = 0.5 * span * (ub - lb)
    lo = max(lb, float(u0[j]) - half)
    hi = min(ub, float(u0[j]) + half)
    # the linearisation point is put on the grid, so the two curves
    # touch on the figure exactly where the theory says they do
    grid = np.unique(np.append(np.linspace(lo, hi, n), float(u0[j])))

    truth, model = [], []
    for value in grid:
        u = u0.at[j].set(value)
        truth.append(float(block.evaluate(u, theta)[i]))
        model.append(float(lin.predict(u)[i]))
    truth, model = np.asarray(truth), np.asarray(model)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))

    step = radius * (ub - lb)
    ax.axvspan(max(lo, float(u0[j]) - step), min(hi, float(u0[j]) + step),
               color=_BLOCK, alpha=0.08, lw=0)
    ax.plot(grid, truth, color=_INK, lw=1.8,
            label=f"the block: {block.name}({variable})")
    ax.plot(grid, model, color=_LEVER, lw=1.6, ls=(0, (5, 2)),
            label=f"delta-vector model  y0 + J (u - u0)")
    ax.plot([float(u0[j])], [float(lin.y0[i])], marker="o", ms=6,
            color=_LEVER, zorder=5)
    _label(ax, (float(u0[j]), float(lin.y0[i])), "  linearisation point",
           color=_LEVER, ha="left", fontsize=8)

    def _gap(x_target, colour, note):
        k = int(np.argmin(np.abs(grid - x_target)))
        err = model[k] - truth[k]
        ax.annotate("", xy=(grid[k], truth[k]), xytext=(grid[k], model[k]),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=1.2))
        right = grid[k] > 0.5 * (grid[0] + grid[-1])
        _label(ax, (grid[k] + (-0.02 if right else 0.02) * (hi - lo),
                    0.5 * (truth[k] + model[k])),
               f"{note}: {err:+.3g}", color=colour,
               ha="right" if right else "left", fontsize=8, box=True)
        return err

    _gap(min(hi, float(u0[j]) + step), _BLOCK, "error at the trust-region edge")
    far = lo if (float(u0[j]) - lo) > (hi - float(u0[j])) else hi
    if abs(far - float(u0[j])) > 1.5 * step:
        _gap(far, _FLAG, "error where the LP is not allowed to go")
    _label(ax, (float(u0[j]), ax.get_ylim()[1]),
           f"trust region, radius {radius:g}", color=_BLOCK, fontsize=7.6,
           va="top")

    ax.set_xlabel(f"{variable}", fontsize=9)
    ax.set_ylabel(f"{output}", fontsize=9)
    ax.set_title(title if title is not None else
                 "A delta vector is a model, and it is only local",
                 fontsize=10, color=_INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=0.25)
    return ax


def draw_trust_region(result, *, decisions=(0, 1), grid: int = 31,
                      ax=None, title: str | None = None,
                      show_boxes: bool = True, levels: int = 14,
                      max_cycles: int = 8):
    """Draw the trust-region loop over the merit surface it walks.

    Two decisions are plotted against each other with the rest held at
    the run's starting point.  The contours are the *nonlinear* merit
    --- priced objective less the realised violation charge --- so the
    picture shows what the acceptance test actually judges each
    proposal against.  Each cycle is drawn as the box the LP was
    allowed to move in, with the proposal it returned: accepted
    proposals move the incumbent, rejected ones shrink the box.

    Evaluating the surface costs ``grid**2`` model runs, so keep
    ``grid`` modest on an expensive flowsheet.

    Args:
        result: A :class:`~difflow.planning.planner.PlanResult`.
        decisions: The two decisions to plot, by name or index.
        grid: Points per axis in the contour.
        ax: Axes to draw on; a new figure is made if omitted.
        title: Axes title.
        show_boxes: Draw the trust-region box of each cycle.
        levels: Number of contour levels.
        max_cycles: Draw only this many cycles.  The tail of a
            converged run is a pile of ever-smaller boxes on top of the
            solution, which hides the part worth looking at.

    Returns:
        The axes drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    planner, net = result.planner, result.network
    names = net.decision_names
    idx = [names.index(d) if isinstance(d, str) else int(d)
           for d in decisions]
    lb, ub = (np.asarray(a, dtype=float) for a in net.decision_bounds())
    base = np.asarray(result.start if result.start is not None
                      else result.decisions, dtype=float)

    gx = np.linspace(lb[idx[0]], ub[idx[0]], grid)
    gy = np.linspace(lb[idx[1]], ub[idx[1]], grid)
    z = np.empty((grid, grid))
    for a, xv in enumerate(gx):
        for b, yv in enumerate(gy):
            u = base.copy()
            u[idx[0]], u[idx[1]] = xv, yv
            z[b, a] = planner.score(u)["merit"]

    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.6))
    cs = ax.contourf(gx, gy, z, levels=levels, cmap="BuGn", alpha=0.6)
    ax.contour(gx, gy, z, levels=levels, colors="white", linewidths=0.5)
    cb = ax.figure.colorbar(cs, ax=ax)
    cb.set_label("merit from the nonlinear blocks", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    incumbent = base.copy()
    shown = result.history[:max_cycles]
    for it in shown:
        proposal = np.asarray(it.decisions, dtype=float)
        if show_boxes:
            half = it.radius * (ub - lb)
            x0 = max(lb[idx[0]], incumbent[idx[0]] - half[idx[0]])
            y0 = max(lb[idx[1]], incumbent[idx[1]] - half[idx[1]])
            x1 = min(ub[idx[0]], incumbent[idx[0]] + half[idx[0]])
            y1 = min(ub[idx[1]], incumbent[idx[1]] + half[idx[1]])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=_BLOCK if it.accepted else _FLAG,
                                   lw=1.0, linestyle=(0, (4, 2)), alpha=0.8))
        ax.annotate("", xy=(proposal[idx[0]], proposal[idx[1]]),
                    xytext=(incumbent[idx[0]], incumbent[idx[1]]),
                    arrowprops=dict(arrowstyle="->", lw=1.1,
                                    color=_INK if it.accepted else _FLAG))
        ax.plot([proposal[idx[0]]], [proposal[idx[1]]],
                marker="o" if it.accepted else "x", ms=6,
                color=_INK if it.accepted else _FLAG, zorder=5)
        _label(ax, (proposal[idx[0]], proposal[idx[1]]),
               f"  {it.index}: rho = {it.rho:.2f}",
               color=_INK_SOFT if it.accepted else _FLAG, fontsize=6.8,
               ha="left", box=True)
        if it.accepted:
            incumbent = proposal

    ax.plot([base[idx[0]]], [base[idx[1]]], marker="s", ms=7, color=_LEVER,
            zorder=6)
    ax.plot([result.decisions[idx[0]]], [result.decisions[idx[1]]],
            marker="*", ms=15, color=_LEVER, zorder=6)
    ax.set_xlabel(names[idx[0]], fontsize=9)
    ax.set_ylabel(names[idx[1]], fontsize=9)
    tail = ("" if len(shown) == len(result.history)
            else f"   (first {len(shown)} drawn)")
    ax.set_title(title if title is not None else
                 f"{result.n_iterations} trust-region cycles, "
                 f"{result.n_accepted} accepted{tail}",
                 fontsize=10, color=_INK, loc="left")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="none", color=_LEVER, label="start"),
        Line2D([], [], marker="o", ls="none", color=_INK,
               label="accepted proposal"),
        Line2D([], [], marker="x", ls="none", color=_FLAG,
               label="rejected: shrink and retry"),
        Line2D([], [], marker="*", ls="none", color=_LEVER, ms=11,
               label="plan"),
    ], fontsize=7.6, frameon=False, loc="best")
    return ax
