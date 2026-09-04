"""Two drawings: where the design runs out, and who is charging for it.

* :func:`draw_flexibility_region` --- the stated envelope, the envelope the
  design actually covers, and the vertex where it stops.  Two parameters at a
  time, because that is what a plane holds honestly.
* :func:`draw_penalty_split` --- the feed penalty and the parameter back-off
  as two stacked bars per constraint, so the reader sees which purchase order
  the margin is on.

matplotlib is imported inside the functions, so importing this module costs
nothing in a headless run.
"""

from __future__ import annotations

import numpy as np

# Light-surface slots of the difflow documentation palette, matching
# difflow.planning.diagram.  Roles are an identity encoding, never cycled.
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"
_LINE = "#8a8985"
_STATED = "#2a78d6"      # the envelope you were given
_COVERED = "#1f7a4d"     # the envelope the design actually covers
_FLAG = "#d03b3b"        # the binding vertex / constraint
_FEED = "#eb6834"        # feed penalty: bought down with controls
_PARAM = "#7a5cc0"       # parameter back-off: bought down with experiments


def draw_flexibility_region(result, axes=(0, 1), ax=None):
    """Draw the stated envelope against the one the design covers.

    Args:
        result: A :class:`~difflow.flexibility.index.FlexibilityResult`.
        axes: Which two parameters to plot, as indices into the set.
        ax: An existing matplotlib axes, or ``None`` to make one.

    Returns:
        The axes drawn on.

    Raises:
        ValueError: If the set has fewer than two parameters.

    Example:
        >>> ax = draw_flexibility_region(res)             # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    T = result.set
    if T.n < 2:
        raise ValueError(
            "a flexibility region needs two parameters to plot; this set has "
            f"{T.n}. Read result.summary() instead.")
    i, j = int(axes[0]), int(axes[1])
    nom = np.asarray(T.nominal, dtype=float)
    lo = np.asarray(T.lower, dtype=float)
    up = np.asarray(T.upper, dtype=float)
    F = float(result.index)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))

    for scale, color, label, lw in ((1.0, _STATED, "stated envelope", 1.6),
                                    (F, _COVERED, f"covered (F = {F:.3g})",
                                     2.0)):
        ax.add_patch(Rectangle(
            (nom[i] - scale * lo[i], nom[j] - scale * lo[j]),
            scale * (lo[i] + up[i]), scale * (lo[j] + up[j]),
            fill=False, edgecolor=color, linewidth=lw,
            linestyle="--" if scale == 1.0 else "-", label=label, zorder=2))

    ax.plot([nom[i]], [nom[j]], "o", color=_INK, ms=5, zorder=4,
            label="nominal")
    crit = np.asarray(result.critical_theta, dtype=float)
    ax.plot([crit[i]], [crit[j]], "X", color=_FLAG, ms=11, zorder=5,
            label=f"binds: {result.binding_constraint}")

    verts = np.asarray(T.vertices(F), dtype=float)
    lim = np.asarray(result.vertex_limits, dtype=float)
    for v in range(verts.shape[0]):
        if v == result.limited_by_vertex:
            continue
        ax.annotate(f"{lim[v]:.2g}", (verts[v, i], verts[v, j]),
                    fontsize=7.5, color=_INK_SOFT,
                    ha="center", va="center", zorder=3)

    ax.set_xlabel(T.names[i])
    ax.set_ylabel(T.names[j])
    ax.set_title("flexibility region", fontsize=10, color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(_LINE)
    return ax


def draw_penalty_split(report, ax=None):
    """Draw the feed penalty and the parameter back-off per constraint.

    Args:
        report: A :class:`~difflow.flexibility.penalties.PenaltyReport`.
        ax: An existing matplotlib axes, or ``None`` to make one.

    Returns:
        The axes drawn on.

    Example:
        >>> ax = draw_penalty_split(report)               # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    names = list(report.constraint_names)
    feed = np.clip(np.asarray(report.feed_penalty, dtype=float), 0.0, None)
    param = np.asarray(report.backoff, dtype=float)
    y = np.arange(len(names))

    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 0.5 * len(names) + 1.8))

    ax.barh(y, feed, color=_FEED, label="feed (controls can respond)")
    ax.barh(y, param, left=feed, color=_PARAM,
            label="parameter (they cannot)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("margin required, in constraint units")
    ax.set_title(f"uncertainty penalties (kappa = {report.kappa:g})",
                 fontsize=10, color=_INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(_LINE)
    return ax
