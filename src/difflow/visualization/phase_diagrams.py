"""Matplotlib plot helpers for the phase-diagram data of :mod:`difflow.phase_diagrams`.

matplotlib is an optional dependency and is imported when a helper is called.
Colour is never the only cue: the bubble and dew curves differ in line style
and marker (solid with circles vs dashed with squares), the 45 degree line in
the xy plot is dotted, and tie lines are thin grey solid lines.
"""

from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("matplotlib is required for phase-diagram plots: "
                          "pip install matplotlib") from e
    return plt


def _ax(ax):
    if ax is None:
        _, ax = _plt().subplots(figsize=(5, 4))
    return ax


def _label(data) -> str:
    return data["species"][0] if "species" in data else "component 1"


def plot_txy(data: dict, ax=None):
    """Plot a Txy diagram from :func:`difflow.phase_diagrams.txy`.

    The bubble curve ``T(x)`` is solid with circle markers, the dew curve
    ``T(y)`` dashed with square markers.

    Args:
        data: Result of ``txy``.
        ax: Optional matplotlib axes.

    Returns:
        The matplotlib axes.
    """
    ax = _ax(ax)
    ax.plot(np.asarray(data["x"]), np.asarray(data["T"]), "-o", color="k",
            ms=3, label="bubble (liquid, x)")
    ax.plot(np.asarray(data["y"]), np.asarray(data["T"]), "--s", color="tab:blue",
            ms=3, label="dew (vapor, y)")
    ax.set_xlabel(f"mole fraction {_label(data)}")
    ax.set_ylabel("T (K)")
    ax.set_xlim(0, 1)
    ax.legend()
    return ax


def plot_pxy(data: dict, ax=None):
    """Plot a Pxy diagram from :func:`difflow.phase_diagrams.pxy` (bubble solid, dew dashed)."""
    ax = _ax(ax)
    P = np.asarray(data["P"]) / 1e3
    ax.plot(np.asarray(data["x"]), P, "-o", color="k", ms=3, label="bubble (liquid, x)")
    ax.plot(np.asarray(data["y"]), P, "--s", color="tab:blue", ms=3, label="dew (vapor, y)")
    ax.set_xlabel(f"mole fraction {_label(data)}")
    ax.set_ylabel("P (kPa)")
    ax.set_xlim(0, 1)
    ax.legend()
    return ax


def plot_xy(data: dict, ax=None):
    """Plot y vs. x with the dotted 45 degree line (``xy_curve`` or ``txy``/``pxy`` data)."""
    ax = _ax(ax)
    ax.plot([0, 1], [0, 1], ":", color="0.4", label="y = x")
    ax.plot(np.asarray(data["x"]), np.asarray(data["y"]), "-o", color="k", ms=3,
            label="equilibrium")
    ax.set_xlabel(f"liquid mole fraction {_label(data)}")
    ax.set_ylabel(f"vapor mole fraction {_label(data)}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()
    return ax


def ternary_xy(comp, coords: str = "equilateral") -> np.ndarray:
    """Map ternary compositions ``(..., 3)`` to 2-D plot coordinates.

    Species 0 sits at the bottom-left, species 1 at the bottom-right and
    species 2 at the top (equilateral), or at the top of the left edge
    (right-triangle, species 2 on the vertical axis, species 1 on the
    horizontal one).
    """
    c = np.asarray(comp, dtype=float)
    if coords == "equilateral":
        return np.stack([c[..., 1] + 0.5 * c[..., 2], (np.sqrt(3.0) / 2.0) * c[..., 2]], axis=-1)
    if coords == "right":
        return np.stack([c[..., 1], c[..., 2]], axis=-1)
    raise ValueError("coords must be 'equilateral' or 'right'")


def plot_ternary(data: dict, ax=None, coords: str = "equilateral",
                 labels: tuple[str, str, str] = ("1", "2", "3")):
    """Plot a ternary LLE diagram from :func:`difflow.phase_diagrams.ternary_lle`.

    Draws the triangle, the binodal (thick solid line), the tie lines (thin
    grey solid lines) and the plait point (a diamond marker).

    Args:
        data: Result of ``ternary_lle``.
        ax: Optional matplotlib axes.
        coords: ``"equilateral"`` or ``"right"`` (right triangle).
        labels: Vertex labels for species 0, 1, 2.

    Returns:
        The matplotlib axes.
    """
    ax = _ax(ax)
    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    ax.plot(*ternary_xy(verts, coords).T, "-", color="0.3", lw=1)
    for lab, v in zip(labels, verts[:3]):
        p = ternary_xy(v, coords)
        ax.annotate(lab, p, textcoords="offset points", xytext=(0, 6), ha="center")
    tl = np.asarray(data["tie_lines"])
    for t in tl:
        xy = ternary_xy(t, coords)
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.6", lw=0.8)
    b = np.asarray(data["binodal"])
    if len(b):
        xy = ternary_xy(b, coords)
        ax.plot(xy[:, 0], xy[:, 1], "-", color="k", lw=2, label="binodal")
        p = ternary_xy(np.asarray(data["plait"]), coords)
        ax.plot([p[0]], [p[1]], "D", color="k", mfc="w", label="plait point")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left")
    return ax


def plot_mccabe_thiele(result: dict, curve, top_line=None, bottom_line=None,
                       q_line=None, ax=None, number_stages: bool = True):
    """Plot a McCabe-Thiele diagram from :func:`difflow.staged.mccabe_thiele`.

    Draws the equilibrium curve (solid, black), the 45 degree line (dotted),
    the rectifying/top line (dashed), the stripping/bottom line
    (dash-dotted), the q-line (long dashes, grey), the staircase (thin
    solid, blue) with the stages numbered, and the feed stage as a larger
    square marker. Colour is never the only cue.

    Args:
        result: Output of ``mccabe_thiele``.
        curve: The :class:`~difflow.staged.EquilibriumCurve` stepped on.
        top_line, bottom_line, q_line: Optional ``(a, b, c)`` lines to draw.
        ax: Optional matplotlib axes.
        number_stages: Number each stage at its equilibrium point.

    Returns:
        The matplotlib axes.
    """
    import jax.numpy as jnp
    from difflow import staged

    ax = _ax(ax)
    credit = np.asarray(result["credit"])
    n = int(np.sum(credit > 0))
    xs = np.asarray(result["x"])
    ys = np.asarray(result["y"])
    xmax = float(max(1.0, np.max(xs[: n + 1]))) if n else 1.0
    grid = jnp.linspace(0.0, xmax, 201)
    ax.plot(np.asarray(grid), np.asarray(curve.ideal()(grid)), "-", color="k", label="equilibrium")
    ax.plot([0, xmax], [0, xmax], ":", color="0.4", label="y = x")

    def draw(line, style, color, label):
        if line is None:
            return
        line = np.asarray(line)
        if abs(line[1]) < 1e-14:     # vertical
            ax.plot([line[2] / line[0]] * 2, [0, xmax], linestyle=style, color=color, label=label)
        else:
            g = np.linspace(0.0, xmax, 2)
            ax.plot(g, (line[2] - line[0] * g) / line[1], linestyle=style, color=color, label=label)

    draw(top_line, "--", "tab:red", "top operating line")
    draw(bottom_line, "-.", "tab:green", "bottom operating line")
    draw(q_line, (0, (8, 3)), "0.5", "q-line")
    xv = np.asarray(result["x_vertices"])[: 3 * n]
    yv = np.asarray(result["y_vertices"])[: 3 * n]
    x_path = np.concatenate([[xs[0]], xv])
    y_path = np.concatenate([[ys[0] if n else xs[0]], yv])
    ax.plot(x_path, y_path, "-", color="tab:blue", lw=1.0, label="stages")
    if number_stages:
        for k in range(n):
            ax.annotate(str(k + 1), (xs[k + 1], ys[k]), textcoords="offset points",
                        xytext=(3, 3), fontsize=7, color="tab:blue")
    fs = float(result["feed_stage"])
    if np.isfinite(fs) and 1 <= round(fs) <= n:
        k = int(round(fs)) - 1
        ax.plot([xs[k + 1]], [ys[k]], "s", color="tab:blue", ms=7, mfc="none", label="feed stage")
    ax.set_xlabel("liquid mole fraction x")
    ax.set_ylabel("vapor mole fraction y")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    return ax


def plot_hunter_nash(result: dict, ax=None, coords: str = "equilateral",
                     labels: tuple[str, str, str] = ("1", "2", "3")):
    """Plot a Hunter-Nash construction from :func:`difflow.phase_diagrams.hunter_nash`.

    Draws the ternary LLE diagram (binodal, tie lines, plait point, as
    :func:`plot_ternary`), the points ``F`` (square), ``S`` (triangle), ``M``
    (circle), ``E_1`` (down triangle), ``R_N`` (plus) and, when it falls near
    the diagram, the difference point ``Delta`` (cross); the lines ``F E_1`` and
    ``S R_N`` (dashed, extended to ``Delta``), ``R_N M E_1`` (dash-dot), each
    stage's tie line (thick solid, numbered at its raffinate end) and the
    operating lines from ``R_n`` to ``E_{n+1}`` (dotted). Colour is not the only
    cue: every element differs in line style or marker.

    Args:
        result: Result of ``hunter_nash``.
        ax: Optional matplotlib axes.
        coords: ``"equilateral"`` or ``"right"``.
        labels: Vertex labels for species 0, 1, 2.

    Returns:
        The matplotlib axes.
    """
    ax = plot_ternary(result["lle_data"], ax=ax, coords=coords, labels=labels)
    xy = lambda c: ternary_xy(np.asarray(c, dtype=float), coords)  # noqa: E731
    F, S, M = xy(result["F"]), xy(result["S"]), xy(result["M"])
    E1, RN = xy(result["E1"]), xy(result["R_N"])
    dc = np.asarray(result["delta"]["composition"], dtype=float)
    delta = None
    if np.all(np.isfinite(dc)):
        d = xy(dc)
        if np.max(np.abs(d)) < 4.0:
            delta = d

    def seg(p, q, style, **kw):
        ax.plot([p[0], q[0]], [p[1], q[1]], style, color="k", lw=1, **kw)

    seg(RN, E1, "-.")
    seg(F, delta if delta is not None else E1, "--")
    seg(S, delta if delta is not None else RN, "--")
    for p, mk, lab in ((F, "s", "F"), (S, "^", "S"), (M, "o", "M"), (E1, "v", "E1"), (RN, "P", "R_N")):
        ax.plot([p[0]], [p[1]], mk, color="k", mfc="w", ms=7)
        ax.annotate(lab, p, textcoords="offset points", xytext=(6, 6), fontsize=8)
    if delta is not None:
        ax.plot([delta[0]], [delta[1]], "X", color="k", mfc="w", ms=8)
        ax.annotate("Delta", delta, textcoords="offset points", xytext=(6, 6), fontsize=8)
    E, R = np.asarray(result["E"], dtype=float), np.asarray(result["R"], dtype=float)
    for n in range(len(E)):
        e, r = xy(E[n]), xy(R[n])
        ax.plot([e[0], r[0]], [e[1], r[1]], "-", color="k", lw=2)
        ax.annotate(str(n + 1), r, textcoords="offset points", xytext=(-10, -10), fontsize=8)
        if n + 1 < len(E):
            e2 = xy(E[n + 1])
            ax.plot([r[0], e2[0]], [r[1], e2[1]], ":", color="k", lw=1.2)
    ax.set_title(f"Hunter-Nash: {result['n_stages']:.2f} stages", fontsize=9)
    all_xy = np.vstack([np.atleast_2d(v) for v in (F, S, M, E1, RN) + ((delta,) if delta is not None else ())])
    lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
    lim_lo = np.minimum(lo, ternary_xy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), coords).min(axis=0))
    lim_hi = np.maximum(hi, ternary_xy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), coords).max(axis=0))
    pad = 0.08 * (lim_hi - lim_lo)
    ax.set_xlim(lim_lo[0] - pad[0], lim_hi[0] + pad[0])
    ax.set_ylim(lim_lo[1] - pad[1], lim_hi[1] + pad[1])
    return ax
