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
