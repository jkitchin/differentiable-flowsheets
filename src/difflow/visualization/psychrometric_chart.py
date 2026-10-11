"""Matplotlib helper for the psychrometric chart of :mod:`difflow.psychrometrics`.

matplotlib is an optional dependency and is imported when the helper is
called. Colour is never the only cue: each family has its own line style
(relative humidity solid, wet bulb dashed, enthalpy dash-dot, specific volume
dotted), the saturation curve is thick and black, and process lines carry
markers and arrows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_T0 = 273.15


def plot_psychrometric_chart(data: dict, ax=None, processes: Sequence | None = None,
                             celsius: bool = True, label_lines: bool = True):
    """Plot a psychrometric chart from :func:`~difflow.psychrometrics.psychrometric_chart_data`.

    Humidity ratio (g water per kg dry air) against dry-bulb temperature.

    Args:
        data: Result of ``psychrometric_chart_data``.
        ax: Optional matplotlib axes.
        processes: Process lines to overlay: each entry is a sequence of states
            (dicts from :func:`~difflow.psychrometrics.state`) joined by arrows,
            or a dict ``{"states": [...], "label": "cooling coil"}``.
        celsius: Plot the temperature axis in degrees C (otherwise K).
        label_lines: Label each family's lines at their ends.

    Returns:
        The matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("matplotlib is required for the psychrometric chart: "
                          "pip install matplotlib") from e
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))
    off = _T0 if celsius else 0.0

    def draw(lines, style, color, lw, name, fmt, end="last"):
        first = True
        for d in lines:
            T = np.asarray(d["T"]) - off
            W = np.asarray(d["W"]) * 1e3
            ax.plot(T, W, style, color=color, lw=lw, label=name if first else None)
            first = False
            if label_lines:
                ok = np.where(np.isfinite(W))[0]
                if ok.size > 2 and fmt is not None:
                    i = ok[-1] if end == "last" else ok[0]
                    dx = 2 if end == "last" else -2
                    ax.annotate(fmt(d["value"]), (T[i], W[i]), fontsize=6, color=color,
                                xytext=(dx, 0 if end == "last" else 3), textcoords="offset points",
                                va="center", ha="left" if end == "last" else "right")

    draw(data["rh"], "-", "0.55", 0.7, "relative humidity", lambda v: f"{v:.0%}")
    draw(data["wb"], "--", "tab:blue", 0.8, "wet bulb (C)" if celsius else "wet bulb (K)",
         lambda v: f"{v - off:.0f}", end="first")
    draw(data["h"], "-.", "tab:red", 0.8, "enthalpy (kJ/kg)", lambda v: f"{v / 1e3:.0f}", end="first")
    draw(data["v"], ":", "tab:green", 1.0, "volume (m$^3$/kg)", None)
    sat = data["saturation"]
    ax.plot(np.asarray(sat["T"]) - off, np.asarray(sat["W"]) * 1e3, "-", color="k", lw=2.0,
            label="saturation")

    for k, proc in enumerate(processes or []):
        states = proc["states"] if isinstance(proc, dict) else proc
        name = proc.get("label") if isinstance(proc, dict) else None
        T = [float(s["T"]) - off for s in states]
        W = [float(s["W"]) * 1e3 for s in states]
        ax.plot(T, W, "-o", color="k", ms=4, lw=1.4, zorder=5, label=name)
        for i in range(len(T) - 1):
            ax.annotate("", xy=(T[i + 1], W[i + 1]), xytext=(T[i], W[i]),
                        arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.4}, zorder=5)

    lo, hi = data["T_range"]
    ax.set_xlim(lo - off, hi - off)
    ymax = float(np.nanmax(np.asarray(sat["W"]))) * 1e3
    ax.set_ylim(0, ymax * 1.02)
    ax.set_xlabel("dry-bulb temperature (C)" if celsius else "dry-bulb temperature (K)")
    ax.set_ylabel("humidity ratio (g/kg dry air)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_title(f"Psychrometric chart, P = {data['P'] / 1e3:.1f} kPa")
    ax.legend(loc="upper left", fontsize=7)
    return ax
