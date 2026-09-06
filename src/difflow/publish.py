"""Publish a flowsheet as a self-contained interactive page.

difflow's solver cannot run in a browser --- jaxlib has no WebAssembly
build --- so a live model cannot simply be shipped next to a paper. But
a *published* model does not need a general solver. Its topology is
fixed and only a couple of numbers vary, and that much can be computed
ahead of time:

    publish(fs, axes=[SweepAxis("reactor.V", 0.2, 5.0)],
            outputs={"product": lambda s: get_flows(s["out"])["B"]},
            path="reactor.html")

:func:`sweep` evaluates the flowsheet across a grid of parameter values
with ``jax.vmap``, and :func:`to_html` writes the result as one HTML
file with sliders. No server, no Python, no JAX --- and nothing to rot,
which matters for something attached to a paper that has to outlive its
hosting.

The page interpolates between grid points, so it is exact *at* them and
approximate between. Take enough points that the curve is smooth and
that approximation is invisible; the sweep is vectorised, so points are
cheap.

Exact derivatives are recorded alongside the values, since difflow has
them for free and nothing else on a static page can supply them. They
are shown as local sensitivities rather than used for interpolation.

The page is the editor's own front end, frozen: the same canvas, the
same graph model and the same palette, with nothing editable and
nothing to solve. So a published model shows the flowsheet it came
from --- click a unit and it says what the unit is, in the words
``difflow.catalog`` reads off the class --- rather than a set of
sliders belonging to nothing visible. ``difflow.gui`` serves that
bundle from disk; this module inlines it, which is the whole
difference between the two.
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin

#: light-surface palette, matching the difflow documentation charts
_PALETTE = {
    "series": "#2a78d6",
    "accent": "#eb6834",
    "surface": "#fcfcfb",
    "panel": "#f4f3f0",
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
    "grid": "#e4e2de",
    "line": "#c9c7c2",
}


@dataclass
class SweepAxis(ParamsMixin):
    """One parameter to vary, and the range to vary it over.

    Attributes:
        key: flowsheet parameter, in the dot notation
            :meth:`~difflow.flowsheet.Flowsheet.make_objective_fn` uses
            --- ``"<unit>.<param>"``, e.g. ``"reactor.V"``.
        lo, hi: inclusive bounds.
        n: grid points. The page interpolates between them, so this
            sets how faithful it is; 21 is usually plenty for a smooth
            response.
        label: axis label; defaults to ``key``.
        units: shown after the value.
    """

    key: str
    lo: float
    hi: float
    n: int = 21
    label: str | None = None
    units: str = ""

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"axis {self.key!r} needs at least 2 points")
        if not self.hi > self.lo:
            raise ValueError(f"axis {self.key!r}: hi must exceed lo")

    @property
    def title(self) -> str:
        return self.label or self.key

    def values(self):
        return jnp.linspace(self.lo, self.hi, self.n)


@dataclass
class SweepResult(ParamsMixin):
    """A flowsheet evaluated over a grid of parameter values.

    Attributes:
        axes: the parameters that were varied.
        values: output name -> array of shape ``(n_1, ..., n_k)``.
        gradients: output name -> axis key -> array of the same shape,
            the exact derivative at each grid point.
        units: output name -> units, for display.
        baseline: the parameter values the flowsheet started from.
    """

    axes: list[SweepAxis]
    values: dict[str, Any]
    gradients: dict[str, dict[str, Any]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.n for axis in self.axes)

    @property
    def n_points(self) -> int:
        n = 1
        for axis in self.axes:
            n *= axis.n
        return n

    def to_dict(self) -> dict:
        """Plain data, ready for JSON and the page."""
        return {
            "axes": [
                {
                    "key": a.key, "label": a.title, "units": a.units,
                    "lo": a.lo, "hi": a.hi, "n": a.n,
                    "values": [float(v) for v in a.values()],
                }
                for a in self.axes
            ],
            "outputs": [
                {
                    "name": name,
                    "units": self.units.get(name, ""),
                    "values": jnp.asarray(grid).tolist(),
                    "gradients": {
                        key: jnp.asarray(g).tolist()
                        for key, g in self.gradients.get(name, {}).items()
                    },
                }
                for name, grid in self.values.items()
            ],
        }


def sweep(
    flowsheet,
    axes: Sequence[SweepAxis],
    outputs: dict[str, Callable],
    *,
    units: dict[str, str] | None = None,
    gradients: bool = True,
    batch: bool = True,
) -> SweepResult:
    """Evaluate a flowsheet across a grid of parameter values.

    Args:
        flowsheet: the flowsheet to sweep.
        axes: parameters to vary. One or two read best on a page; more
            work, but the reader has that many sliders to think about.
        outputs: name -> ``fn(streams) -> scalar``, the quantities to
            record.
        units: name -> units, for display.
        gradients: also record the exact derivative of each output with
            respect to each axis at every grid point.
        batch: evaluate with ``jax.vmap``. Turn off to fall back to a
            plain loop, which is slower but easier to debug.

    Returns:
        A :class:`SweepResult`.

    Raises:
        ValueError: if no axes or no outputs are given.
    """
    axes = list(axes)
    if not axes:
        raise ValueError("a sweep needs at least one axis")
    if not outputs:
        raise ValueError("a sweep needs at least one output")

    keys = [axis.key for axis in axes]
    mesh = jnp.meshgrid(*[axis.values() for axis in axes], indexing="ij")
    flat = [m.reshape(-1) for m in mesh]
    shape = tuple(axis.n for axis in axes)

    values: dict[str, Any] = {}
    grads: dict[str, dict[str, Any]] = {}

    for name, output_fn in outputs.items():
        objective = flowsheet.make_objective_fn(output_fn)

        def at(point, objective=objective):
            return objective({k: point[i] for i, k in enumerate(keys)})

        stacked = jnp.stack(flat, axis=1)          # (n_points, n_axes)
        if batch:
            values[name] = jax.vmap(at)(stacked).reshape(shape)
        else:
            values[name] = jnp.asarray(
                [at(p) for p in stacked]
            ).reshape(shape)

        if gradients:
            def grad_at(point, objective=objective):
                return jax.grad(
                    lambda p: objective({k: p[i] for i, k in enumerate(keys)})
                )(point)

            if batch:
                g = jax.vmap(grad_at)(stacked)
            else:
                g = jnp.stack([grad_at(p) for p in stacked])
            grads[name] = {
                key: g[:, i].reshape(shape) for i, key in enumerate(keys)
            }

    return SweepResult(
        axes=axes, values=values, gradients=grads, units=units or {},
    )


# ---------------------------------------------------------------------
# What the page says about the flowsheet
# ---------------------------------------------------------------------

#: Arrays longer than this are named rather than printed. A stoichiometry
#: vector is worth seeing; a 500-point property table is not.
_MAX_INLINE = 12


def _plain(value, depth: int = 0):
    """One parameter value as something JSON and a reader can both take.

    Whatever cannot be shown as data is named by its type instead of
    dropped, because a parameter that vanishes from a published page
    reads as a parameter the unit does not have. The non-finite floats
    become strings for the same reason `difflow.gui` sends them as
    strings: ``JSON.parse`` rejects the ``Infinity`` Python writes, and
    ``mass_action_kinetics`` puts one in ``K_eq`` for every irreversible
    reaction.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else str(f).replace("inf", "Infinity")
    if isinstance(value, dict):
        if depth >= 2:
            return "{...}"
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_INLINE:
            return f"<{len(value)} values>"
        return [_plain(v, depth + 1) for v in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        array = jnp.asarray(value)
        if array.size > _MAX_INLINE:
            return f"<array {tuple(array.shape)}>"
        return _plain(array.tolist(), depth + 1)
    return f"<{type(value).__name__}>"


def _display_params(operation) -> dict:
    """A unit's parameters, as far as they can be shown."""
    params = getattr(operation, "params", None)
    if params is None:
        return {}
    try:
        return {name: _plain(params[name]) for name in params.keys()}
    except Exception:                      # a params object that is not one
        return {}


def _topology(flowsheet) -> dict | None:
    """The flowsheet in the shape the canvas reads.

    Deliberately not :func:`difflow.serialize.to_dict`. That is a
    document meant to be loaded back, and it refuses a flowsheet whose
    units hold objects JSON cannot carry --- which is most interesting
    flowsheets, and no reason to publish a page without a picture. This
    is a description for looking at: the same node keys the editor uses,
    and every value already reduced to something printable.

    Returns ``None`` if there is nothing to draw.
    """
    from difflow.gui.layout import auto_layout

    units = list(getattr(flowsheet, "units", None) or [])
    if not units:
        return None
    view = dict(getattr(flowsheet, "view", None) or {})
    return {
        "units": [
            {
                "name": u.name,
                "operation": type(u.operation).__name__,
                "inlets": list(u.inlet_names),
                "outlets": list(u.outlet_names),
                "params": _display_params(u.operation),
            }
            for u in units
        ],
        "feeds": {
            name: {k: _plain(v) for k, v in stream.items()}
            for name, stream in (getattr(flowsheet, "feeds", None) or {}).items()
        },
        "recycles": dict(getattr(flowsheet, "recycles", None) or {}),
        # Stored positions on top of the automatic ones, exactly as the
        # editor serves them, so a page published from a flowsheet someone
        # arranged in the GUI opens arranged that way.
        "view": {"nodes": {**auto_layout(flowsheet),
                           **(view.get("nodes") or {})}},
    }


def _catalog_for(flowsheet) -> dict:
    """Catalog entries for the operations this flowsheet actually uses.

    Trimmed to what the panel shows. The whole ``OperationSchema`` would
    carry parameter *defaults*, which are arbitrary objects and not
    JSON, and 87 entries where the page needs three.
    """
    from difflow.catalog import describe_class

    out: dict[str, dict] = {}
    for unit in getattr(flowsheet, "units", None) or []:
        cls = type(unit.operation)
        if cls.__name__ in out:
            continue
        try:
            spec = describe_class(cls)
        except Exception:                  # a class the catalog cannot read
            continue
        out[cls.__name__] = {
            "description": spec.description,
            "equations": list(spec.equations),
            "assumptions": list(spec.assumptions),
            "references": list(spec.references),
            "parameters": [
                {"name": p.name, "units": p.units, "description": p.description}
                for p in spec.parameters
            ],
        }
    return out


# ---------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------

#: Where the built front end lives. The editor serves these files over
#: the loopback; a published page inlines them.
_BUNDLE = ("publish.js", "publish.css")


def _assets() -> tuple[str, str]:
    """The built JavaScript and CSS of the frozen front end.

    Raises:
        FileNotFoundError: if the bundle was not built. It is committed,
            so this means a source checkout with the build not run --- and
            saying that is better than writing a blank page.
    """
    from difflow.gui.server import STATIC

    out = []
    for name in _BUNDLE:
        path = STATIC / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the published page is the editor's own "
                f"bundle, built with `make gui-build`"
            )
        out.append(path.read_text(encoding="utf-8"))
    return out[0], out[1]


def _inline(text: str, tag: str) -> str:
    """Text safe to sit inside a ``<script>`` or ``<style>`` element.

    The parser ends the element at a literal closing tag wherever it
    appears, string literal or not, so the one sequence that can end it
    is broken with a backslash --- which is inert in both JavaScript
    strings and CSS.
    """
    return text.replace(f"</{tag}", f"<\\/{tag}")


def to_html(
    result: SweepResult,
    *,
    title: str = "difflow model",
    description: str = "",
    flowsheet=None,
) -> str:
    """Render a sweep as one self-contained HTML page.

    The page carries the grid, interpolates between points as the
    sliders move, and needs nothing at run time --- no network, no
    Python, no JAX.

    Args:
        result: the sweep to publish.
        title: page and document title.
        description: a paragraph under the title; plain text.
        flowsheet: the flowsheet the sweep came from. Given, the page
            also draws it and describes its units; omitted, the page is
            the sliders alone, which is what it was before.

    Returns:
        A complete HTML document.
    """
    from difflow import __version__

    script, style = _assets()
    payload = {
        "sweep": result.to_dict(),
        "topology": _topology(flowsheet) if flowsheet is not None else None,
        "catalog": _catalog_for(flowsheet) if flowsheet is not None else {},
        "version": __version__,
        "n_points": result.n_points,
    }
    # `<` is escaped rather than left alone: the payload carries names the
    # user chose, and a stream called `</script>` would otherwise end the
    # element it is written into.
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")

    filled = {
        "TITLE": html.escape(title),
        "DESCRIPTION": html.escape(description),
        "STYLE": _inline(style, "style"),
        "DATA": data,
        "SCRIPT": _inline(script, "script"),
    }
    # One pass, so a title of "__SCRIPT__" is a title and not a second copy
    # of the bundle. `.format()` is not an option: the bundle is minified
    # JavaScript and full of braces.
    return re.sub(r"__([A-Z]+)__", lambda m: filled[m.group(1)], _TEMPLATE)


def publish(
    flowsheet,
    axes: Sequence[SweepAxis],
    outputs: dict[str, Callable],
    path: str | Path,
    *,
    title: str = "difflow model",
    description: str = "",
    units: dict[str, str] | None = None,
    topology: bool = True,
    **sweep_kwargs,
) -> Path:
    """Sweep a flowsheet and write the interactive page in one step.

    Args:
        flowsheet: the flowsheet to sweep and to draw.
        axes: parameters to vary.
        outputs: name -> ``fn(streams) -> scalar``.
        path: where to write the page.
        title: page and document title.
        description: a paragraph under the title; plain text.
        units: output name -> units, for display.
        topology: also draw the flowsheet on the page. Off leaves the
            sliders and the chart alone, which is the whole page as it
            was before.
        **sweep_kwargs: passed through to :func:`sweep`.

    Returns:
        The path written.
    """
    result = sweep(flowsheet, axes, outputs, units=units, **sweep_kwargs)
    path = Path(path)
    path.write_text(to_html(
        result, title=title, description=description,
        flowsheet=flowsheet if topology else None,
    ))
    return path


#: The page around the bundle. Everything interactive is Svelte; what is
#: here is what has to be readable before any JavaScript runs, and the
#: title and lede are written by Python because they are the one part of
#: the page that carries text the user typed.
_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__STYLE__
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="lede">__DESCRIPTION__</p>
  <div id="app"></div>
</main>
<script>window.DIFFLOW = __DATA__;</script>
<script>
__SCRIPT__
</script>
</body>
</html>
"""
