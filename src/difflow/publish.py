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
"""

from __future__ import annotations

import html
import itertools
import json
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
        """The label shown on the page, defaulting to the key."""
        return self.label or self.key

    def values(self):
        """The grid points, inclusive of both bounds."""
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
        """Grid shape, one dimension per axis."""
        return tuple(axis.n for axis in self.axes)

    @property
    def n_points(self) -> int:
        """Total operating points solved --- the product of the axes."""
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
# The page
# ---------------------------------------------------------------------


def to_html(
    result: SweepResult,
    *,
    title: str = "difflow model",
    description: str = "",
) -> str:
    """Render a sweep as one self-contained HTML page.

    The page carries the grid, interpolates between points as the
    sliders move, and needs nothing at run time --- no network, no
    Python, no JAX.

    Args:
        result: the sweep to publish.
        title: page and document title.
        description: a paragraph under the title; plain text.

    Returns:
        A complete HTML document.
    """
    from difflow import __version__

    payload = json.dumps(result.to_dict(), separators=(",", ":"))
    return _TEMPLATE.format(
        title=html.escape(title),
        description=html.escape(description),
        version=html.escape(__version__),
        n_points=result.n_points,
        palette=json.dumps(_PALETTE),
        data=payload,
    )


def publish(
    flowsheet,
    axes: Sequence[SweepAxis],
    outputs: dict[str, Callable],
    path: str | Path,
    *,
    title: str = "difflow model",
    description: str = "",
    units: dict[str, str] | None = None,
    **sweep_kwargs,
) -> Path:
    """Sweep a flowsheet and write the interactive page in one step.

    Returns:
        The path written.
    """
    result = sweep(flowsheet, axes, outputs, units=units, **sweep_kwargs)
    path = Path(path)
    path.write_text(to_html(result, title=title, description=description))
    return path


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --surface: #fcfcfb; --panel: #f4f3f0; --ink: #0b0b0b;
    --ink-soft: #52514e; --grid: #e4e2de; --line: #c9c7c2;
    --series: #2a78d6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1.25rem; background: var(--surface);
    color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI",
          Roboto, Helvetica, Arial, sans-serif;
  }}
  main {{ max-width: 62rem; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .4rem; letter-spacing: -.01em; }}
  p.lede {{ color: var(--ink-soft); margin: 0 0 1.75rem; max-width: 46rem; }}
  .layout {{ display: grid; grid-template-columns: 17rem 1fr; gap: 1.5rem;
             align-items: start; }}
  @media (max-width: 46rem) {{ .layout {{ grid-template-columns: 1fr; }} }}
  .panel {{
    background: var(--panel); border: 1px solid var(--grid);
    border-radius: 10px; padding: 1rem 1.1rem;
  }}
  .panel h2 {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
    color: var(--ink-soft); margin: 0 0 .9rem; font-weight: 600;
  }}
  .control {{ margin-bottom: 1.1rem; }}
  .control:last-child {{ margin-bottom: 0; }}
  .control label {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .85rem; margin-bottom: .35rem; gap: .5rem;
  }}
  .control .value {{
    font-variant-numeric: tabular-nums; color: var(--ink);
    font-weight: 600;
  }}
  input[type=range] {{ width: 100%; accent-color: var(--series); }}
  select {{
    width: 100%; padding: .35rem .5rem; border: 1px solid var(--line);
    border-radius: 6px; background: var(--surface); color: var(--ink);
    font: inherit; font-size: .85rem;
  }}
  table {{ width: 100%; border-collapse: collapse; margin-top: .25rem; }}
  th, td {{
    text-align: left; padding: .4rem .3rem; font-size: .85rem;
    border-bottom: 1px solid var(--grid);
  }}
  th {{
    color: var(--ink-soft); font-weight: 600; font-size: .72rem;
    text-transform: uppercase; letter-spacing: .06em;
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.sens {{ text-align: right; font-variant-numeric: tabular-nums;
             color: var(--ink-soft); }}
  figure {{ margin: 0; }}
  footer {{
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--grid);
    color: var(--ink-soft); font-size: .8rem;
  }}
  code {{ font-size: .82em; background: var(--panel); padding: .1em .35em;
          border-radius: 4px; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="lede">{description}</p>

  <div class="layout">
    <div class="panel" id="controls">
      <h2>Parameters</h2>
      <div id="sliders"></div>
    </div>

    <div>
      <figure class="panel" style="margin-bottom:1.5rem">
        <h2>Response</h2>
        <div class="control">
          <label for="output-pick"><span>Plotted quantity</span></label>
          <select id="output-pick"></select>
        </div>
        <svg id="chart" viewBox="0 0 640 300" width="100%"
             role="img" aria-label="Model response"></svg>
      </figure>

      <div class="panel">
        <h2>Values at this point</h2>
        <table>
          <thead>
            <tr><th>Quantity</th><th style="text-align:right">Value</th>
                <th style="text-align:right">Sensitivity</th></tr>
          </thead>
          <tbody id="readout"></tbody>
        </table>
      </div>
    </div>
  </div>

  <footer>
    Precomputed with difflow {version} over {n_points} solved operating
    points; values between grid points are interpolated. Sensitivities are
    exact derivatives from automatic differentiation, reported per unit of
    the first parameter.
  </footer>
</main>

<script>
const DATA = {data};
const C = {palette};
const state = DATA.axes.map(a => Math.floor(a.n / 2));

/* ---- interpolation ------------------------------------------------ */
function axisFrac(ai, x) {{
  const a = DATA.axes[ai];
  const t = (x - a.lo) / (a.hi - a.lo) * (a.n - 1);
  const i = Math.max(0, Math.min(a.n - 2, Math.floor(t)));
  return [i, Math.max(0, Math.min(1, t - i))];
}}
function at(grid, idx) {{ return idx.reduce((g, i) => g[i], grid); }}

/* multilinear over however many axes there are */
function interp(grid, coords) {{
  const parts = coords.map((x, ai) => axisFrac(ai, x));
  let total = 0;
  const n = coords.length;
  for (let mask = 0; mask < (1 << n); mask++) {{
    let weight = 1, idx = [];
    for (let ai = 0; ai < n; ai++) {{
      const [i, f] = parts[ai];
      const up = (mask >> ai) & 1;
      weight *= up ? f : 1 - f;
      idx.push(i + up);
    }}
    if (weight > 0) total += weight * at(grid, idx);
  }}
  return total;
}}
function coords() {{
  return DATA.axes.map((a, i) => a.values[state[i]]);
}}
function fmt(v) {{
  if (!isFinite(v)) return "--";
  const m = Math.abs(v);
  if (m !== 0 && (m < 1e-3 || m >= 1e5)) return v.toExponential(3);
  return v.toFixed(m >= 100 ? 1 : m >= 1 ? 3 : 4);
}}

/* ---- controls ----------------------------------------------------- */
const sliders = document.getElementById("sliders");
DATA.axes.forEach((a, i) => {{
  const wrap = document.createElement("div");
  wrap.className = "control";
  wrap.innerHTML =
    '<label for="ax' + i + '"><span>' + a.label + '</span>' +
    '<span class="value" id="val' + i + '"></span></label>' +
    '<input type="range" id="ax' + i + '" min="0" max="' + (a.n - 1) +
    '" step="1" value="' + state[i] + '">';
  sliders.appendChild(wrap);
  wrap.querySelector("input").addEventListener("input", e => {{
    state[i] = +e.target.value;
    render();
  }});
}});

const pick = document.getElementById("output-pick");
DATA.outputs.forEach((o, i) => {{
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = o.name + (o.units ? " (" + o.units + ")" : "");
  pick.appendChild(opt);
}});
pick.addEventListener("change", render);

/* ---- chart -------------------------------------------------------- */
function drawChart(outIndex) {{
  const svg = document.getElementById("chart");
  const W = 640, H = 300, L = 64, R = 16, T = 16, B = 46;
  const axis = DATA.axes[0], out = DATA.outputs[outIndex];
  const here = coords();

  const xs = axis.values;
  const ys = xs.map(x => interp(out.values, [x, ...here.slice(1)]));
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = (yMax - yMin) * 0.08 || Math.abs(yMax) * 0.08 || 1;
  const lo = yMin - pad, hi = yMax + pad;

  const sx = x => L + (x - axis.lo) / (axis.hi - axis.lo) * (W - L - R);
  const sy = y => H - B - (y - lo) / (hi - lo) * (H - T - B);

  let parts = [];
  /* grid + y labels, recessive */
  for (let k = 0; k <= 4; k++) {{
    const v = lo + (hi - lo) * k / 4, y = sy(v);
    parts.push('<line x1="' + L + '" y1="' + y + '" x2="' + (W - R) +
      '" y2="' + y + '" stroke="' + C.grid + '" stroke-width="1"/>');
    parts.push('<text x="' + (L - 8) + '" y="' + (y + 4) +
      '" text-anchor="end" font-size="11" fill="' + C.ink_soft + '">' +
      fmt(v) + '</text>');
  }}
  /* x labels at the ends and middle */
  [0, 0.5, 1].forEach(f => {{
    const v = axis.lo + (axis.hi - axis.lo) * f;
    parts.push('<text x="' + sx(v) + '" y="' + (H - B + 20) +
      '" text-anchor="middle" font-size="11" fill="' + C.ink_soft + '">' +
      fmt(v) + '</text>');
  }});
  parts.push('<text x="' + ((L + W - R) / 2) + '" y="' + (H - 8) +
    '" text-anchor="middle" font-size="12" fill="' + C.ink_soft + '">' +
    axis.label + (axis.units ? " (" + axis.units + ")" : "") + '</text>');

  const d = xs.map((x, i) => (i ? "L" : "M") + sx(x) + " " + sy(ys[i])).join(" ");
  parts.push('<path d="' + d + '" fill="none" stroke="' + C.series +
    '" stroke-width="2" stroke-linejoin="round"/>');

  /* where the sliders currently sit */
  const cx = sx(here[0]), cy = sy(interp(out.values, here));
  parts.push('<line x1="' + cx + '" y1="' + T + '" x2="' + cx + '" y2="' +
    (H - B) + '" stroke="' + C.line + '" stroke-width="1" ' +
    'stroke-dasharray="4 3"/>');
  parts.push('<circle cx="' + cx + '" cy="' + cy + '" r="5" fill="' +
    C.series + '" stroke="' + C.surface + '" stroke-width="2"/>');

  svg.innerHTML = parts.join("");
}}

/* ---- readout ------------------------------------------------------ */
function render() {{
  const here = coords();
  DATA.axes.forEach((a, i) => {{
    document.getElementById("val" + i).textContent =
      fmt(here[i]) + (a.units ? " " + a.units : "");
  }});

  const rows = DATA.outputs.map(o => {{
    const v = interp(o.values, here);
    const gKey = DATA.axes[0].key;
    const g = o.gradients && o.gradients[gKey]
      ? interp(o.gradients[gKey], here) : null;
    return "<tr><td>" + o.name + "</td><td class='num'>" + fmt(v) +
      (o.units ? " <span style='color:" + C.ink_soft + "'>" + o.units +
        "</span>" : "") +
      "</td><td class='sens'>" + (g === null ? "--" : fmt(g)) + "</td></tr>";
  }});
  document.getElementById("readout").innerHTML = rows.join("");
  drawChart(+pick.value);
}}
render();
</script>
</body>
</html>
"""
