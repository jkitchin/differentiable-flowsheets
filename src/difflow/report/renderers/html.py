"""HTML renderer for :class:`~difflow.report.ir.Report`.

This renderer produces a self-contained HTML document.  Equations are
emitted in MathJax-compatible ``\\(...\\)`` form and a small MathJax CDN
tag is included so the file renders directly in a browser.
"""

from __future__ import annotations

from html import escape
from io import StringIO

from difflow.report.ir import Report


def _tr(cells: list[str], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{escape(c)}</{tag}>" for c in cells) + "</tr>"


def _table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    out = ["<table>"]
    out.append(_tr(header, "th"))
    for row in rows:
        out.append(_tr(row))
    out.append("</table>")
    return "\n".join(out)


_MATHJAX = (
    '<script async '
    'src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">'
    "</script>"
)

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; }
h1 { border-bottom: 2px solid #ccc; }
h2 { margin-top: 2em; border-bottom: 1px solid #eee; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: left; }
th { background: #f5f5f5; }
.eq { margin: 0.25em 0; }
"""


def to_html(report: Report, embed_diagram: bool = True) -> str:
    """Render a Report as a standalone HTML document.

    Args:
        report: Report IR.
        embed_diagram: If True, include the flowsheet diagram rendered by
            :mod:`difflow.visualization.flowsheet_viz`.  Silently skipped if
            ``ipycytoscape`` is not available.
    """
    out = StringIO()
    w = out.write
    w("<!doctype html>\n<html><head><meta charset='utf-8'>")
    w("<title>Flowsheet Report</title>")
    w(f"<style>{_STYLE}</style>")
    w(_MATHJAX)
    w("</head><body>\n")
    w("<h1>Flowsheet Report</h1>\n")

    # Provenance
    p = report.provenance
    prov_rows = [
        ["difflow", p.difflow_version],
        ["jax", f"{p.jax_version} ({p.jax_backend}, x64={p.jax_x64})"],
        ["python", p.python_version],
        ["platform", p.platform],
        ["timestamp", p.timestamp],
    ]
    for name, ver in p.plugin_versions.items():
        prov_rows.append([name, ver])
    if p.git_commit:
        prov_rows.append(["git", f"{p.git_commit[:12]}{' (dirty)' if p.git_dirty else ''}"])
    w("<h2>Provenance</h2>\n")
    w(_table(["Field", "Value"], prov_rows))

    # Topology
    w("<h2>Topology</h2>\n")
    w(
        "<p><b>Units:</b> "
        + escape(", ".join(report.topology.units) or "(none)")
        + "</p>\n"
    )
    w(
        "<p><b>Species order:</b> "
        + escape(", ".join(report.topology.species_order) or "(none)")
        + "</p>\n"
    )
    if report.topology.edges:
        w("<h3>Connections</h3>\n")
        rows = [
            [e.stream, e.source or "(feed)", e.target or "(product)"]
            for e in report.topology.edges
        ]
        w(_table(["stream", "source", "target"], rows))
    if report.topology.recycles:
        w("<h3>Recycles</h3>\n")
        rows = [[r.source_stream, r.dest_stream] for r in report.topology.recycles]
        w(_table(["source", "dest"], rows))

    # Units
    w("<h2>Unit Operations</h2>\n")
    for u in report.units:
        w(f"<h3>{escape(u.name)} — {escape(u.type)} ({escape(u.plugin)})</h3>\n")
        if u.description:
            w(f"<p>{escape(u.description)}</p>\n")
        if u.numerical_method:
            w(f"<p><i>Numerical method:</i> {escape(u.numerical_method)}</p>\n")
        w(f"<p><b>Inlets:</b> {escape(', '.join(u.inlet_names) or '(none)')}<br/>")
        w(f"<b>Outlets:</b> {escape(', '.join(u.outlet_names) or '(none)')}</p>\n")
        if u.equations:
            w("<p><b>Governing equations</b></p>\n")
            for eq in u.equations:
                # MathJax renders \( ... \) as inline; \[ ... \] as display.
                w(f"<div class='eq'>\\[ {eq} \\]</div>\n")
        if u.assumptions:
            w("<p><b>Assumptions</b></p><ul>")
            for a in u.assumptions:
                w(f"<li>{escape(a)}</li>")
            w("</ul>\n")
        if u.parameters:
            rows = [
                [pr.name, pr.symbol, pr.units or "-", pr.value_repr]
                for pr in u.parameters
            ]
            w("<p><b>Parameters</b></p>\n")
            w(_table(["name", "symbol", "units", "value"], rows))
        if u.references:
            w("<p><b>References</b></p><ul>")
            for r in u.references:
                w(f"<li>{escape(r)}</li>")
            w("</ul>\n")

    if report.species:
        w("<h2>Species and Thermophysical Data</h2>\n")
        rows = []
        for s in report.species:
            rows.append([
                s.name,
                "-" if s.MW is None else f"{s.MW:.4g}",
                "-" if s.Tc is None else f"{s.Tc:.4g}",
                "-" if s.Pc is None else f"{s.Pc:.4g}",
                "-" if s.omega is None else f"{s.omega:.4g}",
                "-" if s.Hf is None else f"{s.Hf:.4g}",
                s.source or "-",
            ])
        w(_table(
            ["species", "MW (g/mol)", "Tc (K)", "Pc (Pa)", "omega", "Hf (J/mol)", "source"],
            rows,
        ))

    if report.feeds:
        w("<h2>Feed Streams</h2>\n")
        for f in report.feeds:
            w(f"<h3>{escape(f.name)}</h3>")
            w(f"<p>T = {f.T:.4g} K, P = {f.P:.4g} Pa</p>")
            rows = [[s, f"{v:.4g}"] for s, v in f.flows.items()]
            w(_table(["species", "F (mol/s)"], rows))

    if report.results:
        w("<h2>Solved Streams</h2>\n")
        for r in report.results:
            w(f"<h3>{escape(r.name)}</h3>")
            w(f"<p>T = {r.T:.4g} K, P = {r.P:.4g} Pa</p>")
            rows = [[s, f"{v:.4g}"] for s, v in r.flows.items()]
            w(_table(["species", "F (mol/s)"], rows))

    if report.balance_checks:
        w("<h2>Mass Balance Closure</h2>\n")
        rows = [
            [b.species, f"{b.feed_total:.6g}", f"{b.outlet_total:.6g}", f"{b.residual:+.3g}"]
            for b in report.balance_checks
        ]
        w(_table(["species", "feed total", "outlet total", "residual"], rows))

    if report.optimization is not None:
        o = report.optimization
        w("<h2>Optimization and Sensitivity</h2>\n")
        units = f" {o.objective_units}" if o.objective_units else ""
        w(
            f"<p><b>Objective:</b> {escape(o.objective_name)} ({escape(o.sense)}); "
            f"value = {o.objective_value:.6g}{escape(units)}</p>\n"
        )
        if o.objective_source:
            w(f"<p><b>Source:</b> <code>{escape(o.objective_source)}</code></p>\n")

        rows = []
        for v in o.variables:
            bounds = (
                f"[{v.lower:.4g}, {v.upper:.4g}]"
                if v.lower is not None and v.upper is not None
                else "-"
            )
            rows.append([
                v.name,
                f"{v.value:.6g}",
                bounds,
                "-" if v.gradient is None else f"{v.gradient:.4g}",
                "-" if v.elasticity is None else f"{v.elasticity:.4g}",
            ])
        w("<h3>Decision variables</h3>\n")
        w(_table(["variable", "value", "bounds", "dJ/dx", "elasticity"], rows))

        if o.tornado:
            trows = [
                [
                    t.variable,
                    f"{t.low_value:.4g}",
                    f"{t.high_value:.4g}",
                    f"{t.low_output:.6g}",
                    f"{t.high_output:.6g}",
                    f"{t.swing:.4g}",
                ]
                for t in o.tornado
            ]
            w("<h3>Sensitivity tornado</h3>\n")
            w(_table(
                ["variable", "low", "high", "J(low)", "J(high)", "swing"], trows
            ))

    w("</body></html>\n")
    return out.getvalue()
