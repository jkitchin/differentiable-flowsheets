"""Markdown renderer for :class:`~difflow.report.ir.Report`."""

from __future__ import annotations

from io import StringIO

from difflow.report.ir import Report


def _table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_flow(v: float) -> str:
    if v == 0.0:
        return "0"
    return f"{v:.4g}"


def to_markdown(report: Report) -> str:
    """Render a Report as a Markdown string."""
    out = StringIO()
    w = out.write

    w("# Flowsheet Report\n\n")

    # --- Provenance
    p = report.provenance
    w("## Provenance\n\n")
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
        prov_rows.append([
            "git",
            f"{p.git_commit[:12]}{' (dirty)' if p.git_dirty else ''}",
        ])
    w(_table(["Field", "Value"], prov_rows))
    w("\n\n")

    # --- Topology
    w("## Topology\n\n")
    w(f"- Units: {', '.join(report.topology.units) or '(none)'}\n")
    w(f"- Species order: {', '.join(report.topology.species_order) or '(none)'}\n")
    if report.topology.edges:
        w("\n**Connections**\n\n")
        rows = []
        for e in report.topology.edges:
            rows.append([e.stream, e.source or "(feed)", e.target or "(product)"])
        w(_table(["stream", "source", "target"], rows))
        w("\n")
    if report.topology.recycles:
        w("\n**Recycles**\n\n")
        rows = [[r.source_stream, r.dest_stream] for r in report.topology.recycles]
        w(_table(["source", "dest"], rows))
        w("\n")
    w("\n")

    # --- Units
    w("## Unit Operations\n\n")
    for u in report.units:
        w(f"### {u.name} — {u.type} ({u.plugin})\n\n")
        if u.description:
            w(f"{u.description}\n\n")
        if u.numerical_method:
            w(f"*Numerical method:* {u.numerical_method}\n\n")
        w(f"**Inlets:** {', '.join(u.inlet_names) or '(none)'}  \n")
        w(f"**Outlets:** {', '.join(u.outlet_names) or '(none)'}\n\n")

        if u.equations:
            w("**Governing equations**\n\n")
            for eq in u.equations:
                w(f"- $${eq}$$\n")
            w("\n")

        if u.assumptions:
            w("**Assumptions**\n\n")
            for a in u.assumptions:
                w(f"- {a}\n")
            w("\n")

        if u.parameters:
            w("**Parameters**\n\n")
            rows = [
                [pr.name, pr.symbol, pr.units or "-", pr.value_repr]
                for pr in u.parameters
            ]
            w(_table(["name", "symbol", "units", "value"], rows))
            w("\n\n")

        if u.references:
            w("**References**\n\n")
            for r in u.references:
                w(f"- {r}\n")
            w("\n")

    # --- Species table
    if report.species:
        w("## Species and Thermophysical Data\n\n")
        tracked = any(s.accessed is not None for s in report.species)
        header = ["species", "MW (g/mol)", "Tc (K)", "Pc (Pa)", "ω", "Hf (J/mol)", "source"]
        if tracked:
            header.append("accessed")
        rows = []
        for s in report.species:
            row = [
                s.name,
                "-" if s.MW is None else f"{s.MW:.4g}",
                "-" if s.Tc is None else f"{s.Tc:.4g}",
                "-" if s.Pc is None else f"{s.Pc:.4g}",
                "-" if s.omega is None else f"{s.omega:.4g}",
                "-" if s.Hf is None else f"{s.Hf:.4g}",
                s.source or "-",
            ]
            if tracked:
                row.append("yes" if s.accessed else "no")
            rows.append(row)
        w(_table(header, rows))
        w("\n\n")

    # --- Feeds
    if report.feeds:
        w("## Feed Streams\n\n")
        for f in report.feeds:
            w(f"### {f.name}\n\n")
            w(f"- T = {f.T:.4g} K\n")
            w(f"- P = {f.P:.4g} Pa\n")
            rows = [[s, _fmt_flow(v)] for s, v in f.flows.items()]
            w("\n")
            w(_table(["species", "F (mol/s)"], rows))
            w("\n\n")

    # --- Results
    if report.results:
        w("## Solved Streams\n\n")
        for r in report.results:
            w(f"### {r.name}\n\n")
            w(f"- T = {r.T:.4g} K\n")
            w(f"- P = {r.P:.4g} Pa\n")
            rows = [[s, _fmt_flow(v)] for s, v in r.flows.items()]
            w("\n")
            w(_table(["species", "F (mol/s)"], rows))
            w("\n\n")

    # --- Balance
    if report.balance_checks:
        w("## Mass Balance Closure\n\n")
        rows = [
            [b.species, f"{b.feed_total:.6g}", f"{b.outlet_total:.6g}", f"{b.residual:+.3g}"]
            for b in report.balance_checks
        ]
        w(_table(["species", "feed total", "outlet total", "residual"], rows))
        w("\n\n")

    # --- Recycle convergence (section F)
    if report.convergence is not None:
        c = report.convergence
        w("## Recycle Convergence\n\n")
        w(f"- Method: {c.method}\n")
        w(f"- Tear streams: {', '.join(c.tear_streams) or '(none)'}\n")
        if c.iterations is not None:
            w(f"- Iterations: {c.iterations}\n")
        if c.residual is not None:
            w(f"- Final residual: {c.residual:.3g}\n")
        if c.tolerance is not None:
            w(f"- Tolerance: {c.tolerance:.3g}\n")
        if c.converged is not None:
            w(f"- Converged: {'yes' if c.converged else 'no'}\n")
        w("\n")

    # --- Optimization / sensitivity (section G)
    if report.optimization is not None:
        o = report.optimization
        w("## Optimization and Sensitivity\n\n")
        units = f" {o.objective_units}" if o.objective_units else ""
        w(f"- Objective: {o.objective_name} ({o.sense})\n")
        w(f"- Value at design point: {o.objective_value:.6g}{units}\n")
        if o.objective_source:
            w(f"- Source: `{o.objective_source}`\n")
        w("\n")

        w("**Decision variables**\n\n")
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
        w(_table(
            ["variable", "value", "bounds", "dJ/dx", "elasticity"], rows
        ))
        w("\n\n")

        if o.tornado:
            w("**Sensitivity tornado** (objective swing over each bound)\n\n")
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
            w(_table(
                ["variable", "low", "high", "J(low)", "J(high)", "swing"],
                trows,
            ))
            w("\n\n")

        if o.notes:
            for n in o.notes:
                w(f"- {n}\n")
            w("\n")

    if report.notes:
        w("## Notes\n\n")
        for n in report.notes:
            w(f"- {n}\n")
        w("\n")

    return out.getvalue()
