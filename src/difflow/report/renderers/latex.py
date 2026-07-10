"""LaTeX renderer for :class:`~difflow.report.ir.Report`.

Produces a fragment suitable for inclusion in a ``\\documentclass{article}``
manuscript that loads ``amsmath`` and ``booktabs``.
"""

from __future__ import annotations

from io import StringIO

from difflow.report.ir import Report


_LATEX_ESCAPE = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _esc(s: str) -> str:
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPE.get(ch, ch))
    return "".join(out)


def _tabular(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    spec = "l" * len(header)
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(_esc(h) for h in header) + r" \\")
    out.append(r"\midrule")
    for row in rows:
        out.append(" & ".join(_esc(c) for c in row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def to_latex(report: Report) -> str:
    """Render a Report as a LaTeX fragment."""
    out = StringIO()
    w = out.write
    w("% difflow flowsheet report\n")
    w(r"\section*{Flowsheet Report}" + "\n\n")

    p = report.provenance
    w(r"\subsection*{Provenance}" + "\n")
    rows = [
        ["difflow", p.difflow_version],
        ["jax", f"{p.jax_version} ({p.jax_backend}, x64={p.jax_x64})"],
        ["python", p.python_version],
        ["platform", p.platform],
        ["timestamp", p.timestamp],
    ]
    for name, ver in p.plugin_versions.items():
        rows.append([name, ver])
    if p.git_commit:
        rows.append(["git", f"{p.git_commit[:12]}{' (dirty)' if p.git_dirty else ''}"])
    w(_tabular(["Field", "Value"], rows))
    w("\n\n")

    # Units
    w(r"\subsection*{Unit Operations}" + "\n")
    for u in report.units:
        w(r"\paragraph{" + _esc(f"{u.name} — {u.type} ({u.plugin})") + "}\n")
        if u.description:
            w(_esc(u.description) + "\n\n")
        if u.equations:
            for eq in u.equations:
                w(r"\begin{equation*}" + "\n" + eq + "\n" + r"\end{equation*}" + "\n")
        if u.assumptions:
            w(r"\textbf{Assumptions.} \begin{itemize}" + "\n")
            for a in u.assumptions:
                w(r"\item " + _esc(a) + "\n")
            w(r"\end{itemize}" + "\n")
        if u.parameters:
            prows = [
                [pr.name, pr.symbol, pr.units or "-", pr.value_repr]
                for pr in u.parameters
            ]
            w(r"\textbf{Parameters.}" + "\n\n")
            w(_tabular(["name", "symbol", "units", "value"], prows))
            w("\n\n")
        if u.references:
            w(r"\textbf{References.} \begin{itemize}" + "\n")
            for r in u.references:
                w(r"\item " + _esc(r) + "\n")
            w(r"\end{itemize}" + "\n")

    if report.species:
        w(r"\subsection*{Species and Thermophysical Data}" + "\n")
        tracked = any(s.accessed is not None for s in report.species)
        header = ["species", "MW (g/mol)", "Tc (K)", "Pc (Pa)", "omega", "Hf (J/mol)", "source"]
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
        w(_tabular(header, rows))
        w("\n\n")

    if report.feeds:
        w(r"\subsection*{Feed Streams}" + "\n")
        for f in report.feeds:
            w(r"\paragraph{" + _esc(f.name) + "}\n")
            w(f"$T = {f.T:.4g}$ K, $P = {f.P:.4g}$ Pa.\n\n")
            rows = [[s, f"{v:.4g}"] for s, v in f.flows.items()]
            w(_tabular(["species", "F (mol/s)"], rows))
            w("\n\n")

    if report.results:
        w(r"\subsection*{Solved Streams}" + "\n")
        for r in report.results:
            w(r"\paragraph{" + _esc(r.name) + "}\n")
            w(f"$T = {r.T:.4g}$ K, $P = {r.P:.4g}$ Pa.\n\n")
            rows = [[s, f"{v:.4g}"] for s, v in r.flows.items()]
            w(_tabular(["species", "F (mol/s)"], rows))
            w("\n\n")

    if report.balance_checks:
        w(r"\subsection*{Mass Balance Closure}" + "\n")
        rows = [
            [b.species, f"{b.feed_total:.6g}", f"{b.outlet_total:.6g}", f"{b.residual:+.3g}"]
            for b in report.balance_checks
        ]
        w(_tabular(["species", "feed total", "outlet total", "residual"], rows))
        w("\n\n")

    if report.convergence is not None:
        c = report.convergence
        w(r"\subsection*{Recycle Convergence}" + "\n")
        conv_rows = [
            ["method", c.method],
            ["tear streams", ", ".join(c.tear_streams) or "(none)"],
            ["iterations", "-" if c.iterations is None else str(c.iterations)],
            ["residual", "-" if c.residual is None else f"{c.residual:.3g}"],
            ["tolerance", "-" if c.tolerance is None else f"{c.tolerance:.3g}"],
            ["converged", {True: "yes", False: "no", None: "-"}[c.converged]],
        ]
        w(_tabular(["field", "value"], conv_rows))
        w("\n\n")

    if report.optimization is not None:
        o = report.optimization
        w(r"\subsection*{Optimization and Sensitivity}" + "\n")
        units = f" {o.objective_units}" if o.objective_units else ""
        w(
            _esc(f"Objective: {o.objective_name} ({o.sense}); ")
            + f"value $= {o.objective_value:.6g}$"
            + _esc(units)
            + ".\n\n"
        )
        if o.objective_source:
            w(r"\textbf{Source.} \texttt{" + _esc(o.objective_source) + "}\n\n")

        w(r"\textbf{Decision variables.}" + "\n\n")
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
        w(_tabular(["variable", "value", "bounds", "dJ/dx", "elasticity"], rows))
        w("\n\n")

        if o.tornado:
            w(r"\textbf{Sensitivity tornado.}" + "\n\n")
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
            w(_tabular(
                ["variable", "low", "high", "J(low)", "J(high)", "swing"], trows
            ))
            w("\n\n")

    return out.getvalue()
