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
        w(_tabular(
            ["species", "MW (g/mol)", "Tc (K)", "Pc (Pa)", "omega", "Hf (J/mol)", "source"],
            rows,
        ))
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

    return out.getvalue()
