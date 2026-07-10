"""Markdown renderer for :class:`~difflow.report.diff.ReportDiff`."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difflow.report.diff import ReportDiff


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def diff_to_markdown(diff: "ReportDiff") -> str:
    """Render a :class:`ReportDiff` as Markdown."""
    out = StringIO()
    w = out.write
    w("# Report Diff\n\n")

    if diff.is_empty:
        w("No differences in the compared sections.\n")
        return out.getvalue()

    if diff.provenance:
        w("## Provenance\n\n")
        w(_table(["field", "before", "after"],
                 [[c.name, c.before, c.after] for c in diff.provenance]))
        w("\n\n")

    if diff.units_added or diff.units_removed:
        w("## Units\n\n")
        if diff.units_added:
            w(f"- Added: {', '.join(diff.units_added)}\n")
        if diff.units_removed:
            w(f"- Removed: {', '.join(diff.units_removed)}\n")
        w("\n")

    if diff.unit_param_changes:
        w("## Parameter Changes\n\n")
        for uc in diff.unit_param_changes:
            w(f"### {uc.unit}\n\n")
            w(_table(["parameter", "before", "after"],
                     [[c.name, c.before, c.after] for c in uc.changes]))
            w("\n\n")

    for title, changes in (
        ("Feed Changes", diff.feed_changes),
        ("Result Changes", diff.result_changes),
    ):
        if changes:
            w(f"## {title}\n\n")
            for sc in changes:
                w(f"### {sc.stream}\n\n")
                w(_table(["quantity", "before", "after"],
                         [[c.name, c.before, c.after] for c in sc.changes]))
                w("\n\n")

    if diff.objective_changes:
        w("## Optimization\n\n")
        w(_table(["field", "before", "after"],
                 [[c.name, c.before, c.after] for c in diff.objective_changes]))
        w("\n\n")

    return out.getvalue()
