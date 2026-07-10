"""Structured diff between two :class:`~difflow.report.ir.Report` objects.

Comparing two reports is how a reader sees what changed between two runs:
a design tweak, a parameter sweep step, a difflow-version bump.  The diff is
computed at unit / parameter / stream granularity and renders to Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from difflow.report.ir import Report


def _num_changed(a: Any, b: Any, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """True if two values differ (numeric compare with tolerance, else !=)."""
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) > atol + rtol * abs(fb)
    except (TypeError, ValueError):
        return a != b


@dataclass
class ParamChange:
    """A single changed parameter of a unit (or objective/provenance field)."""

    name: str
    before: str
    after: str


@dataclass
class UnitParamChanges:
    """Parameter changes for one unit present in both reports."""

    unit: str
    changes: list[ParamChange] = field(default_factory=list)


@dataclass
class StreamChange:
    """Changed quantities (T, P, flows) of a feed or result stream."""

    stream: str
    changes: list[ParamChange] = field(default_factory=list)


@dataclass
class ReportDiff:
    """Structured difference between a "before" and an "after" report."""

    provenance: list[ParamChange] = field(default_factory=list)
    units_added: list[str] = field(default_factory=list)
    units_removed: list[str] = field(default_factory=list)
    unit_param_changes: list[UnitParamChanges] = field(default_factory=list)
    feed_changes: list[StreamChange] = field(default_factory=list)
    result_changes: list[StreamChange] = field(default_factory=list)
    objective_changes: list[ParamChange] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the two reports are equivalent in every compared section."""
        return not any([
            self.provenance,
            self.units_added,
            self.units_removed,
            self.unit_param_changes,
            self.feed_changes,
            self.result_changes,
            self.objective_changes,
        ])

    def to_markdown(self) -> str:
        from difflow.report.renderers.diff_markdown import diff_to_markdown

        return diff_to_markdown(self)


def _stream_changes(a_list, b_list) -> list[StreamChange]:
    """Diff two lists of Feed/Result summaries by name (T, P, per-species flow)."""
    a_by = {s.name: s for s in (a_list or [])}
    b_by = {s.name: s for s in (b_list or [])}
    out: list[StreamChange] = []
    for name in sorted(set(a_by) & set(b_by)):
        sa, sb = a_by[name], b_by[name]
        changes: list[ParamChange] = []
        if _num_changed(sa.T, sb.T):
            changes.append(ParamChange("T", f"{sa.T:.6g}", f"{sb.T:.6g}"))
        if _num_changed(sa.P, sb.P):
            changes.append(ParamChange("P", f"{sa.P:.6g}", f"{sb.P:.6g}"))
        for sp in sorted(set(sa.flows) | set(sb.flows)):
            va = sa.flows.get(sp, 0.0)
            vb = sb.flows.get(sp, 0.0)
            if _num_changed(va, vb):
                changes.append(ParamChange(f"F_{sp}", f"{va:.6g}", f"{vb:.6g}"))
        if changes:
            out.append(StreamChange(stream=name, changes=changes))
    return out


def diff_reports(before: Report, after: Report) -> ReportDiff:
    """Compute a :class:`ReportDiff` between two reports.

    Args:
        before: the baseline report.
        after: the report to compare against the baseline.

    Returns:
        A :class:`ReportDiff`; use ``.is_empty`` to test for equivalence and
        ``.to_markdown()`` to render.
    """
    diff = ReportDiff()

    # Provenance (scalar fields).
    pa, pb = before.provenance, after.provenance
    for fname in (
        "difflow_version", "jax_version", "jax_backend", "jax_x64",
        "python_version", "platform", "git_commit", "git_dirty",
    ):
        va, vb = getattr(pa, fname, None), getattr(pb, fname, None)
        if va != vb:
            diff.provenance.append(ParamChange(fname, str(va), str(vb)))
    for name in sorted(set(pa.plugin_versions) | set(pb.plugin_versions)):
        va = pa.plugin_versions.get(name)
        vb = pb.plugin_versions.get(name)
        if va != vb:
            diff.provenance.append(ParamChange(f"plugin:{name}", str(va), str(vb)))

    # Units added / removed / param-changed.
    a_units = {u.name: u for u in before.units}
    b_units = {u.name: u for u in after.units}
    diff.units_added = sorted(set(b_units) - set(a_units))
    diff.units_removed = sorted(set(a_units) - set(b_units))
    for name in sorted(set(a_units) & set(b_units)):
        ua, ub = a_units[name], b_units[name]
        a_params = {p.name: p.value_repr for p in ua.parameters}
        b_params = {p.name: p.value_repr for p in ub.parameters}
        changes: list[ParamChange] = []
        for pname in sorted(set(a_params) | set(b_params)):
            va = a_params.get(pname, "(absent)")
            vb = b_params.get(pname, "(absent)")
            if va != vb:
                changes.append(ParamChange(pname, va, vb))
        if changes:
            diff.unit_param_changes.append(UnitParamChanges(unit=name, changes=changes))

    # Feed and result streams.
    diff.feed_changes = _stream_changes(before.feeds, after.feeds)
    diff.result_changes = _stream_changes(before.results, after.results)

    # Objective (section G).
    oa, ob = before.optimization, after.optimization
    if oa is not None and ob is not None:
        if oa.objective_name != ob.objective_name:
            diff.objective_changes.append(
                ParamChange("name", oa.objective_name, ob.objective_name)
            )
        if _num_changed(oa.objective_value, ob.objective_value):
            diff.objective_changes.append(
                ParamChange(
                    "value",
                    f"{oa.objective_value:.6g}",
                    f"{ob.objective_value:.6g}",
                )
            )
        a_vars = {v.name: v for v in oa.variables}
        b_vars = {v.name: v for v in ob.variables}
        for vname in sorted(set(a_vars) & set(b_vars)):
            va, vb = a_vars[vname].value, b_vars[vname].value
            if _num_changed(va, vb):
                diff.objective_changes.append(
                    ParamChange(f"var:{vname}", f"{va:.6g}", f"{vb:.6g}")
                )
    elif (oa is None) != (ob is None):
        diff.objective_changes.append(
            ParamChange(
                "optimization",
                "absent" if oa is None else oa.objective_name,
                "absent" if ob is None else ob.objective_name,
            )
        )

    return diff
