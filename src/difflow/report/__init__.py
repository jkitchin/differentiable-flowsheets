"""Self-documenting reports for difflow flowsheets.

See issue #162 for motivation.  Typical usage::

    from difflow import Flowsheet, report

    fs = Flowsheet(...)
    # ... add feeds, units, recycles
    streams = fs.solve()

    rep = fs.report(streams)
    print(rep.to_markdown())
    open("report.json", "w").write(rep.to_json())

The ``Report`` intermediate representation lives in :mod:`difflow.report.ir`;
renderers live in :mod:`difflow.report.renderers`; the introspection step
lives in :mod:`difflow.report.builder`.
"""

from __future__ import annotations

from typing import Any

from difflow.report.ir import (
    BalanceCheck,
    Edge,
    FeedSummary,
    ParamRow,
    Provenance,
    RecycleInfo,
    Report,
    ResultSummary,
    SpeciesRow,
    Topology,
    UnitReport,
)
from difflow.report.metadata import UnitMetadata, get_metadata, has_structured_metadata
from difflow.report.builder import build_report
from difflow.report.renderers.markdown import to_markdown
from difflow.report.renderers.json_renderer import to_json
from difflow.report.renderers.latex import to_latex
from difflow.report.renderers.html import to_html


def report(flowsheet, streams: dict[str, Any] | None = None, **kwargs) -> Report:
    """Build a :class:`Report` for ``flowsheet``.

    Thin top-level wrapper around :func:`difflow.report.builder.build_report`.

    Args:
        flowsheet: A :class:`~difflow.flowsheet.Flowsheet` instance.
        streams: Optional solved streams (``fs.solve()``) to include results.
        **kwargs: Forwarded to :func:`build_report`.

    Returns:
        :class:`Report`
    """
    return build_report(flowsheet, streams=streams, **kwargs)


__all__ = [
    "Report",
    "Provenance",
    "Topology",
    "UnitReport",
    "ParamRow",
    "SpeciesRow",
    "FeedSummary",
    "ResultSummary",
    "RecycleInfo",
    "Edge",
    "BalanceCheck",
    "UnitMetadata",
    "get_metadata",
    "has_structured_metadata",
    "build_report",
    "report",
    "to_markdown",
    "to_json",
    "to_latex",
    "to_html",
]
