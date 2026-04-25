"""Build a :class:`Report` from a :class:`Flowsheet`.

The builder introspects a flowsheet's units, feeds, and (optionally) solved
streams and returns a fully populated :class:`~difflow.report.ir.Report`
object.  It relies on :mod:`difflow.report.metadata` to obtain
equations / assumptions / references for each unit, falling back to
docstring parsing when a class has no structured metadata attributes.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields as dc_fields
from typing import Any, Iterable

from difflow.report.ir import (
    BalanceCheck,
    Edge,
    FeedSummary,
    ParamRow,
    RecycleInfo,
    Report,
    ResultSummary,
    SpeciesRow,
    Topology,
    UnitReport,
)
from difflow.report.metadata import get_metadata
from difflow.report.provenance import collect_provenance


def _fmt_value(value: Any) -> str:
    """Compact human-readable repr of a parameter value."""
    if value is None:
        return "None"
    if callable(value) and hasattr(value, "__name__"):
        return f"<fn {value.__name__}>"
    if hasattr(value, "shape"):
        shape = tuple(value.shape)
        if shape == ():
            try:
                return f"{float(value):.6g}"
            except Exception:
                return repr(value)
        return f"Array{list(shape)}"
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"{type(value).__name__}[{len(value)}]"
        return repr(value)
    if isinstance(value, dict):
        if len(value) > 8:
            return f"dict[{len(value)}]"
        return repr(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value)


def _params_rows(operation: Any, metadata) -> list[ParamRow]:
    """Build parameter rows from ``operation.params`` (a ParamsMixin dataclass)."""
    p = getattr(operation, "params", None)
    if p is None or not is_dataclass(p):
        return []
    rows: list[ParamRow] = []
    for f in dc_fields(p):
        value = getattr(p, f.name)
        symbol = metadata.parameter_symbols.get(f.name, f.name)
        units = metadata.parameter_units.get(f.name, "")
        rows.append(
            ParamRow(
                name=f.name,
                symbol=symbol,
                units=units,
                value_repr=_fmt_value(value),
            )
        )
    return rows


def _plugin_for_cls(cls: type) -> str:
    """Return the plugin name a unit class belongs to."""
    module = cls.__module__ or ""
    if module.startswith("difflow_bio"):
        return "difflow_bio"
    if module.startswith("difflow_ree"):
        return "difflow_ree"
    if module.startswith("difflow_cc"):
        return "difflow_cc"
    return "core"


def _stream_flows(stream: dict[str, Any]) -> dict[str, float]:
    flows = {}
    for key, value in stream.items():
        if not isinstance(key, str) or not key.startswith("F_"):
            continue
        try:
            flows[key[2:]] = float(value)
        except Exception:
            continue
    return flows


def _summarise_stream(name: str, stream: dict[str, Any]) -> ResultSummary | None:
    try:
        T = float(stream["T"])
        P = float(stream["P"])
    except Exception:
        return None
    return ResultSummary(name=name, T=T, P=P, flows=_stream_flows(stream))


def _feed_summary(name: str, stream: dict[str, Any]) -> FeedSummary:
    T = float(stream["T"])
    P = float(stream["P"])
    return FeedSummary(name=name, T=T, P=P, flows=_stream_flows(stream))


def _collect_species(
    flowsheet, streams: dict[str, Any] | None
) -> list[SpeciesRow]:
    """Gather thermo + critical properties for every species in any stream."""
    from difflow import database as _db

    names: list[str] = list(getattr(flowsheet, "species_order", []) or [])
    seen = set(names)
    # Harvest additional species that may appear in feeds/streams but not
    # the canonical order (defensive — usually identical).
    pools: list[dict[str, Any]] = []
    if getattr(flowsheet, "feeds", None):
        pools.extend(flowsheet.feeds.values())
    if streams:
        pools.extend(streams.values())
    for stream in pools:
        for key in stream.keys():
            if isinstance(key, str) and key.startswith("F_"):
                s = key[2:]
                if s not in seen:
                    seen.add(s)
                    names.append(s)

    rows: list[SpeciesRow] = []
    citations = getattr(_db, "SOURCE_CITATIONS", {}) or {}
    for name in names:
        row = SpeciesRow(name=name, source=citations.get(name.lower(), ""))
        # Critical properties
        try:
            crit = _db.get_critical_props(name)
            row.Tc = float(crit.Tc)
            row.Pc = float(crit.Pc)
            row.omega = float(crit.omega)
            row.MW = float(crit.MW)
        except Exception:
            pass
        # Ideal thermo
        try:
            sp = _db.get_species_data(name)
            if row.MW is None:
                row.MW = float(sp.MW)
            row.Cp_coeffs = tuple(sp.Cp_coeffs)
            row.Hvap_coeffs = tuple(sp.Hvap_coeffs)
            row.antoine_coeffs = tuple(sp.antoine_coeffs)
            row.Hf = float(getattr(sp, "Hf", 0.0))
        except Exception:
            pass
        rows.append(row)
    return rows


def _edges_from_units(units: Iterable) -> list[Edge]:
    # Map stream name -> producing unit
    producer: dict[str, str] = {}
    for u in units:
        for out in u.outlet_names:
            producer[out] = u.name
    edges: list[Edge] = []
    for u in units:
        for inlet in u.inlet_names:
            edges.append(Edge(stream=inlet, source=producer.get(inlet), target=u.name))
    return edges


def _balance_checks(
    flowsheet,
    streams: dict[str, Any] | None,
) -> list[BalanceCheck] | None:
    if not streams or not flowsheet.feeds:
        return None

    feed_totals: dict[str, float] = {s: 0.0 for s in flowsheet.species_order}
    for stream in flowsheet.feeds.values():
        for s in flowsheet.species_order:
            key = f"F_{s}"
            if key in stream:
                try:
                    feed_totals[s] += float(stream[key])
                except Exception:
                    pass

    # "Terminal" outlet streams: produced by some unit, consumed by none (i.e.,
    # not an inlet to any unit, and not a recycle source).
    consumed = set()
    for unit in flowsheet.units:
        for inlet in unit.inlet_names:
            consumed.add(inlet)
    for source in flowsheet.recycles.keys():
        consumed.add(source)
    produced = set()
    for unit in flowsheet.units:
        for out in unit.outlet_names:
            produced.add(out)
    terminal = [s for s in produced if s not in consumed]

    outlet_totals: dict[str, float] = {s: 0.0 for s in flowsheet.species_order}
    for name in terminal:
        stream = streams.get(name)
        if stream is None:
            continue
        for s in flowsheet.species_order:
            key = f"F_{s}"
            if key in stream:
                try:
                    outlet_totals[s] += float(stream[key])
                except Exception:
                    pass

    return [
        BalanceCheck(
            species=s,
            feed_total=feed_totals[s],
            outlet_total=outlet_totals[s],
            residual=outlet_totals[s] - feed_totals[s],
        )
        for s in flowsheet.species_order
    ]


def build_report(
    flowsheet,
    streams: dict[str, Any] | None = None,
    include_git: bool = True,
    notes: Iterable[str] | None = None,
) -> Report:
    """Build a :class:`Report` from a flowsheet and (optionally) solved streams.

    Args:
        flowsheet: A :class:`~difflow.flowsheet.Flowsheet` instance.
        streams: Optional dict of solved streams from ``fs.solve()``.  When
            omitted, the returned report contains configuration-only sections
            (topology, units, species, feeds) and the ``results`` /
            ``balance_checks`` fields are ``None``.
        include_git: Whether to capture git commit / dirty-flag in provenance.
        notes: Optional free-form notes to attach to the report.

    Returns:
        A fully populated :class:`Report`.
    """
    units_ir: list[UnitReport] = []
    for unit in flowsheet.units:
        op = unit.operation
        op_cls = op if isinstance(op, type) else op.__class__
        meta = get_metadata(op_cls)
        params = _params_rows(op, meta)
        units_ir.append(
            UnitReport(
                name=unit.name,
                type=op_cls.__name__,
                plugin=_plugin_for_cls(op_cls),
                symbol=meta.symbol,
                description=meta.description,
                equations=meta.equations,
                assumptions=meta.assumptions,
                references=meta.references,
                parameters=params,
                inlet_names=list(unit.inlet_names),
                outlet_names=list(unit.outlet_names),
                numerical_method=meta.numerical_method,
            )
        )

    topology = Topology(
        units=[u.name for u in flowsheet.units],
        edges=_edges_from_units(flowsheet.units),
        recycles=[
            RecycleInfo(source_stream=src, dest_stream=dst)
            for src, dst in flowsheet.recycles.items()
        ],
        species_order=list(flowsheet.species_order),
    )

    feeds = [_feed_summary(name, s) for name, s in flowsheet.feeds.items()]

    results: list[ResultSummary] | None = None
    if streams is not None:
        results = []
        for name, stream in streams.items():
            if name in flowsheet.feeds:
                continue
            summary = _summarise_stream(name, stream)
            if summary is not None:
                results.append(summary)

    species = _collect_species(flowsheet, streams)
    balance_checks = _balance_checks(flowsheet, streams)

    return Report(
        provenance=collect_provenance(include_git=include_git),
        topology=topology,
        units=units_ir,
        species=species,
        feeds=feeds,
        results=results,
        balance_checks=balance_checks,
        notes=list(notes) if notes else [],
    )
