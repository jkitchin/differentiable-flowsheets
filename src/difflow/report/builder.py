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
    ConvergenceInfo,
    DecisionVariable,
    Edge,
    FeedSummary,
    OptimizationReport,
    ParamRow,
    RecycleInfo,
    Report,
    ResultSummary,
    SpeciesRow,
    TornadoRow,
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
    flowsheet, streams: dict[str, Any] | None, db_access: Any | None = None
) -> list[SpeciesRow]:
    """Gather thermo + critical properties for every species in any stream.

    When ``db_access`` is a tracker from
    :func:`difflow.database.track_database_access`, each row's ``accessed``
    flag reports whether the solve actually touched that species' database
    entry (precise provenance); otherwise ``accessed`` stays ``None``.
    """
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
        if db_access is not None:
            try:
                row.accessed = db_access.was_accessed(name)
            except Exception:
                row.accessed = None
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


def _convergence_info(flowsheet, streams: dict[str, Any] | None) -> ConvergenceInfo | None:
    """Capture recycle-loop convergence diagnostics from the last solve.

    Reads the ``last_solve_*`` attributes the flowsheet records during
    :meth:`~difflow.flowsheet.Flowsheet.solve`.  Returns ``None`` when no
    streams are supplied or the flowsheet has not been solved.
    """
    if streams is None:
        return None
    method = getattr(flowsheet, "last_solve_method", None)
    if method is None:
        return None
    return ConvergenceInfo(
        method=method,
        iterations=getattr(flowsheet, "last_solve_iterations", None),
        residual=getattr(flowsheet, "last_solve_residual", None),
        tolerance=getattr(flowsheet, "last_solve_tol", None),
        converged=getattr(flowsheet, "last_solve_converged", None),
        tear_streams=list(getattr(flowsheet, "last_solve_tear_streams", []) or []),
    )


def _objective_source(objective: Any, explicit: str | None) -> str:
    """Best-effort description of where an objective function is defined."""
    if explicit:
        return explicit
    name = getattr(objective, "__name__", None)
    module = getattr(objective, "__module__", None)
    if name and module:
        return f"{module}.{name}"
    if name:
        return name
    return ""


def build_optimization_report(
    objective: Any,
    design_point: dict[str, float],
    bounds: dict[str, tuple[float, float]] | None = None,
    objective_name: str = "Objective",
    objective_units: str = "",
    objective_source: str | None = None,
    sense: str = "minimize",
    notes: Iterable[str] | None = None,
) -> OptimizationReport:
    """Build the optimization / sensitivity section (report section G).

    Given the objective function and the design point (usually the optimum),
    this captures the objective value, the per-variable gradient
    ``dJ/dx`` (via JAX automatic differentiation), and — when bounds are
    supplied — a one-at-a-time tornado table of the objective's swing over
    each variable's range.  It performs no optimization itself; the caller
    passes the design point that a solver (or hand analysis) produced.

    Args:
        objective: callable mapping a decision-variable dict to a scalar.
            Must be JAX-differentiable for gradients to be captured; if it
            is not, gradients are reported as ``None`` and the rest of the
            section is still produced.
        design_point: decision variables at the reported point, e.g. the
            optimum ``{"V": 2.3, "reflux": 1.4}``.
        bounds: optional ``{name: (low, high)}`` used both to annotate each
            variable and to build the tornado table.
        objective_name: human-readable objective name.
        objective_units: units of the objective, if any.
        objective_source: where the objective is defined; defaults to the
            callable's ``module.__name__``.
        sense: "minimize" or "maximize" (informational).
        notes: optional free-form notes.

    Returns:
        A populated :class:`~difflow.report.ir.OptimizationReport`.
    """
    import jax
    import jax.numpy as jnp

    names = list(design_point.keys())
    point = {k: float(design_point[k]) for k in names}

    def _scalar(d: dict[str, Any]) -> Any:
        return jnp.asarray(objective(d)).reshape(())

    obj_value = float(_scalar(point))

    # Per-variable gradient via autodiff over the decision-variable pytree.
    grads: dict[str, float | None] = {k: None for k in names}
    try:
        jpoint = {k: jnp.asarray(point[k], dtype=float) for k in names}
        graddict = jax.grad(lambda d: _scalar(d))(jpoint)
        for k in names:
            g = graddict.get(k)
            if g is not None and jnp.all(jnp.isfinite(g)):
                grads[k] = float(g)
    except Exception:
        pass

    variables: list[DecisionVariable] = []
    for k in names:
        lo = hi = None
        if bounds and k in bounds:
            lo, hi = float(bounds[k][0]), float(bounds[k][1])
        g = grads[k]
        elasticity = None
        if g is not None and obj_value != 0.0:
            elasticity = g * point[k] / obj_value
        variables.append(
            DecisionVariable(
                name=k,
                value=point[k],
                lower=lo,
                upper=hi,
                gradient=g,
                elasticity=elasticity,
            )
        )

    tornado: list[TornadoRow] | None = None
    if bounds:
        rows: list[TornadoRow] = []
        for k in names:
            if k not in bounds:
                continue
            lo, hi = float(bounds[k][0]), float(bounds[k][1])
            low_pt = dict(point)
            low_pt[k] = lo
            high_pt = dict(point)
            high_pt[k] = hi
            try:
                y_lo = float(_scalar(low_pt))
                y_hi = float(_scalar(high_pt))
            except Exception:
                continue
            rows.append(
                TornadoRow(
                    variable=k,
                    low_value=lo,
                    high_value=hi,
                    low_output=y_lo,
                    high_output=y_hi,
                    swing=abs(y_hi - y_lo),
                )
            )
        rows.sort(key=lambda r: r.swing, reverse=True)
        tornado = rows or None

    return OptimizationReport(
        objective_name=objective_name,
        objective_value=obj_value,
        objective_units=objective_units,
        objective_source=_objective_source(objective, objective_source),
        sense=sense,
        variables=variables,
        tornado=tornado,
        notes=list(notes) if notes else [],
    )


def build_report(
    flowsheet,
    streams: dict[str, Any] | None = None,
    include_git: bool = True,
    optimization: OptimizationReport | None = None,
    db_access: Any | None = None,
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
        optimization: Optional :class:`~difflow.report.ir.OptimizationReport`
            (build one with :func:`build_optimization_report`) to include the
            optimization / sensitivity section (report section G).
        db_access: Optional tracker from
            :func:`difflow.database.track_database_access` used around the
            solve.  When given, each species row is annotated with whether
            its database entry was actually accessed (precise provenance).
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

    species = _collect_species(flowsheet, streams, db_access=db_access)
    balance_checks = _balance_checks(flowsheet, streams)
    convergence = _convergence_info(flowsheet, streams)

    return Report(
        provenance=collect_provenance(include_git=include_git),
        topology=topology,
        units=units_ir,
        species=species,
        feeds=feeds,
        results=results,
        balance_checks=balance_checks,
        convergence=convergence,
        optimization=optimization,
        notes=list(notes) if notes else [],
    )
