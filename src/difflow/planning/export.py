"""Export delta vectors to LP-style planning systems.

A delta-base plan is only useful inside difflow until its coefficients can be
handed to whatever already runs the plan -- a refinery LP, a spreadsheet, a
Pyomo or GAMS model, an optimisation group that will never import JAX.  This
module is that hand-off.  It has one intermediate representation,
:class:`DeltaVectorSet`, and several renderers over it, following the same
"structured IR plus renderers" shape as :mod:`difflow.report`.

The IR exists because the numbers alone are not enough.  A Jacobian entry
means nothing without the base case it was taken at, the names and units of
its rows and columns, and the radius over which the first-order model was ever
meant to hold.  :class:`~difflow.planning.linearize.Linearization` carries the
numbers, :class:`~difflow.planning.block.Block` carries the names, bounds and
units, and the planner carries the radius; joining the three is most of what
this module does.

Formats:

* :func:`write_json` -- one self-describing manifest.  Read it with anything.
* :func:`write_csv` -- one Jacobian matrix per block plus base, bounds, links,
  specs and duals tables.  This is the shape a planning engineer already
  recognises: a shift-vector table.
* :func:`write_lp` / :func:`write_mps` -- the assembled LP itself, in CPLEX LP
  or free MPS form, for a solver or a planning system that reads them.
* :func:`write_iterations_csv` -- the trust-region audit trail, so the
  provenance of the base case travels with the coefficients.

Example:
    >>> res = DeltaBasePlanner(net, prices).solve()      # doctest: +SKIP
    >>> dvs = DeltaVectorSet.from_result(res)            # doctest: +SKIP
    >>> write_json(dvs, "plan.json")                     # doctest: +SKIP
    >>> write_csv(dvs, "plan_tables/")                   # doctest: +SKIP
    >>> write_mps(res.lp_model, "plan.mps")              # doctest: +SKIP
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from difflow.planning.assemble import trust_region_bounds
from difflow.planning.block import Block
from difflow.planning.linearize import Linearization
from difflow.report.renderers.json_renderer import _Encoder

__all__ = [
    "DeltaVector",
    "DeltaVectorSet",
    "sanitize_name",
    "write_json",
    "write_csv",
    "write_lp",
    "write_mps",
    "write_iterations_csv",
]


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------

_LP_SAFE = set("abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def sanitize_name(name: str) -> str:
    """Make a qualified variable name safe for LP and MPS files.

    difflow's column names carry structure -- ``"ngl.residue_F"``,
    ``"slack[cap]"``, ``"feed:F1.total_flow"`` -- and the ``.``, ``[``, ``]``
    and ``:`` in them are not portable across LP/MPS readers.  Every
    non-alphanumeric character becomes ``_`` and a leading digit is prefixed,
    which is lossy; the mapping is therefore recorded in the JSON manifest and
    in ``names.csv`` so the round trip is recoverable.

    Args:
        name: A difflow column or row name.

    Returns:
        A name matching ``[A-Za-z_][A-Za-z0-9_]*``.

    Example:
        >>> sanitize_name("ngl.residue_F")
        'ngl_residue_F'
        >>> sanitize_name("slack[cap<=]")
        'slack_cap___'
    """
    out = "".join(ch if ch in _LP_SAFE else "_" for ch in name)
    if not out or out[0].isdigit():
        out = "v" + out
    return out


def _name_map(names: Sequence[str]) -> dict[str, str]:
    """Sanitised name per original name, de-duplicated by suffix."""
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for n in names:
        s = sanitize_name(n)
        if s in seen:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        out[n] = s
    return out


# ---------------------------------------------------------------------------
# The IR
# ---------------------------------------------------------------------------


@dataclass
class DeltaVector:
    """One block's linearisation, with everything needed to use it elsewhere.

    Attributes:
        block: Block name.
        u_names: Qualified lever names, ``"<block>.<u>"``.
        y_names: Qualified output names, ``"<block>.<y>"``.
        u_units: Physical unit per lever, or ``None`` where unknown.
        y_units: Physical unit per output, or ``None`` where unknown.
        u0: Base-case lever values.
        y0: Block outputs at ``u0``.
        J: Delta vectors, ``J[i][j] = dy_i/du_j``, one list per output.
        lb: Physical lower bound per lever.
        ub: Physical upper bound per lever.
        radius: Trust-region radius the coefficients were used at, as a
            fraction of each lever's bound range.
        tr_lo: ``radius`` resolved to absolute lower bounds around ``u0``.
        tr_hi: ``radius`` resolved to absolute upper bounds around ``u0``.
        mode: AD mode used, ``"rev"`` or ``"fwd"``.
        phase: Phase-indicator values at ``u0``, or ``None``.
        scaled_J: Dimensionless Jacobian -- fractional change in output ``i``
            per full-bound-range move of lever ``j``.  This is the form a
            shift-vector table is normally read in.  ``None`` when the block
            has no finite bounds to scale by.
        source: Where the block came from, e.g. ``"flowsheet"``.
        u_keys: Original difflow keys behind ``u_names``, when known.
        y_keys: Original difflow keys behind ``y_names``, when known.
    """

    block: str
    u_names: list[str]
    y_names: list[str]
    u_units: list[str | None]
    y_units: list[str | None]
    u0: list[float]
    y0: list[float]
    J: list[list[float]]
    lb: list[float]
    ub: list[float]
    radius: float
    tr_lo: list[float]
    tr_hi: list[float]
    mode: str
    phase: list[float] | None = None
    scaled_J: list[list[float]] | None = None
    source: str | None = None
    u_keys: list[str] | None = None
    y_keys: list[str] | None = None

    @classmethod
    def build(cls, block: Block, lin: Linearization,
              radius: float = 0.0) -> "DeltaVector":
        """Join a block and its linearisation into one exportable record.

        Args:
            block: The block, for names, bounds and metadata units.
            lin: Its linearisation, for the numbers.
            radius: Trust-region radius to record and resolve to absolute
                bounds.  ``0.0`` means "no trust region declared".

        Returns:
            A :class:`DeltaVector`.
        """
        from difflow.planning.health import scaled_jacobian

        core = lin.to_dict(block.qualified_u(), block.qualified_y())
        u0 = np.asarray(core["u0"], dtype=float)
        lb = np.asarray(block.lb, dtype=float)
        ub = np.asarray(block.ub, dtype=float)
        tr_lo, tr_hi = trust_region_bounds(lb, ub, u0, float(radius))

        meta = block.metadata or {}
        n_u, n_y = block.n_u, block.n_y

        def _units(key: str, n: int) -> list[str | None]:
            vals = meta.get(key)
            if not vals or len(vals) != n:
                return [None] * n
            return [None if v is None else str(v) for v in vals]

        # health._u_scale already falls back to max(|u0|, 1) for unbounded
        # levers, so this is finite in every ordinary case; a non-finite y0
        # would make it meaningless, and None says so rather than shipping
        # infs into a planning table.
        Js = np.asarray(scaled_jacobian(block, lin), dtype=float)
        scaled = ([[float(v) for v in row] for row in Js]
                  if np.all(np.isfinite(Js)) else None)

        return cls(
            block=block.name,
            u_names=core["u_names"],
            y_names=core["y_names"],
            u_units=_units("u_units", n_u),
            y_units=_units("y_units", n_y),
            u0=core["u0"],
            y0=core["y0"],
            J=core["J"],
            lb=[float(v) for v in lb],
            ub=[float(v) for v in ub],
            radius=float(radius),
            tr_lo=[float(v) for v in tr_lo],
            tr_hi=[float(v) for v in tr_hi],
            mode=core["mode"],
            phase=core["phase"],
            scaled_J=scaled,
            source=meta.get("source"),
            u_keys=list(meta["u_keys"]) if "u_keys" in meta else None,
            y_keys=list(meta["y_keys"]) if "y_keys" in meta else None,
        )

    def predict(self, u: Sequence[float]) -> list[float]:
        """Evaluate the exported first-order model, ``y0 + J (u - u0)``.

        Present so a consumer can check the export reproduces the source
        model without importing JAX.
        """
        du = np.asarray(u, dtype=float) - np.asarray(self.u0, dtype=float)
        return (np.asarray(self.y0, dtype=float)
                + np.asarray(self.J, dtype=float) @ du).tolist()


@dataclass
class DeltaVectorSet:
    """A full delta-base model: every block, the links, prices and specs.

    Attributes:
        vectors: One :class:`DeltaVector` per block.
        links: ``(source output, target input)`` qualified name pairs.
        prices: Objective coefficients by qualified variable name.
        specs: Constraints as ``{name, coeffs, op, rhs, elastic, penalty,
            backoff}`` dicts.
        sense: ``"max"`` or ``"min"``.
        objective: Objective value at the base case, from the *nonlinear*
            model, or ``None``.
        values: Every network variable at the base case.
        duals: Shadow prices by row/bound name, when a solved LP was
            available.
        health: :class:`~difflow.planning.health.Finding` records, so a dead
            lever or an amplifying recycle travels with the coefficients
            rather than staying a local diagnostic.
        history: The trust-region audit trail, one dict per cycle.
        meta: Provenance -- difflow version, timestamp, convergence.
    """

    vectors: list[DeltaVector] = field(default_factory=list)
    links: list[list[str]] = field(default_factory=list)
    prices: dict[str, float] = field(default_factory=dict)
    specs: list[dict] = field(default_factory=list)
    sense: str = "max"
    objective: float | None = None
    values: dict[str, float] = field(default_factory=dict)
    duals: dict[str, dict[str, float]] | None = None
    health: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_result(cls, result, include_duals: bool = True,
                    include_health: bool = True) -> "DeltaVectorSet":
        """Build the export from a solved plan.

        Args:
            result: A :class:`~difflow.planning.planner.PlanResult`.
            include_duals: Solve the final LP once for its marginals.  See
                :attr:`~difflow.planning.planner.PlanResult.solution`.
            include_health: Run :func:`
                ~difflow.planning.health.check_delta_health` and carry the
                findings.

        Returns:
            A :class:`DeltaVectorSet`.
        """
        net = result.network
        planner = result.planner
        radius = float(result.radius)

        vectors = [
            DeltaVector.build(net.block(name), lin, radius)
            for name, lin in result.linearizations.items()
        ]

        notes: dict[str, str] = {}

        duals = None
        if include_duals:
            try:
                duals = _plain(result.duals)
            except Exception as exc:  # an infeasible or unsolved LP has none
                notes["duals_error"] = f"{type(exc).__name__}: {exc}"

        health: list[dict] = []
        if include_health:
            from difflow.planning.health import check_delta_health
            try:
                health = [asdict(f)
                          for f in check_delta_health(net).findings]
            except Exception as exc:  # diagnostics must never block an export
                notes["health_error"] = f"{type(exc).__name__}: {exc}"

        return cls(
            vectors=vectors,
            links=[[link.source, link.target] for link in net.links],
            prices={k: float(v) for k, v in planner.prices.items()},
            specs=[_spec_dict(s) for s in planner.specs],
            sense=planner.sense,
            objective=float(result.objective),
            values={k: float(v) for k, v in result.values.items()},
            duals=duals,
            health=health,
            history=[_iteration_dict(h) for h in result.history],
            meta=_meta(lp_symbols=_name_map(list(result.lp_model.columns)),
                       lp_sense=result.lp_model.sense,
                       lp_objective_offset=float(
                           result.lp_model.objective_offset),
                       converged=bool(result.converged),
                       reason=str(result.reason),
                       radius=radius,
                       merit=float(result.merit),
                       n_iterations=len(result.history),
                       violations={k: float(v)
                                   for k, v in result.violations.items()},
                       **notes),
        )

    @classmethod
    def from_block(cls, block: Block, lin: Linearization | None = None,
                   radius: float = 0.0, **meta: Any) -> "DeltaVectorSet":
        """Build the export from a single block, with no planner run.

        The common case for a simulation engineer: linearise one flowsheet
        and send the delta vectors to whoever owns the planning model.

        Args:
            block: The block to export.
            lin: Its linearisation.  Computed at ``block.u0`` if omitted.
            radius: Trust-region radius to record.
            **meta: Extra provenance recorded in :attr:`meta`.

        Returns:
            A :class:`DeltaVectorSet` with one vector and no links or specs.
        """
        if lin is None:
            from difflow.planning.linearize import linearize_block
            lin = linearize_block(block)
        return cls(vectors=[DeltaVector.build(block, lin, radius)],
                   meta=_meta(**meta))

    # -- access ----------------------------------------------------------

    def vector(self, block: str) -> DeltaVector:
        """The delta vector for one block."""
        for v in self.vectors:
            if v.block == block:
                return v
        raise KeyError(f"no block {block!r} in this export; "
                       f"have {[v.block for v in self.vectors]}")

    def to_dict(self) -> dict:
        """The whole export as plain JSON-safe Python."""
        return _plain(asdict(self))

    def __repr__(self) -> str:
        return (f"DeltaVectorSet(blocks={[v.block for v in self.vectors]}, "
                f"n_specs={len(self.specs)}, sense={self.sense!r})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain(obj):
    """Round-trip through the report JSON encoder to drop JAX/numpy types."""
    return json.loads(json.dumps(obj, cls=_Encoder))


def _spec_dict(spec) -> dict:
    """A Spec as plain data, including the ``coeffs`` that ``asdict`` misses.

    ``Spec.coeffs`` is set in ``__post_init__`` rather than declared as a
    field, so ``dataclasses.asdict`` silently drops it -- and it is the part
    an external LP actually needs.
    """
    return {
        "name": spec.name,
        "coeffs": {k: float(v) for k, v in spec.coeffs.items()},
        "op": spec.op,
        "rhs": float(spec.rhs),
        "elastic": bool(spec.elastic),
        "penalty": None if spec.penalty is None else float(spec.penalty),
        "backoff": float(spec.backoff),
    }


def _iteration_dict(it) -> dict:
    """One trust-region cycle as plain data."""
    return {
        "index": int(it.index),
        "radius": float(it.radius),
        "merit": float(it.merit),
        "predicted": float(it.predicted),
        "realised": float(it.realised),
        "rho": float(it.rho),
        "accepted": bool(it.accepted),
        "lp_status": str(it.lp_status),
    }


def _meta(**extra: Any) -> dict:
    """Provenance stamp."""
    try:
        from difflow import __version__ as version
    except Exception:
        version = "unknown"
    meta = {
        "generator": "difflow.planning.export",
        "difflow_version": version,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "format": "delta-vectors/1",
    }
    meta.update(extra)
    return meta


def _rows(path: str | os.PathLike, header: Sequence[str],
          rows: Sequence[Sequence[Any]]) -> None:
    """Write one CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def write_json(dvs: DeltaVectorSet, path: str | os.PathLike,
               indent: int = 2) -> str:
    """Write the whole export as one JSON manifest.

    This is the lossless format: every field of the IR, including units,
    trust-region bounds, health findings and provenance.

    Args:
        dvs: The export.
        path: Destination file.
        indent: JSON indent.

    Returns:
        The path written, as a string.
    """
    text = json.dumps(dvs.to_dict(), indent=indent, sort_keys=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return str(path)


def write_csv(dvs: DeltaVectorSet, directory: str | os.PathLike,
              scaled: bool = False) -> list[str]:
    """Write the export as a directory of CSV tables.

    One ``<block>_jacobian.csv`` per block -- outputs down the rows, levers
    across the columns, with the base case in the margins, which is the
    shift-vector table a planning engineer already reads -- plus ``base.csv``,
    ``bounds.csv``, ``links.csv``, ``specs.csv``, ``prices.csv``,
    ``duals.csv``, ``health.csv`` and ``names.csv``.

    Args:
        dvs: The export.
        directory: Destination directory; created if absent.
        scaled: Write the dimensionless :attr:`DeltaVector.scaled_J` instead
            of the raw Jacobian, where it is available.

    Returns:
        The paths written.
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    written: list[str] = []

    def _p(name: str) -> str:
        path = os.path.join(directory, name)
        written.append(path)
        return path

    for v in dvs.vectors:
        J = v.scaled_J if (scaled and v.scaled_J is not None) else v.J
        header = ["y_name", "y_unit", "y0"] + list(v.u_names)
        rows: list[list[Any]] = [
            ["u0", "", ""] + list(v.u0),
            ["u_unit", "", ""] + ["" if u is None else u for u in v.u_units],
        ]
        for i, yn in enumerate(v.y_names):
            unit = v.y_units[i] if v.y_units[i] is not None else ""
            rows.append([yn, unit, v.y0[i]] + list(J[i]))
        _rows(_p(f"{sanitize_name(v.block)}_jacobian.csv"), header, rows)

    _rows(_p("base.csv"), ["variable", "unit", "value", "kind", "block"], [
        *[[n, u if u is not None else "", val, "input", v.block]
          for v in dvs.vectors
          for n, u, val in zip(v.u_names, v.u_units, v.u0)],
        *[[n, u if u is not None else "", val, "output", v.block]
          for v in dvs.vectors
          for n, u, val in zip(v.y_names, v.y_units, v.y0)],
    ])

    _rows(_p("bounds.csv"),
          ["variable", "lb", "ub", "tr_lo", "tr_hi", "radius", "block"],
          [[n, lo, hi, tlo, thi, v.radius, v.block]
           for v in dvs.vectors
           for n, lo, hi, tlo, thi
           in zip(v.u_names, v.lb, v.ub, v.tr_lo, v.tr_hi)])

    _rows(_p("links.csv"), ["source", "target"],
          [list(pair) for pair in dvs.links])

    _rows(_p("prices.csv"), ["variable", "price"],
          sorted(dvs.prices.items()))

    _rows(_p("specs.csv"),
          ["name", "variable", "coefficient", "op", "rhs", "elastic",
           "penalty", "backoff"],
          [[s["name"], var, coef, s["op"], s["rhs"], s["elastic"],
            "" if s["penalty"] is None else s["penalty"], s["backoff"]]
           for s in dvs.specs for var, coef in s["coeffs"].items()])

    _rows(_p("duals.csv"), ["kind", "name", "marginal"],
          [[kind, name, value]
           for kind, group in (dvs.duals or {}).items()
           if isinstance(group, dict)
           for name, value in group.items()])

    _rows(_p("health.csv"),
          ["severity", "kind", "block", "variable", "value", "detail"],
          [[f.get("severity"), f.get("kind"), f.get("block") or "",
            f.get("variable") or "", f.get("value"), f.get("detail")]
           for f in dvs.health])

    names = [n for v in dvs.vectors for n in (*v.u_names, *v.y_names)]
    _rows(_p("names.csv"), ["name", "lp_name"],
          sorted(_name_map(names).items()))

    return written


def _sanitized_model(lp_model):
    """A copy of ``lp_model`` whose column names are LP/MPS-safe.

    Pyomo indexes ``m.x`` by difflow's own qualified names, and those carry
    ``.``, ``[`` and ``]``.  Rather than leave each writer to mangle them its
    own way, rename the columns up front so the file's symbols are exactly the
    ones recorded in the manifest and in ``names.csv``.

    Returns:
        ``(model, symbols)`` where ``symbols`` maps original name to file name.
    """
    from dataclasses import replace

    symbols = _name_map(list(lp_model.columns))
    renamed = replace(lp_model,
                      columns=[symbols[c] for c in lp_model.columns])
    return renamed, symbols


class _ColumnLabeler:
    """Label LP/MPS symbols with difflow's own sanitised names.

    Pyomo indexes ``m.x`` by column name, so its default labeler writes
    ``x(ngl_NGL_C2)``.  Labelling the variable by its index alone makes the
    file's symbol exactly the one recorded in ``names.csv`` and in the
    manifest's ``lp_symbols``, which is what makes the round trip recoverable.
    """

    def __init__(self, var):
        self._var = var

    def __call__(self, comp) -> str:
        if comp.parent_component() is self._var:
            return sanitize_name(str(comp.index()))
        return sanitize_name(comp.getname(fully_qualified=True))

    def update_cache(self, *args, **kwargs) -> None:
        """Part of Pyomo's labeler protocol; nothing is cached here."""

    def remove_obj(self, *args, **kwargs) -> None:
        """Part of Pyomo's labeler protocol; nothing is cached here."""


def _write_pyomo(lp_model, path: str | os.PathLike, fmt: str,
                 name: str) -> tuple[str, dict[str, str]]:
    """Emit an assembled LP through Pyomo's writers."""
    try:
        from pyomo.opt import ProblemFormat
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Writing .{fmt} files requires Pyomo. Install it with "
            "`pip install pyomo` (difflow.planning keeps it optional)."
        ) from exc

    renamed, symbols = _sanitized_model(lp_model)
    model = renamed.to_pyomo(name=name)
    problem_format = (ProblemFormat.cpxlp if fmt == "lp"
                      else ProblemFormat.mps)
    model.write(str(path), format=problem_format,
                io_options={"labeler": _ColumnLabeler(model.x)})
    return str(path), symbols


def write_lp(lp_model, path: str | os.PathLike,
             name: str = "delta_base_plan") -> str:
    """Write the assembled LP in CPLEX LP format.

    Column names are sanitised (see :func:`sanitize_name`); the mapping is in
    ``names.csv`` from :func:`write_csv` and in the JSON manifest.

    Note:
        Pyomo writes a minimisation.  A maximising plan is emitted with
        negated costs, and the constant term is dropped, so recovering
        difflow's objective from a solver's answer needs ``meta["lp_sense"]``
        and ``meta["lp_objective_offset"]`` from the manifest.

    Args:
        lp_model: An :class:`~difflow.planning.lp.LPModel`, typically
            ``result.lp_model``.
        path: Destination file, conventionally ``.lp``.
        name: Model name.

    Returns:
        The path written.

    Raises:
        ImportError: If Pyomo is not installed.
    """
    written, _ = _write_pyomo(lp_model, path, "lp", name)
    return written


def write_mps(lp_model, path: str | os.PathLike,
              name: str = "delta_base_plan") -> str:
    """Write the assembled LP in free MPS format.

    Args:
        lp_model: An :class:`~difflow.planning.lp.LPModel`.
        path: Destination file, conventionally ``.mps``.
        name: Model name.

    Returns:
        The path written.

    Raises:
        ImportError: If Pyomo is not installed.
    """
    written, _ = _write_pyomo(lp_model, path, "mps", name)
    return written


def write_iterations_csv(result, path: str | os.PathLike) -> str:
    """Write the trust-region audit trail as CSV.

    The coefficients only mean something at the base case they were taken
    at, and the history is how that base case was reached: radius, predicted
    versus realised merit, the ratio, and whether the step was accepted.

    Args:
        result: A :class:`~difflow.planning.planner.PlanResult`.
        path: Destination file.

    Returns:
        The path written.
    """
    rows = [[h.index, h.radius, h.merit, h.predicted, h.realised, h.rho,
             h.accepted, h.lp_status] for h in result.history]
    _rows(path, ["index", "radius", "merit", "predicted", "realised", "rho",
                 "accepted", "lp_status"], rows)
    return str(path)
