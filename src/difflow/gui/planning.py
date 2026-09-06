"""Linearize the open flowsheet into delta vectors, for the editor.

`difflow.planning` already computes the object an LP planning system
wants --- an AD Jacobian around a base case, with the bounds and the
trust region that say where it is valid. What it never had was a way to
choose the levers by pointing at them. This is that way:
:func:`linearize` takes the picker's keys, builds a
:class:`~difflow.planning.block.Block` over the live flowsheet, and
returns the :class:`~difflow.planning.export.DeltaVectorSet` the export
writers already know how to render.

Two things are deliberately not here. There is no planner run: the panel
linearizes, it does not optimize, so there are no prices, no
constraints and no shadow prices --- and therefore no ``.lp`` or
``.mps``, which are renderings of an LP that does not exist yet. A
planner run is `difflow plan-export`'s job, and the JSON manifest this
writes is what that run would consume. And the finite-difference check
is offered but never implied: it costs ``2 n_u`` solves against the AD
Jacobian's one, so it is a button rather than something that happens on
every linearization.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from typing import Any, Sequence

#: Formats the editor can hand back. ``lp``/``mps`` are absent on
#: purpose --- see the module docstring.
FORMATS = ("json", "csv")

DEFAULT_RADIUS = 0.3


def _bounds(u: Sequence[str], bounds: dict | None, u0) -> tuple[list, list]:
    """Lever bounds as two aligned lists.

    A lever with no bounds is not an error --- the Jacobian does not
    need them --- but a planning LP does, so an unbounded lever falls
    back to a symmetric window around its own base value rather than to
    infinity, which would make the trust region meaningless and every
    scaled column zero.
    """
    lb: list[float] = []
    ub: list[float] = []
    for key, base in zip(u, u0):
        window = bounds.get(key) if bounds else None
        base = float(base)
        span = abs(base) if base else 1.0
        lo = window.get("lb") if isinstance(window, dict) else None
        hi = window.get("ub") if isinstance(window, dict) else None
        lb.append(float(lo) if lo is not None else base - span)
        ub.append(float(hi) if hi is not None else base + span)
    return lb, ub


def build_block(flowsheet, u: Sequence[str], y: Sequence[str],
                bounds: dict | None = None, name: str = "flowsheet"):
    """A planning block over the open flowsheet.

    Args:
        flowsheet: the live flowsheet. Not modified.
        u: lever keys, in ``_apply_params`` notation. Exactly what
            ``GET /api/levers`` offers.
        y: output keys, ``"<stream>.<quantity>"``.
        bounds: ``{lever: {"lb": float, "ub": float}}``, partial.
        name: block name, which prefixes every qualified variable.

    Returns:
        A :class:`~difflow.planning.block.Block`.
    """
    from difflow.planning import Block

    probe = Block.from_flowsheet(flowsheet, u=list(u), y=list(y), name=name)
    lb, ub = _bounds(u, bounds, probe.u0)
    return Block.from_flowsheet(flowsheet, u=list(u), y=list(y), name=name,
                                lb=lb, ub=ub)


def linearize(flowsheet, u: Sequence[str], y: Sequence[str], *,
              bounds: dict | None = None,
              radius: float = DEFAULT_RADIUS,
              check: bool = False,
              name: str = "flowsheet") -> tuple[Any, dict]:
    """Linearize, and report it the way the panel needs to show it.

    Args:
        flowsheet: the live flowsheet.
        u: lever keys.
        y: output keys.
        bounds: partial lever bounds.
        radius: trust-region radius, as a fraction of each lever's range.
        check: also verify the AD Jacobian against central differences.
        name: block name.

    Returns:
        ``(delta_vector_set, answer)``. The answer is JSON-safe and
        carries the readable table, the health findings and --- if it was
        asked for --- the finite-difference comparison.
    """
    from difflow.planning import check_delta_health, check_delta_vectors
    from difflow.planning.export import DeltaVectorSet
    from difflow.planning.linearize import linearize_block

    block = build_block(flowsheet, u, y, bounds, name)
    lin = linearize_block(block)
    dvs = DeltaVectorSet.from_block(block, lin, radius=radius,
                                    source="difflow.gui")

    notes: dict[str, str] = {}
    try:
        dvs.health = [asdict(f)
                      for f in check_delta_health(block, radius=radius).findings]
    except Exception as exc:      # a diagnostic must never block an export
        notes["health_error"] = f"{type(exc).__name__}: {exc}"

    verified = None
    if check:
        try:
            report = check_delta_vectors(block)
            verified = {
                "passed": bool(report["passed"]),
                "max_rel_error": float(report["max_rel_error"]),
                "max_abs_error": float(report["max_abs_error"]),
            }
        except Exception as exc:
            notes["check_error"] = f"{type(exc).__name__}: {exc}"

    answer = {
        "ok": True,
        "block": name,
        "u": list(u),
        "y": list(y),
        "radius": float(radius),
        "table": lin.as_table(u_names=block.u_names, y_names=block.y_names),
        "delta_vectors": dvs.to_dict(),
        "health": dvs.health,
        "check": verified,
        "notes": notes,
    }
    return dvs, answer


def files(dvs, fmt: str, stem: str = "flowsheet") -> list[dict]:
    """The export as named text files, ready for the browser to save.

    Written through :mod:`difflow.planning.export`'s own writers into a
    temporary directory and read back, rather than re-rendered here, so
    the bytes a user downloads are the bytes ``difflow plan-export``
    would write.

    Args:
        dvs: the linearization to render.
        fmt: one of :data:`FORMATS`.
        stem: base name for the JSON manifest.

    Returns:
        ``[{"name": ..., "text": ...}, ...]``.
    """
    from difflow.planning.export import write_csv, write_json

    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; have {', '.join(FORMATS)}")
    with tempfile.TemporaryDirectory() as scratch:
        if fmt == "json":
            written = [write_json(dvs, os.path.join(scratch, f"{stem}.json"))]
        else:
            written = write_csv(dvs, scratch)
        return [{"name": os.path.basename(path),
                 "text": open(path, encoding="utf-8").read()}
                for path in written]
