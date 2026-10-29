"""Record a whole solve, every checkpoint, to scrub through afterwards.

The live debugger blocks the solve on someone at a prompt. A recording
does not: it steps through every checkpoint with nobody there, keeps what
each one changed, and hands back a trace that a page can replay with a
slider --- no parked thread, no session on the server. It answers most of
"why did this not converge"; the live debugger adds only what needs a
running solve, which is changing it.

A trace is ``difflow-trace/1``::

    {"protocol": "difflow-trace/1",
     "species": [...], "units": [...], "tears": [...],
     "frames": [{"checkpoint", "phase", "iteration", "unit", "residual",
                 "trigger", "error", "set": {stream: {field: value}},
                 "tear": [{"stream", "field", "x", "g", "delta"}]}, ...],
     "truncated": false,
     "result": {"ok", "converged", "iterations", "residual", "method", "error"}}

``set`` holds only what the frame wrote --- the feeds at ``solve_start``,
the outlets at ``after_unit``, the tear iterate at ``iteration`` and the
next guess at ``after_accel`` --- so the state at frame ``i`` is the
frames up to ``i`` replayed in order, and the trace stays small.
"""

from __future__ import annotations

from typing import Any

from . import attach
from .checkpoints import Checkpoint
from .context import DebugContext, plain_stream
from .engine import Control
from .session import DebugSession

PROTOCOL = "difflow-trace/1"

#: Frames kept before recording stops (the solve itself runs on).
MAX_FRAMES = 20000


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _packed(ctx: DebugContext, array) -> dict:
    """A packed tear array as ``{stream: {field: value}}``."""
    out: dict[str, dict] = {}
    for (stream, fname), value in zip(ctx.tear_fields(), array):
        out.setdefault(stream, {})[fname] = _float(value)
    return out


def frame(ctx: DebugContext) -> dict:
    """What one checkpoint contributes to a trace."""
    cp = ctx.checkpoint
    out: dict[str, Any] = {
        "checkpoint": cp.value, "phase": ctx.phase, "iteration": ctx.iteration,
        "unit": ctx.unit.name if ctx.unit is not None else None,
        "residual": ctx.residual, "trigger": ctx.event,
    }
    if ctx.error is not None:
        out["error"] = f"{type(ctx.error).__name__}: {ctx.error}"
    if cp is Checkpoint.SOLVE_START:
        out["set"] = {n: plain_stream(s) for n, s in ctx.flowsheet.feeds.items()}
    elif cp is Checkpoint.AFTER_UNIT and ctx.outlets:
        out["set"] = {n: plain_stream(s) for n, s in ctx.outlets.items()}
    elif cp is Checkpoint.ITERATION and ctx.x is not None:
        out["set"] = _packed(ctx, ctx.x)
        out["tear"] = ctx.tear_breakdown()
    elif cp is Checkpoint.AFTER_ACCEL and ctx.x_next is not None:
        out["set"] = _packed(ctx, ctx.x_next)
    elif cp is Checkpoint.UNIT_ERROR and ctx.inlets:
        out["inlets"] = {n: plain_stream(s) for n, s in ctx.inlets.items()}
    return out


class _Recorder:
    """A front end that keeps every checkpoint and never stops."""

    def __init__(self, max_frames: int) -> None:
        self.frames: list[dict] = []
        self.max_frames = max_frames
        self.truncated = False

    def interact(self, ctx: DebugContext, session: DebugSession) -> Control:
        if len(self.frames) >= self.max_frames:
            self.truncated = True
            session.detach()
            return Control.CONTINUE
        self.frames.append(frame(ctx))
        return Control.STEPI


def record(flowsheet, solve_kwargs: dict | None = None, *,
           max_frames: int = MAX_FRAMES) -> dict:
    """Solve ``flowsheet`` once, recording every checkpoint.

    Args:
        flowsheet: The flowsheet to solve.
        solve_kwargs: Passed to ``flowsheet.solve``.
        max_frames: Stop recording (not solving) after this many frames.

    Returns:
        A ``difflow-trace/1`` dict (see the module docstring). A solve that
        raises is recorded up to the raise, with the error under
        ``result``; ``record`` itself does not raise for it.

    Example:
        >>> trace = record(fs, {"acceleration": "wegstein"})
        >>> [f["residual"] for f in trace["frames"] if f["checkpoint"] == "iteration"]
    """
    recorder = _Recorder(max_frames)
    session = DebugSession(recorder)
    fs = flowsheet
    result: dict[str, Any] = {"ok": True, "error": None}
    try:
        with attach(session):
            fs.solve(**dict(solve_kwargs or {}))
    except Exception as exc:  # noqa: BLE001 -- recorded, not raised
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    result.update(
        converged=getattr(fs, "last_solve_converged", None),
        iterations=getattr(fs, "last_solve_iterations", None),
        residual=_float(getattr(fs, "last_solve_residual", None)),
        method=getattr(fs, "last_solve_method", None))
    return {
        "protocol": PROTOCOL,
        "species": list(fs.species_order),
        "units": [u.name for u in fs.units],
        "tears": sorted(getattr(fs, "last_solve_tear_streams", None)
                        or fs.recycles.values()),
        "frames": recorder.frames,
        "truncated": recorder.truncated,
        "result": result,
    }
