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

*What if.* A solve is deterministic, so solving again and applying
``set tear`` when the replay reaches frame ``i`` continues the recorded
solve exactly, with the accelerator's history intact --- which a fresh
solve seeded with the edited value does not (it forgets the history and
takes a different path). ``record(..., edits=[...], baseline=frames)``
does that, and checks each frame before the first edit against the
baseline: ``result["diverged_at"]`` names the first that differs, which
means the replay is not the recorded solve and its "what if" is about
something else.
"""

from __future__ import annotations

import json
import math
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


def _same(a: dict, b: dict) -> bool:
    """Two frames equal, NaN equal to NaN (it is the same NaN, recorded twice)."""
    return json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)


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
        guess = ctx.x_next if ctx.tear_edit is None else ctx.tear_edit
        out["set"] = _packed(ctx, guess)
    elif cp is Checkpoint.UNIT_ERROR and ctx.inlets:
        out["inlets"] = {n: plain_stream(s) for n, s in ctx.inlets.items()}
    return out


class _Recorder:
    """A front end that keeps every checkpoint and never stops.

    With ``edits`` it also applies ``set tear`` at the frames they name,
    and with ``baseline`` it checks the frames before the first edit.
    """

    def __init__(self, max_frames: int, edits=(), baseline=None) -> None:
        self.frames: list[dict] = []
        self.max_frames = max_frames
        self.truncated = False
        self.edits: dict[int, list[dict]] = {}
        for edit in edits:
            self.edits.setdefault(int(edit["frame"]), []).append(edit)
        self.first_edit = min(self.edits, default=None)
        self.baseline = baseline
        self.diverged_at: int | None = None
        self.applied: list[dict] = []
        self.refused: list[dict] = []

    def interact(self, ctx: DebugContext, session: DebugSession) -> Control:
        k = len(self.frames)
        if k >= self.max_frames:
            self.truncated = True
            session.detach()
            return Control.CONTINUE
        for edit in self.edits.get(k, ()):
            answer = session.engine.execute(
                f"set tear {edit['stream']}.{edit['field']} {float(edit['value'])!r}",
                ctx, session)
            target = self.applied if answer.ok else self.refused
            target.append({**edit, **(answer.data or {}),
                           "detail": " ".join(answer.output)})
        record = frame(ctx)
        if k in self.edits:
            record["edits"] = [e for e in self.applied if e["frame"] == k]
        self._check(k, record)
        self.frames.append(record)
        return Control.STEPI

    def _check(self, k: int, record: dict) -> None:
        """Note the first frame before the first edit that the baseline disagrees on."""
        if (self.baseline is None or self.diverged_at is not None
                or self.first_edit is None or k > self.first_edit):
            return
        theirs = self.baseline[k] if k < len(self.baseline) else None
        mine = {key: v for key, v in record.items() if key != "edits"}
        if theirs is None or not _same(mine, theirs):
            self.diverged_at = k


def record(flowsheet, solve_kwargs: dict | None = None, *,
           max_frames: int = MAX_FRAMES, edits: list[dict] | None = None,
           baseline: list[dict] | None = None) -> dict:
    """Solve ``flowsheet`` once, recording every checkpoint.

    Args:
        flowsheet: The flowsheet to solve.
        solve_kwargs: Passed to ``flowsheet.solve``.
        max_frames: Stop recording (not solving) after this many frames.
        edits: Tear values to set on the way, each ``{"frame", "stream",
            "field", "value"}``: at frame index ``frame`` (an ``iteration``
            or ``after_accel`` frame), ``set tear stream.field value``. An
            edit the debugger refuses is listed under ``result["refused"]``.
        baseline: The frames of the recording being replayed. Each frame
            up to the first edit is checked against it, and the first that
            differs is ``result["diverged_at"]`` (``None`` if none does).

    Returns:
        A ``difflow-trace/1`` dict (see the module docstring). A solve that
        raises is recorded up to the raise, with the error under
        ``result``; ``record`` itself does not raise for it.

    Example:
        >>> trace = record(fs, {"acceleration": "wegstein"})
        >>> [f["residual"] for f in trace["frames"] if f["checkpoint"] == "iteration"]
    """
    recorder = _Recorder(max_frames, edits or (), baseline)
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
    if edits:
        reached = len(recorder.frames)
        unreached = [{**e, "detail": f"the replay ended at frame {reached - 1}"}
                     for e in edits if int(e["frame"]) >= reached]
        result.update(edits=recorder.applied,
                      refused=recorder.refused + unreached,
                      diverged_at=recorder.diverged_at)
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


# ---------------------------------------------------------------------
# Reading a trace: what an agent gets instead of every frame
# ---------------------------------------------------------------------

#: Rows of the per-iteration table before it is cut short.
SUMMARY_ROWS = 200


def state_at(trace: dict, i: int) -> dict:
    """Every stream's values as of frame ``i``: the frames up to ``i`` replayed."""
    state: dict[str, dict] = {}
    for f in trace["frames"][:max(int(i), -1) + 1]:
        for name, values in (f.get("set") or {}).items():
            state.setdefault(name, {}).update(values)
    return state


def _largest(f: dict) -> str | None:
    rows = f.get("tear") or []
    if not rows:
        return None
    r = rows[0]
    return f"{r['stream']}.{r['field']} |g-x|={r['delta']:.3g}" \
        if r.get("delta") is not None else f"{r['stream']}.{r['field']}"


def summarize(trace: dict) -> dict:
    """A trace without its frames: the verdict, one row per tear iteration
    (its frame, residual, and the tear entry the residual is mostly made
    of), the events, and how the residual history reads. Frame numbers are
    what :func:`state_at` and a what-if take."""
    from difflow.diagnostics import classify_history

    frames = trace["frames"]
    iterations = [
        {"frame": k, "iteration": f["iteration"], "phase": f["phase"],
         "residual": f["residual"], "largest": _largest(f)}
        for k, f in enumerate(frames) if f["checkpoint"] == "iteration"]
    events = [
        {"frame": k, "checkpoint": f["checkpoint"], "unit": f["unit"],
         "phase": f["phase"], "iteration": f["iteration"],
         "trigger": f.get("trigger"), "error": f.get("error")}
        for k, f in enumerate(frames) if f.get("trigger") or f.get("error")]
    residuals = [r["residual"] for r in iterations if r["residual"] is not None]
    out = {
        "protocol": trace["protocol"], "units": trace["units"],
        "tears": trace["tears"], "frames": len(frames),
        "truncated": trace["truncated"], "result": trace["result"],
        "regime": classify_history(residuals, trace["result"].get("converged")),
        "iterations": iterations[:SUMMARY_ROWS], "events": events[:SUMMARY_ROWS],
    }
    if len(iterations) > SUMMARY_ROWS:
        out["iterations_omitted"] = len(iterations) - SUMMARY_ROWS
    return out


def compare(recorded: dict, whatif: dict, rtol: float = 1e-6) -> dict:
    """How a what-if replay differs from the recording it replays.

    The residual histories side by side, and whether the two ended at the
    same answer: every stream field at the last frame of each, compared
    relatively. A different answer from a different next guess is the
    signature of more than one steady state.
    """
    def run(trace):
        rows = [f["residual"] for f in trace["frames"] if f["checkpoint"] == "iteration"]
        r = trace["result"]
        return {"converged": r.get("converged"), "iterations": r.get("iterations"),
                "residuals": rows[:SUMMARY_ROWS]}

    a = state_at(recorded, len(recorded["frames"]) - 1)
    b = state_at(whatif, len(whatif["frames"]) - 1)
    worst, where = 0.0, None
    for name in sorted(set(a) & set(b)):
        for field in sorted(set(a[name]) & set(b[name])):
            x, y = a[name][field], b[name][field]
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                gap = 0.0 if (x == y or (math.isnan(x) and math.isnan(y))) else math.inf
            else:
                gap = abs(x - y) / max(abs(x), abs(y), 1e-12)
            if gap > worst:
                worst, where = gap, f"{name}.{field}"
    both = recorded["result"].get("converged") and whatif["result"].get("converged")
    return {
        "recorded": run(recorded), "whatif": run(whatif),
        "edits": whatif["result"].get("edits", []),
        "diverged_at": whatif["result"].get("diverged_at"),
        "same_answer": bool(both) and worst <= rtol,
        "largest_difference": {"where": where, "relative": worst} if where else None,
    }
