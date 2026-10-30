"""The pause decision: step mode and breakpoints over the command engine.

A :class:`DebugSession` is told about every checkpoint the attached solve
reaches, decides whether to stop there, and when it stops hands the
context to its front end. It does not know whether a person or a program
is on the other side.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from .checkpoints import Checkpoint
from .context import stream_is_finite
from .engine import Control, DebugCommandEngine

if TYPE_CHECKING:
    from difflow.flowsheet import Flowsheet

    from .context import DebugContext


class DebugQuit(Exception):
    """Raised out of ``Flowsheet.solve`` when the user stops it (``quit``).

    An exception and not a quietly returned partial answer: a stopped
    solve has no answer, and anything that looks like one would be read
    as one.
    """


class Frontend(Protocol):
    def interact(self, ctx: "DebugContext", session: "DebugSession") -> Control:
        """Take commands until one resumes the solve; return that control."""
        ...


#: Where a session stops in ``continue`` mode without being asked.
DEFAULT_STOP = frozenset({Checkpoint.UNIT_ERROR, Checkpoint.TERMINATED})

#: What each step mode stops at.
_STEP_STOPS = {
    Control.STEP: {Checkpoint.ITERATION},
    Control.NEXT: {Checkpoint.AFTER_UNIT, Checkpoint.ITERATION},
}


def _traced(ctx: "DebugContext") -> bool:
    from difflow.flowsheet import _has_tracer

    return any(_has_tracer(v) for v in (ctx.inlets, ctx.outlets, ctx.streams, ctx.x, ctx.g)
               if v is not None)


class DebugSession:
    """Step mode, breakpoints and the front end for one attached debugger.

    Args:
        frontend: Who to ask at a pause.
        on_error: Stop only post mortem: at the first unit that raises or
            returns a non-finite outlet, or at a solve that ends
            unconverged.
        max_solves: Debug this many solves, then detach. A script that
            solves in a loop would otherwise stop the user once per solve.
            ``None`` debugs every solve.
    """

    def __init__(self, frontend: Frontend, *, on_error: bool = False,
                 max_solves: int | None = 1) -> None:
        self.engine = DebugCommandEngine()
        self.frontend = frontend
        self.on_error = on_error
        self.max_solves = max_solves
        self.stop_checkpoints: set[Checkpoint] = set(DEFAULT_STOP)
        self.flowsheet: "Flowsheet | None" = None   # the solve being debugged
        self.solves = 0
        self.checkpoints_seen = 0
        self._mode: Control | None = Control.STEPI   # the first checkpoint stops
        self._detached = False
        self._stop_requested = False
        self._post_mortem_done = False
        #: tear values set with ``set tear`` during the current solve
        self.edits: list[dict] = []
        self._pause_requested = False

    # -- lifecycle -------------------------------------------------------

    def detach(self) -> None:
        self._detached = True

    def request_pause(self) -> None:
        """Stop at the next checkpoint, whatever the step mode.

        Safe to call from another thread while the solve runs: it only
        sets a flag that the solve's own thread reads at its next
        checkpoint. That is the soonest a pause can happen --- a unit is
        one step, and it is not interrupted --- and pounce's asynchronous
        pause works the same way.
        """
        self._pause_requested = True

    @property
    def detached(self) -> bool:
        return self._detached

    def begin(self, fs: "Flowsheet") -> None:
        self.flowsheet = fs
        self.solves += 1
        self._post_mortem_done = False
        self.edits = []

    def end(self) -> None:
        self.flowsheet = None
        if self.max_solves is not None and self.solves >= self.max_solves:
            self._detached = True

    # -- the fire-site entry ---------------------------------------------

    def fire(self, ctx: "DebugContext") -> None:
        """Handle a checkpoint; raise :class:`DebugQuit` if the user quit."""
        if self._detached or _traced(ctx):
            return
        self.checkpoints_seen += 1
        ctx.event = ctx.event or _derive_event(ctx)
        if self._should_pause(ctx):
            self._apply(self.frontend.interact(ctx, self))
        if self._stop_requested:
            self._detached = True
            raise DebugQuit("solve stopped from the debugger") from ctx.error

    def _should_pause(self, ctx: "DebugContext") -> bool:
        cp = ctx.checkpoint
        if self._pause_requested:
            self._pause_requested = False
            ctx.extra["reason"] = "paused"
            return True
        if self.on_error:
            failed = (
                cp is Checkpoint.UNIT_ERROR
                or ctx.event == "nonfinite"
                or (cp is Checkpoint.TERMINATED
                    and (ctx.error is not None
                         or getattr(ctx.flowsheet, "last_solve_converged",
                                    None) is False))
            )
            if failed and not self._post_mortem_done:
                self._post_mortem_done = True
                ctx.extra["reason"] = f"on-error: {ctx.event or cp.value}"
                return True
            return False

        reason = self.engine.hit_reason(ctx)
        if reason is not None:
            ctx.extra["reason"] = reason
            return True
        mode = self._mode
        if mode is Control.STEPI:
            ctx.extra["reason"] = ("start" if cp is Checkpoint.SOLVE_START
                                   else "step")
            return True
        if mode in _STEP_STOPS and (cp in _STEP_STOPS[mode]
                                    or cp in DEFAULT_STOP):
            ctx.extra["reason"] = "step"
            return True
        if cp in self.stop_checkpoints:
            ctx.extra["reason"] = f"stop-at {cp.value}"
            return True
        return False

    def _apply(self, control: Control) -> None:
        if control is Control.QUIT:
            self._stop_requested = True
        elif control is Control.CONTINUE:
            self._mode = None
        elif control in (Control.STEP, Control.NEXT, Control.STEPI):
            self._mode = control


def _derive_event(ctx: "DebugContext") -> str | None:
    """The event a checkpoint raises, from what it carries."""
    cp = ctx.checkpoint
    if cp is Checkpoint.UNIT_ERROR:
        return "unit_error"
    if cp is Checkpoint.AFTER_UNIT:
        # Only where a NaN is born: a unit fed a NaN passes it on, and
        # stopping at every unit downstream of the first would bury it.
        if (ctx.outlets and all(stream_is_finite(s) for s in (ctx.inlets or {}).values())
                and not all(stream_is_finite(s) for s in ctx.outlets.values())):
            return "nonfinite"
        if isinstance(ctx.info, dict) and ctx.info.get("converged") is not None:
            try:
                if not bool(ctx.info["converged"]):
                    return "inner_unconverged"
            except Exception:  # noqa: BLE001 -- an array flag; not ours to judge
                pass
    if cp is Checkpoint.ITERATION and ctx.residual is not None:
        if not math.isfinite(ctx.residual):
            return "nonfinite"
        if len(ctx.history) >= 2 and ctx.history[-1] > ctx.history[-2]:
            return "residual_rise"
    return None
