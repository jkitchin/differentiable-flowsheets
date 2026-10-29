"""An interactive debugger for flowsheet solves.

Pause a sequential-modular solve unit by unit and tear iteration by tear
iteration, look at every stream, unit and residual on the way, and stop
on a breakpoint: an iteration, a unit, a condition on the residual, or an
event such as the first NaN. One command engine drives three front ends:
a terminal REPL, a newline-delimited JSON protocol for agents and
scripts, and a threaded runner the editor can drive over HTTP.

Use it from a script or a notebook::

    from difflow import debug
    with debug.attach():                 # a REPL on this terminal
        fs.solve(acceleration="wegstein")

    with debug.attach(on_error=True):    # stop only if something fails
        fs.solve()

or from the shell: ``difflow debug plant.py``.

Detached, it costs one thread-local read per checkpoint and changes
nothing. Attached, it only reads: the solve takes the same path and
returns the same answer, and ``quit`` raises :class:`DebugQuit` rather
than returning a partial one.

What it can step: the eager tear iteration (``acceleration="anderson"``,
the default, or ``"wegstein"``) and every unit call in it.
``acceleration="none"`` iterates inside optimistix, where there is no
Python loop to stop in; the debugger sees only that solve's start and
end. Under ``jax.grad``/``jit`` nothing fires at all: a tracer is
not a number anyone can inspect.
"""

from __future__ import annotations

import threading
from typing import Any

from .checkpoints import EVENTS, METRICS, Checkpoint
from .context import DebugContext
from .engine import CommandResult, Control, DebugCommandEngine
from .session import DebugQuit, DebugSession

__all__ = [
    "Checkpoint", "CommandResult", "Control", "DebugCommandEngine",
    "DebugContext", "DebugQuit", "DebugSession", "EVENTS", "METRICS",
    "attach", "current", "debugged_solve", "detach", "hook", "make_session",
    "starts_debugged_solve",
]

#: The attached session, per thread: a solve on another thread never
#: sees this one's debugger.
_TLS = threading.local()


def current() -> DebugSession | None:
    """The session attached on this thread, if any."""
    return getattr(_TLS, "session", None)


def make_session(mode: Any = "repl", *, frontend=None, script=None,
                 on_error: bool = False, max_solves: int | None = 1) -> DebugSession:
    """A session from a mode name: ``"repl"`` or ``"json"``.

    ``on_error=True`` (or ``mode="on-error"``) stops only post mortem.
    """
    if isinstance(mode, DebugSession):
        return mode
    if mode not in ("repl", "json", "on-error"):
        raise ValueError(f"debug: unknown mode {mode!r} "
                         "(expected 'repl', 'json' or 'on-error')")
    if frontend is None:
        if mode == "json":
            from .jsonproto import JsonFrontend
            frontend = JsonFrontend(script=script)
        else:
            from .repl import ReplFrontend
            frontend = ReplFrontend(script=script)
    return DebugSession(frontend, on_error=on_error or mode == "on-error",
                        max_solves=max_solves)


class _Attached:
    def __init__(self, session: DebugSession, previous: DebugSession | None):
        self.session, self.previous = session, previous

    def __enter__(self) -> DebugSession:
        return self.session

    def __exit__(self, *exc) -> None:
        _TLS.session = self.previous


def attach(session: DebugSession | None = None, **kwargs) -> _Attached:
    """Attach a debugger to solves on this thread.

    Works as a context manager (detaching on exit) or as a plain call
    (until :func:`detach`). With no session, builds one with
    :func:`make_session` from ``kwargs``.
    """
    previous = current()
    _TLS.session = session if session is not None else make_session(**kwargs)
    return _Attached(_TLS.session, previous)


def detach() -> None:
    """Detach this thread's debugger."""
    _TLS.session = None


def hook(fs) -> DebugSession | None:
    """The session to fire into from ``fs``'s solve, or None.

    None when nothing is attached, when the session has detached, and when
    ``fs`` is not the flowsheet being debugged --- a flowsheet solved
    inside a unit of the one being debugged stays silent, so its
    checkpoints do not interleave with the user's.
    """
    session = getattr(_TLS, "session", None)
    if session is None or session.detached or session.flowsheet is not fs:
        return None
    return session


def starts_debugged_solve(fs) -> bool:
    """Whether ``fs.solve`` should hand itself to :func:`debugged_solve`.

    Checked first thing in ``Flowsheet.solve``, so that a detached solve
    runs with not even an extra frame (a wrapper would shift the
    ``stacklevel`` of every warning the solve raises). Only the outermost
    solve is instrumented: ``solve`` calling itself on the same flowsheet
    (automatic tearing does) is already inside one and keeps firing into
    the same session; a different flowsheet solved from inside a unit
    stays silent (see :func:`hook`). A traced solve is not debugged:
    there are no numbers to show, and it is being traced once, not run.
    """
    session = getattr(_TLS, "session", None)
    return (session is not None and not session.detached
            and session.flowsheet is None and not fs._is_traced())


def debugged_solve(fs, kwargs: dict):
    """Run ``fs.solve(**kwargs)`` between ``solve_start`` and ``terminated``."""
    session = current()
    session.begin(fs)
    try:
        session.fire(DebugContext(
            Checkpoint.SOLVE_START, fs, phase="start", streams=dict(fs.feeds),
            method=str(kwargs.get("acceleration")),
            tear_names=sorted(fs.recycles.values())))
        try:
            streams = fs.solve(**kwargs)
        except DebugQuit:
            raise
        except Exception as exc:
            session.fire(DebugContext(
                Checkpoint.TERMINATED, fs, phase="end", error=exc,
                history=list(getattr(fs, "last_solve_history", None) or [])))
            raise
        session.fire(DebugContext(
            Checkpoint.TERMINATED, fs, phase="end", streams=streams,
            residual=getattr(fs, "last_solve_residual", None),
            iteration=getattr(fs, "last_solve_iterations", None),
            method=getattr(fs, "last_solve_method", None),
            history=list(getattr(fs, "last_solve_history", None) or []),
            tear_names=list(getattr(fs, "last_solve_tear_streams", None) or [])))
        return streams
    finally:
        session.end()
