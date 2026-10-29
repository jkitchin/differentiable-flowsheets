"""A debugged solve on a worker thread, driven one call at a time.

The REPL and the JSON front end own the thread they run on: the solve
blocks on a prompt. The editor's HTTP server and the agent tools cannot
block a request on a person, so here the solve runs on a thread of its
own and parks on a queue at every pause; :meth:`DebugRunner.command`
hands it one command and returns what came back.

    runner = DebugRunner(fs, {"acceleration": "wegstein"})
    runner.start()                    # -> the first pause event
    runner.command("break unit R1")   # -> a result event
    runner.command("continue")        # -> a result, with the next pause under "next"
    runner.close()

The events are the JSON protocol's, so whatever renders one renders the
other.
"""

from __future__ import annotations

import queue
import threading
from . import attach
from .engine import Control
from .jsonproto import hello, jsonable, pause_event, result_event
from .session import DebugQuit, DebugSession


class _QueueFrontend:
    def __init__(self) -> None:
        self.inbox: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()

    def interact(self, ctx, session) -> Control:
        self.outbox.put(jsonable(pause_event(ctx, session)))
        while True:
            text = self.inbox.get()
            result = session.engine.execute(text, ctx, session)
            event = result_event(result, None, text)
            event["resumed"] = result.control is not Control.NONE
            self.outbox.put(jsonable(event))
            if result.control is not Control.NONE:
                return result.control


class DebugRunner:
    """Solve ``flowsheet`` under the debugger on a worker thread.

    Args:
        flowsheet: The flowsheet to solve.
        solve_kwargs: Passed to ``flowsheet.solve``.
        on_error: Stop only post mortem (see :class:`DebugSession`).
    """

    def __init__(self, flowsheet, solve_kwargs: dict | None = None, *,
                 on_error: bool = False) -> None:
        self.flowsheet = flowsheet
        self.solve_kwargs = dict(solve_kwargs or {})
        self.frontend = _QueueFrontend()
        self.session = DebugSession(self.frontend, on_error=on_error)
        self.result: dict | None = None
        self.error: BaseException | None = None
        self.finished = False
        self.last_event: dict | None = None
        self._thread: threading.Thread | None = None
        # One reader of the outbox at a time: a `wait` parked on it and a
        # `command` arriving beside it would otherwise take each other's
        # events. `pause` and `close` never read it, so they never queue
        # behind a parked `wait`.
        self._io = threading.Lock()

    @property
    def hello(self) -> dict:
        return hello()

    @property
    def state(self) -> str:
        """``idle`` (not started), ``paused``, ``running`` or ``finished``."""
        if self._thread is None:
            return "idle"
        if self.finished:
            return "finished"
        if (self.last_event or {}).get("event") == "pause":
            return "paused"
        return "running"

    def _work(self) -> None:
        fs = self.flowsheet
        try:
            with attach(self.session):
                self.result = fs.solve(**self.solve_kwargs)
            event = {"event": "finished", "ok": True,
                     "converged": getattr(fs, "last_solve_converged", None),
                     "iterations": getattr(fs, "last_solve_iterations", None),
                     "residual": getattr(fs, "last_solve_residual", None)}
        except BaseException as exc:  # noqa: BLE001 -- reported, not lost on a thread
            self.error = exc
            event = {"event": "finished", "ok": False,
                     "stopped": isinstance(exc, DebugQuit),
                     "error": f"{type(exc).__name__}: {exc}"}
        self.finished = True
        self.frontend.outbox.put(jsonable(event))

    def _next(self, timeout: float | None) -> dict:
        try:
            event = self.frontend.outbox.get(timeout=timeout)
        except queue.Empty:
            return {"event": "running", "detail": "no pause yet; call wait()"}
        if event.get("event") != "result" or event.get("resumed"):
            self.last_event = event        # a resume means: no longer paused
        return event

    def start(self, timeout: float | None = 120.0) -> dict:
        """Start the solve; return its first pause (or ``finished``)."""
        if self._thread is not None:
            raise RuntimeError("this runner has already started")
        self._thread = threading.Thread(target=self._work, daemon=True,
                                        name="difflow-debug")
        self._thread.start()
        with self._io:
            return self._next(timeout)

    def wait(self, timeout: float | None = 120.0) -> dict:
        """The next event, for a caller whose last command returned ``running``.

        At a pause, or once finished, there is nothing to wait for: the
        current event is returned at once.
        """
        if self.state == "paused":
            return self.last_event
        if self.state == "finished":
            return self._last_of_finished()
        with self._io:
            return self._next(timeout)

    def _last_of_finished(self) -> dict:
        """The ``finished`` event, past anything a stop left unread before it."""
        with self._io:
            while True:
                try:
                    event = self.frontend.outbox.get_nowait()
                except queue.Empty:
                    break
                if event.get("event") == "finished":
                    self.last_event = event
        return self.last_event

    def pause(self) -> dict:
        """Ask a running solve to stop at its next checkpoint.

        Returns at once; the pause arrives through :meth:`wait`.
        """
        if self.state != "running":
            return {"event": "result", "ok": False, "command": "pause",
                    "output": [f"nothing to pause: the solve is {self.state}"]}
        self.session.request_pause()
        return {"event": "result", "ok": True, "command": "pause",
                "output": ["pausing at the next checkpoint"]}

    def command(self, text: str, timeout: float | None = 120.0) -> dict:
        """Run one command at the current pause.

        Returns its ``result`` event. A command that resumed the solve also
        carries the event it ran to under ``"next"``: the next ``pause``, or
        ``finished``.
        """
        if self.state != "paused":
            return {"event": "result", "ok": False, "command": text,
                    "output": [f"the solve is {self.state}, not paused"
                               + ("; pause it, or wait" if self.state == "running" else "")]}
        with self._io:
            self.frontend.inbox.put(text)
            event = self._next(timeout)
            if event.get("resumed"):
                event["next"] = self._next(timeout)
        return event

    def close(self, timeout: float = 30.0) -> None:
        """Stop the solve: quit it if paused, stop pausing if running."""
        if self._thread is None or self.finished:
            return
        self.session.detach()           # no further pauses ...
        self.frontend.inbox.put("quit")  # ... and stop at the current one
        self._thread.join(timeout)
