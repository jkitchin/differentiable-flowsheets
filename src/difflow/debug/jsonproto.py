"""The program front end: newline-delimited JSON, ``difflow-dbg/1``.

The same engine as the REPL, so an agent, a script or the editor gets the
same debugger a person does. The first line is a ``hello`` that lists
every checkpoint, event, metric, command and capability; a client
feature-detects off it rather than off a version. The protocol follows
pounce's ``pounce-dbg/1`` and discopt's ``discopt-dbg/1``.

Events out (one JSON object per line):

* ``hello`` --- once, first;
* ``pause`` --- at every stop, with the checkpoint, unit, iteration,
  residual, the ``reason`` it stopped, and the rendered ``watches``;
* ``result`` --- one per command, echoing its ``id`` as ``request_id``,
  with ``ok``, the text ``output`` and structured ``data``.

Commands in: a bare string (``"continue"``) or an object
(``{"cmd": "print", "args": ["recycle"], "id": 3}``).
"""

from __future__ import annotations

import json
import math
import sys
from typing import TYPE_CHECKING, Any, TextIO

from .checkpoints import EVENTS, METRICS, Checkpoint
from .engine import COMMANDS, Control
from .repl import read_script

if TYPE_CHECKING:
    from .context import DebugContext
    from .session import DebugSession

PROTOCOL = "difflow-dbg/1"


def capabilities() -> dict[str, Any]:
    """What this debugger can do. Clients branch on these, not on a version."""
    return {
        "inspect": True,
        "unit_breakpoints": True,
        "event_breakpoints": True,
        "conditional_breakpoints": "and",
        "watches": True,
        "request_ids": True,
        "mutate": "tear",
        "rewind": False,
        "traced_solves": False,
        "steppable_accelerations": ["anderson", "wegstein"],
    }


def hello() -> dict:
    from difflow import __version__
    return {
        "event": "hello", "protocol": PROTOCOL, "difflow_version": __version__,
        "checkpoints": [c.value for c in Checkpoint],
        "events": dict(EVENTS), "metrics": dict(METRICS),
        "commands": list(COMMANDS), "capabilities": capabilities(),
    }


def jsonable(value: Any) -> Any:
    """Make a payload JSON-safe: floats that are not finite become strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        return jsonable(float(value))
    except (TypeError, ValueError):
        return str(value)


def pause_event(ctx: "DebugContext", session: "DebugSession") -> dict:
    return {"event": "pause", **ctx.to_event(),
            "watches": session.engine.render_watches(ctx)}


def parse_command(line: str) -> tuple[str, Any]:
    """``(command text, request id)`` from one input line."""
    line = line.strip()
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return line, None              # a bare, unquoted command
    if isinstance(msg, str):
        return msg, None
    if isinstance(msg, dict):
        args = msg.get("args") or []
        text = " ".join([str(msg.get("cmd", ""))] + [str(a) for a in args])
        return text, msg.get("id")
    return "", None


def result_event(result, request_id: Any, command: str) -> dict:
    return {"event": "result", "request_id": request_id,
            "command": command.split(" ", 1)[0], "ok": result.ok,
            "output": result.output, "data": result.data}


class JsonFrontend:
    """Drives the engine over a newline-delimited JSON stream."""

    def __init__(self, script=None, stdin: TextIO | None = None,
                 stdout: TextIO | None = None) -> None:
        self.pending = read_script(script)
        self.stdin = stdin
        self.stdout = stdout
        self._greeted = False

    def emit(self, payload: dict) -> None:
        out = self.stdout or sys.stdout
        out.write(json.dumps(jsonable(payload)) + "\n")
        out.flush()

    def greet(self) -> None:
        if not self._greeted:
            self._greeted = True
            self.emit(hello())

    def interact(self, ctx: "DebugContext", session: "DebugSession") -> Control:
        self.greet()
        self.emit(pause_event(ctx, session))
        while True:
            if self.pending:
                text, request_id = self.pending.pop(0), None
            else:
                line = (self.stdin or sys.stdin).readline()
                if line == "":                 # EOF: let the solve finish
                    session.detach()
                    return Control.CONTINUE
                text, request_id = parse_command(line)
            result = session.engine.execute(text, ctx, session)
            self.emit(result_event(result, request_id, text))
            if result.control is not Control.NONE:
                return result.control
