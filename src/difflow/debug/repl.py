"""The terminal front end: a ``difflow-dbg>`` prompt.

Output goes to stderr, so a script's own stdout is left alone. Line
editing, history and Tab completion come from :mod:`readline` when the
terminal has it; on a pipe it falls back to plain lines and behaves the
same, which is what the tests drive.
"""

from __future__ import annotations

import code
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, TextIO

from .checkpoints import EVENTS, METRICS, Checkpoint
from .engine import COMMANDS, Control

if TYPE_CHECKING:
    from .context import DebugContext
    from .session import DebugSession

PROMPT = "difflow-dbg> "
HISTORY = Path("~/.difflow_dbg_history").expanduser()


def read_script(script) -> list[str]:
    """Commands from a file path or an iterable of lines; comments dropped."""
    if script is None:
        return []
    if isinstance(script, (str, os.PathLike)):
        lines: Iterable[str] = Path(script).read_text().splitlines()
    else:
        lines = script
    return [ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith(("#", "//"))]


def banner(ctx: "DebugContext") -> str:
    """One line saying where the solve stopped."""
    parts = [f"@{ctx.checkpoint.value}"]
    if ctx.unit is not None:
        parts.append(ctx.unit.name)
    if ctx.iteration is not None:
        parts.append(f"iter {ctx.iteration}")
    parts.append(f"phase={ctx.phase}")
    if ctx.residual is not None:
        parts.append(f"residual={ctx.residual:.3e}")
    if ctx.checkpoint is Checkpoint.TERMINATED:
        conv = getattr(ctx.flowsheet, "last_solve_converged", None)
        parts.append("converged" if conv else "NOT converged" if conv is False else "")
    line = "── difflow-dbg ── " + "  ".join(p for p in parts if p)
    reason = ctx.extra.get("reason")
    if reason:
        line += f"\n   ↳ {reason}"
    if ctx.error is not None:
        line += f"\n   ✗ {type(ctx.error).__name__}: {ctx.error}"
    edits = getattr(ctx.flowsheet, "last_solve_debug_edits", None)
    if ctx.checkpoint is Checkpoint.TERMINATED and edits:
        line += f"\n   ✎ the debugger set {len(edits)} tear value(s) on the way"
    return line


class ReplFrontend:
    """A person at a terminal.

    Args:
        script: Commands to run at the first pause, before prompting (a
            path, or lines).
        stdin: Where commands come from (default ``sys.stdin``).
        stdout: Where everything is written (default ``sys.stderr``).
    """

    def __init__(self, script=None, stdin: TextIO | None = None,
                 stdout: TextIO | None = None) -> None:
        self.pending = read_script(script)
        self.stdin = stdin
        self.out = stdout
        self._readline = None
        self._ctx: "DebugContext | None" = None

    # -- I/O -------------------------------------------------------------

    def _write(self, text: str) -> None:
        out = self.out or sys.stderr
        out.write(text + "\n")
        out.flush()

    def _read(self) -> str | None:
        stdin = self.stdin or sys.stdin
        if stdin is sys.stdin and stdin.isatty():
            self._setup_readline()
            try:
                return input(PROMPT)
            except EOFError:
                return None
        (self.out or sys.stderr).write(PROMPT)
        line = stdin.readline()
        return None if line == "" else line

    def _setup_readline(self) -> None:
        if self._readline is not None:
            return
        try:
            import readline
        except ImportError:  # Windows without pyreadline
            self._readline = False
            return
        self._readline = readline
        try:
            readline.read_history_file(HISTORY)
        except OSError:
            pass
        import atexit
        atexit.register(lambda: _save_history(readline))
        readline.set_completer(self._complete)
        readline.set_completer_delims(" ")
        readline.parse_and_bind("tab: complete")

    def _complete(self, text: str, state: int) -> str | None:
        """Verbs first; then units, streams, metrics, events and checkpoints."""
        import readline
        words = readline.get_line_buffer().split()
        if len(words) <= 1 and not readline.get_line_buffer().endswith(" "):
            pool = COMMANDS + ["interact"]
        else:
            ctx = self._ctx
            pool = ["tear", "history", "next", "unit", "if", "on", "del", "clear"]
            pool += list(METRICS) + list(EVENTS) + [c.value for c in Checkpoint]
            if ctx is not None:
                pool += [u.name for u in ctx.flowsheet.units] + ctx.known_streams()
        matches = sorted(w for w in set(pool) if w.startswith(text))
        return matches[state] if state < len(matches) else None

    # -- the pause -------------------------------------------------------

    def interact(self, ctx: "DebugContext", session: "DebugSession") -> Control:
        self._ctx = ctx
        self._write(banner(ctx))
        for watch in session.engine.render_watches(ctx):
            self._write(f"   [{watch['target']}] " + watch["text"].replace("\n", "\n   "))
        while True:
            if self.pending:
                line = self.pending.pop(0)
                self._write(PROMPT + line)
            else:
                line = self._read()
                if line is None:      # EOF: let the solve finish
                    session.detach()
                    self._write("(EOF: detached)")
                    return Control.CONTINUE
            line = line.strip()
            if line in ("interact", "py", "!"):
                self._interact(ctx)
                continue
            result = session.engine.execute(line, ctx, session)
            for text in result.output:
                self._write(text)
            if result.control is not Control.NONE:
                return result.control

    def _interact(self, ctx: "DebugContext") -> None:
        """A Python prompt with the pause's state in scope (Ctrl-D returns)."""
        import jax.numpy as jnp
        scope = {"ctx": ctx, "fs": ctx.flowsheet, "streams": ctx.streams,
                 "unit": ctx.unit, "inlets": ctx.inlets, "outlets": ctx.outlets,
                 "info": ctx.info, "x": ctx.x, "g": ctx.g, "jnp": jnp}
        code.interact(banner="names: " + ", ".join(scope) + "  (Ctrl-D to return)",
                      local=scope, exitmsg="")


def _save_history(readline) -> None:
    try:
        readline.write_history_file(HISTORY)
    except OSError:
        pass
