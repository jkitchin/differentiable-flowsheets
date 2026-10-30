"""A Python console over the objects the editor is already holding.

The panels answer fixed questions --- solve, differentiate, linearize
--- and every one of them was built because the answer was worth a
button. A console is for the question nobody built a button for::

    >>> jax.grad(lambda V: fs._apply_params({"reactor.V": V})
    ...          .solve()["liq"]["F_ethanol"])(1.0)

That is the whole reason this exists rather than a notebook beside the
editor: ``fs`` here is *the* flowsheet, the one on the canvas, with the
code context's own bindings already in scope.

Two things are deliberately not sandboxed. The server already runs
browser-supplied Python --- ``view["code_context"]`` is executed on
load, and that is the trust model of opening a flowsheet at all --- so a
console adds no capability that was not there. And a cell that loops
forever holds the single-threaded server until the process is killed;
there is no safe way to interrupt a running thread in Python, so the
panel says so rather than pretending otherwise.

Figures
-------

Text is what v1 emits, but the wire format and the browser's renderer
both already understand an image, so adding plots is a hook and nothing
else::

    def figures(namespace):
        import matplotlib.pyplot as plt
        out = []
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110)
            out.append(image(buf.getvalue()))
        plt.close("all")
        return out

    Console(display_hooks=[figures])

Hooks run after the cell, in order, and their outputs are appended to
it. A hook that raises is reported in place of its figure rather than
failing the cell --- a broken renderer must not eat a result that was
computed correctly.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import linecache
import traceback
from typing import Any, Callable

#: how a cell's filename is built, so tracebacks can be trimmed to the
#: user's own frames and `linecache` can show the offending line
FILENAME = "<console:{n}>"

MAX_OUTPUT = 200_000
"""Characters of text one cell may return.

A stray ``print`` in a loop can produce megabytes, and the browser has
to render whatever arrives. The cap is per output, and the truncation
says so rather than silently shortening.
"""


def text(body: str, *, stream: str = "stdout") -> dict:
    """One block of captured output."""
    if len(body) > MAX_OUTPUT:
        body = (body[:MAX_OUTPUT]
                + f"\n... truncated at {MAX_OUTPUT:,} characters")
    return {"kind": "text", "stream": stream, "text": body}


def value(repr_text: str) -> dict:
    """The repr of a cell's trailing expression."""
    return {"kind": "value", "stream": "stdout", "text": repr_text}


def image(data: bytes, *, mime: str = "image/png") -> dict:
    """A picture, base64'd for the wire.

    Unused until a display hook produces one; here because it is the
    other half of the format the browser already renders, and a hook
    should not have to invent an encoding.
    """
    return {"kind": "image", "mime": mime,
            "data": base64.b64encode(data).decode("ascii")}


def _trim(exc: BaseException, source: str) -> str:
    """The traceback, from the user's first frame down.

    Everything above it is this module and the request handler, which
    is never what went wrong and is four frames of noise on every typo.
    """
    tb = exc.__traceback__
    while tb is not None and not tb.tb_frame.f_code.co_filename.startswith(
            "<console:"):
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()


class Console:
    """A persistent namespace and the cells run against it.

    Never raises: a cell that fails returns its traceback as the answer,
    because that *is* the answer. The only failure this reports as a
    refusal is one that means the console itself could not run.
    """

    def __init__(self, display_hooks: list[Callable[[dict], list]] | None = None):
        self.namespace: dict[str, Any] = {"__name__": "difflow_console"}
        self.display_hooks = list(display_hooks or ())
        self.count = 0
        #: names the session injected, tracked so `names` can report what
        #: the *user* defined -- a panel listing `jax` and `fs` back at
        #: someone who typed neither is noise
        self._injected: set[str] = set()

    def reset(self) -> None:
        """Forget everything the console defined. Live names come back."""
        self.namespace = {"__name__": "difflow_console"}
        self._injected = set()

    def bind(self, **names) -> None:
        """Put the session's live objects in scope, before a cell runs."""
        self.namespace.update(names)
        self._injected.update(names)

    @property
    def names(self) -> list[str]:
        """What the console itself has defined, for the panel to show."""
        return sorted(n for n in self.namespace
                      if not n.startswith("_") and n not in self._injected)

    def run(self, source: str) -> dict:
        """Execute one cell and report everything it produced.

        A trailing expression is echoed by its ``repr``, as a prompt
        does; anything before it is executed for its effect. The two
        halves are compiled separately rather than with ``mode="single"``
        so that a cell pasted in whole behaves like the script it is.
        """
        self.count += 1
        filename = FILENAME.format(n=self.count)
        outputs: list[dict] = []

        try:
            tree = ast.parse(source, filename, "exec")
        except SyntaxError as exc:
            return {"ok": True, "outputs": outputs,
                    "error": _trim(exc, source), "names": self.names}

        # `linecache` is where traceback looks for source, and a console
        # cell is not on disk. Without this every frame reads `<console:3>`
        # with no line under it, which is the part that helps.
        linecache.cache[filename] = (
            len(source), None, source.splitlines(keepends=True), filename)

        body = list(tree.body)
        tail = body.pop() if body and isinstance(body[-1], ast.Expr) else None

        out, err = io.StringIO(), io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                if body:
                    exec(compile(ast.Module(body, []), filename, "exec"),
                         self.namespace)
                if tail is not None:
                    result = eval(                       # noqa: S307 -- the point
                        compile(ast.Expression(tail.value), filename, "eval"),
                        self.namespace)
                    self.namespace["_"] = result
                    if result is not None:
                        # repr() is user code too, and a broken one must
                        # not look like the cell failed
                        try:
                            shown = repr(result)
                        except Exception as exc:      # noqa: BLE001
                            shown = f"<unreprable {type(result).__name__}: {exc}>"
                        outputs.append(value(shown))
        except BaseException as exc:                  # noqa: BLE001 -- user code
            error = _trim(exc, source)

        # stdout first, then the value, then whatever the hooks drew:
        # the order the cell produced them in
        captured = out.getvalue()
        if captured:
            outputs.insert(0, text(captured))
        if err.getvalue():
            outputs.append(text(err.getvalue(), stream="stderr"))
        outputs.extend(self._display())
        return {"ok": True, "outputs": outputs, "error": error,
                "names": self.names}

    def _display(self) -> list[dict]:
        """Run the display hooks, reporting one that breaks in its place."""
        drawn: list[dict] = []
        for hook in self.display_hooks:
            try:
                drawn.extend(hook(self.namespace) or ())
            except Exception as exc:                  # noqa: BLE001
                drawn.append(text(f"{getattr(hook, '__name__', hook)}: "
                                  f"{type(exc).__name__}: {exc}",
                                  stream="stderr"))
        return drawn
