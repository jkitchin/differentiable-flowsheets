"""The flowsheet the editor is working on, and what it can do to it.

Kept apart from the HTTP layer so the interesting half --- load, edit,
solve, emit code --- is reachable and testable without a socket::

    from difflow.gui import FlowsheetSession
    session = FlowsheetSession(path="plant.json")
    session.solve()

Every method returns a plain dict, and a failure is a value in it
(``{"ok": False, "error": ...}``) rather than an exception: a bad edit
from the browser must not take the server down.
"""

from __future__ import annotations

import threading
from pathlib import Path


class FlowsheetSession:
    """The flowsheet the editor is working on, plus what it can do to it.

    Holds the mutable state so the request handler stays a thin shell
    over :mod:`difflow.serialize`, :mod:`difflow.codegen` and
    :mod:`difflow.catalog`.
    """

    def __init__(self, flowsheet=None, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.flowsheet = flowsheet
        if flowsheet is None and self.path and self.path.exists():
            from difflow import serialize

            self.flowsheet = serialize.load(self.path)
        self._lock = threading.Lock()

    # -- reads --------------------------------------------------------

    def catalog(self) -> dict:
        from difflow.catalog import catalog

        return {
            name: spec.to_dict() for name, spec in catalog().items()
        }

    def document(self) -> dict:
        from difflow import serialize

        if self.flowsheet is None:
            return {"flowsheet": None, "path": str(self.path or "")}
        document = serialize.to_dict(self.flowsheet)
        document.setdefault("view", {})
        if not document["view"].get("nodes"):
            document["view"]["nodes"] = self.layout()
        return {"flowsheet": document, "path": str(self.path or "")}

    def layout(self) -> dict:
        """Canvas positions from the topology, for a flowsheet that has none.

        Filled into the served document rather than into the flowsheet: a
        file that has never been laid out should not acquire coordinates
        merely because someone opened it. The browser owns them from there,
        and they reach disk only when the user saves. :func:`auto_layout` is
        stable, so re-deriving them on every load is not a shuffle.
        """
        from difflow.gui.layout import auto_layout

        if self.flowsheet is None:
            return {}
        return {name: {"x": x, "y": y}
                for name, (x, y) in auto_layout(self.flowsheet).items()}

    def code(self) -> dict:
        from difflow import codegen

        if self.flowsheet is None:
            return {"source": "", "error": "no flowsheet loaded"}
        try:
            return {"source": codegen.to_python(self.flowsheet), "error": None}
        except Exception as exc:                     # surfaced, not swallowed
            return {"source": "", "error": str(exc)}

    # -- writes -------------------------------------------------------

    def replace(self, document: dict) -> dict:
        """Adopt a flowsheet sent from the browser."""
        from difflow import serialize

        with self._lock:
            self.flowsheet = serialize.from_dict(document)
        return {"ok": True}

    def save(self) -> dict:
        from difflow import serialize

        if self.flowsheet is None:
            return {"ok": False, "error": "no flowsheet loaded"}
        if self.path is None:
            return {"ok": False, "error": "no path was given on startup"}
        serialize.save(self.flowsheet, self.path)
        return {"ok": True, "path": str(self.path)}

    def solve(self) -> dict:
        """Solve, and report a failure rather than raising at the socket."""
        if self.flowsheet is None:
            return {"ok": False, "error": "no flowsheet loaded"}
        try:
            with self._lock:
                streams = self.flowsheet.solve()
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "streams": {
                name: {
                    k: (v if isinstance(v, str) else float(v))
                    for k, v in stream.items()
                }
                for name, stream in streams.items()
            },
            "converged": getattr(self.flowsheet, "last_solve_converged", None),
            "iterations": getattr(self.flowsheet, "last_solve_iterations", None),
        }
