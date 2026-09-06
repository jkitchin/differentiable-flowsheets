"""The flowsheet the editor is working on, and what it can do to it.

Kept apart from the HTTP layer so the interesting half --- load, edit,
solve, emit code --- is reachable and testable without a socket::

    from difflow.gui import FlowsheetSession
    session = FlowsheetSession(path="plant.json")
    session.solve()

Every method returns a plain dict, and a failure is a value in it
(``{"ok": False, "error": ...}``) rather than an exception: a bad edit
from the browser must not take the server down.

The code context
----------------

Half the catalog needs a constructor object --- a ``thermo``, an
``eos`` --- and a reactor needs a rate law. Those are objects, not data,
and a palette cannot invent them. So the flowsheet carries a snippet of
Python in ``view["code_context"]``; the session evaluates it and keeps
the bindings, and anything in the flowsheet may refer to them by name
through :func:`difflow.serialize.ref_namespace`. The same snippet is
emitted as the preamble of :func:`difflow.codegen.to_python`, so the
exported script is what actually ran here.

That means **opening a flowsheet in the editor runs the Python it
carries**, exactly as running the exported script would. It is the same
trust as opening a notebook. The server refuses requests that do not
come from the page it served (see :mod:`difflow.gui.server`), because
an ``exec`` reachable from any web page would be something else
entirely.
"""

from __future__ import annotations

import threading
import types
from pathlib import Path


def evaluate_context(source: str) -> tuple[dict, str | None]:
    """Run a code-context snippet and return ``(bindings, error)``.

    Never raises: a snippet that does not compile or does not run is a
    thing the user is in the middle of typing, and the message --- with
    the line number, which is the only part that helps --- is the
    answer. Modules and underscore names are dropped, so what comes back
    is the objects the flowsheet can refer to and nothing else.
    """
    namespace: dict = {"__name__": "difflow_code_context"}
    try:
        exec(compile(source, "<code context>", "exec"), namespace)
    except SyntaxError as exc:
        return {}, f"line {exc.lineno}: {type(exc).__name__}: {exc.msg}"
    except BaseException as exc:                 # user code: catch it all
        import traceback

        line = None
        for frame in traceback.extract_tb(exc.__traceback__):
            if frame.filename == "<code context>":
                line = frame.lineno
        where = f"line {line}: " if line else ""
        return {}, f"{where}{type(exc).__name__}: {exc}"
    return {
        name: value for name, value in namespace.items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }, None


class FlowsheetSession:
    """The flowsheet the editor is working on, plus what it can do to it.

    Holds the mutable state so the request handler stays a thin shell
    over :mod:`difflow.serialize`, :mod:`difflow.codegen` and
    :mod:`difflow.catalog`.
    """

    def __init__(self, flowsheet=None, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.flowsheet = flowsheet
        #: names the flowsheet may refer to, from its code context
        self.bindings: dict = {}
        #: why the code context did not run, if it did not
        self.context_error: str | None = None
        if flowsheet is None and self.path and self.path.exists():
            self._load(self.path)
        elif flowsheet is not None:
            self._evaluate(self._source())
        self._lock = threading.Lock()

    # -- the code context ---------------------------------------------

    def _load(self, path: Path) -> None:
        """Read a file, running its code context first.

        Not :func:`difflow.serialize.load`: a ``$ref`` in the file needs
        the namespace to exist *while* the flowsheet is rebuilt, and the
        namespace comes from the file itself.
        """
        import json

        from difflow import serialize

        data = json.loads(path.read_text())
        self._evaluate((data.get("view") or {}).get("code_context") or "")
        self.flowsheet = serialize.from_dict(data, refs=self.bindings)

    def _source(self) -> str:
        """The code context the flowsheet carries."""
        view = getattr(self.flowsheet, "view", None) or {}
        return view.get("code_context") or ""

    def _evaluate(self, source: str) -> None:
        self.bindings, self.context_error = (
            evaluate_context(source) if source.strip() else ({}, None)
        )

    def code_context(self) -> dict:
        """The snippet, what it defines, and why it did not run."""
        return {
            "source": self._source(),
            "names": sorted(self.bindings),
            "error": self.context_error,
        }

    def set_code_context(self, source: str) -> dict:
        """Adopt a snippet from the browser, or say why it will not run.

        A snippet that fails is not stored: the flowsheet keeps the last
        one that worked, so a half-typed line cannot cost the bindings
        the units already depend on.
        """
        if self.flowsheet is None:
            return {"ok": False, "error": "no flowsheet loaded"}
        if not isinstance(source, str):
            return {"ok": False, "error": "the code context must be a string"}
        bindings, error = evaluate_context(source) if source.strip() else ({}, None)
        if error is not None:
            return {"ok": False, "error": error}
        with self._lock:
            self.bindings, self.context_error = bindings, None
            if source.strip():
                self.flowsheet.view["code_context"] = source
            else:
                self.flowsheet.view.pop("code_context", None)
        return {"ok": True, "names": sorted(bindings)}

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
        document = serialize.to_dict(self.flowsheet, refs=self.bindings)
        document.setdefault("view", {})
        # Auto-layout underneath, stored positions on top. Not "one or the
        # other": the moment the user drags one node the flowsheet has a
        # `view.nodes` with a single entry in it, and serving only that
        # would send every other node back to the browser unplaced.
        document["view"]["nodes"] = {
            **self.layout(), **(document["view"].get("nodes") or {})
        }
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
            source = codegen.to_python(self.flowsheet, refs=self.bindings)
            return {"source": source, "error": None}
        except Exception as exc:                     # surfaced, not swallowed
            return {"source": "", "error": str(exc)}

    # -- writes -------------------------------------------------------

    def replace(self, document: dict) -> dict:
        """Adopt a flowsheet sent from the browser.

        The document's own code context wins, since the document is the
        whole model: loading a file through this route must behave the
        same as opening it on the command line.
        """
        from difflow import serialize

        with self._lock:
            self._evaluate((document.get("view") or {}).get("code_context") or "")
            self.flowsheet = serialize.from_dict(document, refs=self.bindings)
        return {"ok": True}

    # -- incremental edits --------------------------------------------
    #
    # Each one changes the flowsheet in place and rebuilds only what it
    # touched. `replace` is still there for load-a-file; these are for
    # the canvas, where a keystroke should not cost a reconstruction of
    # every unit -- and where a live `thermo` must survive the edit by
    # identity rather than by round-tripping through JSON.

    def _edit(self, fn, *args, **kwargs) -> dict:
        """Run one edit under the lock, reporting a refusal as a value."""
        from difflow.gui.edit import EditError

        if self.flowsheet is None:
            return {"ok": False, "error": "no flowsheet loaded"}
        from difflow.serialize import ref_namespace

        try:
            # Every incremental edit goes through _build_operation, which
            # decodes `$ref` tags -- so the namespace has to be active for
            # all of them, and this is the one place they all pass.
            with self._lock, ref_namespace(self.bindings):
                result = fn(*args, **kwargs)
        except EditError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **(result or {})}

    def patch_unit(self, name: str, changes: dict) -> dict:
        """Set parameters, rename, or move one unit.

        ``changes`` may carry ``params`` (a partial dict, merged over
        what the unit has), ``name`` (a rename) and ``position``
        (``{"x": .., "y": ..}``), in any combination. Anything else is
        refused by name rather than ignored: an editor that drops an
        edit on the floor is worse than one that rejects it.
        """
        from difflow.gui import edit

        def apply() -> dict:
            unknown = sorted(set(changes) - {"params", "name", "position"})
            if unknown:
                raise edit.EditError(
                    f"cannot patch {', '.join(repr(u) for u in unknown)}; "
                    "a unit takes 'params', 'name' and 'position'."
                )
            u = edit.unit(self.flowsheet, name)
            params = changes.get("params")
            if params:
                if not isinstance(params, dict):
                    raise edit.EditError("'params' must be an object")
                u.operation = edit.rebuild(u.operation, name, params)
            new_name = changes.get("name")
            if new_name and new_name != name:
                self._rename_unit(u, new_name)
            position = changes.get("position")
            if position is not None:
                self._place(new_name or name, position)
            return {"name": new_name or name}

        return self._edit(apply)

    def _rename_unit(self, u, new_name: str) -> None:
        from difflow.gui import edit

        if not isinstance(new_name, str) or not new_name.strip():
            raise edit.EditError("a unit name cannot be empty")
        if any(other.name == new_name for other in self.flowsheet.units):
            raise edit.EditError(f"there is already a unit called {new_name!r}")
        nodes = (self.flowsheet.view or {}).get("nodes")
        if isinstance(nodes, dict) and u.name in nodes:
            nodes[new_name] = nodes.pop(u.name)
        u.name = new_name

    def _place(self, key: str, position) -> None:
        from difflow.gui import edit

        try:
            x, y = float(position["x"]), float(position["y"])
        except (TypeError, KeyError, ValueError) as exc:
            raise edit.EditError(
                f"position for {key!r} must be {{'x': number, 'y': number}}"
            ) from exc
        self.flowsheet.view.setdefault("nodes", {})[key] = {"x": x, "y": y}

    def add_unit(self, operation: str, name: str | None = None,
                 position=None, extras: dict | None = None) -> dict:
        """Drop a unit from the palette onto the canvas.

        It arrives unwired, with a dangling stream on every port, and
        with whatever parameters its ``Params`` class defaults to.

        The constructor objects half the catalog needs come from three
        places, in order: ``extras`` as sent (``{"thermo": {"$ref":
        "thermo"}}``, resolved against the code context), then whatever
        :func:`~difflow.gui.edit.known_extras` can find on its own, then
        nothing --- at which point the refusal names what is missing and
        the answer is to define it in the code context.
        """
        from difflow.catalog import _default_registry, describe_class
        from difflow.gui import edit
        from difflow.serialize import (
            SerializationError,
            _build_operation,
            _decode_value,
        )

        def apply() -> dict:
            from difflow.flowsheet import Unit

            info = _default_registry().list_operations().get(operation)
            if info is None:
                raise edit.EditError(f"{operation!r} is not a registered operation")
            taken = {u.name for u in self.flowsheet.units}
            unit_name = edit.unique(name or operation.lower(), taken)
            if name and name in taken:
                raise edit.EditError(f"there is already a unit called {name!r}")
            override = edit.known_extras(self.flowsheet, info.cls, self.bindings)
            try:
                override.update({k: _decode_value(v)
                                 for k, v in (extras or {}).items()})
            except SerializationError as exc:
                raise edit.EditError(str(exc)) from exc
            from difflow.catalog import _params_class

            values, placeholders = edit.known_params(
                self.flowsheet, _params_class(info.cls), self.bindings
            )
            try:
                built = _build_operation(
                    info.cls, values, unit_name, override=override,
                )
            except SerializationError as exc:
                raise edit.EditError(self._missing_hint(str(exc))) from exc
            ports = describe_class(info.cls).to_dict()["ports"]
            inlets, outlets = edit.default_ports(
                unit_name, ports, edit.stream_names(self.flowsheet)
            )
            self.flowsheet.add_unit(
                Unit(unit_name, built, inlets, outlets)
            )
            if position is not None:
                self._place(unit_name, position)
            return {"name": unit_name, "inlets": inlets, "outlets": outlets,
                    "placeholders": placeholders}

        return self._edit(apply)

    def _missing_hint(self, message: str) -> str:
        """Point a "requires X" refusal at the code context."""
        if "requires" not in message:
            return message
        have = ", ".join(sorted(self.bindings)) or "nothing"
        return (f"{message} Define it in the code context (which currently "
                f"defines {have}) and drop the unit again.")

    def remove_unit(self, name: str) -> dict:
        """Delete a unit, and every wire that only existed because of it."""
        from difflow.gui import edit

        def apply() -> dict:
            u = edit.unit(self.flowsheet, name)
            touched = set(u.inlet_names) | set(u.outlet_names)
            self.flowsheet.units = [
                other for other in self.flowsheet.units if other is not u
            ]
            # A recycle naming one of its streams has lost an end. Left in
            # place it would tear a stream nothing produces, and the solve
            # would fail somewhere far from here.
            dropped = {src: dst for src, dst in self.flowsheet.recycles.items()
                       if src in touched or dst in touched}
            for src in dropped:
                del self.flowsheet.recycles[src]
            nodes = (self.flowsheet.view or {}).get("nodes")
            if isinstance(nodes, dict):
                nodes.pop(name, None)
            return {"name": name, "recycles_dropped": dropped}

        return self._edit(apply)

    def connect(self, source: str, outlet: str, target: str, inlet: str) -> dict:
        """Wire an outlet to an inlet, as an arc or as a recycle."""
        from difflow.gui import edit

        return self._edit(
            lambda: edit.connect(self.flowsheet, source, outlet, target, inlet)
        )

    def disconnect(self, source: str, outlet: str, target: str, inlet: str) -> dict:
        """Unwire one arc. The freed inlet keeps a port, with a fresh name."""
        from difflow.gui import edit

        return self._edit(
            lambda: edit.disconnect(self.flowsheet, source, outlet, target, inlet)
        )

    def set_layout(self, nodes: dict) -> dict:
        """Adopt canvas positions. No rebuild, no solve --- coordinates only."""
        from difflow.gui import edit

        def apply() -> dict:
            if not isinstance(nodes, dict):
                raise edit.EditError("layout must be an object of {key: {x, y}}")
            for key, position in nodes.items():
                self._place(key, position)
            return {"nodes": len(nodes)}

        return self._edit(apply)

    def save(self) -> dict:
        from difflow import serialize

        if self.flowsheet is None:
            return {"ok": False, "error": "no flowsheet loaded"}
        if self.path is None:
            return {"ok": False, "error": "no path was given on startup"}
        try:
            serialize.save(self.flowsheet, self.path, refs=self.bindings)
        except serialize.SerializationError as exc:
            return {"ok": False, "error": str(exc)}
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
