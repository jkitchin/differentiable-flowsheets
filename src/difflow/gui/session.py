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
        try:
            with self._lock:
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
                 position=None) -> dict:
        """Drop a unit from the palette onto the canvas.

        It arrives unwired, with a dangling stream on every port, and
        with whatever parameters its ``Params`` class defaults to. An
        operation that cannot be built from defaults alone --- one
        needing a ``thermo`` or a rate law --- is refused with the
        message :mod:`difflow.serialize` gives, which names what is
        missing.
        """
        from difflow.catalog import _default_registry, describe_class
        from difflow.gui import edit
        from difflow.serialize import SerializationError, _build_operation

        def apply() -> dict:
            from difflow.flowsheet import Unit

            info = _default_registry().list_operations().get(operation)
            if info is None:
                raise edit.EditError(f"{operation!r} is not a registered operation")
            taken = {u.name for u in self.flowsheet.units}
            unit_name = edit.unique(name or operation.lower(), taken)
            if name and name in taken:
                raise edit.EditError(f"there is already a unit called {name!r}")
            try:
                built = _build_operation(
                    info.cls, {}, unit_name,
                    override=edit.known_extras(self.flowsheet, info.cls),
                )
            except SerializationError as exc:
                raise edit.EditError(str(exc)) from exc
            ports = describe_class(info.cls).to_dict()["ports"]
            inlets, outlets = edit.default_ports(
                unit_name, ports, edit.stream_names(self.flowsheet)
            )
            self.flowsheet.add_unit(
                Unit(unit_name, built, inlets, outlets)
            )
            if position is not None:
                self._place(unit_name, position)
            return {"name": unit_name, "inlets": inlets, "outlets": outlets}

        return self._edit(apply)

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
