"""Named flowsheet sessions and the operations an agent runs on them.

The engine is :class:`difflow.gui.session.FlowsheetSession`, the same
object behind the browser editor: it holds a live ``Flowsheet``, keeps an
undo history, and answers every call with a dict carrying ``ok`` instead
of raising. A :class:`Workbench` holds several of them by name, so an
agent can keep a base case beside a variant, and adds what a caller
without a canvas needs: one-call unit edits, a compact view of the model,
and a timeout on anything that runs code or solves.

Every public method takes and returns JSON-able values, which is what
lets :mod:`difflow.mcp` register them as tools unchanged and what lets
the tests here call them directly.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from difflow.agent import discovery

#: seconds a solve, a code context or a Python cell may run before the
#: call returns and leaves it running in the background
DEFAULT_TIMEOUT = 300.0


def jsonable(value):
    """``value`` with non-finite floats, arrays and paths made JSON-safe."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [jsonable(v) for v in items]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "tolist"):          # numpy and JAX arrays and scalars
        return jsonable(value.tolist())
    return str(value)


class Workbench:
    """Named :class:`FlowsheetSession` objects, and the tools over them.

    Args:
        allow_exec: Whether the three calls that run Python are available
            (:meth:`run_python`, :meth:`set_code_context`, and
            :meth:`open_file` on a ``.py`` script).
        timeout: Default seconds before a long call returns (see
            :data:`DEFAULT_TIMEOUT`).

    Example:
        >>> wb = Workbench()
        >>> wb.open_example("03_reactor_recycle")["ok"]
        True
        >>> wb.solve()["converged"]
        True
    """

    def __init__(self, allow_exec: bool = True, timeout: float = DEFAULT_TIMEOUT):
        from difflow.gui.session import FlowsheetSession

        self._Session = FlowsheetSession
        self.allow_exec = allow_exec
        self.timeout = timeout
        self.sessions: dict[str, Any] = {"main": FlowsheetSession()}
        self._busy: dict[str, threading.Thread] = {}

    # -- plumbing ----------------------------------------------------------

    def _get(self, session: str):
        """The named session, or an error dict to return as is."""
        if session not in self.sessions:
            return None, {"ok": False, "error": f"no session {session!r} "
                          f"(sessions: {', '.join(self.sessions)})"}
        worker = self._busy.get(session)
        if worker is not None:
            if worker.is_alive():
                return None, {"ok": False, "error": (
                    f"session {session!r} is still busy with a call that "
                    "timed out; wait and try again, or use another session")}
            del self._busy[session]
        return self.sessions[session], None

    def _call(self, session: str, method: str, *args, **kwargs) -> dict:
        s, err = self._get(session)
        if err:
            return err
        return jsonable(getattr(s, method)(*args, **kwargs))

    def _timed(self, session: str, fn, timeout: float | None) -> dict:
        """Run ``fn()`` for ``session`` on a thread, returning by ``timeout``.

        A JAX computation cannot be interrupted, so a call that runs past
        the limit is abandoned, not stopped: it finishes in the background,
        and the session refuses other calls until it has.
        """
        limit = self.timeout if timeout is None else float(timeout)
        box: dict = {}

        def run():
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 -- reported below
                box["result"] = {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"}

        worker = threading.Thread(target=run, daemon=True,
                                  name=f"difflow-{session}")
        worker.start()
        worker.join(limit)
        if worker.is_alive():
            self._busy[session] = worker
            return {"ok": False, "timed_out": True, "error": (
                f"still running after {limit:g} s. It continues in the "
                f"background and session {session!r} is busy until it ends. "
                "A first solve of a large unit can take minutes to compile; "
                "pass a longer timeout if that is the cause.")}
        return jsonable(box["result"])

    def _no_exec(self) -> dict:
        return {"ok": False, "error": "running Python is disabled on this "
                "server (started with --no-exec)"}

    # -- discovery -----------------------------------------------------------

    def plugin_status(self) -> dict:
        """Installed difflow plugins, the operations each registered, and
        any plugin that failed to load (with the reason)."""
        return jsonable(discovery.plugin_status())

    def list_operations(self, query: str = "", category: str = "",
                        plugin: str = "") -> dict:
        """Search the registered unit operations (reactors, separations,
        heat exchange, and every installed plugin's units).

        Args:
            query: Words that must all appear in the name, category,
                plugin or description, e.g. "flash" or "amine absorber".
            category: Only this category, e.g. "reactors".
            plugin: Only this plugin, e.g. "difflow_gas" or "gas".
        """
        return jsonable(discovery.list_operations(query, category, plugin))

    def describe_operation(self, name: str, session: str = "main") -> dict:
        """Everything needed to use one operation: parameters (type,
        default, units, required), ports, equations and assumptions, what
        it still needs in this session before it can be built, and starter
        Python for the code context that supplies it."""
        from difflow.catalog import describe_operation

        s, err = self._get(session)
        if err:
            return err
        try:
            schema = describe_operation(name).to_dict()
        except KeyError:
            return {"ok": False, "error": f"no operation {name!r}; "
                    "search with list_operations"}
        entry = s.catalog().get(name, {})
        schema["needs"] = entry.get("needs", [])
        schema["docs_url"] = entry.get("docs_url")
        if schema["needs"]:
            starter = s.boilerplate(name)
            schema["starter_code"] = starter.get("source")
        return jsonable({"ok": True, **schema})

    def list_api(self, package: str = "difflow", query: str = "") -> dict:
        """The public functions and classes of a difflow package, for the
        libraries that are not unit operations (planning, stochastic,
        estimation, economics, the refinery reformer and FCC, ...).

        Args:
            package: A difflow package or module, e.g. "difflow.planning"
                or "difflow_refinery.reforming". Its submodules are listed
                too, so the library can be walked one level at a time.
            query: Words that must appear in a name or its summary.
        """
        return jsonable(discovery.list_api(package, query))

    def describe_api(self, name: str) -> dict:
        """Full signature and docstring of one difflow function or class,
        e.g. "difflow.solvers.as_nlp" or "difflow.planning.Block"; a class
        also lists its fields and methods."""
        return jsonable(discovery.describe_api(name))

    def search_species(self, query: str = "", limit: int = 25) -> dict:
        """Species in the property database (name, molar mass, and whether
        it has cubic-EOS critical data and ideal-gas thermo data)."""
        return jsonable(discovery.search_species(query, limit))

    def search_docs(self, query: str, limit: int = 5) -> dict:
        """Search the difflow documentation; returns matching sections with
        their text and URL."""
        return jsonable(discovery.search_docs(query, limit))

    def list_examples(self) -> dict:
        """Example flowsheets that open_example can load."""
        return jsonable(discovery.list_examples())

    # -- sessions --------------------------------------------------------------

    def list_sessions(self) -> dict:
        """The open sessions: units, whether solved, and unsaved changes."""
        out = []
        for name, s in self.sessions.items():
            fs = s.flowsheet
            out.append({
                "name": name, "path": None if s.path is None else str(s.path),
                "units": [] if fs is None else [u.name for u in fs.units],
                "solved": s.streams is not None,
                "busy": name in self._busy and self._busy[name].is_alive(),
            })
        return {"ok": True, "sessions": out}

    def new_session(self, name: str) -> dict:
        """Start an empty flowsheet under a new session name."""
        if not name or name in self.sessions:
            return {"ok": False, "error": f"session name {name!r} is empty or taken"}
        self.sessions[name] = self._Session()
        self.sessions[name].new()
        return {"ok": True, "session": name}

    def close_session(self, name: str) -> dict:
        """Forget a session (unsaved changes are lost)."""
        if name not in self.sessions:
            return {"ok": False, "error": f"no session {name!r}"}
        del self.sessions[name]
        self._busy.pop(name, None)
        if not self.sessions:
            self.sessions["main"] = self._Session()
        return {"ok": True, "sessions": list(self.sessions)}

    def open_example(self, key: str, session: str = "main") -> dict:
        """Load an example flowsheet (see list_examples) into a session."""
        return self._call(session, "open_example", key)

    def open_file(self, path: str, session: str = "main",
                  timeout: float | None = None) -> dict:
        """Open a flowsheet file: a .json saved by difflow, or a .py script
        that builds one (the script is run)."""
        s, err = self._get(session)
        if err:
            return err
        if str(path).endswith(".py") and not self.allow_exec:
            return self._no_exec()
        return self._timed(session, lambda: s.open_file(path), timeout)

    def save(self, path: str | None = None, overwrite: bool = False,
             session: str = "main") -> dict:
        """Write the flowsheet as JSON (to its own path when none is given)."""
        return self._call(session, "save", path, overwrite)

    def undo(self, session: str = "main") -> dict:
        """Undo the last change to the flowsheet."""
        return self._call(session, "undo")

    def redo(self, session: str = "main") -> dict:
        """Redo the last undone change."""
        return self._call(session, "redo")

    def get_flowsheet(self, format: str = "summary", session: str = "main") -> dict:
        """The flowsheet as a compact summary, the full JSON document, or
        the equivalent runnable Python.

        Args:
            format: "summary" (units, ports, parameters, feeds, recycles),
                "json" (the file format, everything), or "python".
        """
        s, err = self._get(session)
        if err:
            return err
        if format == "python":
            return jsonable({"ok": True, **s.code()})
        doc = s.document()
        if format == "json":
            return jsonable(doc)
        if format != "summary":
            return {"ok": False, "error": 'format is "summary", "json" or "python"'}
        sheet = doc["flowsheet"] or {}
        return jsonable({
            "ok": True,
            "species": sheet.get("species_order"),
            "units": [{
                "name": u["name"], "operation": u["operation"],
                "inlets": u.get("inlets"), "outlets": u.get("outlets"),
                "params": {k: _brief(v) for k, v in (u.get("params") or {}).items()},
                **({"call_params": u["extra_params"]} if u.get("extra_params") else {}),
            } for u in sheet.get("units", [])],
            "feeds": sheet.get("feeds"),
            "recycles": sheet.get("recycles"),
            "unfed_inlets": s.feeds().get("unfed"),
            "pending": doc.get("pending"),
            "solver": s.solver_options(),
            "solved": s.streams is not None,
            "code_context": bool((sheet.get("view") or {}).get("code_context")),
        })

    # -- building ----------------------------------------------------------------

    def set_species(self, names: list[str], session: str = "main") -> dict:
        """Set the flowsheet's species, in order. Must come before units that
        need them; check names with search_species."""
        return self._call(session, "set_species", names)

    def set_code_context(self, source: str, session: str = "main",
                         timeout: float | None = None) -> dict:
        """Set the Python snippet that defines objects the flowsheet refers
        to by name: thermo or EOS objects, rate laws, solvent names. Units
        waiting on those names build themselves once it defines them. The
        snippet is run, and saved with the flowsheet."""
        if not self.allow_exec:
            return self._no_exec()
        s, err = self._get(session)
        if err:
            return err
        return self._timed(session, lambda: s.set_code_context(source), timeout)

    def add_unit(self, operation: str, name: str | None = None,
                 params: dict | None = None, extras: dict | None = None,
                 session: str = "main") -> dict:
        """Add a unit operation, optionally setting parameters in the same call.

        It arrives unwired, each port carrying its own dangling stream name.
        A unit whose requirements are not met yet is added as pending and
        the reply says what it needs (define it with set_code_context).

        Args:
            operation: A registered operation name (list_operations).
            name: The unit's name; generated when omitted.
            params: Parameter values, e.g. {"V": 2.0}; arrays as lists.
            extras: Constructor objects by name from the code context,
                e.g. {"thermo": {"$ref": "thermo"}}.
        """
        s, err = self._get(session)
        if err:
            return err
        added = jsonable(s.add_unit(operation, name, extras=extras))
        if not added.get("ok") or not params or added.get("pending"):
            return added
        patched = jsonable(s.patch_unit(added["name"], {"params": _decode(params)}))
        if not patched.get("ok"):
            # Keep the model as it was before this call, not half of it.
            s.undo()
            return {"ok": False, "error": f"{operation} was not added: "
                    f"{patched.get('error')}"}
        # A placeholder is a required number the session set to 1.0 for
        # want of one; a value given here is no longer a placeholder.
        left = [p for p in added.get("placeholders", []) if p not in params]
        return {**added, "placeholders": left, "params_set": sorted(params)}

    def update_unit(self, name: str, params: dict | None = None,
                    call_params: dict | None = None, rename: str | None = None,
                    session: str = "main") -> dict:
        """Change a unit's parameters, call parameters (a None value removes
        one) or name. Only the parameters given change."""
        changes: dict = {}
        if params:
            changes["params"] = _decode(params)
        if call_params:
            changes["call_params"] = _decode(call_params)
        if rename:
            changes["name"] = rename
        if not changes:
            return {"ok": False, "error": "nothing to change"}
        return self._call(session, "patch_unit", name, changes)

    def remove_unit(self, name: str, session: str = "main") -> dict:
        """Delete a unit and its wiring."""
        return self._call(session, "remove_unit", name)

    def connect(self, source: str, outlet: str, target: str, inlet: str,
                session: str = "main") -> dict:
        """Wire one unit's outlet stream to another unit's inlet stream.

        Ports are stream names: a unit's outlets and inlets are listed by
        add_unit and get_flowsheet. A connection that closes a loop becomes
        a recycle (tear) automatically.
        """
        return self._call(session, "connect", source, outlet, target, inlet)

    def disconnect(self, source: str, outlet: str, target: str, inlet: str,
                   session: str = "main") -> dict:
        """Remove one connection made by connect."""
        return self._call(session, "disconnect", source, outlet, target, inlet)

    def set_feed(self, stream: str, T: float | None = None,
                 P: float | None = None, flows: dict | None = None,
                 session: str = "main") -> dict:
        """Declare or change the feed on an inlet stream nothing else supplies.

        Args:
            stream: The inlet stream name (get_flowsheet lists unfed inlets).
            T: Temperature in K.
            P: Pressure in Pa.
            flows: Molar flows in mol/s by species, e.g. {"A": 1.0}. Species
                not given keep their current values.
        """
        spec = {k: v for k, v in (("T", T), ("P", P), ("flows", flows))
                if v is not None}
        return self._call(session, "set_feed", stream, spec)

    def remove_feed(self, stream: str, session: str = "main") -> dict:
        """Take the feed off a stream."""
        return self._call(session, "remove_feed", stream)

    # -- running -----------------------------------------------------------------

    def set_solver_options(self, options: dict, session: str = "main") -> dict:
        """Set recycle-solver options, saved with the flowsheet; None restores
        a default.

        Options: tol (step test, default 1e-8), max_iter (100), acceleration
        ("anderson", "wegstein" or "none"), damping (used with "none"),
        anderson_depth (5), clip_negative_flows (true), use_initialization
        (true), tears ("declared", "auto", "heuristic", "minimum"),
        error_probe (2), tol_basis ("step", or "error" to test the measured
        error instead of the step).
        """
        return self._call(session, "set_solver_options", options)

    def solve(self, options: dict | None = None, session: str = "main",
              timeout: float | None = None) -> dict:
        """Solve the flowsheet.

        Read more than ``ok``: ``converged`` is the recycle verdict, ``gain``
        and ``error_estimate`` say how far the answer may be from the step
        ``tol`` tested (at gain g the error is about 1/(1-g) times larger),
        ``warnings`` holds what the solve warned about, and ``audit`` checks
        mass balance, negative flows and NaN. A result is good only when it
        converged AND the audit is clean. Stream values are omitted here;
        read them with get_streams.

        Args:
            options: Solver options to set first (see set_solver_options);
                they are kept with the flowsheet.
        """
        s, err = self._get(session)
        if err:
            return err
        if options:
            changed = jsonable(s.set_solver_options(options))
            if not changed.get("ok"):
                return changed
        result = self._timed(session, s.solve, timeout)
        if result.get("ok"):
            result["streams"] = sorted(result.get("streams") or {})
        return result

    def get_streams(self, names: list[str] | None = None,
                    session: str = "main") -> dict:
        """Stream values from the last solve (T in K, P in Pa, flows in mol/s),
        for the named streams or all of them."""
        s, err = self._get(session)
        if err:
            return err
        if s.streams is None:
            return {"ok": False, "error": "not solved since the last change; call solve"}
        wanted = names or sorted(s.streams)
        missing = [n for n in wanted if n not in s.streams]
        if missing:
            return {"ok": False, "error": f"no stream {', '.join(missing)} "
                    f"(streams: {', '.join(sorted(s.streams))})"}
        return jsonable({"ok": True, "streams": {
            n: {k: (v if isinstance(v, str) else float(v))
                for k, v in s.streams[n].items()} for n in wanted}})

    # -- Python ---------------------------------------------------------------------

    def run_python(self, code: str, session: str = "main",
                   timeout: float | None = None) -> dict:
        """Run Python in the session, for what the other tools cannot express.

        Prefer the structured tools (solve, not fs.solve()) so diagnostics
        and undo are kept. In scope: fs (the live flowsheet), streams (the
        last solve), difflow, jax, jnp, session, and the code context's
        names. Names persist between calls; a trailing expression's value
        is returned.
        """
        if not self.allow_exec:
            return self._no_exec()
        s, err = self._get(session)
        if err:
            return err
        return self._timed(session, lambda: s.console_run(code), timeout)


def _brief(value):
    """A parameter value short enough for a summary."""
    if isinstance(value, dict):
        if "$array" in value:
            return {"$array": value["$array"]} if len(str(value)) < 200 else "array"
        if "$callable" in value:
            return f"{value['$callable'].get('factory')}(...)"
        if "$ref" in value:
            return value
        return "{...}" if len(str(value)) > 200 else value
    return value


def _decode(params: dict) -> dict:
    """Plain JSON parameter values as the session expects them.

    The session takes the file format: a list is a ``$array``, a name in
    the code context is ``{"$ref": name}``. A caller sending a plain list
    for an array parameter means an array, so it is wrapped here.
    """
    out = {}
    for key, value in params.items():
        if isinstance(value, list):
            value = {"$array": value}
        out[key] = value
    return out
