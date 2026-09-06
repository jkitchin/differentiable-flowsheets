"""HTTP plumbing for the editor: the wire encoding, the routes, the server.

The flowsheet itself lives in :class:`~difflow.gui.session.FlowsheetSession`
and the page in ``static/``; this module only moves bytes between them.
Deliberately stdlib only --- see :mod:`difflow.gui`.
"""


from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from difflow.gui.session import FlowsheetSession

#: The front end, as built files on disk rather than a string literal in
#: Python. Serving it from here is what lets the page be built by something
#: other than string concatenation, and lets it be edited without restarting
#: the server.
STATIC = Path(__file__).parent / "static"

DEFAULT_PORT = 8756
#: only ever bound on the loopback interface
HOST = "127.0.0.1"

#: JSON has no literal for the non-finite floats. Python's ``json``
#: writes them as ``Infinity`` / ``NaN``, which the browser's
#: ``JSON.parse`` rejects outright --- and this is the *common* case,
#: not an exotic one: :func:`~difflow.kinetics.mass_action_kinetics`
#: puts ``inf`` in ``K_eq`` for every irreversible reaction. So they go
#: over the wire as strings and are restored on the way back.
NON_FINITE = {"Infinity": float("inf"), "-Infinity": float("-inf"),
              "NaN": float("nan")}


def _json_safe(value: Any) -> Any:
    """Rewrite non-finite floats as the strings in :data:`NON_FINITE`."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _json_restore(value: Any) -> Any:
    """Undo :func:`_json_safe`.

    A string parameter whose value is literally ``"NaN"`` would be
    turned into a float here. No unit declares one, and the alternative
    --- an out-of-band encoding threaded through the whole document ---
    costs more than the case is worth.
    """
    if isinstance(value, str):
        return NON_FINITE.get(value, value)
    if isinstance(value, dict):
        return {k: _json_restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_restore(v) for v in value]
    return value


def page(name: str = "index.html") -> str:
    """A page from :data:`STATIC`, read on every request.

    Read per request, not cached at import: during front-end work the
    rebuilt bundle should show up on reload, not on restart. It is one
    small local file read against a browser round trip.

    Args:
        name: ``"index.html"`` for the canvas editor, ``"classic.html"``
            for the form-and-dropdown one it is replacing.
    """
    return (STATIC / name).read_text(encoding="utf-8")


#: What ``static/`` may serve, and as what. An allow-list rather than
#: ``mimetypes.guess_type`` because this server is reachable from a browser
#: and the tree is ours: anything not built by the front-end build is a bug,
#: not a file to guess a type for.
_CONTENT_TYPES = {
    ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
    ".json": "application/json", ".svg": "image/svg+xml",
    ".map": "application/json", ".woff2": "font/woff2", ".ico": "image/x-icon",
}


def _static_file(path: str) -> tuple[bytes, str] | None:
    """Resolve a URL path under :data:`STATIC`, or ``None``.

    Returns ``None`` for anything that escapes the directory, has an
    unlisted suffix, or does not exist --- the caller answers 404 for all
    three, so a traversal attempt learns nothing a missing file would not.
    """
    candidate = (STATIC / path.lstrip("/")).resolve()
    try:
        candidate.relative_to(STATIC.resolve())
    except ValueError:
        return None
    content = _CONTENT_TYPES.get(candidate.suffix)
    if content is None or not candidate.is_file():
        return None
    return candidate.read_bytes(), content


#: Both ends of a wire, named the way the canvas names them.
_WIRE = ("source", "outlet", "target", "inlet")


def _wire(fn, payload: dict) -> dict:
    """Call a connect/disconnect with a wire read out of a request body."""
    missing = [k for k in _WIRE if not payload.get(k)]
    if missing:
        return {"ok": False, "error": f"a wire needs {', '.join(missing)}"}
    return fn(*(payload[k] for k in _WIRE))


class _Handler(BaseHTTPRequestHandler):
    """Routes. The session does the work."""

    session: FlowsheetSession = None            # set by :func:`serve`
    server_version = "difflow-gui"

    def log_message(self, *args):                # quiet by default
        pass

    def _send(self, payload: Any, status: int = 200, content="application/json"):
        if isinstance(payload, bytes):
            body = payload
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            # allow_nan=False so a leak past _json_safe fails loudly here
            # rather than as a parse error in the browser
            body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        routes = {
            "/": lambda: self._send(page(), content="text/html"),
            # The editor the canvas is replacing, kept reachable until the
            # canvas covers what it does. Losing the working tool for the
            # length of a rewrite is not a trade worth making.
            "/classic": lambda: self._send(page("classic.html"),
                                           content="text/html"),
            # answered so the browser does not log a 404 on every load
            "/favicon.ico": lambda: self._send(b"", content="image/x-icon"),
            "/api/catalog": lambda: self._send(self.session.catalog()),
            "/api/flowsheet": lambda: self._send(self.session.document()),
            "/api/code": lambda: self._send(self.session.code()),
        }
        handler = routes.get(self.path)
        if handler is not None:
            return handler()
        asset = _static_file(self.path)
        if asset is None:
            return self._send({"error": "not found"}, status=404)
        body, content = asset
        self._send(body, content=content)

    def _body(self):
        """The request body as restored JSON, or a 400 already sent."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return _json_restore(json.loads(raw or b"{}")), None
        except json.JSONDecodeError as exc:
            self._send({"ok": False, "error": f"bad JSON: {exc}"}, 400)
            return None, True

    def _dispatch(self, verb: str, payload):
        """The answer to one mutating request, or ``None`` if no route matched.

        Every route here returns the session's own reply dict --- a
        refusal is ``{"ok": False, "error": ...}`` with a 200, because it
        is an answer about the flowsheet rather than a failure of the
        request. Only a malformed request gets a 4xx.
        """
        path, session = self.path, self.session
        unit_path = "/api/unit/"

        if verb == "POST":
            if path == "/api/flowsheet":
                return session.replace(payload)
            if path == "/api/solve":
                return session.solve()
            if path == "/api/save":
                return session.save()
            if path == "/api/layout":
                return session.set_layout(payload.get("nodes", {}))
            if path == "/api/unit":
                return session.add_unit(payload.get("operation", ""),
                                        name=payload.get("name"),
                                        position=payload.get("position"))
            if path == "/api/connect":
                return _wire(session.connect, payload)

        if verb == "PATCH" and path.startswith(unit_path):
            return session.patch_unit(unquote(path[len(unit_path):]), payload)

        if verb == "DELETE":
            if path == "/api/connect":
                return _wire(session.disconnect, payload)
            if path.startswith(unit_path):
                return session.remove_unit(unquote(path[len(unit_path):]))
        return None

    def _mutate(self, verb: str):
        payload, failed = self._body()
        if failed:
            return
        try:
            answer = self._dispatch(verb, payload)
        except Exception as exc:
            # a bad edit from the browser must not take the server down
            return self._send(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400
            )
        if answer is None:
            return self._send({"error": "not found"}, status=404)
        self._send(answer)

    def do_POST(self):
        self._mutate("POST")

    def do_PATCH(self):
        self._mutate("PATCH")

    def do_DELETE(self):
        self._mutate("DELETE")


def make_server(session: FlowsheetSession, port: int = DEFAULT_PORT):
    """Build the HTTP server without starting it.

    Useful for tests, which want a port and a shutdown handle rather
    than a blocking call.
    """
    handler = type("_BoundHandler", (_Handler,), {"session": session})
    return ThreadingHTTPServer((HOST, port), handler)


def serve(
    flowsheet=None,
    path: str | Path | None = None,
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Run the editor until interrupted.

    Args:
        flowsheet: the flowsheet to edit. If omitted and ``path``
            exists, it is loaded from there.
        path: file the editor saves to.
        port: TCP port on the loopback interface.
        open_browser: open a browser window at startup.
    """
    session = FlowsheetSession(flowsheet, path)
    server = make_server(session, port)
    url = f"http://{HOST}:{port}/"
    print(f"difflow editor on {url}   (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """``python -m difflow.gui [flowsheet.json] [--port N] [--no-browser]``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="difflow.gui", description="Local flowsheet editor."
    )
    parser.add_argument("path", nargs="?", help="flowsheet JSON to open and save")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    serve(path=args.path, port=args.port, open_browser=not args.no_browser)
    return 0


