"""HTTP plumbing for the editor: the wire encoding, the routes, the server.

The flowsheet itself lives in :class:`~difflow.gui.session.FlowsheetSession`
and the page in ``static/``; this module only moves bytes between them.
Deliberately stdlib only --- see :mod:`difflow.gui`.
"""


from __future__ import annotations

import errno
import hmac
import html
import json
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from difflow.gui import assistant
from difflow.gui.session import FlowsheetSession

#: The front end, as built files on disk rather than a string literal in
#: Python. Serving it from here is what lets the page be built by something
#: other than string concatenation, and lets it be edited without restarting
#: the server.
STATIC = Path(__file__).parent / "static"

DEFAULT_PORT = 8756
#: only ever bound on the loopback interface
HOST = "127.0.0.1"

#: Prefix of the per-operation documentation route, ``/api/docs/<op>``.
DOCS_PREFIX = "/api/docs/"

#: The header the page must send on every mutating request, and the name of
#: the ``<meta>`` tag it reads the value out of.
TOKEN_HEADER = "X-Difflow-Token"
TOKEN_META = "difflow-token"

#: How often the page says it is still there, in seconds. Read by the
#: front end out of ``/api/about``, so the two cannot disagree.
HEARTBEAT_SECONDS = 15

#: How long after the last word from any page before the editor concludes
#: nobody is watching and stops.
#:
#: Generous on purpose, and the reason is the browser rather than the
#: network: a background tab has its timers throttled to roughly one
#: firing a minute, so a grace of a few heartbeats would shut the editor
#: down every time the user looked at something else. The prompt path
#: for an ordinary close is ``/api/bye``, which the page sends as it
#: unloads; this is the fallback for the cases that send nothing --- a
#: crashed browser, a killed tab, a laptop lid.
IDLE_GRACE_SECONDS = 90

#: How long the editor waits, after the last page has said goodbye,
#: before concluding that none is coming back.
#:
#: A reload is a farewell and then a hello. The farewell goes out from
#: `pagehide`; the hello cannot go out until the new document has
#: fetched the bundle, parsed it and mounted, which is hundreds of
#: milliseconds later at best. Treating the farewell as final closed
#: the server in that gap and the reloaded page landed on nothing --- a
#: browser reload killed the editor.
#:
#: So the farewell means "this page has gone", not "stop now". Five
#: seconds is longer than a reload of a local bundle and short enough
#: that closing the tab still gives the port back while the user is
#: still reaching for the terminal.
RELOAD_LINGER_SECONDS = 5

#: Where the project lives, for the links in the editor's header. Read
#: from the installed metadata so ``pyproject.toml`` stays the one place
#: they are written down, with these as the answer for a source tree
#: that was never installed.
FALLBACK_LINKS = {
    "repository": "https://github.com/jkitchin/differentiable-flowsheets",
    "documentation": "https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/",
    "issues": "https://github.com/jkitchin/differentiable-flowsheets/issues",
}

#: Host names that can only mean this machine. A DNS-rebinding attack
#: reaches the loopback port with the *attacker's* name in ``Host``, so
#: refusing anything else costs nothing and closes it.
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def mint_token() -> str:
    """A fresh per-process token for :data:`TOKEN_HEADER`.

    The editor executes browser-supplied Python (the code context), so a
    request that merely *reaches* the server is not enough: it has to
    come from the page this server itself served. The token is the proof
    of that --- another origin can send a request here, but under the
    same-origin policy it cannot read the page to learn the token.
    """
    return secrets.token_urlsafe(32)


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


def page(name: str = "index.html", token: str = "") -> str:
    """A page from :data:`STATIC`, read on every request.

    Read per request, not cached at import: during front-end work the
    rebuilt bundle should show up on reload, not on restart. It is one
    small local file read against a browser round trip.

    The one thing the server adds to the built file is the token, as a
    ``<meta>`` tag in the head. It has to be delivered *in the page*
    rather than from a route, because a route that hands it out would
    hand it to anyone who can reach the port, which is the whole thing
    the token exists to prevent.

    Args:
        name: ``"index.html"`` for the canvas editor, ``"classic.html"``
            for the form-and-dropdown one it is replacing.
        token: the value for :data:`TOKEN_META`; omitted, no tag is added.
    """
    text = (STATIC / name).read_text(encoding="utf-8")
    if not token:
        return text
    tag = f'<meta name="{TOKEN_META}" content="{html.escape(token, quote=True)}">'
    return text.replace("</head>", f"  {tag}\n  </head>", 1)


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


def links() -> dict[str, str]:
    """The project's own URLs, for the header of the editor.

    Read from the installed distribution's metadata rather than written
    out here, so ``[project.urls]`` in ``pyproject.toml`` stays the one
    place they live and a move of the documentation does not need a
    second edit nobody remembers to make. A source tree that was never
    installed has no metadata to read, and gets :data:`FALLBACK_LINKS`.
    """
    try:
        from importlib.metadata import metadata

        found = {}
        for entry in metadata("difflow").get_all("Project-URL") or ():
            name, _, url = entry.partition(",")
            found[name.strip().lower()] = url.strip()
    except Exception:                            # not installed, or no metadata
        return dict(FALLBACK_LINKS)
    return {key: found.get(key) or fallback
            for key, fallback in FALLBACK_LINKS.items()}


def version() -> str:
    """The installed difflow version, or ``""`` if there is none to read."""
    try:
        from importlib.metadata import version as _version

        return _version("difflow")
    except Exception:
        return ""


class Lifetime:
    """Who is still watching the editor, and therefore whether to stop.

    The editor is a local server behind a browser tab, and the two are
    one tool as far as the user is concerned: closing the tab ought to
    give the port back. Nothing in HTTP says when a page has gone, so
    the page says so itself --- a ``/api/ping`` every
    :data:`HEARTBEAT_SECONDS` while it is open, and an ``/api/bye`` as
    it unloads.

    Clients are counted individually rather than as a number still open.
    Two tabs on the same flowsheet is an ordinary thing to do, and a
    counter gets it wrong in both directions: a reload increments before
    it decrements, and a tab that dies without unloading never
    decrements at all. A dict keyed by a per-page id has neither
    problem, because an id that stops arriving simply ages out.

    Nothing expires until at least one page has checked in. A server
    started with ``--no-browser`` and left for a minute before anyone
    opens it must still be there when they do, and a test that drives
    the routes directly and never pretends to be a page must not have
    the server disappear underneath it.

    Args:
        grace: seconds of silence from every client before the editor
            concludes it is unwatched.
        linger: seconds to keep serving after the last page has gone,
            in case it is a reload rather than a departure.
        clock: the monotonic clock to read, injectable for tests.
    """

    def __init__(self, grace: float = IDLE_GRACE_SECONDS,
                 linger: float = RELOAD_LINGER_SECONDS,
                 clock=time.monotonic):
        self.grace = grace
        self.linger = linger
        self.clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}
        self._arrived = False
        self._asked = False
        #: When the last client left, or ``None`` while one is here.
        self._empty_since: float | None = None

    def ping(self, client: str) -> None:
        """Note that ``client`` is still there."""
        with self._lock:
            self._seen[client or "-"] = self.clock()
            self._arrived = True
            self._empty_since = None

    def bye(self, client: str) -> None:
        """Note that ``client`` has gone.

        The page sends this as it unloads, which is what makes the port
        come back after :attr:`linger` rather than after :attr:`grace`.
        It is not required: a client that vanishes without a word ages
        out.

        Note what it does *not* mean: stop now. `pagehide` fires on a
        reload too, and the page that fired it is about to be replaced
        by one that will ping. Only :meth:`expired` decides, and it
        waits out the linger first.
        """
        with self._lock:
            self._seen.pop(client or "-", None)
            if not self._seen and self._empty_since is None:
                # Timed from the farewell rather than from the watchdog's
                # next look, so the wait is the linger and not the linger
                # plus however long that poll happened to be away.
                self._empty_since = self.clock()

    def quit(self) -> None:
        """Stop, whoever is still watching. The Quit button."""
        with self._lock:
            self._asked = True

    @property
    def asked(self) -> bool:
        """Whether someone pressed Quit."""
        with self._lock:
            return self._asked

    def expired(self) -> bool:
        """Whether the editor should now stop."""
        with self._lock:
            if self._asked:
                return True
            if not self._arrived:
                return False
            now = self.clock()
            self._seen = {client: last for client, last in self._seen.items()
                          if now - last <= self.grace}
            if self._seen:
                self._empty_since = None
                return False
            # Nobody is here. Whether to wait depends on how they went.
            #
            # A client that aged out has already been silent for the
            # whole grace, and that is all the waiting anyone needs.
            # A farewell is different: it is also what a reload sends,
            # on the way to a page that will ping again as soon as it
            # has mounted, and stopping in that gap takes the server
            # out from under the page that was about to arrive.
            if self._empty_since is None:
                return True
            return now - self._empty_since >= self.linger


def watch(server, lifetime: Lifetime, poll: float = 1.0,
          stop: threading.Event | None = None) -> threading.Thread:
    """Stop ``server`` once ``lifetime`` says nobody is watching.

    A thread rather than a check inside a route, because the case that
    matters most --- the tab is gone --- is precisely the case where no
    further request arrives to do the checking.

    ``shutdown`` is called from here and not from the route that asked
    for it: it blocks until ``serve_forever`` returns, and a handler
    that waits for the loop it is itself running in never answers.

    Returns:
        The started daemon thread, so a caller can join it.
    """
    stop = threading.Event() if stop is None else stop

    def loop():
        while not stop.wait(poll):
            if lifetime.expired():
                server.shutdown()
                return

    thread = threading.Thread(target=loop, name="difflow-gui-watch", daemon=True)
    thread.start()
    return thread


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

    session: FlowsheetSession = None            # set by :func:`make_server`
    token: str = ""                             # likewise
    lifetime: "Lifetime" = None                 # likewise
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

    def _guard(self) -> str | None:
        """Why this mutating request must be refused, or ``None``.

        Three checks, cheapest first. ``Host`` is what a DNS-rebinding
        attack cannot control --- it carries the name the browser
        resolved, not the address it reached. ``Origin`` is what an
        ordinary cross-site POST cannot lie about. The token is what is
        left for a request that carries neither, which is every client
        that is not a browser.
        """
        port = self.server.server_address[1]
        host = urlsplit(f"//{self.headers.get('Host', '')}")
        if host.hostname not in LOCAL_HOSTS:
            return "unexpected Host header; this server answers only on loopback"
        origin = self.headers.get("Origin")
        if origin is not None:
            where = urlsplit(origin)
            if where.hostname not in LOCAL_HOSTS or (where.port or 80) != port:
                return f"cross-origin request from {origin!r} refused"
        if not hmac.compare_digest(self.headers.get(TOKEN_HEADER, ""), self.token):
            return (f"missing or wrong {TOKEN_HEADER}; the editor page carries "
                    "it, and a client outside the browser must send it too")
        return None

    def do_GET(self):
        routes = {
            "/": lambda: self._send(page(token=self.token), content="text/html"),
            # The editor the canvas is replacing, kept reachable until the
            # canvas covers what it does. Losing the working tool for the
            # length of a rewrite is not a trade worth making.
            "/classic": lambda: self._send(page("classic.html", self.token),
                                           content="text/html"),
            # answered so the browser does not log a 404 on every load
            "/favicon.ico": lambda: self._send(b"", content="image/x-icon"),
            "/api/catalog": lambda: self._send(self.session.catalog()),
            "/api/flowsheet": lambda: self._send(self.session.document()),
            "/api/code": lambda: self._send(self.session.code()),
            "/api/code-context": lambda: self._send(self.session.code_context()),
            "/api/species": lambda: self._send(self.session.species()),
            "/api/feeds": lambda: self._send(self.session.feeds()),
            "/api/levers": lambda: self._send(self.session.levers()),
            "/api/console": lambda: self._send(self.session.console_names()),
            "/api/diagram": lambda: self._send(self.session.diagram()),
            # Whether the server-side provider can be offered at all.
            # Asked before the option is shown, so "no key here" is a
            # sentence in the settings rather than a failed question.
            "/api/assistant": lambda: self._send(
                {"ok": True, "configured": assistant.configured(),
                 "model": assistant.DEFAULT_MODEL}),
            # Who we are and where to read more. The page draws its
            # header links from this rather than carrying the URLs in
            # the bundle, so a move of the documentation is a metadata
            # edit and not a rebuild of the front end. `heartbeat` is
            # here for the same reason: the interval the page keeps and
            # the grace the server allows are two halves of one
            # agreement, and only one of them should be written down.
            "/api/about": lambda: self._send(
                {"ok": True, "version": version(), "links": links(),
                 "heartbeat": HEARTBEAT_SECONDS}),
            # No separate documentation-index URL here: `links()` already
            # carries `documentation`, read from the packaging metadata,
            # and a second copy is a second thing to move.
        }
        handler = routes.get(self.path)
        if handler is not None:
            return handler()
        # /api/docs/<op>: one operation's rendered docstring. A prefix
        # route rather than a table entry, since the name is the path.
        if self.path.startswith(DOCS_PREFIX):
            operation = unquote(urlsplit(self.path).path[len(DOCS_PREFIX):])
            return self._send(self.session.docs(operation))
        # /api/context?kind=&q=&name=&operation=: the assistant's brief.
        # The only route that takes a query string, because it is the
        # only one whose request is a *question* rather than a resource.
        split = urlsplit(self.path)
        if split.path == "/api/context":
            query = parse_qs(split.query)
            first = lambda key: (query.get(key) or [""])[0]   # noqa: E731
            return self._send(self.session.context(
                kind=first("kind") or "flowsheet",
                question=first("q"),
                name=first("name") or None,
                operation=first("operation") or None,
            ))
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
        feed_path = "/api/feed/"
        stream_path = "/api/stream/"

        if verb == "POST":
            if path == "/api/flowsheet":
                return session.replace(payload)
            if path == "/api/solve":
                return session.solve()
            if path == "/api/sensitivity":
                return session.sensitivity(lever=payload.get("lever"),
                                           target=payload.get("target"))
            if path == "/api/save":
                return session.save()
            if path == "/api/layout":
                return session.set_layout(payload.get("nodes", {}))
            if path == "/api/unit":
                return session.add_unit(payload.get("operation", ""),
                                        name=payload.get("name"),
                                        position=payload.get("position"),
                                        extras=payload.get("extras"))
            if path == "/api/code-context":
                return session.set_code_context(payload.get("source", ""))
            # Writes nothing, but it is a POST: the answer carries the
            # code context back inside `merged`, and `_guard` runs on
            # POST. A GET would be reachable from an unrelated page.
            if path == "/api/boilerplate":
                return session.boilerplate(payload.get("operation", ""),
                                           name=payload.get("name"))
            if path == "/api/species":
                return session.set_species(payload.get("species"))
            # One verb for declaring a feed and for editing one: the
            # browser sends the fields it changed, and a field left out
            # keeps whatever the feed already carried.
            if path == "/api/feed":
                return session.set_feed(payload.get("name"), payload)
            # A POST because it writes: the lever/output selection is
            # persisted in `view["planning"]` so the panel reopens on it.
            if path == "/api/linearize":
                return session.linearize(payload.get("u") or [],
                                         payload.get("y") or [],
                                         bounds=payload.get("bounds"),
                                         radius=payload.get("radius"),
                                         check=bool(payload.get("check")))
            if path == "/api/linearize/files":
                return session.linearization_files(
                    payload.get("format", "json"))
            # A POST because a cell can do anything the process can,
            # the flowsheet included -- `_guard` is the whole protection
            # and a GET would slip past a preflight.
            if path == "/api/console":
                return session.console_run(payload.get("source", ""))
            if path == "/api/console/reset":
                return session.console_reset()
            # Not about the flowsheet, so not on the session: the brief
            # was assembled by a GET and this only forwards it. A POST
            # because it spends money, and so must pass `_guard`.
            if path == "/api/assistant":
                return assistant.answer(payload.get("messages") or [],
                                        model=payload.get("model"))
            if path == "/api/connect":
                return _wire(session.connect, payload)
            # A port, not a wire: a variadic unit's inlet count is a
            # property of this flowsheet rather than of its class, so it
            # is the canvas's to change.
            if path == "/api/inlet":
                return session.add_inlet(payload.get("unit", ""))
            # The three lifetime routes. All POST, so `_guard` runs on
            # them: a GET that stops the server is a link an unrelated
            # page could put in an <img> tag.
            if path == "/api/ping":
                self.lifetime.ping(str(payload.get("client") or ""))
                return {"ok": True}
            if path == "/api/bye":
                self.lifetime.bye(str(payload.get("client") or ""))
                return {"ok": True}
            if path == "/api/quit":
                self.lifetime.quit()
                # Answered before anything stops. The watcher does the
                # shutdown a moment later, so the page gets a reply
                # rather than a dropped connection to report.
                return {"ok": True, "stopped": True}

        if verb == "PATCH":
            if path.startswith(unit_path):
                return session.patch_unit(unquote(path[len(unit_path):]), payload)
            # PATCH rather than POST for the same reason a unit's rename
            # is one: the stream goes on existing, under another name.
            if path.startswith(stream_path):
                return session.rename_stream(unquote(path[len(stream_path):]),
                                             payload.get("name", ""))

        if verb == "DELETE":
            if path == "/api/connect":
                return _wire(session.disconnect, payload)
            if path == "/api/inlet":
                return session.remove_inlet(payload.get("unit", ""),
                                            payload.get("stream", ""))
            if path.startswith(unit_path):
                return session.remove_unit(unquote(path[len(unit_path):]))
            if path.startswith(feed_path):
                return session.remove_feed(unquote(path[len(feed_path):]))
        return None

    def _mutate(self, verb: str):
        refusal = self._guard()
        if refusal is not None:
            return self._send({"ok": False, "error": refusal}, status=403)
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


def make_server(session: FlowsheetSession, port: int = DEFAULT_PORT,
                token: str | None = None, lifetime: Lifetime | None = None):
    """Build the HTTP server without starting it.

    Useful for tests, which want a port and a shutdown handle rather
    than a blocking call. The token is minted here unless one is given,
    and is left on the returned server as ``server.token`` so a caller
    that skipped the page can still speak to it.

    A :class:`Lifetime` is attached either way, because the routes that
    use it must answer whether or not anything is watching it. What a
    test does *not* get is :func:`watch` --- building a server does not
    start a thread that can stop it, so nothing shuts down behind a
    caller that only wanted a port.
    """
    token = mint_token() if token is None else token
    lifetime = Lifetime() if lifetime is None else lifetime
    handler = type("_BoundHandler", (_Handler,),
                   {"session": session, "token": token, "lifetime": lifetime})
    server = ThreadingHTTPServer((HOST, port), handler)
    server.token = token
    server.lifetime = lifetime
    return server


def serve(
    flowsheet=None,
    path: str | Path | None = None,
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    token: str | None = None,
    stay: bool = False,
) -> None:
    """Run the editor until the page goes away, or until interrupted.

    Three things end it, and they are the same ending: the Quit button
    in the header, the last page closing, or ctrl-c here. The first two
    are the point --- the editor is a browser tab as far as the user is
    concerned, and a tab that has been closed should not still be
    holding port 8756 when they start the next one.

    Args:
        flowsheet: the flowsheet to edit. If omitted and ``path``
            exists, it is loaded from there.
        path: file the editor saves to.
        port: TCP port on the loopback interface.
        open_browser: open a browser window at startup.
        token: a fixed :data:`TOKEN_HEADER` value. Only useful for
            front-end development, where the page is served by vite on
            another port and cannot be given a freshly minted one.
        stay: keep serving after the last page closes. For a session
            driven from a script or a notebook rather than from the
            page, where there may be no tab open at all and the Quit
            button is not the thing that ends it.
    """
    session = FlowsheetSession(flowsheet, path)
    server = make_server(session, port, token)
    url = f"http://{HOST}:{port}/"
    how = "ctrl-c to stop" if stay else "Quit, or close the tab, or ctrl-c"
    print(f"difflow editor on {url}   ({how})")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    stop = threading.Event()
    if not stay:
        watch(server, server.lifetime, stop=stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    else:
        # serve_forever returned, so the watcher called shutdown. Say
        # which of the two reasons it was: a user who pressed Quit knows
        # already, but a user who closed a tab they had forgotten about
        # is owed the sentence.
        print("stopped: quit from the editor" if server.lifetime.asked
              else "stopped: the editor page was closed")
    finally:
        stop.set()
        server.server_close()


def free_port_near(port: int, tries: int = 20) -> int:
    """The first bindable port after ``port``, for use in a suggestion.

    A message that names a command has to name one that works, so the
    candidate is actually bound and released rather than guessed. That
    makes it a suggestion and not a reservation --- the socket is closed
    again immediately, and something else may take it in between. The
    fallback if the whole window is busy is ``port + 1``, which is no
    worse than saying nothing.
    """
    for candidate in range(port + 1, port + 1 + tries):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((HOST, candidate))
            except OSError:
                continue
            return candidate
    return port + 1


def listener_on(port: int, host: str = HOST, timeout: float = 2.0) -> dict | None:
    """The process listening on ``host:port``, as far as it can be found.

    "Something is using the port" is only half a sentence: the reader's
    next move is to look at that process and decide whether to kill it,
    and they should not have to go and find out what it is. Nine times
    in ten it is their own forgotten editor and the answer is one
    ``kill`` away.

    ``lsof`` is asked first and ``ss`` second, which between them covers
    macOS and Linux; both are read for a PID, and neither is required.
    A machine with neither, a container without ``/proc``, or a listener
    owned by another user simply yields ``None`` --- the caller then
    says what it always said, which is still true.

    Nothing here is fatal: this runs on the way to an error message, and
    an error message that raises is worse than an incomplete one.

    Args:
        port: the TCP port to look up.
        host: the address it is bound to.
        timeout: seconds to wait for the helper to answer.

    Returns:
        ``{"pid": int, "name": str, "command": str, "mine": bool}``, or
        ``None`` if nothing could be established. ``mine`` is whether
        the process belongs to this user, since only then is ``kill``
        advice worth giving.
    """
    found = _lsof_listener(port, host, timeout) or _ss_listener(port, timeout)
    if found is None:
        return None
    pid = found["pid"]
    found.setdefault("command", "")
    found["command"] = found["command"] or _command_line(pid, timeout)
    found["mine"] = _is_mine(pid)
    return found


def _run(argv: list[str], timeout: float) -> str:
    """``argv``'s stdout, or ``""`` for anything at all going wrong."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout or ""


def _lsof_listener(port: int, host: str, timeout: float) -> dict | None:
    """Read a listening PID out of ``lsof``. macOS, and most Linux."""
    # -nP so it does not stop to resolve names and services; -F asks for
    # the machine-readable form, one field per line tagged by its first
    # character, which is the only lsof output worth parsing.
    out = _run(["lsof", "-nP", f"-iTCP@{host}:{port}", "-sTCP:LISTEN", "-Fpc"],
               timeout)
    pid, name = None, ""
    for line in out.splitlines():
        if line.startswith("p"):
            try:
                pid = int(line[1:])
            except ValueError:
                return None
        elif line.startswith("c"):
            name = line[1:]
    return None if pid is None else {"pid": pid, "name": name}


def _ss_listener(port: int, timeout: float) -> dict | None:
    """Read a listening PID out of ``ss``. Linux without ``lsof``."""
    out = _run(["ss", "-lptnH", f"sport = :{port}"], timeout)
    match = re.search(r'users:\(\("([^"]+)",pid=(\d+)', out)
    if match is None:
        return None
    return {"pid": int(match.group(2)), "name": match.group(1)}


def _command_line(pid: int, timeout: float) -> str:
    """The process's command line, for saying *which* python it is.

    A bare name is not enough here on purpose. Every difflow editor is
    called ``python``, and so is everything else the reader is running;
    the argument list is what tells the forgotten editor from the
    notebook server they are about to kill by mistake.
    """
    out = _run(["ps", "-o", "command=", "-p", str(pid)], timeout).strip()
    return out.splitlines()[0].strip() if out else ""


def _is_mine(pid: int) -> bool:
    """Whether this user could signal ``pid``.

    Asked with signal 0, which checks permission and delivers nothing.
    Only worth knowing so the message does not tell someone to run a
    ``kill`` that will come back ``Operation not permitted``.
    """
    import os

    try:
        os.kill(pid, 0)
    except PermissionError:
        return False
    except OSError:
        return False
    return True


def _listener_lines(port: int) -> str:
    """The part of :func:`port_in_use_message` that names the process."""
    found = listener_on(port)
    if found is None:
        return ""
    pid, name = found["pid"], found["name"] or "?"
    lines = [f"\nIt is held by pid {pid} ({name})."]
    command = found.get("command")
    if command:
        # Long enough to recognise, short enough not to wrap into
        # unreadability on an eighty-column terminal.
        lines.append(f"    {command[:160]}")
    if found.get("mine"):
        lines.append(f"\nTo stop it:\n\n    kill {pid}\n")
    else:
        lines.append("\nIt belongs to another user, so it is not yours to stop.\n")
    return "\n".join(lines)


def port_in_use_message(port: int, prog: str) -> str:
    """What to say when the bind fails because the port is taken.

    Worth spelling out rather than letting the ``OSError`` through: the
    traceback surfaces from inside ``socketserver`` with the port itself
    nowhere in it, so the reader learns that *a* socket is in use and
    nothing about which one or what to do. Nine times in ten the answer
    is that they already have the editor open --- so the process
    holding it is named, with its command line and the ``kill`` that
    would end it, because "in use by *what*" is the reader's actual
    next question and looking it up by hand is three commands they
    should not have to know.

    ``SO_REUSEADDR`` is not the fix and is not offered as one. It is
    already set --- ``HTTPServer`` turns it on --- and it only relieves a
    socket in ``TIME_WAIT``. A live listener is meant to refuse; the
    flag that would override that, ``SO_REUSEPORT``, would leave two
    editors sharing the port and requests landing on whichever won.
    """
    return (
        f"{prog}: port {port} on {HOST} is already in use.\n"
        f"{_listener_lines(port)}"
        f"\n"
        f"An editor is probably running there already --- open\n"
        f"    http://{HOST}:{port}/\n"
        f"before starting a second one. To run another alongside it, or if\n"
        f"that address is something else entirely, pick a free port:\n"
        f"\n"
        f"    {prog} --port {free_port_near(port)}\n"
    )


def main(argv: list[str] | None = None, prog: str = "difflow gui") -> int:
    """``difflow gui [flowsheet.json] [--port N] [--no-browser]``.

    ``prog`` is what usage lines call this, because there are two ways in
    --- the ``difflow`` subcommand and ``python -m difflow.gui`` --- and a
    help text that names the other one sends the reader in a circle.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog=prog, description="Local flowsheet editor."
    )
    parser.add_argument("path", nargs="?", help="flowsheet JSON to open and save")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--stay", action="store_true",
        help="keep serving after the last page closes; otherwise the "
             "editor stops with the tab and gives the port back",
    )
    parser.add_argument(
        "--token",
        help="fixed CSRF token, for `npm run dev` against this server; "
             "otherwise one is minted per run and put into the page",
    )
    args = parser.parse_args(argv)

    try:
        serve(path=args.path, port=args.port, open_browser=not args.no_browser,
              token=args.token, stay=args.stay)
    except OSError as exc:
        # Only the one the caller can act on. Anything else --- a missing
        # flowsheet, a permission --- still gets its traceback, which for
        # those is the useful thing.
        if exc.errno != errno.EADDRINUSE:
            raise
        print(port_in_use_message(args.port, prog), file=sys.stderr)
        return 1
    return 0


