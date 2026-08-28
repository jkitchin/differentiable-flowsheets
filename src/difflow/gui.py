"""A local editor for flowsheets, served to the browser.

Researchers already have difflow installed, so the browser does not need
to run the solver --- it only needs to reach one. This serves a small
HTTP API and a single-page editor on ``localhost``, with the real
package doing the work:

    python -m difflow.gui plant.json

The point is not to replace writing Python. It is to make the parts
that are tedious in Python --- seeing the topology, adjusting a
parameter and re-solving, checking what a unit expects --- quick, while
leaving the door open: the editor exports the model as a script
(:mod:`difflow.codegen`) or as JSON (:mod:`difflow.serialize`), and
reads JSON back. An editor you can only enter is worse than none.

Units are added by clicking the palette, and streams are wired by
naming them: an outlet names a stream, and an inlet chooses one that
something already produces. Renaming a stream follows it to every
consumer, so the wiring survives the edit.

What the palette cannot add, it says so about rather than failing
later. Roughly half the catalog needs something no form can supply ---
a ``thermo`` object, a rate law --- and those entries are dimmed with
the reason (:attr:`~difflow.catalog.OperationSchema.is_buildable`).
For them the route is still Python, then JSON, then here.

Everything comes from :mod:`difflow.catalog`, so the palette lists
whatever is registered, plugins included, with the ports and parameter
schema each unit actually declares.

Deliberately stdlib only: a development tool that made difflow depend
on a web framework would be a poor trade. It binds to 127.0.0.1 and is
meant for a single local user --- it is not hardened for exposure to a
network.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
        return {
            "flowsheet": serialize.to_dict(self.flowsheet),
            "path": str(self.path or ""),
        }

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
            "/": lambda: self._send(_PAGE, content="text/html"),
            # answered so the browser does not log a 404 on every load
            "/favicon.ico": lambda: self._send(b"", content="image/x-icon"),
            "/api/catalog": lambda: self._send(self.session.catalog()),
            "/api/flowsheet": lambda: self._send(self.session.document()),
            "/api/code": lambda: self._send(self.session.code()),
        }
        handler = routes.get(self.path)
        if handler is None:
            return self._send({"error": "not found"}, status=404)
        handler()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = _json_restore(json.loads(raw or b"{}"))
        except json.JSONDecodeError as exc:
            return self._send({"ok": False, "error": f"bad JSON: {exc}"}, 400)

        try:
            if self.path == "/api/flowsheet":
                return self._send(self.session.replace(payload))
            if self.path == "/api/solve":
                return self._send(self.session.solve())
            if self.path == "/api/save":
                return self._send(self.session.save())
        except Exception as exc:
            # a bad edit from the browser must not take the server down
            return self._send(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400
            )
        self._send({"error": "not found"}, status=404)


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


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>difflow editor</title>
<style>
  :root {
    --surface:#fcfcfb; --panel:#f4f3f0; --ink:#0b0b0b; --ink-soft:#52514e;
    --grid:#e4e2de; --line:#c9c7c2; --series:#2a78d6; --accent:#eb6834;
    --bad:#d03b3b; --good:#0ca30c;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--surface); color:var(--ink);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
    Helvetica,Arial,sans-serif; }
  header { display:flex; align-items:center; gap:.75rem; padding:.7rem 1rem;
    border-bottom:1px solid var(--grid); background:var(--panel); }
  header h1 { font-size:.95rem; margin:0; font-weight:650; letter-spacing:-.01em; }
  header .path { color:var(--ink-soft); font-size:.8rem; }
  header .spacer { flex:1; }
  button { font:inherit; font-size:.83rem; padding:.35rem .7rem;
    border:1px solid var(--line); border-radius:6px; background:var(--surface);
    color:var(--ink); cursor:pointer; }
  button:hover { border-color:var(--ink-soft); }
  button.primary { background:var(--series); border-color:var(--series);
    color:#fff; }
  main { display:grid; grid-template-columns:14rem 1fr 21rem; gap:1px;
    background:var(--grid); min-height:calc(100vh - 3.1rem); }
  section { background:var(--surface); padding:.9rem 1rem; overflow:auto;
    max-height:calc(100vh - 3.1rem); }
  h2 { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em;
    color:var(--ink-soft); margin:0 0 .7rem; font-weight:650; }
  .cat { margin-bottom:.9rem; }
  .cat-name { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em;
    color:var(--ink-soft); margin:.5rem 0 .25rem; }
  .op { display:flex; gap:.4rem; align-items:baseline; padding:.25rem .4rem;
    border-radius:5px; cursor:default; font-size:.82rem; }
  .op:hover { background:var(--panel); }
  .op .nm { flex:1; overflow-wrap:anywhere; }
  .op .ports { color:var(--ink-soft); font-size:.72rem;
    font-variant-numeric:tabular-nums; white-space:nowrap; }
  table { width:100%; border-collapse:collapse; }
  th,td { text-align:left; padding:.3rem .35rem; border-bottom:1px solid var(--grid);
    font-size:.82rem; }
  th { color:var(--ink-soft); font-size:.68rem; text-transform:uppercase;
    letter-spacing:.05em; font-weight:650; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  input[type=text] { width:100%; font:inherit; font-size:.8rem; padding:.2rem .35rem;
    border:1px solid var(--line); border-radius:4px; background:var(--surface);
    color:var(--ink); }
  .unit { border:1px solid var(--grid); border-radius:8px; padding:.6rem .75rem;
    margin-bottom:.6rem; background:var(--panel); }
  .unit h3 { margin:0 0 .1rem; font-size:.88rem; }
  .unit .meta { color:var(--ink-soft); font-size:.75rem; margin-bottom:.5rem; }
  .unit td:first-child { width:14rem; }
  .unit .units { color:var(--ink-soft); font-size:.72rem; }
  .code-fields { margin:.5rem 0 0; }
  .op.add { cursor:pointer; width:100%; text-align:left; border:none;
    background:none; font:inherit; font-size:.82rem; }
  .op.add:hover { background:var(--panel); }
  .op.blocked { opacity:.45; }
  .wiring { display:grid; grid-template-columns:auto 1fr; gap:.3rem .5rem;
    align-items:start; margin:.4rem 0 .6rem; font-size:.78rem; }
  .wiring .lbl { color:var(--ink-soft); padding-top:.2rem; }
  .row { display:flex; flex-wrap:wrap; gap:.3rem; align-items:center; }
  select { font:inherit; font-size:.78rem; padding:.15rem .3rem;
    border:1px solid var(--line); border-radius:4px; background:var(--surface);
    color:var(--ink); }
  select.bad, input.bad { border-color:var(--bad); background:#fbeeee; }
  input.sname { width:8.5rem; font-size:.78rem; }
  input.uname { font:inherit; font-size:.88rem; font-weight:650; width:12rem;
    border:1px solid transparent; background:none; padding:.1rem .2rem;
    border-radius:4px; }
  input.uname:hover, input.uname:focus { border-color:var(--line);
    background:var(--surface); }
  .unit .rm { float:right; font-size:.72rem; padding:.1rem .45rem; }
  .tiny { font-size:.72rem; padding:.1rem .4rem; }
  .status.warn { background:#fdf6e8; color:#8a6100; border:1px solid #f0e2c2; }
  .status ul { margin:.3rem 0 0; padding-left:1.1rem; }
  .feeds { margin-bottom:.8rem; }
  .status { font-size:.8rem; padding:.4rem .6rem; border-radius:6px;
    margin-bottom:.7rem; }
  .status.err { background:#fbeeee; color:var(--bad);
    border:1px solid #f3d4d4; }
  .status.ok { background:#eef7ee; color:#166a16; border:1px solid #d3e8d3; }
  pre { background:var(--panel); border:1px solid var(--grid); border-radius:8px;
    padding:.7rem; overflow:auto; font-size:.75rem; line-height:1.45;
    max-height:24rem; }
  /* wide tables scroll in place, so the page never scrolls sideways */
  .scroll { overflow-x:auto; }
  svg text { font:11px ui-sans-serif,system-ui,sans-serif; }
</style>
</head>
<body>
<header>
  <h1>difflow</h1>
  <span class="path" id="path"></span>
  <span class="spacer"></span>
  <button id="solve" class="primary">Solve</button>
  <button id="save">Save</button>
  <button id="showcode">Python</button>
</header>

<main>
  <section>
    <h2>Palette</h2>
    <div id="palette"></div>
  </section>

  <section>
    <h2>Flowsheet</h2>
    <div id="status"></div>
    <svg id="diagram" viewBox="0 0 600 220" width="100%"
         role="img" aria-label="Flowsheet topology"></svg>
    <div class="feeds" id="feeds"></div>
    <div id="units"></div>
  </section>

  <section>
    <h2 id="rightTitle">Streams</h2>
    <div id="right"></div>
  </section>
</main>

<script>
let DOC = null, CATALOG = {};
/* whether the server holds the flowsheet now on screen */
let SYNCED = true, LAST_ERROR = "";

const api = {
  get: p => fetch(p).then(r => r.json()),
  post: (p, b) => fetch(p, {method:"POST", headers:{"Content-Type":"application/json"},
                           body: JSON.stringify(b || {})}).then(r => r.json()),
};
function fmt(v) {
  if (typeof v !== "number" || !isFinite(v)) return String(v);
  const m = Math.abs(v);
  if (m !== 0 && (m < 1e-3 || m >= 1e5)) return v.toExponential(3);
  return v.toFixed(m >= 100 ? 1 : 4);
}
function status(msg, kind) {
  document.getElementById("status").innerHTML =
    msg ? '<div class="status ' + kind + '">' + msg + '</div>' : "";
}

/* ---- the document, as the browser holds it ------------------------ */
/* Two kinds of name live in a flowsheet and they behave differently. A
   *unit* name only labels a box. A *stream* name is the wiring: it is
   how an outlet reaches the inlet that consumes it. So renaming a
   stream has to travel -- to every consumer and to any recycle that
   mentions it -- or the rename quietly disconnects the flowsheet. */
function fsheet() { return DOC && DOC.flowsheet; }

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

/* stream name -> a phrase naming whatever produces it */
function producers() {
  const f = fsheet(), out = {};
  if (!f) return out;
  Object.keys(f.feeds).forEach(n => out[n] = "feed " + n);
  f.units.forEach(u => u.outlets.forEach(o => out[o] = "unit " + u.name));
  /* a recycle republishes a stream under a second name, so the
     destination is produced even though no outlet is called that */
  Object.entries(f.recycles || {}).forEach(([src, dest]) => {
    if (out[src] !== undefined) out[dest] = out[src];
  });
  return out;
}
function streamNames() { return Object.keys(producers()); }

function renameStream(from, to) {
  const f = fsheet();
  if (!f || !to || from === to) return false;
  if (streamNames().includes(to)) return false;
  /* rebuilt, not mutated: the key order of feeds is the display order */
  const feeds = {};
  Object.entries(f.feeds).forEach(([k, v]) => feeds[k === from ? to : k] = v);
  f.feeds = feeds;
  f.units.forEach(u => {
    u.inlets = u.inlets.map(s => s === from ? to : s);
    u.outlets = u.outlets.map(s => s === from ? to : s);
  });
  const rec = {};
  Object.entries(f.recycles || {}).forEach(([src, dest]) =>
    rec[src === from ? to : src] = (dest === from ? to : dest));
  f.recycles = rec;
  return true;
}

/* ---- adding a unit ------------------------------------------------ */
/* The catalog reports required fields but renders their defaults as
   strings, so a seed is chosen from the declared type instead. It only
   has to be the right *kind* of thing -- the point is a unit that
   constructs, with the numbers left visibly wrong for editing. */
function seedValue(p) {
  const t = (p.type || "").toLowerCase();
  if (p.name === "species_order") return (fsheet().species_order || []).slice();
  if (p.is_callable) return null;
  if (t.includes("list") || t.includes("tuple") || t.includes("sequence")) return [];
  if (t.includes("dict") || t.includes("mapping")) return {};
  if (t.includes("bool")) return false;
  if (t.includes("float") || t.includes("int")) return 1.0;
  if (t.includes("str")) return "";
  return null;                       /* an Array, or something unread */
}
function seedParams(spec) {
  const out = {};
  (spec.parameters || []).forEach(p => { if (p.required) out[p.name] = seedValue(p); });
  return out;                        /* optional fields: let the dataclass decide */
}

function uniqueUnitName(base) {
  const taken = new Set(fsheet().units.map(u => u.name));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) if (!taken.has(base + i)) return base + i;
}

/* the last stream nothing reads yet -- bolting a unit onto the end of
   the flowsheet is the common case, and it should need no wiring */
function freeStream() {
  const consumed = new Set();
  fsheet().units.forEach(u => u.inlets.forEach(s => consumed.add(s)));
  const free = streamNames().filter(s => !consumed.has(s));
  return free.length ? free[free.length - 1] : "";
}

function newUnit(opName) {
  const spec = CATALOG[opName], ports = spec.ports || {};
  const name = uniqueUnitName(opName.toLowerCase());
  const nOut = ports.n_outlets === null || ports.n_outlets === undefined
    ? 1 : ports.n_outlets;
  const outlets = nOut === 1 ? [name + "_out"]
    : Array.from({length: nOut}, (unused, i) => name + "_out" + (i + 1));
  const nIn = ports.variadic ? 1 : (ports.n_inlets || 1);
  const inlets = Array.from({length: nIn}, (unused, i) => i === 0 ? freeStream() : "");
  return {name: name, operation: opName, params: seedParams(spec),
          constructor: {}, extra_params: {}, inlets: inlets, outlets: outlets};
}

async function addUnit(opName) {
  if (!fsheet()) { status("No flowsheet loaded.", "err"); return; }
  fsheet().units.push(newUnit(opName));
  /* on success reload: the server writes back every default the Params
     dataclass filled in, which is what makes them editable here */
  if (await push()) await reload();
  render();
}

async function removeUnit(ui) {
  fsheet().units.splice(ui, 1);
  await push();
  render();
}

async function renameUnit(ui, name) {
  const f = fsheet();
  name = name.trim();
  if (name && !f.units.some((u, i) => i !== ui && u.name === name)) {
    f.units[ui].name = name;
    await push();
  }
  render();
}

/* ---- what is wrong with it ---------------------------------------- */
/* Faults the server cannot see. It rebuilds a flowsheet with a dangling
   inlet quite happily; only the solve fails, as a bare KeyError naming
   a stream. Saying so before Solve is pressed is cheaper. */
function problems() {
  const f = fsheet();
  if (!f) return [];
  const out = [], known = streamNames(), seen = {}, named = new Set();
  Object.keys(f.feeds).forEach(n => seen[n] = "feed " + n);
  f.units.forEach(u => {
    if (named.has(u.name)) out.push("two units are named " + u.name);
    named.add(u.name);
    u.inlets.forEach(s => {
      if (!s) out.push(u.name + " has an inlet connected to nothing");
      else if (!known.includes(s))
        out.push(u.name + " reads " + s + ", which nothing produces");
    });
    u.outlets.forEach(s => {
      if (!s) out.push(u.name + " has an unnamed outlet");
      else if (seen[s])
        out.push(s + " is produced by both " + seen[s] + " and unit " + u.name);
      else seen[s] = "unit " + u.name;
    });
  });
  return out;
}

function showProblems() {
  if (!SYNCED) {
    status("The model rejected that edit, and is still running the " +
           "previous version: " + esc(LAST_ERROR), "err");
    return;
  }
  const ps = problems();
  status(ps.length ? "<strong>Not ready to solve</strong><ul>" +
    ps.map(p => "<li>" + esc(p) + "</li>").join("") + "</ul>" : "", "warn");
}

/* ---- palette ------------------------------------------------------ */
function blockedReason(s) {
  const extras = s.constructor_extras || [];
  const calls = (s.parameters || [])
    .filter(p => p.required && p.is_callable).map(p => p.name);
  const parts = [];
  if (extras.length) parts.push(
    (extras.length > 1 ? "the objects " : "the object ") + extras.join(", "));
  if (calls.length) parts.push("code for " + calls.join(", "));
  return s.name + " cannot be built from a form: it needs " +
    parts.join(" and ") + ". Build it in Python, save the JSON, open it here.";
}

function renderPalette() {
  const groups = {};
  Object.values(CATALOG).forEach(s => (groups[s.category] ||= []).push(s));
  const el = document.getElementById("palette");
  el.innerHTML =
    '<div class="hint">Click to add. Dimmed units need an object or a ' +
    'function that only Python can supply.</div>' +
    Object.keys(groups).sort().map(cat =>
      '<div class="cat"><div class="cat-name">' + cat.replace(/_/g, " ") + '</div>' +
      groups[cat].sort((a, b) => a.name.localeCompare(b.name)).map(s => {
        const p = s.ports;
        const arity = (p.variadic ? "n" : p.n_inlets) + "&rarr;" +
                      (p.n_outlets === null ? "?" : p.n_outlets);
        return '<button class="op add' + (s.buildable ? "" : " blocked") + '"' +
               (s.buildable ? "" : " disabled") +
               ' data-op="' + esc(s.name) + '" title="' +
               esc(s.buildable ? (s.description || s.name) : blockedReason(s)) +
               '"><span class="nm">' + esc(s.name) + '</span>' +
               '<span class="ports">' + arity + '</span></button>';
      }).join("") + '</div>'
    ).join("");
  el.querySelectorAll("button.add:not(.blocked)").forEach(b =>
    b.addEventListener("click", () => addUnit(b.dataset.op)));
}

/* ---- diagram ------------------------------------------------------ */
function renderDiagram() {
  const svg = document.getElementById("diagram");
  const f = fsheet();
  if (!f) { svg.innerHTML = ""; return; }
  const units = f.units, feeds = Object.keys(f.feeds);
  const nodes = feeds.map(n => ({id: n, kind: "feed", label: n}))
    .concat(units.map(u => ({id: u.name, kind: "unit", label: u.name, op: u.operation})));

  /* A unit's inlets name *streams*, not nodes. Resolve each to whoever
     produces it -- a feed of the same name, or the unit that lists it
     as an outlet -- or the edge is silently dropped. */
  const producer = {};
  feeds.forEach(n => producer[n] = n);
  units.forEach(u => u.outlets.forEach(o => producer[o] = u.name));
  const recycles = f.recycles || {};
  Object.entries(recycles).forEach(([src, dest]) => {
    if (producer[src] !== undefined) producer[dest] = producer[src];
  });
  const recycled = new Set(Object.values(recycles));

  const W = 620, rowH = 96;
  const cols = Math.max(1, Math.min(nodes.length, 4));
  const rows = Math.ceil(nodes.length / cols);
  const H = Math.max(130, rows * rowH + 30);
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  const step = W / (cols + 1), pos = {};
  nodes.forEach((n, i) => pos[n.id] =
    [step * ((i % cols) + 1), 45 + Math.floor(i / cols) * rowH]);

  const HALF = 36, out = [];
  out.push('<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" ' +
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M0,0 L8,4 L0,8 z" fill="#9d9b96"/></marker></defs>');
  units.forEach(u => {
    u.inlets.forEach(stream => {
      const from = producer[stream];
      if (from === undefined || from === u.name) return;
      const a = pos[from], b = pos[u.name];
      if (!a || !b) return;
      const back = a[0] > b[0], loop = recycled.has(stream);
      /* A right-to-left edge drawn straight would run through every box
         between its ends and read as a second arrowhead on a forward
         line. Route it under the row instead, where a loop belongs. */
      const dip = Math.max(a[1], b[1]) + 52;   // clears the operation label
      const d = back
        ? 'M' + a[0] + ',' + (a[1]+16) + ' C' + a[0] + ',' + dip + ' ' +
          b[0] + ',' + dip + ' ' + b[0] + ',' + (b[1]+16)
        : 'M' + (a[0]+HALF) + ',' + a[1] + ' L' + (b[0]-HALF) + ',' + b[1];
      out.push('<path d="' + d + '" fill="none" stroke="' +
        (loop ? "#eb6834" : "#9d9b96") + '" stroke-width="1.5"' +
        (loop ? ' stroke-dasharray="5 3"' : "") + ' marker-end="url(#ar)"/>');
    });
  });
  nodes.forEach(n => {
    const [x, y] = pos[n.id];
    const feed = n.kind === "feed";
    out.push('<rect x="' + (x-HALF) + '" y="' + (y-16) + '" width="' + (2*HALF) +
      '" height="32" rx="7" fill="' + (feed ? "#2a78d6" : "#fcfcfb") +
      '" stroke="' + (feed ? "#2a78d6" : "#52514e") + '" stroke-width="1.2"/>');
    out.push('<text x="' + x + '" y="' + (y+4) + '" text-anchor="middle" fill="' +
      (feed ? "#fff" : "#0b0b0b") + '">' + esc(n.label) + '</text>');
    if (n.op) out.push('<text x="' + x + '" y="' + (y+28) +
      '" text-anchor="middle" fill="#52514e" font-size="10">' + esc(n.op) + '</text>');
  });
  svg.innerHTML = out.join("");
}

/* ---- feeds --------------------------------------------------------- */
/* Only the names are editable here. A feed's composition is a stream,
   which the right-hand table already shows; renaming is what the
   wiring needs, because a feed name *is* a stream name. */
function renderFeeds() {
  const f = fsheet(), el = document.getElementById("feeds");
  if (!f) { el.innerHTML = ""; return; }
  const names = Object.keys(f.feeds);
  el.innerHTML = '<h2>Feed streams</h2>' + (names.length
    ? '<div class="row">' + names.map(n =>
        '<input class="sname" data-feed="' + esc(n) + '" value="' + esc(n) + '">'
      ).join("") + '</div>'
    : '<div class="hint">none; add one in Python</div>');
  el.querySelectorAll("input[data-feed]").forEach(inp =>
    inp.addEventListener("change", async e => {
      if (renameStream(e.target.dataset.feed, e.target.value.trim())) await push();
      render();
    }));
}

/* ---- units: wiring and parameters ---------------------------------- */
function renderUnits() {
  const el = document.getElementById("units");
  const f = fsheet();
  if (!f) { el.innerHTML = "<p>No flowsheet loaded.</p>"; return; }
  const known = streamNames();

  el.innerHTML = f.units.map((u, ui) => {
    const spec = CATALOG[u.operation] || {};
    const ports = spec.ports || {};
    /* Fields the catalog reports as callable hold code, not data. An
       empty text box beside one invites typing a value that could only
       be rejected, so they are listed as read-only instead. */
    const byName = {};
    (spec.parameters || []).forEach(p => byName[p.name] = p);
    const scalar = v => v === null || ["number","string","boolean"].includes(typeof v);
    const flatList = v => Array.isArray(v) && v.every(scalar);
    const editable = ([k, v]) => !(byName[k] || {}).is_callable &&
      (scalar(v) || flatList(v));

    const rows = Object.entries(u.params).filter(editable).map(([k, v]) => {
      const meta = byName[k] || {};
      const shown = Array.isArray(v) ? v.join(", ") : (v === null ? "" : v);
      const blank = shown === "" || (Array.isArray(v) && !v.length);
      return '<tr><td>' + esc(k) + (meta.units ? ' <span class="units">' +
        esc(meta.units) + '</span>' : "") + '</td><td><input type="text"' +
        (meta.required && blank ? ' class="bad"' : "") +
        ' data-unit="' + ui + '" data-key="' + esc(k) + '"' +
        (Array.isArray(v) ? ' data-list="1"' : "") +
        ' value="' + esc(shown) + '"></td></tr>';
    }).join("");

    const inlets = u.inlets.map((s, i) => {
      const bad = !known.includes(s);
      const opts = ['<option value="">(not connected)</option>'].concat(
        known.filter(n => !u.outlets.includes(n)).map(n =>
          '<option value="' + esc(n) + '"' + (n === s ? " selected" : "") + '>' +
          esc(n) + '</option>'));
      if (s && bad) opts.push('<option value="' + esc(s) + '" selected>' +
        esc(s) + ' (missing)</option>');
      return '<select data-unit="' + ui + '" data-inlet="' + i + '"' +
        (bad ? ' class="bad"' : "") + '>' + opts.join("") + '</select>';
    }).join("");

    const variadic = ports.variadic
      ? '<button class="tiny" data-addin="' + ui + '" title="another inlet">+</button>' +
        (u.inlets.length > 1
          ? '<button class="tiny" data-delin="' + ui + '">&minus;</button>' : "")
      : "";

    const outlets = u.outlets.map((s, i) =>
      '<input class="sname' + (s ? "" : " bad") + '" data-unit="' + ui +
      '" data-outlet="' + i + '" value="' + esc(s) + '">').join("");

    const code = Object.keys(u.params).filter(k => (byName[k] || {}).is_callable);
    return '<div class="unit">' +
      '<button class="rm" data-rm="' + ui + '">Remove</button>' +
      '<h3><input class="uname" data-uname="' + ui + '" value="' + esc(u.name) +
      '" title="unit name"></h3>' +
      '<div class="meta">' + esc(u.operation) +
      (spec.description ? ' &nbsp;·&nbsp; ' + esc(spec.description) : "") + '</div>' +
      '<div class="wiring">' +
      '<span class="lbl">in</span><span class="row">' +
      (inlets || '<span class="hint">none</span>') + variadic + '</span>' +
      '<span class="lbl">out</span><span class="row">' +
      (outlets || '<span class="hint">none</span>') + '</span></div>' +
      (rows ? '<table><tbody>' + rows + '</tbody></table>'
            : '<div class="meta">no editable parameters</div>') +
      (code.length ? '<div class="meta code-fields">set in code: ' +
        esc(code.join(", ")) + '</div>' : "") + '</div>';
  }).join("");

  el.querySelectorAll("input[data-key]").forEach(inp =>
    inp.addEventListener("change", async e => {
      const u = f.units[+e.target.dataset.unit];
      const raw = e.target.value.trim();
      if (e.target.dataset.list) {
        const parts = raw === "" ? [] : raw.split(",").map(x => x.trim());
        const nums = parts.map(Number);
        /* a list of numbers must not arrive as a list of strings */
        u.params[e.target.dataset.key] =
          parts.length && nums.every(n => isFinite(n)) ? nums : parts;
      } else {
        const n = Number(raw);
        /* isFinite, not !isNaN: JSON.stringify writes Infinity as null,
           so a non-finite value has to travel back as the string the
           server sent, which it knows how to restore. */
        u.params[e.target.dataset.key] =
          raw === "" ? null : (isFinite(n) && raw !== "" ? n : raw);
      }
      await push();
      render();
    }));

  el.querySelectorAll("select[data-inlet]").forEach(sel =>
    sel.addEventListener("change", async e => {
      f.units[+e.target.dataset.unit].inlets[+e.target.dataset.inlet] = e.target.value;
      await push();
      render();
    }));

  el.querySelectorAll("input[data-outlet]").forEach(inp =>
    inp.addEventListener("change", async e => {
      const u = f.units[+e.target.dataset.unit];
      if (renameStream(u.outlets[+e.target.dataset.outlet], e.target.value.trim()))
        await push();
      render();
    }));

  el.querySelectorAll("input[data-uname]").forEach(inp =>
    inp.addEventListener("change", e => renameUnit(+e.target.dataset.uname, e.target.value)));

  el.querySelectorAll("button[data-rm]").forEach(b =>
    b.addEventListener("click", () => removeUnit(+b.dataset.rm)));

  el.querySelectorAll("button[data-addin]").forEach(b =>
    b.addEventListener("click", async () => {
      f.units[+b.dataset.addin].inlets.push("");
      await push();
      render();
    }));

  el.querySelectorAll("button[data-delin]").forEach(b =>
    b.addEventListener("click", async () => {
      f.units[+b.dataset.delin].inlets.pop();
      await push();
      render();
    }));
}

function render() { renderDiagram(); renderFeeds(); renderUnits(); showProblems(); }

/* ---- talking to the model ------------------------------------------ */
/* A rejected edit leaves the server running the *previous* flowsheet
   while the browser shows the new one. Solving then would report
   numbers for a model that is not on screen, so the flag gates it. */
async function push() {
  const r = await api.post("/api/flowsheet", fsheet());
  SYNCED = !!r.ok;
  LAST_ERROR = r.ok ? "" : (r.error || "rejected");
  return SYNCED;
}

async function reload() {
  DOC = await api.get("/api/flowsheet");
  document.getElementById("path").textContent = DOC.path || "(unsaved)";
}

/* ---- right panel --------------------------------------------------- */
async function showStreams() {
  if (!SYNCED || problems().length) { showProblems(); return; }
  document.getElementById("rightTitle").textContent = "Streams";
  const r = await api.post("/api/solve");
  if (!r.ok) { status(esc(r.error), "err"); document.getElementById("right").innerHTML = "";
               return; }
  status("Solved" + (r.iterations != null ? " in " + r.iterations + " iterations" : ""), "ok");
  const names = Object.keys(r.streams);
  const keys = [...new Set(names.flatMap(n => Object.keys(r.streams[n])))];
  document.getElementById("right").innerHTML =
    '<div class="scroll"><table><thead><tr><th>Stream</th>' +
    keys.map(k => '<th style="text-align:right">' + k + '</th>').join("") +
    '</tr></thead><tbody>' +
    names.map(n => '<tr><td>' + n + '</td>' +
      keys.map(k => '<td class="num">' +
        (n in r.streams && k in r.streams[n] ? fmt(r.streams[n][k]) : "") +
        '</td>').join("") + '</tr>').join("") +
    '</tbody></table></div>';
}

async function showCode() {
  document.getElementById("rightTitle").textContent = "Python";
  const r = await api.get("/api/code");
  document.getElementById("right").innerHTML = r.error
    ? '<div class="status err">' + esc(r.error) + '</div>'
    : "<pre>" + r.source.replace(/[&<>]/g, c =>
        ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])) + "</pre>";
}

/* ---- wiring -------------------------------------------------------- */
document.getElementById("solve").addEventListener("click", showStreams);
document.getElementById("showcode").addEventListener("click", showCode);
document.getElementById("save").addEventListener("click", async () => {
  const r = await api.post("/api/save");
  status(r.ok ? "Saved to " + esc(r.path) : esc(r.error), r.ok ? "ok" : "err");
});

(async function start() {
  CATALOG = await api.get("/api/catalog");
  await reload();
  renderPalette();
  render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
