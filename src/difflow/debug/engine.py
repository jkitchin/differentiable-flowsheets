"""The command engine: breakpoint state and what every command means.

One engine, several front ends (the terminal REPL, the JSON protocol, the
threaded runner the editor and agents use). The front ends only move
strings; the engine decides, so a person at a prompt and an agent on a
pipe get the same debugger. The design is pounce's and discopt's.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from .checkpoints import ALIASES, EVENTS, METRICS, Checkpoint, resolve_checkpoint
from .context import plain_stream

if TYPE_CHECKING:
    from .context import DebugContext
    from .session import DebugSession


class Control(Enum):
    """What a command tells the paused solve to do next."""

    NONE = auto()      # stay paused (an inspection)
    STEP = auto()      # run to the next tear iteration
    NEXT = auto()      # run to the next unit (or iteration)
    STEPI = auto()     # run to the next checkpoint of any kind
    CONTINUE = auto()  # run to the next breakpoint, or to the end
    QUIT = auto()      # stop the solve: it raises DebugQuit


@dataclass
class CommandResult:
    """One command's outcome: text for a person, data for a program."""

    output: list[str] = field(default_factory=list)
    control: Control = Control.NONE
    data: dict | None = None
    ok: bool = True

    def line(self, text: str) -> "CommandResult":
        self.output.append(text)
        return self


def fail(text: str) -> CommandResult:
    return CommandResult([text], ok=False)


_ATOM = re.compile(r"^\s*([a-z_]+)\s*(<=|>=|==|<|>)\s*([-+0-9.eE]+|nan|inf)\s*$")


@dataclass
class Condition:
    """``metric OP value`` atoms joined by ``&&``."""

    raw: str
    clauses: list[tuple[str, str, float]]

    @classmethod
    def parse(cls, expr: str) -> "Condition":
        if "||" in expr:
            raise ValueError("'||' is not supported; add a second 'break if'")
        clauses = []
        for atom in expr.replace("(", " ").replace(")", " ").split("&&"):
            m = _ATOM.match(atom)
            if not m:
                raise ValueError(f"cannot read condition {atom.strip()!r}; "
                                 "expected e.g. 'residual>1e3 && iter>5'")
            if m.group(1) not in METRICS:
                raise ValueError(f"unknown metric {m.group(1)!r}; "
                                 f"one of {', '.join(METRICS)}")
            clauses.append((m.group(1), m.group(2), float(m.group(3))))
        return cls(expr.strip(), clauses)

    def holds(self, metrics: dict[str, float]) -> bool:
        for name, op, value in self.clauses:
            lhs = metrics.get(name)
            if lhs is None:
                return False
            if not {"<": lhs < value, "<=": lhs <= value, ">": lhs > value,
                    ">=": lhs >= value,
                    "==": abs(lhs - value) <= 1e-12 * max(1.0, abs(value))}[op]:
                return False
        return True


def _fmt(value) -> str:
    if isinstance(value, (str, bool)):
        return str(value)
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return repr(value)


def _stream_lines(name: str, stream: dict) -> list[str]:
    lines = [f"{name}:"]
    for key, value in stream.items():
        lines.append(f"  {key:<12} {_fmt(value)}")
    return lines


class DebugCommandEngine:
    """Holds breakpoints and watches, and executes commands against a context."""

    def __init__(self) -> None:
        self.iter_breaks: set[int] = set()
        self.temp_breaks: set[int] = set()
        self.unit_breaks: set[str] = set()
        self.conditions: list[Condition] = []
        self.events: set[str] = set()
        self.watches: list[str] = []

    # -- the breakpoint predicate ----------------------------------------

    def hit_reason(self, ctx: "DebugContext") -> str | None:
        """Why a breakpoint fires at ``ctx``, or None if none does.

        Iteration and condition breakpoints are tested at ``iteration``;
        unit breakpoints at ``before_unit``; events wherever they occur.
        """
        if ctx.event and ctx.event in self.events:
            return f"event: {ctx.event}"
        if (ctx.checkpoint is Checkpoint.BEFORE_UNIT and ctx.unit is not None
                and ctx.unit.name in self.unit_breaks):
            return f"unit {ctx.unit.name}"
        if ctx.checkpoint is Checkpoint.ITERATION:
            it = ctx.iteration
            if it in self.iter_breaks or it in self.temp_breaks:
                self.temp_breaks.discard(it)
                return f"iteration {it}"
            metrics = ctx.metrics()
            for cond in self.conditions:
                if cond.holds(metrics):
                    return f"condition: {cond.raw}"
        return None

    # -- dispatch --------------------------------------------------------

    def execute(self, raw: str, ctx: "DebugContext",
                session: "DebugSession") -> CommandResult:
        raw = raw.strip()
        if not raw:
            return CommandResult()
        cmd, _, rest = raw.partition(" ")
        handler = _DISPATCH.get(cmd.lower())
        if handler is None:
            return fail(f"unknown command {cmd!r} (try 'help')")
        try:
            return handler(self, rest.strip(), ctx, session)
        except Exception as exc:  # noqa: BLE001 -- a bad command must not kill the solve
            return fail(f"{cmd}: {type(exc).__name__}: {exc}")

    # -- watches, rendered at every pause --------------------------------

    def render_watches(self, ctx: "DebugContext") -> list[dict]:
        out = []
        for target in self.watches:
            res = _print(self, target, ctx, None)
            out.append({"target": target, "ok": res.ok,
                        "text": "\n".join(res.output), "data": res.data})
        return out


# =====================================================================
# Commands. Each is (engine, args, ctx, session) -> CommandResult.
# =====================================================================


def _resume(control: Control, text: str):
    def run(engine, args, ctx, session):
        return CommandResult([text], control=control)
    return run


def _run_to(engine, args, ctx, session):
    target = int(args)
    engine.temp_breaks.add(target)
    return CommandResult([f"running to iteration {target}"],
                         control=Control.CONTINUE)


def _detach(engine, args, ctx, session):
    session.detach()
    return CommandResult(["detached; the solve runs to the end"],
                         control=Control.CONTINUE)


def _break(engine, args, ctx, session, *, temporary=False):
    if not args:
        lines = []
        lines += [f"iteration {i}" for i in sorted(engine.iter_breaks)]
        lines += [f"iteration {i} (once)" for i in sorted(engine.temp_breaks)]
        lines += [f"unit {u}" for u in sorted(engine.unit_breaks)]
        lines += [f"if {c.raw}" for c in engine.conditions]
        lines += [f"on {e}" for e in sorted(engine.events)]
        return CommandResult(lines or ["no breakpoints"], data={"breakpoints": lines})
    head, _, rest = args.partition(" ")
    rest = rest.strip()
    if head == "clear":
        engine.iter_breaks.clear(); engine.temp_breaks.clear()
        engine.unit_breaks.clear(); engine.conditions.clear(); engine.events.clear()
        return CommandResult(["all breakpoints cleared"])
    if head == "del":
        return _break_del(engine, rest)
    if head == "unit":
        known = {u.name for u in ctx.flowsheet.units}
        if rest not in known:
            return fail(f"no unit {rest!r}; units: {', '.join(sorted(known))}")
        engine.unit_breaks.add(rest)
        return CommandResult([f"breakpoint before unit {rest}"])
    if head == "if":
        cond = Condition.parse(rest)
        engine.conditions.append(cond)
        return CommandResult([f"conditional breakpoint: {cond.raw}"])
    if head == "on":
        if rest not in EVENTS:
            return fail(f"unknown event {rest!r}; one of {', '.join(EVENTS)}")
        engine.events.add(rest)
        return CommandResult([f"breakpoint on {rest} ({EVENTS[rest]})"])
    if head.lstrip("-").isdigit():
        (engine.temp_breaks if temporary else engine.iter_breaks).add(int(head))
        return CommandResult([f"breakpoint at iteration {head}"
                              + (" (once)" if temporary else "")])
    # A bare unit name is the common case; allow it without `unit`.
    if head in {u.name for u in ctx.flowsheet.units}:
        engine.unit_breaks.add(head)
        return CommandResult([f"breakpoint before unit {head}"])
    return fail(f"cannot read breakpoint {args!r}; try 'help break'")


def _break_del(engine, what):
    for bucket in (engine.unit_breaks, engine.events):
        if what in bucket:
            bucket.discard(what)
            return CommandResult([f"deleted {what}"])
    if what.isdigit():
        n = int(what)
        engine.iter_breaks.discard(n); engine.temp_breaks.discard(n)
        return CommandResult([f"deleted iteration {n}"])
    before = len(engine.conditions)
    engine.conditions = [c for c in engine.conditions if c.raw != what]
    if len(engine.conditions) < before:
        return CommandResult([f"deleted condition {what}"])
    return fail(f"no breakpoint {what!r}")


def _tbreak(engine, args, ctx, session):
    return _break(engine, args, ctx, session, temporary=True)


def _stop_at(engine, args, ctx, session):
    if not args:
        names = sorted(c.value for c in session.stop_checkpoints)
        return CommandResult([", ".join(names) or "none"], data={"stop_at": names})
    if args == "clear":
        session.stop_checkpoints.clear()
        return CommandResult(["stop-at cleared"])
    cp = resolve_checkpoint(args)
    session.stop_checkpoints.add(cp)
    return CommandResult([f"will stop at every {cp.value}"])


def _info(engine, args, ctx, session):
    ev = ctx.to_event()
    lines = [f"{k:<12} {_fmt(v)}" for k, v in ev.items() if v is not None]
    fs = ctx.flowsheet
    lines.append(f"{'tears':<12} {', '.join(ctx.tear_names) or '-'}")
    lines.append(f"{'units':<12} {len(fs.units)}")
    return CommandResult(lines, data=ev)


def _where(engine, args, ctx, session):
    """The calculation order with a marker on the current unit."""
    fs = ctx.flowsheet
    current = ctx.unit.name if ctx.unit is not None else None
    tears = set(ctx.tear_names or fs.recycles.values())
    lines = [f"{ctx.checkpoint.value}  phase={ctx.phase}  "
             f"iteration={_fmt(ctx.iteration)}"]
    for unit in fs.units:
        mark = "->" if unit.name == current else "  "
        ins = ", ".join(n + ("*" if n in tears else "") for n in unit.inlet_names)
        outs = ", ".join(unit.outlet_names)
        bp = " [b]" if unit.name in engine.unit_breaks else ""
        lines.append(f"{mark} {unit.name}{bp}: ({ins}) -> ({outs})")
    lines.append("   (* = tear stream)")
    return CommandResult(lines, data={"order": [u.name for u in fs.units],
                                      "current": current})


def _units(engine, args, ctx, session):
    names = [u.name for u in ctx.flowsheet.units]
    return CommandResult([", ".join(names)], data={"units": names})


def _streams(engine, args, ctx, session):
    names = ctx.known_streams()
    return CommandResult([", ".join(names)], data={"streams": names})


def _unit_lines(ctx, unit) -> CommandResult:
    fs = ctx.flowsheet
    op = unit.operation
    lines = [f"unit {unit.name} ({type(op).__name__})",
             f"  inlets   {', '.join(unit.inlet_names)}",
             f"  outlets  {', '.join(unit.outlet_names)}"]
    params = getattr(op, "params", None)
    shown = {}
    if params is not None and hasattr(params, "keys"):
        for key in params.keys():
            value = params[key]
            if callable(value):
                continue
            try:
                shown[key] = float(value)
            except (TypeError, ValueError):
                if isinstance(value, (str, int, bool)):
                    shown[key] = value
    shown.update({k: v for k, v in unit.params.items() if not callable(v)})
    for key, value in shown.items():
        lines.append(f"  {key:<14} {_fmt(value)}")
    info = ctx.info if (ctx.unit is unit and ctx.info is not None) else \
        (getattr(fs, "last_solve_unit_info", None) or {}).get(unit.name)
    if info:
        lines.append("  info:")
        for key, value in info.items():
            if isinstance(value, (int, float, bool, str)) or hasattr(value, "shape") \
                    and getattr(value, "shape", None) == ():
                lines.append(f"    {key:<12} {_fmt(value)}")
    data = {"unit": unit.name, "inlets": list(unit.inlet_names),
            "outlets": list(unit.outlet_names),
            "params": {k: (v if isinstance(v, (str, bool)) else _num(v))
                       for k, v in shown.items()}}
    return CommandResult(lines, data=data)


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else str(out)


def _print(engine, args, ctx, session):
    target = args.strip()
    if not target:
        return fail("print what? a stream, a unit, 'tear', 'history' or a metric")
    head, _, rest = target.partition(" ")
    if head in ("tear", "tears"):
        rows = ctx.tear_breakdown()
        if not rows:
            return fail("no tear iterate here (only at iteration / after_accel)")
        limit = len(rows) if rest == "all" else 10
        lines = [f"{'stream':<14}{'field':<10}{'x':>14}{'g(x)':>14}{'|g-x|':>12}"]
        for r in rows[:limit]:
            lines.append(f"{r['stream']:<14}{r['field']:<10}{_fmt(r['x']):>14}"
                         f"{_fmt(r['g']):>14}{_fmt(r['delta']):>12}")
        if len(rows) > limit:
            lines.append(f"... {len(rows) - limit} more ('print tear all')")
        return CommandResult(lines, data={"tear": rows})
    if head == "history":
        from difflow.diagnostics import classify_history
        hist = list(ctx.history)
        regime = classify_history(hist, _converged(ctx))
        lines = [f"{i:>4}  {_fmt(r)}" for i, r in enumerate(hist)]
        lines.append(f"reads as: {regime}")
        return CommandResult(lines, data={"history": hist, "regime": regime})
    if head == "next":
        if ctx.x_next is None:
            return fail("no accelerated guess here (only at after_accel)")
        values = [_num(v) for v in ctx.x_next]
        return CommandResult([", ".join(_fmt(v) for v in values)],
                             data={"x_next": values})
    metrics = ctx.metrics()
    if head in metrics:
        return CommandResult([f"{head} = {_fmt(metrics[head])}"],
                             data={head: metrics[head]})
    if head in METRICS:
        return fail(f"{head} is not defined at {ctx.checkpoint.value}")
    if head.startswith("unit:") or (ctx.stream(head) is None and "." not in head):
        name = head.removeprefix("unit:")
        for unit in ctx.flowsheet.units:
            if unit.name == name:
                return _unit_lines(ctx, unit)
    name, dot, fname = head.removeprefix("stream:").partition(".")
    stream = ctx.stream(name)
    if stream is None:
        return fail(f"no stream or unit {name!r} here; known streams: "
                    f"{', '.join(ctx.known_streams())}")
    if dot:
        if fname not in stream:
            return fail(f"{name} has no {fname!r}; fields: {', '.join(stream)}")
        return CommandResult([f"{name}.{fname} = {_fmt(stream[fname])}"],
                             data={"value": _num(stream[fname])})
    return CommandResult(_stream_lines(name, stream),
                         data={"stream": name, "values": plain_stream(stream)})


def _watch(engine, args, ctx, session):
    if not args:
        return CommandResult(engine.watches or ["no watches"],
                             data={"watches": list(engine.watches)})
    head, _, rest = args.partition(" ")
    if head == "clear":
        engine.watches.clear()
        return CommandResult(["watches cleared"])
    if head == "del":
        engine.watches = [w for w in engine.watches if w != rest.strip()]
        return CommandResult([f"unwatched {rest.strip()}"])
    engine.watches.append(args)
    return CommandResult([f"watching {args}"])


def _diagnose(engine, args, ctx, session):
    from difflow.diagnostics import classify_history, solve_findings
    lines, data = [], {}
    if ctx.checkpoint is Checkpoint.TERMINATED:
        findings = [f.to_dict() for f in solve_findings(ctx.flowsheet)]
        data["findings"] = findings
        for f in findings:
            lines.append(f"[{f['severity']}] {f['detail']}")
            if f["remedies"]:
                lines.append(f"    try: {', '.join(f['remedies'])}")
    regime = classify_history(list(ctx.history), _converged(ctx))
    data["regime"] = regime
    lines.append(f"residual history reads as: {regime}")
    worst = ctx.tear_breakdown()[:3]
    if worst:
        lines.append("largest tear residuals: " + "; ".join(
            f"{r['stream']}.{r['field']} {_fmt(r['delta'])}" for r in worst))
    bad = [n for n in ctx.known_streams()
           if ctx.stream(n) is not None and not _stream_finite(ctx.stream(n))]
    if bad:
        lines.append(f"non-finite streams: {', '.join(bad)}")
        data["nonfinite"] = bad
    return CommandResult(lines, data=data)


def _converged(ctx) -> bool | None:
    """The verdict, once there is one: only at ``terminated``."""
    if ctx.checkpoint is Checkpoint.TERMINATED:
        return getattr(ctx.flowsheet, "last_solve_converged", None)
    return None


def _stream_finite(stream):
    from .context import stream_is_finite
    return stream_is_finite(stream)


def _save(engine, args, ctx, session):
    path = args or "difflow-debug.json"
    payload = {"pause": ctx.to_event(),
               "streams": {n: plain_stream(ctx.stream(n)) for n in ctx.known_streams()},
               "history": list(ctx.history), "tear": ctx.tear_breakdown()}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return CommandResult([f"wrote {path}"], data={"path": path})


HELP = """\
flow       step|s (next iteration)  next|n (next unit)  stepi|si (any checkpoint)
           continue|c  run N  detach  quit|q
break      break [N | UNIT | unit UNIT | if COND | on EVENT | del X | clear]
           tbreak N   stop-at [CHECKPOINT | clear]
inspect    info|i  where|w  units  streams  print|p TARGET  watch [TARGET|del X|clear]
           diagnose  save [PATH]
targets    a stream, stream.field, a unit (or unit:NAME), tear [all], history,
           next (the accelerated guess), or a metric
"""


def _help(engine, args, ctx, session):
    lines = HELP.splitlines()
    if args.startswith("break"):
        lines = ["conditions: " + ", ".join(f"{k} ({v})" for k, v in METRICS.items()),
                 "events: " + ", ".join(f"{k} ({v})" for k, v in EVENTS.items())]
    if args.startswith("stop"):
        lines = ["checkpoints: " + ", ".join(c.value for c in Checkpoint),
                 "aliases: " + ", ".join(f"{k}={v.value}" for k, v in ALIASES.items())]
    return CommandResult(lines)


_DISPATCH: dict[str, Callable] = {
    "step": _resume(Control.STEP, "stepping to the next iteration"),
    "s": _resume(Control.STEP, "stepping to the next iteration"),
    "next": _resume(Control.NEXT, "stepping to the next unit"),
    "n": _resume(Control.NEXT, "stepping to the next unit"),
    "stepi": _resume(Control.STEPI, "stepping to the next checkpoint"),
    "si": _resume(Control.STEPI, "stepping to the next checkpoint"),
    "continue": _resume(Control.CONTINUE, "continuing"),
    "c": _resume(Control.CONTINUE, "continuing"),
    "run": _run_to, "r": _run_to,
    "detach": _detach,
    "quit": _resume(Control.QUIT, "stopping the solve"),
    "q": _resume(Control.QUIT, "stopping the solve"),
    "break": _break, "b": _break,
    "tbreak": _tbreak, "tb": _tbreak,
    "stop-at": _stop_at,
    "info": _info, "i": _info,
    "where": _where, "w": _where,
    "units": _units, "streams": _streams,
    "print": _print, "p": _print,
    "watch": _watch, "display": _watch,
    "diagnose": _diagnose,
    "save": _save,
    "help": _help, "h": _help, "?": _help,
}

#: Canonical verbs, as the JSON hello advertises them.
COMMANDS = ["step", "next", "stepi", "continue", "run", "detach", "quit",
            "break", "tbreak", "stop-at", "info", "where", "units", "streams",
            "print", "watch", "diagnose", "save", "help"]
