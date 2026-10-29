"""The interactive flowsheet debugger (``difflow.debug``).

Contracts only: it pauses where it says, its commands answer, attaching
it does not change the answer, and it stays out of traced solves.
"""

import io
import json

import jax
import jax.numpy as jnp
import pytest

from difflow import debug
from difflow.debug import Checkpoint, DebugQuit, DebugSession
from difflow.debug.jsonproto import JsonFrontend
from difflow.debug.repl import ReplFrontend
from difflow.debug.runner import DebugRunner
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import make_stream


def mixer(feed, tear):
    return make_stream({"A": feed["F_A"] + tear["F_A"]}, feed["T"], feed["P"])


def halve(s):
    return make_stream({"A": 0.5 * s["F_A"]}, s["T"], s["P"])


def split(s):
    return (make_stream({"A": 0.5 * s["F_A"]}, s["T"], s["P"]),
            make_stream({"A": 0.5 * s["F_A"]}, s["T"], s["P"]))


def poisoned(s):
    return make_stream({"A": s["F_A"] * jnp.nan}, s["T"], s["P"])


def broken(s):
    raise RuntimeError("unit blew up")


def loop(middle=halve) -> Flowsheet:
    fs = Flowsheet(species_order=["A"])
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("M1", mixer, ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("R1", middle, ["mixed"], ["reacted"]))
    fs.add_unit(Unit("S1", split, ["reacted"], ["back", "product"]))
    fs.add_recycle("back", "recycle")
    return fs


def scripted(*commands) -> DebugSession:
    return DebugSession(ReplFrontend(script=list(commands), stdin=io.StringIO(""),
                                     stdout=io.StringIO()))


# ---------------------------------------------------------------------
# Runs, pauses, answers
# ---------------------------------------------------------------------


def test_attached_solve_gives_the_detached_answer():
    plain = loop().solve(acceleration="wegstein")["product"]["F_A"]
    session = scripted("break 1", "next", "stepi", "continue", "continue", "continue")
    with debug.attach(session):
        out = loop().solve(acceleration="wegstein")["product"]["F_A"]
    assert float(out) == float(plain)
    assert session.checkpoints_seen > 0


def test_runner_pauses_at_start_then_at_a_unit_breakpoint():
    runner = DebugRunner(loop())
    assert runner.start()["checkpoint"] == "solve_start"
    assert runner.command("break R1")["ok"]
    event = runner.command("continue")["next"]
    assert (event["checkpoint"], event["unit"]) == ("before_unit", "R1")
    runner.close()


@pytest.mark.parametrize("command", [
    "info", "where", "units", "streams", "p feed", "p feed.F_A", "p R1",
    "p unit:R1", "p history", "watch feed.T", "diagnose", "help", "help break",
    "stop-at", "break",
])
def test_inspection_commands_answer(command):
    runner = DebugRunner(loop())
    runner.start()
    result = runner.command(command)
    assert result["ok"], result["output"]
    assert not result["resumed"]
    runner.close()


def test_tear_breakdown_at_an_iteration():
    runner = DebugRunner(loop(), {"acceleration": "anderson"})
    runner.start()
    runner.command("stop-at iteration")
    runner.command("continue")
    result = runner.command("p tear")
    assert result["ok"]
    assert {row["stream"] for row in result["data"]["tear"]} == {"recycle"}
    runner.close()


@pytest.mark.parametrize("command", [
    "frobnicate", "p nowhere", "p feed.nope", "break unit nowhere",
    "break on nothing", "break if bogus<1", "break if residual<<1",
    "stop-at nowhere", "p tear", "p residual", "run x",
])
def test_bad_commands_fail_without_stopping_the_solve(command):
    runner = DebugRunner(loop())
    runner.start()
    assert runner.command(command)["ok"] is False
    assert runner.command("continue")["next"]["checkpoint"] == "terminated"
    runner.close()


def test_quit_raises_debug_quit():
    with pytest.raises(DebugQuit):
        with debug.attach(scripted("quit")):
            loop().solve()


def test_eof_detaches_and_finishes():
    session = DebugSession(ReplFrontend(stdin=io.StringIO(""), stdout=io.StringIO()))
    with debug.attach(session):
        loop().solve()
    assert session.detached


# ---------------------------------------------------------------------
# Post mortem and events
# ---------------------------------------------------------------------


def test_on_error_stops_where_a_nan_is_born():
    runner = DebugRunner(loop(poisoned), {"on_nonconvergence": "ignore"},
                         on_error=True)
    event = runner.start()
    assert (event["unit"], event["trigger"]) == ("R1", "nonfinite")
    runner.close()


def test_a_raising_unit_stops_at_unit_error_and_still_raises():
    runner = DebugRunner(loop(broken))
    runner.start()
    event = runner.command("continue")["next"]
    assert (event["checkpoint"], event["unit"]) == ("unit_error", "R1")
    assert "unit blew up" in event["error"]
    runner.command("continue")
    final = runner.command("continue")["next"]
    assert final["event"] == "finished" and final["ok"] is False


def test_on_error_never_stops_a_clean_solve():
    runner = DebugRunner(loop(), on_error=True)
    assert runner.start()["event"] == "finished"


# ---------------------------------------------------------------------
# Scope: traced solves, nested solves, threads
# ---------------------------------------------------------------------


def test_a_traced_solve_does_not_pause_and_its_gradient_is_finite():
    def objective(a):
        fs = loop()
        fs.feeds["feed"] = make_stream({"A": a}, 300.0, 1e5)
        return fs.solve(acceleration="anderson")["product"]["F_A"]

    session = scripted()
    with debug.attach(session):
        grad = jax.grad(objective)(1.0)
    assert jnp.isfinite(grad)
    assert session.checkpoints_seen == 0


class Recorder:
    """A front end that steps through everything and remembers where."""

    def __init__(self):
        self.seen = []

    def interact(self, ctx, session):
        self.seen.append(ctx)
        return debug.Control.STEPI


def test_a_flowsheet_solved_inside_a_unit_stays_silent():
    inner = loop()

    def nested(s):
        inner.solve()
        return halve(s)

    outer = loop(nested)
    recorder = Recorder()
    with debug.attach(DebugSession(recorder)):
        outer.solve()
    assert recorder.seen
    assert all(ctx.flowsheet is outer for ctx in recorder.seen)


def test_only_the_first_solve_is_debugged_by_default():
    session = scripted("continue", "continue")
    with debug.attach(session):
        loop().solve()
        loop().solve()
    assert session.solves == 1


def test_attach_restores_what_was_attached_before():
    outer = scripted()
    with debug.attach(outer):
        with debug.attach(scripted()):
            pass
        assert debug.current() is outer
    assert debug.current() is None


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        debug.make_session("jsn")


# ---------------------------------------------------------------------
# The JSON protocol and the command line
# ---------------------------------------------------------------------


def test_json_protocol_hello_pause_result():
    stdin = io.StringIO('{"cmd": "p", "args": ["feed"], "id": 7}\n"continue"\n"continue"\n')
    stdout = io.StringIO()
    with debug.attach(DebugSession(JsonFrontend(stdin=stdin, stdout=stdout))):
        loop().solve()
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert events[0]["event"] == "hello"
    assert set(events[0]["checkpoints"]) == {c.value for c in Checkpoint}
    assert events[1]["event"] == "pause"
    result = next(e for e in events if e.get("request_id") == 7)
    assert result["ok"] and result["data"]["stream"] == "feed"


def test_cli_dispatches_debug():
    from difflow import cli

    assert cli.COMMANDS["debug"] == ("difflow.debug.cli", "main")


def test_cli_debugs_a_script(tmp_path, monkeypatch):
    from difflow.debug import cli

    model = tmp_path / "plant.py"
    model.write_text(
        "from difflow.flowsheet import Flowsheet, Unit\n"
        "from difflow.streams import make_stream\n"
        "def halve(s):\n"
        "    return make_stream({'A': 0.5 * s['F_A']}, s['T'], s['P'])\n"
        "fs = Flowsheet(species_order=['A'])\n"
        "fs.add_feed('feed', make_stream({'A': 1.0}, 300.0, 1e5))\n"
        "fs.add_unit(Unit('H', halve, ['feed'], ['out']))\n")
    commands = tmp_path / "go.dbg"
    commands.write_text("# run it\nwhere\ncontinue\ncontinue\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main([str(model), "--script", str(commands)]) == 0
