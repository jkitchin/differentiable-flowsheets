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


# ---------------------------------------------------------------------
# Editing the tear
# ---------------------------------------------------------------------


@pytest.mark.parametrize("acceleration", ["anderson", "wegstein"])
@pytest.mark.parametrize("where", ["iteration", "accel"])
def test_set_tear_runs_the_next_pass_on_it_and_is_recorded(acceleration, where):
    fs = loop()
    runner = DebugRunner(fs, {"acceleration": acceleration, "max_iter": 200})
    runner.start()
    runner.command(f"stop-at {where}")
    runner.command("continue")
    edit = runner.command("set tear recycle.F_A 5")
    assert edit["ok"], edit["output"]
    assert runner.command("p next")["data"]["x_next"][0] == 5.0
    runner.command("stop-at clear")
    end = runner.command("continue")["next"]
    assert end["checkpoint"] == "terminated"
    assert [e["value"] for e in end["edits"]] == [5.0]
    runner.command("continue")
    assert fs.last_solve_converged
    assert len(fs.last_solve_debug_edits) == 1
    fs.solve()
    assert fs.last_solve_debug_edits == []


@pytest.mark.parametrize("command", [
    "set tear recycle.F_A",          # no value
    "set param R1.k 2",              # only tears are editable
    "set tear nowhere.F_A 1",
    "set tear recycle.F_Z 1",
])
def test_set_refuses_what_it_cannot_do(command):
    runner = DebugRunner(loop())
    runner.start()
    runner.command("stop-at iteration")
    runner.command("continue")
    assert runner.command(command)["ok"] is False
    runner.close()


def test_set_tear_needs_an_iteration_pause():
    runner = DebugRunner(loop())
    runner.start()                                    # at solve_start
    assert runner.command("set tear recycle.F_A 1")["ok"] is False
    runner.close()


# ---------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------


def test_record_keeps_every_checkpoint_and_the_answer():
    fs = loop()
    trace = debug.record(fs, {"acceleration": "wegstein"})
    assert trace["protocol"] == "difflow-trace/1"
    assert trace["units"] == ["M1", "R1", "S1"]
    assert trace["tears"] == ["recycle"]
    kinds = {f["checkpoint"] for f in trace["frames"]}
    assert {"solve_start", "after_unit", "iteration", "terminated"} <= kinds
    assert trace["result"]["ok"] and trace["result"]["converged"]
    json.dumps(trace)                                 # it crosses a wire
    assert debug.current() is None


def test_record_reports_a_raising_solve_instead_of_raising():
    trace = debug.record(loop(broken))
    assert trace["result"]["ok"] is False
    assert "unit blew up" in trace["result"]["error"]
    assert any(f["checkpoint"] == "unit_error" for f in trace["frames"])


def test_record_stops_recording_not_solving_at_its_limit():
    fs = loop()
    trace = debug.record(fs, max_frames=3)
    assert len(trace["frames"]) == 3 and trace["truncated"]
    assert trace["result"]["converged"]


# ---------------------------------------------------------------------
# What-if replays
# ---------------------------------------------------------------------


def _iteration_frames(trace):
    return [i for i, f in enumerate(trace["frames"]) if f["checkpoint"] == "iteration"]


def test_a_replay_with_an_edit_matches_the_recording_up_to_the_edit():
    fs = loop()
    base = debug.record(fs, {"max_iter": 200})
    k = _iteration_frames(base)[0]
    edit = {"frame": k, "stream": "recycle", "field": "F_A", "value": 5.0}
    replay = debug.record(fs, {"max_iter": 200}, edits=[edit],
                          baseline=base["frames"])
    result = replay["result"]
    assert result["diverged_at"] is None and result["refused"] == []
    assert [e["value"] for e in result["edits"]] == [5.0]
    assert replay["frames"][k]["edits"][0]["was"] is not None
    assert result["converged"]
    assert fs.last_solve_debug_edits                 # the replay says it was edited


def test_a_replay_of_a_changed_flowsheet_reports_where_it_diverged():
    fs = loop()
    base = debug.record(fs)
    k = _iteration_frames(base)[0]
    fs.feeds["feed"] = make_stream({"A": 2.0}, 300.0, 1e5)
    replay = debug.record(fs, edits=[{"frame": k, "stream": "recycle",
                                      "field": "F_A", "value": 5.0}],
                          baseline=base["frames"])
    assert replay["result"]["diverged_at"] == 0


def test_a_replay_lists_the_edits_it_could_not_make():
    fs = loop()
    base = debug.record(fs)
    replay = debug.record(fs, edits=[
        {"frame": 0, "stream": "recycle", "field": "F_A", "value": 1.0},     # solve_start
        {"frame": 10_000, "stream": "recycle", "field": "F_A", "value": 1.0},
    ], baseline=base["frames"])
    assert len(replay["result"]["refused"]) == 2
    assert replay["result"]["edits"] == []
