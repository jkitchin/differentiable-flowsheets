# The Flowsheet Debugger

A recycle that will not close usually fails somewhere specific: one unit
returns a NaN on the first pass, a tear field oscillates while the others
settle, a reactor's own Newton solve quietly gives up. The solve's summary
(`last_solve_converged`, the residual history, `difflow.diagnostics`) tells
you *that* it failed. The debugger lets you watch it fail: pause the
sequential-modular solve unit by unit and tear iteration by tear iteration,
look at every stream on the way, and stop on the thing you are hunting.

It is not `pdb`. `pdb` stops on lines of Python; this stops on the
flowsheet's own structure --- *before unit R1*, *at iteration 12*, *when the
residual rises*, *where the first NaN is born* --- and its commands speak
streams, units and tears.

## Table of Contents

- [Quick start](#quick-start)
- [Where it stops](#where-it-stops)
- [Commands](#commands)
- [Breakpoints](#breakpoints)
- [Post mortem: `on_error`](#post-mortem-on-error)
- [Driving it from a program](#driving-it-from-a-program)
- [What it does not change](#what-it-does-not-change)
- [Limits](#limits)

## Quick start

From the shell, on a script that builds a flowsheet or on a saved
flowsheet:

```bash
difflow debug plant.py                      # stop before the first pass
difflow debug plant.py --on-error           # run; stop only if something fails
difflow debug plant.json --acceleration wegstein
difflow debug plant.py --script warmup.dbg  # run these commands first
difflow debug plant.py --json               # JSON protocol on stdin/stdout
```

A script runs with the debugger already attached, so the first solve it
makes is the one you step through, with its own arguments. From Python or
a notebook:

```python
from difflow import debug

with debug.attach():                    # a prompt on this terminal
    fs.solve(acceleration="wegstein")

with debug.attach(on_error=True):       # stop only post mortem
    fs.solve()
```

A short session:

```text
── difflow-dbg ── @solve_start  phase=start
   ↳ start
difflow-dbg> where
   M1: (feed, recycle*) -> (mixed)
   R1: (mixed) -> (reacted)
   S1: (reacted) -> (recycle_out, product)
difflow-dbg> break if residual<1e-4
difflow-dbg> watch recycle.F_A
difflow-dbg> continue
── difflow-dbg ── @iteration  iter 4  phase=solve  residual=1.110e-16
   ↳ condition: residual<1e-4
   [recycle.F_A] recycle.F_A = 0.5625
difflow-dbg> p tear
stream        field                  x          g(x)       |g-x|
recycle       F_B               0.9375        0.9375 1.11022e-16
...
difflow-dbg> p history
difflow-dbg> continue
```

The prompt and everything it prints go to stderr, so a script's own output
is left alone. Tab completes commands, unit names and stream names, and
history is kept in `~/.difflow_dbg_history`.

## Where it stops

| Checkpoint | Fires | What is fresh |
|---|---|---|
| `solve_start` | once, before anything runs | the feeds |
| `before_unit` | before each unit call | the unit's inlets |
| `after_unit` | after each unit call | its outlets and its `info` dict |
| `unit_error` | a unit raised (the solve re-raises after) | the inlets that broke it, the exception |
| `iteration` | the tear map was evaluated | `x`, `g(x)`, the residual, the history |
| `after_accel` | Wegstein/Anderson computed the next guess | `x_next` (`p next`) |
| `terminated` | the solve is about to return or raise | the verdict, every stream |

Unit checkpoints carry a **phase**: `init` (the pass from the feeds that
guesses the tears), `solve` (the tear iteration), `probe` (the
error-probe passes after convergence, see
[convergence](convergence.md)), `final` (the pass from the converged
tears) and `sequential` (a flowsheet with no recycle).

Stepping:

| Command | Runs to |
|---|---|
| `step` / `s` | the next `iteration` |
| `next` / `n` | the next unit (`after_unit`) or `iteration` |
| `stepi` / `si` | the next checkpoint of any kind |
| `continue` / `c` | the next breakpoint, a unit error, or the end |
| `run N` | iteration `N` |
| `stop-at CHECKPOINT` | (not a step) always stop at that checkpoint |
| `detach` | the end, without stopping again |
| `quit` / `q` | nowhere: the solve raises `DebugQuit` |

## Commands

| Command | Shows |
|---|---|
| `info` / `i` | where the solve is: checkpoint, phase, iteration, unit, residual |
| `where` / `w` | the calculation order, `->` at the current unit, `*` on tears, `[b]` on breakpoints |
| `p STREAM` | every field of a stream |
| `p STREAM.FIELD` | one value |
| `p UNIT` (or `p unit:NAME`) | the unit's ports, parameters and its last `info` |
| `p tear [all]` | every tear entry with `x`, `g(x)` and `|g - x|`, largest first: *which* field the residual is made of |
| `p history` | the residual history, and what `classify_history` reads in it |
| `p next` | the accelerated next guess, at `after_accel` |
| `p residual`, `p ratio`, ... | a metric |
| `watch TARGET` | print a target at every stop (`watch del T`, `watch clear`) |
| `diagnose` | the history's regime, the worst tear entries, non-finite streams; at `terminated` the full `solve_findings` with remedies |
| `save [PATH]` | the current streams, history and tear table as JSON |
| `interact` | a Python prompt with `ctx`, `fs`, `streams`, `unit`, `inlets`, `outlets`, `x`, `g` in scope (REPL only) |

## Breakpoints

```text
break 12                       # at iteration 12   (tbreak 12: once)
break R1                       # before unit R1, every pass (or: break unit R1)
break if residual>1e3 && iter>5
break on residual_rise         # an event
break                          # list them
break del R1                   # delete one; break clear: all
```

Conditions are `metric OP value` joined by `&&`, over `iter`,
`residual`, `log_residual` and `ratio` (residual over the previous
residual: the observed contraction; a ratio near 1 is a loop gain near 1).

Events:

| Event | When |
|---|---|
| `nonfinite` | a unit turned finite inlets into a NaN or inf outlet --- where the NaN is *born*, not every unit it passes through --- or the tear residual is not finite |
| `inner_unconverged` | a unit's `info` says its own solve did not converge |
| `residual_rise` | the tear residual went up |
| `unit_error` | a unit raised |

## Post mortem: `on_error`

`--on-error` (`debug.attach(on_error=True)`) runs the solve freely and
stops once, at the first of: a unit that raises, a unit that produces a
NaN from finite inlets, or a solve that ends unconverged. A clean solve
never stops. It is the quickest way to answer "which unit broke it?",
including when it broke during the tear-guessing pass before the iteration
began.

## Driving it from a program

`difflow debug --json` speaks `difflow-dbg/1`: newline-delimited JSON,
one object per line. The first line is a `hello` listing every checkpoint,
event, metric, command and capability, so a client feature-detects off it
rather than off a version. Then a `pause` at each stop, and a `result` for
each command, echoing its `id` as `request_id`. A command is a bare string
(`"continue"`) or an object (`{"cmd": "print", "args": ["recycle"], "id": 3}`).

Inside Python, `difflow.debug.runner.DebugRunner` runs the solve on a
worker thread and hands back the same events one call at a time --- the
shape a server, an editor or an agent needs:

```python
from difflow.debug.runner import DebugRunner

runner = DebugRunner(fs, {"acceleration": "wegstein"})
runner.start()                          # {'event': 'pause', 'checkpoint': 'solve_start', ...}
runner.command("break on nonfinite")    # {'event': 'result', 'ok': True, ...}
runner.command("continue")["next"]      # the next pause, or {'event': 'finished', ...}
runner.close()
```

## What it does not change

Detached, a solve is exactly what it was: each checkpoint costs one
thread-local read. Attached, the debugger only reads: the same units run
in the same order on the same values, and the answer is the same. `quit`
raises `DebugQuit` instead of returning a partial answer, because a
stopped solve has none.

Only the outermost solve is debugged. A flowsheet solved inside a unit of
the one being debugged stays silent, and by default only the first solve
on the thread is debugged, so a script that sweeps does not stop you once
per point (`--all-solves` changes that).

## Limits

- **What it can step** is the eager tear iteration: `acceleration="anderson"`
  (the default) and `"wegstein"`. `acceleration="none"` iterates inside
  optimistix, where there is no Python loop to stop in; the debugger sees
  that solve's start and end only.
- **Nothing fires under `jax.grad` or `jit`.** A traced solve is traced
  once, not run, and a tracer is not a number anyone can inspect.
- **A unit is one step.** A unit's own inner solve (a CSTR's Newton
  iteration, a flash's Rachford-Rice) runs inside JAX and is not stepped;
  what it reports in its `info` dict is shown, and `inner_unconverged`
  stops on it.
- **It reads, it does not write.** Changing a tear value or a parameter
  mid-solve is not supported yet.
