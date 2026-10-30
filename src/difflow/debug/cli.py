"""``difflow debug`` --- solve a flowsheet under the debugger.

    difflow debug plant.py                      # REPL, stops before the first pass
    difflow debug plant.json --on-error         # run; stop only if something fails
    difflow debug plant.py --json               # difflow-dbg/1 on stdin/stdout
    difflow debug plant.py --script warmup.dbg  # run these commands first

A script is run with the debugger already attached, so the first solve
it makes is the one debugged, with the arguments the script gave it. A
script that never solves (or a ``.json`` flowsheet) is solved afterwards
with the options given here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="difflow debug", description=__doc__.split("\n\n")[0])
    p.add_argument("model", help="a flowsheet script (.py) or saved flowsheet (.json)")
    p.add_argument("--json", action="store_true",
                   help="speak difflow-dbg/1 JSON on stdin/stdout instead of a prompt")
    p.add_argument("--on-error", action="store_true",
                   help="run freely; stop only at a failing unit or an unconverged end")
    p.add_argument("--script", help="debugger commands to run at the first pause")
    p.add_argument("--acceleration", choices=["anderson", "wegstein", "none"],
                   help="tear acceleration (anderson and wegstein can be stepped)")
    p.add_argument("--tol", type=float, help="tear tolerance")
    p.add_argument("--max-iter", type=int, help="tear iteration limit")
    p.add_argument("--all-solves", action="store_true",
                   help="debug every solve the script makes, not only the first")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from difflow import debug
    from difflow import scripts

    kwargs = {k: v for k, v in (("acceleration", args.acceleration), ("tol", args.tol),
                                ("max_iter", args.max_iter)) if v is not None}
    session = debug.make_session("json" if args.json else "repl", script=args.script,
                                 on_error=args.on_error,
                                 max_solves=None if args.all_solves else 1)
    if args.json:
        session.frontend.greet()

    path = Path(args.model)
    if not path.exists():
        sys.stderr.write(f"difflow debug: no such file: {path}\n")
        return 2
    try:
        with debug.attach(session):
            if scripts.is_script(path):
                fs = scripts.load_flowsheet(path)
            else:
                from difflow import serialize
                fs = serialize.load(path)
            if session.solves == 0:
                fs.solve(**kwargs)
    except debug.DebugQuit:
        sys.stderr.write("difflow debug: stopped\n")
        return 1
    except scripts.ScriptError as exc:
        if isinstance(exc.__cause__, debug.DebugQuit):   # quit inside the script's solve
            sys.stderr.write("difflow debug: stopped\n")
            return 1
        sys.stderr.write(f"difflow debug: {exc}\n")
        return 2
    converged = getattr(fs, "last_solve_converged", None)
    sys.stderr.write(f"difflow debug: solve finished, converged={converged}\n")
    return 0 if converged is not False else 1


if __name__ == "__main__":
    sys.exit(main())
