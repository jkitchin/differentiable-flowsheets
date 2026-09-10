"""The ``difflow`` command --- the front door to everything with a shell face.

Three things here are worth running without writing a script, and they
had three different invocations: ``python -m difflow.gui`` for the
editor, ``difflow`` for a report, ``difflow plan-export`` for delta
vectors. Only the last of those was discoverable. They are subcommands
now::

    difflow                              # open the editor
    difflow gui plant.json --port 9000   # ...on a flowsheet, on a port
    difflow report model.py --format html -o report.html
    difflow plan-export model.py -u reactor.V --format csv -o tables/

A bare ``difflow`` opens the editor rather than printing usage. It is
the one subcommand with nothing to say on stdout, the one a new user
wants first, and the alternative --- an argparse usage error --- taught
nobody anything.

``difflow model.py`` still means ``difflow report model.py``. That form
predates the subcommands and is what the older docs and scripts say, so
anything that is not a known subcommand falls through to ``report``.
The fall-through is deliberately narrow: a first argument that is not a
subcommand *and* not an existing file is a typo, and is reported as one
with the subcommand list, rather than handed to ``report`` to fail on a
missing script three frames later.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: subcommand -> (module, callable), imported only when dispatched to.
#: Importing jax to print `--help` is a second of nothing.
COMMANDS = {
    "gui": ("difflow.gui.server", "main"),
    "report": ("difflow.report.cli", "main"),
    "plan-export": ("difflow.planning.cli", "main"),
}

USAGE = """\
usage: difflow [<command>] [<args>]

Open the local flowsheet editor when given no command.

commands:
  gui           open the editor in a browser (the default)
  report        run a script and write its flowsheet report
  plan-export   linearize a model and write delta vectors for a planner

  difflow <command> --help   for that command's own options

options:
  -h, --help    show this message
  -V, --version show the installed version
"""


def _dispatch(name: str, argv: list[str]) -> int:
    module, attr = COMMANDS[name]
    import importlib

    return getattr(importlib.import_module(module), attr)(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        return _dispatch("gui", [])

    head = argv[0]
    if head in ("-h", "--help", "help"):
        sys.stdout.write(USAGE)
        return 0
    if head in ("-V", "--version"):
        from difflow import __version__

        print(__version__)
        return 0
    if head in COMMANDS:
        return _dispatch(head, argv[1:])
    if head.startswith("-") or Path(head).exists():
        return _dispatch("report", argv)      # `difflow model.py`, as before

    sys.stderr.write(
        f"difflow: no such command or file: {head!r}\n"
        f"commands: {', '.join(sorted(COMMANDS))}\n"
        "try `difflow --help`\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
