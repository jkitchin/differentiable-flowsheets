"""The ``difflow`` command --- the front door to everything with a shell face.

The things worth running without writing a script are subcommands::

    difflow gui plant.json --port 9000   # the editor, if the book's app is installed
    difflow report model.py --format html -o report.html
    difflow plan-export model.py -u reactor.V --format csv -o tables/
    difflow mcp                          # tools for an agent, over stdio
    difflow debug model.py               # step through a solve

A bare ``difflow`` opens the editor when one is installed (it is the
first thing a new user wants, and has nothing to say on stdout), and
prints this usage when not. The editor is an app that comes with the
difflow book and registers itself as a ``difflow.apps`` entry point;
``gui`` runs whatever is registered (see :mod:`difflow.apps`).

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
    "gui": ("difflow.apps", "gui"),             # the book's app, if installed
    "report": ("difflow.report.cli", "main"),
    "plan-export": ("difflow.planning.cli", "main"),
    "mcp": ("difflow.mcp.server", "main"),
    "debug": ("difflow.debug.cli", "main"),
}

USAGE = """\
usage: difflow [<command>] [<args>]

Open the flowsheet editor when given no command (if it is installed).

commands:
  gui           open the browser editor (it comes with the difflow book)
  report        run a script and write its flowsheet report
  plan-export   linearize a model and write delta vectors for a planner
  mcp           serve the flowsheet tools to an agent over stdio (MCP)
  debug         solve a flowsheet under the interactive debugger

  difflow <command> --help   for that command's own options

options:
  -h, --help    show this message
  -V, --version show the installed version
"""


def _dispatch(name: str, argv: list[str]) -> int:
    module, attr = COMMANDS[name]
    if module == "difflow.apps":
        from difflow import apps

        return apps.run(attr, argv)
    import importlib

    return getattr(importlib.import_module(module), attr)(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        from difflow import apps

        if apps.find("gui") is None:
            sys.stdout.write(USAGE)
            return 0
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
