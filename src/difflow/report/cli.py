"""``difflow report`` CLI entry point.

Run a user script, locate the first :class:`~difflow.flowsheet.Flowsheet`
instance it creates, and emit a report in the requested format.

Usage::

    difflow report script.py                         # Markdown to stdout
    difflow report script.py --format json -o out.json
    difflow report script.py --solve                 # run fs.solve() first
"""

from __future__ import annotations

import argparse
import runpy
import sys


def _find_flowsheet(namespace: dict):
    """Return the first Flowsheet-like value defined in ``namespace``."""
    from difflow.flowsheet import Flowsheet

    for name, value in namespace.items():
        if isinstance(value, Flowsheet):
            return name, value
    return None, None


def _find_streams(namespace: dict):
    """Return the first ``dict[str, Stream]`` that looks like solved streams."""
    for name, value in namespace.items():
        if not isinstance(value, dict):
            continue
        ok = True
        for v in value.values():
            if not isinstance(v, dict) or "T" not in v or "P" not in v:
                ok = False
                break
        if ok and value:
            return name, value
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="difflow report",
        description="Generate a self-documenting report for a difflow flowsheet.",
    )
    parser.add_argument("script", help="Python script that builds a Flowsheet")
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "html", "latex"),
        default="markdown",
    )
    parser.add_argument("-o", "--output", help="Output path (default: stdout)")
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Call fs.solve() before building the report",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip git commit capture in provenance",
    )
    parser.add_argument(
        "--subcommand", nargs="?", default=None, help=argparse.SUPPRESS
    )

    # Support both ``difflow report script.py`` (script entry wires through
    # ``main``) and a bare ``python -m difflow.report.cli script.py`` invocation.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "report":
        argv = argv[1:]
    args = parser.parse_args(argv)

    ns = runpy.run_path(args.script, run_name="__difflow_report__")
    name, fs = _find_flowsheet(ns)
    if fs is None:
        print("error: no Flowsheet instance found in script", file=sys.stderr)
        return 2

    streams = None
    if args.solve:
        streams = fs.solve()
    else:
        _, streams = _find_streams(ns)

    from difflow.report import build_report

    rep = build_report(fs, streams=streams, include_git=not args.no_git)

    if args.format == "markdown":
        text = rep.to_markdown()
    elif args.format == "json":
        text = rep.to_json()
    elif args.format == "html":
        text = rep.to_html()
    else:
        text = rep.to_latex()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
