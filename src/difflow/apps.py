"""Front ends installed beside difflow, found by entry point.

difflow is the engine. The browser editor --- the canvas, the inspector,
the Debug and Trace panels --- is an app that comes with the difflow
book, installed as its own package. It registers itself under the
``difflow.apps`` entry-point group::

    [project.entry-points."difflow.apps"]
    gui = "difflow_app.server:main"

and ``difflow gui`` (and the agent tool ``open_in_editor``) run whatever
is registered there. With nothing registered they say where the editor
comes from rather than failing on an import.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

#: The entry-point group front ends register under.
GROUP = "difflow.apps"

#: What to say when the editor is asked for and is not installed.
MISSING = (
    "The difflow editor (the browser app with the Debug and Trace panels) "
    "is not part of difflow: it comes with the difflow book and installs as "
    "its own package. Everything else works without it --- `difflow debug` "
    "steps a solve in the terminal, and `difflow mcp` serves the same "
    "session, debugger and record/replay tools to an agent."
)


def find(name: str = "gui") -> EntryPoint | None:
    """The entry point registered as ``name``, or None."""
    for entry in entry_points(group=GROUP):
        if entry.name == name:
            return entry
    return None


def run(name: str, argv: list[str]) -> int:
    """Run the front end registered as ``name`` with ``argv``; 2 if none is."""
    import sys

    entry = find(name)
    if entry is None:
        sys.stderr.write(f"difflow {name}: {MISSING}\n")
        return 2
    return entry.load()(argv) or 0


def module(name: str = "gui") -> str | None:
    """The top-level module of the front end ``name``, for ``python -m``."""
    entry = find(name)
    return None if entry is None else entry.module.split(".")[0]
