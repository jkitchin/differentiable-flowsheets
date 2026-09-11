"""Open a flowsheet that is written as a Python script rather than a file.

A flowsheet has two forms. ``plant.json`` is what the editor reads and
writes; ``plant.py`` is what someone wrote by hand, and there are far
more of the second, because the package was a library for two years
before it had an editor.

Running the script is the only way to read one. A flowsheet is built by
arbitrary code --- a rate law is a function, a parameter is whatever
arithmetic produced it --- so there is nothing to parse. That is also
why this is not, and cannot be made, safe against a hostile file: it
executes it. The rule is the one the shell already uses. ``difflow
report plant.py`` and ``difflow plan-export plant.py`` have run user
scripts since before the editor existed; this is the same act, asked
for the same way, by someone who is about to run the file anyway.

What it does NOT do is write one back. :mod:`difflow.codegen` can emit
a script from a flowsheet, but emitting it over the script it came from
would delete everything in that file that is not the flowsheet --- the
comments, the sweep at the bottom, the ``if __name__`` block. So a
script is a way IN only, and the editor saves the JSON beside it.
"""

from __future__ import annotations

import runpy
from pathlib import Path

#: Suffixes read by running them. Not a guess about content: a file
#: named ``.py`` is asking to be run, and one that is not is not.
SCRIPT_SUFFIXES = (".py",)


class ScriptError(Exception):
    """A script did not yield a flowsheet, or did not run at all."""


def is_script(path: str | Path | None) -> bool:
    """Whether ``path`` names a flowsheet to run rather than to read."""
    return bool(path) and Path(path).suffix.lower() in SCRIPT_SUFFIXES


def find_flowsheet(namespace: dict):
    """The first flowsheet in a namespace, as ``(name, flowsheet)``.

    First by definition order, which for a module namespace is the order
    the file defines things in. A script that builds an intermediate
    flowsheet and then the real one is rare; a script whose flowsheet is
    the first one it makes is the normal case.
    """
    from difflow.flowsheet import Flowsheet

    for name, value in namespace.items():
        if isinstance(value, Flowsheet):
            return name, value
    return None, None


def load_flowsheet(path: str | Path, *, run_name: str = "__difflow__"):
    """Run ``path`` and return the flowsheet it built.

    Args:
        path: a Python script that builds a :class:`~difflow.flowsheet.Flowsheet`.
        run_name: the ``__name__`` the script runs under. The default is
            deliberately not ``"__main__"``: a script's ``if __name__ ==
            "__main__"`` block is the part that solves, sweeps or plots,
            and opening a flowsheet in an editor should not run an hour
            of optimization first. A script that wants to be opened puts
            the flowsheet at module level, which is where it already is.

    Raises:
        ScriptError: the script raised, or built no flowsheet. Either
            way with the script named and the cause kept, because "no
            flowsheet found" and "your script raised on line 40" are
            different problems and look identical from the outside.
    """
    path = Path(path)
    try:
        namespace = runpy.run_path(str(path), run_name=run_name)
    except Exception as exc:                      # the script's, not ours
        raise ScriptError(
            f"{path.name} raised while building the flowsheet: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _, flowsheet = find_flowsheet(namespace)
    if flowsheet is None:
        raise ScriptError(
            f"{path.name} ran, but defines no Flowsheet. The editor opens "
            "the first one a script leaves at module level -- a flowsheet "
            "built inside a function, or only under `if __name__ == "
            '"__main__"`, is not visible from outside it.'
        )
    return flowsheet


def save_target(path: str | Path) -> Path:
    """Where the editor saves a flowsheet that was opened from a script.

    Beside the script, under the same stem: ``plant.py`` saves to
    ``plant.json``. Never back over the script itself --- see the module
    docstring for why that would be a deletion rather than a save.
    """
    return Path(path).with_suffix(".json")
