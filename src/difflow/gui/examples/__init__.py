"""Example flowsheets the editor offers to open.

Each is an ordinary flowsheet file, the same JSON ``python -m difflow.gui
plant.json`` opens and **File > Flowsheet JSON** writes, so an example is
also a working answer to "what does a file look like". Its title and
one-line description ride in ``view``, which the format already carries
through untouched.

Files rather than Python that builds them: the point of an example here
is that it opens in the editor, and the only way to be sure a file opens
is to ship the file. ``tests/test_gui_examples.py`` opens and solves
every one, so a change to the format that breaks them fails the suite
rather than the menu.

To add one, drop a ``NN_name.json`` in this directory with
``view.title`` and ``view.description`` set; the ``NN`` prefix orders
the menu.
"""

from __future__ import annotations

import json
from pathlib import Path

#: where the example files live
DIRECTORY = Path(__file__).parent


def _files() -> dict[str, Path]:
    return {p.stem: p for p in sorted(DIRECTORY.glob("*.json"))}


def listing() -> list[dict]:
    """``[{key, title, description}]``, in menu order."""
    out = []
    for key, path in _files().items():
        view = json.loads(path.read_text()).get("view") or {}
        out.append({
            "key": key,
            "title": view.get("title") or key,
            "description": view.get("description") or "",
        })
    return out


def document(key: str) -> dict:
    """The flowsheet document for one example.

    Looked up among the files that exist rather than joined onto the
    directory, so a key is a name and never a path.

    Raises:
        KeyError: if there is no example by that name.
    """
    path = _files().get(key)
    if path is None:
        raise KeyError(key)
    return json.loads(path.read_text())
