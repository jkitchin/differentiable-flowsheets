"""Where the published documentation talks about each unit operation.

The editor can already render a unit's *docstring* --- that is
:mod:`difflow.gui.docs`, and it is what the inspector shows. This module
answers a different question: which page of the built book, at which
heading, discusses this operation. A docstring says what the arguments
are; the book says what the unit is for, what equations it solves and
what it assumes, and a reader who wants that should be one click away
from it rather than one search.

The mapping is derived rather than written down. A hand-kept table of 87
operations against a book that gets reorganised is a table that is wrong
within a release, and wrong silently --- a link to a heading that has
been renamed still returns 200 and simply lands at the top of the page.
So the answer is read out of ``static/docs-index.json``, the retrieval
index the assistant already builds from ``docs/`` (see
:mod:`difflow.gui.docs_index`), which is regenerated in CI whenever the
prose changes and therefore cannot drift from it.

Two strengths of match, and the difference is worth keeping:

* A **heading** that begins with the operation's name --- ``ShortcutColumn``,
  ``DistillationColumn (Rigorous)`` --- is the unit's own section, and
  wins outright.
* Failing that, the shallowest section that **mentions** the name. The gas
  and power plugins document their units in one reference table rather
  than a section apiece, so ``GasPipe`` lands on "Unit operation
  reference" in ``unit-operations-gas.md``: the right page and the right
  table, just not a heading of its own. Shallowest, because a mention in
  "Example Usage" is a worse destination than the same mention in the
  section that owns it.

A handful of operations are in no page at all and get ``None`` --- at the
time of writing ``EnthalpyCounterCurrentHX``, ``GroupSeparator``,
``LLEEquilibrium``, ``MultistageMembrane`` and ``ShellAndTubeHX``. That
is reported as an absence rather than papered over with a link to the
front page: the caller can offer the book's index instead, and a missing
entry here is a true statement that the unit is undocumented.
"""


from __future__ import annotations

import functools
import json
import re
from pathlib import Path

#: The built book. The trailing slash matters --- the page paths below
#: are relative to it, and ``urljoin`` on a base without one would eat
#: the last segment.
DOCS_BASE = "https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/"

#: Where the book puts a source file. ``_toc.yml`` lists the prose as
#: ``docs/unit-operations-chemical``, and jupyter-book mirrors the source
#: tree into the output, so ``docs/foo.md`` is served as ``docs/foo.html``.
DOCS_PATH_PREFIX = "docs/"

#: The retrieval index, as built by :mod:`difflow.gui.docs_index`.
INDEX = Path(__file__).parent / "static" / "docs-index.json"

#: Sources whose sections are about unit operations. Preferred over any
#: other page: ``Compressor`` is named in the planning tutorial too, and
#: the tutorial is not where a reader who clicked a compressor wants to
#: land.
_PREFERRED = "unit-operations"


def _sections() -> list[dict]:
    """The index's sections, or ``[]`` if it cannot be read.

    Built output, so a source checkout that has never run the build has
    no file here. That is a reason to offer no links, not to fail to
    start the editor.
    """
    try:
        return json.loads(INDEX.read_text(encoding="utf-8"))["sections"]
    except (OSError, ValueError, KeyError):
        return []


def _rank(section: dict) -> int:
    """0 for a unit-operations page, 1 for anything else."""
    return 0 if section.get("source", "").startswith(_PREFERRED) else 1


def _depth(section: dict) -> int:
    """How far down the heading tree a section sits.

    ``path`` is ``"Chemical Unit Operations > Separators > Flash Drum"``,
    so counting the separators counts the ancestors.
    """
    return section.get("path", "").count(">")


def _page(section: dict) -> str:
    """The published URL of the section, anchor and all."""
    stem = section["source"].rsplit(".", 1)[0]
    anchor = section.get("anchor", "")
    return f"{DOCS_BASE}{DOCS_PATH_PREFIX}{stem}.html" + (f"#{anchor}" if anchor else "")


@functools.lru_cache(maxsize=1)
def _cached_sections() -> tuple[dict, ...]:
    """:func:`_sections`, read once per process."""
    return tuple(_sections())


@functools.lru_cache(maxsize=512)
def url_for(operation: str) -> str | None:
    """The book's URL for one operation, or ``None`` if it has no page.

    Matched against the name it is asked about rather than against a
    guess at what an operation name looks like: a CamelCase pattern
    would have found ``ShortcutColumn`` and missed ``Junction`` and
    ``Transformer``, which are ordinary words and are exactly the names
    a heuristic cannot tell from prose.

    Args:
        operation: the registered name, e.g. ``"DistillationColumn"``.

    Returns:
        An absolute URL into the published book, anchored at the section
        that documents the operation, or ``None`` when the book never
        names it.
    """
    if not operation:
        return None
    sections = _cached_sections()
    owns = re.compile(r"^" + re.escape(operation) + r"\b")
    names = re.compile(r"\b" + re.escape(operation) + r"\b")
    best, found = None, None
    for order, section in enumerate(sections):
        heading = section.get("heading", "")
        if owns.match(heading):
            # A section of its own always wins, so depth is spent as -1
            # and the mention count does not come into it.
            key = (_rank(section), -1, 0, order)
        else:
            n = 3 * len(names.findall(heading)) + len(names.findall(section.get("text", "")))
            if not n:
                continue
            key = (_rank(section), _depth(section), -n, order)
        if best is None or key < best:
            best, found = key, section
    return _page(found) if found is not None else None
