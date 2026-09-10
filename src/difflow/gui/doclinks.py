"""Where to read about a unit operation.

Every entry in the editor's palette is a registered operation, and the
question a user asks of an unfamiliar one --- *what is this, and what do
its parameters mean?* --- is answered in ``docs/``, not in the editor.
So each operation gets a link, and this module is the one function that
decides where the link points.

It resolves against the committed section index
(:mod:`difflow.gui.docs_index`), because that is the only description of
``docs/`` a pip-installed difflow has: the prose itself is not in the
wheel. Resolution is therefore a lookup over headings and explicit MyST
targets, not a table hand-maintained next to the registry --- a table
would be one more thing to forget when a unit is renamed, and the
symptom would be a link that quietly goes to the wrong page.

Three ways an operation can be found, strongest first:

``target``
    An explicit MyST label: either the operation's own lowercased name,
    as the reference pages already label their sections
    (``(membraneseparator)=``), or ``op-<lowercase name>`` with an
    optional page prefix (``gas-op-gaspipe``), for a unit that has no
    section of its own to label. That second form is how a unit
    documented as one *row* of a reference table gets a link that lands
    on the row: the 25 equation-set units of the gas and power plugins
    are understood together and described together, and an anchor each
    is all they were missing. The prefix is not decoration --- MyST
    labels are global to the book, and two pages both documenting a
    ``Compressor`` may not both call the anchor ``compressor``.

``heading``
    A section whose heading names the operation. An exact heading wins
    over one that merely starts with the name, which wins over a mention
    anywhere in the heading, so ``CSTR`` prefers "### CSTR" to
    "#### CSTR Design Considerations".

``mention``
    The name appears in the body of a section and nowhere better. A
    usable link, but a weak one --- it lands on a section *about
    something else*. :mod:`tests.test_doclinks` asserts that no
    registered operation needs this path, so a new unit that only gets a
    passing mention fails the suite rather than shipping a vague link.

Nothing here raises. An index that was never built, or prose that never
names the unit, gives ``None``, and the caller shows no link: a missing
documentation link is a smaller defect than an editor that will not
open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from difflow.gui import docs_index

#: Where the book is published. The ``docs/`` prefix is part of the URL
#: because that is where the chapters live in the Jupyter Book source
#: tree, and the HTML mirrors it.
BASE_URL = "https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/docs/"

#: Resolution kinds, strongest first. See the module docstring.
KINDS = ("target", "heading", "mention")

#: Pages that document unit operations. A name can appear in a tutorial
#: or a release note too, and the reference page is the better landing
#: place when both match equally well.
_UNIT_PAGES = "unit-operations-"


def target_name(operation: str) -> str:
    """The MyST label that makes a link land exactly on *operation*.

    Authored in ``docs/`` as ``(op-gaspipe)=`` on the line before the
    paragraph, row or heading that describes the unit. A page-specific
    prefix (``gas-op-gaspipe``) is equally good and is what the network
    plugins use, since MyST labels are global to the book and two pages
    may both document a ``Compressor``. A label that is simply the
    lowercased name --- the convention the reference pages already use
    for their own sections --- is recognised too.
    """
    return f"op-{operation.lower()}"


@dataclass(frozen=True)
class DocLink:
    """Where one operation is documented.

    Attributes:
        operation: The registered operation name that was resolved.
        source: The markdown file in ``docs/``, e.g. ``unit-operations-gas.md``.
        anchor: The in-page anchor, without its ``#``.
        heading: The heading of the section the link lands in, for a
            tooltip or a link title.
        kind: One of :data:`KINDS` --- how it was found, and therefore
            how good the landing place is.
    """

    operation: str
    source: str
    anchor: str
    heading: str
    kind: str

    @property
    def page(self) -> str:
        """The published HTML file name for :attr:`source`."""
        return re.sub(r"\.md$", ".html", self.source)

    @property
    def url(self) -> str:
        """The full published URL, anchor included."""
        anchor = f"#{self.anchor}" if self.anchor else ""
        return f"{BASE_URL}{self.page}{anchor}"


def _heading_rank(heading: str, word: re.Pattern) -> int | None:
    """How squarely a heading names the operation: 0 is best, ``None`` no match.

    The three tiers are "the heading *is* the name" (surrounding
    backticks or emphasis stripped, since a heading may be written
    ``### `Mixer```), "the heading starts with it" --- which is what
    ``### CSTR (Continuous Stirred-Tank Reactor)`` is --- and "the name
    is in there somewhere", as in ``### The Saponifier``.
    """
    match = word.search(heading)
    if match is None:
        return None
    stripped = heading.strip().strip("`*_ ")
    if stripped == match.group(0):
        return 0
    return 1 if match.start() == 0 else 2


def _resolve(operation: str, sections: list[dict]) -> DocLink | None:
    """The best link for *operation* over already-loaded sections."""
    # An empty name is not a unit that happens to be undocumented, it is
    # a caller with nothing to ask about --- and it must be rejected
    # here rather than left to match: `\b\b` matches at every position,
    # so every heading in the book would rank as naming it.
    if not operation.strip():
        return None
    word = re.compile(rf"\b{re.escape(operation)}\b")
    label = target_name(operation)
    bare = operation.lower()

    best: tuple[tuple, DocLink] | None = None

    def offer(key: tuple, link: DocLink) -> None:
        nonlocal best
        if best is None or key < best[0]:
            best = (key, link)

    for order, section in enumerate(sections):
        source = section["source"]
        # A reference page outranks a tutorial that happens to match as
        # well; document order breaks what is left.
        page_rank = 0 if source.startswith(_UNIT_PAGES) else 1

        for target in section.get("targets") or ():
            if target in (label, bare) or target.endswith(f"-{label}"):
                offer(
                    (KINDS.index("target"), page_rank, order),
                    DocLink(operation, source, target,
                            section["heading"], "target"),
                )

        rank = _heading_rank(section["heading"], word)
        if rank is not None:
            offer(
                (KINDS.index("heading"), rank, page_rank, order),
                DocLink(operation, source, section["anchor"],
                        section["heading"], "heading"),
            )

        hits = len(word.findall(section["text"]))
        if hits:
            # Negated, so the section that says the name most often --
            # the one most likely to be about it -- sorts first.
            offer(
                (KINDS.index("mention"), page_rank, -hits, order),
                DocLink(operation, source, section["anchor"],
                        section["heading"], "mention"),
            )

    return None if best is None else best[1]


@lru_cache(maxsize=None)
def resolve(operation: str) -> DocLink | None:
    """Where *operation* is documented, or ``None`` if nowhere.

    Cached: the index does not change while the server is up, and the
    palette asks for all 87 operations on every catalog fetch.

    Example:
        >>> resolve("CSTR").source
        'unit-operations-chemical.md'
    """
    index = docs_index.load()
    if not index or not index.get("sections"):
        return None
    return _resolve(operation, index["sections"])


def url_for(operation: str) -> str | None:
    """The documentation URL for *operation*, or ``None`` if undocumented.

    Example:
        >>> url_for("GasPipe")
        'https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/docs/unit-operations-gas.html#gas-op-gaspipe'
    """
    link = resolve(operation)
    return None if link is None else link.url
