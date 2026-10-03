"""Rendering a unit's own docstring for the inspector.

The catalog already carries every unit's docstring, equations,
assumptions and references (:mod:`difflow.report.metadata`). What it
cannot do is decide how they should look, and a docstring pasted into a
panel as a wall of text is not documentation --- the sections, the
bullet lists and the doctest that shows the unit being constructed are
the part worth reading.

So the docstring is rendered as reStructuredText with :mod:`docutils`,
which is the same choice the FastSim editor made and which costs
difflow nothing: docutils is an *optional* extra, and when it is absent
:func:`render` falls back to the escaped source in a ``<pre>``. The
inspector says which of the two it got, so a missing extra reads as a
plainer panel rather than as a broken one.

Two accommodations are needed for the docstrings difflow actually has:

* They are Google-style, not reST. That mostly works out --- ``Args:``
  followed by an indented block *is* a reST definition list --- so
  nothing is rewritten. The stylesheet in the front end does the rest.
* They use Sphinx roles (``:class:`~difflow.streams.Stream```) that
  bare docutils does not know, and an unknown role is an error that
  swallows the line. :func:`register_roles` teaches docutils the dozen
  roles the codebase uses, rendering each as literal text with the
  ``~`` abbreviation honoured.

Messages are suppressed rather than emitted into the output: a handful
of docstrings have indentation docutils reads as a block quote, and a
red "System Message" box in the panel would be noise about difflow's
own prose, not about the user's flowsheet.
"""

from __future__ import annotations

import html
import re

#: Sphinx roles the difflow docstrings use, rendered as literal text.
SPHINX_ROLES = (
    "attr", "class", "data", "doc", "exc", "func", "meth", "mod",
    "obj", "ref", "term",
)

#: docutils settings. ``report_level=5`` silences messages, ``halt_level=5``
#: keeps even a severe one from raising, and file insertion and raw HTML
#: are off because a docstring is not a document we are publishing.
SETTINGS = {
    "report_level": 5,
    "halt_level": 5,
    "doctitle_xform": False,
    "syntax_highlight": "none",
    "embed_stylesheet": False,
    "input_encoding": "unicode",
    "file_insertion_enabled": False,
    "raw_enabled": False,
    # An image is a reference, never read from disk and inlined: with
    # `:loading: embed` docutils would paste an SVG file's markup in.
    "image_loading": "link",
    "_disable_config": True,
}

#: Link schemes a rendered docstring may keep. ``raw_enabled`` keeps
#: markup out, but a reST link target is copied into ``href`` as written,
#: so `` `x <javascript:...>`_ `` would run in the editor's page.
SAFE_SCHEMES = ("http", "https", "mailto")

_HREF = re.compile(r'\shref="([^"]*)"')
_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*):")
_IMG = re.compile(r"<img\b[^>]*>")
_ALT = re.compile(r'\salt="([^"]*)"')
_FETCH = re.compile(r'\s(?:src|poster|data)="[^"]*"')
_SVG = re.compile(r"<svg\b.*?</svg>", re.S | re.I)

_registered = False


def register_roles() -> None:
    """Teach docutils the Sphinx roles, once per process.

    Each renders as inline literal text, with a leading ``~`` meaning
    "show only the last component" as it does in Sphinx.
    """
    global _registered
    if _registered:
        return
    from docutils import nodes
    from docutils.parsers.rst import roles

    def role(name, rawtext, text, lineno, inliner, options=None, content=None):
        shown = text[1:].rsplit(".", 1)[-1] if text.startswith("~") else text
        return [nodes.literal(rawtext, shown)], []

    for name in SPHINX_ROLES:
        roles.register_local_role(name, role)
    _registered = True


def available() -> bool:
    """Whether docutils is importable, and so whether rendering is real."""
    try:
        import docutils.core  # noqa: F401
    except Exception:
        return False
    return True


def render(text: str) -> tuple[str, str]:
    """Render a docstring.

    Args:
        text: the docstring, already cleaned of its indentation.

    Returns:
        ``(html, format)``. ``format`` is ``"rst"`` when docutils
        rendered it and ``"text"`` when it did not, so a caller can say
        which it is showing rather than guessing from the markup.
    """
    if not text.strip():
        return "", "text"
    try:
        from docutils.core import publish_parts

        register_roles()
        parts = publish_parts(text, writer_name="html5", settings_overrides=SETTINGS)
    except Exception:
        return f"<pre>{html.escape(text)}</pre>", "text"
    return _no_fetches(_safe_links(parts["fragment"].strip())), "rst"


def _safe_links(fragment: str) -> str:
    """Drop every ``href`` whose scheme is not in :data:`SAFE_SCHEMES`.

    The value is read as a browser would read it: entities decoded, and
    the whitespace and control characters it ignores inside a URL
    removed, so ``java&#9;script:`` is not a way round the check. A link
    without a scheme (``#anchor``, a relative path) is kept.
    """
    def keep(match):
        url = re.sub(r"[\x00-\x20]", "", html.unescape(match.group(1))).lower()
        scheme = _SCHEME.match(url)
        return match.group(0) if scheme is None or scheme.group(1) in SAFE_SCHEMES else ""

    return _HREF.sub(keep, fragment)


def _no_fetches(fragment: str) -> str:
    """Show no image or media the browser would have to fetch.

    An ``.. image::`` in a plugin's docstring renders as an ``<img>`` the
    page loads on its own, from wherever it points --- a request the
    user never chose to make, which is what a tracking pixel is. A
    relative one would 404 against the editor's server anyway. So an
    image is shown as its alt text, and a ``<video>``'s source is
    dropped (it keeps the link docutils puts inside it, which is a
    request only when clicked).
    """
    def alt(match):
        found = _ALT.search(match.group(0))
        text = found.group(1) if found else "image"
        return f'<span class="image-alt">[{text}]</span>'

    fragment = _SVG.sub('<span class="image-alt">[image]</span>', fragment)
    return _FETCH.sub("", _IMG.sub(alt, fragment))
