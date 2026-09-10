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
    "_disable_config": True,
}

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
    return parts["fragment"].strip(), "rst"
